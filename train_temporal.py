import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.optimizer import (
    build_optimizer_with_prefix_multipliers,
    build_warmup_cosine_factor_scheduler,
    linear_ramp,
    _prob_sched_epoch
)
from my_logging import TrainStats, log_train_step, log_train_epoch_summary, _extract_losses

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from xmem_det.util import load_xmem_train_cfg
from config_utils import (
    MissingConfigError,
    cfg_req_bool,
    cfg_req_float,
    cfg_req_int,
    cfg_req_list_str,
    cfg_req_nn,
    cfg_req_str,
    cfg_req_key,
    _require_pos_int,
    _require_prob_01
)


#return probability flag for stochastic decisions
def sample_flag(p: float) -> bool:
    p = _require_prob_01("p", float(p))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(torch.rand((), device="cpu").item() < p)

# set trainable parameters based on prefixes
def set_trainable_prefixes(model, prefixes: List[str]) -> None:
    prefixes_t = tuple(prefixes)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(prefixes_t)

# get relative transform between current and previous frame
def rel_T_curr_prev(T_world_lidar: np.ndarray, t: int) -> np.ndarray:
    T_prev = T_world_lidar[t - 1]
    T_curr = T_world_lidar[t]
    return (np.linalg.inv(T_curr) @ T_prev).astype(np.float32)

# convert frame dict to torch batch dict
def to_torch_batch_dict(frame_dict: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch_dict: Dict[str, Any] = {}
    for k, v in frame_dict.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                batch_dict[k] = v
            else:
                batch_dict[k] = torch.from_numpy(v).to(device, non_blocking=True)
        elif isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device, non_blocking=True)
        else:
            batch_dict[k] = v
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict

# build sequence data loader
def build_seq_loader(cfg_obj, logger, workers_arg: int):
    seq_len = cfg_req_int(cfg_obj, "TRAIN.SEQ_LEN")
    stride = cfg_req_int(cfg_obj, "TRAIN.STRIDE")

    dl_cfg = cfg_req_nn(cfg_obj, "ADDONS.DATALOADER")
    batch_size = cfg_req_int(dl_cfg, "BATCH_SIZE")
    shuffle = cfg_req_bool(dl_cfg, "SHUFFLE")
    pin_memory = cfg_req_bool(dl_cfg, "PIN_MEMORY")
    drop_last = cfg_req_bool(dl_cfg, "DROP_LAST")

    dataset_cfg = cfg_obj.DATA_CONFIG
    class_names = cfg_obj.CLASS_NAMES

    train_set = NuScenesSeqDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=True,
        root_path=None,
        logger=logger,
        num_groups=int(seq_len),
    )

    num_workers_cap = cfg_req_int(cfg_obj, "OPTIMIZATION.NUM_WORKERS")
    num_workers = min(int(workers_arg), int(num_workers_cap))

    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_seq,
        drop_last=bool(drop_last),
    )

    return train_set, train_loader

# parse group specifications for optimizer
def _parse_group_specs(specs: Any) -> List[Tuple[Tuple[str, ...], float]]:
    if not isinstance(specs, (list, tuple)) or len(specs) == 0:
        raise MissingConfigError("Missing or empty config key: ADDONS.PHASES.<phase>.OPT_GROUP_SPECS")
    out: List[Tuple[Tuple[str, ...], float]] = []
    for s in list(specs):
        if not isinstance(s, dict):
            raise MissingConfigError("Each OPT_GROUP_SPECS entry must be a dict")
        if "PREFIXES" not in s or "LR_MULT" not in s:
            raise MissingConfigError("Each OPT_GROUP_SPECS entry must have PREFIXES and LR_MULT")
        prefixes = tuple([str(x) for x in s["PREFIXES"]])
        mult = float(s["LR_MULT"])
        out.append((prefixes, mult))
    return out


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    train_loader,
    epoch: int,
    total_epochs: int,
    logger,
    device: torch.device,
    alpha_temporal: float,
    loop_cfg: Any,
    run_cfg: Any,
    probs: Dict[str, float],
):
    model.train()

    stats_window = cfg_req_int(loop_cfg, "STATS_WINDOW")
    log_every = cfg_req_int(loop_cfg, "LOG_EVERY")
    max_grad_norm = cfg_req_float(loop_cfg, "MAX_GRAD_NORM")

    stats = TrainStats(w=int(stats_window))

    det_mask_prev = None
    for seq_idx, seq in enumerate(train_loader):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Get sequence frames
        frames = seq["frames"]
        T = len(frames)
        if T == 0:
            continue
        
        # Reset model state
        if hasattr(model, "reset_sequence"):
            model.reset_sequence(seq_idx)

        # Convert all frames to torch
        frames_list = []
        for t in range(T):
            frame = frames[t]
            batch_dict = to_torch_batch_dict(frame, device)
            frames_list.append(batch_dict)
        
        vis_every = int(cfg_req_int(run_cfg, "VIS_EVERY_SEQS"))
        vis_dir = str(cfg_req_str(run_cfg, "VIS_DIR"))
        vis_tpl = str(cfg_req_str(run_cfg, "VIS_TAG_TEMPLATE"))
        vis_b = int(cfg_req_int(run_cfg, "VIS_B"))

        do_vis = (vis_every > 0) and ((seq_idx % vis_every) == 0)

        last_bd = frames_list[-1]
        last_bd["_vis"] = bool(do_vis)
        last_bd["_vis_dir"] = vis_dir
        last_bd["_vis_b"] = vis_b
        last_bd["_vis_tag"] = vis_tpl.format(tag=str(run_cfg.get("PHASE", "phase")), epoch=epoch + 1, seq=seq_idx)

        
        # === FORWARD PASS ALL FRAMES AT ONCE ===
        use_det_t0 = not sample_flag(probs.get("p_teacher"))
        # print("use_det_t0:", use_det_t0)
        ret_dict, tb_dict, disp_dict, det_masks = model(
            frames_list=frames_list,
            # alpha_temporal=float(alpha_temporal),
            compute_det_loss=True,
            # compute_aux_loss=False,
            # use_det_t0=use_det_t0,
        )


        loss = ret_dict["loss"]
        
        # Backprop
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Logging
        loss_total_v = loss.item()
        loss_det_v = tb_dict.get("loss_det", torch.zeros(())).item()
        loss_occ_v = tb_dict.get("loss_aux_occ", torch.zeros(())).item()
        loss_cons_v = 0.0
        
        stats.update_main(loss_total_v, loss_det_v, loss_occ_v, loss_cons_v)
        stats.update_optional(tb_dict)

        if (seq_idx + 1) % int(log_every) == 0:
            lr = optimizer.param_groups[0]["lr"]
            log_train_step(
                logger=logger,
                stats=stats,
                epoch=epoch,
                total_epochs=total_epochs,
                seq_idx=seq_idx,
                num_seqs=len(train_loader),
                lr=lr,
                loss_total_v=loss_total_v,
                loss_det_v=loss_det_v,
                loss_occ_v=loss_occ_v,
                loss_cons_v=loss_cons_v,
                tb_dict=tb_dict,
                batch_dict=frames_list[-1],
            )

    log_train_epoch_summary(logger=logger, stats=stats, epoch=epoch, total_epochs=total_epochs)
    return {
        "avg_total": stats.total.avg(),
        "avg_det": stats.det.avg(),
        "avg_occ": stats.occ.avg(),
        "avg_cons": stats.cons.avg(),
        "win200_total": stats.total.win_avg(),
        "win200_det": stats.det.win_avg(),
        "win200_occ": stats.occ.avg(),
        "win200_cons": stats.cons.win_avg(),
    }

def train_phase(
    phase_id: str,
    extra_tag: str,
    model,
    train_loader,
    cfg_obj,
    logger,
    device: torch.device,
    run_cfg: Any,
    loop_cfg: Any,
    phase_cfg: Any,
    resume_blob: Any,
):
    prefixes = cfg_req_list_str(phase_cfg, "PREFIXES")
    set_trainable_prefixes(model, prefixes)

    start_epoch_cfg = cfg_req_int(phase_cfg, "START_EPOCH")
    end_epoch = cfg_req_int(phase_cfg, "END_EPOCH")

    use_ckpt_epoch = cfg_req_bool(run_cfg, "USE_CKPT_EPOCH")
    start_epoch = int(start_epoch_cfg)

    if resume_blob is not None and use_ckpt_epoch:
        start_epoch = int(resume_blob.get("epoch", start_epoch))

    if start_epoch >= end_epoch:
        logger.info(f"Phase {phase_id}: start_epoch={start_epoch} >= end_epoch={end_epoch}, nothing to do")
        return

    group_specs = _parse_group_specs(cfg_req_nn(phase_cfg, "OPT_GROUP_SPECS"))

    lr_start = cfg_req_float(phase_cfg, "LR_START")
    lr_max = cfg_req_float(phase_cfg, "LR_MAX")
    lr_end = cfg_req_float(phase_cfg, "LR_END")
    warmup_epochs = cfg_req_int(phase_cfg, "WARMUP_EPOCHS")

    optimizer = build_optimizer_with_prefix_multipliers(
        model=model,
        cfg=cfg_obj,
        base_lr=float(lr_max),
        group_specs=group_specs,
    )

    scheduler = build_warmup_cosine_factor_scheduler(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        epochs=(int(end_epoch) - int(start_epoch)),
        lr_start=float(lr_start),
        lr_max=float(lr_max),
        lr_end=float(lr_end),
        warmup_epochs=int(warmup_epochs),
    )

    resume_opt = cfg_req_bool(run_cfg, "RESUME_OPTIM_SCHED")
    if resume_blob is not None and resume_opt:
        opt_state = resume_blob.get("optimizer_state", None)
        sch_state = resume_blob.get("scheduler_state", None)
        if opt_state is None or sch_state is None:
            raise MissingConfigError("Checkpoint missing optimizer_state/scheduler_state but RESUME_OPTIM_SCHED is true")
        optimizer.load_state_dict(opt_state)
        scheduler.load_state_dict(sch_state)

    alpha_start = cfg_req_float(phase_cfg, "ALPHA_START")
    alpha_end = cfg_req_float(phase_cfg, "ALPHA_END")
    alpha_ramp_epochs = cfg_req_int(phase_cfg, "ALPHA_RAMP_EPOCHS")

    ckpt_dir = cfg_req_str(run_cfg, "CKPT_DIR")
    ckpt_name_tpl = cfg_req_str(phase_cfg, "CKPT_NAME_TEMPLATE")
    os.makedirs(str(ckpt_dir), exist_ok=True)

    logger.info(
        f"Phase {phase_id} start: start_epoch={start_epoch}, end_epoch={end_epoch}, steps_per_epoch={len(train_loader)}, "
        f"lr_start={float(lr_start):.3e}, lr_max={float(lr_max):.3e}, lr_end={float(lr_end):.3e}, warmup_epochs={int(warmup_epochs)}, "
        f"alpha_start={float(alpha_start):.3f}, alpha_end={float(alpha_end):.3f}, alpha_ramp_epochs={int(alpha_ramp_epochs)}"
    )

    for epoch in range(int(start_epoch), int(end_epoch)):
        alpha = linear_ramp(
            epoch_idx=epoch,
            start_epoch=int(start_epoch),
            start=float(alpha_start),
            end=float(alpha_end),
            ramp_epochs=int(alpha_ramp_epochs),
        )

        probs = _prob_sched_epoch(phase_cfg=phase_cfg, epoch=epoch, start_epoch=int(start_epoch))

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}/{end_epoch} "
            f"alpha={alpha:.3f} lr={lr:.3e} "
            f"pT={probs['p_teacher']:.3f} "
            # f"pT_last={probs['p_teacher_last']:.3f} pC_last={probs['p_corrupt_last']:.3f}"
        )

        train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=int(end_epoch),
            logger=logger,
            device=device,
            alpha_temporal=float(alpha),
            loop_cfg=loop_cfg,
            run_cfg=run_cfg,
            probs=probs,
        )

        ckpt_path = os.path.join(
            str(ckpt_dir),
            ckpt_name_tpl.format(phase=phase_id, tag=extra_tag, epoch=epoch + 1),
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "phase": str(phase_id),
                "phase_cfg": dict(phase_cfg),
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--xmem_cfg", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra_tag", type=str, default="default")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    xmem_train_cfg = load_xmem_train_cfg(args.xmem_cfg)

    run_cfg = cfg_req_nn(cfg, "ADDONS.RUN")
    loop_cfg = cfg_req_nn(cfg, "ADDONS.TRAIN_LOOP")
    phases_cfg = cfg_req_nn(cfg, "ADDONS.PHASES")

    phase_id = cfg_req_str(run_cfg, "PHASE")
    phase_cfg = cfg_req_nn(phases_cfg, str(phase_id))

    device_str = cfg_req_str(run_cfg, "DEVICE").lower()
    if device_str not in ("cpu", "cuda"):
        raise MissingConfigError("ADDONS.RUN.DEVICE must be 'cpu' or 'cuda'")

    if device_str == "cuda" and not torch.cuda.is_available():
        raise MissingConfigError("ADDONS.RUN.DEVICE is 'cuda' but CUDA is not available")

    device = torch.device(device_str)

    log_dir = cfg_req_str(run_cfg, "LOG_DIR")
    log_file_tpl = cfg_req_str(run_cfg, "LOG_FILE_TEMPLATE")
    os.makedirs(str(log_dir), exist_ok=True)

    log_file = os.path.join(str(log_dir), log_file_tpl.format(tag=str(args.extra_tag)))
    logger = common_utils.create_logger(log_file)

    seq_len = cfg_req_int(cfg, "TRAIN.SEQ_LEN")
    stride = cfg_req_int(cfg, "TRAIN.STRIDE")
    logger.info(str(cfg))
    logger.info(f"seq_len={seq_len}, stride={stride}")

    train_set, train_loader = build_seq_loader(cfg_obj=cfg, logger=logger, workers_arg=int(args.workers))

    num_class = len(cfg.CLASS_NAMES)
    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=num_class,
        dataset=train_set.base,
        # xmem_train_cfg=xmem_train_cfg,
        pc_range=pc_range,
    )
    model.to(device)

    names = [n for n, _ in model.named_parameters()]
    tops = sorted({n.split(".")[0] for n in names})
    print(tops)

    resume_ckpt = cfg_req_key(run_cfg, "RESUME_CKPT")
    pretrained_pp_ckpt = cfg_req_key(run_cfg, "PRETRAINED_PP_CKPT")

    if resume_ckpt is not None and pretrained_pp_ckpt is not None:
        raise MissingConfigError("Set only one of ADDONS.RUN.RESUME_CKPT or ADDONS.RUN.PRETRAINED_PP_CKPT (the other must be null)")

    resume_blob = None
    if resume_ckpt is not None:
        resume_blob = torch.load(str(resume_ckpt), map_location="cpu")
        model.load_state_dict(resume_blob["model_state"], strict=False)
        logger.info(f"Resumed model from {resume_ckpt} at epoch {int(resume_blob.get('epoch', 0))}")
    elif pretrained_pp_ckpt is not None:
        # ckpt = torch.load(str(pretrained_pp_ckpt), map_location="cpu")
        # state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        # model.load_state_dict(state, strict=False)
        # logger.info(f"Loaded pretrained PP from {pretrained_pp_ckpt}")
    
        ckpt = torch.load(str(pretrained_pp_ckpt), map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        # Filter out dense_head
        backbone_state = {k: v for k, v in state.items() if not k.startswith('dense_head.')}
        skipped = len(state) - len(backbone_state)
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        logger.info(f"✅ Loaded backbone only from {pretrained_pp_ckpt}")
        logger.info(f"⏭️  Skipped {skipped} dense_head parameters (will be random)")
        logger.info(f"🎲 Missing (random): {len(missing)} parameters")


    # check required config keys
    cfg_req_int(phase_cfg, "START_EPOCH")
    cfg_req_int(phase_cfg, "END_EPOCH")
    cfg_req_float(phase_cfg, "LR_START")
    cfg_req_float(phase_cfg, "LR_MAX")
    cfg_req_float(phase_cfg, "LR_END")
    cfg_req_int(phase_cfg, "WARMUP_EPOCHS")
    cfg_req_float(phase_cfg, "ALPHA_START")
    cfg_req_float(phase_cfg, "ALPHA_END")
    cfg_req_int(phase_cfg, "ALPHA_RAMP_EPOCHS")
    cfg_req_str(phase_cfg, "CKPT_NAME_TEMPLATE")
    cfg_req_nn(phase_cfg, "PROB_SCHEDULE")
    cfg_req_nn(phase_cfg, "OPT_GROUP_SPECS")

    cfg_req_int(loop_cfg, "STATS_WINDOW")
    cfg_req_int(loop_cfg, "LOG_EVERY")
    cfg_req_float(loop_cfg, "MAX_GRAD_NORM")
    cfg_req_bool(loop_cfg, "FORCE_TEACHER_ON_T0")
    cfg_req_bool(loop_cfg, "BURN_IN_TEACHER_FORCE")
    cfg_req_bool(loop_cfg, "BURN_IN_OCC_CORRUPT")
    cfg_req_float(loop_cfg, "OCC_DROP_RATE")
    cfg_req_int(loop_cfg, "OCC_BLOCK")
    cfg_req_float(loop_cfg, "XMEM_PREV_THR")

    cfg_req_str(run_cfg, "VIS_DIR")
    cfg_req_int(run_cfg, "VIS_EVERY_SEQS")
    cfg_req_str(run_cfg, "VIS_TAG_TEMPLATE")
    cfg_req_int(run_cfg, "VIS_B")
    cfg_req_str(run_cfg, "CKPT_DIR")
    cfg_req_bool(run_cfg, "USE_CKPT_EPOCH")
    cfg_req_bool(run_cfg, "RESUME_OPTIM_SCHED")

    logger.info("Start training temporal PointPillar")

    train_phase(
        phase_id=str(phase_id),
        extra_tag=str(args.extra_tag),
        model=model,
        train_loader=train_loader,
        cfg_obj=cfg,
        logger=logger,
        device=device,
        run_cfg=run_cfg,
        loop_cfg=loop_cfg,
        phase_cfg=phase_cfg,
        resume_blob=resume_blob,
    )


if __name__ == "__main__":
    main()
