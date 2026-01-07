import argparse
import os
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.optimizer import build_optimizer_trainable_only, build_warmup_cosine_scheduler , build_optimizer_with_prefix_multipliers, build_warmup_cosine_factor_scheduler

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from xmem_det.util import load_xmem_train_cfg

def set_trainable_prefixes(model, prefixes):
    prefixes = tuple(prefixes)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(prefixes)


def rel_T_curr_prev(T_world_lidar: np.ndarray, t: int) -> np.ndarray:
    T_prev = T_world_lidar[t - 1]
    T_curr = T_world_lidar[t]
    return (np.linalg.inv(T_curr) @ T_prev).astype(np.float32)


def to_torch_batch_dict(frame_dict, device):
    batch_dict = {}
    for k, v in frame_dict.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                batch_dict[k] = v
            else:
                tensor = torch.from_numpy(v)
                batch_dict[k] = tensor.to(device, non_blocking=True)
        elif isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device, non_blocking=True)
        else:
            batch_dict[k] = v
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict


def build_seq_loader(cfg, logger, workers, seq_len, stride):
    dataset_cfg = cfg.DATA_CONFIG
    class_names = cfg.CLASS_NAMES

    train_set = NuScenesSeqDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=True,
        root_path=None,
        logger=logger,
        seq_len=seq_len,
        stride=stride,
        nusc_version=dataset_cfg.VERSION,
        nusc_dataroot=dataset_cfg.DATA_PATH,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=min(workers, cfg.OPTIMIZATION.NUM_WORKERS),
        pin_memory=True,
        collate_fn=collate_seq,
        drop_last=False,
    )

    return train_set, train_loader

from collections import deque
def train_one_epoch(
    model,
    optimizer,
    scheduler,
    train_loader,
    epoch,
    total_epochs,
    logger,
    device,
    max_grad_norm,
    alpha_temporal: float,
    supervise_det: bool,
    supervise_aux: bool,
    tbptt_k: int = 8,
):
    model.train()

    from collections import deque
    w = 200
    w_loss = deque(maxlen=w)
    w_cls = deque(maxlen=w)
    w_loc = deque(maxlen=w)
    w_dir = deque(maxlen=w)
    w_aux = deque(maxlen=w)

    sum_loss = 0.0
    sum_cls = 0.0
    sum_loc = 0.0
    sum_dir = 0.0
    sum_aux = 0.0

    n_loss = 0
    n_cls = 0
    n_loc = 0
    n_dir = 0
    n_aux = 0

    def _to_float(x):
        return float(x.detach().item()) if hasattr(x, "detach") else float(x)

    def _avg(dq):
        return (sum(dq) / len(dq)) if len(dq) > 0 else None

    for seq_idx, seq in enumerate(train_loader):
        torch.cuda.reset_peak_memory_stats()

        frames = seq["frames"]
        T_world_lidar = seq["T_world_lidar"]
        T = len(frames)

        if T == 0:
            continue

        if hasattr(model, "reset_sequence"):
            model.reset_sequence(seq_idx)

        det_instance_masks_prev = None

        K = int(tbptt_k)
        burn = max(T - K, 0)

        for t in range(burn):
            frame = frames[t]
            batch_dict = to_torch_batch_dict(frame, device)

            if t == 0:
                T_rel = None
            else:
                T_rel_np = rel_T_curr_prev(T_world_lidar, t)
                T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

            with torch.no_grad():
                _, _, _, det_instance_masks_prev = model(
                    batch_dict,
                    t_seq=t,
                    det_instance_masks_prev=det_instance_masks_prev,
                    T_rel=T_rel,
                    alpha_temporal=alpha_temporal,
                    compute_det_loss=False,
                    compute_aux_loss=False,
                    keep_state_grad=False,
                )

            if isinstance(det_instance_masks_prev, torch.Tensor):
                det_instance_masks_prev = det_instance_masks_prev.detach()

        if hasattr(model, "detach_temporal_state"):
            model.detach_temporal_state()

        for t in range(burn, T - 1):
            frame = frames[t]
            batch_dict = to_torch_batch_dict(frame, device)

            if t == 0:
                T_rel = None
            else:
                T_rel_np = rel_T_curr_prev(T_world_lidar, t)
                T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

            _, _, _, det_instance_masks_prev = model(
                batch_dict,
                t_seq=t,
                det_instance_masks_prev=det_instance_masks_prev,
                T_rel=T_rel,
                alpha_temporal=alpha_temporal,
                compute_det_loss=False,
                compute_aux_loss=False,
                keep_state_grad=True,
            )

            if isinstance(det_instance_masks_prev, torch.Tensor):
                det_instance_masks_prev = det_instance_masks_prev.detach()

        t_last = T - 1
        frame = frames[t_last]
        batch_dict = to_torch_batch_dict(frame, device)

        if t_last == 0:
            T_rel = None
        else:
            T_rel_np = rel_T_curr_prev(T_world_lidar, t_last)
            T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

        ret_dict, tb_dict, disp_dict, det_masks_last = model(
            batch_dict,
            t_seq=t_last,
            det_instance_masks_prev=det_instance_masks_prev,
            T_rel=T_rel,
            alpha_temporal=alpha_temporal,
            compute_det_loss=bool(supervise_det),
            compute_aux_loss=bool(supervise_aux),
            keep_state_grad=True,
        )

        loss = ret_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        # print(torch.cuda.max_memory_allocated() / (1024**2))

        # def grad_norm(m):
        #     s = 0.0
        #     for p in m.parameters():
        #         if p.grad is not None:
        #             s += float(p.grad.detach().norm().item())
        #     return s

        # print("state_gate", grad_norm(model.state_gate))
        # print("state_cand", grad_norm(model.state_cand))
        # print("motion_tf", grad_norm(model.motion_transform_net))


        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_v = _to_float(loss)
        w_loss.append(loss_v)
        sum_loss += loss_v
        n_loss += 1

        if "rpn_loss_cls" in tb_dict:
            v = _to_float(tb_dict["rpn_loss_cls"])
            w_cls.append(v)
            sum_cls += v
            n_cls += 1

        if "rpn_loss_loc" in tb_dict:
            v = _to_float(tb_dict["rpn_loss_loc"])
            w_loc.append(v)
            sum_loc += v
            n_loc += 1

        if "rpn_loss_dir" in tb_dict:
            v = _to_float(tb_dict["rpn_loss_dir"])
            w_dir.append(v)
            sum_dir += v
            n_dir += 1

        if "aux_motion_tf" in tb_dict:
            v = _to_float(tb_dict["aux_motion_tf"])
            w_aux.append(v)
            sum_aux += v
            n_aux += 1

        if (seq_idx + 1) % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]

            win_loss = _avg(w_loss)
            win_cls = _avg(w_cls)
            win_loc = _avg(w_loc)
            win_dir = _avg(w_dir)
            win_aux = _avg(w_aux)

            loss_str = (
                f"epoch {epoch + 1}/{total_epochs}, seq {seq_idx + 1}/{len(train_loader)}, "
                f"loss {loss_v:.4f}, win200 {win_loss:.4f}, "
            )

            if "rpn_loss_cls" in tb_dict:
                loss_str += f"cls {_to_float(tb_dict['rpn_loss_cls']):.4f}, win200_cls {win_cls:.4f}, "
            if "rpn_loss_loc" in tb_dict:
                loss_str += f"loc {_to_float(tb_dict['rpn_loss_loc']):.4f}, win200_loc {win_loc:.4f}, "
            if "rpn_loss_dir" in tb_dict:
                loss_str += f"dir {_to_float(tb_dict['rpn_loss_dir']):.4f}, win200_dir {win_dir:.4f}, "
            if "aux_motion_tf" in tb_dict:
                loss_str += f"aux {_to_float(tb_dict['aux_motion_tf']):.4f}, win200_aux {win_aux:.4f}, "

            loss_str += f"lr {lr:.6e}"
            logger.info(loss_str)

        if (seq_idx + 1) % 2 == 0:
            keys = [
                "_dbg_alpha",
                "_dbg_hidden_prev_has_fn",
                "_dbg_hidden_to_bev",
                "_dbg_temporal_delta",
                "_dbg_bev_delta",
                "_dbg_gate_used_mean",
                "_dbg_gate_mean",
                "_dbg_z_mean",
            ]
            vals = []
            for k in keys:
                if k in tb_dict:
                    vals.append(f"{k}={float(tb_dict[k].detach().item()):.6f}")
            if len(vals) > 0:
                print("TEMPDBG", " ".join(vals))


    epoch_avg_loss = sum_loss / max(n_loss, 1)
    msg = f"epoch {epoch + 1}/{total_epochs} summary: avg_loss {epoch_avg_loss:.4f}"
    if n_cls > 0:
        msg += f", avg_cls {sum_cls / n_cls:.4f}"
    if n_loc > 0:
        msg += f", avg_loc {sum_loc / n_loc:.4f}"
    if n_dir > 0:
        msg += f", avg_dir {sum_dir / n_dir:.4f}"
    if n_aux > 0:
        msg += f", avg_aux {sum_aux / n_aux:.4f}"
    if len(w_loss) > 0:
        msg += f", win200_loss {_avg(w_loss):.4f}"
    if len(w_aux) > 0:
        msg += f", win200_aux {_avg(w_aux):.4f}"
    logger.info(msg)



def train_phase(
    args,
    model,
    train_loader,
    cfg,
    logger,
    device,
    prefixes,
    resume_blob=None,
    start_epoch=0,
    epochs=30,
    lr_start=2e-5,
    lr_max=1e-4,
    lr_end=1e-6,
    warmup_epochs=1,
    alpha_start=1,
    alpha_end=1.0,
    alpha_ramp_epochs=20,
):
    def alpha_ramp_epoch(epoch_idx):
        s = float(alpha_start)
        e = float(alpha_end)
        r = int(alpha_ramp_epochs)
        if r <= 0:
            return e
        if epoch_idx >= r:
            return e
        x = (epoch_idx + 1) / r
        return s + (e - s) * x


    set_trainable_prefixes(model, prefixes)

    # optimizer = build_optimizer_trainable_only(model, cfg, lr=lr_start)

    # steps_per_epoch = len(train_loader)

    # scheduler = build_warmup_cosine_scheduler(
    #     optimizer=optimizer,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     lr_start=lr_start,
    #     lr_max=lr_max,
    #     lr_end=lr_end,
    #     warmup_epochs=warmup_epochs,
    # )

    HEAD_LR_MULT = 0.2 
    TEMPORAL_LR_MULT = 1.0
    BACKBONE2D_LR_MULT = 0.05
    group_specs = [
        (("xmem", "motion_transform_net", "aux_head", "temporal_fusion", "state_gate", "state_cand"), TEMPORAL_LR_MULT),
        (("dense_head",), HEAD_LR_MULT),
        (("backbone_2d",), BACKBONE2D_LR_MULT),
    ]
    total_epochs = int(epochs)
    start_epoch = int(start_epoch)

    optimizer = build_optimizer_with_prefix_multipliers(
        model=model,
        cfg=cfg,
        base_lr=lr_max,
        group_specs=group_specs,
    )

    scheduler = build_warmup_cosine_factor_scheduler(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        epochs=(total_epochs - start_epoch),
        lr_start=lr_start,
        lr_max=lr_max,
        lr_end=lr_end,
        warmup_epochs=warmup_epochs,
    )


    # if resume_blob is not None:
    #     opt_state = resume_blob.get("optimizer_state", None)
    #     sch_state = resume_blob.get("scheduler_state", None)
    #     if opt_state is not None:
    #         optimizer.load_state_dict(opt_state)
    #     if sch_state is not None:
    #         scheduler.load_state_dict(sch_state)


    if start_epoch >= total_epochs:
        logger.info(f"Phase1: start_epoch={start_epoch} >= epochs={total_epochs}, nothing to do")
        return

    logger.info(
        f"Phase {args.phase} start: epochs={total_epochs}, steps_per_epoch={len(train_loader)}, "
        f"lr_start={float(lr_start):.3e}, lr_max={float(lr_max):.3e}, lr_end={float(lr_end):.3e}, warmup_epochs={int(warmup_epochs)}, "
        f"alpha_start={float(alpha_start):.3f}, alpha_end={float(alpha_end):.3f}, alpha_ramp_epochs={int(alpha_ramp_epochs)}"
    )

    for epoch in range(start_epoch, total_epochs):
        alpha = alpha_ramp_epoch(epoch)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}/{total_epochs} alpha={alpha:.3f} lr={lr:.3e}")

        train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
            logger=logger,
            device=device,
            max_grad_norm=10.0,
            alpha_temporal=alpha,
            supervise_det=True,
            supervise_aux=True,
        )

        ckpt_path = os.path.join("log", f"phase_ckpt_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "phase": "phase1",
                "phase1_cfg": {
                    "epochs": int(total_epochs),
                    "lr_start": float(lr_start),
                    "lr_max": float(lr_max),
                    "lr_end": float(lr_end),
                    "warmup_epochs": int(warmup_epochs),
                    "alpha_start": float(alpha_start),
                    "alpha_end": float(alpha_end),
                    "alpha_ramp_epochs": int(alpha_ramp_epochs),
                },
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
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=35.0)

    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_pp_ckpt", type=str, default=None)

    parser.add_argument('--phase', type=int, default=1, help='Training phase (1 or 2)')


    # SETUP 
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    xmem_train_cfg = load_xmem_train_cfg(args.xmem_cfg)

    os.makedirs("log", exist_ok=True)
    log_file = os.path.join("log", f"train_temporal_pp_{args.extra_tag}.txt")
    logger = common_utils.create_logger(log_file)

    logger.info(str(cfg))
    logger.info(f"seq_len={args.seq_len}, stride={args.stride}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, train_loader = build_seq_loader(
        cfg=cfg,
        logger=logger,
        workers=args.workers,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    num_class = len(cfg.CLASS_NAMES)
    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=num_class,
        dataset=train_set.base,
        xmem_train_cfg=xmem_train_cfg,
        pc_range=pc_range,
    )
    model.to(device)
    names = [n for n, _ in model.named_parameters()]
    tops = sorted({n.split(".")[0] for n in names})
    print(tops)

    resume_blob = None
    start_epoch = 0

    if args.resume_ckpt:

        resume_blob = torch.load(args.resume_ckpt, map_location="cpu")

        model.load_state_dict(resume_blob["model_state"], strict=True)
        start_epoch = int(resume_blob.get("epoch", 0))
    elif args.pretrained_pp_ckpt:
        ckpt = torch.load(args.pretrained_pp_ckpt, map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)


    total_epochs = int(cfg.OPTIMIZATION.NUM_EPOCHS)

    # Training loop with staged training
    logger.info("Start training temporal PointPillar with cyclical XMem gating")

    if args.phase == 1:
        PHASE1_PREFIXES = (
            "xmem",
            "hidden_to_bev",
            "dense_head",
        )

        train_phase(
            args=args,
            prefixes=PHASE1_PREFIXES,
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            logger=logger,
            device=device,
            resume_blob=resume_blob,
            start_epoch=start_epoch,
            epochs=6,
            lr_start=5e-5,
            lr_max=5e-4,
            lr_end=5e-6,
            warmup_epochs=1,
            alpha_start=0.0,
            alpha_end=1.0,
            alpha_ramp_epochs=6,
        )
        return

    if args.phase == 2:
        print("Starting Phase 2 Training")
        PHASE2_PREFIXES = (
            "dense_head",
            "xmem",
            "motion_transform_net",
            "aux_head",
            "temporal_fusion",
            "state_gate",
            "state_cand",
        )
        train_phase(
            args=args,
            prefixes=PHASE2_PREFIXES,
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            logger=logger,
            device=device,
            resume_blob=resume_blob,
            start_epoch=start_epoch,
            epochs=18,
            lr_start=5e-5,
            lr_max=1e-4,
            lr_end=5e-7,
            warmup_epochs=1,
            alpha_start=0.0,
            alpha_end=1.0,
            alpha_ramp_epochs=12
        )



if __name__ == "__main__":
    main()