import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.util import load_xmem_train_cfg


def deduce_alpha_from_ckpt_if_possible(blob: dict, fallback: float) -> float:
    if not isinstance(blob, dict):
        return float(fallback)

    epoch_1based = int(blob.get("epoch", 0))
    phase1_cfg = blob.get("phase1_cfg", None)

    if epoch_1based <= 0 or not isinstance(phase1_cfg, dict):
        return float(fallback)

    s = float(phase1_cfg.get("alpha_start", fallback))
    e = float(phase1_cfg.get("alpha_end", fallback))
    r = int(phase1_cfg.get("alpha_ramp_epochs", 0))

    if r <= 0:
        return float(e)

    epoch_idx = max(epoch_1based - 1, 0)
    if epoch_idx >= r:
        return float(e)

    x = (epoch_idx + 1) / r
    return float(s + (e - s) * x)


def to_torch_batch_dict(frame_dict, device):
    # Converts OpenPCDet batch dict values into torch tensors on GPU when appropriate.
    # Keeps strings/objects as-is (OpenPCDet stores tokens, frame ids, etc. as object arrays).
    batch_dict = {}
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

    # OpenPCDet modules assume batch_size exists in batch_dict. We run batch_size=1 always.
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict


def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(seconds + 0.5)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def steps_for_scene(n: int, seq_len: int) -> int:
    # Counts how many forward steps total the rolling-window evaluation will do for progress/ETA.
    # For each frame position pos in the scene, we run a window of length <= seq_len ending at pos.
    # Total steps is sum of window lengths across all pos.
    n = int(n)
    L = int(seq_len)
    if n <= 0:
        return 0
    if n <= L:
        return n * (n + 1) // 2
    return (L * (L + 1) // 2) + (n - L) * L


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--xmem_cfg", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--seq_len", type=int, default=8)

    p.add_argument("--extra_tag", type=str, default="default")
    p.add_argument("--eval_tag", type=str, default="opt1_window_ends_each_frame")
    p.add_argument("--log_interval", type=int, default=100)

    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Loads the OpenPCDet YAML config into global cfg.
    cfg_from_yaml_file(args.cfg_file, cfg)

    # Optional: allow overriding cfg keys from CLI (OpenPCDet convention).
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # This is important: OpenPCDet uses DATA_SPLIT['test'] when building evaluation dataset.
    # Setting --split ensures you evaluate exactly val or test as intended.
    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    return args


def main():
    args = parse_args()

    root_dir = getattr(cfg, "ROOT_DIR", Path.cwd())

    # OpenPCDet-style output folder; eval results go into eval_temporal/<eval_tag>/.
    output_dir = Path(root_dir) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    eval_output_dir = output_dir / "eval_temporal" / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ("log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info("**********************Start logging**********************")
    logger.info(f"cfg_file={args.cfg_file}")
    logger.info(f"ckpt={args.ckpt}")
    logger.info(f"split={cfg.DATA_CONFIG.DATA_SPLIT['test']}")
    logger.info(f"seq_len={args.seq_len}")
    log_config_to_file(cfg, logger=logger)

    # Build NuScenesDataset in evaluation mode.
    # Even though build_dataloader returns a DataLoader, we will not iterate it;
    # we use test_set.__getitem__ directly so we can control temporal grouping and rolling windows.
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    # NuScenes API is used ONLY to map each sample token -> scene_token,
    # so we can evaluate sequences in correct temporal order.
    dataroot_for_nusc = Path(cfg.DATA_CONFIG.DATA_PATH) / cfg.DATA_CONFIG.VERSION
    nusc = NuScenes(version=cfg.DATA_CONFIG.VERSION, dataroot=str(dataroot_for_nusc), verbose=False)

    # Group dataset indices by scene, and sort by timestamp so window order matches time.
    by_scene = {}
    for i, info in enumerate(test_set.infos):
        tok = info.get("token", None)
        if tok is None:
            raise KeyError("token missing in infos")
        scene_token = nusc.get("sample", tok)["scene_token"]
        by_scene.setdefault(scene_token, []).append(i)

    for scene_token, idxs in by_scene.items():
        idxs.sort(key=lambda j: test_set.infos[j]["timestamp"])

    total_samples = len(test_set)

    # Total forward steps is larger than total_samples because rolling window replays past frames.
    total_steps = 0
    for _, idxs in by_scene.items():
        total_steps += steps_for_scene(len(idxs), int(args.seq_len))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build your temporal detector. It contains XMem + fusion logic internally.
    xmem_train_cfg = load_xmem_train_cfg(args.xmem_cfg)
    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set,
        xmem_train_cfg=xmem_train_cfg,
        pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
    ).to(device)

    # Load your checkpoint. If training saved {"model_state": ...}, we load that.
    blob = torch.load(args.ckpt, map_location="cpu")
    alpha_used = deduce_alpha_from_ckpt_if_possible(blob if isinstance(blob, dict) else {}, float(args.alpha))
    logger.info(f"alpha={alpha_used}")

    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state, strict=True)
    model.eval()

    # We must generate ONE prediction per dataset frame.
    # det_annos[i] is the OpenPCDet-format dict for sample i, which OpenPCDet will serialize to NuScenes JSON.
    det_annos = [None] * total_samples

    done_samples = 0
    done_steps = 0
    start_t = time.time()

    # Rolling-window evaluation:
    # For each frame position 'pos' in a scene:
    #   window = last up to seq_len frames ending at pos
    #   run temporal model across that window
    #   store prediction for the LAST frame only (the frame at pos)
    with torch.no_grad():
        win_id = 0
        for scene_token, idxs in by_scene.items():
            n = len(idxs)

            for pos in range(n):
                cur_idx = idxs[pos]

                # Build the window ending at current frame.
                w_start = max(0, pos - int(args.seq_len) + 1)
                window = idxs[w_start:pos + 1]

                # Reset temporal state for determinism: each evaluated frame uses only its own past context window.
                # win_id is just a unique identifier; your model may ignore it.
                model.reset_sequence(win_id)

                frames_gpu = []
                last_batch_cpu = None

                # Collect all frames in the window as OpenPCDet batch_dicts on GPU.
                # We keep last_batch_cpu because OpenPCDet generate_prediction_dicts expects CPU batch_dict
                # corresponding to the frame that pred_dicts refer to (the LAST frame in the window).
                for idx in window:
                    item = test_set.__getitem__(idx)
                    batch_cpu = test_set.collate_batch([item])
                    batch_gpu = to_torch_batch_dict(batch_cpu, device)
                    frames_gpu.append(batch_gpu)
                    last_batch_cpu = batch_cpu

                # This is the key call:
                # forward_eval() should run your XMem inference core across frames_gpu in time order,
                # extract g4 from segmentation for the last timestep, fuse into BEV, then run dense_head.
                # Returned pred_dicts must be OpenPCDet-style predictions for the last frame only.
                pred_dicts, recall_dicts, det_masks_next = model.forward_eval(
                    frames_gpu,
                    alpha_temporal=float(alpha_used),
                    use_det_t0=True,
                )

                # Convert OpenPCDet pred_dicts to NuScenes JSON-compatible format.
                # This produces a list (batch size = 1), so we take [0].
                annos = test_set.generate_prediction_dicts(
                    batch_dict=last_batch_cpu,
                    pred_dicts=pred_dicts,
                    class_names=cfg.CLASS_NAMES,
                    output_path=None,
                )
                det_annos[cur_idx] = annos[0]

                done_samples += 1
                done_steps += len(window)
                win_id += 1

                # Progress logging uses done_steps because that is the true workload for rolling windows.
                if done_samples % int(args.log_interval) == 0:
                    now = time.time()
                    elapsed = now - start_t
                    rate = done_steps / max(elapsed, 1e-9)
                    pct_s = 100.0 * done_samples / max(total_samples, 1)
                    pct_t = 100.0 * done_steps / max(total_steps, 1)
                    eta = (total_steps - done_steps) / max(rate, 1e-9)
                    logger.info(
                        f"Eval progress: samples {pct_s:6.2f}% ({done_samples}/{total_samples}) "
                        f"steps {pct_t:6.2f}% ({done_steps}/{total_steps}) "
                        f"elapsed={fmt_hms(elapsed)} eta={fmt_hms(eta)} rate={rate:.2f} it/s"
                    )

    # Safety: ensure every frame got a prediction.
    missing = [i for i, a in enumerate(det_annos) if a is None]
    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} samples, first missing index={missing[0]}")

    # NuScenes evaluation sometimes crashes if the FIRST sample has zero predicted boxes.
    # Workaround: rotate list so first entry is non-empty. This does not change the content, only order.
    first_nonempty = -1
    for i, a in enumerate(det_annos):
        names = a.get("name", [])
        if hasattr(names, "__len__") and len(names) > 0:
            first_nonempty = i
            break

    det_annos_eval = det_annos
    if first_nonempty > 0:
        det_annos_eval = [det_annos[first_nonempty]] + det_annos[:first_nonempty] + det_annos[first_nonempty + 1 :]

    # OpenPCDet runs official NuScenes detection eval here:
    # - dumps results JSON
    # - runs NuScenes evaluator
    # - returns metric string + dict (NDS, mAP, per-class metrics)
    eval_metric = getattr(cfg.MODEL.POST_PROCESSING, "EVAL_METRIC", "nuscenes")
    result_str, result_dict = test_set.evaluation(
        det_annos=det_annos_eval,
        class_names=cfg.CLASS_NAMES,
        eval_metric=eval_metric,
        output_path=str(eval_output_dir),
    )

    logger.info(result_str)
    logger.info(str(result_dict))
    print(result_str)


if __name__ == "__main__":
    main()
