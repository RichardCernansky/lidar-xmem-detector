"""
Evaluation script for TemporalPointPillar.
"""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils

from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp_GRU import TemporalPointPillar


def to_torch_batch_dict(frame_dict: dict, device: torch.device) -> dict:
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
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file",     type=str, required=True)
    p.add_argument("--ckpt",         type=str, required=True)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--split",        type=str, default=None)
    p.add_argument("--extra_tag",    type=str, default="default")
    p.add_argument("--eval_tag",     type=str, default="sweep_eval")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--random_window", action="store_true",
                   help="Diagnostic: per-sample, replicate training's exact window-length "
                        "sampling (num_warmup = randint(max(1, T-3), T), a uniformly random "
                        "4-6 frame window for T=6) instead of always feeding the full window. "
                        "Default: no truncation (full window, matching normal eval).")
    p.add_argument("--window_len", type=int, default=None,
                   help="Diagnostic: truncate every sequence to a fixed last N frames "
                        "(keyframe + N-1 preceding sweeps), e.g. --window_len 3 to match "
                        "the paper's Table 1 'Scans: 3' reporting convention. Mutually "
                        "exclusive with --random_window.")
    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()

    if args.window_len is not None and args.random_window:
        p.error("--window_len and --random_window are mutually exclusive")

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    cfg.TAG            = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])
    return args


def main():
    args = parse_args()

    root_dir        = getattr(cfg, "ROOT_DIR", Path.cwd())
    output_dir      = Path(root_dir) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    eval_output_dir = output_dir / "eval_sweep" / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ("log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger   = common_utils.create_logger(log_file, rank=0)

    logger.info("**********************Start logging**********************")
    logger.info(f"cfg_file={args.cfg_file}")
    logger.info(f"ckpt={args.ckpt}")
    logger.info(f"split={cfg.DATA_CONFIG.DATA_SPLIT['test']}")
    logger.info(f"random_window={args.random_window}")
    logger.info(f"window_len={args.window_len if args.window_len is not None else 'full (no truncation)'}")
    log_config_to_file(cfg, logger=logger)

    # ------------------------------------------------------------------
    # Dataset  (keyframe_stride=1 is critical: sequence_indices[i] == i == kf_idx)
    # ------------------------------------------------------------------
    seq_len      = int(getattr(cfg.TRAIN, "SEQ_LEN", 4))
    sweep_stride = int(getattr(cfg.DATA_CONFIG, "SWEEP_STRIDE", 2))

    test_set = NuScenesSeqDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger,
        seq_len=seq_len,
        keyframe_stride=1,
        nusc_version=cfg.DATA_CONFIG.VERSION,
        nusc_dataroot=cfg.DATA_CONFIG.DATA_PATH,
        root_path=None,
        sweep_stride=sweep_stride,
    )

    assert len(test_set) == len(test_set.base), (
        f"Dataset length mismatch: seq={len(test_set)} base={len(test_set.base)}. "
        f"keyframe_stride must be 1 for eval."
    )
    logger.info(f"Dataset: {len(test_set)} sequences, seq_len={seq_len}, sweep_stride={sweep_stride}")

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_seq,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set.base,
        pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
    ).to(device)

    blob  = torch.load(args.ckpt, map_location="cpu")
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.eval()

    logger.info("Model loaded and set to eval mode.")

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    det_annos = [None] * len(test_set)
    start_t   = time.time()

    with torch.no_grad():
        for sample_idx, seq in enumerate(test_loader):

            if hasattr(model, "reset_sequence"):
                model.reset_sequence(sample_idx)

            frames      = seq["frames"]
            frames_list = [to_torch_batch_dict(f, device) for f in frames]
            if args.random_window:
                T = len(frames_list)
                if T > 1:
                    num_warmup = int(np.random.randint(max(1, T - 3), T))
                    frames_list = frames_list[-(num_warmup + 1):]
            elif args.window_len is not None:
                frames_list = frames_list[-int(args.window_len):]

            pred_dicts, recall_dicts, dbg = model(
                frames_list=frames_list,
                compute_det_loss=False,
            )

       

            base_item      = test_set.base.__getitem__(sample_idx)
            base_batch_cpu = test_set.base.collate_batch([base_item])


            annos = test_set.base.generate_prediction_dicts(
                batch_dict=base_batch_cpu,
                pred_dicts=pred_dicts,
                class_names=cfg.CLASS_NAMES,
                output_path=None,
            )
        

            # In eval.py, after generating annos, add:
            if sample_idx == 0:
                import json
                a = annos[0]
                print("ANNO KEYS:", list(a.keys()))
                print("sample_token:", a.get('metadata', {}).get('token', 'MISSING'))
                print("n_boxes:", len(a.get('boxes_lidar', [])))
                print("first 3 boxes_lidar:\n", a.get('boxes_lidar', [])[:3])
                print("first 3 names:", a.get('name', [])[:3])
                print("first 3 scores:", a.get('score', [])[:3])
                # Save to json for comparison
                with open("anno_sample0.json", "w") as f:
                    json.dump({k: v.tolist() if hasattr(v, 'tolist') else v 
                            for k, v in a.items()}, f, indent=2)


            det_annos[sample_idx] = annos[0]

            if (sample_idx + 1) % int(args.log_interval) == 0:
                elapsed = time.time() - start_t
                pct     = 100.0 * (sample_idx + 1) / len(test_set)
                eta     = elapsed / (sample_idx + 1) * (len(test_set) - sample_idx - 1)
                logger.info(
                    f"Eval: {sample_idx + 1}/{len(test_set)} ({pct:.1f}%)  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                )

    missing_idxs = [i for i, a in enumerate(det_annos) if a is None]
    if missing_idxs:
        raise RuntimeError(
            f"Missing predictions for {len(missing_idxs)} samples. "
            f"First missing: idx={missing_idxs[0]}"
        )

    logger.info("All predictions generated.")

    eval_metric = getattr(cfg.MODEL.POST_PROCESSING, "EVAL_METRIC", "nuscenes")
    result_str, result_dict = test_set.base.evaluation(
        det_annos=det_annos,
        class_names=cfg.CLASS_NAMES,
        eval_metric=eval_metric,
        output_path=str(eval_output_dir),
    )

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(result_str)
    logger.info(f"Results dict: {result_dict}")
    logger.info("=" * 80 + "\n")
    print("\n" + result_str)


if __name__ == "__main__":
    main()
