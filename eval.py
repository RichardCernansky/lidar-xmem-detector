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
from xmem_det.lstm_vis import TemporalDebugger  # NEW!


def to_torch_batch_dict(frame_dict, device):
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


def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(seconds + 0.5)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def steps_for_scene(n: int, seq_len: int) -> int:
    n = int(n)
    L = int(seq_len)
    if n <= 0:
        return 0
    if n <= L:
        return n * (n + 1) // 2
    return (L * (L + 1) // 2) + (n - L) * L


def infer_nusc_dataroot(data_path: str, version: str) -> str:
    p = Path(str(data_path))
    v = str(version)
    if (p / v).exists():
        return str(p)
    if p.name == v and (p.parent / v).exists():
        return str(p.parent)
    if (p / "samples").exists() or (p / "sweeps").exists():
        return str(p)
    return str(p)


def token_to_scene_token(nusc: NuScenes, tok: str) -> str:
    try:
        return nusc.get("sample", tok)["scene_token"]
    except Exception:
        sd = nusc.get("sample_data", tok)
        sample_tok = sd["sample_token"]
        return nusc.get("sample", sample_tok)["scene_token"]





# ... [keep all your existing helper functions] ...


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--extra_tag", type=str, default="default")
    p.add_argument("--eval_tag", type=str, default="temporal_eval")
    p.add_argument("--log_interval", type=int, default=50)
    
    # NEW: Visualization options
    p.add_argument("--visualize", action="store_true", help="Enable visualization")
    p.add_argument("--vis_scenes", type=int, default=3, help="Number of scenes to visualize")
    p.add_argument("--vis_interval", type=int, default=1, help="Visualize every N frames")
    
    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    if args.seq_len is None:
        args.seq_len = int(getattr(cfg.TRAIN, "SEQ_LEN", 4))

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    return args


def main():
    args = parse_args()

    root_dir = getattr(cfg, "ROOT_DIR", Path.cwd())

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
    logger.info(f"visualize={args.visualize}")
    if args.visualize:
        logger.info(f"vis_scenes={args.vis_scenes}, vis_interval={args.vis_interval}")
    log_config_to_file(cfg, logger=logger)

    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    dataroot_for_nusc = Path(cfg.DATA_CONFIG.DATA_PATH) / cfg.DATA_CONFIG.VERSION
    nusc = NuScenes(version=cfg.DATA_CONFIG.VERSION, dataroot=str(dataroot_for_nusc), verbose=False)

    by_scene = {}
    for i, info in enumerate(test_set.infos):
        tok = info.get("token", None)
        if tok is None:
            raise KeyError("token missing in infos")
        scene_token = token_to_scene_token(nusc, tok)
        by_scene.setdefault(scene_token, []).append(i)

    for scene_token, idxs in by_scene.items():
        idxs.sort(key=lambda j: test_set.infos[j]["timestamp"])

    total_samples = len(test_set)

    total_steps = 0
    for _, idxs in by_scene.items():
        total_steps += steps_for_scene(len(idxs), int(args.seq_len))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set,
        pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
    ).to(device)

    blob = torch.load(args.ckpt, map_location="cpu")
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    logger.info("Model loaded and set to eval mode")

    # NEW: Setup visualization
    if args.visualize:
        vis_dir = "visualizations"
        debugger = TemporalDebugger(
            save_dir=str(vis_dir),
            log_every=args.vis_interval,
            max_batches=1
        )
        model.debugger = debugger  # Attach to model
        logger.info(f"Visualization enabled, saving to {vis_dir}")
    else:
        debugger = None

    det_annos = [None] * total_samples

    done_samples = 0
    done_steps = 0
    start_t = time.time()

    with torch.no_grad():
        win_id = 0
        scene_items = list(by_scene.items())
        for scene_idx, (scene_token, idxs) in enumerate(scene_items):
            n = len(idxs)
            logger.info(f"Processing scene {scene_idx+1}/{len(scene_items)}: {scene_token} ({n} frames)")

            # NEW: Start visualization for this scene
            visualize_this_scene = args.visualize and scene_idx < args.vis_scenes
            if visualize_this_scene:
                debugger.start_sequence(f"scene_{scene_idx:03d}_{scene_token[:8]}")
                logger.info(f"  → Visualizing this scene")

            for pos in range(n):
                cur_idx = idxs[pos]
                w_start = max(0, pos - int(args.seq_len) + 1)
                window = idxs[w_start:pos + 1]

                if hasattr(model, "reset_sequence"):
                    model.reset_sequence(win_id)

                frames_list = []
                for idx in window:
                    item = test_set.__getitem__(idx)
                    batch_cpu = test_set.collate_batch([item])
                    batch_gpu = to_torch_batch_dict(batch_cpu, device)
                    frames_list.append(batch_gpu)

                last_item = test_set.__getitem__(cur_idx)
                last_batch_cpu = test_set.collate_batch([last_item])

                # Run model (debugger logs internally if attached)
                pred_dicts, recall_dicts, debug_info = model(
                    frames_list=frames_list,
                    compute_det_loss=False,
                    return_debug_info=False  # Debug info logged internally via debugger
                )

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

                if done_samples % int(args.log_interval) == 0:
                    now = time.time()
                    elapsed = now - start_t
                    rate = done_steps / max(elapsed, 1e-9)
                    pct_s = 100.0 * done_samples / max(total_samples, 1)
                    pct_t = 100.0 * done_steps / max(total_steps, 1)
                    eta = (total_steps - done_steps) / max(rate, 1e-9)
                    logger.info(
                        f"Eval: samples {pct_s:6.2f}% ({done_samples}/{total_samples}) "
                        f"steps {pct_t:6.2f}% ({done_steps}/{total_steps}) "
                        f"elapsed={fmt_hms(elapsed)} eta={fmt_hms(eta)} rate={rate:.2f} it/s"
                    )

            # NEW: Finish visualization for this scene
            if visualize_this_scene:
                debugger.finish_sequence()

    # ... [rest of your evaluation code stays exactly the same] ...

    missing_idxs = [i for i, a in enumerate(det_annos) if a is None]
    if missing_idxs:
        raise RuntimeError(f"Missing predictions for {len(missing_idxs)} samples, first missing index={missing_idxs[0]}")

    logger.info("All predictions generated successfully")

    first_nonempty = -1
    for i, a in enumerate(det_annos):
        names = a.get("name", [])
        if hasattr(names, "__len__") and len(names) > 0:
            first_nonempty = i
            break

    det_annos_eval = det_annos
    if first_nonempty > 0:
        logger.info(f"Rotating predictions list (first non-empty at index {first_nonempty})")
        det_annos_eval = [det_annos[first_nonempty]] + det_annos[:first_nonempty] + det_annos[first_nonempty + 1:]

    logger.info("Running evaluation...")
    eval_metric = getattr(cfg.MODEL.POST_PROCESSING, "EVAL_METRIC", "nuscenes")
    result_str, result_dict = test_set.evaluation(
        det_annos=det_annos_eval,
        class_names=cfg.CLASS_NAMES,
        eval_metric=eval_metric,
        output_path=str(eval_output_dir),
    )

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(result_str)
    logger.info("=" * 80)
    logger.info(f"Results dict: {result_dict}")
    logger.info("=" * 80 + "\n")

    print("\n" + result_str)


if __name__ == "__main__":
    main()