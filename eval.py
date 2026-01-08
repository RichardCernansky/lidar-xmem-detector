import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.util import load_xmem_train_cfg

def deduce_alpha_from_ckpt_if_possible(blob: dict, fallback: float) -> float:
    return 0
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
    """
    OpenPCDet datasets often return a dict with numpy arrays.
    Your TemporalPointPillar forward expects tensors on GPU for numeric arrays,
    but should keep string/object arrays as-is.
    """
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

    """
    OpenPCDet code relies on batch_size existing (some modules do).
    Because we always collate with one sample here, we force batch_size=1 if missing.
    """
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict


def world_T_lidar_from_token(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Computes world_T_lidar for the LiDAR_TOP sensor at a given sample token.
    This is needed to compute the relative motion between consecutive frames (T_rel).
    """
    sample = nusc.get("sample", sample_token)
    sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd_lidar["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])

    world_T_ego = transform_matrix(ep["translation"], Quaternion(ep["rotation"]), inverse=False)
    ego_T_lidar = transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=False)
    return (world_T_ego @ ego_T_lidar).astype(np.float32)


def rel_T_curr_prev(T_world_prev: np.ndarray, T_world_curr: np.ndarray) -> np.ndarray:
    """
    Your model uses T_rel to motion-compensate masks/state between frames.
    We want a transform that maps points from prev-lidar frame into curr-lidar frame:
        T_rel = inv(T_world_curr) @ T_world_prev
    """
    return (np.linalg.inv(T_world_curr) @ T_world_prev).astype(np.float32)


def fmt_hms(seconds: float) -> str:
    """
    Pretty printing for progress logs.
    """
    seconds = max(0.0, float(seconds))
    s = int(seconds + 0.5)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def steps_for_scene(n: int, seq_len: int) -> int:
    """
    Option 1 evaluates *each* frame as the end of a rolling window (stride=1 by definition).
    For a scene with n frames and max context length L:

    - for early frames, window sizes grow: 1,2,3,...,L
    - then it stays L: L,L,L,... (n-L times)

    Total forward steps in that scene:
        sum_{k=1..min(n,L)} k  +  max(0, n-L) * L
    """
    n = int(n)
    L = int(seq_len)
    if n <= 0:
        return 0
    if n <= L:
        return n * (n + 1) // 2
    return (L * (L + 1) // 2) + (n - L) * L


def parse_args():
    p = argparse.ArgumentParser()

    """
    --cfg_file: OpenPCDet YAML, contains dataset config, class names, post-processing, etc.
    --xmem_cfg: your XMem train config (dims, checkpoint usage, etc.)
    --ckpt: the trained TemporalPointPillar checkpoint (your custom ckpt)
    """
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--xmem_cfg", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    """
    --workers: dataloader workers (only used for OpenPCDet dataset build; we do manual per-item access).
    --split: overrides cfg.DATA_CONFIG.DATA_SPLIT['test'] to ensure you really evaluate val/test as intended.
    --alpha: your temporal gate strength; alpha=1.0 means full temporal enhancement.
    """
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--alpha", type=float, default=1.0)

    """
    Option 1 parameter:
    --seq_len is the maximum context length used for each "window ending at current frame".
    Stride is implicitly 1 in option 1 (because every frame is evaluated with its own rolling context).
    """
    p.add_argument("--seq_len", type=int, default=8)

    """
    Output folder controls:
    --extra_tag matches OpenPCDet convention for experiment folder separation.
    --eval_tag creates a subfolder so you can store multiple eval variants side-by-side.
    --log_interval controls progress logging frequency.
    """
    p.add_argument("--extra_tag", type=str, default="default")
    p.add_argument("--eval_tag", type=str, default="opt1_window_ends_each_frame")
    p.add_argument("--log_interval", type=int, default=100)

    """
    --set allows overriding YAML keys from CLI, same as OpenPCDet scripts.
    Example: --set DATA_CONFIG.DATA_SPLIT.test val
    """
    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)

    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    """
    OpenPCDet uses cfg.DATA_CONFIG.DATA_SPLIT['test'] to decide which split annotations to load.
    If you want val metrics, force it here.
    """
    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    return args


def main():
    args = parse_args()

    """
    Some OpenPCDet setups define cfg.ROOT_DIR (usually repository root).
    If it is missing, we fall back to current working directory so path creation still works.
    """
    root_dir = getattr(cfg, "ROOT_DIR", Path.cwd())

    """
    Output layout matches OpenPCDet:
      output/<EXP_GROUP_PATH>/<TAG>/<extra_tag>/eval_temporal/<eval_tag>/
    """
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

    """
    We build the OpenPCDet NuScenesDataset in eval mode (training=False).
    This gives us:
      - test_set.infos (tokens, timestamps, etc.)
      - test_set.__getitem__ and test_set.collate_batch
      - test_set.generate_prediction_dicts and test_set.evaluation
    """
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    """
    We need NuScenes API access only to group frames by scene and compute transforms (T_world_lidar).
    """
    dataroot_for_nusc = Path(cfg.DATA_CONFIG.DATA_PATH) / cfg.DATA_CONFIG.VERSION
    nusc = NuScenes(version=cfg.DATA_CONFIG.VERSION, dataroot=str(dataroot_for_nusc), verbose=False)

    """
    Group dataset indices by scene_token, and sort each scene by timestamp.
    This provides the temporal order we want for feeding sequences.
    """
    by_scene = {}
    for i, info in enumerate(test_set.infos):
        tok = info.get("token", None)
        if tok is None:
            raise KeyError("token missing in infos")
        scene_token = nusc.get("sample", tok)["scene_token"]
        by_scene.setdefault(scene_token, []).append(i)

    for scene_token, idxs in by_scene.items():
        idxs.sort(key=lambda j: test_set.infos[j]["timestamp"])

    """
    Precompute totals for progress logging:
    - total_samples = number of frames in the split (e.g. 6019 for val in trainval)
    - total_steps = number of model forward steps (larger, because option 1 replays context windows)
    """
    total_samples = len(test_set)
    total_steps = 0
    for _, idxs in by_scene.items():
        total_steps += steps_for_scene(len(idxs), int(args.seq_len))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Build the temporal model.
    NOTE: Your XMem wrapper prints messages at construction time (NOT RESUMING XMEM etc.).
    That is normal, and afterwards we load the full trained state_dict from --ckpt.
    """
    xmem_train_cfg = load_xmem_train_cfg(args.xmem_cfg)
    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set,
        xmem_train_cfg=xmem_train_cfg,
        pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
    ).to(device)

    """
    Load your trained checkpoint.
    Your training saves {"model_state": ...}, so we handle both that and raw state_dict.
    """
    blob = torch.load(args.ckpt, map_location="cpu")

    alpha_used = deduce_alpha_from_ckpt_if_possible(
        blob if isinstance(blob, dict) else {},
        float(args.alpha),
    )
    logger.info(f"alpha={alpha_used}")

    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state, strict=True)
    model.eval()


    """
    We must produce a prediction for every dataset index (every frame).
    det_annos[i] will hold the OpenPCDet-formatted annotation dict for frame i.
    """
    det_annos = [None] * total_samples

    done_samples = 0
    done_steps = 0
    start_t = time.time()

    """
    OPTION 1 EVAL LOGIC (rolling window ending at each frame):
      For each scene:
        For each frame position pos:
          window = last up to seq_len frames ending at pos
          reset temporal state
          replay the window frames in order with t_seq=0..len(window)-1
          take the prediction of the LAST frame in that window
          store it for the dataset index corresponding to that last frame
    This means: every sample gets evaluated once, but with temporal context.
    """
    with torch.no_grad():
        win_id = 0
        for scene_token, idxs in by_scene.items():
            n = len(idxs)

            for pos in range(n):
                cur_idx = idxs[pos]

                w_start = max(0, pos - int(args.seq_len) + 1)
                window = idxs[w_start:pos + 1]

                """
                Very important:
                We reset sequence state for each rolling window so evaluation is deterministic
                and reflects "using only the past context up to seq_len".
                """
                if hasattr(model, "reset_sequence"):
                    model.reset_sequence(win_id)

                det_instance_masks_prev = None
                T_world_prev = None

                last_batch_cpu = None
                last_pred_dicts = None

                for step, idx in enumerate(window):
                    info = test_set.infos[idx]
                    sample_token = info["token"]

                    """
                    Build T_rel from previous to current lidar pose (for motion compensation).
                    First step has no previous.
                    """
                    T_world_curr = world_T_lidar_from_token(nusc, sample_token)
                    if T_world_prev is None:
                        T_rel = None
                    else:
                        T_rel_np = rel_T_curr_prev(T_world_prev, T_world_curr)
                        T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

                    """
                    Fetch OpenPCDet frame item and collate into a batch dict (CPU),
                    then move numeric tensors to GPU.
                    """
                    item = test_set.__getitem__(idx)
                    batch_cpu = test_set.collate_batch([item])
                    batch_gpu = to_torch_batch_dict(batch_cpu, device)

                    """
                    Forward temporal model:
                      - t_seq is the step inside the window
                      - we pass det_instance_masks_prev and T_rel so your temporal path works
                      - compute_det_loss/aux_loss disabled for eval
                    The model returns OpenPCDet-style pred_dicts.
                    """
                    pred_dicts, _, det_masks_next = model(
                        batch_gpu,
                        t_seq=int(step),
                        det_instance_masks_prev=det_instance_masks_prev,
                        T_rel=T_rel,
                       alpha_temporal=float(alpha_used),
                        compute_det_loss=False,
                        compute_aux_loss=False,
                    )
                    
                    # pd = pred_dicts[0]
                    # num = int(pd["pred_boxes"].shape[0])
                    # mx = float(pd["pred_scores"].max().item()) if num > 0 else 0.0
                    # if done_samples % int(args.log_interval) == 0:
                    #     logger.info(f"sample idx={idx} step={step} num_boxes={num} max_score={mx:.4f}")


                    """
                    Track the last frame’s prediction, because option 1 evaluates only the last timestep.
                    """
                    last_batch_cpu = batch_cpu
                    last_pred_dicts = pred_dicts

                    """
                    Update det_instance_masks_prev for next timestep in the window.
                    """
                    if isinstance(det_masks_next, torch.Tensor):
                        det_instance_masks_prev = det_masks_next.detach()
                    else:
                        det_instance_masks_prev = det_masks_next

                    T_world_prev = T_world_curr
                    done_steps += 1

                """
                Convert last frame pred_dicts into nuScenes-format dict expected by OpenPCDet evaluation.
                """
                annos = test_set.generate_prediction_dicts(
                    batch_dict=last_batch_cpu,
                    pred_dicts=last_pred_dicts,
                    class_names=cfg.CLASS_NAMES,
                    output_path=None,
                )
                det_annos[cur_idx] = annos[0]

                done_samples += 1
                win_id += 1

                """
                Progress logging:
                  - sample progress: how many frames already have final predictions
                  - step progress: how many forward steps executed (useful because option 1 is heavier)
                """
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

    """
    Safety check: ensure every dataset frame got a prediction.
    """
    missing = [i for i, a in enumerate(det_annos) if a is None]
    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} samples, first missing index={missing[0]}")

    """
    IMPORTANT nuScenes eval edge case:
    The nuScenes evaluation code (filter_eval_boxes) assumes the first sample in the results dict
    has at least one predicted box; if the first sample token has an empty list, it can crash with:
      IndexError: list index out of range

    A practical workaround is to rotate det_annos so the first entry has at least one prediction.
    This does NOT change the set of predictions; it only changes the serialization order.
    """
    first_nonempty = -1
    for i, a in enumerate(det_annos):
        names = a.get("name", [])
        if hasattr(names, "__len__") and len(names) > 0:
            first_nonempty = i
            break

    det_annos_eval = det_annos
    if first_nonempty > 0:
        det_annos_eval = [det_annos[first_nonempty]] + det_annos[:first_nonempty] + det_annos[first_nonempty + 1 :]

    """
    Run OpenPCDet’s official evaluation for NuScenes.
    This will:
      - dump predictions to results_nusc.json under eval_output_dir
      - run nuScenes detection evaluation
      - print NDS/mAP and per-class metrics
    """
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
