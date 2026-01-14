import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.util import load_xmem_train_cfg


def to_torch_batch_dict(frame_dict, device):
    out = {}
    for k, v in frame_dict.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                out[k] = v
            else:
                out[k] = torch.from_numpy(v).to(device, non_blocking=True)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    if "batch_size" not in out:
        out["batch_size"] = 1
    return out


def pad16_hw(h, w):
    h16 = ((int(h) + 15) // 16) * 16
    w16 = ((int(w) + 15) // 16) * 16
    return h16, w16


def occ_prob_from_xmem_masks(masks):
    if masks.dim() != 4:
        raise ValueError(f"Expected masks with shape [B,K,H,W], got {tuple(masks.shape)}")
    if masks.size(1) > 1:
        return masks[:, 1:].max(dim=1, keepdim=True)[0]
    return masks[:, 0:1]


def iou_binary(pred_bool, gt_bool):
    pred_bool = pred_bool.bool()
    gt_bool = gt_bool.bool()
    inter = (pred_bool & gt_bool).sum(dim=(1, 2, 3)).to(torch.float32)
    union = (pred_bool | gt_bool).sum(dim=(1, 2, 3)).to(torch.float32)
    ones = torch.ones_like(union)
    return torch.where(union > 0.0, inter / union, ones)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--xmem_cfg", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default=None)

    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--keep_last_partial", action="store_true")

    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--include_first", action="store_true")

    p.add_argument("--extra_tag", type=str, default="default")
    p.add_argument("--eval_tag", type=str, default="xmem_miou")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)

    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    args.seq_len = int(max(1, args.seq_len))
    args.stride = int(max(1, args.stride))
    args.thr = float(args.thr)

    return args


def main():
    args = parse_args()

    root_dir = getattr(cfg, "ROOT_DIR", Path.cwd())
    output_dir = Path(root_dir) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    eval_output_dir = output_dir / "eval_xmem" / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ("log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info("**********************Start logging**********************")
    logger.info(f"cfg_file={args.cfg_file}")
    logger.info(f"xmem_cfg={args.xmem_cfg}")
    logger.info(f"ckpt={args.ckpt}")
    logger.info(f"split={cfg.DATA_CONFIG.DATA_SPLIT['test']}")
    logger.info(f"seq_len={args.seq_len} stride={args.stride} keep_last_partial={bool(args.keep_last_partial)}")
    logger.info(f"thr={args.thr} include_first={bool(args.include_first)}")
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
        scene_token = nusc.get("sample", tok)["scene_token"]
        by_scene.setdefault(scene_token, []).append(i)

    for scene_token, idxs in by_scene.items():
        idxs.sort(key=lambda j: test_set.infos[j]["timestamp"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xmem_train_cfg = load_xmem_train_cfg(args.xmem_cfg)
    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set,
        xmem_train_cfg=xmem_train_cfg,
        pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
    ).to(device)

    blob = torch.load(args.ckpt, map_location="cpu")
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state, strict=True)
    model.eval()

    sum_iou_frames = 0.0
    cnt_iou_frames = 0

    sum_iou_seq_mean = 0.0
    cnt_iou_seqs = 0

    sum_iou_t3 = 0.0
    cnt_iou_t3 = 0

    sum_iou_last = 0.0
    cnt_iou_last = 0

    total_seqs = 0
    start_t = time.time()

    with torch.no_grad():
        seq_id = 0
        for scene_token, idxs in by_scene.items():
            cache = {}

            starts = list(range(0, len(idxs), args.stride))
            if not args.keep_last_partial:
                starts = [s for s in starts if s + args.seq_len <= len(idxs)]

            for s in starts:
                window = idxs[s : s + args.seq_len]
                if len(window) < 2:
                    continue

                def get_batch_cpu(idx):
                    if idx in cache:
                        return cache[idx]
                    item = test_set.__getitem__(idx)
                    batch_cpu = test_set.collate_batch([item])
                    cache[idx] = batch_cpu
                    return batch_cpu

                bev_list = []
                gt_list = []
                H16 = None
                W16 = None

                for idx in window:
                    batch_cpu = get_batch_cpu(idx)
                    batch_gpu = to_torch_batch_dict(batch_cpu, device)

                    for cur_module in model.module_list:
                        if cur_module is model.dense_head:
                            break
                        batch_gpu = cur_module(batch_gpu)

                    bev = batch_gpu["spatial_features_2d"]
                    B, C, H, W = bev.shape
                    H16_i, W16_i = pad16_hw(H, W)
                    if H16 is None:
                        H16, W16 = H16_i, W16_i

                    gt_occ = model._build_gt_occ_target(batch_gpu, H16, W16)
                    if gt_occ is None:
                        gt_occ = torch.zeros(B, 1, H16, W16, device=bev.device, dtype=bev.dtype)

                    bev_list.append(bev)
                    gt_list.append(gt_occ)

                frames_rgb_list = []
                for bev in bev_list:
                    rgb = model.bev_adapter(bev)
                    if rgb.shape[-2:] != (H16, W16):
                        rgb = F.pad(rgb, (0, W16 - rgb.shape[-1], 0, H16 - rgb.shape[-2]))
                    frames_rgb_list.append(rgb)

                frames_rgb = torch.cat(frames_rgb_list, dim=0)
                first_frame_gt = gt_list[0]

                xmem_out, _ = model.xmem_processor.do_pass(frames=frames_rgb, first_frame_gt=first_frame_gt, T=len(window))

                seq_ious = []
                start_ti = 0 if args.include_first else 1

                for ti in range(start_ti, len(window)):
                    key = f"masks_{ti}"
                    if key not in xmem_out:
                        continue
                    masks = xmem_out[key]
                    occ_prob = occ_prob_from_xmem_masks(masks)
                    gt_occ = gt_list[ti]
                    pred = occ_prob > float(args.thr)
                    gt = gt_occ > 0.5
                    iou_b = iou_binary(pred, gt)
                    iou = float(iou_b.mean().item())

                    seq_ious.append(iou)
                    sum_iou_frames += iou
                    cnt_iou_frames += 1

                    if ti == 2:
                        sum_iou_t3 += iou
                        cnt_iou_t3 += 1

                    if ti == len(window) - 1:
                        sum_iou_last += iou
                        cnt_iou_last += 1

                if len(seq_ious) > 0:
                    sum_iou_seq_mean += float(sum(seq_ious) / len(seq_ious))
                    cnt_iou_seqs += 1

                total_seqs += 1
                seq_id += 1

                if total_seqs % int(args.log_interval) == 0:
                    elapsed = time.time() - start_t
                    miou_frames = (sum_iou_frames / max(cnt_iou_frames, 1)) if cnt_iou_frames else 0.0
                    miou_seq = (sum_iou_seq_mean / max(cnt_iou_seqs, 1)) if cnt_iou_seqs else 0.0
                    miou_t3 = (sum_iou_t3 / max(cnt_iou_t3, 1)) if cnt_iou_t3 else 0.0
                    miou_last = (sum_iou_last / max(cnt_iou_last, 1)) if cnt_iou_last else 0.0
                    logger.info(
                        f"seqs={total_seqs} frames={cnt_iou_frames} "
                        f"mIoU_frames={miou_frames:.4f} mIoU_seq={miou_seq:.4f} "
                        f"mIoU_t3={miou_t3:.4f} mIoU_last={miou_last:.4f} "
                        f"elapsed={elapsed:.1f}s"
                    )

    miou_frames = (sum_iou_frames / max(cnt_iou_frames, 1)) if cnt_iou_frames else 0.0
    miou_seq = (sum_iou_seq_mean / max(cnt_iou_seqs, 1)) if cnt_iou_seqs else 0.0
    miou_t3 = (sum_iou_t3 / max(cnt_iou_t3, 1)) if cnt_iou_t3 else 0.0
    miou_last = (sum_iou_last / max(cnt_iou_last, 1)) if cnt_iou_last else 0.0

    results = {
        "split": cfg.DATA_CONFIG.DATA_SPLIT["test"],
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "keep_last_partial": bool(args.keep_last_partial),
        "thr": float(args.thr),
        "include_first": bool(args.include_first),
        "num_sequences": int(total_seqs),
        "num_frames_scored": int(cnt_iou_frames),
        "num_t3_scored": int(cnt_iou_t3),
        "num_last_scored": int(cnt_iou_last),
        "mIoU_frames": float(miou_frames),
        "mIoU_seq": float(miou_seq),
        "mIoU_3rd_frame": float(miou_t3),
        "mIoU_last_frame": float(miou_last),
    }

    out_json = Path(eval_output_dir) / "xmem_miou.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Final results:")
    logger.info(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
