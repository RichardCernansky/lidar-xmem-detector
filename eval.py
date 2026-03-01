"""
eval_temporal.py — Evaluation script for TemporalPointPillar with attention visualization.

Fixes vs original:
  1. Correct return-value unpacking: inference forward() returns (final_box, {}, None)
     — pred_dicts IS final_box (list of box dicts), not a nested structure.
  2. base_item index: uses seq_dataset keyframe_indices to map sequence → base dataset index,
     instead of assuming sample_idx == base_idx (breaks when len(seq) != len(base)).
  3. Bank reset ordering: reset_sequence() call before forward() is now the canonical reset;
     the in-forward reset() is a safety net only.
  4. Attention visualization: after every vis_interval sequences, dumps a ReasonNet-style
     attention grid (one query location per object, T-1…T-4 frames) to PNG.
  5. Proper eval mode throughout; no accidental train-mode BN updates.

Usage:
  python eval_temporal.py \
      --cfg_file tools/cfgs/nuscenes_models/temporal_pp.yaml \
      --ckpt output/.../checkpoint_epoch_20.pth \
      --split val \
      --num_groups 4 \
      --vis_interval 50 \
      --vis_dir ./attn_vis
"""

import argparse
import datetime
import time
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")                   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils

from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.memory_fuser import ReasonNetTemporalBank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Attention visualization
# ---------------------------------------------------------------------------

class AttentionLogger:
    """
    Hooks into ReasonNetTemporalBank to capture per-frame attention maps
    during inference, then renders a ReasonNet Figure-4-style grid.

    For each sequence we pick up to `max_query_pts` query locations
    (chosen near detected objects) and show their attention weights
    over the last `n_hist` short-term memory frames.
    """

    def __init__(
        self,
        save_dir: str,
        n_hist: int = 4,            # number of past frames to visualise (T-1…T-n)
        max_query_pts: int = 2,     # rows in the grid (one per query location)
        q_chunk: int = 512,
    ):
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_hist       = n_hist
        self.max_query_pts = max_query_pts
        self.q_chunk      = q_chunk

        # Filled by capture() after the forward pass
        self._attn_maps: Optional[np.ndarray] = None   # [n_pts, n_hist, H, W]
        self._bev_maps:  Optional[np.ndarray] = None   # [n_hist, H, W]

    # ------------------------------------------------------------------

    def _l2_similarity(
        self,
        q_flat: torch.Tensor,  # [HW_q, key_dim]
        k_flat: torch.Tensor,  # [M,    key_dim]
    ) -> torch.Tensor:
        """Normalised L2 similarity: S(q,k) = dist(q,k)^2 / sum_k dist(q,k)^2"""
        # [HW_q, 1] + [1, M] - 2*[HW_q, M]  ->  [HW_q, M]
        q2  = (q_flat * q_flat).sum(dim=1, keepdim=True)   # [HW_q, 1]
        k2  = (k_flat * k_flat).sum(dim=1).unsqueeze(0)    # [1,    M]
        dot = q_flat @ k_flat.T                             # [HW_q, M]
        dist = (q2 + k2 - 2.0 * dot).clamp_min(0.0)       # [HW_q, M]
        S    = dist / (dist.sum(dim=1, keepdim=True) + 1e-8)
        return S                                            # [HW_q, M]

    # ------------------------------------------------------------------

    @torch.no_grad()
    def capture(
        self,
        bank: ReasonNetTemporalBank,
        pred_dicts: list,
    ):
        """
        Call AFTER forward() — bank._st_keys still has the stored frames.
        We compute the keyframe query from the most recent stored key directly
        (q_t was stored as the key, so bank._st_keys[-1] IS the keyframe query).
        """
        if not bank._st_keys:
            return

        n_st  = len(bank._st_keys)
        n_vis = min(n_st - 1, self.n_hist)  # exclude keyframe's own entry
        if n_vis == 0:
            return

        b, key_dim, h, w = bank._st_keys[-1].shape  # keyframe key = query
        hw = h * w

        # The keyframe's query is exactly _st_keys[-1] (q_t was stored as key)
        q_map  = bank._st_keys[-1]                          # [1, key_dim, H, W]
        q_flat = q_map[0].reshape(key_dim, hw).T            # [HW, key_dim]

        # Pick query locations from top-scoring detections
        query_pixels = []
        if pred_dicts and "pred_scores" in pred_dicts[0]:
            scores = pred_dicts[0]["pred_scores"]
            boxes  = pred_dicts[0]["pred_boxes"]
            if scores.numel() > 0:
                topk_n = min(self.max_query_pts, int(scores.shape[0]))
                topk   = torch.topk(scores, k=topk_n).indices
                for idx in topk:
                    bx, by = float(boxes[idx, 0]), float(boxes[idx, 1])
                    px_frac = np.clip((bx + 51.2) / 102.4, 0.0, 1.0)
                    py_frac = np.clip((by + 51.2) / 102.4, 0.0, 1.0)
                    qi = int(py_frac * (h - 1))
                    qj = int(px_frac * (w - 1))
                    query_pixels.append((qi, qj))

        if not query_pixels:
            # Fallback: highest-magnitude locations in the keyframe key map
            mag = q_map[0].abs().mean(0)  # [H, W]
            flat_idx = mag.reshape(-1).topk(self.max_query_pts).indices
            for fi in flat_idx:
                query_pixels.append((int(fi // w), int(fi % w)))

        n_pts = len(query_pixels)
        attn_maps = np.zeros((n_pts, n_vis, h, w), dtype=np.float32)
        bev_maps  = np.zeros((n_vis, h, w), dtype=np.float32)

        # Iterate history frames: T-1 is _st_keys[-2], T-2 is _st_keys[-3], etc.
        for vi in range(n_vis):
            st_idx = n_st - 2 - vi          # skip [-1] which is the keyframe itself
            k_map  = bank._st_keys[st_idx]  # [1, key_dim, H, W]
            k_flat = k_map[0].reshape(key_dim, hw).T  # [HW, key_dim]

            v_map = bank._st_vals[st_idx]
            bev_maps[vi] = v_map[0].abs().mean(0).cpu().numpy()

            for pi, (qi, qj) in enumerate(query_pixels):
                q_vec = q_flat[qi * w + qj].unsqueeze(0)    # [1, key_dim]
                # L2 similarity (STCN)
                q2   = (q_vec * q_vec).sum(dim=1, keepdim=True)
                k2   = (k_flat * k_flat).sum(dim=1).unsqueeze(0)
                dot  = q_vec @ k_flat.T
                dist = (q2 + k2 - 2.0 * dot).clamp_min(0.0)  # [1, HW]
                S    = dist / (dist.sum(dim=1, keepdim=True) + 1e-8)
                attn_maps[pi, vi] = S.reshape(h, w).cpu().numpy()

        self._attn_maps    = attn_maps
        self._bev_maps     = bev_maps
        self._n_vis        = n_vis
        self._n_pts        = n_pts
        self._query_pixels = query_pixels

    # ------------------------------------------------------------------

    def save(self, seq_idx: int, tag: str = ""):
        """Render and save a ReasonNet-style attention grid."""
        if self._attn_maps is None:
            return

        n_pts  = self._n_pts
        n_vis  = self._n_vis
        fig    = plt.figure(figsize=(3.5 * n_vis, 3.2 * n_pts + 0.6))
        gs     = gridspec.GridSpec(
            n_pts, n_vis,
            figure=fig,
            hspace=0.35, wspace=0.08,
            left=0.05, right=0.88, top=0.88, bottom=0.05,
        )

        vmin_global = float(np.percentile(self._attn_maps, 1))
        vmax_global = float(np.percentile(self._attn_maps, 99))

        for pi in range(n_pts):
            for vi in range(n_vis):
                ax = fig.add_subplot(gs[pi, vi])
                # Background: BEV value magnitude (grey-ish context)
                bev_norm = self._bev_maps[vi]
                bev_norm = (bev_norm - bev_norm.min()) / (bev_norm.ptp() + 1e-8)
                im = ax.imshow(
                    self._attn_maps[pi, vi],
                    cmap="coolwarm",
                    vmin=vmin_global, vmax=vmax_global,
                    interpolation="bilinear",
                    origin="upper",
                )
                # Draw grid lines (ReasonNet style)
                h, w = self._attn_maps.shape[2], self._attn_maps.shape[3]
                for gx in np.linspace(0, w, 5):
                    ax.axvline(gx, color="black", lw=0.5, alpha=0.4)
                for gy in np.linspace(0, h, 5):
                    ax.axhline(gy, color="black", lw=0.5, alpha=0.4)

                # Mark query location with a white cross
                qi, qj = self._query_pixels[pi]
                ax.plot(qj, qi, "w+", markersize=8, markeredgewidth=1.5)

                # Column title (only top row)
                if pi == 0:
                    ax.set_title(f"T − {vi + 1}", fontsize=10, pad=4)
                ax.axis("off")

        # Shared colorbar
        cbar_ax = fig.add_axes([0.90, 0.10, 0.025, 0.75])
        sm = plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([vmin_global, (vmin_global + vmax_global) / 2, vmax_global])
        cbar.set_ticklabels(["0.0", "0.5", "1.0"])
        cbar.ax.tick_params(labelsize=8)

        tag_str = f"_{tag}" if tag else ""
        fname   = self.save_dir / f"attn_seq{seq_idx:04d}{tag_str}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fname


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file",       type=str,  required=True)
    p.add_argument("--ckpt",           type=str,  required=True)
    p.add_argument("--workers",        type=int,  default=4)
    p.add_argument("--split",          type=str,  default=None)
    p.add_argument("--num_groups",     type=int,  default=None,
                   help="Sweep groups per keyframe. Defaults to TRAIN.SEQ_LEN.")
    p.add_argument("--extra_tag",      type=str,  default="default")
    p.add_argument("--eval_tag",       type=str,  default="sweep_eval")
    p.add_argument("--log_interval",   type=int,  default=50)
    p.add_argument("--vis_interval",   type=int,  default=50,
                   help="Save attention visualization every N sequences (0 = disable).")
    p.add_argument("--vis_dir",        type=str,  default="./attn_vis",
                   help="Directory for attention visualization PNGs.")
    p.add_argument("--vis_n_hist",     type=int,  default=4,
                   help="Number of past frames to visualize (T-1 ... T-n).")
    p.add_argument("--vis_n_pts",      type=int,  default=2,
                   help="Number of query locations (rows) per visualization.")
    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    if args.num_groups is None:
        args.num_groups = int(getattr(cfg.TRAIN, "SEQ_LEN", 4))

    cfg.TAG           = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    root_dir       = getattr(cfg, "ROOT_DIR", Path.cwd())
    output_dir     = Path(root_dir) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    eval_output_dir = output_dir / "eval_sweep" / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ("log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger   = common_utils.create_logger(log_file, rank=0)

    logger.info("**********************Start logging**********************")
    logger.info(f"cfg_file={args.cfg_file}")
    logger.info(f"ckpt={args.ckpt}")
    logger.info(f"split={cfg.DATA_CONFIG.DATA_SPLIT['test']}")
    logger.info(f"num_groups={args.num_groups}")
    log_config_to_file(cfg, logger=logger)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    test_set = NuScenesSeqDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger,
        seq_len=int(getattr(cfg.TRAIN, "SEQ_LEN", 4)),
        stride=1,           # always 1 for eval — never subsample the val set
        nusc_version=cfg.DATA_CONFIG.VERSION,
        nusc_dataroot=cfg.DATA_CONFIG.DATA_PATH,
        root_path=None,
        sweep_stride=int(getattr(cfg.DATA_CONFIG, "SWEEP_STRIDE", 2)),
    )

    # FIX: sanity-check that the sequence dataset and base dataset are aligned.
    # len(test_set) should equal len(test_set.base) because each sequence maps
    # to exactly one annotated keyframe.  Log a warning if not.
    if len(test_set) != len(test_set.base):
        logger.warning(
            f"[WARN] len(seq_dataset)={len(test_set)} != len(base_dataset)={len(test_set.base)}. "
            "base_item lookup by sample_idx may be wrong — check NuScenesSeqDataset.keyframe_indices."
        )

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
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # ------------------------------------------------------------------
    # Attention logger (optional)
    # ------------------------------------------------------------------
    do_vis   = args.vis_interval > 0
    attn_log = AttentionLogger(
        save_dir=args.vis_dir,
        n_hist=args.vis_n_hist,
        max_query_pts=args.vis_n_pts,
    ) if do_vis else None

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    det_annos = [None] * len(test_set)
    start_t   = time.time()

    with torch.no_grad():
        for sample_idx, seq in enumerate(test_loader):

            # FIX: reset bank BEFORE forward so persistent GRU state from the
            # previous sequence never leaks into this one.
            # (forward() also calls bank.reset() internally as a safety net.)
            if hasattr(model, "reset_sequence"):
                model.reset_sequence(sample_idx)

            frames      = seq["frames"]
            frames_list = [to_torch_batch_dict(f, device) for f in frames]

            # FIX: correct return-value unpacking.
            # Inference path of forward() returns: (final_box_dicts, {}, None)
            # — exactly 3 values, and final_box_dicts IS the pred_dicts for
            # generate_prediction_dicts (a list of per-batch-element dicts with
            # 'pred_boxes', 'pred_scores', 'pred_labels').
            pred_dicts, _recall_dicts, _ = model(
                frames_list=frames_list,
                compute_det_loss=False,
            )
            # pred_dicts: list[dict], len == batch_size == 1

            # ------------------------------------------------------------------
            # FIX: map sequence index → base dataset index correctly.
            # If NuScenesSeqDataset exposes a keyframe_indices list, use it.
            # Otherwise fall back to sample_idx (valid only when 1:1 mapping holds).
            # ------------------------------------------------------------------
            if hasattr(test_set, "keyframe_indices"):
                base_idx = int(test_set.keyframe_indices[sample_idx])
            else:
                base_idx = sample_idx   # works when len(seq_set) == len(base_set)

            base_item     = test_set.base.__getitem__(base_idx)
            base_batch_cpu = test_set.base.collate_batch([base_item])

            # generate_prediction_dicts only uses metadata from base_batch_cpu
            # (frame_id, lidar_path, etc.) and the prediction values from pred_dicts.
            # It does NOT need spatial_features_2d from base_batch_cpu.
            annos = test_set.base.generate_prediction_dicts(
                batch_dict=base_batch_cpu,
                pred_dicts=pred_dicts,
                class_names=cfg.CLASS_NAMES,
                output_path=None,
            )
            det_annos[sample_idx] = annos[0]

            # ------------------------------------------------------------------
            # Attention visualization (every vis_interval sequences)
            # ------------------------------------------------------------------
            if do_vis and (sample_idx % args.vis_interval == 0):
                try:
                    # Reconstruct bev_list for the logger:
                    # After forward() the frames_list entries [0..T-2] are set to None
                    # (freed in the forward pass). We only have the keyframe BEV available
                    # via model's bank._st_vals (stored after backbone).
                    # We pass an empty list and let AttentionLogger use bank internal state.
                    attn_log.capture(
                        bank=model.bank,
                        pred_dicts=pred_dicts,
                    )
                    saved = attn_log.save(seq_idx=sample_idx)
                    if saved:
                        logger.info(f"Saved attention vis → {saved}")
                except Exception as e:
                    logger.warning(f"Attention vis failed at seq {sample_idx}: {e}")

            # ------------------------------------------------------------------
            # Progress log
            # ------------------------------------------------------------------
            if (sample_idx + 1) % int(args.log_interval) == 0:
                elapsed = time.time() - start_t
                pct     = 100.0 * (sample_idx + 1) / len(test_set)
                eta     = elapsed / (sample_idx + 1) * (len(test_set) - sample_idx - 1)
                logger.info(
                    f"Eval: {sample_idx + 1}/{len(test_set)} ({pct:.1f}%)  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                )

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    missing_idxs = [i for i, a in enumerate(det_annos) if a is None]
    if missing_idxs:
        raise RuntimeError(
            f"Missing predictions for {len(missing_idxs)} samples. "
            f"First missing: idx={missing_idxs[0]}"
        )

    logger.info("All predictions generated.")

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------
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