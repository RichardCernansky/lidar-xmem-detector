
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
    over all frames currently stored in the short-term memory bank.

    KEY DESIGN:
      - q_t is passed in explicitly from dbg["q_t"] — the output of query_enc
        applied to the current (not-yet-stored) BEV frame.
      - ALL entries in bank._st_keys are historical frames (stored every tau steps).
        With tau=2: _st_keys[0]=T-2*ts, ..., _st_keys[-1]=T-tau (last stored).
      - Column titles reflect actual temporal distance: T-tau, T-2*tau, etc.
    """

    def __init__(
        self,
        save_dir: str,
        n_hist: int = 4,
        max_query_pts: int = 2,
        tau: int = 2,
        q_chunk: int = 512,
    ):
        self.save_dir      = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_hist        = n_hist
        self.max_query_pts = max_query_pts
        self.tau           = tau
        self.q_chunk       = q_chunk

        self._attn_maps:    Optional[np.ndarray] = None   # [n_pts, n_vis, H, W]
        self._bev_maps:     Optional[np.ndarray] = None   # [n_vis, H, W]
        self._n_vis:        int = 0
        self._n_pts:        int = 0
        self._query_pixels: list = []

    # ------------------------------------------------------------------

    @torch.no_grad()
    def capture(
        self,
        bank:       ReasonNetTemporalBank,
        pred_dicts: list,
        q_t:        torch.Tensor,   # [B, key_dim, H, W] — current frame query from query_enc
    ):
        """
        Call AFTER forward().

        q_t  : the actual current-frame query (NOT stored in bank).
        bank : _st_keys contains only historical frames stored every tau steps.
               _st_keys[-1] = most recent stored frame = T - tau
               _st_keys[-2] = T - 2*tau
               etc.
        """
        n_st = len(bank._st_keys)
        if n_st == 0:
            return

        n_vis = min(n_st, self.n_hist)   # ALL stored frames are history

        b, key_dim, h, w = q_t.shape
        hw = h * w

        # Flatten current query
        q_flat = q_t[0].reshape(key_dim, hw).T   # [HW, key_dim]

        # ------------------------------------------------------------------
        # Pick query pixel locations from top-scoring detections
        # ------------------------------------------------------------------
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
            # Fallback: highest-magnitude locations in current query map
            mag      = q_t[0].abs().mean(0)   # [H, W]
            flat_idx = mag.reshape(-1).topk(self.max_query_pts).indices
            for fi in flat_idx:
                query_pixels.append((int(fi // w), int(fi % w)))

        n_pts     = len(query_pixels)
        attn_maps = np.zeros((n_pts, n_vis, h, w), dtype=np.float32)
        bev_maps  = np.zeros((n_vis, h, w),        dtype=np.float32)

        # ------------------------------------------------------------------
        # Compute attention: q_t vs each historical stored frame
        # Iterate newest → oldest: _st_keys[-1]=T-tau, _st_keys[-2]=T-2*tau, ...
        # ------------------------------------------------------------------
        for vi in range(n_vis):
            st_idx = n_st - 1 - vi              # newest first
            k_map  = bank._st_keys[st_idx]      # [1, key_dim, H, W]
            k_flat = k_map[0].reshape(key_dim, hw).T   # [HW, key_dim]

            # BEV context from stored values (for background in save())
            bev_maps[vi] = bank._st_vals[st_idx][0].abs().mean(0).cpu().numpy()

            for pi, (qi, qj) in enumerate(query_pixels):
                q_vec = q_flat[qi * w + qj].unsqueeze(0)   # [1, key_dim]

                # L2 similarity (STCN):  S = dist^2 / sum_k dist^2
                q2   = (q_vec * q_vec).sum(dim=1, keepdim=True)          # [1, 1]
                k2   = (k_flat * k_flat).sum(dim=1).unsqueeze(0)         # [1, HW]
                dot  = q_vec @ k_flat.T                                   # [1, HW]
                dist = (q2 + k2 - 2.0 * dot).clamp_min(0.0)             # [1, HW]
                S    = dist / (dist.sum(dim=1, keepdim=True) + 1e-8)    # [1, HW]
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

        n_pts = self._n_pts
        n_vis = self._n_vis
        fig   = plt.figure(figsize=(3.5 * n_vis, 3.2 * n_pts + 0.6))
        gs    = gridspec.GridSpec(
            n_pts, n_vis,
            figure=fig,
            hspace=0.35, wspace=0.08,
            left=0.05, right=0.88, top=0.88, bottom=0.05,
        )

        vmin_global = float(np.percentile(self._attn_maps, 1))
        vmax_global = float(np.percentile(self._attn_maps, 99))

        h, w = self._attn_maps.shape[2], self._attn_maps.shape[3]

        for pi in range(n_pts):
            for vi in range(n_vis):
                ax = fig.add_subplot(gs[pi, vi])

                ax.imshow(
                    self._attn_maps[pi, vi],
                    cmap="coolwarm",
                    vmin=vmin_global, vmax=vmax_global,
                    interpolation="bilinear",
                    origin="upper",
                )

                # Grid lines (ReasonNet style)
                for gx in np.linspace(0, w, 5):
                    ax.axvline(gx, color="black", lw=0.5, alpha=0.4)
                for gy in np.linspace(0, h, 5):
                    ax.axhline(gy, color="black", lw=0.5, alpha=0.4)

                # White cross = query location (from current frame q_t)
                qi, qj = self._query_pixels[pi]
                ax.plot(qj, qi, "w+", markersize=8, markeredgewidth=1.5)

                # Column title: actual temporal distance in frames
                if pi == 0:
                    frames_ago = (vi + 1) * self.tau
                    ax.set_title(f"T − {frames_ago}", fontsize=10, pad=4)

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
                   help="Number of past frames to visualize.")
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

    cfg.TAG            = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        stride=1,
        nusc_version=cfg.DATA_CONFIG.VERSION,
        nusc_dataroot=cfg.DATA_CONFIG.DATA_PATH,
        root_path=None,
        sweep_stride=int(getattr(cfg.DATA_CONFIG, "SWEEP_STRIDE", 2)),
    )

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

    # Read tau from bank config for correct column labels in visualisation
    tau = int(getattr(model.bank, "tau", 2))

    # ------------------------------------------------------------------
    # Attention logger (optional)
    # ------------------------------------------------------------------
    do_vis   = args.vis_interval > 0
    attn_log = AttentionLogger(
        save_dir=args.vis_dir,
        n_hist=args.vis_n_hist,
        max_query_pts=args.vis_n_pts,
        tau=tau,
    ) if do_vis else None

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    det_annos = [None] * len(test_set)
    start_t   = time.time()

    with torch.no_grad():
        for sample_idx, seq in enumerate(test_loader):

            # Reset bank BEFORE forward so state from previous sequence never leaks
            if hasattr(model, "reset_sequence"):
                model.reset_sequence(sample_idx)

            frames      = seq["frames"]
            frames_list = [to_torch_batch_dict(f, device) for f in frames]

            # forward() returns (pred_dicts, recall_dicts, dbg)
            # dbg["q_t"] is the query_enc output for the current (keyframe) BEV —
            # this is the correct query to use for attention visualisation.
            pred_dicts, _recall_dicts, dbg = model(
                frames_list=frames_list,
                compute_det_loss=False,
            )

            # ------------------------------------------------------------------
            # Map sequence index → base dataset index
            # ------------------------------------------------------------------
            if hasattr(test_set, "keyframe_indices"):
                base_idx = int(test_set.keyframe_indices[sample_idx])
            else:
                base_idx = sample_idx

            base_item      = test_set.base.__getitem__(base_idx)
            base_batch_cpu = test_set.base.collate_batch([base_item])

            annos = test_set.base.generate_prediction_dicts(
                batch_dict=base_batch_cpu,
                pred_dicts=pred_dicts,
                class_names=cfg.CLASS_NAMES,
                output_path=None,
            )
            det_annos[sample_idx] = annos[0]

            # ------------------------------------------------------------------
            # Attention visualization
            # ------------------------------------------------------------------
            if do_vis and (sample_idx % args.vis_interval == 0):
                try:
                    # q_t from dbg: the actual current-frame query [B, key_dim, H, W]
                    # This is NEVER stored in the bank — it is the live query that
                    # attends to all _st_keys[] which are the historical stored frames.
                    q_t = dbg["q_t"]   # [1, key_dim, H, W]

                    attn_log.capture(
                        bank=model.bank,
                        pred_dicts=pred_dicts,
                        q_t=q_t,
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