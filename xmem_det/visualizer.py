import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class TemporalDebugger:
    def __init__(self, save_dir: str, log_every: int = 1, max_batches: int = 1,
                 attn_top_k_objects: int = 3):
        self.save_dir = save_dir
        self.log_every = int(log_every)
        self.max_batches = int(max_batches)
        self.attn_top_k_objects = attn_top_k_objects
        self.seq_dir: Optional[str] = None

    def start_sequence(self, seq_name: str):
        self.seq_dir = os.path.join(self.save_dir, seq_name)
        os.makedirs(self.seq_dir, exist_ok=True)

    def finish_sequence(self):
        self.seq_dir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_cpu_2d(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    def _bev_energy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(x, dim=0)

    def _get_top_object_positions(
        self,
        mp0: torch.Tensor,      # [H, W]
        k: int,
    ) -> List[Tuple[int, int]]:
        h, w = mp0.shape
        flat = mp0.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat, k=min(k, flat.numel()), largest=True)
        positions = []
        for idx, val in zip(topk_idx, topk_vals):
            if val.item() < 0.05:
                break
            r = int(idx.item() // w)
            c = int(idx.item() % w)
            positions.append((r, c))
        return positions

    def _compute_attention_map(
        self,
        query_pos: Tuple[int, int],
        q_t: torch.Tensor,              # [key_dim, H, W]
        st_keys: List[torch.Tensor],    # list of [key_dim, H, W]
    ) -> Optional[np.ndarray]:
        if len(st_keys) == 0:
            return None

        r, c = query_pos
        key_dim, h, w = q_t.shape
        hw = h * w

        q_vec = q_t[:, r, c]
        q_vec = F.normalize(q_vec.unsqueeze(0), dim=1)      # [1, key_dim]

        k_maps = []
        for k_map in st_keys:
            k_flat = k_map.reshape(key_dim, hw).permute(1, 0)
            k_maps.append(k_flat)
        k_all = torch.cat(k_maps, dim=0)                    # [M_total, key_dim]
        k_all = F.normalize(k_all, dim=1)

        diff = k_all - q_vec
        dist = (diff * diff).sum(dim=1)
        attn = torch.softmax(-dist, dim=0)                  # [M_total]




    # ------------------------------------------------------------------
    # GT / pred box drawing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_gt_boxes(batch_dict: dict, batch_idx: int = 0) -> Optional[np.ndarray]:
        """
        Pull gt_boxes out of an OpenPCDet batch_dict for one sample.
        Strips all-zero padding rows.  Returns [N_valid, 7+] or None.
        """
        gt = batch_dict.get("gt_boxes", None)
        if gt is None:
            return None
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        gt_sample = gt[batch_idx]                           # [N, 8]
        valid = np.abs(gt_sample).sum(axis=1) > 0
        return gt_sample[valid] if valid.any() else None

    def _draw_gt_boxes_bev(
        self,
        ax,
        boxes_np: np.ndarray,       # [N, 7+]  (x, y, z, dx, dy, dz, heading, ...)
        pc_range: list,             # [xmin, ymin, zmin, xmax, ymax, zmax]
        H: int,
        W: int,
        color: str = "lime",
        heading_color: str = "yellow",
    ):
        """
        Overlay boxes as rotated BEV rectangles.

        Coordinate mapping (no y-flip needed):
          LiDAR x → image col  (xmin = col 0,   xmax = col W)
          LiDAR y → image row  (ymin = row 0,   ymax = row H)
        """
        if boxes_np is None or len(boxes_np) == 0:
            return

        xmin, ymin = float(pc_range[0]), float(pc_range[1])
        xmax, ymax = float(pc_range[3]), float(pc_range[4])
        x_scale = W / (xmax - xmin)
        y_scale = H / (ymax - ymin)

        for box in boxes_np:
            cx, cy, _cz, dx, dy, _dz, heading = box[:7]

            if dx <= 0 or dy <= 0:
                continue
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                continue

            # LiDAR → pixel  (no y flip)
            px = (cx - xmin) * x_scale
            py = (cy - ymin) * y_scale

            hdx = dx * x_scale / 2.0
            hdy = dy * y_scale / 2.0

            corners_local = np.array([
                [ hdx,  hdy],
                [-hdx,  hdy],
                [-hdx, -hdy],
                [ hdx, -hdy],
            ], dtype=np.float32)

            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            rot   = np.array([[cos_h, -sin_h],
                               [sin_h,  cos_h]], dtype=np.float32)
            corners_rot = corners_local @ rot.T             # [4, 2]

            corners_px = corners_rot.copy()
            corners_px[:, 0] = corners_rot[:, 0] + px      # col
            corners_px[:, 1] = corners_rot[:, 1] + py      # row  (no flip)

            poly = plt.Polygon(
                corners_px, closed=True,
                fill=False, edgecolor=color, linewidth=1.5,
                zorder=5, clip_on=True,
            )
            ax.add_patch(poly)

            # Heading arrow
            fwd = rot @ np.array([hdx, 0.0])
            ax.annotate(
                '',
                xy=(px + fwd[0], py + fwd[1]),
                xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color=heading_color, lw=1.2),
                zorder=6, annotation_clip=True,
            )

        # Lock axes back to image extent after adding patches
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)   # row 0 at top, matches origin="upper"

    # ------------------------------------------------------------------
    # Main logging entry point
    # ------------------------------------------------------------------

    def log_timestep(
        self,
        t: int,
        bev_t: torch.Tensor,        # [B, C, H, W]
        mfused_t: torch.Tensor,     # [B, C, H, W]
        mp_t: torch.Tensor,         # [B, 7, H, W]
        bank_state: Dict[str, Any],
        bank_maps: Dict[str, List[torch.Tensor]],
        mem_kinds: List[str],
        q_t: Optional[torch.Tensor] = None,         # [B, key_dim, H, W]
        st_keys: Optional[List[torch.Tensor]] = None,
        batch_idx: int = 0,
        gt_boxes: Optional[np.ndarray] = None,      # [N, 7+] already extracted
        pred_boxes: Optional[np.ndarray] = None,    # [N, 7+] post-NMS predictions
        pc_range: Optional[list] = None,
    ):
        if self.seq_dir is None:
            raise RuntimeError("start_sequence() must be called before log_timestep()")
        if self.log_every > 1 and (t % self.log_every) != 0:
            return
        if batch_idx >= self.max_batches:
            return

        bev = bev_t[batch_idx]      # [C, H, W]
        mf  = mfused_t[batch_idx]
        mp  = mp_t[batch_idx]

        bev_e = self._bev_energy(bev)
        mf_e  = self._bev_energy(mf)
        diff  = (mf - bev).abs().mean(dim=0)
        mp0   = mp[0]               # [H, W] existence channel

        H_bev, W_bev = int(mp0.shape[0]), int(mp0.shape[1])

        st_usage_list = bank_maps.get("st_usage", [])
        st_mp0_list   = bank_maps.get("st_mp0",   [])

        st_usage_mean = (torch.stack(st_usage_list, dim=0).mean(dim=0)
                         if len(st_usage_list) > 0 else None)
        st_mp0_mean   = (torch.stack(st_mp0_list,  dim=0).mean(dim=0)
                         if len(st_mp0_list)  > 0 else None)

        # ── Attention spotlights ──────────────────────────────────────
        spotlight_maps: List[Tuple[Tuple[int, int], np.ndarray]] = []
        if q_t is not None and st_keys is not None and len(st_keys) > 0:
            q_single    = q_t[batch_idx]
            keys_single = [k[batch_idx] for k in st_keys]
            for pos in self._get_top_object_positions(mp0, k=self.attn_top_k_objects):
                attn_map = self._compute_attention_map(pos, q_single, keys_single)
                if attn_map is not None:
                    spotlight_maps.append((pos, attn_map))

        # ── Layout: 2 rows (+ optional spotlight row) ────────────────
        n_spots = len(spotlight_maps)
        n_rows  = 3 if n_spots > 0 else 2
        n_cols  = max(3, n_spots)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        def _show(ax, data, title, cmap="viridis"):
            if isinstance(data, torch.Tensor):
                data = self._to_cpu_2d(data)
            ax.imshow(data, cmap=cmap, origin="upper")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        def _hide(ax):
            ax.axis("off")

        # Row 0: bev_energy | mfused_energy | abs_mean(mfused-bev)
        _show(axes[0, 0], bev_e, "bev_energy")
        _show(axes[0, 1], mf_e,  "mfused_energy")
        _show(axes[0, 2], diff,  "abs_mean(mfused-bev)")
        for col in range(3, n_cols):
            _hide(axes[0, col])

        # Row 1, col 0: mp_ch0  +  GT boxes (lime)  +  pred boxes (red)
        _show(axes[1, 0], mp0, "mp_ch0")
        n_gt   = 0
        n_pred = 0
        if pc_range is not None:
            if gt_boxes is not None and len(gt_boxes) > 0:
                self._draw_gt_boxes_bev(axes[1, 0], gt_boxes, pc_range,
                                        H_bev, W_bev, color="lime", heading_color="yellow")
                n_gt = len(gt_boxes)
            if pred_boxes is not None and len(pred_boxes) > 0:
                self._draw_gt_boxes_bev(axes[1, 0], pred_boxes, pc_range,
                                        H_bev, W_bev, color="red", heading_color="orange")
                n_pred = len(pred_boxes)

        title_str = "mp_ch0"
        if n_gt   > 0: title_str += f"  lime=GT({n_gt})"
        if n_pred > 0: title_str += f"  red=pred({n_pred})"
        axes[1, 0].set_title(title_str, fontsize=9)

        # Row 1, col 1-2: st_usage | st_mp0
        if st_usage_mean is not None:
            _show(axes[1, 1], st_usage_mean, "st_usage_mean")
        else:
            axes[1, 1].text(0.1, 0.5, "no ST usage", fontsize=10)
            _hide(axes[1, 1])
        if st_mp0_mean is not None:
            _show(axes[1, 2], st_mp0_mean, "st_mp0_mean")
        else:
            _hide(axes[1, 2])
        for col in range(3, n_cols):
            _hide(axes[1, col])

        # Row 2: per-object attention spotlights
        if n_spots > 0:
            for col, ((r, c), attn_map) in enumerate(spotlight_maps):
                ax = axes[2, col]
                ax.imshow(attn_map, cmap="hot", origin="upper")
                ax.set_title(f"attn from ({r},{c})  mp0={mp0[r,c]:.3f}", fontsize=8)
                ax.axis("off")
                ax.plot(c, r, marker="+", color="cyan", markersize=12, markeredgewidth=2)
                axes[1, 0].plot(c, r, marker="+", color="red",
                                markersize=10, markeredgewidth=1.5)
            for col in range(n_spots, n_cols):
                _hide(axes[2, col])

        # ── Save ──────────────────────────────────────────────────────
        info = {
            "t":             int(t),
            "step":          int(bank_state.get("step",      -1)),
            "st_len":        int(bank_state.get("st_len",    -1)),
            "lt_len":        int(bank_state.get("lt_len",    -1)),
            "lt_tokens":     int(bank_state.get("lt_tokens", -1)),
            "did_write":     int(bank_state.get("did_write", -1)),
            "mem_kinds":     ",".join(mem_kinds),
            "mp0_mean":      float(mp0.detach().float().mean().item()),
            "mfused_delta":  float((mf - bev).abs().mean().item()),
            "n_gt":          n_gt,
            "n_pred":        n_pred,
            "n_spotlights":  n_spots,
        }

        out_png = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.png")
        fig.suptitle(str(info), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)

        out_npz = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.npz")
        np.savez_compressed(
            out_npz,
            bev_energy   = self._to_cpu_2d(bev_e),
            mfused_energy= self._to_cpu_2d(mf_e),
            diff         = self._to_cpu_2d(diff),
            mp0          = self._to_cpu_2d(mp0),
            st_usage_mean= (self._to_cpu_2d(st_usage_mean)
                            if st_usage_mean is not None else np.array([])),
            gt_boxes     = (gt_boxes   if gt_boxes   is not None else np.array([])),
            pred_boxes   = (pred_boxes if pred_boxes is not None else np.array([])),
        )