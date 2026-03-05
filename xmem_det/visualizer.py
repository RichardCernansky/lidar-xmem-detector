import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TemporalDebugger:
    def __init__(self, save_dir: str, log_every: int = 1, max_batches: int = 1,
                 attn_top_k_objects: int = 3):
        self.save_dir = save_dir
        self.log_every = int(log_every)
        self.max_batches = int(max_batches)
        self.attn_top_k_objects = attn_top_k_objects  # how many car spots to spotlight
        self.seq_dir: Optional[str] = None

    def start_sequence(self, seq_name: str):
        self.seq_dir = os.path.join(self.save_dir, seq_name)
        os.makedirs(self.seq_dir, exist_ok=True)

    def _to_cpu_2d(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    def _bev_energy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(x, dim=0)

    def _get_top_object_positions(
        self,
        mp0: torch.Tensor,      # [H, W] existence probability
        k: int,
    ) -> List[Tuple[int, int]]:
        """
        Return (row, col) of the top-k highest object existence positions.
        These are used as query probe points for attention spotlight.
        """
        h, w = mp0.shape
        flat = mp0.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat, k=min(k, flat.numel()), largest=True)
        # Only keep positions with meaningful object probability
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
        query_pos: Tuple[int, int],   # (row, col) in BEV grid
        q_t: torch.Tensor,            # [key_dim, H, W] query map for this frame
        st_keys: List[torch.Tensor],  # list of [key_dim, H, W] stored keys
    ) -> Optional[np.ndarray]:
        """
        For a single query position (r, c), compute its attention distribution
        over ALL short-term memory positions (concatenated).

        Returns a [H, W] attention map showing where this query attends in memory.
        If no memory, returns None.
        """
        if len(st_keys) == 0:
            return None

        r, c = query_pos
        key_dim, h, w = q_t.shape
        hw = h * w

        # Extract query vector at this position: [key_dim]
        q_vec = q_t[:, r, c]  # [key_dim]
        q_vec = F.normalize(q_vec.unsqueeze(0), dim=1)  # [1, key_dim]

        # Concatenate all ST memory keys: [M_total, key_dim]
        k_maps = []
        for k_map in st_keys:
            # k_map: [key_dim, H, W] → [HW, key_dim]
            k_flat = k_map.reshape(key_dim, hw).permute(1, 0)
            k_maps.append(k_flat)
        k_all = torch.cat(k_maps, dim=0)           # [M_total, key_dim]
        k_all = F.normalize(k_all, dim=1)          # [M_total, key_dim]

        # Compute squared L2 distances: [M_total]
        diff = k_all - q_vec                        # [M_total, key_dim]
        dist = (diff * diff).sum(dim=1)             # [M_total]

        # Softmax over all memory positions
        attn = torch.softmax(-dist, dim=0)          # [M_total]

        # Use only the LAST (most recent) ST frame's portion for spatial display
        # (most interpretable — same H×W as current frame)
        last_hw = h * w
        attn_last = attn[-last_hw:]                 # [HW] — last ST frame
        attn_map = attn_last.reshape(h, w)          # [H, W]

        return self._to_cpu_2d(attn_map)

    def log_timestep(
        self,
        t: int,
        bev_t: torch.Tensor,       # [B, C, H, W]
        mfused_t: torch.Tensor,    # [B, C, H, W]
        mp_t: torch.Tensor,        # [B, 7, H, W]
        bank_state: Dict[str, Any],
        bank_maps: Dict[str, List[torch.Tensor]],
        mem_kinds: List[str],
        q_t: Optional[torch.Tensor] = None,         # [B, key_dim, H, W] — query map
        st_keys: Optional[List[torch.Tensor]] = None,  # raw stored keys [B, key_dim, H, W]
        batch_idx: int = 0,
    ):
        if self.seq_dir is None:
            raise RuntimeError("start_sequence() must be called before log_timestep()")
        if self.log_every > 1 and (t % self.log_every) != 0:
            return
        if batch_idx >= self.max_batches:
            return

        bev = bev_t[batch_idx]      # [C, H, W]
        mf  = mfused_t[batch_idx]   # [C, H, W]
        mp  = mp_t[batch_idx]       # [7, H, W]

        bev_e = self._bev_energy(bev)
        mf_e  = self._bev_energy(mf)
        diff  = (mf - bev).abs().mean(dim=0)
        mp0   = mp[0]               # [H, W] existence channel

        st_usage_list = bank_maps.get("st_usage", [])
        st_mp0_list   = bank_maps.get("st_mp0",   [])

        st_usage_mean = (torch.stack(st_usage_list, dim=0).mean(dim=0)
                         if len(st_usage_list) > 0 else None)
        st_mp0_mean   = (torch.stack(st_mp0_list,  dim=0).mean(dim=0)
                         if len(st_mp0_list)  > 0 else None)

        # ----------------------------------------------------------------
        # Compute per-object attention spotlights
        # ----------------------------------------------------------------
        spotlight_maps: List[Tuple[Tuple[int,int], np.ndarray]] = []
        if q_t is not None and st_keys is not None and len(st_keys) > 0:
            q_single  = q_t[batch_idx]                          # [key_dim, H, W]
            keys_single = [k[batch_idx] for k in st_keys]      # list of [key_dim, H, W]
            obj_positions = self._get_top_object_positions(
                mp0, k=self.attn_top_k_objects
            )
            for pos in obj_positions:
                attn_map = self._compute_attention_map(pos, q_single, keys_single)
                if attn_map is not None:
                    spotlight_maps.append((pos, attn_map))

        # ----------------------------------------------------------------
        # Layout: 2 rows of 3 (existing) + 1 row of N spotlights
        # ----------------------------------------------------------------
        n_spots   = len(spotlight_maps)
        n_rows    = 3 if n_spots > 0 else 2
        n_cols    = max(3, n_spots)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        # Make axes always indexable as [row][col]
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        def _show(ax, data, title, cmap="viridis"):
            if isinstance(data, torch.Tensor):
                data = self._to_cpu_2d(data)
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        def _hide(ax):
            ax.axis("off")

        # Row 0: bev_energy | mfused_energy | abs_mean(mfused-bev)
        _show(axes[0, 0], bev_e,  "bev_energy")
        _show(axes[0, 1], mf_e,   "mfused_energy")
        _show(axes[0, 2], diff,   "abs_mean(mfused-bev)")
        for col in range(3, n_cols):
            _hide(axes[0, col])

        # Row 1: mp_ch0 | st_usage_mean | st_mp0_mean
        _show(axes[1, 0], mp0, "mp_ch0")
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
            h_bev, w_bev = mp0.shape
            for col, ((r, c), attn_map) in enumerate(spotlight_maps):
                ax = axes[2, col]
                ax.imshow(attn_map, cmap="hot")
                ax.set_title(f"attn from query ({r},{c})\nmp0={mp0[r,c]:.3f}", fontsize=8)
                ax.axis("off")
                # Mark the query position with a cyan cross
                ax.plot(c, r, marker="+", color="cyan", markersize=12, markeredgewidth=2)

                # Also annotate row 1 (mp_ch0) with same marker so you can
                # cross-reference which object each spotlight came from
                axes[1, 0].plot(c, r, marker="+", color="red",
                                markersize=10, markeredgewidth=1.5,
                                label=f"obj{col}")

            for col in range(n_spots, n_cols):
                _hide(axes[2, col])

        # ----------------------------------------------------------------
        info = {
            "t":           int(t),
            "step":        int(bank_state.get("step", -1)),
            "st_len":      int(bank_state.get("st_len", -1)),
            "lt_len":      int(bank_state.get("lt_len", -1)),
            "lt_tokens":   int(bank_state.get("lt_tokens", -1)),
            "did_write":   int(bank_state.get("did_write", -1)),
            "mem_sources": int(len(mem_kinds)),
            "mem_kinds":   ",".join(mem_kinds),
            "mp0_mean":    float(mp0.detach().float().mean().item()),
            "mfused_delta": float((mfused_t[batch_idx] - bev_t[batch_idx]).abs().mean().item()),
            "n_spotlights": n_spots,
        }

        out_png = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.png")
        fig.suptitle(str(info), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)

        out_npz = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.npz")
        np.savez_compressed(
            out_npz,
            bev_energy=self._to_cpu_2d(bev_e),
            mfused_energy=self._to_cpu_2d(mf_e),
            diff=self._to_cpu_2d(diff),
            mp0=self._to_cpu_2d(mp0),
            st_usage_mean=(self._to_cpu_2d(st_usage_mean)
                           if st_usage_mean is not None else np.array([])),
        )

    def finish_sequence(self):
        self.seq_dir = None