import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


class TemporalDebugger:
    def __init__(self, save_dir: str, log_every: int = 1, max_batches: int = 1):
        self.save_dir = save_dir
        self.log_every = int(log_every)
        self.max_batches = int(max_batches)
        self.seq_dir: Optional[str] = None

    def start_sequence(self, seq_name: str):
        self.seq_dir = os.path.join(self.save_dir, seq_name)
        os.makedirs(self.seq_dir, exist_ok=True)

    def _to_cpu_2d(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    def _bev_energy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(x, dim=0)

    def log_timestep(
        self,
        t: int,
        bev_t: torch.Tensor,
        mfused_t: torch.Tensor,
        mp_t: torch.Tensor,
        bank_state: Dict[str, Any],
        bank_maps: Dict[str, List[torch.Tensor]],
        mem_kinds: List[str],
        batch_idx: int = 0,
    ):
        if self.seq_dir is None:
            raise RuntimeError("start_sequence() must be called before log_timestep()")
        if self.log_every > 1 and (t % self.log_every) != 0:
            return
        if batch_idx >= self.max_batches:
            return

        bev = bev_t[batch_idx]
        mf = mfused_t[batch_idx]
        mp = mp_t[batch_idx]

        bev_e = self._bev_energy(bev)
        mf_e = self._bev_energy(mf)
        diff = (mf - bev).abs().mean(dim=0)

        mp0 = mp[0]
        mp1 = mp[1] if mp.shape[0] > 1 else mp[0]
        mp2 = mp[2] if mp.shape[0] > 2 else mp[0]

        st_usage_list = bank_maps.get("st_usage", [])
        st_mp0_list = bank_maps.get("st_mp0", [])

        st_usage_mean = None
        if len(st_usage_list) > 0:
            st_usage_mean = torch.stack(st_usage_list, dim=0).mean(dim=0)

        st_mp0_mean = None
        if len(st_mp0_list) > 0:
            st_mp0_mean = torch.stack(st_mp0_list, dim=0).mean(dim=0)

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)

        ax1.imshow(self._to_cpu_2d(bev_e))
        ax1.set_title("bev_energy")
        ax1.axis("off")

        ax2.imshow(self._to_cpu_2d(mf_e))
        ax2.set_title("mfused_energy")
        ax2.axis("off")

        ax3.imshow(self._to_cpu_2d(diff))
        ax3.set_title("abs_mean(mfused-bev)")
        ax3.axis("off")

        ax4.imshow(self._to_cpu_2d(mp0))
        ax4.set_title("mp_ch0")
        ax4.axis("off")

        if st_usage_mean is not None:
            ax5.imshow(self._to_cpu_2d(st_usage_mean))
            ax5.set_title("st_usage_mean")
            ax5.axis("off")
        else:
            ax5.text(0.01, 0.5, "no ST usage", fontsize=12)
            ax5.axis("off")

        if st_mp0_mean is not None:
            ax6.imshow(self._to_cpu_2d(st_mp0_mean))
            ax6.set_title("st_mp0_mean")
            ax6.axis("off")
        else:
            ax6.imshow(self._to_cpu_2d(mp1))
            ax6.set_title("mp_ch1")
            ax6.axis("off")

        info = {
            "t": int(t),
            "step": int(bank_state.get("step", -1)),
            "st_len": int(bank_state.get("st_len", -1)),
            "lt_len": int(bank_state.get("lt_len", -1)),
            "lt_tokens": int(bank_state.get("lt_tokens", -1)),
            "did_write": int(bank_state.get("did_write", -1)),
            "mem_sources": int(len(mem_kinds)),
            "mem_kinds": ",".join(mem_kinds),
            "mp0_mean": float(mp0.detach().float().mean().item()),
            "mfused_delta": float((mfused_t[batch_idx] - bev_t[batch_idx]).abs().mean().item()),
        }

        out_png = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.png")
        fig.suptitle(str(info), fontsize=9)
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
            mp1=self._to_cpu_2d(mp2),
        )

    def finish_sequence(self):
        self.seq_dir = None
