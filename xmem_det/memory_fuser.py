import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasonNetValueEnc(nn.Module):
    """
    ValueEnc(concat(Mfused_t, Mp_t)) -> v_t   (same HxW, outputs C_bev channels)
    """
    def __init__(self, c_bev: int, mp_ch: int = 7, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_bev + mp_ch, hidden, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden, c_bev, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(c_bev, c_bev, kernel_size=1, bias=True),
        )

    def forward(self, mfused: torch.Tensor, mp: torch.Tensor) -> torch.Tensor:
        if mp.dim() == 3:
            mp = mp.unsqueeze(1)
        # if mp.shape[-2:] != mfused.shape[-2:]:
        #     mp = F.interpolate(mp, size=mfused.shape[-2:], mode="nearest")
        x = torch.cat([mfused, mp], dim=1)
        return self.net(x)


class ReasonNetTemporalBank(nn.Module):
    """
    Online ReasonNet-style temporal bank:

      - q_t = QueryEnc(BEV_t)                        -> key tensor (Ck x H x W)
      - m_t = AttentionRead(q_t, {k_i, v_i}_{i<t})   -> readout (Cv x H x W)
      - Mfused_t = GRU( concat(BEV_t, m_t) )         -> fused BEV (Cv x H x W)
      - v_t = ValueEnc( concat(Mfused_t, Mp_t) )     -> value tensor (Cv x H x W)
      - store: k_t = q_t, v_t = v_t

    Notes:
      - This uses a standard nn.GRU, applied per BEV cell (flatten H*W into batch).
      - The bank stores detached keys/values (no backprop through stored memory).
    """
    def __init__(self, c_bev: int, key_dim: int = 64, max_frames: int = 8):
        super().__init__()
        self.c_bev = int(c_bev)
        self.key_dim = int(key_dim)
        self.max_frames = int(max_frames)

        self.query_enc = nn.Conv2d(self.c_bev, self.key_dim, kernel_size=1, bias=True)

        self.readout_proj = nn.Conv2d(self.c_bev, self.c_bev, kernel_size=1, bias=True)
        self.fuse_proj = nn.Conv2d(self.c_bev * 2, self.c_bev, kernel_size=1, bias=True)

        self.gru = nn.GRU(input_size=self.c_bev, hidden_size=self.c_bev, num_layers=1, batch_first=True)

        self.value_enc = ReasonNetValueEnc(self.c_bev, mp_ch=7, hidden=256)

        self._keys: List[torch.Tensor] = []
        self._vals: List[torch.Tensor] = []
        self._gru_h: Optional[torch.Tensor] = None
        self._hw: Optional[Tuple[int, int]] = None

    def reset(self):
        self._keys = []
        self._vals = []
        self._gru_h = None
        self._hw = None

    def _truncate(self):
        if self.max_frames > 0 and len(self._keys) > self.max_frames:
            self._keys = self._keys[-self.max_frames :]
            self._vals = self._vals[-self.max_frames :]

    def _read_memory(self, qk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, H, W = qk.shape

        if len(self._keys) == 0:
            m = torch.zeros((B, self.c_bev, H, W), device=qk.device, dtype=qk.dtype)
            w = torch.zeros((B, 0, H, W), device=qk.device, dtype=qk.dtype)
            return m, w

        keys_hist = torch.stack(self._keys, dim=1)   # [B, Th, K, H, W]
        vals_hist = torch.stack(self._vals, dim=1)   # [B, Th, C, H, W]

        scale = 1.0 / math.sqrt(float(K))
        scores = (keys_hist * qk.unsqueeze(1)).sum(dim=2) * scale  # [B, Th, H, W]
        w = torch.softmax(scores, dim=1)                            # softmax over time
        m = (w.unsqueeze(2) * vals_hist).sum(dim=1)                 # [B, C, H, W]
        return m, w

    def compute_mfused(self, bev_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = bev_t.shape

        if self._gru_h is None:
            self._gru_h = torch.zeros((1, B * H * W, self.c_bev), device=bev_t.device, dtype=bev_t.dtype)
            self._hw = (H, W)
        else:
            H0, W0 = self._hw
            if (H, W) != (H0, W0):
                raise ValueError(f"GRU state HW {(H0, W0)} != input HW {(H, W)}")

        q_t = self.query_enc(bev_t)                       # [B, K, H, W]
        m_raw, attn_t = self._read_memory(q_t)            # [B, C, H, W], [B, Th, H, W]
        m_t = self.readout_proj(m_raw)                    # optional projection
        x_t = self.fuse_proj(torch.cat([bev_t, m_t], 1))   # [B, C, H, W]

        x_seq = x_t.permute(0, 2, 3, 1).reshape(B * H * W, 1, self.c_bev)  # [BHW, 1, C]
        out_seq, h_new = self.gru(x_seq, self._gru_h)                      # one-step update
        self._gru_h = h_new

        mfused_t = out_seq[:, -1].reshape(B, H, W, self.c_bev).permute(0, 3, 1, 2).contiguous()

        dbg = {
            "q_t": q_t,
            "m_t": m_t,
            "attn_t": attn_t,
            "mfused_t": mfused_t,
        }
        return mfused_t, dbg

    def update_bank(self, q_t: torch.Tensor, mfused_t: torch.Tensor, mp_t: torch.Tensor) -> torch.Tensor:
        v_t = self.value_enc(mfused_t, mp_t)  # [B, C, H, W]
        self._keys.append(q_t.detach())
        self._vals.append(v_t.detach())
        self._truncate()
        return v_t
