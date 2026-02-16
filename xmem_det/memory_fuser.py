from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ConvGRU cell (spatial: operates on [B, C, H, W] directly)
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """
    Single ConvGRU cell following TimePillars Eq.(1-3):

        [r(t), z(t)] = σ( [Wr, Wz] * [h(t-1), x(t)] + [br, bz] )
        h̃(t)        = tanh( Wh * [r(t) · h(t-1), x(t)] + bh )
        h(t)        = (1 - z(t)) · h(t-1) + z(t) · h̃(t)

    All convolutions are 3×3 to preserve spatial structure.
    Input x  : [B, input_dim,  H, W]
    Hidden h : [B, hidden_dim, H, W]
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2

        # Shared conv for reset+update gates (2×hidden_dim output channels)
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

        # Candidate hidden state conv
        self.candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Returns new hidden state h(t) with shape [B, hidden_dim, H, W]."""
        xh = torch.cat([x, h], dim=1)                  # [B, C_in + C_hid, H, W]

        rz = torch.sigmoid(self.gates(xh))              # [B, 2*C_hid, H, W]
        r, z = rz.chunk(2, dim=1)                       # each [B, C_hid, H, W]

        xrh = torch.cat([x, r * h], dim=1)              # [B, C_in + C_hid, H, W]
        h_cand = torch.tanh(self.candidate(xrh))        # [B, C_hid, H, W]

        h_new = (1.0 - z) * h + z * h_cand             # [B, C_hid, H, W]
        return h_new


# ---------------------------------------------------------------------------
# Value encoder
# ---------------------------------------------------------------------------

class ReasonNetValueEnc(nn.Module):
    """
    Value encoder:  v_t = ValueEnc(concat(Mfused_t, Mp_t))
    Mp_t is a 7-channel BEV map prediction.
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
        x = torch.cat([mfused, mp], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Main temporal bank
# ---------------------------------------------------------------------------

class ReasonNetTemporalBank(nn.Module):
    """
    Paper-close memory bank with ConvGRU replacing the standard GRU.

    Key differences from the flat-GRU version
    ------------------------------------------
    * `self.gru` is now a `ConvGRUCell` that operates on [B, C, H, W] tensors.
    * Hidden state `_gru_h` is [B, C, H, W] instead of [1, B*H*W, C].
    * The fusion loop steps through Th+1 spatial maps sequentially instead of
      reshaping everything to [B*H*W, Th+1, C].
    * No intermediate `.reshape(b*h*w, ...)` / `.permute` gymnastics needed.

    Everything else (short-term / long-term bank, attention read, update
    stride tau, LT selection) is unchanged.
    """

    def __init__(
        self,
        c_bev: int,
        key_dim: int = 64,

        ts: int = 4,
        tl: int = 2,
        tau: int = 2,

        long_frame_tokens: int = 2048,

        obj_prob_thresh: float = 0.3,
        topk_usage: int = 512,
        max_from_discard: int = 2048,

        q_chunk: int = 512,

        # ConvGRU-specific
        convgru_kernel: int = 3,
    ):
        super().__init__()
        self.c_bev   = int(c_bev)
        self.key_dim = int(key_dim)

        self.ts  = int(ts)
        self.tl  = int(tl)
        self.tau = int(tau)

        self.long_frame_tokens = int(long_frame_tokens)

        self.obj_prob_thresh  = float(obj_prob_thresh)
        self.topk_usage       = int(topk_usage)
        self.max_from_discard = int(max_from_discard)

        self.q_chunk = int(q_chunk)

        # Query encoder: q_t = Conv1x1(BEV_t)
        self.query_enc = nn.Conv2d(self.c_bev, self.key_dim, kernel_size=1, bias=True)

        # ---- ConvGRU replaces nn.GRU ----------------------------------------
        # Input  to cell: one [B, C_bev, H, W] frame at a time
        # Hidden state  : [B, C_bev, H, W]
        self.gru = ConvGRUCell(
            input_dim=self.c_bev,
            hidden_dim=self.c_bev,
            kernel_size=convgru_kernel,
        )
        # LayerNorm on the final fused output (applied over channel dim)
        self.gru_output_norm = nn.GroupNorm(num_groups=1, num_channels=self.c_bev)
        # ---------------------------------------------------------------------

        self.value_enc = ReasonNetValueEnc(self.c_bev, mp_ch=7, hidden=256)

        # ====== State ======
        self._st_keys:  List[torch.Tensor] = []
        self._st_vals:  List[torch.Tensor] = []
        self._st_mp0:   List[torch.Tensor] = []
        self._st_usage: List[torch.Tensor] = []

        self._lt_keys: List[torch.Tensor] = []
        self._lt_vals: List[torch.Tensor] = []
        self._lt_fill: List[int]          = []

        self._gru_h: Optional[torch.Tensor] = None   # [B, C, H, W]
        self._hw:    Optional[Tuple[int, int]] = None
        self._step:  int = 0

    # ------------------------------------------------------------------
    def reset(self):
        self._st_keys  = []
        self._st_vals  = []
        self._st_mp0   = []
        self._st_usage = []

        self._lt_keys = []
        self._lt_vals = []
        self._lt_fill = []

        self._gru_h = None
        self._hw    = None
        self._step  = 0

    # -------- Long-term frame management (unchanged) ------------------

    def _ensure_long_frame(self, b: int, device, dtype):
        if self.tl <= 0:
            return
        if len(self._lt_keys) == 0:
            self._lt_keys.append(torch.empty(b, 0, self.key_dim, device=device, dtype=dtype))
            self._lt_vals.append(torch.empty(b, 0, self.c_bev,   device=device, dtype=dtype))
            self._lt_fill.append(0)
            return
        if self._lt_fill[-1] >= self.long_frame_tokens:
            self._lt_keys.append(torch.empty(b, 0, self.key_dim, device=device, dtype=dtype))
            self._lt_vals.append(torch.empty(b, 0, self.c_bev,   device=device, dtype=dtype))
            self._lt_fill.append(0)
            if len(self._lt_keys) > self.tl:
                self._lt_keys = self._lt_keys[-self.tl:]
                self._lt_vals = self._lt_vals[-self.tl:]
                self._lt_fill = self._lt_fill[-self.tl:]

    def _append_long_tokens(self, k_sel: torch.Tensor, v_sel: torch.Tensor):
        if self.tl <= 0:
            return
        b, n, _ = k_sel.shape
        if n == 0:
            return
        device, dtype = k_sel.device, k_sel.dtype
        ptr, n_left = 0, n
        while n_left > 0:
            self._ensure_long_frame(b, device, dtype)
            fill  = self._lt_fill[-1]
            space = self.long_frame_tokens - fill
            take  = min(space, n_left)
            self._lt_keys[-1] = torch.cat([self._lt_keys[-1], k_sel[:, ptr:ptr + take]], dim=1)
            self._lt_vals[-1] = torch.cat([self._lt_vals[-1], v_sel[:, ptr:ptr + take]], dim=1)
            self._lt_fill[-1] = self._lt_keys[-1].shape[1]
            ptr    += take
            n_left -= take
            if self._lt_fill[-1] >= self.long_frame_tokens:
                self._ensure_long_frame(b, device, dtype)

    def _select_for_long_term(self, st_idx: int):
        if self.tl <= 0:
            return
        k_map = self._st_keys[st_idx]
        v_map = self._st_vals[st_idx]
        mp0   = self._st_mp0[st_idx]
        usage = self._st_usage[st_idx]
        b, _, h, w = k_map.shape
        hw = h * w

        k_flat    = k_map.reshape(b, self.key_dim, hw).permute(0, 2, 1)
        v_flat    = v_map.reshape(b, self.c_bev,   hw).permute(0, 2, 1)
        mp0_flat  = mp0.reshape(b, hw)
        usage_flat = usage.reshape(b, hw)

        obj_mask  = mp0_flat > self.obj_prob_thresh
        k_top     = min(self.topk_usage, hw)
        topk_idx  = torch.topk(usage_flat, k=k_top, dim=1, largest=True).indices
        topk_mask = torch.zeros_like(obj_mask, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        sel_mask  = obj_mask | topk_mask
        sel_idx   = sel_mask.nonzero(as_tuple=False)

        if sel_idx.numel() == 0:
            return

        per_b: List[torch.Tensor] = []
        for bi in range(b):
            idx_b = sel_idx[sel_idx[:, 0] == bi][:, 1]
            if idx_b.numel() == 0:
                per_b.append(idx_b); continue
            if idx_b.numel() > self.max_from_discard:
                u    = usage_flat[bi, idx_b]
                keep = torch.topk(u, k=self.max_from_discard, largest=True).indices
                idx_b = idx_b[keep]
            per_b.append(idx_b)

        nmax = max(int(x.numel()) for x in per_b)
        if nmax == 0:
            return

        k_sel = torch.zeros(b, nmax, self.key_dim, device=k_map.device, dtype=k_map.dtype)
        v_sel = torch.zeros(b, nmax, self.c_bev,   device=v_map.device, dtype=v_map.dtype)
        for bi in range(b):
            idx_b = per_b[bi]
            if idx_b.numel() == 0:
                continue
            k_sel[bi, :idx_b.numel()] = k_flat[bi, idx_b]
            v_sel[bi, :idx_b.numel()] = v_flat[bi, idx_b]

        self._append_long_tokens(k_sel, v_sel)

    # -------- Memory read (unchanged) ----------------------------------

    def _dist_sq_block(self, q_blk: torch.Tensor, k_all: torch.Tensor) -> torch.Tensor:
        q2  = (q_blk * q_blk).sum(dim=2, keepdim=True)
        k2  = (k_all * k_all).sum(dim=2).unsqueeze(1)
        dot = torch.bmm(q_blk, k_all.transpose(1, 2))
        return (q2 + k2 - 2.0 * dot).clamp_min(0.0)

    def _read_one_memory(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, hw_q, _ = q_flat.shape
        step = self.q_chunk if self.q_chunk > 0 else hw_q

        out_blocks:   List[torch.Tensor] = []
        usage_blocks: List[torch.Tensor] = []

        for s in range(0, hw_q, step):
            e     = min(s + step, hw_q)
            q_blk = q_flat[:, s:e, :]
            dist  = self._dist_sq_block(q_blk, k_flat)
            denom = dist.sum(dim=2, keepdim=True).add(1e-8)
            S     = dist / denom
            out_blocks.append(torch.bmm(S, v_flat))
            usage_blocks.append(S.detach().sum(dim=1))

        out   = torch.cat(out_blocks, dim=1)
        usage = torch.stack(usage_blocks, dim=0).sum(dim=0)
        return out, usage

    def _collect_memory(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        keys:  List[torch.Tensor] = []
        vals:  List[torch.Tensor] = []
        kinds: List[str]          = []

        for i in range(len(self._st_keys)):
            b, _, h, w = self._st_keys[i].shape
            hw     = h * w
            k_flat = self._st_keys[i].reshape(b, self.key_dim, hw).permute(0, 2, 1)
            v_flat = self._st_vals[i].reshape(b, self.c_bev,   hw).permute(0, 2, 1)
            keys.append(k_flat); vals.append(v_flat); kinds.append("st")

        for j in range(len(self._lt_keys)):
            if self._lt_fill[j] <= 0:
                continue
            keys.append(self._lt_keys[j]); vals.append(self._lt_vals[j]); kinds.append("lt")

        return keys, vals, kinds

    # -------- Top-level temporal fusion (ConvGRU version) --------------

    def compute_mfused(
        self, bev_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, c, h, w = bev_t.shape

        # Initialise ConvGRU hidden state [B, C, H, W]
        if self._gru_h is None:
            self._gru_h = torch.zeros(b, self.c_bev, h, w,
                                      device=bev_t.device, dtype=bev_t.dtype)
            self._hw = (h, w)
        else:
            h0, w0 = self._hw
            if (h, w) != (h0, w0):
                raise ValueError(f"ConvGRU state HW mismatch: {(h0, w0)} != {(h, w)}")

        # Query encoding q_t  [B, K, H, W]
        q_t   = self.query_enc(bev_t)
        hw    = h * w
        q_flat = q_t.reshape(b, self.key_dim, hw).permute(0, 2, 1)  # [B, HW, K]

        # Read from memory
        mem_keys, mem_vals, mem_kinds = self._collect_memory()

        m_frames:   List[torch.Tensor] = []
        attn_diag:  List[torch.Tensor] = []

        for mi, (k_flat, v_flat, kind) in enumerate(zip(mem_keys, mem_vals, mem_kinds)):
            m_flat, usage = self._read_one_memory(q_flat, k_flat, v_flat)  # [B, HW, C]
            m_map = m_flat.permute(0, 2, 1).reshape(b, self.c_bev, h, w)   # [B, C, H, W]
            m_frames.append(m_map)

            if kind == "st":
                self._st_usage[mi] = self._st_usage[mi] + usage.reshape(b, h, w)

            attn_diag.append(torch.zeros(b, h, w, device=bev_t.device, dtype=bev_t.dtype))

        # ---- ConvGRU sequential fusion ------------------------------------
        # Step through: bev_t first, then each readout frame M_1 … M_Th
        # Hidden state starts from the stored _gru_h across timesteps.
        h_t = self._gru_h.detach()                                  # [B, C, H, W]

        # Frame 0: current BEV features
        h_t = self.gru(bev_t, h_t)

        # Frames 1…Th: memory readout maps
        for m_map in m_frames:
            h_t = self.gru(m_map, h_t)

        self._gru_h = h_t.detach()

        # Normalise the final fused output
        mfused_t = self.gru_output_norm(h_t)                       # [B, C, H, W]
        # ------------------------------------------------------------------

        dbg = {
            "q_t":       q_t,
            "mfused_t":  mfused_t,
            "m_frames":  m_frames,
            "mem_kinds": mem_kinds,
        }
        return mfused_t, dbg

    # -------- Bank update (unchanged) ----------------------------------

    def update_bank(
        self,
        q_t:      torch.Tensor,
        mfused_t: torch.Tensor,
        mp_t:     torch.Tensor,
    ) -> torch.Tensor:
        self._step += 1
        if self.tau > 1 and (self._step % self.tau) != 0:
            return torch.zeros((), device=mfused_t.device, dtype=mfused_t.dtype)

        k_t   = q_t.detach()
        v_t   = self.value_enc(mfused_t, mp_t).detach()
        mp0   = mp_t[:, 0:1].detach()
        b, _, h_k, w_k = k_t.shape
        usage0 = torch.zeros(b, h_k, w_k, device=k_t.device, dtype=k_t.dtype)

        self._st_keys.append(k_t)
        self._st_vals.append(v_t)
        self._st_mp0.append(mp0)
        self._st_usage.append(usage0)

        while len(self._st_keys) > self.ts:
            self._select_for_long_term(0)
            self._st_keys.pop(0)
            self._st_vals.pop(0)
            self._st_mp0.pop(0)
            self._st_usage.pop(0)

        return v_t

    # -------- Debug helpers (unchanged) --------------------------------

    def get_debug_state(self) -> Dict[str, Any]:
        lt_tokens = int(sum(self._lt_fill)) if self._lt_fill else 0
        return {
            "step":      int(self._step),
            "st_len":    int(len(self._st_keys)),
            "lt_len":    int(len(self._lt_keys)),
            "lt_tokens": lt_tokens,
            "lt_fills":  [int(x) for x in self._lt_fill],
            "did_write": int(1 if (self.tau <= 1 or (self._step % self.tau) == 0) else 0),
        }

    def get_debug_maps(self, batch_idx: int = 0) -> Dict[str, List[torch.Tensor]]:
        st_usage = [u[batch_idx].detach()     for u in self._st_usage]
        st_mp0   = [m[batch_idx, 0].detach()  for m in self._st_mp0]
        return {"st_usage": st_usage, "st_mp0": st_mp0}