from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from typing import Any, Dict, List

import torch


class ReasonNetValueEnc(nn.Module):
    """
    Value encoder from paper:
      v_t = ValueEnc(concat(Mfused_t, Mp_t))
    Mp_t is 7-channel BEV map prediction.
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


class ReasonNetTemporalBank(nn.Module):
    """
    Paper-close memory bank:

    - Short-term buffer (ST): dense key/value maps for recent frames, capped by Ts.
    - Long-term buffer (LT): selective storage of representative key/value features, capped by Tl.
    - Update stride tau: only push a new ST frame every tau timesteps.

    Memory read implements Eq.1/Eq.2 in a mathematically equivalent way but avoids OOM:
      * Never materialize [B, HWq, HWk, K].
      * Compute squared distances via norms + dot products, optionally in query chunks.
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
    ):
        super().__init__()
        self.c_bev = int(c_bev)
        self.key_dim = int(key_dim)

        # Paper hyperparams (reported in experiments)
        self.ts = int(ts)    # short-term size Ts
        self.tl = int(tl)    # long-term size Tl
        self.tau = int(tau)  # update stride tau

        # Long-term storage capacity: LT stores "tokens" rather than full HxW maps
        self.long_frame_tokens = int(long_frame_tokens)

        # LT selection rules:
        #  (1) object-based: Mp existence prob > threshold
        #  (2) usage-based: keep top-K by accumulated similarity/usage
        self.obj_prob_thresh = float(obj_prob_thresh)
        self.topk_usage = int(topk_usage)
        self.max_from_discard = int(max_from_discard)

        # Query chunk size to bound peak memory. Chunking does not change math.
        self.q_chunk = int(q_chunk)

        # Query encoder: q_t = Conv1x1(BEV_t)
        self.query_enc = nn.Conv2d(self.c_bev, self.key_dim, kernel_size=1, bias=True)

        # GRU fuse: paper says concatenate [BEV_t, M_1..M_Th] then GRU progressive fusion
        self.gru = nn.GRU(
            input_size=self.c_bev,
            hidden_size=self.c_bev,
            num_layers=1,
            batch_first=True
        )

        self.gru_output_norm = nn.LayerNorm(c_bev)

        self.value_enc = ReasonNetValueEnc(self.c_bev, mp_ch=7, hidden=256)

        # ====== State ======
        # Short-term dense frames (each is full [B, *, H, W])
        self._st_keys: List[torch.Tensor] = []   # each: [B, K, H, W]
        self._st_vals: List[torch.Tensor] = []   # each: [B, C, H, W]
        self._st_mp0:  List[torch.Tensor] = []   # each: [B, 1, H, W] objectness/existence
        self._st_usage: List[torch.Tensor] = []  # each: [B, H, W] accumulated usage score

        # Long-term token frames (each is [B, N, *])
        # We store LT in "frames" for similarity with paper's Tl, but each LT frame is a token set.
        self._lt_keys: List[torch.Tensor] = []   # each: [B, Ncap, K]
        self._lt_vals: List[torch.Tensor] = []   # each: [B, Ncap, C]
        self._lt_fill: List[int] = []            # how many tokens are used in each LT frame

        self._gru_h: Optional[torch.Tensor] = None
        self._hw: Optional[Tuple[int, int]] = None
        self._step: int = 0

    def reset(self):
        self._st_keys = []
        self._st_vals = []
        self._st_mp0 = []
        self._st_usage = []

        self._lt_keys = []
        self._lt_vals = []
        self._lt_fill = []

        self._gru_h = None
        self._hw = None
        self._step = 0

    # -------- Long-term frame management --------
    def _ensure_long_frame(self, b: int, device, dtype):
        if self.tl <= 0:
            return
        if len(self._lt_keys) == 0:
            self._lt_keys.append(torch.empty(b, 0, self.key_dim, device=device, dtype=dtype))
            self._lt_vals.append(torch.empty(b, 0, self.c_bev, device=device, dtype=dtype))
            self._lt_fill.append(0)
            return
        if self._lt_fill[-1] >= self.long_frame_tokens:
            self._lt_keys.append(torch.empty(b, 0, self.key_dim, device=device, dtype=dtype))
            self._lt_vals.append(torch.empty(b, 0, self.c_bev, device=device, dtype=dtype))
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

        device = k_sel.device
        dtype = k_sel.dtype

        ptr = 0
        n_left = n

        while n_left > 0:
            self._ensure_long_frame(b, device, dtype)
            fill = self._lt_fill[-1]
            space = self.long_frame_tokens - fill
            take = min(space, n_left)

            k_take = k_sel[:, ptr:ptr + take]
            v_take = v_sel[:, ptr:ptr + take]

            self._lt_keys[-1] = torch.cat([self._lt_keys[-1], k_take], dim=1)
            self._lt_vals[-1] = torch.cat([self._lt_vals[-1], v_take], dim=1)
            self._lt_fill[-1] = self._lt_keys[-1].shape[1]

            ptr += take
            n_left -= take

            if self._lt_fill[-1] >= self.long_frame_tokens:
                self._ensure_long_frame(b, device, dtype)

    def _select_for_long_term(self, st_idx: int):
        """
        Called when a short-term frame is about to be discarded (ST size > Ts).
        This implements the paper's idea:
          - store "important/representative" features in long-term
          - object-based: Mp existence prob (Mp channel 0)
          - usage-based: top-K by accumulated usage score

        We convert the dense ST map to tokens by selecting spatial indices.
        """
        if self.tl <= 0:
            return

        k_map = self._st_keys[st_idx]    # [B, K, H, W]
        v_map = self._st_vals[st_idx]    # [B, C, H, W]
        mp0   = self._st_mp0[st_idx]     # [B, 1, H, W]
        usage = self._st_usage[st_idx]   # [B, H, W]

        b, _, h, w = k_map.shape
        hw = h * w

        # Flatten dense maps into token candidate lists [B, HW, *]
        k_flat = k_map.reshape(b, self.key_dim, hw).permute(0, 2, 1)   # [B, HW, K]
        v_flat = v_map.reshape(b, self.c_bev, hw).permute(0, 2, 1)     # [B, HW, C]
        mp0_flat = mp0.reshape(b, hw)                                  # [B, HW]
        usage_flat = usage.reshape(b, hw)                              # [B, HW]

        # Object-based selection: existence prob threshold
        obj_mask = mp0_flat > self.obj_prob_thresh                     # [B, HW]

        # Usage-based selection: keep top-K by accumulated usage
        k_top = min(self.topk_usage, hw)
        topk_idx = torch.topk(usage_flat, k=k_top, dim=1, largest=True).indices  # [B, k_top]
        topk_mask = torch.zeros_like(obj_mask, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)

        # Union of both criteria
        sel_mask = obj_mask | topk_mask                                 # [B, HW]
        sel_idx = sel_mask.nonzero(as_tuple=False)                      # [Nsel_total, 2] (batch, index)

        if sel_idx.numel() == 0:
            return

        # Cap per-batch selected tokens so LT doesn't explode when many cells pass threshold
        per_b: List[torch.Tensor] = []
        for bi in range(b):
            idx_b = sel_idx[sel_idx[:, 0] == bi][:, 1]                  # [n_b]
            if idx_b.numel() == 0:
                per_b.append(idx_b)
                continue
            if idx_b.numel() > self.max_from_discard:
                u = usage_flat[bi, idx_b]
                keep = torch.topk(u, k=self.max_from_discard, largest=True).indices
                idx_b = idx_b[keep]
            per_b.append(idx_b)

        nmax = max(int(x.numel()) for x in per_b)
        if nmax == 0:
            return

        # Pack variable selection into dense [B, nmax, *] (padding where needed)
        k_sel = torch.zeros(b, nmax, self.key_dim, device=k_map.device, dtype=k_map.dtype)
        v_sel = torch.zeros(b, nmax, self.c_bev, device=v_map.device, dtype=v_map.dtype)
        for bi in range(b):
            idx_b = per_b[bi]
            if idx_b.numel() == 0:
                continue
            k_sel[bi, :idx_b.numel()] = k_flat[bi, idx_b]
            v_sel[bi, :idx_b.numel()] = v_flat[bi, idx_b]

        self._append_long_tokens(k_sel, v_sel)

    def _dist_sq_block(self, q_blk: torch.Tensor, k_all: torch.Tensor) -> torch.Tensor:
        q2 = (q_blk * q_blk).sum(dim=2, keepdim=True)
        k2 = (k_all * k_all).sum(dim=2).unsqueeze(1)
        dot = torch.bmm(q_blk, k_all.transpose(1, 2))
        dist = q2 + k2 - 2.0 * dot
        dist = dist.clamp_min(0.0)
        return dist

    def _read_one_memory(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, hw_q, _ = q_flat.shape
        _, hw_k, _ = k_flat.shape
        step = self.q_chunk if self.q_chunk > 0 else hw_q

        out_blocks: List[torch.Tensor] = []
        usage_blocks: List[torch.Tensor] = []

        for s in range(0, hw_q, step):
            e = min(s + step, hw_q)
            q_blk = q_flat[:, s:e, :]

            dist = self._dist_sq_block(q_blk, k_flat)               # [B, Q, M]
            denom = dist.sum(dim=2, keepdim=True).add(1e-8)         # no in-place
            S = dist / denom                                        # [B, Q, M]

            out_blocks.append(torch.bmm(S, v_flat))                 # [B, Q, C]

            # usage is NOT part of training signal; detach so it cannot participate in autograd
            usage_blocks.append(S.detach().sum(dim=1))              # [B, M]

        out = torch.cat(out_blocks, dim=1)                          # [B, HWq, C]
        usage = torch.stack(usage_blocks, dim=0).sum(dim=0)         # [B, M]
        return out, usage


    def _collect_memory(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        keys: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        kinds: List[str] = []

        for i in range(len(self._st_keys)):
            b, _, h, w = self._st_keys[i].shape
            hw = h * w
            k_flat = self._st_keys[i].reshape(b, self.key_dim, hw).permute(0, 2, 1)
            v_flat = self._st_vals[i].reshape(b, self.c_bev, hw).permute(0, 2, 1)
            keys.append(k_flat)
            vals.append(v_flat)
            kinds.append("st")

        for j in range(len(self._lt_keys)):
            fill = self._lt_fill[j]
            if fill <= 0:
                continue
            keys.append(self._lt_keys[j])
            vals.append(self._lt_vals[j])
            kinds.append("lt")

        return keys, vals, kinds


    # -------- Top-level temporal fusion --------
    def compute_mfused(self, bev_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, h, w = bev_t.shape

        # Init GRU hidden state for each new sequence (bank.reset() must be called per sequence)
        if self._gru_h is None:
            self._gru_h = torch.zeros((1, b * h * w, self.c_bev), device=bev_t.device, dtype=bev_t.dtype)
            self._hw = (h, w)
        else:
            h0, w0 = self._hw
            if (h, w) != (h0, w0):
                raise ValueError(f"GRU state HW mismatch: {(h0, w0)} != {(h, w)}")

        # Query encoding q_t
        q_t = self.query_enc(bev_t)                             # [B, K, H, W]
        hw = h * w
        q_flat = q_t.reshape(b, self.key_dim, hw).permute(0, 2, 1)  # [B, HW, K]

        # Read from all memory sources
        mem_keys, mem_vals, mem_kinds = self._collect_memory()

        m_frames: List[torch.Tensor] = []
        attn_diag: List[torch.Tensor] = []

        if len(mem_keys) > 0:
            for mi, (k_flat, v_flat, kind) in enumerate(zip(mem_keys, mem_vals, mem_kinds)):
                m_flat, usage = self._read_one_memory(q_flat, k_flat, v_flat)         # [B, HW, C], [B, M]
                m_map = m_flat.permute(0, 2, 1).reshape(b, self.c_bev, h, w)          # [B, C, H, W]
                m_frames.append(m_map)

                # Accumulate usage ONLY for ST frames because LT selection uses discarded ST frames.
                if kind == "st":
                    self._st_usage[mi] = self._st_usage[mi] + usage.reshape(b, h, w)

                # Debug attention is not well-defined for LT tokens. For ST we can approximate a diag view.
                if kind == "st":
                    attn_diag.append(torch.zeros(b, h, w, device=bev_t.device, dtype=bev_t.dtype))
                else:
                    attn_diag.append(torch.zeros(b, h, w, device=bev_t.device, dtype=bev_t.dtype))

        # Concatenate [bev_t, M_1..M_Th] as in paper
        if len(m_frames) == 0:
            m_prime = bev_t
        else:
            m_prime = torch.cat([bev_t] + m_frames, dim=1)        # [B, C*(Th+1), H, W]

        # GRU expects sequence length Th+1 with feature dim C
        th_plus_1 = m_prime.shape[1] // self.c_bev
        m_seq = m_prime.permute(0, 2, 3, 1).reshape(b * h * w, th_plus_1, self.c_bev)  # [B*HW, Th+1, C]

        if self._gru_h is not None:
            self._gru_h = self._gru_h.detach()
        out_seq, h_new = self.gru(m_seq, self._gru_h)
        
        self._gru_h = h_new.detach()

        mfused_t = out_seq[:, -1, :].reshape(b, h, w, self.c_bev).permute(0, 3, 1, 2)  # [B, C, H, W]

        attn_t = torch.stack(attn_diag, dim=1) if len(attn_diag) > 0 else torch.zeros((b, 0, h, w), device=bev_t.device, dtype=bev_t.dtype)
        m_t_debug = m_frames[0] if len(m_frames) > 0 else torch.zeros_like(bev_t)

        dbg = {
            "q_t": q_t,
            "mfused_t": mfused_t,
            "m_frames": m_frames,
            "mem_kinds": mem_kinds,
        }
        return mfused_t, dbg


    def update_bank(self, q_t: torch.Tensor, mfused_t: torch.Tensor, mp_t: torch.Tensor) -> torch.Tensor:
        """
        Paper: update memory every tau frames.
        Store new ST key/value maps and eventually move oldest ST into LT using selection rules.
        """
        self._step += 1
        if self.tau > 1 and (self._step % self.tau) != 0:
            return torch.zeros((), device=mfused_t.device, dtype=mfused_t.dtype)

        # Key = q_t directly copied (paper statement)
        k_t = q_t.detach()                                       # [B, K, H, W]

        # Value = ValueEnc(concat(Mfused, Mp))
        v_t = self.value_enc(mfused_t, mp_t).detach()            # [B, C, H, W]

        # Store objectness (Mp channel 0) for object-based LT selection
        mp0 = mp_t[:, 0:1].detach()                              # [B, 1, H, W]

        # Initialize usage map for this ST frame
        b, _, h, w = k_t.shape
        usage0 = torch.zeros(b, h, w, device=k_t.device, dtype=k_t.dtype)

        # Append to ST
        self._st_keys.append(k_t)
        self._st_vals.append(v_t)
        self._st_mp0.append(mp0)
        self._st_usage.append(usage0)

        # If ST exceeds Ts, discard oldest, but before that, push selected tokens into LT
        while len(self._st_keys) > self.ts:
            self._select_for_long_term(0)
            self._st_keys.pop(0)
            self._st_vals.pop(0)
            self._st_mp0.pop(0)
            self._st_usage.pop(0)

        return v_t

    def get_debug_state(self) -> Dict[str, Any]:
        lt_tokens = int(sum(self._lt_fill)) if len(self._lt_fill) > 0 else 0
        return {
            "step": int(self._step),
            "st_len": int(len(self._st_keys)),
            "lt_len": int(len(self._lt_keys)),
            "lt_tokens": lt_tokens,
            "lt_fills": [int(x) for x in self._lt_fill],
            "did_write": int(1 if (self.tau <= 1 or (self._step % self.tau) == 0) else 0),
        }


    def get_debug_maps(self, batch_idx: int = 0) -> Dict[str, List[torch.Tensor]]:
        st_usage = []
        st_mp0 = []
        for u in self._st_usage:
            st_usage.append(u[batch_idx].detach())
        for m in self._st_mp0:
            st_mp0.append(m[batch_idx, 0].detach())
        return {"st_usage": st_usage, "st_mp0": st_mp0}