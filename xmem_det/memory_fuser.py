from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ConvGRU cell
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """
    Single ConvGRU cell following TimePillars Eq.(1-3).

    This is the SPATIAL variant — convolutions instead of linear projections,
    so the hidden state is [B, C, H, W] instead of [B, C].
    This preserves spatial structure across temporal steps, unlike a flat GRU
    which would require flattening H×W into the batch dimension.

    Equations:
        r(t) = σ( Conv([h(t-1), x(t)]) )          reset gate
        z(t) = σ( Conv([h(t-1), x(t)]) )          update gate
        h̃(t) = tanh( Conv([r(t)·h(t-1), x(t)]) ) candidate state
        h(t) = (1 - z(t))·h(t-1) + z(t)·h̃(t)    new hidden state

    z close to 1 → accept new input (forget old state)
    z close to 0 → keep old state (ignore new input)

    Input x  : [B, input_dim,  H, W]
    Hidden h : [B, hidden_dim, H, W]
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding so H, W are preserved

        # Single conv computes BOTH reset and update gates simultaneously
        # by producing 2×hidden_dim channels, then splitting.
        # Input: concat(x, h) → [B, input_dim + hidden_dim, H, W]
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,           # first half = reset gate, second = update gate
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

        # Candidate hidden state — uses reset-gated h(t-1)
        self.candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim,  H, W]  — current input (e.g. a memory readout or bev_t)
        h: [B, hidden_dim, H, W]  — previous hidden state
        Returns new hidden state [B, hidden_dim, H, W]
        """
        xh = torch.cat([x, h], dim=1)                   # [B, C_in + C_hid, H, W]

        rz = torch.sigmoid(self.gates(xh))               # [B, 2*C_hid, H, W]
        r, z = rz.chunk(2, dim=1)                        # each [B, C_hid, H, W]

        # Reset gate r controls how much of h(t-1) the candidate sees.
        # r≈0 → candidate ignores past hidden state (fresh start)
        # r≈1 → candidate uses full past hidden state
        xrh    = torch.cat([x, r * h], dim=1)            # [B, C_in + C_hid, H, W]
        h_cand = torch.tanh(self.candidate(xrh))         # [B, C_hid, H, W]

        # Update gate z interpolates between old state and candidate
        h_new = (1.0 - z) * h + z * h_cand              # [B, C_hid, H, W]
        return h_new


# ---------------------------------------------------------------------------
# Value encoder
# ---------------------------------------------------------------------------

class ReasonNetValueEnc(nn.Module):
    """
    Encodes (mfused_t, Mp_t) into the value v_t stored in the memory bank.

    v_t = ValueEnc( concat(mfused_t, Mp_t) )
    """

    def __init__(self, c_bev: int, mp_ch: int = 7, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 1×1 conv: cheap channel mixing to project (c_bev + 7) → hidden
            nn.Conv2d(c_bev + mp_ch, hidden, kernel_size=1, bias=True),
            nn.ReLU(),
            # 3×3 conv: spatial context (neighbouring cells inform encoding)
            nn.Conv2d(hidden, c_bev, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            # 1×1 conv: final projection back to c_bev (same dim as keys)
            nn.Conv2d(c_bev, c_bev, kernel_size=1, bias=True),
        )

    def forward(self, mfused: torch.Tensor, mp: torch.Tensor) -> torch.Tensor:
        if mp.dim() == 3:
            mp = mp.unsqueeze(1)              # handle accidental [B, H, W] input
        x = torch.cat([mfused, mp], dim=1)   # [B, c_bev + 7, H, W]
        return self.net(x)                    # [B, c_bev, H, W]


# ---------------------------------------------------------------------------
# Main temporal bank
# ---------------------------------------------------------------------------

class ReasonNetTemporalBank(nn.Module):
    """
    XMem-style memory bank with a persistent ConvGRU hidden state.

    This is a TimePillars-style architecture (persistent GRU state across real
    timesteps), NOT ReasonNet's stateless GRU (which resets to zero each frame
    and uses the GRU purely as a memory-slot aggregator).

    Two sources of temporal context per frame:
      1. _gru_h: ConvGRU hidden state that carries compressed history forward
                 across real timesteps via truncated BPTT (detached between steps)
      2. Memory bank: explicit key/value store read via L2 attention,
                 gives direct access to specific past feature maps

    Memory layout:
      Short-term buffer (_st_*): last `ts` stored frames, full spatial resolution.
        Discarded frames are selectively promoted to long-term.
      Long-term buffer (_lt_*): sparse token set (promoted from short-term).
        Stores important/frequently-attended locations indefinitely.
        Capacity: tl frames × long_frame_tokens tokens each.

    Bank is updated every `tau` steps to control memory usage.
    At tau=2 on 10Hz data, we store one frame every ~0.2s.
    """

    def __init__(
        self,
        c_bev: int,
        key_dim: int = 64,
        ts: int = 2,                    # short-term buffer size (frames)
        tl: int = 0,                    # long-term buffer size (frames), 0 = disabled
        tau: int = 2,                   # update bank every tau steps
        long_frame_tokens: int = 2048,  # max tokens per long-term frame
        obj_prob_thresh: float = 0.3,   # existence threshold for LT selection
        topk_usage: int = 512,          # top-K by usage frequency for LT selection
        max_from_discard: int = 2048,   # max tokens promoted from one ST frame
        q_chunk: int = 512,             # chunk size for memory read (memory saving)
        convgru_kernel: int = 3,
    ):
        super().__init__()
        self.c_bev   = int(c_bev)
        self.key_dim = int(key_dim)
        self.ts  = int(ts)
        self.tl  = int(tl)
        self.tau = int(tau)
        self.long_frame_tokens = int(long_frame_tokens)
        self.obj_prob_thresh   = float(obj_prob_thresh)
        self.topk_usage        = int(topk_usage)
        self.max_from_discard  = int(max_from_discard)
        self.q_chunk           = int(q_chunk)

        # Query encoder: projects BEV features to a lower-dim key space.
        # Keys are in R^key_dim, values stay in R^c_bev.
        # Smaller key_dim reduces attention computation cost.
        self.query_enc = nn.Sequential(
            nn.Conv2d(self.c_bev, self.c_bev // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.c_bev // 2, self.key_dim, kernel_size=3, padding=1, bias=True),
        )

        # ConvGRU: persistent hidden state [B, c_bev, H, W].
        # input_dim == hidden_dim == c_bev because both memory readouts
        # and current BEV features are in the same c_bev space.
        self.gru = ConvGRUCell(
            input_dim=self.c_bev,
            hidden_dim=self.c_bev,
            kernel_size=convgru_kernel,
        )

        # GroupNorm after GRU stabilises feature statistics.
        # num_groups=1 = LayerNorm over channel dim (operates on [B, C, H, W]).
        self.gru_output_norm = nn.GroupNorm(num_groups=1, num_channels=self.c_bev)

        self.value_enc = ReasonNetValueEnc(self.c_bev, mp_ch=7, hidden=256)

        # Runtime state (non-parameter) 
        # Short-term: list of tensors, one per stored frame, index 0 = oldest
        self._st_keys:  List[torch.Tensor] = []  # [B, key_dim, H, W] each
        self._st_vals:  List[torch.Tensor] = []  # [B, c_bev,   H, W] each
        self._st_mp0:   List[torch.Tensor] = []  # [B, 1,       H, W] existence ch
        self._st_usage: List[torch.Tensor] = []  # [B, H, W] cumulative attention

        # Long-term: sparse tokens in [B, N, dim] format
        self._lt_keys: List[torch.Tensor] = []
        self._lt_vals: List[torch.Tensor] = []
        self._lt_fill: List[int]          = []   # how many tokens in each LT frame

        # GRU hidden state — persists across real timesteps, reset between sequences
        self._gru_h: Optional[torch.Tensor] = None   # [B, c_bev, H, W]
        self._hw:    Optional[Tuple[int, int]] = None
        self._step:  int = 0  # counts update_bank calls (for tau stride)

    # ------------------------------------------------------------------
    def reset(self):
        """
        Clear all state between sequences
        Must be called at the start of every new keyframe episode
        Failing to call this lets memory from one scene contaminate the next
        """
        self._st_keys  = []
        self._st_vals  = []
        self._st_mp0   = []
        self._st_usage = []
        self._lt_keys  = []
        self._lt_vals  = []
        self._lt_fill  = []
        self._gru_h    = None   # GRU hidden state reset — next frame starts from zeros
        self._hw       = None
        self._step     = 0

    # -------- Long-term frame management ------------------

    def _ensure_long_frame(self, b: int, device, dtype):
        """
        Ensure there is an open (not-full) long-term frame to write into.
        Creates a new empty frame if needed, and prunes oldest if over limit.
        """
        if self.tl <= 0:
            return
        if len(self._lt_keys) == 0:
            # No LT frames yet — create first empty frame
            self._lt_keys.append(torch.empty(b, 0, self.key_dim, device=device, dtype=dtype))
            self._lt_vals.append(torch.empty(b, 0, self.c_bev,   device=device, dtype=dtype))
            self._lt_fill.append(0)
            return
        if self._lt_fill[-1] >= self.long_frame_tokens:
            # Current last frame is full — open a new one
            self._lt_keys.append(torch.empty(b, 0, self.key_dim, device=device, dtype=dtype))
            self._lt_vals.append(torch.empty(b, 0, self.c_bev,   device=device, dtype=dtype))
            self._lt_fill.append(0)
            # Prune to keep only most recent `tl` frames
            if len(self._lt_keys) > self.tl:
                self._lt_keys = self._lt_keys[-self.tl:]
                self._lt_vals = self._lt_vals[-self.tl:]
                self._lt_fill = self._lt_fill[-self.tl:]

    def _append_long_tokens(self, k_sel: torch.Tensor, v_sel: torch.Tensor):
        """
        Append selected tokens to the long-term buffer, spanning frames if needed.
        k_sel: [B, N, key_dim]
        v_sel: [B, N, c_bev]
        """
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
        """
        When evicting short-term frame st_idx, promote important tokens to LT.

        Two XMem-style selection criteria (union):
          1. Object-based: locations where Mp_t channel 0 > obj_prob_thresh
             (keep tokens near detected objects — most semantically important)
          2. Usage-based: top-K tokens by cumulative attention weight
             (keep tokens that were actually queried frequently — useful history)

        This avoids storing empty space (criterion 1) and ensures frequently
        accessed locations are retained even if objects moved away (criterion 2).
        """
        if self.tl <= 0:
            return
        k_map  = self._st_keys[st_idx]   # [B, key_dim, H, W]
        v_map  = self._st_vals[st_idx]   # [B, c_bev,   H, W]
        mp0    = self._st_mp0[st_idx]    # [B, 1,       H, W]
        usage  = self._st_usage[st_idx]  # [B,          H, W]
        b, _, h, w = k_map.shape
        hw = h * w

        # Flatten spatial dims for token-level operations
        k_flat     = k_map.reshape(b, self.key_dim, hw).permute(0, 2, 1)  # [B, HW, key_dim]
        v_flat     = v_map.reshape(b, self.c_bev,   hw).permute(0, 2, 1)  # [B, HW, c_bev]
        mp0_flat   = mp0.reshape(b, hw)                                    # [B, HW]
        usage_flat = usage.reshape(b, hw)                                  # [B, HW]

        # Criterion 1: object locations
        obj_mask = mp0_flat > self.obj_prob_thresh                         # [B, HW] bool

        # Criterion 2: top-K usage
        k_top     = min(self.topk_usage, hw)
        topk_idx  = torch.topk(usage_flat, k=k_top, dim=1, largest=True).indices
        topk_mask = torch.zeros_like(obj_mask, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)                             # [B, HW] bool

        sel_mask = obj_mask | topk_mask                                    # union
        sel_idx  = sel_mask.nonzero(as_tuple=False)                       # [N, 2]

        if sel_idx.numel() == 0:
            return

        # Gather selected tokens per batch element
        per_b: List[torch.Tensor] = []
        for bi in range(b):
            idx_b = sel_idx[sel_idx[:, 0] == bi][:, 1]
            if idx_b.numel() == 0:
                per_b.append(idx_b)
                continue
            if idx_b.numel() > self.max_from_discard:
                # Too many tokens — keep only the highest-usage ones
                u    = usage_flat[bi, idx_b]
                keep = torch.topk(u, k=self.max_from_discard, largest=True).indices
                idx_b = idx_b[keep]
            per_b.append(idx_b)

        nmax = max(int(x.numel()) for x in per_b)
        if nmax == 0:
            return

        # Pad to uniform length for batched append
        k_sel = torch.zeros(b, nmax, self.key_dim, device=k_map.device, dtype=k_map.dtype)
        v_sel = torch.zeros(b, nmax, self.c_bev,   device=v_map.device, dtype=v_map.dtype)
        for bi in range(b):
            idx_b = per_b[bi]
            if idx_b.numel() == 0:
                continue
            k_sel[bi, :idx_b.numel()] = k_flat[bi, idx_b]
            v_sel[bi, :idx_b.numel()] = v_flat[bi, idx_b]

        self._append_long_tokens(k_sel, v_sel)

    # -------- Memory read ----------------------------------

    def _dist_sq_block(self, q_blk: torch.Tensor, k_all: torch.Tensor) -> torch.Tensor:
        """
        Compute squared L2 distance between query block and all keys.
        ||q - k||^2 = ||q||^2 + ||k||^2 - 2 q·k^T

        q_blk: [B, chunk, key_dim]
        k_all: [B, M,     key_dim]
        returns: [B, chunk, M]  (squared distances)

        L2 similarity (from STCN) is more stable than dot-product attention
        because it doesn't require normalisation of key/query magnitudes.
        """
        q2  = (q_blk * q_blk).sum(dim=2, keepdim=True)   # [B, chunk, 1]
        k2  = (k_all * k_all).sum(dim=2).unsqueeze(1)     # [B, 1,     M]
        dot = torch.bmm(q_blk, k_all.transpose(1, 2))     # [B, chunk, M]
        return (q2 + k2 - 2.0 * dot).clamp_min(0.0)

    def _read_one_memory(
        self,
        q_flat: torch.Tensor,   # [B, HW, key_dim]
        k_flat: torch.Tensor,   # [B, M,  key_dim]
        v_flat: torch.Tensor,   # [B, M,  c_bev]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-based readout from one memory frame.

        Similarity S(q, k) = dist(q, k)^2 / sum_k dist(q, k)^2
        This is the STCN L2 softmax: nearby keys get high weight,
        but the denominator normalises over the spatial extent of the memory.

        Chunked over query positions to avoid a full [B, HW, M] attention matrix
        in memory (HW can be 512×512/4 = 65536 for full BEV grid).

        Returns:
          out:   [B, HW, c_bev]  — attended value readout
          usage: [B, M]          — cumulative attention weight per memory token
                                   (used for LT promotion selection)
        """
        b, hw_q, _ = q_flat.shape
        step = self.q_chunk if self.q_chunk > 0 else hw_q

        out_blocks:   List[torch.Tensor] = []
        usage_blocks: List[torch.Tensor] = []

        for s in range(0, hw_q, step):
            e     = min(s + step, hw_q)
            q_blk = q_flat[:, s:e, :]                           # [B, chunk, key_dim]
            dist  = self._dist_sq_block(q_blk, k_flat)          # [B, chunk, M]
            denom = dist.sum(dim=2, keepdim=True).add(1e-8)     # [B, chunk, 1]
            S     = dist / denom                                 # [B, chunk, M] normalised
            out_blocks.append(torch.bmm(S, v_flat))             # [B, chunk, c_bev]
            usage_blocks.append(S.detach().sum(dim=1))          # [B, M] — detached, no grad needed

        out   = torch.cat(out_blocks, dim=1)                    # [B, HW, c_bev]
        usage = torch.stack(usage_blocks, dim=0).sum(dim=0)    # [B, M]
        return out, usage

    def _collect_memory(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        """
        Flatten all short-term and long-term stored frames into lists of
        [B, N, dim] tensors ready for attention.
        Order: ST frames (oldest→newest), then LT frames.
        """
        keys:  List[torch.Tensor] = []
        vals:  List[torch.Tensor] = []
        kinds: List[str]          = []

        # Short-term: stored as [B, key_dim, H, W], flatten to [B, HW, key_dim]
        for i in range(len(self._st_keys)):
            b, _, h, w = self._st_keys[i].shape
            hw     = h * w
            k_flat = self._st_keys[i].reshape(b, self.key_dim, hw).permute(0, 2, 1)
            v_flat = self._st_vals[i].reshape(b, self.c_bev,   hw).permute(0, 2, 1)
            # lists of [B, N, dim] tensors 
            keys.append(k_flat)
            vals.append(v_flat)
            kinds.append("st")


        # Long-term: already stored as [B, N, dim] sparse tokens
        for j in range(len(self._lt_keys)):
            if self._lt_fill[j] <= 0:
                continue
            keys.append(self._lt_keys[j])
            vals.append(self._lt_vals[j])
            kinds.append("lt")

        return keys, vals, kinds

    # -------- Top-level temporal fusion  --------------

    def compute_mfused(
        self, bev_t: torch.Tensor  # [B, c_bev, H, W]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
 
        b, c, h, w = bev_t.shape

        # Step 1: initialise GRU hidden state to zeros on first call
        if self._gru_h is None:
            self._gru_h = torch.zeros(b, self.c_bev, h, w,
                                      device=bev_t.device, dtype=bev_t.dtype)
            self._hw = (h, w)
        else:
            h0, w0 = self._hw
            if (h, w) != (h0, w0):
                raise ValueError(f"ConvGRU HW mismatch: stored {(h0,w0)}, got {(h,w)}")

        # Step 2: encode bev_t to query [B, key_dim, H, W]
        q_t    = self.query_enc(bev_t)                                    # [B, key_dim, H, W]
        hw     = h * w
        q_flat = q_t.reshape(b, self.key_dim, hw).permute(0, 2, 1)       # [B, HW, key_dim]

        # Step 3: read from all stored memory frames
        mem_keys, mem_vals, mem_kinds = self._collect_memory()

        m_frames: List[torch.Tensor] = []
        for mi, (k_flat, v_flat, kind) in enumerate(zip(mem_keys, mem_vals, mem_kinds)):
            m_flat, usage = self._read_one_memory(q_flat, k_flat, v_flat)  # [B, HW, c_bev]
            m_map = m_flat.permute(0, 2, 1).reshape(b, self.c_bev, h, w)  # [B, c_bev, H, W]
            m_frames.append(m_map)

            # Accumulate usage for short-term frames (used in LT promotion selection)
            if kind == "st":
                self._st_usage[mi] = self._st_usage[mi] + usage.reshape(b, h, w)


        # Step 4: ConvGRU sequential fusion
        # process history FIRST (oldest → newest), current frame LAST.
        # h_t = self._gru_h  # start from persistent state (detached value from last step)
        h_t = torch.zeros_like(self._gru_h)  # begin from zero - aggregator only

        # History: memory readouts ordered oldest → newest (as returned by _collect_memory)
        for m_map in m_frames:
            h_t = self.gru(m_map, h_t)

        # Current frame last: h_t now represents "current BEV given all temporal context"
        h_t = self.gru(bev_t, h_t)

        # DEBUG VALS
        # with torch.no_grad():
        #     xh = torch.cat([bev_t, h_t], dim=1)
        #     rz = torch.sigmoid(self.gru.gates(xh))
        #     _, z = rz.chunk(2, dim=1)
        #     print(f"GRU z gate mean={z.mean():.3f} std={z.std():.3f}")

        # _gru_h is used as the starting h in the NEXT call — it carries temporal
        # information forward in time without carrying the computation graph.
        # self._gru_h = h_t.detach()

        # Step 6: normalise and return.
        # mfused_t uses the ORIGINAL h_t (not detached) so gradients from loss
        # flow back through this step's GRU calls into bev_t and GRU weights.
        mfused_t = self.gru_output_norm(h_t)   # [B, c_bev, H, W]

        dbg = {
            "q_t":       q_t,
            "mfused_t":  mfused_t,
            "m_frames":  m_frames,
            "mem_kinds": mem_kinds,
        }
        return mfused_t, dbg

    # -------- Bank update ----------------------------------

    def update_bank(
        self,
        q_t:      torch.Tensor,   # [B, key_dim, H, W] — query from this timestep
        mfused_t: torch.Tensor,   # [B, c_bev,   H, W] — fused features
        mp_t:     torch.Tensor,   # [B, 7,       H, W] — BEV map from head
    ) -> torch.Tensor:
        """
        Store a new key/value pair in the short-term memory bank.

        Only runs every `tau` steps to limit memory usage and compute cost.
        At tau=2 and 10Hz, this stores one frame every ~0.2 seconds.

        Key   = q_t (the query encoder output — same space as stored keys,
                so future queries can compare directly).
        Value = value_enc(mfused_t, mp_t) — enriched with object geometry info.

        Both are detached — the bank is a store of past information,
        not part of the current step's computation graph.

        When short-term buffer overflows (> ts frames), the oldest frame
        is evicted, with important tokens promoted to long-term memory first.
        """
        self._step += 1
        if self.tau > 1 and (self._step % self.tau) != 0:
            # Not an update step — skip storing
            return torch.zeros((), device=mfused_t.device, dtype=mfused_t.dtype)

        # Detach everything stored in the bank — these tensors must not be
        # part of the gradient graph (they'd be reused across sequence steps
        # and cause incorrect cross-step gradients).
        k_t  = q_t.detach()                               # [B, key_dim, H, W]
        v_t  = self.value_enc(mfused_t, mp_t).detach()   # [B, c_bev,   H, W]
        mp0  = mp_t[:, 0:1].detach()                      # [B, 1, H, W] existence channel only

        b, _, h_k, w_k = k_t.shape
        usage0 = torch.zeros(b, h_k, w_k, device=k_t.device, dtype=k_t.dtype)

        self._st_keys.append(k_t)
        self._st_vals.append(v_t)
        self._st_mp0.append(mp0)
        self._st_usage.append(usage0)

        # Evict oldest short-term frame(s) if over capacity
        # Before discarding, selectively promote important tokens to long-term memory
        while len(self._st_keys) > self.ts:
            self._select_for_long_term(0)   # index 0 = oldest
            self._st_keys.pop(0)
            self._st_vals.pop(0)
            self._st_mp0.pop(0)
            self._st_usage.pop(0)

        return v_t

    # -------- Debug helpers --------------------------------

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
        st_usage = [u[batch_idx].detach()    for u in self._st_usage]
        st_mp0   = [m[batch_idx, 0].detach() for m in self._st_mp0]
        return {"st_usage": st_usage, "st_mp0": st_mp0}