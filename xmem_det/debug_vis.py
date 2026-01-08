import os
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, None]


def _agg2d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 4:
        return x.abs().mean(dim=1, keepdim=False)
    if x.dim() == 3:
        return x.abs().mean(dim=1, keepdim=False)
    if x.dim() == 2:
        return x
    return x.view(-1)


def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _to_numpy_rgb(x: torch.Tensor) -> np.ndarray:
    y = x.detach().float().cpu().numpy()
    y = np.transpose(y, (1, 2, 0))
    out = np.zeros_like(y, dtype=np.float32)
    for c in range(3):
        out[..., c] = _norm01(y[..., c])
    return out


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    a = np.percentile(x, 1.0)
    b = np.percentile(x, 99.0)
    d = float(b - a) if float(b - a) > 1e-8 else 1.0
    y = (x - a) / d
    return np.clip(y, 0.0, 1.0)


def _as_image(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 0:
        return None
    if p.ndim == 1:
        return p.reshape(1, -1)
    if p.ndim == 2:
        return p
    if p.ndim == 3:
        if p.shape[-1] == 3:
            return p
        if p.shape[0] == 3 and p.shape[-1] != 3:
            return np.transpose(p, (1, 2, 0))
        if p.shape[-1] == 1:
            return p[..., 0]
        return p.mean(axis=-1)
    return p.reshape(p.shape[0], -1)


def _save_grid(out_path: str, panels, titles, ncols: int = 3) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(panels)
    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 4.2 * nrows), dpi=120)
    for i, (p, t) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title(t)
        ax.axis("off")
        if p is None:
            continue
        p2 = _as_image(p)
        if p2 is None:
            continue
        if p2.ndim == 3 and p2.shape[-1] == 3:
            ax.imshow(p2)
        else:
            ax.imshow(p2, cmap="gray", vmin=0.0, vmax=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)



def dump_temporal_debug(
    batch_dict: dict,
    t_seq: int,
    bev: torch.Tensor,
    bev_fused: torch.Tensor,
    temp: torch.Tensor,
    hidden_cur: torch.Tensor,
    occ_logits: Optional[torch.Tensor],
    det_prev_raw: Optional[torch.Tensor],
    det_prev_warped: Optional[torch.Tensor],
    scene_mask: Optional[torch.Tensor],
    det_next: Optional[torch.Tensor],
    frames_img: Optional[torch.Tensor],
) -> None:
    if not bool(batch_dict.get("_vis", False)):
        return

    vis_dir = str(batch_dict.get("_vis_dir", "log/vis"))
    vis_tag = str(batch_dict.get("_vis_tag", "sample"))
    b = int(batch_dict.get("_vis_b", 0))

    B = int(bev.shape[0])
    if B <= 0:
        return
    if b < 0:
        b = 0
    if b >= B:
        b = B - 1

    bev_mag = _agg2d(bev[b:b+1])[0]
    bevf_mag = _agg2d(bev_fused[b:b+1])[0]
    temp_mag = _agg2d(temp[b:b+1])[0]
    hid_mag = _agg2d(hidden_cur[b:b+1])[0]
    hid_std = hidden_cur[b].detach().float().std(dim=0)
    delta_mag = (bev_fused[b:b+1] - bev[b:b+1]).abs().mean(dim=1)[0]

    m_prev_raw = None
    if det_prev_raw is not None:
        m_prev_raw = det_prev_raw[b:b+1].max(dim=1)[0][0]

    m_prev_warp = None
    if det_prev_warped is not None:
        m_prev_warp = det_prev_warped[b:b+1].max(dim=1)[0][0]

    m_next = None
    if det_next is not None:
        m_next = det_next[b:b+1].max(dim=1)[0][0]

    sm = None
    if scene_mask is not None:
        sm = scene_mask[b, 0]

    occ_prob = None
    if occ_logits is not None:
        occ_prob = torch.sigmoid(occ_logits[b, 0])

    rgb = None
    if frames_img is not None and frames_img.dim() == 4 and frames_img.shape[1] == 3:
        rgb = _to_numpy_rgb(frames_img[b])

    panels: List[ArrayLike] = [
        _norm01(_to_numpy_2d(bev_mag)),
        _norm01(_to_numpy_2d(temp_mag)),
        _norm01(_to_numpy_2d(delta_mag)),
        _norm01(_to_numpy_2d(bevf_mag)),
        _norm01(_to_numpy_2d(hid_mag)),
         _norm01(_to_numpy_2d(hid_std)),
        None if occ_prob is None else _norm01(_to_numpy_2d(occ_prob)),
        None if m_prev_raw is None else _norm01(_to_numpy_2d(m_prev_raw)),
        None if m_prev_warp is None else _norm01(_to_numpy_2d(m_prev_warp)),
        None if m_next is None else _norm01(_to_numpy_2d(m_next)),
        None if sm is None else _norm01(_to_numpy_2d(sm)),
        rgb,
    ]

    titles = [
        "bev_mag",
        "temp_mag",
        "delta_mag",
        "bev_fused_mag",
        "hidden_mag",
         "hidden_std",
        "occ_prob",
        "det_prev_raw",
        "det_prev_warped",
        "det_next",
        "scene_mask",
        "bev_adapter_rgb",
    ]

    out_path = os.path.join(vis_dir, f"{vis_tag}_t{int(t_seq):03d}.png")
    _save_grid(out_path, panels, titles, ncols=3)
