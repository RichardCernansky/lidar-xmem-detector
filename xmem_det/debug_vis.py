import os
from typing import Optional, List, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, None]


def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    a = np.percentile(x, 1.0)
    b = np.percentile(x, 99.0)
    d = float(b - a) if float(b - a) > 1e-8 else 1.0
    y = (x - a) / d
    return np.clip(y, 0.0, 1.0)


def _to_numpy_rgb(x: torch.Tensor) -> np.ndarray:
    y = x.detach().float().cpu().numpy()
    y = np.transpose(y, (1, 2, 0))
    out = np.zeros_like(y, dtype=np.float32)
    for c in range(3):
        out[..., c] = _norm01(y[..., c])
    return out


def _as_image(p: np.ndarray) -> Optional[np.ndarray]:
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



def _agg2d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 4:
        return x.abs().mean(dim=1, keepdim=False)
    if x.dim() == 3:
        return x.abs().mean(dim=0, keepdim=False)
    if x.dim() == 2:
        return x
    return x.reshape(-1)

def _ref_percentiles(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0):
    x = x.astype(np.float32)
    a = float(np.percentile(x, p_lo))
    b = float(np.percentile(x, p_hi))
    if b - a < 1e-8:
        b = a + 1.0
    return a, b

def _norm01_ref(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = x.astype(np.float32)
    d = float(b - a) if float(b - a) > 1e-8 else 1.0
    y = (x - float(a)) / d
    return np.clip(y, 0.0, 1.0)

def dump_temporal_debug(
    batch_dict: dict,
    t_seq: int,
    bev: torch.Tensor,
    bev_fused: torch.Tensor,
    temp: torch.Tensor,
    hidden_cur: torch.Tensor,
    occ_logits: Optional[torch.Tensor],
    det_next: Optional[torch.Tensor],
    frames_img: Optional[torch.Tensor],
    gt_occ: Optional[torch.Tensor] = None,
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

    bev_mag_t = _agg2d(bev[b:b + 1])[0]
    bevf_mag_t = _agg2d(bev_fused[b:b + 1])[0]
    temp_mag_t = _agg2d(temp[b:b + 1])[0]
    delta_mag_t = (bev_fused[b:b + 1] - bev[b:b + 1]).abs().mean(dim=1)[0]

    bev_mag = _to_numpy_2d(bev_mag_t)
    bevf_mag = _to_numpy_2d(bevf_mag_t)
    temp_mag = _to_numpy_2d(temp_mag_t)
    delta_mag = _to_numpy_2d(delta_mag_t)

    a_ref, b_ref = _ref_percentiles(bev_mag, 1.0, 99.0)

    bev_mean = float(np.mean(np.abs(bev_mag)))
    temp_mean = float(np.mean(np.abs(temp_mag)))
    delta_mean = float(np.mean(np.abs(delta_mag)))
    eps = 1e-8
    r_temp = temp_mean / (bev_mean + eps)
    r_delta = delta_mean / (bev_mean + eps)

    hid_mag = None
    hid_std = None
    if hidden_cur is not None:
        hid_mag = _agg2d(hidden_cur[b:b + 1])[0]
        hid_std = hidden_cur[b].detach().float().std(dim=0)

    occ_prob = None
    if occ_logits is not None:
        occ_prob = torch.sigmoid(occ_logits[b, 0])

    gt = None
    if gt_occ is not None:
        if gt_occ.dim() == 4:
            gt = gt_occ[b, 0]
        elif gt_occ.dim() == 3:
            gt = gt_occ[b]

    occ_err = None
    if occ_prob is not None and gt is not None:
        occ_err = (occ_prob - gt).abs()

    m_next = None
    if det_next is not None:
        if det_next.dim() == 4:
            m_next = det_next[b:b + 1].max(dim=1)[0][0]
        elif det_next.dim() == 3:
            m_next = det_next[b]

    rgb = None
    if frames_img is not None and frames_img.dim() == 4 and frames_img.shape[1] == 3:
        rgb = _to_numpy_rgb(frames_img[b])

    panels: List[ArrayLike] = [
        _norm01_ref(bev_mag, a_ref, b_ref),
        _norm01_ref(temp_mag, a_ref, b_ref),
        _norm01_ref(delta_mag, a_ref, b_ref),
        _norm01_ref(bevf_mag, a_ref, b_ref),
        None if hid_mag is None else _norm01(_to_numpy_2d(hid_mag)),
        None if hid_std is None else _norm01(_to_numpy_2d(hid_std)),
        None if occ_prob is None else _norm01(_to_numpy_2d(occ_prob)),
        None if gt is None else _norm01(_to_numpy_2d(gt)),
        None if occ_err is None else _norm01(_to_numpy_2d(occ_err)),
        None if m_next is None else _norm01(_to_numpy_2d(m_next)),
        rgb,
    ]

    titles = [
        f"bev_mag (ref)",
        f"temp_mag (ref) r={r_temp:.3e}",
        f"delta_mag (ref) r={r_delta:.3e}",
        "bev_fused_mag (ref)",
        "hidden_mag",
        "hidden_std",
        "occ_prob",
        "gt_occ",
        "occ_abs_err",
        "det_next",
        "bev_adapter_rgb",
    ]

    out_path = os.path.join(vis_dir, f"{vis_tag}_t{int(t_seq):03d}.png")
    _save_grid(out_path, panels, titles, ncols=3)



from pathlib import Path

import numpy as np
import torch


def _norm_u8_rgb(chw):
    x = chw.astype(np.float32)
    x = np.transpose(x, (1, 2, 0))
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = (x - mn) / (mx - mn)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dump_xmem_sequence_vis(
    out_dir,
    scene_token,
    seq_idx,
    frames_rgb_bt3hw,
    occ_prob_bt1hw,
    gt_occ_bt1hw,
    thr=0.5,
    ious=None,
    max_cols=16,
    dpi=140,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _to_np(frames_rgb_bt3hw)
    occ = _to_np(occ_prob_bt1hw)
    gt = _to_np(gt_occ_bt1hw)

    if frames.ndim != 5:
        raise ValueError(f"frames_rgb_bt3hw must be [B,T,3,H,W], got {frames.shape}")
    if occ.ndim != 5:
        raise ValueError(f"occ_prob_bt1hw must be [B,T,1,H,W], got {occ.shape}")
    if gt.ndim != 5:
        raise ValueError(f"gt_occ_bt1hw must be [B,T,1,H,W], got {gt.shape}")

    B, T, _, H, W = frames.shape
    b = 0

    t_show = min(int(T), int(max_cols))
    frames_b = frames[b, :t_show]
    occ_b = occ[b, :t_show, 0]
    gt_b = gt[b, :t_show, 0]

    pred_b = occ_b > float(thr)

    fig_w = max(4.0, 2.2 * t_show)
    fig_h = 6.6
    fig, axes = plt.subplots(3, t_show, figsize=(fig_w, fig_h), squeeze=False)

    for ti in range(t_show):
        axes[0, ti].imshow(_norm_u8_rgb(frames_b[ti]))
        axes[0, ti].set_axis_off()
        title0 = f"t={ti}"
        if ious is not None and ti < len(ious):
            v = ious[ti]
            if v is not None and np.isfinite(v):
                title0 = title0 + f"  IoU={float(v):.3f}"
        axes[0, ti].set_title(title0, fontsize=9)

        axes[1, ti].imshow(pred_b[ti].astype(np.uint8), vmin=0, vmax=1, interpolation="nearest")
        axes[1, ti].set_axis_off()
        axes[1, ti].set_title("pred", fontsize=9)

        axes[2, ti].imshow((gt_b[ti] > 0.5).astype(np.uint8), vmin=0, vmax=1, interpolation="nearest")
        axes[2, ti].set_axis_off()
        axes[2, ti].set_title("gt", fontsize=9)

    fig.suptitle(f"scene={scene_token}  seq={int(seq_idx)}  thr={float(thr):.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = out_dir / f"xmem_seq_{int(seq_idx):06d}.png"
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

    return str(out_path)
