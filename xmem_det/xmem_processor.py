import sys
import torch
import torch.nn as nn
import numpy as np

XMEM_ROOT = "external/XMem/"
if XMEM_ROOT not in sys.path:
    sys.path.insert(0, XMEM_ROOT)

from model.network import XMem


class XMemProcessor:
    """
    XMem processor matching XMemTrainer.do_pass() exactly.
    """
    def __init__(self, config, device):
        self.config = config
        self.num_ref_frames = config.get('num_ref_frames', 3)
        self.deep_update_prob = config.get('deep_update_prob', 0.2)
        
        # XMem config
        xmem_cfg = {
            "single_object": True,
            "hidden_dim": config.get('hidden_dim', 64)
        }
        
        self.XMem = XMem(xmem_cfg, model_path=None, map_location="cpu")
        
        # Load pretrained if specified
        if config.get("xmem_resume", False):
            ckpt_path = config.get("xmem_model")
            state = torch.load(ckpt_path, map_location="cpu")
            self.XMem.load_weights(state, init_as_zero_if_needed=True)
        
        self.XMem.to(device)
        
        # Make trainable
        for p in self.XMem.parameters():
            p.requires_grad = True
    
    def do_pass(self, frames, first_frame_gt, T):
        out = {}

        if frames.dim() != 4 or frames.shape[1] != 3:
            raise RuntimeError(f"Expected frames [B*T,3,H,W], got {tuple(frames.shape)}")

        bt, _, H, W = frames.shape
        b = int(first_frame_gt.shape[0])
        if bt != b * T:
            raise RuntimeError(f"frames has {bt} images but expected B*T={b*T} (B={b}, T={T})")

        num_objects = 1
        selector = torch.ones((b, 1, 1, 1), device=frames.device)

        frames_bt = frames
        frames_bthw = frames_bt.view(b, T, 3, H, W)

        key_bt, shrinkage_bt, selection_bt, f16_bt, f8_bt, f4_bt = self.XMem('encode_key', frames_bt)

        Ck, hk, wk = key_bt.shape[1], key_bt.shape[2], key_bt.shape[3]
        key = key_bt.view(b, T, Ck, hk, wk).permute(0, 2, 1, 3, 4).contiguous()

        if shrinkage_bt is not None:
            Cs, hs, ws = shrinkage_bt.shape[1], shrinkage_bt.shape[2], shrinkage_bt.shape[3]
            shrinkage = shrinkage_bt.view(b, T, Cs, hs, ws).permute(0, 2, 1, 3, 4).contiguous()
        else:
            shrinkage = None

        if selection_bt is not None:
            Csel, hsel, wsel = selection_bt.shape[1], selection_bt.shape[2], selection_bt.shape[3]
            selection = selection_bt.view(b, T, Csel, hsel, wsel).permute(0, 2, 1, 3, 4).contiguous()
        else:
            selection = None

        C16, h16, w16 = f16_bt.shape[1], f16_bt.shape[2], f16_bt.shape[3]
        C8, h8, w8 = f8_bt.shape[1], f8_bt.shape[2], f8_bt.shape[3]
        C4, h4, w4 = f4_bt.shape[1], f4_bt.shape[2], f4_bt.shape[3]

        f16 = f16_bt.view(b, T, C16, h16, w16).contiguous()
        f8 = f8_bt.view(b, T, C8, h8, w8).contiguous()
        f4 = f4_bt.view(b, T, C4, h4, w4).contiguous()

        filler_one = torch.zeros(1, dtype=torch.int64, device=frames.device)
        hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], hk, wk), device=frames.device, dtype=frames.dtype)

        v16, hidden = self.XMem('encode_value', frames_bthw[:, 0], f16[:, 0], hidden, first_frame_gt)
        values = v16.unsqueeze(3)

        out['masks_0'] = first_frame_gt
        out['logits_0'] = None

        readouts = []

        for ti in range(1, T):
            if ti <= self.num_ref_frames:
                ref_values = values
                ref_keys = key[:, :, :ti]
                ref_shrinkage = shrinkage[:, :, :ti] if shrinkage is not None else None
            else:
                indices = [
                    torch.cat([filler_one, torch.randperm(ti - 1, device=frames.device)[:self.num_ref_frames - 1] + 1])
                    for _ in range(b)
                ]
                ref_values = torch.stack([values[bi, :, :, indices[bi]] for bi in range(b)], 0)
                ref_keys = torch.stack([key[bi, :, indices[bi]] for bi in range(b)], 0)
                ref_shrinkage = torch.stack([shrinkage[bi, :, indices[bi]] for bi in range(b)], 0) if shrinkage is not None else None

            memory_readout = self.XMem(
                'read_memory',
                key[:, :, ti],
                selection[:, :, ti] if selection is not None else None,
                ref_keys,
                ref_shrinkage,
                ref_values
            )

            readouts.append(memory_readout[:, 0])

            hidden, logits, masks = self.XMem(
                'segment',
                (f16[:, ti], f8[:, ti], f4[:, ti]),
                memory_readout,
                hidden,
                selector,
                h_out=True
            )

            if ti < (T - 1):
                is_deep_update = np.random.rand() < self.deep_update_prob
                v16, hidden = self.XMem(
                    'encode_value',
                    frames_bthw[:, ti],
                    f16[:, ti],
                    hidden,
                    masks,
                    is_deep_update=is_deep_update
                )
                values = torch.cat([values, v16.unsqueeze(3)], 3)

            out[f'masks_{ti}'] = masks
            out[f'logits_{ti}'] = logits

        if len(readouts) > 0:
            readouts = torch.stack(readouts, dim=1)
        else:
            readouts = torch.zeros((b, 0, 512, hk, wk), device=frames.device, dtype=frames.dtype)

        return out, readouts, hidden
