import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from data.configs.filenames import XMEM_CHECKPOINT, XMEM_CONFIG, REPO_ROOT
# from trainer.utils import open_config

from xmem_det.util import load_xmem_train_cfg
XMEM_ROOT = "external/XMem/"
if XMEM_ROOT not in sys.path:
    sys.path.insert(0, XMEM_ROOT)

from inference.memory_manager import MemoryManager
from model.aggregate import aggregate
from util.tensor_util import pad_divide_by
from model.network import XMem


def load_xmem(device: torch.device, train_config):
    cfg = {"single_object": False}  # multi-object-capable XMem; still works for 1 foreground label
    xmem_resume = bool(train_config.get("xmem_resume", False))
    if xmem_resume:
        pass 
        # ckpt_path = train_config.get("xmem_model", XMEM_CHECKPOINT)
        # net = XMem(cfg, model_path=None, map_location="cpu")
        # state = torch.load(ckpt_path, map_location="cpu")
        # net.load_weights(state, init_as_zero_if_needed=True)  # load pretrained weights
    else:
        print("NOT RESUMING XMEM")
        net = XMem(cfg, model_path=None, map_location="cpu")
    net.to(device)
    return net


class XMemBackboneWrapper(nn.Module):
    def __init__(self, device: str, train_config: dict, bev_channels):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.train_config = train_config
        self.xmem_config = load_xmem_train_cfg("xmem_det/configs/xmem.yaml")
        self.xmem = load_xmem(self.device, self.train_config)
        self.xmem_core = self.xmem.to(self.device)
        self.hidden_dim = getattr(self.xmem_core, "hidden_dim")
        self.xmem_config["hidden_dim"] = self.hidden_dim
        self.mem_every = self.xmem_config["mem_every"]
        self.deep_update_every = self.xmem_config["deep_update_every"]
        self.enable_long_term = self.xmem_config["enable_long_term"]
        self.deep_update_sync = self.deep_update_every < 0
        
        for p in self.xmem_core.parameters():
            p.requires_grad = True 
        
        self.bev_adapter = nn.Conv2d(bev_channels, 3, kernel_size=1)
        self.mms = []
    
    def clear_memory(self):
        """Reset memory at start of each sequence"""
        self.mms = []
    
    def forward_step(self, t: int, bev_features: torch.Tensor, scene_mask: torch.Tensor):
        """
        Args:
            t: timestep in sequence (0-indexed)
            bev_features: (B, C_bev, H, W) - BEV features from backbone
            scene_mask: (B, 1, H, W) - binary mask of occupied regions
            
        Returns:
            occupancy_logits: (B, 1, H, W) - predicted occupancy for gating
        """
        B = bev_features.size(0)
        
        # Initialize memory managers at t=0
        if t == 0 or len(self.mms) != B:
            self.mms = []
            for _ in range(B):
                mm = MemoryManager(config=self.xmem_config.copy())
                mm.ti = -1
                mm.set_hidden(None)
                self.mms.append(mm)
        
        # Convert BEV to pseudo-RGB
        frames_img = self.bev_adapter(bev_features)  # (B, 3, H, W)
        
        # Pad to multiple of 16 (XMem requirement)
        H, W = frames_img.shape[-2:]
        H16 = ((H + 15) // 16) * 16
        W16 = ((W + 15) // 16) * 16
        if (H, W) != (H16, W16):
            frames_img = F.pad(frames_img, (0, W16-W, 0, H16-H))
            scene_mask_padded = F.pad(scene_mask, (0, W16-W, 0, H16-H))
        else:
            scene_mask_padded = scene_mask
        
        occupancy_logits_list = []
        hidden_features_list = []
        
        for b in range(B):
            mm = self.mms[b]
            mm.ti = t
            mm.all_labels = [1]  # Single foreground label
            
            # Encode current frame
            k_l, sh_l, sel_l, f16, f8, f4 = self.xmem_core.encode_key(
                frames_img[b:b+1], need_ek=True, need_sk=True
            )
            
            if sel_l.dim() == 2:
                sel_l = sel_l.unsqueeze(0)
            
            # Write to memory every mem_every frames
            do_write = (t % self.mem_every == 0) or (mm.work_mem.key is None if hasattr(mm, 'work_mem') else True)
            
            if do_write:
                mask_for_mem = scene_mask_padded[b:b+1]  # (1, 1, H16, W16)
                
                # Simple binary threshold - no padding needed, already padded to H16, W16
                masks_for_encode = (mask_for_mem > 0.5).float()  # (1, 1, H16, W16)
                
                # print(f"frames_img shape: {frames_img[b:b+1].shape}")
                # print(f"f16 shape: {f16.shape}")
                # print(f"masks_for_encode shape: {masks_for_encode.shape}")
                
                K_write = 1  # Always 1 object
                
                h_cur = mm.get_hidden()
                if h_cur is None:
                    mm.create_hidden_state(K_write, k_l)
                    h_cur = mm.get_hidden()
                
                is_deep = (self.deep_update_every > 0) and (t % self.deep_update_every == 0)
                v_l, h_new = self.xmem_core.encode_value(
                    frames_img[b:b+1],  # (1, 3, H16, W16)
                    f16,                 # (1, C, 8, 8)
                    h_cur,               # (1, 1, D, H, W)
                    masks_for_encode,    # (1, 1, H16, W16) ✓ Always this shape!
                    is_deep_update=is_deep
                )
                
                mm.add_memory(k_l, sh_l, v_l, [1], selection=sel_l)
                mm.set_hidden(h_new)
            
            # Query memory to get prediction
            has_memory = hasattr(mm, 'work_mem') and mm.work_mem.key is not None and mm.work_mem.size > 0
            
            if has_memory:
                mem_readout = mm.match_memory(k_l, sel_l if self.enable_long_term else None).unsqueeze(0)
                hidden_local, _, pred_prob_with_bg = self.xmem_core.segment(
                    (f16, f8, f4), mem_readout, mm.get_hidden(),
                    h_out=True, strip_bg=False
                )
                # pred_prob_with_bg: (1, K+1, H_out, W_out)
                pred_prob_with_bg = pred_prob_with_bg[0]  # (K+1, H_out, W_out)
                
                if pred_prob_with_bg.shape[0] > 1:
                    pred_fg = pred_prob_with_bg[1:].max(dim=0, keepdim=True)[0]  # (1, H_out, W_out)
                else:
                    pred_fg = pred_prob_with_bg[0:1]

                hidden_local = hidden_local.squeeze(1)  # (1, D_hidden, H, W)
                hidden_features_list.append(hidden_local)
            else:
                # No memory yet, use current mask
                pred_fg = scene_mask_padded[b, 0:1]  # (1, H16, W16)
            
            # Convert to logits
            pred_fg = pred_fg.clamp(1e-5, 1-1e-5)
            logits = torch.log(pred_fg / (1 - pred_fg))  # (1, H_out, W_out)
            occupancy_logits_list.append(logits)
        
        # Concatenate along batch dimension (dim=0)
        occupancy_logits = torch.cat(occupancy_logits_list, dim=0)  # (B, 1, H_out, W_out)
        hidden_features = torch.cat(hidden_features_list, dim=0)     # (B, D_hidden, H_feat, W_feat)
        
        # Resize both to original BEV resolution
        if occupancy_logits.shape[-2:] != (H, W):
            occupancy_logits = F.interpolate(
                occupancy_logits, size=(H, W),
                mode='bilinear', align_corners=False
            )
        
        if hidden_features.shape[-2:] != (H, W):
            hidden_features = F.interpolate(
                hidden_features, size=(H, W),
                mode='bilinear', align_corners=False
            )
            
        return occupancy_logits, hidden_features