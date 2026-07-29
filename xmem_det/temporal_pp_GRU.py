import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar


class ConvGRUCell(nn.Module):
    """
    ConvGRU cell (simpler than LSTM, prevents overfitting)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        
        # Combined conv for reset and update gates
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,  # r, z
            kernel_size,
            padding=padding,
            bias=True
        )
        
        # Conv for candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x, h):
        """
        Args:
            x: [B, C, H, W]
            h: [B, hidden_dim, H, W]
        Returns:
            h_next: [B, hidden_dim, H, W]
        """
        combined = torch.cat([x, h], dim=1)  # stack channel-wise: [B, C+hidden_dim, H, W]

        # Reset and update gates
        gates = self.conv_gates(combined)  # [B, C+hidden_dim, H, W] -> [B, 2*hidden_dim, H, W]
        r, z = torch.split(gates, self.hidden_dim, dim=1)  # each [B, hidden_dim, H, W]: channels 0:hidden_dim -> r, hidden_dim:2*hidden_dim -> z
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        # Candidate hidden state
        combined_r = torch.cat([x, r * h], dim=1)  # r gates how much old memory h to keep per-cell
        h_candidate = torch.tanh(self.conv_candidate(combined_r))

        # New hidden state: z interpolates per-cell between old h and the new candidate
        h_next = (1 - z) * h + z * h_candidate

        return h_next


class TemporalPointPillar(PointPillar):
    """
    Temporal PointPillar with ConvGRU, matching TimePillars (Fig. 2):
    backbone -> convGRU -> detection head. The GRU's own update gate
    is the sole "blend old vs. new information" mechanism (Eq. 3 in the
    paper) -- there is no separate fusion layer recombining the GRU
    output with the current single-frame features; the detection head
    consumes the GRU's (projected) hidden state directly.

    Two-stage training:
        Stage 1: Pretrain single-frame PointPillar
        Stage 2: Add GRU, freeze early layers, fine-tune
    """
    def __init__(self, model_cfg, num_class, dataset, pc_range=None, lstm_hidden_dim: int = None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        if pc_range is None:
            raise ValueError("pc_range must be provided")
        self.pc_range = pc_range
        
        # Get BEV feature dimension from backbone
        self.c_bev = int(self.backbone_2d.num_bev_features)
        
        # Use same dimension as BEV features (no bottleneck)
        self.lstm_hidden_dim = lstm_hidden_dim if lstm_hidden_dim is not None else self.c_bev
        
        # ConvGRU for temporal processing (simpler than LSTM)
        self.conv_gru = ConvGRUCell(
            input_dim=self.c_bev,
            hidden_dim=self.lstm_hidden_dim,
            kernel_size=3
        )
        
        # Project GRU output back to BEV dimension
        # GroupNorm(num_groups=1, ...) is used instead of BatchNorm2d
        # throughout -- this pipeline trains with batch_size=1, and
        # BatchNorm's running stats are unusable/noisy when computed over
        # a single sample every step.
        if self.lstm_hidden_dim != self.c_bev:
            self.gru_proj = nn.Sequential(
                nn.Conv2d(self.lstm_hidden_dim, self.c_bev, kernel_size=1),
                nn.GroupNorm(num_groups=1, num_channels=self.c_bev),
                nn.ReLU(inplace=True)
            )
        else:
            self.gru_proj = nn.Identity()

        self.debugger = None
    
    def freeze_early_layers(self):
        """
        Freeze early layers to prevent distribution drift.
        Call this AFTER loading pretrained weights!
        """
        # Freeze VFE (pillar encoder)
        if hasattr(self, 'vfe'):
            print("Freezing VFE (pillar encoder)")
            for param in self.vfe.parameters():
                param.requires_grad = False
        
        # Freeze early backbone blocks
        if hasattr(self.backbone_2d, 'blocks'):
            num_freeze = min(2, len(self.backbone_2d.blocks))
            print(f"Freezing first {num_freeze} backbone blocks")
            for block in self.backbone_2d.blocks[:num_freeze]:
                for param in block.parameters():
                    param.requires_grad = False
    
    def reset_sequence(self, seq_id: int):
        """Reset for new sequence"""
        pass
    
    def forward(self, frames_list, compute_det_loss: bool = True, return_debug_info: bool = False):
        """
        Args:
            frames_list: List of T frame dictionaries, oldest -> newest,
              frames_list[-1] is the annotated keyframe.
            compute_det_loss: Whether to compute detection loss
            return_debug_info: Whether to return debug information

        Returns:
            If training: (loss_dict, tb_dict, disp_dict)
            If inference: (final_box_dicts, {}, debug_info)

        TimePillars-style training: a random number of frames immediately
        preceding the keyframe are used to warm up the GRU hidden state.
        There is no no_grad() anywhere in this window -- gradients flow
        through every warm-up step (real backprop-through-time), matching
        TimePillars Fig. 4, where the backprop arrow spans the whole chain.
        Only the *loss* is restricted to the keyframe.
        """
        T = len(frames_list)
        if T <= 0:
            raise ValueError("frames_list is empty")

        # Randomize warmup length during training (TimePillars strategy).
        # num_warmup = how many frames before the keyframe are used; a
        # smaller num_warmup trims from the FAR end of history, so the
        # frames closest to the keyframe are always kept.
        if self.training and T > 1:
            num_warmup = np.random.randint(max(1, T - 3), T)
        else:
            num_warmup = T - 1

        start_idx = T - 1 - num_warmup
        window = frames_list[start_idx:]  # oldest USED frame ... keyframe

        # Step 1: Extract BEV features for every frame in the window.
        # No no_grad here -- every frame's backbone pass keeps its graph.
        bev_list = []
        for i, bd in enumerate(window):
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                bd = cur_module(bd)
            window[i] = bd
            bev_list.append(bd["spatial_features_2d"])  # [B, C, H, W]
        frames_list[start_idx:] = window

        # Step 2: Process through ConvGRU, oldest -> newest, all with gradient.
        B, C, H, W = bev_list[0].shape
        device = bev_list[0].device

        h = torch.zeros(B, self.lstm_hidden_dim, H, W, device=device)
        for bev in bev_list:
            h = self.conv_gru(bev, h)

        # Step 3: Project GRU output to BEV dimension. This is fed to the
        # detection head directly (TimePillars Fig. 2) -- no separate
        # fusion layer recombining it with bev_list[-1].
        bev_out = self.gru_proj(h)  # [B, c_bev, H, W]

        # Step 4: Run detection head
        frames_list[-1]["spatial_features_2d"] = bev_out
        frames_list[-1] = self.dense_head(frames_list[-1])
        
        # Extract predictions for debugging
        pred_boxes = None
        pred_scores = None
        pred_labels = None
        
        if return_debug_info or self.debugger is not None:
            final_dict = frames_list[-1]
            if "batch_cls_preds" in final_dict and "batch_box_preds" in final_dict:
                cls_preds = final_dict["batch_cls_preds"][0]
                box_preds = final_dict["batch_box_preds"][0]
                
                pred_scores, pred_labels = torch.max(torch.sigmoid(cls_preds), dim=1)
                pred_boxes = box_preds
                
                mask = pred_scores > 0.1
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
        
        # Logging
        if self.debugger is not None:
            self.debugger.log_timestep(
                t=T-1,
                bev_single=bev_list[-1],
                bev_temporal=bev_out,
                lstm_hidden=h,
                lstm_cell=None,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                batch_idx=0,
            )

        # Prepare debug info
        debug_info = None
        if return_debug_info:
            debug_info = {
                'bev_single': bev_list[-1].detach(),
                'bev_out': bev_out.detach(),
                'lstm_hidden': h.detach(),
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
            }
        
        # Step 7: Compute loss or return predictions
        if self.training and compute_det_loss:
            loss_det, tb_dict, disp_dict = self.get_training_loss()
            
            loss_total = loss_det if loss_det is not None else torch.tensor(0.0, device=device)
            
            tb_dict["loss_total"] = loss_total.detach()
            tb_dict["loss_det"] = loss_det.detach() if loss_det is not None else torch.tensor(0.0, device=device)

            return {"loss": loss_total}, tb_dict, disp_dict
        
        # Inference mode
        final_box = frames_list[-1].get("final_box_dicts", None)
        if final_box is None:
            raise KeyError(f"final_box_dicts missing. keys={list(frames_list[-1].keys())}")
        
        return final_box, {}, debug_info
