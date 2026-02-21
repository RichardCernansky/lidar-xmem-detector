import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar


class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell for processing spatial features with temporal dynamics.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g gates
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x, hidden_state):
        """
        Args:
            x: [B, C, H, W] - current input
            hidden_state: tuple of (h, c) each [B, hidden_dim, H, W]
        Returns:
            new hidden state tuple (h, c)
        """
        h, c = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)  # [B, C + hidden_dim, H, W]
        
        # Compute gates
        gates = self.conv(combined)  # [B, 4*hidden_dim, H, W]
        
        # Split into i, f, o, g
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        # Update cell state and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class TemporalPointPillar(PointPillar):
    """
    Simplified temporal PointPillar with ConvLSTM processing.
    
    Flow:
        1. Extract BEV features from all frames using backbone
        2. Process through ConvLSTM to aggregate temporal information
        3. Run detection head on final timestep only
    """
    def __init__(self, model_cfg, num_class, dataset, pc_range, lstm_hidden_dim: int = 384):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        if pc_range is None:
            raise ValueError("pc_range must be provided")
        self.pc_range = pc_range
        
        # Get BEV feature dimension from backbone
        self.c_bev = int(self.backbone_2d.num_bev_features)

        # ConvLSTM for temporal processing
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        
        self.conv_lstm = ConvLSTMCell(
            input_dim=self.c_bev,
            hidden_dim=self.lstm_hidden_dim,
            kernel_size=3
        )
        
        # Project LSTM output back to BEV feature dimension for detection head
        self.lstm_proj = nn.Sequential(
            nn.Conv2d(self.lstm_hidden_dim, self.c_bev, kernel_size=1),
            nn.BatchNorm2d(self.c_bev),
            nn.ReLU(inplace=True)
        )
        
        # Debugger (optional, set externally)
        self.debugger = None
    
    def reset_sequence(self, seq_id: int):
        """Reset LSTM hidden state for new sequence"""
        pass  # Will initialize fresh state in forward()
    
    def forward(self, frames_list, compute_det_loss: bool = True, return_debug_info: bool = False):
        """
        Args:
            frames_list: List of T frame dictionaries
            compute_det_loss: Whether to compute detection loss
            return_debug_info: Whether to return debug information for visualization
        
        Returns:
            If training: (loss_dict, tb_dict, disp_dict, debug_info)
            If inference: (final_box_dicts, {}, debug_info)
        """
        T = len(frames_list)
        if T <= 0:
            raise ValueError("frames_list is empty")
        
        # Step 1: Extract BEV features from all frames
        bev_list = []
        for t in range(T):
            bd = frames_list[t]
            # Run through all modules except detection head
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                bd = cur_module(bd)
            
            frames_list[t] = bd  # Save processed batch_dict
            bev = bd["spatial_features_2d"]  # [B, C, H, W]
            bev_list.append(bev)
        
        # Step 2: Process through ConvLSTM
        B, C, H, W = bev_list[0].shape
        device = bev_list[0].device
        
        # Initialize hidden state
        h = torch.zeros(B, self.lstm_hidden_dim, H, W, device=device)
        c = torch.zeros(B, self.lstm_hidden_dim, H, W, device=device)
        
        # Process sequence through LSTM
        for t in range(T):
            h, c = self.conv_lstm(bev_list[t], (h, c))
            
            # Optional: Log intermediate timesteps
            if self.debugger is not None and t < T - 1:
                # For intermediate frames, we don't have detections yet
                self.debugger.log_timestep(
                    t=t,
                    bev_single=bev_list[t],
                    bev_temporal=self.lstm_proj(h),  # Projected for visualization
                    lstm_hidden=h,
                    lstm_cell=c,
                    batch_idx=0,
                )
        
        # Step 3: Project LSTM output to BEV feature space
        bev_temporal = self.lstm_proj(h)  # [B, c_bev, H, W]
        
        # Step 4: Run detection head on final frame
        frames_list[-1]["spatial_features_2d"] = bev_temporal
        frames_list[-1] = self.dense_head(frames_list[-1])
        
        # Extract predictions for visualization
        pred_boxes = None
        pred_scores = None
        pred_labels = None
        
        if return_debug_info or self.debugger is not None:
            final_dict = frames_list[-1]
            
            # Extract from batch_cls_preds and batch_box_preds
            if "batch_cls_preds" in final_dict and "batch_box_preds" in final_dict:
                cls_preds = final_dict["batch_cls_preds"]  # [B, H*W*anchors, num_class]
                box_preds = final_dict["batch_box_preds"]  # [B, H*W*anchors, 7]
                
                # Get predictions for first batch element
                cls_preds_b0 = cls_preds[0]  # [N, num_class]
                box_preds_b0 = box_preds[0]  # [N, 7]
                
                # Get max class scores
                pred_scores, pred_labels = torch.max(torch.sigmoid(cls_preds_b0), dim=1)
                pred_boxes = box_preds_b0
                
                # Filter by confidence threshold
                conf_thresh = 0.1
                mask = pred_scores > conf_thresh
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
        
        # Log final timestep with detections
        if self.debugger is not None:
            self.debugger.log_timestep(
                t=T-1,
                bev_single=bev_list[-1],
                bev_temporal=bev_temporal,
                lstm_hidden=h,
                lstm_cell=c,
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
                'bev_temporal': bev_temporal.detach(),
                'lstm_hidden': h.detach() if h is not None else None,
                'lstm_cell': c.detach() if c is not None else None,
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
            }
        
        # Step 5: Compute loss or return predictions
        if self.training and compute_det_loss:
            loss_det, tb_dict, disp_dict = self.get_training_loss()
            
            dev = bev_temporal.device
            loss_total = loss_det if loss_det is not None else torch.zeros((), device=dev)
            
            tb_dict["loss_total"] = loss_total.detach()
            tb_dict["loss_det"] = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)
            
            return {"loss": loss_total}, tb_dict, disp_dict, debug_info
        
        # Inference mode
        final_box = frames_list[-1].get("final_box_dicts", None)
        if final_box is None:
            raise KeyError(f"final_box_dicts missing. keys={list(frames_list[-1].keys())}")
        
        return final_box, {}, debug_info