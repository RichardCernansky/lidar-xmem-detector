import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.util import boxes_to_bev_masks
from xmem_det.visualizer import TemporalDebugger
from xmem_det.memory_fuser import ReasonNetTemporalBank

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


class TemporalPointPillar(PointPillar):
    """
    ReasonNet-style online temporal pipeline (no special-casing t==0):

      For each timestep t:
        1) run PP backbone -> BEV_t
        2) Mfused_t = Bank.compute_mfused(BEV_t)
        3) Mp_t     = pretrained BEV head output turned into a mask (no-grad)
        4) v_t      = Bank.update_bank(q_t, Mfused_t, Mp_t)

      Training:
        - compute detection loss ONLY on the last timestep (t = T-1) using Mfused_{T-1}
        - earlier timesteps are used to build memory, but do not contribute loss directly
    """
    def __init__(self, model_cfg, num_class, dataset, pc_range, key_dim: int = 64, max_bank_frames: int = 8):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if pc_range is None:
            raise ValueError("pc_range must be provided")
        self.pc_range = pc_range

        self.c_bev = int(self.backbone_2d.num_bev_features)
        self.bank = ReasonNetTemporalBank(c_bev=self.c_bev, key_dim=int(key_dim))

        self.debug_last = {}
        self.vis_counter = 0

    def reset_sequence(self, seq_id: int):
        self.bank.reset()

    def _get_gt_boxes_tensor(self, batch_dict):
        for k in ("gt_boxes", "gt_boxes_lidar", "gt_boxes3d"):
            v = batch_dict.get(k, None)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                return v[0]
        return None

    def _build_gt_occ_target(self, batch_dict, H: int, W: int):
        gt = self._get_gt_boxes_tensor(batch_dict)
        if gt is None:
            return None
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)

        dev = batch_dict["spatial_features_2d"].device
        dtype = batch_dict["spatial_features_2d"].dtype
        gt = gt.to(device=dev, dtype=dtype)

        gt7 = gt[..., :7]
        valid = (gt7[..., 3] > 0) & (gt7[..., 4] > 0) & (gt7[..., 5] > 0)
        B = gt7.size(0)
        nmax = int(valid.sum(dim=1).max().item())
        if nmax == 0:
            return torch.zeros(B, 1, H, W, device=dev, dtype=dtype)

        boxes = torch.zeros(B, nmax, 7, device=dev, dtype=dtype)
        for b in range(B):
            idx = valid[b].nonzero(as_tuple=False).squeeze(1)
            k = int(idx.numel())
            if k > 0:
                boxes[b, :k] = gt7[b, idx]

        scores = torch.ones(B, nmax, device=dev, dtype=dtype)
        occ = boxes_to_bev_masks(boxes, scores, H, W, self.pc_range, score_thresh=0.0)

        if occ is None:
            return torch.zeros(B, 1, H, W, device=dev, dtype=dtype)
        if occ.dim() == 3:
            occ = occ.unsqueeze(1)
        if occ.dim() == 4 and occ.size(1) != 1:
            occ = (occ.sum(dim=1, keepdim=True) > 0).to(dtype=dtype)
        else:
            occ = (occ > 0).to(dtype=dtype)
        return occ

    def _fill_postproc_inputs(self, batch_dict):
        fwd_ret = self.dense_head.forward_ret_dict

        cls_preds = fwd_ret["cls_preds"]
        box_preds = fwd_ret["box_preds"]
        dir_cls_preds = fwd_ret.get("dir_cls_preds", None)


        batch_cls_preds, batch_box_preds = self.dense_head.generate_predicted_boxes(
            batch_size=batch_dict["batch_size"],
            cls_preds=cls_preds,
            box_preds=box_preds,
            dir_cls_preds=dir_cls_preds,
        )

        batch_dict["batch_cls_preds"] = batch_cls_preds
        batch_dict["batch_box_preds"] = batch_box_preds
        batch_dict["cls_preds_normalized"] = False

        if isinstance(batch_cls_preds, list):
            batch_dict["multihead_label_mapping"] = [
                self.dense_head.rpn_heads[i].head_label_indices for i in range(len(batch_cls_preds))
            ]
        return batch_dict

    def _bev_to_rgb3(self, bev):
        B, C, H, W = bev.shape
        if C >= 3:
            return bev[:, :3]
        out = torch.zeros(B, 3, H, W, device=bev.device, dtype=bev.dtype)
        out[:, :C] = bev
        return out

    def _extract_mp_from_centerpoint(self, pred_dict: dict, H: int, W: int) -> torch.Tensor:
        """
        Extract 7-channel Mp_t from CenterPoint predictions.
        
        CenterPoint pred_dict structure:
        {
            'hm': [B, num_classes, H, W],     # Heatmap
            'center': [B, 2, H, W],           # XY offset
            'center_z': [B, 1, H, W],         # Z coordinate
            'dim': [B, 3, H, W],              # log(l, w, h)
            'rot': [B, 2, H, W],              # cos, sin
            'vel': [B, 2, H, W],              # vx, vy (optional)
        }
        
        ReasonNet Mp_t channels:
        [0] existence_prob
        [1] x_offset
        [2] y_offset  
        [3] w (width)
        [4] h (length)
        [5] heading
        [6] velocity_magnitude
        """
        B = pred_dict['hm'].shape[0]
        device = pred_dict['hm'].device
        dtype = pred_dict['hm'].dtype
        
        mp = torch.zeros(B, 7, H, W, device=device, dtype=dtype)
        
        # Channel 0: Object existence (max probability over all classes)
        # Apply sigmoid to convert logits to probabilities
        hm_sigmoid = pred_dict['hm'].sigmoid()  # [B, num_classes, H, W]
        mp[:, 0:1] = hm_sigmoid.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        
        # Channels 1-2: XY offset from grid center
        mp[:, 1:3] = pred_dict['center']  # [B, 2, H, W]
        
        # Channels 3-4: Box dimensions (width, length)
        # CenterPoint predicts log(dims), so exp to get actual dimensions
        dim_exp = pred_dict['dim'].exp()  # [B, 3, H, W] -> [l, w, h]
        mp[:, 3] = dim_exp[:, 1]  # w (width)
        mp[:, 4] = dim_exp[:, 0]  # l (length) - often called 'h' in BEV
        
        # Channel 5: Heading angle
        # Convert (cos, sin) to angle
        rot_cos = pred_dict['rot'][:, 0:1]  # [B, 1, H, W]
        rot_sin = pred_dict['rot'][:, 1:2]  # [B, 1, H, W]
        mp[:, 5:6] = torch.atan2(rot_sin, rot_cos)  # [B, 1, H, W]
        
        # Channel 6: Velocity magnitude
        if 'vel' in pred_dict:
            vel_x = pred_dict['vel'][:, 0:1]  # [B, 1, H, W]
            vel_y = pred_dict['vel'][:, 1:2]  # [B, 1, H, W]
            mp[:, 6:7] = torch.sqrt(vel_x**2 + vel_y**2 + 1e-8)  # [B, 1, H, W]
        else:
            mp[:, 6:7] = 0.0
        
        return mp


    def _get_mp_from_head_nograd(self, batch_dict_in: dict, mfused_t: torch.Tensor) -> torch.Tensor:
        """
        Generate Mp_t: 7-channel BEV map from CenterPoint head.
        This runs the detection head in eval mode with no gradients.
        """
        # Save training state
        was_training = self.dense_head.training
        self.dense_head.eval()
        
        with torch.no_grad():
            # Create a copy to avoid modifying original
            bd = dict(batch_dict_in)
            bd['spatial_features_2d'] = mfused_t
            
            # Forward through CenterPoint head
            bd = self.dense_head(bd)
            
            # Get predictions from forward_ret_dict
            pred_dicts = self.dense_head.forward_ret_dict['pred_dicts']
            
            # CenterPoint uses multiple heads (one per class group)
            # We'll aggregate them by taking max existence probability
            B, C, H, W = mfused_t.shape
            
            mp_combined = torch.zeros(B, 7, H, W, device=mfused_t.device, dtype=mfused_t.dtype)
            
            # Aggregate predictions from all heads
            for head_idx, pred_dict in enumerate(pred_dicts):
                mp_head = self._extract_mp_from_centerpoint(pred_dict, H, W)
                
                # For existence channel, take max across heads
                mp_combined[:, 0] = torch.max(mp_combined[:, 0], mp_head[:, 0])
                
                # For other channels, use values from head with highest existence
                mask = mp_head[:, 0] > mp_combined[:, 0]  # [B, H, W]
                for c in range(1, 7):
                    mp_combined[:, c] = torch.where(
                        mask,
                        mp_head[:, c],
                        mp_combined[:, c]
                    )
        
        # Restore training state
        self.dense_head.train(was_training)
        
        return mp_combined

    def forward(self, frames_list, compute_det_loss: bool = True):
        T = len(frames_list)
        if T <= 0:
            raise ValueError("frames_list is empty")

        bev_list = []

        # Step 1: Extract BEV features from all frames
        for t in range(T):
            bd = frames_list[t]
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                bd = cur_module(bd)
            frames_list[t] = bd
            bev = bd["spatial_features_2d"]
            bev_list.append(bev)

        # Step 2: Reset temporal bank for new sequence
        self.bank.reset()
        
        mfused_last = None
        dbg_last = None
        mp_last = None

        # Optional visualization
        do_viz = self.training and (self.vis_counter < 5)
        debugger = None
        if do_viz:
            debugger = TemporalDebugger(
                save_dir="./temporal_debug",
                log_every=1,
                max_batches=1
            )
            debugger.start_sequence(seq_name=f"seq{self.vis_counter:03d}")

        # Step 2: Temporal reasoning loop
        for t in range(T):
            bev_t = bev_list[t]
            
            # Compute temporally-fused features
            mfused_t, dbg_t = self.bank.compute_mfused(bev_t)
            
            # Generate Mp_t from CenterPoint predictions (no gradient)
            mp_t = self._get_mp_from_head_nograd(frames_list[t], mfused_t)
            
            # Update memory bank
            self.bank.update_bank(dbg_t["q_t"], mfused_t, mp_t)
            
            # Visualization
            if do_viz:
                bank_state = self.bank.get_debug_state()
                bank_maps = self.bank.get_debug_maps(batch_idx=0)
                debugger.log_timestep(
                    t=t,
                    bev_t=bev_t,
                    mfused_t=mfused_t,
                    mp_t=mp_t,
                    bank_state=bank_state,
                    bank_maps=bank_maps,
                    mem_kinds=dbg_t["mem_kinds"],
                    batch_idx=0
                )

            # Save last frame outputs
            if t == T - 1:
                mfused_last = mfused_t
                dbg_last = dbg_t
                mp_last = mp_t

        if do_viz:
            debugger.finish_sequence()
            self.vis_counter += 1

        # Step 3: Detection on last frame only
        frames_list[-1]["spatial_features_2d"] = mfused_last
        frames_list[-1] = self.dense_head(frames_list[-1])

        # Step 4: Compute loss or return predictions
        if self.training and compute_det_loss:
            loss_det, tb_dict, disp_dict = self.get_training_loss()
            dev = mfused_last.device
            loss_total = loss_det if loss_det is not None else torch.zeros((), device=dev)

            tb_dict["loss_total"] = loss_total.detach()
            tb_dict["loss_det"] = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)

            # Save debug info
            self.debug_last = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in dbg_last.items()}
            self.debug_last["mp_last"] = mp_last.detach()

            return {"loss": loss_total}, tb_dict, disp_dict, None

        # Inference mode
        frames_list[-1] = self._fill_postproc_inputs(frames_list[-1])
        pred_dicts, recall_dicts = self.post_processing(frames_list[-1])
        return pred_dicts, recall_dicts, None