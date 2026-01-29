import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.util import boxes_to_bev_masks
from xmem_det.debug_vis import dump_temporal_debug
from xmem_det.memory_fuser import ReasonNetTemporalBank


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
        self.bank = ReasonNetTemporalBank(c_bev=self.c_bev, key_dim=int(key_dim), max_frames=int(max_bank_frames))

        self.debug_last = {}

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

    def _get_bev_map_from_head_nograd(self, batch_dict_in: dict, mfused_t: torch.Tensor) -> torch.Tensor:
        """
        Generate Mp: the 7-channel BEV map prediction from mfused_t
        Channels: [existence_prob, x_offset, y_offset, w, h, heading, velocity]
        
        Handles flattened predictions from multi-head configuration.
        """
        dh_was_training = bool(self.dense_head.training)
        self.dense_head.eval()

        with torch.no_grad():
            bd = dict(batch_dict_in)
            bd["spatial_features_2d"] = mfused_t
            
            # Forward through detection head
            bd = self.dense_head(bd)
            
            # Get the raw predictions from the head
            fwd_ret = self.dense_head.forward_ret_dict
            
            cls_preds = fwd_ret["cls_preds"]  # List of [B, num_anchors*H*W, num_classes]
            box_preds = fwd_ret["box_preds"]  # List of [B, num_anchors*H*W, box_code_size]
            
            B, C, H, W = mfused_t.shape
            
            # Generate Mp from flattened predictions
            mp = self._aggregate_flattened_predictions(
                cls_preds, box_preds, B, H, W, 
                mfused_t.device, mfused_t.dtype
            )
            
        self.dense_head.train(dh_was_training)
        return mp

    def _aggregate_flattened_predictions(self, cls_preds_list, box_preds_list, B, H, W, device, dtype):
        """
        Aggregate flattened predictions from multiple detection heads.
        
        Args:
            cls_preds_list: List of [B, num_anchors*H*W, num_classes]
            box_preds_list: List of [B, num_anchors*H*W, box_code_size]
        """
        # Initialize output: 7 channels [existence, x, y, w, h, heading, vel]
        mp = torch.zeros(B, 7, H, W, device=device, dtype=dtype)
        max_existence = torch.zeros(B, H, W, device=device, dtype=dtype)
        
        for head_idx, (cls_pred, box_pred) in enumerate(zip(cls_preds_list, box_preds_list)):
            # cls_pred: [B, num_anchors*H*W, num_classes]
            # box_pred: [B, num_anchors*H*W, box_code_size]
            
            total_predictions = cls_pred.shape[1]
            num_classes = cls_pred.shape[2]
            box_code_size = box_pred.shape[2]
            
            # Infer number of anchors
            num_anchors = total_predictions // (H * W)
            
            # Reshape to spatial format: [B, num_anchors, H, W, num_classes]
            cls_pred_spatial = cls_pred.view(B, num_anchors, H, W, num_classes)
            box_pred_spatial = box_pred.view(B, num_anchors, H, W, box_code_size)
            
            # Permute to [B, num_anchors, num_classes, H, W]
            cls_pred_spatial = cls_pred_spatial.permute(0, 1, 4, 2, 3)
            # Permute to [B, num_anchors, box_code_size, H, W]
            box_pred_spatial = box_pred_spatial.permute(0, 1, 4, 2, 3)
            
            # Get existence probability (max over classes, then over anchors)
            cls_prob = torch.sigmoid(cls_pred_spatial)  # [B, num_anchors, num_classes, H, W]
            
            # Max over classes
            max_prob_per_anchor, _ = cls_prob.max(dim=2)  # [B, num_anchors, H, W]
            # Max over anchors
            existence_prob, max_anchor_idx = max_prob_per_anchor.max(dim=1)  # [B, H, W]
            
            # Update mask: where this head has higher confidence
            update_mask = existence_prob > max_existence  # [B, H, W]
            max_existence = torch.where(update_mask, existence_prob, max_existence)
            
            # Extract box predictions for best anchor at each location
            # box_pred_spatial: [B, num_anchors, box_code_size, H, W]
            # We need to select along the num_anchors dimension using max_anchor_idx
            
            # Expand max_anchor_idx to match box dimensions
            max_anchor_idx_expanded = max_anchor_idx.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, H, W]
            max_anchor_idx_expanded = max_anchor_idx_expanded.expand(-1, -1, box_code_size, -1, -1)  # [B, 1, box_code_size, H, W]
            
            # Gather the boxes from the selected anchors
            selected_boxes = torch.gather(box_pred_spatial, dim=1, index=max_anchor_idx_expanded)  # [B, 1, box_code_size, H, W]
            selected_boxes = selected_boxes.squeeze(1)  # [B, box_code_size, H, W]
            
            # Update mp where this head is more confident
            mp[:, 0] = torch.where(update_mask, existence_prob, mp[:, 0])
            
            # Update box parameters (up to 6 channels: x, y, z, w, l, h)
            for i in range(min(6, box_code_size)):
                mp[:, i + 1] = torch.where(
                    update_mask,
                    selected_boxes[:, i],  # [B, H, W]
                    mp[:, i + 1]           # [B, H, W]
                )
        
        return mp

    def forward(self, frames_list, compute_det_loss: bool = True):
        T = len(frames_list)
        if T <= 0:
            raise ValueError("frames_list is empty")

        bev_list = []
        gt_occ_list = []

        # Step 1: Extract BEV features from all frames
        for t in range(T):
            bd = frames_list[t]
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                bd = cur_module(bd)
            frames_list[t] = bd
            bev = bd["spatial_features_2d"]
            B, C, H, W = bev.shape
            
            gt_occ = self._build_gt_occ_target(bd, H, W)
            if gt_occ is None:
                gt_occ = torch.zeros(B, 1, H, W, device=bev.device, dtype=bev.dtype)
            
            bev_list.append(bev)
            gt_occ_list.append(gt_occ)

        self.bank.reset()
        
        mfused_last = None
        dbg_last = None
        mp_last = None

        # Step 2: Temporal processing with memory bank
        for t in range(T):
            bev_t = bev_list[t]
            
            # Compute fused features
            mfused_t, dbg_t = self.bank.compute_mfused(bev_t)
            
            # Generate Mp: 7-channel BEV map (NOT binary masks!)
            mp_t = self._get_bev_map_from_head_nograd(frames_list[t], mfused_t)
            
            # Update memory bank with query, fused features, and BEV map
            self.bank.update_bank(dbg_t["q_t"], mfused_t, mp_t)
            
            if t == T - 1:
                mfused_last = mfused_t
                dbg_last = dbg_t
                mp_last = mp_t

        # Step 3: Final detection on last frame
        frames_list[-1]["spatial_features_2d"] = mfused_last
        frames_list[-1] = self.dense_head(frames_list[-1])

        if self.training and compute_det_loss:
            loss_det, tb_dict, disp_dict = self.get_training_loss()
            dev = mfused_last.device
            loss_total = loss_det if loss_det is not None else torch.zeros((), device=dev)

            tb_dict["loss_total"] = loss_total.detach()
            tb_dict["loss_det"] = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)

            self.debug_last = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in dbg_last.items()}
            self.debug_last["mp_last"] = mp_last.detach()

            return {"loss": loss_total}, tb_dict, disp_dict, None

        frames_list[-1] = self._fill_postproc_inputs(frames_list[-1])
        pred_dicts, recall_dicts = self.post_processing(frames_list[-1])
        return pred_dicts, recall_dicts, None
