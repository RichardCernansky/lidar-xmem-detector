import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.util import boxes_to_bev_masks
from xmem_det.xmem_processor import XMemProcessor

from xmem_det.debug_vis import dump_temporal_debug
from xmem_det.losses import boot_bce_plus_dice


class TemporalPointPillar(PointPillar):
    def __init__(self, model_cfg, num_class, dataset, xmem_train_cfg, pc_range):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c_bev = self.backbone_2d.num_bev_features

        # XMem processor
        self.xmem_processor = XMemProcessor(xmem_train_cfg, device)
        self.xmem = self.xmem_processor.XMem

        self.hidden_dim = self.xmem_processor.config['hidden_dim']
        
        self.pc_range = pc_range

        # Adapters
        self.bev_adapter = nn.Conv2d(c_bev, 3, kernel_size=1)
        self.readout_to_bev = nn.Conv2d(512, c_bev, 1)

        self.readout_gru = nn.GRU(input_size=512, hidden_size=512, num_layers=1, batch_first=True)


        # Loss weights
        self.aux_occ_w = 0.5

        self.readout_to_bev = nn.Sequential(
            nn.Conv2d(512, c_bev, 1, bias=False),
            nn.GroupNorm(8, c_bev),
            nn.GELU(),
        )

        self.temp_up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_bev, c_bev, 3, padding=1, bias=False),
                nn.GroupNorm(8, c_bev),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        self.temp_refine = nn.Sequential(
            nn.Conv2d(c_bev, c_bev, 3, padding=1, bias=False),
            nn.GroupNorm(8, c_bev),
            nn.GELU(),
            nn.Conv2d(c_bev, c_bev, 3, padding=1, bias=True),
        )

        self.temp_gain = nn.Parameter(torch.tensor(0.0))

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c_bev, c_bev, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_bev, c_bev, 3, padding=1),
        )


    def reset_sequence(self, seq_id: int):
        """Called at start of sequence"""
        pass

    def _build_det_masks(self, pred_dicts, batch_dict):
        batch_bev = batch_dict["spatial_features_2d"]
        bev_h, bev_w = batch_bev.shape[-2], batch_bev.shape[-1]
        B = len(pred_dicts)
        max_det = max([d["pred_boxes"].shape[0] for d in pred_dicts]) if B > 0 else 0

        if max_det == 0:
            return torch.zeros(B, 1, bev_h, bev_w, device=batch_bev.device)

        boxes_batch = []
        scores_batch = []
        for d in pred_dicts:
            boxes = d["pred_boxes"]
            scores = d["pred_scores"]
            if boxes.shape[0] < max_det:
                pad_n = max_det - boxes.shape[0]
                boxes = torch.cat([boxes, torch.zeros(pad_n, boxes.shape[1], device=boxes.device)], dim=0)
                scores = torch.cat([scores, torch.zeros(pad_n, device=scores.device)], dim=0)
            boxes_batch.append(boxes.unsqueeze(0))
            scores_batch.append(scores.unsqueeze(0))

        boxes_batch = torch.cat(boxes_batch, dim=0)
        scores_batch = torch.cat(scores_batch, dim=0)
        return boxes_to_bev_masks(boxes_batch, scores_batch, bev_h, bev_w, self.pc_range, score_thresh=0.3)

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

    def forward(
        self,
        frames_list,  # List of T batch_dicts
        alpha_temporal: float = 1.0,
        compute_det_loss=True,
        compute_aux_loss=True,
        use_det_t0: bool = True,
    ):
        """
        Process all frames at once.
        
        Args:
            frames_list: List of T batch_dicts, one per frame
            alpha_temporal: Temporal fusion weight
            compute_det_loss: Whether to compute detection loss (only on last frame)
            compute_aux_loss: Whether to compute XMem occupancy loss (on all frames)
        """
        T = len(frames_list)
        
        # === PROCESS ALL FRAMES THROUGH POINTPILLAR ===
        bev_list = []
        gt_occ_list = []
        
        for t, batch_dict in enumerate(frames_list):
            # Standard PointPillar forward
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                batch_dict = cur_module(batch_dict)
            
            bev = batch_dict["spatial_features_2d"]
            B, C, H, W = bev.shape
            
            # Pad to multiple of 16
            H16 = ((H + 15) // 16) * 16
            W16 = ((W + 15) // 16) * 16
            
            # Get GT
            gt_occ = self._build_gt_occ_target(batch_dict, H16, W16)
            if gt_occ is None:
                gt_occ = torch.zeros(B, 1, H16, W16, device=bev.device)
            
            bev_list.append(bev)
            gt_occ_list.append(gt_occ)
        
        # === PREPARE FOR XMEM ===
        bev_stack = torch.stack(bev_list, dim=0)  # [T, B, C, H, W]
        bev_stack = bev_stack.permute(1, 2, 0, 3, 4)  # [B, C, T, H, W]
        
        # Convert to RGB
        B, C, T, H, W = bev_stack.shape
        bev_flat = bev_stack.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        frames_rgb = self.bev_adapter(bev_flat)  # [B*T, 3, H, W]
        
        # Pad to 16
        if (H, W) != (H16, W16):
            frames_rgb = F.pad(frames_rgb, (0, W16 - W, 0, H16 - H))
        
        #get first detection masks
        first_frame_gt = gt_occ_list[0]
        mask_t0 = None
        if use_det_t0 == True:
            bd0 = dict(frames_list[0])
            dh_was_training = bool(self.dense_head.training)
            self.dense_head.eval()
            with torch.no_grad():
                bd0 = self.dense_head(bd0)

                fwd_ret0 = self.dense_head.forward_ret_dict
                cls_preds0 = fwd_ret0["cls_preds"]
                box_preds0 = fwd_ret0["box_preds"]
                dir_cls_preds0 = fwd_ret0.get("dir_cls_preds", None)

                batch_cls_preds0, batch_box_preds0 = self.dense_head.generate_predicted_boxes(
                    batch_size=bd0["batch_size"],
                    cls_preds=cls_preds0,
                    box_preds=box_preds0,
                    dir_cls_preds=dir_cls_preds0,
                )

                bd0["batch_cls_preds"] = batch_cls_preds0
                bd0["batch_box_preds"] = batch_box_preds0
                bd0["cls_preds_normalized"] = False

                if isinstance(batch_cls_preds0, list):
                    bd0["multihead_label_mapping"] = [
                        self.dense_head.rpn_heads[i].head_label_indices for i in range(len(batch_cls_preds0))
                    ]

                pred_dicts0, _ = self.post_processing(bd0)
                mask_t0 = self._build_det_masks(pred_dicts0, bd0)
                if mask_t0.dim() == 3:
                    mask_t0 = mask_t0.unsqueeze(1)
                if mask_t0.size(1) != 1:
                    mask_t0 = (mask_t0.sum(dim=1, keepdim=True) > 0).to(dtype=frames_rgb.dtype)
                else:
                    mask_t0 = (mask_t0 > 0).to(dtype=frames_rgb.dtype)

                mask_t0 = mask_t0.to(device=frames_rgb.device, dtype=frames_rgb.dtype)

                if mask_t0.shape[-2:] != (H16, W16):
                    mask_t0 = F.interpolate(mask_t0, size=(H16, W16), mode="nearest")
            self.dense_head.train(dh_was_training)
        else:
            mask_t0 = first_frame_gt


        # === CALL XMEM do_pass ===
        xmem_out, readouts, hidden = self.xmem_processor.do_pass(
            frames=frames_rgb,
            first_frame_gt=mask_t0,
            T=T
        )

        S = int(readouts.shape[1])
        Bx, Sx, Cx, hk, wk = readouts.shape
        x = readouts.permute(0, 3, 4, 1, 2).reshape(Bx * hk * wk, Sx, Cx)
        _, h = self.readout_gru(x)
        readout_fused = h[-1].reshape(Bx, hk, wk, Cx).permute(0, 3, 1, 2)

        
        # === COMPUTE XMEM LOSS ON ALL FRAMES ===
        aux_occ_loss = torch.zeros((), device=frames_rgb.device)
        
        if self.training and compute_aux_loss:
            for ti in range(1, T):
                masks_ti = xmem_out[f'masks_{ti}']
                
                # Extract occupancy
                if masks_ti.shape[1] > 1:
                    occ_prob = masks_ti[:, 1:].max(dim=1, keepdim=True)[0]
                else:
                    occ_prob = masks_ti[:, 0:1]
                
                occ_prob = occ_prob.clamp(1e-5, 1 - 1e-5)
                occ_logits = torch.log(occ_prob / (1 - occ_prob))
                
                # GT for frame ti
                gt_ti = gt_occ_list[ti]
                if gt_ti is not None:
                    target = gt_ti.detach()
                    loss_ti = boot_bce_plus_dice(occ_logits, target, ratio=0.25, min_k=1024)
                    aux_occ_loss = aux_occ_loss + loss_ti


            # Average over frames
            aux_occ_loss = aux_occ_loss / (T - 1)
        
        # === FUSE TEMPORAL FOR LAST FRAME ===
        ti_last = T - 1
        masks_last = xmem_out[f'masks_{ti_last}']
        
        # Extract occupancy for gating
        if masks_last.shape[1] > 1:
            occ_prob_last = masks_last[:, 1:].max(dim=1, keepdim=True)[0]
        else:
            occ_prob_last = masks_last[:, 0:1]
        
        occ_prob_last = occ_prob_last.clamp(1e-5, 1 - 1e-5)
        occ_logits_last = torch.log(occ_prob_last / (1 - occ_prob_last))
        
        hidden_cur = hidden[:, 0]

        temp = self.readout_to_bev(readout_fused)

        # for blk in self.temp_up_blocks:
        #     if temp.shape[-2] >= H and temp.shape[-1] >= W:
        #         break
        #     temp = F.interpolate(temp, scale_factor=2, mode="bilinear", align_corners=False)
        #     temp = blk(temp)

        if temp.shape[-2:] != (H, W):
            temp = F.interpolate(temp, size=(H, W), mode="bilinear", align_corners=False)

        # temp = self.temp_refine(temp)
        # temp = torch.tanh(self.temp_gain) * temp

        # Fuse
        bev_last = bev_list[-1] 
        a = float(alpha_temporal)
        bev_fused = bev_last + a *  temp
        
        #SCALE
        # bev_scale = bev_last.abs().mean(dim=1, keepdim=True) + 1e-6
        # temp_scale = temp.abs().mean(dim=1, keepdim=True) + 1e-6
        # temp = temp * (bev_scale / temp_scale)
        eps = 1e-6
        # bev_l1 = bev_last.abs().mean()
        # temp_l1 = temp.abs().mean()
        # delta_l1 = (bev_fused - bev_last).abs().mean()

        # gate_mean = gate.mean()
        # occ_mean = torch.sigmoid(occ_logits_last).mean()

        # temp_ratio = temp_l1 / (bev_l1 + eps)
        # delta_ratio = delta_l1 / (bev_l1 + eps)

        
        # Update last frame's batch_dict
        frames_list[-1]["spatial_features_2d"] = bev_fused
        frames_list[-1] = self.dense_head(frames_list[-1])

        
        # === DETECTION LOSS (ONLY LAST FRAME) ===
        if self.training and compute_det_loss:
            tb_dict = {}
            disp_dict = {}
            
            # Run detection head
            fwd_ret = self.dense_head.forward_ret_dict
            batch_cls_preds, batch_box_preds = self.dense_head.generate_predicted_boxes(
                batch_size=frames_list[-1]["batch_size"],
                cls_preds=fwd_ret["cls_preds"],
                box_preds=fwd_ret["box_preds"],
                dir_cls_preds=fwd_ret.get("dir_cls_preds", None),
            )

            frames_list[-1]["batch_cls_preds"] = batch_cls_preds
            frames_list[-1]["batch_box_preds"] = batch_box_preds
            frames_list[-1]["cls_preds_normalized"] = False

            if isinstance(batch_cls_preds, list):
                frames_list[-1]["multihead_label_mapping"] = [
                    self.dense_head.rpn_heads[i].head_label_indices for i in range(len(batch_cls_preds))
                ]

            pred_dicts, _ = self.post_processing(frames_list[-1])
            det_masks_next = self._build_det_masks(pred_dicts, frames_list[-1])
            
            # Detection loss
            loss_det, tb_dict, disp_dict = self.get_training_loss()
            
            # Total loss
            dev = bev_fused.device
            loss_total = loss_det if loss_det is not None else torch.zeros((), device=dev)
            
            if compute_aux_loss:
                aux_occ_w = self.aux_occ_w * aux_occ_loss
                loss_total = loss_total + aux_occ_w
                tb_dict["loss_aux_occ"] = aux_occ_loss.detach()
                tb_dict["loss_aux_occ_w"] = aux_occ_w.detach()
            else:
                tb_dict["loss_aux_occ"] = torch.zeros((), device=dev)
                tb_dict["loss_aux_occ_w"] = torch.zeros((), device=dev)
            
            tb_dict["loss_det"] = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)
            tb_dict["loss_total"] = loss_total.detach()
            
            # tb_dict["bev_l1"] = bev_l1.detach()
            # tb_dict["temp_l1"] = temp_l1.detach()
            # tb_dict["delta_l1"] = delta_l1.detach()
            # tb_dict["temp_ratio"] = temp_ratio.detach()
            # tb_dict["delta_ratio"] = delta_ratio.detach()
            # tb_dict["gate_mean"] = gate_mean.detach()
            # tb_dict["occ_mean"] = occ_mean.detach()
            # tb_dict["alpha_temporal"] = torch.as_tensor(float(alpha_temporal), device=bev_fused.device)

            B0 = int(bev_last.shape[0])
            frames_img_last = None
            try:
                frames_img_last = frames_rgb.reshape(B0, T, 3, H16, W16)[:, T - 1]
            except Exception:
                frames_img_last = None

            occ_vis = occ_logits_last[:, :, :H, :W]
            gt_vis = gt_occ_list[-1][:, :, :H, :W] if gt_occ_list[-1] is not None else None

            dump_temporal_debug(
                batch_dict=frames_list[-1],
                t_seq=int(T - 1),
                bev=bev_last,
                bev_fused=bev_fused,
                temp=temp,
                hidden_cur=hidden_cur,
                occ_logits=occ_vis,
                det_next=det_masks_next,
                frames_img=frames_img_last,
                gt_occ=gt_vis,
            )

            return {"loss": loss_total}, tb_dict, disp_dict, det_masks_next
        
        # Inference
        pred_dicts, recall_dicts = self.post_processing(frames_list[-1])
        det_masks_next = self._build_det_masks(pred_dicts, frames_list[-1])


        return pred_dicts, recall_dicts, det_masks_next