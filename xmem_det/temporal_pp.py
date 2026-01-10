import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.xmem_wrapper import XMemBackboneWrapper
from xmem_det.util import boxes_to_bev_masks

from xmem_det.debug_vis import dump_temporal_debug

class TemporalPointPillar(PointPillar):
    def __init__(self, model_cfg, num_class, dataset, xmem_train_cfg, pc_range):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c_bev = self.backbone_2d.num_bev_features

        self.xmem = XMemBackboneWrapper(
            device=str(device),
            train_config=xmem_train_cfg,
            bev_channels=c_bev,
        )

        self.pc_range = pc_range
        # self.motion_transform_net = nn.Conv2d(self.xmem.hidden_dim + 6, self.xmem.hidden_dim, 3, padding=1)

        mid = max(self.xmem.hidden_dim // 2, 1)

        self.hidden_prev = None

        D = self.xmem.hidden_dim

        self.occ_prev = None
        self.aux_occ_w = 0.5
        self.aux_cons_w = 0.1

        self.hidden_to_bev = nn.Conv2d(self.xmem.hidden_dim, c_bev, kernel_size=1)

    def reset_sequence(self, seq_id: int):
        self.xmem.clear_memory()
        self.occ_prev = None
        self.hidden_prev = None

    def _build_scene_mask_from_bev(self, spatial_features_2d: torch.Tensor):
        with torch.no_grad():
            mag = spatial_features_2d.abs().sum(dim=1, keepdim=True)
            mask = (mag > 0).float()
        return mask

    def _motion6_map(self, T_rel: torch.Tensor, H: int, W: int, device, dtype):
        if T_rel is None:
            return torch.zeros(1, 6, H, W, device=device, dtype=dtype)

        T = T_rel.unsqueeze(0) if T_rel.dim() == 2 else T_rel

        if T.size(-1) == 4:
            r11 = T[:, 0, 0]
            r12 = T[:, 0, 1]
            r21 = T[:, 1, 0]
            r22 = T[:, 1, 1]
            tx = T[:, 0, 3]
            ty = T[:, 1, 3]
        else:
            r11 = T[:, 0, 0]
            r12 = T[:, 0, 1]
            r21 = T[:, 1, 0]
            r22 = T[:, 1, 1]
            tx = T[:, 0, 2]
            ty = T[:, 1, 2]

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        sx = float(x_max - x_min)
        sy = float(y_max - y_min)
        tx = tx / sx
        ty = ty / sy

        v = torch.stack([r11, r12, r21, r22, tx, ty], dim=1).to(dtype)
        return v.view(v.shape[0], 6, 1, 1).expand(v.shape[0], 6, H, W)

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

        det_masks_next = boxes_to_bev_masks(
            boxes_batch,
            scores_batch,
            bev_h,
            bev_w,
            self.pc_range,
            score_thresh=0.3,
        )
        return det_masks_next

    def _transform_mask(self, mask_prev: torch.Tensor, T_rel: torch.Tensor, H: int, W: int, mode: str = "nearest"):
        if mask_prev is None or T_rel is None:
            return mask_prev

        if T_rel.dim() == 2:
            T_rel = T_rel.unsqueeze(0)

        B = mask_prev.size(0)
        if T_rel.size(0) == 1 and B > 1:
            T_rel = T_rel.expand(B, -1, -1)
        if T_rel.size(0) != B:
            raise ValueError(f"T_rel batch {T_rel.size(0)} != mask_prev batch {B}")

        if T_rel.size(-1) == 4:
            R = T_rel[:, :2, :2]
            t = T_rel[:, :2, 3]
        else:
            R = T_rel[:, :2, :2]
            t = T_rel[:, :2, 2]

        R_inv = torch.inverse(R)
        t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        sx = float(x_max - x_min)
        sy = float(y_max - y_min)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5

        r11 = R_inv[:, 0, 0]
        r12 = R_inv[:, 0, 1]
        r21 = R_inv[:, 1, 0]
        r22 = R_inv[:, 1, 1]
        tx = t_inv[:, 0]
        ty = t_inv[:, 1]

        theta = torch.zeros(B, 2, 3, device=mask_prev.device, dtype=mask_prev.dtype)
        theta[:, 0, 0] = r11
        theta[:, 0, 1] = r12 * (sy / sx)
        theta[:, 1, 0] = r21 * (sx / sy)
        theta[:, 1, 1] = r22
        theta[:, 0, 2] = (2.0 / sx) * (r11 * cx + r12 * cy + tx - cx)
        theta[:, 1, 2] = (2.0 / sy) * (r21 * cx + r22 * cy + ty - cy)

        grid = F.affine_grid(theta, size=(B, 1, H, W), align_corners=False)
        return F.grid_sample(mask_prev, grid, mode=mode, padding_mode="zeros", align_corners=False)

    def _transform_feat(self, x: torch.Tensor, T_rel: torch.Tensor, mode: str = "bilinear"):
        if x is None or T_rel is None:
            return x

        if T_rel.dim() == 2:
            T_rel = T_rel.unsqueeze(0)

        B = x.size(0)
        if T_rel.size(0) == 1 and B > 1:
            T_rel = T_rel.expand(B, -1, -1)
        if T_rel.size(0) != B:
            raise ValueError(f"T_rel batch {T_rel.size(0)} != x batch {B}")

        H, W = x.shape[-2], x.shape[-1]

        if T_rel.size(-1) == 4:
            R = T_rel[:, :2, :2]
            t = T_rel[:, :2, 3]
        else:
            R = T_rel[:, :2, :2]
            t = T_rel[:, :2, 2]

        R_inv = torch.inverse(R)
        t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        sx = float(x_max - x_min)
        sy = float(y_max - y_min)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5

        r11 = R_inv[:, 0, 0]
        r12 = R_inv[:, 0, 1]
        r21 = R_inv[:, 1, 0]
        r22 = R_inv[:, 1, 1]
        tx = t_inv[:, 0]
        ty = t_inv[:, 1]

        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = r11
        theta[:, 0, 1] = r12 * (sy / sx)
        theta[:, 1, 0] = r21 * (sx / sy)
        theta[:, 1, 1] = r22
        theta[:, 0, 2] = (2.0 / sx) * (r11 * cx + r12 * cy + tx - cx)
        theta[:, 1, 2] = (2.0 / sy) * (r21 * cx + r22 * cy + ty - cy)

        grid = F.affine_grid(theta, size=(B, x.size(1), H, W), align_corners=False)
        return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


    def _get_gt_boxes_tensor(self, batch_dict):
        for k in ("gt_boxes", "gt_boxes_lidar", "gt_boxes3d"):
            v = batch_dict.get(k, None)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                return v[0]
            if hasattr(v, "dtype") and hasattr(v, "shape"):
                try:
                    return torch.from_numpy(v)
                except Exception:
                    pass
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

        occ = boxes_to_bev_masks(
            boxes,
            scores,
            H,
            W,
            self.pc_range,
            score_thresh=0.0,
        )
        if occ is None:
            return torch.zeros(B, 1, H, W, device=dev, dtype=dtype)

        if occ.dim() == 3:
            occ = occ.unsqueeze(1)

        if occ.dim() == 4 and occ.size(1) != 1:
            occ = (occ.sum(dim=1, keepdim=True) > 0).to(dtype=dtype)
        else:
            occ = (occ > 0).to(dtype=dtype)

        return occ

    def _bev_block_dropout(self, bev: torch.Tensor, drop_rate: float, block: int):
        if drop_rate <= 0:
            return bev
        B, C, H, W = bev.shape
        b = int(max(block, 1))
        h2 = int((H + b - 1) // b)
        w2 = int((W + b - 1) // b)
        m_small = (torch.rand(B, 1, h2, w2, device=bev.device) > float(drop_rate)).to(dtype=bev.dtype)
        m = F.interpolate(m_small, size=(H, W), mode="nearest")
        return bev * m


    def detach_temporal_state(self):
        if isinstance(self.hidden_prev, torch.Tensor):
            self.hidden_prev = self.hidden_prev.detach()
        if hasattr(self.xmem, "detach_memory"):
            self.xmem.detach_memory()
    
    def tinfo(name, x):
        if x is None:
            print(name, "None")
            return
        if isinstance(x, list):
            print(name, "list", "len", len(x))
            if len(x) > 0 and torch.is_tensor(x[0]):
                y = x[0]
                print(name + "[0]", "req", y.requires_grad, "has_fn", y.grad_fn is not None, "shape", tuple(y.shape))
            return
        if torch.is_tensor(x):
            print(name, "req", x.requires_grad, "has_fn", x.grad_fn is not None, "shape", tuple(x.shape))
            return
        print(name, "type", type(x))


    def forward(
        self,
        batch_dict,
        t_seq: int = 0,
        det_instance_masks_prev: torch.Tensor = None,
        T_rel: torch.Tensor = None,
        alpha_temporal: float = 1.0,
        keep_state_grad: bool = False,
    ):
        aux_occ_loss = None
        aux_cons_loss = None

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

            if cur_module is self.backbone_2d:
                occ_logits = None
                aux_occ = None
                aux_cons = None
                bev = batch_dict["spatial_features_2d"]
                H, W = bev.shape[-2], bev.shape[-1]

                det_prev_raw = det_instance_masks_prev
                det_prev_warped = det_instance_masks_prev

                #warp previous detection masks and hidden state
                if t_seq > 0 and det_instance_masks_prev is not None and T_rel is not None:
                    det_prev_warped = self._transform_mask(det_instance_masks_prev, T_rel, H, W, mode="nearest")
                    det_instance_masks_prev = det_prev_warped
                if t_seq > 0 and T_rel is not None and hasattr(self.xmem, "mms") and len(self.xmem.mms) > 0:
                    for mm in self.xmem.mms:
                        h = mm.get_hidden()
                        if h is None:
                            continue
                        h2 = h.squeeze(1)
                        T_rel_h = T_rel.to(device=h2.device, dtype=h2.dtype)
                        h2w = self._transform_feat(h2, T_rel_h, mode="bilinear")
                        h_warped = h2w.unsqueeze(1)
                        mm.set_hidden(h_warped if keep_state_grad else h_warped.detach())
                
                xmem_teacher = self.training and bool(batch_dict.get("_xmem_teacher"))
                occ_corrupt = self.training and bool(batch_dict.get("_occ_corrupt"))
                prev_thr = float(batch_dict.get("_xmem_prev_thr", 0.5))
                # print("t_seq", t_seq, "xmem_teacher", xmem_teacher, "occ_corrupt", occ_corrupt, "prev_thr", prev_thr)

                if det_instance_masks_prev is not None:
                    scene_mask_gate = (det_instance_masks_prev.sum(dim=1, keepdim=True) > 0).to(dtype=bev.dtype)
                else:
                    scene_mask_gate = self._build_scene_mask_from_bev(bev).to(dtype=bev.dtype)

                if xmem_teacher:
                    scene_mask_xmem = scene_mask_gate
                else:
                    if (self.occ_prev is not None) and (T_rel is not None) and (t_seq > 0):
                        prev_w = self._transform_mask(self.occ_prev, T_rel, H, W, mode="bilinear")
                        scene_mask_xmem = (prev_w > prev_thr).to(dtype=bev.dtype)
                    else:
                        scene_mask_xmem = scene_mask_xmem = torch.zeros(bev.size(0), 1, H, W, device=bev.device, dtype=bev.dtype)# change to only zeros

                bev_xmem = bev
                if occ_corrupt:
                    drop_rate = float(batch_dict.get("_occ_drop_rate", 0.5))
                    block = int(batch_dict.get("_occ_block", 8))
                    bev_xmem = self._bev_block_dropout(bev, drop_rate=drop_rate, block=block)

                occ_logits, hidden_cur = self.xmem.forward_step(
                    t_seq,
                    bev_xmem,
                    scene_mask=scene_mask_xmem,
                    keep_state_grad=keep_state_grad,
                    history_mode="full",
                )


                if occ_logits is not None and occ_logits.dim() == 3:
                    occ_logits = occ_logits.unsqueeze(1)
                occ_prob = torch.sigmoid(occ_logits) if occ_logits is not None else None

                gt_occ = self._build_gt_occ_target(batch_dict, H, W)
                if self.training and gt_occ is not None:
                    target = gt_occ.detach().to(dtype=bev.dtype)
                else:
                    target = scene_mask_gate.detach().to(dtype=bev.dtype)


                # compute aux loss
                if self.training and occ_logits is not None and target is not None:
                    pos = target.mean().clamp(1e-4, 1 - 1e-4)
                    pos_weight = ((1 - pos) / pos).to(device=occ_logits.device, dtype=occ_logits.dtype)
                    aux_occ = F.binary_cross_entropy_with_logits(
                        occ_logits,
                        target.to(dtype=occ_logits.dtype),
                        pos_weight=pos_weight,
                    )
                if self.training and occ_prob is not None and self.occ_prev is not None and T_rel is not None:
                    prev_w = self._transform_mask(self.occ_prev, T_rel, H, W, mode="bilinear")
                    aux_cons = (occ_prob - prev_w.detach()).abs().mean()
                if occ_prob is not None:
                    self.occ_prev = occ_prob.detach()
                aux_occ_loss = aux_occ
                aux_cons_loss = aux_cons            

                # combine bev and temporal
                a = float(alpha_temporal)
                temp = self.hidden_to_bev(hidden_cur)
                bev_scale = bev.detach().abs().mean(dim=1, keepdim=True)
                temp_scale = temp.detach().abs().mean(dim=1, keepdim=True)
                temp = temp * (bev_scale / (temp_scale + 1e-6))

                gate = scene_mask_gate if occ_logits is None else (0.2 + 0.8 * torch.sigmoid(occ_logits))
                if self.training:
                    gate = gate.detach()
                bev_fused = bev + a * gate * temp
                batch_dict["spatial_features_2d"] = bev_fused



        if self.training:
            loss_det = None
            tb_dict = {}
            disp_dict = {}

            loss_det, tb_dict, disp_dict = self.get_training_loss()

            fwd_ret = self.dense_head.forward_ret_dict
            batch_cls_preds, batch_box_preds = self.dense_head.generate_predicted_boxes(
                batch_size=batch_dict["batch_size"],
                cls_preds=fwd_ret["cls_preds"],
                box_preds=fwd_ret["box_preds"],
                dir_cls_preds=fwd_ret.get("dir_cls_preds", None),
            )

            batch_dict["batch_cls_preds"] = batch_cls_preds
            batch_dict["batch_box_preds"] = batch_box_preds
            batch_dict["cls_preds_normalized"] = False

            if isinstance(batch_cls_preds, list):
                batch_dict["multihead_label_mapping"] = [
                    self.dense_head.rpn_heads[i].head_label_indices for i in range(len(batch_cls_preds))
                ]

            pred_dicts, _ = self.post_processing(batch_dict)
            det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
            dump_temporal_debug(
                batch_dict=batch_dict,
                t_seq=t_seq,
                bev=bev,
                bev_fused=batch_dict["spatial_features_2d"],
                temp=temp,
                hidden_cur=hidden_cur,
                occ_logits=occ_logits,
                det_prev_raw=det_prev_raw,
                det_prev_warped=det_prev_warped,
                scene_mask=scene_mask_gate,
                det_next=det_masks_next,
                frames_img=self.xmem.bev_adapter(bev),
                gt_occ=gt_occ,
            )

            loss_total = loss_det if loss_det is not None else torch.zeros((), device=batch_dict["spatial_features_2d"].device)

            aux_occ_raw = aux_occ_loss
            aux_cons_raw = aux_cons_loss

            aux_occ_w = None
            aux_cons_w = None

            if aux_occ_raw is not None:
                aux_occ_w = self.aux_occ_w * aux_occ_raw
                loss_total = loss_total + aux_occ_w

            if aux_cons_raw is not None:
                aux_cons_w = self.aux_cons_w * aux_cons_raw
                loss_total = loss_total + aux_cons_w

            dev = batch_dict["spatial_features_2d"].device

            tb_dict["loss_det"] = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)
            tb_dict["loss_aux_occ"] = aux_occ_raw.detach() if aux_occ_raw is not None else torch.zeros((), device=dev)
            tb_dict["loss_aux_cons"] = aux_cons_raw.detach() if aux_cons_raw is not None else torch.zeros((), device=dev)
            tb_dict["loss_aux_occ_w"] = aux_occ_w.detach() if aux_occ_w is not None else torch.zeros((), device=dev)
            tb_dict["loss_aux_cons_w"] = aux_cons_w.detach() if aux_cons_w is not None else torch.zeros((), device=dev)
            tb_dict["loss_total"] = loss_total.detach()

            return {"loss": loss_total}, tb_dict, disp_dict, det_masks_next

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
        return pred_dicts, recall_dicts, det_masks_next