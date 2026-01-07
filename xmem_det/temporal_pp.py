import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.xmem_wrapper import XMemBackboneWrapper
from xmem_det.util import boxes_to_bev_masks


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
        self.motion_transform_net = nn.Conv2d(self.xmem.hidden_dim + 6, self.xmem.hidden_dim, 3, padding=1)

        mid = max(self.xmem.hidden_dim // 2, 1)
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.xmem.hidden_dim, mid, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, 6, 3, padding=1),
        )

        self.aux_weight = 1.0
        self.hidden_prev = None

        D = self.xmem.hidden_dim
        self.state_gate = nn.Conv2d(2 * D, D, kernel_size=1)
        self.state_cand = nn.Conv2d(2 * D, D, kernel_size=3, padding=1)


        self.hidden_to_bev = nn.Conv2d(self.xmem.hidden_dim, c_bev, kernel_size=1)

    def reset_sequence(self, seq_id: int):
        self.xmem.clear_memory()
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
        compute_det_loss: bool = True,
        compute_aux_loss: bool = True,
        keep_state_grad: bool = False,
    ):
        aux_loss = None

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

            if cur_module is self.backbone_2d:
                bev = batch_dict["spatial_features_2d"]
                H, W = bev.shape[-2], bev.shape[-1]

                if t_seq > 0 and det_instance_masks_prev is not None and T_rel is not None:
                    det_instance_masks_prev = self._transform_mask(det_instance_masks_prev, T_rel, H, W, mode="nearest")

                if det_instance_masks_prev is not None:
                    scene_mask = (det_instance_masks_prev.sum(dim=1, keepdim=True) > 0).float()
                else:
                    scene_mask = self._build_scene_mask_from_bev(bev)

                occ_logits, hidden_cur = self.xmem.forward_step(
                    t_seq,
                    bev,
                    scene_mask=scene_mask,
                    keep_state_grad=keep_state_grad,
                )

                a = float(alpha_temporal)
                temp = self.hidden_to_bev(hidden_cur)
                batch_dict["spatial_features_2d"] = bev + a * temp


                # if t_seq > 0 and self.hidden_prev is not None and T_rel is not None:
                #     motion_gt = self._motion6_map(T_rel, H, W, bev.device, bev.dtype)
                #     prior_to_cur = self.motion_transform_net(torch.cat([self.hidden_prev, motion_gt], dim=1))
                # else:
                #     prior_to_cur = torch.zeros_like(hidden_cur)

                # u = torch.cat([hidden_cur, prior_to_cur], dim=1)
                # z = torch.sigmoid(self.state_gate(u))
                # h_tilde = torch.tanh(self.state_cand(u))
                # hidden_out = (1.0 - z) * prior_to_cur + z * h_tilde
                # batch_dict["spatial_features_2d"] = self.hidden_to_bev(hidden_out)
                
                # if self.training and compute_aux_loss and t_seq > 0 and self.hidden_prev is not None and T_rel is not None:
                #     motion_gt = self._motion6_map(T_rel, H, W, bev.device, bev.dtype).detach()
                #     motion_pred = self.aux_head(prior_to_cur)
                #     valid_area = self._build_scene_mask_from_bev(bev).detach()
                #     diff = torch.abs(motion_pred - motion_gt)
                #     aux_loss = (diff * valid_area).sum() / (valid_area.sum() + 1e-6)

                # self.hidden_prev = hidden_out if keep_state_grad else hidden_out.detach()


        if self.training:
            loss_det = None
            tb_dict = {}
            disp_dict = {}

            if compute_det_loss:
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

            if loss_det is None:
                loss = self.aux_weight * aux_loss if aux_loss is not None else torch.zeros((), device=bev.device)
            else:
                loss = loss_det
                if aux_loss is not None:
                    loss = loss + self.aux_weight * aux_loss

            if aux_loss is not None:
                tb_dict["aux_motion_tf"] = aux_loss.detach()

            for k in [
                "_dbg_gate_mean",
                "_dbg_gate_used_mean",
                "_dbg_hidden_used_l1",
                "_dbg_bev_l1",
                "_dbg_hidden_to_bev",
                "_dbg_prior_l1",
                "_dbg_hidden_cur_l1",
                "_dbg_z_mean",
                "_dbg_bev_delta",
                "_dbg_temporal_delta",
                "_dbg_hidden_prev_has_fn",
                "_dbg_alpha",
            ]:
                if k in batch_dict:
                    tb_dict[k] = batch_dict[k]


            return {"loss": loss}, tb_dict, disp_dict, det_masks_next

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
        return pred_dicts, recall_dicts, det_masks_next