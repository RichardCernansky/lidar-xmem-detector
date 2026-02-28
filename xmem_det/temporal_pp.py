import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.util import boxes_to_bev_masks
from xmem_det.visualizer import TemporalDebugger
from xmem_det.memory_fuser import ReasonNetTemporalBank

import os
import matplotlib.pyplot as plt


class TemporalPointPillar(PointPillar):
    """
    Temporal 3D detection pipeline combining:
      - A pretrained PointPillar/CenterPoint backbone+head for per-frame detection
      - A ReasonNetTemporalBank (ConvGRU + XMem-style memory) for temporal fusion

    Architecture (TimePillars-style persistent GRU, NOT ReasonNet's stateless GRU):
      - The ConvGRU hidden state _gru_h persists across real timesteps within a sequence
      - The memory bank stores explicit key/value pairs for attention-based readout
      - Both the persistent hidden state AND the bank provide temporal context

    Per-sequence forward pass:
      For each frame t in [sweep_0, sweep_1, ..., sweep_{N-1}, KEYFRAME]:
        1) backbone(frame_t)  -> bev_t          [B, C, H, W]
        2) bank.compute_mfused(bev_t) -> mfused_t  (GRU over history + current)
        3) head(mfused_t, no_grad) -> mp_t      7-channel BEV map for bank conditioning
        4) bank.update_bank(q_t, mfused_t, mp_t) -> stores new key/value pair

      Only the final KEYFRAME contributes detection loss (it has gt_boxes).
      Sweep frames build temporal memory but have no GT and no loss.

    Training setup (phase 0):
      - backbone_2d: FROZEN (pretrained weights, no gradient)
      - dense_head:  trained (starts from pretrained, fine-tuned with temporal features)
      - bank:        trained from scratch (query_enc, ConvGRU, value_enc)
    """

    def __init__(self, model_cfg, num_class, dataset, pc_range, key_dim: int = 64, max_bank_frames: int = 8):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if pc_range is None:
            raise ValueError("pc_range must be provided")
        self.pc_range = pc_range  # [xmin, ymin, zmin, xmax, ymax, zmax] for BEV grid

        # c_bev: number of BEV feature channels output by backbone_2d (e.g. 384 for BaseBEVBackbone)
        self.c_bev = int(self.backbone_2d.num_bev_features)
        self.bank = ReasonNetTemporalBank(c_bev=self.c_bev, key_dim=int(key_dim))

        self.debug_last = {}
        self.vis_counter = 0

    def reset_sequence(self, seq_id: int):
        # Called between sequences to clear GRU hidden state and memory bank.
        # Without this, memory from one scene would leak into the next.
        self.bank.reset()

    # ------------------------------------------------------------------
    # Helpers for building Mp_t (7-channel BEV map used to condition bank)
    # ------------------------------------------------------------------

    def _extract_mp_from_centerpoint(self, pred_dict: dict, H: int, W: int) -> torch.Tensor:
        """
        Convert one CenterPoint head's prediction dict into a 7-channel Mp map.

        CenterPoint pred_dict keys:
          'hm'     : [B, num_classes, H, W]  raw logits (need sigmoid)
          'center' : [B, 2, H, W]            sub-voxel XY offset from grid center
          'center_z': [B, 1, H, W]           Z coordinate
          'dim'    : [B, 3, H, W]            log(l, w, h) — need exp()
          'rot'    : [B, 2, H, W]            (cos θ, sin θ)
          'vel'    : [B, 2, H, W]            (vx, vy) — optional

        Output Mp channels:
          [0] existence probability  (sigmoid of max heatmap logit)
          [1] x sub-voxel offset
          [2] y sub-voxel offset
          [3] width  (exp of log(w))
          [4] length (exp of log(l))
          [5] heading angle (atan2 of sin/cos)
          [6] velocity magnitude

        Mp is used by the value encoder to condition what gets stored in the bank.
        High-existence locations get prioritised in long-term memory selection.
        """
        B = pred_dict['hm'].shape[0]
        device = pred_dict['hm'].device
        dtype = pred_dict['hm'].dtype

        mp = torch.zeros(B, 7, H, W, device=device, dtype=dtype)

        # Channel 0: max existence probability over all classes in this head
        hm_sigmoid = pred_dict['hm'].sigmoid()                          # [B, num_cls, H, W]
        mp[:, 0:1] = hm_sigmoid.max(dim=1, keepdim=True)[0]            # [B, 1, H, W]

        # Channels 1-2: sub-voxel center offset (already in voxel units)
        mp[:, 1:3] = pred_dict['center']                                # [B, 2, H, W]

        # Channels 3-4: box dimensions (exp to undo log encoding)
        # CenterPoint dim order: [log_l, log_w, log_h]
        dim_exp = pred_dict['dim'].exp()                                 # [B, 3, H, W]
        mp[:, 3] = dim_exp[:, 1]   # width
        mp[:, 4] = dim_exp[:, 0]   # length

        # Channel 5: heading angle from (cos, sin) encoding
        rot_cos = pred_dict['rot'][:, 0:1]
        rot_sin = pred_dict['rot'][:, 1:2]
        mp[:, 5:6] = torch.atan2(rot_sin, rot_cos)                     # [B, 1, H, W]

        # Channel 6: velocity magnitude
        if 'vel' in pred_dict:
            vel_x = pred_dict['vel'][:, 0:1]
            vel_y = pred_dict['vel'][:, 1:2]
            mp[:, 6:7] = torch.sqrt(vel_x ** 2 + vel_y ** 2 + 1e-8)
        # else: stays zero (no velocity head)

        return mp

    def _get_mp_from_head_nograd(self, batch_dict_in: dict, mfused_t: torch.Tensor) -> torch.Tensor:

        was_training = self.dense_head.training
        self.dense_head.eval()  # switch to eval so BN uses running stats, not batch stats

        with torch.no_grad():
            bd = dict(batch_dict_in)           # shallow copy — don't mutate original
            bd['spatial_features_2d'] = mfused_t

            bd = self.dense_head(bd)           # forward through all heads
            pred_dicts = self.dense_head.forward_ret_dict['pred_dicts']  # list, one per head

            B, C, H, W = mfused_t.shape
            mp_combined = torch.zeros(B, 7, H, W, device=mfused_t.device, dtype=mfused_t.dtype)

            for pred_dict in pred_dicts:
                mp_head = self._extract_mp_from_centerpoint(pred_dict, H, W)  # [B, 7, H, W]

                # FIX: compute mask BEFORE updating channel 0.
                # Previously the mask was computed AFTER torch.max updated mp_combined[:, 0],
                # making mask always False and channels 1-6 never getting written.
                mask = mp_head[:, 0] > mp_combined[:, 0]  # [B, H, W] — this head wins here

                # Now update existence channel (take max across heads)
                mp_combined[:, 0] = torch.max(mp_combined[:, 0], mp_head[:, 0])

                # For geometry channels, use values from whichever head had higher existence
                mask_expanded = mask.unsqueeze(1).expand_as(mp_combined[:, 1:])  # [B, 6, H, W]
                mp_combined[:, 1:] = torch.where(mask_expanded, mp_head[:, 1:], mp_combined[:, 1:])

        self.dense_head.train(was_training)  # restore training mode
        return mp_combined

    def _create_mp_from_gt(self, batch_dict: dict) -> torch.Tensor:
        """
        Alternative Mp_t from GT boxes via CenterPoint's target assignment.
        Only valid for the keyframe (sweep frames have no gt_boxes).

        Uses assign_targets which is parameter-free — pure geometric computation.
        This avoids any dependency on the head's learned weights and gives a
        perfect, noise-free Mp_t for the keyframe during training.

        Currently not used (we rely on pretrained head predictions for consistency
        between training and inference), but kept for ablation experiments.
        """
        spatial_features = batch_dict['spatial_features_2d']
        B, C, H, W = spatial_features.shape
        device = spatial_features.device
        dtype = spatial_features.dtype

        with torch.no_grad():
            target_dict = self.dense_head.assign_targets(
                gt_boxes=batch_dict['gt_boxes'],
                feature_map_size=(H, W),
                feature_map_stride=batch_dict.get('spatial_features_2d_strides', None)
            )

        mp_t = self._convert_target_dict_to_mp(target_dict, B, H, W, device, dtype)
        return mp_t

    def _convert_target_dict_to_mp(
        self,
        target_dict: dict,
        B: int, H: int, W: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert CenterPoint's sparse target_dict to dense [B, 7, H, W] Mp_t.

        target_dict layout (per head):
          'heatmaps'    : [B, num_cls, H, W]   Gaussian heatmap at object centers
          'target_boxes': [B, num_max_objs, K]  box params at each object (sparse)
          'inds'        : [B, num_max_objs]     linear index (y*W + x) into H×W grid
          'masks'       : [B, num_max_objs]     1 = valid object, 0 = padding

        target_boxes column layout (K dims):
          [0,1]: center xy offset within voxel
          [2]  : z
          [3,4,5]: log(w), log(l), log(h)
          [6,7]: cos(heading), sin(heading)
          [8,9]: vx, vy  (if velocity head present)

        We scatter the sparse object params to their spatial locations using 'inds'.
        Uses vectorised scatter_ instead of Python loops for speed.
        """
        mp_t = torch.zeros(B, 7, H, W, device=device, dtype=dtype)

        for head_idx in range(len(target_dict['heatmaps'])):
            heatmap     = target_dict['heatmaps'][head_idx]      # [B, num_cls, H, W]
            target_boxes = target_dict['target_boxes'][head_idx]  # [B, num_max_objs, K]
            inds        = target_dict['inds'][head_idx]           # [B, num_max_objs]
            masks       = target_dict['masks'][head_idx]          # [B, num_max_objs] bool/int

            # Channel 0: existence — max Gaussian value over all classes
            heatmap_max = heatmap.max(dim=1, keepdim=True)[0]    # [B, 1, H, W]
            mp_t[:, 0:1] = torch.max(mp_t[:, 0:1], heatmap_max)

            # Build the 6 geometry channels to scatter: [B, num_max_objs, 6]
            K = target_boxes.shape[2]
            print(K)
            box_feats = torch.zeros(B, inds.shape[1], 6, device=device, dtype=dtype)

            box_feats[:, :, 0] = target_boxes[:, :, 0]           # x offset
            box_feats[:, :, 1] = target_boxes[:, :, 1]           # y offset
            if K > 4:
                box_feats[:, :, 2] = target_boxes[:, :, 4].exp() # width  = exp(log_w)
                box_feats[:, :, 3] = target_boxes[:, :, 3].exp() # length = exp(log_l)
            if K > 7:
                cos_h = target_boxes[:, :, 6]
                sin_h = target_boxes[:, :, 7]
                box_feats[:, :, 4] = torch.atan2(sin_h, cos_h)   # heading angle
            if K > 9:
                vx = target_boxes[:, :, 8]
                vy = target_boxes[:, :, 9]
                box_feats[:, :, 5] = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)  # speed

            # Zero out padding objects using mask
            valid_mask = masks.bool().unsqueeze(-1)               # [B, num_max_objs, 1]
            box_feats = box_feats * valid_mask

            # Vectorised scatter: for each valid object, write its 6 geometry values
            # to the spatial location given by its linear index.
            # inds: [B, num_max_objs] — each entry is y*W + x
            inds_long = inds.long().clamp(0, H * W - 1)           # [B, num_max_objs]

            for c_offset, mp_ch in enumerate(range(1, 7)):
                # src: [B, num_max_objs] — the c_offset-th geometry value per object
                src = box_feats[:, :, c_offset]                    # [B, num_max_objs]
                # target: [B, H*W] flat spatial grid
                flat = mp_t[:, mp_ch].reshape(B, H * W)
                # scatter_: for each obj, write src value at its linear index
                # 'reduce=max' would be ideal but not available everywhere; default overwrites
                flat.scatter_(1, inds_long, src)
                mp_t[:, mp_ch] = flat.reshape(B, H, W)

        return mp_t

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, frames_list, compute_det_loss: bool = True):
        """
        frames_list: list of batch_dicts, length T = num_sweeps + 1
          frames_list[0]   = oldest sweep  (no gt_boxes)
          frames_list[1]   = next sweep    (no gt_boxes)
          ...
          frames_list[T-2] = newest sweep  (no gt_boxes)
          frames_list[T-1] = KEYFRAME      (has gt_boxes, is the annotated nuScenes frame)

        The sequence is one independent episode — bank is reset at the start.
        No memory carries over from the previous keyframe's sequence.
        """
        T = len(frames_list)
        if T <= 0:
            raise ValueError("frames_list is empty")

        # ------------------------------------------------------------------
        # Step 1: Run backbone on ALL frames upfront (before temporal loop).
        # We run all frames through VFE → MAP_TO_BEV → BACKBONE_2D but stop
        # before the dense_head (temporal fusion happens between backbone and head).
        # ------------------------------------------------------------------
        # bev_list = []
        # for t in range(T):
        #     bd = frames_list[t]
        #     for cur_module in self.module_list:
        #         if cur_module is self.dense_head:
        #             break                          # stop before detection head
        #         bd = cur_module(bd)
        #     frames_list[t] = bd
        #     bev_list.append(bd["spatial_features_2d"])  # [B, C, H, W]

        # ------------------------------------------------------------------
        # Step 2: Reset bank for this new sequence.
        # Done AFTER backbone pass so any exception in backbone doesn't leave
        # the bank in a half-reset state for the temporal loop.
        # ------------------------------------------------------------------
        # self.bank.reset()

        mfused_last = None
        dbg_last    = None
        mp_last     = None

        # Optional per-sequence visualisation (every 50 sequences during training)
        do_viz  = self.training and (self.vis_counter % 50 == 0)
        debugger = None
        if do_viz:
            debugger = TemporalDebugger(
                save_dir="./temporal_debug",
                log_every=1,
                max_batches=1
            )
            debugger.start_sequence(seq_name=f"seq{self.vis_counter:03d}")

        # ------------------------------------------------------------------
        # Step 3: Temporal reasoning loop — oldest frame to newest (keyframe).
        # ------------------------------------------------------------------
        self.bank.reset()

        for t in range(T):
            bd = frames_list[t]
            
            # Run backbone for this frame only
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                bd = cur_module(bd)
            frames_list[t] = bd
            bev_t = bd["spatial_features_2d"]
            
            # Immediately fuse and update bank
            # bev_t from early frames can be freed after bank.update_bank()
            mfused_t, dbg_t = self.bank.compute_mfused(bev_t)
            mp_t = self._get_mp_from_head_nograd(frames_list[t], mfused_t)
            self.bank.update_bank(dbg_t["q_t"], mfused_t, mp_t)

            if do_viz:
                bank_state = self.bank.get_debug_state()
                bank_maps  = self.bank.get_debug_maps(batch_idx=0)
                debugger.log_timestep(
                    t=t, bev_t=bev_t, mfused_t=mfused_t, mp_t=mp_t,
                    bank_state=bank_state, bank_maps=bank_maps,
                    mem_kinds=dbg_t["mem_kinds"], batch_idx=0
                )
            
            # Explicitly free early frame BEV — no longer needed
            if t < T - 1:
                del bev_t
                del mfused_t
                frames_list[t] = None   # free the batch dict too
            else:
                mfused_last = mfused_t
                dbg_last = dbg_t
                mp_last = mp_t

        
        if do_viz:
            debugger.finish_sequence()
        self.vis_counter += 1

        # ------------------------------------------------------------------
        # Step 4: Detection head on keyframe only.
        # Replace keyframe's raw BEV features with temporally-fused mfused_last.
        # frames_list[-1] has gt_boxes → dense_head.forward stores targets internally
        # for loss computation in get_training_loss().
        # ------------------------------------------------------------------
        frames_list[-1]["spatial_features_2d"] = mfused_last
        frames_list[-1] = self.dense_head(frames_list[-1])

        # ------------------------------------------------------------------
        # Step 5: Loss or prediction output.
        # ------------------------------------------------------------------
        if self.training and compute_det_loss:
            # get_training_loss() reads from dense_head.forward_ret_dict which was
            # populated in Step 4. It computes heatmap loss + box regression loss
            # against the gt_boxes in frames_list[-1].
            loss_det, tb_dict, disp_dict = self.get_training_loss()

            dev        = mfused_last.device
            loss_total = loss_det if loss_det is not None else torch.zeros((), device=dev)

            tb_dict["loss_total"] = loss_total.detach()
            tb_dict["loss_det"]   = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)

            # Store debug tensors (detached — no graph retention after backward)
            self.debug_last = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in dbg_last.items()
            }
            self.debug_last["mp_last"] = mp_last.detach()

            return {"loss": loss_total}, tb_dict, disp_dict, None

        # Inference path: dense_head populates final_box_dicts via post-processing
        final_box = frames_list[-1].get("final_box_dicts", None)
        if final_box is None:
            raise KeyError(f"final_box_dicts missing. keys={list(frames_list[-1].keys())}")
        return final_box, {}, None