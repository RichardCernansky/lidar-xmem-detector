import torch

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.visualizer import TemporalDebugger
from xmem_det.memory_fuser import ReasonNetTemporalBank


class TemporalPointPillar(PointPillar):

    def __init__(self, model_cfg, num_class, dataset, pc_range, key_dim: int = 64):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if pc_range is None:
            raise ValueError("pc_range must be provided")
        self.pc_range = pc_range  # [xmin, ymin, zmin, xmax, ymax, zmax] for BEV grid

        # c_bev: number of BEV feature channels output by backbone_2d (e.g. 384 for BaseBEVBackbone)
        self.c_bev = int(self.backbone_2d.num_bev_features)
        self.bank = ReasonNetTemporalBank(c_bev=self.c_bev, key_dim=int(key_dim))

        self.vis_counter = 0
        self.eval_vis_dir: str = None   # set to a path to enable vis during eval

    def reset_sequence(self, seq_id: int):
        # Called between sequences to clear GRU hidden state and memory bank.
        # Without this, memory from one scene would leak into the next.
        self.bank.reset()

    # ------------------------------------------------------------------
    # Helpers for building Mp_t (7-channel BEV map used to condition bank)
    # ------------------------------------------------------------------

    def _extract_mp_from_centerpoint(self, pred_dict: dict, H: int, W: int) -> torch.Tensor:
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

        mfused_last = None
        dbg_last    = None
        mp_last     = None

        # Training: visualize every 50 steps. Eval: visualize if eval_vis_dir is set.
        _eval_vis = not self.training and self.eval_vis_dir is not None
        do_viz    = (self.training and (self.vis_counter % 50 == 0)) or _eval_vis
        debugger  = None
        if do_viz:
            save_dir = self.eval_vis_dir if _eval_vis else "./temporal_debug"
            debugger = TemporalDebugger(save_dir=save_dir, log_every=1, max_batches=1)
            debugger.start_sequence(seq_name=f"seq{self.vis_counter:03d}")

        # ------------------------------------------------------------------
        # Temporal reasoning loop
        # ------------------------------------------------------------------
        self.bank.reset()

        for t in range(T):
            bd = frames_list[t]
            
            for cur_module in self.module_list:
                if cur_module is self.dense_head:
                    break
                bd = cur_module(bd)
            frames_list[t] = bd
            bev_t = bd["spatial_features_2d"]
            
            mfused_t, dbg_t = self.bank.compute_mfused(bev_t)
            mp_t = self._get_mp_from_head_nograd(frames_list[t], mfused_t)
            self.bank.update_bank(dbg_t["q_t"], mfused_t, mp_t)

            if do_viz and debugger is not None:
                bank_state = self.bank.get_debug_state()
                bank_maps  = self.bank.get_debug_maps(batch_idx=0)
                mem_kinds  = dbg_t["mem_kinds"]
                gt_boxes_np = TemporalDebugger.extract_gt_boxes(bd, batch_idx=0)
                debugger.log_timestep(
                    t, bev_t, mfused_t, mp_t,
                    bank_state, bank_maps, mem_kinds,
                    q_t=dbg_t["q_t"],
                    st_keys=self.bank._st_keys,
                    batch_idx=0,
                    gt_boxes=gt_boxes_np,
                    pc_range=self.pc_range,
                )
            
            if t < T - 1:
                del bev_t
                del mfused_t
                frames_list[t] = None
            else:
                mfused_last = mfused_t
                dbg_last = dbg_t
                mp_last = mp_t

        # ------------------------------------------------------------------
        # Step 4: Detection head on keyframe only
        # ------------------------------------------------------------------
        frames_list[-1]["spatial_features_2d"] = mfused_last
        frames_list[-1] = self.dense_head(frames_list[-1])

        # After Step 4, before finish_sequence
        if do_viz and debugger is not None:
            final_boxes = frames_list[-1].get("final_box_dicts", None)
            pred_boxes_np = None
            if final_boxes is not None:
                pb = final_boxes[0].get('pred_boxes', None)
                if pb is not None:
                    pred_boxes_np = pb.detach().cpu().numpy()

            debugger.log_timestep(
                T,                          # t=T to distinguish from the in-loop t=T-1 log
                mfused_last, mfused_last, mp_last,
                self.bank.get_debug_state(), {}, [],
                batch_idx=0,
                gt_boxes=TemporalDebugger.extract_gt_boxes(frames_list[-1], batch_idx=0),
                pred_boxes=pred_boxes_np,
                pc_range=self.pc_range,
            )
            debugger.finish_sequence()

        self.vis_counter += 1

        # ------------------------------------------------------------------
        # Loss or prediction output.
        # ------------------------------------------------------------------
        if self.training and compute_det_loss:
            # get_training_loss() reads from dense_head.forward_ret_dict which was
            # populated above. It computes heatmap loss + box regression loss
            # against the gt_boxes in frames_list[-1].
            loss_det, tb_dict, disp_dict = self.get_training_loss()

            dev        = mfused_last.device
            loss_total = loss_det if loss_det is not None else torch.zeros((), device=dev)

            tb_dict["loss_total"] = loss_total.detach()
            tb_dict["loss_det"]   = loss_det.detach() if loss_det is not None else torch.zeros((), device=dev)

            return {"loss": loss_total}, tb_dict, disp_dict

        # Inference path: dense_head populates final_box_dicts via post-processing
        final_box = frames_list[-1].get("final_box_dicts", None)
        if final_box is None:
            raise KeyError(f"final_box_dicts missing. keys={list(frames_list[-1].keys())}")
        return final_box, {}, dbg_last