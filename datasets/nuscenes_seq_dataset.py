import copy
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pcdet.datasets.augmentor import augmentor_utils
from pcdet.datasets.augmentor.database_sampler import DataBaseSampler
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset


def collate_seq(batch):
    if len(batch) != 1:
        raise RuntimeError("Use batch_size=1 with collate_seq for now")
    return batch[0]


class NuScenesSeqDataset(Dataset):
    """
    Each dataset item = one NuScenes keyframe + its preceding LiDAR sweeps
    as individual frames, forming a ~10 Hz temporal sequence.
    Layout of one item (oldest → newest):
        [sweep_{-N}, sweep_{-N+2}, ..., sweep_{-2}, sweep_{-1}, KEYFRAME]

    - Sweep frames: each aligned into the KEYFRAME's LIDAR frame via the
      sweep's precomputed ego-motion transform_matrix (same alignment
      OpenPCDet's baseline get_sweep() applies). This keeps BEV grid cell
      (i, j) at the same physical location across every frame in the
      sequence, which the ConvGRU (local 3x3 convs) and the bank's
      key/value memory both assume holds between timesteps. Moving
      objects, occlusion and sparsity still differ frame to frame, so
      this does not make the BEVs identical -- it only removes the ego's
      own motion, which was never itself useful signal.
    - Keyframe    : backbone + detection head + loss (has gt_boxes).
    - Bank resets at the start of every forward() call.

    Two unrelated "stride" knobs, easy to conflate:
    - keyframe_stride: spacing BETWEEN dataset items, i.e. which keyframes
      each get their own sample. len(self) == len(base.infos) / keyframe_stride.
      keyframe_stride=1 -> every keyframe becomes a sample (required by eval,
      so predictions line up 1:1 with ground truth). keyframe_stride=2 would
      use only every other keyframe, halving the number of samples.
    - sweep_stride: spacing WITHIN one sample's history, i.e. which of the
      raw ~20Hz sweeps between two keyframes are selected as the seq_len
      history frames for that one sample. Does not affect how many samples
      the dataset has.

    Augmentation (training only): gt_sampling is applied to the keyframe
    alone (history frames have no labels to match sampled objects against,
    so there's nothing to keep consistent there). random_world_flip/
    rotation/scaling, however, are geometric transforms of the whole scene,
    so they draw ONE random parameter set per sequence and apply it
    identically to the keyframe (points + gt_boxes) and every history frame
    (points only) -- applying them independently per frame would misalign
    the sequence, breaking the same BEV grid-cell correspondence invariant
    described above.
    """

    def __init__(
        self,
        dataset_cfg,
        class_names,
        training,
        logger,
        seq_len,
        keyframe_stride,
        nusc_version,
        nusc_dataroot,
        root_path=None,
        sweep_stride: int = 2,
    ):
        super().__init__()
        self.base = NuScenesDataset(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.seq_len         = int(seq_len)
        self.keyframe_stride = int(keyframe_stride)
        self.sweep_stride     = int(sweep_stride)
        self.class_names      = self.base.class_names
        self.training         = bool(training)
        self.sequence_indices = self._build_sequences()

        self.db_sampler     = None
        self.aug_flip_axes  = None
        self.aug_rot_range  = None
        self.aug_scale_range = None

        aug_cfg = dataset_cfg.get("DATA_AUGMENTOR", None)
        if self.training and aug_cfg is not None:
            disabled = set(aug_cfg.DISABLE_AUG_LIST)
            for cur_cfg in aug_cfg.AUG_CONFIG_LIST:
                if cur_cfg.NAME in disabled:
                    continue
                if cur_cfg.NAME == "gt_sampling":
                    self.db_sampler = DataBaseSampler(
                        root_path=self.base.root_path,
                        sampler_cfg=cur_cfg,
                        class_names=self.class_names,
                        logger=logger,
                    )
                elif cur_cfg.NAME == "random_world_flip":
                    self.aug_flip_axes = list(cur_cfg.ALONG_AXIS_LIST)
                elif cur_cfg.NAME == "random_world_rotation":
                    rot = cur_cfg.WORLD_ROT_ANGLE
                    self.aug_rot_range = rot if isinstance(rot, list) else [-rot, rot]
                elif cur_cfg.NAME == "random_world_scaling":
                    self.aug_scale_range = list(cur_cfg.WORLD_SCALE_RANGE)

    def _build_sequences(self):
        return list(range(0, len(self.base.infos), self.keyframe_stride))

    def __len__(self):
        return len(self.sequence_indices)

    def _load_sweep_points(self, sweep_info: dict) -> np.ndarray:
        """
        Load sweep points and align them into the keyframe's LIDAR frame,
        same as OpenPCDet's baseline get_sweep(). transform_matrix is a
        precomputed 4x4 homogeneous transform (sweep sensor -> sweep ego ->
        global -> keyframe ego -> keyframe sensor) already stored per sweep
        in the dataset's info pkl; the first sweep in a scene has no `prev`
        and stores transform_matrix=None (left unwarped, matching upstream).
        """
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) &
                     (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.base.root_path / sweep_info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
        points = points.reshape([-1, 5])[:, :4]        # x,y,z,intensity
        points = remove_ego_points(points)

        tm = sweep_info.get('transform_matrix', None)
        if tm is not None:
            xyz1 = np.concatenate(
                [points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
            )
            points[:, :3] = (tm.astype(np.float32) @ xyz1.T).T[:, :3]

        times = sweep_info['time_lag'] * np.ones(
            (points.shape[0], 1), dtype=np.float32
        )
        return np.concatenate([points, times], axis=1) # [N, 5]

    def _points_to_batch_dict(self, points: np.ndarray) -> dict:
        data_dict = {"points": points}
        data_dict = self.base.data_processor.forward(data_dict=data_dict)
        return self.base.collate_batch([data_dict])

    def _load_keyframe_input_dict(self, kf_idx: int) -> dict:
        """
        Mirrors the first half of NuScenesDataset.__getitem__ (load points +
        attach gt_boxes/gt_names), stopping short of prepare_data so
        gt_sampling can run on genuinely raw data first.
        """
        base = self.base
        info = copy.deepcopy(base.infos[kf_idx])
        points = base.get_lidar_with_sweeps(kf_idx, max_sweeps=base.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
        }
        if 'gt_boxes' in info:
            if base.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > base.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None
            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask],
            })
        return input_dict

    def _finish_keyframe(self, input_dict: dict) -> dict:
        """
        Runs the rest of NuScenesDataset's per-item pipeline (class-name
        filtering, point_feature_encoder, data_processor) without its
        internal DataAugmentor -- gt_sampling already ran (if enabled) on
        genuinely raw data in __getitem__, and the shared flip/rotation/
        scaling run afterward in __getitem__ too, so the base augmentor
        must stay off here to avoid double-applying / drawing independent
        randomness.
        """
        base = self.base
        was_training = base.training
        base.training = False
        data_dict = base.prepare_data(data_dict=input_dict)
        base.training = was_training

        if base.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in data_dict:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not base.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def _apply_shared_geometric_aug(self, gt_boxes: np.ndarray, kf_points: np.ndarray,
                                     history_points_list: list) -> tuple:
        """
        Draws ONE random parameter set per sequence and applies it to the
        keyframe's gt_boxes+points and every history frame's points, so the
        whole sequence stays geometrically consistent (see class docstring).
        """
        n_cols = gt_boxes.shape[1]

        if self.aug_flip_axes:
            for axis in self.aug_flip_axes:
                enable = bool(np.random.choice([False, True]))
                fn = getattr(augmentor_utils, f"random_flip_along_{axis}")
                gt_boxes, kf_points = fn(gt_boxes, kf_points, enable=enable)
                dummy_boxes = np.zeros((0, n_cols), dtype=np.float32)
                for i, pts in enumerate(history_points_list):
                    _, history_points_list[i] = fn(dummy_boxes, pts, enable=enable)

        if self.aug_rot_range:
            noise_rotation = float(np.random.uniform(self.aug_rot_range[0], self.aug_rot_range[1]))
            gt_boxes, kf_points = augmentor_utils.global_rotation(
                gt_boxes, kf_points, rot_range=self.aug_rot_range, noise_rotation=noise_rotation,
            )
            dummy_boxes = np.zeros((0, n_cols), dtype=np.float32)
            for i, pts in enumerate(history_points_list):
                _, history_points_list[i] = augmentor_utils.global_rotation(
                    dummy_boxes, pts, rot_range=self.aug_rot_range, noise_rotation=noise_rotation,
                )

        if self.aug_scale_range and (self.aug_scale_range[1] - self.aug_scale_range[0] >= 1e-3):
            noise_scale = float(np.random.uniform(self.aug_scale_range[0], self.aug_scale_range[1]))
            kf_points[:, :3] *= noise_scale
            gt_boxes[:, :6] *= noise_scale
            if n_cols > 7:
                gt_boxes[:, 7:] *= noise_scale
            for pts in history_points_list:
                pts[:, :3] *= noise_scale

        return gt_boxes, kf_points, history_points_list

    def __getitem__(self, index):
        kf_idx = self.sequence_indices[index]
        info   = self.base.infos[kf_idx]

        sweeps     = info.get("sweeps", [])
        subsampled = sweeps[::self.sweep_stride][:self.seq_len][::-1]  # oldest→newest

        history_points = [self._load_sweep_points(sw) for sw in subsampled]  # own ego frame each

        input_dict = self._load_keyframe_input_dict(kf_idx)

        if self.db_sampler is not None and 'gt_boxes' in input_dict:
            # DataBaseSampler pops 'gt_boxes_mask' itself and applies the
            # class-name filtering internally (only when it actually adds
            # sampled boxes) -- prepare_data's own unconditional class-name
            # filtering below covers the case where it doesn't.
            gt_boxes_mask = np.array(
                [n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_
            )
            input_dict = self.db_sampler(data_dict={**input_dict, 'gt_boxes_mask': gt_boxes_mask})

        kf_dict = self._finish_keyframe(input_dict)

        if (self.aug_flip_axes or self.aug_rot_range or self.aug_scale_range) and 'gt_boxes' in kf_dict:
            gt_boxes, kf_points, history_points = self._apply_shared_geometric_aug(
                kf_dict['gt_boxes'], kf_dict['points'], history_points,
            )
            kf_dict['gt_boxes'] = gt_boxes
            kf_dict['points']   = kf_points

        frames = [self._points_to_batch_dict(pts) for pts in history_points]
        frames.append(self.base.collate_batch([kf_dict]))

        return {
            "frames"      : frames,
            "sample_token": info["token"],
            "timestamp"   : info["timestamp"],
        }
