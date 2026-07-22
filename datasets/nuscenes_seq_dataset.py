import numpy as np
from torch.utils.data import Dataset
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
    """

    def __init__(
        self,
        dataset_cfg,
        class_names,
        training,
        logger,
        seq_len,
        stride,
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
        self.seq_len      = int(seq_len)
        self.stride       = int(stride)
        self.sweep_stride = int(sweep_stride)
        self.class_names  = self.base.class_names
        self.sequence_indices = self._build_sequences()

    def _build_sequences(self):
        return list(range(0, len(self.base.infos), self.stride))

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

    def __getitem__(self, index):
        kf_idx = self.sequence_indices[index]
        info   = self.base.infos[kf_idx]

        sweeps     = info.get("sweeps", [])
        subsampled = sweeps[::self.sweep_stride][:self.seq_len][::-1]  # oldest→newest

        frames = []

        for sw in subsampled:
            pts = self._load_sweep_points(sw)           # own ego frame
            frames.append(self._points_to_batch_dict(pts))

        # Keyframe: disable augmentor to keep consistent coordinate frame
        was_training       = self.base.training
        self.base.training = False
        kf_raw             = self.base.__getitem__(kf_idx)
        self.base.training = was_training
        frames.append(self.base.collate_batch([kf_raw]))

        return {
            "frames"      : frames,
            "sample_token": info["token"],
            "timestamp"   : info["timestamp"],
        }