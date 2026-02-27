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

    - Sweep frames: backbone only, no gt_boxes, no detection loss.
    - Keyframe    : backbone + detection head + loss (has gt_boxes).
    - Bank resets at the start of every forward() call, so each item is
      an independent episode — no cross-keyframe memory.

    Args:
        seq_len     : max sweep frames BEFORE the keyframe.
                      Set to MAX_SWEEPS // sweep_stride in config.
        stride      : keyframe-level stride, subsample training set for speed.
        sweep_stride: every N-th sweep is kept; 2 gives ~10 Hz from ~20 Hz.
    """

    def __init__(
        self,
        dataset_cfg,
        class_names,
        training,
        logger,
        seq_len,
        stride,
        nusc_version,    # kept for API compat, unused
        nusc_dataroot,   # kept for API compat, unused
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
        # One index per keyframe; stride subsamples for training speed.
        return list(range(0, len(self.base.infos), self.stride))

    def __len__(self):
        return len(self.sequence_indices)

    def _load_sweep_points(self, sweep_info: dict) -> np.ndarray:
        # Delegate entirely to OpenPCDet's get_sweep which handles
        # file loading, ego-point removal, transform_matrix application,
        # and time_lag — returns (points [N,4], times [N,1]).
        points, times = self.base.get_sweep(sweep_info)
        return np.concatenate([points, times], axis=1).astype(np.float32)  # [N,5]

    def _points_to_batch_dict(self, points: np.ndarray) -> dict:
        # Run voxelization only (no augmentation) then collate.
        data_dict = {"points": points}
        data_dict = self.base.data_processor.forward(data_dict=data_dict)
        return self.base.collate_batch([data_dict])

    def __getitem__(self, index):
        kf_idx = self.sequence_indices[index]
        info   = self.base.infos[kf_idx]

        # info['sweeps']: newest→oldest, length up to MAX_SWEEPS.
        # Take every sweep_stride-th, cap at seq_len, reverse to oldest→newest.
        sweeps     = info.get("sweeps", [])
        subsampled = sweeps[::self.sweep_stride][:self.seq_len][::-1]

        frames = []

        # Sweep frames: backbone only, no gt_boxes
        for sw in subsampled:
            pts = self._load_sweep_points(sw)
            frames.append(self._points_to_batch_dict(pts))

        # Keyframe: disable augmentor so it stays in the same coordinate
        # frame as the sweep frames (augmentation would break bank attention).
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
