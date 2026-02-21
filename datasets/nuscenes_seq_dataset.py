import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset


def relative_T_lidar(T_world_lidar):
    T_world_lidar = np.asarray(T_world_lidar)
    assert T_world_lidar.ndim == 3 and T_world_lidar.shape[1:] == (4, 4)
    T_rel = []
    T0 = T_world_lidar[0]
    for t in range(T_world_lidar.shape[0]):
        Tt = T_world_lidar[t]
        T = np.linalg.inv(Tt) @ T0
        T_rel.append(T.astype(np.float32))
    return np.stack(T_rel, axis=0)


def collate_seq(batch):
    if len(batch) != 1:
        raise RuntimeError("Use batch_size=1 with collate_seq for now")
    return batch[0]


class NuScenesSeqDataset(Dataset):
    def __init__(self, dataset_cfg, class_names, training, logger,
                 num_groups, root_path=None):
        super().__init__()
        self.base = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger,
        )
        self.num_groups = num_groups
        self.max_sweeps = self.base.dataset_cfg.MAX_SWEEPS
        self.class_names = self.base.class_names

    def __len__(self):
        return len(self.base)

    def get_sweep_groups(self, index):
        """
        Load all sweeps for one keyframe and split into num_groups groups.
        All sweeps are already transformed to the current frame by the base loader.
        """
        info = self.base.infos[index]
        lidar_path = self.base.root_path / info['lidar_path']

        # Current frame — timestamp = 0
        cur_pts = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)[:, :4]
        cur_times = np.zeros((cur_pts.shape[0], 1), dtype=np.float32)

        # Past sweeps — already in current frame's coordinate system
        all_sweeps = [(cur_pts, cur_times)]  # newest first
        for sweep in info['sweeps'][:self.max_sweeps - 1]:
            pts, times = self.base.get_sweep(sweep)
            all_sweeps.append((pts, times))

        # Reverse so oldest → newest
        all_sweeps = list(reversed(all_sweeps))

        # Split into num_groups and concatenate each group
        n = len(all_sweeps)
        groups = []
        for g in range(self.num_groups):
            start = (g * n) // self.num_groups
            end = ((g + 1) * n) // self.num_groups
            pts = np.concatenate([s[0] for s in all_sweeps[start:end]], axis=0)
            times = np.concatenate([s[1] for s in all_sweeps[start:end]], axis=0).astype(pts.dtype)
            groups.append(np.concatenate([pts, times], axis=1))  # [N, 5]

        return groups

    def __getitem__(self, index):
        sweep_groups = self.get_sweep_groups(index)

        # Build a batch_dict for each group through the base prepare_data
        frames = []
        info = self.base.infos[index]
        for g, pts in enumerate(sweep_groups):
            input_dict = {'points': pts, 'frame_id': f"{Path(info['lidar_path']).stem}_g{g}"}

            # Only the final group carries ground truth labels
            if g == self.num_groups - 1:
                base_sample = self.base.__getitem__(index)
                input_dict['gt_names'] = base_sample.get('gt_names', np.array([]))
                input_dict['gt_boxes'] = base_sample.get('gt_boxes', np.zeros((0, 9)))
            else:
                input_dict['gt_names'] = np.array([])
                input_dict['gt_boxes'] = np.zeros((0, 9))

            data_dict = self.base.prepare_data(data_dict=input_dict)
            frames.append(self.base.collate_batch([data_dict]))

        return {'frames': frames}
