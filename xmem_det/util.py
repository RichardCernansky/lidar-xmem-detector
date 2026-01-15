import torch
import yaml

def load_xmem_train_cfg(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def boxes_to_bev_masks(
    boxes_lidar: torch.Tensor,
    scores: torch.Tensor,
    bev_h: int,
    bev_w: int,
    pc_range,
    score_thresh: float = 0.3,
):
    device = boxes_lidar.device
    B, N, _ = boxes_lidar.shape

    x_min_range, y_min_range, _, x_max_range, y_max_range, _ = pc_range
    x_size = x_max_range - x_min_range
    y_size = y_max_range - y_min_range

    voxel_x = x_size / float(bev_w)
    voxel_y = y_size / float(bev_h)

    masks = torch.zeros(B, N, bev_h, bev_w, device=device, dtype=torch.float32)

    valid = scores > score_thresh
    if valid.sum() == 0:
        return masks

    cx = boxes_lidar[..., 0]
    cy = boxes_lidar[..., 1]
    dx = boxes_lidar[..., 3]
    dy = boxes_lidar[..., 4]

    bx_min = cx - 0.5 * dx
    bx_max = cx + 0.5 * dx
    by_min = cy - 0.5 * dy
    by_max = cy + 0.5 * dy

    ix_min = ((bx_min - x_min_range) / voxel_x).floor().clamp(0, bev_w - 1).long()
    ix_max = ((bx_max - x_min_range) / voxel_x).ceil().clamp(0, bev_w - 1).long()
    iy_min = ((by_min - y_min_range) / voxel_y).floor().clamp(0, bev_h - 1).long()
    iy_max = ((by_max - y_min_range) / voxel_y).ceil().clamp(0, bev_h - 1).long()

    for b in range(B):
        for n in range(N):
            if not bool(valid[b, n]):
                continue
            x0 = int(ix_min[b, n])
            x1 = int(ix_max[b, n]) + 1
            y0 = int(iy_min[b, n])
            y1 = int(iy_max[b, n]) + 1
            if x0 >= x1 or y0 >= y1:
                continue
            masks[b, n, y0:y1, x0:x1] = 1.0

    
    return masks
