import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils

from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar


def to_torch_batch_dict(frame_dict, device):
    batch_dict = {}
    for k, v in frame_dict.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                batch_dict[k] = v
            else:
                batch_dict[k] = torch.from_numpy(v).to(device, non_blocking=True)
        elif isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device, non_blocking=True)
        else:
            batch_dict[k] = v
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--num_groups", type=int, default=None,
                   help="Number of sweep groups per keyframe. Defaults to TRAIN.SEQ_LEN from config.")
    p.add_argument("--extra_tag", type=str, default="default")
    p.add_argument("--eval_tag", type=str, default="sweep_eval")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.split is not None:
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    if args.num_groups is None:
        args.num_groups = int(getattr(cfg.TRAIN, "SEQ_LEN", 4))

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    return args


def main():
    args = parse_args()

    root_dir = getattr(cfg, "ROOT_DIR", Path.cwd())
    output_dir = Path(root_dir) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    eval_output_dir = output_dir / "eval_sweep" / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print(eval_output_dir, "------------------")
    log_file = eval_output_dir / ("log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info("**********************Start logging**********************")
    logger.info(f"cfg_file={args.cfg_file}")
    logger.info(f"ckpt={args.ckpt}")
    logger.info(f"split={cfg.DATA_CONFIG.DATA_SPLIT['test']}")
    logger.info(f"num_groups={args.num_groups}")
    log_config_to_file(cfg, logger=logger)

    test_set = NuScenesSeqDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=None,
        logger=logger,
        num_groups=int(args.num_groups),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_seq,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set.base,
        pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
    ).to(device)

    blob = torch.load(args.ckpt, map_location="cpu")
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    logger.info("Model loaded and set to eval mode")

    det_annos = [None] * len(test_set)
    start_t = time.time()

    with torch.no_grad():
        for sample_idx, seq in enumerate(test_loader):
            if hasattr(model, "reset_sequence"):
                model.reset_sequence(sample_idx)

            frames = seq["frames"]
            frames_list = [to_torch_batch_dict(f, device) for f in frames]

            pred_dicts, recall_dicts, _ = model(
                frames_list=frames_list,
                compute_det_loss=False,
            )

            # Use a clean base item for annotation generation so that
            # frame_id and metadata are correct (not the _gN suffixed ones)
            base_item = test_set.base.__getitem__(sample_idx)
            base_batch_cpu = test_set.base.collate_batch([base_item])

            annos = test_set.base.generate_prediction_dicts(
                batch_dict=base_batch_cpu,
                pred_dicts=pred_dicts,
                class_names=cfg.CLASS_NAMES,
                output_path=None,
            )
            det_annos[sample_idx] = annos[0]

            if (sample_idx + 1) % int(args.log_interval) == 0:
                elapsed = time.time() - start_t
                pct = 100.0 * (sample_idx + 1) / len(test_set)
                eta = elapsed / (sample_idx + 1) * (len(test_set) - sample_idx - 1)
                logger.info(
                    f"Eval: {sample_idx + 1}/{len(test_set)} ({pct:.1f}%) "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

    missing_idxs = [i for i, a in enumerate(det_annos) if a is None]
    if missing_idxs:
        raise RuntimeError(
            f"Missing predictions for {len(missing_idxs)} samples, "
            f"first missing index={missing_idxs[0]}"
        )

    logger.info("All predictions generated successfully")

    eval_metric = getattr(cfg.MODEL.POST_PROCESSING, "EVAL_METRIC", "nuscenes")
    result_str, result_dict = test_set.base.evaluation(
        det_annos=det_annos,
        class_names=cfg.CLASS_NAMES,
        eval_metric=eval_metric,
        output_path=str(eval_output_dir),
    )

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(result_str)
    logger.info("=" * 80)
    logger.info(f"Results dict: {result_dict}")
    logger.info("=" * 80 + "\n")

    print("\n" + result_str)


if __name__ == "__main__":
    main()
