#!/usr/bin/env bash
set -euo pipefail

CFG_FILE="xmem_det/configs/temporal_pp_xmem_nuscenes.yaml"
XMEM_CFG="xmem_det/configs/xmem.yaml"
PP_BASELINE_CKPT="/home/cernanskyr/OpenPCDet/output/cfgs/nuscenes_models/cbgs_pp_multihead/default/ckpt/checkpoint_epoch_20.pth"

EXTRA_TAG="phase1"
WORKERS=4
SEQ_LEN=8
STRIDE=4
MAX_GRAD_NORM=10.0


python train_temporal.py \
  --cfg_file "$CFG_FILE" \
  --xmem_cfg "$XMEM_CFG" \
  --pretrained_pp_ckpt "$PP_BASELINE_CKPT" \
  --extra_tag "$EXTRA_TAG" \
  --workers "$WORKERS" \
  --seq_len "$SEQ_LEN" \
  --stride "$STRIDE" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --phase 1

