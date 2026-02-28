#!/usr/bin/env bash
set -euo pipefail

CFG_FILE="xmem_det/configs/temporal_pp_xmem_nuscenes.yaml"
XMEM_CFG="xmem_det/configs/xmem.yaml"

EXTRA_TAG="10hz"
WORKERS=4

CUDA_VISIBLE_DEVICES=0

python train_temporal.py \
  --cfg_file "$CFG_FILE" \
  --xmem_cfg "$XMEM_CFG" \
  --extra_tag "$EXTRA_TAG" \
  --workers "$WORKERS" \

