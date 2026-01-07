set -euo pipefail

CFG_FILE="xmem_det/configs/temporal_pp_xmem_nuscenes.yaml"
XMEM_CFG="xmem_det/configs/xmem.yaml"

LATEST_CKPT="./log/phase_ckpt_epoch_5.pth"

EXTRA_TAG="resume_phase1"
WORKERS=4
SEQ_LEN=8
STRIDE=4
MAX_GRAD_NORM=10.0

python train_temporal.py \
  --cfg_file "$CFG_FILE" \
  --xmem_cfg "$XMEM_CFG" \
  --resume_ckpt "$LATEST_CKPT" \
  --extra_tag "$EXTRA_TAG" \
  --workers "$WORKERS" \
  --seq_len "$SEQ_LEN" \
  --stride "$STRIDE" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --phase 1 
