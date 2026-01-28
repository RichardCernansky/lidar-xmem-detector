python eval.py \
  --cfg_file xmem_det/configs/temporal_pp_xmem_nuscenes.yaml \
  --xmem_cfg xmem_det/configs/xmem.yaml \
  --ckpt log/ckpt/phase1_only_seq8_epoch_7.pth \
  --split val \
  --seq_len 9 \
  --alpha 0.55 \
  --eval_tag winner_run 
