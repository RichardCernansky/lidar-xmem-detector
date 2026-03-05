
python eval.py \
    --cfg_file xmem_det/configs/temporal_pp_xmem_nuscenes.yaml \
    --ckpt log/ckpt/phase0_10hz_epoch_3_seq14000.pth \
    --split val \
    --extra_tag default \
    --eval_tag 10hz \
    --log_interval 50 \
    --vis_interval 50 \
    --vis_dir ./attn_vis \
    --vis_n_hist 4 \
    --vis_n_pts 2