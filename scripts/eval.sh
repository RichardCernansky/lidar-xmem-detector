
# python eval.py \
#     --cfg_file xmem_det/configs/temporal_pp_xmem_nuscenes.yaml \
#     --ckpt log/ckpt/phase0_10hz_epoch_3.pth \
#     --split val \
#     --extra_tag default \
#     --eval_tag 10hz \
#     --log_interval 50 \
#     --vis_interval 50 \
#     --vis_dir ./attn_vis \
#     --vis_n_hist 4 \
#     --vis_n_pts 2
    #  --ckpt /home/cernanskyr/pp_xmem_det/log/ckpt/phase0_10hz_epoch_10_seq6000.pth \

python eval.py \
    --cfg_file xmem_det/configs/temporal_pp_xmem_nuscenes.yaml \
     --ckpt log/ckpt/phase0_10hz_epoch_9.pth      \
    --vis_dir ./eval_vis