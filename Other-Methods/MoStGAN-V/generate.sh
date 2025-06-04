CUDA_VISIBLE_DEVICES=0 python src/scripts/generate.py \
--network_pkl /data1/Endora/Endora/pre_trained/mostgan-v/col_network-snapshot-last.pkl \
--num_videos 3125 \
--as_grids false \
--save_as_mp4 true \
--fps 25 \
--video_len 16 \
--batch_size 25 \
--outdir mostgan_col \
--truncation_psi 0.9