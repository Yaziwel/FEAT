#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python sample/sample_ddp.py \
    --config ./configs/kva/kva_sample.yaml \
    --ckpt /path/to/ckpt \
    --port 29757 \
    --save_video_path /path/to/save
