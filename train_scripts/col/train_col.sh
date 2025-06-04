#!/bin/bash

export CUDA_VISIBLE_DEVICES=5,6,7
export MASTER_PORT=29585

config=./configs/col/col_train.yaml

torchrun --nnodes=1 --nproc_per_node=3 --master_port=$MASTER_PORT train.py \
  --config $config \




