#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3
export MASTER_PORT=29624

config=./configs/kva/kva_train.yaml

torchrun --nnodes=1 --nproc_per_node=3 --master_port=$MASTER_PORT train.py \
  --config $config \




