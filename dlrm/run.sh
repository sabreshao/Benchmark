#!/bin/bash

HIP_VISIBLE_DEVICES=$3 python3 -u -m bind_pyt --nproc_per_node $1 --nsockets_per_node 2 --ncores_per_socket 64 --no_hyperthreads --nnodes=1 --node_rank=0 dlrm_s_pytorch.py --arch-embedding-size="80000-80000-80000-80000-80000-80000-80000-80000" --arch-sparse-feature-size=128 --arch-mlp-bot="128-128-128-128" --arch-mlp-top="512-512-512-256-1" --max-ind-range=40000000 --data-generation=random --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=10 --print-time --memory-map --use-gpu --num-batches=1000 2>&1|tee $2
