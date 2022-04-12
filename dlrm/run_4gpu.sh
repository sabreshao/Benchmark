#!/bin/bash
cmd="--arch-embedding-size="80000-80000-80000-80000-80000-80000-80000-80000" --arch-sparse-feature-size=128 --arch-mlp-bot="128-128-128-128" --arch-mlp-top="512-512-512-256-1" --max-ind-range=40000000 --data-generation=random --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=10 --print-time --memory-map --use-gpu --num-batches=1000"

HIP_VISIBLE_DEVICES=4,5,6,7 numactl --cpunodebind=1 --membind=1 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 dlrm_s_pytorch.py ${cmd} 2>&1|tee $1 
