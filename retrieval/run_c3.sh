#!/bin/bash
# C3 end-to-end: contrastive backbone fine-tune -> re-index @384 -> head -> grid.
# Dropped vs C2 (measured duds): query TTA, multi-res 512+384 ensemble, gate-mass sweep.
set -e
cd /root/auto
export PYTHONPATH=/root/auto
PY=/venv/main/bin/python
IX=retrieval_index
TR="$PY -m torch.distributed.run --standalone --nproc_per_node=4"
STEPS=${STEPS:-2500}
ts() { date +"[%H:%M:%S] $*"; }

ts "=== 1/5 contrastive backbone fine-tune ($STEPS steps) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/train_c3_contrastive.py \
    --ckpt run_full2/ckpt_best.pt --out run_c3 --steps $STEPS --img-size 384 --pairs 32

ts "=== 2/5 re-index @384 with the fine-tuned encoder ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split train --img-size 384 --tag c3 \
    --ckpt run_c3/ckpt.pt --out $IX --bs 64 --workers 10

ts "=== 3/5 val sweep @384 with the fine-tuned encoder ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split val --img-size 384 --tag c3 \
    --ckpt run_c3/ckpt.pt --out $IX --bs 64 --workers 8

ts "=== 4/5 geo-smooth head on the new mean-patch descriptor ==="
cp run_c3/c3_head.pt $IX/c3_head.pt
CUDA_VISIBLE_DEVICES=0 $PY retrieval/train_geo_head.py --index-dir $IX \
    --emb-prefix c2_mean_train_c3_r --d-pos 10 --out geo10_mean_c3.pt

ts "=== 5/5 grid (gate = old-checkpoint 384 logits, unchanged) ==="
CUDA_VISIBLE_DEVICES=0 $PY retrieval/eval_c2_gpu.py --index $IX --gate 384 --tag c3 \
    --heads "c3head:c3_head.pt:cls,geo10m:geo10_mean_c3.pt:mean" --stages ABDF
ts "C3 PIPELINE DONE"
