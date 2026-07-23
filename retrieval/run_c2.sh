#!/bin/bash
# C2 end-to-end on the box: 512px multi-descriptor index -> geo heads -> staged grid.
# Requires HF_TOKEN in the environment. Run with nohup from /root/auto.
set -e
cd /root/auto
export PYTHONPATH=/root/auto
PY=/venv/main/bin/python
IX=retrieval_index
TR="$PY -m torch.distributed.run --standalone --nproc_per_node=4"
ts() { date +"[%H:%M:%S] $*"; }

ts "rows: $($PY -c "import prepare,pandas as pd;print(len(pd.read_parquet(prepare.TRAIN_META)))")"

ts "=== 1/6 train index sweep @512px ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split train --img-size 512 \
    --ckpt run_full2/ckpt_best.pt --out $IX --bs 48 --workers 10

ts "=== 2/6 val sweep @512px (+flip/zoom TTA) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split val --img-size 512 \
    --ckpt run_full2/ckpt_best.pt --out $IX --bs 32 --workers 8 --tta

ts "=== 3/6 val sweep @384px (gate logits + old-index query side) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split val --img-size 384 --tag s384 \
    --ckpt run_full2/ckpt_best.pt --out $IX --bs 48 --workers 8 --no-regions

ts "=== 4/6 row-order alignment check vs the old 384 index ==="
$PY -c "
import numpy as np
z=np.load('$IX/train_latlon.npz'); lat,lon=z['lat'],z['lon']
print('N', len(lat))
idx=[0,1,2,3,100000,599035,1000000,len(lat)-1]
print('sample lat/lon:', [(round(float(lat[i]),5), round(float(lon[i]),5)) for i in idx])
"

ts "=== 5/6 geo-smooth heads on the 512 descriptors ==="
CUDA_VISIBLE_DEVICES=0 $PY retrieval/train_geo_head.py --index-dir $IX \
    --emb-prefix c2_cls_train_s512_r --d-pos 10 --out geo10_cls512.pt
CUDA_VISIBLE_DEVICES=0 $PY retrieval/train_geo_head.py --index-dir $IX \
    --emb-prefix c2_mean_train_s512_r --d-pos 10 --out geo10_mean512.pt

ts "=== 6/6 staged grid ==="
CUDA_VISIBLE_DEVICES=0 $PY retrieval/eval_c2_gpu.py --index $IX --gate 384 --stages ABCDF
ts "C2 PIPELINE DONE"
