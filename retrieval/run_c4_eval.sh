#!/bin/bash
# C4 EVAL rescue (2026-07-24): the encoder is already trained and mirrored (HF c4/ckpt.pt) —
# the first box died before producing any number. This rebuilds the index and runs the grids:
#   fetch ckpt -> re-index train -> val sweep -> offline band heads -> grids -> mirror.
# No training. Mirrors aggressively (big npys right after they exist, not at the end).
set -e
cd /root/auto
export PYTHONPATH=/root/auto
PY=/venv/main/bin/python
IX=retrieval_index
TR="$PY -m torch.distributed.run --standalone --nproc_per_node=4"
ts() { date +"[%H:%M:%S] $*"; }

mirror() {   # push whatever exists right now; safe to call repeatedly
  $PY - <<'EOF' || echo "(mirror failed, continuing)"
import glob
import os

from huggingface_hub import HfApi

api, R = HfApi(), 'josefbednar/neuroguessr-fullrun2-ckpt'
files = (glob.glob('run_c4/*.pt') + glob.glob('retrieval_index/geo*_c4.pt') +
         glob.glob('retrieval_index/c2_*_c4_r*.npy') + glob.glob('retrieval_index/c2_regproj.npz') +
         glob.glob('retrieval_index/train_latlon.npz') + glob.glob('retrieval_index/val_meta.npz') +
         glob.glob('*.log'))
for f in files:
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo='c4/' + os.path.basename(f), repo_id=R)
        print('up', f, flush=True)
    except Exception as e:
        print('FAIL', f, type(e).__name__, flush=True)
print('MIRROR DONE', flush=True)
EOF
}

ts "=== 0/5 fetch the C4 encoder from HF ==="
$PY - <<'EOF'
import os
import shutil

from huggingface_hub import hf_hub_download

os.makedirs('run_c4', exist_ok=True)
for f in ['ckpt.pt', 'c4_head.pt']:
    p = hf_hub_download('josefbednar/neuroguessr-fullrun2-ckpt', 'c4/' + f)
    shutil.copy(p, 'run_c4/' + f)
    print('got', f, flush=True)
EOF
mkdir -p $IX
cp run_c4/c4_head.pt $IX/c4_head.pt

ts "=== 1/5 re-index train @384 with the C4 encoder ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split train --img-size 384 --tag c4 \
    --ckpt run_c4/ckpt.pt --out $IX --bs 64 --workers 10
mirror > mirror1.log 2>&1 &   # push the index arrays NOW, overlapped with the val sweep

ts "=== 2/5 val sweep @384 (C4 logits -> single-pass gate) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split val --img-size 384 --tag c4 \
    --ckpt run_c4/ckpt.pt --out $IX --bs 64 --workers 8

ts "=== 3/5 offline band heads 5/10/25/50 km, one per GPU ==="
for i in 0 1 2 3; do
  D=$(echo "5 10 25 50" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$i $PY retrieval/train_geo_head.py --index-dir $IX \
      --emb-prefix c2_cls_train_c4_r --d-pos $D --out geo${D}_cls_c4.pt > head_c4_$D.log 2>&1 &
done
wait

ts "=== 4/5 grids IN PARALLEL (the morning run had them sequential) ==="
CUDA_VISIBLE_DEVICES=0 $PY retrieval/eval_levers.py --index $IX --tag c4 --gate c4 --combo > combo_c4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY retrieval/eval_c2_gpu.py --index $IX --gate c4 --tag c4 \
    --heads "c4head:c4_head.pt:cls,geo10c:geo10_cls_c4.pt:cls" --stages ABDF > grid_gate_c4.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY retrieval/eval_levers.py --index $IX --tag c4 --gate c4 > levers_c4.log 2>&1 &
wait || true

ts "=== 5/5 final mirror (heads, logs, val arrays) ==="
mirror
ts "C4 EVAL DONE"
