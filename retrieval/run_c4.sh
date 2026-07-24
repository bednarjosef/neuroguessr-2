#!/bin/bash
# C4 end-to-end, unattended: joint fine-tune -> re-index -> offline band heads -> grids ->
# MIRROR EVERYTHING TO HF (the 2026-07-24 lesson) . The local driver destroys the box after.
set -e
cd /root/auto
export PYTHONPATH=/root/auto
PY=/venv/main/bin/python
IX=retrieval_index
TR="$PY -m torch.distributed.run --standalone --nproc_per_node=4"
STEPS=${STEPS:-1800}
ts() { date +"[%H:%M:%S] $*"; }

mirror() {   # push whatever exists right now; safe to call repeatedly
  $PY - <<'EOF' || echo "(mirror failed, continuing)"
import os, glob
from huggingface_hub import HfApi
api, R = HfApi(), 'josefbednar/neuroguessr-fullrun2-ckpt'
files = (glob.glob('run_c4/*.pt') + glob.glob('retrieval_index/geo*_c4.pt') +
         glob.glob('retrieval_index/c2_*_c4_r*.npy') + glob.glob('retrieval_index/c2_regproj.npz') +
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

ts "=== 1/6 C4 joint fine-tune ($STEPS steps) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/train_c4.py --ckpt run_c3/ckpt.pt --out run_c4 \
    --steps $STEPS --pairs 16 --img-size 384 --no-ckpt --workers 8 --probe-every 300 --kill-step 1200
cp run_c4/c4_head.pt $IX/c4_head.pt || true
mirror

ts "=== 2/6 re-index @384 with the C4 encoder ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split train --img-size 384 --tag c4 \
    --ckpt run_c4/ckpt.pt --out $IX --bs 64 --workers 10

ts "=== 3/6 val sweep @384 (C4 logits — the gate should now be usable single-pass) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR retrieval/embed_c2.py --split val --img-size 384 --tag c4 \
    --ckpt run_c4/ckpt.pt --out $IX --bs 64 --workers 8

ts "=== 4/6 offline band heads 5/10/25/50 km, one per GPU ==="
for i in 0 1 2 3; do
  D=$(echo "5 10 25 50" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$i $PY retrieval/train_geo_head.py --index-dir $IX \
      --emb-prefix c2_cls_train_c4_r --d-pos $D --out geo${D}_cls_c4.pt > head_c4_$D.log 2>&1 &
done
wait
mirror

ts "=== 5/6 grids: gate comparison, then the full lever combo ==="
CUDA_VISIBLE_DEVICES=0 $PY retrieval/eval_c2_gpu.py --index $IX --gate c4 --tag c4 \
    --heads "c4head:c4_head.pt:cls,geo10c:geo10_cls_c4.pt:cls" --stages ABDF > grid_gate_c4.log 2>&1 || true
CUDA_VISIBLE_DEVICES=1 $PY retrieval/eval_c2_gpu.py --index $IX --gate s384 --tag c4 \
    --heads "c4head:c4_head.pt:cls,geo10c:geo10_cls_c4.pt:cls" --stages ABDF > grid_gate_old.log 2>&1 || true
CUDA_VISIBLE_DEVICES=2 $PY retrieval/eval_levers.py --index $IX --tag c4 --gate c4 > levers_c4.log 2>&1 || true
CUDA_VISIBLE_DEVICES=3 $PY retrieval/eval_levers.py --index $IX --tag c4 --gate c4 --combo > combo_c4.log 2>&1 || true

ts "=== 6/6 final mirror (everything, including the index arrays) ==="
mirror
ts "C4 PIPELINE DONE"
