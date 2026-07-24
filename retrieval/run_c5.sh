#!/bin/bash
# C5 bed run, unattended (docs/C5_PLAN.md). NO `|| true`: a failed stage fails loudly and the
# driver mirrors whatever exists. Mirror after every artifact-producing stage.
set -e
cd /root/auto
export PYTHONPATH=/root/auto
PY=/venv/main/bin/python
IX=retrieval_index
TR="$PY -m torch.distributed.run --standalone --nproc_per_node=4"
export AR_CKPT_HF_REPO=${AR_CKPT_HF_REPO:-josefbednar/neuroguessr-fullrun2-ckpt}
ts() { date +"[%H:%M:%S] $*"; }

mirror() {   # push whatever exists right now; safe to call repeatedly
  $PY - <<'EOF' || echo "(mirror failed, continuing)"
import glob
import os

from huggingface_hub import HfApi

api, R = HfApi(), os.environ.get("AR_CKPT_HF_REPO", "josefbednar/neuroguessr-fullrun2-ckpt")
files = (glob.glob('run_c5/ckpt_best.pt') + glob.glob('run_c5/ckpt_final.pt') +
         glob.glob('retrieval_index/c2_*_c5_r*.npy') + glob.glob('retrieval_index/c5_ep_*.npy') +
         glob.glob('retrieval_index/geo*_c5.pt') + glob.glob('retrieval_index/c2_regproj.npz') +
         glob.glob('retrieval_index/centroids.npz') + glob.glob('retrieval_index/train_latlon.npz') +
         glob.glob('retrieval_index/val_meta.npz') + glob.glob('retrieval_index/val_patches_c5.npy') +
         glob.glob('*.log') + glob.glob('run_c5_p*/pilot.json'))
for f in files:
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo='c5/' + os.path.basename(f), repo_id=R)
        print('up', f, flush=True)
    except Exception as e:
        print('FAIL', f, type(e).__name__, flush=True)
print('MIRROR DONE', flush=True)
EOF
}

ts "=== 1/8 smoke (40 steps, fresh dirs, probe resolution assert) ==="
rm -rf run_c5_smoke && AR_RESUME=none AR_RUN_DIR=run_c5_smoke AR_MAX_STEPS=40 AR_RUN_NAME=c5-smoke \
  $TR train_c5.py 2>&1 | tee smoke_c5.log | grep -E "Probe:|probe @|step |FAIL|Traceback" | tail -20
grep -q "Probe: .* ceiling @25km 100" smoke_c5.log || { echo "SMOKE FAIL: probe resolution"; exit 1; }

ts "=== 2/8 pilots (3 x 200 steps, ranked by probe@200) ==="
rm -rf run_c5_p0 run_c5_p1 run_c5_p2
AR_RESUME=none AR_RUN_DIR=run_c5_p0 AR_MAX_STEPS=200 AR_RUN_NAME=c5-pilot-base \
  $TR train_c5.py > pilot0.log 2>&1
AR_RESUME=none AR_RUN_DIR=run_c5_p1 AR_MAX_STEPS=200 AR_RUN_NAME=c5-pilot-fullpatch AR_PATCH_KEEP=1.0 \
  $TR train_c5.py > pilot1.log 2>&1
AR_RESUME=none AR_RUN_DIR=run_c5_p2 AR_MAX_STEPS=200 AR_RUN_NAME=c5-pilot-epw035 AR_EP_W=0.35 \
  $TR train_c5.py > pilot2.log 2>&1
PICK=$($PY - <<'EOF'
import re
def probe(f):
    vals = re.findall(r"probe @ 200\] near-recall@1: @25km ([0-9.]+)%", open(f).read())
    return float(vals[-1]) if vals else -1.0
p0, p1, p2 = probe("pilot0.log"), probe("pilot1.log"), probe("pilot2.log")
print(f"pilots probe@200: base {p0} fullpatch {p1} epw035 {p2}", file=__import__("sys").stderr)
env = []
if p1 > p0 + 1.0:                       # full patches only if clearly better (costs ~2x time)
    env.append("AR_PATCH_KEEP=1.0")
if p2 > max(p0, p1) + 1.0:
    env.append("AR_EP_W=0.35")
print(" ".join(env))
EOF
)
echo "[pilots] chosen env: '$PICK'"

ts "=== 3/8 MAIN TRAIN (1 epoch CE + equal episodes, ~9360 steps) ==="
rm -rf run_c5
env $PICK AR_RESUME=none AR_RUN_DIR=run_c5 AR_RUN_NAME=c5-bed-main \
  $TR train_c5.py 2>&1 | tee train_c5.log | grep -E "step .*img/s|probe @|median_km|\[ckpt\]|FAIL"
grep -q "median_km:" train_c5.log || { echo "TRAIN FAIL: no final eval"; exit 1; }
mirror > mirror_train.log 2>&1 &

ts "=== 4/8 embed train (cls/mean/reg + gate top-50) ==="
$TR retrieval/embed_c2.py --split train --img-size 384 --tag c5 \
  --ckpt run_c5/ckpt_best.pt --out $IX --bs 64 --workers 10 --train-gate-topk 50
mirror > mirror_embed.log 2>&1 &

ts "=== 5/8 embed val (logits gate) + centroids export + val patches ==="
$TR retrieval/embed_c2.py --split val --img-size 384 --tag c5 \
  --ckpt run_c5/ckpt_best.pt --out $IX --bs 64 --workers 8
$PY - <<'EOF'
import numpy as np, torch
ck = torch.load("run_c5/ckpt_best.pt", map_location="cpu", weights_only=False)
c = torch.nn.functional.normalize(ck["buffers"]["centroids"].float(), dim=-1).numpy()
np.savez("retrieval_index/centroids.npz", cent=c)
print("centroids exported", c.shape)
EOF
CUDA_VISIBLE_DEVICES=0 $PY tools/dump_val_patches.py --ckpt run_c5/ckpt_best.pt &
VPPID=$!

ts "=== 6/8 episode cache (train-side gated top-50) ==="
$TR tools/build_episode_cache.py --index $IX --tag c5 --ckpt run_c5/ckpt_best.pt
wait $VPPID
mirror > mirror_cache.log 2>&1 &

ts "=== 7/8 band heads 5/10/25 (offline, one per GPU) ==="
for i in 0 1 2; do
  D=$(echo "5 10 25" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$i $PY retrieval/train_geo_head.py --index-dir $IX \
      --emb-prefix c2_cls_train_c5_r --d-pos $D --out geo${D}_cls_c5.pt > head_c5_$D.log 2>&1 &
done
wait

ts "=== 8/8 manifest check + final mirror ==="
$PY tools/check_bed_manifest.py --index $IX --tag c5
mirror
ts "C5 PIPELINE DONE"
