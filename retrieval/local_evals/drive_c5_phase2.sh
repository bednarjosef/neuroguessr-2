#!/bin/bash
# C5 driver PHASE 2 (recovery): poll -> pull -> mirror-if-needed -> down. Box already
# running run_c5_cont.sh (main train started 17:32 box time).
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SP=/tmp/claude-1000/-home-josef-everything-coding-neuroguessr-2-research/cbfdf82d-e4aa-4b9a-859a-2381fe68b5f8/scratchpad
PY="${PY:-python}"
V=$(command -v vastai || echo "$HOME/.local/bin/vastai")
cd "$REPO" || exit 1
set -a; . "$REPO/.env"; set +a

# --- 4. poll (cap 280 min) + balance guard --------------------------------------------
STATUS=timeout
for i in $(seq 1 240); do
  timeout 90 $PY vast.py run "tail -n 25 /root/auto/c5.log" > "$SP/c5_tail.log" 2>&1
  if grep -q "C5 PIPELINE DONE" "$SP/c5_tail.log"; then STATUS=done; break; fi
  if grep -qiE "SMOKE FAIL|TRAIN FAIL|MANIFEST MISSING|CUDA out of memory|Killed|FAIL: loss diverged" "$SP/c5_tail.log"; then STATUS=error; break; fi
  if [ $((i % 10)) -eq 0 ]; then
    BAL=$($V show user --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('balance',0)+d.get('credit',0))" 2>/dev/null | cut -d. -f1)
    if [ -n "$BAL" ] && [ "$BAL" -lt 1 ] 2>/dev/null; then STATUS=low-balance; break; fi
  fi
  sleep 60
done
echo "[driver] pipeline status: $STATUS"

# --- 5. pull results BEFORE teardown --------------------------------------------------
mkdir -p "$REPO/research/logs/run_c5"
timeout 300 $PY vast.py run "cd /root/auto && tail -n 800 c5.log" > "$REPO/research/logs/run_c5/c5_full.log" 2>&1
for f in train_c5.log smoke_c5.log pilot0.log pilot1.log pilot2.log head_c5_5.log head_c5_10.log head_c5_25.log; do
  timeout 200 $PY vast.py run "cd /root/auto && tail -n 150 $f 2>/dev/null | grep -v 'img/s, ETA'" > "$REPO/research/logs/run_c5/$f" 2>&1
done

# --- 6. emergency mirror if the pipeline didn't finish its own ------------------------
if [ "$STATUS" != "done" ]; then
  timeout 1200 $PY vast.py run "cd /root/auto && PYTHONPATH=/root/auto HF_TOKEN=$HF_TOKEN AR_CKPT_HF_REPO=josefbednar/neuroguessr-fullrun2-ckpt /venv/main/bin/python - <<'PYEOF'
import glob, os
from huggingface_hub import HfApi
api, R = HfApi(), 'josefbednar/neuroguessr-fullrun2-ckpt'
for f in (glob.glob('run_c5/ckpt_best.pt') + glob.glob('run_c5/ckpt_last.pt') +
          glob.glob('retrieval_index/c2_*_c5_r*.npy') + glob.glob('retrieval_index/c5_ep_*.npy') +
          glob.glob('retrieval_index/geo*_c5.pt') + glob.glob('retrieval_index/*.npz') + glob.glob('*.log')):
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo='c5/'+os.path.basename(f), repo_id=R); print('up', f, flush=True)
    except Exception as e:
        print('FAIL', f, type(e).__name__, flush=True)
print('EMERGENCY MIRROR DONE', flush=True)
PYEOF" >> "$SP/c5_mirror.log" 2>&1
fi

# --- 7. teardown + verify -------------------------------------------------------------
$PY vast.py down >> "$SP/c5_down.log" 2>&1
sleep 5
$PY vast.py ps >> "$SP/c5_down.log" 2>&1
tail -3 "$SP/c5_down.log"
echo "C5 DRIVER FINISHED ($STATUS)"
