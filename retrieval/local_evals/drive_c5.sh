#!/bin/bash
# C5 bed-run driver: rent (screened) -> deps -> full cache -> run_c5.sh -> pull -> down.
# Never leaves a box billing; mirrors before any teardown; balance-guarded.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SP=/tmp/claude-1000/-home-josef-everything-coding-neuroguessr-2-research/cbfdf82d-e4aa-4b9a-859a-2381fe68b5f8/scratchpad
PY="${PY:-python}"
V=$(command -v vastai || echo "$HOME/.local/bin/vastai")
cd "$REPO" || exit 1
set -a; . "$REPO/.env"; set +a

if [ -f .vast_state.json ]; then
  echo "C5 DRIVER FINISHED (state-file-exists — refusing to double-rent)"; exit 1
fi

# --- 1. rent + GPU screen + net screen (Estonia box proven today; then HU, SE) --------
BOX=none
for OFFER in 45543732 45702192 39399824; do
  echo "[driver] renting offer='$OFFER'"
  if ! timeout 900 $PY vast.py up --gpu RTX_5090 --gpus 4 --hours 4.5 --max-price 0.60 --disk 140 --offer-id $OFFER; then
    $PY vast.py down > /dev/null 2>&1; rm -f .vast_state.json; continue
  fi
  SYNCED=no
  for t in 1 2 3 4 5 6; do
    if timeout 300 $PY vast.py sync; then SYNCED=yes; break; fi
    echo "[driver] sync attempt $t failed; waiting 30s for sshd"; sleep 30
  done
  if [ "$SYNCED" != yes ]; then $PY vast.py down; rm -f .vast_state.json; continue; fi
  timeout 420 $PY vast.py run "/venv/main/bin/python /root/auto/tools/screen_gpus.py" > "$SP/c5_screen.log" 2>&1
  cat "$SP/c5_screen.log"
  if ! grep -q "SCREEN PASS" "$SP/c5_screen.log"; then
    echo "[driver] GPU SCREEN FAILED -> destroying"; $PY vast.py down; rm -f .vast_state.json; continue
  fi
  SPEED=$(timeout 60 $PY vast.py run "curl -sL -o /dev/null -w '%{speed_download}' --max-time 20 --range 0-419430400 -H 'Authorization: Bearer $HF_TOKEN' 'https://huggingface.co/datasets/josefbednar/streetview-acw-ar-cache/resolve/main/cache_n1198072_v3000_s1337.tar'" 2>/dev/null | tr -d '\r' | tail -1 | cut -d. -f1)
  echo "[driver] HF LFS speed: ${SPEED:-0} B/s"
  if [ -z "$SPEED" ] || [ "$SPEED" -lt 15000000 ] 2>/dev/null; then
    echo "[driver] NET SCREEN FAILED -> destroying"; $PY vast.py down; rm -f .vast_state.json; continue
  fi
  BOX=ok; break
done
if [ "$BOX" != ok ]; then echo "C5 DRIVER FINISHED (no-healthy-box)"; exit 1; fi
nohup $PY vast.py watchdog > "$SP/c5_watchdog.log" 2>&1 &

# --- 2. deps + full 1.2M cache --------------------------------------------------------
DEPS=no
for t in 1 2 3; do
  if timeout 700 $PY vast.py run "cd /root/auto && /venv/main/bin/python -m pip install -q --no-input --timeout 20 --retries 10 'transformers>=4.55' 'peft>=0.13' 'accelerate>=0.34' 'datasets>=3.0' 'huggingface_hub>=0.25' 'wandb>=0.18' pillow pyarrow pandas requests 'numpy>=1.26' safetensors && echo DEPS_OK" | grep -q DEPS_OK; then DEPS=yes; break; fi
  echo "[driver] pip attempt $t failed; retrying"
done
[ "$DEPS" = yes ] || { $PY vast.py down; echo "C5 DRIVER FINISHED (deps-failed)"; exit 1; }
timeout 120 $PY vast.py run "cd /root/auto && HF_TOKEN=$HF_TOKEN nohup /venv/main/bin/python -u tools/restore_full_cache.py > cache_restore.log 2>&1 & echo RESTORE_LAUNCHED"
CACHE=no
for i in $(seq 1 40); do
  sleep 60
  timeout 90 $PY vast.py run "tail -5 /root/auto/cache_restore.log" > "$SP/c5_cache_tail.log" 2>&1
  if grep -q "CACHE READY" "$SP/c5_cache_tail.log"; then CACHE=yes; break; fi
  if grep -qiE "Traceback|AssertionError" "$SP/c5_cache_tail.log"; then break; fi
done
cat "$SP/c5_cache_tail.log"
[ "$CACHE" = yes ] || { $PY vast.py down; echo "C5 DRIVER FINISHED (cache-failed)"; exit 1; }

# --- 3. launch the pipeline -----------------------------------------------------------
timeout 120 $PY vast.py run "cd /root/auto && HF_TOKEN=$HF_TOKEN WANDB_API_KEY=$WANDB_API_KEY AR_CKPT_HF_REPO=josefbednar/neuroguessr-fullrun2-ckpt nohup bash retrieval/run_c5.sh > c5.log 2>&1 & echo LAUNCHED"

# --- 4. poll (cap 280 min) + balance guard --------------------------------------------
STATUS=timeout
for i in $(seq 1 240); do
  timeout 90 $PY vast.py run "tail -n 25 /root/auto/c5.log" > "$SP/c5_tail.log" 2>&1
  if grep -q "C5 PIPELINE DONE" "$SP/c5_tail.log"; then STATUS=done; break; fi
  if grep -qiE "SMOKE FAIL|TRAIN FAIL|MANIFEST MISSING|CUDA out of memory|Killed|FAIL: loss diverged" "$SP/c5_tail.log"; then STATUS=error; break; fi
  if [ $((i % 10)) -eq 0 ]; then
    BAL=$($V show user --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('balance',0)+d.get('credit',0))" 2>/dev/null | cut -d. -f1)
    if [ -n "$BAL" ] && [ "$BAL" -lt 2 ] 2>/dev/null; then STATUS=low-balance; break; fi
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
