#!/bin/bash
# Unattended C4 EVAL driver: rent -> SCREEN GPUS -> setup -> run_c4_eval.sh -> pull -> down.
# Never leaves a box billing; mirrors before any teardown.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SP=/tmp/claude-1000/-home-josef-everything-coding-neuroguessr-2-research/cbfdf82d-e4aa-4b9a-859a-2381fe68b5f8/scratchpad
PY="${PY:-python}"
cd "$REPO" || exit 1
set -a; . "$REPO/.env"; set +a

if [ -f .vast_state.json ]; then
  echo "C4 EVAL DRIVER FINISHED (state-file-exists — refusing to double-rent)"; exit 1
fi

# --- 1. rent + GPU screen (up to 3 attempts; first two are handpicked EU offers) -----
BOX=none
for OFFER in 45410072 44842587 45256631; do
  EXTRA=""; [ -n "$OFFER" ] && EXTRA="--offer-id $OFFER"
  echo "[driver] renting offer='${OFFER:-cheapest-qualifying}'"
  if ! timeout 900 $PY vast.py up --gpu RTX_5090 --gpus 4 --hours 3 --max-price 0.60 --disk 140 $EXTRA; then
    $PY vast.py down > /dev/null 2>&1; rm -f .vast_state.json; continue
  fi
  SYNCED=no                       # sshd can lag 'running' by minutes — retry, don't reject
  for t in 1 2 3 4 5 6; do
    if timeout 300 $PY vast.py sync; then SYNCED=yes; break; fi
    echo "[driver] sync attempt $t failed; waiting 30s for sshd"; sleep 30
  done
  if [ "$SYNCED" != yes ]; then
    $PY vast.py down; rm -f .vast_state.json; continue
  fi
  timeout 420 $PY vast.py run "/venv/main/bin/python /root/auto/tools/screen_gpus.py" \
      > "$SP/screen.log" 2>&1
  cat "$SP/screen.log"
  if ! grep -q "SCREEN PASS" "$SP/screen.log"; then
    echo "[driver] GPU SCREEN FAILED -> destroying this box"
    $PY vast.py down; rm -f .vast_state.json; continue
  fi
  # network screen: the 45GB cache tar and the HF mirrors need real egress (>=30 MB/s).
  # The UK host advertised 1.7Gbps and delivered stalls to both PyPI and HF.
  SPEED=$(timeout 60 $PY vast.py run "curl -sL -o /dev/null -w '%{speed_download}' --max-time 20 --range 0-419430400 -H 'Authorization: Bearer $HF_TOKEN' 'https://huggingface.co/datasets/josefbednar/streetview-acw-ar-cache/resolve/main/cache_n1198072_v3000_s1337.tar'" 2>/dev/null | tr -d '\r' | tail -1 | cut -d. -f1)
  echo "[driver] HF LFS-CDN speed on the actual tar: ${SPEED:-0} B/s"
  if [ -z "$SPEED" ] || [ "$SPEED" -lt 15000000 ] 2>/dev/null; then
    echo "[driver] NET SCREEN FAILED (<15 MB/s on the real artifact) -> destroying this box"
    $PY vast.py down; rm -f .vast_state.json; continue
  fi
  BOX=ok; break
done
if [ "$BOX" != ok ]; then echo "C4 EVAL DRIVER FINISHED (no-healthy-box)"; exit 1; fi
nohup $PY vast.py watchdog > "$SP/watchdog.log" 2>&1 &

# --- 2. deps + FULL 1.2M data cache (vast.py setup restores the wrong 60k subset,
# and prepare.py aborts in interpreter teardown even on success -> grep markers, not codes)
DEPS=no
for t in 1 2 3; do
  if timeout 700 $PY vast.py run "cd /root/auto && /venv/main/bin/python -m pip install -q --no-input --timeout 20 --retries 10 'transformers>=4.55' 'peft>=0.13' 'accelerate>=0.34' 'datasets>=3.0' 'huggingface_hub>=0.25' 'wandb>=0.18' pillow pyarrow pandas requests 'numpy>=1.26' safetensors hf_transfer && echo DEPS_OK" | grep -q DEPS_OK; then DEPS=yes; break; fi
  echo "[driver] pip attempt $t failed; retrying"
done
if [ "$DEPS" != yes ]; then
  echo "[driver] deps failed 3x -> down"
  $PY vast.py down; echo "C4 EVAL DRIVER FINISHED (deps-failed)"; exit 1
fi
timeout 120 $PY vast.py run "cd /root/auto && HF_TOKEN=$HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1 nohup /venv/main/bin/python -u tools/restore_full_cache.py > cache_restore.log 2>&1 & echo RESTORE_LAUNCHED"
CACHE=no
for i in $(seq 1 40); do
  sleep 60
  timeout 90 $PY vast.py run "tail -5 /root/auto/cache_restore.log" > "$SP/cache_tail.log" 2>&1
  if grep -q "CACHE READY" "$SP/cache_tail.log"; then CACHE=yes; break; fi
  if grep -qiE "Traceback|AssertionError" "$SP/cache_tail.log"; then break; fi
done
cat "$SP/cache_tail.log"
if [ "$CACHE" != yes ]; then
  echo "[driver] cache restore failed -> down"
  $PY vast.py down; echo "C4 EVAL DRIVER FINISHED (cache-failed)"; exit 1
fi

# --- 3. launch the pipeline ----------------------------------------------------------
timeout 120 $PY vast.py run "cd /root/auto && HF_TOKEN=$HF_TOKEN WANDB_API_KEY=$WANDB_API_KEY nohup bash retrieval/run_c4_eval.sh > c4eval.log 2>&1 & echo LAUNCHED"

# --- 4. poll (progress tail kept fresh for the human; cap 150 min) -------------------
STATUS=timeout
for i in $(seq 1 150); do
  timeout 90 $PY vast.py run "tail -n 30 /root/auto/c4eval.log" > "$SP/c4eval_tail.log" 2>&1
  if grep -q "C4 EVAL DONE" "$SP/c4eval_tail.log"; then STATUS=done; break; fi
  if grep -qiE "Traceback|CUDA out of memory|Killed" "$SP/c4eval_tail.log"; then STATUS=error; break; fi
  sleep 60
done
echo "[driver] pipeline status: $STATUS"

# --- 5. pull results locally BEFORE teardown -----------------------------------------
timeout 300 $PY vast.py run "cd /root/auto && tail -n 600 c4eval.log" > "$SP/c4eval_full.log" 2>&1
for f in combo_c4.log grid_gate_c4.log levers_c4.log head_c4_5.log head_c4_10.log head_c4_25.log head_c4_50.log; do
  timeout 200 $PY vast.py run "cd /root/auto && cat $f 2>/dev/null | grep -v 'img/s, ETA'" \
      > "$REPO/research/logs/run_c2/c4eval_$f" 2>&1
done

# --- 6. emergency mirror if the pipeline didn't finish its own ----------------------
if [ "$STATUS" != "done" ]; then
  timeout 900 $PY vast.py run "cd /root/auto && PYTHONPATH=/root/auto HF_TOKEN=$HF_TOKEN /venv/main/bin/python - <<'PYEOF'
import glob, os
from huggingface_hub import HfApi
api, R = HfApi(), 'josefbednar/neuroguessr-fullrun2-ckpt'
for f in (glob.glob('run_c4/*.pt') + glob.glob('retrieval_index/geo*_c4.pt') +
          glob.glob('retrieval_index/c2_*_c4_r*.npy') + glob.glob('retrieval_index/c2_regproj.npz') +
          glob.glob('retrieval_index/train_latlon.npz') + glob.glob('retrieval_index/val_meta.npz') +
          glob.glob('*.log')):
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo='c4/'+os.path.basename(f), repo_id=R); print('up', f, flush=True)
    except Exception as e:
        print('FAIL', f, type(e).__name__, flush=True)
print('EMERGENCY MIRROR DONE', flush=True)
PYEOF" >> "$SP/c4eval_mirror.log" 2>&1
fi

# --- 7. teardown + verify ------------------------------------------------------------
$PY vast.py down >> "$SP/c4eval_down.log" 2>&1
sleep 5
$PY vast.py ps >> "$SP/c4eval_down.log" 2>&1
tail -3 "$SP/c4eval_down.log"
echo "C4 EVAL DRIVER FINISHED ($STATUS)"
