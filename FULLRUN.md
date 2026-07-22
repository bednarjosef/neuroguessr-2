# Full-dataset training run — runbook

One long training run of the champion stack on **all 1,198,072 train images** (300k locations
× 4 views, 512×512, ~115 GB) instead of the frozen 60k ratchet subset. Same architecture as
`train.py` @ champion, same frozen evaluator, same val split — scores are directly comparable
to the ratchet history. This is **not** a ratchet experiment: it uses `train_full.py`,
which never touches `train.py`/`results.tsv`.

## The decisions (and why)

| Question | Decision |
|---|---|
| **Steps** | **3 epochs** (`AR_EPOCHS=3`) — 37,437 steps @ 1 GPU / 18,717 @ 2 / 9,360 @ 4 (global batch 96×N). LoRA r16 (10M params) underfits before it overfits, and champion runs were still improving at cutoff, so full epochs > early stop. |
| **Geocells** | **5000** fine cells (`AR_CELLS`, was 2048 at 60k). 1.2M/5000 = 240 imgs/cell — the same density PIGEON trained at (500k/2000). Lowers the quantization floor (the S5 lesson: the floor binds once classification gets good). The floor is printed at startup (`Val quantization floor @ …`). Set `AR_CELLS=2048` to replicate champion cells exactly. |
| **Eval cadence** | Every **~100k samples** (= every 1041 steps @ 1 GPU, 260 @ 4 — ~36 evals total, ~12 min of eval over the run) on the fixed 1000-image val subset. **Step-0 eval included.** Final official eval on the full 2998 val. |
| **Checkpoints** | Every **~200k samples** (`ckpt_last.pt`) + on every **new best** val median (`ckpt_best.pt`) + `ckpt_final.pt`. ~130 MB each (trainable params + ALL buffers incl. geocells + optimizer + exact step/epoch/batch position + W&B id) — full-fidelity resume, ~2 s to save, atomic write. Optional `AR_CKPT_HF_REPO=user/repo` uploads best/last to a private HF repo in the background → a dead box costs ≤ 200k samples. |
| **Resume** | `AR_RESUME=auto` (default) picks up `run_full/ckpt_last.pt`: same step count, same LR-schedule position, same data order (per-epoch seeded permutation + batch offset), same W&B run. `AR_RESUME=none` forces fresh. |
| **Long-run optimizations** | Step-based cosine (warmup 2%, floor 5%) instead of the 8-min time-based schedule; torch.compile before the clock; PatchDropout 0.6 kept (regularizer + 231 img/s throughput); raw-bytes data download (`AR_RAW_TRAIN=1`, network-bound ~15–30 min instead of CPU-bound ~2.3 h); per-GPU LRs sqrt-scaled with world size (×2 at 4 GPUs, standard for larger global batch). |
| **Fastest finish** | **4×RTX 5090, one run sharded via torchrun** (vast.py already launches multi-GPU this way). The backbone is frozen so only ~10M trainable params all-reduce per step (~40 MB, a few ms) — near-linear scaling. `train_full.py` runs unchanged on 1, 2 or 4 GPUs. |

## Time & cost (measured 231 img/s per 5090 at bs96)

| Config | Train time | Wall total* | Cost (~$0.60–0.80/GPU/hr) |
|---|---|---|---|
| 1× 5090 | ~4.3 h | **~5.3 h** | ~$3.5 |
| 2× 5090 | ~2.2 h | **~3.2 h** | ~$4 |
| 4× 5090 (recommended) | ~1.2 h | **~2.1 h** | ~$5–7 |

*includes setup ~5 min, data download ~15–30 min (prefer hosts with ≥1 Gbit inet_down —
`vast.py search` prints it), k-means + compile ~5 min, evals ~12 min, final eval + pulls.

## Commands

### 1. Bring up the box (4×5090, 200 GB disk for the 115 GB cache)

```bash
export AR_N_TRAIN=1198072 AR_RAW_TRAIN=1
python vast.py start --metric median_km --goal min --gpu RTX_5090 --gpus 4 \
    --max-price 0.80 --disk 200 --hours 5
```

(1-GPU variant: `--gpus 1 --hours 9`. Setup streams the full dataset via the raw-bytes path;
watch worker-0 progress lines. Idempotent — re-run `python vast.py setup` if SSH drops;
finished workers skip via `.done` markers.)

### 2. Smoke test (~5 min, 40 steps, no W&B, separate run dir)

```bash
source .env && python vast.py run "cd /root/auto && env HF_TOKEN=$HF_TOKEN \
  WANDB_MODE=disabled AR_MAX_STEPS=40 AR_RESUME=none AR_RUN_DIR=/root/auto/run_smoke \
  AR_TIME_BUDGET=1800 PYTHONPATH=/root/auto TORCHINDUCTOR_CACHE_DIR=/root/auto/.inductor \
  CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_full.py 2>&1 | tail -30"
```

Expect: quantization-floor line, step-0 eval, ~40 steps, a `median_km:` panel. On 1 GPU drop
the torchrun part (`/venv/main/bin/python train_full.py`) and use `CUDA_VISIBLE_DEVICES=0`.

### 3. Launch the real run (nohup — NEVER via `vast.py exp`, SSH drops would look like crashes)

```bash
source .env && python vast.py run "cd /root/auto && mkdir -p run_full && env \
  HF_TOKEN=$HF_TOKEN WANDB_API_KEY=$WANDB_API_KEY \
  AR_CKPT_HF_REPO=josefbednar/neuroguessr-fullrun-ckpt \
  AR_TIME_BUDGET=21600 PYTHONPATH=/root/auto TORCHINDUCTOR_CACHE_DIR=/root/auto/.inductor \
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /venv/main/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_full.py > run_full/full.log 2>&1 & echo LAUNCHED"
```

(`AR_TIME_BUDGET` is only the SIGALRM backstop — set ~4× the planned train time:
21600 for 4 GPUs, 43200 for 1 GPU. The run stops itself at TOTAL_STEPS.)

### 4. Monitor

```bash
python vast.py run "tail -n 4 run_full/full.log"     # or watch the W&B run fullrun-5000c-3ep-w4
python vast.py status                                 # cost + deadline
```

### 5. If it dies

- **Same box:** re-run the launch command from step 3 — `AR_RESUME=auto` continues from
  `ckpt_last.pt` (same W&B run, same step count).
- **New box:** `vast.py start` again (data re-downloads in ~20 min), then restore the
  checkpoint from HF before launching:
  `python vast.py run "cd /root/auto && mkdir -p run_full && env HF_TOKEN=... /venv/main/bin/python -c \"from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('josefbednar/neuroguessr-fullrun-ckpt','ckpt_last.pt'),'run_full/ckpt_last.pt')\""`
  — buffers in the checkpoint carry the geocells, so a new box's k-means jitter doesn't matter.

### 6. Wind down (pull artifacts BEFORE the watchdog deadline)

```bash
mkdir -p run_full
python vast.py pull run_full/ckpt_best.pt run_full/   # ~130 MB — the app-export artifact
python vast.py pull run_full/full.log run_full/
python vast.py down
```

## What to expect

Reference points: the champion gets ~209–213 km median from 60k images / 8 min; the original
neuroguessr repo hit **194 km** with 1.2M images + ~2 h on a weaker stack (full-FT CLIP
ViT-L@336, plain cells, no hierarchy); PIGEON's 44 km needed 4-view panoramas + retrieval.
This run = champion architecture × 20× data × ~30× compute: beating 194 decisively is the
expectation; the 100–150 km band is the realistic target. Watch `val/median_km` and
`val/country_acc` (~46% at 60k — the single biggest driver of the median; PIGEON is ~92%).
