# Mission & config — EDIT THIS FILE

This is the **editable** part of autoresearch: **what** to research, **how** to score it,
and the **knobs**. The fixed machinery (how the agent runs the loop) is in
[`ENGINE.md`](ENGINE.md) — **do not edit that one.** When the human starts a session, read
this file, then run the loop in `ENGINE.md`.

> This repo is configured for **image geolocalization**: given a street-view image, predict
> where on Earth it was taken. The *experiment* is `train.py` (a LoRA-finetuned DINOv3 ViT-L
> backbone + geocell head), the *harness* is `prepare.py` (a fixed, seed-pinned data subset
> + the frozen haversine scoring), and the *objective* is **`median_km`** (median great-circle
> error on the held-out val split, **lower is better**). Data: `josefbednar/streetview-acw-300k`.
> Each experiment is recorded as a run in a **Weights & Biases** project.

---

## 1. What we're optimizing

**Goal:** find the `train.py` that reaches the **lowest median great-circle error (`median_km`)**
on the held-out val split within the fixed per-experiment training budget. We **finetune** a
pretrained vision backbone rather than train from scratch — the default is a frozen
**DINOv3 ViT-L/16 (LVD-1689M)** adapted with **LoRA** at **448px**, with a **geocell-classification head**
(PIGEON-style haversine-smoothed labels) that predicts location via a probability-weighted
spherical mean over cell centroids. Everything in `train.py` is fair game: backbone choice,
LoRA config, head architecture, geocell scheme, loss, optimizer, augmentation, resolution,
pooling.

**Bias toward big wins.** Prefer **bold, novel** changes with large upside (a better head,
a hierarchical geocell scheme, a smarter loss, retrieval, mixture-of-experts over regions)
over tiny hyperparameter tweaks. (See "SWING FOR BIG WINS" in `ENGINE.md`.)

The loop is **single-agent, one experiment at a time**: pick one idea, run it once on the
rented box, and keep it only if it beats the champion.

## 2. The metric

- **`median_km`** — median haversine distance (km) between predicted and true coordinates on
  the **full val split**. **Lower is better.** Robust to the heavy tail of catastrophic
  (wrong-continent) misses, and the standard headline metric in the geolocation literature.
  This is the score.
- Every run also prints a full diagnostic panel (do **not** optimize these directly, just
  watch them): `mean_km`, `acc_1km` / `acc_25km` / `acc_200km` / `acc_750km` / `acc_2500km`
  (street/city/region/country/continent accuracy), and `geoguessr_score`
  (`5000·exp(-d/1492.7)`, higher better). `peak_vram_mb` too — watch it.
- **A step-0 eval runs before any training** (untrained baseline point), then **every
  `EVAL_EVERY` (=100) steps** `train.py` prints a monitoring line with all of these,
  evaluated on a **fixed 1000-image val subset** (fast); the final score uses the full val.
- All of these (plus train loss) are logged to **W&B** — one run per experiment, grouped in
  the `WANDB_PROJECT` project (default `streetview-geoloc`).
- The metric is computed by `evaluate_geo` in `prepare.py` and is **ground truth** — never
  change it. The model only ever emits coordinates via `predict_latlon`; the scoring lives in
  the frozen harness, so it can't be gamed.
  - To optimize a *different* headline number instead, re-launch with e.g.
    `--metric geoguessr_score --goal max` or `--metric acc_25km --goal max` — all are printed
    every run, so switching costs nothing.

## 3. Config knobs

Passed to `vast.py start`. This session's defaults:

There are **two independent time knobs**:
- **`--minutes`** — how long **each single experiment** trains (one `train.py` run).
- **`--hours`** — how long the **whole session** runs before the box auto-destroys.

| Knob | Value | What it controls |
|------|---------|------------------|
| **Objective** | `--metric median_km` | the metric name the experiment prints. |
| **Direction** | `--goal min` | lower `median_km` is better. |
| **Per-experiment budget** | `--minutes 8` | how long **one** experiment trains. |
| **Session length** | `--hours 3` | whole-session hard auto-destroy deadline. |
| **GPU type** | `--gpu RTX_5090` | 32GB Blackwell — fits ViT-L @448 with a big batch. |
| **GPUs** | `--gpus 1` | one GPU, one experiment (train.py is single-GPU). |
| **Price cap** | `--max-price 0.90` | ceiling in $/GPU/hr (5090s run pricier than 4090s). |

One-shot bring-up:
`python vast.py start --metric median_km --goal min --hours 3 --minutes 8 --gpu RTX_5090 --max-price 0.90`

**Data (frozen subset).** `prepare.py` caches a **fixed, seed-pinned** subset once so every
experiment sees the same images (env-tunable at setup, then frozen): `AR_N_TRAIN` (default
60k training images ≈ 15k locations × 4 views), `AR_N_VAL` (3000 = full val), seed 1337.
`AR_QUICK_VAL_N` (1000) is the val subset used for the in-training monitoring evals.

**Backbone note.** Default is `facebook/dinov3-vitl16-pretrain-lvd1689m` (0.3B, natural-image
pretraining — the right domain for street-view). DINOv3 weights are gated, so the box needs an
accepted HF token. Alternatives to try: `...vith16plus-pretrain-lvd1689m` (0.8B, more capacity
but slower → fewer steps in the 8-min budget), or `facebook/dinov2-large` (ungated fallback).
Avoid the `sat493m` checkpoint — it's pretrained on satellite imagery, wrong domain.

**Secrets (`.env`).** Put `HF_TOKEN`, `WANDB_API_KEY`, and optionally `WANDB_PROJECT` in a
gitignored `.env` in the repo root. `vast.py` loads it and forwards these to the box (setup +
every experiment); they are never committed or uploaded in the repo tarball.

## 4. The search directions

A menu of idea families the agent **rotates through** so the search stays broad. Each
experiment picks **one** concrete idea. Grouped by which part of `train.py` it touches:

- **Backbone & features.** DINOv2 size (base/large/giant) or DINOv3; which layers/tokens to
  pool (`POOL`: cls / mean patch tokens / both / multi-layer concat); input **resolution**
  (`IMG_SIZE`, must stay a multiple of the patch size); registers variant.
- **LoRA / adaptation.** `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGETS` (attention only
  vs +MLP `fc1`/`fc2`), which blocks get adapters (last-N only), or partial full-finetune of
  the top blocks; `GRAD_CHECKPOINT` for VRAM.
- **Geocells & output head.** `N_CELLS`, k-means vs adaptive/balanced/hierarchical cells (S2),
  `HEAD_HIDDEN`/depth/dropout, classification vs classification+intra-cell regression offset,
  `PRED_TOPK` and the spherical-mean prediction, direct 3D-vector regression head.
- **Loss.** PIGEON-style haversine label smoothing (`SMOOTH_TAU_KM`), hard CE, focal loss for
  the tail, hierarchical/coarse-to-fine losses, auxiliary country/continent classification,
  contrastive/retrieval objectives.
- **Optimization & data.** `LORA_LR` / `HEAD_LR` split, `WEIGHT_DECAY`, betas, warmup/cosine
  shape, `DEVICE_BATCH_SIZE` / `GRAD_ACCUM`, **augmentation** (careful: horizontal flip
  destroys driving-side cues), using side info (heading, panorama's 4 views jointly), label
  balancing across regions.

## 5. What's editable — and when (no cheating)

The boundary is **temporal**:

- **Before research (setup):** the human/agent may edit anything to configure the study —
  `program.md`, `train.py`, and even **`prepare.py`** (the data subset, splits, the evaluator).
- **During research (the ratchet loop):** **only `train.py` may change.** Do NOT touch
  `prepare.py`, `evaluate_geo`, the `predict_latlon`→coordinates contract, or
  [`ENGINE.md`](ENGINE.md). Every gain must come from `train.py` alone. Enforced structurally:
  `vast.py exp` uploads **only** `train.py`, so the box keeps the frozen `prepare.py`/metric.
- **Keep the time-budget + eval scaffolding in `train.py`** when editing it: the
  `start_training_clock()` call, the `try … except TrainingTimeUp` around the loop, and the
  final `evaluate_geo` + `median_km:` print. They hard-bound the run and guarantee a graceful
  final score.

---

**To run:** the agent follows [`ENGINE.md`](ENGINE.md) — bring up the box with `vast.py start`,
run the baseline once, then loop: pick one idea, edit `train.py`, run it with `python vast.py
exp --train train.py`, and keep it only if it beats the champion. The whole control plane is
`python vast.py --help`.
