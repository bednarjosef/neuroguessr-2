# neuroguessr-2 — agent notes

Single-image geolocalization: a DINOv3 ViT-L/16 + LoRA geocell classifier gates a
nearest-neighbour search over 1.2 M geotagged Street View images, and a small learned reranker
picks the answer. Champion: **34.10 km median** on the frozen 2,998-image val split
(`paper/neuroguessr.pdf`, `README.md`).

Where things are:
- `prepare.py` — the **frozen harness** (data subset, splits, haversine, metric panel). Never edit it.
- `train.py` / `train_full.py` / `train_c5.py` — classifier trainers (ratchet-phase / full-corpus / joint).
- `retrieval/` — index building, place heads, contrastive encoders, lever sweeps, reranker scoring.
- `vast.py` — Vast.ai control plane (`python vast.py --help`).
- `research/` — how the work was done: `RESEARCH_LOG.md` (durable cross-session memory, read it
  first), `findings.md` + `results.tsv` (the ratchet ledger), `ENGINE.md` + `program.md` (the
  keep/reset loop and its config, driven by `/goal`), `plans/`, run `logs/`.

Research runs follow the keep/reset ratchet in `research/ENGINE.md`: change one experiment file,
run it once against the frozen evaluator, keep it only if the objective improves. Cheap
experiments on cached vectors (`retrieval/eval_levers.py`) come before any GPU run.

---

## Session discipline (learned the hard way — 2026-07-24)

**Mirror before teardown, always.** A rented box is ephemeral storage. Push every artefact worth
keeping to HF (`josefbednar/neuroguessr-fullrun2-ckpt`, folder per phase) **as it is produced**,
and re-run the mirror as the **last step before `vast.py down`**. On 2026-07-24 the mirror ran
mid-session while three band heads were still training; the box was destroyed after they finished
and the champion config became non-reproducible without retraining them. Cheap files (`.pt`
heads, configs, logs) cost seconds to upload — push them immediately. Big index arrays
(`c2_*_train_*.npy`, regional descriptors) are the ones that hurt to lose: they are the input to
every cheap experiment.

**The best known retrieval recipe (2026-07-24, full val: median 36.96 km / @25 km 46.06% /
@1 km 12.78% / GeoGuessr 4452 per round):**

1. encoder = contrastively fine-tuned backbone (C3/C4), descriptors = backbone CLS
2. **band heads trained OFFLINE on cached descriptors** at d_pos 5 / 10 / 25 / 50 km — a head
   trained *inside* the fine-tune scored 10 points worse than the same recipe fit offline
3. **blend all four bands + the raw descriptor** (a single wider head does not work; the blend does)
4. **CSLS hubness correction** — biggest cheap win of the night, ~40 s of arithmetic
5. **PCA whitening** of the raw descriptor
6. gate = 95 % posterior mass (cap 400 cells), prior weight λ = 0.05, k = 1 snap
7. **regional chamfer rerank** of the top 50 (R50, α 0.25)

Steps 2–5 run on cached vectors for cents (`retrieval/eval_levers.py`) — **always exhaust them
before proposing another GPU training run.**

**TODO carried forward:** the 5/25/50 km band heads and the regional descriptors from the C3
index were lost with that box. They retrain in ~15 min from `c3/c2_cls_train_c3_r*.npy` on HF —
fold this into the next session rather than renting a box for it alone.

## Screen the box BEFORE any long run (learned 2026-07-24, cost ~$5 + a whole eval)

A rented box can pass the rent-time filters (dlperf, GPU count) and still be a dud at RUN time —
throttled clocks, or a torchrun job silently running on one GPU. **Before committing to any
multi-hour run, screen it:**

1. `nvidia-smi --query-gpu=index,utilization.gpu,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader`
   — every GPU's `clocks.sm` must be near `clocks.max.sm` (e.g. ~2800-3100 MHz on a 5090, NOT
   ~550 MHz). A GPU pinned far below max clock is throttled → destroy and re-race.
2. Run a ~30-step throughput smoke and confirm **img/s matches the historical baseline**
   (4×5090 train ≈ 200-260 img/s; embed_c2 re-index ≈ 900 img/s total) AND that a multi-GPU job
   shows **load on ALL GPUs**, not just GPU 0.

What went wrong: a Bulgaria 4×5090 trained C4 at 33 img/s (vs 260 elsewhere) and re-indexed at
23 img/s on GPU 0 only (1-3 idle, GPU 0 at 547/3090 MHz). ~$5 went into training at ~1/7 speed
and credit ran out before the eval — a completed training produced NO usable number. A 30-second
clock check would have caught it.
