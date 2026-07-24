# C5 — the bed run

*Planned 2026-07-24 with Josef. Successor to C4_PLAN.md. Everything here is grounded in a
measurement (this repo) or a cited paper (docs/RETRIEVAL_ROADMAP.md). C4's post-mortem is in
RESEARCH_LOG.md 2026-07-24.*

## The philosophy (why C5 exists)

**The GPU run's product is not a model — it is a *bed*:** the complete set of cached
artifacts that lets every subsequent experiment run on a CPU in minutes. The C4 era proved
both halves of this: cheap cached-vector experiments produced the best numbers of the
project (E7: −7.7% median in one evening, $0), and the mean campaign died *only* because
the bed lacked evidence channels no decision rule could conjure. C5's success criterion is
therefore twofold: strong L1 metrics AND a bed that never blocks an experiment again.

One session, ~2.5–3h, ~$5–6, then weeks of light-speed iteration.

## What C5 trains — one script, one joint phase, from scratch

**Init: pretrained DINOv3 ViT-L + fresh LoRA (attn q,k,v,o + MLP up/down, r16) + fresh
heads. No checkpoint lineage** — C3 carries classifier damage, C4 carries descriptor
erosion, and warm-starting either makes every C5 number un-attributable. The classifier
recipe being re-run is proven (train_full.py, 104.37 median twice). The whole run is
`DINOv3 + dataset + one script → bed`, reproducible by anyone.

Heads on the shared backbone (all jointly trained, all shipped or bed-emitted — C5 trains
nothing it doesn't use; C4's orphan PlaceHead and the never-blended geo50 head are cut):

1. **Classifier stack** — 5000-cell fine head + coarse hier + country head with
   tau-smoothed targets (the full-run champion stack incl. MSL; country-hier is the one
   supervision with a measured mean effect, −81).
2. **Attribute heads** (new, tiny, free labels from coordinates): driving side, climate
   class, hemisphere, lat band (+coastal/elevation if Phase 0b lands the rasters).
   Purpose: force meta-cue encoding into the representation AND emit per-image attribute
   posteriors as bed evidence channels for the mean campaign.

Losses, sharing one forward per image:

- **CE** on smoothed cell targets + hier + country (proven stack).
- **Graded listwise episode loss**: batches structured as anchor + candidates sampled at
  graded distance bands (0–5/5–25/25–100/100–500 km + global), target ∝ exp(−d/τ_geo),
  listwise CE over the episode. Pools mined from **geography alone** — no model in the
  mining loop, fully deterministic.
- **Climate-matched cross-continent negatives** (the training-stage mean attack): global
  negatives sampled preferentially from *same climate class, different continent* — the
  Estonian-forest/Canadian-forest pair that owns the confidently-wrong tail, presented
  every batch instead of almost never.
- **Attribute CE** (small weight).
- Episode-loss weight ramps over the first ~500 steps (fresh classifier settles first).

Batch mix: alternate episode batches with pure-global i.i.d. batches (CE health + long-range
negatives); ratio is pilot #2.

## Instrumentation — the C3/C4 blindness is banned

- **Probe with verified resolution** (checked in the smoke test before the long run):
  held-out pool with ceiling ≥30% and expected hits in the hundreds; near-recall@1 every
  250 steps (~30s — never a full re-index during training).
- Classifier quick-val (1000 imgs) + attribute-head accuracy on the same cadence.
- Pilots (~20 min total, 200 steps each, ranked by probe slope + CE health):
  #1 PatchDropout 0.5 on/off (banked ratchet win, +110% steps — if the probe tolerates it,
  training is ~2× faster); #2 batch mix; #3 λ_episode / τ_geo coarse grid.
- W&B: one run, all curves.

## The bed contract (checked by manifest before the box may be destroyed)

| artifact | size | unlocks |
|---|---|---|
| cls + mean + regional descriptors, train AND val | ~10GB | everything; tail-routing (M4) |
| val logits (full) + train top-50 cells+probs | ~0.6GB | gate, priors, **train-side episode mining** |
| **train-side episode cache** (per train image: gated top-50 ids+sims) | ~0.4GB | big-E7 at 1M episodes, calibration, hub discount, second-hop, graph denoising, C6 confusion mining |
| attribute posteriors (both splits) + coordinate label table | ~50MB | attribute-consistency features, masks — the mean campaign |
| val patch tokens | ~3.5GB | patch-verification + SALAD prototyping |
| band heads 5/10/25 + CSLS stats + whitening + calibration curves | ~60MB | the lever stack |
| centroids.npz, train_latlon, val_meta, quick-val ids, pilot logs | ~25MB | the boring completeness that burned C4's eval |
| L1/L2 reports + baseline rows (C4, S384) in results.json | — | attribution forever |
| val images cached locally (one 150MB download, Phase 0) | — | OCR / any query-side signal, no box |

Mirror **after every artifact**, not at the end. A cheap-experiment idea that can't run on
this list is a *bed bug*: fix = add a manifest line.

Honest exceptions (physics, not planning): a SALAD-aggregated index and at-scale patch
verification need train-side patch tokens (~1.4TB, uncacheable) — prototype locally on val
patches, productionize in one cheap box-hour with the manifest runner.

## Metrics contract ("legendary evals")

Three layers, each judged only by its own report; one CPU-runnable entrypoint writes all of
it to results.json:

- **L1 encoder**: within-pool matcher accuracy (THE number: answer-in-pool → picked, now
  45.1%), descriptor recall@k curves, probe, classifier solo panel (median/mean/@25/@200/
  @2500/cell_top1/GG), attribute accuracies, **gate recall at 0.95 mass** (the classifier's
  real job — its solo median is diagnostic only).
- **L2 bed**: gate recall vs mass curve, pool-contains-answer rates, tail decomposition
  (mean-km per error band — the autopsy, standardized).
- **L3 decision layer**: end-to-end panel + **auto-printed attribution table**
  (encoder → +bands → +CSLS → +E7 → +rules, delta by delta) + the λ dial curve
  (median↔@1km frontier).

## Session plan & budget

| stage | time | notes |
|---|---|---|
| rent + GPU screen + net screen (tools/screen_gpus.py, real-artifact curl) | ~15 min | lemon/slow-egress boxes destroyed pre-spend |
| restore full cache (tools/restore_full_cache.py, marker-checked) | ~12 min | |
| pilots ×3 | ~20 min | |
| joint training, 2 epochs (2.4M views) | 60–90 min | PatchDropout-dependent |
| bed derivation (embeds, logits, episode cache, heads, stats, reports — 4 GPUs parallel) | ~40 min | mirror per artifact |
| manifest check → destroy | ~5 min | |
| **total** | **~2.5–3h** | **~$5–6 @ $1.60–2/hr screened 4×5090** |

Launch gate: C5 does not rent until the Phase-0 dry-run passes (preflight validates every
stage's inputs against the local repo + HF listing — the check that would have caught
C4-eval's centroids.npz in 5 seconds).

## Expectations (honest)

- Classifier: within a few km of the 104 standalone bar; cell_top1 25–27%; **gate recall
  unchanged** (the consumed metric). Fallback if pilots show CE lagging: +half epoch (~25 min).
- Matcher/median/@25: encoder trained *directly* on graded episode ranking + the proven
  lever stack + big-E7 afterwards → the credible path at 45.1% → 55–65% within-pool,
  median toward ~20–30, @25 toward 50–60%.
- Mean: climate-matched negatives + attribute channels + big-E7 + calibration →
  328 → ~230–270 well-supported; sub-200 is the stretch goal, measured by the standard
  tail decomposition, not vibes.

## Phase 0 checklist (no box, $0)

- [x] GPU + network screen (tools/screen_gpus.py; net probe on the real artifact)
- [x] full-cache restore with success markers (tools/restore_full_cache.py)
- [ ] attribute label table, train+val (tools/build_attribute_table.py) — country/driving
      side/hemisphere/lat-band (+Köppen; coastal/elevation Phase 0b)
- [ ] manifest runner + preflight (promote scratchpad driver v5 → tools/, kill `|| true`)
- [ ] CPU-first eval entrypoint (port of local_c4_eval.py + metrics contract)
- [ ] E7 module in retrieval/ (episode-cache consumer, graded target — the @1km lesson)
- [ ] probe builder with resolution validation
- [ ] vast.py: sync retry, low-balance watchdog guard (mirror-then-destroy), `screen` subcommand
- [ ] val images cached locally (150MB)
- [ ] preflight dry-run passes end-to-end

## Dropped / dead (this cycle)

Chamfer rerank (struggle/benefit ratio — regional arrays stay in the bed for M4, the
default recipe drops the stage) · C3-val backfill eval · warm starts · geo50 band head ·
in-training PlaceHead · all items on RETRIEVAL_ROADMAP's standing dead list.
