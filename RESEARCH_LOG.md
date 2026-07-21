# Research log — neuroguessr-2 (image geolocalization)

**Durable, cross-session memory.** This is the **distilled** knowledge base — the champion,
what's banked, what's a dead end, and what to try next — kept readable at a glance. The raw
`results.tsv` (every run's score) and `findings.md` (the live notebook) are **also committed
and pushed** now, as the full history behind this summary. The `/goal` command reads this file
FIRST at session start and updates + pushes it (with the raw ledger + notebook) during and at
the end of a session, so every future session — even a fresh clone — starts with the
accumulated knowledge and doesn't repeat dead ends.

Objective: **`median_km`** (median great-circle error on the val split), **lower is better**.

---

## Current champion

- **median_km:** 263.9
- **train.py commit:** 7acb12a (branch autoresearch/2026-07-21b)
- **one-line:** no-grad-ckpt (bs24x2, 32w) + 2048 geocells/topk16 + mode-seeking prediction
  (T=0.5, 1000km locality) + hierarchical heads 64/512/2048 (log-space combine, w 0.25/0.5/1.0)
- **full metric panel (last full-val eval):** mean_km 1219 · acc@200km 41.5% · acc@2500km 89.9% · geoguessr 3596 · val cell_top1 13.3% (lift 273x)

## Banked wins (confirmed to help — keep these, don't re-litigate)

_(each entry: the change, the median_km delta, and why it likely helped)_

- **No grad-checkpoint + bs24×2 accum + 32 workers** (535.4→433.3, −19%): checkpointing was a
  ~30% compute tax; at 8-min budgets, throughput ≈ data seen ≈ accuracy. bs48 no-ckpt OOMs 32GB.
- **2048 k-means geocells + PRED_TOPK 16** (433.3→398.5, −8%): median floored by cell size.
- **Mode-seeking prediction rule** (398.5→283.4, −29%!): softmax at T=0.5 + only average top-k
  cells within 1000 km of the top-1. Median rewards mode-seeking; the old global spherical mean
  averaged cross-continent mass into oceans. mean_km worsens slightly — expected, fine.
- **Hierarchical multi-resolution heads** (282.8→263.9, −6.7%, session 2): shared trunk + linear
  heads at 64/512/2048 cells; coarse logits added onto child fine cells (log-space) at prediction;
  coarse CE aux w=0.25/0.5. Fixes wrong-region errors (mean 1399→1219, acc@2500 87.7→89.9%).

## Dead ends & mistakes (tried, did NOT help or broke — do NOT repeat)

_(each entry: what was tried, what happened, and the takeaway so it isn't retried blind)_

- **No-ckpt at bs48**: OOM on 32GB (activations). Pair no-ckpt with device bs ≤24 @448px.
- **cls_mean pooling**: 398.5→395.5, within run noise (±3–5 km). Neutral at best solo.
- **Per-cell offset regression head** (OSV-5M hybrid): 395.7, within noise. Takeaway: at this
  accuracy the error is dominated by WRONG-CELL selection, not within-cell resolution — skip
  further within-cell refinement (incl. retrieval refinement) until cell_top1 is much higher.
- Run-to-run noise on the full-val median is ~±3–5 km: don't crown wins smaller than ~8 km.
- **Unfreezing last 2 blocks** (alone 278.3, +T0.35 combo 280.6 vs champion 283.4): ~1–2% real
  at best, never cleared the noise bar in two runs. Not worth re-trying as-is; maybe with
  longer budgets.
- **tau 75→40 at 2048 cells**: 316.4, clearly WORSE (−12%). The 75km haversine smoothing is
  load-bearing; if anything try tau UP, not down.
- **Panorama InfoNCE aux (λ=0.5, pair batches)**: 306.1 vs 282.8 baseline, +23 WORSE; val
  cell_top1 fell 14.0→11.9%. The same-pano pair sampler halves distinct locations per batch —
  at 8-min budgets diversity beats the contrastive signal. Retry only with much longer budgets
  or a sampler that keeps ≥75% unique locations.

## Parked for LONG-budget runs (better per-step, worse per-second — revisit when --minutes grows)

- **bs48×1 @384** (exp 14): quick-val median hit ~221 km by step ~800 vs ~235 km for champion
  bs24×2 at similar steps — clearly better per-step learning (bigger real batch), but the fused
  pass ran at 92 vs 106 img/s so it lost under the 8-min clock (246.9 vs 238.0 full-val).
  First thing to re-try in any long/final training run.
- **Practice (Josef, 2026-07-21): when discarding an idea that lost on throughput, ALWAYS check
  W&B val/median_km at matched step counts; if it's better per-step, park it here instead of
  calling it a dead end. The session constraint is TIME, but long runs are constrained differently.**

## Open ideas / next to try (ranked)

_(carry unfinished/promising directions forward across sessions)_

Refreshed at end of session 2026-07-21 (champion 283.4). Cell SELECTION is the bottleneck —
prioritize ideas that improve which cell wins, not within-cell refinement:

1. **Panorama-aware InfoNCE auxiliary** (OSV-5M's best aux; needs a same-panorama batch
   sampler — 4 views share a panoid/location in the train metadata). Untried, top pick.
2. **Hierarchical multi-resolution heads** (64/512/2048, combine in log-space) — targets
   wrong-continent/region errors that locality can't fix. Untried.
3. **Prediction-rule micro-sweep is DONE** (grid diag): T0.35/r1000 ≈ −5km on quick-val but
   didn't confirm on full val; r=400 neutral; no-locality catastrophic (+54). Don't re-sweep.
4. **tau UP (75→110/150)** at 2048 cells — 40 was much worse, so the gradient points up.
   One cheap shot.
5. **Bigger effective capacity via throughput**: IMG_SIZE 448→384 (−27% tokens ≈ +35% steps,
   epoch coverage 0.53→~0.7) — accuracy/steps tradeoff unknown, worth one run.
6. **Geolocation-safe augmentation** (RandomResizedCrop 0.5–1.0, color jitter; NO flips) —
   only 0.5 epochs seen so overfitting is mild, but cheap to test.
7. TTA without flips (3 crops, average probs after temp-sharpening).
8. EMA of trainables.
9. GeM pooling over patch tokens (cls_mean was neutral; GeM is the stronger variant).
10. Retrieval refinement & offset regression: PARKED until cell_top1 improves substantially
    (both tested within-noise at current accuracy).

Original scout list 2026-07-21 (PIGEON CVPR'24, OSV-5M CVPR'24, GeoCLIP NeurIPS'23), for reference:

1. **Finer geocells 512→2048** — median_km is floored by cell size; PIGEON uses ~2000 cells. Grow PRED_TOPK 8→16, drop cells with <5 points. (big)
2. **Retrieval refinement at prediction** — cache train-image embeddings + coords; at eval, refine top-K cell guess by cosine-similarity match against train embeddings within those cells (PIGEON's biggest ablation win). (big)
3. **Hybrid classification + within-cell offset regression** — regress tangent-plane offset from cell centroid, guess = centroid + offset; OSV-5M found class-then-regress best head. (big/medium)
4. **CLS + mean-patch-token concat pooling** (optionally + mid-layer) — distributed cues (vegetation, signage) live in patch stats. (medium, cheap)
5. **Sharper prediction rule** — temperature-sharpen before spherical mean; restrict mean to cells near top-1 (multimodal posteriors average into the ocean); median rewards mode-seeking. (medium, free)
6. **Panorama-aware InfoNCE auxiliary** — positives = same-panorama views in batch (train-time only, legal); OSV-5M's best auxiliary. Needs group sampler. (medium)
7. **Hierarchical multi-resolution heads** (64/512/2048 cells, combined in log-space). (medium)
8. **Unfreeze last 1–2 blocks** (LR ~2e-5) — OSV-5M: beats LoRA for geolocation. (medium)
9. GeoCLIP-style RFF coordinate-regression auxiliary. (medium/small)
10. Tau sweep 75→30/150 km — only after finer cells (at 512 cells smoothing is near-inert). (small/medium)
11. TTA without flips (3 crops, average probs). (small/medium)
12. EMA of trainables, eval EMA copy. (small)
13. GeM pooling over patch tokens. (small/medium)
14. Geolocation-safe augmentation: RandomResizedCrop(0.5–1.0)+color jitter (no flips!). (small)

## Environment / gotchas learned

_(anything about the box, dataset, VRAM ceilings, throughput, DINOv3 quirks, etc.)_

- **Screen boxes for real HF throughput before setup** — advertised `inet_down` is meaningless
  for the HF route (a 4,471 Mbps box did 143 KB/s to HF). Rent → `curl` a dataset shard → keep
  if ≥4 MB/s else destroy (scratchpad script `screen_and_rent.py` pattern). California DC
  209.146.116.50 also drops idle SSH.
- **Run long remote jobs under `nohup` on the box** — plain `ssh cmd` dies with the connection
  and kills the child. `vast.py exp` streams output constantly so it survives.
- **Parallel data pull** (banked in prepare.py 8a28c43): 16 sharded streams; the dataset's
  parquet row groups are ~1000 rows (~390 MB), so each stream is silent for minutes before its
  first image — not a hang. Full 60k pull ≈ 20 min at 33 MB/s.
- **Cache tar DONE (session 2):** `josefbednar/streetview-acw-ar-cache/cache_n60000_v3000_s1337.tar`
  (4.65 GB, private) uploaded; prepare.py pulls it automatically (66c390d) before falling back to
  streaming. Setup is now a one-tar download. Also: per-worker part-parquets (8bdf4bc) make
  streaming reruns resume instead of restart, and workers have socket timeouts + retries (f027237).
- **HF streaming stalls root-caused (probably):** unauthenticated/parallel hammering triggers silent
  rate-limit backoff; one worker (w08) hung twice with zero traffic on live sockets. If a worker
  stalls: kill + rerun prepare (parts resume). ALWAYS pass HF_TOKEN to remote nohup commands —
  the env does not follow you.
- transformers 5.14 DINOv3: LoRA targets are `q_proj/k_proj/v_proj/o_proj`.
- RTX 5090 32GB: baseline uses only 6.9 GB with checkpointing on; throughput 46 img/s → 0.36
  epochs per 8-min budget. Compute/throughput, not VRAM, is the binding constraint.

---

## Session history

_(one dated block per session: dates, champion at start → end, headline results)_

### 2026-07-21 (session 1)
- Champion at start: none → at end: **283.4 km** (ae5e29f)
- Experiments run: 9 (1 baseline, 3 KEEP, 4 discard, 1 OOM crash)
- Headline: 535.4 → 283.4 (−47%) via throughput unbrake (−19%), 2048 geocells (−8%),
  mode-seeking prediction rule (−29%). Big lesson: median_km rewards mode-seeking inference;
  locality restriction around the top-1 cell is essential.
- Infra: ~50 min lost to two bad Vast boxes (broken HF peering). Fixes now durable: parallel
  sharded downloader in prepare.py (committed), rent-screen-by-curl pattern, and the prepared
  cache upload to HF `josefbednar/streetview-acw-ar-cache` was attempted but DID NOT COMPLETE
  (box's HF upload throttled to <1MB/s; ~85% of new data at teardown). Next session: either
  re-download via the parallel downloader (~20 min) or redo the tar upload early from a box
  with good HF peering, THEN switch prepare to the tar path.
- Follow-ups for next session: panorama InfoNCE aux (top pick), hierarchical heads, tau up,
  IMG_SIZE 384 throughput trade. W&B has per-run cell_top1/top5 + lift metrics now.

<!-- template:
### 2026-07-21
- Champion at start: <median_km or "baseline"> → at end: <median_km>
- Experiments run: <N>, kept: <k>
- Headline: <what moved the metric>
- Notes / follow-ups for next session: <...>
-->
