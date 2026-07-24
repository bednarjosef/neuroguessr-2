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

- **median_km (60k/8min ratchet): 205.46** on the Italy box whose baseline anchor read 215.65
  (−10.2 vs anchor; boxes differ ~4-6 km). Stack = S5 champion + **MSL** (median-seeking loss)
  + **staggered tessellation** (2nd 2048-cell fine head on k-means seed 1, union readout).
- **Separate lineage — FULL-RUN model (train_full.py): official 104.66 on full val** (best
  subset eval 99.91), ckpt local at run_full/ckpt_best.pt + HF josefbednar/neuroguessr-fullrun-ckpt;
  served by webapp/server.py.
- **Full run #2 (2026-07-23, MSL+stagger+cc-cells+aug): official 104.37 — a WASH vs run #1**
  (mean 479 worse, @25km 7.8 slightly better, cell_top1 26.2 lower; best quick 102.54). Ckpt
  local run_full2/ckpt_best.pt + HF josefbednar/neuroguessr-fullrun2-ckpt. Lesson: ratchet
  keeps did NOT stack at scale; two different recipes both land ~104 ⇒ 3-epoch regime ceiling.
  Next levers: more epochs (ckpt-chain) and the retrieval stage (built in retrieval/).
- **RETRIEVAL STAGE (2026-07-23, inference-time on run #2 ckpt_best): median 78.5 km /
  @1km 10.2% / @25km 35.2% on FULL val** — the biggest single-day move in the project.
  Two-stage: honest-posterior mass gate over tessellation-A cells → cosine kNN over 1.2M
  train embeddings → neighbor spherical mean (or k=1 snap). Configs: median-optimal
  mass0.6/k6/T.005 → 78.52 | precision-optimal mass0.8/k1 → 79.67, @1 10.24%, @25 35.16%.
  **Backbone-CLS tap beats post-trunk tap in every top config** (geo-training collapses
  instance detail; tap BEFORE the trunk). Beats single-image PIGEON everywhere; @1km beats
  even their panorama number. Index: 2.4GB fp16 (bb tap) on HF fullrun2-ckpt repo
  (retrieval_index/); rebuildable in ~25 min/$1 via retrieval/embed_full.py.
- **train.py commit:** see autoresearch/2026-07-22-ideas branch (S6 keeps committed there;
  master carries the same file)
- **one-line:** session-4 stack + **country-level hierarchy** (w=0.5, majority-vote parents)
  + **tau-smoothed country targets** (300km) + **PATCH_KEEP 0.6** (post-saturation: richer
  tokens beat extra steps; VRAM 31.0GB peak on 5090 — NO headroom, use 0.5 on smaller GPUs).
- **Process rules now standing (Josef): NO confirm re-runs; KEEP at ≥1 km; report every run
  in chat immediately; ViT-L locked (mobile-app target); EVAL_EVERY 250 fixed.**
- prior: 221.5 @ ae5c260 = no-grad-ckpt + bs48×1 (32w) + 2048 geocells/topk16 + mode-seeking
  prediction (T=0.5, 1000km locality) + hier heads 64/512/2048 (log-space, w 0.25/0.5/1.0)
  + IMG 384 + EVAL_EVERY 250 + torch.compile(dynamic=True) with pre-clock warmup
- **full metric panel:** mean_km ~1094–1124 · acc@200km 46.7% · acc@2500km ~91% · geoguessr
  ~3722 · val cell_top1 16.4–17.7% · top5 45.8–46.3% · ~1168 steps/8min · vram 26GB (5090-only:
  would OOM a 24GB 4090 — drop to bs24×2 there)

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
- **IMG_SIZE 448→384** (263.9→238.0, −9.8%, session 2): −27% tokens ≈ +50% steps in-budget;
  resolution loss « step gain at 8-min budgets. cell_top1 13.3→15.2%. (Long runs may prefer 448.)
- **EVAL_EVERY 100→250** (238.0→230.9/229.9 CONFIRMED, session 2): quick evals were eating ~90s
  of the WALL-clock SIGALRM window (480+45s) — fewer evals = full 480s of real training, 1056
  steps. Harness-efficiency win, NOT a modeling insight (label it as such). TTA/eval-time ideas
  must go in the final eval only (alarm disarmed there).
- **bs48×1 @384 under compile** (224.8→221.5 CONFIRMED 221.47/221.56, session 3): the parked
  exp-14 idea (bigger real batch, better per-step) graduates — compile bought back the
  throughput the fused bs48 pass used to lose (1168 steps vs 1205 at bs24×2; per-step gain
  wins). VRAM 26GB of 32 — fine on 5090, would OOM a 4090.
- **torch.compile(backbone, dynamic=True) + pre-clock warmup** (232.4→224.8 CONFIRMED twice,
  session 3): compile+warmup (fwd+bwd + eval-mode fwd at 2-3 shapes) runs BEFORE
  start_training_clock, so the compile tax lands outside the wall alarm and every in-alarm step
  is compiled → +14% steps (1205 vs 1057), cell_top1 15.4→17.4%. Costs ~6-10 min extra wall per
  run (first run on a box slowest; inductor cache helps after). dynamic=True + multi-shape eval
  warmup prevents recompile stalls during quick evals.

- **PatchDropout 0.5** (221.5→217.4/215.4 CONFIRMED, session 4): drop a random 50% of patch
  tokens in TRAIN forwards only (index-select hidden-state patch slice AND RoPE cos/sin with
  the same mask; prefix cls+register tokens exempt); eval keeps all tokens. +110% steps,
  VRAM halved to 13.8GB. cell_top1 slightly down, median clearly up — throughput converts.
  tf 5.14 note: encoder module is `core.model` (DINOv3ViTEncoder, called as `enc(hs, (cos,sin))`),
  NOT `.layer`.
- **bs96×1** (215.4→209.6/210.1 CONFIRMED, session 4): PatchDropout's VRAM dividend spent on
  doubling the real batch. Steps 2750→1500 yet clearly better — step count SATURATES at this
  budget (~2750); past it, buy per-step quality (batch), not more steps. cell_top1 18.6% best.
- **PATCH_KEEP 0.5→0.6** (214.2→213.0 on Korea anchor, session 5, ≥1km rule): post-saturation,
  keeping 60% of patch tokens (1295 steps) beats 50% (1536 steps) — per-step token richness
  now outbids step count. VRAM 31.0GB peak at bs96/5090: knob FROZEN, no headroom (pair any
  memory-adding idea with a batch drop; GeM OOMed on top of this).
- **Tau-smoothed country targets** (→209.53/210.30 CONFIRMED, session 5): country CE targets
  = softmax(-d(true, country_centroid)/300km) instead of hard one-hot — near-miss countries
  penalized less (HierLoc-lite distance-weighting). The ONE follow-up on the country win; both
  knobs (w=0.5, tau=300) now FROZEN.
- **Country-level geographic hierarchy** (212.2→210.34/210.58 CONFIRMED, session 5): 115-way
  hard-CE country head (w=0.5) + log_softmax(country) added onto fine logits via majority-vote
  fine-cell→country parents (empty cells → global mode). Median −1.7 (small) but mean −81 and
  acc@2500 +1.1pp — it fixes wrong-region mass, exactly like the k-means hier win. Semantic
  supervision wired INTO the combine works where pure aux losses (S3) failed.
- **Geocell disk cache** (session 4, plumbing not a win): k-means cells cached per
  (n_cells,iters,npts) in CACHE_DIR — deterministic cells across runs (CUDA index_add_ is
  nondeterministic). Josef asked for this; keep it.

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
- **tau 75→110 on hier champion**: 271.0 (+7.1). With 40 much worse and 110 somewhat worse,
  tau=75 is the optimum — sweep DONE, frozen.
- **TTA (3 center crops, avg sharpened probs, final eval)**: 247.3 vs 238.0, +9.3 WORSE. Tight
  crops drop peripheral cues and averaging blurs the mode our mode-seeking rule needs. Skip TTA.
- **Geo-safe augmentation (RRC 0.5–1.0 + jitter 0.15)**: 239.3, neutral at 0.5 epochs — nothing
  to regularize yet. Parked for long runs (below).
- **Geo-contrastive hard negatives v1 (S3 exp19: proj128, margin .3, far>1500km, λ=.1)**: 238.7
  (+6.3 vs 232.4 box-baseline). cell_top1 ticked UP; median down; −5% steps. One retune allowed
  someday (semivariogram range-weighting instead of fixed margin), low priority.
- **Prototype cosine head, both variants (S3 exps 20+22)**: log-space ensemble 239.7; surgical
  rerank (fused selects cells, original posterior weights the mean) 236.3. Consistent pattern:
  cell_top1/top5 improve (16.2–16.4/44.9), median does NOT. At 8-min budgets a +1pp top1 gain
  moves too few val rows to shift the 50th percentile; the aux-classifier-head direction is
  EXHAUSTED at this budget (geo-contrastive, proto head, and pano-InfoNCE all show it).
- **EMA (S3 exp21, decay .999 warmup-corrected)**: 236.1 (+3.7 noise) — ~1000 steps is too few
  for EMA to average anything. Parked for long runs.
- **LoRA on gated MLP (up/gate/down_proj) at bs48**: OOM (26GB base + MLP adapter activations
  >32GB). Untested at bs24×2 — if retried, pair with smaller device batch. DINOv3 MLP module
  names are up_proj/gate_proj/down_proj (gated MLP, NOT fc1/fc2).
- **Geocells 2048→4096 + topk24 (S4)**: 223.8 (+8.4 vs 215.4). acc@25km UP (3.4%, first
  nonzero acc@1km) but 4096-way CE too hard at ~2750 steps (top1 10.6%). Cell-count sweep
  DONE at 8-min budgets; finer cells only make sense with much longer training.
- **FixRes train 288 / eval 384 (S4)**: 219.2 (+3.8) despite 4410 steps (+60%) — the step-
  scaling curve BENDS past ~2750 steps; further throughput no longer converts. Diagnostic
  moment for the whole "buy steps" era.
- **LR ×2 (S4)**: 227.2 (+11.8). 1e-4/1e-3 optimal even at 2x step count. LR sweep DONE, frozen.
- **Muon on head matrices, lr 0.02 (S4)**: 241.4 (+26) — far too hot for this soft-CE head.
  If EVER retried: lr ≤5e-3 and expect little; low priority.
- **Subdivision level (S5)**: 210.2 — median flat, panel regressed (1895-way CE eats gradient;
  country granularity is where semantic supervision stops paying at 8-min).
- **IMG 448 + PatchDropout (S5)**: OOM @ bs96, then 224.4 (+14) @ bs72 — 384 is now TRIPLE
  confirmed (S2 win, S4 FixRes, S5). Resolution FROZEN at 8-min budgets, even post-saturation.
- **Retrieval refinement v1 (S5)**: 222.6 (+12.7) BUT acc@25km 3.97% best ever — always-on
  top-8-NN blend (w .5, top-5 cells) fixes close cases and wrecks typical ones at top1=18%.
  ONE retune allowed: sim-GATED + OPTICS-style cluster centroids + multiply-with-cell-probs
  (PIGEON's actual mechanism; their ablation: median 44.4→50.0 without it). Needs top1 much
  higher to be a median lever; keep it eval-time only.
- **Semantic geocells / country-constrained k-means (S5)**: 216.6 (+2.4, ~neutral) — border
  alignment isn't the lever at this budget.
- **H3 merged cells (S5, from old neuroguessr repo artifacts)**: AS fine cells 240.7 (+26.5)
  with top1 25.7/top5 52.8 RECORDS — balanced cells classify easier but guess coarser
  (quantization floor). As an EXTRA hier level: 222.7 (+8.5) — any ~2k-way aux vocab steals
  gradient (matches subdivision). Cell-scheme chapter CLOSED at 8-min: kmeans-2048 + 64/512
  + country is the optimum. Mappings staged at CACHE_DIR/h3_to_class_*.json on future boxes
  via old repo h3_utils/ if ever needed.
- **Within-cell offset head (S5, Josef's idea)**: 219.9 (+5.7) — regressing position inside a
  ~250km cell is as hard as the classification itself at 1500 steps; PARK for long budgets
  (S1 exp4 was neutral for the same reason). Zero-init + tanh-clamp + per-cell RMS-radius
  scaling implementation is in git history (S5).
- **GeM pooling cls+concat (S5)**: 212.1 vs 213.0 — missed the ≥1km bar by 0.11km at bs84
  (memory-paired after OOM at bs96). Panel BETTER (mean 950 best-ViT-L). ONE retune allowed:
  GeM-replace instead of concat, or PATCH_KEEP 0.5 + bs96 to avoid the batch drop.
- **LoRA r32/a64 (S5)**: 214.8 (+1.8) — steps 1295→1158; capacity costs steps and doesn't
  pay. r16/a32 FROZEN.
- **ViT-H+ 0.84B (S5)**: 211.2 (−2.9 vs anchor, would have KEPT under ≥1km rule) with mean
  876 / acc@2500 93.4% / geoguessr 3836 ALL BEST-EVER — Josef dropped it deliberately (3x
  inference RAM, mobile-app target). Knowledge: capacity buys the TAIL (coarse geography),
  not the median. Right candidate if the mission ever stops caring about model size.

## Parked for LONG-budget runs (better per-step, worse per-second — revisit when --minutes grows)

- ~~bs48×1 @384~~ **GRADUATED in session 3** (banked win above) — compile fixed its throughput.
- **Geo-safe augmentation** (exp 15): neutral at 0.5 epochs; will matter once long runs do
  multiple epochs. Code is in git history (exp 15, commit range around a2f907c).
- **IMG_SIZE 448** may re-win at long budgets where steps aren't the binding constraint.
- **ViT-H+ backbone (0.84B)**: untried; ~2.5x FLOPs of ViT-L → wrong trade at 8 min, right
  candidate for long runs (fits 32GB with LoRA+bf16 at 384px, moderate batch).
- **Practice (Josef, 2026-07-21): when discarding an idea that lost on throughput, ALWAYS check
  W&B val/median_km at matched step counts; if it's better per-step, park it here instead of
  calling it a dead end. The session constraint is TIME, but long runs are constrained differently.**

## Open ideas / next to try (ranked)

_(carry unfinished/promising directions forward across sessions)_

### Measured 2026-07-23 — retrieval-readiness of the full-run model (ckpt_best, ALL 2998 val, local CPU)

Full logit cache + analysis script live in `run_full/val_analysis/` (re-slicing is instant, no re-inference).
Sanity: reproduced official numbers (median 104.4, @25km 7.6%). Key facts, all on 5000 k-means cells:

- **Exact-cell hit**: top1 27.6 / top5 57.0 / top10 70.0 / top25 84.3 / top50 91.6. **Median rank of the
  true cell = 3.**
- **Region recall (truth within R of ANY top-K centroid)** — the retrieval KPI, cell-count invariant:
  top10@100km **75.6%**, top25@100km 83.6, top50@100km 88.4, top50@200km **95.3%**.
- **Honest 90%-posterior-mass set**: median only **42 cells**, covers truth-within-100km **91.6%** of
  the time → a mass-based search region beats fixed top-K for retrieval.
- **Conditional medians**: top5-HIT cases (57%) sit at 67 km (cell-floor territory — exactly what a
  retrieval stage converts to <25km); top5-MISS cases already at 243 km median → retrieval can't
  make them worse. Asymmetry confirmed empirically.
- **Perfect-snap ceiling: @25km = 36.2% even with a PERFECT top-1 classifier** (5000-cell tessellation
  limit; PIGEON's headline is 40.4 with panoramas, 24.2 single-image). So @25km needs BOTH better
  classification AND floor-breaking (finer cells raise the ceiling; retrieval/offset removes it).
- PIGEON facts (arXiv 2307.05845 ablations): their smoothing == ours exactly (softmax(-d/τ), τ=75);
  semantic geocells worth ~10km median vs naive grids (k-means already density-adaptive, so our gap
  is only border-purity); refinement = the @1km mechanism (5.4%→1.3% without) but only −5pts @25km;
  panoramas = +16pts @25km (closed to us by mobile constraint — single-image PIGEON: 131 km median,
  24.2% @25km, i.e. WE beat their single-image median already).
- Eval panels (train.py + train_full.py) now log cell_top10/25/50 + region_recall_top10/50@100km.

New cheap ideas from this: **country-constrained k-means cells** (per-country clustering, budget ∝ data
share — makes the cell→country term exact, PIGEON's real semantic-cell edge); **confidence-adaptive
readout** (snap to top-1 data centroid when confident instead of spherical-mean smearing — @25km
booster, zero training cost).

### Invented 2026-07-22 (during full run) — novel mechanisms, ranked; designed from OUR evidence

1. **MSL — Median-Seeking Loss.** Differentiable prediction: v = normalize(Σ softmax(logits/T)·centroid),
   penalize haversine(v, true) with a REDESCENDING robust loss (Geman–McClure d²/(d²+s²), s≈300km).
   Kills the train(CE-on-cells)/test(spherical-mean) mismatch; gradient vanishes on hopeless tail
   cases = literally optimizes the median; net learns sub-cell interpolation → quantization floor
   dissolves without more cells. Soft-argmax/integral regression (pose estimation) ported to the
   sphere + median-robustifier. ONE loss term — 8-min-testable. TRY FIRST next session.
2. **PanoDistill.** Train data is 300k locations × 4 headings treated as independent — waste.
   Distill the fused 4-view posterior (geometric mean over the location's views, from an EMA copy,
   no separate teacher) into each single view: KL(fused_EMA ‖ view). Single-view mobile contract
   untouched; each view learns what the other three would reveal. Needs location-grouped sampler.
3. **FreeEarth heads.** Coordinates → free small-vocab labels via public rasters: Köppen climate
   (~30), biome (~14), coastal-distance band, elevation band, driving side (2). We PROVED small
   semantic vocabs work (country 115 ✓) and big ones fail (subdiv 1895 ✗). Köppen first.
4. **Confusion-forge sampling.** Online country-confusion matrix from quick evals → upsample the
   top-confused country pairs. Country_acc (43% vs PIGEON 92%) IS the median gap.
5. **EmbedOffset.** Why the S5 offset head died: N_CELLS×2 independent slots, ~30 samples each.
   Fix: ONE shared tangent-offset MLP conditioned (FiLM) on the chosen cell's learned embedding.
   Composes with MSL. Needs converged selector → full-run scale.
6. **GeoKernel alignment.** In-batch Gram matrix of features aligned to exp(-d_ij/σ): feature
   space becomes a metric atlas of Earth; makes future retrieval/kNN refinement strong.
7. **vMF mixture head (moonshot).** 16-component von Mises–Fisher mixture on the sphere, NLL
   loss — continuous, cell-free, no quantization floor by construction. Run as AUX head first
   (mixture-collapse risk).

### Invented 2026-07-22 part 2 — beating the CELL BOUND itself (floor @5000 cells = 34.2 km,
### measured; NOTE: floor applies to argmax readout only — the spherical-mean readout can go
### BELOW it, and a 100%-confident perfect classifier would score exactly 34.2, i.e. 100%
### top1 is not even optimal under our prediction rule)

8. **Barycentric geo-labels (TRY FIRST of this batch — pure label engineering, 8-min-testable).**
   Construct soft targets that are EXACTLY INVERTIBLE by the readout: per point, weights over
   K≈8 nearest centroids (w≥0, Σw=1, + entropy reg) s.t. weighted spherical mean of centroids
   = true point. Labels carry sub-cell coordinates; perfect learning ⇒ exact location through
   the EXISTING head/readout. Floor gone in expectation. Composes with MSL.
9. **Staggered tessellation ensemble.** 2-3 fine heads on different k-means seeds; Voronoi
   boundaries don't align → combined posterior (concat centroid/weight pairs into one
   spherical mean) has ~√2-√3 lower effective floor for ~5MB/head. Dithered quantization.
10. **Within-cell exemplar memory (retrieval, scoped).** S5 global retrieval failed; fix =
    per-cell scope: ~16 PCA-64 exemplars/cell + exact coords (20MB — mobile-viable). Predict
    weighted mean of EXEMPLAR coords after cell choice → nearest-neighbor floor (few km).
    Needs full-run checkpoint features — natural follow-up to the 2026-07-22 full run.
11. **Regional GPS heads.** Offset-head autopsy: per-FINE-cell slots starved (~30 samples).
    Flip granularity: 64 COARSE-region tangent-plane regression experts (~19k samples each),
    gated by coarse posterior. Dense supervision, region-miss-proof.
12. **Progressive cell splitting.** Split highest-residual cells during training, warm-start
    child logits from parent → adaptive mesh refinement; end ~12-15k cells (floor ~20km)
    without cold-starting a big head.
    Cell_top1 levers (it's a pessimistic proxy; don't chase 100%): confusion-forge, FreeEarth,
    PanoDistill, TTA self-ensemble over PatchDropout masks (2-3 fwd, mobile-affordable).

Refreshed at end of session 2 (2026-07-21, champion ~231–238). Session 2 burned through most
of the cheap menu — remaining ranked ideas:

1. **Geo-contrastive hard negatives (Josef's idea, 2026-07-21):** within ordinary shuffled
   batches, mine pairs that are visually similar (embedding cosine) but geographically FAR;
   push those apart in a projection space (or penalize overlapping top cells), weight
   ~ sim × distance, λ small (~0.1). Positives are already handled by tau-smoothing; do NOT
   use special batch samplers (exp 10 lesson — diversity is sacred). Targets the ~10%
   wrong-continent look-alike errors directly.
2. **EMA of trainables** (eval the EMA copy) — classic free win, untried.
3. **GeM pooling over patch tokens** (cls_mean was neutral; GeM is the stronger variant). Also
   try cls+GeM concat.
4. **Deeper hierarchy / rebalanced weights** (e.g. add 8-cell level, or w 0.15/0.35/1.0) — ONE
   follow-up allowed on the banked hier win.
5. **LoRA on MLP fc1/fc2 too** (currently attention-only) — more adaptation capacity, small
   speed cost.
6. **Country-code auxiliary head** (metadata has country_code; ~150-way CE aux) — different
   supervision signal than cells.
7. **Focal/entropy tweaks on fine CE** for the tail.
7b. **(Process, from Josef): run the `research` skill / literature sweeps BETWEEN experiments
   every session — don't only work down this list; hunt for new SOTA ideas each time.**
8. **Top-K cell RERANKING (Josef's framing, 2026-07-21):** top1=16% but top5=44% — in ~28% of
   val the right cell is in the top-5 but ranked wrong. Choosing among 5 candidates is much
   easier than 2048-way CE. Options: (a) PIGEON-style retrieval rerank — cache train embeddings
   per cell, rerank top-K by cosine to query (their biggest ablation win; eval-time only, no
   training cost); (b) a small learned reranker head over top-K (cell embedding + image feature).
   NOTE: this is DIFFERENT from the parked within-cell offset (that refines inside a chosen
   cell; this fixes CHOOSING). Un-parked.
9. **Retrieval for within-cell refinement & offset regression**: still PARKED until top1 ≫ 16%.
9. Prediction-rule sweep is DONE (don't re-sweep); tau is DONE (75).

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

## Literature sweep (2026-07-21, post-session 2) — new idea sources

- **Semivariogram hard negatives** (arxiv 2509.21573): formalizes exactly Josef's geo-contrastive
  idea — image dissimilarity grows with geo distance only up to a RANGE, beyond which it
  plateaus; they mine hard negatives as pairs whose feature distance is far below the
  semivariogram-expected value for their geo distance. Supports open-idea #1; use their
  range-based weighting instead of raw sim×distance.
- **HierLoc (ICLR'26)**: hierarchy of geographic ENTITIES (country→region→city) embedded in
  hyperbolic space; images aligned to entity embeddings via geo-weighted contrastive. SOTA on
  OSV-5M. Cheap partial adoption: add entity-level (country_code) alignment aux — we already
  bank hierarchical CELL heads; an entity head is complementary (open idea #6 upgraded).
- **Pinpoint (arxiv 2606.04133)**: retrieval + RERANKING; reranking stage adds +23.9pp acc@1km
  over retrieval-only. Strong evidence for open-idea #8 (top-K rerank) being the next big win.
- **Scaling Geo-Localization to Continent Level (NeurIPS'25, Lindenberger)**: classification
  prototypes + aerial-image embeddings; 68% within 200m over Europe. Aerial cross-view is out
  of scope for our harness (no aerial data in the frozen subset) but "per-cell PROTOTYPE
  embeddings learned via proxy classification" is adoptable: rerank top-K cells by cosine
  between image feature and learned cell prototype (= a learned reranker with zero extra data).
- **PIGEON semantic geocells**: admin-boundary/OSM-based cells instead of k-means — better
  cell semantics claimed. Only worth it if we can build cells from train metadata
  (country_code/subdivision) without new downloads: hierarchy country→subdivision→kmeans-within.

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

### 2026-07-24 (night 5) — C3 ENCODER FINE-TUNE + LEVER SWEEP: 43.75 -> 36.96 km, @25 42.9 -> 46.1%

- **CHAMPION: C3 descriptors, 4-band head blend (d_pos 5/10/25/50) + CSLS + whitened cls +
  regional rerank (R50 a0.25), gate = OLD ckpt logits, mass .95 lam .05, k1 snap.
  Full val: median 36.96 | mean 378.3 | @1 12.78% | @25 46.06% | @200 74.32% | GG 4452/round
  (22 258 per 5-round game).** Day: 104.7 -> 36.96 median, 7.6 -> 46.06% @25, 0 -> 12.8% @1.
- **C3 = contrastive fine-tune of the BACKBONE** (retrieval/train_c3_contrastive.py): LoRA warm
  start from run #2, positives <=10 km, negatives >=50 km masked, region-restricted batches
  (~500 km) for hard negatives, gathered NT-Xent across 4 GPUs, joint projection head, 2500
  steps (0.53 epochs, 42 min, 260 img/s). **Raw CLS descriptor 36.32 -> 42.46% @25 (+6.1 pts,
  the largest single-component jump of the project).**
- **BUT the naive C3 champion was a wash** (47.57 / 42.73%) until the missing piece was added:
  run_c3.sh only trained a head on the mean-patch branch. With an offline geo10 head on the C3
  CLS: 40.97 / 44.33%. **The head trained ONLINE inside the fine-tune scored 33.89% — 10 pts
  WORSE than the same recipe fit offline on cached vectors.** Train encoders online, heads offline.
- **Josef's diagnosis of the 25-200 km regression was correct and is now the C4 centrepiece:**
  the binary loss masks 10-50 km and pushes 50-500 km apart at full strength, and region batches
  made 50-500 km nearly the only negative seen. @1 km improved while the 25-200 km band decayed.
  Fix = graded targets exp(-d/tau_geo), the same smoothing the classifier already uses.
- **LEVER SWEEP on cached descriptors (retrieval/eval_levers.py), all cents:**
  | lever | median | @25 |
  | CSLS hubness correction | 39.94 | 44.90% | **biggest cheap win, 40 s of arithmetic** |
  | 4-band blend 5/10/25/50 | 39.60 | 45.06% | a single wider head (25 km) FAILED; the blend won |
  | whitened cls | 41.00 | 44.83% |
  | **all three combined** | **36.96** | **46.06%** | orthogonal, they add |
  Duds: location top-2 evidence (44.03), adaptive lambda by entropy (43.40 / 41.79), confidence
  router to the classifier readout (43.96 @25 — but it DOES improve mean 378->368 and @200).
- **Gate must stay on the OLD checkpoint**: gating with C3's own logits scores 42.93% vs 44.20%.
  The contrastive fine-tune damages the classifier on the shared trunk -> inference is currently
  TWO encoder passes. Joint CE+contrastive training (C4) is the fix.
- Plan for the next run: **docs/C4_PLAN.md** (graded loss, joint objective, MLP LoRA r32, mixed
  batches, 2 epochs, offline band heads, probe every 250 steps + kill criterion). ~3 h, ~$5.5.

### 2026-07-24 — INFRA: cost/time optimisations (measured, apply to every future session)

1. **Prebuilt cache tar — DONE and live.** tools/build_cache_tar.py re-encodes the cache to 384 px
   and streams it straight into a tar (staging a copy needs 2x disk; the box had 64 GB free of the
   90 GB required). Uploaded as `josefbednar/streetview-acw-ar-cache/cache_n1198072_v3000_s1337.tar`
   (50.1 GB, 83 MB/s). **Setup drops ~28 min -> ~10 min, ~$0.9 saved per session.**
   Gotcha found by the smoke test: the filename embeds AR_N_TRAIN — build it with
   `AR_N_TRAIN=1198072` or setup will never find it.
2. **Freeze the bottom 8 blocks** when fine-tuning (LoRA only in blocks 9-24): backward stops
   early, ~-30 % step time, low-level texture layers do not need adapting.
3. **Patch dropout 0.25 + torch.compile** during contrastive training: ~-35 % combined
   (precedent: the classifier trained at 0.6 dropout and evaluates at full patches).
4. **Keep every idea in cached-descriptor land as long as possible** — heads and levers cost
   cents there ($0.15/head) and dollars once images are involved. 104.7 -> 36.96 cost ~$9 total.
5. **Interruptible instances: REJECTED by Josef** — only ~$1.4/hr vs $1.89 on-demand, not worth
   the preemption risk.
6. **Two jobs share a GPU fine** — the limit is VRAM, not policy (head training 12.4 GB, lever
   sweep 22 GB: two heads per card yes, head + sweep no).

### 2026-07-24 — MISTAKE: mirror artefacts BEFORE the box dies, not once

The HF mirror ran while geo5/geo25/geo50 were still training, so **the 4-band heads and the
regional descriptors were lost when the box was destroyed** — the 46.06% champion is not exactly
reproducible without ~15 min of head retraining on a future box (the cached C3 descriptors they
train from ARE on HF, so nothing is unrecoverable, just re-spendable).
**Rule: re-run the mirror as the LAST step before `vast.py down`, and mirror every new .pt as it
appears, not in one batch.**

### 2026-07-24 (night 5, late) — C4 joint fine-tune: TRAINED, NOT EVALUATED (broken box)

- **C4 = joint encoder+classifier+place-head fine-tune with graded-distance contrastive loss**
  (Josef's diagnosis of C3's 25-200km regression -> soft targets exp(-d/tau_geo), tau_geo=100).
  Trains all three at once so the classifier stays valid on the moving encoder (C3's flaw: it
  froze the classifier, so gate needed the OLD ckpt = 2 forward passes). LoRA now on MLP too
  (up/down_proj): 7.1M LoRA + 12M classifier + 4.2M place head.
- **Training COMPLETED (1800 steps, ~1h): con-KL 1.19 -> ~0.4, ce-KL 4.2 -> ~2.7 — both learned.**
  W&B run c4-joint-graded. Checkpoint mirrored to HF (c4/ckpt.pt, c4/c4_head.pt).
- **NOT EVALUATED: the box (Bulgaria 45679713) was pathologically slow** — training 33 img/s
  (vs C3's 260 on prior boxes), and the re-index ran at 23 img/s on GPU 0 ONLY (GPUs 1-3 idle at
  0%, GPU0 throttled to 547/3090 MHz). Re-index ETA was 3h vs the ~22min it takes on a healthy
  4x5090. Credit hit $1.78 so the box was destroyed before the re-index finished. **We have NO
  C4 retrieval number.**
- **The C4 encoder is saved and re-evaluable in ~30 min on a healthy box** (racing for high
  clocks/dlperf, verify nvidia-smi shows all 4 GPUs working + full clocks BEFORE the long run):
  pull c4/ckpt.pt, run embed_c2 --tag c4 (train+val), train the 4 band heads offline, run the
  CSLS/whitening/rerank grid (eval_levers.py + eval_c2_gpu.py). Champion to beat: 36.96km/46.06%.
- **Bugs found & fixed tonight (all committed)**: cache-tar filename (AR_N_VAL renames it);
  blind kill criterion on a low-resolution probe (crashed run 1 via NCCL timeout); probe running
  on all ranks desynced them past the collective timeout (-> rank 0 only + barrier + 2h PG
  timeout); autograd all_gather deadlock once a 2nd loss branch was added (-> local contrastive +
  bucketed manual grad all-reduce); gradient checkpointing hung the first step (-> --no-ckpt).
- **Cost lesson: SCREEN THE BOX BEFORE THE LONG RUN.** nvidia-smi clocks + a 30-step throughput
  check would have caught this dud box before ~$5 was spent training on it at 1/7 speed. The
  race-rent filters dlperf but a host can still be throttled/misconfigured at run time.

### 2026-07-24 (night 4) — C2 512px re-index: 49.0 -> 43.75 median, @25 41.2 -> 42.9%

- **Champion: geo10_cls512 (+) geo10_mean512 blend + regional chamfer rerank (R200, a0.5),
  gate m0.95 lam0.05, k1 snap. Full val: median 43.75 | mean 373.2 | @1 11.27% | @25 42.93% |
  @200 73.92% | GeoGuessr 4438/round (22188 per 5-round game, 69.3% of rounds >=4500).**
- **Per-lever attribution (Stage A measures every space ALONE, before blending):**
  512px vs 384px: **+0.33 pts** (35.99 -> 36.32) — resolution is NOT the lever, the single
  most surprising result of the program. mean-patch alone 32.99 (worst descriptor solo);
  cls alone 36.32; geo10_cls512 39.79; geo10_mean512 38.33; **blend of the two heads 42.39
  (+2.6 over the best single)**; + regional rerank 42.93.
- **DEAD ENDS (measured, do not retry): query TTA** (flip/zoom, -0.4 @25, helps mean/@200 only);
  **multi-resolution 512(+)384 ensemble** (-0.9 vs pure-512 blend — same model, correlated);
  **gate-mass sweep** (0.90/0.95/0.99 identical; lam 0.05 optimal, 0.0 and 0.15 both worse).
- Regional descriptors: 4x4 block-mean of patch tokens, PCA-whitened to 128d (4.9 GB for 1.2M),
  chamfer (max over query regions, mean over candidate regions) rerank of top-R. +0.5 pts.
- Row order is nshards-independent (contiguous IterableDataset.shard): a 32-worker download
  reproduced the 16-worker run-#2 order exactly, verified sample-for-sample. Old indices stay
  reusable across boxes.
- **The read: two rounds of head engineering (41.2 -> 42.9) are hitting diminishing returns
  while the in-pool oracle sits at 92-98%. The ceiling is in the FEATURES, not the pooling.**

### 2026-07-24 (night 4) — LESSONS (process, not results)

1. **Instrument every training loop with the metric the final decision uses.** The C3 fine-tune
   ran blind: its loss swings 1.6-2.9 purely on region difficulty, so it predicts nothing, and
   the 5 intermediate checkpoints can't be ranked without a full re-index each. train_geo_head's
   probe (near-recall@1 <=10km, own location excluded) costs ~2 s and would have given both a
   forecast and checkpoint selection. Add it to any future fine-tune BEFORE launching.
2. **Blend diversity is a resource to be protected.** +2.6 pts came from CLS and mean-patch
   making *different* mistakes. Fine-tuning one descriptor risks correlating them. A previous
   model's descriptors are decorrelated by construction and (see row-order note) still aligned
   -> always keep them as candidate blend partners in the grid, even when they lost as a
   same-model partner.
3. **Measure every component alone before combining** (Stage A). It is the only reason
   "mean-patch is the worst solo descriptor AND essential to the champion" is a known fact
   rather than a guess.
4. **Decouple what you change from what you rely on.** C3 changes the encoder but the retrieval
   gate keeps using the OLD checkpoint's logits, so a worse encoder cannot poison the gate.
5. **Hard negatives matter for geo-contrastive training**: globally random anchors gave in-batch
   acc 0.81 by step 25 (trivial task); drawing all anchors of a batch from one ~500 km region
   dropped it to 0.49 at the same step, with <50 km pairs still masked out of the loss.

### 2026-07-23 (night 3) — C1.5 GEO-SMOOTH heads (~$0.35): median 61.2 -> 49.0, @25 38.7 -> 41.2%
- **Diagnosis that unlocked it: C1's same-place objective was subtly WRONG for eval** — val
  locations are never in the index, so the best reachable match is a different location
  1-15 km away, and same-place contrastive training explicitly pushes those neighbors apart.
  Fix = geo-smooth contrastive (retrieval/train_geo_head.py): positives = pairs within
  d_pos km (same OR neighboring location), in-batch negatives <50km masked (false negs).
- Two heads (d_pos 25 & 10; ~$0.15 of 1x5090 each, trained on cached embeddings in ~1 min).
  Eval-realistic probe (own location excluded): raw 4.96% -> geo25 8.9% (+79%).
- **Full-val GPU grid (40 configs, retrieval/eval_spaces_gpu.py, ~2 min on a 5090):
  champion = geo10⊕c1 sim blend 50/50: median 49.04 | @1 11.57% | @25 41.16% | @200 73.32%.**
  Every geo-space config beats every non-geo config at @25. Duds: location-max-pool (no-op
  under k=1 snap), alpha-QE (consistently ~-0.5pt). Grid log: run_full2/retrieval_index/c15_grid.log.
- Day summary: **median 104.7 -> 49.0 | @25 7.6 -> 41.2% | @1 0 -> 11.6%**, retrieval total
  cost ~$1.6. Webapp serves the blend (engine tag retrieval-snap-geo, ~1.2s CPU/guess).
  Heads mirrored on HF (place_head, geo_head_25, geo_head_10 in fullrun2-ckpt/retrieval_index).
- **Champion full distance profile (recomputed locally, reproduces the box grid exactly):**
  mean 388.0 | p10 0.8 | p25 5.5 | p50 49.0 | p75 219 | p90 600 | p95 1219 | p99 8241 km;
  @0.5km 8.0 | @5km 23.8 | @10km 31.8 | @50km 50.2 | @100km 60.7 | @500km 88.0 | @2500km 97.4%.
  **GeoGuessr world-map score (5000*e^(-d/1492.7)): 4417/round avg, 4838 median round,
  22085 per 5-round game; 68.7% of rounds >=4500, 43.5% >=4900.** Classifier-only readout
  ~4274/round. Mean is tail-dominated (2.6% beyond 2500 km) -> mean is a country-accuracy
  metric, not a fine-grained one. Script: scratchpad/score_champion.py; per-image distances
  cached at run_full2/retrieval_index/champion_dist_km.npy.
- Next for 50%: C2 patch-aware descriptors at 512px (in-pool oracle 92-97% still leaves
  50+ pts of matching headroom); possibly geo-head retrained on C2 descriptors, same recipe.

### 2026-07-23 (night 2) — PHASE C1 place head (~$0.15): 65.7 -> 61.2 median, @25 36.0 -> 38.7%
- Contrastive head (residual MLP 1024→2048→1024, zero-init out = identity start, NT-Xent
  τ.07, 2 views/loc positives) trained on CACHED bb embeddings — no images, 10 epochs in
  ~30s on a $0.44/hr 1x5090. **Heldout view-recall@1: 84.9% -> 99.0%** — raw DINOv3 CLS
  fails same-place-different-heading 15% of the time; the head almost never.
- Full-val (k1 snap m0.95 lam0.05 in projected space): **median 61.16 | @1 11.47% |
  @25 38.69% | @200 71.55% | mean 390** — better than Phase A champion on every axis but
  @200 (−1.1). Day total: 104.7 -> 61.2 median; @25 7.6 -> 38.7%; @1 0 -> 11.5%.
- Artifacts: retrieval/train_place_head.py (committed); place_head.pt + train_proj.npy +
  val_proj.npy in run_full2/retrieval_index/ (local). Head is 17MB — mobile-fine.
- **Webapp now serves the full engine**: run2 ckpt + place head + mass-.95/λ.05/k1 snap
  (engine tag + match-similarity shown; classifier readout kept as fallback + belief radii).
  Spot checks: val#0 Uruguay 21 km, val#7 Sweden 3.4 km, ~950 ms CPU.
- C2 still queued (patch-aware head + 512px re-index, needs images + 4 GPUs): in-pool
  oracle 92-97% @25 says the road to ~50% runs through richer descriptors.

### 2026-07-23 (night) — RETRIEVAL PHASE A (all-local, $0): 79.7 -> 65.7 median; ceilings measured
- **Ceilings on full val: ORACLE coverage @25km = 100.0%** (median nearest-train-image 2.2 km,
  @1km 38.5%); in-pool oracle @25: 92.1% (mass .8) / 97.6% (mass .95). ⇒ coverage and gate are
  NOT constraints; the whole 36%->92% @25km gap is MATCHING quality (Phase B/C target).
- **New champion readout: k=1 snap, mass 0.95 gate, soft prior sim+0.05·log p(cell): median
  65.66 | @1 10.11% | @25 35.99% | @200 72.65%** (beats even the classifier's @200). Day total:
  104.7 -> 65.7 official median. Consensus voting = dud (median-neutral, dulls @1/@25 — snap wins).
- Infra: train lat/lon table reconstructed order-faithfully from HF parquet column reads
  (datasets 5.0 contiguous-shard guarantee; 12-thread reads ~4 min; verification gate
  reproduced box numbers exactly). Index now fully local: run_full2/retrieval_index/
  (emb_bb train 2.4GB + val arrays + train_latlon.npz). Analysis: scratchpad analyze_phaseA.py,
  results in this block; local CPU runtime ~4 min.
- Next (Phase B, ~$2-3): 512px re-embed, CLS⊕mean-patch descriptor, query multi-crop, patch
  rerank. (Phase C, ~$5): contrastive place head on frozen bb — 4 views/location = free
  positives; 56 pts of in-pool @25 headroom says this is the highest-leverage move left.

### 2026-07-23 (later) — RETRIEVAL ENGINE v1: 104.9 -> 78.5 median, 0->10.2% @1km, 7.7->35.2% @25km
- Built retrieval/ (embed_full.py sweep, engine.py mass-gated kNN, eval_retrieval.py 2-phase
  tuner). Swept 1.2M train imgs on the run-#2 box (~905 img/s 4x5090, 21.7 min) at TWO taps.
- Findings: (1) **bb (backbone-CLS) tap wins every top config** — confirms the collapse
  hypothesis directionally: geo-trained trunk features lose instance detail; (2) sharp
  matching wins (k=1-6, temp .005-.01) — "trust the nearest visual match"; (3) tighter mass
  gates (0.6-0.8) beat 0.9+ — fewer distractors > recall, exactly the distractor-risk tradeoff;
  (4) similarity floor irrelevant (0.0-0.6 identical) — bad matches are already far, fallback
  rarely engaged (asymmetry confirmed end-to-end); (5) mean also improves (482->406-428).
- vs PIGEON: beats their single-image numbers on every metric (median 78.5 vs 131, @25 35 vs
  24, @1 10.2 vs 0.9); @1km even beats their 4-view PANORAMA 5.36%.
- Costs: sweep ~$0.7, tuning ~2 min/GPU. All grids in run_full2/retrieval_eval*.log.
- Next: PCA/PQ-compress index for mobile (target ~20-40MB); wire top-config into webapp;
  retrain-then-reindex loop (every future model improvement compounds through this stage);
  per-cell exemplar curation; consider contrastive projection head if a finer tap is needed.

### 2026-07-23 — FULL RUN #2 (Bulgaria 4x5090 $1.815/hr, 62 min train): official 104.37 (wash)
- Recipe = run #1 + MSL(0.5/300/T.5) + staggered tessellation B + country-constrained k-means
  (5000 cells, floor 33.8 vs 34.2 plain) + parked RRC 0.5-1.0/jitter-.15 aug. 15.16M trainable.
- Full-val final: median 104.37 | mean 478.7 | @25 7.77% | @200 70.1% | country-run best quick
  102.54. vs run #1: median −0.3, mean +23, @25 +0.2, cell_top1 −1.2. **A wash.**
- Lessons: (1) ratchet keeps don't stack at full scale either (S6 integration lesson repeats);
  (2) two different recipes → same ~104 ⇒ the 3-epoch/LoRA-r16 classifier regime is the
  binding constraint, not the loss/tessellation; (3) aug at 3 epochs ≠ the long-run it was
  parked for — earn it back at 5+ epochs or drop it; (4) this box did 1040 img/s trained /
  ~905 swept (dlperf 647) — 2x run #1's effective throughput at 75% of the $/hr.
- Infra: stale run-#1 ckpts uploaded by repo sync tripped the new stagger/cc resume assert
  (assert did its job — delete run_full/*.pt on the box before a fresh-architecture launch).
  W&B netrc interpolation worked (run live from step 0). Race-rent v2: ssh-url shows the
  PROXY until direct port provisioning lands — poll for the flip, don't reject on first sight.
- Retrieval-readiness at final: cell_top1 26.2 / top10 70.5 / rr10@100 75.7 — same regime as
  run #1 ⇒ retrieval index built on run2 ckpt_best (stagger head + aug-trained embeddings).

### 2026-07-22 evening — S6 idea-batch ratchet (Italy 1x5090, 8 exps + integration): 215.65 -> 205.46
- **KEEP: MSL** (differentiable spherical-mean + Geman-McClure km loss, w0.5/s300/T0.5):
  215.65->209.49 (−6.2, biggest since PatchDropout). cell_top1 UNCHANGED — pure mass-placement
  win, the quantization-floor thesis validated. **KEEP: staggered tessellation** (2nd fine head
  seed1, union readout): 209.49->205.46 (−4.0). **KEEP-but-does-not-stack: vMF-aux** (16 comp
  w0.25): 207.69 on MSL base (−1.8) BUT integration MSL+stagger+vMF = 211.89 — aux crowding;
  champion stays MSL+stagger.
- Discards: pure barycentric labels 212.11 (acc@25 3.8% box-best — tau+bary BLEND parked);
  elev+season pure-aux heads 213.54 (aux without readout wiring taxes gradient); GeoKernel
  209.85 (neutral — retrieval enabler, pair with exemplar memory); confusion-forge 216.51
  (sampler diversity law, 3rd confirmation); PanoDistill@84 234.08 (21 locs/batch — park for
  full-run epochs).
- Full-run queue: add MSL to train_full.py next full run (+ consider stagger); vMF/PanoDistill
  candidates at scale only. Infra: variant-generator class-scope bug cost 2 crashed runs —
  ALWAYS AST-check GeoModel methods after scripted edits; OOM pairing rule applied 2x (bs84).

### 2026-07-22 — FULL-DATASET RUN (not a ratchet session): 209.5 -> 104.7 official
- train_full.py @ master: champion stack, ALL 1,198,072 imgs (300k locs x 4 views), 3 epochs,
  10,695 steps @ global bs336 (4x5090 Hungary $1.87/hr, bs84/GPU after NCCL-buffer OOM at 96),
  5000 k-means cells (floor 34.2 km), step-based cosine, evals/100k samples, ckpts/200k + HF
  mirror. 2.0h train, ~$5.6 all-in.
- OFFICIAL full-val median 104.66 | best quick-subset 99.91 (<100!) | mean 455.6 | acc@200 70.6%
  | acc@2500 96.9% | geoguessr 4286 | cell_top1 27.4%@5000. Beat original neuroguessr (194km,
  same 1.2M imgs, full FT) by 46% with LoRA r16. Curve still descending at cutoff ->
  epochs 4-5 via checkpoint chaining (run_full/ckpt_best.pt local + josefbednar/
  neuroguessr-fullrun-ckpt) is the cheapest next win.
- Infra: raw-bytes download = 1.2M imgs in 266s (~6.8k img/s); smoke test caught the DDP OOM;
  W&B key must come from ~/.netrc for manual nohup launches (silent 'disabled' otherwise);
  pkill self-match killed a relaunch shell (separate the reap from the run — again).
- HARD INSTANCE RULES (Josef, permanent): no California, direct SSH only, 5-min
  startup-or-destroy, time-boxed setup, race EU regions first.


_(one dated block per session: dates, champion at start → end, headline results)_

### 2026-07-22 (session 5 part 2, after credit top-up — Korea box 45534538, $0.40/hr)
- Anchor on this box: 214.2 (same stack Estonia read at ~209.9 — boxes differ ~4km; ALWAYS
  re-anchor after a box change). End: **213.0** (1dfceb0) = est. ~208.8 Estonia-scale.
- 8 scored runs + 2 OOMs: PATCH_KEEP 0.6 KEEP (−1.1); ViT-H+ 211.2 dropped by Josef (size);
  semantic cells, H3-as-fine, H3-as-level, offset head, LoRA r32 discards; GeM missed by 0.11.
- New val metric: country_acc (predicted cell's country vs true) — ViT-L ~44-46%, H3-fine
  54.5%, PIGEON reports ~92% — the country gap is the clearest remaining signal deficit.
- Infra: Vast CONFIRM PROMPT eats destroys silently (`input='y\n'` required — two "destroyed"
  losers kept billing; always `vast.py ps` after destroys). SSH stream drop mid-exp shows as
  CRASH while the run continues on the box — recover score from run/run.log. Old-repo assets
  (h3 jsons) scp'd to CACHE_DIR + `pip install h3` on box. progress.png + analysis.ipynb
  rewritten for median_km with session bands.
- **Next-session queue:** 1) GeM retune (replace-not-concat @ bs96/keep0.5 — missed by 0.11!),
  2) sim-gated cluster-centroid retrieval retune, 3) season/elevation SMALL aux heads (4-way
  works, 2k-way doesn't), 4) HARNESS DECISION for ≤100km goal (Josef): AR_N_TRAIN 60k→150-300k
  + --minutes 20-30 + optionally checkpoint chaining; original neuroguessr hit 194km with
  1.2M imgs / 2h full-FT — data+budget IS the gap (we're at ~209-213 with 60k/8min).

### 2026-07-22 (session 5 part 1 — mandate: aux heads + geographic hierarchy, then
### structural swings toward ≤100km; PAUSED: Vast credit ran out)
- Champion at start: 210.1 (box baseline 212.2) → at end: **~209.9** (07bd934); wins were
  tail-fixers: country-hier + tau-smoothed country targets (mean 1079→~1000 best ever).
- Experiments: 11 runs (baseline, 2 keeps ×2 confirms each, subdivision discard, 448
  OOM+discard, retrieval-v1 discard, semantic-geocells NEVER RAN — box died first).
- **BLOCKER at end: Vast account balance went NEGATIVE (-$0.02) mid-session** → host
  force-stopped the box ("exited", restart queued-unavailable), new creates return None.
  Also: "destroyed" instances can linger as `exited` zombies billing storage — always verify
  with `vast.py ps` after destroys. TOP UP CREDIT before next session.
- Infra learned: offer-id rent is RACY (find+create must be ONE process); race-rent 3 regional
  candidates → first with working SSH wins (Estonia won in 2 min; scratchpad/race_rent.py
  pattern worth folding into vast.py pre-loop next session). Cache tar on HF was val-only/
  INCOMPLETE since S1 — rebuilt+re-uploaded correct 4.65GB tar (TAR_UPLOAD_OK); next setup
  should be genuinely ~2-3 min. Boot filter: prefer hosts that pull the image fast; two duds
  (proxy-only SSH; 10-min docker build) cost ~35 min.
- Next session queue: 1) semantic geocells (code READY in tree), 2) ViT-H+ compute-matched
  (bs48+dropout, GRAD_CHECKPOINT maybe), 3) sim-gated cluster retrieval retune, 4) PATCH_KEEP
  0.6, 5) EVAL_EVERY 500 (harness-efficiency, ~50s/run). Honest 100km read: needs harness
  changes (data/panoramas/budget) — raise with Josef at session start.

### 2026-07-22 (session 4, night — mandate: big swings)
- Champion at start: 221.5 (baseline re-run 221.50 exactly) → at end: **210.1** (63e090c),
  −5.1% this session, −60.8% cumulative from 535.4
- Experiments run: 9 (baseline + 6 ideas + 2 confirms; 2 KEEP confirmed, 4 discard, 1 3-min crash)
- Headline: **PatchDropout 0.5** (−6.1: drop half the patch tokens in train forwards,
  index-select RoPE cos/sin; +110% steps, VRAM halved) then **bs96×1** (−5.3: spend the freed
  VRAM on batch). Key scientific yield: **step-scaling saturates ~2750 steps at 8-min** —
  FixRes hit 4410 steps and LOST; past saturation the winning currency flips from steps to
  per-step quality. LR and cell-count sweeps now DONE/frozen. Muon-on-head flopped at lr .02.
- Infra: Austria box (offer 43122662, $0.40/hr). Lost ~65 min to (a) setup's single SSH
  session dying silently (local hang, remote orphan kept running — TWO racing setups), and
  (b) prepare.py cache-tar download stuck in CLOSE-WAIT (HF CDN hung up, no socket timeout).
  Recovery pattern that worked: kill both, relaunch prepare.py directly under nohup with
  `python -u` + log file, stream the log. TODO NEXT SESSION (pre-loop, allowed): make
  `vast.py setup` run remote-side under nohup+log with local tail/reattach + heartbeat; add
  socket timeout/retry to prepare.py tar download. Spend: ~$1.35.
- For next session: bs96 needs 25.8GB (5090-only); at 4090 use bs48. PATCH_KEEP itself swept
  only at 0.5 — 0.4/0.6 is ONE allowed follow-up. ViT-B compute-matched swap still untried
  (less appealing now that steps saturated). Long-run parked list unchanged (EMA, aug,
  IMG 448, ViT-H+, finer cells, rerank family).

### 2026-07-22 (session 3)
- Champion at start: 229.9 (box baseline 232.4) → at end: **221.5** (ae5c260), −4.7% this session
- Experiments run: 10 (baseline + 7 ideas + 2 confirms; 2 KEEP confirmed, 5 discard)
- Headline: **torch.compile + pre-clock warmup** (−7.6, +14% steps) then **bs48×1 graduates
  from the parked list** (−3.3 confirmed twice) — both wins are throughput/optimization-scale;
  ALL classifier-side auxiliaries (geo-contrastive, proto ensemble, surgical rerank, EMA)
  improved cell_top1/top5 but left median flat → at 8-min budgets, don't buy classification,
  buy steps. Reranking/aux ideas should be re-tested only at long budgets.
- Infra: CA DC 209.146.116.50 confirmed broken (2 dead boxes, ~$0.30); vast.py now sends SSH
  keepalives; `--offer-id` handpick of the Thailand host worked; cache-tar setup ~2 min.
- Follow-ups: LoRA-on-MLP result (exp 25, see findings), GeM pooling untried, country-code aux
  untried (but see aux-head pattern above), deeper-hierarchy follow-up still unused.

### 2026-07-21 (session 2, evening)
- Champion at start: 283.4 (baseline re-run 282.8) → at end: **229.9** (32075c7), −19% this session
- Experiments run: 9 (baseline + 8; 3 KEEP incl. confirm, 5 discard, 1 parser-CRASH that was
  really a finish — score recovered from run.log)
- Headline: hier heads 64/512/2048 (−18.9), IMG 384 (−25.9), EVAL_EVERY 250 (−8, confirmed).
  Dead ends: pano-InfoNCE pair batches (+23), tau 110 (+7), bs48×1 (+9, but parked — better
  per-step), TTA crops (+9), aug (neutral, parked).
- Infra now durable: cache tar on HF (2-min setups), part-parquet resume, socket timeouts.
  Box: Thailand 5090 $0.38/hr, screened 20 MB/s HF. ~50 min lost to w08 stream stalls before
  the resume fix landed.
- Key diagnostics: top1 16% / top5 44% → RERANKING top-K is the big open direction (see open
  ideas #8); Josef's geo-contrastive hard negatives is #1.

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
