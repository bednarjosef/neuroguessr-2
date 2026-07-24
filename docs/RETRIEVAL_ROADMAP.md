# Retrieval roadmap — median → ~20 km, @25km → ~60%, mean → <200 km

*Written 2026-07-24 evening, during the C4 eval rescue. Everything here is grounded either
in our own measurements (cited by session) or in literature (OpenAlex ids). Local E-series
experiment scripts live in the session scratchpad (`local_ab*.py`); port the winners into
`retrieval/` once validated on the C4 index.*

## The arithmetic that tells us where to aim

For every val query a train image within 25 km **exists (100%)** and **survives the gate
92–98%** of the time — but the matcher picks one only **45.1%** of the time (C4_PLAN §1).
So @25km = 60% requires within-pool matcher accuracy ~63–65%. And the moment @25km crosses
50%, **the median is mathematically below 25 km** — one lever, both metrics. At 60% the
median lands ~12–18 km.

The failure mode: a Finnish forest road query whose top-1 is a near-identical Finnish road
300 km away while the true ≤25km neighbour sits at rank ~7. This is **perceptual aliasing**
— a visual-place-recognition problem, not a geolocalization problem. That reframe is the
map to the right literature.

The mean is a different problem: it is owned by ~5% catastrophic misses where **both**
evidence channels (descriptor space AND classifier posterior) point to the wrong region
(measured, S384 bed, 2026-07-24: >1000km tail = 174 queries carrying 259 of 377 mean-km;
only 29% of tail pools contained a ≤25km candidate; wide re-gating at mass .995 changed
nothing; classifier fallback hurts because classifier-only mean is 471 — failures are
correlated). **The mean needs new evidence, not new arithmetic** — see §Mean below.

## Measured tonight (S384 bed, top-100 gated pools, 5-fold CV where learned)

| config | median | mean | @1 | @25 | GG |
|---|---|---|---|---|---|
| baseline (geo10\|c1, m.95, λ.05, k1) | 49.04 | 388 | 11.6 | 41.2 | 4417 |
| E1 geo-consensus R25 | 45.08 | 371 | 10.2 | 42.5 | 4427 |
| **E7 learned rerank + blend λ1.0** | **42.74** | 377 | 11.0 | **43.1** | **4430** |

- **VALIDATED: candidate-consensus reranking** (cluster of mutually-close candidates beats
  a lone high-sim outlier) and a **learned listwise reranker** — a 16-feature logistic on
  2998 queries already buys −13% median. The full version (set transformer, millions of
  train-side episodes, graded exp(−d/τ) target — NOT binary ≤25km, that crashed @1km) has
  clear headroom.
- **KILLED with evidence** (do not retry): k-reciprocal (2 fair variants), visual/geo graph
  diffusion, SuperGlobal-style query expansion (2 variants), classifier fallback (×2, the
  second with corrected centroids), wide re-gate for the tail, geometric-median snap.
- **BUG to remember:** run-1 `cells.npz` coordinates were paired with run-2 logits in early
  rounds — any cross-run cell mapping must be checked against `centroids.npz` of the SAME run.

## Tier 1 — cached vectors, cents, no GPU (exhaust first — standing rule)

1. **Learned listwise reranker over the gated pool** — replace the hand-tuned blend
   (fixed band weights + CSLS + λ=0.05) with a small set-transformer over the top-50,
   per-candidate features: band sims, raw sim, CSLS sim, gate posterior, rank, margin,
   **pairwise km between candidates**, sibling-view sims. Graded soft target ∝ exp(−d/τ).
   Train on train-side self-retrieval episodes, offline. Literature: Reranking Transformers;
   pillar re-ranking (W4367060669). *Status: validated in miniature (E7). Next: episode
   miner on the C4 index + graded target + more capacity.*
2. **Calibrated fusion**: isotonic sim→P(≤25km), product-of-experts with the gate posterior
   instead of λ=0.05. (The abstain half is dead; the calibration half is untested.)
3. ~~k-reciprocal / diffusion~~ ~~query expansion~~ — dead, see above.

## Tier 2 — one re-index run each (~$1–2, ~40 min)

4. **SALAD aggregation instead of CLS** (W4402660141, CVPR'24, Sinkhorn OT over DINO patch
   tokens; largest literature-endorsed descriptor jump; we already compute the patch tokens
   in `embed_c2.py` and throw them away). Train aggregation on the frozen C4 encoder, re-index.
   Family: BoQ, CriSALAD (W4410237700), EffoVPR (W4399151845 — zero-shot ViT-feature rerank,
   cheap pilot first).
5. **Panorama-aware reranker features** — each train location has 4 views; give the reranker
   sibling-view sims (learned version of the failed hand-crafted location-max-pool). Zero
   index cost.
6. **Database-side multi-scale entries** — one 0.7-zoom crop per train image as extra rows
   (asymmetric: adds recall paths; query-side TTA stays dead).

## Tier 3 — the big hammers (~$2–3)

7. **Patch-level geometric verification of the top-10** (Patch-NetVLAD lineage; learned
   homography W4392224206; model-free W4398249504; region mining W4409366443). Re-embed
   top-10 candidates' patch tokens on the fly (~2 min for full val), mutual-NN matches +
   spatial consistency, fuse into rank. True neighbours share scene GEOMETRY; 300-km
   lookalikes share only statistics. Eval-time only — no index rebuild.
8. **C5: fine-tune the encoder on its own confusions** — mine champion-recipe failures
   (top-1 >25km while a ≤25km candidate sat in the pool) as hard listwise episodes
   (CosPlace/EigenPlaces canon; graded supervision independently validated by W4386076349).
   Gate this on C4's result and Tier-1/2 plateau.

## Mean < 200 (and the road to <100) — new evidence channels

The S384 mean is floored at ~370 by ~170 queries whose true region appears in NO cached
channel. Ideas ranked by (evidence-add × cheapness):

M1. ~~**Bayes prior fallback**~~ **KILLED 2026-07-24 (E11/E12), with the decisive lesson:**
    (a) the dataset is world-spread — the train-density minimiser sits ~6100 km from the
    average train location, so "guess the prior" costs more than the tail it replaces;
    (b) **outcome-selected tail ≠ detector-selected tail**: the confidence signal finds
    queries E7 already handles decently (top_mass<0.2 set: mean 1079, 30% already ≤200km),
    while the true catastrophic misses are **confidently wrong** — their evidence points
    firmly at the wrong region, so no confidence threshold isolates them. This closes the
    ENTIRE fallback/hedging family (classifier ×2, prior-jump, prior-shrink, geomedian):
    post-hoc decision rules cannot move the mean on a fixed evidence bed. Mean progress =
    evidence progress (M2–M5, better encoder), full stop.
M2. **Geo-attribute constraint masks with FREE labels**: every train coordinate yields
    labels for driving side, hemisphere, climate zone (Köppen), coastal proximity,
    elevation band — no annotation needed. Fit linear probes on cached CLS descriptors;
    at inference each confident probe MASKS the cell posterior (left-driving detected →
    right-driving countries excluded → gate renormalises). Even a 90%-accurate binary
    attribute halves the feasible world for exactly the generic-scenery queries that kill
    the mean. Literature: OSV-5M (W4396821246) trains auxiliary heads (climate/etc.) but
    uses them as training signal — using them as INFERENCE-TIME hard constraints on a
    retrieval gate is our twist. All CPU on cached vectors once labels are derived
    (country/Köppen lookup table from coordinates).
M3. **OCR / script-ID channel**: signs and text pin the country (Cyrillic vs Latin-with-
    diacritics vs Thai…). Val is 3000 images (~150MB download); CPU OCR + script ID is
    hours-cheap. High precision where it fires; no coverage on empty roads. Fuse as another
    posterior mask.
M4. **Tail-routed descriptor channels**: for low-confidence queries only, re-rank by
    mean-patch and regional-chamfer sims (cached in the C4 index) instead of CLS — texture
    cues (vegetation, road surface, soil colour) fail differently from CLS. Overall blends
    were neutral, but the tail has nothing to lose. Local, cheap, C4 arrays.
M5. **Train–train hub discount**: images that are hubs of the train–train kNN graph are
    aliasing attractors; discount them index-wide. Needs one GPU pass (1.2M×1.2M chunked)
    — fold into the next box session.

Realistic path: M1 (+100ish mean-km on the S384 bed if the detector is precise) stacks
with the C4-bed evidence upgrade; <200 plausible with C4 + M1 + M2; <100 needs the
encoder loop (Tier-3 #8) plus M2/M3 landing well.

## Standing dead list (measured, do not retry)

Query-side TTA · single wide band head · 512px re-index · cross-model blend · α-QE ·
adaptive-λ by entropy · k-reciprocal · graph diffusion · SuperGlobal QE (query-side) ·
classifier fallback for the tail · wide re-gate for the tail · geometric-median snap ·
MLP reranker at 3k-query scale (capacity without data).
