# C4 — joint classifier + retrieval fine-tune

**Target: median < 30 km, @25 km ≈ 55%.** Starting point (24 Jul 2026): **38.74 km / 45.13%**.

Everything below is justified by a measurement from C1–C3, not by intuition. Measured dead ends
are listed at the end so they are never retried.

---

## 1 · What the gap actually is

| | @25 km |
|---|---|
| a training image within 25 km exists | 100 % |
| …and is inside the gated candidate pool | 92–98 % |
| …and our matcher picks it | **45.1 %** |

The gate and the data are not the constraint. ~50 points sit in **matching quality**, and C3
proved that moving the encoder moves it: the raw descriptor went **36.3 % → 42.5 % (+6.1)** with
only 0.53 epochs of attention-only LoRA. C4 is the same lever, pulled properly.

## 2 · The five changes

### 2.1 Graded-distance contrastive loss *(the fix for C3's regression)*

C3's loss was **binary** — positive under 10 km, masked 10–50 km, full negative beyond 50 km —
and because every batch came from one ~500 km region, **50–500 km was nearly the only negative it
ever saw**. It was trained to treat "80 km away" exactly like "another continent". Result: @1 km
improved (11.3 → 12.4 %) while the 25–200 km band degraded, which is precisely what showed up in
the C3 champion (median 43.8 → 47.6).

Replace the hard split with soft targets that decay with distance — the same trick the classifier
already uses (softmax(−d/τ), τ = 75 km):

```
target_ij  ∝ exp(-d_ij / TAU_GEO)      # TAU_GEO ≈ 100 km, self excluded
loss       = cross_entropy(sims_ij / TAU_TEMP, target_ij)   # soft-target InfoNCE
```

No positives, no negatives, no masking — one continuous objective that preserves ordering across
every distance band. **Pilot TAU_GEO ∈ {75, 150} km before committing.**

### 2.2 Joint training with the classifier *(fixes the two-pass problem)*

Measured: gating with C3's own logits scores **42.93 %** vs **44.20 %** with the old checkpoint's
logits — the contrastive fine-tune damaged the classifier riding on the same backbone. So today
inference needs two encoder passes (~2.4 s CPU).

```
loss = CE_cells(existing recipe: hierarchy + country + MSL) + LAMBDA_C * soft_contrastive
```

The CE term also acts as an anchor that stops the encoder from drifting away from coarse
geography — the likely second cause of the 25–200 km regression. **Pilot LAMBDA_C ∈ {0.3, 0.7,
1.5}.** One model, one forward pass, both jobs.

### 2.3 Mixed batches

Contrastive wants region-restricted batches (hard negatives); CE wants i.i.d. samples. Use
**half the batch from one ~500 km region, half global**. Region batches alone were what starved
C3 of long-range negatives.

### 2.4 LoRA on the MLP blocks, rank 32

| | params in ViT-L | adapted in C3 | adapted in C4 |
|---|---|---|---|
| attention q,k,v,o | ~100 M | r16 → 3.15 M | r32 → 6.3 M |
| **MLP fc1/fc2** | **~201 M** | **nothing** | **r32 → 7.9 M** |

We have been adapting the third of the network that *routes* information and ignoring the
two-thirds that *stores* it. Feature content — what distinguishes one Finnish road from the next —
lives in the FFN blocks. Cost is ~14 M trainable (still 4.7 % of the backbone), negligible extra
compute, free at inference after merging.

### 2.5 Four times the training

C3 saw 0.53 epochs. C4 runs **2 full epochs** (~2.4 M image views). With the speed-ups below this
is ~90 min on 4×5090.

## 3 · Speed-ups (measured or standard, all safe)

| change | effect |
|---|---|
| LoRA only in blocks 9–24, bottom 8 frozen | backward stops early, −30 % step time |
| patch dropout 0.25 during training (full patches at eval — precedent: the classifier trained at 0.6) | −20 % |
| `torch.compile` on the backbone | −15 % |
| prebuilt **384 px** dataset cache tar on HF (~45 GB vs 115 GB raw) | −25 min per session |

Target throughput ≈ 450 img/s vs C3's 260.

## 4 · Post-training pipeline (all known-good, do not improvise)

1. **Re-index** 1.2 M images at 384 px (~22 min).
2. **Train the band heads OFFLINE** — d_pos 5 / 10 / 25 / 50 km, in parallel, ~15 min total.
   Measured: the head trained *online inside* C3 scored 33.9 % vs 44.2 % for the offline head on
   the same descriptors. Always fit heads offline afterwards, on cached vectors, big batches.
3. **Grid**: 4-band blend + **CSLS** + whitening + regional rerank (R50, α0.25), gate now from the
   same model. CSLS alone was worth +0.8 pts / −2.2 km tonight, for 40 s of arithmetic.

## 5 · Instrumentation — non-negotiable

C3 ran blind: its loss swung 1.6–2.9 on region difficulty and forecast nothing, and its 6
checkpoints could not be ranked without a 22-minute re-index each.

- **every 250 steps**: near-recall@1 within 10 km on a cached 5 000-location held-out set, query's
  own location excluded (the probe already in `train_geo_head.py`, ~2 s)
- **every 1000 steps**: classifier median on the quick val subset (catches CE/contrastive imbalance)
- **kill criterion**: if the probe has not beaten C3's raw baseline (1.80 %) by step 800, stop the
  run and re-pilot the loss weights. Saves ~$3 on a bad configuration.

## 6 · Schedule and cost

| phase | time | $ |
|---|---|---|
| rent + setup + data (with cache tar) | 15 min | 0.5 |
| pilot: 3 × LAMBDA_C, 2 × TAU_GEO, 200 steps each, ranked by the probe | 20 min | 0.6 |
| main run, 2 epochs | 90 min | 2.8 |
| re-index | 22 min | 0.7 |
| band heads ×4 (parallel) + grids | 30 min | 0.9 |
| **total** | **~3 h** | **≈ $5.5** (+$1.5 contingency) |

## 7 · Honest expectation

C3 bought +6.1 pts on the raw descriptor with a quarter of the training, a third of the adapter
capacity, and a loss that actively damaged the 25–200 km band. C4 fixes all three.

- raw descriptor: **47–52 %** @25 km
- after band heads + CSLS + rerank: **@25 km 48–54 %**, **median 28–34 km**

So **median < 30 km is likely; @25 km = 55 % is at the optimistic edge** of the range. If C4 lands
at ~50 % the next lever is no longer the encoder — it is index density (more exemplars per
location) and a second retrieval round (re-query using the matched image's neighbours).

## 8 · Explicitly NOT in C4 (measured dead ends)

512 px re-index (+0.33 pts for 1.8× compute) · query-side TTA (−0.4) · multi-resolution
same-model ensemble (−0.9) · cross-model blend with C2 descriptors (43.8 vs 44.2) · gate-mass
sweep (identical at 0.90/0.95/0.99) · adaptive λ by posterior entropy (−0.9) · confidence router
to the classifier readout (−0.4 @25, though it does help mean/@200) · location max-pool and
top-2-view evidence (−0.3) · α-query-expansion (−0.5) · post-trunk features instead of backbone
CLS (worse everywhere) · a single wider-band head instead of a multi-band blend (25 km head alone
did not beat the champion; the 4-band blend did).
