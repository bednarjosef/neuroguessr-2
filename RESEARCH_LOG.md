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

- **median_km:** 535.4
- **train.py commit:** 7d22c37 (branch autoresearch/2026-07-21)
- **one-line:** stock baseline: DINOv3-L LoRA r16 (q/k/v/o proj), 448px, 512 k-means cells, tau 75km, bs48
- **full metric panel (last full-val eval):** mean_km 1473 · acc@25km 0.4% · acc@2500km 86.7% · geoguessr 3070

## Banked wins (confirmed to help — keep these, don't re-litigate)

_(each entry: the change, the median_km delta, and why it likely helped)_

- —

## Dead ends & mistakes (tried, did NOT help or broke — do NOT repeat)

_(each entry: what was tried, what happened, and the takeaway so it isn't retried blind)_

- —

## Open ideas / next to try (ranked)

_(carry unfinished/promising directions forward across sessions)_

From literature scout 2026-07-21 (PIGEON CVPR'24, OSV-5M CVPR'24, GeoCLIP NeurIPS'23):

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
- **Speedup TODO:** tar `~/.cache/autoresearch_geo` (~4 GB) and push once to HF; future setups
  become a single 2-min download.
- transformers 5.14 DINOv3: LoRA targets are `q_proj/k_proj/v_proj/o_proj`.
- RTX 5090 32GB: baseline uses only 6.9 GB with checkpointing on; throughput 46 img/s → 0.36
  epochs per 8-min budget. Compute/throughput, not VRAM, is the binding constraint.

---

## Session history

_(one dated block per session: dates, champion at start → end, headline results)_

<!-- template:
### 2026-07-21
- Champion at start: <median_km or "baseline"> → at end: <median_km>
- Experiments run: <N>, kept: <k>
- Headline: <what moved the metric>
- Notes / follow-ups for next session: <...>
-->
