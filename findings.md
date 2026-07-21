# Findings — autoresearch sessions (image geolocalization, median_km ↓)

## Session 2 — 2026-07-21b (starts from champion 283.4 @ ae5e29f + val-cell instrumentation 337e4a7)

Box: Thailand 5090 @ $0.38/hr, 20.4 MB/s HF (screened). Plan: 9. panorama InfoNCE aux (top pick),
hierarchical heads, tau up 110, IMG_SIZE 384, augmentation, TTA, EMA, GeM.

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 9 | baseline re-run (337e4a7, smoke-tests val-cell metrics) | 282.8 | champion seed (=283.4 within noise); cell_top1 14.0%, lift 288x |
| 10 | panorama InfoNCE aux (pair sampler, λ=0.5, T=0.1, proj 256) | 306.1 | discard (+23 WORSE; cell_top1 14.0→11.9 — pair batches halve per-batch location diversity at 8-min budget) |
| 11 | hierarchical heads 64/512/2048, log-space combine, CE w=0.25/0.5/1.0 | 263.9 | **KEEP** (−18.9; mean 1399→1219, acc@2500 87.7→89.9% — fixes wrong-region mass) |
| 12 | tau 75→110 (on hier champion) | 271.0 | discard (+7.1; tau sweep DONE — 75 frozen) |
| 13 | IMG_SIZE 448→384 (+35% steps at same budget) | 238.0 | **KEEP** (−25.9! 961 steps, cell_top1 15.2%, vram 13.7GB) |
| 14 | bs48×1 @384 (same eff. batch, single fused fwd) | | running |


## Champion

- **263.9 km** @ 7acb12a: prev champion + hierarchical heads 64/512/2048 (log-space combine,
  coarse CE w 0.25/0.5). Panel: mean 1219 | acc@200 41.5% | acc@2500 89.9% | geoguessr 3596.

## Notes

- Throughput is the binding constraint (0.36 epochs/budget). VRAM massively underused.
- transformers 5.x DINOv3 module names are q_proj/k_proj/v_proj/o_proj (not query/key/value/dense).

## Tried

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 0 | baseline (fixed LoRA targets) | 535.4 | KEEP (champion seed) |
| 1 | no-ckpt + 32 workers @ bs48 | OOM | crash (activations >32GB at bs48) |
| 1b | no-ckpt, bs24x2, 32 workers | 433.3 | **KEEP** (−19%, +47% steps) |
| 2 | geocells 512→2048, topk 16 | 398.5 | **KEEP** (−8%) |
| 3 | cls_mean pooling | 395.5 | discard (within ±3–5km noise) |
| 4 | per-cell offset regression head | 395.7 | discard (within noise → within-cell res NOT the bottleneck; cell selection is) |
| 5 | mode-seeking pred rule (T=0.5 + 1000km locality) | 283.4 | **KEEP** (−29%!) |
| 6 | unfreeze last 2 blocks (+grid diag) | 278.3 | discard (−5.1, below 8km noise bar; grid: T0.35/r1000 best on quick-val, no-locality=332 terrible) |
| 7 | unfreeze + T=0.35 combo | 280.6 | discard (within noise again) |
| 8 | tau 75→40 (sharper labels @ 2048 cells) | 316.4 | discard (clearly worse — smoothing at 75 is load-bearing) |

## Banked

- —

## Dead ends

- —
