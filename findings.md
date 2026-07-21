# Findings — autoresearch/2026-07-21 (image geolocalization, median_km ↓)

## Champion

- **433.3 km** @ 9890292: champion = baseline + no grad-ckpt (bs24x2 accum, 32 workers).
  661 steps = 0.53 epochs, 17.8GB VRAM. Panel: mean 1292 | acc@25 0.9% | acc@200 25.9% |
  acc@2500 88.6% | geoguessr 3249.

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
| 5 | mode-seeking pred rule (T=0.5 + 1000km locality) | running | |

## Banked

- —

## Dead ends

- —
