# Findings — autoresearch/2026-07-21 (image geolocalization, median_km ↓)

## Champion

- **535.4 km** — baseline @ 7d22c37: DINOv3-L LoRA r16 (q/k/v/o), 448px, 512 k-means cells,
  tau 75km, bs48, 8-min budget. 449 steps = 0.36 epochs, 46 img/s, 6.9GB VRAM.
  Panel: mean 1473 | acc@25 0.4% | acc@200 20.8% | acc@2500 86.7% | geoguessr 3070.

## Notes

- Throughput is the binding constraint (0.36 epochs/budget). VRAM massively underused.
- transformers 5.x DINOv3 module names are q_proj/k_proj/v_proj/o_proj (not query/key/value/dense).

## Tried

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 0 | baseline (fixed LoRA targets) | 535.4 | KEEP (champion seed) |
| 1 | unbrake: GRAD_CHECKPOINT off + 32 workers | running | |

## Banked

- —

## Dead ends

- —
