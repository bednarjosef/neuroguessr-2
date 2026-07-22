# Findings — autoresearch sessions (image geolocalization, median_km ↓)

## Session 4 — 2026-07-22 (starts from champion 221.5 @ ae5c260, 3h, mandate: BIG swings)

Plan (scout-refreshed): 1. PatchDropout 50% (FLIP/PatchDropout — drop patch tokens in train
fwd only, index-select RoPE cos/sin to match; ~2x step throughput), 2. finer geocells
2048→4096 (+topk; we're quantization-limited: ~250km mean cell radius vs 221km median),
3. FixRes train 256–288 / eval 384, 4. Muon on the head, 5. ViT-B compute-matched swap,
6. LR ×2 for bs48. Scout flagged ToMe as NOT feasible in HF DINOv3 (RoPE surgery) — skip.
Infra: Austria box (43122662); first setup SSH-hung 50 min (silent), then HF CDN CLOSE-WAIT
stall in prepare.py — killed + nohup rerun with live log fixed it. TODO for vast.py: setup
heartbeat + direct-nohup mode.

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 26 | baseline re-run (ae5c260 champion) | 221.5 | champion seed (=221.5 exactly; 1307 steps, 26GB) |
| 26b | PatchDropout v1 (walk core.layer) | CRASH | tf 5.14 renamed: encoder module is core.model, not .layer (3-min crash, pre-clock) |
| 27 | PatchDropout 0.5: train fwds keep 288/576 patch tokens (random per batch), RoPE cos/sin index-selected to match; eval full tokens; compile moved to encoder walk | 217.4 | −4.1, steps 1307→2748 (+110%), vram 26→13.8GB; cell_top1 DOWN 17.3→16.0 yet median better — throughput>classification again. Confirming |
| 27b | confirm re-run | 215.4 | **KEEP** (217.35/215.39 — champion ≈215.4 @ 948fe2e, −2.8% vs 221.5) |
| 28 | geocells 2048→4096 + PRED_TOPK 24 (+ geocell disk cache, det. cells) | 223.8 | discard (+8.4; acc@25km UP 2.9→3.4 + first nonzero acc@1km, but 4096-way CE too hard at 2750 steps — top1 10.6%. Cell count DONE at 8-min; revisit only at long budgets. Cache plumbing kept) |
| 29 | FixRes: train 288px / eval 384px (324→162 kept tokens w/ dropout) | 219.2 | discard (+3.8; 4410 steps (+60%) yet worse — STEP-SCALING EXHAUSTED past ~2750 steps; res loss nets negative now. acc@25 up again) |
| 30 | LR ×2 (2e-4/2e-3) — steps doubled since LRs were tuned | 227.2 | discard (+11.8, clearly worse — LR sweep DONE, 1e-4/1e-3 frozen) |
| 31 | Muon (NS-orthogonalized momentum, lr .02) on 2D head/trunk matrices; AdamW keeps LoRA+biases | 241.4 | discard (+26 — lr .02 way too hot for this head/loss; retry only at ≤5e-3, low priority) |
| 32 | bs96×1 (PatchDropout freed VRAM; steps saturated → buy per-step quality via bigger real batch) | running | LR kept 1e-4/1e-3 (LR×2 already ruled out) |

## Session 3 results are below.

## Session 3 — 2026-07-21c (starts from champion 229.9 @ 32075c7, 4-hour session)

Plan (ranked from RESEARCH_LOG): 1. geo-contrastive hard negatives (semivariogram-weighted,
no special samplers), 2. top-K cell reranking via learned per-cell prototypes (Lindenberger),
3. EMA of trainables, 4. country-code aux head (HierLoc-lite), 5. GeM pooling, 6. deeper
hierarchy follow-up, 7. LoRA on fc1/fc2. Literature sweeps between experiments.

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 18 | baseline re-run (32075c7 champion) | 232.4 | champion seed (=229.9 within noise); cell_top1 15.4%, top5 44.1% |
| 19 | geo-contrastive hard negatives (proj128, margin .3, far>1500km, λ=.1) | 238.7 | discard (+6.3; cell_top1 15.7 ticked UP but median down; −5% steps. Margin/λ may be off — one retune allowed later) |
| 20 | per-cell prototype cosine head (256d, temp .07, w .5) + log-space ensemble at pred | 239.7 | discard (+7.2; cell_top1 16.4% BEST + acc25 up, median down — fused sharper posterior hurts the spherical-mean geometry) |
| 21 | EMA of trainables (0.999, warmup-corrected), evals on EMA copy | 236.1 | discard (+3.7 noise; ~1000 steps too few for EMA — PARK for long runs) |
| 22 | surgical rerank: fused (cls+proto) picks/orders top-k cells, spherical-mean weights stay on original posterior | 236.3 | discard (+3.9 noise; top1 16.2/top5 44.9 up again, median flat — classifier-aux direction EXHAUSTED at 8-min budget) |
| 23 | torch.compile(backbone, dynamic) + pre-clock warmup (compile cost outside wall alarm) | 224.8 | WIN −7.6 (+14% steps → 1205; top1 17.4% best; all panel metrics up). Costs ~10 min compile per run (wall 24 min). Confirm below |
| 23b | confirm re-run of 23 | 224.8 | **KEEP** (224.80/224.85 — champion 224.8 @ 804fe7f) |
| 24 | bs48×1 (compile bought throughput back; parked exp14 said better per-step) | 221.5 | marginal −3.3 (1168 steps, vram 25.8GB, top5 46.3%), confirming |
| 24b | confirm re-run of 24 | 221.6 | **KEEP** (221.47/221.56 — champion 221.5 @ ae5c260; parked exp14 idea graduates thanks to compile) |
| 25 | LoRA also on gated MLP (up/gate/down_proj), r16 | OOM | crash (bs48 base is 26GB; MLP adapter activations blow 32GB — if ever retried, pair with bs24×2) |

## Champion (end of session 3)

- **221.5 km** @ ae5c260: prior stack + torch.compile(dynamic, pre-clock warmup) + bs48×1.
  Panel: mean ~1094–1124 | acc@200 46.7% | acc@2500 90.7–91.4% | geoguessr ~3722 |
  top1 16.4–17.7% | top5 45.8–46.3% | 1168 steps/8min | vram 25.5–26.4GB.

**Emerging pattern (exps 19+20):** ideas that improve cell_top1/top5 keep WORSENING median —
the bottleneck may now be the prediction geometry (how probs → coordinates), not classification.
If 21 doesn't move things, revisit prediction rule interaction instead of more classifier aux.

Box: Thailand 5090 @ $0.40/hr (handpicked — 3 cheapest CA offers all in the broken-network DC;
vast.py now sends SSH keepalives). Infra cost of bring-up: ~50 min + $0.30 across 2 dead boxes.

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
| 14 | bs48×1 @384 (same eff. batch, single fused fwd) | 246.9 | discard (+8.9; fused pass SLOWER, 92 vs 106 img/s — bs24×2 frozen) |
| 15 | geo-safe augmentation (RRC 0.5–1.0 + jitter 0.15, no flips) | 239.3 | discard (+1.3 noise-neutral at 0.5 epochs — parked for long runs) |
| 16 | TTA at final eval (3 center crops, avg sharpened probs) | 247.3 | discard (+9.3; crop-averaging blurs the mode — same run's single-view quick-val was 223 @ step 900. exp parser showed CRASH but score was in run.log/W&B — SSH tail drop) |
| 17 | EVAL_EVERY 100→250 (reclaim ~70s of wall-alarm window for training) | 230.9 | marginal (−7.1), confirm below |
| 17b | confirm re-run of 17 | 229.9 | **KEEP** (held twice → champion 229.9 @ 32075c7) |

**Found:** SIGALRM cap is WALL clock (480+45s); 9 quick evals/run eat ~90s of it → training
cut at ~435s not 480. TTA therefore final-eval-only (alarm disarmed there). Next: EVAL_EVERY
100→250 to reclaim ~70s training (+15% steps).


## Champion

- **229.9 km** @ 32075c7: hier heads + IMG 384 + EVAL_EVERY 250 (confirmed 230.9/229.9).
  Panel: mean 1111 | acc@200 45.6% | acc@2500 91.0% | geoguessr 3693 | top1 ~15–16% top5 ~44%.

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
