# Findings — autoresearch sessions (image geolocalization, median_km ↓)

## Session 5 — 2026-07-22 (starts from champion 210.1 @ 6d8ca60; 3h; mandate: aux heads +
## better/geographic hierarchy first, then other stuff)

Plan: 1. country-level geographic hierarchy (hard-CE aux wired into the log-space combine via
majority-vote fine-cell→country parents), 2. subdivision level / deeper hierarchy, 3. IMG 448
+ PatchDropout 0.5 (post-saturation: buy per-step quality; scout: PatchDropout ablation says
half tokens ≈ free), 4. distance-weighted negatives on coarse CE (HierLoc-lite), 5. PATCH_KEEP
0.4/0.6 (one allowed follow-up). Scout notes: PIGEON bakes admin structure into CELL
CONSTRUCTION (semantic geocells), HierLoc gets 2x median cut from entity hierarchy; GeoRanker
pairwise ranking loss = denser early gradient (medium confidence).

Infra: offer-id rent is RACY (ids churn between search and create — 4 failed rents); fixed by
find+create in ONE process. Two dud hosts wasted ~35 min (Virginia proxy-only SSH unreachable;
Canada wedged 10 min in docker build) → race-rent 3 regional candidates in parallel, keep first
with working SSH (Estonia won in 2 min), destroy losers. Cache tar on HF was INCOMPLETE all
along (val only, no train!) — every "2-min tar setup" actually streamed; rebuilt + re-uploaded
correct 4.65GB tar from this box (TAR_UPLOAD_OK). Estonia box: 116 MB/s HF, 60k train in 409s.

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 33 | baseline re-run (6d8ca60 champion) | 212.2 | champion seed (=210.1 within noise; 1484 steps, 25.8GB) |
| 34 | country-level geographic hier: hard-CE (w .5) country head + log-softmax broadcast onto fine cells via majority-vote parents (115-country level above 64/512/2048) | 210.3 | −1.9 marginal BUT the right panel signature: mean 1079→1019, acc@2500 90.8→91.8, acc@750 +1.2pp, top5 47.1% best ever. Confirming |
| 34b | confirm re-run | 210.6 | **KEEP** (210.34/210.58 both beat 212.2; mean 998 best ever @ c557b72) |
| 35 | + subdivision level (1895-way hard CE, w .5) | 210.2 | discard (median flat, panel regressed: mean +40, top1 −1.3pp — country granularity is where semantic supervision stops paying) |
| 36a | IMG 448 + PatchDropout @ bs96 | OOM | crash at warmup (31.3GB, pre-clock, cheap) |
| 36b | IMG 448 + PatchDropout @ bs72 | 224.4 | discard (+14; 1410 steps ≈ matched — 384 FROZEN, 3rd independent confirmation: S2 win, S4 FixRes, this) |
| 37 | country targets tau-smoothed (300km over country spherical centroids) instead of hard CE | 209.5 | best of session; confirming |
| 37b | confirm re-run | 210.3 | **KEEP** (209.53/210.30 both beat 210.34/210.58 — champion ~209.9 @ 77b4082) |
| 38 | PIGEON retrieval refinement v1: embed 60k train post-clock, snap to sim-weighted top-8 NN within top-5 cells, blend .5 | 222.6 | discard (+12.7 median BUT acc@25 3.97% best ever — snap fixes close cases, wrecks typical ones at top1=18%. One retune: sim-gated cluster-centroid version) |
| 39 | semantic geocells (country-constrained k-means, largest-remainder allocation) | 216.6 | discard (+2.4 vs Korea anchor 214.2 — neutral-to-worse; border alignment isn't the lever). Ran after credit top-up on Korea box 45534538 |
| 40 | ANCHOR: champion re-run on Korea box | 214.2 | box reads +4 vs Estonia (209.5/210.3 there); all S5-afternoon comparisons vs 214.2. First country_acc: 44.6% |
| 41 | ViT-H+ 0.84B @ bs48 | 211.2 | −2.9; mean 876 / acc@2500 93.4% / geoguessr 3836 ALL BEST-EVER at 1279 steps. Josef verdict: drop (too large for app target); capacity → tail quality banked as knowledge; confirm aborted |
| 42 | H3-2122 merged cells AS FINE cells | 240.7 | discard (+26.5) BUT top1 25.7/top5 52.8/country 54.5 all records — quantization-floor lesson: balanced cells classify easier, guess coarser |
| 43 | H3-2122 as EXTRA hier level (evidence not geometry) | 222.7 | discard (+8.5) — big aux vocab steals gradient regardless of wiring (matches subdivision). Cell-scheme chapter CLOSED at 8-min |
| 44 | within-cell offset regression head (Josef's idea; shared per-cell tangent MLP, zero-init) | 219.9 | discard (+5.7) — within-cell position ≈ as hard as classification at 1500 steps; PARK for long budgets (needs converged selector; S1 exp4 was neutral at 396km for the same reason) |
| 45 | PATCH_KEEP 0.5→0.6 (post-saturation: richer tokens > extra steps) | 213.0 | **KEEP** (−1.1, ≥1km rule; 1295 steps) — champion @ 3f2c4ce. VRAM 31.0GB peak: knob FROZEN, no headroom |
| 46a | GeM pooling (cls+GeM concat) @ bs96 | OOM | crash at warmup — 31GB champion + GeM activations |
| 46b | GeM @ bs84 (memory-paired) | 212.1 | discard by 0.11km (−0.89 vs ≥1km rule); SSH-drop false-CRASH, score recovered from run.log. Panel BETTER (mean 950 best-ViT-L, acc2500 92.5, country 46.1) — ONE retune allowed |
| 47 | LoRA r32/a64 (capacity, never swept) | 214.8 | discard (+1.8; steps 1295→1158 — capacity costs steps, doesn't pay; r16 FROZEN) |

**Process rules (Josef, mid-session): no confirm re-runs; KEEP at ≥1 km improvement; report
every result in chat immediately. ViT-L locked as backbone (mobile-app target: ~600MB fp16 /
300MB int8 — ViT-H+ would be ~3x). EVAL_EVERY stays 250; no panorama tricks at eval (val is
one view per panorama anyway — checked panoid: 2998 unique).**

**Session pivot (Josef, mid-session): mandate changed to substantial structural swings, target
median ≤100 km eventually. Extended +2h. Honest read: 100 km needs harness-level changes (more
data / panoramas / longer budget) — PIGEON's 44 km used 500k imgs + 4-view panoramas + days of
training. In-harness moonshots ranked: retrieval refinement (their biggest ablation win),
semantic geocells, ViT-H+.**

## Champion (end of session 4) — see below for session-4 details

- **~210 km** @ 6d8ca60 (confirmed 209.61/210.11): session-3 stack + PatchDropout 0.5 + bs96×1.

## Session 4 — 2026-07-22 (starts from champion 221.5 @ ec1b843, 3h, mandate: BIG swings)

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
| 26 | baseline re-run (ec1b843 champion) | 221.5 | champion seed (=221.5 exactly; 1307 steps, 26GB) |
| 26b | PatchDropout v1 (walk core.layer) | CRASH | tf 5.14 renamed: encoder module is core.model, not .layer (3-min crash, pre-clock) |
| 27 | PatchDropout 0.5: train fwds keep 288/576 patch tokens (random per batch), RoPE cos/sin index-selected to match; eval full tokens; compile moved to encoder walk | 217.4 | −4.1, steps 1307→2748 (+110%), vram 26→13.8GB; cell_top1 DOWN 17.3→16.0 yet median better — throughput>classification again. Confirming |
| 27b | confirm re-run | 215.4 | **KEEP** (217.35/215.39 — champion ≈215.4 @ 608ab83, −2.8% vs 221.5) |
| 28 | geocells 2048→4096 + PRED_TOPK 24 (+ geocell disk cache, det. cells) | 223.8 | discard (+8.4; acc@25km UP 2.9→3.4 + first nonzero acc@1km, but 4096-way CE too hard at 2750 steps — top1 10.6%. Cell count DONE at 8-min; revisit only at long budgets. Cache plumbing kept) |
| 29 | FixRes: train 288px / eval 384px (324→162 kept tokens w/ dropout) | 219.2 | discard (+3.8; 4410 steps (+60%) yet worse — STEP-SCALING EXHAUSTED past ~2750 steps; res loss nets negative now. acc@25 up again) |
| 30 | LR ×2 (2e-4/2e-3) — steps doubled since LRs were tuned | 227.2 | discard (+11.8, clearly worse — LR sweep DONE, 1e-4/1e-3 frozen) |
| 31 | Muon (NS-orthogonalized momentum, lr .02) on 2D head/trunk matrices; AdamW keeps LoRA+biases | 241.4 | discard (+26 — lr .02 way too hot for this head/loss; retry only at ≤5e-3, low priority) |
| 32 | bs96×1 (PatchDropout freed VRAM; steps saturated → buy per-step quality via bigger real batch) | 209.6 | −5.8, top1 18.6% + acc@25 3.7% best ever, 1500 steps, vram 25.8GB. Confirming |
| 32b | confirm re-run | 210.1 | **KEEP** (209.61/210.11 — champion ≈210.1 @ 6d8ca60) |

## Champion (end of session 4)

- **~210 km** @ 6d8ca60 (confirmed 209.61/210.11): session-3 stack + **PatchDropout 0.5**
  + **bs96×1**. Panel: mean ~1048–1058 | acc@25 3.7–3.8% | acc@200 48.0–48.2% | acc@2500
  ~90.6% | geoguessr ~3740 | top1 18.2–18.6% | top5 45.8–45.9% | 1500 steps/8min | vram 25.8GB.
- Session arc: steps 1307→2750 (PatchDropout) proved step-scaling then SATURATED (FixRes's
  4410 steps lost); bs96 converted the same throughput into per-step quality instead. The
  parked-idea compounding chain: exp14 bs48 (parked) → compile un-parks it (S3) → PatchDropout
  VRAM dividend doubles it (S4).

## Session 3 results are below.

## Session 3 — 2026-07-21c (starts from champion 229.9 @ c7202d3, 4-hour session)

Plan (ranked from RESEARCH_LOG): 1. geo-contrastive hard negatives (semivariogram-weighted,
no special samplers), 2. top-K cell reranking via learned per-cell prototypes (Lindenberger),
3. EMA of trainables, 4. country-code aux head (HierLoc-lite), 5. GeM pooling, 6. deeper
hierarchy follow-up, 7. LoRA on fc1/fc2. Literature sweeps between experiments.

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 18 | baseline re-run (c7202d3 champion) | 232.4 | champion seed (=229.9 within noise); cell_top1 15.4%, top5 44.1% |
| 19 | geo-contrastive hard negatives (proj128, margin .3, far>1500km, λ=.1) | 238.7 | discard (+6.3; cell_top1 15.7 ticked UP but median down; −5% steps. Margin/λ may be off — one retune allowed later) |
| 20 | per-cell prototype cosine head (256d, temp .07, w .5) + log-space ensemble at pred | 239.7 | discard (+7.2; cell_top1 16.4% BEST + acc25 up, median down — fused sharper posterior hurts the spherical-mean geometry) |
| 21 | EMA of trainables (0.999, warmup-corrected), evals on EMA copy | 236.1 | discard (+3.7 noise; ~1000 steps too few for EMA — PARK for long runs) |
| 22 | surgical rerank: fused (cls+proto) picks/orders top-k cells, spherical-mean weights stay on original posterior | 236.3 | discard (+3.9 noise; top1 16.2/top5 44.9 up again, median flat — classifier-aux direction EXHAUSTED at 8-min budget) |
| 23 | torch.compile(backbone, dynamic) + pre-clock warmup (compile cost outside wall alarm) | 224.8 | WIN −7.6 (+14% steps → 1205; top1 17.4% best; all panel metrics up). Costs ~10 min compile per run (wall 24 min). Confirm below |
| 23b | confirm re-run of 23 | 224.8 | **KEEP** (224.80/224.85 — champion 224.8 @ 62e92d2) |
| 24 | bs48×1 (compile bought throughput back; parked exp14 said better per-step) | 221.5 | marginal −3.3 (1168 steps, vram 25.8GB, top5 46.3%), confirming |
| 24b | confirm re-run of 24 | 221.6 | **KEEP** (221.47/221.56 — champion 221.5 @ ec1b843; parked exp14 idea graduates thanks to compile) |
| 25 | LoRA also on gated MLP (up/gate/down_proj), r16 | OOM | crash (bs48 base is 26GB; MLP adapter activations blow 32GB — if ever retried, pair with bs24×2) |

## Champion (end of session 3)

- **221.5 km** @ ec1b843: prior stack + torch.compile(dynamic, pre-clock warmup) + bs48×1.
  Panel: mean ~1094–1124 | acc@200 46.7% | acc@2500 90.7–91.4% | geoguessr ~3722 |
  top1 16.4–17.7% | top5 45.8–46.3% | 1168 steps/8min | vram 25.5–26.4GB.

**Emerging pattern (exps 19+20):** ideas that improve cell_top1/top5 keep WORSENING median —
the bottleneck may now be the prediction geometry (how probs → coordinates), not classification.
If 21 doesn't move things, revisit prediction rule interaction instead of more classifier aux.

Box: Thailand 5090 @ $0.40/hr (handpicked — 3 cheapest CA offers all in the broken-network DC;
vast.py now sends SSH keepalives). Infra cost of bring-up: ~50 min + $0.30 across 2 dead boxes.

## Session 2 — 2026-07-21b (starts from champion 283.4 @ 359b895 + val-cell instrumentation 3bd2775)

Box: Thailand 5090 @ $0.38/hr, 20.4 MB/s HF (screened). Plan: 9. panorama InfoNCE aux (top pick),
hierarchical heads, tau up 110, IMG_SIZE 384, augmentation, TTA, EMA, GeM.

| # | idea | median_km | verdict |
|---|------|-----------|---------|
| 9 | baseline re-run (3bd2775, smoke-tests val-cell metrics) | 282.8 | champion seed (=283.4 within noise); cell_top1 14.0%, lift 288x |
| 10 | panorama InfoNCE aux (pair sampler, λ=0.5, T=0.1, proj 256) | 306.1 | discard (+23 WORSE; cell_top1 14.0→11.9 — pair batches halve per-batch location diversity at 8-min budget) |
| 11 | hierarchical heads 64/512/2048, log-space combine, CE w=0.25/0.5/1.0 | 263.9 | **KEEP** (−18.9; mean 1399→1219, acc@2500 87.7→89.9% — fixes wrong-region mass) |
| 12 | tau 75→110 (on hier champion) | 271.0 | discard (+7.1; tau sweep DONE — 75 frozen) |
| 13 | IMG_SIZE 448→384 (+35% steps at same budget) | 238.0 | **KEEP** (−25.9! 961 steps, cell_top1 15.2%, vram 13.7GB) |
| 14 | bs48×1 @384 (same eff. batch, single fused fwd) | 246.9 | discard (+8.9; fused pass SLOWER, 92 vs 106 img/s — bs24×2 frozen) |
| 15 | geo-safe augmentation (RRC 0.5–1.0 + jitter 0.15, no flips) | 239.3 | discard (+1.3 noise-neutral at 0.5 epochs — parked for long runs) |
| 16 | TTA at final eval (3 center crops, avg sharpened probs) | 247.3 | discard (+9.3; crop-averaging blurs the mode — same run's single-view quick-val was 223 @ step 900. exp parser showed CRASH but score was in run.log/W&B — SSH tail drop) |
| 17 | EVAL_EVERY 100→250 (reclaim ~70s of wall-alarm window for training) | 230.9 | marginal (−7.1), confirm below |
| 17b | confirm re-run of 17 | 229.9 | **KEEP** (held twice → champion 229.9 @ c7202d3) |

**Found:** SIGALRM cap is WALL clock (480+45s); 9 quick evals/run eat ~90s of it → training
cut at ~435s not 480. TTA therefore final-eval-only (alarm disarmed there). Next: EVAL_EVERY
100→250 to reclaim ~70s training (+15% steps).


## Champion

- **229.9 km** @ c7202d3: hier heads + IMG 384 + EVAL_EVERY 250 (confirmed 230.9/229.9).
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
