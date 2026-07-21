---
description: Run a full autonomous autoresearch session (rent GPU → baseline → ratchet loop until the deadline)
argument-hint: "[optional overrides/notes, e.g. 'minutes 10, hours 4' or 'focus on the head']"
---

You are the single research agent described in `ENGINE.md`. Run a **complete, autonomous
autoresearch session** for this repo and **DO NOT STOP** until the session deadline, an
unrecoverable error, or I interrupt you. Invoking this command is my explicit authorization
to **rent a GPU and spend money** for the configured session length.

Follow this exactly:

1. **Read the mission and the engine.** Read `program.md` (what to optimize + config) and
   `ENGINE.md` (the fixed loop). Obey `ENGINE.md` to the letter. The two non-negotiables:
   **experiments are synchronous — run `python vast.py exp` in the FOREGROUND, never poll,
   background, or build a wait-loop**; and **during the loop only `train.py` may change** —
   never edit `prepare.py`, `evaluate_geo`, the `predict_latlon` contract, or `ENGINE.md`.

2. **Bring up the box** (one command — rent + watchdog + setup):
   ```
   python vast.py start --metric median_km --goal min --hours 3 --minutes 8 --gpu RTX_5090 --max-price 0.90
   ```
   If I passed overrides in the arguments below, apply them (e.g. different `--minutes`,
   `--hours`, `--gpu`, `--max-price`). Setup downloads the frozen data subset and forwards
   `HF_TOKEN` (for gated DINOv3) and the W&B key from `.env`/netrc automatically.

3. **Seed the champion.** Create the champion branch off master
   (`git checkout -b autoresearch/<today>`), init `results.tsv` + `findings.md`, then run the
   **baseline once**: `python vast.py exp --train train.py`. Confirm it prints `median_km` and
   fits VRAM comfortably. If it OOMs or crashes, fix `train.py` minimally (e.g. lower
   `DEVICE_BATCH_SIZE`) until it runs clean — that becomes the champion. Each experiment is
   also logged to the W&B project `neuroguessr-2-research`.

4. **Run the ratchet loop.** Repeat until the deadline: pick **one** idea (rotate the search
   directions in `program.md` §4, **swing for big wins** over tiny tweaks), edit `train.py`,
   run `python vast.py exp --train train.py` in the foreground, log the result with
   `python vast.py log`, then **KEEP** if `median_km` improved vs the champion (commit on the
   branch) or **RESET** otherwise (`git checkout -- train.py`). Update `findings.md` every
   iteration. Between experiments, use the `research` skill for literature-grounded ideas.

5. **NEVER STOP.** When one experiment finishes, immediately pick the next and run it — keep
   the GPU busy every minute until the session's hard deadline auto-destroys the box. Before
   each run, glance at `python vast.py status`; if under ~9 minutes remain, wind down.

Give me a brief status line when the box is up and after the baseline, then run the loop.
Report the champion's score and the best ideas as you go.

Additional instructions / overrides for this run: $ARGUMENTS
