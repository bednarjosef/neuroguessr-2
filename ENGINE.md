<!-- ============================================================================
     ENGINE.md — the fixed autoresearch engine. DO NOT EDIT.

     This file is the GENERAL machinery: how the agent runs a session, regardless
     of WHAT is being researched. The "what / how to optimize" lives in program.md
     (the mission), which IS meant to be edited. The agent and the human edit
     program.md and train.py — never this file.
     ============================================================================ -->

# The autoresearch engine (fixed — do not edit)

`program.md` says **what** to optimize and the directions to explore. **This file says how the
machine runs.** Read `program.md` first (the mission + config), then run the loop below.

This is a **single agent** running a **single experiment at a time** — a keep/reset ratchet, in
the spirit of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). There is no
parallelism of experiments, no swarm, no per-experiment subagent. You think of one idea, edit
`train.py`, run it once on the rented box, and either **keep** it (it beat the champion) or
**reset** it (it didn't). The champion only ever moves to a strictly better score. That's the
whole engine.

The one twist versus the original: the experiment runs on a **rented Vast.ai box** instead of a
local GPU, so you can drive the loop from a laptop with no GPU. The control plane is one file:
**`vast.py`** (`python vast.py --help`). It rents/destroys the box, prepares it, and runs one
experiment (`exp`), blocking until it finishes and printing the objective.

**One experiment, but it may use more than one GPU.** By default the box has one GPU. If a model
is too big for one, rent an N-GPU box with `vast.py start --gpus N`: the **single** experiment is
then sharded across all N GPUs with **torch.distributed** (`exp` launches it under `torchrun`).
This is *not* N experiments in parallel — it's one model/one run using N GPUs. A multi-GPU
experiment's `train.py` must be **distributed-aware** (init the process group, shard the model,
print the objective on **rank 0** only); the default LLM `train.py` is single-GPU, so keep
`--gpus 1` unless you've made `train.py` distributed.

> **This engine is domain-agnostic.** The examples below use the default LLM-pretraining
> vocabulary; read these terms as their general roles, whatever `program.md` actually configures:
> **`train.py`** = the *experiment* (the artifact you edit), **`prepare.py`** = the *frozen
> harness/evaluator*, **`val_bpb`** = the *objective* the experiment prints. "Lower `val_bpb` is
> better" is the default direction; if `program.md`/`vast.py start` set **`--goal max`**, flip
> every "lower/improved" comparison to "higher". Everything else — the ratchet, the frozen
> harness, the safety model — is identical for any research target.

---

## The two rules that are non-negotiable

**You (the agent) run the whole loop yourself.** You think, edit `train.py`, run one experiment,
record the result, keep or reset, and repeat — building the single champion.

1. **EXPERIMENTS ARE SYNCHRONOUS — NEVER POLL OR BUILD A WAIT-LOOP.** `python vast.py exp` blocks
   until the run finishes (the full training budget + compile/eval) and then **prints the
   result** (`val_bpb` / `RESULT_JSON`). Call it in the **foreground** and let it return. **Do
   NOT** background it, redirect it to a log and poll the log, set up a "wait loop", or repeatedly
   read files waiting for it to finish. One foreground `exp` call returns a finished experiment
   with its objective in the output. (Polling is the #1 way autonomous loops get stuck — don't.)

2. **DURING RESEARCH, ONLY `train.py` MAY CHANGE — NO CHEATING.** Once the loop is running, you
   may not edit `prepare.py`, `evaluate_bpb`, the `forward`→logits contract, or `ENGINE.md`. Every
   gain must come from `train.py` alone, so the score can't be gamed. This is enforced
   structurally: `vast.py exp` uploads **only** `train.py` to the box, so the box's `prepare.py`
   and metric — frozen when you ran `start`/`setup` — score every run no matter what you edit
   locally. (The human or their agent MAY edit `prepare.py` to set the regime **before** launching
   research; during the loop it stays frozen.)

---

## Session bring-up

Do this once, when the human starts a session:

1. **Pick a per-experiment budget and session length** (from `program.md`'s config, or ask).
2. **Bring up the box in ONE command** — rent → watchdog → setup:
   ```
   python vast.py start --hours <H> --minutes <M> --max-price 0.60          # 1 GPU (default)
   python vast.py start --gpus 4 --hours <H> --minutes <M> --max-price 0.60 # 4 GPUs, one sharded experiment
   ```
   `start` rents the cheapest qualifying box (`--gpus N` for a multi-GPU box; `--max-price` is
   per GPU/hr), launches the deadline **watchdog in the background** (the box can never outlive
   `--hours`), and prepares it (template torch + light deps — no torch download — plus
   data/tokenizer). Use `--gpus N` **only** when the model needs to shard across GPUs and
   `train.py` is distributed-aware; otherwise leave it at 1.
3. **Create the champion branch** off master: `git checkout -b autoresearch/<tag>` (`<tag>` from
   today's date). You work in the normal working tree — no worktrees, no slots.
4. **Init shared state** (both untracked/gitignored): `results.tsv` (auto-created by `vast.py
   log`; the append-only ledger you write one row per experiment to) and `findings.md` (seed
   sections **Champion**, **Tried**, **Dead ends**).
5. **Establish the baseline + confirm it fits VRAM.** Run the unmodified `train.py` ONCE to seed
   the champion: `python vast.py exp --train train.py`.
   - Returns a `val_bpb` and `peak_vram` comfortably under the GPU (e.g. < ~22 GB on a 24 GB
     4090) → log `keep`, commit `train.py` on the champion branch, record it in `findings.md`,
     begin the loop. (This first run also warms the compile cache, so later runs start faster.)
   - **OOM** → baseline too big. Lower `DEVICE_BATCH_SIZE` (then `DEPTH`/`n_embd`) until it fits
     with headroom; make THAT the champion baseline before the loop.
6. **(Optional) Offer the dashboard**: `python vast.py dashboard` (localhost, auto-refresh).

---

## The ratchet loop

One experiment is the unit: **one idea → run it → keep or reset.** The champion branch only ever
moves to a strictly better score. Repeat — forever, the champion improving by building UPON
itself.

Each iteration, in order:

1. **Pick the next idea.** The champion is the best `train.py` so far (the tip of your branch).
   Choose **one** concrete change to try on top of it — drawn from the search directions in
   `program.md` §4 and from your own reasoning + the literature (below). Change only what the
   idea needs; hold everything else at the champion's values so the delta is attributable.

   > **SWING FOR BIG WINS — bold and novel over tiny and safe.** Spend most experiments on ideas
   > with **large upside**: new mechanisms, **structural/architectural** changes, fundamentally
   > different approaches — the things that can move the metric *a lot*. Micro-tuning (nudging a
   > hyperparameter, ±0.001 chasing) is **secondary**: do it sparingly. A bold idea that fails is
   > fine and expected; a session of timid tweaks is the real failure mode.

   > **ROTATE the directions; don't tunnel.** `program.md` §4 lists families of ideas (optimizer,
   > attention, MLP, capacity, embeddings, …). Move across them so the search stays broad. The
   > moment an idea is confirmed and banked into the champion, it is **DONE** — stop re-tuning it
   > and go find the NEXT new win. At most ONE follow-up may fine-tune a banked knob (a single
   > sweep), after which its value is frozen. Chasing the "ideal" value of an already-good
   > hyperparameter is diminishing-returns busywork.

   > **RESEARCH, don't just recall.** Ground your ideas in active research + reasoning, not
   > memory. Use the **`research` skill** (scholarly + web) to find **current SOTA**, genuinely
   > **novel ideas**, and evidence on what's known to work or fail — so you don't burn a run
   > rediscovering it. **Triage abstracts first** (`--abstracts --json`) to scan many cheaply,
   > then fetch full text for the most promising. Do this **between experiments**, when you're
   > unblocked. *(Optional: because a single `exp` blocks you for the whole training budget, you
   > may spin up one research-scout subagent in the same message as the `exp` call to mine the
   > literature during that window — but it's just a convenience, not required. The loop is
   > single-agent by default.)*

2. **Edit `train.py`** with that one idea. Keep the time-budget + eval scaffolding intact (the
   `start_training_clock()` call, the `try … except TrainingTimeUp` around the loop, and the
   final `evaluate_bpb` + `val_bpb:` print) — they hard-bound the run and guarantee a graceful
   final score.

3. **Run it once, FOREGROUND:** `python vast.py exp --train train.py`. It blocks until the run
   finishes and prints the objective + `RESULT_JSON` (score / `peak_vram` / `OOM` / `CRASH`).
   Before running, glance at `python vast.py status`: if under ~9 minutes remain to the deadline,
   skip it and wind down.

4. **Record the result, then KEEP or RESET.**
   - **Log it** to the ledger: `python vast.py log <commit> <score> <mem_gb>
     <keep|discard|crash> "<description>"`. (`<score>` is the objective the run printed;
     `results.tsv` feeds the dashboard and is the durable record.)
   - **KEEP** if the score improved vs the champion (per the goal direction): commit the change on
     the champion branch (`git commit -am "<idea>: val_bpb X → Y"`). The champion has ratcheted
     forward by one tooth. **Confirm before counting a marginal win** — a small delta can be
     noise; re-run it once, and if it holds, keep it.
   - **RESET** otherwise: `git checkout -- train.py` (or `git reset --hard HEAD`) to throw the
     idea away and return to the champion. An `OOM` counts as a failure → log `crash`, reset, and
     don't retry it bigger (batch/size is the capacity direction's job).
   - **Bank + move on.** A kept win is now part of the champion — add it to the **Banked** list in
     `findings.md` and stop re-visiting it. Next idea builds on top.

5. **Update `findings.md`** every iteration from the results — keep **Champion** (val + the
   one-line change), **Tried** (every idea + result), **Banked** (folded in), **Dead ends**
   current. Read it before picking each next idea so nothing repeats and the search stays broad.

6. **Loop** (see NEVER STOP). Each iteration starts from the current champion, so every experiment
   is "champion + one new idea" by construction and the search never restarts from scratch.

---

## How an experiment runs (and is bounded)

`vast.py exp` syncs `train.py` to the box, exposes the box's GPUs (`CUDA_VISIBLE_DEVICES=0..N-1`,
`AR_NUM_GPUS=N`), runs it, parses the result, and prints `val_bpb` / `RESULT_JSON`. On a one-GPU
box it's a plain `python train.py`; on a multi-GPU box the single experiment is launched under
`torchrun` (`python -m torch.distributed.run --nproc_per_node=N`), so one model shards across all
the GPUs (standard `RANK`/`WORLD_SIZE`/`LOCAL_RANK` are set). Force the launcher with
`exp --launcher {auto,torchrun,single}` if needed. **Each run is HARD-bounded by the
training budget, gracefully — no brutal kill:** `train.py` self-stops when its elapsed training
time hits the budget, and a frozen alarm in `prepare.py` (`start_training_clock`, SIGALRM at the
budget + grace) is the hard backstop — when it fires, training stops *immediately* even on a
slow/hung step, and `train.py` catches it and **still runs the final eval**, so the run always
returns a `val_bpb` for the model trained up to that moment. The budget is set per session via
`vast.py` (default 5 min). `exp` does **not** SIGKILL the run; it only reaps a *ghost* still on
the GPU **before** launching. For a genuinely wedged process (e.g. a stuck compile, which a
Python-level alarm can't interrupt), `python vast.py reap` clears it.

---

## VRAM & OOM (no OOMs)

Keep peak VRAM well under the GPU's limit (< ~22 GB on a 24 GB 4090). The baseline is sized to
fit; an OOM is a failure: log `crash`, reset, don't retry larger. Memory-heavy ideas must be
paired with a smaller `DEVICE_BATCH_SIZE`. `reap` clears a stuck run holding memory.

---

## Logging & objectivity

Log every run via `python vast.py log` (file-locked), 5 tab-separated columns: `commit  score
memory_gb  status  description` (`score` = the objective the experiment printed; `0000000` /
`0.000000` / `0.0` on failure). The fixed budget makes runs comparable. When a `val_bpb` delta is
small, **re-run before crowning a new champion**. Don't `git add` `results.tsv` or `findings.md`
(untracked).

---

## Findings

`findings.md` is your notebook. You log every run to `results.tsv` and curate `findings.md` from
it. Update it every iteration, right after you record a result. Keep **Champion** (val + the
one-line change), **Tried** (every idea + result), **Banked** (folded in), **Dead ends** current.

---

## Teardown

The watchdog auto-destroys the box at the deadline. To stop early: `python vast.py down`, then
`python vast.py ps` to confirm nothing is billing. `extend --hours <H>` pushes the deadline;
`ps`/`nuke` catch and kill orphans; `reap` clears a stray run without destroying the box.

---

## NEVER STOP

Once the loop begins, don't pause to ask whether to continue. The human may be asleep and expects
research **until the deadline, an interrupt, or teardown**. When an experiment finishes,
immediately pick the next idea and run it — keep the GPU busy every minute. If ideas run dry:
re-read `train.py`, mine the literature with the `research` skill, combine near-misses, try more
radical changes, rotate directions. The loop runs until stopped, period.
