# autoresearch — session start

This is a general **autonomous research agent**: a single agent rents a GPU box and runs a
**keep/reset ratchet** — it mutates one **experiment artifact**, runs it once, and keeps the
change only if it improved a **measured objective**, building a steadily-better champion over a
session. One experiment at a time, no parallel experiments — the simplicity of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), with Vast wired in so the
experiments run on a rented GPU when there's no local one. (A single experiment can still shard
across multiple GPUs via `torch.distributed` when a model needs them — `--gpus N` — it's just
never *several* experiments at once.) **It can research almost anything**
that fits a simple contract (below) — optimizing an ML model, an algorithm, a GPU kernel, a
solver, a prompt, a trading rule, a compression scheme, a config… **LLM pretraining is just
the default instantiation that ships in the repo** (`train.py` + `prepare.py`, objective
`val_bpb`); it's filler to be reshaped to whatever the user wants.

Two docs drive a session:
- **`program.md`** — the **mission & config** (what to optimize, the metric, the knobs, the
  search directions). Editable by you and the human.
- **`ENGINE.md`** — the **fixed engine** (how the agent runs the loop). **Never edit it.**

The roles (the LLM default in parentheses):
- **The experiment** (`train.py`) — the one artifact the agent edits; running it prints a
  `OBJECTIVE: <number>` line.
- **The harness** (`prepare.py`) — the **frozen** task/data/evaluator that computes the
  objective honestly, so the score can't be gamed.
- **The objective** — a metric name + direction, set via `vast.py start --metric NAME --goal min|max`.
- **The search directions** — a menu of idea families the agent rotates through so the search
  stays broad instead of tunneling on one knob.

At the **start of every session**, do ONE of the following:

## A) Fresh clone — run onboarding FIRST

**Check `program.md` for the marker `<!-- AUTORESEARCH:UNCONFIGURED -->`.** If present, the
repo hasn't been pointed at a goal yet. Before anything else (even if the user asked something
else, say you'll get them set up first), **run onboarding**:

1. **Ask what they want to research**, with `AskUserQuestion` (one question at a time, drilling
   in with follow-ups until it's unambiguous):
   - **The research target.** What gets optimized? It can be **anything** — keep the LLM
     default, or point it elsewhere (another ML task, an algorithm/heuristic, a kernel, a
     solver, a prompt, a strategy…). Offer a few concrete options plus "something else (type it)".
   - **The objective + direction.** What single number defines success, and is **lower or
     higher** better? (LLM default: `val_bpb`, lower.) Note any secondary signal to watch.
   - **How a run is scored.** What does one experiment *do*, and how is the number computed —
     so we can make it a **frozen, un-gameable evaluator**? (For LLM: train, then `evaluate_bpb`.)
   - **Two time budgets, separately:** per-experiment minutes (one run; default 5) and session
     hours (whole run before auto-destroy; default 3, ~2–3 h typical).
   - **Hardware:** GPU type (default `RTX_4090`) and **how many GPUs** (default 1). Use `>1`
     **only** when a single model must shard across GPUs via `torch.distributed` (e.g. 2–4× 5090
     for a big model) — it's still one experiment at a time, never parallel experiments, and
     needs a distributed-aware `train.py`. Price cap is per-GPU/hr (default $0.60). If the task
     doesn't need a GPU, the box still has CPUs — note it.
   - **Constraints / must-keeps / ideas to try first.**

2. **Tailor the repo to their answers.**
   - **If keeping the LLM default:** just rewrite `program.md` §1/§2/§4 for their angle, set §3
     config, optionally tune `prepare.py` (e.g. `TRAIN_TOKENS`) and the `train.py` baseline.
   - **If retargeting to another domain — the adaptation playbook:**
     1. **Rewrite `train.py` as the experiment** for the new task. It must: run one trial and
        **print exactly one `OBJECTIVE: <number>`** summary line (named whatever you choose),
        plus any diagnostics as extra `name: number` lines. Keep it self-contained and
        deterministic where possible. (If it needs to shard across GPUs via `torch.distributed`
        for `--gpus N`, init the process group from the torchrun env — `RANK`/`WORLD_SIZE`/
        `LOCAL_RANK`, or read `AR_NUM_GPUS` — and print the `OBJECTIVE:` line on **rank 0 only**.)
     2. **Rewrite `prepare.py` as the frozen harness/evaluator** — one-time setup (data/assets)
        + the function that computes the objective. **Reuse its deadline helpers**
        (`start_training_clock` / `except TrainingTimeUp` / `stop_training_clock`) so every run
        is hard-bounded to the per-experiment budget and still emits a final score.
     3. **Set the objective:** the session will use `vast.py start --metric OBJECTIVE --goal min|max`.
        Record that name/direction in `program.md` §2/§3.
     4. **Define the search directions in `program.md` §4** for the new domain — a menu of idea
        families (grouped by *what part of the experiment* a change touches) the agent rotates
        through to keep the search broad.
     5. **Adjust deps** in `pyproject.toml` if the task needs different libraries (installed at setup).
   - In all cases set §3 config to their choices and **remove the `<!-- AUTORESEARCH:UNCONFIGURED -->`
     marker** from `program.md`.

3. **Smoke-test the contract before declaring done.** Once a box is up (or locally if it has the
   right compute), run the experiment once and confirm it prints the `OBJECTIVE:` line and that
   `vast.py exp` parses it. Fix until it runs clean — the whole point is that it works out of the box.

4. **Confirm** the tailored mission to the human in 2–3 lines, then offer to kick off a session
   (`vast.py start --metric … --goal … …`). Don't rent GPUs until they say go.

## B) Already configured — proceed normally

No marker → the mission is set. Read `program.md` + `ENGINE.md`, then do what the user asked.
To start a session, follow `ENGINE.md`: `python vast.py start --metric … --goal … …`, run the
baseline once (`python vast.py exp --train train.py`), then run the ratchet loop — **pick one
idea, edit `train.py`, run it once with `python vast.py exp`, and keep it only if it beats the
champion** — using the `research` skill between experiments for fresh, literature-grounded ideas.

---

**Always:** never edit `ENGINE.md`. During research only the **experiment** file is edited; the
**harness/evaluator is frozen** so the objective can't be gamed. Full control plane:
`python vast.py --help`.
