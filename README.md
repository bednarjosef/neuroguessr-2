# autoresearch

An autonomous research agent you drive from a laptop with **no local GPU**. You tell it what to
optimize; it rents a GPU box on [Vast.ai](https://vast.ai), then runs a simple **keep/reset
ratchet** — mutate one **experiment** file, run it once, keep the change if it beat the champion,
otherwise throw it away — improving a measured result over a session. It's a single agent running
one experiment at a time, in the spirit of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), with Vast wired in so the
experiments run on a rented GPU when you don't have one locally. (That one experiment can shard
across **multiple GPUs** via `torch.distributed` when a model needs them — `--gpus N` — it's
just never *several* experiments in parallel.) Ideas are grounded in the literature via the
`research` skill, progress is visible in a live dashboard, and the box tears itself down at a
hard deadline so you can leave it running overnight.

**It can research almost anything.** The repo ships configured for **LLM pretraining** (the
*experiment* is `train.py`, the *harness* is `prepare.py`, the *objective* is `val_bpb`) — but
on a fresh clone the agent **onboards you and reshapes it to your goal**: any task where one
artifact can be mutated and scored by a frozen evaluator (an algorithm, a kernel, a solver, a
prompt, a strategy…). Set the objective + direction with `vast.py start --metric NAME --goal
min|max`; the loop is the same regardless of domain.

The default training core is a single-file GPT (Muon + AdamW), forked from
[nanochat](https://github.com/karpathy/nanochat) / [autoresearch](https://github.com/karpathy/nanochat).

---

## How it works

- **You** pick the goal (lowest `val_bpb`, a better optimizer, a loss that generalizes, …), the
  per-experiment training budget, and how long the session runs.
- **The agent** rents a Vast box (one GPU by default; `--gpus N` for one experiment sharded
  across N via `torch.distributed`) and runs the ratchet loop. Each iteration it:
  1. picks **one** idea (from the search directions in `program.md` + its own reasoning and the
     literature),
  2. edits `train.py`,
  3. runs it once — `vast.py exp` trains for a fixed budget on the rented GPU and reports the
     score (**`val_bpb`**: held-out bits-per-byte, lower = better, vocab-independent so changes
     compare fairly),
  4. **keeps** the change (commits it as the new champion) if it beat the previous best, else
     **resets** `train.py` and tries something else.
- **The champion only ratchets forward.** Every kept win becomes the base the next idea builds on,
  so the score improves monotonically across the whole session.

The metric (`evaluate_bpb`) and the data regime (`prepare.py`) are **frozen during research** so
the score can't be gamed — gains must come from `train.py` alone.

## The files

| File | Role |
|------|------|
| **`CLAUDE.md`** | Session-start router. On a fresh clone it **onboards you** (asks what to research) and tailors `program.md`. |
| **`program.md`** | The **mission & config** you edit: what to optimize, the metric, the knobs, the search directions. |
| **`ENGINE.md`** | The **fixed engine**: how the agent runs the ratchet loop. Not edited. |
| **`train.py`** | The research target — the only file edited *during* research. |
| **`prepare.py`** | Data, tokenizer, regime, and the `evaluate_bpb` metric. Frozen during research. |
| **`vast.py`** | The control plane (rent / setup / run one experiment / dashboard / teardown). |

## Quick start

```bash
# 1. One-time: authenticate Vast (an ssh key is auto-created/registered).
vastai set api-key <YOUR_KEY>

# 2. Open Claude Code in this repo (grant permissions for autonomy).
#    A fresh clone auto-onboards: it asks what you want to research, the per-experiment
#    minutes, the session hours, and the hardware — then tailors program.md to your goal.

# 3. Say "kick off a session". The agent brings a box up in one command:
python vast.py start --hours 3 --minutes 5 --max-price 0.60
#    start = rent cheapest qualifying box → background watchdog → setup.
#    Need multiple GPUs for one big model? add --gpus 4 (one experiment, sharded via torchrun).
```

Then the agent runs the loop on its own until the deadline (or you stop it). Two independent time
knobs: **`--minutes`** = how long *one* experiment trains; **`--hours`** = how long the *whole*
session runs before the box auto-destroys (typically 2–3 h).

> Already have a GPU and just want to smoke-test the trainer locally?
> `uv sync && uv run prepare.py && uv run train.py`.

## Research-driven ideas — the `research` skill

The agent doesn't only brainstorm from memory. It uses the
[**`research` skill**](https://github.com/bednarjosef/claude-research-skill) — which searches
[OpenAlex](https://openalex.org), fetches open-access PDFs, and converts them to Markdown so the
agent *reads* papers (line-numbered) instead of just citing them. **Install it once** (keyless —
no API key):

```bash
git clone https://github.com/bednarjosef/claude-research-skill
cd claude-research-skill && ./install.sh   # symlinks into ~/.claude/skills/research/
```

Between experiments the agent leans on it to:

- surface **novel ideas** and **current SOTA** for whatever is being optimized,
- check whether an idea is already known to work (or to fail), before spending a run on it,
- ground each next experiment in the literature rather than guesswork.

It scans **abstracts first** (cheap) and only fetches full text for the most promising leads.
`ENGINE.md` directs the agent to use the skill for ideation, SOTA-hunting, and verification — not
to rely on what it already knows. (Because a single `exp` run blocks for the whole training
budget, the agent may optionally spin up a research-scout subagent during that window to mine the
literature — but the loop is single-agent by default.)

## Control plane (`vast.py`)

`python vast.py --help` for everything. The ones you'll see most:

| Command | What it does |
|---------|--------------|
| `start` | One-shot bring-up: rent → watchdog → setup (`--gpus N` for a multi-GPU box). |
| `exp --train F` | Run one experiment (`train.py`); print `val_bpb`/vram (foreground/blocking). Sharded across all GPUs via `torchrun` on a multi-GPU box. |
| `dashboard` | Live browser view (chart, leaderboard, cost, deadline). |
| `log` | Append one row to the results ledger. |
| `reap` | Kill a stray/ghost run; show what's on the GPU. |
| `status` / `ps` | Tracked-box status / all account instances. |
| `down` / `nuke` | Destroy the box / destroy all boxes. |

## Watch it live

`python vast.py dashboard` starts a tiny local web server (stdlib only, no deps) and opens a page
that auto-refreshes every 5s: a **`val_bpb` chart** with a best-so-far line, a leaderboard, recent
runs, and a live box panel (GPU, $/hr, uptime, **spend so far**, **deadline countdown**). It reads
the local `results.tsv` + `.vast_state.json`, so it updates as the agent logs results — leave it
open overnight.

## Safety & cost

- **One tracked box at a time** with a **hard auto-destroy deadline** and a background `watchdog`;
  `ps`/`nuke` catch orphans, `reap` clears a stray run.
- At ~$0.34/hr for an RTX 4090, a 3-hour single-GPU session is roughly **$1** (`--max-price` is
  per-GPU/hr, so an N-GPU box costs ~N×).

## Credits & license

Forked from Andrej Karpathy's autoresearch; training core adapted from
[nanochat](https://github.com/karpathy/nanochat). MIT.
