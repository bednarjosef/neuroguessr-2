#!/usr/bin/env python3
"""Vast.ai control plane for autoresearch — one single-GPU box per session.

The research loop and the agent run on YOUR laptop. The only thing that goes to
Vast is the GPU work: each experiment is `train.py` running for the fixed
per-experiment budget (default 5 min) on a rented box. This exists so you can
drive an autonomous research loop from a machine with no local GPU. It runs ONE
experiment at a time — never several in parallel — but that one experiment may be
sharded across several GPUs with torch.distributed (torchrun) when a model needs
more than one. Rent 1 GPU (default) or N with `--gpus N`; the box tears itself
down at a hard deadline.

Safety model (same spirit as the battle-tested arc script):
  * Exactly ONE tracked box at a time, recorded in .vast_state.json with its price
    and a hard deadline. `watchdog` auto-destroys at the deadline. `status` shows
    cost so far. `down` tears it down; `ps`/`nuke` catch orphans across the account.
  * `up` refuses offers above the price cap and always prints the bill rate.

Auth: run `vastai set api-key <KEY>` once (stored locally). An ssh key is
auto-generated + registered if missing.

Typical session (what ENGINE.md drives the agent to do):
  vast.py start --hours 3 --minutes 5      # rent + watchdog + setup, one command
  vast.py exp --train train.py             # run one experiment, print val_bpb/vram
  vast.py log <commit> <score> <mem> keep "baseline"   # append to results.tsv
  vast.py down                             # destroy + clear state (or let watchdog)

Commands:
  search   list cheapest qualifying offers (1 GPU by default, N with --gpus)
  up       rent cheapest qualifying box + record price/deadline
  start    one-shot bring-up: up + watchdog + setup
  status   instance status, uptime, est. cost, deadline
  sync     upload the repo to the box (tar over ssh)
  setup    sync + install deps + prepare.py (data/tokenizer) + detect GPU
  exp      --train PATH: run that train.py on the GPU, print val_bpb/vram
  run      run an arbitrary command in the remote repo dir
  log      atomically append a row to local results.tsv
  pull     download a file/dir back from the box
  dashboard  live local web view (chart, leaderboard, cost, deadline)
  reap     kill a stray/ghost train.py run on the box
  watchdog loop; destroy at the recorded deadline (run in background)
  extend   push the deadline out
  down     destroy the tracked box + clear state
  ps       list ALL instances on the account (catch orphans)
  nuke     destroy ALL instances on the account
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from string import Template

REPO = Path(__file__).resolve().parent
STATE = REPO / ".vast_state.json"
REMOTE_DIR = "/root/auto"
RESULTS = REPO / "results.tsv"

# Use the proven "PyTorch (Vast)" template: it ships a cached image with CUDA + python
# + a working sshd AND torch preinstalled at /venv/main. We RUN that torch directly
# (no multi-GB torch download per box) and only pip-install the few light deps this
# repo adds. `uv sync` is a fallback only if the template torch is somehow unusable.
TEMPLATE_HASH = "a33b72bd045341cfcd678ce7c932a614"  # PyTorch (Vast)
DEFAULT_REMOTE_PY = "/venv/main/bin/python"          # the template's torch interpreter
DEFAULT_GPU = "RTX_4090"

# Offer filter (applied in Python on the raw JSON, robust to query syntax).
MIN_CUDA = 12.1          # template torch is cu12x; host driver must support >= 12.1
MIN_RELIABILITY = 0.98
MIN_INET_DOWN = 300      # Mbit/s — HF data + torch wheels download fast
MIN_DLPERF = 110         # avoid throttled/junk hosts (a real 4090 is ~150)
MIN_CPU_PER_GPU = 6      # the dataloader tokenizes on CPU; give each GPU real cores
MIN_RAM_PER_GPU = 16     # GB
BLOCK_COUNTRIES = ("CN",)  # HuggingFace is often blocked/slow here


def _vastai() -> str:
    return shutil.which("vastai") or str(Path.home() / ".local/bin/vastai")


def vast(args: list[str], raw: bool = False, check: bool = True):
    cmd = [_vastai(), *args]
    if raw:
        cmd.append("--raw")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed:\n{p.stderr or p.stdout}")
    if raw:
        try:
            return json.loads(p.stdout or "null")
        except json.JSONDecodeError:
            sys.exit(f"could not parse JSON from: vastai {' '.join(args)}\n{p.stdout}")
    return p.stdout


def load_state() -> dict | None:
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(d: dict) -> None:
    STATE.write_text(json.dumps(d, indent=2))


def require_state() -> dict:
    s = load_state()
    if not s:
        sys.exit("no tracked instance (.vast_state.json). Run: vast.py up")
    return s


# --- ssh -----------------------------------------------------------------------

SSH_KEY = Path.home() / ".ssh/id_ed25519"


def ensure_ssh_key() -> None:
    pub = SSH_KEY.with_suffix(".pub")
    if not pub.exists():
        print("generating an ssh key…")
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(SSH_KEY)], check=True)
    subprocess.run([_vastai(), "create", "ssh-key", pub.read_text().strip()],
                   capture_output=True, text=True)  # idempotent


def _ssh_opts() -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR"]
    if SSH_KEY.exists():
        opts += ["-i", str(SSH_KEY)]
    return opts


def ssh_base(s: dict) -> list[str]:
    return ["ssh", *_ssh_opts(), "-p", str(s["port"]), f"root@{s['host']}"]


def ssh_run(s: dict, remote_cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_base(s) + [remote_cmd], capture_output=True, text=True, check=check)


def scp_to(s: dict, local: str, remote: str) -> None:
    subprocess.run(["scp", *_ssh_opts(), "-P", str(s["port"]), local,
                    f"root@{s['host']}:{remote}"], check=True)


def attach_ssh(s: dict) -> None:
    pub = SSH_KEY.with_suffix(".pub")
    if pub.exists():
        subprocess.run([_vastai(), "attach", "ssh", str(s["instance_id"]), pub.read_text().strip()],
                       capture_output=True, text=True)


def wait_ssh(s: dict, tries: int = 36) -> bool:
    r = None
    for _ in range(tries):
        r = ssh_run(s, "true")
        if r.returncode == 0:
            return True
        time.sleep(5)
    print("WARN: ssh not ready:", (r.stderr if r else "").strip()[:200])
    return False


def _ssh_endpoint(iid) -> tuple[str | None, int | None]:
    out = vast(["ssh-url", str(iid)], check=False) or ""
    m = re.match(r"ssh://[^@]+@([^:\s]+):(\d+)", out.strip())
    return (m.group(1), int(m.group(2))) if m else (None, None)


# --- offers --------------------------------------------------------------------

def find_offers(gpu: str, gpus: int, max_price_per_gpu: float) -> list[dict]:
    # verified=true keeps Vast's vetted hosts; num_gpus pins a whole N-GPU machine (all GPUs
    # identical) so ONE experiment can shard a model across them with torch.distributed.
    offers = vast(["search", "offers",
                   f"gpu_name={gpu} num_gpus={gpus} rentable=true verified=true",
                   "-o", "dph_total"], raw=True) or []
    machine_cap = max_price_per_gpu * gpus
    ok = []
    for o in offers:
        country = str(o.get("geolocation", "")).split(",")[-1].strip()
        cpu = o.get("cpu_cores_effective") or o.get("cpu_cores", 0)
        ram = o.get("cpu_ram", 0) / 1024  # MB -> GB (total box RAM)
        if (o.get("cuda_max_good", 0) >= MIN_CUDA
                and o.get("reliability2", 0) >= MIN_RELIABILITY
                and o.get("inet_down", 0) >= MIN_INET_DOWN
                and o.get("dlperf", 0) >= MIN_DLPERF
                and cpu >= MIN_CPU_PER_GPU * gpus
                and ram >= MIN_RAM_PER_GPU * gpus
                and o.get("dph_total", 1e9) <= machine_cap
                and country not in BLOCK_COUNTRIES):
            ok.append(o)
    ok.sort(key=lambda o: o["dph_total"])
    return ok


def fmt_offer(o: dict) -> str:
    n = o.get("num_gpus", 1)
    cpu = o.get("cpu_cores_effective") or o.get("cpu_cores", 0)
    return (f"id={o['id']:<10} ${o['dph_total']:.3f}/hr (${o['dph_total']/n:.3f}/gpu)  "
            f"{n}x {o['gpu_name']}  cpu={cpu:.0f}  dlperf={o.get('dlperf', 0):.0f}  "
            f"cuda={o.get('cuda_max_good')}  inet={o.get('inet_down', 0):.0f}↓  "
            f"{o.get('geolocation', '?')}")


# --- commands ------------------------------------------------------------------

def cmd_search(a) -> None:
    offers = find_offers(a.gpu, a.gpus, a.max_price)
    if not offers:
        print(f"no {a.gpus}x {a.gpu} offers <= ${a.max_price}/gpu/hr meeting filters")
        return
    print(f"top {min(8, len(offers))} cheapest qualifying {a.gpus}x {a.gpu} offers:")
    for o in offers[:8]:
        print("  " + fmt_offer(o))


def cmd_up(a) -> None:
    if load_state():
        sys.exit("an instance is already tracked. `status`/`down` first (or rm .vast_state.json).")
    ensure_ssh_key()
    offers = find_offers(a.gpu, a.gpus, a.max_price)
    if not offers:
        sys.exit(f"no {a.gpus}x {a.gpu} offers <= ${a.max_price}/gpu/hr meeting filters "
                 f"(try --max-price, fewer --gpus, or a different --gpu)")
    if a.offer_id:
        best = next((o for o in offers if o["id"] == a.offer_id), None)
        if not best:
            sys.exit(f"offer {a.offer_id} not in qualifying set; run `search`")
    else:
        best = offers[0]
    print("renting: " + fmt_offer(best))
    create = ["create", "instance", str(best["id"]), "--disk", str(a.disk)]
    create += ["--image", a.image] if a.image else ["--template_hash", TEMPLATE_HASH]
    res = vast(create, raw=True)
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not iid:
        sys.exit(f"create failed: {res}")
    now = time.time()
    save_state({
        "instance_id": iid, "offer_id": best["id"], "dph": best["dph_total"],
        "gpu": best["gpu_name"], "num_gpus": best.get("num_gpus", a.gpus),
        "launcher": getattr(a, "launcher", "auto"),  # auto | torchrun | single
        "created_at": now, "deadline": now + a.hours * 3600,
        "time_budget_s": int(getattr(a, "minutes", 5) * 60),  # per-experiment compute budget
        "metric": getattr(a, "metric", PRIMARY_METRIC_DEFAULT),  # objective name (domain-agnostic)
        "goal": getattr(a, "goal", "min"),                       # "min" or "max"
        "host": None, "port": None, "ready": False, "remote_py": DEFAULT_REMOTE_PY,
    })
    print(f"created instance {iid} at ${best['dph_total']:.3f}/hr "
          f"({best.get('num_gpus', a.gpus)} GPUs); auto-destroy deadline in {a.hours}h; "
          f"per-experiment budget {getattr(a, 'minutes', 5):.0f} min.")
    print("IMPORTANT: start the watchdog now so the box can't be left billing:\n"
          "  python vast.py watchdog &")
    print("waiting for it to boot…")
    _wait_running()


def _spawn_watchdog() -> None:
    """Launch the deadline watchdog as a detached background process so the box can't be
    left billing — survives this command returning."""
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    print("watchdog: launched in background (auto-destroys the box at the deadline).")


def cmd_start(a) -> None:
    """One-shot session bring-up: rent → watchdog → setup. Collapses the manual steps into
    one robust command so the agent starts fast with sane defaults. The first `exp` run
    (the baseline) doubles as the smoke test and warms the compile cache."""
    cmd_up(a)            # rents + waits until running (sys.exit on failure)
    _spawn_watchdog()
    cmd_setup(a)         # template torch + light deps + data + detect GPU
    print("\n=== READY ===")
    print("box up + prepared. Next (agent): run the baseline once with "
          "`python vast.py exp --train train.py`, then start the ratchet loop. See ENGINE.md.")


def _wait_running() -> None:
    s = require_state()
    for _ in range(60):
        info = vast(["show", "instance", str(s["instance_id"])], raw=True)
        status = info.get("actual_status")
        print(f"  status={status}")
        if status == "running":
            host, port = _ssh_endpoint(s["instance_id"])
            if host:
                s["host"], s["port"] = host, port
                save_state(s)
                print(f"READY. ssh -p {port} root@{host}\nnext: python vast.py setup")
                return
        time.sleep(10)
    print("still not running after 10 min; check `vast.py status` / the Vast UI")


def cmd_status(a) -> None:
    s = load_state()
    if not s:
        print("no tracked instance.")
        return
    info = vast(["show", "instance", str(s["instance_id"])], raw=True, check=False)
    up_h = (time.time() - s["created_at"]) / 3600
    print(f"instance {s['instance_id']}  {s.get('num_gpus', 1)}x {s['gpu']}  "
          f"${s['dph']:.3f}/hr  exp_budget={s.get('time_budget_s', 300)//60}min  "
          f"launcher={_effective_launcher(s)}  "
          f"objective={s.get('metric', PRIMARY_METRIC_DEFAULT)}({s.get('goal', 'min')})")
    print(f"  status: {info.get('actual_status') if isinstance(info, dict) else '?'}")
    print(f"  uptime: {up_h:.2f}h   est. cost so far: ${up_h * s['dph']:.2f}")
    print(f"  deadline in: {(s['deadline'] - time.time()) / 3600:.2f}h")
    if s.get("host"):
        print(f"  ssh -p {s['port']} root@{s['host']}")


def _resolve_endpoint(s: dict) -> dict:
    host, port = _ssh_endpoint(s["instance_id"])
    if host:
        s["host"], s["port"] = host, port
        save_state(s)
    elif not s.get("host"):
        _wait_running()
        s = require_state()
    return s


def cmd_sync(a) -> None:
    s = _resolve_endpoint(require_state())
    attach_ssh(s)
    if not wait_ssh(s):
        sys.exit("ssh never came up; check `vast.py status` / the Vast UI")
    print(f"uploading repo -> {REMOTE_DIR} (tar over ssh)…")
    excludes = " ".join(f"--exclude=./{x}" for x in
                        (".venv", ".git", ".vast_state.json", "results.tsv",
                         "__pycache__", "findings.md", "dev", ".env"))
    remote = " ".join(ssh_base(s)) + f" 'mkdir -p {REMOTE_DIR} && tar xzf - -C {REMOTE_DIR}'"
    subprocess.run(f"tar czf - {excludes} -C {REPO} . | {remote}", shell=True, check=True)
    print("upload complete.")


SETUP_SCRIPT = r"""
set -e
cd {REMOTE_DIR}
export HF_HUB_DISABLE_PROGRESS_BARS=1
{FORWARD_EXPORTS}
TPY=/venv/main/bin/python

# Prefer the template's PREINSTALLED torch -> no multi-GB torch download per box.
# Require CUDA + torch>=2.4 + F.rms_norm (train.py needs it); else fall back to uv.
if [ -x "$TPY" ] && "$TPY" -c "import torch,sys; from torch.nn.functional import rms_norm; sys.exit(0 if torch.cuda.is_available() and tuple(map(int,torch.__version__.split('.')[:2]))>=(2,4) else 1)" 2>/dev/null; then
  PY="$TPY"
  echo "=== using template torch (no download) ==="
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'on', torch.cuda.get_device_name(0))"
  echo "=== installing light deps only (torch already present) ==="
  "$PY" -m pip install -q --no-input "transformers>=4.55" "peft>=0.13" "accelerate>=0.34" \
    "datasets>=3.0" "huggingface_hub>=0.25" "wandb>=0.18" pillow pyarrow pandas requests "numpy>=1.26" safetensors
else
  echo "=== template torch unusable -> building .venv with uv (downloads torch) ==="
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || pip install -q uv
  uv sync
  PY={REMOTE_DIR}/.venv/bin/python
fi
echo "REMOTE_PY=$PY"

echo "=== backbone smoke (transformers + peft import + CUDA visible) ==="
"$PY" -c "import torch, transformers, peft; print('transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())"

echo "=== prepare.py (download fixed geolocalization subset; idempotent, cached) ==="
"$PY" prepare.py --num-shards {NUM_SHARDS}

echo "GPU_COUNT=$(nvidia-smi -L | wc -l)"
echo "CPU_COUNT=$(nproc)"
echo "=== setup complete ==="
"""


def cmd_setup(a) -> None:
    cmd_sync(a)
    s = require_state()
    exports = "\n".join(f"export {kv}" for kv in _forward_env_fragment().split()) if _forward_env_fragment() else ""
    script = SETUP_SCRIPT.format(REMOTE_DIR=REMOTE_DIR, NUM_SHARDS=a.num_shards, FORWARD_EXPORTS=exports)
    # Stream setup output live so the user/agent can watch the (slow) data prep.
    p = subprocess.run(ssh_base(s) + [script])
    if p.returncode != 0:
        sys.exit("setup failed; see output above")
    # Detect which python to run experiments with (template torch vs uv fallback).
    detect = ssh_run(s, (
        f'if [ -x {DEFAULT_REMOTE_PY} ] && {DEFAULT_REMOTE_PY} -c '
        '"import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; '
        f'then echo PY={DEFAULT_REMOTE_PY}; '
        f'elif [ -x {REMOTE_DIR}/.venv/bin/python ]; then echo PY={REMOTE_DIR}/.venv/bin/python; fi; '
        'echo GPU=$(nvidia-smi -L | wc -l)')).stdout
    pym = re.search(r"PY=(\S+)", detect)
    gm = re.search(r"GPU=(\d+)", detect)
    if pym:
        s["remote_py"] = pym.group(1)
    if gm:  # authoritative GPU count from the box itself (offer metadata can be approximate)
        s["num_gpus"] = int(gm.group(1))
    s["ready"] = True
    save_state(s)
    print(f"\nremote python: {s.get('remote_py', DEFAULT_REMOTE_PY)}  "
          f"({'template torch — no download' if s.get('remote_py') == DEFAULT_REMOTE_PY else 'uv .venv'})")
    print(f"GPUs on box: {s.get('num_gpus', 1)}  (launcher: {_effective_launcher(s)} — "
          f"one experiment {'sharded across all GPUs' if _effective_launcher(s) == 'torchrun' else 'on one GPU'})")
    print("next: python vast.py exp --train train.py   (run the baseline)")


# --- experiment run ------------------------------------------------------------

# The default (LLM) objective. Override per session with `start --metric NAME --goal min|max`.
# The experiment just has to print a "NAME: <number>" summary line; nothing here is
# LLM-specific, so re-targeting the research to another domain needs no change in this file.
PRIMARY_METRIC_DEFAULT = "val_bpb"
METRIC_LINE_RE = r"^([A-Za-z_][A-Za-z0-9_]*):[ \t]*(-?[0-9.]+)[ \t]*$"

REMOTE_RUN = f"{REMOTE_DIR}/run"          # where the experiment's train.py is uploaded/run
REMOTE_CACHE = f"{REMOTE_DIR}/.inductor"  # persistent torch.compile cache (warm starts)


def parse_metrics(text: str) -> dict:
    """Parse EVERY `name: number` summary line the experiment prints (domain-agnostic) —
    so whatever objective + diagnostics a re-targeted experiment emits are all captured."""
    out = {}
    for k, v in re.findall(METRIC_LINE_RE, text, re.MULTILINE):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


# Secrets/config forwarded to the experiment box (HF gated-model access + W&B tracking).
# Loaded from a gitignored .env in the repo root and/or the local environment (env wins).
FORWARD_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY",
                    "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE", "AR_RUN_NAME",
                    "AR_N_TRAIN", "AR_N_VAL", "AR_QUICK_VAL_N")


def _load_dotenv() -> dict:
    """Parse simple KEY=VALUE lines from REPO/.env (if present). Never uploaded to the box."""
    out = {}
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _wandb_key_from_netrc() -> str | None:
    """Read the W&B API key from ~/.netrc (machine api.wandb.ai) — the standard place the
    `wandb` CLI stores it after `wandb login`. Lets us forward it to the box without copying
    the secret into any file. Returns None if not present."""
    try:
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth[2] if auth else None
    except Exception:
        return None


def _forward_env_fragment() -> str:
    """Build the ` KEY=val ...` fragment of forwarded secrets/config for the remote command.
    Precedence: os.environ > .env > (for WANDB_API_KEY only) ~/.netrc. Only whitelisted keys."""
    merged = {**_load_dotenv(), **os.environ}
    if not merged.get("WANDB_API_KEY"):
        k = _wandb_key_from_netrc()
        if k:
            merged["WANDB_API_KEY"] = k
    parts = []
    for k in FORWARD_ENV_KEYS:
        v = merged.get(k)
        if v:
            parts.append(f"{k}={shlex.quote(v)}")
    return (" " + " ".join(parts)) if parts else ""


def _effective_launcher(s: dict) -> str:
    """Resolve the stored launcher preference to a concrete mode for this box. 'auto' ->
    'torchrun' on a multi-GPU box (one model sharded across the GPUs with torch.distributed),
    else 'single' (one process on one GPU). Force either with --launcher."""
    pref = s.get("launcher", "auto")
    if pref in ("torchrun", "single"):
        return pref
    return "torchrun" if s.get("num_gpus", 1) > 1 else "single"


def _launch_cmd(s: dict, script: str) -> str:
    """Build the remote command that runs ONE experiment (never several in parallel).

    On a multi-GPU box the single experiment is sharded across every GPU with
    torch.distributed (torchrun / `python -m torch.distributed.run`); on one GPU it's a
    plain python run. Either way it's ONE model/experiment. All N GPUs are exposed
    (CUDA_VISIBLE_DEVICES=0..N-1) and AR_NUM_GPUS tells the experiment how many it has;
    torchrun also sets the standard RANK / WORLD_SIZE / LOCAL_RANK. PYTHONPATH lets train.py
    import prepare.py regardless of cwd; a persistent TORCHINDUCTOR_CACHE_DIR makes
    torch.compile reuse artifacts across runs (warm starts after the first)."""
    py = s.get("remote_py", DEFAULT_REMOTE_PY)  # template torch by default; uv .venv fallback
    n = s.get("num_gpus", 1)
    budget = s.get("time_budget_s", 300)  # variable per session; prepare.py reads AR_TIME_BUDGET
    devices = ",".join(str(i) for i in range(n))
    env = (f"env CUDA_VISIBLE_DEVICES={devices} PYTHONPATH={REMOTE_DIR} "
           f"AR_TIME_BUDGET={budget} AR_NUM_GPUS={n} TORCHINDUCTOR_CACHE_DIR={REMOTE_CACHE}"
           f"{_forward_env_fragment()}")  # + forwarded HF_TOKEN / WANDB_* / AR_N_* from .env
    if _effective_launcher(s) == "torchrun":
        # `python -m torch.distributed.run` (== torchrun) via the same interpreter, so it
        # doesn't depend on PATH. --standalone = single-node rendezvous on a free port.
        return f"{env} {py} -m torch.distributed.run --standalone --nproc_per_node={n} {script}"
    return f"{env} {py} {script}"


def _exp_run(s: dict, train: Path) -> dict:
    """Run ONE experiment on the box's GPU and return its result dict: push this train.py,
    run it, parse + print the objective."""
    ssh_run(s, f"mkdir -p {REMOTE_RUN} {REMOTE_CACHE}")
    scp_to(s, str(train), f"{REMOTE_RUN}/train.py")
    log = f"{REMOTE_RUN}/run.log"
    inv = _launch_cmd(s, f"{REMOTE_RUN}/train.py")
    # Reap any GHOST run still holding the GPU, in a SEPARATE ssh call. The reap and the run
    # must NOT share a shell: pkill -f matches the *whole* command line, so if the reap ran in
    # the same shell as the train.py invocation it would SIGKILL its own wrapper shell before
    # python launched. The `[t]rain.py` regex class matches real train.py processes but keeps
    # the literal "train.py" out of pkill's own command line, so it can't self-match. The run
    # itself is NOT force-killed — train.py self-stops at the budget.
    ssh_run(s, f'pkill -9 -f "{REMOTE_RUN}/[t]rain.py" 2>/dev/null; sleep 1; true')
    # Capture EVERY `name: number` summary line (domain-agnostic), then pick the objective.
    remote = (f"{inv} > {log} 2>&1; "
              f"grep -E '^[A-Za-z_][A-Za-z0-9_]*:[ \\t]*-?[0-9.]+[ \\t]*$' {log} || "
              f"(echo '--- CRASH (tail) ---'; tail -n 40 {log})")
    n = s.get("num_gpus", 1)
    how = f"{n} GPUs via {_effective_launcher(s)}" if n > 1 else "1 GPU"
    print(f"running experiment ({how})…", flush=True)
    t0 = time.time()
    r = ssh_run(s, remote)
    dt = time.time() - t0
    m = parse_metrics(r.stdout)
    primary = s.get("metric", PRIMARY_METRIC_DEFAULT)
    oom = "out of memory" in r.stdout.lower() or "outofmemory" in r.stdout.lower()
    if primary in m:
        result = {"ok": True, "metric": primary, "score": m[primary],
                  "wall_seconds": round(dt, 1), **m}
        extra = f"  vram={m['peak_vram_mb']/1024:.1f}GB" if "peak_vram_mb" in m else ""
        print(f"{primary}={m[primary]:.6f}{extra}  wall={dt:.0f}s")
    else:
        reason = "OOM" if oom else "CRASH"
        result = {"ok": False, "reason": reason, "wall_seconds": round(dt, 1)}
        hint = "  (out of memory — shrink the run; treat as discard)" if oom else ""
        print(f"{reason} / no '{primary}'{hint}. tail:\n{r.stdout[-1500:]}")
    return result


def cmd_exp(a) -> None:
    """Run ONE experiment and report the objective / peak_vram. This is the unit of the
    ratchet loop: the agent edits train.py, runs `exp`, and keeps or resets on the score."""
    s = require_state()
    if not s.get("ready"):
        print("WARN: box not marked ready; run `vast.py setup` first.", file=sys.stderr)
    if a.minutes:  # per-run budget override (default: the session's time_budget_s)
        s = {**s, "time_budget_s": int(a.minutes * 60)}
    if a.launcher:  # per-run launcher override (default: the session's)
        s = {**s, "launcher": a.launcher}
    train = Path(a.train)
    if not train.exists():
        sys.exit(f"train file not found: {train}")
    result = _exp_run(s, train)
    print("RESULT_JSON:" + json.dumps(result))
    sys.exit(0 if result["ok"] else 1)


# --- dashboard (local live visualization) --------------------------------------

RESULT_COLS = ("commit", "score", "memory_gb", "status", "description")


def _read_results() -> list[dict]:
    """Parse results.tsv (the ledger the agent appends to via `vast.py log`)."""
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text().splitlines():
        if not line.strip() or line.startswith("commit\t"):
            continue
        p = (line.split("\t") + [""] * len(RESULT_COLS))[:len(RESULT_COLS)]
        rows.append(dict(zip(RESULT_COLS, p)))
    return rows


def _fval(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _svg_chart(rows: list[dict], metric: str = "val_bpb", goal: str = "min",
               w: int = 920, h: int = 340) -> str:
    """Inline SVG (no JS/CDN): the objective per experiment, colored by status, with a
    'best so far' line. Domain-agnostic — works for any metric name + direction."""
    # Plot non-crash rows with a numeric score (scores may be 0 or negative for some
    # domains). Index by POSITION among plotted points (j) so filtering a crash/discard
    # doesn't push later points off the right edge (which would look blank).
    pts = [(v, r["status"]) for r in rows
           if r["status"] != "crash" and (v := _fval(r.get("score"))) is not None]
    if not pts:
        return "<p class=muted>No completed experiments yet — the chart fills in as results log.</p>"
    pl, pr, pt, pb = 64, 18, 18, 34
    pw, ph = w - pl - pr, h - pt - pb
    vals = [v for v, _ in pts]
    ymin, ymax = min(vals), max(vals)
    if ymax - ymin < 1e-9:
        ymin, ymax = ymin - 0.01, ymax + 0.01
    n = len(pts)
    better = min if goal == "min" else max
    fx = lambda j: pl + (j / (n - 1) * pw if n > 1 else pw / 2)  # j = position among plotted
    fy = lambda v: pt + (ymax - v) / (ymax - ymin) * ph          # higher value -> higher up

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#0f1419" rx="8"/>')
    # y gridlines + labels
    for k in range(5):
        v = ymax - k / 4 * (ymax - ymin)
        y = pt + k / 4 * ph
        parts.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{w-pr}" y2="{y:.1f}" stroke="#1f2937"/>')
        parts.append(f'<text x="{pl-8}" y="{y+4:.1f}" fill="#6b7280" font-size="11" text-anchor="end">{v:.4g}</text>')
    # best-so-far line (running best per goal: min for lower-is-better, else max)
    best, bpath = None, []
    for j, (v, _) in enumerate(pts):
        best = v if best is None else better(best, v)
        bpath.append(f"{fx(j):.1f},{fy(best):.1f}")
    parts.append(f'<polyline points="{" ".join(bpath)}" fill="none" stroke="#34d399" stroke-width="2"/>')
    # points
    for j, (v, st) in enumerate(pts):
        c = {"keep": "#34d399", "discard": "#6b7280"}.get(st, "#f59e0b")
        parts.append(f'<circle cx="{fx(j):.1f}" cy="{fy(v):.1f}" r="3.4" fill="{c}"/>')
    parts.append(f'<text x="{pl}" y="{h-10}" fill="#6b7280" font-size="11">experiments → ({n} plotted)</text>')
    parts.append(f'<text x="{w-pr}" y="{h-10}" fill="#34d399" font-size="11" text-anchor="end">best {html.escape(metric)} {better(vals):g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _box_panel(s: dict | None) -> str:
    if not s:
        return '<div class="card warn">No active box (no .vast_state.json). Start one: <code>python vast.py up</code></div>'
    up_h = (time.time() - s["created_at"]) / 3600
    total_h = (s["deadline"] - s["created_at"]) / 3600
    left_h = (s["deadline"] - time.time()) / 3600
    frac = max(0.0, min(1.0, up_h / total_h)) if total_h else 0
    cost = up_h * s["dph"]
    chips = [
        f'<b>{s.get("num_gpus", 1)}× {html.escape(str(s["gpu"]))}</b>',
        f'${s["dph"]:.3f}/hr',
        f'up {up_h:.2f}h',
        f'spent ${cost:.2f}',
        (f'deadline in {left_h:.2f}h' if left_h > 0 else 'DEADLINE PASSED'),
    ]
    bar = (f'<div class=bar><div class=fill style="width:{frac*100:.1f}%"></div></div>'
           f'<div class=muted style="margin-top:4px">session {up_h:.2f}h / {total_h:.1f}h</div>')
    return f'<div class=card><div class=chips>{" ".join(f"<span>{c}</span>" for c in chips)}</div>{bar}</div>'


def _stats_panel(rows: list[dict], metric: str = "val_bpb", goal: str = "min") -> str:
    keeps = [r for r in rows if r["status"] == "keep"]
    disc = sum(1 for r in rows if r["status"] == "discard")
    crash = sum(1 for r in rows if r["status"] == "crash")
    valid = [v for v in (_fval(r.get("score")) for r in keeps) if v is not None]
    best = (min if goal == "min" else max)(valid) if valid else None
    cells = [("experiments", len(rows)), ("keeps", len(keeps)), ("discards", disc),
             ("crashes", crash), (f"best {metric}", f"{best:g}" if best is not None else "—")]
    return '<div class=stats>' + "".join(
        f'<div><div class=k>{html.escape(str(v))}</div><div class=muted>{lbl}</div></div>'
        for lbl, v in cells) + '</div>'


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{c}</th>" for c, _ in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for _, k in cols) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


PAGE = Template("""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=5><title>autoresearch</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0e13;color:#e5e7eb;
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;padding:22px}
h1{margin:0 0 4px;font-size:20px}h2{font-size:13px;text-transform:uppercase;
letter-spacing:.05em;color:#9ca3af;margin:0 0 10px}.sub{color:#34d399;font-size:12px}
.card{background:#111722;border:1px solid #1f2937;border-radius:10px;padding:16px;margin:14px 0}
.warn{border-color:#f59e0b;color:#fbbf24}.muted{color:#6b7280;font-size:12px}
.chips span{display:inline-block;background:#0f1419;border:1px solid #1f2937;border-radius:999px;
padding:4px 12px;margin:0 6px 6px 0}.chips b{color:#fff}
.bar{height:8px;background:#0f1419;border-radius:999px;overflow:hidden;margin-top:12px}
.fill{height:100%;background:linear-gradient(90deg,#34d399,#10b981)}
.stats{display:flex;gap:26px;flex-wrap:wrap}.stats .k{font-size:22px;color:#fff;font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:12.5px}th{text-align:left;color:#6b7280;
font-weight:500;padding:5px 8px;border-bottom:1px solid #1f2937}td{padding:5px 8px;
border-bottom:1px solid #141b26;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:340px}
pre{white-space:pre-wrap;color:#cbd5e1;font-size:12px;margin:0;max-height:280px;overflow:auto}
code{background:#0f1419;padding:1px 6px;border-radius:4px}a{color:#34d399}
</style></head><body>
<h1>autoresearch <span class=sub>● live · refreshes every 5s</span></h1>
$box$stats
<div class=card><h2>$metrictitle</h2>$chart</div>
<div class=grid>
<div class=card><h2>Leaderboard (best keeps)</h2>$leaderboard</div>
<div class=card><h2>Recent</h2>$recent</div></div>
<div class=card><h2>findings.md</h2><pre>$findings</pre></div>
</body></html>""")


def _render_page() -> str:
    rows = _read_results()
    state = load_state() or {}
    metric = state.get("metric", PRIMARY_METRIC_DEFAULT)
    goal = state.get("goal", "min")
    valid_keeps = [r for r in rows if r["status"] == "keep" and _fval(r.get("score")) is not None]
    keeps = sorted(valid_keeps, key=lambda r: _fval(r["score"]), reverse=(goal == "max"))
    cols = [(metric, "score"), ("mem", "memory_gb"), ("commit", "commit"),
            ("description", "description")]
    findings = (REPO / "findings.md").read_text()[-4000:] if (REPO / "findings.md").exists() else "(no findings.md yet)"
    return PAGE.safe_substitute(
        box=_box_panel(load_state()),
        stats=f'<div class=card>{_stats_panel(rows, metric, goal)}</div>',
        chart=_svg_chart(rows, metric, goal),
        metrictitle=f"{html.escape(metric)} over experiments ({'lower' if goal == 'min' else 'higher'} = better)",
        leaderboard=_table(keeps[:10], cols) or "<p class=muted>none yet</p>",
        recent=_table(list(reversed(rows))[:14],
                      [(metric, "score"), ("status", "status"), ("description", "description")]),
        findings=html.escape(findings),
    )


def cmd_dashboard(a) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                body = _render_page().encode()
            except Exception as e:  # never let a transient read crash the server
                body = f"<pre>dashboard error: {html.escape(str(e))}</pre>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet
            pass

    url = f"http://127.0.0.1:{a.port}"
    srv = HTTPServer(("127.0.0.1", a.port), Handler)
    print(f"dashboard live at {url}  (Ctrl-C to stop; reads results.tsv + .vast_state.json live)")
    if not a.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped.")


# --- misc commands -------------------------------------------------------------

def cmd_reap(a) -> None:
    """Kill a stray train.py run on the box (a ghost from a wedged/abandoned run) and show
    what's still on the GPU. Use it if a run got stuck and is holding VRAM."""
    s = require_state()
    # `[t]rain.py` matches the real train.py process but keeps the literal "train.py" out of
    # pkill's own command line, so it never SIGKILLs its own wrapper shell (see _exp_run).
    ssh_run(s, f'pkill -9 -f "{REMOTE_RUN}/[t]rain.py" 2>/dev/null; true')
    print("reaped any stray train.py run")
    out = ssh_run(s, "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory "
                     "--format=csv,noheader 2>/dev/null").stdout.strip()
    print("GPU compute processes now:\n  " + (out.replace("\n", "\n  ") if out else "(none)"))


def cmd_run(a) -> None:
    s = require_state()
    sys.exit(subprocess.run(ssh_base(s) + [f"cd {REMOTE_DIR} && {a.command}"]).returncode)


def cmd_pull(a) -> None:
    s = require_state()
    src = f"root@{s['host']}:{REMOTE_DIR}/{a.remote}"
    subprocess.run(["scp", *_ssh_opts(), "-P", str(s["port"]), "-r", src, a.local], check=True)


def cmd_log(a) -> None:
    """Atomically append one tab-separated row to results.tsv (file-locked). The agent logs
    one row per experiment: commit  score  memory_gb  status  description."""
    import fcntl
    row = "\t".join(a.fields)
    if not RESULTS.exists():
        RESULTS.write_text("\t".join(RESULT_COLS) + "\n")
    with open(RESULTS, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    print("logged:", row)


def cmd_watchdog(a) -> None:
    print("watchdog: will auto-destroy at the recorded deadline; Ctrl-C to stop.")
    while True:
        s = load_state()
        if not s:
            print("watchdog: no tracked instance; exiting.")
            return
        if time.time() >= s["deadline"]:
            print("watchdog: deadline reached -> destroying instance.")
            _destroy(s["instance_id"])
            STATE.unlink(missing_ok=True)
            return
        time.sleep(60)


def _destroy(iid) -> None:
    subprocess.run([_vastai(), "destroy", "instance", str(iid)], input="y\n", text=True)


def cmd_extend(a) -> None:
    s = require_state()
    s["deadline"] = time.time() + a.hours * 3600
    save_state(s)
    print(f"deadline extended to {a.hours}h from now")


def cmd_down(a) -> None:
    s = require_state()
    _destroy(s["instance_id"])
    STATE.unlink(missing_ok=True)
    print(f"destroyed {s['instance_id']} and cleared state.")


def cmd_ps(a) -> None:
    rows = vast(["show", "instances"], raw=True) or []
    if not rows:
        print("no instances on the account. (nothing being billed)")
        return
    print("ALL instances on the account:")
    for r in rows:
        print(f"  id={r.get('id')}  {r.get('num_gpus')}x {r.get('gpu_name')}  "
              f"{r.get('actual_status')}  ${r.get('dph_total', 0):.3f}/hr")


def cmd_nuke(a) -> None:
    rows = vast(["show", "instances"], raw=True) or []
    if not rows:
        print("nothing to nuke.")
        return
    for r in rows:
        print(f"destroying {r.get('id')} ({r.get('gpu_name')})")
        _destroy(r.get("id"))
    STATE.unlink(missing_ok=True)
    print("all instances destroyed.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        return p

    def add_rent_args(p, full):
        p.add_argument("--gpu", default=DEFAULT_GPU)
        p.add_argument("--gpus", type=int, default=1,
                       help="GPUs on the box; ONE experiment is sharded across them with "
                            "torch.distributed (NOT parallel experiments). Default 1.")
        p.add_argument("--max-price", type=float, default=0.60, help="cap in $/GPU/hr")
        if full:
            p.add_argument("--hours", type=float, default=3.0, help="session length (auto-destroy)")
            p.add_argument("--minutes", type=float, default=5.0,
                           help="per-experiment compute budget, minutes (variable per session)")
            p.add_argument("--metric", default=PRIMARY_METRIC_DEFAULT,
                           help="objective name the experiment prints (default: val_bpb)")
            p.add_argument("--goal", choices=("min", "max"), default="min",
                           help="optimize the metric to min (default) or max")
            p.add_argument("--launcher", choices=("auto", "torchrun", "single"), default="auto",
                           help="how to launch the one experiment: auto (torchrun iff >1 GPU), "
                                "torchrun (force torch.distributed), or single (one process)")
            p.add_argument("--disk", type=int, default=80)
            p.add_argument("--image", default=None, help="raw docker image (default: Vast PyTorch template)")
            p.add_argument("--offer-id", type=int, default=None, help="rent this exact offer id")

    add_rent_args(add("search", cmd_search), full=False)
    add_rent_args(add("up", cmd_up), full=True)
    stp = add("start", cmd_start)  # one-shot: up + watchdog + setup
    add_rent_args(stp, full=True)
    stp.add_argument("--num-shards", type=int, default=8)

    add("status", cmd_status)
    add("sync", cmd_sync)
    sp = add("setup", cmd_setup)
    sp.add_argument("--num-shards", type=int, default=8, help="train shards to download")
    ep = add("exp", cmd_exp)
    ep.add_argument("--train", default="train.py", help="path to the train.py to run")
    ep.add_argument("--minutes", type=float, default=None,
                    help="override this run's training budget (default: the session's)")
    ep.add_argument("--launcher", choices=("auto", "torchrun", "single"), default=None,
                    help="override this run's launcher (default: the session's)")
    add("reap", cmd_reap)
    rp = add("run", cmd_run)
    rp.add_argument("command")
    pp = add("pull", cmd_pull)
    pp.add_argument("remote")
    pp.add_argument("local")
    lp = add("log", cmd_log)
    lp.add_argument("fields", nargs="+", help="commit score memory_gb status description")
    dp = add("dashboard", cmd_dashboard)
    dp.add_argument("--port", type=int, default=8723)
    dp.add_argument("--no-open", action="store_true", help="don't auto-open a browser")
    add("watchdog", cmd_watchdog)
    xp = add("extend", cmd_extend)
    xp.add_argument("--hours", type=float, default=2.0)
    add("down", cmd_down)
    add("ps", cmd_ps)
    add("nuke", cmd_nuke)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
