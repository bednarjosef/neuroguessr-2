#!/usr/bin/env python3
"""C5 launch gate: validate everything locally BEFORE renting (research/plans/C5_PLAN.md).
The check that would have caught the C4-eval centroids.npz crash in 5 seconds."""
import os
import py_compile
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'XX'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("compile:")
for f in ("train_c5.py", "retrieval/embed_c2.py", "retrieval/run_c5.sh",
          "tools/build_episode_cache.py", "tools/dump_val_patches.py",
          "tools/check_bed_manifest.py", "retrieval/train_geo_head.py",
          "tools/screen_gpus.py", "tools/restore_full_cache.py"):
    p = os.path.join(REPO, f)
    if not os.path.exists(p):
        check(f, False, "missing")
        continue
    if f.endswith(".py"):
        try:
            py_compile.compile(p, doraise=True)
            check(f, True)
        except py_compile.PyCompileError as e:
            check(f, False, str(e)[:120])
    else:
        r = subprocess.run(["bash", "-n", p], capture_output=True)
        check(f, r.returncode == 0, r.stderr.decode()[:120])

print("data:")
import numpy as np  # noqa: E402
for split, nrows in (("train", 1_198_072), ("val", 2_998)):
    p = os.path.join(REPO, "data", f"attr_labels_{split}.npz")
    if not os.path.exists(p):
        check(f"attr_labels_{split}", False, "missing")
        continue
    z = np.load(p, allow_pickle=True)
    check(f"attr_labels_{split}", len(z["lat"]) == nrows,
          f"{len(z['lat'])} rows, drive_left {100*z['drive_left'].mean():.1f}%")

print("secrets/env:")
env = {}
envp = os.path.join(REPO, ".env")
if os.path.exists(envp):
    for line in open(envp):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
check("HF_TOKEN in .env", bool(env.get("HF_TOKEN")))
check("WANDB_API_KEY in .env", bool(env.get("WANDB_API_KEY")))

print("account:")
try:
    vast = subprocess.run(["vastai", "show", "user", "--raw"], capture_output=True, text=True,
                          timeout=30)
    import json
    d = json.loads(vast.stdout)
    bal = d.get("balance", 0) + d.get("credit", 0)
    check("balance >= $6.5", bal >= 6.5, f"${bal:.2f}")
except Exception as e:
    check("balance check", False, str(e)[:80])

print("HF reachability (auth):")
try:
    import requests
    r = requests.head("https://huggingface.co/datasets/josefbednar/streetview-acw-ar-cache/"
                      "resolve/main/cache_n1198072_v3000_s1337.tar",
                      headers={"Authorization": f"Bearer {env.get('HF_TOKEN', '')}"},
                      timeout=20, allow_redirects=True)
    check("cache tar HEAD", r.status_code == 200, f"http {r.status_code}")
except Exception as e:
    check("cache tar HEAD", False, str(e)[:80])

print()
if fails:
    print(f"PREFLIGHT FAILED: {fails}")
    sys.exit(1)
print("PREFLIGHT OK — C5 may launch")
