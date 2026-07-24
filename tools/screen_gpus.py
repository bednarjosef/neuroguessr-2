#!/usr/bin/env python3
"""Pre-run GPU health screen (CLAUDE.md 2026-07-24 lesson): every GPU must deliver real
matmul throughput at healthy clocks BEFORE any long run is committed to the box.

The 2026-07-24 lemon (GPU 0 at 547/3090 MHz under full power draw, ~1/7 throughput) passes
every rent-time filter and only shows up under load — this catches it in ~1 minute.

Exit 0 + "SCREEN PASS" iff all GPUs are healthy; exit 1 + "SCREEN FAIL [ids]" otherwise.
"""
import subprocess
import sys
import time

import torch

MIN_TFLOPS = 80.0   # healthy 5090 dense bf16 reads ~150-250; the lemon would read ~30
REL = 0.6           # every GPU must reach 60% of the best sibling

n = torch.cuda.device_count()
res = []
for i in range(n):
    torch.cuda.set_device(i)
    a = torch.randn(8192, 8192, device=f"cuda:{i}", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=f"cuda:{i}", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(i)
    t0, it = time.time(), 30
    for _ in range(it):
        a @ b
    torch.cuda.synchronize(i)
    tf = 2 * 8192 ** 3 * it / (time.time() - t0) / 1e12
    res.append(tf)
    print(f"GPU{i}: {tf:.0f} TFLOPs bf16", flush=True)
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu",
     "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
bad = [i for i, t in enumerate(res) if t < MIN_TFLOPS or t < REL * max(res)]
print(f"SCREEN {'FAIL ' + str(bad) if bad else 'PASS'} ({n} GPUs)", flush=True)
sys.exit(1 if bad else 0)
