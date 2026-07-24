#!/usr/bin/env python3
"""Restore the FULL 1.2M-image 384px prepare-cache from the prebuilt HF tar.

vast.py setup runs prepare.py with the default AR_N_TRAIN=60000, which restores only the
60k ratchet subset — the retrieval index needs all 1,198,072 images. This replicates the
2026-07-24 morning restore (c4/cache_restore.log: downloaded 463s, ready 674s) with an
explicit CACHE READY marker so drivers can grep for success instead of trusting exit codes.
Deletes the tar after extraction (tar 45GB + extracted 45GB would not fit an 80GB disk).
"""
import os
import tarfile
import time

import pandas as pd
from huggingface_hub import hf_hub_download

CACHE = "/root/.cache/autoresearch_geo"
t0 = time.time()
p = hf_hub_download("josefbednar/streetview-acw-ar-cache",
                    "cache_n1198072_v3000_s1337.tar", repo_type="dataset")
print(f"downloaded {time.time() - t0:.0f} s", flush=True)
with tarfile.open(p) as tf:
    tf.extractall(os.path.dirname(CACHE))
os.remove(p)
tr = pd.read_parquet(os.path.join(CACHE, "train.parquet"))
va = pd.read_parquet(os.path.join(CACHE, "val.parquet"))
print("train", len(tr), "val", len(va), f"{time.time() - t0:.0f} s", flush=True)
assert len(tr) == 1198072 and len(va) == 2998, "row counts wrong — refusing to continue"
print("CACHE READY", flush=True)
