#!/usr/bin/env python3
"""Build the prebuilt-cache tar that prepare.py's fast path looks for, and push it to HF.

Why: a fresh box spends ~28 min streaming 115 GB of raw JPEGs before any GPU work starts
(~$0.9 per session, every session). prepare.py already tries
  {CACHE_TAR_REPO}/{CACHE_TAR_NAME}  (dataset repo)
and extracts it into the parent of CACHE_DIR before falling back to streaming — this script
produces exactly that artefact from a box that already holds the cache.

Re-encoding to 384 px (--size 384) roughly halves the tar (~115 GB -> ~45 GB), which is both a
faster upload once and a faster download for every future session. Training and every sweep
resize to 384 anyway, so the pixels the model sees are unchanged; --size 0 keeps the bytes
exactly as downloaded if you want a byte-identical cache.

Run ON THE BOX, after prepare.py has completed:
  python tools/build_cache_tar.py --size 384 --workers 96          # build + verify + upload
  python tools/build_cache_tar.py --size 384 --limit 200 --no-upload  # 1-minute smoke test
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare as P


def _reencode(args):
    src, dst, size = args
    from PIL import Image
    try:
        im = Image.open(src)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if size and im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        im.save(dst, "JPEG", quality=92, optimize=False)
        return 1
    except Exception:
        return 0


def build_stage(stage, size, workers, limit):
    """Materialise a cache-shaped copy of CACHE_DIR under `stage` (re-encoded if size>0)."""
    root = os.path.join(stage, os.path.basename(P.CACHE_DIR))
    os.makedirs(root, exist_ok=True)
    for f in os.listdir(P.CACHE_DIR):
        p = os.path.join(P.CACHE_DIR, f)
        if os.path.isfile(p):                      # train.parquet, val.parquet, part parquets
            shutil.copy2(p, os.path.join(root, f))
    total = 0
    for split in ("train", "val"):
        src_dir = os.path.join(P.CACHE_DIR, split)
        dst_dir = os.path.join(root, split)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        names = sorted(os.listdir(src_dir))
        if limit:
            names = names[:limit]
        if not size:
            for n in names:
                os.link(os.path.join(src_dir, n), os.path.join(dst_dir, n))
            total += len(names)
            continue
        jobs = [(os.path.join(src_dir, n), os.path.join(dst_dir, n), size) for n in names]
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, ok in enumerate(ex.map(_reencode, jobs, chunksize=256), 1):
                done += ok
                if i % 50_000 == 0:
                    r = i / (time.time() - t0)
                    print(f"  {split}: {i}/{len(jobs)} ({r:.0f} img/s, "
                          f"ETA {(len(jobs)-i)/r/60:.1f} min)", flush=True)
        print(f"  {split}: {done}/{len(names)} re-encoded in {(time.time()-t0)/60:.1f} min",
              flush=True)
        total += done
    return root, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384, help="re-encode to NxN (0 = keep bytes)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N images per split")
    ap.add_argument("--stage", default="/root/cache_stage")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()

    name = P.CACHE_TAR_NAME if not a.limit else P.CACHE_TAR_NAME.replace(".tar", "_smoke.tar")
    out = a.out or os.path.join("/root", name)
    print(f"cache dir  : {P.CACHE_DIR}", flush=True)
    print(f"target tar : {out}  ->  {P.CACHE_TAR_REPO} (dataset)", flush=True)

    if os.path.exists(a.stage):
        shutil.rmtree(a.stage)
    os.makedirs(a.stage)
    t0 = time.time()
    root, n = build_stage(a.stage, a.size, a.workers, a.limit)
    print(f"staged {n} images in {(time.time()-t0)/60:.1f} min", flush=True)

    t0 = time.time()
    with tarfile.open(out, "w") as tf:             # no compression: JPEGs are already compressed
        tf.add(root, arcname=os.path.basename(P.CACHE_DIR))
    gb = os.path.getsize(out) / 2 ** 30
    print(f"tar {gb:.1f} GB in {(time.time()-t0)/60:.1f} min", flush=True)

    # verify: the tar must extract to a layout prepare.py accepts
    ver = "/root/cache_verify"
    if os.path.exists(ver):
        shutil.rmtree(ver)
    os.makedirs(ver)
    with tarfile.open(out) as tf:
        members = tf.getnames()
    top = {m.split("/")[0] for m in members}
    need = {os.path.basename(P.CACHE_DIR)}
    assert top == need, f"tar root is {top}, prepare.py extracts expecting {need}"
    assert any(m.endswith("train.parquet") for m in members), "train.parquet missing from tar"
    assert any(m.endswith("val.parquet") for m in members), "val.parquet missing from tar"
    print(f"VERIFY OK — {len(members)} members, root '{list(top)[0]}/'", flush=True)
    shutil.rmtree(ver, ignore_errors=True)

    if a.no_upload:
        print("skipping upload (--no-upload)", flush=True)
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(P.CACHE_TAR_REPO, repo_type="dataset", exist_ok=True, private=True)
    t0 = time.time()
    api.upload_file(path_or_fileobj=out, path_in_repo=name,
                    repo_id=P.CACHE_TAR_REPO, repo_type="dataset")
    print(f"uploaded {gb:.1f} GB in {(time.time()-t0)/60:.1f} min "
          f"({gb*1024/(time.time()-t0):.0f} MB/s)", flush=True)
    print("CACHE TAR DONE", flush=True)


if __name__ == "__main__":
    main()
