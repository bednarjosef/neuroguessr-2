"""
FROZEN harness for the geolocalization study — NOT edited during research.

Caches a fixed, seed-pinned image subset once (~/.cache/autoresearch_geo/) and defines the
objective. `evaluate_geo(predict_fn, ...)`: predict_fn takes PIL images and returns
(lat, lon) in degrees; scoring (haversine + metrics) lives here so it can't be gamed.
Primary objective `median_km` (lower better), plus mean_km / acc@thresholds / geoguessr.
"""

import os
import sys
import time
import math
import signal
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify during research)
# ---------------------------------------------------------------------------

# Per-experiment training time budget in seconds. Default 8 min; the human sets it per
# session via vast.py (AR_TIME_BUDGET). Constant within a session so runs stay comparable.
TIME_BUDGET = int(os.environ.get("AR_TIME_BUDGET", "480"))

# The frozen, seed-pinned data subset. Fixed so every experiment sees the SAME images.
DATASET_NAME = os.environ.get("AR_DATASET", "josefbednar/streetview-acw-300k")
N_TRAIN = int(os.environ.get("AR_N_TRAIN", "60000"))   # cached training pool (fixed subset)
N_VAL = int(os.environ.get("AR_N_VAL", "3000"))        # full held-out val split
DATA_SEED = 1337                                        # frozen sampling seed
SAVE_MAX_SIDE = 512                                     # cap stored JPEG long side (disk/speed)
# Fixed subset of val used for the fast every-N-steps monitoring eval during training.
# The FINAL official score always uses the full val split.
QUICK_VAL_N = int(os.environ.get("AR_QUICK_VAL_N", "1000"))

# Earth radius (km) for the great-circle distance.
EARTH_RADIUS_KM = 6371.0088
# GeoGuessr score constant: 5000 * exp(-distance_km / 1492.7).
GEOGUESSR_TAU_KM = 1492.7
# Accuracy thresholds (km): street / city / region / country / continent.
ACC_THRESHOLDS_KM = [1.0, 25.0, 200.0, 750.0, 2500.0]

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_geo")
TRAIN_IMG_DIR = os.path.join(CACHE_DIR, "train")
VAL_IMG_DIR = os.path.join(CACHE_DIR, "val")
TRAIN_META = os.path.join(CACHE_DIR, "train.parquet")
VAL_META = os.path.join(CACHE_DIR, "val.parquet")

# Metadata columns we keep if present (image + coords are required; rest are optional
# side-information train.py may exploit).
META_COLS = ["latitude", "longitude", "country_code", "subdivision", "date",
             "elevation", "season", "heading", "panoid", "image_id"]

# ---------------------------------------------------------------------------
# Great-circle distance + metrics (FROZEN — this is the objective)
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km. Inputs in degrees (numpy arrays or scalars)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, dtype=np.float64),
                                              np.asarray(lon1, dtype=np.float64),
                                              np.asarray(lat2, dtype=np.float64),
                                              np.asarray(lon2, dtype=np.float64)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def geo_metrics(pred_lat, pred_lon, true_lat, true_lon):
    """Compute the full panel of geolocation metrics from predicted vs true coordinates.
    Returns a dict; `median_km` is the primary objective (lower is better)."""
    d = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
    out = {
        "median_km": float(np.median(d)),
        "mean_km": float(np.mean(d)),
        "geoguessr_score": float(np.mean(5000.0 * np.exp(-d / GEOGUESSR_TAU_KM))),
    }
    names = {1.0: "acc_1km", 25.0: "acc_25km", 200.0: "acc_200km",
             750.0: "acc_750km", 2500.0: "acc_2500km"}
    for thr in ACC_THRESHOLDS_KM:
        out[names[thr]] = float(np.mean(d <= thr))
    return out

# ---------------------------------------------------------------------------
# Data download + caching (one-time; frozen subset)
# ---------------------------------------------------------------------------

def _resize_for_storage(img):
    """Downscale so the long side <= SAVE_MAX_SIDE (keeps disk + decode cheap). RGB."""
    from PIL import Image
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > SAVE_MAX_SIDE:
        s = SAVE_MAX_SIDE / m
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return img


def _find_split(candidates):
    """Return the first streaming split that loads, else None."""
    from datasets import load_dataset
    for name in candidates:
        try:
            ds = load_dataset(DATASET_NAME, split=name, streaming=True)
            _ = next(iter(ds))  # force a real fetch so bad split names raise
            return name
        except Exception as e:
            print(f"  split '{name}' unavailable ({type(e).__name__}); trying next")
    return None


def _download_split(split_name, n, img_dir, meta_path, shuffle):
    """Stream `n` rows from `split_name`, save JPEGs + a metadata parquet. Idempotent."""
    import pandas as pd
    from datasets import load_dataset

    if os.path.exists(meta_path):
        df = pd.read_parquet(meta_path)
        if len(df) >= n or len(df) >= QUICK_VAL_N:
            print(f"  {split_name}: {len(df)} rows already cached at {meta_path}")
            return
    os.makedirs(img_dir, exist_ok=True)

    from PIL import Image as _PILImage
    ds = load_dataset(DATASET_NAME, split=split_name, streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=DATA_SEED, buffer_size=20000)

    # Auto-detect the image column (a PIL image value) from the first row.
    first = next(iter(ds))
    img_key = next((k for k, v in first.items() if isinstance(v, _PILImage.Image)), None)
    if img_key is None:
        img_key = "image"
    print(f"  {split_name}: image column = '{img_key}'")

    rows = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        if len(rows) >= n:
            break
        try:
            lat = float(ex["latitude"]); lon = float(ex["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        img = ex.get(img_key)
        if img is None:
            continue
        fname = f"{len(rows):07d}.jpg"
        fpath = os.path.join(img_dir, fname)
        try:
            _resize_for_storage(img).save(fpath, "JPEG", quality=90)
        except Exception as e:
            print(f"  skip image {i}: {e}")
            continue
        rec = {"path": fname, "latitude": lat, "longitude": lon}
        for c in META_COLS:
            if c in ("latitude", "longitude"):
                continue
            if c in ex:
                rec[c] = ex[c]
        rows.append(rec)
        if len(rows) % 2000 == 0:
            rate = len(rows) / (time.time() - t0 + 1e-9)
            print(f"  {split_name}: {len(rows)}/{n} ({rate:.0f} img/s)")

    df = pd.DataFrame(rows)
    df.to_parquet(meta_path, index=False)
    print(f"  {split_name}: cached {len(df)} rows -> {meta_path}  ({time.time()-t0:.0f}s)")


DOWNLOAD_WORKERS = int(os.environ.get("AR_DOWNLOAD_WORKERS", "16"))
SHARD_SHUFFLE_BUFFER = 1000   # per-worker shuffle buffer (part of the frozen subset definition)


def _stream_shard_worker(args):
    """One parallel stream: worker widx takes every nshards-th parquet file of the train
    split (deterministic), shuffles within its stream (seed-pinned), and saves its quota.
    The union over workers IS the frozen train subset — fixed given (nshards, seed, quota)."""
    widx, nshards, quota, img_dir = args
    # Each worker persists its finished rows as a part-parquet: reruns skip completed
    # workers entirely (no re-download), and the part is the durable record of the rows.
    # Deterministic (shard, seed) stream -> identical rows on any retry. Subset unchanged.
    import pandas as pd
    part_path = os.path.join(CACHE_DIR, f"train_part_w{widx:02d}.parquet")
    if os.path.exists(part_path):
        try:
            part = pd.read_parquet(part_path)
            if len(part) >= quota:
                print(f"  train[w{widx:02d}]: part cached ({len(part)} rows), skipping", flush=True)
                return part.to_dict("records")
        except Exception:
            pass
    import socket
    socket.setdefaulttimeout(90)
    from datasets import load_dataset
    for attempt in range(5):
        try:
            ds = load_dataset(DATASET_NAME, split="train", streaming=True)
            ds = ds.shard(num_shards=nshards, index=widx)
            ds = ds.shuffle(seed=DATA_SEED + widx, buffer_size=SHARD_SHUFFLE_BUFFER)
            rows = []
            t0 = time.time()
            for ex in ds:
                if len(rows) >= quota:
                    break
                try:
                    lat = float(ex["latitude"]); lon = float(ex["longitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                img = ex.get("image")
                if img is None:
                    continue
                fname = f"w{widx:02d}_{len(rows):06d}.jpg"
                try:
                    _resize_for_storage(img).save(os.path.join(img_dir, fname), "JPEG", quality=90)
                except Exception:
                    continue
                rec = {"path": fname, "latitude": lat, "longitude": lon}
                for c in META_COLS:
                    if c not in ("latitude", "longitude") and c in ex:
                        rec[c] = ex[c]
                rows.append(rec)
                if widx == 0 and len(rows) % 250 == 0:
                    rate = len(rows) / (time.time() - t0 + 1e-9)
                    print(f"  train[w0]: {len(rows)}/{quota} ({rate:.1f} img/s/worker, ~{rate*nshards:.0f} img/s total)",
                          flush=True)
            pd.DataFrame(rows).to_parquet(part_path, index=False)
            print(f"  train[w{widx:02d}]: done ({len(rows)} rows -> part)", flush=True)
            return rows
        except Exception as e:
            print(f"  train[w{widx:02d}]: stream died ({type(e).__name__}: {e}); retry {attempt+1}/5",
                  flush=True)
    raise RuntimeError(f"worker {widx}: stream failed 5x")


def _download_train_parallel(n, img_dir, meta_path):
    """Parallel sharded train download: DOWNLOAD_WORKERS concurrent streams, each over a
    disjoint slice of the split's files. ~WORKERSx faster than one stream. Idempotent."""
    import pandas as pd
    from datasets import load_dataset
    if os.path.exists(meta_path):
        df = pd.read_parquet(meta_path)
        if len(df) >= n:
            print(f"  train: {len(df)} rows already cached at {meta_path}")
            return
    os.makedirs(img_dir, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    nshards = max(1, min(DOWNLOAD_WORKERS, getattr(ds, "num_shards", 1) or 1))
    quota = -(-n // nshards)
    print(f"  train: {nshards} parallel shard streams x {quota} rows (seed={DATA_SEED})")
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(nshards) as pool:
        parts = pool.map(_stream_shard_worker,
                         [(i, nshards, quota, img_dir) for i in range(nshards)])
    rows = [r for part in parts for r in part][:n]
    df = pd.DataFrame(rows)
    df.to_parquet(meta_path, index=False)
    print(f"  train: cached {len(df)} rows -> {meta_path}  ({time.time()-t0:.0f}s)")


def download_data():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cache:   {CACHE_DIR}")

    if _try_cache_tar():
        return

    val_split = _find_split(["validation", "val", "test", "valid"])
    if val_split is None:
        sys.exit("Could not find a validation split (tried validation/val/test/valid).")
    print(f"Downloading val split '{val_split}' (up to {N_VAL})...")
    _download_split(val_split, N_VAL, VAL_IMG_DIR, VAL_META, shuffle=False)

    print(f"Downloading train subset ({N_TRAIN} images, parallel)...")
    _download_train_parallel(N_TRAIN, TRAIN_IMG_DIR, TRAIN_META)

CACHE_TAR_REPO = "josefbednar/streetview-acw-ar-cache"   # prebuilt-cache fast path
CACHE_TAR_NAME = f"cache_n{N_TRAIN}_v{N_VAL}_s{DATA_SEED}.tar"


def _cache_complete():
    """True if the local cache already holds the full frozen subset."""
    import pandas as pd
    try:
        return (len(pd.read_parquet(TRAIN_META)) >= N_TRAIN
                and len(pd.read_parquet(VAL_META)) >= N_VAL)
    except Exception:
        return False


def _try_cache_tar():
    """Fast path: pull the prebuilt cache tar (identical bytes to a fresh streaming
    download of the frozen subset) from HF. Falls back to streaming on any failure."""
    if _cache_complete():
        print("  cache already complete — skipping download")
        return True
    try:
        from huggingface_hub import hf_hub_download
        print(f"  trying prebuilt cache tar {CACHE_TAR_REPO}/{CACHE_TAR_NAME} ...")
        t0 = time.time()
        tar_path = hf_hub_download(repo_id=CACHE_TAR_REPO, repo_type="dataset",
                                   filename=CACHE_TAR_NAME)
        import tarfile
        with tarfile.open(tar_path) as tf:
            tf.extractall(os.path.dirname(CACHE_DIR))
        ok = _cache_complete()
        print(f"  cache tar {'restored' if ok else 'INCOMPLETE'} in {time.time()-t0:.0f}s")
        return ok
    except Exception as e:
        print(f"  no prebuilt cache tar ({type(e).__name__}); falling back to streaming")
        return False

# ---------------------------------------------------------------------------
# Data access helpers (imported by train.py — train.py builds its own dataloader)
# ---------------------------------------------------------------------------

def load_index(split):
    """Return (image_dir, DataFrame) for 'train' or 'val'. The df has at least
    path/latitude/longitude, plus any available side columns."""
    import pandas as pd
    if split == "train":
        return TRAIN_IMG_DIR, pd.read_parquet(TRAIN_META)
    elif split == "val":
        return VAL_IMG_DIR, pd.read_parquet(VAL_META)
    raise ValueError(split)


def open_image(img_dir, path):
    from PIL import Image
    img = Image.open(os.path.join(img_dir, path))
    return img.convert("RGB") if img.mode != "RGB" else img

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_geo(predict_fn, split="val", subset=None, batch_size=64):
    """Score predict_fn's coordinates against ground truth. predict_fn(pil_images) ->
    (lat, lon) in degrees. subset=int: first N rows (fast monitoring); None: full split
    (official score). Returns the geo_metrics dict."""
    img_dir, df = load_index(split)
    if subset is not None:
        df = df.iloc[:subset]
    n = len(df)
    paths = df["path"].tolist()
    true_lat = df["latitude"].to_numpy(dtype=np.float64)
    true_lon = df["longitude"].to_numpy(dtype=np.float64)

    pred_lat = np.empty(n, dtype=np.float64)
    pred_lon = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        imgs = [open_image(img_dir, paths[j]) for j in range(start, end)]
        lat, lon = predict_fn(imgs)
        pred_lat[start:end] = np.asarray(lat, dtype=np.float64).reshape(-1)
        pred_lon[start:end] = np.asarray(lon, dtype=np.float64).reshape(-1)
    return geo_metrics(pred_lat, pred_lon, true_lat, true_lon)

# ---------------------------------------------------------------------------
# Hard training-time deadline (DO NOT CHANGE)
# ---------------------------------------------------------------------------
# SIGALRM backstop: raises TrainingTimeUp in the main thread at TIME_BUDGET + grace, so
# train.py stops even on a hang/overrun and still reaches the final eval. Wrap the loop in
# try/except TrainingTimeUp.

class TrainingTimeUp(Exception):
    pass


def start_training_clock(grace_seconds=45):
    """Arm the wall-clock cap; call right before the training loop."""
    def _on_deadline(signum, frame):
        raise TrainingTimeUp()
    signal.signal(signal.SIGALRM, _on_deadline)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, TIME_BUDGET + grace_seconds))


def stop_training_clock():
    """Disarm the cap; call after the loop, before eval."""
    signal.setitimer(signal.ITIMER_REAL, 0.0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare geolocalization data for autoresearch")
    # --num-shards is accepted for vast.py setup-script compatibility but ignored here;
    # the subset size is controlled by AR_N_TRAIN / AR_N_VAL.
    parser.add_argument("--num-shards", type=int, default=8, help="(ignored; kept for compatibility)")
    parser.add_argument("--download-workers", type=int, default=8, help="(ignored)")
    args = parser.parse_args()

    download_data()
    print()
    print("Done! Ready to train.")
