"""
One-time data preparation + FROZEN evaluator for autoresearch (image geolocalization).

Downloads a FIXED, seed-pinned subset of the streetview dataset once and caches it to
local JPEGs + a metadata parquet, so every experiment trains/evaluates on exactly the
same images (comparable scores, no re-downloading 1.2M rows). Also defines the frozen,
un-gameable objective: the great-circle (haversine) error between the model's predicted
lat/lon and the ground truth on the held-out val split.

Usage:
    python prepare.py                 # full prep (download subset + val)
    python prepare.py --num-shards 8  # arg accepted for vast.py compatibility (ignored)

Everything is cached under ~/.cache/autoresearch_geo/.

CONTRACT (frozen during research — agents edit train.py only):
    - The train subset (which images) and the val split are fixed here.
    - `evaluate_geo(predict_fn, ...)` computes the objective. `predict_fn` takes a list of
      PIL images and returns (lat_array, lon_array) in degrees. The model ONLY ever emits
      coordinates; the scoring (haversine + metrics) lives here and cannot be gamed from
      train.py.
    - `median_km` (lower is better) is the primary objective. mean_km, acc@{1,25,200,750,
      2500}km and the GeoGuessr game score are computed too, as diagnostics.
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


def download_data():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cache:   {CACHE_DIR}")

    val_split = _find_split(["validation", "val", "test", "valid"])
    if val_split is None:
        sys.exit("Could not find a validation split (tried validation/val/test/valid).")
    print(f"Downloading val split '{val_split}' (up to {N_VAL})...")
    _download_split(val_split, N_VAL, VAL_IMG_DIR, VAL_META, shuffle=False)

    print(f"Downloading train subset (seed={DATA_SEED}, {N_TRAIN} images)...")
    _download_split("train", N_TRAIN, TRAIN_IMG_DIR, TRAIN_META, shuffle=True)

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
    """FROZEN evaluator. Feeds val images to the model's `predict_fn` and scores its
    coordinate predictions against ground truth.

    predict_fn(pil_images: list[PIL.Image]) -> (lat, lon) as array-likes of length B,
    in degrees. The model may preprocess however it likes internally; it only ever
    returns coordinates, so the objective can't be gamed from train.py.

    `subset`: if an int, evaluate on the first `subset` rows (fixed, for fast in-training
    monitoring). None = full split (the official score). Returns the geo_metrics dict.
    """
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
# Hard training-time deadline (DO NOT CHANGE — keeps the per-experiment limit honest)
# ---------------------------------------------------------------------------
# train.py self-stops via its per-step time check, but that can drift (a slow step, an
# uncounted warm-up, or a hang). This arms a real wall-clock alarm: when it fires, SIGALRM
# raises TrainingTimeUp in the main thread, so training stops promptly and gracefully —
# train.py catches it and still runs the final eval, so you always get a score.

class TrainingTimeUp(Exception):
    """Raised in the main thread when the hard training-time deadline fires."""


def start_training_clock(grace_seconds=45):
    """Arm a HARD wall-clock cap on the training phase. Call right before the training loop.
    Fires at TIME_BUDGET + grace; the grace covers uncounted warm-up/compile so a healthy
    run stops via its own per-step check first and the alarm only bites on overruns/hangs."""
    def _on_deadline(signum, frame):
        raise TrainingTimeUp()
    signal.signal(signal.SIGALRM, _on_deadline)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, TIME_BUDGET + grace_seconds))


def stop_training_clock():
    """Disarm the deadline. Call after the loop, before eval, so the alarm can't interrupt
    evaluation."""
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
