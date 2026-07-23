#!/usr/bin/env python3
"""Recompute the champion (geo10|c1 blend, mass 0.95, lam 0.05, k=1 snap) on full val,
locally on CPU, and report the full distance distribution + GeoGuessr score."""
import os
import time

import numpy as np

REPO = "/home/josef/everything/coding/neuroguessr-2-research"
IX = os.path.join(REPO, "run_full2", "retrieval_index")
EARTH = 6371.0088


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, map(np.asarray, (lat1, lon1, lat2, lon2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cat(prefix):
    fs = sorted([f for f in os.listdir(IX) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(IX, f)) for f in fs])


t0 = time.time()
z = np.load(os.path.join(IX, "train_latlon.npz"))
trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
cell_a = np.load(os.path.join(IX, "cell_a.npy"))
vz = np.load(os.path.join(REPO, "run_full", "val_analysis", "val_meta.npz"))
tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
n = len(tlat)

lga = cat("logits_a_val_r").astype(np.float64)
lga -= lga.max(1, keepdims=True)
p = np.exp(lga)
p /= p.sum(1, keepdims=True)
logp = np.log(np.clip(p, 1e-12, None))
rank = np.argsort(-p, axis=1)
csum = np.take_along_axis(p, rank, 1).cumsum(1)
m_of = (csum < 0.95).sum(1) + 1

# cell-bucketed order
order = np.argsort(cell_a, kind="stable")
cs = cell_a[order]
C = p.shape[1]
starts = np.searchsorted(cs, np.arange(C), "left")
ends = np.searchsorted(cs, np.arange(C), "right")

geo_t = np.load(os.path.join(IX, "geo_head_10_train.npy"), mmap_mode="r")
c1_t = np.load(os.path.join(IX, "train_proj.npy"), mmap_mode="r")
geo_v = np.load(os.path.join(IX, "geo_head_10_val.npy")).astype(np.float32)
c1_v = np.load(os.path.join(IX, "val_proj.npy")).astype(np.float32)
print(f"loaded ({time.time()-t0:.0f}s) N={len(trlat)} val={n}", flush=True)

pred = np.zeros((n, 2))
nc = np.zeros(n)
for i in range(n):
    m = min(int(m_of[i]), 400)
    cells = rank[i, :m]
    ids = np.concatenate([order[starts[c]:ends[c]] for c in cells])
    ids.sort()
    nc[i] = len(ids)
    s = 0.5 * (geo_t[ids].astype(np.float32) @ geo_v[i]) + 0.5 * (c1_t[ids].astype(np.float32) @ c1_v[i])
    s = s.astype(np.float64) + 0.05 * logp[i, cell_a[ids]]
    j = ids[int(s.argmax())]
    pred[i] = (trlat[j], trlon[j])
    if i % 250 == 0:
        print(f"  {i}/{n} ({time.time()-t0:.0f}s, cand~{nc[:i+1].mean():.0f})", flush=True)

d = hav(pred[:, 0], pred[:, 1], tlat, tlon)
np.save(os.path.join(IX, "champion_dist_km.npy"), d)
np.save(os.path.join(IX, "champion_pred.npy"), pred)

gg = 5000 * np.exp(-d / 1492.7)
print("\n=== CHAMPION geo10|c1  m0.95 lam0.05 k1 (full val, n=%d) ===" % n)
print(f"median {np.median(d):.2f} km | mean {d.mean():.1f} km | cand/query {nc.mean():.0f}")
for r in (0.1, 0.5, 1, 5, 10, 25, 50, 100, 200, 500, 750, 1000, 2500):
    print(f"  @{r:>6} km : {100*(d<=r).mean():5.2f}%")
for q in (10, 25, 50, 75, 90, 95, 99):
    print(f"  p{q:<3d} : {np.percentile(d, q):9.1f} km")
print(f"\nGeoGuessr (world map, 5000*exp(-d/1492.7)):")
print(f"  mean per round : {gg.mean():.0f} / 5000")
print(f"  median round   : {np.median(gg):.0f}")
print(f"  per 5-round game: {5*gg.mean():.0f} / 25000")
print(f"  rounds >=4500  : {100*(gg>=4500).mean():.1f}%   >=4900: {100*(gg>=4900).mean():.1f}%")
print("DONE")
