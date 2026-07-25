#!/usr/bin/env python3
"""Local no-GPU A/B tests on the fully-materialised S384 index (run_full2/retrieval_index).

Phase 1: replicate score_champion.py's gated candidate scoring (geo10|c1 blend, mass .95,
lam .05) but keep the TOP-100 candidates per query, with all score components + coords.
Phase 2: on those cached pools, evaluate:
  E0 baseline argmax                  (sanity: must match champion_dist_km.npy 49.04/41.16)
  E1 geo-consensus rerank             (candidate cluster agreement — hand-crafted reranker)
  E2 1-step graph diffusion           (k-reciprocal-lite on the pool)
  E3 query expansion, max-fuse        (SuperGlobal-style, query side only)
  E4 abstain -> classifier fallback   (mean-killer: low-consensus queries take cell mode)
Deltas measured here transfer directionally to the C4 index when it lands.
"""
import os
import time

import numpy as np

REPO = "/home/josef/everything/coding/neuroguessr-2-research"
IX = os.path.join(REPO, "run_full2", "retrieval_index")
SP = os.path.dirname(os.path.abspath(__file__))
EP = os.path.join(SP, "episodes_s384.npz")
EARTH = 6371.0088
K = 100


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, map(np.asarray, (lat1, lon1, lat2, lon2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cat(prefix, ix=IX):
    fs = sorted([f for f in os.listdir(ix) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(ix, f)) for f in fs])


def report(tag, d, extra=""):
    gg = 5000 * np.exp(-d / 1492.7)
    print(f"{tag:28s} median {np.median(d):7.2f} | mean {d.mean():7.1f} | "
          f"@1 {100*(d<=1).mean():5.2f} | @25 {100*(d<=25).mean():5.2f} | "
          f"@200 {100*(d<=200).mean():5.2f} | GG {gg.mean():4.0f} {extra}", flush=True)


t0 = time.time()
z = np.load(os.path.join(IX, "train_latlon.npz"))
trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
cell_a = np.load(os.path.join(IX, "cell_a.npy"))
vz = np.load(os.path.join(REPO, "run_full", "val_analysis", "val_meta.npz"))
tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
n = len(tlat)
cz = np.load(os.path.join(REPO, "run_full", "val_analysis", "cells.npz"))
clat, clon = cz["lat"].astype(np.float64), cz["lon"].astype(np.float64)

lga = cat("logits_a_val_r").astype(np.float64)
lga -= lga.max(1, keepdims=True)
p = np.exp(lga)
p /= p.sum(1, keepdims=True)
logp = np.log(np.clip(p, 1e-12, None))
rank = np.argsort(-p, axis=1)
csum = np.take_along_axis(p, rank, 1).cumsum(1)
m_of = (csum < 0.95).sum(1) + 1

# classifier-only fallback prediction: mode-seeking spherical mean of top-16 cells w/in 1000 km
T = 0.5
ps = np.exp(lga / T)
ps /= ps.sum(1, keepdims=True)
cls_pred = np.zeros((n, 2))
for i in range(n):
    top = rank[i, :16]
    keep = top[hav(clat[top[0]], clon[top[0]], clat[top], clon[top]) <= 1000.0]
    w = ps[i, keep]
    w = w / w.sum()
    la, lo = np.radians(clat[keep]), np.radians(clon[keep])
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    m = (w[:, None] * v).sum(0)
    m /= np.linalg.norm(m)
    cls_pred[i] = (np.degrees(np.arcsin(m[2])), np.degrees(np.arctan2(m[1], m[0])))

if os.path.exists(EP):
    e = np.load(EP)
    ids_k, s_k, bb_k = e["ids"], e["s"], e["bb"]
    print(f"episodes loaded from cache ({time.time()-t0:.0f}s)", flush=True)
else:
    order = np.argsort(cell_a, kind="stable")
    cs = cell_a[order]
    C = p.shape[1]
    starts = np.searchsorted(cs, np.arange(C), "left")
    ends = np.searchsorted(cs, np.arange(C), "right")

    geo_t = np.load(os.path.join(IX, "geo_head_10_train.npy"), mmap_mode="r")
    c1_t = np.load(os.path.join(IX, "train_proj.npy"), mmap_mode="r")
    geo_v = np.load(os.path.join(IX, "geo_head_10_val.npy")).astype(np.float32)
    c1_v = np.load(os.path.join(IX, "val_proj.npy")).astype(np.float32)
    bb_t = np.lib.format.open_memmap
    bb_train = [np.load(os.path.join(IX, f"emb_bb_train_r{r}.npy"), mmap_mode="r") for r in range(4)]
    bb_lens = np.cumsum([0] + [len(b) for b in bb_train])
    bb_v = cat("emb_bb_val_r").astype(np.float32)
    bb_v /= np.linalg.norm(bb_v, axis=1, keepdims=True) + 1e-9

    def bb_rows(ids):
        out = np.empty((len(ids), bb_train[0].shape[1]), np.float32)
        for r in range(4):
            m = (ids >= bb_lens[r]) & (ids < bb_lens[r + 1])
            if m.any():
                out[m] = bb_train[r][ids[m] - bb_lens[r]].astype(np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out

    print(f"loaded ({time.time()-t0:.0f}s) N={len(trlat)} val={n} -> building top-{K} pools", flush=True)
    ids_k = np.zeros((n, K), np.int64)
    s_k = np.full((n, K), -1e9, np.float32)
    bb_k = np.zeros((n, K), np.float32)
    for i in range(n):
        m = min(int(m_of[i]), 400)
        cells = rank[i, :m]
        ids = np.concatenate([order[starts[c]:ends[c]] for c in cells])
        ids.sort()
        s = 0.5 * (geo_t[ids].astype(np.float32) @ geo_v[i]) + 0.5 * (c1_t[ids].astype(np.float32) @ c1_v[i])
        s = s.astype(np.float64) + 0.05 * logp[i, cell_a[ids]]
        k = min(K, len(ids))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        ids_k[i, :k] = ids[top]
        s_k[i, :k] = s[top]
        bb_k[i, :k] = bb_rows(ids[top]) @ bb_v[i]
        if i % 250 == 0:
            print(f"  {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(EP, ids=ids_k, s=s_k, bb=bb_k)
    print(f"episodes cached -> {EP} ({time.time()-t0:.0f}s)", flush=True)

la_k = trlat[ids_k]
lo_k = trlon[ids_k]
valid = s_k > -1e8

# pairwise candidate distances per query (n, K, K)
print("pairwise candidate geometry...", flush=True)
d_kk = np.zeros((n, K, K), np.float32)
for i in range(n):
    d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])

# E0 baseline
d0 = hav(la_k[:, 0], lo_k[:, 0], tlat, tlon)
report("E0 baseline (argmax)", d0)

# softmax weights over candidate scores
def soft(s, tau):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * valid
    return w / (w.sum(1, keepdims=True) + 1e-12)

# E1 geo-consensus: add mass of nearby candidates
for R in (10.0, 25.0):
    for beta in (0.5, 1.0, 2.0):
        for tau in (0.05,):
            w = soft(s_k, tau)
            cons = ((d_kk <= R) * w[:, None, :]).sum(2)          # support near each candidate
            sc = s_k + beta * cons
            j = sc.argmax(1)
            d = hav(la_k[np.arange(n), j], lo_k[np.arange(n), j], tlat, tlon)
            report(f"E1 consensus R{R:.0f} b{beta} t{tau}", d)

# E2 1-step diffusion on the pool graph (bb affinity)
for alpha in (0.7, 0.85):
    w = soft(s_k, 0.05)
    aff = np.clip(bb_k[:, None, :] * 0 + 1, 0, 1)  # placeholder to keep shape clear
    # affinity between candidates approximated by geo proximity kernel (bb c-c sims not cached)
    A = np.exp(-d_kk / 50.0)
    A /= A.sum(2, keepdims=True) + 1e-12
    sc = alpha * s_k + (1 - alpha) * (A @ w[..., None]).squeeze(-1) * 10.0
    j = sc.argmax(1)
    d = hav(la_k[np.arange(n), j], lo_k[np.arange(n), j], tlat, tlon)
    report(f"E2 diffusion a{alpha}", d)

# E3 query-expansion via bb sims: fuse rank-1..3 bb sims into the score
for gamma in (0.3, 0.6):
    sc = s_k + gamma * bb_k
    j = sc.argmax(1)
    d = hav(la_k[np.arange(n), j], lo_k[np.arange(n), j], tlat, tlon)
    report(f"E3 bb-fuse g{gamma}", d)

# E4 abstain: low consensus around the chosen candidate -> classifier mode fallback
w = soft(s_k, 0.05)
cons_top = ((d_kk[:, 0, :] <= 25.0) * w).sum(1)
for thr in (0.15, 0.30, 0.50):
    use_cls = cons_top < thr
    d = d0.copy()
    dc = hav(cls_pred[:, 0], cls_pred[:, 1], tlat, tlon)
    d[use_cls] = dc[use_cls]
    report(f"E4 abstain<{thr}", d, extra=f"(fallback {100*use_cls.mean():.0f}%)")

# E1+E3 combo (the two most promising, if both help)
w = soft(s_k, 0.05)
cons = ((d_kk <= 25.0) * w[:, None, :]).sum(2)
sc = s_k + 1.0 * cons + 0.3 * bb_k
j = sc.argmax(1)
d = hav(la_k[np.arange(n), j], lo_k[np.arange(n), j], tlat, tlon)
report("E5 consensus+bb combo", d)
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
