#!/usr/bin/env python3
"""Local A/B round 2 on the S384 bed: proper visual-affinity reranks + learned reranker pilot.

Uses the episode cache from round 1 (top-100 gated pools) and adds candidate-candidate
VISUAL sims (bb space), enabling:
  E2' k-reciprocal + diffusion with real visual affinity (round 1 used a geo kernel)
  E3' SuperGlobal-style query expansion (max-fuse of top-3) re-scored in bb space
  E4' retrieval-side tail handling: snap to the best member of the max-mass geo cluster
  E6  learned per-candidate reranker (numpy logistic on rich features), 5-fold CV so every
      query is scored by a model that never saw it. Honest small-scale preview of Tier-1 #1.
"""
import os
import time

import numpy as np

REPO = "/home/josef/everything/coding/neuroguessr-2-research"
IX = os.path.join(REPO, "run_full2", "retrieval_index")
SP = os.path.dirname(os.path.abspath(__file__))
EARTH = 6371.0088
K = 100


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, map(np.asarray, (lat1, lon1, lat2, lon2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def report(tag, d, extra=""):
    gg = 5000 * np.exp(-d / 1492.7)
    print(f"{tag:30s} median {np.median(d):7.2f} | mean {d.mean():7.1f} | "
          f"@1 {100*(d<=1).mean():5.2f} | @25 {100*(d<=25).mean():5.2f} | "
          f"@200 {100*(d<=200).mean():5.2f} | GG {gg.mean():4.0f} {extra}", flush=True)


t0 = time.time()
e = np.load(os.path.join(SP, "episodes_s384.npz"))
ids_k, s_k, bb_k = e["ids"], e["s"].astype(np.float64), e["bb"].astype(np.float64)
z = np.load(os.path.join(IX, "train_latlon.npz"))
trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
vz = np.load(os.path.join(REPO, "run_full", "val_analysis", "val_meta.npz"))
tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
n = len(tlat)
la_k, lo_k = trlat[ids_k], trlon[ids_k]
valid = s_k > -1e8

CC = os.path.join(SP, "ccsims_s384.npy")
if os.path.exists(CC):
    cc = np.load(CC)
else:
    bb_train = [np.load(os.path.join(IX, f"emb_bb_train_r{r}.npy"), mmap_mode="r") for r in range(4)]
    lens = np.cumsum([0] + [len(b) for b in bb_train])

    def bb_rows(ids):
        out = np.empty((len(ids), bb_train[0].shape[1]), np.float32)
        for r in range(4):
            m = (ids >= lens[r]) & (ids < lens[r + 1])
            if m.any():
                out[m] = bb_train[r][ids[m] - lens[r]].astype(np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out

    cc = np.zeros((n, K, K), np.float32)
    for i in range(n):
        v = bb_rows(ids_k[i])
        cc[i] = v @ v.T
        if i % 250 == 0:
            print(f"  cc {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.save(CC, cc)
print(f"cc sims ready ({time.time()-t0:.0f}s)", flush=True)
cc = cc.astype(np.float64)
d_kk = np.zeros((n, K, K), np.float32)
for i in range(n):
    d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])

d0 = hav(la_k[:, 0], lo_k[:, 0], tlat, tlon)
report("E0 baseline (argmax)", d0)
ar = np.arange(n)


def soft(s, tau=0.05):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * valid
    return w / (w.sum(1, keepdims=True) + 1e-12)


def snap(sc):
    j = sc.argmax(1)
    return hav(la_k[ar, j], lo_k[ar, j], tlat, tlon)


# ---- E2' k-reciprocal with visual affinity: candidate score boosted when it is visually
# reciprocal-close to OTHER high-scoring candidates (query-anchored expanded neighbourhood).
w0 = soft(s_k)
for kk in (5, 10):
    # reciprocal indicator: candidate j is in top-kk visual neighbours of candidate i and vice versa
    top = np.argsort(-cc, axis=2)[:, :, :kk]
    R = np.zeros((n, K, K), bool)
    for i in range(n):
        M = np.zeros((K, K), bool)
        M[np.repeat(np.arange(K), kk), top[i].ravel()] = True
        R[i] = M & M.T
    for beta in (0.5, 1.0):
        boost = (R * w0[:, None, :]).sum(2)
        report(f"E2' k-recip k{kk} b{beta}", snap(s_k + beta * boost))

# visual diffusion: propagate softmax mass over the visual graph
A = np.where(cc > 0.5, cc, 0)
A = A / (A.sum(2, keepdims=True) + 1e-12)
for alpha in (0.7, 0.85):
    sc = alpha * s_k + (1 - alpha) * 10.0 * np.einsum('nkj,nj->nk', A, w0)
    report(f"E2' vis-diffuse a{alpha}", snap(sc))

# ---- E3' SuperGlobal-style QE: refine query implicitly by max-fusing each candidate's sim
# to the top-3 candidates (bb space) with its own query sim.
qe = np.maximum(bb_k, np.max(cc[:, :, :3] * w0[:, None, :3] * 3.0, axis=2))
for g in (0.3, 0.6, 1.0):
    report(f"E3' QE-maxfuse g{g}", snap(s_k + g * qe))

# ---- E4' retrieval-side tail handling: pick the max-mass geo cluster, snap to its best member
w = soft(s_k)
for R_ in (25.0, 50.0):
    mass = ((d_kk <= R_) * w[:, None, :]).sum(2)          # mass around each candidate
    best_anchor = mass.argmax(1)
    d4 = np.zeros(n)
    for i in range(n):
        m = (d_kk[i, best_anchor[i]] <= R_) & valid[i]
        j = np.where(m, s_k[i], -1e9).argmax()
        d4[i] = hav(la_k[i, j], lo_k[i, j], tlat[i], tlon[i])
    report(f"E4' cluster-snap R{R_:.0f}", d4)

# ---- E6 learned reranker (numpy logistic, 5-fold CV over queries)
print("building features...", flush=True)
rank_pos = np.tile(np.arange(K, dtype=np.float64), (n, 1))
margin = s_k[:, :1] - s_k
cons10 = ((d_kk <= 10) * w[:, None, :]).sum(2)
cons25 = ((d_kk <= 25) * w[:, None, :]).sum(2)
cons50 = ((d_kk <= 50) * w[:, None, :]).sum(2)
ccmean = (cc * w[:, None, :]).sum(2)
ccmax3 = np.sort(cc, axis=2)[:, :, -4:-1].mean(2)
ent = -(w * np.log(np.clip(w, 1e-12, None))).sum(1, keepdims=True) * np.ones((1, K))
F = np.stack([s_k, bb_k, np.log1p(rank_pos), margin, cons10, cons25, cons50,
              ccmean, ccmax3, ent, s_k * cons25, bb_k * cons25], axis=2)
mu = F.reshape(-1, F.shape[2]).mean(0)
sd = F.reshape(-1, F.shape[2]).std(0) + 1e-9
F = (F - mu) / sd
d_all = hav(la_k, lo_k, tlat[:, None], tlon[:, None])
y = (d_all <= 25.0).astype(np.float64)

def fit_logistic(X, yy, iters=300, lr=0.5):
    wgt = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        z_ = X @ wgt + b
        pr = 1 / (1 + np.exp(-z_))
        g = pr - yy
        wgt -= lr * (X.T @ g) / len(yy)
        b -= lr * g.mean()
    return wgt, b

folds = np.arange(n) % 5
sc6 = np.zeros((n, K))
for f in range(5):
    tr = folds != f
    Xtr = F[tr].reshape(-1, F.shape[2])
    ytr = y[tr].ravel()
    vmask = valid[tr].ravel()
    wgt, b = fit_logistic(Xtr[vmask], ytr[vmask])
    sc6[~tr] = F[~tr] @ wgt + b
sc6[~valid] = -1e9
report("E6 learned rerank (5-fold CV)", snap(sc6))
# blend learned score with the original ranking score
for lam in (0.5, 1.0, 2.0):
    report(f"E6 blend lam{lam}", snap(s_k / s_k.std() + lam * sc6))
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
