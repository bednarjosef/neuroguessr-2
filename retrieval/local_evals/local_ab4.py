#!/usr/bin/env python3
"""Round 4: MEAN-targeted. Tail autopsy first, then risk-aware prediction rules.

Median rewards mode-seeking (argmax). Mean rewards mass-seeking: the Bayes rule under
absolute-distance loss is the weighted GEOMETRIC MEDIAN of the posterior cloud, and a
per-query switch between the two (unimodal -> argmax, diffuse -> mass-seeking) should cut
the tail without touching the well-solved queries. Cluster-hypothesis selection adds the
classifier's regional mass as EVIDENCE (not a fallback — the E4 lesson)."""
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IX = os.path.join(REPO, "run_full2", "retrieval_index")
SP = os.path.dirname(os.path.abspath(__file__))
EARTH = 6371.0088
K = 100


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, map(np.asarray, (a1, o1, a2, o2)))
    x = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def report(tag, d, extra=""):
    gg = 5000 * np.exp(-d / 1492.7)
    print(f"{tag:36s} median {np.median(d):7.2f} | mean {d.mean():7.1f} | "
          f"@1 {100*(d<=1).mean():5.2f} | @25 {100*(d<=25).mean():5.2f} | "
          f"@200 {100*(d<=200).mean():5.2f} | GG {gg.mean():4.0f} {extra}", flush=True)


t0 = time.time()
e = np.load(os.path.join(SP, "episodes_s384.npz"))
ids_k, s_k, bb_k = e["ids"], e["s"].astype(np.float64), e["bb"].astype(np.float64)
cc = np.load(os.path.join(SP, "ccsims_s384.npy")).astype(np.float64)
z = np.load(os.path.join(IX, "train_latlon.npz"))
trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
vz = np.load(os.path.join(REPO, "run_full", "val_analysis", "val_meta.npz"))
tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
cz = np.load(os.path.join(REPO, "run_full", "val_analysis", "cells.npz"))
clat, clon = cz["lat"].astype(np.float64), cz["lon"].astype(np.float64)
n = len(tlat)
la_k, lo_k = trlat[ids_k], trlon[ids_k]
valid = s_k > -1e8
ar = np.arange(n)


def cat(prefix):
    fs = sorted([f for f in os.listdir(IX) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(IX, f)) for f in fs])


lga = cat("logits_a_val_r").astype(np.float64)
lga -= lga.max(1, keepdims=True)
pc = np.exp(lga)
pc /= pc.sum(1, keepdims=True)

d_kk = np.zeros((n, K, K), np.float32)
for i in range(n):
    d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])
d_all = hav(la_k, lo_k, tlat[:, None], tlon[:, None])


def soft(s, tau=0.05):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * valid
    return w / (w.sum(1, keepdims=True) + 1e-12)


# ---- rebuild the E7-logistic CV score (the current best ranker) -----------------------
w = soft(s_k)
A = np.where(cc > 0.5, cc, 0)
A = A / (A.sum(2, keepdims=True) + 1e-12)
diff = np.einsum('nkj,nj->nk', A, w)
cons10 = ((d_kk <= 10) * w[:, None, :]).sum(2)
cons25 = ((d_kk <= 25) * w[:, None, :]).sum(2)
cons50 = ((d_kk <= 50) * w[:, None, :]).sum(2)
ccmean = (cc * w[:, None, :]).sum(2)
ccmax3 = np.sort(cc, axis=2)[:, :, -4:-1].mean(2)
rank_pos = np.tile(np.arange(K, dtype=np.float64), (n, 1))
margin = s_k[:, :1] - s_k
ent = -(w * np.log(np.clip(w, 1e-12, None))).sum(1, keepdims=True) * np.ones((1, K))
nvalid = valid.sum(1, keepdims=True) * np.ones((1, K)) / K
F = np.stack([s_k, bb_k, np.log1p(rank_pos), margin, cons10, cons25, cons50,
              diff * 10, ccmean, ccmax3, ent, nvalid,
              s_k * cons25, bb_k * cons25, margin * cons25, diff * cons25 * 10], axis=2)
mu = F.reshape(-1, F.shape[2]).mean(0)
sd = F.reshape(-1, F.shape[2]).std(0) + 1e-9
F = (F - mu) / sd
y = (d_all <= 25.0).astype(np.float64)


def fit_logistic(X, yy, iters=400, lr=0.5):
    wg = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(X @ wg + b)))
        g = pr - yy
        wg -= lr * (X.T @ g) / len(yy); b -= lr * g.mean()
    return lambda X_: X_ @ wg + b


folds = ar % 5
sc = np.zeros((n, K))
for f in range(5):
    tr = folds != f
    Xtr = F[tr].reshape(-1, F.shape[2])
    m = valid[tr].ravel()
    model = fit_logistic(Xtr[m], y[tr].ravel()[m])
    sc[~tr] = model(F[~tr].reshape(-1, F.shape[2])).reshape((~tr).sum(), K)
sc[~valid] = -1e9
sblend = s_k / s_k.std() + 1.0 * sc
j7 = sblend.argmax(1)
d7 = hav(la_k[ar, j7], lo_k[ar, j7], tlat, tlon)
report("E7 blend lam1.0 (current best)", d7)

# ---- TAIL AUTOPSY ---------------------------------------------------------------------
print("\n--- tail autopsy on E7 ---", flush=True)
for lo_b, hi_b in ((0, 25), (25, 200), (200, 1000), (1000, 2500), (2500, 1e9)):
    m = (d7 >= lo_b) & (d7 < hi_b)
    contrib = d7[m].sum() / n
    print(f"  band {lo_b:>5.0f}-{hi_b:<7.0f} km: {m.sum():4d} q ({100*m.mean():4.1f}%) "
          f"-> {contrib:6.1f} km of the mean", flush=True)
tail = d7 > 1000
best_in_pool = d_all.min(1)
rank_of_best = d_all.argmin(1)
print(f"\n  tail (>1000km): {tail.sum()} queries")
print(f"    pool contained a <=25km candidate : {100*(best_in_pool[tail]<=25).mean():.0f}%")
print(f"    pool contained a <=100km candidate: {100*(best_in_pool[tail]<=100).mean():.0f}%")
print(f"    median pool-rank of best candidate: {np.median(rank_of_best[tail]):.0f}")
cd = hav(clat[pc.argmax(1)], clon[pc.argmax(1)], tlat, tlon)
print(f"    classifier top-cell <=500km of truth: {100*(cd[tail]<=500).mean():.0f}% "
      f"(vs {100*(cd<=500).mean():.0f}% overall)", flush=True)
wl = soft(np.where(valid, sc, -1e9), tau=1.0)
top_mass = ((d_kk[ar, j7] <= 100) * wl).sum(1)
print(f"    mean top-pick 100km-mass: tail {top_mass[tail].mean():.2f} vs ok {top_mass[~tail].mean():.2f}")

# ---- risk-aware prediction rules ------------------------------------------------------
print("\n--- mean-targeted rules on E7 scores ---", flush=True)

def geomed(i, wts):
    v = np.stack([np.cos(np.radians(la_k[i])) * np.cos(np.radians(lo_k[i])),
                  np.cos(np.radians(la_k[i])) * np.sin(np.radians(lo_k[i])),
                  np.sin(np.radians(la_k[i]))], 1)
    x = (wts[:, None] * v).sum(0)
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(25):                       # Weiszfeld on the sphere (tangent approx)
        dd = np.maximum(np.arccos(np.clip(v @ x, -1, 1)), 1e-6)
        wp = wts / dd
        x = (wp[:, None] * v).sum(0)
        x /= np.linalg.norm(x) + 1e-12
    return np.degrees(np.arcsin(x[2])), np.degrees(np.arctan2(x[1], x[0]))


# cluster-hypothesis selection with classifier regional mass as evidence
cell_of = None
def rule_eval(theta, gamma, use_geomed):
    pred = np.stack([la_k[ar, j7], lo_k[ar, j7]], 1)
    switched = 0
    for i in np.where(top_mass < theta)[0]:
        wts = wl[i] * valid[i]
        anchors = np.argsort(-wts)[:8]
        best_score, best_j = -1e18, j7[i]
        for a_ in anchors:
            m_ = d_kk[i, a_] <= 100
            rmass = float((wts * m_).sum())
            cmass = float(pc[i][hav(la_k[i, a_], lo_k[i, a_], clat, clon) <= 300].sum())
            sc_ = rmass + gamma * cmass
            if sc_ > best_score:
                best_score = sc_
                mem = np.where(m_ & valid[i], sc[i], -1e9)
                best_j = int(mem.argmax())
        if use_geomed and best_score < 0.35:
            pred[i] = geomed(i, wts)
        else:
            pred[i] = (la_k[i, best_j], lo_k[i, best_j])
        switched += 1
    d = hav(pred[:, 0], pred[:, 1], tlat, tlon)
    report(f"R th{theta} g{gamma}{' gm' if use_geomed else '  '}", d,
           extra=f"(rescored {100*switched/n:.0f}%)")
    return d


for theta in (0.25, 0.4):
    for gamma in (0.0, 0.5, 1.5):
        rule_eval(theta, gamma, False)
rule_eval(0.4, 0.5, True)
rule_eval(0.6, 0.5, False)
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
