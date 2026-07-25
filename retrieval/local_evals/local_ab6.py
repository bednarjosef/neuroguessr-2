#!/usr/bin/env python3
"""Round 6: the Bayes-prior fallback for hopeless queries.

Autopsy fact: the >2500km band (78 queries) averages ~8100 km per miss — these are
antipodal disasters where the evidence points nowhere. Decision theory: when the posterior
carries no information, the loss-minimising guess is the PRIOR's minimiser — the point
minimising expected distance to the training density (~ the dataset's geographic median).
If the hopeless set is identifiable (top-mass signal) and the truth distribution is
concentrated, replacing ~8100km misses with ~prior-radius misses is a large mean win that
touches nothing else. Sweep the confidence threshold; also try a continent-conditioned
variant (prior mode within the classifier's best continent-scale mass region)."""
import os
import time

import numpy as np

REPO = "/home/josef/everything/coding/neuroguessr-2-research"
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
    print(f"{tag:40s} median {np.median(d):7.2f} | mean {d.mean():7.1f} | "
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
n = len(tlat)
la_k, lo_k = trlat[ids_k], trlon[ids_k]
valid = s_k > -1e8
ar = np.arange(n)

# ---- the training-density prior minimiser (geometric median of train locations) ------
sub = np.random.default_rng(0).choice(len(trlat), 40000, replace=False)
sl, so = np.radians(trlat[sub]), np.radians(trlon[sub])
V = np.stack([np.cos(sl) * np.cos(so), np.cos(sl) * np.sin(so), np.sin(sl)], 1)
x = V.mean(0)
x /= np.linalg.norm(x)
for _ in range(50):
    dd = np.maximum(np.arccos(np.clip(V @ x, -1, 1)), 1e-6)
    wp = 1.0 / dd
    x = (wp[:, None] * V).sum(0)
    x /= np.linalg.norm(x)
prior_lat = float(np.degrees(np.arcsin(x[2])))
prior_lon = float(np.degrees(np.arctan2(x[1], x[0])))
dp = hav(prior_lat, prior_lon, trlat[sub], trlon[sub])
print(f"train prior minimiser: ({prior_lat:.2f}, {prior_lon:.2f}) — "
      f"mean dist to train locs {dp.mean():.0f} km, median {np.median(dp):.0f} km", flush=True)
d_prior_all = hav(prior_lat, prior_lon, tlat, tlon)
print(f"  expected val error if ALWAYS guessing prior: mean {d_prior_all.mean():.0f} km", flush=True)

# ---- rebuild E7 (same as rounds 4/5) --------------------------------------------------
d_kk = np.zeros((n, K, K), np.float32)
for i in range(n):
    d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])
d_all = hav(la_k, lo_k, tlat[:, None], tlon[:, None])


def soft(s, tau=0.05):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * (s > -1e8)
    return w / (w.sum(1, keepdims=True) + 1e-12)


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
F = (F - F.reshape(-1, F.shape[2]).mean(0)) / (F.reshape(-1, F.shape[2]).std(0) + 1e-9)
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
report("E7 blend (current best)", d7)
wl = soft(np.where(valid, sc, -1e9), tau=1.0)
top_mass = ((d_kk[ar, j7] <= 100) * wl).sum(1)

# hopeless-detector quality: what do the fallback sets currently cost?
print("\n--- hopeless-set precision (current E7 error within each candidate fallback set) ---")
for thr in (0.1, 0.15, 0.2, 0.3, 0.4):
    m = top_mass < thr
    if m.sum() == 0:
        continue
    print(f"  top_mass<{thr}: {m.sum():4d} q | E7 mean there {d7[m].mean():6.0f} km | "
          f"median {np.median(d7[m]):6.0f} | already <=200km: {100*(d7[m]<=200).mean():.0f}% | "
          f"prior-guess mean would be {d_prior_all[m].mean():6.0f} km", flush=True)

# ---- E11: prior fallback ------------------------------------------------------------
print()
for thr in (0.1, 0.15, 0.2, 0.3):
    m = top_mass < thr
    d = d7.copy()
    d[m] = d_prior_all[m]
    report(f"E11 prior-fallback<{thr}", d, extra=f"(fallback {100*m.mean():.0f}%)")

# ---- E12: shrink toward prior instead of jumping (weighted point between pick & prior)
for thr in (0.2, 0.3):
    m = top_mass < thr
    pred = np.stack([la_k[ar, j7], lo_k[ar, j7]], 1)
    for i in np.where(m)[0]:
        t = min(1.0, top_mass[i] / thr)          # confidence in the pick
        a1, o1 = np.radians(pred[i]); a2, o2 = np.radians([prior_lat, prior_lon])
        v1 = np.array([np.cos(a1) * np.cos(o1), np.cos(a1) * np.sin(o1), np.sin(a1)])
        v2 = np.array([np.cos(a2) * np.cos(o2), np.cos(a2) * np.sin(o2), np.sin(a2)])
        vv = t * v1 + (1 - t) * v2
        vv /= np.linalg.norm(vv)
        pred[i] = (np.degrees(np.arcsin(vv[2])), np.degrees(np.arctan2(vv[1], vv[0])))
    d = hav(pred[:, 0], pred[:, 1], tlat, tlon)
    report(f"E12 prior-shrink<{thr}", d, extra=f"(shrunk {100*m.mean():.0f}%)")
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
