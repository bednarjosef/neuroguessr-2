#!/usr/bin/env python3
"""Round 3: stack all winners. Learned reranker with the full winning feature set
(consensus + diffusion + cluster mass), logistic AND tiny-MLP variants, 5-fold CV,
and cluster-snap as an alternative prediction rule on the learned scores."""
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
    print(f"{tag:34s} median {np.median(d):7.2f} | mean {d.mean():7.1f} | "
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
d_kk = np.zeros((n, K, K), np.float32)
for i in range(n):
    d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])
d_all = hav(la_k, lo_k, tlat[:, None], tlon[:, None])
d0 = hav(la_k[:, 0], lo_k[:, 0], tlat, tlon)
report("E0 baseline", d0)


def soft(s, tau=0.05):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * valid
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
mu = F.reshape(-1, F.shape[2]).mean(0)
sd = F.reshape(-1, F.shape[2]).std(0) + 1e-9
F = (F - mu) / sd
y = (d_all <= 25.0).astype(np.float64)
print(f"features ready ({time.time()-t0:.0f}s)", flush=True)


def fit_logistic(X, yy, iters=400, lr=0.5):
    wg = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(X @ wg + b)))
        g = pr - yy
        wg -= lr * (X.T @ g) / len(yy); b -= lr * g.mean()
    return lambda X_: X_ @ wg + b


def fit_mlp(X, yy, h=24, iters=500, lr=0.3, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.3, (X.shape[1], h)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.3, h); b2 = 0.0
    for _ in range(iters):
        H = np.tanh(X @ W1 + b1)
        pr = 1 / (1 + np.exp(-(H @ W2 + b2)))
        g = (pr - yy) / len(yy)
        gW2 = H.T @ g; gb2 = g.sum()
        gH = np.outer(g, W2) * (1 - H ** 2)
        W1 -= lr * X.T @ gH; b1 -= lr * gH.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2
    return lambda X_: np.tanh(X_ @ W1 + b1) @ W2 + b2


folds = ar % 5
for name, fit in (("logistic", fit_logistic), ("mlp24", fit_mlp)):
    sc = np.zeros((n, K))
    for f in range(5):
        tr = folds != f
        Xtr = F[tr].reshape(-1, F.shape[2])
        m = valid[tr].ravel()
        model = fit(Xtr[m], y[tr].ravel()[m])
        sc[~tr] = model(F[~tr].reshape(-1, F.shape[2])).reshape((~tr).sum(), K)
    sc[~valid] = -1e9
    j = sc.argmax(1)
    report(f"E7 {name} (CV, all winners)", hav(la_k[ar, j], lo_k[ar, j], tlat, tlon))
    for lam in (0.25, 0.5, 1.0):
        sb = s_k / s_k.std() + lam * sc
        jj = sb.argmax(1)
        report(f"E7 {name} blend lam{lam}", hav(la_k[ar, jj], lo_k[ar, jj], tlat, tlon))
    # cluster-snap prediction rule on learned weights
    wl = soft(np.where(valid, sc, -1e9), tau=1.0)
    for R_ in (25.0,):
        mass = ((d_kk <= R_) * wl[:, None, :]).sum(2)
        anc = mass.argmax(1)
        d4 = np.zeros(n)
        for i in range(n):
            m_ = (d_kk[i, anc[i]] <= R_) & valid[i]
            d4[i] = hav(la_k[i, np.where(m_, sc[i], -1e9).argmax()],
                        lo_k[i, np.where(m_, sc[i], -1e9).argmax()], tlat[i], tlon[i])
        report(f"E7 {name} cluster-snap R{R_:.0f}", d4)
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
