#!/usr/bin/env python3
"""Round 5: mean-targeted, informed by the round-4 autopsy.

Fixes: (1) correct run_full2 cell centroids (round 1-4 paired run2 logits with run1 cell
coords — classifier evidence was garbage, E4 was executed on false testimony);
(2) WIDE re-gate (mass 0.995, cap 1500) for low-confidence queries — 58% of tail pools
lacked even a <=100km candidate, so the gate, not the matcher, owns the tail;
(3) classifier fallback retried with correct centroids, on the queries where even the wide
pool has no support."""
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
    print(f"{tag:38s} median {np.median(d):7.2f} | mean {d.mean():7.1f} | "
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

# CORRECT cell centroids for the run2 tessellation
cent = np.load(os.path.join(IX, "centroids.npz"))["cent"].astype(np.float64)
cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
clat = np.degrees(np.arcsin(np.clip(cent[:, 2], -1, 1)))
clon = np.degrees(np.arctan2(cent[:, 1], cent[:, 0]))
cell_a = np.load(os.path.join(IX, "cell_a.npy"))


def cat(prefix):
    fs = sorted([f for f in os.listdir(IX) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(IX, f)) for f in fs])


lga = cat("logits_a_val_r").astype(np.float64)
lga -= lga.max(1, keepdims=True)
pc = np.exp(lga)
pc /= pc.sum(1, keepdims=True)
logp = np.log(np.clip(pc, 1e-12, None))
rank_c = np.argsort(-pc, axis=1)
cd = hav(clat[rank_c[:, 0]], clon[rank_c[:, 0]], tlat, tlon)
print(f"sanity: classifier top-cell <=500km overall = {100*(cd<=500).mean():.0f}% "
      f"(round-4's broken mapping said 1%)", flush=True)

# mode-seeking classifier prediction with CORRECT centroids
T = 0.5
ps = np.exp(lga / T)
ps /= ps.sum(1, keepdims=True)
cls_pred = np.zeros((n, 2))
for i in range(n):
    top = rank_c[i, :16]
    keep = top[hav(clat[top[0]], clon[top[0]], clat[top], clon[top]) <= 1000.0]
    wv = ps[i, keep]
    wv = wv / wv.sum()
    la, lo = np.radians(clat[keep]), np.radians(clon[keep])
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    m = (wv[:, None] * v).sum(0)
    m /= np.linalg.norm(m)
    cls_pred[i] = (np.degrees(np.arcsin(m[2])), np.degrees(np.arctan2(m[1], m[0])))
dcls = hav(cls_pred[:, 0], cls_pred[:, 1], tlat, tlon)
report("classifier-only (fixed centroids)", dcls)

# ---- rebuild E7 CV score (same as round 4) -------------------------------------------
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
MU = F.reshape(-1, F.shape[2]).mean(0)
SD = F.reshape(-1, F.shape[2]).std(0) + 1e-9
F = (F - MU) / SD
y = (d_all <= 25.0).astype(np.float64)


def fit_logistic(X, yy, iters=400, lr=0.5):
    wg = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(X @ wg + b)))
        g = pr - yy
        wg -= lr * (X.T @ g) / len(yy); b -= lr * g.mean()
    return wg, b


folds = ar % 5
Wf, Bf = {}, {}
sc = np.zeros((n, K))
for f in range(5):
    tr = folds != f
    Xtr = F[tr].reshape(-1, F.shape[2])
    m = valid[tr].ravel()
    Wf[f], Bf[f] = fit_logistic(Xtr[m], y[tr].ravel()[m])
    sc[~tr] = (F[~tr] @ Wf[f] + Bf[f])
sc[~valid] = -1e9
sblend = s_k / s_k.std() + 1.0 * sc
j7 = sblend.argmax(1)
d7 = hav(la_k[ar, j7], lo_k[ar, j7], tlat, tlon)
report("E7 blend (current best)", d7)
wl = soft(np.where(valid, sc, -1e9), tau=1.0)
top_mass = ((d_kk[ar, j7] <= 100) * wl).sum(1)

# ---- E8: classifier fallback with CORRECT centroids ----------------------------------
for thr in (0.2, 0.35, 0.5):
    use = top_mass < thr
    d = d7.copy()
    d[use] = dcls[use]
    report(f"E8 cls-fallback<{thr}", d, extra=f"(fallback {100*use.mean():.0f}%)")

# ---- E9: WIDE re-gate for low-confidence queries -------------------------------------
LOW = np.where(top_mass < 0.5)[0]
print(f"wide re-gate for {len(LOW)} queries ({time.time()-t0:.0f}s)...", flush=True)
order = np.argsort(cell_a, kind="stable")
cs = cell_a[order]
C = pc.shape[1]
starts = np.searchsorted(cs, np.arange(C), "left")
ends = np.searchsorted(cs, np.arange(C), "right")
geo_t = np.load(os.path.join(IX, "geo_head_10_train.npy"), mmap_mode="r")
c1_t = np.load(os.path.join(IX, "train_proj.npy"), mmap_mode="r")
geo_v = np.load(os.path.join(IX, "geo_head_10_val.npy")).astype(np.float32)
c1_v = np.load(os.path.join(IX, "val_proj.npy")).astype(np.float32)
bb_train = [np.load(os.path.join(IX, f"emb_bb_train_r{r}.npy"), mmap_mode="r") for r in range(4)]
lens = np.cumsum([0] + [len(b) for b in bb_train])
bb_v = cat("emb_bb_val_r").astype(np.float32)
bb_v /= np.linalg.norm(bb_v, axis=1, keepdims=True) + 1e-9


def bb_rows(ids):
    out = np.empty((len(ids), bb_train[0].shape[1]), np.float32)
    for r in range(4):
        m = (ids >= lens[r]) & (ids < lens[r + 1])
        if m.any():
            out[m] = bb_train[r][ids[m] - lens[r]].astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


csum = np.take_along_axis(pc, rank_c, 1).cumsum(1)
m_of_wide = (csum < 0.995).sum(1) + 1
pred9 = np.stack([la_k[ar, j7], lo_k[ar, j7]], 1).astype(np.float64)
wide_mass = top_mass.copy()
for i in LOW:
    m = min(int(m_of_wide[i]), 1500)
    cells = rank_c[i, :m]
    ids = np.concatenate([order[starts[c]:ends[c]] for c in cells])
    ids.sort()
    s = 0.5 * (geo_t[ids].astype(np.float32) @ geo_v[i]) + 0.5 * (c1_t[ids].astype(np.float32) @ c1_v[i])
    s = s.astype(np.float64) + 0.05 * logp[i, cell_a[ids]]
    k = min(K, len(ids))
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top])]
    idk, sk = ids[top], s[top]
    lak, lok = trlat[idk], trlon[idk]
    bbk = bb_rows(idk) @ bb_v[i]
    dkk = hav(lak[:, None], lok[:, None], lak[None, :], lok[None, :])
    vv = bb_rows(idk)
    cck = (vv @ vv.T).astype(np.float64)
    ww = np.exp((sk - sk.max()) / 0.05); ww /= ww.sum() + 1e-12
    Ak = np.where(cck > 0.5, cck, 0); Ak = Ak / (Ak.sum(1, keepdims=True) + 1e-12)
    dfk = Ak @ ww
    c10 = ((dkk <= 10) * ww[None, :]).sum(1)
    c25 = ((dkk <= 25) * ww[None, :]).sum(1)
    c50 = ((dkk <= 50) * ww[None, :]).sum(1)
    ccm = (cck * ww[None, :]).sum(1)
    ccx = np.sort(cck, axis=1)[:, -4:-1].mean(1)
    en = -(ww * np.log(np.clip(ww, 1e-12, None))).sum() * np.ones(k)
    nv = np.ones(k) * k / K
    mg = sk[0] - sk
    Fi = np.stack([sk, bbk, np.log1p(np.arange(k, dtype=np.float64)), mg, c10, c25, c50,
                   dfk * 10, ccm, ccx, en, nv, sk * c25, bbk * c25, mg * c25, dfk * c25 * 10], 1)
    Fi = (Fi - MU) / SD
    f = int(folds[i])
    sci = Fi @ Wf[f] + Bf[f]
    sbl = sk / s_k.std() + 1.0 * sci
    jj = int(sbl.argmax())
    pred9[i] = (lak[jj], lok[jj])
    wli = np.exp(sci - sci.max()); wli /= wli.sum() + 1e-12
    wide_mass[i] = float(((dkk[jj] <= 100) * wli).sum())
d9 = hav(pred9[:, 0], pred9[:, 1], tlat, tlon)
report("E9 wide re-gate (low-conf only)", d9)

# ---- E10: wide re-gate + classifier fallback where even the wide pool has no support --
for thr in (0.2, 0.35):
    use = wide_mass < thr
    d = d9.copy()
    d[use] = dcls[use]
    report(f"E10 wide+cls-fallback<{thr}", d, extra=f"(fallback {100*use.mean():.0f}%)")
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
