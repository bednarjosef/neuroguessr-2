#!/usr/bin/env python3
"""C4 headline numbers, computed LOCALLY on CPU from the HF-mirrored index.

Faithful port of eval_levers.py's champion/combo variants (gate mass .95 cap 400,
space blends, CSLS 2s-r correction, PCA whitening, lam*logp prior, top-50 k=1 snap)
minus the regional chamfer rerank (reg_train not downloaded; historically ~+0.5-1pt @25).
Then the E7 learned reranker (5-fold CV) on the best variant's top-100 pools.

Comparison target: the C3-recipe champion — median 36.96 / @25 46.06 / @1 12.78 / GG 4452.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IX = os.path.join(REPO, "run_c4_index")
sys.path.insert(0, REPO)
from retrieval.train_place_head import PlaceHead  # noqa: E402

torch.set_num_threads(os.cpu_count())
EARTH = 6371.0088
T0 = time.time()


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, map(lambda v: np.asarray(v, np.float64), (a1, o1, a2, o2)))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def rep(tag, dd):
    print(f"{tag:34s} median {np.median(dd):7.2f}  mean {dd.mean():6.0f}  "
          f"@1 {100*(dd<=1).mean():5.2f}%  @25 {100*(dd<=25).mean():5.2f}%  "
          f"@200 {100*(dd<=200).mean():5.2f}%  GG {(5000*np.exp(-dd/1492.7)).mean():5.0f}", flush=True)


def cat(p):
    fs = sorted([f for f in os.listdir(IX) if f.startswith(p)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(IX, f)) for f in fs])


needed = ([f"c2_cls_train_c4_r{r}.npy" for r in range(4)] +
          [f"c2_cls_val_c4_r{r}.npy" for r in range(4)] +
          [f"c2_logits_val_c4_r{r}.npy" for r in range(4)] +
          ["geo5_cls_c4.pt", "geo10_cls_c4.pt", "geo25_cls_c4.pt",
           "train_latlon.npz", "val_meta.npz", "ckpt.pt"])
while True:
    missing = [f for f in needed if not os.path.exists(os.path.join(IX, f))]
    if not missing:
        break
    print(f"waiting for download: {len(missing)} missing ({missing[:2]}...)", flush=True)
    time.sleep(30)
print("all inputs present — starting", flush=True)

z = np.load(os.path.join(IX, "train_latlon.npz"))
trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
N = len(trlat)
vz = np.load(os.path.join(IX, "val_meta.npz"))
tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
n = len(tlat)

lg = torch.tensor(cat("c2_logits_val_c4_r"), dtype=torch.float32)
p = F.softmax(lg, dim=-1)
logp = p.clamp(1e-12).log().double()
rank_order = torch.argsort(p, dim=1, descending=True)
csum = torch.gather(p, 1, rank_order).cumsum(1)
m_of = (csum < 0.95).sum(1) + 1

ck = torch.load(os.path.join(IX, "ckpt.pt"), map_location="cpu", weights_only=False)
cent = ck["buffers"]["centroids"].float()
cent = F.normalize(cent, dim=-1)
del ck
lar, lor = np.radians(trlat), np.radians(trlon)
tru = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                             np.sin(lar)], -1), dtype=torch.float32)
cell_a = torch.empty(N, dtype=torch.long)
for i in range(0, N, 200_000):
    cell_a[i:i + 200_000] = (tru[i:i + 200_000] @ cent.T).argmax(1)
del tru
order = torch.argsort(cell_a, stable=True)
cs = cell_a[order]
C = cent.shape[0]
ar = torch.arange(C)
starts, ends = torch.searchsorted(cs, ar), torch.searchsorted(cs, ar, right=True)
print(f"gate ready ({time.time()-T0:.0f}s): {C} cells, median gate size "
      f"{int(np.median(m_of.numpy()))} cells", flush=True)

spaces = {}
tr_cls = torch.from_numpy(cat("c2_cls_train_c4_r"))
for i in range(0, N, 200_000):
    tr_cls[i:i + 200_000] = F.normalize(tr_cls[i:i + 200_000].float(), dim=-1).half()
va_cls = F.normalize(torch.from_numpy(cat("c2_cls_val_c4_r")).float(), dim=-1)
spaces["cls"] = (tr_cls, va_cls)
for nm, f in (("geo5c", "geo5_cls_c4.pt"), ("geo10c", "geo10_cls_c4.pt"),
              ("geo25c", "geo25_cls_c4.pt")):
    hck = torch.load(os.path.join(IX, f), map_location="cpu", weights_only=False)
    head = PlaceHead(hck["dim"], hck["hidden"]).eval()
    head.load_state_dict(hck["state_dict"])
    with torch.no_grad():
        out = torch.empty_like(tr_cls)
        for i in range(0, N, 100_000):
            out[i:i + 100_000] = head(tr_cls[i:i + 100_000].float()).half()
        spaces[nm] = (out, head(va_cls))
    print(f"space {nm} ready ({time.time()-T0:.0f}s)", flush=True)

# PCA whitening of cls on a 200k sample
idx = torch.randint(0, N, (200_000,))
X = tr_cls[idx].float()
mu = X.mean(0)
Xc = X - mu
cov = (Xc.T @ Xc) / X.shape[0]
ev, Vv = torch.linalg.eigh(cov.double())
W = (Vv / ev.clamp(min=1e-6).sqrt()).float()
out = torch.empty_like(tr_cls)
for i in range(0, N, 200_000):
    out[i:i + 200_000] = F.normalize((tr_cls[i:i + 200_000].float() - mu) @ W, dim=-1).half()
spaces["cls_wh"] = (out, F.normalize((va_cls - mu) @ W, dim=-1))
del X, Xc, cov, ev, Vv
print(f"whitened cls ready ({time.time()-T0:.0f}s)", flush=True)

# CSLS r(): mean top-10 sim to 5k pseudo-queries, per space
csls_r = {}
q = torch.randint(0, N, (5_000,))
for src in ("cls", "geo5c", "geo10c", "geo25c"):
    tr, _ = spaces[src]
    Q = tr[q].float()
    r = torch.empty(N, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, N, 50_000):
            sims = tr[i:i + 50_000].float() @ Q.T
            r[i:i + 50_000] = sims.topk(10, dim=1).values.mean(1)
            del sims
    csls_r[src] = r
    print(f"csls r() {src}: mean {r.mean():.3f} ({time.time()-T0:.0f}s)", flush=True)

b4 = {"cls": .34, "geo5c": .22, "geo10c": .22, "geo25c": .22}
V = [("CHAMPION-recipe cls|geo10c", {"cls": .5, "geo10c": .5}, {}),
     ("4band+CSLS", b4, {"csls": True}),
     ("4band+CSLS+wh", {**{k: v for k, v in b4.items() if k != "cls"}, "cls_wh": .34},
      {"csls": True})]
need = sorted({s for _, w, _ in V for s in w})
preds = {t: np.zeros((n, 2)) for t, _, _ in V}
K = 100
ep_ids = np.zeros((n, K), np.int64)
ep_s = np.full((n, K), -1e9, np.float32)
ep_bb = np.zeros((n, K), np.float32)
BEST_FOR_EP = "4band+CSLS+wh"

with torch.no_grad():
    for i in range(n):
        m = min(int(m_of[i]), 400)
        ids = torch.cat([order[starts[c]:ends[c]] for c in rank_order[i, :m]])
        lp = logp[i, cell_a[ids]]
        sims = {nm: spaces[nm][0][ids].float() @ spaces[nm][1][i] for nm in need}
        for tag, w, ex in V:
            s = sum(wt * sims[nm] for nm, wt in w.items())
            if ex.get("csls"):
                corr = sum(wt * csls_r[nm][ids] for nm, wt in w.items() if nm in csls_r)
                s = 2 * s - corr
            sc = s.double() + 0.05 * lp
            top = sc.topk(min(50, sc.numel()))
            j = int(ids[top.indices[0]])
            preds[tag][i] = (trlat[j], trlon[j])
            if tag == BEST_FOR_EP:
                k = min(K, sc.numel())
                tk = sc.topk(k)
                ep_ids[i, :k] = ids[tk.indices].numpy()
                ep_s[i, :k] = tk.values.float().numpy()
                ep_bb[i, :k] = sims["cls"][tk.indices].numpy()
        if i and i % 500 == 0:
            print(f"  {i}/{n} ({time.time()-T0:.0f}s)", flush=True)

print("\n=== C4 RESULTS, full val (champion C3-recipe = 36.96 / 46.06 / 12.78 / 4452) ===",
      flush=True)
res = {t: hav(v[:, 0], v[:, 1], tlat, tlon) for t, v in preds.items()}
for t, _, _ in V:
    rep(t, res[t])
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes_c4.npz"),
         ids=ep_ids, s=ep_s, bb=ep_bb)
print("episodes cached (episodes_c4.npz)", flush=True)

# ---- E7 learned rerank on the best variant's pools -----------------------------------
la_k, lo_k = trlat[ep_ids], trlon[ep_ids]
valid = ep_s > -1e8
s_k = ep_s.astype(np.float64)
bb_k = ep_bb.astype(np.float64)
arn = np.arange(n)
d_kk = np.zeros((n, K, K), np.float32)
for i in range(n):
    d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])
d_all = hav(la_k, lo_k, tlat[:, None], tlon[:, None])
cc = np.zeros((n, K, K), np.float32)
for i in range(n):
    v = spaces["cls"][0][torch.from_numpy(ep_ids[i])].float().numpy()
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    cc[i] = v @ v.T
print(f"E7 features ({time.time()-T0:.0f}s)", flush=True)


def soft(s, tau=0.05):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * (s > -1e8)
    return w / (w.sum(1, keepdims=True) + 1e-12)


w = soft(s_k)
A = np.where(cc > 0.5, cc, 0)
A = A / (A.sum(2, keepdims=True) + 1e-12)
diffu = np.einsum('nkj,nj->nk', A, w)
cons10 = ((d_kk <= 10) * w[:, None, :]).sum(2)
cons25 = ((d_kk <= 25) * w[:, None, :]).sum(2)
cons50 = ((d_kk <= 50) * w[:, None, :]).sum(2)
ccmean = (cc * w[:, None, :]).sum(2)
ccmax3 = np.sort(cc, axis=2)[:, :, -4:-1].mean(2)
rank_pos = np.tile(np.arange(K, dtype=np.float64), (n, 1))
margin = s_k[:, :1] - s_k
entf = -(w * np.log(np.clip(w, 1e-12, None))).sum(1, keepdims=True) * np.ones((1, K))
nvalid = valid.sum(1, keepdims=True) * np.ones((1, K)) / K
Ff = np.stack([s_k, bb_k, np.log1p(rank_pos), margin, cons10, cons25, cons50,
               diffu * 10, ccmean, ccmax3, entf, nvalid,
               s_k * cons25, bb_k * cons25, margin * cons25, diffu * cons25 * 10], axis=2)
Ff = (Ff - Ff.reshape(-1, Ff.shape[2]).mean(0)) / (Ff.reshape(-1, Ff.shape[2]).std(0) + 1e-9)
y = (d_all <= 25.0).astype(np.float64)


def fit_logistic(X, yy, iters=400, lr=0.5):
    wg = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(X @ wg + b)))
        g = pr - yy
        wg -= lr * (X.T @ g) / len(yy); b -= lr * g.mean()
    return lambda X_: X_ @ wg + b


folds = arn % 5
sc7 = np.zeros((n, K))
for f in range(5):
    tr = folds != f
    Xtr = Ff[tr].reshape(-1, Ff.shape[2])
    mm = valid[tr].ravel()
    model = fit_logistic(Xtr[mm], y[tr].ravel()[mm])
    sc7[~tr] = model(Ff[~tr].reshape(-1, Ff.shape[2])).reshape((~tr).sum(), K)
sc7[~valid] = -1e9
print(f"\n=== E7 on the C4 bed (base: {BEST_FOR_EP}) ===", flush=True)
for lam in (0.5, 1.0, 2.0):
    sb = s_k / s_k[valid].std() + lam * sc7
    j = sb.argmax(1)
    rep(f"E7 blend lam{lam}", hav(la_k[arn, j], lo_k[arn, j], tlat, tlon))
print(f"ALL DONE ({time.time()-T0:.0f}s)", flush=True)
