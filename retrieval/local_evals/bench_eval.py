#!/usr/bin/env python3
"""Score external benchmarks (im2gps / im2gps3k) with the frozen champion recipe:
C4 bed -> gate mass .95 cap 400 -> 4band+CSLS+wh blend + 0.05*logp prior -> E7 rerank.

Protocol (no benchmark leakage):
  1. Bed prep identical to local_c4_eval.py (seeded so whitening/CSLS are reproducible).
  2. Val (2998) is scored first; E7 is 5-fold-CV'd on val to reproduce the 34.10 champion
     as a sanity anchor, then REFIT ONCE on all of val.
  3. Benchmarks are scored with everything frozen: same blend, same E7 weights, feature
     normalization stats taken from val. lam=2 is the champion dial.
Also reports a classifier-only baseline (argmax cell -> centroid) per benchmark set.

Usage:
  uv run python retrieval/local_evals/bench_eval.py \
      --bench im2gps=benchmarks/embeds/im2gps --bench im2gps3k=benchmarks/embeds/im2gps3k
(each dir must hold {tag}_cls.npy, {tag}_logits.npy, {tag}_latlon.npz from bench_embed.py)
"""
import argparse
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
torch.manual_seed(0)
EARTH = 6371.0088
T0 = time.time()
K = 100
LAM = 2.0
THRESH = (1, 25, 200, 750, 2500)


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, map(lambda v: np.asarray(v, np.float64), (a1, o1, a2, o2)))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def rep(tag, dd):
    accs = "  ".join(f"@{r} {100*(dd<=r).mean():5.2f}%" for r in THRESH)
    print(f"{tag:40s} median {np.median(dd):7.2f}  mean {dd.mean():6.0f}  {accs}  "
          f"GG {(5000*np.exp(-dd/1492.7)).mean():5.0f}", flush=True)


def cat(p):
    fs = sorted([f for f in os.listdir(IX) if f.startswith(p)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(IX, f)) for f in fs])


ap = argparse.ArgumentParser()
ap.add_argument("--bench", action="append", default=[],
                help="name=dir with {name}_cls.npy/_logits.npy/_latlon.npz")
ap.add_argument("--out", default=os.path.join(REPO, "benchmarks", "results"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------- bed prep (faithful to local_c4_eval.py) ----------------
z = np.load(os.path.join(IX, "train_latlon.npz"))
trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
N = len(trlat)

ck = torch.load(os.path.join(IX, "ckpt.pt"), map_location="cpu", weights_only=False)
cent = F.normalize(ck["buffers"]["centroids"].float(), dim=-1)
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
cent_lat = np.degrees(np.arcsin(np.clip(cent[:, 2].numpy(), -1, 1)))
cent_lon = np.degrees(np.arctan2(cent[:, 1].numpy(), cent[:, 0].numpy()))
print(f"gate ready ({time.time()-T0:.0f}s): {C} cells", flush=True)

spaces_tr = {}
tr_cls = torch.from_numpy(cat("c2_cls_train_c4_r"))
for i in range(0, N, 200_000):
    tr_cls[i:i + 200_000] = F.normalize(tr_cls[i:i + 200_000].float(), dim=-1).half()
spaces_tr["cls"] = tr_cls
heads = {}
for nm, f in (("geo5c", "geo5_cls_c4.pt"), ("geo10c", "geo10_cls_c4.pt"),
              ("geo25c", "geo25_cls_c4.pt")):
    hck = torch.load(os.path.join(IX, f), map_location="cpu", weights_only=False)
    head = PlaceHead(hck["dim"], hck["hidden"]).eval()
    head.load_state_dict(hck["state_dict"])
    heads[nm] = head
    with torch.no_grad():
        out = torch.empty_like(tr_cls)
        for i in range(0, N, 100_000):
            out[i:i + 100_000] = head(tr_cls[i:i + 100_000].float()).half()
        spaces_tr[nm] = out
    print(f"space {nm} ready ({time.time()-T0:.0f}s)", flush=True)

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
spaces_tr["cls_wh"] = out
del X, Xc, cov, ev, Vv
print(f"whitened cls ready ({time.time()-T0:.0f}s)", flush=True)

csls_r = {}
q = torch.randint(0, N, (5_000,))
for src in ("cls", "geo5c", "geo10c", "geo25c"):
    tr = spaces_tr[src]
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
V = [("cls|geo10c (C3 recipe)", {"cls": .5, "geo10c": .5}, {}),
     ("4band+CSLS", b4, {"csls": True}),
     ("4band+CSLS+wh", {**{k: v for k, v in b4.items() if k != "cls"}, "cls_wh": .34},
      {"csls": True})]
BEST = "4band+CSLS+wh"
need = sorted({s for _, w, _ in V for s in w})


def query_spaces(cls_raw):
    """raw fp16 cls (m,1024) -> per-space normalized query vectors."""
    va = F.normalize(torch.from_numpy(cls_raw).float(), dim=-1)
    out = {"cls": va, "cls_wh": F.normalize((va - mu) @ W, dim=-1)}
    with torch.no_grad():
        for nm, head in heads.items():
            out[nm] = head(va)
    return out


def score_set(tag, cls_raw, logits_raw):
    """Gated blend scoring. Returns preds per variant + E7 pools + argmax-cell baseline."""
    lg = torch.tensor(logits_raw, dtype=torch.float32)
    p = F.softmax(lg, dim=-1)
    logp = p.clamp(1e-12).log().double()
    rank_order = torch.argsort(p, dim=1, descending=True)
    csum = torch.gather(p, 1, rank_order).cumsum(1)
    m_of = (csum < 0.95).sum(1) + 1
    qs = query_spaces(cls_raw)
    n = qs["cls"].shape[0]
    preds = {t: np.zeros((n, 2)) for t, _, _ in V}
    ep_ids = np.zeros((n, K), np.int64)
    ep_s = np.full((n, K), -1e9, np.float32)
    ep_bb = np.zeros((n, K), np.float32)
    with torch.no_grad():
        for i in range(n):
            m = min(int(m_of[i]), 400)
            ids = torch.cat([order[starts[c]:ends[c]] for c in rank_order[i, :m]])
            lp = logp[i, cell_a[ids]]
            sims = {nm: spaces_tr[nm][ids].float() @ qs[nm][i] for nm in need}
            for t, w, ex in V:
                s = sum(wt * sims[nm] for nm, wt in w.items())
                if ex.get("csls"):
                    corr = sum(wt * csls_r[nm][ids] for nm, wt in w.items() if nm in csls_r)
                    s = 2 * s - corr
                sc = s.double() + 0.05 * lp
                j = int(ids[int(sc.argmax())])
                preds[t][i] = (trlat[j], trlon[j])
                if t == BEST:
                    k = min(K, sc.numel())
                    tk = sc.topk(k)
                    ep_ids[i, :k] = ids[tk.indices].numpy()
                    ep_s[i, :k] = tk.values.float().numpy()
                    ep_bb[i, :k] = sims["cls"][tk.indices].numpy()
            if i and i % 500 == 0:
                print(f"  [{tag}] {i}/{n} ({time.time()-T0:.0f}s)", flush=True)
    base = np.stack([cent_lat[rank_order[:, 0].numpy()],
                     cent_lon[rank_order[:, 0].numpy()]], 1)
    return preds, (ep_ids, ep_s, ep_bb), base


def e7_features(pools):
    """The 16 E7 features per (query, candidate). Returns Ff (n,K,16), valid, la_k, lo_k."""
    ep_ids, ep_s, ep_bb = pools
    n = ep_ids.shape[0]
    la_k, lo_k = trlat[ep_ids], trlon[ep_ids]
    valid = ep_s > -1e8
    s_k = ep_s.astype(np.float64)
    bb_k = ep_bb.astype(np.float64)
    d_kk = np.zeros((n, K, K), np.float32)
    for i in range(n):
        d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])
    cc = np.zeros((n, K, K), np.float32)
    for i in range(n):
        v = spaces_tr["cls"][torch.from_numpy(ep_ids[i])].float().numpy()
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        cc[i] = v @ v.T

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
    return Ff, valid, la_k, lo_k, s_k


def fit_logistic(X, yy, iters=400, lr=0.5):
    wg = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(X @ wg + b)))
        g = pr - yy
        wg -= lr * (X.T @ g) / len(yy); b -= lr * g.mean()
    return wg, b


def e7_apply(Ff, valid, s_k, la_k, lo_k, wg, b, lam=LAM):
    sc7 = (Ff.reshape(-1, Ff.shape[2]) @ wg + b).reshape(Ff.shape[0], K)
    sc7[~valid] = -1e9
    sb = s_k / s_k[valid].std() + lam * sc7
    j = sb.argmax(1)
    arn = np.arange(Ff.shape[0])
    return np.stack([la_k[arn, j], lo_k[arn, j]], 1)


# ---------------- val: sanity anchor + E7 fit ----------------
vz = np.load(os.path.join(IX, "val_meta.npz"))
vlat, vlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
v_cls = cat("c2_cls_val_c4_r")
v_log = cat("c2_logits_val_c4_r")
preds, pools, _ = score_set("val", v_cls, v_log)
print(f"\n=== VAL (n={len(vlat)}) — expect base 40.98, E7 lam2 CV 34.10 ===", flush=True)
for t, _, _ in V:
    rep("val " + t, hav(preds[t][:, 0], preds[t][:, 1], vlat, vlon))

Ff, valid, la_k, lo_k, s_k = e7_features(pools)
fmu = Ff.reshape(-1, Ff.shape[2]).mean(0)
fsd = Ff.reshape(-1, Ff.shape[2]).std(0) + 1e-9
Ffn = (Ff - fmu) / fsd
n = Ffn.shape[0]
y = (hav(la_k, lo_k, vlat[:, None], vlon[:, None]) <= 25.0).astype(np.float64)

folds = np.arange(n) % 5
pcv = np.zeros((n, 2))
for f in range(5):
    tr = folds != f
    Xtr = Ffn[tr].reshape(-1, Ffn.shape[2])
    mm_ = valid[tr].ravel()
    wg, b = fit_logistic(Xtr[mm_], y[tr].ravel()[mm_])
    pcv[~tr] = e7_apply(Ffn[~tr], valid[~tr], s_k[~tr], la_k[~tr], lo_k[~tr], wg, b)
rep("val E7 lam2 (5-fold CV, champion)", hav(pcv[:, 0], pcv[:, 1], vlat, vlon))

Xall = Ffn.reshape(-1, Ffn.shape[2])
wg, b = fit_logistic(Xall[valid.ravel()], y.ravel()[valid.ravel()])
pfull = e7_apply(Ffn, valid, s_k, la_k, lo_k, wg, b)
rep("val E7 lam2 (full-val fit, optimistic)", hav(pfull[:, 0], pfull[:, 1], vlat, vlon))

# ---------------- benchmarks: frozen application ----------------
for spec in a.bench:
    name, d = spec.split("=", 1)
    bcls = np.load(os.path.join(d, f"{name}_cls.npy"))
    blog = np.load(os.path.join(d, f"{name}_logits.npy"))
    bz = np.load(os.path.join(d, f"{name}_latlon.npz"))
    blat, blon = bz["lat"], bz["lon"]
    preds, pools, base = score_set(name, bcls, blog)
    print(f"\n=== {name.upper()} (n={len(blat)}) — frozen champion, no benchmark fitting ===",
          flush=True)
    res = {"classifier argmax cell": hav(base[:, 0], base[:, 1], blat, blon)}
    for t, _, _ in V:
        res[t] = hav(preds[t][:, 0], preds[t][:, 1], blat, blon)
    BFf, Bval, Bla, Blo, Bs = e7_features(pools)
    BFfn = (BFf - fmu) / fsd
    for lam in (1.0, LAM):
        pb = e7_apply(BFfn, Bval, Bs, Bla, Blo, wg, b, lam=lam)
        res[f"E7 lam{lam:g} (champion)" if lam == LAM else f"E7 lam{lam:g}"] = \
            hav(pb[:, 0], pb[:, 1], blat, blon)
        if lam == LAM:
            champ_pred = pb
    for t, dd in res.items():
        rep(f"{name} {t}", dd)
    np.savez(os.path.join(a.out, f"{name}_results.npz"),
             dist_champion=res[f"E7 lam{LAM:g} (champion)"], pred_champion=champ_pred,
             dist_base=res[BEST], lat=blat, lon=blon, path=bz["path"],
             **{f"dist_{t.replace(' ', '_')}": dd for t, dd in res.items()})
    print(f"saved {a.out}/{name}_results.npz", flush=True)

print(f"ALL DONE ({time.time()-T0:.0f}s)", flush=True)
