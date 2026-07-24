#!/usr/bin/env python3
"""Cheap retrieval levers, all on CACHED descriptors — no retraining, no re-inference.

Each lever is evaluated as a delta against the current champion
(cls (+) geo10c, mass .95, lam .05, k1 snap, regional rerank R50 a0.25):

  1 multi-band head ensemble   blends of geo heads trained at d_pos 5/10/25/50 km
  2 confidence router          low top-1 margin -> fall back to the classifier readout
  3 adaptive prior             lambda scaled by how peaked the classifier posterior is
  4 whitening + CSLS           PCA-whitened descriptors; hubness-corrected similarity
  6 location evidence          score a LOCATION by its best two views, then snap

(5 learned score fusion is deliberately left out — it needs a held-out fit and is the one
lever that can leak if done carelessly.)

Usage:
  python retrieval/eval_levers.py --index retrieval_index --tag c3 --gate s384
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.train_place_head import PlaceHead

device = torch.device("cuda")
EARTH = 6371.0088
T0 = time.time()


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, map(lambda v: np.asarray(v, np.float64), (a1, o1, a2, o2)))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def cat(d, p):
    fs = sorted([f for f in os.listdir(d) if f.startswith(p)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs]) if fs else None


def rep(tag, dd, champ=None):
    star = ""
    if champ is not None:
        star = "   <<< NEW CHAMPION" if (dd <= 25).mean() > champ else ""
    print(f"{tag:38s} median {np.median(dd):7.2f}  mean {dd.mean():6.0f}  @1 {100*(dd<=1).mean():5.2f}%  "
          f"@25 {100*(dd<=25).mean():5.2f}%  @200 {100*(dd<=200).mean():5.2f}%  "
          f"GG {(5000*np.exp(-dd/1492.7)).mean():5.0f}{star}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="retrieval_index")
    ap.add_argument("--tag", default="c3")
    ap.add_argument("--gate", default="s384")
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--levers", default="1234 6")
    ap.add_argument("--combo", action="store_true",
                    help="only the winning levers, combined (fast pass)")
    a = ap.parse_args()
    d = a.index

    z = np.load(os.path.join(d, "train_latlon.npz"))
    trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    N = len(trlat)
    vz = np.load(os.path.join(d, "val_meta.npz"))
    tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
    n = len(tlat)
    trlat_t = torch.tensor(trlat, device=device)
    trlon_t = torch.tensor(trlon, device=device)
    _, loc = np.unique(np.stack([trlat, trlon], 1), axis=0, return_inverse=True)
    loc_t = torch.tensor(loc, device=device)

    lg = cat(d, f"c2_logits_val_{a.gate}_r")
    p = F.softmax(torch.tensor(lg, dtype=torch.float32, device=device), dim=-1)
    logp = p.clamp(1e-12).log()
    rank_order = torch.argsort(p, dim=1, descending=True)
    csum = torch.gather(p, 1, rank_order).cumsum(1)
    ent = -(p * logp).sum(1)                                  # posterior entropy per query
    ent_n = (ent / np.log(p.shape[1])).clamp(0, 1)
    cent = torch.tensor(np.load(os.path.join(d, "centroids.npz"))["cent"], dtype=torch.float32,
                        device=device)
    C = cent.shape[0]
    clat = torch.rad2deg(torch.asin(cent[:, 2].clamp(-1, 1)))
    clon = torch.rad2deg(torch.atan2(cent[:, 1], cent[:, 0]))
    lar, lor = np.radians(trlat), np.radians(trlon)
    tru = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                                 np.sin(lar)], -1), dtype=torch.float32, device=device)
    cell_a = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, 200_000):
        cell_a[i:i + 200_000] = (tru[i:i + 200_000] @ cent.T).argmax(1)
    del tru
    order = torch.argsort(cell_a, stable=True)
    cs = cell_a[order]
    ar = torch.arange(C, device=device)
    starts, ends = torch.searchsorted(cs, ar), torch.searchsorted(cs, ar, right=True)

    # classifier readout (fallback for the router): top-16 cells within 1000 km, spherical mean
    w16, i16 = p.topk(16, dim=1)
    d1 = 2 * EARTH * torch.asin(torch.sqrt(torch.clamp(
        torch.sin(torch.deg2rad(clat[i16] - clat[i16[:, :1]]) / 2) ** 2 +
        torch.cos(torch.deg2rad(clat[i16[:, :1]])) * torch.cos(torch.deg2rad(clat[i16])) *
        torch.sin(torch.deg2rad(clon[i16] - clon[i16[:, :1]]) / 2) ** 2, 0, 1)))
    wm = w16 * (d1 <= 1000.0)
    wm = wm / wm.sum(1, keepdim=True).clamp(min=1e-9)
    v = (wm.unsqueeze(-1) * cent[i16]).sum(1)
    v = F.normalize(v, dim=-1)
    cls_lat = torch.rad2deg(torch.asin(v[:, 2].clamp(-1, 1))).cpu().numpy().astype(np.float64)
    cls_lon = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])).cpu().numpy().astype(np.float64)
    print(f"classifier-readout fallback: median {np.median(hav(cls_lat, cls_lon, tlat, tlon)):.1f} km",
          flush=True)

    spaces = {}

    def norm_big(x):
        t = torch.from_numpy(x).to(device)
        for i in range(0, t.shape[0], 200_000):
            t[i:i + 200_000] = F.normalize(t[i:i + 200_000].float(), dim=-1).half()
        return t

    base_tr = norm_big(cat(d, f"c2_cls_train_{a.tag}_r"))
    base_va = F.normalize(torch.from_numpy(cat(d, f"c2_cls_val_{a.tag}_r")).float().to(device), dim=-1)
    spaces["cls"] = (base_tr, base_va)
    sfx = f"_cls_{a.tag}.pt"
    heads = sorted(f for f in os.listdir(d) if f.startswith("geo") and f.endswith(sfx))
    for f in heads:
        nm = f.replace(sfx, "") + "c"                   # geo10_cls_c4.pt -> geo10c
        pth = os.path.join(d, f)
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        head = PlaceHead(ck["dim"], ck["hidden"]).to(device).eval()
        head.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            out = torch.empty_like(base_tr)
            for i in range(0, N, 200_000):
                out[i:i + 200_000] = head(base_tr[i:i + 200_000].float()).half()
            spaces[nm] = (out, head(base_va))
    print(f"spaces {list(spaces)} | VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

    # lever 4a: PCA whitening fitted on a 200k sample of the index
    if "4" in a.levers:
        for src in ("cls",):
            if src not in spaces:
                continue
            tr, va = spaces[src]
            idx = torch.randint(0, N, (200_000,), device=device)
            X = tr[idx].float()
            mu = X.mean(0)
            Xc = X - mu
            cov = (Xc.T @ Xc) / X.shape[0]
            ev, V = torch.linalg.eigh(cov.double())
            W = (V / ev.clamp(min=1e-6).sqrt()).float()
            out = torch.empty_like(tr)
            for i in range(0, N, 200_000):
                out[i:i + 200_000] = F.normalize((tr[i:i + 200_000].float() - mu) @ W, dim=-1).half()
            spaces[f"{src}_wh"] = (out, F.normalize((va - mu) @ W, dim=-1))
            del X, Xc, cov, ev, V
            torch.cuda.empty_cache()
            print(f"whitened {src}", flush=True)

    # lever 4b: CSLS hubness correction — r(x) = mean top-10 sim of x to 20k pseudo-queries
    csls_r = {}
    if "4" in a.levers:
        q = torch.randint(0, N, (10_000,), device=device)
        for src in [k for k in list(spaces) if not k.endswith("_wh")]:
            tr, _ = spaces[src]
            Q = tr[q]                       # keep fp16: 10k x 1024
            r = torch.empty(N, device=device, dtype=torch.float16)
            for i in range(0, N, 20_000):
                sims = tr[i:i + 20_000] @ Q.T            # 20k x 10k fp16 = 0.4 GB
                r[i:i + 20_000] = sims.topk(10, dim=1).values.mean(1)
                del sims
            torch.cuda.empty_cache()
            csls_r[src] = r
            print(f"csls r() for {src}: mean {r.mean():.3f}", flush=True)

    reg_all = reg_va = None
    if os.path.exists(os.path.join(d, f"c2_reg_train_{a.tag}_r0.npy")):
        reg_all = torch.from_numpy(cat(d, f"c2_reg_train_{a.tag}_r"))
        reg_va = torch.from_numpy(cat(d, f"c2_reg_val_{a.tag}_r")).float().to(device)

    # ---- variants: (tag, weights, lam-mode, extras)
    have = set(spaces)
    V = []
    V.append(("CHAMPION cls|geo10c +rr", {"cls": .5, "geo10c": .5}, "fixed", {}))
    if "1" in a.levers:
        for nm in ("geo5c", "geo25c", "geo50c"):
            if nm in have:
                V.append((f"L1 cls|{nm}", {"cls": .5, nm: .5}, "fixed", {}))
        if {"geo5c", "geo25c"} <= have:
            V.append(("L1 3-band cls|geo10c|geo25c", {"cls": .4, "geo10c": .3, "geo25c": .3}, "fixed", {}))
            V.append(("L1 4-band all", {"cls": .34, "geo5c": .22, "geo10c": .22, "geo25c": .22},
                      "fixed", {}))
    if "3" in a.levers:
        V.append(("L3 adaptive lambda (peaked)", {"cls": .5, "geo10c": .5}, "adaptive", {}))
        V.append(("L3 adaptive lambda strong", {"cls": .5, "geo10c": .5}, "adaptive2", {}))
    if "4" in a.levers:
        if "cls_wh" in have:
            V.append(("L4 whitened cls|geo10c", {"cls_wh": .5, "geo10c": .5}, "fixed", {}))
        if "geo10c_wh" in have:
            V.append(("L4 whitened both", {"cls_wh": .5, "geo10c_wh": .5}, "fixed", {}))
        if csls_r:
            V.append(("L4 CSLS", {"cls": .5, "geo10c": .5}, "fixed", {"csls": True}))
    if "6" in a.levers:
        V.append(("L6 location evidence (top2)", {"cls": .5, "geo10c": .5}, "fixed", {"locagg": True}))

    if a.combo:
        b4 = {"cls": .34, "geo5c": .22, "geo10c": .22, "geo25c": .22}
        b3 = {"cls": .4, "geo10c": .3, "geo25c": .3}
        V = [("CHAMPION cls|geo10c +rr", {"cls": .5, "geo10c": .5}, "fixed", {}),
             ("L4 CSLS (prev best)", {"cls": .5, "geo10c": .5}, "fixed", {"csls": True}),
             ("COMBO 4band+CSLS", b4, "fixed", {"csls": True}),
             ("COMBO 3band+CSLS", b3, "fixed", {"csls": True}),
             ("COMBO 4band+CSLS+wh", {**{k: v for k, v in b4.items() if k != "cls"},
                                      "cls_wh": .34}, "fixed", {"csls": True})]
        V = [v for v in V if all(k in spaces for k in v[1])]
    need = sorted({s for _, w, _, _ in V for s in w})
    m_of = (csum < 0.95).sum(1) + 1
    preds = {t: np.zeros((n, 2)) for t, _, _, _ in V}
    margins = {t: np.zeros(n) for t, _, _, _ in V}
    with torch.no_grad():
        for i in range(n):
            m = min(int(m_of[i]), a.cap)
            ids = torch.cat([order[starts[c]:ends[c]] for c in rank_order[i, :m]])
            lp = logp[i, cell_a[ids]].double()
            sims = {nm: spaces[nm][0][ids].float() @ spaces[nm][1][i] for nm in need}
            for tag, w, lam_mode, ex in V:
                s = sum(wt * sims[nm] for nm, wt in w.items())
                if ex.get("csls"):
                    corr = sum(wt * csls_r[nm][ids].float() for nm, wt in w.items() if nm in csls_r)
                    s = 2 * s - corr
                lam = 0.05
                if lam_mode == "adaptive":
                    lam = 0.05 * float(2.0 - ent_n[i])
                elif lam_mode == "adaptive2":
                    lam = 0.05 * float(3.0 - 2.0 * ent_n[i])
                sc = s.double() + lam * lp
                if ex.get("locagg"):
                    ls = loc_t[ids]
                    uniq, inv = torch.unique(ls, return_inverse=True)
                    best = torch.full((uniq.numel(),), -1e9, dtype=torch.float64, device=device)
                    best.scatter_reduce_(0, inv, sc, reduce="amax", include_self=True)
                    sc2 = sc.clone()
                    sc2[sc == best[inv]] = -1e9          # remove each location's best -> 2nd best
                    second = torch.full_like(best, -1e9)
                    second.scatter_reduce_(0, inv, sc2, reduce="amax", include_self=True)
                    locscore = torch.where(second > -1e8, 0.7 * best + 0.3 * second, best)
                    bl = int(locscore.argmax())
                    cand = (inv == bl).nonzero().flatten()
                    j = ids[cand[int(sc[cand].argmax())]]
                else:
                    top = sc.topk(min(50, sc.numel()))
                    if reg_all is not None:
                        rt = reg_all[ids[top.indices].cpu()].float().to(device)
                        ch = (rt @ reg_va[i].T).max(2).values.mean(1)
                        sc_r = top.values + 0.25 * ch.double()
                        j = ids[top.indices[int(sc_r.argmax())]]
                    else:
                        j = ids[top.indices[0]]
                    margins[tag][i] = float(top.values[0] - top.values[min(4, top.values.numel() - 1)])
                jj = int(j)
                preds[tag][i] = (float(trlat_t[jj]), float(trlon_t[jj]))
            if i and i % 1000 == 0:
                print(f"  {i}/{n} ({time.time()-T0:.0f}s)", flush=True)

    print("\n=== RESULTS (full val) ===", flush=True)
    res = {t: hav(v[:, 0], v[:, 1], tlat, tlon) for t, v in preds.items()}
    champ = (res["CHAMPION cls|geo10c +rr"] <= 25).mean()
    rep("CHAMPION cls|geo10c +rr", res["CHAMPION cls|geo10c +rr"])
    for t in sorted([k for k in res if k != "CHAMPION cls|geo10c +rr"],
                    key=lambda k: -(res[k] <= 25).mean()):
        rep(t, res[t], champ)

    if "2" in a.levers:                     # router: swap in the classifier when the match is weak
        print("\n--- L2 confidence router on the best variant ---", flush=True)
        bt = max(res, key=lambda k: (res[k] <= 25).mean())
        mg = margins[bt]
        for q in (0.05, 0.1, 0.2, 0.3):
            thr = np.quantile(mg[mg > 0], q) if (mg > 0).any() else 0
            use_cls = mg < thr
            lat = np.where(use_cls, cls_lat, preds[bt][:, 0])
            lon = np.where(use_cls, cls_lon, preds[bt][:, 1])
            dd = hav(lat, lon, tlat, tlon)
            res[f"L2 router q{q} ({100*use_cls.mean():.0f}% routed)"] = dd
            rep(f"L2 router q{q} ({100*use_cls.mean():.0f}% routed)", dd, champ)

    best = max(res, key=lambda k: ((res[k] <= 25).mean(), -np.median(res[k])))
    dd = res[best]
    gg = 5000 * np.exp(-dd / 1492.7)
    print(f"\n=== BEST: {best} ===", flush=True)
    print(f"median {np.median(dd):.2f} | mean {dd.mean():.1f}", flush=True)
    for r in (1, 5, 10, 25, 50, 100, 200, 1000):
        print(f"  @{r:>5} km : {100*(dd<=r).mean():5.2f}%", flush=True)
    print(f"GeoGuessr {gg.mean():.0f}/round | 5-round {5*gg.mean():.0f}", flush=True)
    with open(os.path.join(d, "levers_results.json"), "w") as fh:
        json.dump({k: {"median": float(np.median(v)), "acc25": float((v <= 25).mean()),
                       "acc1": float((v <= 1).mean()), "mean": float(v.mean())}
                   for k, v in res.items()}, fh, indent=1)
    print("LEVERS DONE", flush=True)


if __name__ == "__main__":
    main()
