#!/usr/bin/env python3
"""C2 retrieval grid on GPU — spaces, blends, gate retune, query TTA, regional rerank.

Stages (each narrows to the best config of the previous one, ranked by acc@25km):
  A  single spaces (raw 512 CLS / mean-patch / old 384 index / geo-head projections)
  B  50/50 similarity blends of the best spaces
  C  gate mass x classifier-prior lambda retune + query TTA (base | +flip | +flip+zoom)
  D  regional (4x4 patch-block) rerank of the top-R candidates, chamfer similarity
  F  final champion report: median, mean, percentile profile, accuracy curve, GeoGuessr score

Usage on the box:
  /venv/main/bin/python retrieval/eval_c2_gpu.py --index retrieval_index --gate 384
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


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, map(lambda v: np.asarray(v, np.float64),
                                                 (lat1, lon1, lat2, lon2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cat(d, prefix):
    fs = sorted([f for f in os.listdir(d) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs]) if fs else None


def report(tag, dd):
    print(f"{tag:44s} median {np.median(dd):7.2f}  mean {dd.mean():6.0f}  "
          f"@1 {100*(dd<=1).mean():5.2f}%  @25 {100*(dd<=25).mean():5.2f}%  "
          f"@200 {100*(dd<=200).mean():5.2f}%  GG {(5000*np.exp(-dd/1492.7)).mean():5.0f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="retrieval_index")
    ap.add_argument("--ckpt", default="run_full2/ckpt_best.pt")
    ap.add_argument("--gate", default="s384", help="tag of the val logits used to gate")
    ap.add_argument("--stages", default="ABCDF")
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--tag", default="s512", help="descriptor tag to load (s512 | s384)")
    ap.add_argument("--old", action="store_true", help="also load the old 384 index as a space")
    ap.add_argument("--extra-tags", default="",
                    help="also load cls/mean descriptors from these tags as <kind>_<tag> spaces")
    ap.add_argument("--heads", default="",
                    help="extra heads as name:file:base[,...] (base = cls|mean)")
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

    # ---- gate: classifier posterior over cells + cell buckets over the index
    lg = cat(d, f"c2_logits_val_{a.gate}_r")
    p = F.softmax(torch.tensor(lg, dtype=torch.float32, device=device), dim=-1)
    logp = p.clamp(1e-12).log()
    rank_order = torch.argsort(p, dim=1, descending=True)
    csum = torch.gather(p, 1, rank_order).cumsum(1)
    cpath = os.path.join(d, "centroids.npz")
    if os.path.exists(cpath):
        cent_np = np.load(cpath)["cent"]
    else:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cent_np = ck["buffers"]["centroids"].numpy()
        np.savez(cpath, cent=cent_np)
        print("[grid] centroids taken from ckpt buffers", flush=True)
    cent = torch.tensor(cent_np, dtype=torch.float32, device=device)
    C = cent.shape[0]
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
    print(f"gate={a.gate} | N={N} val={n} cells={C} ({time.time()-T0:.0f}s)", flush=True)

    # ---- embedding spaces (train fp16 on GPU, val fp32) + query-TTA variants
    spaces, valvars = {}, {}

    def norm_big(x):
        t = torch.from_numpy(x).to(device)
        for i in range(0, t.shape[0], 200_000):
            t[i:i + 200_000] = F.normalize(t[i:i + 200_000].float(), dim=-1).half()
        return t

    def norm_v(x):
        return F.normalize(torch.from_numpy(x).float().to(device), dim=-1)

    def add(name, tr_np, va_np, variants=None):
        spaces[name] = (norm_big(tr_np), norm_v(va_np))
        if variants:
            vv = {k: norm_v(v) for k, v in variants.items() if v is not None}
            if vv:
                valvars[name] = vv

    tg = a.tag
    add("cls", cat(d, f"c2_cls_train_{tg}_r"), cat(d, f"c2_cls_val_{tg}_r"))
    add("mean", cat(d, f"c2_mean_train_{tg}_r"), cat(d, f"c2_mean_val_{tg}_r"))
    for xt in [x for x in a.extra_tags.split(",") if x]:
        for kind in ("cls", "mean"):
            tr_np = cat(d, f"c2_{kind}_train_{xt}_r")
            if tr_np is not None:
                add(f"{kind}_{xt}", tr_np, cat(d, f"c2_{kind}_val_{xt}_r"))
    if a.old:
        old_tr = cat(d, "emb_bb_train_r")
        if old_tr is not None:
            old_va = cat(d, "c2_cls_val_s384_r")
            add("old384", old_tr, old_va if old_va is not None else cat(d, "emb_bb_val_r"))

    def head_space(name, fname, base):
        pth = os.path.join(d, fname)
        if not os.path.exists(pth) or base not in spaces:
            return
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        head = PlaceHead(ck["dim"], ck["hidden"]).to(device).eval()
        head.load_state_dict(ck["state_dict"])
        tr, va = spaces[base]
        with torch.no_grad():
            out = torch.empty_like(tr)
            for i in range(0, tr.shape[0], 200_000):
                out[i:i + 200_000] = head(tr[i:i + 200_000].float()).half()
            spaces[name] = (out, head(va))
            if base in valvars:
                valvars[name] = {k: head(v) for k, v in valvars[base].items()}

    for spec in [x for x in a.heads.split(",") if x]:
        nm, fn, base = spec.split(":")
        head_space(nm, fn, base)
    if a.old:
        head_space("geo10_old384", "geo_head_10.pt", "old384")
        head_space("c1_old384", "place_head.pt", "old384")
    print(f"spaces: {list(spaces)} | VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB "
          f"({time.time()-T0:.0f}s)", flush=True)

    # ---- regional descriptors (CPU tensor; only top-R rows are gathered per query)
    reg_all = reg_va = None
    if os.path.exists(os.path.join(d, f"c2_reg_train_{a.tag}_r0.npy")):
        reg_all = torch.from_numpy(cat(d, f"c2_reg_train_{a.tag}_r"))
        reg_va = torch.from_numpy(cat(d, f"c2_reg_val_{a.tag}_r")).float().to(device)
        print(f"regions {tuple(reg_all.shape)} on CPU ({time.time()-T0:.0f}s)", flush=True)

    def run(cfgs, mass=0.95, tta=None, rerank=None):
        """cfgs: [(tag, {space: weight}, lam)] — all share one gate pass over the queries.
        rerank: (max_R, [(R, alpha), ...]) — chamfer computed once for max_R per config."""
        need = sorted({s for _, w, _ in cfgs for s in w})
        m_of = (csum < mass).sum(1) + 1
        tags = [t for t, _, _ in cfgs] if not rerank else \
            [f"{t} rr:R{R}a{al}" for t, _, _ in cfgs for R, al in rerank[1]]
        preds = {t: np.zeros((n, 2)) for t in tags}
        with torch.no_grad():
            for i in range(n):
                m = min(int(m_of[i]), a.cap)
                cells = rank_order[i, :m]
                ids = torch.cat([order[starts[c]:ends[c]] for c in cells])
                lp = logp[i, cell_a[ids]].double()
                sims = {}
                for nm in need:
                    tr, va = spaces[nm]
                    q = va[i]
                    if tta and nm in valvars:
                        q = F.normalize(sum([va[i]] + [valvars[nm][k][i] for k in tta
                                                       if k in valvars[nm]]), dim=-1)
                    sims[nm] = tr[ids].float() @ q
                for tag, w, lam in cfgs:
                    s = sum(wt * sims[nm] for nm, wt in w.items()).double() + lam * lp
                    if rerank is None:
                        jj = int(ids[int(s.argmax())])
                        preds[tag][i] = (float(trlat_t[jj]), float(trlon_t[jj]))
                        continue
                    maxR, combos = rerank
                    top = s.topk(min(maxR, s.numel())).indices
                    rt = reg_all[ids[top].cpu()].float().to(device)
                    ch = (rt @ reg_va[i].T).max(2).values.mean(1)     # chamfer over 16 regions
                    for R, al in combos:
                        k = min(R, top.numel())
                        sc = s[top[:k]] + al * ch[:k].double()
                        jj = int(ids[top[:k][int(sc.argmax())]])
                        preds[f"{tag} rr:R{R}a{al}"][i] = (float(trlat_t[jj]), float(trlon_t[jj]))
                if i and i % 1500 == 0:
                    print(f"    {i}/{n} ({time.time()-T0:.0f}s)", flush=True)
        return {t: hav(v[:, 0], v[:, 1], tlat, tlon) for t, v in preds.items()}

    RES, META = {}, {}

    def record(tag, dd, cfg, mass, lam, tta, rerank):
        RES[tag] = dd
        META[tag] = dict(cfg=cfg, mass=mass, lam=lam, tta=tta, rerank=rerank)
        report(tag, dd)

    def best_tag():
        return max(RES, key=lambda t: ((RES[t] <= 25).mean(), -np.median(RES[t])))

    top = []
    if "A" in a.stages:
        print("\n=== STAGE A: single spaces (mass .95, lam .05, k1 snap) ===", flush=True)
        cfgs = [(nm, {nm: 1.0}, 0.05) for nm in spaces]
        res = run(cfgs)
        for t in sorted(res, key=lambda t: -(res[t] <= 25).mean()):
            record(t, res[t], {t: 1.0}, 0.95, 0.05, None, None)
        top = sorted(res, key=lambda t: -(res[t] <= 25).mean())[:3]
        print(f"best single: {best_tag()} | top3 {top}", flush=True)

    if "B" in a.stages:
        print("\n=== STAGE B: 50/50 similarity blends ===", flush=True)
        seen, pairs = set(), []
        for i, x in enumerate(top):
            for y in list(top[i + 1:]) + [s for s in spaces if s not in top]:
                k = tuple(sorted((x, y)))
                if x != y and k not in seen:
                    seen.add(k)
                    pairs.append(k)
        cfgs = [(f"{x}|{y}", {x: 0.5, y: 0.5}, 0.05) for x, y in pairs]
        res = run(cfgs)
        for t in sorted(res, key=lambda t: -(res[t] <= 25).mean()):
            record(t, res[t], dict(zip(t.split("|"), [0.5, 0.5])), 0.95, 0.05, None, None)
        print(f"best after blends: {best_tag()}", flush=True)

    if "C" in a.stages:
        b = best_tag()
        cfg = META[b]["cfg"]
        print(f"\n=== STAGE C: gate / lambda / query-TTA on [{b}] ===", flush=True)
        for mass in (0.90, 0.95, 0.99):
            cfgs = [(f"{b} m{mass} l{lam}", cfg, lam) for lam in (0.0, 0.05, 0.15)]
            res = run(cfgs, mass=mass)
            for tag, _, lam in cfgs:
                record(tag, res[tag], cfg, mass, lam, None, None)
        b2 = best_tag()
        mass, lam = META[b2]["mass"], META[b2]["lam"]
        for tta in (["flip"], ["flip", "zoom"]):
            tag = f"{b} m{mass} l{lam} tta:{'+'.join(tta)}"
            record(tag, run([(tag, cfg, lam)], mass=mass, tta=tta)[tag], cfg, mass, lam, tta, None)
        print(f"best after C: {best_tag()}", flush=True)

    if "D" in a.stages and reg_all is not None:
        b = best_tag()
        M = META[b]
        print(f"\n=== STAGE D: regional rerank on [{b}] ===", flush=True)
        combos = [(R, al) for R in (50, 200) for al in (0.25, 0.5, 1.0)]
        res = run([(b, M["cfg"], M["lam"])], mass=M["mass"], tta=M["tta"],
                  rerank=(200, combos))
        for tag in sorted(res, key=lambda t: -(res[t] <= 25).mean()):
            R, al = tag.rsplit("rr:R", 1)[1].split("a")
            record(tag, res[tag], M["cfg"], M["mass"], M["lam"], M["tta"], (int(R), float(al)))
        print(f"best after D: {best_tag()}", flush=True)

    if "F" in a.stages:
        b = best_tag()
        dd = RES[b]
        gg = 5000 * np.exp(-dd / 1492.7)
        bm = min(RES, key=lambda t: np.median(RES[t]))
        print(f"\n=== CHAMPION: {b} (full val, n={n}) ===", flush=True)
        print(f"config: {META[b]}", flush=True)
        print(f"median {np.median(dd):.2f} km | mean {dd.mean():.1f} km", flush=True)
        for r in (0.1, 0.5, 1, 5, 10, 25, 50, 100, 200, 500, 750, 1000, 2500):
            print(f"  @{r:>6} km : {100*(dd<=r).mean():5.2f}%", flush=True)
        for q in (10, 25, 50, 75, 90, 95, 99):
            print(f"  p{q:<3d} : {np.percentile(dd, q):9.1f} km", flush=True)
        print(f"GeoGuessr: mean/round {gg.mean():.0f} | median round {np.median(gg):.0f} | "
              f"5-round {5*gg.mean():.0f} | >=4500 {100*(gg>=4500).mean():.1f}% | "
              f">=4900 {100*(gg>=4900).mean():.1f}%", flush=True)
        print(f"(best-by-median config: {bm} -> {np.median(RES[bm]):.2f} km, "
              f"@25 {100*(RES[bm]<=25).mean():.2f}%)", flush=True)
        np.save(os.path.join(d, "c2_champion_dist_km.npy"), dd)
        with open(os.path.join(d, "c2_results.json"), "w") as fh:
            json.dump({k: {"median": float(np.median(v)), "mean": float(v.mean()),
                           "acc1": float((v <= 1).mean()), "acc25": float((v <= 25).mean()),
                           "acc200": float((v <= 200).mean()),
                           "gg": float((5000 * np.exp(-v / 1492.7)).mean()),
                           "meta": {kk: str(vv) for kk, vv in META[k].items()}}
                       for k, v in RES.items()}, fh, indent=1)
    print(f"GRID DONE ({(time.time()-T0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
