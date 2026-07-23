#!/usr/bin/env python3
"""Tune + score the retrieval engine on the full val split — runs ON THE BOX after
retrieval/embed_full.py has produced the index shards.

Two-phase search so the grid is cheap:
  phase 1: per (mass, tap) sweep the 2998 queries ONCE, caching top-32 neighbor sims +
           unit vectors (the expensive gather+matmul work);
  phase 2: every (k, sim_temp, smax_floor) config is vectorized math on the cache —
           the full grid costs seconds.

Usage (box):
  cd /root/auto && PYTHONPATH=/root/auto /venv/main/bin/python retrieval/eval_retrieval.py \
      --ckpt run_full/ckpt_best.pt --index retrieval_index
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare import load_index
from retrieval.engine import (EARTH_RADIUS_KM, RetrievalIndex, haversine_km,
                              latlon_to_unit, unit_to_latlon)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_TEMP = 0.5
PRED_RADIUS_KM = 1000.0
PRED_TOPK = 16


def cat_shards(d, prefix, split):
    fs = sorted([f for f in os.listdir(d) if f.startswith(f"{prefix}_{split}_r")],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    assert fs, f"no shards {prefix}_{split}_r*.npy in {d}"
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs])


def metrics(pred, tlat, tlon):
    d = haversine_km(torch.as_tensor(pred[:, 0]), torch.as_tensor(pred[:, 1]),
                     torch.as_tensor(tlat, dtype=torch.float64),
                     torch.as_tensor(tlon, dtype=torch.float64)).numpy()
    return {"median": float(np.median(d)), "mean": float(d.mean()),
            **{f"acc{r}": float((d <= r).mean()) for r in (1, 25, 200, 750, 2500)}}


def fmt(m):
    return (f"median {m['median']:7.2f}  mean {m['mean']:6.0f}  @1 {100*m['acc1']:.2f}%  "
            f"@25 {100*m['acc25']:5.2f}%  @200 {100*m['acc200']:5.2f}%  "
            f"@750 {100*m['acc750']:5.2f}%  @2500 {100*m['acc2500']:5.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="run_full/ckpt_best.pt")
    ap.add_argument("--index", default="retrieval_index")
    ap.add_argument("--taps", default="bb,trunk")
    ap.add_argument("--masses", default="0.8,0.9,0.95")
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--temps", default="0.02,0.05,0.1,0.2")
    a = ap.parse_args()
    TAPS = a.taps.split(",")
    MASSES = [float(x) for x in a.masses.split(",")]
    KS = [int(x) for x in a.ks.split(",")]
    TEMPS = [float(x) for x in a.temps.split(",")]

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    bufs = ck["buffers"]
    n_cells = int(ck["config"]["n_cells"])
    cent_a = bufs["centroids"].to(device)
    cla, clo = bufs["cell_lat"].to(device), bufs["cell_lon"].to(device)
    has_b = "centroids_b" in bufs
    if has_b:
        clb, clob = bufs["cell_lat_b"].to(device), bufs["cell_lon_b"].to(device)
        cent_b = bufs["centroids_b"].to(device)

    # ---- truths ----------------------------------------------------------------
    _, vdf = load_index("val")
    tlat, tlon = vdf["latitude"].to_numpy(), vdf["longitude"].to_numpy()
    _, tdf = load_index("train")
    trlat = torch.tensor(tdf["latitude"].to_numpy(), dtype=torch.float32, device=device)
    trlon = torch.tensor(tdf["longitude"].to_numpy(), dtype=torch.float32, device=device)

    # ---- train A-cell assignment (chunked) ---------------------------------------
    pts = latlon_to_unit(trlat, trlon)
    cell_a = torch.empty(pts.size(0), dtype=torch.long, device=device)
    for i in range(0, pts.size(0), 100_000):
        cell_a[i:i + 100_000] = (pts[i:i + 100_000] @ cent_a.T).argmax(1)

    # ---- val arrays ---------------------------------------------------------------
    lga = torch.tensor(cat_shards(a.index, "logits_a", "val"), dtype=torch.float32, device=device)
    lgb = torch.tensor(cat_shards(a.index, "logits_b", "val"), dtype=torch.float32, device=device)
    n = lga.size(0)
    v_emb = {t: torch.tensor(cat_shards(a.index, f"emb_{t}", "val")).to(device)
             for t in ("bb", "trunk")}

    # ---- classifier baseline: exact training readout (A+B union, mode-seeking) ----
    probs = F.softmax(lga / PRED_TEMP, dim=-1)
    if has_b:
        probs_b = F.softmax(lgb / PRED_TEMP, dim=-1)
        probs_u = torch.cat([probs * 0.5, probs_b * 0.5], dim=-1)
        all_lat, all_lon = torch.cat([cla, clb]), torch.cat([clo, clob])
        all_cent = torch.cat([cent_a, cent_b])
    else:
        probs_u, all_lat, all_lon, all_cent = probs, cla, clo, cent_a
    w, idx = probs_u.topk(min(2 * PRED_TOPK, probs_u.size(-1)), dim=-1)
    d1 = haversine_km(all_lat[idx[:, :1]], all_lon[idx[:, :1]], all_lat[idx], all_lon[idx])
    w = w * (d1 <= PRED_RADIUS_KM)
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
    v = torch.einsum("nk,nkd->nd", w, all_cent[idx])
    blat, blon = unit_to_latlon(v)
    base = np.stack([blat.cpu().numpy(), blon.cpu().numpy()], axis=1)
    print("BASELINE (classifier readout, full val):")
    print("  " + fmt(metrics(base, tlat, tlon)), flush=True)
    fallbacks = [(float(base[i, 0]), float(base[i, 1])) for i in range(n)]

    # ---- phase 1: cache top-32 neighbors per (mass, tap) ---------------------------
    K_CACHE = 32
    p_hon = F.softmax(lga, dim=-1)
    ws, order = p_hon.sort(dim=-1, descending=True)
    cums = ws.cumsum(-1)
    caches = {}
    for tap in TAPS:
        ix = RetrievalIndex(cat_shards(a.index, f"emb_{tap}", "train"),
                            tdf["latitude"].to_numpy(), tdf["longitude"].to_numpy(),
                            cell_a.cpu().numpy(), n_cells, device=device)
        q_all = F.normalize(v_emb[tap].float(), dim=-1).half()
        for mass in MASSES:
            m_counts = (cums < mass).sum(-1) + 1
            sims_c = torch.full((n, K_CACHE), -2.0, device=device)
            units_c = torch.zeros((n, K_CACHE, 3), device=device)
            pool_c = torch.zeros(n, dtype=torch.long, device=device)
            for i in range(n):
                cells = order[i, :min(int(m_counts[i]), 400)]
                cand = ix.candidates(cells)
                pool_c[i] = cand.numel()
                if cand.numel() == 0:
                    continue
                s = (ix.emb[cand] @ q_all[i]).float()
                k = min(K_CACHE, s.numel())
                sw, st = s.topk(k)
                sims_c[i, :k] = sw
                units_c[i, :k] = ix.unit[cand[st]]
            caches[(tap, mass)] = (sims_c, units_c, pool_c)
            print(f"cached tap={tap} mass={mass}: median pool "
                  f"{int(pool_c.median())} imgs", flush=True)
        del ix
        torch.cuda.empty_cache()

    # ---- phase 2: vectorized grid ---------------------------------------------------
    base_t = torch.tensor(base, dtype=torch.float32, device=device)
    base_unit = latlon_to_unit(base_t[:, 0], base_t[:, 1])
    results = []
    for (tap, mass), (sims_c, units_c, _) in caches.items():
        for k in KS:
            sk, uk = sims_c[:, :k], units_c[:, :k]
            for temp in TEMPS:
                ww = F.softmax(sk / temp, dim=-1) * (sk > -1.5)
                ww = ww / ww.sum(-1, keepdim=True).clamp(min=1e-9)
                vv = torch.einsum("nk,nkd->nd", ww, uk)
                rlat, rlon = unit_to_latlon(vv)
                for floor in (0.0, 0.3, 0.45, 0.6):
                    use = (sims_c[:, 0] >= floor).float().unsqueeze(-1)
                    fv = F.normalize(vv, dim=-1) * use + base_unit * (1 - use)
                    fla, flo = unit_to_latlon(fv)
                    pred = np.stack([fla.cpu().numpy(), flo.cpu().numpy()], 1)
                    m = metrics(pred, tlat, tlon)
                    results.append(((tap, mass, k, temp, floor), m))
    results.sort(key=lambda r: r[1]["median"])
    print("\nTOP 12 by median:")
    for cfg, m in results[:12]:
        print(f"  tap={cfg[0]:5s} mass={cfg[1]} k={cfg[2]:2d} temp={cfg[3]} floor={cfg[4]}  "
              + fmt(m), flush=True)
    best25 = sorted(results, key=lambda r: -r[1]["acc25"])[:6]
    print("\nTOP 6 by acc@25km:")
    for cfg, m in best25:
        print(f"  tap={cfg[0]:5s} mass={cfg[1]} k={cfg[2]:2d} temp={cfg[3]} floor={cfg[4]}  "
              + fmt(m), flush=True)
    best1 = sorted(results, key=lambda r: -r[1]["acc1"])[:3]
    print("\nTOP 3 by acc@1km:")
    for cfg, m in best1:
        print(f"  tap={cfg[0]:5s} mass={cfg[1]} k={cfg[2]:2d} temp={cfg[3]} floor={cfg[4]}  "
              + fmt(m), flush=True)


if __name__ == "__main__":
    main()
