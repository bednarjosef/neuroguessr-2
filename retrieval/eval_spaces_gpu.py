#!/usr/bin/env python3
"""GPU grid over embedding spaces x readout tricks — the full C1.5 evaluation in minutes.

Spaces: raw bb, C1 place head, geo heads (25/10km), plus 50/50 similarity blends.
Tricks: k=1 snap | location-max-pool | alpha-query-expansion. Gate: mass 0.95, prior 0.05.

Box usage:
  python retrieval/eval_spaces_gpu.py --index retrieval_index
Expects in --index: emb_bb_train_r*.npy, emb_bb_val_r*.npy, logits_a_val_r*.npy,
train_latlon.npz, centroids.npz, val_meta.npz, place_head.pt, geo_head_25.pt, geo_head_10.pt
"""
import argparse
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


def hav_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, np.float64), np.asarray(lon1, np.float64),
                                              np.asarray(lat2, np.float64), np.asarray(lon2, np.float64)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cat(d, prefix):
    fs = sorted([f for f in os.listdir(d) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="retrieval_index")
    a = ap.parse_args()
    d = a.index

    print("loading…", flush=True)
    z = np.load(os.path.join(d, "train_latlon.npz"))
    trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    N = len(trlat)
    vz = np.load(os.path.join(d, "val_meta.npz"))
    tlat, tlon = vz["lat"], vz["lon"]
    n = len(tlat)
    lga = torch.tensor(cat(d, "logits_a_val_r"), dtype=torch.float32, device=device)

    raw_t = torch.from_numpy(cat(d, "emb_bb_train_r")).to(device)
    raw_t = F.normalize(raw_t.float(), dim=-1).half()
    raw_v = F.normalize(torch.from_numpy(cat(d, "emb_bb_val_r")).float().to(device), dim=-1)

    spaces = {"raw": (raw_t, raw_v)}
    for name, f in (("c1", "place_head.pt"), ("geo25", "geo_head_25.pt"), ("geo10", "geo_head_10.pt")):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        head = PlaceHead(ck["dim"], ck["hidden"])
        head.load_state_dict(ck["state_dict"])
        head.to(device).eval()
        with torch.no_grad():
            pt = torch.empty_like(raw_t)
            for i in range(0, N, 200_000):
                pt[i:i + 200_000] = head(raw_t[i:i + 200_000].float()).half()
            pv = head(raw_v)
        spaces[name] = (pt, pv)
        print(f"projected {name}", flush=True)
    print("spaces:", list(spaces), f"VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

    # cells + location ids
    cent = torch.tensor(np.load(os.path.join(d, "centroids.npz"))["cent"], device=device)
    C = cent.shape[0]
    lar, lor = np.radians(trlat), np.radians(trlon)
    tru = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                                 np.sin(lar)], -1), dtype=torch.float32, device=device)
    cell_a = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, 200_000):
        cell_a[i:i + 200_000] = (tru[i:i + 200_000] @ cent.T).argmax(1)
    order = torch.argsort(cell_a, stable=True)
    cs = cell_a[order]
    starts = torch.searchsorted(cs, torch.arange(C, device=device))
    ends = torch.searchsorted(cs, torch.arange(C, device=device), right=True)
    _, loc = np.unique(np.stack([trlat, trlon], 1), axis=0, return_inverse=True)
    loc_t = torch.tensor(loc, device=device)
    trlat_t = torch.tensor(trlat, dtype=torch.float64, device=device)
    trlon_t = torch.tensor(trlon, dtype=torch.float64, device=device)

    p = F.softmax(lga, dim=-1)
    logp = p.clamp(1e-12).log()
    rank_order = torch.argsort(p, dim=1, descending=True)
    csum = torch.gather(p, 1, rank_order).cumsum(1)
    m_of = (csum < 0.95).sum(1) + 1
    LAM = 0.05

    blends = [("raw", None), ("c1", None), ("geo25", None), ("geo10", None),
              ("geo25|raw", ("geo25", "raw")), ("geo25|c1", ("geo25", "c1")),
              ("geo10|raw", ("geo10", "raw")), ("geo10|c1", ("geo10", "c1")),
              ("c1|raw", ("c1", "raw")), ("geo25|geo10", ("geo25", "geo10"))]
    blends = [(t, pr) for t, pr in blends
              if (pr is None and t in spaces) or (pr and pr[0] in spaces and pr[1] in spaces)]
    configs = [(f"{t}{'+locmax' if lm else ''}{'+qe' if qe else ''}", t, pr, lm, qe)
               for t, pr in blends for lm in (False, True) for qe in (False, True)]
    print(f"{len(configs)} configs", flush=True)

    preds = {c[0]: np.zeros((n, 2)) for c in configs}
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            m = min(int(m_of[i]), 400)
            cells = rank_order[i, :m]
            segs = [order[starts[c]:ends[c]] for c in cells]
            ids = torch.cat(segs)
            prior = LAM * logp[i, cell_a[ids]].double()
            sims_by = {nm: (te[ids].float() @ ve[i]) for nm, (te, ve) in spaces.items()}
            l_ids = loc_t[ids]
            for tag, base, pr, lm, qe in configs:
                s = sims_by[base] if pr is None else 0.5 * sims_by[pr[0]] + 0.5 * sims_by[pr[1]]
                if qe:
                    sp = base if pr is None else pr[0]
                    te, ve = spaces[sp]
                    top5 = (s.double() + prior).topk(min(5, s.numel())).indices
                    qv = F.normalize(ve[i] + te[ids[top5]].float().mean(0), dim=-1)
                    s = 0.5 * s + 0.5 * (te[ids].float() @ qv)
                score = s.double() + prior
                if lm:
                    k = min(64, score.numel())
                    tv, ti = score.topk(k)
                    lsub = l_ids[ti]
                    first = torch.ones(k, dtype=torch.bool, device=device)
                    seen = {}
                    lcpu = lsub.cpu().numpy()
                    for kk in range(k):
                        if lcpu[kk] in seen:
                            first[kk] = False
                        else:
                            seen[lcpu[kk]] = True
                    j = ids[ti[first.nonzero()[0]]]  # best image of best new location
                else:
                    j = ids[score.argmax()]
                jj = int(j)
                preds[tag][i] = (float(trlat_t[jj]), float(trlon_t[jj]))
            if i % 500 == 0:
                print(f"  {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== RESULTS (full val, gate m0.95 lam0.05) ===", flush=True)
    rows = []
    for tag, *_ in configs:
        dd = hav_np(preds[tag][:, 0], preds[tag][:, 1], tlat, tlon)
        rows.append((tag, np.median(dd), (dd <= 1).mean(), (dd <= 25).mean(),
                     (dd <= 200).mean(), dd.mean()))
    for r in sorted(rows, key=lambda r: -r[3]):
        print(f"{r[0]:24s} median {r[1]:7.2f}  mean {r[5]:6.0f}  @1 {100*r[2]:5.2f}%  "
              f"@25 {100*r[3]:5.2f}%  @200 {100*r[4]:5.2f}%", flush=True)
    print("GRID DONE", flush=True)


if __name__ == "__main__":
    main()
