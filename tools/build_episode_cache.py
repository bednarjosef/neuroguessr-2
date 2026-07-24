#!/usr/bin/env python3
"""C5 bed: per-TRAIN-image gated retrieval episodes — the artifact that unlocks training
the full-scale learned reranker (and calibration, hub stats, second-hop) on CPU.

For each train image: gate cells from its own top-K gate (mass 0.95), gather candidate
images in those cells, score by cls cosine, store top-50 (ids int32 + sims fp16),
excluding the query's own location (its images would be trivial positives).

Run ON THE BOX after the train embed:
  torchrun --standalone --nproc_per_node=4 tools/build_episode_cache.py --index retrieval_index --tag c5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)
K_OUT = 50
MASS = 0.95


def cat(d, p):
    fs = sorted([f for f in os.listdir(d) if f.startswith(p)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    assert fs, f"missing {p}* in {d}"
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="retrieval_index")
    ap.add_argument("--tag", default="c5")
    ap.add_argument("--ckpt", default="run_c5/ckpt_best.pt")
    a = ap.parse_args()
    t0 = time.time()
    d = a.index

    z = np.load(os.path.join(d, "train_latlon.npz"))
    trlat, trlon = z["lat"], z["lon"]
    N = len(trlat)
    _, loc_of = np.unique(np.round(np.stack([trlat, trlon], 1), 6), axis=0, return_inverse=True)
    loc_t = torch.tensor(loc_of, device=device)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cent = torch.nn.functional.normalize(ck["buffers"]["centroids"].float(), dim=-1).to(device)
    del ck
    lar, lor = np.radians(trlat), np.radians(trlon)
    pts = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                                 np.sin(lar)], -1), dtype=torch.float32, device=device)
    cell = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, 200_000):
        cell[i:i + 200_000] = (pts[i:i + 200_000] @ cent.T).argmax(1)
    del pts
    order = torch.argsort(cell, stable=True)
    cs = cell[order]
    C = cent.shape[0]
    ar = torch.arange(C, device=device)
    starts = torch.searchsorted(cs, ar)
    ends = torch.searchsorted(cs, ar, right=True)

    emb = torch.from_numpy(cat(d, f"c2_cls_train_{a.tag}_r")).to(device)
    for i in range(0, N, 200_000):
        emb[i:i + 200_000] = torch.nn.functional.normalize(
            emb[i:i + 200_000].float(), dim=-1).half()
    gids = torch.from_numpy(cat(d, f"c2_gateids_train_{a.tag}_r").astype(np.int64)).to(device)
    gps = torch.from_numpy(cat(d, f"c2_gatep_train_{a.tag}_r").astype(np.float32)).to(device)
    m_of = (gps.cumsum(1) < MASS).sum(1) + 1

    chunk = -(-N // WORLD)
    lo, hi = RANK * chunk, min(N, (RANK + 1) * chunk)
    mq = hi - lo
    ids_out = np.lib.format.open_memmap(
        os.path.join(d, f"c5_ep_ids_r{RANK}.npy"), mode="w+", dtype=np.int32, shape=(mq, K_OUT))
    sim_out = np.lib.format.open_memmap(
        os.path.join(d, f"c5_ep_sims_r{RANK}.npy"), mode="w+", dtype=np.float16, shape=(mq, K_OUT))
    print(f"[r{RANK}] episodes for rows {lo}:{hi}", flush=True)

    with torch.no_grad():
        for j, q in enumerate(range(lo, hi)):
            m = int(m_of[q])
            cells = gids[q, :m]
            ids = torch.cat([order[starts[c]:ends[c]] for c in cells])
            ids = ids[loc_t[ids] != loc_t[q]]
            if ids.numel() == 0:
                ids_out[j] = -1
                continue
            s = emb[ids].float() @ emb[q].float()
            k = min(K_OUT, s.numel())
            top = s.topk(k)
            ids_out[j, :k] = ids[top.indices].cpu().numpy().astype(np.int32)
            sim_out[j, :k] = top.values.cpu().numpy().astype(np.float16)
            if k < K_OUT:
                ids_out[j, k:] = -1
            if j % 20_000 == 0:
                r = j / max(1e-9, time.time() - t0)
                print(f"[r{RANK}] {j}/{mq} ({r:.0f} q/s, ETA {(mq-j)/max(r,1e-9)/60:.1f} min)",
                      flush=True)
    ids_out.flush(); sim_out.flush()
    print(f"[r{RANK}] EPISODE CACHE DONE {mq} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
