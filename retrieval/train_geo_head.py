#!/usr/bin/env python3
"""Phase C1.5: GEO-SMOOTH contrastive head — fixes C1's objective mismatch.

At eval time the query's own location is never in the index (val is held out); the best
reachable match is a DIFFERENT location 1-15 km away. C1's same-place objective actively
pushed those neighbors away. Here the positive for an anchor is a view from any location
within --d-pos km (occasionally its own other view), so similarity is trained to decay
with geographic distance — exactly the @25km ranking the engine is scored on. In-batch
negatives whose true distance to the anchor is < 50 km are masked out (false negatives).

Run (box, 1 GPU): python retrieval/train_geo_head.py --index-dir retrieval_index --d-pos 25
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARTH = 6371.0088


class PlaceHead(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return F.normalize(x + self.fc2(F.gelu(self.fc1(self.ln(x)))), dim=-1)


def cat_shards(d, prefix):
    fs = sorted([f for f in os.listdir(d) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="retrieval_index")
    ap.add_argument("--d-pos", type=float, default=25.0)
    ap.add_argument("--p-same", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--locs-per-batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--out", default="geo_head.pt")
    ap.add_argument("--emb-prefix", default="emb_bb_train_r",
                    help="shard prefix of the index embeddings to train on")
    a = ap.parse_args()

    print("loading…", flush=True)
    emb = torch.from_numpy(cat_shards(a.index_dir, a.emb_prefix)).to(device)
    emb = F.normalize(emb.float(), dim=-1)
    z = np.load(os.path.join(a.index_dir, "train_latlon.npz"))
    ll = np.stack([z["lat"], z["lon"]], 1)
    N = len(ll)
    uniq, loc = np.unique(ll, axis=0, return_inverse=True)
    n_loc = len(uniq)
    order = np.argsort(loc, kind="stable")
    counts = np.bincount(loc, minlength=n_loc)
    starts = np.zeros(n_loc + 1, np.int64)
    starts[1:] = counts.cumsum()

    lar = np.radians(uniq[:, 0])
    lor = np.radians(uniq[:, 1])
    loc_unit = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                                      np.sin(lar)], -1), dtype=torch.float32, device=device)
    cos_pos = float(np.cos(a.d_pos / EARTH))
    cos_neg = float(np.cos(50.0 / EARTH))       # mask in-batch negatives closer than 50 km
    print(f"{N} imgs, {n_loc} locs | d_pos={a.d_pos}km p_same={a.p_same}", flush=True)

    rng = np.random.default_rng(0)
    heldout = rng.choice(n_loc, size=5000, replace=False)
    held_mask = np.zeros(n_loc, bool)
    held_mask[heldout] = True
    train_locs = np.nonzero(~held_mask)[0]

    # probe mimicking eval: query = 1 view of a heldout loc; its OWN views are excluded
    # from the pool; hit = top pool image within 25 km.
    q_locs = heldout[:2500]
    q_rows = np.array([order[starts[l]] for l in q_locs])
    pool_rows = []
    for l in heldout[2500:]:
        pool_rows.extend(order[starts[l]:starts[l] + 2])
    pool_rows = np.array(pool_rows)
    pool_t = torch.tensor(pool_rows, device=device)
    q_t = torch.tensor(q_rows, device=device)
    pu = loc_unit[torch.tensor(loc[pool_rows], device=device)]
    qu = loc_unit[torch.tensor(loc[q_rows], device=device)]
    hit_ok = (qu @ pu.T) >= cos_pos             # (2500, pool) within d_pos

    def probe(head):
        with torch.no_grad():
            zp = head(emb[pool_t])
            zq = head(emb[q_t])
            best = (zq @ zp.T).argmax(1)
            return float(hit_ok[torch.arange(len(q_rows), device=device), best].float().mean())

    dim = emb.shape[1]
    head = PlaceHead(dim, 2 * dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(train_locs) // a.locs_per_batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * spe)
    print(f"baseline probe near-recall@1 (<= {a.d_pos}km): {probe(head):.4f}", flush=True)

    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        perm = rng.permutation(train_locs)
        for s in range(spe):
            locs = perm[s * a.locs_per_batch:(s + 1) * a.locs_per_batch]
            lt = torch.tensor(locs, device=device)
            # geographic neighbors within d_pos (GPU brute force over all locations)
            d = loc_unit[lt] @ loc_unit.T                       # (B, n_loc) cos
            near = d >= cos_pos
            near[torch.arange(len(locs), device=device), lt] = False
            i1 = np.empty(len(locs), np.int64)
            i2 = np.empty(len(locs), np.int64)
            near_cpu = near.cpu().numpy()
            for j, l in enumerate(locs):
                rows = order[starts[l]:starts[l + 1]]
                i1[j] = rows[rng.integers(len(rows))]
                cand = np.nonzero(near_cpu[j])[0]
                if len(cand) == 0 or rng.random() < a.p_same:
                    others = rows[rows != i1[j]]
                    i2[j] = others[rng.integers(len(others))] if len(others) else i1[j]
                else:
                    l2 = cand[rng.integers(len(cand))]
                    rows2 = order[starts[l2]:starts[l2 + 1]]
                    i2[j] = rows2[rng.integers(len(rows2))]
            x = emb[np.concatenate([i1, i2])]
            zb = head(x)
            B = len(locs)
            sims = zb @ zb.T / a.tau
            sims.fill_diagonal_(-1e4)
            # mask geographically-close in-batch pairs from the negatives
            au = loc_unit[torch.tensor(np.concatenate([loc[i1], loc[i2]]), device=device)]
            close = (au @ au.T) >= cos_neg
            tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device)
            close[torch.arange(2 * B, device=device), tgt] = False
            sims = sims.masked_fill(close, -1e4)
            loss = F.cross_entropy(sims, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep+1} step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
        print(f"epoch {ep+1}: probe near-recall@1 = {probe(head):.4f}", flush=True)

    torch.save({"state_dict": head.state_dict(), "dim": dim, "hidden": 2 * dim,
                "d_pos": a.d_pos, "p_same": a.p_same},
               os.path.join(a.index_dir, a.out))
    print(f"saved {a.out}", flush=True)
    print("GEO HEAD DONE", flush=True)


if __name__ == "__main__":
    main()
