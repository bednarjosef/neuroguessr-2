#!/usr/bin/env python3
"""Phase C1: contrastive place-matching head on FROZEN cached embeddings (no images).

Supervision is free: the dataset has 4 views per location (identical lat/lon). NT-Xent
pulls views of the same place together and pushes other places away — reshaping the
metric from "looks alike" to "IS the same place". The head is residual with a zero-init
output layer, so training starts exactly at the identity metric (current baseline) and
can only move away where the data says so.

Run (box, 1 GPU):  python retrieval/train_place_head.py --index-dir retrieval_index
Outputs: place_head.pt (the head) + val_proj.npy (projected val embeddings).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    assert fs, f"no {prefix}* in {d}"
    return np.concatenate([np.load(os.path.join(d, f)) for f in fs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="retrieval_index")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--locs-per-batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    a = ap.parse_args()

    print("loading embeddings…", flush=True)
    emb = torch.from_numpy(cat_shards(a.index_dir, "emb_bb_train_r")).to(device)
    emb = F.normalize(emb.float(), dim=-1)          # (N,1024) fp32 on GPU (~4.9GB)
    z = np.load(os.path.join(a.index_dir, "train_latlon.npz"))
    ll = np.stack([z["lat"], z["lon"]], 1)
    N = len(ll)
    assert emb.shape[0] == N

    # location ids: rows with identical (lat,lon) are views of one place
    uniq, loc = np.unique(ll, axis=0, return_inverse=True)
    n_loc = len(uniq)
    order = np.argsort(loc, kind="stable")
    counts = np.bincount(loc, minlength=n_loc)
    starts = np.zeros(n_loc + 1, np.int64)
    starts[1:] = counts.cumsum()
    multi = np.nonzero(counts >= 2)[0]
    print(f"{N} imgs, {n_loc} locations ({len(multi)} with >=2 views; "
          f"median views {int(np.median(counts))})", flush=True)

    rng = np.random.default_rng(0)
    heldout = rng.choice(multi, size=5000, replace=False)
    held_set = np.zeros(n_loc, bool)
    held_set[heldout] = True
    train_locs = multi[~held_set[multi]]

    # heldout probe: view-retrieval recall@1 among a 20k-image pool
    probe_pairs = []
    pool = []
    for l in heldout[:2500]:
        rows = order[starts[l]:starts[l + 1]]
        probe_pairs.append((rows[0], rows[1]))
        pool.extend(rows[:2])
    pool = torch.tensor(pool, device=device)
    q_idx = torch.tensor([p[0] for p in probe_pairs], device=device)
    t_idx = torch.tensor([p[1] for p in probe_pairs], device=device)

    def probe(head):
        with torch.no_grad():
            zp = head(emb[pool])
            zq = head(emb[q_idx])
            sims = zq @ zp.T
            # mask self-matches (query is inside the pool)
            self_pos = (pool.unsqueeze(0) == q_idx.unsqueeze(1))
            sims.masked_fill_(self_pos, -2)
            hit = pool[sims.argmax(1)] == t_idx
        return float(hit.float().mean())

    head = PlaceHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_locs) // a.locs_per_batch
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    print(f"baseline heldout view-recall@1: {probe(head):.4f}", flush=True)

    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        perm = rng.permutation(train_locs)
        for s in range(steps_per_epoch):
            locs = perm[s * a.locs_per_batch:(s + 1) * a.locs_per_batch]
            i1 = np.empty(len(locs), np.int64)
            i2 = np.empty(len(locs), np.int64)
            for j, l in enumerate(locs):
                rows = order[starts[l]:starts[l + 1]]
                pick = rng.choice(len(rows), 2, replace=False)
                i1[j], i2[j] = rows[pick[0]], rows[pick[1]]
            x = emb[np.concatenate([i1, i2])]
            zb = head(x)
            B = len(locs)
            sims = zb @ zb.T / a.tau
            sims.fill_diagonal_(-1e4)
            tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device)
            loss = F.cross_entropy(sims, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep+1} step {step}/{total} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"epoch {ep+1}: heldout view-recall@1 = {probe(head):.4f}", flush=True)

    torch.save({"state_dict": head.state_dict(), "dim": 1024, "hidden": 2048},
               os.path.join(a.index_dir, "place_head.pt"))
    v = torch.from_numpy(cat_shards(a.index_dir, "emb_bb_val_r")).to(device)
    with torch.no_grad():
        vp = head(F.normalize(v.float(), dim=-1)).cpu().numpy().astype(np.float16)
    np.save(os.path.join(a.index_dir, "val_proj.npy"), vp)
    print("saved place_head.pt + val_proj.npy", flush=True)
    print("C1 TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
