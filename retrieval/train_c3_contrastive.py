#!/usr/bin/env python3
"""C3: fine-tune the BACKBONE on the geo-smooth objective (not classification).

Every head so far (C1/C1.5/C2) could only re-weight information the frozen trunk kept.
C2 proved the ceiling is in the features themselves: 512px added +0.33 pts, descriptor
diversity added +2.2. So here we train the encoder itself so that cosine similarity IS
geographic proximity:
  * warm start from the run-#2 LoRA weights (keeps the geo semantics already learned)
  * positives = two images from locations within --d-pos km (or 2 views of one location)
  * in-batch negatives closer than 50 km are masked (false negatives)
  * embeddings all-gathered across ranks (autograd-aware) -> global NT-Xent batch
  * a residual projection head is trained jointly, so the descriptor head comes free
The classifier is NOT touched and NOT used here: the retrieval gate keeps using the old
checkpoint's logits, so a better encoder cannot degrade the gate.

Saves run_c3/ckpt.pt in the same format embed_c2.py/GeoModelEval expect (["trainable"] holds
the updated LoRA params) plus run_c3/c3_head.pt in PlaceHead format for eval_c2_gpu.

Launch:
  cd /root/auto && PYTHONPATH=/root/auto /venv/main/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=4 retrieval/train_c3_contrastive.py --steps 3500
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prepare import load_index, open_image
from retrieval.embed_full import GeoModelEval, IMAGENET_MEAN, IMAGENET_STD
from retrieval.train_place_head import PlaceHead

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{LOCAL_RANK}")
torch.cuda.set_device(device)
EARTH = 6371.0088


class PairDataset(Dataset):
    """One item = (anchor image, positive image) — a geo-smooth positive pair."""

    def __init__(self, img_dir, paths, order, starts, neigh, size, p_same, seed):
        self.img_dir, self.paths = img_dir, paths
        self.order, self.starts, self.neigh = order, starts, neigh
        self.size, self.p_same = size, p_same
        self.rng = np.random.default_rng(seed)
        self.locs = np.array([l for l in range(len(starts) - 1)])

    def __len__(self):
        return len(self.locs)

    def _img(self, row):
        img = open_image(self.img_dir, self.paths[row])
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size))
        return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)

    def __getitem__(self, i):
        l = self.locs[i]
        rows = self.order[self.starts[l]:self.starts[l + 1]]
        r1 = rows[self.rng.integers(len(rows))]
        cand = self.neigh[l]
        if len(cand) == 0 or self.rng.random() < self.p_same:
            others = rows[rows != r1]
            r2 = others[self.rng.integers(len(others))] if len(others) else r1
        else:
            l2 = cand[self.rng.integers(len(cand))]
            rows2 = self.order[self.starts[l2]:self.starts[l2 + 1]]
            r2 = rows2[self.rng.integers(len(rows2))]
        return self._img(r1), self._img(r2), float(self.lat[l]), float(self.lon[l])


def build_locations(lat, lon, d_pos):
    """Location ids, row buckets, and per-location neighbour lists within d_pos km."""
    ll = np.stack([lat, lon], 1)
    uniq, loc = np.unique(ll, axis=0, return_inverse=True)
    n_loc = len(uniq)
    order = np.argsort(loc, kind="stable")
    counts = np.bincount(loc, minlength=n_loc)
    starts = np.zeros(n_loc + 1, np.int64)
    starts[1:] = counts.cumsum()
    lar, lor = np.radians(uniq[:, 0]), np.radians(uniq[:, 1])
    unit = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                                  np.sin(lar)], -1), dtype=torch.float32, device=device)
    cos_pos = float(np.cos(d_pos / EARTH))
    neigh = []
    for i in range(0, n_loc, 4096):
        blk = (unit[i:i + 4096] @ unit.T) >= cos_pos
        blk[torch.arange(blk.shape[0], device=device), torch.arange(i, i + blk.shape[0],
                                                                   device=device)] = False
        idx = blk.nonzero().cpu().numpy()
        by = [[] for _ in range(blk.shape[0])]
        for r, c in idx:
            by[r].append(c)
        neigh.extend([np.array(b, np.int64) for b in by])
    return uniq, loc, order, starts, neigh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="run_full2/ckpt_best.pt")
    ap.add_argument("--index-dir", default="retrieval_index")
    ap.add_argument("--out", default="run_c3")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--pairs", type=int, default=48, help="pairs per GPU per step")
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--d-pos", type=float, default=10.0)
    ap.add_argument("--p-same", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if WORLD > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    os.makedirs(a.out, exist_ok=True)

    img_dir, df = load_index("train")
    lat = df["latitude"].to_numpy(np.float64)
    lon = df["longitude"].to_numpy(np.float64)
    t0 = time.time()
    uniq, loc, order, starts, neigh = build_locations(lat, lon, a.d_pos)
    if RANK == 0:
        nn_ = np.array([len(x) for x in neigh])
        print(f"[r0] {len(uniq)} locations, neighbours<= {a.d_pos}km: mean {nn_.mean():.1f}, "
              f"{100*(nn_>0).mean():.1f}% have any ({time.time()-t0:.0f}s)", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GeoModelEval(ck)
    model.load_from_ckpt(ck)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = []
    for n, p in model.backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            lora_params.append(p)
    core = model._core()
    try:
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:                                   # noqa: BLE001
        print(f"[r{RANK}] no gradient checkpointing ({e})", flush=True)
    head = PlaceHead(1024, 2048).to(device)
    if RANK == 0:
        print(f"[r0] trainable: {sum(p.numel() for p in lora_params)/1e6:.1f}M LoRA + "
              f"{sum(p.numel() for p in head.parameters())/1e6:.1f}M head", flush=True)

    ds = PairDataset(img_dir, df["path"].tolist(), order, starts, neigh, a.img_size,
                     a.p_same, seed=1234 + RANK)
    ds.lat, ds.lon = uniq[:, 0], uniq[:, 1]
    dl = DataLoader(ds, batch_size=a.pairs, num_workers=a.workers, pin_memory=True,
                    shuffle=True, drop_last=True, persistent_workers=True)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 *
                                                       (1 + np.cos(np.pi * min(1.0, s / a.steps)))))
    cos_neg = float(np.cos(50.0 / EARTH))
    scaler = torch.amp.GradScaler("cuda")

    def gather(t):
        if WORLD == 1:
            return t
        try:
            from torch.distributed.nn.functional import all_gather as ag
            return torch.cat(ag(t))
        except Exception:                                    # noqa: BLE001
            out = [torch.zeros_like(t) for _ in range(WORLD)]
            dist.all_gather(out, t.detach())
            out[RANK] = t
            return torch.cat(out)

    step = 0
    t0 = time.time()
    model.train()
    while step < a.steps:
        for x1, x2, la, lo in dl:
            if step >= a.steps:
                break
            x = torch.cat([x1, x2]).to(device, non_blocking=True).float().div_(255.0)
            x = (x - mean_t) / std_t
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats = model.features(x)
            z = head(feats.float())                          # (2P, D), L2-normalised
            P = x1.shape[0]
            zg = gather(z)
            lag = gather(la.to(device).float()).detach()
            log = gather(lo.to(device).float()).detach()
            G = zg.shape[0]
            # positive of global row i: the other half of its own rank's block
            base = torch.arange(G, device=device)
            blk = base // (2 * P)
            within = base % (2 * P)
            tgt = blk * (2 * P) + (within + P) % (2 * P)
            lar, lor = torch.deg2rad(lag), torch.deg2rad(log)
            u = torch.stack([lar.cos() * lor.cos(), lar.cos() * lor.sin(), lar.sin()], -1)
            close = (u @ u.T) >= cos_neg
            close[base, tgt] = False
            sims = zg @ zg.T / a.tau
            sims[base, base] = -1e4
            sims = sims.masked_fill(close, -1e4)
            loss = F.cross_entropy(sims, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if RANK == 0 and step % 25 == 0:
                acc = (sims.argmax(1) == tgt).float().mean().item()
                ips = step * 2 * P * WORLD / (time.time() - t0)
                print(f"step {step}/{a.steps} loss {loss.item():.4f} in-batch-acc {acc:.3f} "
                      f"({ips:.0f} img/s, ETA {(a.steps-step)*2*P*WORLD/ips/60:.1f} min)", flush=True)
            if RANK == 0 and step % 500 == 0:
                save(a, ck, model, head, step)
    if RANK == 0:
        save(a, ck, model, head, step)
        print("C3 TRAIN DONE", flush=True)


def save(a, ck, model, head, step):
    trainable = dict(ck["trainable"])
    for n, p in model.named_parameters():
        if n in trainable and p.requires_grad:
            trainable[n] = p.detach().float().cpu()
    out = {k: v for k, v in ck.items() if k != "trainable"}
    out["trainable"] = trainable
    out["c3_step"] = step
    torch.save(out, os.path.join(a.out, "ckpt.pt"))
    torch.save({"state_dict": {k: v.detach().float().cpu() for k, v in head.state_dict().items()},
                "dim": 1024, "hidden": 2048, "c3_step": step},
               os.path.join(a.out, "c3_head.pt"))
    print(f"[r0] saved at step {step}", flush=True)


if __name__ == "__main__":
    main()
