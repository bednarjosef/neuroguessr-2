#!/usr/bin/env python3
"""C4: joint classifier + retrieval fine-tune with a GRADED-distance contrastive loss.

Fixes everything C1-C3 measured (see docs/C4_PLAN.md):

  * GRADED targets instead of binary positives/negatives. C3 masked 10-50 km and pushed
    50-500 km apart at full strength, which — with region-restricted batches — was nearly the
    only negative it ever saw. @1 km improved, the 25-200 km band decayed. Here the target
    similarity decays smoothly, target_ij ∝ exp(-d_ij / TAU_GEO), so ordering is preserved at
    every scale. Same trick the classifier already uses on cell targets.
  * JOINT with the classifier: the retrieval gate needs logits from the SAME encoder, and C3
    proved a pure contrastive fine-tune degrades them (44.20% -> 42.93%). CE also anchors the
    coarse geography the contrastive term erodes.
  * LoRA on the MLP blocks too (up_proj/down_proj) — ~201 M of ViT-L's parameters that no run
    has ever adapted. Attention LoRA keeps rank 16 so C3's weights warm-start exactly; the MLP
    adapters are new and zero-init, so training starts where C3 finished.
  * MIXED batches: half from one ~500 km region (hard negatives), half global (long-range
    negatives C3 never saw, and i.i.d.-ish labels for CE).
  * INSTRUMENTED: held-out retrieval probe every --probe-every steps -> W&B, plus a kill
    criterion, because C3 ran blind and its loss forecast nothing.

Launch (4 GPUs):
  torchrun --standalone --nproc_per_node=4 retrieval/train_c4.py --steps 5000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prepare import load_index, open_image
from retrieval.embed_full import GeoModelEval, IMAGENET_MEAN, IMAGENET_STD
from retrieval.train_c3_contrastive import PairDataset, build_locations, build_region_index
from retrieval.train_place_head import PlaceHead

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{LOCAL_RANK}")
torch.cuda.set_device(device)
EARTH = 6371.0088
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


class MixedBatchSampler(torch.utils.data.Sampler):
    """Half the anchors from one ~500 km region (hard negatives), half globally random."""

    def __init__(self, cell_of_loc, near_cells, n_batches, batch, region_frac, seed):
        self.near = near_cells
        self.by_cell = {}
        for l, c in enumerate(cell_of_loc):
            self.by_cell.setdefault(int(c), []).append(l)
        self.by_cell = {c: np.array(v, np.int64) for c, v in self.by_cell.items()}
        self.cells = np.array(sorted(self.by_cell))
        self.n_batches, self.batch = n_batches, batch
        self.n_reg = int(round(batch * region_frac))
        self.rng = np.random.default_rng(seed)
        self.n_loc = len(cell_of_loc)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            c = int(self.rng.choice(self.cells))
            pool = np.concatenate([self.by_cell[int(x)] for x in self.near[c]
                                   if int(x) in self.by_cell])
            k = min(self.n_reg, len(pool))
            reg = self.rng.choice(pool, size=k, replace=False) if k else np.array([], np.int64)
            glob = self.rng.integers(0, self.n_loc, size=self.batch - len(reg))
            yield list(np.concatenate([reg, glob]))


def haversine_t(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (torch.deg2rad(x) for x in (lat1, lon1, lat2, lon2))
    a = (torch.sin((lat2 - lat1) / 2) ** 2 +
         torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH * torch.asin(a.clamp(0, 1).sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="run_c3/ckpt.pt", help="warm start (C3 encoder)")
    ap.add_argument("--out", default="run_c4")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07, help="softmax temperature on similarities")
    ap.add_argument("--tau-geo", type=float, default=100.0, help="km scale of the graded target")
    ap.add_argument("--tau-cell", type=float, default=75.0, help="km scale of the cell targets")
    ap.add_argument("--lam-c", type=float, default=0.7, help="weight of the contrastive term")
    ap.add_argument("--d-pos", type=float, default=10.0)
    ap.add_argument("--p-same", type=float, default=0.25)
    ap.add_argument("--region-frac", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--kill-step", type=int, default=1500)
    ap.add_argument("--wandb", default="neuroguessr-2-research")
    a = ap.parse_args()
    if WORLD > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    os.makedirs(a.out, exist_ok=True)

    img_dir, df = load_index("train")
    lat = df["latitude"].to_numpy(np.float64)
    lon = df["longitude"].to_numpy(np.float64)
    paths = df["path"].tolist()
    uniq, loc, order, starts, neigh = build_locations(lat, lon, a.d_pos)
    if RANK == 0:
        print(f"[r0] {len(uniq)} locations", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck_cent = ck["buffers"]["centroids"].float()
    model = GeoModelEval(ck, lora_targets=LORA_TARGETS, lora_r=ck["config"]["lora_r"])
    model.load_from_ckpt(ck, strict=False)
    model.to(device)
    core = model._core()
    try:
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:                                          # noqa: BLE001
        print(f"[r{RANK}] no gradient checkpointing ({e})", flush=True)
    head = PlaceHead(1024, 2048).to(device)

    lora_p, cls_p = [], []
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            lora_p.append(p)
        elif n.startswith(("trunk", "fine_head", "coarse_heads", "country_head")):
            p.requires_grad_(True)
            cls_p.append(p)
        else:
            p.requires_grad_(False)
    if RANK == 0:
        print(f"[r0] trainable: {sum(p.numel() for p in lora_p)/1e6:.1f}M LoRA "
              f"(attn+MLP) + {sum(p.numel() for p in cls_p)/1e6:.1f}M classifier + "
              f"{sum(p.numel() for p in head.parameters())/1e6:.1f}M place head", flush=True)

    cell_lat = model.cell_lat.float()
    cell_lon = model.cell_lon.float()
    cell_of_loc, near = build_region_index(uniq, ck_cent.to(device), 500.0)
    ds = PairDataset(img_dir, paths, order, starts, neigh, a.img_size, a.p_same, seed=7 + RANK)
    ds.lat, ds.lon = uniq[:, 0], uniq[:, 1]
    bs = MixedBatchSampler(cell_of_loc, near, 10 ** 6, a.pairs, a.region_frac, seed=11 + RANK)
    dl = DataLoader(ds, batch_sampler=bs, num_workers=a.workers, pin_memory=True,
                    persistent_workers=True)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # ---- held-out retrieval probe (the metric the final grid ranks by)
    rng = np.random.default_rng(0)
    held = rng.choice(len(uniq), size=1800, replace=False)
    q_loc, p_loc = held[:600], held[600:]
    q_rows = np.array([order[starts[l]] for l in q_loc])
    p_rows = np.array([r for l in p_loc for r in order[starts[l]:starts[l] + 1]])
    qu = torch.tensor(uniq[q_loc], dtype=torch.float64, device=device)
    pu = torch.tensor(uniq[p_loc], dtype=torch.float64, device=device)
    probe_ok = (haversine_t(qu[:, :1].float(), qu[:, 1:].float(),
                            pu[:, 0].float().unsqueeze(0), pu[:, 1].float().unsqueeze(0))
                <= a.d_pos)

    def embed_rows(rows, bs_=64):
        outs = []
        with torch.no_grad():
            for i in range(0, len(rows), bs_):
                ims = []
                for r in rows[i:i + bs_]:
                    im = open_image(img_dir, paths[r])
                    if im.size != (a.img_size, a.img_size):
                        im = im.resize((a.img_size, a.img_size))
                    ims.append(torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1))
                x = torch.stack(ims).to(device).float().div_(255.0)
                x = (x - mean_t) / std_t
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    f = model.features(x)
                outs.append(head(f.float()))
        return torch.cat(outs)

    def probe():
        model.eval()
        head.eval()
        zq, zp = embed_rows(q_rows), embed_rows(p_rows)
        best = (zq @ zp.T).argmax(1)
        model.train()
        head.train()
        return float(probe_ok[torch.arange(len(q_rows), device=device), best].float().mean())

    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr},
                             {"params": cls_p, "lr": a.lr * 2},
                             {"params": head.parameters(), "lr": a.head_lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) *
        (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))))

    run = None
    if RANK == 0 and a.wandb:
        try:
            import wandb
            run = wandb.init(project=a.wandb, name="c4-joint-graded", config=vars(a))
        except Exception as e:                                       # noqa: BLE001
            print(f"[r0] wandb off ({e})", flush=True)

    def gather(t):
        if WORLD == 1:
            return t
        try:
            from torch.distributed.nn.functional import all_gather as ag
            return torch.cat(ag(t))
        except Exception:                                            # noqa: BLE001
            out = [torch.zeros_like(t) for _ in range(WORLD)]
            dist.all_gather(out, t.detach())
            out[RANK] = t
            return torch.cat(out)

    base_probe = probe() if RANK == 0 else 0.0
    if RANK == 0:
        print(f"[r0] probe @ step 0 (warm start): {base_probe:.4f}", flush=True)
    model.train()
    head.train()
    step = 0
    t0 = time.time()
    while step < a.steps:
        for x1, x2, la, lo in dl:
            if step >= a.steps:
                break
            x = torch.cat([x1, x2]).to(device, non_blocking=True).float().div_(255.0)
            x = (x - mean_t) / std_t
            la2 = torch.cat([la, la]).to(device).float()
            lo2 = torch.cat([lo, lo]).to(device).float()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats = model.features(x)
            feats = feats.float()

            # ---- graded contrastive
            z = head(feats)
            zg, lag, log_ = gather(z), gather(la2).detach(), gather(lo2).detach()
            G = zg.shape[0]
            dkm = haversine_t(lag.unsqueeze(1), log_.unsqueeze(1),
                              lag.unsqueeze(0), log_.unsqueeze(0))
            tgt = torch.exp(-dkm / a.tau_geo)
            eye = torch.eye(G, dtype=torch.bool, device=device)
            tgt = tgt.masked_fill(eye, 0.0)
            tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-9)
            sims = (zg @ zg.T / a.tau).masked_fill(eye, -1e4)
            loss_con = -(tgt * F.log_softmax(sims, dim=-1)).sum(1).mean()

            # ---- classifier CE on distance-smoothed cell targets
            _, comb, fine_b = model.heads(feats)
            dcell = haversine_t(la2.unsqueeze(1), lo2.unsqueeze(1),
                                cell_lat.unsqueeze(0), cell_lon.unsqueeze(0))
            ct = F.softmax(-dcell / a.tau_cell, dim=-1)
            loss_ce = -(ct * F.log_softmax(comb, dim=-1)).sum(1).mean()
            if fine_b is not None:
                loss_ce = loss_ce + 0.5 * -(ct * F.log_softmax(fine_b, dim=-1)).sum(1).mean()

            loss = loss_ce + a.lam_c * loss_con
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_p + cls_p + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1

            if RANK == 0 and step % 25 == 0:
                ips = step * 2 * a.pairs * WORLD / (time.time() - t0)
                print(f"step {step}/{a.steps} loss {loss.item():.4f} "
                      f"(ce {loss_ce.item():.4f} con {loss_con.item():.4f}) "
                      f"({ips:.0f} img/s, ETA {(a.steps-step)*2*a.pairs*WORLD/ips/60:.0f} min)",
                      flush=True)
                if run:
                    run.log({"loss": loss.item(), "loss_ce": loss_ce.item(),
                             "loss_con": loss_con.item(), "img_s": ips,
                             "lr": sched.get_last_lr()[0]}, step=step)
            if RANK == 0 and step % a.probe_every == 0:
                pv = probe()
                print(f"[r0] probe @ {step}: {pv:.4f} (start {base_probe:.4f})", flush=True)
                if run:
                    run.log({"probe_near_recall@1": pv}, step=step)
                if step >= a.kill_step and pv < base_probe * 1.02:
                    print(f"[r0] KILL CRITERION: probe {pv:.4f} has not beaten the warm start "
                          f"{base_probe:.4f} by step {step} — stopping.", flush=True)
                    save(a, ck, model, head, step)
                    if run:
                        run.finish()
                    return
            if RANK == 0 and step % 1000 == 0:
                save(a, ck, model, head, step)
    if RANK == 0:
        save(a, ck, model, head, step)
        print(f"[r0] final probe: {probe():.4f}", flush=True)
        if run:
            run.finish()
        print("C4 TRAIN DONE", flush=True)


def save(a, ck, model, head, step):
    trainable = dict(ck["trainable"])
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable[n] = p.detach().float().cpu()
    out = {k: v for k, v in ck.items() if k != "trainable"}
    out["trainable"] = trainable
    # record the LoRA layout so every later loader (embed_c2, webapp) rebuilds the same model
    out["config"] = {**ck["config"], "lora_targets": LORA_TARGETS}
    out["c4_step"] = step
    torch.save(out, os.path.join(a.out, "ckpt.pt"))
    torch.save({"state_dict": {k: v.detach().float().cpu() for k, v in head.state_dict().items()},
                "dim": 1024, "hidden": 2048, "c4_step": step},
               os.path.join(a.out, "c4_head.pt"))
    print(f"[r0] saved at step {step}", flush=True)


if __name__ == "__main__":
    main()
