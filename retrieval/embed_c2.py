#!/usr/bin/env python3
"""C2 embedding sweep — richer descriptors at higher resolution. Runs ON THE BOX.

vs embed_full.py (384px, CLS only) this writes, per rank:
  c2_cls_{split}_{tag}_r{R}.npy   (m, 1024)      fp16 — backbone CLS after final norm
  c2_mean_{split}_{tag}_r{R}.npy  (m, 1024)      fp16 — mean over patch tokens (distributed cues)
  c2_reg_{split}_{tag}_r{R}.npy   (m, 16, RDIM)  fp16 — 4x4 regional descriptors, PCA-whitened
  c2_logits_{split}_{tag}_r{R}.npy(m, C)         fp16 — classifier logits (gate), val only
  + with --tta (val only) the same cls/mean for a horizontal flip and a 0.8 center zoom.
Row order = parquet row order; ranks own contiguous chunks -> concatenation restores order.

The regional PCA basis is fit ONCE on the train sweep (covariance all-reduced across ranks)
and cached as c2_regproj.npz — later sweeps (val) load it, so all descriptors share a basis.

Launch (4-way shard):
  cd /root/auto && PYTHONPATH=/root/auto /venv/main/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=4 retrieval/embed_c2.py --split train --img-size 512 \
    --ckpt run_full2/ckpt_best.pt --out retrieval_index
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prepare import load_index, open_image
from retrieval.embed_full import GeoModelEval, IMAGENET_MEAN, IMAGENET_STD, NUM_PREFIX_TOKENS

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

REG_GRID = 4          # 4x4 = 16 regional descriptors per image
PCA_IMGS = 1500       # images per rank used to fit the regional PCA basis


class SweepDataset(Dataset):
    """Returns (base, flip, zoom) uint8 tensors; flip/zoom only materialised when tta=True."""

    def __init__(self, split, lo, hi, size, tta=False):
        self.img_dir, df = load_index(split)
        self.paths = df["path"].tolist()[lo:hi]
        self.size = size
        self.tta = tta

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = open_image(self.img_dir, self.paths[i])
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        a = np.asarray(img, dtype=np.uint8)
        out = [torch.from_numpy(a).permute(2, 0, 1)]
        if self.tta:
            out.append(torch.from_numpy(np.ascontiguousarray(a[:, ::-1])).permute(2, 0, 1))
            w = img.size[0]
            m = int(round(w * 0.1))
            z = img.crop((m, m, w - m, w - m)).resize((self.size, self.size), Image.BICUBIC)
            out.append(torch.from_numpy(np.asarray(z, dtype=np.uint8)).permute(2, 0, 1))
        return tuple(out)


def full_tokens(model, x):
    """Normed last hidden state (B, T, D) — CLS at 0, NUM_PREFIX_TOKENS-1 registers, then patches."""
    core = model._core()
    hs = core.embeddings(x.to(core.embeddings.patch_embeddings.weight.dtype))
    cos, sin = core.rope_embeddings(x)
    enc = getattr(core, "model", None) or core.layer
    out = enc(hs, (cos, sin))
    return core.norm(out.last_hidden_state)


def split_tokens(tok):
    """-> cls (B,D), mean-patch (B,D), regions (B,16,D) via 4x4 block mean pooling."""
    cls = tok[:, 0]
    pt = tok[:, NUM_PREFIX_TOKENS:]
    B, P, D = pt.shape
    g = int(round(P ** 0.5))
    assert g * g == P, f"patch count {P} is not square"
    grid = pt.view(B, g, g, D)
    b = g // REG_GRID
    reg = grid[:, :b * REG_GRID, :b * REG_GRID].reshape(B, REG_GRID, b, REG_GRID, b, D).mean((2, 4))
    return cls, pt.mean(1), reg.reshape(B, REG_GRID * REG_GRID, D)


def fit_region_pca(model, dl, norm, rdim, n_imgs):
    """Mean + top-rdim whitened PCA basis of regional descriptors, all-reduced across ranks."""
    D = None
    s = c = None
    seen = 0
    with torch.no_grad():
        for batch in dl:
            x = norm(batch[0])
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, reg = split_tokens(full_tokens(model, x))
            v = reg.float().reshape(-1, reg.shape[-1])
            if s is None:
                D = v.shape[1]
                s = torch.zeros(D, dtype=torch.float64, device=device)
                c = torch.zeros(D, D, dtype=torch.float64, device=device)
            s += v.double().sum(0)
            c += v.double().T @ v.double()
            seen += reg.shape[0]
            if seen >= n_imgs:
                break
    n = torch.tensor([float(seen * REG_GRID * REG_GRID)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        for t in (s, c, n):
            dist.all_reduce(t)
    mu = s / n
    cov = c / n - torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(cov.float())
    idx = torch.argsort(ev, descending=True)[:rdim]
    proj = V[:, idx] / (ev[idx].clamp(min=1e-6).sqrt())     # PCA-whitened
    return mu.float().cpu().numpy(), proj.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--ckpt", default="run_full2/ckpt_best.pt")
    ap.add_argument("--out", default="retrieval_index")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rdim", type=int, default=128)
    ap.add_argument("--tta", action="store_true", help="also write hflip + 0.8-zoom descriptors")
    ap.add_argument("--regions", action="store_true", default=True)
    ap.add_argument("--no-regions", dest="regions", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap rows per rank")
    a = ap.parse_args()
    tag = a.tag or f"s{a.img_size}"
    os.makedirs(a.out, exist_ok=True)
    if WORLD > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    _, df = load_index(a.split)
    n = len(df)
    chunk = -(-n // WORLD)
    lo, hi = RANK * chunk, min(n, (RANK + 1) * chunk)
    if a.limit:
        hi = min(hi, lo + a.limit)
    print(f"[r{RANK}] rows {lo}:{hi} of {n} @ {a.img_size}px tag={tag}", flush=True)

    if RANK == 0:                      # ground truth for THIS split, straight from parquet order
        fn = "train_latlon.npz" if a.split == "train" else "val_meta.npz"
        pth = os.path.join(a.out, fn)
        if not os.path.exists(pth):
            np.savez(pth, lat=df["latitude"].to_numpy(np.float64),
                     lon=df["longitude"].to_numpy(np.float64))
            print(f"[r0] wrote {fn} ({len(df)} rows)", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GeoModelEval(ck)
    model.load_from_ckpt(ck)
    model.to(device).eval()
    n_cells = int(ck["config"]["n_cells"])
    if RANK == 0:
        print(f"[r0] model ready: {n_cells} cells, step {ck.get('step')}, "
              f"best {ck.get('best_median', float('nan')):.2f} km", flush=True)

    ds = SweepDataset(a.split, lo, hi, a.img_size, tta=a.tta)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, shuffle=False)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def norm(xb):
        return (xb.to(device, non_blocking=True).float().div_(255.0) - mean_t) / std_t

    pca_path = os.path.join(a.out, "c2_regproj.npz")
    reg_mu = reg_proj = None
    if a.regions:
        if os.path.exists(pca_path):
            z = np.load(pca_path)
            reg_mu, reg_proj = z["mu"], z["proj"]
            print(f"[r{RANK}] loaded regional PCA basis {reg_proj.shape}", flush=True)
        else:
            t0 = time.time()
            reg_mu, reg_proj = fit_region_pca(model, dl, norm, a.rdim, PCA_IMGS)
            if RANK == 0:
                np.savez(pca_path, mu=reg_mu, proj=reg_proj)
            print(f"[r{RANK}] fit regional PCA {reg_proj.shape} in {time.time()-t0:.0f}s", flush=True)
        reg_mu_t = torch.tensor(reg_mu, device=device)
        reg_proj_t = torch.tensor(reg_proj, device=device)

    m = len(ds)
    def mm(kind, shape):
        return np.lib.format.open_memmap(
            os.path.join(a.out, f"c2_{kind}_{a.split}_{tag}_r{RANK}.npy"),
            mode="w+", dtype=np.float16, shape=shape)

    outs = {"cls": mm("cls", (m, 1024)), "mean": mm("mean", (m, 1024))}
    if a.regions:
        outs["reg"] = mm("reg", (m, REG_GRID * REG_GRID, a.rdim))
    if a.split == "val":
        outs["logits"] = mm("logits", (m, n_cells))
    if a.tta:
        for k in ("clsflip", "meanflip", "clszoom", "meanzoom"):
            outs[k] = mm(k, (m, 1024))

    t0 = time.time()
    off = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            x = norm(batch[0])
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                tok = full_tokens(model, x)
            cls, mp, reg = split_tokens(tok.float())
            k = cls.size(0)
            outs["cls"][off:off + k] = cls.cpu().numpy().astype(np.float16)
            outs["mean"][off:off + k] = mp.cpu().numpy().astype(np.float16)
            if a.regions:
                r = F.normalize((reg - reg_mu_t) @ reg_proj_t, dim=-1)
                outs["reg"][off:off + k] = r.cpu().numpy().astype(np.float16)
            if a.split == "val":
                _, comb, _ = model.heads(cls)
                outs["logits"][off:off + k] = comb.float().cpu().numpy().astype(np.float16)
            if a.tta:
                for src, (ck_, mk_) in ((1, ("clsflip", "meanflip")), (2, ("clszoom", "meanzoom"))):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        t2 = full_tokens(model, norm(batch[src]))
                    c2, m2, _ = split_tokens(t2.float())
                    outs[ck_][off:off + k] = c2.cpu().numpy().astype(np.float16)
                    outs[mk_][off:off + k] = m2.cpu().numpy().astype(np.float16)
            off += k
            if bi % 50 == 0:
                r = off / max(1e-9, time.time() - t0)
                print(f"[r{RANK}] {off}/{m} ({r:.0f} img/s, ETA {(m-off)/max(r,1e-9)/60:.1f} min)",
                      flush=True)
    for v in outs.values():
        v.flush()
    print(f"[r{RANK}] DONE {m} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
