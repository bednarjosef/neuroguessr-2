#!/usr/bin/env python3
"""Embedding sweep for the retrieval engine — runs ON THE BOX after a full run.

Sweeps a split through the trained checkpoint (eval mode, full patches) and writes:
  emb_bb_{split}_r{R}.npy     (n_r, 1024) fp16 — backbone CLS after final norm (pre-trunk)
  emb_trunk_{split}_r{R}.npy  (n_r, 1024) fp16 — post-trunk feature (pre-heads)
  logits_a_{split}_r{R}.npy   (n_r, C)    fp16 — combined head logits, tessellation A (val only)
  logits_b_{split}_r{R}.npy   (n_r, C)    fp16 — fine head logits, tessellation B   (val only)
Row order = parquet row order, ranks own contiguous chunks -> plain concatenation restores order.

Launch (4-way shard, no collectives):
  cd /root/auto && TORCHINDUCTOR_CACHE_DIR=/root/auto/.inductor PYTHONPATH=/root/auto \
    /venv/main/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
    retrieval/embed_full.py --split train --ckpt run_full/ckpt_best.pt --out retrieval_index
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prepare import load_index, open_image

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

IMG_SIZE = 384
NUM_PREFIX_TOKENS = 5
HEAD_HIDDEN = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GeoModelEval(nn.Module):
    """Eval-only mirror of train_full's GeoModel (incl. staggered head B)."""

    def __init__(self, ck, lora_targets=None, lora_r=None):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        cfg = ck["config"]
        bufs = ck["buffers"]
        n_cells = cfg["n_cells"]
        n_country = len(ck["countries"])
        backbone = AutoModel.from_pretrained(cfg["model"])
        lora = LoraConfig(r=lora_r or cfg["lora_r"], lora_alpha=32, lora_dropout=0.05,
                          target_modules=lora_targets or cfg.get("lora_targets") or
                          ["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
        self.backbone = get_peft_model(backbone, lora)
        hidden = backbone.config.hidden_size
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, HEAD_HIDDEN),
                                   nn.GELU(), nn.Dropout(0.0))
        self.fine_head = nn.Linear(HEAD_HIDDEN, n_cells)
        self.fine_head_b = (nn.Linear(HEAD_HIDDEN, n_cells)
                            if any(n == "centroids_b" for n in bufs) else None)
        self.coarse_heads = nn.ModuleList([nn.Linear(HEAD_HIDDEN, c) for c in cfg["hier"]])
        self.country_head = nn.Linear(HEAD_HIDDEN, n_country)
        for name, b in bufs.items():
            if "." not in name:
                self.register_buffer(name, b.clone())

    def load_from_ckpt(self, ck, strict=True):
        """strict=False tolerates ckpt params the model no longer has (e.g. a different LoRA
        layout); params the model has but the ckpt lacks always keep their init (new adapters
        are zero-init, so the model starts exactly where the checkpoint left off)."""
        params = dict(self.named_parameters())
        missed = []
        for n, t in ck["trainable"].items():
            if n in params and params[n].shape == t.shape:
                params[n].data.copy_(t)
            else:
                missed.append(n)
        if missed and strict:
            raise SystemExit(f"ckpt/model mismatch: {missed[:8]}")
        if missed:
            print(f"[model] {len(missed)} ckpt params not in model (ok): {missed[:3]}", flush=True)
        own = dict(self.named_buffers())
        for n, b in ck["buffers"].items():
            if "." in n and n in own and own[n].shape == b.shape:
                own[n].data.copy_(b)

    def _core(self):
        bb = self.backbone
        core = getattr(bb, "base_model", bb)
        return getattr(core, "model", core)

    def features(self, pixel_values):
        core = self._core()
        hs = core.embeddings(pixel_values.to(core.embeddings.patch_embeddings.weight.dtype))
        cos, sin = core.rope_embeddings(pixel_values)
        enc = getattr(core, "model", None) or core.layer
        out = enc(hs, (cos, sin))
        return core.norm(out.last_hidden_state)[:, 0]

    def heads(self, feats):
        h = self.trunk(feats)
        comb = self.fine_head(h)
        for l, ch in enumerate(self.coarse_heads):
            comb = comb + ch(h)[:, getattr(self, f"parent_{l}")]
        comb = comb + F.log_softmax(self.country_head(h), dim=-1)[:, self.cell_country]
        fine_b = self.fine_head_b(h) if self.fine_head_b is not None else None
        return h, comb, fine_b


class SweepDataset(Dataset):
    def __init__(self, split, lo, hi):
        self.img_dir, df = load_index(split)
        self.paths = df["path"].tolist()[lo:hi]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = open_image(self.img_dir, self.paths[i]).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--ckpt", default="run_full/ckpt_best.pt")
    ap.add_argument("--out", default="retrieval_index")
    ap.add_argument("--bs", type=int, default=84)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    _, df = load_index(a.split)
    n = len(df)
    chunk = -(-n // WORLD)
    lo, hi = RANK * chunk, min(n, (RANK + 1) * chunk)
    print(f"[r{RANK}] rows {lo}:{hi} of {n}", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GeoModelEval(ck)
    model.load_from_ckpt(ck)
    model.to(device).eval()
    n_cells = int(ck["config"]["n_cells"])
    if RANK == 0:
        print(f"[r0] model ready: {n_cells} cells, step {ck.get('step')}, "
              f"best {ck.get('best_median', float('nan')):.2f} km", flush=True)

    ds = SweepDataset(a.split, lo, hi)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=10, pin_memory=True, shuffle=False)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    m = len(ds)
    emb_bb = np.zeros((m, 1024), np.float16)
    emb_tr = np.zeros((m, 1024), np.float16)
    want_logits = a.split == "val"
    lg_a = np.zeros((m, n_cells), np.float16) if want_logits else None
    lg_b = np.zeros((m, n_cells), np.float16) if want_logits else None

    t0 = time.time()
    off = 0
    with torch.no_grad():
        for bi, xb in enumerate(dl):
            x = xb.to(device, non_blocking=True).float().div_(255.0)
            x = (x - mean_t) / std_t
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                fb = model.features(x).float()
            h, comb, fine_b = model.heads(fb)
            k = fb.size(0)
            emb_bb[off:off + k] = fb.cpu().numpy().astype(np.float16)
            emb_tr[off:off + k] = h.cpu().numpy().astype(np.float16)
            if want_logits:
                lg_a[off:off + k] = comb.float().cpu().numpy().astype(np.float16)
                if fine_b is not None:
                    lg_b[off:off + k] = fine_b.float().cpu().numpy().astype(np.float16)
            off += k
            if bi % 50 == 0:
                r = off / max(1e-9, time.time() - t0)
                print(f"[r{RANK}] {off}/{m} ({r:.0f} img/s, ETA {(m-off)/max(r,1e-9)/60:.1f} min)",
                      flush=True)

    np.save(os.path.join(a.out, f"emb_bb_{a.split}_r{RANK}.npy"), emb_bb)
    np.save(os.path.join(a.out, f"emb_trunk_{a.split}_r{RANK}.npy"), emb_tr)
    if want_logits:
        np.save(os.path.join(a.out, f"logits_a_{a.split}_r{RANK}.npy"), lg_a)
        np.save(os.path.join(a.out, f"logits_b_{a.split}_r{RANK}.npy"), lg_b)
    print(f"[r{RANK}] DONE {m} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
