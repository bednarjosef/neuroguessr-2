#!/usr/bin/env python3
"""Embed an external benchmark image set (manifest CSV: path,lat,lon) with the C4 encoder,
on CPU. Writes {tag}_cls.npy (m,1024) fp16, {tag}_logits.npy (m,C) fp16, {tag}_latlon.npz
in manifest row order — the exact query-side artifacts the retrieval eval consumes.

Preprocessing matches embed_c2.py val sweeps: square resize to --img-size (BICUBIC),
/255, ImageNet norm. CLS via GeoModelEval.features() (normed last hidden state, token 0).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = os.environ.get("NG_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from retrieval.embed_full import GeoModelEval, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

torch.set_num_threads(os.cpu_count())
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True


class ManifestDataset(Dataset):
    def __init__(self, root, rows, size):
        self.root, self.rows, self.size = root, rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(os.path.join(self.root, self.rows[i][0]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with header path,lat,lon")
    ap.add_argument("--root", default=None, help="image root (default: manifest's dir)")
    ap.add_argument("--ckpt", default=os.path.join(REPO, "run_c4_index", "ckpt.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    root = a.root or os.path.dirname(os.path.abspath(a.manifest))
    os.makedirs(a.out, exist_ok=True)

    with open(a.manifest) as f:
        rd = csv.DictReader(f)
        rows = [(r["path"], float(r["lat"]), float(r["lon"])) for r in rd]
    m = len(rows)
    print(f"{a.tag}: {m} images from {a.manifest}", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GeoModelEval(ck)
    model.load_from_ckpt(ck, strict=False)
    model.to(DEV).eval()
    n_cells = int(ck["config"]["n_cells"])
    print(f"model ready: {n_cells} cells, step {ck.get('step')}, device {DEV}", flush=True)

    dl = DataLoader(ManifestDataset(root, rows, a.img_size), batch_size=a.bs,
                    num_workers=a.workers, shuffle=False)
    mean_t = torch.tensor(IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=DEV).view(1, 3, 1, 1)

    cls_out = np.zeros((m, 1024), np.float16)
    log_out = np.zeros((m, n_cells), np.float16)
    t0, off = time.time(), 0
    with torch.no_grad():
        for batch in dl:
            x = (batch.to(DEV).float().div_(255.0) - mean_t) / std_t
            if DEV.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    cls = model.features(x)
            else:
                cls = model.features(x)
            cls = cls.float()
            _, comb, _ = model.heads(cls)
            k = cls.size(0)
            cls_out[off:off + k] = cls.cpu().numpy().astype(np.float16)
            log_out[off:off + k] = comb.float().cpu().numpy().astype(np.float16)
            off += k
            r = off / max(1e-9, time.time() - t0)
            print(f"  {off}/{m} ({r:.1f} img/s, ETA {(m-off)/max(r,1e-9)/60:.1f} min)", flush=True)

    np.save(os.path.join(a.out, f"{a.tag}_cls.npy"), cls_out)
    np.save(os.path.join(a.out, f"{a.tag}_logits.npy"), log_out)
    np.savez(os.path.join(a.out, f"{a.tag}_latlon.npz"),
             lat=np.array([r[1] for r in rows], np.float64),
             lon=np.array([r[2] for r in rows], np.float64),
             path=np.array([r[0] for r in rows]))
    print(f"DONE {m} rows in {(time.time()-t0)/60:.1f} min -> {a.out}/{a.tag}_*.npy", flush=True)


if __name__ == "__main__":
    main()
