#!/usr/bin/env python3
"""C5 bed: dump full val patch tokens (2998 x 576 x 1024 fp16, ~3.5GB) so patch-level
verification and SALAD prototyping stay possible on CPU, forever, without a box.

Run ON THE BOX (single GPU):  python tools/dump_val_patches.py --ckpt run_c5/ckpt_best.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare import load_index, open_image  # noqa: E402
from retrieval.embed_full import GeoModelEval, IMAGENET_MEAN, IMAGENET_STD, NUM_PREFIX_TOKENS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="run_c5/ckpt_best.pt")
    ap.add_argument("--out", default="retrieval_index")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--tag", default="c5")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    device = torch.device("cuda")
    t0 = time.time()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targets = ck.get("config", {}).get("lora_targets", None)
    model = (GeoModelEval(ck, lora_targets=targets, lora_r=ck["config"]["lora_r"])
             if targets else GeoModelEval(ck))
    model.load_from_ckpt(ck, strict=False)
    model.to(device).eval()

    img_dir, df = load_index("val")
    paths = df["path"].tolist()
    n = len(paths)
    n_patch = (a.img_size // 16) ** 2
    out = np.lib.format.open_memmap(
        os.path.join(a.out, f"val_patches_{a.tag}.npy"), mode="w+",
        dtype=np.float16, shape=(n, n_patch, 1024))
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    from PIL import Image
    with torch.no_grad():
        for i in range(0, n, a.bs):
            ims = []
            for p in paths[i:i + a.bs]:
                im = open_image(img_dir, p)
                if im.size != (a.img_size, a.img_size):
                    im = im.resize((a.img_size, a.img_size), Image.BICUBIC)
                ims.append(torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1))
            x = torch.stack(ims).to(device).float().div_(255.0)
            x = (x - mean_t) / std_t
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                tok = model.tokens(x) if hasattr(model, "tokens") else None
            if tok is None:
                from retrieval.embed_c2 import full_tokens
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tok = full_tokens(model, x)
            out[i:i + len(ims)] = tok.float()[:, NUM_PREFIX_TOKENS:].cpu().numpy().astype(np.float16)
    out.flush()
    print(f"VAL PATCHES DONE {n} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
