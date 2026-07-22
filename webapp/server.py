#!/usr/bin/env python3
"""Local demo: paste an image -> predicted location + confidence on a world map.

Loads the full-dataset checkpoint (run_full/ckpt_best.pt) and serves a tiny FastAPI app.
CPU inference (a few seconds per image on a laptop; uses CUDA automatically if present).

Run from the repo root:
    .venv/bin/python webapp/server.py            # then open http://127.0.0.1:8765
Options: --ckpt run_full/ckpt_best.pt --port 8765
"""
import argparse
import io
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HF_TOKEN for the gated DINOv3 backbone (never committed; read from the local .env)
_envf = os.path.join(REPO, ".env")
if os.path.exists(_envf):
    for _line in open(_envf):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

IMG_SIZE = 384
NUM_PREFIX_TOKENS = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRED_TEMP = 0.5
PRED_TOPK = 16
PRED_RADIUS_KM = 1000.0
HEAD_HIDDEN = 1024
EARTH_RADIUS_KM = 6371.0088

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 1))


def unit_to_latlon(v):
    v = F.normalize(v, dim=-1)
    lat = torch.rad2deg(torch.asin(v[..., 2].clamp(-1, 1)))
    lon = torch.rad2deg(torch.atan2(v[..., 1], v[..., 0]))
    return lat, lon


def haversine_km_t(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))


class GeoModel(nn.Module):
    """Eval-only mirror of the training GeoModel; weights come from the checkpoint."""

    def __init__(self, ck):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        cfg = ck["config"]
        bufs = ck["buffers"]
        n_cells = cfg["n_cells"]
        n_country = len(ck["countries"])

        backbone = AutoModel.from_pretrained(cfg["model"])
        lora = LoraConfig(r=cfg["lora_r"], lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
        self.backbone = get_peft_model(backbone, lora)
        hidden = backbone.config.hidden_size

        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.0),
        )
        self.fine_head = nn.Linear(HEAD_HIDDEN, n_cells)
        self.coarse_heads = nn.ModuleList([nn.Linear(HEAD_HIDDEN, c) for c in cfg["hier"]])
        self.country_head = nn.Linear(HEAD_HIDDEN, n_country)
        # Top-level buffers (geocells, parents, cell->country) are registered fresh; dotted
        # names are the backbone's own internal buffers — copied into place further down.
        for name, b in bufs.items():
            if "." not in name:
                self.register_buffer(name, b.clone())

    def load_nested_buffers(self, bufs):
        own = dict(self.named_buffers())
        for name, b in bufs.items():
            if "." in name and name in own and own[name].shape == b.shape:
                own[name].data.copy_(b)

    def _core(self):
        bb = self.backbone
        core = getattr(bb, "base_model", bb)
        return getattr(core, "model", core)

    def encode(self, hs, cos, sin):
        core = self._core()
        enc = getattr(core, "model", None) or core.layer
        out = enc(hs, (cos, sin))
        return core.norm(out.last_hidden_state)

    def features(self, pixel_values):
        core = self._core()
        hs = core.embeddings(pixel_values.to(core.embeddings.patch_embeddings.weight.dtype))
        cos, sin = core.rope_embeddings(pixel_values)
        h = self.encode(hs, cos, sin)
        return h[:, 0]

    def head_logits(self, feats):
        h = self.trunk(feats)
        comb = self.fine_head(h)
        for l, ch in enumerate(self.coarse_heads):
            comb = comb + ch(h)[:, getattr(self, f"parent_{l}")]
        country = self.country_head(h)
        comb = comb + F.log_softmax(country, dim=-1)[:, self.cell_country]
        return comb


print(f"Loading checkpoint… (device={device})")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(REPO, "run_full", "ckpt_best.pt"))
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--host", default="127.0.0.1")
args = ap.parse_args()

CK = torch.load(args.ckpt, map_location="cpu", weights_only=False)
COUNTRIES = CK["countries"]
model = GeoModel(CK)
_params = dict(model.named_parameters())
_loaded, _missed = 0, []
for n, t in CK["trainable"].items():
    if n in _params and _params[n].shape == t.shape:
        _params[n].data.copy_(t)
        _loaded += 1
    else:
        _missed.append(n)
if _missed:
    raise SystemExit(f"checkpoint/model mismatch, unmatched params: {_missed[:8]} …")
model.load_nested_buffers(CK["buffers"])
model.to(device).eval()
print(f"Model ready: {_loaded} trainable tensors loaded, "
      f"{int(CK['config']['n_cells'])} cells, step {CK.get('step')}, "
      f"best val median {CK.get('best_median', float('nan')):.1f} km")

MEAN_T = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    """Center-crop to square (training images are square) then bicubic-resize to 384."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device).float().div_(255.0)
    return (x - MEAN_T) / STD_T


@torch.no_grad()
def predict(img: Image.Image) -> dict:
    t0 = time.time()
    x = preprocess(img)
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.head_logits(model.features(x).float())[0]
    else:
        logits = model.head_logits(model.features(x).float())[0]
    p = F.softmax(logits.float() / PRED_TEMP, dim=-1)        # (C,) sharpened posterior

    # Prediction: mode-seeking spherical mean (same rule as training/eval)
    w, idx = p.topk(PRED_TOPK)
    d_top1 = haversine_km_t(model.cell_lat[idx[:1]], model.cell_lon[idx[:1]],
                            model.cell_lat[idx], model.cell_lon[idx])
    wm = w * (d_top1 <= PRED_RADIUS_KM)
    wm = wm / wm.sum().clamp(min=1e-9)
    v = (wm.unsqueeze(-1) * model.centroids[idx]).sum(dim=0, keepdim=True)
    plat, plon = unit_to_latlon(v)
    plat, plon = float(plat[0]), float(plon[0])

    # Regional confidence: total posterior mass within PRED_RADIUS of the top-1 cell
    d_all = haversine_km_t(model.cell_lat[idx[:1]], model.cell_lon[idx[:1]],
                           model.cell_lat, model.cell_lon)
    conf = float(p[d_all <= PRED_RADIUS_KM].sum())

    # r50 / r90: radius around the *prediction* containing 50% / 90% of posterior mass
    d_pred = haversine_km_t(torch.tensor([plat], device=device),
                            torch.tensor([plon], device=device),
                            model.cell_lat, model.cell_lon)
    ds, order = d_pred.sort()
    cum = p[order].cumsum(0)
    r50 = float(ds[int((cum >= 0.5).nonzero()[0])]) if float(cum[-1]) >= 0.5 else float(ds[-1])
    r90 = float(ds[int((cum >= 0.9).nonzero()[0])]) if float(cum[-1]) >= 0.9 else float(ds[-1])

    # Country posterior: aggregate cell mass by the cell->country map
    cp = torch.zeros(len(COUNTRIES), device=device).index_add_(0, model.cell_country, p)
    cw, ci = cp.topk(5)
    countries = [{"code": COUNTRIES[int(i)], "p": round(float(x), 4)}
                 for x, i in zip(cw, ci) if float(x) > 0.005]

    # Top cells for the map (display only)
    tw, ti = p.topk(24)
    cells = [{"lat": round(float(model.cell_lat[int(i)]), 4),
              "lon": round(float(model.cell_lon[int(i)]), 4),
              "p": round(float(x), 5)}
             for x, i in zip(tw, ti) if float(x) > 0.002]

    return {"lat": round(plat, 5), "lon": round(plon, 5),
            "confidence": round(conf, 4), "r50_km": round(r50, 1), "r90_km": round(r90, 1),
            "countries": countries, "cells": cells,
            "time_ms": int((time.time() - t0) * 1000)}


# --------------------------------------------------------------------------- FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="neuroguessr-2 demo")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))


@app.get("/health")
def health():
    return {"ok": True, "cells": int(CK["config"]["n_cells"]),
            "best_val_median_km": float(CK.get("best_median", 0.0)), "device": str(device)}


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        return JSONResponse({"error": f"not an image: {e}"}, status_code=400)
    return predict(img)


if __name__ == "__main__":
    import uvicorn
    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
