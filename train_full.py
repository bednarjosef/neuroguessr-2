"""
FULL-DATASET training run (all ~1.2M train images) — not a ratchet experiment.

Same champion architecture as train.py (DINOv3 ViT-L/16 + LoRA r16, 384px, PatchDropout 0.6,
k-means geocells + 64/512 hier levels + tau-smoothed country level, mode-seeking predict),
adapted for one long run:
  - step-based cosine schedule over AR_EPOCHS (default 3) full passes of the dataset
  - runs on 1..N GPUs from the same file: under torchrun each rank takes a disjoint shard
    of every epoch's permutation and gradients are all-reduced (the backbone is frozen, so
    only the ~10M trainable params sync — a few ms/step)
  - eval every ~100k samples on the fixed 1000-image val subset, plus a step-0 eval and a
    final full-val eval (the official median_km panel)
  - checkpoints every ~200k samples + on every new best median: trainable params, ALL model
    buffers (geocells — so a resume on a different box keeps identical cells), optimizer,
    exact step/epoch/batch position, W&B run id. Resume with AR_RESUME=auto.
  - optional off-box safety: AR_CKPT_HF_REPO=<user/repo> uploads best/last checkpoints to a
    private HF repo in a background thread (box death then costs at most CKPT_SAMPLES).

Env knobs: AR_EPOCHS (3), AR_CELLS (5000), AR_MAX_STEPS (smoke test), AR_RESUME (auto|none|path),
AR_CKPT_HF_REPO, AR_RUN_NAME (W&B run id), AR_TIME_BUDGET (wall backstop seconds, default 24h).
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import re
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import prepare as _prepare
from prepare import (QUICK_VAL_N, EARTH_RADIUS_KM, load_index, open_image,
                     evaluate_geo, TrainingTimeUp, start_training_clock, stop_training_clock)

# ---------------------------------------------------------------------------
# Distributed setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; plain python = 1 GPU)
# ---------------------------------------------------------------------------

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DIST = WORLD > 1
if DIST:
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(LOCAL_RANK)
is_main = RANK == 0


def p0(*a, **kw):
    if is_main:
        print(*a, **kw, flush=True)


@contextmanager
def rank0_first():
    """Rank 0 runs the body first (builds caches / downloads weights); others wait, then
    run the same body against the warm cache."""
    if DIST and not is_main:
        dist.barrier()
    yield
    if DIST and is_main:
        dist.barrier()

# ---------------------------------------------------------------------------
# Hyperparameters — champion stack (train.py @ 1dfceb0) + full-run knobs
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_SIZE = 384
NUM_PREFIX_TOKENS = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
GRAD_CHECKPOINT = False

HIER_CELLS = [64, 512]
HIER_LOSS_W = [0.25, 0.5]
COUNTRY_LOSS_W = 0.5
COUNTRY_TAU_KM = 300.0
# 5000 fine cells (vs 2048 at 60k imgs): 1.2M/5000 = 240 imgs/cell — the same per-cell density
# PIGEON trained at (500k/2000). Lowers the quantization floor the S5 H3 experiments hit.
N_CELLS = int(os.environ.get("AR_CELLS", "5000"))
KMEANS_ITERS = 25
SMOOTH_TAU_KM = 75.0
PRED_TOPK = 16
PRED_TEMP = 0.5
PRED_RADIUS_KM = 1000.0
HEAD_HIDDEN = 1024
HEAD_DROPOUT = 0.1
POOL = "cls"
PATCH_KEEP = 0.6

# Optimization — per-GPU batch is the proven 96 @ 31GB; LRs sqrt-scale with world size.
DEVICE_BATCH_SIZE = 96
LR_SCALE = math.sqrt(WORLD)
LORA_LR = 1e-4 * LR_SCALE
HEAD_LR = 1e-3 * LR_SCALE
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.9, 0.95)
FINAL_LR_FRAC = 0.05
NUM_WORKERS = max(4, min(16, (os.cpu_count() or 32) // WORLD - 2))

EPOCHS = int(os.environ.get("AR_EPOCHS", "3"))
GLOBAL_BS = DEVICE_BATCH_SIZE * WORLD
EVAL_SAMPLES = 100_000          # quick eval every ~100k samples (36 evals over 3 epochs)
CKPT_SAMPLES = 200_000          # periodic checkpoint every ~200k samples
DATA_ORDER_SEED = 1234          # per-epoch permutation seed (part of exact resume)
RUN_DIR = os.environ.get("AR_RUN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "run_full"))
CKPT_HF_REPO = os.environ.get("AR_CKPT_HF_REPO", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "neuroguessr-2-research")
MAX_STEPS_OVERRIDE = int(os.environ.get("AR_MAX_STEPS", "0"))   # smoke-test cap
WALL_BACKSTOP_S = int(os.environ.get("AR_TIME_BUDGET", str(24 * 3600)))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)                      # identical CPU init on every rank
np.random.seed(42)
torch.cuda.manual_seed_all(42 + RANK)      # rank-decorrelated PatchDropout masks
torch.set_float32_matmul_precision("high")
device = torch.device("cuda", LOCAL_RANK)
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

MEAN_T = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)


def latlon_to_unit(lat, lon):
    lat = torch.deg2rad(lat); lon = torch.deg2rad(lon)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


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


def assign_cells(pts, centroids, chunk=100_000):
    """Nearest-centroid assignment, chunked over points (1.2M x 5000 sims would be 24GB)."""
    out = torch.empty(pts.size(0), dtype=torch.long, device=pts.device)
    for i in range(0, pts.size(0), chunk):
        out[i:i + chunk] = (pts[i:i + chunk] @ centroids.T).argmax(dim=1)
    return out

# ---------------------------------------------------------------------------
# Geocells (disk-cached; rank 0 builds, other ranks read the cache)
# ---------------------------------------------------------------------------

def build_geocells(train_df, n_cells, iters):
    from prepare import CACHE_DIR
    _ck = os.path.join(CACHE_DIR, f"geocells_{n_cells}_{iters}_{len(train_df)}.pt")
    if os.path.exists(_ck):
        d = torch.load(_ck, map_location=device)
        return d["centroids"], d["cell_lat"], d["cell_lon"]
    lat = torch.tensor(train_df["latitude"].to_numpy(), dtype=torch.float32, device=device)
    lon = torch.tensor(train_df["longitude"].to_numpy(), dtype=torch.float32, device=device)
    pts = latlon_to_unit(lat, lon)
    g = torch.Generator(device=device).manual_seed(0)
    idx = torch.randperm(pts.size(0), generator=g, device=device)[:n_cells]
    centroids = pts[idx].clone()
    for _ in range(iters):
        assign = assign_cells(pts, centroids)
        new = torch.zeros_like(centroids)
        new.index_add_(0, assign, pts)
        counts = torch.zeros(n_cells, device=device).index_add_(
            0, assign, torch.ones_like(lat))
        mask = counts > 0
        new[mask] = F.normalize(new[mask], dim=-1)
        centroids[mask] = new[mask]
    cell_lat, cell_lon = unit_to_latlon(centroids)
    centroids, cell_lat, cell_lon = centroids.detach(), cell_lat.detach(), cell_lon.detach()
    try:
        torch.save({"centroids": centroids, "cell_lat": cell_lat, "cell_lon": cell_lon}, _ck)
    except OSError:
        pass
    return centroids, cell_lat, cell_lon

# ---------------------------------------------------------------------------
# Dataset / dataloader
# ---------------------------------------------------------------------------

class GeoDataset(Dataset):
    def __init__(self, split, size):
        self.img_dir, self.df = load_index(split)
        self.size = size
        self.lat = self.df["latitude"].to_numpy(dtype=np.float32)
        self.lon = self.df["longitude"].to_numpy(dtype=np.float32)
        self.paths = self.df["path"].tolist()
        self.country = (self.df["country_code"].fillna("??").astype(str)
                        .map(lambda c: C2I.get(c, 0)).to_numpy(dtype=np.int64))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = open_image(self.img_dir, self.paths[i]).resize((self.size, self.size), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
        return arr, self.lat[i], self.lon[i], self.country[i]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    lat = torch.tensor([b[1] for b in batch])
    lon = torch.tensor([b[2] for b in batch])
    ctry = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return imgs, lat, lon, ctry


def normalize_batch(imgs_uint8):
    x = imgs_uint8.to(device, non_blocking=True).float().div_(255.0)
    return (x - MEAN_T) / STD_T

# ---------------------------------------------------------------------------
# Model (identical to champion train.py)
# ---------------------------------------------------------------------------

class GeoModel(nn.Module):
    def __init__(self, centroids, cell_lat, cell_lon, hier=(), cell_country=None, n_country=0):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        backbone = AutoModel.from_pretrained(MODEL_NAME)
        import inspect
        self._fwd_extra = ({"interpolate_pos_encoding": True}
                           if "interpolate_pos_encoding" in inspect.signature(backbone.forward).parameters
                           else {})
        if GRAD_CHECKPOINT:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGETS, bias="none")
        self.backbone = get_peft_model(backbone, lora)
        hidden = backbone.config.hidden_size
        feat_dim = hidden * (2 if POOL == "cls_mean" else 1)

        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
        )
        self.fine_head = nn.Linear(HEAD_HIDDEN, N_CELLS)
        self.coarse_heads = nn.ModuleList([nn.Linear(HEAD_HIDDEN, c) for c in HIER_CELLS])
        self.country_head = nn.Linear(HEAD_HIDDEN, n_country) if n_country else None
        if cell_country is not None:
            self.register_buffer("cell_country", cell_country)
        self.register_buffer("centroids", centroids)
        self.register_buffer("cell_lat", cell_lat)
        self.register_buffer("cell_lon", cell_lon)
        for l, (cc, clat, clon) in enumerate(hier):
            self.register_buffer(f"coarse_lat_{l}", clat)
            self.register_buffer(f"coarse_lon_{l}", clon)
            self.register_buffer(f"parent_{l}", (centroids @ cc.T).argmax(dim=1))

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
        if self.training and PATCH_KEEP < 1.0:
            n_patch = cos.shape[0]
            keep = max(1, int(n_patch * PATCH_KEEP))
            idx = torch.randperm(n_patch, device=hs.device)[:keep]
            hs = torch.cat([hs[:, :NUM_PREFIX_TOKENS], hs[:, NUM_PREFIX_TOKENS:][:, idx]], dim=1)
            cos, sin = cos[idx], sin[idx]
        h = self.encode(hs, cos, sin)
        if POOL == "cls":
            f = h[:, 0]
        elif POOL == "mean":
            f = h[:, NUM_PREFIX_TOKENS:].mean(dim=1)
        else:
            f = torch.cat([h[:, 0], h[:, NUM_PREFIX_TOKENS:].mean(dim=1)], dim=-1)
        return f

    def head_logits(self, feats):
        h = self.trunk(feats)
        fine = self.fine_head(h)
        coarse = [ch(h) for ch in self.coarse_heads]
        comb = fine
        for l, cl in enumerate(coarse):
            comb = comb + cl[:, getattr(self, f"parent_{l}")]
        country = None
        if self.country_head is not None:
            country = self.country_head(h)
            comb = comb + F.log_softmax(country, dim=-1)[:, self.cell_country]
        return comb, coarse, country

    def logits(self, pixel_values):
        return self.head_logits(self.features(pixel_values).float())[0]

    @torch.no_grad()
    def predict_latlon(self, pixel_values, topk=PRED_TOPK, collect=None):
        logits = self.logits(pixel_values)
        probs = F.softmax(logits.float() / PRED_TEMP, dim=-1)
        k = min(topk, probs.size(-1))
        w, idx = probs.topk(k, dim=-1)
        if collect is not None:
            collect.append(idx[:, :5].cpu())
        d_top1 = haversine_km_t(self.cell_lat[idx[:, :1]], self.cell_lon[idx[:, :1]],
                                self.cell_lat[idx], self.cell_lon[idx])
        w = w * (d_top1 <= PRED_RADIUS_KM)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        cents = self.centroids[idx]
        v = (w.unsqueeze(-1) * cents).sum(dim=1)
        return unit_to_latlon(v)


def soft_targets(true_lat, true_lon, cell_lat, cell_lon, tau_km):
    d = haversine_km_t(true_lat.unsqueeze(1), true_lon.unsqueeze(1),
                       cell_lat.unsqueeze(0), cell_lon.unsqueeze(0))
    return F.softmax(-d / tau_km, dim=-1)

# ---------------------------------------------------------------------------
# Build cells, country level, model (rank 0 first so caches/downloads happen once)
# ---------------------------------------------------------------------------

_, train_df = load_index("train")
N_TRAIN_IMGS = len(train_df)
p0(f"Train pool: {N_TRAIN_IMGS} images | Backbone: {MODEL_NAME} @ {IMG_SIZE}px | "
   f"{WORLD} GPU(s) x bs{DEVICE_BATCH_SIZE} = global {GLOBAL_BS}")

with rank0_first():
    centroids, cell_lat, cell_lon = build_geocells(train_df, N_CELLS, KMEANS_ITERS)
    hier = [build_geocells(train_df, c, KMEANS_ITERS) for c in HIER_CELLS]
p0(f"Geocells: {N_CELLS} fine + hierarchy {HIER_CELLS} (k-means, {KMEANS_ITERS} iters)")

_cc = train_df["country_code"].fillna("??").astype(str)
COUNTRIES = sorted(_cc.unique().tolist())
C2I = {c: i for i, c in enumerate(COUNTRIES)}
N_COUNTRY = len(COUNTRIES)
with torch.no_grad():
    _tlat = torch.tensor(train_df["latitude"].to_numpy(), dtype=torch.float32, device=device)
    _tlon = torch.tensor(train_df["longitude"].to_numpy(), dtype=torch.float32, device=device)
    _pts = latlon_to_unit(_tlat, _tlon)
    _assign = assign_cells(_pts, centroids)
    _pc = torch.tensor(_cc.map(C2I).to_numpy(dtype=np.int64), device=device)
    _counts = torch.zeros(N_CELLS, N_COUNTRY, device=device)
    _counts.index_put_((_assign, _pc), torch.ones_like(_pc, dtype=torch.float32), accumulate=True)
    _empty = _counts.sum(dim=1) == 0
    _counts[_empty] = _counts.sum(dim=0)
    CELL_COUNTRY = _counts.argmax(dim=1)
    _csum = torch.zeros(N_COUNTRY, 3, device=device).index_add_(0, _pc, _pts)
    COUNTRY_LAT, COUNTRY_LON = unit_to_latlon(F.normalize(_csum, dim=-1))
    del _tlat, _tlon, _pts, _assign, _pc, _counts, _csum
p0(f"Countries: {N_COUNTRY} (majority-vote parents for {N_CELLS} cells)")

with rank0_first():
    model = GeoModel(centroids, cell_lat, cell_lon, hier,
                     cell_country=CELL_COUNTRY, n_country=N_COUNTRY).to(device)
model.trunk.to(torch.float32); model.fine_head.to(torch.float32)
model.coarse_heads.to(torch.float32)
if model.country_head is not None:
    model.country_head.to(torch.float32)

lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
head_params = (list(model.trunk.parameters()) + list(model.fine_head.parameters())
               + list(model.coarse_heads.parameters())
               + (list(model.country_head.parameters()) if model.country_head is not None else []))
trainable_params = lora_params + head_params
n_train_params = sum(p.numel() for p in trainable_params)
p0(f"Trainable params: {n_train_params/1e6:.2f}M (LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M "
   f"+ head {sum(p.numel() for p in head_params)/1e6:.2f}M) | LR scale x{LR_SCALE:.2f}")

optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": LORA_LR},
    {"params": head_params, "lr": HEAD_LR},
], betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

train_ds = GeoDataset("train", IMG_SIZE)

# Rank-synchronized step math: every rank must run the SAME number of steps per epoch or the
# gradient all-reduce deadlocks; truncate each rank's shard to the common minimum.
STEPS_PER_EPOCH = (N_TRAIN_IMGS // WORLD) // DEVICE_BATCH_SIZE
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
if MAX_STEPS_OVERRIDE:
    TOTAL_STEPS = min(TOTAL_STEPS, MAX_STEPS_OVERRIDE)
WARMUP_STEPS = max(300, int(0.02 * TOTAL_STEPS))
EVAL_EVERY = max(1, EVAL_SAMPLES // GLOBAL_BS)
CKPT_EVERY = max(1, CKPT_SAMPLES // GLOBAL_BS)
p0(f"Plan: {EPOCHS} epochs x {STEPS_PER_EPOCH} steps = {TOTAL_STEPS} steps "
   f"(global batch {GLOBAL_BS}) | warmup {WARMUP_STEPS} | eval every {EVAL_EVERY} | "
   f"ckpt every {CKPT_EVERY}")


def make_loader(epoch, skip_batches):
    """Deterministic per-epoch permutation -> per-rank shard -> optional resume offset."""
    g = torch.Generator().manual_seed(DATA_ORDER_SEED + epoch)
    perm = torch.randperm(N_TRAIN_IMGS, generator=g)
    mine = perm[RANK::WORLD][:STEPS_PER_EPOCH * DEVICE_BATCH_SIZE]
    mine = mine[skip_batches * DEVICE_BATCH_SIZE:]
    sub = Subset(train_ds, mine.tolist())
    return DataLoader(sub, batch_size=DEVICE_BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, drop_last=True, collate_fn=collate,
                      pin_memory=True, prefetch_factor=4)


def lr_mult(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    cos = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * cos

# ---------------------------------------------------------------------------
# Eval plumbing (rank 0 only; other ranks wait at a barrier)
# ---------------------------------------------------------------------------

def make_predict_fn(collect=None):
    from PIL import Image
    model.eval()

    def predict_fn(pil_images):
        arr = np.stack([np.asarray(im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8)
                        for im in pil_images])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        with torch.no_grad(), autocast_ctx:
            lat, lon = model.predict_latlon(normalize_batch(x), collect=collect)
        return lat.float().cpu().numpy(), lon.float().cpu().numpy()
    return predict_fn


_, _val_df = load_index("val")
with torch.no_grad():
    _vlat = torch.tensor(_val_df["latitude"].to_numpy(), dtype=torch.float32, device=device)
    _vlon = torch.tensor(_val_df["longitude"].to_numpy(), dtype=torch.float32, device=device)
    _d = haversine_km_t(_vlat.unsqueeze(1), _vlon.unsqueeze(1),
                        cell_lat.unsqueeze(0), cell_lon.unsqueeze(0))
    VAL_TRUE_CELLS = _d.argmin(dim=1).cpu()
    # Quantization floor: median error IF classification were perfect (top-1 = nearest cell).
    _floor = float(_d.min(dim=1).values.median())
    del _d
p0(f"Val quantization floor @ {N_CELLS} cells: {_floor:.1f} km "
   f"(median dist to nearest cell centroid)")
VAL_TRUE_COUNTRY = torch.tensor(_val_df["country_code"].fillna("??").astype(str)
                                .map(lambda c: C2I.get(c, -1)).to_numpy(dtype=np.int64))
CELL_COUNTRY_CPU = CELL_COUNTRY.cpu()


def val_cell_metrics(collected):
    if not collected:
        return {}
    top5 = torch.cat(collected)
    true = VAL_TRUE_CELLS[:top5.size(0)]
    top1 = (top5[:, 0] == true).float().mean().item()
    top5a = (top5 == true.unsqueeze(1)).any(dim=1).float().mean().item()
    country_acc = (CELL_COUNTRY_CPU[top5[:, 0]]
                   == VAL_TRUE_COUNTRY[:top5.size(0)]).float().mean().item()
    return {"cell_top1": top1, "cell_top5": top5a, "country_acc": country_acc,
            "cell_top1_lift": top1 * N_CELLS, "cell_top5_lift": top5a * N_CELLS / 5}


def quick_eval(tag, step):
    m = None
    if is_main:
        coll = []
        pf = make_predict_fn(collect=coll)
        m = evaluate_geo(pf, split="val", subset=QUICK_VAL_N, batch_size=DEVICE_BATCH_SIZE)
        cm = val_cell_metrics(coll)
        print(f"\n[{tag}] median_km={m['median_km']:.1f} mean_km={m['mean_km']:.1f} "
              f"acc@25km={m['acc_25km']*100:.1f}% acc@200km={m['acc_200km']*100:.1f}% "
              f"acc@2500km={m['acc_2500km']*100:.1f}% geoguessr={m['geoguessr_score']:.0f} "
              f"cell_top1={cm.get('cell_top1', 0)*100:.1f}% "
              f"country={cm.get('country_acc', 0)*100:.1f}%", flush=True)
        if wandb_run is not None:
            wandb.log({f"val/{k}": v for k, v in {**m, **cm}.items()}, step=step)
    model.train()
    if DIST:
        dist.barrier()
    return m

# ---------------------------------------------------------------------------
# Checkpointing (rank 0 writes; every rank can load)
# ---------------------------------------------------------------------------

os.makedirs(RUN_DIR, exist_ok=True)
_upload_lock = threading.Lock()


def _hf_upload(path, tag):
    """Best-effort background upload of a checkpoint to a private HF repo."""
    if not CKPT_HF_REPO:
        return

    def _up():
        if not _upload_lock.acquire(blocking=False):
            return          # previous upload still running; skip this one
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(CKPT_HF_REPO, repo_type="model", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=path, path_in_repo=f"ckpt_{tag}.pt",
                            repo_id=CKPT_HF_REPO, repo_type="model")
            print(f"\n[ckpt] uploaded ckpt_{tag}.pt -> {CKPT_HF_REPO}", flush=True)
        except Exception as e:
            print(f"\n[ckpt] HF upload failed ({type(e).__name__}: {e}) — continuing", flush=True)
        finally:
            _upload_lock.release()
    threading.Thread(target=_up, daemon=True).start()


def save_ckpt(tag, step, epoch, batches_in_epoch, best_median):
    if not is_main:
        return
    t0 = time.time()
    ck = {
        "step": step, "epoch": epoch, "batches_in_epoch": batches_in_epoch,
        "best_median": best_median,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "buffers": {n: b.detach().cpu() for n, b in model.named_buffers()},
        "optimizer": optimizer.state_dict(),
        "country_lat": COUNTRY_LAT.cpu(), "country_lon": COUNTRY_LON.cpu(),
        "countries": COUNTRIES,
        "wandb_id": WANDB_ID,
        "config": {"n_cells": N_CELLS, "hier": HIER_CELLS, "epochs": EPOCHS,
                   "world": WORLD, "bs": DEVICE_BATCH_SIZE, "img": IMG_SIZE,
                   "model": MODEL_NAME, "lora_r": LORA_R, "n_train": N_TRAIN_IMGS},
    }
    path = os.path.join(RUN_DIR, f"ckpt_{tag}.pt")
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)
    print(f"\n[ckpt] saved {tag} @ step {step} ({time.time()-t0:.1f}s)", flush=True)
    _hf_upload(path, tag)


def try_resume():
    """Returns (step, epoch, batches_in_epoch, best_median, wandb_id) or zeros."""
    spec = os.environ.get("AR_RESUME", "auto")
    if spec == "none":
        return 0, 0, 0, float("inf"), None
    path = spec if spec not in ("auto", "") else os.path.join(RUN_DIR, "ckpt_last.pt")
    if not os.path.exists(path):
        best_p = os.path.join(RUN_DIR, "ckpt_best.pt")
        if spec in ("auto", "") and os.path.exists(best_p):
            path = best_p
        else:
            return 0, 0, 0, float("inf"), None
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    assert cfg["n_cells"] == N_CELLS and cfg["world"] == WORLD and cfg["bs"] == DEVICE_BATCH_SIZE, \
        f"checkpoint config mismatch: {cfg} vs cells={N_CELLS} world={WORLD} bs={DEVICE_BATCH_SIZE}"
    assert ck["countries"] == COUNTRIES, "country vocabulary changed between runs"
    with torch.no_grad():
        bufs = dict(model.named_buffers())
        for n, b in ck["buffers"].items():
            bufs[n].copy_(b.to(device))
        params = dict(model.named_parameters())
        for n, t in ck["trainable"].items():
            params[n].data.copy_(t.to(device))
        COUNTRY_LAT.copy_(ck["country_lat"].to(device))
        COUNTRY_LON.copy_(ck["country_lon"].to(device))
    optimizer.load_state_dict(ck["optimizer"])
    p0(f"RESUMED from {path}: step {ck['step']}, epoch {ck['epoch']}, "
       f"batch {ck['batches_in_epoch']}, best {ck['best_median']:.1f}")
    return ck["step"], ck["epoch"], ck["batches_in_epoch"], ck["best_median"], ck.get("wandb_id")


start_step, start_epoch, start_batches, best_median, _resume_wandb = try_resume()

if DIST:   # identical weights/buffers everywhere (same seed + same cache, but be certain)
    for t in trainable_params:
        dist.broadcast(t.data, src=0)
    for b in model.buffers():
        dist.broadcast(b, src=0)

# ---------------------------------------------------------------------------
# W&B (rank 0; resumed runs continue the same W&B run at the right step)
# ---------------------------------------------------------------------------

WANDB_ID = None
wandb_run = None
if is_main:
    try:
        import wandb
        _cfg = {k: v for k, v in globals().items()
                if k.isupper() and isinstance(v, (int, float, str, bool, tuple, list))}
        _cfg["n_train_params_M"] = round(n_train_params / 1e6, 2)
        WANDB_ID = (_resume_wandb or os.environ.get("AR_RUN_NAME")
                    or f"fullrun-{N_CELLS}c-{EPOCHS}ep-w{WORLD}")
        WANDB_ID = re.sub(r"[^a-zA-Z0-9_-]", "-", WANDB_ID)
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
        wandb_run = wandb.init(project=WANDB_PROJECT, config=_cfg, mode=wandb_mode,
                               id=WANDB_ID, name=WANDB_ID, resume="allow")
        print(f"W&B: {wandb_mode} (project={WANDB_PROJECT}, id={WANDB_ID})")
    except Exception as e:
        print(f"W&B init failed ({e}); continuing without tracking")
        wandb, wandb_run = None, None

# ---------------------------------------------------------------------------
# Compile + warmup (once, before the loop; eval shapes included)
# ---------------------------------------------------------------------------

model.train()
model.encode = torch.compile(model.encode, dynamic=True)
p0("compile warmup…")
_t_c = time.time()
_wb = collate([train_ds[i] for i in range(DEVICE_BATCH_SIZE)])
_x = normalize_batch(_wb[0])
with autocast_ctx:
    _feats = model.features(_x)
_logits, _, _ = model.head_logits(_feats.float())
_logits.mean().backward()
optimizer.zero_grad(set_to_none=True)
model.eval()
with torch.no_grad(), autocast_ctx:
    model.features(_x[:16]); model.features(_x[:7])
model.train()
del _wb, _x, _feats, _logits
p0(f"compile warmup done in {time.time()-_t_c:.0f}s")

if start_step == 0:
    quick_eval("step 0", 0)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

_prepare.TIME_BUDGET = WALL_BACKSTOP_S   # SIGALRM backstop only; the step count is the plan
start_training_clock()
step = start_step
smooth_loss = 0.0
t_train0 = time.time()
_win_t0, _win_step0 = time.time(), step

try:
    for epoch in range(start_epoch, EPOCHS):
        if step >= TOTAL_STEPS:
            break
        in_epoch = start_batches if epoch == start_epoch else 0
        loader = make_loader(epoch, in_epoch)
        for imgs, blat, blon, bctry in loader:
            optimizer.zero_grad(set_to_none=True)
            x = normalize_batch(imgs)
            blat = blat.to(device); blon = blon.to(device); bctry = bctry.to(device)
            with autocast_ctx:
                feats = model.features(x)
            logits, coarse_logits, country_logits = model.head_logits(feats.float())
            tgt = soft_targets(blat, blon, cell_lat, cell_lon, SMOOTH_TAU_KM)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(tgt * logp).sum(dim=-1).mean()
            for l, cl in enumerate(coarse_logits):
                tgt_l = soft_targets(blat, blon, getattr(model, f"coarse_lat_{l}"),
                                     getattr(model, f"coarse_lon_{l}"), SMOOTH_TAU_KM)
                loss = loss + HIER_LOSS_W[l] * -(tgt_l * F.log_softmax(cl, dim=-1)).sum(dim=-1).mean()
            if country_logits is not None:
                tgt_c = soft_targets(blat, blon, COUNTRY_LAT, COUNTRY_LON, COUNTRY_TAU_KM)
                loss = loss + COUNTRY_LOSS_W * -(tgt_c * F.log_softmax(country_logits, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            loss_val = loss.item()

            if DIST:
                for p in trainable_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            lrm = lr_mult(step)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * lrm
            optimizer.step()

            if math.isnan(loss_val) or loss_val > 1e4:
                p0("FAIL: loss diverged"); raise SystemExit(1)

            step += 1
            in_epoch += 1
            beta = 0.9
            smooth_loss = beta * smooth_loss + (1 - beta) * loss_val
            deb = smooth_loss / (1 - beta ** (step - start_step))

            if is_main and step % 20 == 0:
                dt_win = time.time() - _win_t0
                ips = (step - _win_step0) * GLOBAL_BS / max(1e-9, dt_win)
                eta_h = (TOTAL_STEPS - step) / max(1e-9, (step - _win_step0) / dt_win) / 3600
                print(f"step {step}/{TOTAL_STEPS} ({100*step/TOTAL_STEPS:.1f}%) | "
                      f"loss {deb:.4f} | lrm {lrm:.2f} | {ips:.0f} img/s | "
                      f"ep {epoch+1}/{EPOCHS} | eta {eta_h:.2f}h", flush=True)
                _win_t0, _win_step0 = time.time(), step
                if wandb_run is not None:
                    wandb.log({"train/loss": deb, "train/lr_mult": lrm, "train/img_per_s": ips,
                               "train/progress": step / TOTAL_STEPS, "epoch": epoch + 1,
                               "train/samples_seen": step * GLOBAL_BS}, step=step)

            if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
                m = quick_eval(f"step {step}", step)
                if is_main and m is not None and m["median_km"] < best_median:
                    best_median = m["median_km"]
                    save_ckpt("best", step, epoch, in_epoch, best_median)
            if step % CKPT_EVERY == 0:
                save_ckpt("last", step, epoch, in_epoch, best_median)
                if DIST:
                    dist.barrier()

            if step >= TOTAL_STEPS:
                break
        start_batches = 0
except TrainingTimeUp:
    p0("\n[hard-deadline] wall backstop reached — stopping for final eval")
finally:
    stop_training_clock()

total_training_time = time.time() - t_train0
p0("")

# ---------------------------------------------------------------------------
# Final eval on the FULL val split (official score) + final checkpoint
# ---------------------------------------------------------------------------

save_ckpt("final", step, EPOCHS - 1, 0, best_median)

if is_main:
    model.eval()
    final_coll = []
    predict_fn = make_predict_fn(collect=final_coll)
    with autocast_ctx:
        m = evaluate_geo(predict_fn, split="val", subset=None, batch_size=DEVICE_BATCH_SIZE)
    final_cm = val_cell_metrics(final_coll)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    t_end = time.time()

    print("---")
    print(f"median_km:        {m['median_km']:.6f}")
    print(f"mean_km:          {m['mean_km']:.6f}")
    print(f"acc_1km:          {m['acc_1km']:.6f}")
    print(f"acc_25km:         {m['acc_25km']:.6f}")
    print(f"acc_200km:        {m['acc_200km']:.6f}")
    print(f"acc_750km:        {m['acc_750km']:.6f}")
    print(f"acc_2500km:       {m['acc_2500km']:.6f}")
    print(f"geoguessr_score:  {m['geoguessr_score']:.6f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"best_val_median:  {best_median:.6f}")
    print(f"num_params_M:     {n_train_params/1e6:.2f}")
    for k, v in final_cm.items():
        print(f"{k}: {v:.6f}")

    if wandb_run is not None:
        wandb.log({f"final/{k}": v for k, v in {**m, **final_cm}.items()}, step=step)
        wandb.summary.update({f"final_{k}": v for k, v in {**m, **final_cm}.items()})
        wandb.summary.update({"num_steps": step, "peak_vram_mb": peak_vram_mb,
                              "best_val_median": best_median})
        wandb.finish()

if DIST:
    dist.barrier()
    dist.destroy_process_group()
