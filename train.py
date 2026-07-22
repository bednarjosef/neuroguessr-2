"""
Image-geolocalization experiment. Must print one `median_km: <number>` summary line (the
objective) plus any `name: number` diagnostics. Everything here is editable.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prepare import (TIME_BUDGET, QUICK_VAL_N, EARTH_RADIUS_KM, load_index, open_image,
                     evaluate_geo, TrainingTimeUp, start_training_clock, stop_training_clock)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_SIZE = 384                 # must be a multiple of the patch size (16 dinov3 / 14 dinov2)
NUM_PREFIX_TOKENS = 5          # CLS + register tokens before patches (dinov3=5, dinov2=1)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # transformers 5.x DINOv3 naming
GRAD_CHECKPOINT = False   # bs24: activations fit without the ~30% recompute tax

# Head / geocells
# Hierarchical multi-resolution heads: coarse levels regularize the fine head and their
# logits are added (log-space) onto the fine cells at prediction, fixing wrong-region errors.
HIER_CELLS = [64, 512]         # coarse levels (fine level is N_CELLS)
HIER_LOSS_W = [0.25, 0.5]      # per-coarse-level CE weight (fine CE has weight 1.0)
N_CELLS = 2048
KMEANS_ITERS = 25
SMOOTH_TAU_KM = 75.0
PRED_TOPK = 16
PRED_TEMP = 0.5                # sharpen posterior before the spherical mean (mode-seeking)
PRED_RADIUS_KM = 1000.0        # only average cells within this radius of the top-1 cell
HEAD_HIDDEN = 1024
HEAD_DROPOUT = 0.1
POOL = "cls"                   # "cls" | "mean" | "cls_mean"
PATCH_KEEP = 0.5               # PatchDropout/FLIP: fraction of patch tokens kept in TRAIN
                               # forwards (eval always uses all tokens); RoPE cos/sin are
                               # index-selected with the same mask so positions stay correct

# Optimization
DEVICE_BATCH_SIZE = 96         # exp S4.6: PatchDropout halved VRAM (13.8GB @ 48) — double the
GRAD_ACCUM = 1                 # real batch; steps saturated, so buy per-step quality instead
LORA_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.05
FINAL_LR_FRAC = 0.05
NUM_WORKERS = 32

EVAL_EVERY = 250               # steps between monitoring evals (the SIGALRM cap is WALL time;
                               # fewer quick evals -> more of the alarm window spent training)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "neuroguessr-2-research")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

MEAN_T = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)


def latlon_to_unit(lat, lon):
    """(lat, lon) degrees -> unit xyz on the sphere. Tensors."""
    lat = torch.deg2rad(lat); lon = torch.deg2rad(lon)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def unit_to_latlon(v):
    """unit xyz -> (lat, lon) degrees. Tensor (..., 3)."""
    v = F.normalize(v, dim=-1)
    lat = torch.rad2deg(torch.asin(v[..., 2].clamp(-1, 1)))
    lon = torch.rad2deg(torch.atan2(v[..., 1], v[..., 0]))
    return lat, lon


def haversine_km_t(lat1, lon1, lat2, lon2):
    """Great-circle km, torch, broadcasting. Degrees in."""
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))

# ---------------------------------------------------------------------------
# Geocells: k-means on train coordinates (built fresh each run — modeling choice)
# ---------------------------------------------------------------------------

def build_geocells(train_df, n_cells, iters):
    # Disk-cache per (n_cells, iters, n_points): identical cells for every run on this box
    # (CUDA index_add_ is nondeterministic, so rebuilding each run adds tiny cell jitter).
    from prepare import CACHE_DIR
    _ck = os.path.join(CACHE_DIR, f"geocells_{n_cells}_{iters}_{len(train_df)}.pt")
    if os.path.exists(_ck):
        d = torch.load(_ck, map_location=device)
        return d["centroids"], d["cell_lat"], d["cell_lon"]
    lat = torch.tensor(train_df["latitude"].to_numpy(), dtype=torch.float32, device=device)
    lon = torch.tensor(train_df["longitude"].to_numpy(), dtype=torch.float32, device=device)
    pts = latlon_to_unit(lat, lon)  # (N, 3)
    g = torch.Generator(device=device).manual_seed(0)
    idx = torch.randperm(pts.size(0), generator=g, device=device)[:n_cells]
    centroids = pts[idx].clone()
    for _ in range(iters):
        sims = pts @ centroids.T             # (N, C) cosine similarity
        assign = sims.argmax(dim=1)
        new = torch.zeros_like(centroids)
        new.index_add_(0, assign, pts)
        counts = torch.zeros(n_cells, device=device).index_add_(
            0, assign, torch.ones_like(lat))
        mask = counts > 0
        new[mask] = F.normalize(new[mask], dim=-1)
        centroids[mask] = new[mask]          # empty cells keep their seed
    cell_lat, cell_lon = unit_to_latlon(centroids)
    centroids, cell_lat, cell_lon = centroids.detach(), cell_lat.detach(), cell_lon.detach()
    try:
        torch.save({"centroids": centroids, "cell_lat": cell_lat, "cell_lon": cell_lon}, _ck)
    except OSError:
        pass
    return centroids, cell_lat, cell_lon

# ---------------------------------------------------------------------------
# Dataset / dataloader (JPEG -> normalized 448 tensor; on CPU workers)
# ---------------------------------------------------------------------------

class GeoDataset(Dataset):
    def __init__(self, split, size):
        self.img_dir, self.df = load_index(split)
        self.size = size
        self.lat = self.df["latitude"].to_numpy(dtype=np.float32)
        self.lon = self.df["longitude"].to_numpy(dtype=np.float32)
        self.paths = self.df["path"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = open_image(self.img_dir, self.paths[i]).resize((self.size, self.size), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)  # uint8 CHW
        return arr, self.lat[i], self.lon[i]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    lat = torch.tensor([b[1] for b in batch])
    lon = torch.tensor([b[2] for b in batch])
    return imgs, lat, lon


def normalize_batch(imgs_uint8):
    """uint8 CHW batch on device -> normalized float. Kept on GPU so workers stay light."""
    x = imgs_uint8.to(device, non_blocking=True).float().div_(255.0)
    return (x - MEAN_T) / STD_T

# ---------------------------------------------------------------------------
# Model: frozen DINOv2 backbone (LoRA) + geocell head
# ---------------------------------------------------------------------------

class GeoModel(nn.Module):
    def __init__(self, centroids, cell_lat, cell_lon, hier=()):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        backbone = AutoModel.from_pretrained(MODEL_NAME)
        # Only pass interpolate_pos_encoding to backbones that accept it (dinov2 does, dinov3 doesn't).
        import inspect
        self._fwd_extra = ({"interpolate_pos_encoding": True}
                           if "interpolate_pos_encoding" in inspect.signature(backbone.forward).parameters
                           else {})
        if GRAD_CHECKPOINT:
            # use_reentrant=False required: backbone is frozen, so reentrant checkpointing drops LoRA grads.
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
        # Geocell geometry (buffers — fixed per run, not learned)
        self.register_buffer("centroids", centroids)      # (C, 3) unit vectors
        self.register_buffer("cell_lat", cell_lat)         # (C,)
        self.register_buffer("cell_lon", cell_lon)         # (C,)
        # Coarse-level geometry + fine->coarse parent maps (nearest coarse centroid)
        for l, (cc, clat, clon) in enumerate(hier):
            self.register_buffer(f"coarse_lat_{l}", clat)
            self.register_buffer(f"coarse_lon_{l}", clon)
            self.register_buffer(f"parent_{l}", (centroids @ cc.T).argmax(dim=1))

    def _core(self):
        bb = self.backbone
        core = getattr(bb, "base_model", bb)               # PEFT wrapper -> LoraModel
        return getattr(core, "model", core)                # LoraModel -> DINOv3ViTModel

    def encode(self, hs, cos, sin):
        core = self._core()
        enc = getattr(core, "model", None) or core.layer   # tf5.14: .model (encoder module)
        out = enc(hs, (cos, sin))
        return core.norm(out.last_hidden_state)

    def features(self, pixel_values):
        core = self._core()
        hs = core.embeddings(pixel_values.to(core.embeddings.patch_embeddings.weight.dtype))
        cos, sin = core.rope_embeddings(pixel_values)      # (n_patches, head_dim) each
        if self.training and PATCH_KEEP < 1.0:
            n_patch = cos.shape[0]
            keep = max(1, int(n_patch * PATCH_KEEP))
            idx = torch.randperm(n_patch, device=hs.device)[:keep]
            hs = torch.cat([hs[:, :NUM_PREFIX_TOKENS], hs[:, NUM_PREFIX_TOKENS:][:, idx]], dim=1)
            cos, sin = cos[idx], sin[idx]
        h = self.encode(hs, cos, sin)                      # (B, prefix+kept, D)
        if POOL == "cls":
            f = h[:, 0]
        elif POOL == "mean":
            f = h[:, NUM_PREFIX_TOKENS:].mean(dim=1)        # patch tokens only
        else:  # cls_mean
            f = torch.cat([h[:, 0], h[:, NUM_PREFIX_TOKENS:].mean(dim=1)], dim=-1)
        return f

    def head_logits(self, feats):
        """(combined fine logits, [coarse logits per level]). Combined = fine + each
        coarse level's logit broadcast onto its child fine cells (log-space product)."""
        h = self.trunk(feats)
        fine = self.fine_head(h)
        coarse = [ch(h) for ch in self.coarse_heads]
        comb = fine
        for l, cl in enumerate(coarse):
            comb = comb + cl[:, getattr(self, f"parent_{l}")]
        return comb, coarse

    def logits(self, pixel_values):
        return self.head_logits(self.features(pixel_values).float())[0]

    @torch.no_grad()
    def predict_latlon(self, pixel_values, topk=PRED_TOPK, collect=None):
        """Mode-seeking: temperature-sharpened weights over top-k cells, restricted to the
        neighborhood of the top-1 cell, then prob-weighted spherical mean.
        collect: optional list; appends this batch's top-5 cell ids (diagnostics only)."""
        logits = self.logits(pixel_values)
        probs = F.softmax(logits.float() / PRED_TEMP, dim=-1)
        k = min(topk, probs.size(-1))
        w, idx = probs.topk(k, dim=-1)                     # (B, k)
        if collect is not None:
            collect.append(idx[:, :5].cpu())
        # keep only cells near the argmax cell (kills cross-continent averaging)
        d_top1 = haversine_km_t(self.cell_lat[idx[:, :1]], self.cell_lon[idx[:, :1]],
                                self.cell_lat[idx], self.cell_lon[idx])   # (B, k)
        w = w * (d_top1 <= PRED_RADIUS_KM)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        cents = self.centroids[idx]                        # (B, k, 3)
        v = (w.unsqueeze(-1) * cents).sum(dim=1)           # (B, 3)
        return unit_to_latlon(v)


def soft_targets(true_lat, true_lon, cell_lat, cell_lon, tau_km):
    """PIGEON-style haversine label smoothing: softmax(-d_geo(true, cell)/tau) over cells."""
    d = haversine_km_t(true_lat.unsqueeze(1), true_lon.unsqueeze(1),
                       cell_lat.unsqueeze(0), cell_lon.unsqueeze(0))   # (B, C)
    return F.softmax(-d / tau_km, dim=-1)

# ---------------------------------------------------------------------------
# Build model, optimizer, data
# ---------------------------------------------------------------------------

_, train_df = load_index("train")
print(f"Train pool: {len(train_df)} images | Backbone: {MODEL_NAME} @ {IMG_SIZE}px")
centroids, cell_lat, cell_lon = build_geocells(train_df, N_CELLS, KMEANS_ITERS)
print(f"Geocells: {N_CELLS} (k-means, {KMEANS_ITERS} iters)")
hier = [build_geocells(train_df, c, KMEANS_ITERS) for c in HIER_CELLS]
print(f"Hierarchy: {HIER_CELLS} + {N_CELLS} (log-space combined)")

model = GeoModel(centroids, cell_lat, cell_lon, hier).to(device)
model.trunk.to(torch.float32); model.fine_head.to(torch.float32)
model.coarse_heads.to(torch.float32)

lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
head_params = (list(model.trunk.parameters()) + list(model.fine_head.parameters())
               + list(model.coarse_heads.parameters()))
n_train_params = sum(p.numel() for p in lora_params + head_params)
print(f"Trainable params: {n_train_params/1e6:.2f}M (LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M "
      f"+ head {sum(p.numel() for p in head_params)/1e6:.2f}M)")

optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": LORA_LR},
    {"params": head_params, "lr": HEAD_LR},
], betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

train_ds = GeoDataset("train", IMG_SIZE)
train_loader = DataLoader(train_ds, batch_size=DEVICE_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True, collate_fn=collate,
                          pin_memory=True, persistent_workers=True, prefetch_factor=4)


def make_predict_fn(collect=None):
    """Closure passed to the frozen evaluator: PIL images -> (lat, lon) arrays."""
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


# True (nearest-centroid) cell of every val row, in evaluate_geo's row order — lets the eval
# pass double as a cell-classification probe at zero extra GPU cost.
_, _val_df = load_index("val")
with torch.no_grad():
    _vlat = torch.tensor(_val_df["latitude"].to_numpy(), dtype=torch.float32, device=device)
    _vlon = torch.tensor(_val_df["longitude"].to_numpy(), dtype=torch.float32, device=device)
    _d = haversine_km_t(_vlat.unsqueeze(1), _vlon.unsqueeze(1),
                        cell_lat.unsqueeze(0), cell_lon.unsqueeze(0))
    VAL_TRUE_CELLS = _d.argmin(dim=1).cpu()               # (N_val,)
    del _d


def val_cell_metrics(collected):
    """Top-1/5 cell accuracy + chance-normalized lift from collected top-5 ids."""
    if not collected:
        return {}
    top5 = torch.cat(collected)                            # (n, 5), eval row order
    true = VAL_TRUE_CELLS[:top5.size(0)]
    top1 = (top5[:, 0] == true).float().mean().item()
    top5a = (top5 == true.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"cell_top1": top1, "cell_top5": top5a,
            "cell_top1_lift": top1 * N_CELLS, "cell_top5_lift": top5a * N_CELLS / 5}


def quick_eval(tag, step):
    coll = []
    pf = make_predict_fn(collect=coll)
    m = evaluate_geo(pf, split="val", subset=QUICK_VAL_N, batch_size=DEVICE_BATCH_SIZE)
    cm = val_cell_metrics(coll)
    model.train()
    print(f"\n[{tag}] median_km={m['median_km']:.1f} mean_km={m['mean_km']:.1f} "
          f"acc@25km={m['acc_25km']*100:.1f}% acc@200km={m['acc_200km']*100:.1f}% "
          f"acc@2500km={m['acc_2500km']*100:.1f}% geoguessr={m['geoguessr_score']:.0f} "
          f"cell_top1={cm.get('cell_top1', 0)*100:.1f}% (lift {cm.get('cell_top1_lift', 0):.0f}x)",
          flush=True)
    if wandb_run is not None:
        wandb.log({f"val/{k}": v for k, v in {**m, **cm}.items()}, step=step)
    return m

# ---------------------------------------------------------------------------
# LR schedule (progress = training_time / TIME_BUDGET)
# ---------------------------------------------------------------------------

def lr_mult(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    p = (progress - WARMUP_RATIO) / max(1e-9, 1 - WARMUP_RATIO)
    cos = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * cos

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# --- W&B: record this experiment as a run in the shared project ---
try:
    import wandb
    _cfg = {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, tuple, list))}
    _cfg["n_train_params_M"] = round(n_train_params / 1e6, 2)
    wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    wandb_run = wandb.init(project=WANDB_PROJECT, config=_cfg, mode=wandb_mode,
                           name=os.environ.get("AR_RUN_NAME"))
    print(f"W&B: {wandb_mode} (project={WANDB_PROJECT})")
except Exception as e:
    print(f"W&B init failed ({e}); continuing without tracking")
    wandb, wandb_run = None, None

print(f"Time budget: {TIME_BUDGET}s | batch {DEVICE_BATCH_SIZE} x accum {GRAD_ACCUM}")
model.train()
step = 0
total_training_time = 0.0
smooth_loss = 0.0
data_iter = iter(train_loader)
epoch = 1

# torch.compile the backbone — the compile cost is paid HERE, before start_training_clock(),
# so the whole 480s wall-alarm window runs compiled steps. dynamic=True + multi-shape eval
# warmup below prevents recompile stalls inside the alarm (eval sees partial last batches).
model.encode = torch.compile(model.encode, dynamic=True)   # compile the layer stack; the
# eager pre-encode part (embeddings + rope + patch dropout) stays outside the graph
print("compile warmup (before the training clock)…", flush=True)
_t_c = time.time()
_wimgs, _, _ = next(data_iter)
_x = normalize_batch(_wimgs)
with autocast_ctx:
    _feats = model.features(_x)
_logits, _ = model.head_logits(_feats.float())
_logits.mean().backward()                       # compile the backward graph too
optimizer.zero_grad(set_to_none=True)
model.eval()
with torch.no_grad(), autocast_ctx:
    model.features(_x[:16]); model.features(_x[:7])   # eval-mode + varied shapes
model.train()
del _wimgs, _x, _feats, _logits
print(f"compile warmup done in {time.time()-_t_c:.0f}s", flush=True)

# Eval BEFORE any training (step 0) so every run has an untrained baseline point.
quick_eval("step 0", 0)

start_training_clock()
try:
    while True:
        torch.cuda.synchronize(); t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                imgs, blat, blon = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(train_loader)
                imgs, blat, blon = next(data_iter)
            x = normalize_batch(imgs)
            blat = blat.to(device); blon = blon.to(device)
            with autocast_ctx:
                feats = model.features(x)
            logits, coarse_logits = model.head_logits(feats.float())
            tgt = soft_targets(blat, blon, cell_lat, cell_lon, SMOOTH_TAU_KM)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(tgt * logp).sum(dim=-1).mean()
            for l, cl in enumerate(coarse_logits):
                tgt_l = soft_targets(blat, blon, getattr(model, f"coarse_lat_{l}"),
                                     getattr(model, f"coarse_lon_{l}"), SMOOTH_TAU_KM)
                loss = loss + HIER_LOSS_W[l] * -(tgt_l * F.log_softmax(cl, dim=-1)).sum(dim=-1).mean()
            loss = loss / GRAD_ACCUM
            loss.backward()
            loss_val += loss.item()

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = lr_mult(progress)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        optimizer.step()

        if math.isnan(loss_val) or loss_val > 1e4:
            print("FAIL"); raise SystemExit(1)

        torch.cuda.synchronize(); dt = time.time() - t0
        if step > 3:
            total_training_time += dt

        beta = 0.9
        smooth_loss = beta * smooth_loss + (1 - beta) * loss_val
        deb = smooth_loss / (1 - beta ** (step + 1))
        remaining = max(0, TIME_BUDGET - total_training_time)
        ips = DEVICE_BATCH_SIZE * GRAD_ACCUM / dt
        print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss {deb:.4f} | lrm {lrm:.2f} | "
              f"{dt*1000:.0f}ms | {ips:.0f} img/s | ep {epoch} | {remaining:.0f}s left    ",
              end="", flush=True)

        if wandb_run is not None and step % 20 == 0:
            with torch.no_grad():
                true_cell = tgt.argmax(dim=-1)
                top5 = logits.float().topk(5, dim=-1).indices
                cell_top1 = (top5[:, 0] == true_cell).float().mean().item()
                cell_top5 = (top5 == true_cell.unsqueeze(1)).any(dim=1).float().mean().item()
            wandb.log({"train/loss": deb, "train/lr_mult": lrm, "train/img_per_s": ips,
                       "train/progress": progress, "epoch": epoch,
                       "train/cell_top1": cell_top1, "train/cell_top5": cell_top5,
                       # chance-normalized lift (1.0 = random), comparable across N_CELLS
                       "train/cell_top1_lift": cell_top1 * N_CELLS,
                       "train/cell_top5_lift": cell_top5 * N_CELLS / 5}, step=step)

        if EVAL_EVERY and step > 0 and step % EVAL_EVERY == 0:
            quick_eval(f"step {step}", step)

        step += 1
        if step > 3 and total_training_time >= TIME_BUDGET:
            break
except TrainingTimeUp:
    print("\n[hard-deadline] training budget reached — stopping for final eval")
finally:
    stop_training_clock()

print()

# ---------------------------------------------------------------------------
# Final eval on the FULL val split (the official score)
# ---------------------------------------------------------------------------

model.eval()
final_coll = []
predict_fn = make_predict_fn(collect=final_coll)
with autocast_ctx:
    m = evaluate_geo(predict_fn, split="val", subset=None, batch_size=DEVICE_BATCH_SIZE)
final_cm = val_cell_metrics(final_coll)

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
t_end = time.time()

print("---")
print(f"median_km:        {m['median_km']:.6f}")   # <-- PRIMARY OBJECTIVE (lower is better)
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
print(f"num_params_M:     {n_train_params/1e6:.2f}")
for k, v in final_cm.items():
    print(f"{k}: {v:.6f}")

if wandb_run is not None:
    wandb.log({f"final/{k}": v for k, v in {**m, **final_cm}.items()}, step=step)
    wandb.summary.update({f"final_{k}": v for k, v in {**m, **final_cm}.items()})
    wandb.summary.update({"num_steps": step, "peak_vram_mb": peak_vram_mb})
    wandb.finish()
