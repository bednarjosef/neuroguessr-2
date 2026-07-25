#!/usr/bin/env python3
"""Generate train_c5.py from train_full.py via anchored surgical patches.

C5 additions (docs/C5_PLAN.md): MLP LoRA, attribute heads (free labels), graded listwise
episode loss on geography-mined pools with climate-matched cross-continent negatives,
resolution-verified retrieval probe, per-component W&B logging. Interleaved CE/episode
batches; AR_EPOCHS=1 => one full CE pass + equal episode views (~2.4M views total).
"""
import re

SRC = "/home/josef/everything/coding/neuroguessr-2-research/train_full.py"
DST = "/home/josef/everything/coding/neuroguessr-2-research/train_c5.py"
s = open(SRC).read()
n0 = len(s)


def rep(old, new, count=1):
    global s
    assert s.count(old) >= count, f"ANCHOR MISSING: {old[:80]!r}"
    s = s.replace(old, new, count)


# ---- config changes -------------------------------------------------------------------
rep('LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]',
    'LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]')
rep('PATCH_KEEP = 0.6', 'PATCH_KEEP = float(os.environ.get("AR_PATCH_KEEP", "0.6"))')
rep('DEVICE_BATCH_SIZE = int(os.environ.get("AR_BS", "84"))',
    'DEVICE_BATCH_SIZE = int(os.environ.get("AR_BS", "64"))  # MLP-LoRA activations need headroom')
rep('EPOCHS = int(os.environ.get("AR_EPOCHS", "3"))',
    'EPOCHS = int(os.environ.get("AR_EPOCHS", "1"))  # 1 epoch = full CE pass + equal episodes')
rep('''RUN_DIR = os.environ.get("AR_RUN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "run_full"))''',
    '''RUN_DIR = os.environ.get("AR_RUN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "run_c5"))
# C5 knobs
EP_E = int(os.environ.get("AR_EP_E", "4"))            # episodes per batch
EP_M = DEVICE_BATCH_SIZE // EP_E                       # members per episode (16 @ bs64)
EP_TAU = float(os.environ.get("AR_EP_TAU", "0.07"))
EP_TAU_GEO = float(os.environ.get("AR_EP_TAU_GEO", "100.0"))
EP_W = float(os.environ.get("AR_EP_W", "0.7"))
EP_RAMP = int(os.environ.get("AR_EP_RAMP", "500"))
ATTR_W = float(os.environ.get("AR_ATTR_W", "0.15"))
PROBE_EVERY = int(os.environ.get("AR_PROBE_EVERY", "500"))
PROBE_POOL_LOCS = 6000
PROBE_QUERIES = 800''')

# ---- GeoModel: attribute heads --------------------------------------------------------
rep('''        self.country_head = nn.Linear(HEAD_HIDDEN, n_country) if n_country else None''',
    '''        self.country_head = nn.Linear(HEAD_HIDDEN, n_country) if n_country else None
        # C5 attribute heads (free coordinate-derived labels): drive side / Köppen / lat band
        self.attr_heads = nn.ModuleDict({
            "drive": nn.Linear(HEAD_HIDDEN, 2),
            "kop": nn.Linear(HEAD_HIDDEN, 6),
            "band": nn.Linear(HEAD_HIDDEN, 12),
        })''')
rep('''if model.fine_head_b is not None:
    model.fine_head_b.to(torch.float32)''',
    '''if model.fine_head_b is not None:
    model.fine_head_b.to(torch.float32)
model.attr_heads.to(torch.float32)''')
rep('''               + (list(model.fine_head_b.parameters()) if model.fine_head_b is not None else []))''',
    '''               + (list(model.fine_head_b.parameters()) if model.fine_head_b is not None else [])
               + list(model.attr_heads.parameters()))''')

# ---- dataset returns its global index (attr lookups) ----------------------------------
rep('''        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
        return arr, self.lat[i], self.lon[i], self.country[i]''',
    '''        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
        return arr, self.lat[i], self.lon[i], self.country[i], i''')
rep('''def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    lat = torch.tensor([b[1] for b in batch])
    lon = torch.tensor([b[2] for b in batch])
    ctry = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return imgs, lat, lon, ctry''',
    '''def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    lat = torch.tensor([b[1] for b in batch])
    lon = torch.tensor([b[2] for b in batch])
    ctry = torch.tensor([b[3] for b in batch], dtype=torch.long)
    idx = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return imgs, lat, lon, ctry, idx''')

# ---- episode + attribute + probe infrastructure (inserted after the country block) ----
rep('''p0(f"Countries: {N_COUNTRY} (majority-vote parents for {N_CELLS} cells)")''',
    '''p0(f"Countries: {N_COUNTRY} (majority-vote parents for {N_CELLS} cells)")

# ---------------------------------------------------------------------------
# C5: attribute labels, location tables, episode sampler, retrieval probe
# ---------------------------------------------------------------------------

_attr = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "attr_labels_train.npz"), allow_pickle=True)
assert len(_attr["lat"]) == N_TRAIN_IMGS, "attr table rows != train rows (order mismatch?)"
assert np.abs(_attr["lat"] - train_df["latitude"].to_numpy()).max() < 1e-6, \\
    "attr table coordinate mismatch — rebuild tools/build_attribute_table.py"
KOP_CLASSES = ["A", "B", "C", "D", "E", "?"]
_k2i = {c: i for i, c in enumerate(KOP_CLASSES)}
KOP_T = torch.tensor([_k2i.get(str(k), 5) for k in _attr["koppen_major"]],
                     dtype=torch.long, device=device)
BAND_T = torch.tensor(np.clip(_attr["lat_band"].astype(np.int64), 0, 11), device=device)
_drive_img = torch.tensor(_attr["drive_left"].astype(np.int64), device=device)
# drive side per country vocab index (majority vote of that country's images)
DRIVE_OF_COUNTRY = torch.zeros(N_COUNTRY, dtype=torch.long, device=device)
_ctry_img = torch.tensor(_cc.map(C2I).to_numpy(dtype=np.int64), device=device)
for _ci in range(N_COUNTRY):
    _m = _ctry_img == _ci
    if _m.any():
        DRIVE_OF_COUNTRY[_ci] = (_drive_img[_m].float().mean() > 0.5).long()

_lat_np = train_df["latitude"].to_numpy()
_lon_np = train_df["longitude"].to_numpy()
_locu, LOC_OF_IMG = np.unique(np.round(np.stack([_lat_np, _lon_np], 1), 6),
                              axis=0, return_inverse=True)
N_LOCS = len(_locu)
_order = np.argsort(LOC_OF_IMG, kind="stable")
_starts = np.searchsorted(LOC_OF_IMG[_order], np.arange(N_LOCS))
_ends = np.searchsorted(LOC_OF_IMG[_order], np.arange(N_LOCS), side="right")
IMGS_OF_LOC = [_order[_starts[i]:_ends[i]] for i in range(N_LOCS)]
_loc_pts = latlon_to_unit(torch.tensor(_locu[:, 0], dtype=torch.float32, device=device),
                          torch.tensor(_locu[:, 1], dtype=torch.float32, device=device))
LOC_CELL = assign_cells(_loc_pts, centroids).cpu().numpy()
_cd = haversine_km_t(cell_lat.unsqueeze(1), cell_lon.unsqueeze(1),
                     cell_lat.unsqueeze(0), cell_lon.unsqueeze(0))
CELL_KM = _cd.cpu().numpy()
del _cd
LOCS_OF_CELL = [np.where(LOC_CELL == c)[0] for c in range(N_CELLS)]
KOP_OF_LOC = KOP_T.cpu().numpy()[np.array([IMGS_OF_LOC[i][0] for i in range(N_LOCS)])]
LOC_LAT, LOC_LON = _locu[:, 0], _locu[:, 1]
BY_KOP = {k: np.where(KOP_OF_LOC == k)[0] for k in range(6)}
p0(f"C5 episode tables: {N_LOCS} locations | Köppen pools "
   f"{[len(BY_KOP[k]) for k in range(6)]} | EP {EP_E}x{EP_M} tau_geo {EP_TAU_GEO}")


def _hav_np(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class EpisodeSampler(torch.utils.data.Sampler):
    """Batches of EP_E episodes x EP_M members: anchor + graded distance bands + global +
    climate-matched cross-continent negatives (same Köppen major, >3000 km away)."""

    BANDS = [((0.0, 25.0), 4), ((25.0, 100.0), 3), ((100.0, 500.0), 2)]

    def __init__(self, n_batches, seed):
        self.n_batches = n_batches
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_batches

    def _sample_loc_in_band(self, ca, lo, hi):
        cells = np.where((CELL_KM[ca] >= lo) & (CELL_KM[ca] < hi))[0]
        for _ in range(4):
            if len(cells) == 0:
                break
            c = int(self.rng.choice(cells))
            locs = LOCS_OF_CELL[c]
            if len(locs):
                return int(self.rng.choice(locs))
        return int(self.rng.integers(N_LOCS))

    def _img_of(self, loc):
        return int(self.rng.choice(IMGS_OF_LOC[loc]))

    def __iter__(self):
        for _ in range(self.n_batches):
            flat = []
            for _e in range(EP_E):
                a = int(self.rng.integers(N_LOCS))
                ca = int(LOC_CELL[a])
                members = [self._img_of(a)]
                if len(IMGS_OF_LOC[a]) > 1:            # second view of the anchor location
                    members.append(self._img_of(a))
                for (lo, hi), cnt in self.BANDS:
                    for _ in range(cnt):
                        members.append(self._img_of(self._sample_loc_in_band(ca, lo, hi)))
                # climate-matched cross-continent negatives
                kop = int(KOP_OF_LOC[a])
                pool = BY_KOP.get(kop)
                got = 0
                for _ in range(10):
                    if pool is None or len(pool) == 0 or got >= 2:
                        break
                    cand = int(self.rng.choice(pool))
                    if _hav_np(LOC_LAT[a], LOC_LON[a], LOC_LAT[cand], LOC_LON[cand]) > 3000.0:
                        members.append(self._img_of(cand))
                        got += 1
                while len(members) < EP_M:             # fill with global randoms
                    members.append(self._img_of(int(self.rng.integers(N_LOCS))))
                flat.extend(members[:EP_M])
            yield flat


def make_episode_loader(epoch):
    return DataLoader(train_ds, batch_sampler=EpisodeSampler(STEPS_PER_EPOCH,
                                                             DATA_ORDER_SEED * 7 + epoch * 131 + RANK),
                      num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True,
                      prefetch_factor=4)''')

# ---- probe (inserted after the val plumbing / quantization floor block) ---------------
rep('''VAL_TRUE_COUNTRY = torch.tensor(_val_df["country_code"].fillna("??").astype(str)''',
    '''# ---------------------------------------------------------------------------
# C5 retrieval probe — resolution-verified (ceiling 100% @25km BY CONSTRUCTION)
# ---------------------------------------------------------------------------
_prng = np.random.default_rng(1234)
_pool_locs = _prng.choice(N_LOCS, size=min(PROBE_POOL_LOCS, N_LOCS), replace=False)
_pool_set = set(_pool_locs.tolist())
_pl = LOC_LAT[_pool_locs]; _po = LOC_LON[_pool_locs]
_cand_q = _prng.permutation(N_LOCS)
PROBE_Q, PROBE_Q_NN = [], []
for _l in _cand_q:
    if int(_l) in _pool_set or len(PROBE_Q) >= PROBE_QUERIES:
        if len(PROBE_Q) >= PROBE_QUERIES:
            break
        continue
    _d = _hav_np(LOC_LAT[_l], LOC_LON[_l], _pl, _po)
    if _d.min() <= 25.0:
        PROBE_Q.append(int(_l)); PROBE_Q_NN.append(float(_d.min()))
PROBE_Q = np.array(PROBE_Q)
_pool_imgs = np.array([IMGS_OF_LOC[l][k] for l in _pool_locs
                       for k in range(min(2, len(IMGS_OF_LOC[l])))])
_pool_of_img = np.array([j for j, l in enumerate(_pool_locs)
                         for _ in range(min(2, len(IMGS_OF_LOC[l])))])
_q_imgs = np.array([IMGS_OF_LOC[l][0] for l in PROBE_Q])
_pd = _hav_np(LOC_LAT[PROBE_Q][:, None], LOC_LON[PROBE_Q][:, None],
              _pl[_pool_of_img][None, :], _po[_pool_of_img][None, :])
PROBE_OK25 = torch.tensor(_pd <= 25.0)
PROBE_OK10 = torch.tensor(_pd <= 10.0)
_ceil25 = float(PROBE_OK25.any(1).float().mean())
p0(f"Probe: {len(PROBE_Q)} queries x {len(_pool_imgs)} pool imgs | ceiling @25km "
   f"{100*_ceil25:.0f}% (must be ~100), @10km {100*float(PROBE_OK10.any(1).float().mean()):.0f}%")
assert _ceil25 > 0.99, "probe has no resolution — refuse to run blind (C3/C4 lesson)"
_PROBE_CACHE = {}


def run_probe(step):
    r25 = r10 = 0.0
    if is_main:
        t0p = time.time()
        model.eval()
        if "pool" not in _PROBE_CACHE:
            from PIL import Image
            def _load(rows):
                out = torch.empty(len(rows), 3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
                for j, r in enumerate(rows):
                    im = open_image(train_ds.img_dir, train_ds.paths[r])
                    if im.size != (IMG_SIZE, IMG_SIZE):
                        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                    out[j] = torch.from_numpy(np.asarray(im, np.uint8)).permute(2, 0, 1)
                return out
            _PROBE_CACHE["pool"] = _load(_pool_imgs)
            _PROBE_CACHE["q"] = _load(_q_imgs)
            p0(f"[probe] image cache built ({time.time()-t0p:.0f}s)")
        def _embed(t):
            zs = []
            with torch.no_grad(), autocast_ctx:
                for i in range(0, len(t), 96):
                    zs.append(F.normalize(model.features(
                        normalize_batch(t[i:i + 96])).float(), dim=-1))
            return torch.cat(zs)
        zq = _embed(_PROBE_CACHE["q"]); zp = _embed(_PROBE_CACHE["pool"])
        best = (zq @ zp.T).argmax(1).cpu()
        ar_ = torch.arange(len(_q_imgs))
        r25 = float(PROBE_OK25[ar_, best].float().mean())
        r10 = float(PROBE_OK10[ar_, best].float().mean())
        print(f"\\n[probe @ {step}] near-recall@1: @25km {100*r25:.1f}% | @10km {100*r10:.1f}% "
              f"({time.time()-t0p:.0f}s)", flush=True)
        if wandb_run is not None:
            wandb.log({"probe/near_recall25": r25, "probe/near_recall10": r10}, step=step)
        model.train()
    if DIST:
        dist.barrier()
    return r25


VAL_TRUE_COUNTRY = torch.tensor(_val_df["country_code"].fillna("??").astype(str)''')

# ---- training loop: interleave, per-component losses, attr loss, probe ----------------
rep('''        loader = make_loader(epoch, in_epoch)
        for imgs, blat, blon, bctry in loader:
            optimizer.zero_grad(set_to_none=True)
            x = normalize_batch(imgs)
            blat = blat.to(device); blon = blon.to(device); bctry = bctry.to(device)''',
    '''        loader = make_loader(epoch, in_epoch)
        ep_loader = make_episode_loader(epoch)

        def _interleave(a, b):
            ita, itb = iter(a), iter(b)
            while True:
                try:
                    yield next(ita), False
                    yield next(itb), True
                except StopIteration:
                    return

        for (imgs, blat, blon, bctry, bidx), IS_EP in _interleave(loader, ep_loader):
            optimizer.zero_grad(set_to_none=True)
            x = normalize_batch(imgs)
            blat = blat.to(device); blon = blon.to(device); bctry = bctry.to(device)
            bidx = bidx.to(device)''')

rep('''            if MSL_W > 0:''', '''            parts = {"ce": float(loss.item())}
            if MSL_W > 0:''')

rep('''                gm = d_msl.pow(2) / (d_msl.pow(2) + MSL_S_KM ** 2)
                loss = loss + MSL_W * gm.mean()
            loss.backward()''',
    '''                gm = d_msl.pow(2) / (d_msl.pow(2) + MSL_S_KM ** 2)
                loss = loss + MSL_W * gm.mean()
                parts["msl"] = float(gm.mean().item())
            # C5: graded listwise episode loss (episode batches only)
            if IS_EP:
                fe = F.normalize(feats.float(), dim=-1).view(EP_E, EP_M, -1)
                el = blat.view(EP_E, EP_M); eo = blon.view(EP_E, EP_M)
                dkm = haversine_km_t(el.unsqueeze(2), eo.unsqueeze(2),
                                     el.unsqueeze(1), eo.unsqueeze(1))
                tgt_e = torch.exp(-dkm / EP_TAU_GEO)
                eye = torch.eye(EP_M, device=device, dtype=torch.bool).unsqueeze(0)
                tgt_e = tgt_e.masked_fill(eye, 0.0)
                tgt_e = tgt_e / tgt_e.sum(-1, keepdim=True).clamp(min=1e-9)
                sims_e = (fe @ fe.transpose(1, 2)) / EP_TAU
                sims_e = sims_e.masked_fill(eye, -1e4)
                loss_ep = -(tgt_e * F.log_softmax(sims_e, dim=-1)).sum(-1).mean()
                ep_floor = -(tgt_e * tgt_e.clamp(min=1e-12).log()).sum(-1).mean()
                epw = EP_W * min(1.0, (step + 1) / max(1, EP_RAMP))
                loss = loss + epw * loss_ep
                parts["ep"] = float(loss_ep.item())
                parts["ep_kl"] = float((loss_ep - ep_floor).item())
            # C5: attribute CE (every batch; labels are free)
            h_attr = model.trunk(feats.float())
            att = model.attr_heads
            la_drive = DRIVE_OF_COUNTRY[bctry]
            loss_attr = (F.cross_entropy(att["drive"](h_attr), la_drive)
                         + F.cross_entropy(att["kop"](h_attr), KOP_T[bidx])
                         + F.cross_entropy(att["band"](h_attr), BAND_T[bidx])) / 3
            loss = loss + ATTR_W * loss_attr
            parts["attr"] = float(loss_attr.item())
            loss.backward()''')

rep('''                if wandb_run is not None:
                    wandb.log({"train/loss": deb, "train/lr_mult": lrm, "train/img_per_s": ips,
                               "train/progress": step / TOTAL_STEPS, "epoch": epoch + 1,
                               "train/samples_seen": step * GLOBAL_BS}, step=step)''',
    '''                if wandb_run is not None:
                    with torch.no_grad():
                        gn = torch.norm(torch.stack([p.grad.norm() for p in trainable_params
                                                     if p.grad is not None])).item()
                        acc_d = (att["drive"](h_attr).argmax(1) == la_drive).float().mean().item()
                        acc_k = (att["kop"](h_attr).argmax(1) == KOP_T[bidx]).float().mean().item()
                    wandb.log({"train/loss": deb, "train/lr_mult": lrm, "train/img_per_s": ips,
                               "train/progress": step / TOTAL_STEPS, "epoch": epoch + 1,
                               "train/samples_seen": step * GLOBAL_BS,
                               "train/grad_norm": gn, "train/is_episode": float(IS_EP),
                               "attr/acc_drive": acc_d, "attr/acc_koppen": acc_k,
                               **{f"loss/{k}": v for k, v in parts.items()},
                               "train/vram_gb": torch.cuda.max_memory_allocated() / 2**30},
                              step=step)''')

rep('''            if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:''',
    '''            if step % PROBE_EVERY == 0 or step == TOTAL_STEPS:
                run_probe(step)
            if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:''')

# steps-per-epoch doubles with interleave
rep('''STEPS_PER_EPOCH = (N_TRAIN_IMGS // WORLD) // DEVICE_BATCH_SIZE
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH''',
    '''STEPS_PER_EPOCH = (N_TRAIN_IMGS // WORLD) // DEVICE_BATCH_SIZE
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH * 2   # interleaved: CE pass + equal episode batches''')

# ---- checkpoints: c5 config + HF subfolder --------------------------------------------
rep('''                   "stagger": STAGGER, "cc_cells": CC_CELLS, "msl_w": MSL_W, "aug": AUG},''',
    '''                   "stagger": STAGGER, "cc_cells": CC_CELLS, "msl_w": MSL_W, "aug": AUG,
                   "c5": True, "lora_targets": LORA_TARGETS, "ep_w": EP_W,
                   "ep_tau_geo": EP_TAU_GEO, "attr_w": ATTR_W, "kop_classes": KOP_CLASSES},''')
rep('''            api.upload_file(path_or_fileobj=path, path_in_repo=f"ckpt_{tag}.pt",
                            repo_id=CKPT_HF_REPO, repo_type="model")''',
    '''            api.upload_file(path_or_fileobj=path, path_in_repo=f"c5/ckpt_{tag}.pt",
                            repo_id=CKPT_HF_REPO, repo_type="model")''')
rep('''                    or (f"fullrun-{N_CELLS}c-{EPOCHS}ep-w{WORLD}"''',
    '''                    or (f"c5-bed-{N_CELLS}c-{EPOCHS}ep-w{WORLD}"''')
# probe at step 0 too (baseline point)
rep('''if start_step == 0:
    quick_eval("step 0", 0)''',
    '''if start_step == 0:
    quick_eval("step 0", 0)
    run_probe(0)''')

open(DST, "w").write(s)
print(f"train_c5.py written: {n0} -> {len(s)} chars (+{len(s)-n0})")
