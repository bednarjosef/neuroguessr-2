#!/usr/bin/env python3
"""Two-stage retrieval engine: classifier posterior gates a candidate region, embedding
kNN against the training index refines the guess to (potentially) street level.

Design (measured 2026-07-23 on run #1):
- Search region = the smallest cell set holding `mass` of the HONEST (T=1) posterior over
  tessellation A — adapts to per-image uncertainty (median ~42 cells, ~92% recall@100km).
- Within the region: cosine kNN over train embeddings; guess = sim-softmax-weighted
  spherical mean of neighbor locations.
- Confidence guard: if the best similarity is below `smax_floor`, keep the classifier's
  own point (unseen-area fallback) — retrieval never has to be worse than the classifier.

All arrays stay torch tensors on `device`; queries loop (2998 val queries -> seconds).
"""
import numpy as np
import torch
import torch.nn.functional as F

EARTH_RADIUS_KM = 6371.0088


def latlon_to_unit(lat, lon):
    lat = torch.deg2rad(lat)
    lon = torch.deg2rad(lon)
    return torch.stack([torch.cos(lat) * torch.cos(lon),
                        torch.cos(lat) * torch.sin(lon),
                        torch.sin(lat)], dim=-1)


def unit_to_latlon(v):
    v = F.normalize(v, dim=-1)
    return (torch.rad2deg(torch.asin(v[..., 2].clamp(-1, 1))),
            torch.rad2deg(torch.atan2(v[..., 1], v[..., 0])))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))
    a = (torch.sin((lat2 - lat1) / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))


class RetrievalIndex:
    """Train-side database: L2-normalized embeddings + locations + A-cell ids, cell-bucketed."""

    def __init__(self, emb, lat, lon, cell, n_cells, device="cuda"):
        self.device = torch.device(device)
        emb = torch.as_tensor(np.ascontiguousarray(emb)).to(self.device, torch.float16)
        self.emb = F.normalize(emb.float(), dim=-1).half()
        self.lat = torch.as_tensor(lat, dtype=torch.float32, device=self.device)
        self.lon = torch.as_tensor(lon, dtype=torch.float32, device=self.device)
        self.unit = latlon_to_unit(self.lat, self.lon)
        cell = torch.as_tensor(cell, dtype=torch.long, device=self.device)
        # bucket rows by cell for fast candidate gathering
        order = torch.argsort(cell)
        self.rows_by_cell = order
        counts = torch.bincount(cell, minlength=n_cells)
        self.cell_start = torch.zeros(n_cells + 1, dtype=torch.long, device=self.device)
        self.cell_start[1:] = counts.cumsum(0)

    def candidates(self, cells):
        """Row indices of all train images whose A-cell is in `cells` (LongTensor)."""
        segs = [self.rows_by_cell[self.cell_start[c]:self.cell_start[c + 1]] for c in cells]
        return torch.cat(segs) if segs else torch.empty(0, dtype=torch.long, device=self.device)


class RetrievalEngine:
    def __init__(self, index: RetrievalIndex, mass=0.9, k=16, sim_temp=0.07,
                 smax_floor=0.0, max_cells=400):
        self.ix = index
        self.mass = mass
        self.k = k
        self.sim_temp = sim_temp
        self.smax_floor = smax_floor
        self.max_cells = max_cells

    def mass_set(self, logits_a):
        """Smallest top-cell set covering `mass` of the honest posterior (capped)."""
        p = F.softmax(logits_a.float(), dim=-1)
        w, idx = p.sort(descending=True)
        m = int((w.cumsum(0) < self.mass).sum().item()) + 1
        return idx[:min(m, self.max_cells)]

    def predict_one(self, logits_a, emb, fallback_latlon):
        cells = self.mass_set(logits_a)
        cand = self.ix.candidates(cells)
        if cand.numel() == 0:
            return fallback_latlon, 0.0, 0
        q = F.normalize(emb.float(), dim=-1).half()
        sims = (self.ix.emb[cand] @ q).float()
        k = min(self.k, sims.numel())
        w, top = sims.topk(k)
        smax = float(w[0])
        if smax < self.smax_floor:
            return fallback_latlon, smax, int(cand.numel())
        ww = F.softmax(w / self.sim_temp, dim=0)
        v = (ww.unsqueeze(-1) * self.ix.unit[cand[top]]).sum(0, keepdim=True)
        plat, plon = unit_to_latlon(v)
        return (float(plat[0]), float(plon[0])), smax, int(cand.numel())

    def predict_batch(self, logits_a, embs, fallbacks):
        """logits_a (n, C), embs (n, D), fallbacks list[(lat, lon)] -> arrays + diagnostics."""
        out = np.zeros((len(fallbacks), 2), np.float64)
        smaxs = np.zeros(len(fallbacks))
        pools = np.zeros(len(fallbacks), np.int64)
        for i in range(len(fallbacks)):
            (la, lo), sm, np_ = self.predict_one(logits_a[i], embs[i], fallbacks[i])
            out[i] = (la, lo)
            smaxs[i] = sm
            pools[i] = np_
        return out, smaxs, pools
