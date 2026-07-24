#!/usr/bin/env python3
"""Build the coordinate-derived attribute label table (C5 Phase 0).

Free supervision for the attribute heads and the mean-campaign evidence channels:
every train/val coordinate yields country (offline reverse geocode), driving side
(static country table), Köppen climate class (kgcpy lookup, cached on a 0.25° grid),
hemisphere, and latitude band. No annotation, no network.

Usage:
  python tools/build_attribute_table.py --split val
  python tools/build_attribute_table.py --split train      # 1.2M points, a few minutes
Outputs data/attr_labels_{split}.npz
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Countries driving on the LEFT (ISO-2). Sources: standard traffic-side tables.
LEFT = {
    "GB", "IE", "MT", "CY", "IM", "JE", "GG", "GI",
    "AU", "NZ", "FJ", "PG", "SB", "TO", "WS", "KI", "TV", "CK", "NU",
    "JP", "TH", "ID", "MY", "SG", "BN", "TL", "MO", "HK",
    "IN", "PK", "BD", "LK", "NP", "BT", "MV",
    "ZA", "NA", "BW", "ZW", "ZM", "MW", "MZ", "TZ", "KE", "UG", "LS", "SZ",
    "MU", "SC", "SH",
    "KN", "LC", "VC", "GD", "DM", "AG", "BB", "BS", "JM", "TT", "KY", "VG", "MS", "AI", "BM",
    "GY", "SR", "FK",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--train-latlon", default="run_c4_index/train_latlon.npz")
    ap.add_argument("--val-meta", default="run_full/val_analysis/val_meta.npz")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    t0 = time.time()
    src = a.train_latlon if a.split == "train" else a.val_meta
    z = np.load(os.path.join(REPO, src))
    lat, lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    n = len(lat)
    print(f"{a.split}: {n} coordinates", flush=True)

    # country via offline reverse geocode (unique 0.05-degree grid keeps it fast)
    import reverse_geocoder as rg
    key = np.round(np.stack([lat, lon], 1) / 0.05).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    pts = [(k[0] * 0.05, k[1] * 0.05) for k in uniq]
    print(f"reverse-geocoding {len(uniq)} unique grid points…", flush=True)
    res = rg.search(pts, mode=1)
    cc_u = np.array([r["cc"] for r in res])
    cc = cc_u[inv]
    print(f"countries done ({time.time()-t0:.0f}s): "
          f"{len(np.unique(cc))} distinct, top: "
          f"{[f'{c}:{s}' for c, s in zip(*np.unique(cc, return_counts=True))][:1]}", flush=True)

    drive_left = np.isin(cc, sorted(LEFT))

    # Köppen class via kgcpy, cached on a 0.25-degree grid
    from kgcpy import lookupCZ
    key2 = np.round(np.stack([lat, lon], 1) / 0.25).astype(np.int64)
    uniq2, inv2 = np.unique(key2, axis=0, return_inverse=True)
    print(f"Köppen lookup for {len(uniq2)} unique grid points…", flush=True)
    kz_u = []
    for i, k in enumerate(uniq2):
        try:
            kz_u.append(lookupCZ(float(k[0] * 0.25), float(k[1] * 0.25)))
        except Exception:
            kz_u.append("??")
        if i and i % 20000 == 0:
            print(f"  {i}/{len(uniq2)} ({time.time()-t0:.0f}s)", flush=True)
    koppen = np.array(kz_u)[inv2]
    # major class = first letter (A tropics / B arid / C temperate / D continental / E polar)
    kmajor = np.array([k[0] if k and k[0] in "ABCDE" else "?" for k in koppen])

    hemi_n = lat >= 0.0
    lat_band = np.digitize(lat, np.arange(-75, 76, 15))

    os.makedirs(os.path.join(REPO, a.out_dir), exist_ok=True)
    out = os.path.join(REPO, a.out_dir, f"attr_labels_{a.split}.npz")
    np.savez_compressed(out, lat=lat, lon=lon, cc=cc, drive_left=drive_left,
                        koppen=koppen, koppen_major=kmajor, hemi_north=hemi_n,
                        lat_band=lat_band)
    print(f"\nwrote {out} ({time.time()-t0:.0f}s)")
    print(f"  drive_left: {100*drive_left.mean():.1f}%  | hemi north: {100*hemi_n.mean():.1f}%")
    ku, kc = np.unique(kmajor, return_counts=True)
    print(f"  Köppen major: {dict(zip(ku.tolist(), (100*kc/n).round(1).tolist()))}")
    cu, cn = np.unique(cc, return_counts=True)
    topc = np.argsort(-cn)[:10]
    print(f"  top countries: {[(cu[i], int(cn[i])) for i in topc]}")
    print("ATTR TABLE DONE", flush=True)


if __name__ == "__main__":
    main()
