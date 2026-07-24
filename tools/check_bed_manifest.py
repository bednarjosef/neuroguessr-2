#!/usr/bin/env python3
"""C5 bed manifest: the box may not be destroyed until every artifact exists with a sane
size. A cheap-experiment idea that can't run on this list is a bed bug -> add a line here."""
import argparse
import os
import sys

EXPECTED = [
    # (pattern with {tag}, min_bytes, count)
    ("c2_cls_train_{tag}_r{r}.npy", 400_000_000, 4),
    ("c2_mean_train_{tag}_r{r}.npy", 400_000_000, 4),
    ("c2_reg_train_{tag}_r{r}.npy", 400_000_000, 4),
    ("c2_cls_val_{tag}_r{r}.npy", 1_000_000, 4),
    ("c2_mean_val_{tag}_r{r}.npy", 1_000_000, 4),
    ("c2_reg_val_{tag}_r{r}.npy", 1_000_000, 4),
    ("c2_logits_val_{tag}_r{r}.npy", 5_000_000, 4),
    ("c2_gateids_train_{tag}_r{r}.npy", 20_000_000, 4),
    ("c2_gatep_train_{tag}_r{r}.npy", 20_000_000, 4),
    ("c5_ep_ids_r{r}.npy", 40_000_000, 4),
    ("c5_ep_sims_r{r}.npy", 20_000_000, 4),
    ("geo5_cls_{tag}.pt", 10_000_000, 1),
    ("geo10_cls_{tag}.pt", 10_000_000, 1),
    ("geo25_cls_{tag}.pt", 10_000_000, 1),
    ("c2_regproj.npz", 100_000, 1),
    ("centroids.npz", 40_000, 1),
    ("train_latlon.npz", 10_000_000, 1),
    ("val_meta.npz", 20_000, 1),
    ("val_patches_{tag}.npy", 3_000_000_000, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="retrieval_index")
    ap.add_argument("--tag", default="c5")
    a = ap.parse_args()
    missing = []
    for pat, min_b, cnt in EXPECTED:
        for r in range(cnt):
            f = pat.format(tag=a.tag, r=r)
            p = os.path.join(a.index, f)
            if not os.path.exists(p):
                missing.append(f"{f} (absent)")
            elif os.path.getsize(p) < min_b:
                missing.append(f"{f} ({os.path.getsize(p)} < {min_b} B)")
    also = [f for f in ("run_c5/ckpt_best.pt",) if not os.path.exists(f)]
    missing += [f"{f} (absent)" for f in also]
    if missing:
        print("MANIFEST MISSING:")
        for m in missing:
            print("  -", m)
        sys.exit(1)
    print(f"MANIFEST OK ({sum(c for _, _, c in EXPECTED) + 1} artifacts)")


if __name__ == "__main__":
    main()
