#!/usr/bin/env python3
"""
Author: Wesley Ducharme
Date: 2025-11-23

Make a 3D MoDMap for HIV-1 LANL whole genomes (pure subtypes only).

- Dataset: hiv1_lanl_whole
- k-mer length: default k = 6
- Distance: Manhattan in k-mer frequency space
- Embedding: 3D metric MDS
- Labels kept: pure HIV-1 subtypes
    A1, A6, B, C, D, F1, G, O, U

Outputs:
  results/modmap_hiv1_lanl_whole__k6__pure.tsv
    columns: id, label, x, y, z
"""

import os
import csv
import argparse

import numpy as np

from kameris_reimp import datasets, modmap

# Pure HIV-1 subtypes used in Figure 2 of the paper
PURE_SUBTYPES = {"A1", "A6", "B", "C", "D", "F1", "G", "O", "U"}


def main():
    ap = argparse.ArgumentParser(
        description="Build 3D MoDMap for hiv1_lanl_whole (pure subtypes only)."
    )
    ap.add_argument(
        "--dataset",
        default="hiv1_lanl_whole",
        choices=datasets.list_dataset_keys(),
        help="Dataset key (default: hiv1_lanl_whole).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=6,
        help="k-mer length (default: 6).",
    )
    ap.add_argument(
        "--metric",
        default="manhattan",
        choices=["manhattan", "euclidean"],
        help="Distance metric in k-mer space (default: manhattan).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MDS (default: 42).",
    )
    ap.add_argument(
        "--out-tsv",
        default=None,
        help="Output TSV path (default: data/modmap_<dataset>__k<k>__pure.tsv).",
    )
    args = ap.parse_args()

    dataset_key = args.dataset
    k = args.k

    # Load dataset to (optionally) get IDs
    ds = datasets.load_dataset(dataset_key)

    # Build MoDMap using the helper in kameris_repo.modmap
    coords, labels_kept, indices_kept = modmap.modmap_for_dataset(
        dataset_key=dataset_key,
        k=k,
        metric=args.metric,
        n_components=3,
        label_filter=lambda lab: lab in PURE_SUBTYPES,
    )

    coords = np.asarray(coords)
    n_points, n_dims = coords.shape
    assert n_dims == 3, f"Expected 3D embedding, got {n_dims}D"

    # Derive IDs if available, otherwise just use row indices
    ids = getattr(ds, "ids", None)
    if ids is not None:
        id_list = [ids[i] for i in indices_kept]
    else:
        # Fallback: 0..N-1 as IDs
        id_list = [str(i) for i in indices_kept]

    # Determine output path
    if args.out_tsv is None:
        os.makedirs("results", exist_ok=True)
        out_tsv = os.path.join(
            "results",
            f"modmap_{dataset_key}__k{k}__pure.tsv",
        )
    else:
        out_tsv = args.out_tsv
        os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)

    # Write TSV: id, label, x, y, z
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "label", "x", "y", "z"])
        for sid, lab, (x, y, z) in zip(id_list, labels_kept, coords):
            w.writerow([sid, lab, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    # Small summary
    from collections import Counter
    counts = Counter(labels_kept)

    print(f"[OK] MoDMap for {dataset_key}")
    print(f"  k          = {k}")
    print(f"  metric     = {args.metric}")
    print(f"  n_points   = {n_points}")
    print(f"  labels     = {sorted(counts.keys())}")
    print(f"  per-label counts:")
    for lab in sorted(counts.keys()):
        print(f"    {lab}: {counts[lab]}")
    print(f"  wrote: {out_tsv}")


if __name__ == "__main__":
    main()
