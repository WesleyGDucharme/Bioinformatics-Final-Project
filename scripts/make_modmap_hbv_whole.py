#!/usr/bin/env python3
"""
scripts/make_modmap_hbv_whole.py

Author: Wesley Ducharme
Date: 2025-11-23

Purpose:
    Prepare a MoDMap-style 3D embedding for hepatitis B whole genomes.

    - Dataset: hbv_whole
    - k-mer length: k = 6
    - Keep ONLY pure genotypes: A, B, C, D, E, F, G, H
      (all other labels are completely discarded)
    - Embedding: classical MDS on Manhattan k-mer distances (3D)

Output:
    data/modmap_hbv_whole__k6.tsv with columns:
        sample_id, subtype, x, y, z
"""

import os
import numpy as np

from kameris_reimp import datasets, modmap

DATASET_KEY = "hbv_whole"
K = 6
OUT_TSV = os.path.join("data", "modmap_hbv_whole__k6.tsv")

PURE_LABELS = {"A", "B", "C", "D", "E", "F", "G", "H"}


def main():
    os.makedirs("data", exist_ok=True)

    # Load dataset
    ds = datasets.load_dataset(DATASET_KEY)
    seqs = np.array(ds.sequences)
    labels = np.array(ds.labels)

    # Keep ONLY genotypes A–H
    mask = np.isin(labels, list(PURE_LABELS))
    seqs_sub = seqs[mask]
    labels_sub = labels[mask]

    print(f"[INFO] Total sequences: {len(seqs)}")
    print(f"[INFO] Kept pure A–H genotypes: {len(seqs_sub)}")

    # 3D MoDMap via Manhattan distance + classical MDS
    print("[INFO] Computing MoDMap (Manhattan distance + classical MDS)...")
    coords, _D = modmap.kmer_modmap_from_sequences(
        seqs_sub,
        k=K,
        metric="manhattan",
        n_components=3,
        normalize=True,
    )

    # Write TSV
    print(f"[INFO] Writing TSV to {OUT_TSV}")
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("sample_id\tsubtype\tx\ty\tz\n")
        for i, (lab, (x, y, z)) in enumerate(zip(labels_sub, coords)):
            f.write(f"{i}\t{lab}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
