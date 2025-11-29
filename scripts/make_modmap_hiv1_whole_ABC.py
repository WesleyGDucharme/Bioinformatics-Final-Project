#!/usr/bin/env python3
"""
scripts/make_modmap_hiv1_whole_ABC.py

Author: Wesley Ducharme
Date: 2025-11-23

Purpose:
    Prepare a MoDMap-style 3D embedding for HIV-1 LANL whole-genome
    sequences, restricted to (pure) subtypes A, B, and C.

    - Uses dataset: hiv1_lanl_whole
    - k-mer length: k = 6
    - Subtype mapping:
        A1, A6 -> "A"
        B      -> "B"
        C      -> "C"
    - Embedding: classical MDS on Manhattan k-mer distances (3D)

Output:
    results/modmap_hiv1_ABC_k6.tsv with columns:
        sample_id, group, x, y, z
"""

import os
import numpy as np

from kameris_reimp import datasets, modmap

DATASET_KEY = "hiv1_lanl_whole"
K = 6
OUT_TSV = os.path.join("data", "modmap_hiv1_ABC_k6.tsv")

PURE_A = {"A1", "A6"}
PURE_B = {"B"}
PURE_C = {"C"}


def main():
    os.makedirs("data", exist_ok=True)

    # Load dataset
    ds = datasets.load_dataset(DATASET_KEY)
    seqs = np.array(ds.sequences)
    labels = np.array(ds.labels)

    # Filter to A1, A6, B, C
    keep_labels = PURE_A | PURE_B | PURE_C
    mask = np.isin(labels, list(keep_labels))

    seqs_sub = seqs[mask]
    labels_sub = labels[mask]

    print(f"[INFO] Subset size for A/B/C: {len(seqs_sub)} sequences")

    # Map A1/A6 -> A; B -> B; C -> C
    groups = []
    for lab in labels_sub:
        if lab in PURE_A:
            groups.append("A")
        elif lab in PURE_B:
            groups.append("B")
        elif lab in PURE_C:
            groups.append("C")
        else:
            # Should not happen due to mask
            groups.append("OTHER")

    groups = np.array(groups)

    # 3D MoDMap via Manhattan distances + classical MDS
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
        f.write("sample_id\tgroup\tx\ty\tz\n")
        for i, (g, (x, y, z)) in enumerate(zip(groups, coords)):
            f.write(f"{i}\t{g}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
