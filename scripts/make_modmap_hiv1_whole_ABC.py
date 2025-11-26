#!/usr/bin/env python3
"""
scripts/make_modmap_hiv1_whole_ABC.py

Author: Wesley Ducharme
Date: 2025-11-23

Purpose:
    Prepare a MoDMap-style 3D embedding for HIV-1 LANL whole-genome
    sequences, restricted to (pure) subtypes A, B, and C.

    - Uses dataset: hiv1_lanl_whole
    - k-mer length: k = 5
    - Subtype mapping:
        A1, A6 -> "A"
        B      -> "B"
        C      -> "C"
    - Embedding: TruncatedSVD (3 components) on centered k-mer matrix.

Output:
    results/modmap_hiv1_ABC_k5.tsv with columns:
        sample_id, group, x, y, z
"""

import os
import numpy as np
from sklearn.decomposition import TruncatedSVD

from kameris_reimp import datasets, kmer

DATASET_KEY = "hiv1_lanl_whole"
K = 5
OUT_TSV = os.path.join("data", "modmap_hiv1_ABC_k5.tsv")

PURE_A = {"A1", "A6"}
PURE_B = {"B"}
PURE_C = {"C"}


def compute_embedding(X: np.ndarray, n_components: int = 3, random_state: int = 42) -> np.ndarray:
    """
    Simple MoDMap-style embedding: center columns and apply TruncatedSVD
    to get 3D coordinates.
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    coords = svd.fit_transform(X_centered)
    return coords


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

    # Compute k-mer matrix (k=5)
    print("[INFO] Computing k-mer matrix...")
    X, vocab, _valid_counts = kmer.batch_kmer_matrix(seqs_sub, k=K, normalize=True)

    # 3D embedding
    print("[INFO] Computing 3D embedding (TruncatedSVD)...")
    coords = compute_embedding(X, n_components=3, random_state=42)

    # Write TSV
    print(f"[INFO] Writing TSV to {OUT_TSV}")
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("sample_id\tgroup\tx\ty\tz\n")
        for i, (g, (x, y, z)) in enumerate(zip(groups, coords)):
            f.write(f"{i}\t{g}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
