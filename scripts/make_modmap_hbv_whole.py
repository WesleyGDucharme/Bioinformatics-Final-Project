#!/usr/bin/env python3
"""
scripts/make_modmap_hbv_whole.py

Author: Wesley Ducharme
Date: 2025-11-23

Purpose:
    Prepare a MoDMap-style 3D embedding for hepatitis B whole genomes.

    - Dataset: hbv_whole
    - k-mer length: k = 5
    - Keep ONLY pure genotypes: A, B, C, D, E, F, G, H
      (all other labels are completely discarded)
    - Embedding: TruncatedSVD (3 components) on centered k-mer matrix.

Output:
    data/modmap_hbv_whole__k5.tsv with columns:
        sample_id, subtype, x, y, z
"""

import os
import numpy as np
from sklearn.decomposition import TruncatedSVD

from kameris_repo import datasets, kmer

DATASET_KEY = "hbv_whole"
K = 5
OUT_TSV = os.path.join("data", "modmap_hbv_whole__k5.tsv")

PURE_LABELS = {"A", "B", "C", "D", "E", "F", "G", "H"}


def compute_embedding(X: np.ndarray, n_components: int = 3, random_state: int = 42) -> np.ndarray:
    """
    MoDMap-style embedding: center columns and apply TruncatedSVD
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

    # Keep ONLY genotypes A–H
    mask = np.isin(labels, list(PURE_LABELS))
    seqs_sub = seqs[mask]
    labels_sub = labels[mask]

    print(f"[INFO] Total sequences: {len(seqs)}")
    print(f"[INFO] Kept pure A–H genotypes: {len(seqs_sub)}")

    # Compute k-mer matrix (k=5)
    print("[INFO] Computing k-mer matrix...")
    X, vocab, _valid_counts = kmer.batch_kmer_matrix(seqs_sub, k=K, normalize=True)

    # 3D embedding
    print("[INFO] Computing 3D embedding (TruncatedSVD)...")
    coords = compute_embedding(X, n_components=3, random_state=42)

    # Write TSV
    print(f"[INFO] Writing TSV to {OUT_TSV}")
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("sample_id\tsubtype\tx\ty\tz\n")
        for i, (lab, (x, y, z)) in enumerate(zip(labels_sub, coords)):
            f.write(f"{i}\t{lab}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
