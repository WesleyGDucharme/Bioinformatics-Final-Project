#!/usr/bin/env python3
"""
scripts/make_modmap_pol_nat_syn.py

Author: Wesley Ducharme
Date: 2025-11-23

Purpose:
    Prepare a MoDMap-style 3D embedding for HIV-1 pol sequences:
      - "natural" = all pol genes from LANL (hiv1_lanl_pol)
      - "synthetic" = synthetic pol fragments (hiv1_synthetic_polfragments)

    This corresponds to the setup for Figure 4 in the Kameris paper:
      - Train/visualize on natural + synthetic pol sequences
      - Same k-mer representation as other experiments
      - Here we use k = 5 for consistency with the rest of this repro.

Output:
    results/modmap_hiv1_pol_nat_syn__k5.tsv with columns:

        sample_id    (integer index 0..N-1)
        type         ("natural" or "synthetic")
        subtype      (HIV-1 subtype label, e.g. A1, B, C, ...)
        x            (MoDMap coordinate 1)
        y            (MoDMap coordinate 2)
        z            (MoDMap coordinate 3)

    You can then use your generic modmap_plot.py to make:
      - Left panel: color by `type` (natural vs synthetic)
      - Right panel: color by `subtype`
"""

import os
import numpy as np
from sklearn.decomposition import TruncatedSVD

from kameris_reimp import datasets, kmer

DATASET_NAT = "hiv1_lanl_pol"
DATASET_SYN = "hiv1_synthetic_polfragments"
K = 5
OUT_TSV = os.path.join("data", "modmap_hiv1_pol_nat_syn__k5.tsv")


def compute_embedding(X: np.ndarray, n_components: int = 3, random_state: int = 42) -> np.ndarray:
    """
    Simple MoDMap-style embedding:
      1. Center columns of the k-mer matrix
      2. Apply TruncatedSVD to get 3D coordinates
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    coords = svd.fit_transform(X_centered)
    return coords


def main():
    os.makedirs("data", exist_ok=True)

    # --- Load datasets ---
    ds_nat = datasets.load_dataset(DATASET_NAT)
    ds_syn = datasets.load_dataset(DATASET_SYN)

    seqs_nat = np.array(ds_nat.sequences)
    seqs_syn = np.array(ds_syn.sequences)

    labels_nat = np.array(ds_nat.labels)    # subtype labels for natural pol
    labels_syn = np.array(ds_syn.labels)    # subtype labels for synthetic pol

    n_nat = len(seqs_nat)
    n_syn = len(seqs_syn)
    print(f"[INFO] Natural pol sequences:   {n_nat}")
    print(f"[INFO] Synthetic pol sequences: {n_syn}")

    # --- Concatenate sequences & metadata ---
    all_seqs = np.concatenate([seqs_nat, seqs_syn], axis=0)
    all_subtypes = np.concatenate([labels_nat, labels_syn], axis=0)
    all_types = np.array(["natural"] * n_nat + ["synthetic"] * n_syn)

    # --- Build k-mer matrix (k=5) ---
    print("[INFO] Computing k-mer matrix for natural + synthetic pol (k=5)...")
    X, vocab, valid_counts = kmer.batch_kmer_matrix(all_seqs, k=K, normalize=True)
    print(f"[INFO] k-mer matrix shape: {X.shape}")
    print(f"[INFO] #valid k-mers per sequence (min/mean/max): "
          f"{min(valid_counts)} / {np.mean(valid_counts):.1f} / {max(valid_counts)}")

    # --- 3D embedding via TruncatedSVD ---
    print("[INFO] Computing 3D embedding (TruncatedSVD)...")
    coords = compute_embedding(X, n_components=3, random_state=42)
    assert coords.shape[0] == len(all_seqs)

    # --- Write TSV ---
    print(f"[INFO] Writing TSV to {OUT_TSV}")
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("sample_id\ttype\tsubtype\tx\ty\tz\n")
        for i, (typ, sub, (x, y, z)) in enumerate(zip(all_types, all_subtypes, coords)):
            f.write(f"{i}\t{typ}\t{sub}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
