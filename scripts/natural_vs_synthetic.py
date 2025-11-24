#!/usr/bin/env python3
"""
scripts/natural_vs_synthetic.py

Author: Wesley Ducharme
Date: 2025-11-23

Experiment: distinguish 'natural' vs 'synthetic' HIV-1 pol sequences.

- 'natural'   = all sequences from dataset key 'hiv1_lanl_pol'
- 'synthetic' = all sequences from dataset key 'hiv1_synthetic_polfragments'

This script will:
  * build k-mer vectors (k given on CLI, default 5)
  * preprocess with the same Preprocessor as run_experiment.py
    (scaling + TruncatedSVD with fraction_of_avg_nnz)
  * run Stratified 10-fold cross-validation
  * train the chosen model (default: linear_svm) in each fold
  * report per-fold accuracy + mean/std accuracy
  * write a JSON summary to results/

Usage example:

    PYTHONPATH=src python3 scripts/natural_vs_synthetic.py \\
      --model linear_svm \\
      --k 5 \\
      --cv-splits 10 \\
      --seed 42
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import Dict, Any, List

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from kameris_repo import datasets, kmer, preprocess, models


def run_natural_vs_synthetic(
    model_name: str,
    k: int = 5,
    cv_splits: int = 10,
    seed: int = 42,
    out_path: str | None = None,
) -> Dict[str, Any]:
    """
    Run the natural vs synthetic experiment and return a result dict.
    """

    # Load datasets
    ds_nat = datasets.load_dataset("hiv1_lanl_pol")
    ds_syn = datasets.load_dataset("hiv1_synthetic_polfragments")

    seqs_nat = list(ds_nat.sequences)
    seqs_syn = list(ds_syn.sequences)

    n_nat = len(seqs_nat)
    n_syn = len(seqs_syn)

    # Combine sequences + build binary labels
    seqs_all = seqs_nat + seqs_syn
    y = np.array(
        ["natural"] * n_nat + ["synthetic"] * n_syn,
        dtype=object,
    )

    # Build k-mer matrix (shared vocab)
    print(f"[INFO] Building {k}-mer matrix for {len(seqs_all)} sequences...")
    X, vocab, valid_counts = kmer.batch_kmer_matrix(seqs_all, k=k)
    # X: (n_samples, 4^k), vocab: dict kmer->index, valid_counts: per-seq nnz

    # Set up CV
    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )

    fold_results: List[Dict[str, Any]] = []

    print(f"[INFO] Running {cv_splits}-fold Stratified CV with model={model_name!r}...")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocessing: same style as run_experiment.py
        pre = preprocess.Preprocessor(
            scale=True,
            svd_rule="fraction_of_avg_nnz",
            svd_fraction=0.10,
            svd_fixed_k=None,
            random_state=seed,
        )

        fold_start = time.time()

        # Fit on train, transform both
        X_train_proc = pre.fit_transform(X_train)
        X_test_proc = pre.transform(X_test)

        # Model
        clf = models.make_model(model_name)

        train_start = time.time()
        clf.fit(X_train_proc, y_train)
        train_end = time.time()

        # Evaluate
        y_pred = clf.predict(X_test_proc)
        acc = float(accuracy_score(y_test, y_pred))

        fold_end = time.time()

        fold_results.append(
            {
                "fold_index": fold_idx,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "accuracy": acc,
                "train_time_seconds": float(train_end - train_start),
                "fold_time_seconds": float(fold_end - fold_start),
            }
        )

        print(
            f"[INFO] Fold {fold_idx}/{cv_splits}: "
            f"accuracy={acc:.4f}, fold_time={fold_end - fold_start:.2f}s"
        )

    # Aggregate metrics
    accs = [fr["accuracy"] for fr in fold_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs, ddof=0))

    # mean runtimes (may or may not use)
    mean_fold_time = float(np.mean([fr["fold_time_seconds"] for fr in fold_results]))
    mean_train_time = float(np.mean([fr["train_time_seconds"] for fr in fold_results]))

    result: Dict[str, Any] = {
        "experiment": "natural_vs_synthetic_pol",
        "dataset_natural": "hiv1_lanl_pol",
        "dataset_synthetic": "hiv1_synthetic_polfragments",
        "model_name": model_name,
        "k": int(k),
        "cv_splits": int(cv_splits),
        "seed": int(seed),
        "n_samples_total": int(len(seqs_all)),
        "n_natural": int(n_nat),
        "n_synthetic": int(n_syn),
        "folds": fold_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_fold_time_seconds": mean_fold_time,
        "mean_train_time_seconds": mean_train_time,
        "vocab_size": int(len(vocab)),
        "notes": (
            "Binary classification: 'natural' = all pol genes from LANL; "
            "'synthetic' = 1500 synthetic pol sequences. "
            "Preprocessing: scaling + TruncatedSVD(rule='fraction_of_avg_nnz', fraction=0.10)."
        ),
    }

    # Save JSON if requested
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join(
            "results",
            f"natural_vs_synthetic_pol__{model_name}__k{k}.json",
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"[OK] mean_accuracy={mean_acc:.4f}, std={std_acc:.4f}, "
        f"output -> {out_path}"
    )
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Distinguish natural vs synthetic HIV-1 pol sequences."
    )
    ap.add_argument(
        "--model",
        default="linear_svm",
        help="Model name (as understood by kameris_repo.models.make_model).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="k-mer length (paper used k=6; we default to 5 for consistency).",
    )
    ap.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Number of StratifiedKFold splits (default: 10).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV split and preprocessing.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path (default: results/natural_vs_synthetic_pol__MODEL__kK.json).",
    )
    args = ap.parse_args()

    run_natural_vs_synthetic(
        model_name=args.model,
        k=args.k,
        cv_splits=args.cv_splits,
        seed=args.seed,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
