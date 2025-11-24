#!/usr/bin/env python3
"""
scripts/run_experiment.py

Author: Wesley Ducharme
Date: 2025-11-22

Run a single K-fold cross-validation experiment for one dataset/model at a
given k-mer length (default k=6, as in the Kameris paper).

Usage example (from repo root):

    PYTHONPATH=src python3 scripts/run_experiment.py \
        --dataset hiv1_lanl_whole \
        --model linear_svm \
        --k 6 \
        --cv-splits 10 \
        --seed 42 \
        --out results/hiv1_lanl_whole_linear_svm_k6.json

Repeatable for other datasets / models and then aggregate results.

Outputs a JSON file with detailed results, including per-fold metrics and confusion matrices.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from kameris_repo import datasets, kmer, preprocess, models



# Helpers to pull data out of Dataset objects without assuming exact attribute

def _get_sequences(ds) -> List[str]:
    """
    Try to obtain the raw nucleotide sequences from a Dataset object.
    We support both 'seqs' and 'sequences' attributes.
    """
    if hasattr(ds, "seqs"):
        return list(ds.seqs)
    if hasattr(ds, "sequences"):
        return list(ds.sequences)
    raise AttributeError(
        "Dataset object has neither 'seqs' nor 'sequences'. "
        "Please ensure datasets.load_dataset(...) returns one of these."
    )


def _get_labels_and_classes(ds) -> Tuple[np.ndarray, List[str]]:
    """
    Return (y_int, classes_str) for a Dataset.

    If ds has 'y' and 'classes', we trust them. Otherwise we fall back to
    'labels' and use a LabelEncoder.
    """
    if hasattr(ds, "y") and hasattr(ds, "classes"):
        y = np.asarray(ds.y)
        classes = list(ds.classes)
        return y, classes

    # Fallback: infer from string labels
    if hasattr(ds, "labels"):
        from sklearn.preprocessing import LabelEncoder

        labels = np.asarray(ds.labels)
        le = LabelEncoder()
        y = le.fit_transform(labels)
        classes = list(le.classes_)
        return y, classes

    raise AttributeError(
        "Dataset object has neither ('y' and 'classes') nor 'labels'. "
        "Please check datasets.load_dataset."
    )



# Core experiment runner

def run_experiment(
    dataset_key: str,
    model_key: str,
    k: int = 6,
    cv_splits: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load one dataset, build k-mer features, apply Preprocessor, and run
    Stratified K-fold CV with the selected model.

    Returns a dict that can be dumped to JSON.
    """
    t0 = time.time()

    # Load dataset
    ds = datasets.load_dataset(dataset_key)
    seqs = _get_sequences(ds)
    y, classes = _get_labels_and_classes(ds)
    n_samples = len(seqs)

    if len(y) != n_samples:
        raise ValueError(
            f"Length mismatch: {n_samples} sequences vs {len(y)} labels "
            f"for dataset '{dataset_key}'."
        )

    # Build k-mer matrix
    # Assuming kmer.build_kmer_matrix(seqs, k) -> (X, vocab)
    X, vocab, valid_counts = kmer.batch_kmer_matrix(seqs, k=k)
    # X can be numpy array or scipy sparse; Preprocessor handles both.

    # Preprocessing: matching the paper's "dim = avg # nonzeros" rule
    pre = preprocess.Preprocessor(
        svd_rule="avg_nnz",
        svd_fraction=1.0, # unused for 'avg_nnz' but kept explicit
        svd_fixed_k=None,
        scale=True,
        random_state=seed,
    )
    X_red = pre.fit_transform(X)

    # Stratified K-fold CV
    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )

    fold_metrics: List[Dict[str, Any]] = []
    confusions: List[List[List[int]]] = []

    accs: List[float] = []
    f1s: List[float] = []
    recs: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_red, y), start=1):
        # Fresh model for each fold
        clf = models.make_model(model_key)

        X_train, X_test = X_red[train_idx], X_red[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        accs.append(acc)
        f1s.append(macro_f1)
        recs.append(macro_rec)
        confusions.append(cm)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "macro_recall": macro_rec,
            }
        )

    t1 = time.time()

    # Aggregate results
    result: Dict[str, Any] = {
        "dataset": dataset_key,
        "model": model_key,
        "k": int(k),
        "n_samples": int(n_samples),
        "n_classes": int(len(classes)),
        "classes": [str(c) for c in classes],
        "cv_splits": int(cv_splits),
        "seed": int(seed),
        "vocab_size": int(len(vocab)),
        "preprocessor": {
            "svd_rule": pre.svd_rule,
            "svd_fraction": pre.svd_fraction,
            "svd_fixed_k": pre.svd_fixed_k,
            "scale": pre.scale,
            "random_state": pre.random_state,
        },
        "fold_metrics": fold_metrics,
        "confusion_matrices": confusions,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "mean_macro_recall": float(np.mean(recs)),
        "std_macro_recall": float(np.std(recs)),
        "runtime_seconds": float(t1 - t0),
    }

    return result



# CLI

def main() -> None:
    # Discover dataset keys from your datasets.py
    dataset_choices = datasets.list_dataset_keys()

    # Try to discover model keys from models.py, if such a helper exists
    model_choices = None
    if hasattr(models, "list_model_keys"):
        try:
            model_choices = models.list_model_keys()
        except Exception:
            model_choices = None

    ap = argparse.ArgumentParser(
        description="Run a single K-fold CV experiment (dataset + model) with k-mer features."
    )
    ap.add_argument(
        "--dataset",
        required=True,
        choices=dataset_choices,
        help=f"Dataset key (one of: {', '.join(dataset_choices)})",
    )
    if model_choices:
        ap.add_argument(
            "--model",
            required=True,
            choices=model_choices,
            help=f"Model key (one of: {', '.join(model_choices)})",
        )
    else:
        ap.add_argument(
            "--model",
            required=True,
            help="Model key (must be accepted by models.get_model).",
        )

    ap.add_argument(
        "--k",
        type=int,
        default=6,
        help="k-mer length (default: 6, as used in the paper).",
    )
    ap.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Number of Stratified K-fold splits (default: 10).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV shuffling and preprocessing (default: 42).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Path to JSON file where results will be written. "
            "If omitted, a name will be auto-generated in ./results/"
        ),
    )

    args = ap.parse_args()

    # Auto-generate output path if needed
    out_path = args.out
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        out_name = f"{args.dataset}__{args.model}__k{args.k}.json"
        out_path = os.path.join("results", out_name)

    print(f"[INFO] Running experiment:")
    print(f"       dataset = {args.dataset}")
    print(f"       model   = {args.model}")
    print(f"       k       = {args.k}")
    print(f"       splits  = {args.cv_splits}")
    print(f"       seed    = {args.seed}")
    print(f"       out     = {out_path}")
    print()

    result = run_experiment(
        dataset_key=args.dataset,
        model_key=args.model,
        k=args.k,
        cv_splits=args.cv_splits,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"[OK] Wrote results to {out_path}")
    print(
        f"     mean_accuracy={result['mean_accuracy']:.4f}, "
        f"mean_macro_f1={result['mean_macro_f1']:.4f}, "
        f"mean_macro_recall={result['mean_macro_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
