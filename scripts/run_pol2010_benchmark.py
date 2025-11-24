#!/usr/bin/env python3
"""
Train on HIV-1 Web pol 2010 and test on the HIV-1 benchmark mixed pol fragments.

Author: Wesley Ducharme
Date: 2025-11-23

Usage (from repo root):

    PYTHONPATH=src python3 scripts/run_pol2010_benchmark.py \
      --model linear_svm \
      --k 5 \
      --seed 42

This will write a JSON result file under results/.
"""

import os
import json
import time
import argparse
import random

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from kameris_repo import datasets, kmer, preprocess, models


def run_pol2010_benchmark(
    train_key: str = "hiv1_web_pol2010",
    test_key: str = "hiv1_benchmark_mixed_polfragments",
    model_name: str = "linear_svm",
    k: int = 5,
    seed: int = 42,
) -> dict:
    """Train on train_key, test on test_key with k-mer features and a given model."""


    random.seed(seed)
    np.random.seed(seed)

    # Load datasets
    ds_train = datasets.load_dataset(train_key)
    ds_test = datasets.load_dataset(test_key)

    y_train = np.array(ds_train.labels)
    y_test  = np.array(ds_test.labels)

    # Build k-mer feature matrices (same deterministic index since k is fixed)
    # batch_kmer_matrix uses a deterministic kmer_index(k), so calling it twice
    # for the same k gives the same feature ordering.
    t0 = time.perf_counter()
    X_train, _idx_train, valid_train = kmer.batch_kmer_matrix(ds_train.sequences, k=k)
    X_test, _idx_test, valid_test = kmer.batch_kmer_matrix(ds_test.sequences, k=k)
    t_kmer = time.perf_counter() - t0

    # Preprocessing: scale + SVD (same config as CV pipeline)
    pre = preprocess.Preprocessor(
        scale=True,
        svd_rule="fraction_of_avg_nnz",
        svd_fraction=0.10,
        svd_fixed_k=None,
        random_state=seed,
    )

    t0 = time.perf_counter()
    pre.fit(X_train)
    X_train_proc = pre.transform(X_train)
    X_test_proc = pre.transform(X_test)
    t_preproc = time.perf_counter() - t0

    # Model
    clf = models.make_model(model_name)

    t0 = time.perf_counter()
    clf.fit(X_train_proc, y_train)
    t_fit = time.perf_counter() - t0

    # Predictions & metrics
    t0 = time.perf_counter()
    y_pred_train = clf.predict(X_train_proc)
    t_pred_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_test = clf.predict(X_test_proc)
    t_pred_test = time.perf_counter() - t0

    acc_train = float(accuracy_score(y_train, y_pred_train))
    acc_test = float(accuracy_score(y_test, y_pred_test))

    # Classification report + confusion matrix on the benchmark set
    labels_sorted = sorted(sorted(set(y_test)))  # stable ordering
    conf_mat = confusion_matrix(y_test, y_pred_test, labels=labels_sorted).tolist()
    cls_report = classification_report(
        y_test, y_pred_test, labels=labels_sorted, output_dict=True, zero_division=0
    )

    result = {
        "train_dataset": train_key,
        "test_dataset": test_key,
        "model": model_name,
        "k": k,
        "seed": seed,
        "n_train": int(ds_train.n_samples),
        "n_test": int(ds_test.n_samples),
        "train_classes": list(ds_train.classes),
        "test_classes": list(ds_test.classes),
        "valid_kmers_train": valid_train,
        "valid_kmers_test": valid_test,
        "metrics": {
            "train_accuracy": acc_train,
            "test_accuracy": acc_test,
        },
        "classification_report_test": cls_report,
        "confusion_matrix_test": {
            "labels": labels_sorted,
            "matrix": conf_mat,
        },
        "timing": {
            "kmer_seconds": t_kmer,
            "preprocess_seconds": t_preproc,
            "fit_seconds": t_fit,
            "predict_train_seconds": t_pred_train,
            "predict_test_seconds": t_pred_test,
            "total_seconds": t_kmer + t_preproc + t_fit + t_pred_train + t_pred_test,
        },
        "notes": (
            "Single train–test experiment: train on hiv1_web_pol2010, "
            "test on hiv1_benchmark_mixed_polfragments, linear SVM, k-mer features."
        ),
    }

    return result


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train linear SVM on HIV-1 Web pol 2010 and evaluate on "
            "the benchmark mixed pol fragments dataset."
        )
    )
    ap.add_argument(
        "--train-dataset",
        default="hiv1_web_pol2010",
        help="Training dataset key (default: hiv1_web_pol2010).",
    )
    ap.add_argument(
        "--test-dataset",
        default="hiv1_benchmark_mixed_polfragments",
        help="Test dataset key (default: hiv1_benchmark_mixed_polfragments).",
    )
    ap.add_argument(
        "--model",
        default="linear_svm",
        help="Model name (default: linear_svm).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="k-mer length (default: 5).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path. If omitted, a name is derived in results/.",
    )
    args = ap.parse_args()

    result = run_pol2010_benchmark(
        train_key=args.train_dataset,
        test_key=args.test_dataset,
        model_name=args.model,
        k=args.k,
        seed=args.seed,
    )

    os.makedirs("results", exist_ok=True)
    if args.out is None:
        out_name = (
            f"pol2010_train__mixed_pol_benchmark_test__"
            f"{args.model}__k{args.k}__seed{args.seed}.json"
        )
        out_path = os.path.join("results", out_name)
    else:
        out_path = args.out

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"[OK] Wrote cross-dataset result to: {out_path}")
    print(f"Test accuracy: {result['metrics']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
