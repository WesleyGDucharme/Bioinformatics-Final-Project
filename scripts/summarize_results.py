#!/usr/bin/env python3
"""
Author: Wesley Ducharme
Date: 2025-11-23

Summarize HIV1 LANL whole experiments into a Markdown table.

Usage:
    PYTHONPATH=src python3 scripts/summarize_results.py
"""

import os
import json
from glob import glob
from statistics import mean, pstdev

RESULTS_DIR = "results"
DATASET_KEY = "hiv1_lanl_whole"
K = 5 # k-mer length used in the experiments to summarize

# Map model key used in filenames to nice label in the table
MODEL_DISPLAY = {
    "linear_svm": "Linear SVM",
    "quadratic_svm": "Quadratic SVM",
    "cubic_svm": "Cubic SVM",
    "10_nearest_neighbors": "10-Nearest Neighbors",
    "nearest_centroid_mean": "Nearest Centroid (mean)",
    "nearest_centroid_median": "Nearest Centroid (median)",
    "logistic_regression": "Logistic Regression",
    "sgd": "SGD",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "gaussian_naive_bayes": "Gaussian NB",
    "lda": "LDA",
    "qda": "QDA",
    "multilayer_perceptron": "MLP",
}

def load_result_files():
    pattern = os.path.join(
        RESULTS_DIR,
        f"{DATASET_KEY}__*__k{K}.json"
    )
    files = sorted(glob(pattern))
    if not files:
        raise SystemExit(f"No result files found matching {pattern}")
    return files

def extract_metrics_from_result(result):
    """
    Adjust this function if your JSON structure differs.
    It tries two patterns:

    1) result["folds"] = list of { "accuracy": float, "runtime": float, ... }
    2) result["folds"] = list of { "metrics": {"accuracy": float}, "timing": {...} }
    
    Returns:
        acc_mean, acc_std, runtime_mean
        
    """
    folds = result.get("folds") or result.get("cv_folds") or result.get("per_fold") or []

    if not folds:
        # Fallback: try top-level aggregate means
        acc_mean = result.get("mean_accuracy")
        acc_std = result.get("std_accuracy")
        runtime_mean = result.get("runtime_seconds")
        return acc_mean, acc_std, runtime_mean
    return acc_mean, acc_std, runtime_mean

def main():
    files = load_result_files()
    rows = []

    for path in files:
        base = os.path.basename(path)
        # Expect filenames like: hiv1_lanl_whole__MODEL__k5.json
        try:
            _, model_key, _ = base.split("__")
            model_key = model_key.replace(".json", "")
        except ValueError:
            print(f"[WARN] Unexpected filename format: {base}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)

        acc_mean, acc_std, runtime_mean = extract_metrics_from_result(result)

        rows.append({
            "model_key": model_key,
            "display": MODEL_DISPLAY.get(model_key, model_key),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "runtime_mean": runtime_mean,
        })

    # Sort rows by mean accuracy (descending), fall back to model name
    rows.sort(key=lambda r: (r["acc_mean"] is not None, r["acc_mean"]), reverse=True)

    # Print Markdown table
    print("| Model | Mean Accuracy | Std Accuracy | Mean Runtime |")
    print("|-------|--------------:|-------------:|-------------:|")

    for r in rows:
        def fmt(x, *, as_percent=False):
            if x is None:
                return "N/A"
            if as_percent:
                # assume accuracy is stored as 0–1, convert to %
                return f"{x * 100:.2f}"
            return f"{x:.2f}"

        print(
            f"| {r['display']} | "
            f"{fmt(r['acc_mean'], as_percent=True)}% | "
            f"{fmt(r['acc_std'], as_percent=True)}% | "
            f"{fmt(r['runtime_mean'])}s |")

if __name__ == "__main__":
    main()
