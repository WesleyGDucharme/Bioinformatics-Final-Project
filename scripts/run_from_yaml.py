#!/usr/bin/env python3
"""
scripts/run_from_yaml.py

Author: Wesley Ducharme
Date: 2025-11-22

Run a batch of experiments defined in one or more YAML config files.

Each YAML file should look like:

  dataset: hiv1_lanl_whole
  k: 5                # optional, default 5
  cv_splits: 10       # optional, default 10
  seed: 42            # optional, default 42
  models:
    - linear_svm
    - quadratic_svm
    - random_forest
    # or with per-model overrides:
    # - name: linear_svm
    #   k: 6
    #   seed: 123
    #   cv_splits: 5
    #   out: results/custom_name.json

Usage examples:

  # Run one config
  PYTHONPATH=src python3 scripts/run_from_yaml.py experiments/hiv1_lanl_whole.yml

  # Run several configs in sequence
  PYTHONPATH=src python3 scripts/run_from_yaml.py experiments/hiv1_lanl_whole.yml \
                                                experiments/hiv1_lanl_pol.yml

  # Just show what would be run, without executing
  PYTHONPATH=src python3 scripts/run_from_yaml.py --dry-run experiments/hiv1_lanl_whole.yml
"""

import argparse
import os
import sys
import subprocess
from typing import Any, Dict, List

import yaml  # make sure pyyaml is installed in your environment


def _normalize_models(raw_models: Any) -> List[Dict[str, Any]]:
    """
    Normalize the 'models' entry from the YAML into a list of dicts
    each having at least a 'name' field.

    Accepts:
      - ["linear_svm", "random_forest", ...]
      - [{"name": "linear_svm"}, {"model": "random_forest", "seed": 123}, ...]
    """
    if not isinstance(raw_models, list):
        raise ValueError("'models' must be a list in the YAML config")

    norm: List[Dict[str, Any]] = []
    for m in raw_models:
        if isinstance(m, str):
            norm.append({"name": m})
        elif isinstance(m, dict):
            d = dict(m)  # shallow copy
            # allow 'model' or 'name'
            if "name" not in d and "model" in d:
                d["name"] = d.pop("model")
            if "name" not in d:
                raise ValueError(f"Model entry {m!r} is missing 'name'/'model'")
            norm.append(d)
        else:
            raise ValueError(f"Unsupported model entry in YAML: {m!r}")
    return norm


def run_config(config_path: str, dry_run: bool = False) -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML in {config_path} must be a mapping")

    dataset = cfg.get("dataset")
    if not dataset:
        raise ValueError(f"{config_path}: 'dataset' key is required")

    default_k = int(cfg.get("k", 5))
    default_cv = int(cfg.get("cv_splits", 10))
    default_seed = int(cfg.get("seed", 42))

    models_cfg = _normalize_models(cfg.get("models", []))
    if not models_cfg:
        raise ValueError(f"{config_path}: 'models' list is empty")

    print(f"[CONFIG] {config_path}")
    print(f"  dataset = {dataset}")
    print(f"  default k = {default_k}, cv_splits = {default_cv}, seed = {default_seed}")
    print(f"  models   = {[m['name'] for m in models_cfg]}")

    env = os.environ.copy()  # keep existing PYTHONPATH etc.

    for idx, m in enumerate(models_cfg):
        name = m["name"]
        k = int(m.get("k", default_k))
        cv_splits = int(m.get("cv_splits", default_cv))
        seed = int(m.get("seed", default_seed))
        out = m.get("out")

        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--dataset", dataset,
            "--model", name,
            "--k", str(k),
            "--cv-splits", str(cv_splits),
            "--seed", str(seed),
        ]
        if out:
            cmd.extend(["--out", out])

        print(f"\n[RUN {idx+1}/{len(models_cfg)}] {' '.join(cmd)}")
        if dry_run:
            continue

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[ERROR] Command failed with exit code {result.returncode}", file=sys.stderr)
            # you can choose to break or continue; here we break so you notice
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Run experiments defined in YAML config files.")
    ap.add_argument("configs", nargs="+", help="One or more YAML config files in experiments/")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would be run, but do not execute them.")
    args = ap.parse_args()

    for cfg_path in args.configs:
        run_config(cfg_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
