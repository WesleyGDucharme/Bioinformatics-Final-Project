#!/usr/bin/env python3
"""
General MoDMap plotting script.

Author: Wesley Ducharme
Date: 2025-11-23

Usage examples:

  # Figure 2: 3D pure subtypes (label column = 'label')
  PYTHONPATH=src python3 scripts/modmap_plot.py \
      --input results/modmap_hiv1_lanl_whole__k5__pure.tsv \
      --label-col label \
      --dims 3 \
      --title "MoDMap – HIV-1 LANL whole (k=5, pure subtypes)" \
      --out figures/figure2_modmap_pure_k5.png

  # Figure 3: 2D A/B/C plot (label column = 'group')
  PYTHONPATH=src python3 scripts/modmap_plot.py \
      --input figures/modmap_hiv1_ABC_k5.tsv \
      --label-col group \
      --dims 2 \
      --title "MoDMap – HIV-1 LANL whole (A, B, C), k=5" \
      --out figures/figure3_modmap_ABC_k5.png
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="General MoDMap plotter (2D/3D) from TSV.")
    ap.add_argument(
        "--input",
        required=True,
        help="Path to TSV file with at least columns: x, y, and optional z.",
    )
    ap.add_argument(
        "--label-col",
        default=None,
        help="Name of label column (e.g. 'label' or 'group'). "
             "If omitted, tries 'label' then 'group'.",
    )
    ap.add_argument(
        "--dims",
        type=int,
        choices=[2, 3],
        default=None,
        help="Number of dimensions to plot: 2 or 3. "
             "Default: 3 if 'z' column exists, otherwise 2.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title (optional). If omitted, a simple title is inferred.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Path to save figure (e.g. figures/figure2.png). "
             "If omitted, just shows the plot interactively.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input, sep="\t")

    # Determine label column
    if args.label_col is not None:
        label_col = args.label_col
        if label_col not in df.columns:
            raise SystemExit(f"Label column '{label_col}' not found in {args.input}")
    else:
        # Try in order: 'label', 'group', 'type', 'subtype'
        for cand in ("label", "group", "type", "subtype"):
            if cand in df.columns:
                label_col = cand
                break
        else:
            raise SystemExit(
                f"No label column specified and none of "
                f"'label', 'group', 'type', or 'subtype' found in {args.input}")

    # Determine dimensionality
    if args.dims is not None:
        dims = args.dims
    else:
        dims = 3 if "z" in df.columns else 2

    if dims == 3 and "z" not in df.columns:
        raise SystemExit(
            f"Requested 3D plot but 'z' column is missing in {args.input}"
        )

    labels = sorted(df[label_col].unique())
    print(f"[INFO] Loaded {len(df)} points from {args.input}")
    print(f"[INFO] Using label column '{label_col}' with {len(labels)} unique labels")
    print(f"[INFO] Plotting in {dims}D")

    # Build figure
    if dims == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for lab in labels:
            sub = df[df[label_col] == lab]
            ax.scatter(
                sub["x"],
                sub["y"],
                sub["z"],
                label=lab,
                s=10,
                alpha=0.7,
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        for lab in labels:
            sub = df[df[label_col] == lab]
            ax.scatter(
                sub["x"],
                sub["y"],
                label=lab,
                s=10,
                alpha=0.7,
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    # Title
    if args.title:
        ax.set_title(args.title)
    else:
        base = os.path.basename(args.input)
        ax.set_title(f"MoDMap from {base}")

    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    # Save or show
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out, dpi=300)
        print(f"[OK] Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
