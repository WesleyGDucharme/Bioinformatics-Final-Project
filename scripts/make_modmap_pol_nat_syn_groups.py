#!/usr/bin/env python3
"""
Group MoDMap points for natural + synthetic pol embeddings.

Two grouping modes:
  - type (left panel): group = natural vs synthetic
  - subtype (right panel): collapse subtypes into A/B/C/D/other
"""

import os
import re
import argparse
import pandas as pd

IN_TSV = os.path.join("data", "modmap_hiv1_pol_nat_syn__k6.tsv")
OUT_TSV = os.path.join("data", "modmap_hiv1_pol_nat_syn__k6__groups.tsv")


def collapse_subtype(st: str) -> str:
    """
    Map raw subtype strings into A/B/C/D/other.
    We treat pure major subtypes like A, A1, A2, A6, B, C, D, etc. as
    their letter, and everything else as 'other'.
    """
    if not isinstance(st, str):
        return "other"
    s = st.strip().upper()

    if re.fullmatch(r"A\d*", s):
        return "A"
    if re.fullmatch(r"B\d*", s):
        return "B"
    if re.fullmatch(r"C\d*", s):
        return "C"
    if re.fullmatch(r"D\d*", s):
        return "D"

    # e.g. 0107, 01BC, 01_AE, BC, BF, CD, F1, G, O, U, etc.
    return "other"


def main():
    ap = argparse.ArgumentParser(
        description="Group MoDMap TSV for natural+synthetic pol (type vs synthetic OR collapsed subtypes)."
    )
    ap.add_argument(
        "--mode",
        choices=["type", "subtype"],
        default="type",
        help="Grouping mode: 'type' = natural vs synthetic (left panel); "
             "'subtype' = A/B/C/D/other (right panel). Default: type.",
    )
    ap.add_argument(
        "--input",
        default=IN_TSV,
        help=f"Input TSV path (default: {IN_TSV})",
    )
    ap.add_argument(
        "--out",
        default=OUT_TSV,
        help=f"Output TSV path (default: {OUT_TSV})",
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input TSV not found: {args.input}")

    df = pd.read_csv(args.input, sep="\t")
    for col in ("sample_id", "type", "subtype", "x", "y"):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in {args.input}")

    if args.mode == "type":
        df["group"] = df["type"].astype(str)
    else:
        df["group"] = df["subtype"].apply(collapse_subtype)

    cols = ["sample_id", "type", "subtype", "group", "x", "y"]
    if "z" in df.columns:
        cols.append("z")
    df_out = df[cols]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_out.to_csv(args.out, sep="\t", index=False)

    print(f"[OK] Wrote grouped TSV to {args.out}")
    print("Group counts:")
    print(df_out["group"].value_counts().sort_index())


if __name__ == "__main__":
    main()
