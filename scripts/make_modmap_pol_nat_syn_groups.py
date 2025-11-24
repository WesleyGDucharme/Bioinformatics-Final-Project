#!/usr/bin/env python3
"""
scripts/make_modmap_pol_nat_syn_groups.py

Author: Wesley Ducharme
Date: 2025-11-23

Purpose:
    Take the MoDMap TSV for natural + synthetic pol (with a 'subtype'
    column) and collapse subtypes into 5 groups:

        A, B, C, D, other

    Rough rule:
      - 'A', 'A1', 'A2', 'A6', ...  -> 'A'
      - 'B', 'B1', ...              -> 'B'
      - 'C', 'C1', ...              -> 'C'
      - 'D', 'D1', ...              -> 'D'
      - everything else             -> 'other'

Input:
    data/modmap_hiv1_pol_nat_syn__k5.tsv
       columns: sample_id, type, subtype, x, y, z

Output:
    data/modmap_hiv1_pol_nat_syn__k5__groups.tsv
       columns: sample_id, type, subtype, group, x, y, z
"""

import os
import re
import pandas as pd

IN_TSV = os.path.join("data", "modmap_hiv1_pol_nat_syn__k5.tsv")
OUT_TSV = os.path.join("data", "modmap_hiv1_pol_nat_syn__k5__groups.tsv")


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
    if not os.path.exists(IN_TSV):
        raise SystemExit(f"Input TSV not found: {IN_TSV}")

    df = pd.read_csv(IN_TSV, sep="\t")
    if "subtype" not in df.columns:
        raise SystemExit(f"'subtype' column not found in {IN_TSV}")

    df["group"] = df["subtype"].apply(collapse_subtype)

    # Keep useful columns in a nice order
    cols = ["sample_id", "type", "subtype", "group", "x", "y"]
    if "z" in df.columns:
        cols.append("z")
    df_out = df[cols]

    os.makedirs("results", exist_ok=True)
    df_out.to_csv(OUT_TSV, sep="\t", index=False)

    print(f"[OK] Wrote grouped TSV to {OUT_TSV}")
    print("Group counts:")
    print(df_out["group"].value_counts().sort_index())


if __name__ == "__main__":
    main()
