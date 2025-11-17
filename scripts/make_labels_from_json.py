#!/usr/bin/env python3
"""
Author: Wesley Ducharme
Date: 2025-11-16

Make labels.tsv directly from a Kameris-style JSON mapping (no FASTA scan).

Usage examples:
  # HIV-1 LANL whole
  python3 scripts/make_labels_from_json.py \
    --json data/hiv1_lanl_whole/hiv1-lanl-whole.json \
    --out-dir data/hiv1_lanl_whole --project hiv1

  # HCV LANL whole
  python3 scripts/make_labels_from_json.py \
    --json data/hcv_whole/hcv-lanl-whole.json \
    --out-dir data/hcv_whole --project hcv
"""

from __future__ import annotations
import os, re, csv, json, argparse
from collections import Counter
from typing import Dict

# ---------------- Label normalization ----------------

CRF_RE = re.compile(r"(CRF\d{1,3}(?:_[A-Z0-9]{1,2})+)", re.IGNORECASE)

def canon(s: str) -> str:
    s = s.strip().replace(" ", "_").replace("/", "_").upper()
    while "__" in s:
        s = s.replace("__", "_")
    return s

def normalize_label(lab: str, project: str) -> str:
    """
    Normalize a label per project conventions.
    IMPORTANT: In HIV-1, 'U' is a legitimate class and is kept as 'U'.
    """
    if project == "hcv":
        L = str(lab).strip().lower()
        allowed = {"1a","1b","2a","2b","3a","6a"}
        if L in allowed:
            return L
        if re.fullmatch(r"[1-7]", L):
            return L  # keep broad genotype if present
        return L

    L = canon(str(lab))
    if project == "hiv1":
        # Include U explicitly as a valid class.
        allowed_pure = {"A","A1","A2","A6","B","C","D","F","F1","F2","G","H","J","K","U"}
        if L in allowed_pure or L.startswith("CRF"):
            return L
        # Keep A1 and A6 as-is; collapse other A# to A.
        if re.fullmatch(r"A\d+", L):
            return L if L in {"A1", "A6"} else "A"
        # Collapse F# to F
        if re.fullmatch(r"F\d+", L):
            return "F"
        # Tokens like AE, AG (recombinants written as X_Y)
        if re.fullmatch(r"[A-Z]{1,2}_[A-Z]{1,2}", L):
            return L
        return L

    if project == "dengue":
        if L in {"DENV1","DENV2","DENV3","DENV4"}:
            return L
        return L

    if project == "hbv":
        if re.fullmatch(r"[A-H]", L):
            return L
        # e.g., C1, C2 -> collapse to C
        if re.fullmatch(r"[A-H]\d", L):
            return L[0]
        return L

    return L

# ---------------- Robust JSON loader (accepts various shapes) ----------------

ID_KEYS = ("id", "accession", "accession_id", "acc", "name")
LABEL_KEYS = ("label", "subtype", "genotype", "serotype", "type", "class")

def _pick_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return d[k]
    return None

def load_json_mapping(json_path: str) -> Dict[str, str]:
    """
    Load id->label from many plausible JSON shapes:
      - direct dict {id: label}
      - {"labels": {...}} or {"mapping": {...}}
      - {"items"/"data"/"rows": [ {id, subtype/...} or [id,label], ... ]}
      - {id: {subtype/...}}, i.e., accession -> inner object with a label-ish field
      - top-level list of dicts or [id,label] pairs
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    mapping: Dict[str, str] = {}

    def register(k, v):
        k = str(k).strip()
        v = str(v).strip()
        if k and v:
            mapping[k] = v

    if isinstance(obj, dict):
        # direct dict: {id: label}
        if all(isinstance(k, str) for k in obj.keys()) and all(
            isinstance(v, (str, int, float)) for v in obj.values()
        ):
            for k, v in obj.items():
                register(k, v)
            return mapping

        # wrappers: labels/mapping
        for key in ("labels", "mapping"):
            if key in obj and isinstance(obj[key], dict):
                for k, v in obj[key].items():
                    register(k, v)
                return mapping

        # arrays under items/data/rows (dict rows or [id,label])
        for key in ("items", "data", "rows"):
            if key in obj and isinstance(obj[key], list):
                for row in obj[key]:
                    if isinstance(row, dict):
                        rid = _pick_key(row, ID_KEYS)
                        lab = _pick_key(row, LABEL_KEYS)
                        if rid is not None and lab is not None:
                            register(rid, lab)
                    elif isinstance(row, (list, tuple)) and len(row) >= 2:
                        register(row[0], row[1])
                return mapping

        # accession -> { label-ish fields }
        inserted = False
        for k, v in obj.items():
            if isinstance(v, dict):
                lab = _pick_key(v, LABEL_KEYS)
                if lab is not None:
                    register(k, lab)
                    inserted = True
        if inserted:
            return mapping

    if isinstance(obj, list):
        inserted = False
        for row in obj:
            if isinstance(row, dict):
                rid = _pick_key(row, ID_KEYS)
                lab = _pick_key(row, LABEL_KEYS)
                if rid is not None and lab is not None:
                    register(rid, lab); inserted = True
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                register(row[0], row[1]); inserted = True
        if inserted:
            return mapping

    raise ValueError(f"Unrecognized JSON structure in {json_path}")

# ---------------- Main: JSON → labels.tsv (+ report) ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to Kameris JSON mapping file.")
    ap.add_argument("--out-dir", required=True, help="Output directory (dataset folder).")
    ap.add_argument("--project", required=True, choices=["hiv1","dengue","hbv","hcv"])
    ap.add_argument("--min", type=int, default=18, help="Minimum sequences per class (paper rule).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_map = load_json_mapping(args.json)

    # Normalize labels
    id2lab_norm: Dict[str, str] = {sid: normalize_label(lab, args.project)
                                   for sid, lab in raw_map.items()}

    # Thresholding (≥ min)
    counts = Counter(id2lab_norm.values())
    keep_classes = {c for c, n in counts.items() if n >= args.min}
    kept_items = [(sid, lab) for sid, lab in id2lab_norm.items() if lab in keep_classes]

    # Write labels.tsv
    out_tsv = os.path.join(args.out_dir, "labels.tsv")
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "label"])
        for sid, lab in kept_items:
            w.writerow([sid, lab])

    # Report (no unknown logic here)
    report = {
        "json_path": args.json,
        "project": args.project,
        "n_total_in_json": len(raw_map),
        "n_after_normalize": len(id2lab_norm),
        "n_classes_total": len(counts),
        "class_counts_total": dict(counts),
        "min_per_class": args.min,
        "n_classes_kept": len(keep_classes),
        "n_labeled_kept": len(kept_items),
    }
    out_json = os.path.join(args.out_dir, "labels_report.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"[OK] wrote {out_tsv} ({len(kept_items)} items) and {out_json}")

if __name__ == "__main__":
    main()
