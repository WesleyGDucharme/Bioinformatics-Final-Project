#!/usr/bin/env python3
"""
scripts/make_flu_labels.py

Purpose
-------
Create labels.tsv for the Influenza A dataset from a metadata.json that maps
final genome FASTA filenames to their subtype. Keeps only whitelisted subtypes
and applies the ≥ min-per-class rule. Also emits a summary report JSON.

Inputs
------
--metadata PATH        Path to metadata.json (list of records with at least
                       'filename' and 'subtype'; accepts a few alias keys).
--out-dir DIR          Where to write labels.tsv and labels_report.json.
--min INT              Minimum items per class to keep (default: 18).
--fasta-dir DIR        (Optional) Directory where FASTA files live; if given,
                       the script verifies that each mentioned file exists.
--whitelist-file PATH  (Optional) Override the built-in whitelist with a file
                       containing one subtype per line.

Outputs
-------
out-dir/labels.tsv            (id<TAB>label)  where id is filename without .fasta
out-dir/labels_report.json    summary counts and diagnostics
out-dir/labels_unmatched.txt  (subtypes in metadata that are NOT in whitelist)

Notes
-----
- subtype is taken as-is from metadata.
"""

from __future__ import annotations
import os, json, csv, argparse
from collections import Counter, defaultdict

# whitelist 
DEFAULT_WHITELIST = {
    "H1N1","H1N2","H1N3","H1N6","H1N9",
    "H2N1","H2N2","H2N3","H2N7","H2N9",
    "H3N1","H3N2","H3N3","H3N6","H3N8",
    "H4N2","H4N6","H4N8","H4N9",
    "H5N1","H5N2","H5N3","H5N5","H5N6","H5N8",
    "H6N1","H6N2","H6N5","H6N6","H6N8",
    "H7N1","H7N2","H7N3","H7N4","H7N6","H7N7","H7N9",
    "H8N4","H9N2","H10N1","H10N3","H10N4","H10N5","H10N6",
    "H10N7","H10N8","H11N1","H11N2","H11N3","H11N9",
    "H12N5","H13N2","H13N6","H13N8","H16N3","mixed",
}

#  Helpers 
ID_KEYS = ("filename","file","fname","final_filename","name")
SUBTYPE_KEYS = ("subtype","label","class","type")

def _pick(d: dict, keys: tuple[str,...]):
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return str(d[k]).strip()
    return None

def _load_whitelist(path: str|None):
    if not path:
        return set(DEFAULT_WHITELIST)
    keep = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                keep.add(s)
    return keep

def _load_metadata(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if isinstance(obj, list):
        return obj
    # try common wrappers
    for key in ("items","data","rows","entries"):
        if key in obj and isinstance(obj[key], list):
            return obj[key]
    raise ValueError("Unsupported metadata.json structure: expected list at top-level.")

def _file_id_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    if base.lower().endswith(".fasta"):
        base = base[:-6]
    return base

#  Main 
def main():
    ap = argparse.ArgumentParser(description="Build labels.tsv for Influenza A from metadata.json.")
    ap.add_argument("--metadata", required=True, help="Path to metadata.json.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--min", type=int, default=18, help="Minimum items per class to keep (default: 18).")
    ap.add_argument("--fasta-dir", default=None, help="Optional directory where FASTA files live (for existence check).")
    ap.add_argument("--whitelist-file", default=None, help="Optional path to a file with one subtype per line to override built-in whitelist.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    whitelist = _load_whitelist(args.whitelist_file)

    # Load metadata rows
    rows = _load_metadata(args.metadata)

    # Collect id -> label (exact strings from metadata)
    id2label = {}
    non_whitelisted = Counter()
    missing_files = []
    total_rows = 0
    dup_conflicts = 0

    for r in rows:
        total_rows += 1
        fname = _pick(r, ID_KEYS)
        sub   = _pick(r, SUBTYPE_KEYS)
        if fname is None or sub is None:
            continue  # skip malformed rows

        file_id = _file_id_from_filename(fname)
        label = sub  # no normalization beyond trimming

        # optional file existence check
        if args.fasta_dir:
            fpath = os.path.join(args.fasta_dir, os.path.basename(fname))
            if not os.path.isfile(fpath):
                missing_files.append(os.path.basename(fname))

        # track non-whitelisted to report later (still collect counts, but they won't be kept)
        if label not in whitelist:
            non_whitelisted[label] += 1

        # dedupe by id; if conflicting labels appear for same id, keep first and count conflict
        if file_id in id2label and id2label[file_id] != label:
            dup_conflicts += 1
            continue
        id2label.setdefault(file_id, label)

    # Compute counts and apply filters
    counts_total = Counter(id2label.values())
    kept_labels = {lab for lab, n in counts_total.items() if (lab in whitelist and n >= args.min)}
    dropped_under_min = {lab: n for lab, n in counts_total.items() if (lab in whitelist and n < args.min)}

    kept_pairs = [(sid, lab) for sid, lab in id2label.items() if lab in kept_labels]

    # Write labels.tsv
    out_tsv = os.path.join(args.out_dir, "labels.tsv")
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id","label"])
        for sid, lab in kept_pairs:
            w.writerow([sid, lab])

    # Write report
    kept_counts = {lab: counts_total[lab] for lab in sorted(kept_labels)}
    report = {
        "metadata_path": os.path.abspath(args.metadata),
        "out_dir": os.path.abspath(args.out_dir),
        "min_per_class": args.min,
        "whitelist_source": args.whitelist_file if args.whitelist_file else "built_in",
        "whitelist_size": len(whitelist),
        "total_rows_in_metadata": total_rows,
        "distinct_ids_seen": len(id2label),
        "duplicate_label_conflicts_ignored": dup_conflicts,
        "verify_files": bool(args.fasta_dir),
        "n_missing_files": len(missing_files),
        "missing_files_sample": missing_files[:20],
        "class_counts_total": dict(counts_total),
        "kept_labels": sorted(kept_labels),
        "kept_class_counts": kept_counts,
        "dropped_under_min": dropped_under_min,
        "non_whitelisted_class_counts": dict(non_whitelisted),
        "notes": "No normalization of subtype strings; exact match against whitelist; id = filename without .fasta",
    }
    with open(os.path.join(args.out_dir, "labels_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Emit unmatched (non-whitelisted) for inspection
    if non_whitelisted:
        with open(os.path.join(args.out_dir, "labels_unmatched.txt"), "w", encoding="utf-8") as fh:
            for lab, n in sorted(non_whitelisted.items()):
                fh.write(f"{lab}\t{n}\n")

    print(f"[OK] Wrote:\n  {out_tsv}\n  {os.path.join(args.out_dir, 'labels_report.json')}"
          + (f"\n  {os.path.join(args.out_dir, 'labels_unmatched.txt')}" if non_whitelisted else ""))
    if args.fasta_dir and missing_files:
        print(f"[WARN] {len(missing_files)} filenames from metadata not found under --fasta-dir (see report).")

if __name__ == "__main__":
    main()
