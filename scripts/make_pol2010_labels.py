#!/usr/bin/env python3
"""
scripts/make_pol2010_labels.py

Author: Wesley Ducharme
Date: 2025-11-16

Purpose: Create labels.tsv for the HIV-1 pol 2010 dataset only.

Inputs:
  --fasta PATH              (or) --dataset DIR containing sequences.fasta[.gz]
  --whitelist PATH          text file: one label per line (EXACT strings to match)
  --min INT                 default 18
  --out-dir DIR             default: folder of the FASTA

Outputs in out-dir:
  labels.tsv                (id<TAB>label) using EXACT label strings from headers
  labels_report.json        (summary, counts)
  labels_unmatched.txt      (headers that didn’t map into the whitelist)

Rules:
  - ID is an accession (version stripped) if present; else the header's first token.
  - Label is taken VERBATIM as:
        first whitespace token of the header  -> take the substring before the first '.'
        e.g.,  ">01B.TH.2010...."   -> label "01B"
               ">A1.CY.05...."      -> label "A1"
               ">B.FR.83...."       -> label "B"
  - NO normalization of label strings (no case change, no separator change, no 'CRF' handling).
  - Only labels that exactly appear in the whitelist are kept; then apply ≥ min rule.
"""

from __future__ import annotations
import os, re, json, csv, argparse, gzip
from collections import Counter
from typing import Iterable, Tuple, Dict, Optional, List

FA_EXTS = (".fa", ".fasta", ".fna")
GZ_EXTS = tuple(ext + ".gz" for ext in FA_EXTS)
ALL_FASTA_EXTS = FA_EXTS + GZ_EXTS

# Accession detector (GB/RefSeq style, with optional version)
ACC_FULL = re.compile(r'(?:[A-Z]{1,2}\d{5,6}|[A-Z]{3}\d{5}|NC_\d+|[A-Z]{2}_\d+)(?:\.\d+)?')

def _open_text(path: str):
    return gzip.open(path, "rt") if path.lower().endswith(".gz") else open(path, "rt", encoding="utf-8", errors="ignore")

def _norm_accession(token_or_header: str) -> Optional[str]:
    """
    Extract an accession-like ID and strip version (e.g., AB016785.1 -> AB016785).
    Returns None if nothing accession-like is found.
    """
    s = token_or_header.strip()
    parts = s.split('|') if '|' in s else [s]
    cand = None
    for p in parts:
        p = p.strip()
        m = ACC_FULL.fullmatch(p) or ACC_FULL.search(p)
        if m:
            cand = m.group(0)
            break
    if cand is None:
        tok = s.split()[0]
        m = ACC_FULL.fullmatch(tok) or ACC_FULL.search(tok)
        if m:
            cand = m.group(0)
    if cand is None:
        return None
    if '.' in cand:
        cand = cand.split('.', 1)[0]
    return cand

def _parse_label_from_header_exact(header: str) -> Optional[str]:
    """
    VERBATIM label extraction:
      label = (first whitespace token).split('.', 1)[0]
    No normalization.
    """
    tok0 = header.strip().split()[0] if header.strip() else ""
    if not tok0:
        return None
    label = tok0.split('.', 1)[0]
    return label if label else None

def _iter_fasta_headers(path: str) -> Iterable[Tuple[str, str]]:
    """Yield (id, header) from a single FASTA file (header starts after '>')."""
    with _open_text(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            sid = _norm_accession(header) or header.split()[0]
            yield sid, header

def _resolve_fasta(dataset: Optional[str], fasta: Optional[str]) -> Tuple[str, str]:
    if fasta:
        if not os.path.isfile(fasta):
            raise FileNotFoundError(f"--fasta not found: {fasta}")
        out_dir = os.path.dirname(os.path.abspath(fasta))
        return fasta, out_dir
    if dataset:
        for cand in ["sequences.fasta", "sequences.fa", "sequences.fna",
                     "sequences.fasta.gz", "sequences.fa.gz", "sequences.fna.gz"]:
            p = os.path.join(dataset, cand)
            if os.path.isfile(p):
                return p, os.path.abspath(dataset)
        raise FileNotFoundError(f"--dataset given but no sequences.fasta[.gz] in {dataset}")
    raise ValueError("Provide exactly one of --fasta or --dataset")

def _load_whitelist(path: str) -> List[str]:
    """Load whitelist EXACTLY as given (strip newline/outer spaces only)."""
    keep: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            keep.append(s)  # no case/format normalization
    return keep

def main():
    ap = argparse.ArgumentParser(description="Make labels.tsv for HIV-1 pol 2010 (verbatim labels; no normalization).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", default=None, help="Path to the HIV-1 pol 2010 FASTA file.")
    src.add_argument("--dataset", default=None, help="Folder containing sequences.fasta[.gz].")
    ap.add_argument("--whitelist", required=True, help="Text file with allowed labels (exact strings).")
    ap.add_argument("--min", type=int, default=18, help="Minimum sequences per class (default: 18).")
    ap.add_argument("--out-dir", default=None, help="Where to write outputs (default: FASTA folder or dataset).")
    args = ap.parse_args()

    fasta_path, inferred_out = _resolve_fasta(args.dataset, args.fasta)
    out_dir = os.path.abspath(args.out_dir or inferred_out)
    os.makedirs(out_dir, exist_ok=True)

    whitelist = _load_whitelist(args.whitelist)
    whitelist_set = set(whitelist)  # exact-match set

    id2label_raw: Dict[str, str] = {}
    unmatched_headers: List[str] = []
    unmatched_labels: Counter[str] = Counter()

    # Extract labels exactly as written
    for sid, header in _iter_fasta_headers(fasta_path):
        lab = _parse_label_from_header_exact(header)
        if lab is None:
            unmatched_headers.append(header)
            unmatched_labels["<missing>"] += 1
            continue
        if lab not in whitelist_set:
            unmatched_headers.append(header)
            unmatched_labels[lab] += 1
            continue
        if sid not in id2label_raw:
            id2label_raw[sid] = lab  # verbatim

    # Apply >= min
    counts_all = Counter(id2label_raw.values())
    keep_labels = {c for c, n in counts_all.items() if n >= args.min}
    kept_pairs = [(sid, lab) for sid, lab in id2label_raw.items() if lab in keep_labels]

    # Write labels.tsv
    out_tsv = os.path.join(out_dir, "labels.tsv")
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "label"])
        for sid, lab in kept_pairs:
            w.writerow([sid, lab])

    # Write report
    n_headers_seen = sum(1 for _ in _iter_fasta_headers(fasta_path))
    report = {
        "dataset": args.dataset if args.dataset else None,
        "fasta": fasta_path,
        "out_dir": out_dir,
        "n_headers_seen": n_headers_seen,
        "n_labeled_parsed": len(id2label_raw),
        "n_labels_total_before_min": len(counts_all),
        "class_counts_before_min": dict(counts_all),
        "min_per_class": args.min,
        "n_labels_after_min": len(keep_labels),
        "class_counts_after_min": {c: counts_all[c] for c in sorted(keep_labels)},
        "whitelist_count": len(whitelist),
        "whitelist_path": os.path.abspath(args.whitelist),
        "unmatched_count": len(unmatched_headers),
        "unmatched_labels_example_counts": dict(unmatched_labels.most_common(25)),
        "notes": "Labels compared verbatim (no normalization). Label = first header token before first '.'.",
    }
    with open(os.path.join(out_dir, "labels_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Write unmatched headers for audit
    with open(os.path.join(out_dir, "labels_unmatched.txt"), "w", encoding="utf-8") as fh:
        for h in unmatched_headers:
            fh.write(h + "\n")

    print(f"[OK] Wrote:\n  {out_tsv}\n  {os.path.join(out_dir,'labels_report.json')}\n  {os.path.join(out_dir,'labels_unmatched.txt')}")

if __name__ == "__main__":
    main()
