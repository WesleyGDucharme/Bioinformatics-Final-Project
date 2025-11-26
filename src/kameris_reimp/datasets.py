#!/usr/bin/env python3
"""
datasets.py

Central loader for all datasets used in the Kameris reproduction.

Given the final data layout, this module does exactly three things:

1. Locate the FASTA files for a named dataset (per-file layout or a single
   multi-FASTA).
2. Read the sequences and assign each sample a stable ID:
      - Per-file layout:    ID = file stem (e.g. "AB016785" from "AB016785.fasta")
      - hiv1_web_pol2010:   ID = accession parsed from the FASTA header
3. Read labels.tsv in the dataset directory and intersect IDs so that we
   return only samples that have both a sequence and a label.

Authors: Wesley Ducharme
Date: 2025-11-19
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional



# Data root detection
# -------------------

def _detect_data_root() -> Path:
    """
    Walk up from this file until we find a parent that contains a 'data' dir.
    Falls back to './data' if nothing is found (useful for tests).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    return Path("data")


DATA_ROOT: Path = _detect_data_root()



# Dataset specifications
# ----------------------

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    root: Path
    fasta_subdir: Optional[str] = None  # for per-file layout
    fasta_file: Optional[str] = None    # for single multi-FASTA file
    labels_file: str = "labels.tsv"


DATASETS: Dict[str, DatasetSpec] = {
    # HIV-1 LANL whole genomes
    "hiv1_lanl_whole": DatasetSpec(
        key="hiv1_lanl_whole",
        root=DATA_ROOT / "hiv1_lanl_whole",
        fasta_subdir="lanl-whole",
    ),

    # HIV-1 LANL pol genes (full set)
    "hiv1_lanl_pol": DatasetSpec(
        key="hiv1_lanl_pol",
        root=DATA_ROOT / "hiv1_lanl_pol",
        fasta_subdir="lanl-pol",
    ),

    # Mixed benchmark pol fragments
    "hiv1_benchmark_mixed_polfragments": DatasetSpec(
        key="hiv1_benchmark_mixed_polfragments",
        root=DATA_ROOT / "hiv1_benchmark_mixed_polfragments",
        fasta_subdir="mixed-polfragments",
    ),

    # Synthetic pol fragments
    "hiv1_synthetic_polfragments": DatasetSpec(
        key="hiv1_synthetic_polfragments",
        root=DATA_ROOT / "hiv1_synthetic_polfragments",
        fasta_subdir="synthetic-polfragments",
    ),

    # 2010 Web pol alignment (single multi-FASTA)
    "hiv1_web_pol2010": DatasetSpec(
        key="hiv1_web_pol2010",
        root=DATA_ROOT / "hiv1_web_pol2010",
        fasta_file="sequences.fasta",
    ),

    # Dengue whole genomes
    "dengue_whole": DatasetSpec(
        key="dengue_whole",
        root=DATA_ROOT / "dengue_whole",
        fasta_subdir="dengue-whole",
    ),

    # HBV whole genomes
    "hbv_whole": DatasetSpec(
        key="hbv_whole",
        root=DATA_ROOT / "hbv_whole",
        fasta_subdir="ibcp-whole-hbv",
    ),

    # HCV whole genomes
    "hcv_whole": DatasetSpec(
        key="hcv_whole",
        root=DATA_ROOT / "hcv_whole",
        fasta_subdir="lanl-whole-hcv",
    ),

    # Influenza A genomes (assembled from metadata.json; we already built labels.tsv)
    "flu_a": DatasetSpec(
        key="flu_a",
        root=DATA_ROOT / "flu_a",
        fasta_subdir="fasta",
    ),
}



# FASTA helpers
# -------------

FA_EXTS = (".fa", ".fasta", ".fna")


def _clean_seq(seq: str) -> str:
    """
    Basic nucleotide clean-up: keep A/C/G/T/N, uppercase, drop everything else.
    """
    import re as _re
    return _re.sub(r"[^ACGTNacgtn]", "N", seq).upper()


def _read_single_fasta_file(path: Path) -> str:
    """
    Read a one-record FASTA file and return the concatenated sequence.
    We ignore the header content here – the *ID* will be derived from the
    file name (stem) for per-file datasets.
    """
    seq_chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq_chunks.append(line.strip())
    return _clean_seq("".join(seq_chunks))


def _read_per_file_fasta(root: Path) -> Dict[str, str]:
    """
    Recursively read all FASTA files under `root`, assuming exactly one record
    per file. The sample ID is the file stem (without extension).
    """
    out: Dict[str, str] = {}
    for dirpath, _dirs, files in os_walk_str(root):
        for fname in files:
            p = Path(dirpath) / fname
            if p.suffix.lower() in FA_EXTS:
                sid = p.stem  # <-- ID convention for per-file datasets
                out[sid] = _read_single_fasta_file(p)
    return out


# For the 2010 Web pol alignment we need accession IDs from headers.
ACC_FULL = re.compile(r"(?:[A-Z]{1,2}\d{5,6}|[A-Z]{3}\d{5}|NC_\d+|[A-Z]{2}_\d+)(?:\.\d+)?")


def _norm_accession_from_header(header: str) -> Optional[str]:
    """
    Extract a GenBank/RefSeq-like accession from a full FASTA header and
    drop any version suffix (e.g. AB016785.1 -> AB016785).
    Returns None if nothing accession-like is found.
    """
    s = header.strip()

    parts = s.split("|") if "|" in s else [s]
    cand = None
    for part in parts:
        part = part.strip()
        m = ACC_FULL.fullmatch(part) or ACC_FULL.search(part)
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

    if "." in cand:
        cand = cand.split(".", 1)[0]
    return cand


def _read_multi_fasta_with_accession_ids(path: Path) -> Dict[str, str]:
    """
    Read a multi-FASTA file (used for hiv1_web_pol2010) and use the
    accession parsed from the header as the sample ID. This matches the
    IDs we used when building labels.tsv for that dataset.
    """
    out: Dict[str, str] = {}
    header: Optional[str] = None
    seq_chunks: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                # flush previous record
                if header is not None:
                    sid = _norm_accession_from_header(header) or header.split()[0]
                    out[sid] = _clean_seq("".join(seq_chunks))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())

    # flush last record
    if header is not None:
        sid = _norm_accession_from_header(header) or header.split()[0]
        out[sid] = _clean_seq("".join(seq_chunks))

    return out


# Small wrapper so we don't import os at the top just for walk()
def os_walk_str(root: Path):
    import os
    return os.walk(str(root))



# Labels helpers
# --------------

def _read_labels_tsv(path: Path) -> Dict[str, str]:
    """
    Read labels.tsv (id<TAB>label) into a dict[id] -> label.
    """
    if not path.is_file():
        raise FileNotFoundError(f"labels file not found: {path}")

    id2label: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path} must have 'id' and 'label' columns")
        for row in reader:
            sid = row["id"].strip()
            lab = row["label"].strip()
            if not sid or not lab:
                continue
            id2label[sid] = lab
    return id2label



# Public dataset object + loader
# ------------------------------
@dataclass
class Dataset:
    key: str
    ids: List[str]
    sequences: List[str]
    labels: List[str]
    classes: List[str]

    @property
    def n_samples(self) -> int:
        return len(self.ids)


def list_dataset_keys() -> List[str]:
    """Return the list of available dataset keys."""
    return sorted(DATASETS.keys())


def load_dataset(key: str) -> Dataset:
    """
    Load one dataset by key. Returns a Dataset object with:

      - ids:       list of sample IDs (strings)
      - sequences: list of nucleotide sequences (A/C/G/T/N)
      - labels:    subtype / genotype / serotype labels (strings)
      - classes:   sorted list of unique labels present

    Raises informative errors if:
      - The dataset key is unknown
      - FASTA files cannot be found
      - labels.tsv is missing
      - There is no intersection between sequences and labels
    """
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset key '{key}'. "
                       f"Known keys: {', '.join(list_dataset_keys())}")

    spec = DATASETS[key]

    # --- labels ---
    labels_path = spec.root / spec.labels_file
    id2label = _read_labels_tsv(labels_path)

    # --- sequences ---
    if spec.fasta_file is not None:
        fasta_path = spec.root / spec.fasta_file
        if not fasta_path.is_file():
            raise FileNotFoundError(f"FASTA file not found for dataset '{key}': {fasta_path}")
        id2seq = _read_multi_fasta_with_accession_ids(fasta_path)
    elif spec.fasta_subdir is not None:
        fasta_root = spec.root / spec.fasta_subdir
        if not fasta_root.is_dir():
            raise FileNotFoundError(f"FASTA subdir not found for dataset '{key}': {fasta_root}")
        id2seq = _read_per_file_fasta(fasta_root)
    else:
        raise RuntimeError(f"DatasetSpec for '{key}' has neither fasta_file nor fasta_subdir set.")

    # --- intersection ---
    seq_ids = set(id2seq.keys())
    label_ids = set(id2label.keys())
    common = sorted(seq_ids & label_ids)

    if not common:
        raise ValueError(
            f"No overlapping IDs between sequences and labels for dataset '{key}'.\n"
            f"  #seq_ids={len(seq_ids)}, #label_ids={len(label_ids)}\n"
            "Check that your labels.tsv 'id' column uses the same convention as the FASTA IDs "
            "(for per-file datasets: file stem; for hiv1_web_pol2010: accession)."
        )

    sequences = [id2seq[sid] for sid in common]
    labels = [id2label[sid] for sid in common]
    classes = sorted(set(labels))

    return Dataset(
        key=key,
        ids=common,
        sequences=sequences,
        labels=labels,
        classes=classes,
    )
