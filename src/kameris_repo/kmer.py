"""K-mer featurizer for DNA sequences.

Provides functions to convert DNA sequences into k-mer frequency vectors
and matrices.
A k-mer is a substring of length k composed of the characters A, C, G, and T.
A k-mer frequency vector counts occurrences of each possible k-mer in a sequence.
Example:
  k = 2
  Sequence: "ACGTACGT"
  Possible 2-mers: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
  Observed 2-mers (with repeats): "AC", "CG", "GT", "TA", "AC", "CG", "GT"
  Frequency vector (A,C,G,T order): [2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (16 elements for 4^2=16 possible 2-mers)
  
  Normalized vector: [0.2857, 0.2857, 0.2857, 0.1429]
  This normalized vecotor shows that "AC", "CG", and "GT" each make up about 28.57% of the total 2-mers, while "TA" makes up about 14.29%.
"""

from itertools import product
from typing import Dict, Tuple, Iterable, List
import numpy as np

ALPHABET = ("A", "C", "G", "T")
ALPHABET_SET = set(ALPHABET)

def all_kmers(k: int) -> List[str]:
    """Lexicographic A/C/G/T k-mers."""
    return ["".join(p) for p in product(ALPHABET, repeat=k)]

def kmer_index(k: int) -> Dict[str, int]:
    """Map each k-mer to a column index [0..4^k-1] in lexicographic order."""
    return {kmer: i for i, kmer in enumerate(all_kmers(k))}

def kmer_vector(seq: str, k: int, index: Dict[str, int], normalize: bool = True) -> Tuple[np.ndarray, int]:
    """
    Convert a DNA sequence to a k-mer frequency vector.
    - Skips any window containing non-ACGT.
    - If normalize=True, divides by number of valid windows (so the vector sums to 1.0 when valid>0).
    Returns: (vector, valid_windows)
    """
    seq_u = seq.upper()
    V = np.zeros(len(index), dtype=np.float64)
    L = len(seq_u)
    if L < k:
        return V, 0

    valid = 0
    for i in range(L - k + 1):
        kmer = seq_u[i:i + k]
        if set(kmer) <= ALPHABET_SET:
            V[index[kmer]] += 1.0
            valid += 1

    if normalize and valid > 0:
        V /= float(valid)
    return V, valid

def batch_kmer_matrix(seqs: Iterable[str], k: int, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, int], List[int]]:
    """
    Vectorize multiple sequences into an (n_samples, 4^k) dense matrix.
    Returns: (X, index_map, valid_counts_per_seq)
    """
    idx = kmer_index(k)
    X = np.zeros((len(seqs), len(idx)), dtype=np.float64)
    valids: List[int] = []
    for r, s in enumerate(seqs):
        v, valid = kmer_vector(s, k, idx, normalize=normalize)
        X[r, :] = v
        valids.append(valid)
    return X, idx, valids
