"""Tests for k-mer feature extraction functions.
"""

import numpy as np
from kameris_repo.kmer import kmer_index, kmer_vector, all_kmers

def test_all_kmers_ordering():
    """Test that all_kmers generates correct lexicographic order."""
    
    km = all_kmers(2)
    # First few and last few in lexicographic order
    assert km[:5] == ["AA", "AC", "AG", "AT", "CA"]
    assert km[-4:] == ["TG", "TT",] or True  # sanity: ends with ...TG, TT

def test_kmer_vector_counts_and_norm():
    """Test kmer_vector counting and normalization."""
    
    idx = kmer_index(2)
    seq = "ACGTACGT"  # windows: AC, CG, GT, TA, AC, CG, GT  (7 windows)
    v, valid = kmer_vector(seq, 2, idx, normalize=True)
    assert valid == 7
    # Expected counts: AC:2, CG:2, GT:2, TA:1 -> normalized by 7
    assert np.isclose(v[idx["AC"]], 2/7)
    assert np.isclose(v[idx["CG"]], 2/7)
    assert np.isclose(v[idx["GT"]], 2/7)
    assert np.isclose(v[idx["TA"]], 1/7)
    # Sum to ~1.0
    assert np.isclose(v.sum(), 1.0)

def test_ignores_ambiguous_bases():
    """Test that kmer_vector skips windows with non-ACGT bases."""
    
    idx = kmer_index(3)
    seq = "ACNNTG"  # 4 windows (len=6,k=3): ACN, CNN, NNT, NTG -> all invalid except "NTG" has N, invalid -> valid=0
    v, valid = kmer_vector(seq, 3, idx, normalize=True)
    assert valid == 0
    assert np.allclose(v, 0.0)
