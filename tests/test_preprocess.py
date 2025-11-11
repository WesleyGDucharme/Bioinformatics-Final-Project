"""Unit tests for the preprocess module.
Tests for average non-zero calculation, component selection rules,
and the Preprocessor class functionality.
Authors: Wesley Ducharme
Date: 2025-11-11
"""

import numpy as np
from kameris_repo.preprocess import Preprocessor, avg_nnz_per_row, choose_n_components

def test_avg_nnz_and_choose_components():
    """Test average non-zero calculation and component selection rules."""
    # 3 samples, 6 dims
    X = np.array([
        [1,0,2,0,0,3],   # 3 nnz
        [0,0,0,0,0,0],   # 0 nnz
        [4,5,0,0,1,0],   # 3 nnz
    ], dtype=float)
    avg = avg_nnz_per_row(X)
    assert np.isclose(avg, (3+0+3)/3)  # = 2.0
    assert choose_n_components(X, "avg_nnz") == 2
    assert choose_n_components(X, "fraction_of_avg_nnz", fraction=0.5) == 1
    expected = min(3, min(X.shape) - 1)  # min(3, 3-1) = 2
    assert choose_n_components(X, "fixed", fixed_k=3) == expected

def test_scale_then_svd_runs_and_shapes():
    """Test that Preprocessor with scaling and SVD runs and produces expected shapes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 100))
    pp = Preprocessor(scale=True, svd_rule="fraction_of_avg_nnz", svd_fraction=0.1, random_state=0)
    Z = pp.fit_transform(X)
    # ~10% of avg_nnz; for dense normal data avg_nnz≈100, so ~10 comps
    assert 5 <= Z.shape[1] <= 15
    assert Z.shape[0] == X.shape[0]
