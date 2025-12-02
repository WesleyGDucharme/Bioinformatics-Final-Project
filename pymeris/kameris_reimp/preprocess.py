"""Preprocessing module: scaling and dimensionality reduction via SVD.
Implements a Preprocessor class that can scale data and reduce its dimensionality
using Truncated SVD.

Authors: Wesley Ducharme
Date: 2025-11-11

"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

ReductionRule = Literal["avg_nnz", "fraction_of_avg_nnz", "fixed"]

def avg_nnz_per_row(X) -> float:
    """Average number of non-zero entries per sample vector."""
    if sparse.issparse(X):
        row_nnzs = np.diff(X.tocsr().indptr)
        return float(row_nnzs.mean()) if row_nnzs.size else 0.0
    return float(np.count_nonzero(X, axis=1).mean()) if X.size else 0.0

def choose_n_components(X_raw, rule: ReductionRule, fraction: float = 0.10, fixed_k: Optional[int] = None) -> int:
    """
    Dimension rules from the paper (configurable), computed on the *original*
    k-mer matrix (before scaling) to preserve sparsity information:
    - 'avg_nnz': n_components = round(average # of nonzeros per row)
    - 'fraction_of_avg_nnz': round(fraction * average # of nonzeros)
    - 'fixed': use fixed_k (sanity-capped)
    Returns the chosen number of components.
    """
    m, d = X_raw.shape
    if rule == "avg_nnz":
        n = int(round(avg_nnz_per_row(X_raw)))
    elif rule == "fraction_of_avg_nnz":
        n = int(round(fraction * avg_nnz_per_row(X_raw)))
    elif rule == "fixed":
        if fixed_k is None:
            raise ValueError("fixed_k must be set when rule='fixed'")
        n = int(fixed_k)
    else:
        raise ValueError(f"unknown rule: {rule}")
    # cap to valid SVD range
    n = max(1, min(n, min(m, d) - 1)) if min(m, d) > 1 else 1
    return n

@dataclass
class Preprocessor:
    """Data preprocessor: scaling and dimensionality reduction via SVD.
    Configurable options for scaling and SVD reduction.
    Usage:
      pre = Preprocessor(scale=True, svd_rule="avg_nnz", svd_fraction
      pre.fit(X_train)
      X_train_proc = pre.transform(X_train)
      X_test_proc = pre.transform(X_test)
    """

    def __init__(self, *, scale: bool = True, svd_rule: Optional[str] = None, svd_fraction: float = 0.1, svd_fixed_k: Optional[int] = None, random_state: Optional[int] = 42,) -> None:
        """Initialize the Preprocessor with given options."""
        self.scale = scale
        self.svd_rule = svd_rule
        self.svd_fraction = svd_fraction
        self.svd_fixed_k = svd_fixed_k
        self.random_state = random_state

        self._scaler = None
        self._svd = None
        self._n_components_ = None  # set in fit() if SVD is used

    def _decide_k(self, X) -> Optional[int]:
        """Decide number of SVD components based on rule and data X."""
        if self.svd_rule is None:
            return None

        n_samples, n_features = X.shape

        if self.svd_rule == "fraction_of_avg_nnz":
            if sparse.issparse(X):
                # nnz per row (CSR/CSC safe)
                row_nnzs = np.diff(X.tocsr().indptr)
            else:
                # dense path
                row_nnzs = np.count_nonzero(X, axis=1)
            avg_nnz = float(np.mean(row_nnzs))
            k = int(np.floor(self.svd_fraction * avg_nnz))
        elif self.svd_rule in ("fixed_k", "fixed"):
            if self.svd_fixed_k is None:
                raise ValueError("svd_rule='fixed_k' requires svd_fixed_k")
            k = int(self.svd_fixed_k)
        else:
            raise ValueError(f"Unknown svd_rule: {self.svd_rule!r}")

        # Guard rails for sklearn TruncatedSVD
        # must be 1 <= k < min(n_samples, n_features)
        k = max(1, k)
        k = min(k, n_features - 1, n_samples - 1)
        return k

    def fit(self, X: np.ndarray) -> "Preprocessor":
        """Fit the preprocessor to data X."""
        Z = X
        if self.scale:
            # with_mean=False keeps sparsity
            self._scaler = StandardScaler(with_mean=False, with_std=True, copy=True)
            Z = self._scaler.fit_transform(Z)
        if self.svd_rule:
            k = choose_n_components(X, self.svd_rule, self.svd_fraction, self.svd_fixed_k)
            self._svd = TruncatedSVD(n_components=k, algorithm="randomized",
                                     random_state=self.random_state)
            Z = self._svd.fit_transform(Z)
            self._n_components_ = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data X using the fitted preprocessor."""
        Z = X
        if self._scaler is not None:
            Z = self._scaler.transform(Z)
        if self._svd is not None:
            Z = self._svd.transform(Z)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data X in one step."""
        self.fit(X)
        return self.transform(X)
