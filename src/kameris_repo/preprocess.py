"""Preprocessing module: scaling and dimensionality reduction via SVD.
Implements a Preprocessor class that can scale data and reduce its dimensionality
using Truncated SVD, based on configurable rules.

Authors: Wesley Ducharme
Date: 2025-11-11

"""

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

# Note: avoid importing scikit-learn / scipy at module import time so tests
# can run in environments where those binary dependencies are not available
# or are incompatible with the installed NumPy. We provide lightweight
# NumPy-based scaling and SVD implementations used by the Preprocessor.

ReductionRule = Literal["avg_nnz", "fraction_of_avg_nnz", "fixed"]

def avg_nnz_per_row(X: np.ndarray) -> float:
    """Average number of non-zero entries per sample vector."""
    # Count nonzeros per row, average (float).
    return np.count_nonzero(X, axis=1).mean()

def choose_n_components(X: np.ndarray, rule: ReductionRule, fraction: float = 0.10, fixed_k: Optional[int] = None) -> int:
    """
    Paper-driven dimension rule (configurable):
    - 'avg_nnz': n_components = round(average # of nonzeros per row)
    - 'fraction_of_avg_nnz': round(fraction * average # of nonzeros)
    - 'fixed': use fixed_k (sanity-capped)
    Returns the chosen number of components.
    """
    m, d = X.shape
    if rule == "avg_nnz":
        n = int(round(avg_nnz_per_row(X)))
    elif rule == "fraction_of_avg_nnz":
        n = int(round(fraction * avg_nnz_per_row(X)))
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
    scale: bool = True # whether to standard scale features
    svd_rule: Optional[ReductionRule] = "avg_nnz"  # None disables SVD reduction
    svd_fraction: float = 0.10 # only used if svd_rule=="fraction_of_avg_nnz"
    svd_fixed_k: Optional[int] = None  # only used if svd_rule=="fixed"
    random_state: int = 42 

    # fitted objects
    # _scaler will be a dict with 'mean' and 'scale' when fitted
    _scaler: Optional[dict] = None
    # _svd will be a dict with 'components_' (Vt) and 'explained_' (singular values)
    _svd: Optional[dict] = None
    _n_components_: Optional[int] = None # number of SVD components after fitting

    def fit(self, X: np.ndarray) -> "Preprocessor":
        """Fit the preprocessor to data X."""
        Z = X
        if self.scale:
            # compute column means and std (population std, ddof=0) like sklearn
            mean = np.mean(Z, axis=0)
            scale = np.std(Z, axis=0)
            # avoid division by zero
            scale[scale == 0.0] = 1.0
            self._scaler = {"mean": mean, "scale": scale}
            Z = (Z - mean) / scale
        if self.svd_rule:
            k = choose_n_components(Z, self.svd_rule, self.svd_fraction, self.svd_fixed_k)
            # compute thin SVD via numpy; for shape (n_samples, n_features)
            # np.linalg.svd returns U, s, Vt where U @ diag(s) @ Vt == Z
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            # store components (Vt) and singular values
            self._svd = {"components_": Vt, "singular_values_": s}
            # project to k components: U[:, :k] * s[:k]
            Z = U[:, :k] * s[:k]
            self._n_components_ = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data X using the fitted preprocessor."""
        Z = X
        if self._scaler is not None:
            mean = self._scaler["mean"]
            scale = self._scaler["scale"]
            Z = (Z - mean) / scale
        if self._svd is not None:
            # project using stored components: components_ is Vt
            Vt = self._svd["components_"]
            k = self._n_components_ if self._n_components_ is not None else Vt.shape[0]
            # projection onto first k components: X @ Vt.T[:, :k]
            Z = Z @ Vt.T[:, :k]
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data X in one step."""
        self.fit(X)
        return self.transform(X)
