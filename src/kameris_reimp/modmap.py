# src/kameris_repo/modmap.py
"""
MoDMap-style helpers: distance matrices and low-dimensional embeddings
from k-mer frequency vectors.

Author: Wesley Ducharme
Date: 2025-11-23

This module is for reproducing the MoDMap figures (e.g., Fig. 2–5) from
Solis-Reyes et al. for this reimplementation.

Key ideas:
- Compute normalized k-mer feature vectors (using kmer.py).
- Build a pairwise distance matrix (Manhattan distance, as in the paper).
- Apply classical MDS to get 2D/3D coordinates suitable for plotting.

Usage (for a dataset already loaded via datasets.py):

    from kameris_repo import datasets, modmap

    ds = datasets.load_dataset("hiv1_lanl_whole")
    seqs = ds.sequences
    labels = ds.labels

    coords, D = modmap.kmer_modmap_from_sequences(
        seqs, k=5, metric="manhattan", n_components=3
    )

    # coords is an (n_samples, 3) array of embedding coordinates

"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Callable, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from . import kmer, datasets as ds_mod



# Core math: classical MDS on a distance matrix

def classical_mds(
    D: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """
    Classical MDS (a.k.a. Torgerson–Gower scaling) on a distance matrix.

    Parameters
    ----------
    D : (n, n) ndarray
        Symmetric distance matrix (e.g., Manhattan distances between
        normalized k-mer frequency vectors).
    n_components : int, default=3
        Target embedding dimensionality (2 or 3 for plots).

    Returns
    -------
    coords : (n, n_components) ndarray
        Low-dimensional coordinates. Each row corresponds to one sample.
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")

    n = D.shape[0]
    # Double-centering
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * J @ D2 @ J  # Gram matrix

    # Eigen-decomposition (symmetric matrix -> eigh)
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort eigenvalues/vectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take the top n_components dimensions with non-negative eigenvalues
    k = min(n_components, eigvals.shape[0])
    lam = np.clip(eigvals[:k], a_min=0.0, a_max=None)
    # coords = V * sqrt(Lambda)
    coords = eigvecs[:, :k] * np.sqrt(lam)

    return coords



# K-mer → distance → embedding helpers

def kmer_distance_matrix(
    seqs: Iterable[str],
    k: int = 5,
    metric: str = "manhattan",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-mer frequency vectors and a pairwise distance matrix.

    Parameters
    ----------
    seqs : iterable of str
        Nucleotide sequences (already filtered as desired).
    k : int, default=5
        k-mer length (paper uses k=6; you are standardizing on k=5).
    metric : str, default="manhattan"
        Distance metric for pairwise_distances (e.g. "manhattan", "euclidean").
        The paper uses Manhattan distance.
    normalize : bool, default=True
        Passed through to kmer.batch_kmer_matrix to get normalized frequencies.

    Returns
    -------
    D : (n_samples, n_samples) ndarray
        Pairwise distance matrix.
    X : (n_samples, 4**k) ndarray
        K-mer feature matrix used to compute distances.
    """
    # Ensure we have a concrete list so len() works
    seq_list: List[str] = list(seqs)
    if not seq_list:
        raise ValueError("kmer_distance_matrix: 'seqs' is empty")

    X, index_map, valids = kmer.batch_kmer_matrix(seq_list, k=k, normalize=normalize)

    # pairwise_distances returns a dense (n, n) matrix
    D = pairwise_distances(X, metric=metric)

    return D, X


def kmer_modmap_from_sequences(
    seqs: Iterable[str],
    *,
    k: int = 5,
    metric: str = "manhattan",
    n_components: int = 3,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper: sequences -> k-mer matrix -> distance matrix -> MDS coords.

    Parameters
    ----------
    seqs : iterable of str
        Input sequences (already filtered to the subset you care about:
        e.g., just pure subtypes).
    k : int, default=5
        K-mer length.
    metric : str, default="manhattan"
        Distance metric for pairwise distances.
    n_components : int, default=3
        Target embedding dimension (2D or 3D).
    normalize : bool, default=True
        Whether to use normalized k-mer frequencies (recommended).

    Returns
    -------
    coords : (n_samples, n_components) ndarray
        Low-dimensional embedding coordinates.
    D : (n_samples, n_samples) ndarray
        Pairwise distance matrix (same order as coords).
    """
    D, X = kmer_distance_matrix(seqs, k=k, metric=metric, normalize=normalize)
    coords = classical_mds(D, n_components=n_components)
    return coords, D


# dataset-level helper (so don’t re-write boilerplate)

def modmap_for_dataset(
    dataset_key: str,
    *,
    k: int = 5,
    metric: str = "manhattan",
    n_components: int = 3,
    label_filter: Optional[Callable[[str], bool]] = None,
):
    """
    Build a MoDMap embedding for one of your registered datasets.

    Parameters
    ----------
    dataset_key : str
        One of the keys from datasets.list_dataset_keys(), e.g. "hiv1_lanl_whole".
    k : int, default=5
        K-mer length.
    metric : str, default="manhattan"
        Distance metric.
    n_components : int, default=3
        Embedding dimension.
    label_filter : callable or None
        If provided, it will be called on each label and only samples
        where label_filter(label) is True will be kept. This is where
        you’ll plug in the “pure subtypes only” rule for Fig. 2.

    Returns
    -------
    coords : (n_kept, n_components) ndarray
        Embedding coordinates.
    labels_kept : list of str
        Labels in the same order as coords.
    indices_kept : list of int
        Indices into the original dataset (0-based) that were kept.
    """
    ds = ds_mod.load_dataset(dataset_key)
    seqs = ds.sequences
    labels = ds.labels

    if label_filter is None:
        mask = [True] * len(seqs)
    else:
        mask = [bool(label_filter(lbl)) for lbl in labels]

    seqs_kept: List[str] = []
    labels_kept: List[str] = []
    indices_kept: List[int] = []

    for i, (s, lab, keep) in enumerate(zip(seqs, labels, mask)):
        if keep:
            seqs_kept.append(s)
            labels_kept.append(lab)
            indices_kept.append(i)

    if not seqs_kept:
        raise ValueError(
            f"modmap_for_dataset: label_filter removed all samples for dataset '{dataset_key}'"
        )

    coords, D = kmer_modmap_from_sequences(
        seqs_kept,
        k=k,
        metric=metric,
        n_components=n_components,
        normalize=True,
    )
    return coords, labels_kept, indices_kept
