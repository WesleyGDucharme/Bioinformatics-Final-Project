"""
Cross-validation functionality for classification models.
Authors: Wesley Ducharme
Date: 2025-11-11
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from .preprocess import Preprocessor
from .models import make_model

@dataclass
class CVResult:
    """Cross-validation result summary.
    Attributes:
      mean_accuracy: Average accuracy over all folds.
      fold_accuracies: List of accuracies per fold.
      fit_time_mean: Average model fit time per fold (seconds).
      predict_time_mean: Average model predict time per fold (seconds).
      n_components_per_fold: List of number of SVD components used per fold.
      confusion_matrix: Aggregated confusion matrix over all folds.
      classes_: Array of class labels corresponding to confusion matrix rows/cols.
      random_state: Random state used for CV splitting.
    """
    mean_accuracy: float
    fold_accuracies: List[float]
    fit_time_mean: float
    predict_time_mean: float
    n_components_per_fold: List[int]
    confusion_matrix: np.ndarray
    classes_: np.ndarray
    random_state: int

def _clone_preprocessor(pp: Preprocessor) -> Preprocessor:
    """Create a fresh Preprocessor instance with the same configuration as pp.
    This ensures no data leakage between CV folds.
    """
    # shallow "config clone" — new instance with same params, fresh state
    return Preprocessor(
        scale=pp.scale,
        svd_rule=pp.svd_rule,
        svd_fraction=pp.svd_fraction,
        svd_fixed_k=pp.svd_fixed_k,
        random_state=pp.random_state,
    )

def cross_validate(X: np.ndarray, y: np.ndarray, model_name: str, preprocessor: Optional[Preprocessor] = None, n_splits: int = 10,
                   random_state: int = 42, timing_repeats: int = 5,) -> CVResult:
    """ Function to perform cross-validation of a classifier model with optional preprocessing.
    Stratified K-fold CV. Per fold:
      - fit scale -> SVD on X_train
      - transform X_train/X_val
      - fit model; time fit and predict
    
    Args:
      X: Feature matrix (n_samples, n_features).
      y: Class labels (n_samples,).
      model_name: Name of the classifier model to use (passed to make_model).
      preprocessor: Optional Preprocessor instance for scaling/SVD. If None, a default Preprocessor is used.
      n_splits: Number of CV folds (default 10).
      random_state: Random state for reproducibility.
      timing_repeats: Number of times to repeat fit/predict timing for averaging.
      
    Returns averaged accuracy and timing; confusion matrix aggregated over all folds.
    """
    if preprocessor is None:
        preprocessor = Preprocessor(scale=True, svd_rule="avg_nnz", random_state=random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    accs: List[float] = []
    fit_times: List[float] = []
    pred_times: List[float] = []
    ncomps: List[int] = []

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    classes = np.unique(y)

    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]

        # fresh preprocessor per fold; fit on train only
        pp = _clone_preprocessor(preprocessor)
        Ztr = pp.fit_transform(Xtr)
        Zval = pp.transform(Xval)

        # pick up chosen k (if SVD enabled)
        k_fold = getattr(pp, "_n_components_", None)
        ncomps.append(int(k_fold) if k_fold else 0)

        # model
        model = make_model(model_name)

        # time fit/predict (optionally repeated)
        fit_elapsed = 0.0
        pred_elapsed = 0.0
        for _ in range(max(1, timing_repeats)):
            t0 = time.perf_counter()
            model.fit(Ztr, ytr)
            fit_elapsed += time.perf_counter() - t0

            t1 = time.perf_counter()
            yhat = model.predict(Zval)
            pred_elapsed += time.perf_counter() - t1

        fit_times.append(fit_elapsed / max(1, timing_repeats))
        pred_times.append(pred_elapsed / max(1, timing_repeats))

        # metrics
        acc = accuracy_score(yval, yhat)
        accs.append(acc)
        y_true_all.extend(yval.tolist())
        y_pred_all.extend(yhat.tolist())

    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    return CVResult(
        mean_accuracy=float(np.mean(accs)),
        fold_accuracies=[float(a) for a in accs],
        fit_time_mean=float(np.mean(fit_times)),
        predict_time_mean=float(np.mean(pred_times)),
        n_components_per_fold=ncomps,
        confusion_matrix=cm,
        classes_=classes,
        random_state=random_state,
    )
