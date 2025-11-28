"""Tests for models and cross-validation functionality.
Authors: Wesley Ducharme
Date: 2025-11-11
"""

import numpy as np
from sklearn.datasets import make_classification

from kameris_reimp.models import make_model
from kameris_reimp.cv import cross_validate
from kameris_reimp.preprocess import Preprocessor

def test_model_registry_basic():
    """ Test that model factory returns models for known names."""
    names = [
        "knn_10", "nearest_centroid_mean", "nearest_centroid_median",
        "logreg", "linear_svm", "fquadratic_svm", "cubic_svm",
        "sgd", "decision_tree", "random_forest", "adaboost",
        "gnb", "lda", "qda", "mlp",
    ]
    for n in names:
        assert make_model(n) is not None

def test_cv_runner_no_svd_dense_data():
    """Test cross-validation runner without SVD on dense data."""
    # simple, linearly separable data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                               n_redundant=5, n_classes=3, random_state=0)
    pp = Preprocessor(scale=True, svd_rule=None)  # disable SVD for this test
    res = cross_validate(X, y, model_name="linear_svm", preprocessor=pp, n_splits=5, random_state=0)
    assert 0.0 <= res.mean_accuracy <= 1.0
    assert len(res.fold_accuracies) == 5
    # confusion matrix shape matches #classes
    assert res.confusion_matrix.shape == (3, 3)
    assert res.confusion_matrix.sum() == len(y)

def test_cv_runner_with_svd():
    """Test cross-validation runner with SVD on dense data."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 200))
    y = np.repeat([0,1,2], repeats=40)
    pp = Preprocessor(scale=True, svd_rule="fraction_of_avg_nnz", svd_fraction=0.1, random_state=123)
    res = cross_validate(X, y, model_name="logreg", preprocessor=pp, n_splits=6, random_state=123)
    assert len(res.n_components_per_fold) == 6
    assert all(k >= 1 for k in res.n_components_per_fold)
