""" 
Classifier model factory function.
Authors: Wesley Ducharme
Date: 2025-11-11
"""

# src/kameris_repro/models.py
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def make_model(name: str):
    """Factory function to create classifier models by name."""
    key = name.lower()
    if key in ("10_nearest_neighbors", "knn_10", "knn"):
        return KNeighborsClassifier(n_neighbors=10, weights="uniform", metric="euclidean")
    if key == "nearest_centroid_mean":
        return NearestCentroid(metric="euclidean")       # mean centroid
    if key == "nearest_centroid_median":
        return NearestCentroid(metric="manhattan")       # median centroid
    if key in ("logistic_regression", "logreg"):
        return LogisticRegression(
            penalty="l2", multi_class="ovr",
            C=1.0, tol=1e-4, solver="lbfgs", max_iter=1000
        )
    if key == "linear_svm":
        return SVC(kernel="linear", C=1.0, tol=1e-3, probability=False, random_state=42)
    if key == "quadratic_svm":
        return SVC(kernel="poly", degree=2, coef0=0.0, C=1.0, tol=1e-3, probability=False, random_state=42)
    if key == "cubic_svm":
        return SVC(kernel="poly", degree=3, coef0=0.0, C=1.0, tol=1e-3, probability=False, random_state=42)
    if key == "sgd":
        return SGDClassifier(loss="hinge", max_iter=5, tol=1e-3, random_state=42)
    if key == "decision_tree":
        return DecisionTreeClassifier(criterion="gini", random_state=42)
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=10, criterion="gini", random_state=42)
    if key == "adaboost":
        return AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=50, algorithm="SAMME.R", random_state=42
        )
    if key in ("gaussian_naive_bayes", "gnb"):
        return GaussianNB()
    if key == "lda":
        return LinearDiscriminantAnalysis()
    if key == "qda":
        return QuadraticDiscriminantAnalysis()
    if key in ("multilayer_perceptron", "mlp"):
        return MLPClassifier(
            hidden_layer_sizes=(100,), activation="relu",
            solver="adam", alpha=1e-4, learning_rate_init=1e-3,
            max_iter=200, random_state=42
        )
    raise ValueError(f"Unknown model '{name}'")
