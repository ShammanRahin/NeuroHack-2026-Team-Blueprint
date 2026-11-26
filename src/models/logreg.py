"""MEMBER C. One file per model. Add a file + a YAML entry; touch nothing else."""
from sklearn.linear_model import LogisticRegression
from src.common.registry import model


@model("logreg")
class LogReg:
    def __init__(self, seed=0, **params):
        self.m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, **params)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)
