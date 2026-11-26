from lightgbm import LGBMClassifier
from src.common.registry import model


@model("lgbm")
class LGBM:
    def __init__(self, seed=0, **params):
        defaults = dict(n_estimators=200, learning_rate=0.05,
                        min_child_samples=2, class_weight="balanced",
                        verbose=-1)
        defaults.update(params)
        self.m = LGBMClassifier(random_state=seed, **defaults)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)
