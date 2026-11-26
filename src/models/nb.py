from sklearn.naive_bayes import ComplementNB
from src.common.registry import model


@model("nb")
class NB:
    def __init__(self, seed=0, **params):   # seed is passed to every model, even if unused
        self.m = ComplementNB(**params)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):             # columns = sorted class labels
        return self.m.predict_proba(X)
