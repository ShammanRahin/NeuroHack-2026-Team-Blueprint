from sklearn.metrics import f1_score


def score(y_true, y_pred):
    """ONE metric for everyone. Change here if the competition metric differs."""
    return f1_score(y_true, y_pred, average="macro")
