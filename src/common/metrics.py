"""Metrics mirroring the competition (Overview tab). Hidden subsets (domain/robust/novel) can't be scored locally."""
import numpy as np
from sklearn.metrics import f1_score

WEIGHTS = {"A": 0.40, "B": 0.15, "C": 0.10, "robust": 0.10, "calib": 0.05}
# S = 0.40A + 0.15B + 0.10C + 0.10domain + 0.10robust + 0.05calib + 0.10novel; "robust" here = Task A macro F1 on a
# seeded corrupted copy of the validation rows (estimate of the hidden robustness subset)


def score(y_true, y_pred):
    """Legacy single-task metric (macro F1)."""
    return f1_score(y_true, y_pred, average="macro")


def macro_f1(y_true, y_pred):
    """Macro F1; a class absent from both truth and prediction is skipped (competition rule)."""
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def techniques_f1(T_true, T_pred):
    """Macro F1 over T1..T6, binary per technique over all rows; techniques absent from truth and prediction skipped."""
    fs = [f1_score(T_true[:, j], T_pred[:, j], zero_division=0) for j in range(T_true.shape[1])
          if T_true[:, j].any() or T_pred[:, j].any()]
    return float(np.mean(fs)) if fs else 0.0


def calibration(conf, correct):
    """1 - Brier score of the confidence against 'Task A label was correct'."""
    return 1.0 - float(np.mean((np.asarray(conf) - np.asarray(correct, dtype=float)) ** 2))


def local_composite(parts):
    """Weighted mean of the available local components, weights renormalised (like the public LB with missing subsets)."""
    w = {k: WEIGHTS[k] for k in parts}
    return sum(w[k] * parts[k] for k in parts) / sum(w.values())
