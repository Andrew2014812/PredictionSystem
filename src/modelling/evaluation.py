"""Evaluation metrics for the classifier, the count regressors and the
markets derived from them."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, mean_absolute_error, mean_poisson_deviance,
                             mean_squared_error, r2_score)

from .tasks import RESULT_CLASSES


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def classifier_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = proba.argmax(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier": multiclass_brier(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "predicted_share": {c: float(np.mean(y_pred == i)) for i, c in enumerate(RESULT_CLASSES)},
        "actual_share": {c: float(np.mean(y_true == i)) for i, c in enumerate(RESULT_CLASSES)},
    }


def regressor_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, None)
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "poisson_deviance": float(mean_poisson_deviance(y_true, pred)),
        "mean_actual": float(y_true.mean()),
        "mean_predicted": float(pred.mean()),
    }


def binary_metrics(outcome: np.ndarray, proba: np.ndarray) -> dict:
    outcome = np.asarray(outcome, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(outcome)),
        "accuracy": float(np.mean((proba >= 0.5) == outcome)),
        "brier": float(np.mean((proba - outcome) ** 2)),
        "log_loss": float(log_loss(outcome, proba, labels=[0, 1])),
        "base_rate": float(outcome.mean()),
    }


def calibration_table(outcome: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict]:
    """Predicted vs observed frequency per probability bin (reliability diagram)."""
    df = pd.DataFrame({"p": proba, "y": outcome})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"),
                                                   n=("y", "size"))
    return [{"predicted": float(r.predicted), "observed": float(r.observed), "n": int(r.n)}
            for r in grouped.itertuples() if r.n > 0]


def market_benchmark(df: pd.DataFrame, y_true: np.ndarray) -> dict | None:
    """Log loss / accuracy of bookmaker-implied 1X2 probabilities.

    Reported for comparison only: the odds are never model inputs.
    """
    odds = df[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    mask = np.all(odds > 1, axis=1)
    if mask.sum() < 50:
        return None
    inv = 1 / odds[mask]
    proba = inv / inv.sum(axis=1, keepdims=True)
    metrics = classifier_metrics(np.asarray(y_true)[mask], proba)
    metrics.pop("confusion_matrix")
    return metrics


def prior_benchmark(y_train: np.ndarray, y_true: np.ndarray) -> dict:
    """Naive baseline: always predict the training class frequencies."""
    freq = np.bincount(np.asarray(y_train, dtype=int), minlength=3) / len(y_train)
    proba = np.tile(freq, (len(y_true), 1))
    metrics = classifier_metrics(np.asarray(y_true), proba)
    metrics.pop("confusion_matrix")
    return metrics
