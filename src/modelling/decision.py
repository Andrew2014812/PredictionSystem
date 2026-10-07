"""Draw-aware decision rule for the 1X2 classifier.

The classifier's draw probabilities are well calibrated (the average
predicted draw probability matches the draw rate), but a draw is rarely the
single most likely outcome, so a plain arg-max almost never predicts one.

The decision rule keeps the probabilities untouched (log loss and Brier do
not change) and only changes which outcome is *named* as the prediction:

    predicted outcome = argmax(P(H), m * P(D), P(A))

The draw multiplier ``m`` is chosen on the VALIDATION season only:
the value with the highest macro F1 among those whose predicted draw share
does not exceed the actual draw share of the training seasons. The cap
stops the rule from simply predicting more draws than really happen.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

DRAW = 1
MULTIPLIER_GRID = np.round(np.arange(1.0, 1.801, 0.05), 2)


def decide(proba: np.ndarray, draw_multiplier: float = 1.0) -> np.ndarray:
    """Outcome index (0 = H, 1 = D, 2 = A) for every row of ``proba``."""
    weights = np.array([1.0, draw_multiplier, 1.0])
    return np.argmax(np.asarray(proba) * weights, axis=1)


def choose_draw_multiplier(proba: np.ndarray, y: np.ndarray, max_draw_share: float) -> dict:
    """Pick the multiplier on validation data; returns the choice and the grid."""
    y = np.asarray(y, dtype=int)
    grid = []
    for m in MULTIPLIER_GRID:
        pred = decide(proba, m)
        grid.append({"multiplier": float(m), "macro_f1": float(f1_score(y, pred, average="macro")),
                     "draw_share": float(np.mean(pred == DRAW))})
    allowed = [g for g in grid if g["draw_share"] <= max_draw_share] or grid[:1]
    best = max(allowed, key=lambda g: (g["macro_f1"], -g["multiplier"]))
    return {"multiplier": best["multiplier"], "max_draw_share": float(max_draw_share), "grid": grid}
