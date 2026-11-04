"""Class metrics of the outcome rule FootPredict really uses, for the Model Analysis page.

The model metadata stores the test-season confusion matrix of a plain arg-max. Since v6 an outcome is named
with the draw-aware rule (``decide`` with DRAW_THRESHOLD / DRAW_MAX_GAP), so the page computes the matrix,
accuracy, macro F1, balanced accuracy, recall / precision and predicted shares of THAT rule on the same
matches: the evaluation model's stored test-season probabilities (snapshots, source "backtest"). Log loss and
Brier belong to the probabilities and stay as stored (the rule does not change any probability).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from .. import config
from ..modelling.decision import decide

CLASSES = ("H", "D", "A")


def rule_metrics(snapshots: pd.DataFrame, results: pd.DataFrame, season: int | None = None) -> dict | None:
    """Metrics of the current rule on the backtest snapshots of ``season`` (default: the latest one)."""
    if snapshots.empty or results.empty:
        return None
    bt = snapshots.loc[snapshots["source"] == "backtest"]
    if bt.empty:
        return None
    season = season if season is not None else int(bt["season"].max())
    bt = bt.loc[bt["season"] == season].drop_duplicates("match_id", keep="last")
    played = results.loc[results["played"].astype(bool), ["match_id", "home_goals", "away_goals"]]
    df = bt.merge(played, on="match_id", how="inner")
    if df.empty:
        return None
    y = np.select([df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]], [0, 1], 2)
    proba = df[["p_home", "p_draw", "p_away"]].to_numpy(float)
    pred = decide(proba, 1.0, config.DRAW_THRESHOLD, config.DRAW_MAX_GAP)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    recall = {c: (cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan) for i, c in enumerate(CLASSES)}
    precision = {c: (cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else np.nan) for i, c in enumerate(CLASSES)}
    return {
        "season": season, "n": int(len(df)), "rule": {"draw_threshold": config.DRAW_THRESHOLD,
                                                      "max_gap": config.DRAW_MAX_GAP},
        "confusion_matrix": cm.tolist(),                      # rows: actual H / D / A, columns: named H / D / A
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall": recall, "precision": precision,
        "predicted_share": {c: float(np.mean(pred == i)) for i, c in enumerate(CLASSES)},
        "actual_share": {c: float(np.mean(y == i)) for i, c in enumerate(CLASSES)},
        "mean_probability": {c: float(proba[:, i].mean()) for i, c in enumerate(CLASSES)},
    }
