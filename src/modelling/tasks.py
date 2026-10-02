"""The four prediction tasks, each solved by its own XGBoost model."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RESULT_CLASSES = ["H", "D", "A"]


@dataclass(frozen=True)
class Task:
    name: str            # also the feature-set key (features/sets.py)
    title: str
    kind: str            # "classifier" | "regressor"
    target: str
    objective: str
    eval_metric: str
    description: str

    @property
    def file_name(self) -> str:
        return f"{self.name}_model.pkl"

    def target_values(self, df: pd.DataFrame) -> pd.Series:
        y = df[self.target]
        if self.kind == "classifier":
            return y.map({c: i for i, c in enumerate(RESULT_CLASSES)}).astype(int)
        return y.astype(float)

    def trainable(self, df: pd.DataFrame) -> pd.Series:
        """Rows with a known target."""
        return df["played"].astype(bool) & df[self.target].notna()


TASKS: dict[str, Task] = {
    "result": Task(
        "result", "Match result (1X2)", "classifier", "result",
        "multi:softprob", "mlogloss",
        "Probabilities of home win, draw and away win."),
    "home_goals": Task(
        "home_goals", "Home goals", "regressor", "home_goals",
        "count:poisson", "poisson-nloglik",
        "Expected number of goals of the home team (Poisson rate)."),
    "away_goals": Task(
        "away_goals", "Away goals", "regressor", "away_goals",
        "count:poisson", "poisson-nloglik",
        "Expected number of goals of the away team (Poisson rate)."),
    "corners": Task(
        "corners", "Total corners", "regressor", "total_corners",
        "count:poisson", "poisson-nloglik",
        "Expected total number of corners in the match."),
}
