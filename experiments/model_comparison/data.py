"""Folds and feature matrices shared by every algorithm (identical for all of them).

Walk-forward folds, one per evaluated season s:

    tune   train = seasons < s-1, validation = s-1   (hyperparameters, early stopping, weights)
    fit    all seasons < s with the chosen settings
    score  season s

Winners are chosen on the folds 2023/24 and 2024/25. The fold 2025/26 is the
v1–v4 "test season", which has been looked at during development, so it is
reported but not used to choose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features.sets import GROUPS, MODEL_GROUPS
from src.features.store import load_features
from src.modelling.pipeline import _subset
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.tasks import TASKS
from src.modelling.training import corner_leagues

SELECTION_SEASONS = (2023, 2024)     # folds used to choose winners
REPORT_SEASON = 2025                  # reported, not used to choose


@dataclass
class Fold:
    season: int
    tune_train: pd.DataFrame
    tune_valid: pd.DataFrame
    fit: pd.DataFrame
    test: pd.DataFrame

    @property
    def label(self) -> str:
        return config.season_label(self.season)


def load() -> pd.DataFrame:
    return load_features()


def folds(features: pd.DataFrame, seasons=(2023, 2024, 2025)) -> list[Fold]:
    out = []
    for s in seasons:
        before = sorted(x for x in features["season"].unique() if x < s)
        out.append(Fold(s, _subset(features, before[:-1]), _subset(features, [before[-1]]),
                        _subset(features, before), _subset(features, [s])))
    return out


def numeric_columns(task: str, columns, drop: re.Pattern | None = None) -> list[str]:
    pats = [re.compile(p) for g in MODEL_GROUPS[task] for p in GROUPS[g]]
    cols = [c for c in columns if any(p.search(c) for p in pats)]
    return [c for c in cols if not (drop and drop.search(c))]


def task_rows(task: str, df: pd.DataFrame, leagues: list[str] | None = None) -> pd.DataFrame:
    t = TASKS[task] if task in TASKS else None
    target = t.target if t else task
    rows = df.loc[df["played"].astype(bool) & df[target].notna()]
    if leagues is not None:
        rows = rows.loc[rows["league"].isin(leagues)]
    return rows


class Matrix:
    """Tree-model matrix (NaN kept, league one-hot) and a linear-model matrix
    (median-imputed, standardised, plus missing-value indicators), both fitted on
    the training rows only."""

    def __init__(self, numeric: list[str]):
        self.numeric = numeric
        self.prep = FeaturePreprocessor(numeric, ["league"])

    def fit(self, df: pd.DataFrame) -> "Matrix":
        self.prep.fit(df)
        X = self.prep.transform(df)
        self.median = X.median()
        self.missing_cols = [c for c in self.numeric if X[c].isna().any()]
        filled = self._linear_raw(X)
        self.mean, self.std = filled.mean(), filled.std().replace(0, 1)
        return self

    def tree(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.prep.transform(df)

    def _linear_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        filled = X.fillna(self.median).fillna(0)
        miss = X[self.missing_cols].isna().astype(np.float32).add_suffix("__missing")
        return pd.concat([filled, miss], axis=1)

    def linear(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._linear_raw(self.prep.transform(df))
        return ((raw - self.mean) / self.std).to_numpy(np.float64)


def corner_scope(features: pd.DataFrame) -> list[str]:
    return corner_leagues(features.loc[features["played"].astype(bool)])
