"""One fitted transformation shared by training, evaluation and prediction.

The encoder is fitted on the training rows only and saved inside the model
bundle, so validation, test and live prediction matrices always have the
same columns in the same order. Unknown leagues at prediction time become
an all-zero one-hot block instead of an error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def _one_hot_name(feature: str, category) -> str:
    return f"{feature}={category}"


class FeaturePreprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.transformer = ColumnTransformer(
            [("categorical",
              OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            feature_name_combiner=_one_hot_name),
              self.categorical),
             ("numeric", "passthrough", self.numeric)],
            verbose_feature_names_out=False,
        )
        self.feature_names: list[str] = []

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # Missing numeric columns (e.g. a statistic absent for a league) are
        # created as NaN, which XGBoost treats as "missing".
        frame = df.reindex(columns=self.categorical + self.numeric)
        frame[self.numeric] = frame[self.numeric].astype("float64")
        frame[self.categorical] = frame[self.categorical].astype(str)
        return frame

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        self.transformer.fit(self._frame(df))
        self.feature_names = list(self.transformer.get_feature_names_out())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        values = self.transformer.transform(self._frame(df)).astype(np.float32)
        return pd.DataFrame(values, columns=self.feature_names, index=df.index)
