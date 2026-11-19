"""The model inputs of every stored prediction, so it can be explained later exactly as it was made.

Features of a match are rebuilt every day (results arrive, a club's spelling is corrected...). A finished
match opened a month later must still be explained with the values the models saw when its stored
prediction was generated, not with today's table. Each stored prediction therefore keeps, in
``inputs.parquet``: the match id, when it was generated, the kick-off, the model version, an id of the
feature set and the raw input values of the models (the union of the columns the version's models read,
float32). Only the stored version of a prediction is kept (it is replaced while the prediction is
refreshed before kick-off and frozen with it), so the file stays small.

Predictions made before inputs were stored (before v6.2) have no row: they are explained with the
current features and the page says so.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..storage import predictions_storage

INPUTS = "inputs.parquet"
ID_COLUMNS = ["match_id", "source", "generated_at", "kickoff", "model_version", "feature_set"]


def model_input_columns(bundles: dict) -> tuple[list[str], list[str]]:
    """(categorical, numeric) raw columns read by the models of one version."""
    categorical, numeric = set(), set()
    for bundle in bundles.values():
        categorical |= set(bundle.preprocessor.categorical)
        numeric |= set(bundle.preprocessor.numeric)
    return sorted(categorical), sorted(numeric - categorical)


def feature_set_id(categorical: list[str], numeric: list[str]) -> str:
    """Short, stable id of a list of input columns (the same columns always give the same id)."""
    text = "|".join(categorical) + "#" + "|".join(numeric)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def build_inputs(matches: pd.DataFrame, bundles: dict, version: str, source: str,
                 generated_at: pd.Timestamp) -> pd.DataFrame:
    categorical, numeric = model_input_columns(bundles)
    if matches.empty or not numeric:
        return pd.DataFrame()
    frame = matches.reindex(columns=numeric).apply(pd.to_numeric, errors="coerce").astype(np.float32)
    for col in categorical:
        frame.insert(0, col, matches[col].astype(str) if col in matches else "")
    ids = pd.DataFrame({"match_id": matches["match_id"].to_numpy(), "source": source,
                        "generated_at": generated_at,
                        "kickoff": pd.to_datetime(matches.get("kickoff", pd.Series(pd.NaT, index=matches.index))
                                                  ).to_numpy(),
                        "model_version": version, "feature_set": feature_set_id(categorical, numeric)},
                       index=matches.index)
    return pd.concat([ids, frame], axis=1).reset_index(drop=True)


def load_inputs() -> pd.DataFrame:
    return predictions_storage.read_parquet(INPUTS) if predictions_storage.exists(INPUTS) else pd.DataFrame()


def stored_inputs(match_id: str, generated_at, inputs: pd.DataFrame | None = None) -> pd.Series | None:
    """The inputs of the stored prediction of ``match_id`` made at ``generated_at``, or None."""
    inputs = load_inputs() if inputs is None else inputs
    if inputs.empty:
        return None
    rows = inputs.loc[(inputs["match_id"] == match_id)
                      & (pd.to_datetime(inputs["generated_at"]) == pd.Timestamp(generated_at))]
    return rows.iloc[-1] if len(rows) else None


def explanation_row(current: pd.Series | None, stored: pd.Series | None) -> tuple[pd.Series | None, bool]:
    """The feature row an explanation must use: the stored inputs over the current features.
    Returns (row, reproduced) — ``reproduced`` is False when no stored inputs exist."""
    if stored is None:
        return current, False
    values = stored.drop(labels=[c for c in ID_COLUMNS if c != "match_id"], errors="ignore")
    if current is None:
        return values, True
    row = current.copy()
    for col, value in values.items():
        row[col] = value
    return row, True


REPRODUCE_TOLERANCE = 0.005


def reproduces(row: pd.Series | None, snapshot: pd.Series | None, bundles: dict,
               tolerance: float = REPRODUCE_TOLERANCE) -> bool:
    """True when the prediction's own model version, given ``row``, returns the stored prediction (1X2
    probabilities within ``tolerance``, expected goals within 2 × ``tolerance``). The features of a past match are
    rebuilt from the matches before it, so for most predictions made before the inputs were stored the current row
    IS the original input; when it is not (a data correction since), an explanation of ``row`` would explain a
    different number and must not be shown as the explanation of the stored prediction."""
    if row is None or snapshot is None or not bundles or "result" not in bundles:
        return False
    try:
        frame = pd.DataFrame([row])
        p = bundles["result"].predict(frame)[0]
        stored = np.array([snapshot.get("p_home"), snapshot.get("p_draw"), snapshot.get("p_away")], dtype=float)
        if not np.all(np.isfinite(stored)) or np.max(np.abs(p - stored)) > tolerance:
            return False
        for name, col in (("home_goals", "xg_home"), ("away_goals", "xg_away")):
            value = snapshot.get(col)
            if name in bundles and value is not None and value == value:
                if abs(float(bundles[name].predict(frame)[0]) - float(value)) > 2 * tolerance:
                    return False
        return True
    except Exception:                                   # noqa: BLE001 - an incompatible row is "not reproduced"
        return False
