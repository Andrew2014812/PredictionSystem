"""SHAP explanations (TreeSHAP) of the XGBoost models.

Local explanations are computed on demand with the model version that
produced the stored prediction, from the stored pre-match features, so an
explanation can never use information the prediction did not have.

SHAP values are in the model's margin space: log-odds of a class for the
1X2 classifier and log of the expected count for the Poisson regressors.
Positive values push the explained selection up, negative values down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from ..features.names import feature_label
from ..markets.markets import market_group
from ..modelling.tasks import RESULT_CLASSES
from ..modelling.training import ModelBundle

_EXPLAINERS: dict[int, shap.TreeExplainer] = {}


def _explainer(bundle: ModelBundle) -> shap.TreeExplainer:
    key = id(bundle.model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(bundle.model)
    return _EXPLAINERS[key]


def shap_matrix(bundle: ModelBundle, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(feature matrix, SHAP values). For the classifier: (n, features, 3)."""
    X = bundle.matrix(df)
    values = np.asarray(_explainer(bundle).shap_values(X))
    return X, values


def _collapse_one_hot(contrib: pd.Series, values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Merge the one-hot league columns into a single 'league' feature."""
    one_hot = [c for c in contrib.index if c.startswith("league=")]
    if not one_hot:
        return contrib, values
    active = [c for c in one_hot if values[c] == 1]
    league_value = active[0].split("=", 1)[1] if active else "other"
    contrib = pd.concat([contrib.drop(one_hot), pd.Series({"league": contrib[one_hot].sum()})])
    values = pd.concat([values.drop(one_hot), pd.Series({"league": league_value})])
    return contrib, values


# How a market selection maps onto model outputs: (task, class or None, sign)
def explanation_plan(market: str, selection: str) -> list[tuple[str, int | None, float]]:
    group = market_group(market)
    cls = {c: i for i, c in enumerate(RESULT_CLASSES)}
    if group in ("1X2", "HANDICAP"):
        return [("result", cls[selection], 1.0)]
    if group == "DOUBLE_CHANCE":
        # "1X" = not an away win, so explain the away class with opposite sign
        excluded = {"1X": "A", "X2": "H", "12": "D"}[selection]
        return [("result", cls[excluded], -1.0)]
    if group == "EXACT_SCORE":
        h, a = (int(x) for x in selection.split("-"))
        return [("result", cls["H" if h > a else ("A" if h < a else "D")], 1.0)]
    if group in ("TOTAL", "BTTS"):
        sign = 1.0 if selection in ("OVER", "YES") else -1.0
        return [("home_goals", None, sign), ("away_goals", None, sign)]
    if group == "CORNERS":
        return [("corners", None, 1.0 if selection == "OVER" else -1.0)]
    raise ValueError(market)


def explain_selection(row: pd.DataFrame, market: str, selection: str,
                      bundles: dict[str, ModelBundle]) -> pd.DataFrame | None:
    """Per-feature contributions to one selection of one match.

    Returns columns: feature, label, value, shap, mean (training average).
    """
    contrib_total: pd.Series | None = None
    values_all: pd.Series | None = None
    means: dict = {}
    for task, cls, sign in explanation_plan(market, selection):
        bundle = bundles.get(task)
        if bundle is None:
            return None
        X, sv = shap_matrix(bundle, row)
        vals = sv[0, :, cls] if cls is not None else sv[0]
        contrib = pd.Series(sign * vals, index=X.columns)
        values = X.iloc[0]
        contrib, values = _collapse_one_hot(contrib, values)
        contrib_total = contrib if contrib_total is None else contrib_total.add(contrib, fill_value=0)
        values_all = values if values_all is None else values_all.combine_first(values)
        means.update(bundle.extra.get("feature_means", {}))
    out = pd.DataFrame({"feature": contrib_total.index, "shap": contrib_total.to_numpy()})
    out["value"] = out["feature"].map(values_all)
    out["mean"] = out["feature"].map(means)
    out["label"] = out["feature"].map(feature_label)
    out = out.loc[out["shap"].abs() > 1e-6]
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def global_importance(bundle: ModelBundle, df: pd.DataFrame, top: int = 30,
                      sample: int = 3000) -> list[dict]:
    """Mean |SHAP| per feature over a sample of matches."""
    if len(df) > sample:
        df = df.sample(sample, random_state=0)
    X, sv = shap_matrix(bundle, df)
    mean_abs = np.abs(sv).mean(axis=0)
    if mean_abs.ndim == 2:                      # classifier: average over classes
        mean_abs = mean_abs.mean(axis=1)
    imp = pd.Series(mean_abs, index=X.columns)
    one_hot = [c for c in imp.index if c.startswith("league=")]
    if one_hot:
        imp = pd.concat([imp.drop(one_hot), pd.Series({"league": imp[one_hot].sum()})])
    imp = imp.sort_values(ascending=False).head(top)
    return [{"feature": f, "label": feature_label(f), "mean_abs_shap": float(v)} for f, v in imp.items()]


def gain_importance(bundle: ModelBundle, top: int = 30) -> list[dict]:
    scores = bundle.model.get_booster().get_score(importance_type="gain")
    imp = pd.Series(scores, dtype=float)
    if imp.empty:
        return []
    imp = imp / imp.sum()
    one_hot = [c for c in imp.index if c.startswith("league=")]
    if one_hot:
        imp = pd.concat([imp.drop(one_hot), pd.Series({"league": imp[one_hot].sum()})])
    imp = imp.sort_values(ascending=False).head(top)
    return [{"feature": f, "label": feature_label(f), "gain_share": float(v)} for f, v in imp.items()]
