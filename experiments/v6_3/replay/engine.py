"""Replaying the production prediction engine on the season 2026/27 with any model outputs.

* ``production_outputs`` — the active production models (``prod_20261012``, trained on 2021/22–2025/26, i.e. only on
  matches played before every 2026/27 match): Baseline B's model.
* ``batch_from_outputs`` — any (p_home, p_draw, p_away, λ_home, λ_away, μ_corners) per match through the unchanged
  production mathematics (``src.prediction.engine.predict_matches``: score matrix reconciled with the 1X2
  classifier, every market, real prices, Main / Risk rules). The engine is fed "array bundles" that return the given
  outputs, so the market mathematics and the Main rule are production's own code, not a copy.
* ``fit_task`` / ``walk_forward`` — refit a production task (same features, same hyper-parameters unless told
  otherwise) on matches played before a cut-off date, with optional sample weights or another estimator; used by the
  model studies (recent-season weighting, other algorithms, corners, squad strength).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.registry import load_bundles
from src.modelling.tasks import TASKS
from src.modelling.training import ModelBundle, TaskTrainer, recency_weights
from src.prediction import engine as E

PRODUCTION_VERSION = "prod_20261012"
OUTPUT_COLS = ["p_home", "p_draw", "p_away", "lam_home", "lam_away", "corner_mu"]


# ------------------------------------------------------------------------------------------------ outputs
def production_bundles() -> dict:
    return load_bundles(PRODUCTION_VERSION)


def outputs_from_bundles(bundles: dict, matches: pd.DataFrame) -> pd.DataFrame:
    p = bundles["result"].predict(matches)
    out = pd.DataFrame({"match_id": matches["match_id"].to_numpy(), "p_home": p[:, 0], "p_draw": p[:, 1],
                        "p_away": p[:, 2], "lam_home": bundles["home_goals"].predict(matches),
                        "lam_away": bundles["away_goals"].predict(matches)})
    cb = bundles.get("corners")
    out["corner_mu"] = cb.predict(matches) if cb is not None else np.nan
    return out


def production_outputs(matches: pd.DataFrame) -> pd.DataFrame:
    return outputs_from_bundles(production_bundles(), matches)


@dataclass
class ArrayBundle:
    """Stands in for a ModelBundle inside ``predict_matches``: returns stored outputs by match_id."""
    values: pd.DataFrame              # index match_id
    columns: list[str]
    extra: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        v = self.values.reindex(df["match_id"])[self.columns].to_numpy(float)
        return v if len(self.columns) > 1 else v[:, 0]


def array_bundles(outputs: pd.DataFrame, extras: dict | None = None) -> dict:
    """Bundles for ``predict_matches`` from an outputs table; ``extras`` default to the production ones
    (draw multiplier 1, Dixon–Coles rho 0, corner leagues and negative-binomial size of prod_20261012)."""
    prod = production_bundles()
    ex = {k: copy.deepcopy(b.extra) for k, b in prod.items()}
    for k, v in (extras or {}).items():
        ex.setdefault(k, {}).update(v)
    o = outputs.set_index("match_id")
    out = {"result": ArrayBundle(o, ["p_home", "p_draw", "p_away"], ex["result"]),
           "home_goals": ArrayBundle(o, ["lam_home"], ex["home_goals"]),
           "away_goals": ArrayBundle(o, ["lam_away"], ex["away_goals"])}
    if "corner_mu" in o and o["corner_mu"].notna().any():
        out["corners"] = ArrayBundle(o, ["corner_mu"], ex["corners"])
    return out


def batch_from_outputs(matches: pd.DataFrame, outputs: pd.DataFrame, odds_lookup: dict, label: str,
                       extras: dict | None = None):
    """(snapshots, markets, picks) of ``matches`` with the given outputs, through production's engine."""
    bundles = array_bundles(outputs, extras)
    keep = E.build_inputs
    E.build_inputs = lambda *a, **k: pd.DataFrame()          # stored model inputs are not needed in a replay
    try:
        batch = E.predict_matches(matches, bundles, odds_lookup, label, "replay")
    finally:
        E.build_inputs = keep
    return batch


# ------------------------------------------------------------------------------------------------ refitting
def _xgb(task, params: dict, n_estimators: int):
    from xgboost import XGBClassifier, XGBRegressor
    cls = XGBClassifier if task.kind == "classifier" else XGBRegressor
    return cls(n_estimators=n_estimators, objective=task.objective, eval_metric=task.eval_metric, tree_method="hist",
               random_state=42, n_jobs=C.N_JOBS, **{k: v for k, v in params.items() if k != "half_life_years"})


def fit_task(name: str, train: pd.DataFrame, base: ModelBundle | None = None, params: dict | None = None,
             n_estimators: int | None = None, weights=None, estimator=None) -> ModelBundle:
    """Fit task ``name`` on the played rows of ``train`` with production's feature set.

    params / n_estimators default to the production bundle's; ``weights`` is an array aligned with the
    trainable rows, a function rows -> array, or None (then production's recency half-life is used);
    ``estimator`` replaces XGBoost (any sklearn-like model with fit / predict(_proba))."""
    base = base or production_bundles()[name]
    task = TASKS[name]
    trainer = TaskTrainer(task, train.columns)
    rows = train.loc[task.trainable(train)]
    if name == "corners" and base.extra.get("leagues"):
        rows = rows.loc[rows["league"].isin(base.extra["leagues"])]
    params = dict(params or base.params)
    n_estimators = n_estimators or base.n_estimators
    prep = FeaturePreprocessor(trainer.numeric, trainer.categorical).fit(rows)
    X, y = prep.transform(rows), task.target_values(rows)
    if callable(weights):
        w = weights(rows)
    elif weights is not None:
        w = np.asarray(weights, float)
    else:
        w = recency_weights(rows["date"], params.get("half_life_years", 0))
    model = estimator if estimator is not None else _xgb(task, params, n_estimators)
    model.fit(X, y, sample_weight=w)
    return ModelBundle(task=task, preprocessor=prep, model=model, params=params, n_estimators=n_estimators,
                       extra=copy.deepcopy(base.extra),
                       trained_on={"date_to": str(rows["date"].max().date()), "rows": int(len(rows))})


def month_starts(matches: pd.DataFrame) -> list[pd.Timestamp]:
    first = matches["date"].min().to_period("M").to_timestamp()
    last = matches["date"].max().to_period("M").to_timestamp()
    return list(pd.date_range(first, last, freq="MS"))


def walk_forward(features: pd.DataFrame, matches: pd.DataFrame, fit_fn, cutoffs: list[pd.Timestamp] | None = None,
                 log=None) -> pd.DataFrame:
    """Outputs for ``matches`` from models refitted at every cut-off on matches played BEFORE it.

    ``fit_fn(train_rows, cutoff) -> dict of bundles`` (keys result / home_goals / away_goals / corners; a missing
    key falls back to production's model). A match dated in [cutoff_i, cutoff_{i+1}) is predicted by the models
    of cutoff_i. Default cut-offs: the first day of every month of the season (monthly refit)."""
    cutoffs = sorted(cutoffs or month_starts(matches))
    if matches["date"].min() < cutoffs[0]:
        # a match before the first cut-off would be predicted by a model trained on data that includes it
        raise ValueError(f"first cut-off {cutoffs[0].date()} is after the first match {matches['date'].min().date()}")
    prod = production_bundles()
    frames = []
    for i, cut in enumerate(cutoffs):
        end = cutoffs[i + 1] if i + 1 < len(cutoffs) else pd.Timestamp.max
        target = matches.loc[(matches["date"] >= cut) & (matches["date"] < end)]
        if target.empty:
            continue
        train = features.loc[features["played"].astype(bool) & (features["date"] < cut)]
        bundles = {**prod, **fit_fn(train, cut)}
        out = outputs_from_bundles(bundles, target)
        out["refit_cutoff"] = cut
        frames.append(out)
        if log:
            log.info("cut-off %s: trained on %d rows, predicted %d matches", cut.date(), len(train), len(target))
    return pd.concat(frames, ignore_index=True)
