"""Walk-forward fits of the production XGBoost models (shared by the leagues and temporal studies).

For every season s the models are fitted on seasons < s only and predict season s (2026/27: the matches played
so far, models fitted on 2021/22 ... 2025/26). Parameters and tree counts are those of the active production
version (local_data/models/<active>/metadata.json); preprocessing (league one-hot + numeric passthrough) and
recency weights exactly as src/modelling/training.py. Only the random seed (production 42) and the thread
count (2, shared machine) are set here. Nothing is re-tuned.

Market reference: Football-Data Bet365 opening 1X2 prices (published before the decision), de-margined.
Main: the production rule (v6.1 vectorised copy, experiments/v6_1/main/strategies.py, checked against the
production functions there) on the candidate table built with the production mathematics
(experiments/v6_1/main/candidates.py), priced at the Bet365 opening price as production attaches it.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from experiments.v6_research import common as V6
from src import config
from src.features.sets import model_features
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.tasks import TASKS
from src.modelling.training import corner_leagues, negative_binomial_size, recency_weights

from . import common as C

GOAL_TASKS = ("result", "home_goals", "away_goals")

# ------------------------------------------------------------------------------------------------ data
_F: pd.DataFrame | None = None


def features() -> pd.DataFrame:
    """Played matches of the feature table (seasons >= FIRST_SEASON), as the production training uses them."""
    global _F
    if _F is None:
        _F = V6.features()
    return _F


def prod_numeric(task: str) -> list[str]:
    return model_features(task, list(features().columns))[0]


# ------------------------------------------------------------------------------------------------ one model
class Fit:
    """Production-faithful XGBoost fit with a chosen seed, numeric feature list and categorical columns."""

    def __init__(self, task: str, numeric: list[str] | None = None, categorical=("league",), seed: int = C.PROD_SEED,
                 n_estimators: int | None = None):
        t = TASKS[task]
        params, n = V6.prod_params(task)
        self.task, self.kind = task, t.kind
        self.objective, self.eval_metric = t.objective, t.eval_metric
        self.params, self.n = params, int(n_estimators or n)
        self.numeric = list(numeric if numeric is not None else prod_numeric(task))
        self.categorical = list(categorical)
        self.seed = seed
        self.seconds = 0.0

    def _est(self, n: int):
        cls = XGBClassifier if self.kind == "classifier" else XGBRegressor
        return cls(n_estimators=n, objective=self.objective, eval_metric=self.eval_metric, tree_method="hist",
                   random_state=self.seed, n_jobs=C.N_JOBS,
                   **{k: v for k, v in self.params.items() if k != "half_life_years"})

    def _weights(self, rows, extra=None):
        w = recency_weights(rows["date"], self.params.get("half_life_years", 0))
        if extra is not None:
            w = np.asarray(extra, float) * (1.0 if w is None else w)
        return w

    def fit(self, rows: pd.DataFrame, sample_weight=None) -> "Fit":
        t0 = time.time()
        y = TASKS[self.task].target_values(rows).to_numpy()
        self.prep = FeaturePreprocessor(self.numeric, self.categorical).fit(rows)
        self.model = self._est(self.n).fit(self.prep.transform(rows), y, sample_weight=self._weights(rows, sample_weight))
        self.seconds = time.time() - t0
        return self

    def fine_tune(self, rows: pd.DataFrame, extra_trees: int) -> "Fit":
        """Continue boosting the fitted model on ``rows`` only (same preprocessing, same parameters)."""
        out = Fit.__new__(Fit)
        out.__dict__.update(self.__dict__)
        y = TASKS[self.task].target_values(rows).to_numpy()
        est = self._est(extra_trees)
        est.fit(self.prep.transform(rows), y, sample_weight=self._weights(rows), xgb_model=self.model.get_booster())
        out.model = est
        return out

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.prep.transform(rows)
        if self.kind == "classifier":
            return self.model.predict_proba(X)
        return self.model.predict(X)


def predict_goal_tasks(train: pd.DataFrame, score: pd.DataFrame, seed: int, numeric: dict | None = None,
                       categorical=("league",), tasks=GOAL_TASKS, sample_weight=None) -> pd.DataFrame:
    """pH, pD, pA, lh, la of ``score`` from models fitted on ``train``."""
    numeric = numeric or {}
    out = pd.DataFrame({"match_id": score["match_id"].to_numpy()})
    for task in tasks:
        m = Fit(task, numeric.get(task), categorical, seed).fit(train, sample_weight)
        p = m.predict(score)
        if task == "result":
            out["pH"], out["pD"], out["pA"] = p[:, 0], p[:, 1], p[:, 2]
        else:
            out["lh" if task == "home_goals" else "la"] = p
    return out


def before(df, s):
    return df.loc[df["season"] < s]


def season(df, s):
    return df.loc[df["season"] == s]


# ------------------------------------------------------------------------------------------------ base predictions
def base_predictions(s: int, seed: int = C.PROD_SEED, refresh: bool = False) -> pd.DataFrame:
    """Production setup, walk-forward: fitted on < s, predicts s. Cached per (season, seed)."""
    path = C.CACHE / f"base_{s}_{seed}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = features()
    out = predict_goal_tasks(before(df, s), season(df, s), seed)
    out.to_parquet(path)
    C.log.info("base %s seed %s: %d rows", s, seed, len(out))
    return out


def corners(s: int, refresh: bool = False) -> tuple[pd.DataFrame, float]:
    """Production corner model (seed 42) for season s and its negative-binomial size (Main needs both)."""
    path, pinfo = C.CACHE / f"corners_{s}.parquet", C.CACHE / f"corners_{s}.json"
    if path.exists() and pinfo.exists() and not refresh:
        return pd.read_parquet(path), json.loads(pinfo.read_text())["nb_size"]
    df = features()
    train, score = before(df, s), season(df, s)
    cl = corner_leagues(train)
    ctr = train.loc[train["total_corners"].notna() & train["league"].isin(cl)]
    m = Fit("corners").fit(ctr)
    out = pd.DataFrame({"match_id": score["match_id"].to_numpy(),
                        "cmu": np.where(score["league"].isin(cl), m.predict(score), np.nan)})
    nb = negative_binomial_size(ctr["total_corners"].to_numpy(float), m.predict(ctr))
    out.to_parquet(path)
    pinfo.write_text(json.dumps({"nb_size": nb, "corner_leagues": cl}))
    return out, nb


def seed_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average of several seeds' predictions (same rows in the same order)."""
    out = frames[0][["match_id"]].copy()
    for c in ("pH", "pD", "pA", "lh", "la"):
        if c in frames[0]:
            out[c] = np.mean([f[c].to_numpy() for f in frames], 0)
    if "pH" in out:
        s = out[["pH", "pD", "pA"]].sum(1).to_numpy()
        for c in ("pH", "pD", "pA"):
            out[c] = out[c] / s
    return out


# ------------------------------------------------------------------------------------------------ evaluation frame
_MKT: dict[int, pd.DataFrame] = {}


def fd_table(s: int) -> pd.DataFrame:
    """Football-Data opening / closing prices of season s (experiments/v6_1/main/fd_prices.py), cached."""
    from experiments.v6_1.main import fd_prices
    path = C.CACHE / f"fd_{s}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fd = fd_prices.prices(s)
    fd.to_parquet(path)
    return fd


def market(s: int) -> pd.DataFrame:
    """match_id + de-margined Bet365 opening (qH, qD, qA) and closing (cH, cD, cA) 1X2 probabilities."""
    if s not in _MKT:
        fd = fd_table(s)
        o = fd[["x_H_b365", "x_D_b365", "x_A_b365"]].to_numpy(float)
        c = fd[["x_H_b365_c", "x_D_b365_c", "x_A_b365_c"]].to_numpy(float)
        q, qc = C.demargin(o, C.OPEN_METHOD), C.demargin(c, C.CLOSE_METHOD)
        _MKT[s] = pd.DataFrame({"match_id": fd["match_id"].to_numpy(), "qH": q[:, 0], "qD": q[:, 1], "qA": q[:, 2],
                                "cH": qc[:, 0], "cD": qc[:, 1], "cA": qc[:, 2]})
    return _MKT[s]


def frame(s: int) -> pd.DataFrame:
    """Evaluation rows of season s: every played match WITH a full Bet365 opening 1X2 price (the same matches for
    every variant), with result, goals and the market probabilities. Ordered like features()."""
    df = season(features(), s)
    base = df[["match_id", "league", "season", "date", "home_team", "away_team", "home_goals", "away_goals",
               "total_corners"]].reset_index(drop=True)
    base["y"] = np.where(base["home_goals"] > base["away_goals"], 0,
                         np.where(base["home_goals"] == base["away_goals"], 1, 2))
    base = base.merge(market(s), on="match_id", how="left")
    base["in_eval"] = base[["qH", "qD", "qA"]].notna().all(1)
    return base


def P(pred: pd.DataFrame) -> np.ndarray:
    return pred[["pH", "pD", "pA"]].to_numpy(float)


def Q(fr: pd.DataFrame, closing: bool = False) -> np.ndarray:
    return fr[["cH", "cD", "cA"] if closing else ["qH", "qD", "qA"]].to_numpy(float)


# ------------------------------------------------------------------------------------------------ Main
def main_outcome(fr: pd.DataFrame, pred: pd.DataFrame, s: int) -> pd.DataFrame:
    """The production Main of every evaluation match from 1X2 probabilities and goal rates ``pred`` (aligned with
    ``fr``) plus the production corner model; one row per match with p, odds, status, profit."""
    from experiments.v6_1.main import candidates as CD
    from experiments.v6_1.main import strategies as ST
    cm, nb = corners(s)
    p = fr[["match_id", "league", "season", "date", "home_goals", "away_goals", "total_corners"]].copy()
    p[["pH", "pD", "pA"]] = P(pred)
    p["lh"], p["la"] = pred["lh"].to_numpy(), pred["la"].to_numpy()
    p = p.merge(cm, on="match_id", how="left")
    p = p.loc[fr["in_eval"].to_numpy()].reset_index(drop=True)
    long = CD.long_from_pred(p, nb, s, fd_table(s))
    keep = ["match_id", "league", "date", "market", "selection", "p", "p_push", "status", "group", "line", "odds"]
    long = long[keep].reset_index(drop=True)
    long["mi"] = pd.factorize(long["match_id"])[0]
    idx = ST.choose(long)
    o = ST.outcome(long, idx)
    return o[["match_id", "league", "market", "selection", "group", "p", "odds", "status", "decided", "priced",
              "won", "profit"]].reset_index(drop=True)
