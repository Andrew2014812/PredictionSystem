"""Out-of-sample predictions of the production models and their Main pick at real prices.

* 2023/24, 2024/25 — walk-forward refits: the production trainer (src/modelling/training.TaskTrainer) with the
  tuned parameters and tree counts of the active version, fitted on all seasons before the season predicted;
  the corners model on the leagues with corner statistics in its training rows. The fixed extras of the active
  bundle are reused (draw multiplier 1, Dixon–Coles rho 0, negative-binomial size of corners).
* 2025/26 — the production back-test bundle ``backtest_2526`` (trained on 2021/22–2024/25).
* 2026/27 — the active production bundle (trained on 2021/22–2025/26), played matches so far.

Every match is pushed through the production engine (src/prediction/engine.predict_matches) with the real
Football-Data Bet365 pre-match prices of the match, so Main / Risk are chosen exactly as production chooses
them; settlement uses src/markets/markets.settle_selection. Threads per model: N_JOBS.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from src.markets.markets import settle_selection
from src.modelling import training as T
from src.modelling.registry import load_bundles
from src.modelling.tasks import TASKS
from src.prediction.engine import predict_matches
from src.providers.football_data import odds_from_matches
from src.providers.schema import odds_lookup

from . import common as C

_orig_estimator = T._estimator


def _estimator_limited(task, params, n_estimators, early_stopping):
    est = _orig_estimator(task, params, n_estimators, early_stopping)
    est.set_params(n_jobs=C.N_JOBS)
    return est


T._estimator = _estimator_limited          # study-only: <= N_JOBS threads per model on the shared machine

KEEP = ["match_id", "league", "country", "season", "date", "time", "kickoff", "home_team", "away_team", "played",
        "result", "home_goals", "away_goals", "total_corners", "odds_home", "odds_draw", "odds_away"]
_FEATURES: pd.DataFrame | None = None


def all_features() -> pd.DataFrame:
    global _FEATURES
    if _FEATURES is None:
        f = C.features()
        _FEATURES = f.loc[f["season"] >= 2021].reset_index(drop=True)
    return _FEATURES


def played(season: int) -> pd.DataFrame:
    f = all_features()
    return f.loc[(f["season"] == season) & f["played"].astype(bool) & f["home_goals"].notna()].reset_index(drop=True)


def _limit(bundles: dict) -> dict:
    for b in bundles.values():
        try:
            b.model.set_params(n_jobs=C.N_JOBS)
        except Exception:                                           # noqa: BLE001
            pass
    return bundles


def refit(season: int) -> dict:
    path = C.CACHE / f"bundles_{season}.pkl"
    if path.exists():
        return _limit(pickle.loads(path.read_bytes()))
    prod = load_bundles(C.PROD_VERSION)
    f = all_features()
    train = f.loc[(f["season"] < season) & f["played"].astype(bool)]
    cl = T.corner_leagues(train)
    out = {}
    for name, task in TASKS.items():
        rows = train.loc[train["league"].isin(cl)] if name == "corners" else train
        p = prod[name]
        b = T.TaskTrainer(task, f.columns).fit(rows, p.params, p.n_estimators)
        extra = {k: v for k, v in p.extra.items() if k != "feature_means"}
        if name == "corners":
            extra["leagues"] = cl
        b.extra.update(extra)
        out[name] = b
        C.log.info("refit %s for %s: %d rows", name, season, int(task.trainable(rows).sum()))
    path.write_bytes(pickle.dumps(out))
    return out


def bundles_for(season: int) -> dict:
    if season == C.BACKTEST_SEASON:
        return _limit(load_bundles(C.BACKTEST_VERSION))
    if season == C.LIVE_SEASON:
        return _limit(load_bundles(C.PROD_VERSION))
    return refit(season)


def model_label(season: int) -> str:
    return {C.BACKTEST_SEASON: C.BACKTEST_VERSION, C.LIVE_SEASON: C.PROD_VERSION}.get(season, f"refit < {season}")


def predict(rows: pd.DataFrame, bundles: dict, odds: dict | None = None) -> pd.DataFrame:
    """Production engine on ``rows``; one row per match with probabilities, Main / Risk and settlement."""
    if odds is None:
        odds = odds_lookup(odds_from_matches(rows))
    batch = predict_matches(rows, bundles, odds, "study", "study")
    snap = batch.snapshots.set_index("match_id")
    picks = batch.picks
    out = rows[[c for c in KEEP if c in rows]].copy().set_index("match_id")
    for c in ("p_home", "p_draw", "p_away", "xg_home", "xg_away", "exp_corners"):
        out[c] = snap[c]
    for role in ("main", "risk"):
        p = picks.loc[picks["role"] == role].set_index("match_id")
        out[f"{role}_market"] = p["market"]
        out[f"{role}_selection"] = p["selection"]
        out[f"{role}_prob"] = p["probability"]
        out[f"{role}_odds"] = p["odds"]
    out = out.reset_index()
    out["y"] = out["result"].map({"H": 0, "D": 1, "A": 2})
    for role in ("main", "risk"):
        status = [settle_selection(m, s, hg, ag, tc) if isinstance(m, str) else None
                  for m, s, hg, ag, tc in zip(out[f"{role}_market"], out[f"{role}_selection"], out["home_goals"],
                                              out["away_goals"], out["total_corners"])]
        status = pd.Series(status, index=out.index, dtype=object)
        out[f"{role}_won"] = np.where(status == "WON", 1.0, np.where(status == "LOST", 0.0, np.nan))
        priced = out[f"{role}_odds"].notna() & (out[f"{role}_odds"] > 1)
        out[f"{role}_profit"] = np.where(priced & (status == "WON"), out[f"{role}_odds"] - 1,
                                         np.where(priced & (status == "LOST"), -1.0, np.nan))
    known = out["y"].notna().to_numpy()
    yi = out["y"].fillna(0).to_numpy(int)
    p = np.clip(out[["p_home", "p_draw", "p_away"]].to_numpy(float), 1e-15, 1)
    out["ll"] = np.where(known, -np.log(p[np.arange(len(out)), yi]), np.nan)
    inv = 1 / out[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    q = inv / inv.sum(1, keepdims=True)
    out["ll_market"] = np.where(known, -np.log(np.clip(q[np.arange(len(out)), yi], 1e-15, 1)), np.nan)
    return out


def season_predictions(season: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"pred_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    rows = played(season)
    out = predict(rows, bundles_for(season))
    out["model"] = model_label(season)
    out.to_parquet(path)
    C.log.info("predictions %s: %d matches", season, len(out))
    return out


def all_predictions() -> pd.DataFrame:
    return pd.concat([season_predictions(s) for s in C.EVAL_SEASONS], ignore_index=True)


if __name__ == "__main__":
    C.setup_logging("predictions")
    for s in C.EVAL_SEASONS:
        season_predictions(s)
