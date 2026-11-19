"""Model fits of the Goal-Model study (HEAVY job, run through experiments/v6_3/heavy.sh).

    experiments/v6_3/heavy.sh 5400 experiments/v6_3/goals/logs/base.log \
        .venv/bin/python -m experiments.v6_3.goals.base

Stages (each writes a cache file and is skipped when the file exists):

1. ``base_<S>`` S = 2023, 2024, 2025 — production tasks (result, home_goals, away_goals) refitted with production's
   features / hyper-parameters / recency weights on every season before S, predicting S (the analogue of the
   production model, trained once before the season). Used to choose the parameters of every variant; 2026/27 uses
   the production model prod_20261012 itself (Baseline B).
2. ``ratfeat_<S>`` S = 2024, 2025, 2026 — home / away goal models with the dynamic ratings (ratings.py, chosen
   on 2023–2025) as six extra features; otherwise production's set-up, trained on seasons before S.
3. ``stacked_<S>`` S = 2024, 2025, 2026 — one "team goals" model instead of separate home / away models: every
   match gives two rows (team / opponent features, is_home flag), production's away-goal hyper-parameters.
4. ``maher_2026`` — v6.2's MH2 Maher ratings (weekly Poisson GLM per league, 730-day window, 180-day half-life,
   L2 0.005; re-implemented from experiments/v6_2/goal_distributions/maher.py) for the 2026/27 matches, so that
   v6.2's frozen B candidate HY+R can be replayed.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.recent_weighting import lib as L
from src.features.sets import model_features
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.tasks import TASKS
from src.modelling.training import recency_weights

from . import ratings as R

DIRS = C.study_dirs("goals")
RAT_COLS = ["r_lh", "r_la", "r_h_att", "r_h_def", "r_a_att", "r_a_def"]


def before(season: int) -> pd.DataFrame:
    f = L.features()
    return f.loc[f["played"].astype(bool) & (f["season"] < season)]


def stage_base(log) -> None:
    for s in (2023, 2024, 2025):
        path = DIRS["cache"] / f"base_{s}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        train, target = before(s), L.season_rows(s)
        p = EN.fit_task("result", train).predict(target)
        out = pd.DataFrame({"match_id": target["match_id"], "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
                            "lam_home": EN.fit_task("home_goals", train).predict(target),
                            "lam_away": EN.fit_task("away_goals", train).predict(target)})
        out.to_parquet(path)
        log.info("base %s: %d rows, %.0fs", s, len(out), time.time() - t0)


def ratings_table() -> pd.DataFrame:
    path = DIRS["cache"] / "ratings_plain.parquet"
    if not path.exists():
        raise SystemExit("run `python -m experiments.v6_3.goals.run --stage ratings` first")
    return pd.read_parquet(path)


def _fit_xgb(task: str, rows: pd.DataFrame, numeric: list[str], target_col: str | None = None,
             params: dict | None = None, n_estimators: int | None = None):
    prod = EN.production_bundles()[task]
    params = dict(params or prod.params)
    prep = FeaturePreprocessor(numeric, ["league"]).fit(rows)
    y = rows[target_col or TASKS[task].target].astype(float)
    model = EN._xgb(TASKS[task], params, n_estimators or prod.n_estimators)
    model.fit(prep.transform(rows), y, sample_weight=recency_weights(rows["date"], params.get("half_life_years", 0)))
    return prep, model


def stage_ratfeat(log) -> None:
    rat = ratings_table().set_index("match_id")
    f = L.features()
    cols = f.columns
    for s in (2024, 2025, 2026):
        path = DIRS["cache"] / f"ratfeat_{s}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        train = before(s).join(rat[RAT_COLS], on="match_id")
        target = L.season_rows(s).join(rat[RAT_COLS], on="match_id")
        out = pd.DataFrame({"match_id": target["match_id"]})
        for task, col in (("home_goals", "lam_home"), ("away_goals", "lam_away")):
            num, _ = model_features(task, cols)
            rows = train.loc[TASKS[task].trainable(train)]
            prep, model = _fit_xgb(task, rows, num + RAT_COLS)
            out[col] = model.predict(prep.transform(target))
        out.to_parquet(path)
        log.info("ratfeat %s: %.0fs", s, time.time() - t0)


# ------------------------------------------------------------------------------------------------ stacked model
def _canon(col: str, home_row: bool) -> str:
    me, opp = ("h_", "a_") if home_row else ("a_", "h_")
    if col.startswith(me):
        c = "t_" + col[2:]
    elif col.startswith(opp):
        c = "o_" + col[2:]
    else:
        return col
    return c.replace("_vs_a_", "_vs_o_").replace("_vs_h_", "_vs_o_")


def stacked_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Two rows per match: (home team vs away team, is_home = 1) and (away team vs home team, is_home = 0)."""
    parts = []
    for home_row in (True, False):
        d = pd.DataFrame({"match_id": df["match_id"].to_numpy(), "league": df["league"].to_numpy(),
                          "date": df["date"].to_numpy(), "is_home": float(home_row)})
        for c in cols:
            v = df[c].to_numpy(float)
            name = _canon(c, home_row)
            if not home_row and c.startswith("diff_"):
                v = -v
            if c == "elo_diff" and not home_row:
                v = -v
            if c == "elo_exp_home" and not home_row:
                v = 1 - v
            if c == "h2h_home_team_goals":
                name = "h2h_team_goals"
                if not home_row:
                    v = df["h2h_avg_goals"].to_numpy(float) - v
            if c in ("league_avg_home_goals", "league_avg_away_goals"):
                mine = (c == "league_avg_home_goals") == home_row
                name = "league_avg_team_goals" if mine else "league_avg_opp_goals"
            d[name] = v
        y = df["home_goals" if home_row else "away_goals"].to_numpy(float)
        d["goals"] = y
        d["played"] = df["played"].to_numpy()
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def stage_stacked(log) -> None:
    f = L.features()
    num_h, _ = model_features("home_goals", f.columns)
    num_a, _ = model_features("away_goals", f.columns)
    cols = sorted(set(num_h) | set(num_a))
    for s in (2024, 2025, 2026):
        path = DIRS["cache"] / f"stacked_{s}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        tr = stacked_frame(before(s), cols)
        tr = tr.loc[tr["played"].astype(bool) & tr["goals"].notna()]
        tr["date"] = pd.to_datetime(tr["date"])
        num = [c for c in tr.columns if c not in ("match_id", "league", "date", "goals", "played")]
        prep, model = _fit_xgb("away_goals", tr, num, target_col="goals")
        target = L.season_rows(s)
        te = stacked_frame(target, cols)
        pred = model.predict(prep.transform(te))
        n = len(target)
        out = pd.DataFrame({"match_id": target["match_id"].to_numpy(), "lam_home": pred[:n], "lam_away": pred[n:]})
        out.to_parquet(path)
        log.info("stacked %s: %d training rows, %.0fs", s, len(tr), time.time() - t0)


# ------------------------------------------------------------------------------------------------ v6.2 Maher MH2
HALF_LIFE, WINDOW, ALPHA = 180.0, 730, 0.005


def _maher_design(train, teams):
    n, k = len(train), len(teams)
    hi = train["home_team"].map(teams).to_numpy()
    ai = train["away_team"].map(teams).to_numpy()
    r = np.arange(n)
    X = np.zeros((2 * n, 4 * k + 1))
    X[r, hi] = 1
    X[r, k + ai] = -1
    X[r, 4 * k] = 1
    X[n + r, 2 * k + ai] = 1
    X[n + r, 3 * k + hi] = -1
    y = np.concatenate([train["home_goals"].to_numpy(float), train["away_goals"].to_numpy(float)])
    return X, y


def _maher_rows(score, teams):
    k, n = len(teams), len(score)
    X = np.zeros((2 * n, 4 * k + 1))
    for i, (h, a) in enumerate(zip(score["home_team"].map(teams), score["away_team"].map(teams))):
        if h == h:
            X[i, int(h)] = 1
            X[n + i, 3 * k + int(h)] = -1
        if a == a:
            X[i, k + int(a)] = -1
            X[n + i, 2 * k + int(a)] = 1
        X[i, 4 * k] = 1
    return X


def stage_maher(log) -> None:
    from sklearn.linear_model import PoissonRegressor
    path = DIRS["cache"] / "maher_2026.parquet"
    if path.exists():
        return
    warnings.filterwarnings("ignore")
    t0 = time.time()
    f = L.features()
    played = f.loc[f["played"].astype(bool) & f["home_goals"].notna()]
    out = []
    for lg, d in f.loc[f["season"] == 2026].groupby("league"):
        hist = played.loc[played["league"] == lg]
        week = (d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")).dt.normalize()
        for w, score in d.groupby(week):
            train = hist.loc[(hist["date"] < w) & (hist["date"] >= w - pd.Timedelta(days=WINDOW))]
            if len(train) < 30:
                continue
            teams = {t: i for i, t in enumerate(sorted(set(train["home_team"]) | set(train["away_team"])))}
            wt = np.power(2.0, -(w - train["date"]).dt.days.to_numpy(float) / HALF_LIFE)
            X, y = _maher_design(train, teams)
            m = PoissonRegressor(alpha=ALPHA, max_iter=300).fit(X, y, sample_weight=np.concatenate([wt, wt]))
            mu = m.predict(_maher_rows(score, teams))
            n = len(score)
            out.append(pd.DataFrame({"match_id": score["match_id"].to_numpy(), "lh_m2": mu[:n], "la_m2": mu[n:]}))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(path)
    log.info("maher 2026: %d rows, %.0fs", len(res), time.time() - t0)


def main() -> None:
    log = C.setup_logging("goals.base")
    stage_base(log)
    stage_ratfeat(log)
    stage_stacked(log)
    stage_maher(log)
    log.info("done")


if __name__ == "__main__":
    main()
