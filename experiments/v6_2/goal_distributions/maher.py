"""Attack / defence ratings (Maher 1982; Dixon & Coles 1997 time weighting), refitted every week per league.

For the matches of league L in the week starting on Monday w, a Poisson GLM is fitted on the matches of L
played in the 730 days before w, each weighted by 2^(-age / 180 days):

    MH1  log E[goals of team t against o] = c + gamma * home + att_t - def_o       (one strength pair per team)
    MH2  home goals: log E = c_h + att^H_home - def^A_away;  away goals: log E = c_a + att^A_away - def^H_home
         (Maher's separate home / away strengths: four parameters per team)

A small L2 penalty (alpha = 0.005, sklearn PoissonRegressor; intercepts unpenalised) keeps the strengths of
teams with few matches near the league average; a team without matches in the window gets the average.
Nothing of week w (or later) is used to predict week w. The half-life, window and penalty were fixed in advance
(the half-life is close to the Dixon-Coles estimate); they are not tuned on the evaluated seasons.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from . import common as C

HALF_LIFE = 180.0
WINDOW = 730
ALPHA = 0.005
FIRST = 2022


def _design(train: pd.DataFrame, teams: dict, separate: bool):
    n, k = len(train), len(teams)
    hi = train["home_team"].map(teams).to_numpy()
    ai = train["away_team"].map(teams).to_numpy()
    r = np.arange(n)
    if not separate:
        X = np.zeros((2 * n, 2 * k + 1))
        X[r, hi] = 1
        X[r, k + ai] = -1
        X[r, 2 * k] = 1                                     # home indicator
        X[n + r, ai] = 1
        X[n + r, k + hi] = -1
    else:
        X = np.zeros((2 * n, 4 * k + 1))
        X[r, hi] = 1                                        # attack at home
        X[r, k + ai] = -1                                   # defence away
        X[r, 4 * k] = 1
        X[n + r, 2 * k + ai] = 1                            # attack away
        X[n + r, 3 * k + hi] = -1                           # defence at home
    y = np.concatenate([train["home_goals"].to_numpy(float), train["away_goals"].to_numpy(float)])
    return X, y


def _predict_rows(model, score: pd.DataFrame, teams: dict, separate: bool) -> tuple[np.ndarray, np.ndarray]:
    k = len(teams)
    n = len(score)
    hi = score["home_team"].map(teams)
    ai = score["away_team"].map(teams)
    X = np.zeros((2 * n, (4 if separate else 2) * k + 1))
    r = np.arange(n)
    for i, (h, a) in enumerate(zip(hi, ai)):
        h_ok, a_ok = h == h, a == a
        h, a = (int(h) if h_ok else 0), (int(a) if a_ok else 0)
        if not separate:
            if h_ok:
                X[i, h] = 1
                X[n + i, k + h] = -1
            if a_ok:
                X[i, k + a] = -1
                X[n + i, a] = 1
            X[i, 2 * k] = 1
        else:
            if h_ok:
                X[i, h] = 1
                X[n + i, 3 * k + h] = -1
            if a_ok:
                X[i, k + a] = -1
                X[n + i, 2 * k + a] = 1
            X[i, 4 * k] = 1
    mu = model.predict(X)
    return mu[r], mu[n + r]


def fit_predict(matches: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import PoissonRegressor
    warnings.filterwarnings("ignore")
    out = []
    for lg, d in matches.groupby("league"):
        d = d.sort_values("date")
        week = d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")
        d = d.assign(_week=week.dt.normalize())
        for w, score in d.loc[d["season"] >= FIRST].groupby("_week"):
            train = d.loc[(d["date"] < w) & (d["date"] >= w - pd.Timedelta(days=WINDOW))]
            if len(train) < 30:
                continue
            teams = {t: i for i, t in enumerate(sorted(set(train["home_team"]) | set(train["away_team"])))}
            age = (w - train["date"]).dt.days.to_numpy(float)
            wt = np.power(2.0, -age / HALF_LIFE)
            row = {"match_id": score["match_id"].to_numpy()}
            for name, sep in (("m1", False), ("m2", True)):
                X, y = _design(train, teams, sep)       # the last column marks home-goal rows (home advantage)
                m = PoissonRegressor(alpha=ALPHA, max_iter=300).fit(X, y, sample_weight=np.concatenate([wt, wt]))
                lh, la = _predict_rows(m, score, teams, sep)
                row[f"lh_{name}"], row[f"la_{name}"] = lh, la
            out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True)


def build(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    path, ipath = C.CACHE / "maher.parquet", C.CACHE / "maher.json"
    if path.exists() and ipath.exists() and not refresh:
        return pd.read_parquet(path), json.loads(ipath.read_text())
    from src.features.store import load_features
    t0 = time.time()
    f = load_features()[["match_id", "league", "season", "date", "home_team", "away_team", "played", "home_goals",
                         "away_goals"]]
    f = f.loc[f["played"].astype(bool) & f["home_goals"].notna()].copy()
    f["date"] = pd.to_datetime(f["date"])
    res = fit_predict(f)
    info = {"rows": int(len(res)), "seconds": round(time.time() - t0, 1), "half_life_days": HALF_LIFE,
            "window_days": WINDOW, "alpha": ALPHA}
    res.to_parquet(path)
    ipath.write_text(json.dumps(info))
    return res, info


if __name__ == "__main__":
    C.setup_logging()
    _, info = build(refresh=True)
    print(info)
