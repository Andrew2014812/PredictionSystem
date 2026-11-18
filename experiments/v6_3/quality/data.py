"""Per-match frame of the quality-vs-coverage study: data completeness strata, model losses, Main outcomes.

Strata of a match (everything is known before kick-off):

* ``n_min`` — the smaller number of league matches the two clubs have played this season before the match
  (features ``h_season_games`` / ``a_season_games``: what the system itself sees);
* ``hist_min`` — the smaller ``form10_n`` of the two clubs (matches in the last-10 form window, across seasons):
  0–9 means a club the system has (almost) no history for — a newcomer from a league we do not load, or a
  phantom created by a misspelt name;
* club status (after v6.2 ``data_quality/teams.py``): established / moved_within (promoted or relegated inside
  our leagues) / returner / outside, the worst of the two clubs, and the club's match number;
* ``dq_missing`` — share of missing result-model features; ``stats_missing`` — share of missing shots /
  shots-on-target / corner features of the corner and result models;
* league and league tier; price availability at T (number of priced selections, OddsPapi price present, Main
  priced at T).

Losses per match: 1X2 log loss of the model and of the de-margined market (closing and at T; excess = model −
market), unit Poisson deviance of home / away goals and corners.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.alt_ml import mainroi as R
from experiments.v6_3.replay import data as RD
from src.features.sets import model_features
from src.leagues import get_league

D = C.study_dirs("quality")
ID = ["match_id", "league", "country", "season", "date", "home_team", "away_team", "played"]
STATUS_ORDER = ["outside", "returner", "moved_within", "established"]
NMIN_BINS = [-1, 2, 5, 9, 100]
NMIN_LABELS = ["0–2", "3–5", "6–9", "10+"]


@lru_cache(maxsize=None)
def features() -> pd.DataFrame:
    return RD.load_features()


# ------------------------------------------------------------------------------------------------ club status
def _sides(f: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([f[ID].rename(columns={"home_team": "team", "away_team": "opp"}).assign(side="h"),
                      f[ID].rename(columns={"away_team": "team", "home_team": "opp"}).assign(side="a")],
                     ignore_index=True)


def team_seasons(f: pd.DataFrame) -> pd.DataFrame:
    """Status of every (league, season, club) — the definition of v6.2 data_quality/teams.py."""
    long = _sides(f.loc[f["played"].astype(bool) | (f["season"] == C.CURRENT)])
    ts = long.groupby(["league", "country", "season", "team"]).size().rename("games").reset_index()
    first = int(ts["season"].min())
    seen = {(r.country, r.team, r.season): r.league for r in ts.itertuples()}
    seasons_of = ts.groupby(["country", "team"])["season"].apply(set).to_dict()
    status = []
    for r in ts.itertuples():
        prev = seen.get((r.country, r.team, r.season - 1))
        earlier = [s for s in seasons_of[(r.country, r.team)] if s < r.season - 1]
        if r.season == first:
            status.append("first_season")
        elif prev == r.league:
            status.append("established")
        elif prev is not None:
            status.append("moved_within")
        elif earlier:
            status.append("returner")
        else:
            status.append("outside")
    ts["status"] = status
    return ts


def match_status(f: pd.DataFrame) -> pd.DataFrame:
    ts = team_seasons(f).set_index(["league", "season", "team"])["status"]
    long = _sides(f)
    long["status"] = [ts.get((lg, s, t), "outside") for lg, s, t in zip(long["league"], long["season"], long["team"])]
    st = long.pivot(index="match_id", columns="side", values="status")
    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    out = pd.DataFrame({"h_status": st["h"], "a_status": st["a"]})
    out["club_group"] = [min((h, a), key=lambda s: rank.get(s, 9)) for h, a in zip(out["h_status"], out["a_status"])]
    return out


# ------------------------------------------------------------------------------------------------ losses
def poisson_unit_deviance(y, mu) -> np.ndarray:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def add_losses(x: pd.DataFrame, prefix_q: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    y = x["result"].map({"H": 0, "D": 1, "A": 2})
    P = x[["p_home", "p_draw", "p_away"]].to_numpy(float)
    ok = y.notna().to_numpy()
    yi = y.fillna(0).astype(int).to_numpy()
    x["ll"] = np.where(ok, -np.log(np.clip(P[np.arange(len(x)), yi], 1e-9, 1)), np.nan)
    for name, cols in prefix_q.items():
        Q = x[list(cols)].to_numpy(float)
        x[f"ll_{name}"] = np.where(ok, -np.log(np.clip(Q[np.arange(len(x)), yi], 1e-9, 1)), np.nan)
        x[f"excess_{name}"] = x["ll"] - x[f"ll_{name}"]
    x["dev_home"] = np.where(x["home_goals"].notna(), poisson_unit_deviance(x["home_goals"].fillna(0), x["lam_home"]), np.nan)
    x["dev_away"] = np.where(x["away_goals"].notna(), poisson_unit_deviance(x["away_goals"].fillna(0), x["lam_away"]), np.nan)
    ok_c = x["total_corners"].notna() & x["corner_mu"].notna()
    x["dev_corners"] = np.where(ok_c, poisson_unit_deviance(x["total_corners"].fillna(0), x["corner_mu"].fillna(1)), np.nan)
    return x


# ------------------------------------------------------------------------------------------------ frame
def strata_columns(f: pd.DataFrame) -> pd.DataFrame:
    num, _ = model_features("result", list(f.columns))
    corner_num, _ = model_features("corners", list(f.columns))
    stat_cols = sorted({c for c in num + corner_num if re.search(r"(shot|sot|corner)", c)})
    q = pd.DataFrame({"match_id": f["match_id"].to_numpy()}, index=f.index)
    q["dq_missing"] = f[num].isna().mean(axis=1)
    q["stats_missing"] = f[stat_cols].isna().mean(axis=1)
    q["n_min"] = np.minimum(f["h_season_games"], f["a_season_games"])
    q["hist_min"] = np.minimum(f["h_form10_n"], f["a_form10_n"])
    q["n_bucket"] = pd.cut(q["n_min"], NMIN_BINS, labels=NMIN_LABELS).astype(str)
    q["tier"] = f["league"].map(lambda lg: get_league(lg).tier)
    return q


def frame_current() -> pd.DataFrame:
    """Every played 2026/27 match with strata, production outputs, losses and Baseline B Main (both universes)."""
    m = R.matches()
    f = features()
    st = match_status(f.loc[f["season"] >= 2021])
    x = m[["match_id", "league", "date", "home_team", "away_team", "played", "result", "home_goals", "away_goals",
           "total_corners", "h_season_games", "a_season_games", "h_form10_n", "a_form10_n"]].copy()
    x = x.loc[x["played"].astype(bool)]
    x = x.merge(strata_columns(m), on="match_id").join(st, on="match_id")
    out = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    x = x.merge(out, on="match_id", how="left")
    for u in R.UNIVERSES:
        q = R.market_probs(u).add_suffix(f"_{u}")
        x = x.join(q, on="match_id")
    x = add_losses(x, {"close": ("q_home_close", "q_draw_close", "q_away_close"),
                       "t09": ("q_home_t09", "q_draw_t09", "q_away_t09")})
    # price availability at T
    pt = R.prices("t09")
    cands_t = R.candidates_b("t09")
    x = x.join(cands_t.groupby("match_id")["odds"].apply(lambda s: int(s.notna().sum())).rename("n_priced_t09"),
               on="match_id")
    op = set(pt.loc[pt["provider"] == "oddspapi", "match_id"])
    x["oddspapi_at_t09"] = x["match_id"].isin(op)
    for u in R.UNIVERSES:
        b = R.baseline_b(u).set_index("match_id")
        x[f"main_{u}_profit"] = x["match_id"].map(b["profit"])
        x[f"main_{u}_bet"] = x["match_id"].map(b["bet"]).fillna(0).astype(int)
        x[f"main_{u}_won"] = x["match_id"].map((b["status"] == "WON").astype(float).where(b["status"].isin(["WON", "LOST"])))
        x[f"main_{u}_group"] = x["match_id"].map(b["group"])
        x[f"main_{u}_priced"] = x["match_id"].map(b["odds"].notna())
    return x.reset_index(drop=True)


def frame_hist(season: int, outputs: pd.DataFrame) -> pd.DataFrame:
    """A selection season (2024/25 or 2025/26): out-of-sample outputs, strata and losses against the
    Football-Data Bet365 de-margined probabilities (market_prob_* of the feature table)."""
    f = features()
    st = match_status(f.loc[f["season"] >= 2021])
    s = f.loc[(f["season"] == season) & f["played"].astype(bool)]
    x = s[["match_id", "league", "date", "home_team", "away_team", "played", "result", "home_goals", "away_goals",
           "total_corners", "h_season_games", "a_season_games", "h_form10_n", "a_form10_n",
           "market_prob_home", "market_prob_draw", "market_prob_away"]].copy()
    x = x.merge(strata_columns(s), on="match_id").join(st, on="match_id").merge(outputs, on="match_id", how="left")
    x = x.rename(columns={"market_prob_home": "q_home_fd", "market_prob_draw": "q_draw_fd", "market_prob_away": "q_away_fd"})
    return add_losses(x, {"fd": ("q_home_fd", "q_draw_fd", "q_away_fd")}).reset_index(drop=True)
