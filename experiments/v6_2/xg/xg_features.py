"""Pre-match xG features (step 2 of the study).

For a match on day d only matches of the same league played on an EARLIER day (d' < d - lag) with xG enter
its features — never the match's own xG, never a same-day match. With ``lag = 0`` the latest earlier match
counts as known; ``lag = 4`` days is a sensitivity check for the publication delay of the Football-Data files.

Per team and match (prefix ``h_`` for the home team, ``a_`` for the away team):

    xg_n                 matches with xG before the day (season start: 0, 1, 2, ...)
    xgf5, xga5, xgf10, xga10, xgd10      mean xG for / against over the last 5 / 10 matches (NaN when 0)
    gmx10, gamx10        mean (goals - xG) for and (conceded - xG against) over the last 10: finishing
    xgf_v5, xga_v5       the same at the venue of this match (home team at home, away team away)
    xgf_ewm, xga_ewm     exponentially weighted means, half-life 4 matches
    xgf_s, xga_s         shrunk last-10 means: (n m + k prior) / (n + k), k = 3, prior = league mean xG
    gmx_s                finishing shrunk towards 0: n m / (n + 5)
    att, dfn             opponent-adjusted xG ratings (log scale) of the iterative model below
    gatt, gdfn           the same ratings fitted on goals (control: is xG better than goals in-season?)

Per match: ``exp_xg_h``, ``exp_xg_a`` (expected xG from the ratings), ``exp_g_h``, ``exp_g_a`` (from the goal
ratings), ``lg_xg_h``, ``lg_xg_a`` (league mean xG before the day).

Rating model (per league, refitted for every match day on the earlier matches; Poisson quasi-likelihood,
valid for a non-negative continuous target such as xG):

    E[xG_home] = exp(mu + hfa + att_home - dfn_away),   E[xG_away] = exp(mu + att_away - dfn_home)
    minimise  sum(E - y log E) + lambda / 2 * (sum att^2 + sum dfn^2),   lambda = RIDGE (fixed a priori)

The ridge term pulls a team with few matches to the league average (att = dfn = 0) — the explicit
treatment of the season start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import common as C
from . import data as D

K_SHRINK = 3.0
K_FINISH = 5.0
HALF_LIFE = 4.0
RIDGE = 5.0
MIN_LEAGUE_MATCHES = 6
TEAM_COLS = ["xg_n", "xgf5", "xga5", "xgf10", "xga10", "xgd10", "gmx10", "gamx10", "xgf_v5", "xga_v5",
             "xgf_ewm", "xga_ewm", "xgf_s", "xga_s", "gmx_s", "att", "dfn", "gatt", "gdfn"]


def fit_ratings(hi: np.ndarray, ai: np.ndarray, yh: np.ndarray, ya: np.ndarray, n_teams: int,
                ridge: float = RIDGE) -> tuple[float, float, np.ndarray, np.ndarray]:
    """(mu, hfa, att, dfn) of the multiplicative rating model (see module docstring)."""
    T = n_teams

    def f(theta):
        mu, hfa, att, dfn = theta[0], theta[1], theta[2:2 + T], theta[2 + T:]
        eh = np.exp(mu + hfa + att[hi] - dfn[ai])
        ea = np.exp(mu + att[ai] - dfn[hi])
        loss = np.sum(eh - yh * np.log(eh)) + np.sum(ea - ya * np.log(ea)) + 0.5 * ridge * (att @ att + dfn @ dfn)
        rh, ra = eh - yh, ea - ya
        g = np.zeros_like(theta)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        ga = np.bincount(hi, rh, T) + np.bincount(ai, ra, T) + ridge * att
        gd = -np.bincount(ai, rh, T) - np.bincount(hi, ra, T) + ridge * dfn
        g[2:2 + T], g[2 + T:] = ga, gd
        return loss, g

    m = max(np.mean(np.r_[yh, ya]), 0.1)
    theta0 = np.r_[np.log(m), 0.1, np.zeros(2 * T)]
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    th = res.x
    return float(th[0]), float(th[1]), th[2:2 + T], th[2 + T:]


def _team_stats(rec: pd.DataFrame, venue: str, prior: float) -> dict:
    """rec: the team's earlier records (newest first) with columns xgf, xga, gf, ga, venue."""
    n = len(rec)
    out = {"xg_n": n}
    if n == 0:
        out.update({c: np.nan for c in TEAM_COLS if c not in ("xg_n", "att", "dfn", "gatt", "gdfn")})
        out["xgf_s"] = out["xga_s"] = prior
        out["gmx_s"] = 0.0
        return out
    xf, xa, gf, ga = (rec[c].to_numpy(float) for c in ("xgf", "xga", "gf", "ga"))
    out["xgf5"], out["xga5"] = xf[:5].mean(), xa[:5].mean()
    out["xgf10"], out["xga10"] = xf[:10].mean(), xa[:10].mean()
    out["xgd10"] = out["xgf10"] - out["xga10"]
    out["gmx10"] = (gf[:10] - xf[:10]).mean()
    out["gamx10"] = (ga[:10] - xa[:10]).mean()
    v = rec.loc[rec["venue"] == venue]
    out["xgf_v5"] = v["xgf"].head(5).mean() if len(v) else np.nan
    out["xga_v5"] = v["xga"].head(5).mean() if len(v) else np.nan
    w = 0.5 ** (np.arange(n) / HALF_LIFE)
    out["xgf_ewm"], out["xga_ewm"] = np.sum(w * xf) / w.sum(), np.sum(w * xa) / w.sum()
    k = min(n, 10)
    out["xgf_s"] = (k * xf[:10].mean() + K_SHRINK * prior) / (k + K_SHRINK) if np.isfinite(prior) else np.nan
    out["xga_s"] = (k * xa[:10].mean() + K_SHRINK * prior) / (k + K_SHRINK) if np.isfinite(prior) else np.nan
    out["gmx_s"] = k * out["gmx10"] / (k + K_FINISH)
    return out


def build(lag_days: int = 0, refresh: bool = False) -> pd.DataFrame:
    """Features of every 2026/27 match of the xG leagues (cached per lag)."""
    path = C.CACHE / f"xg_features_lag{lag_days}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = D.raw_xg("2627")
    x = D.clean_xg(raw)
    xg_leagues = sorted(raw.loc[raw["has_xg_column"], "league"].unique())
    sched = D.season_features(columns=["match_id", "league", "date", "home_team", "away_team"])
    sched = sched.loc[sched["league"].isin(xg_leagues) & sched["league"].isin(D.enabled_leagues())]
    lag = pd.Timedelta(days=lag_days)
    rows = []
    for lg, games in sched.groupby("league"):
        hist_lg = x.loc[x["league"] == lg].sort_values("date")
        teams = sorted(set(games["home_team"]) | set(games["away_team"]) | set(hist_lg["home_team"]) | set(hist_lg["away_team"]))
        tix = {t: i for i, t in enumerate(teams)}
        rec = pd.concat([
            pd.DataFrame({"team": hist_lg["home_team"], "date": hist_lg["date"], "xgf": hist_lg["xg_home"],
                          "xga": hist_lg["xg_away"], "gf": hist_lg["home_goals"], "ga": hist_lg["away_goals"], "venue": "H"}),
            pd.DataFrame({"team": hist_lg["away_team"], "date": hist_lg["date"], "xgf": hist_lg["xg_away"],
                          "xga": hist_lg["xg_home"], "gf": hist_lg["away_goals"], "ga": hist_lg["home_goals"], "venue": "A"}),
        ]).sort_values("date", ascending=False)
        for d, day in games.groupby("date"):
            h = hist_lg.loc[hist_lg["date"] < d - lag]
            r = rec.loc[rec["date"] < d - lag]
            n_lg = len(h)
            lg_h = h["xg_home"].mean() if n_lg >= MIN_LEAGUE_MATCHES else np.nan
            lg_a = h["xg_away"].mean() if n_lg >= MIN_LEAGUE_MATCHES else np.nan
            prior = np.nanmean([lg_h, lg_a]) if n_lg >= MIN_LEAGUE_MATCHES else np.nan
            if n_lg >= MIN_LEAGUE_MATCHES:
                hi = h["home_team"].map(tix).to_numpy()
                ai = h["away_team"].map(tix).to_numpy()
                mu, hfa, att, dfn = fit_ratings(hi, ai, h["xg_home"].to_numpy(float), h["xg_away"].to_numpy(float), len(teams))
                gmu, ghfa, gatt, gdfn = fit_ratings(hi, ai, h["home_goals"].to_numpy(float), h["away_goals"].to_numpy(float), len(teams))
            else:
                mu = hfa = gmu = ghfa = np.nan
                att = dfn = gatt = gdfn = np.full(len(teams), np.nan)
            by_team = {t: g for t, g in r.groupby("team")}
            empty = r.iloc[:0]
            for m in day.itertuples(index=False):
                row = {"match_id": m.match_id, "league": lg, "date": d, "lg_n": n_lg, "lg_xg_h": lg_h, "lg_xg_a": lg_a}
                for side, team, venue in (("h", m.home_team, "H"), ("a", m.away_team, "A")):
                    st = _team_stats(by_team.get(team, empty), venue, prior)
                    i = tix[team]
                    st.update({"att": att[i], "dfn": dfn[i], "gatt": gatt[i], "gdfn": gdfn[i]})
                    row.update({f"{side}_{k}": v for k, v in st.items()})
                ih, ia = tix[m.home_team], tix[m.away_team]
                row["exp_xg_h"] = np.exp(mu + hfa + att[ih] - dfn[ia])
                row["exp_xg_a"] = np.exp(mu + att[ia] - dfn[ih])
                row["exp_g_h"] = np.exp(gmu + ghfa + gatt[ih] - gdfn[ia])
                row["exp_g_a"] = np.exp(gmu + gatt[ia] - gdfn[ih])
                rows.append(row)
        C.log.info("xg features %s: %d matches", lg, len(games))
    out = pd.DataFrame(rows)
    out["xg_ok"] = out["exp_xg_h"].notna() & out["exp_xg_a"].notna()
    out["min_xg_n"] = out[["h_xg_n", "a_xg_n"]].min(axis=1)
    out.to_parquet(path)
    return out


def coverage(feat: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Season-start coverage: how many matches have how much xG history (played matches)."""
    f = feat.merge(base[["match_id", "played"]], on="match_id")
    f = f.loc[f["played"].astype(bool)].copy()
    f["month"] = f["date"].dt.strftime("%Y-%m")
    bins = [-1, 0, 2, 4, 7, 100]
    labels = ["0", "1-2", "3-4", "5-7", "8+"]
    f["band"] = pd.cut(f["min_xg_n"], bins, labels=labels)
    by_month = f.groupby("month").agg(matches=("match_id", "size"), ratings=("xg_ok", "mean"),
                                      min_n_mean=("min_xg_n", "mean"),
                                      ge3=("min_xg_n", lambda s: float((s >= 3).mean())),
                                      ge5=("min_xg_n", lambda s: float((s >= 5).mean()))).reset_index()
    band = f["band"].value_counts().reindex(labels).fillna(0).astype(int).to_dict()
    eval_ = f.loc[f["date"] >= C.EVAL_START]
    return {"by_month": by_month.to_dict("records"), "band_all": band,
            "band_eval": eval_["band"].value_counts().reindex(labels).fillna(0).astype(int).to_dict(),
            "eval_matches": int(len(eval_)), "eval_ratings_share": float(eval_["xg_ok"].mean()),
            "eval_min_n_median": float(eval_["min_xg_n"].median())}
