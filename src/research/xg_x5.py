"""Shadow X5 of the xG study (experiments/v6_2/xg, preregistration "primary").

    log mu_side = (1 - w) log lambda_B0_side + w log E[xG_side]

* lambda_B0 — the goal rates of the stored production prediction (snapshot ``xg_home`` / ``xg_away``);
* E[xG] — the opponent-adjusted xG rating of the league, refitted for the match day on the league's matches
  with xG played on an EARLIER day (lag 0): E[xG_home] = exp(mu + hfa + att_home - dfn_away),
  E[xG_away] = exp(mu + att_away - dfn_home), Poisson quasi-likelihood with ridge 5 on att / dfn; at least
  6 earlier matches of the league, otherwise X5 = B0 (fallback);
* w — one weight for both sides, refitted for every 7-day week (weeks start on Tuesday 2026-09-01) by Poisson
  maximum likelihood on all season matches of the xG leagues played before the week, w in [0, 1];
* 1X2: p_k = p_B0,k P_k(mu) / P_k(lambda_B0), renormalised (independent Poisson); markets: the production score
  matrix reconcile_with_1x2(score_matrix(mu_home, mu_away, rho), p); corners unchanged; Main = production
  main_prediction() on those selections with the prices stored for the match at this moment.

xG comes from the Football-Data season files (``HxG`` / ``AxG``) as they are on disk when the shadow runs.
Differences to the study, documented: lambda_B0 of past matches (for w) are the stored production predictions
(the study applied the same model set to the current feature table); w uses only the results on disk at the run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from .. import config
from ..markets.distributions import reconcile_with_1x2, score_matrix
from ..markets.markets import goal_markets
from ..markets.odds import price
from ..markets.recommendation import main_prediction
from . import variants as V

EVAL_START = pd.Timestamp("2026-09-01")
WEEK_DAYS = 7
RIDGE = 5.0
MIN_LEAGUE_MATCHES = 6
MAXG = 10
SEASON_CODE = "2627"


# ------------------------------------------------------------------------------------------------ data
def load_xg(results_dir: Path | None = None, season_code: str = SEASON_CODE) -> pd.DataFrame:
    """Played matches with xG of one season (production match ids), from the Football-Data files on disk."""
    from ..data.cleaning import _parse_dates, normalise
    from ..data.dataset import _read_raw_csv
    from ..leagues import league_codes
    folder = Path(results_dir or config.RESULTS_DIR) / season_code
    frames = []
    for path in sorted(folder.glob("*.csv")):
        raw = _read_raw_csv(path)
        if raw.empty or "HxG" not in raw.columns or "AxG" not in raw.columns:
            continue
        raw = raw.loc[raw["HomeTeam"].notna() & raw["AwayTeam"].notna() & raw["Date"].notna()]
        raw = raw.loc[_parse_dates(raw["Date"].astype(str).str.strip()).notna()].reset_index(drop=True)
        norm = normalise(raw)
        if len(norm) != len(raw):
            continue
        out = norm[["match_id", "league", "date", "home_team", "away_team", "home_goals", "away_goals",
                    "played"]].copy()
        out["xg_home"] = pd.to_numeric(raw["HxG"], errors="coerce").to_numpy()
        out["xg_away"] = pd.to_numeric(raw["AxG"], errors="coerce").to_numpy()
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["match_id", "league", "date", "home_team", "away_team", "home_goals",
                                     "away_goals", "xg_home", "xg_away"])
    x = pd.concat(frames, ignore_index=True)
    ok = (x["league"].isin(league_codes()) & x["played"].astype(bool) & x["xg_home"].between(0, 8)
          & x["xg_away"].between(0, 8))
    return x.loc[ok].drop(columns="played").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ ratings
def fit_ratings(hi, ai, yh, ya, n_teams: int, ridge: float = RIDGE):
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
        g[2:2 + T] = np.bincount(hi, rh, T) + np.bincount(ai, ra, T) + ridge * att
        g[2 + T:] = -np.bincount(ai, rh, T) - np.bincount(hi, ra, T) + ridge * dfn
        return loss, g

    m = max(np.mean(np.r_[yh, ya]), 0.1)
    res = minimize(f, np.r_[np.log(m), 0.1, np.zeros(2 * T)], jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    th = res.x
    return float(th[0]), float(th[1]), th[2:2 + T], th[2 + T:]


def _league_counts(xg: pd.DataFrame, games: pd.DataFrame) -> np.ndarray:
    """Number of xG matches of the game's league played on an earlier day (the data of its rating fit)."""
    out = np.zeros(len(games), dtype=int)
    days = pd.to_datetime(games["date"]).dt.normalize().to_numpy()
    xd = {lg: np.sort(pd.to_datetime(g["date"]).dt.normalize().to_numpy()) for lg, g in xg.groupby("league")}
    for i, (lg, d) in enumerate(zip(games["league"].to_numpy(), days)):
        arr = xd.get(lg)
        out[i] = 0 if arr is None else int(np.searchsorted(arr, d, side="left"))
    return out


def expected_xg(xg: pd.DataFrame, games: pd.DataFrame, cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """E[xG] (home, away) of ``games`` (match_id, league, date, home_team, away_team): per league and match day a
    rating fit on that league's xG matches of earlier days. NaN when fewer than MIN_LEAGUE_MATCHES.

    ``cache``: earlier results (match_id, league_xg_matches, exp_xg_h, exp_xg_a); a cached value is reused only
    when the number of earlier league matches it was fitted on is unchanged."""
    games = games.assign(league_xg_matches=_league_counts(xg, games))
    out = []
    if cache is not None and not cache.empty:
        key = cache.drop_duplicates("match_id", keep="last").set_index("match_id")
        hit = games["match_id"].isin(key.index)
        hit &= games["match_id"].map(key["league_xg_matches"]).to_numpy() == games["league_xg_matches"].to_numpy()
        if hit.any():
            h = games.loc[hit]
            out.append(pd.DataFrame({"match_id": h["match_id"].to_numpy(),
                                     "exp_xg_h": h["match_id"].map(key["exp_xg_h"]).to_numpy(float),
                                     "exp_xg_a": h["match_id"].map(key["exp_xg_a"]).to_numpy(float),
                                     "league_xg_matches": h["league_xg_matches"].to_numpy()}))
        games = games.loc[~hit]
    for (lg, day), g in games.groupby(["league", pd.to_datetime(games["date"]).dt.normalize()]):
        h = xg.loc[(xg["league"] == lg) & (pd.to_datetime(xg["date"]) < day)]
        eh = np.full(len(g), np.nan)
        ea = np.full(len(g), np.nan)
        if len(h) >= MIN_LEAGUE_MATCHES:
            teams = sorted(set(h["home_team"]) | set(h["away_team"]) | set(g["home_team"]) | set(g["away_team"]))
            tix = {t: i for i, t in enumerate(teams)}
            mu, hfa, att, dfn = fit_ratings(h["home_team"].map(tix).to_numpy(), h["away_team"].map(tix).to_numpy(),
                                            h["xg_home"].to_numpy(float), h["xg_away"].to_numpy(float), len(teams))
            ih, ia = g["home_team"].map(tix).to_numpy(), g["away_team"].map(tix).to_numpy()
            eh = np.exp(mu + hfa + att[ih] - dfn[ia])
            ea = np.exp(mu + att[ia] - dfn[ih])
        out.append(pd.DataFrame({"match_id": g["match_id"].to_numpy(), "exp_xg_h": eh, "exp_xg_a": ea,
                                 "league_xg_matches": len(h)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["match_id", "exp_xg_h", "exp_xg_a",
                                                                               "league_xg_matches"])


def week_start(day) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    k = (day - EVAL_START).days // WEEK_DAYS
    return EVAL_START + pd.Timedelta(days=WEEK_DAYS * k)


def _log(x):
    return np.log(np.clip(np.asarray(x, float), 1e-3, None))


def fit_w(xg: pd.DataFrame, lambdas: pd.DataFrame, before: pd.Timestamp, cache: pd.DataFrame | None = None) -> dict:
    """w of the week starting ``before``: Poisson ML on every xG-league match with xG played before it."""
    played = xg.loc[pd.to_datetime(xg["date"]) < before]
    if played.empty:
        return {"w": np.nan, "n": 0, "expected": None}
    e = expected_xg(xg, played[["match_id", "league", "date", "home_team", "away_team"]], cache)
    d = played.merge(e, on="match_id").merge(lambdas, on="match_id")
    d = d.loc[d["exp_xg_h"].notna() & d["lam_h"].notna() & d["home_goals"].notna()]
    if len(d) < 20:
        return {"w": np.nan, "n": int(len(d)), "expected": e}
    lb = np.r_[_log(d["lam_h"]), _log(d["lam_a"])]
    lx = np.r_[_log(d["exp_xg_h"]), _log(d["exp_xg_a"])]
    y = np.r_[d["home_goals"], d["away_goals"]].astype(float)

    def nll(w):
        eta = (1 - w) * lb + w * lx
        return np.sum(np.exp(eta) - y * eta)

    w = float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)
    return {"w": w, "n": int(len(d)), "expected": e}


def pois_1x2(lh, la) -> np.ndarray:
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], np.atleast_1d(lh)[:, None])
    pa = poisson.pmf(k[None, :], np.atleast_1d(la)[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    i, j = np.indices((MAXG + 1, MAXG + 1))
    out = np.stack([m[:, i > j].sum(1), m[:, i == j].sum(1), m[:, i < j].sum(1)], 1)
    return out / out.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ shadow rows
def shadow_rows(ctx: V.Context, doc: dict) -> pd.DataFrame:
    results_dir = ctx.model_meta.get("results_dir")
    xg = ctx.model_meta.get("xg_table")
    if xg is None:
        xg = load_xg(results_dir)
    leagues = set(xg["league"])
    snaps = ctx.snapshots.loc[ctx.snapshots["league"].isin(leagues)]
    if snaps.empty:
        return V._frame([])
    rho = float(ctx.model_meta.get("dc_rho", 0.0) or 0.0)
    hist = ctx.model_meta.get("history_snapshots")
    lambdas = (hist.rename(columns={"xg_home": "lam_h", "xg_away": "lam_a"})[["match_id", "lam_h", "lam_a"]]
               if hist is not None and not hist.empty else pd.DataFrame(columns=["match_id", "lam_h", "lam_a"]))
    cache_path = Path(ctx.research_dir) / "cache" / "x5_expected_xg.parquet" if ctx.research_dir else None
    cache = pd.read_parquet(cache_path) if cache_path is not None and cache_path.exists() else None
    exp = expected_xg(xg, snaps[["match_id", "league", "date", "home_team", "away_team"]]).set_index("match_id")
    weeks = {}
    rows = []
    for snap in snaps.itertuples(index=False):
        mid = snap.match_id
        ws = week_start(snap.date)
        if ws not in weeks:
            weeks[ws] = fit_w(xg, lambdas, ws, cache) if ws >= EVAL_START else {"w": np.nan, "n": 0, "expected": None}
            e_new = weeks[ws].get("expected")
            if e_new is not None and len(e_new):
                cache = e_new if cache is None or cache.empty else pd.concat([cache, e_new]).drop_duplicates(
                    "match_id", keep="last")
        w = weeks[ws]["w"]
        lam_h, lam_a = float(snap.xg_home), float(snap.xg_away)
        e_h, e_a = (float(exp.at[mid, "exp_xg_h"]), float(exp.at[mid, "exp_xg_a"])) if mid in exp.index else (np.nan,
                                                                                                             np.nan)
        fallback = not (np.isfinite(e_h) and np.isfinite(e_a) and np.isfinite(w))
        if fallback:
            mu_h, mu_a = lam_h, lam_a
        else:
            mu_h = float(np.exp((1 - w) * _log(lam_h) + w * _log(e_h)))
            mu_a = float(np.exp((1 - w) * _log(lam_a) + w * _log(e_a)))
        p0 = np.array([snap.p_home, snap.p_draw, snap.p_away], float)
        p0 = p0 / p0.sum()
        r = pois_1x2(mu_h, mu_a)[0] / np.clip(pois_1x2(lam_h, lam_a)[0], 1e-12, None)
        p5 = p0 * r
        p5 = p5 / p5.sum()
        payload = {"lam": [lam_h, lam_a], "mu": [mu_h, mu_a], "exp_xg": [e_h, e_a], "w": w,
                   "w_week": str(ws.date()), "w_n": weeks[ws]["n"], "fallback": bool(fallback),
                   "p_b0": p0.tolist(), "p_x5": p5.tolist(),
                   "league_xg_matches": int(exp.at[mid, "league_xg_matches"]) if mid in exp.index else 0}
        rows.append(V._row(mid, "goals", "GOALS", "MU", np.nan, None, payload=payload))
        # Main of X5 with the prices stored for the match now
        mk = ctx.markets.loc[ctx.markets["match_id"] == mid]
        main = V._pick(ctx.picks, mid, "main")
        if mk.empty or main is None:
            continue
        matrix = reconcile_with_1x2(score_matrix(mu_h, mu_a, rho=rho), p5)
        sels = goal_markets(matrix)
        corners = [s for s in V.selections(mk.loc[mk["group"] == "CORNERS"])]
        quotes = {(r_.market, r_.selection): {"odds": float(r_.odds), "bookmaker": r_.bookmaker,
                                              "provider": r_.provider, "priced_at": V._ts(r_.priced_at)}
                  for r_ in mk.itertuples(index=False) if V._f(r_.odds) == V._f(r_.odds) and V._f(r_.odds) > 1.0}
        priced = price(sels, quotes) + corners
        rec = main_prediction(priced)
        if rec is None:
            continue
        s = rec.pick
        rows.append(V._row(mid, "main", s.market, s.selection, s.probability, main, s.odds if s.has_odds else np.nan,
                           s.bookmaker, s.provider, s.priced_at,
                           {"push": s.push_probability, "kind": rec.kind, "fallback": bool(fallback),
                            "same_as_production": (s.market, s.selection) == (main["market"], main["selection"])}))
    if cache_path is not None and cache is not None and not cache.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.reset_index(drop=True).to_parquet(cache_path, index=False)
    return V._frame(rows)
