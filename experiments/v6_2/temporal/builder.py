"""Candidate temporal features, computed strictly from matches played BEFORE the kick-off day.

Same timeline as the production builder (src/features/builder.py): matches are processed day by day; the
features of every match of day D are read from the state built from days < D, and only then are the results of
day D added. History is kept per (country, team) across seasons and divisions, as in production.

The builder is self-contained: it needs only the raw match columns (date, teams, goals, shots, shots on target),
so the leakage test (leakage.py) can perturb future results and rebuild everything, Elo included. Elo is the
production class (src/features/state.EloRatings), used read-only; opponents' pre-match season averages are kept
here exactly as production keeps them.

Families (h_ / a_ prefix per side; diff_* = home − away):
    form3   last-3 means: points, goal difference, goals for / against, shots on target for / against
    ewm     exponentially weighted means (half-life 3 and 8 matches, last 40 matches): the same statistics
    oadj    opponent-adjusted form over 5 and 15 matches (+ EW half-life 8): goals minus what that opponent usually
            concedes / scores (pre-match season average), Elo surprise = result − Elo expectation
    venue   home form at home / away form away: points (5, 10), goal difference, goals for / against (10)
    slope   OLS slope over the last 10 matches of goal difference, shot-on-target difference and points;
            momentum = last-5 minus last-15 mean (points, goal difference)
    elo     Elo trend: sum of the team's own Elo updates over its last 5 / 10 matches and over this season
    cong    fixture congestion: league matches in the last 7 / 14 / 21 / 28 days, days since the 2nd-last match
    cp      current runs (wins, unbeaten, winless, losses) and a change-point statistic on the last 15 points
            sequences: shift of the mean after the best split (min. 3 matches each side) and matches since it
Only league matches are in the data (Football-Data): congestion ignores cup and European matches.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.features.state import EloRatings
from src.leagues import get_league

RAW = ["match_id", "league", "country", "season", "date", "kickoff", "home_team", "away_team", "home_goals",
       "away_goals", "home_shots", "away_shots", "home_sot", "away_sot"]
MAXH = 40
REST_CAP = 30
# record layout of one team-match (team perspective)
R = {k: i for i, k in enumerate(("day", "home", "gf", "ga", "shf", "sha", "sotf", "sota", "pts", "win", "draw", "loss",
                                  "surp", "adjgf", "adjga", "delo", "season"))}

FAMILIES: dict[str, list[str]] = {
    "form3": [r"^[ha]_f3_", r"^diff_f3_"],
    "ewm": [r"^[ha]_ew(3|8)_(ppg|gd|gf|ga|sotf|sota)$", r"^diff_ew8_ppg$"],
    "oadj": [r"^[ha]_f(5|15)_(adjgf|adjga|surp)$", r"^[ha]_ew8_(adjgf|adjga|surp)$"],
    "venue": [r"^[ha]_vf", r"^diff_vf10_ppg$"],
    "slope": [r"^[ha]_(sl10_|mom_)", r"^diff_sl10_gd$"],
    "elo": [r"^[ha]_elo_d", r"^diff_elo_d10$"],
    "cong": [r"^[ha]_(n7|n14|n21|n28|rest2)$", r"^diff_n14$"],
    "cp": [r"^[ha]_(streak_|cp_)"],
}
FAMILY_RU = {"form3": "форма за 3 матча", "ewm": "экспоненциальные средние", "oadj": "поправка на силу соперника",
             "venue": "форма дома / в гостях", "slope": "наклоны и импульс", "elo": "тренд Elo",
             "cong": "плотность календаря", "cp": "серии и точка изменения"}


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def _mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _slope(y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    y = y[ok]
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _run(flags: np.ndarray) -> int:
    """Length of the current run of True at the end of ``flags``."""
    n = 0
    for f in flags[::-1]:
        if not f:
            break
        n += 1
    return n


def _change_point(pts: np.ndarray, min_seg: int = 3) -> tuple[float, float]:
    """Best single split of the last points sequence by the two-sample t statistic: (mean after − mean before,
    matches since the split)."""
    n = len(pts)
    if n < 2 * min_seg:
        return np.nan, np.nan
    best, shift, since = -1.0, np.nan, np.nan
    for k in range(min_seg, n - min_seg + 1):
        a, b = pts[:k], pts[k:]
        sd = np.sqrt(a.var(ddof=0) / len(a) + b.var(ddof=0) / len(b)) + 0.25
        t = abs(b.mean() - a.mean()) / sd
        if t > best:
            best, shift, since = t, float(b.mean() - a.mean()), float(n - k)
    return shift, since


def team_features(hist: np.ndarray, venue: int, day: int, season: int, out: dict, side: str) -> None:
    """Features of one side from its past records ``hist`` (rows = matches, oldest first)."""
    n = len(hist)
    if n == 0:
        out[f"{side}_n7"] = out[f"{side}_n14"] = out[f"{side}_n21"] = out[f"{side}_n28"] = 0.0
        return
    col = {k: hist[:, i] for k, i in R.items()}
    gd = col["gf"] - col["ga"]
    sotd = col["sotf"] - col["sota"]
    # form3 / reproduction windows
    for k, tag in ((3, "f3"),):
        sl = slice(max(0, n - k), n)
        out[f"{side}_{tag}_ppg"] = _mean(col["pts"][sl])
        out[f"{side}_{tag}_gd"] = _mean(gd[sl])
        out[f"{side}_{tag}_gf"] = _mean(col["gf"][sl])
        out[f"{side}_{tag}_ga"] = _mean(col["ga"][sl])
        out[f"{side}_{tag}_sotf"] = _mean(col["sotf"][sl])
        out[f"{side}_{tag}_sota"] = _mean(col["sota"][sl])
    # exponentially weighted means (most recent match: age 0)
    age = np.arange(n - 1, -1, -1, dtype=float)
    for hl in (3, 8):
        w = 0.5 ** (age / hl)
        out[f"{side}_ew{hl}_ppg"] = _wmean(col["pts"], w)
        out[f"{side}_ew{hl}_gd"] = _wmean(gd, w)
        out[f"{side}_ew{hl}_gf"] = _wmean(col["gf"], w)
        out[f"{side}_ew{hl}_ga"] = _wmean(col["ga"], w)
        out[f"{side}_ew{hl}_sotf"] = _wmean(col["sotf"], w)
        out[f"{side}_ew{hl}_sota"] = _wmean(col["sota"], w)
        if hl == 8:
            out[f"{side}_ew8_adjgf"] = _wmean(col["adjgf"], w)
            out[f"{side}_ew8_adjga"] = _wmean(col["adjga"], w)
            out[f"{side}_ew8_surp"] = _wmean(col["surp"], w)
    # opponent-adjusted windows (5, 15) and reproduction of production's form10 / form5 columns
    for k in (5, 10, 15):
        sl = slice(max(0, n - k), n)
        if k != 10:
            out[f"{side}_f{k}_adjgf"] = _mean(col["adjgf"][sl])
            out[f"{side}_f{k}_adjga"] = _mean(col["adjga"][sl])
            out[f"{side}_f{k}_surp"] = _mean(col["surp"][sl])
        else:
            out[f"{side}_rep_f10_adjgf"] = _mean(col["adjgf"][sl])
            out[f"{side}_rep_f10_surp"] = _mean(col["surp"][sl])
    out[f"{side}_rep_f5_gf"] = _mean(col["gf"][-5:])
    out[f"{side}_rep_f5_ppg"] = _mean(col["pts"][-5:])
    # venue form (same venue as this match)
    v = hist[col["home"] == (1.0 if venue == 0 else 0.0)]
    if len(v):
        vgd = v[:, R["gf"]] - v[:, R["ga"]]
        out[f"{side}_vf5_ppg"] = _mean(v[-5:, R["pts"]])
        out[f"{side}_vf10_ppg"] = _mean(v[-10:, R["pts"]])
        out[f"{side}_vf10_gd"] = _mean(vgd[-10:])
        out[f"{side}_vf10_gf"] = _mean(v[-10:, R["gf"]])
        out[f"{side}_vf10_ga"] = _mean(v[-10:, R["ga"]])
        out[f"{side}_rep_vf5_gf"] = _mean(v[-5:, R["gf"]])
    # slopes and momentum
    out[f"{side}_sl10_gd"] = _slope(gd[-10:])
    out[f"{side}_sl10_sotd"] = _slope(sotd[-10:])
    out[f"{side}_sl10_pts"] = _slope(col["pts"][-10:])
    if n >= 5:
        out[f"{side}_mom_ppg"] = _mean(col["pts"][-5:]) - _mean(col["pts"][-15:])
        out[f"{side}_mom_gd"] = _mean(gd[-5:]) - _mean(gd[-15:])
    # Elo trend (sum of the team's own Elo updates)
    out[f"{side}_elo_d5"] = float(np.nansum(col["delo"][-5:]))
    out[f"{side}_elo_d10"] = float(np.nansum(col["delo"][-10:]))
    out[f"{side}_elo_dseason"] = float(np.nansum(col["delo"][col["season"] == season]))
    # congestion and rest
    days = day - col["day"]
    for w in (7, 14, 21, 28):
        out[f"{side}_n{w}"] = float((days <= w).sum())
    out[f"{side}_rep_rest"] = float(min(days[-1], REST_CAP))
    if n >= 2:
        out[f"{side}_rest2"] = float(min(days[-2], 60))
    # runs and change point
    out[f"{side}_streak_w"] = float(_run(col["win"] == 1))
    out[f"{side}_streak_unb"] = float(_run(col["loss"] == 0))
    out[f"{side}_streak_winless"] = float(_run(col["win"] == 0))
    out[f"{side}_streak_l"] = float(_run(col["loss"] == 1))
    shift, since = _change_point(col["pts"][-15:])
    out[f"{side}_cp_shift"], out[f"{side}_cp_since"] = shift, since


def _diffs(f: dict) -> None:
    for name in ("f3_ppg", "ew8_ppg", "vf10_ppg", "sl10_gd", "elo_d10", "n14"):
        a, b = f.get(f"h_{name}", np.nan), f.get(f"a_{name}", np.nan)
        f[f"diff_{name}"] = a - b


def build(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match of ``matches`` (played matches with RAW columns) with every candidate feature."""
    m = matches[RAW].copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "kickoff", "league", "home_team"], kind="mergesort").reset_index(drop=True)
    day0 = m["date"].min()
    m["_day"] = (m["date"] - day0).dt.days.astype(int)
    hist: dict[tuple, list] = defaultdict(list)
    season_acc: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))   # (league, season, team): games, gf, ga
    elo = EloRatings()
    rows = []
    cols = {c: m[c].to_numpy() for c in m.columns}
    order = np.flatnonzero(np.r_[True, cols["_day"][1:] != cols["_day"][:-1]])
    bounds = list(order) + [len(m)]
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        ctx = []
        for i in range(b0, b1):
            country, lg, s, day = cols["country"][i], cols["league"][i], int(cols["season"][i]), int(cols["_day"][i])
            th, ta = (country, cols["home_team"][i]), (country, cols["away_team"][i])
            tier = get_league(lg).tier
            rh, ra = elo.get(th, s, tier), elo.get(ta, s, tier)
            exp_h = elo.expected_home(rh, ra)
            f = {"match_id": cols["match_id"][i], "x_elo_h": rh, "x_elo_a": ra}
            for side, team, venue in (("h", th, 0), ("a", ta, 1)):
                hh = hist.get(team)
                team_features(np.array(hh[-MAXH:]) if hh else np.empty((0, len(R))), venue, day, s, f, side)
            _diffs(f)
            rows.append(f)
            acc_h, acc_a = season_acc.get((lg, s, th[1])), season_acc.get((lg, s, ta[1]))
            ctx.append((th, ta, exp_h, acc_h, acc_a, s, tier, lg))
        # results of the day enter the state only after every match of the day has its features
        for i, (th, ta, exp_h, acc_h, acc_a, s, tier, lg) in zip(range(b0, b1), ctx):
            hg, ag = float(cols["home_goals"][i]), float(cols["away_goals"][i])
            score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            gdiff = abs(hg - ag)
            g = 1.0 if gdiff <= 1 else (1.5 if gdiff == 2 else (11 + gdiff) / 8)
            delta = elo.k * g * (score - exp_h)
            # opponent's pre-match season averages (NaN when the opponent has no game this season yet)
            a_ga = acc_a[2] / acc_a[0] if acc_a is not None and acc_a[0] > 0 else np.nan
            a_gf = acc_a[1] / acc_a[0] if acc_a is not None and acc_a[0] > 0 else np.nan
            h_ga = acc_h[2] / acc_h[0] if acc_h is not None and acc_h[0] > 0 else np.nan
            h_gf = acc_h[1] / acc_h[0] if acc_h is not None and acc_h[0] > 0 else np.nan
            day = int(cols["_day"][i])
            for team, gf, ga, shf, sha, sotf, sota, home, surp, og, oc, de in (
                    (th, hg, ag, cols["home_shots"][i], cols["away_shots"][i], cols["home_sot"][i], cols["away_sot"][i],
                     1.0, score - exp_h, a_ga, a_gf, delta),
                    (ta, ag, hg, cols["away_shots"][i], cols["home_shots"][i], cols["away_sot"][i], cols["home_sot"][i],
                     0.0, (1 - score) - (1 - exp_h), h_ga, h_gf, -delta)):
                win, draw = float(gf > ga), float(gf == ga)
                hist[team].append([day, home, gf, ga, float(shf), float(sha), float(sotf), float(sota),
                                   3 * win + draw, win, draw, float(gf < ga), surp, gf - og, ga - oc, de, float(s)])
                if len(hist[team]) > 2 * MAXH:
                    del hist[team][:MAXH]
            season_acc[(lg, s, th[1])] += (1, hg, ag)
            season_acc[(lg, s, ta[1])] += (1, ag, hg)
            elo.update(th, ta, hg, ag, s, tier)
    return pd.DataFrame(rows)


def family_columns(columns, families) -> list[str]:
    import re
    pats = [re.compile(p) for fam in families for p in FAMILIES[fam]]
    return [c for c in columns if any(p.search(c) for p in pats)]
