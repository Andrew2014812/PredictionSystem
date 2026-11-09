"""Independent recomputation of rolling features from the raw files, compared with the production feature table.

The raw Football-Data CSVs are parsed here with plain pandas (not src/data/cleaning.normalise), the API
providers' played rows are added only when Football-Data has no row of that league, day and club, and every
rolling feature is recomputed with vectorised group operations (shift + rolling / expanding means) instead of
the production state machine (src/features/state.py). Elo is recomputed in a separate loop from the published
rule (K = 20, home advantage 60, carry-over 0.8, base 1500 − 100·(tier − 1), goal-difference multiplier).

Compared on every played match of the production table that the independent table also has:
form windows 5/10/15, season and venue averages, points, games, rest days, head-to-head count, Elo.
Writes results/recompute_summary.csv, recompute_examples.csv, recompute.json.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.leagues import LEAGUES, league_codes

from . import common as C

RAW = {"FTHG": "hg", "FTAG": "ag", "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac"}
COUNTRY = {lg.code: lg.country for lg in LEAGUES}
TIER = {lg.code: lg.tier for lg in LEAGUES}


def _date(s: pd.Series) -> pd.Series:
    a = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return a.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))


def raw_matches() -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(C.DATA / "results" / "*" / "*.csv"))):
        if Path(p).stem not in league_codes():
            continue
        try:
            raw = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(p, encoding="latin-1", low_memory=False)
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG"])
        d = pd.DataFrame({"league": raw["Div"].str.strip(), "date": _date(raw["Date"].str.strip()),
                          "home": raw["HomeTeam"].str.strip(), "away": raw["AwayTeam"].str.strip()})
        for src, dst in RAW.items():
            d[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
        d["source"] = "fd"
        frames.append(d)
    fd = pd.concat(frames, ignore_index=True)
    live = pd.read_parquet(C.LIVE_FILE)
    live = live.loc[live["played"].astype(bool) & live["league"].isin(league_codes())]
    live = pd.DataFrame({"league": live["league"], "date": pd.to_datetime(live["date"]), "home": live["home_team"],
                         "away": live["away_team"], "hg": live["home_goals"].astype(float),
                         "ag": live["away_goals"].astype(float), "source": "api"})
    busy = set(zip(fd["league"], fd["date"], fd["home"])) | set(zip(fd["league"], fd["date"], fd["away"]))
    live = live.loc[[(lg, d, h) not in busy and (lg, d, a) not in busy
                     for lg, d, h, a in zip(live["league"], live["date"], live["home"], live["away"])]]
    live = live.drop_duplicates(["league", "date", "home", "away"])
    m = pd.concat([fd, live], ignore_index=True)
    m["country"] = m["league"].map(COUNTRY)
    m["season"] = np.where(m["date"].dt.month >= 7, m["date"].dt.year, m["date"].dt.year - 1)
    slug = lambda s: s.str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True).str.strip("-")   # noqa: E731
    m["match_id"] = m["league"] + "-" + m["date"].dt.strftime("%Y%m%d") + "-" + slug(m["home"]) + "-" + slug(m["away"])
    return m.sort_values(["date", "league", "home"]).reset_index(drop=True)


def _long(m: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame({"match_id": m["match_id"], "country": m["country"], "league": m["league"], "season": m["season"],
                      "date": m["date"], "team": m["home"], "venue": "home", "gf": m["hg"], "ga": m["ag"],
                      "shots_for": m["hs"], "sot_against": m["ast"], "corners_for": m["hc"]})
    a = pd.DataFrame({"match_id": m["match_id"], "country": m["country"], "league": m["league"], "season": m["season"],
                      "date": m["date"], "team": m["away"], "venue": "away", "gf": m["ag"], "ga": m["hg"],
                      "shots_for": m["as_"], "sot_against": m["hst"], "corners_for": m["ac"]})
    t = pd.concat([h, a], ignore_index=True)
    t["pts"] = np.where(t["gf"] > t["ga"], 3.0, np.where(t["gf"] == t["ga"], 1.0, 0.0))
    t["over25"] = (t["gf"] + t["ga"] > 2.5).astype(float)
    return t.sort_values(["country", "team", "date"]).reset_index(drop=True)


def _prior_rolling(t: pd.DataFrame, keys: list[str], col: str, k: int | None) -> pd.Series:
    g = t.groupby(keys, sort=False)[col]
    if k is None:
        return g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return g.transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())


def team_features(t: pd.DataFrame) -> pd.DataFrame:
    team, season, venue = ["country", "team"], ["league", "season", "team"], ["country", "team", "venue"]
    out = t[["match_id", "venue"]].copy()
    n_prior = t.groupby(team).cumcount()
    for k in (5, 10, 15):
        out[f"form{k}_n"] = np.minimum(n_prior, k).astype(float)
    out["form5_ppg"] = _prior_rolling(t, team, "pts", 5)
    out["form10_gf"] = _prior_rolling(t, team, "gf", 10)
    out["form10_ga"] = _prior_rolling(t, team, "ga", 10)
    out["form10_shots_for"] = _prior_rolling(t, team, "shots_for", 10)
    out["form10_corners_for"] = _prior_rolling(t, team, "corners_for", 10)
    out["form15_sot_against"] = _prior_rolling(t, team, "sot_against", 15)
    out["freq_over25"] = _prior_rolling(t, team, "over25", 10)
    out["season_games"] = t.groupby(season).cumcount().astype(float)
    out["points"] = t.groupby(season)["pts"].transform(lambda s: s.shift(1).cumsum())
    out["season_ppg"] = _prior_rolling(t, season, "pts", None)
    out["season_gf"] = _prior_rolling(t, season, "gf", None)
    out["venue_gf"] = _prior_rolling(t, ["league", "season", "team", "venue"], "gf", None)
    out["venue_form5_gf"] = _prior_rolling(t, venue, "gf", 5)
    prev = t.groupby(team)["date"].shift(1)
    out["rest_days"] = np.minimum((t["date"] - prev).dt.days, 30)
    # with no previous match in the season the production builder leaves season values out (season_games = 0)
    for c in ("points", "season_ppg", "season_gf"):
        out.loc[out["season_games"] == 0, c] = np.nan
    return out


def h2h_counts(m: pd.DataFrame) -> pd.Series:
    pair = m["country"] + "|" + np.where(m["home"] < m["away"], m["home"] + "|" + m["away"], m["away"] + "|" + m["home"])
    n = m.assign(pair=pair).sort_values("date").groupby("pair").cumcount()
    return np.minimum(n, 5).astype(float).reindex(m.index)


def elo(m: pd.DataFrame) -> pd.DataFrame:
    ratings, season_of = {}, {}
    out_h, out_a = np.empty(len(m)), np.empty(len(m))

    def get(key, season, tier):
        base = 1500.0 - 100.0 * (tier - 1)
        if key not in ratings:
            ratings[key], season_of[key] = base, season
        elif season_of[key] != season:
            ratings[key] = 0.8 * ratings[key] + 0.2 * base
            season_of[key] = season
        return ratings[key]

    order = m.sort_values(["date"]).index
    dates = m.loc[order, "date"].to_numpy()
    i = 0
    idx = list(order)
    while i < len(idx):
        j = i
        while j < len(idx) and dates[j] == dates[i]:
            j += 1
        day = idx[i:j]
        for r in day:                           # read the whole day first
            row = m.loc[r]
            tier = TIER[row["league"]]
            out_h[r] = get((row["country"], row["home"]), row["season"], tier)
            out_a[r] = get((row["country"], row["away"]), row["season"], tier)
        for r in day:                           # then update with the day's results
            row = m.loc[r]
            tier = TIER[row["league"]]
            kh, ka = (row["country"], row["home"]), (row["country"], row["away"])
            rh, ra = get(kh, row["season"], tier), get(ka, row["season"], tier)
            e = 1 / (1 + 10 ** (-(rh + 60 - ra) / 400))
            s = 1.0 if row["hg"] > row["ag"] else (0.5 if row["hg"] == row["ag"] else 0.0)
            gd = abs(row["hg"] - row["ag"])
            g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
            d = 20 * g * (s - e)
            ratings[kh], ratings[ka] = rh + d, ra - d
        i = j
    return pd.DataFrame({"h_elo": out_h, "a_elo": out_a}, index=m.index)


COMPARE = ["form5_n", "form10_n", "form15_n", "form5_ppg", "form10_gf", "form10_ga", "form10_shots_for",
           "form10_corners_for", "form15_sot_against", "freq_over25", "season_games", "points", "season_ppg",
           "season_gf", "venue_gf", "venue_form5_gf", "rest_days"]


def run() -> dict:
    m = raw_matches()
    t = _long(m)
    tf = team_features(t)
    home = tf.loc[tf["venue"] == "home"].drop(columns="venue").set_index("match_id").add_prefix("h_")
    away = tf.loc[tf["venue"] == "away"].drop(columns="venue").set_index("match_id").add_prefix("a_")
    ind = m.set_index("match_id")[["league", "season", "source"]].join(home).join(away)
    ind["h2h_n"] = h2h_counts(m).to_numpy()
    e = elo(m)
    ind["h_elo"], ind["a_elo"] = e["h_elo"].to_numpy(), e["a_elo"].to_numpy()
    cols = [f"{s}_{c}" for c in COMPARE for s in ("h", "a")] + ["h2h_n", "h_elo", "a_elo"]
    prod = C.features(["match_id", "played", "season"] + [c for c in cols if c != "h2h_n"] + ["h2h_n"])
    prod = prod.loc[prod["played"].astype(bool)].set_index("match_id")
    both = prod.index.intersection(ind.index)
    only_prod = prod.index.difference(ind.index)
    only_ind = ind.index.difference(prod.index)
    rows, examples = [], []
    for c in cols:
        a, b = prod.loc[both, c].astype(float), ind.loc[both, c].astype(float)
        nan_mismatch = (a.isna() != b.isna())
        diff = (a - b).abs()
        tol = 1e-6 * np.maximum(1.0, b.abs())
        bad = nan_mismatch | (diff > tol)
        rows.append({"feature": c, "compared": int(len(both)), "both_missing": int((a.isna() & b.isna()).sum()),
                     "missing_mismatch": int(nan_mismatch.sum()), "value_mismatch": int(((diff > tol) & ~nan_mismatch).sum()),
                     "mismatch_share": float(bad.mean()), "max_abs_diff": float(diff.max()) if diff.notna().any() else 0.0})
        for mid in list(both[bad.to_numpy()])[:3]:
            examples.append({"feature": c, "match_id": mid, "production": a.at[mid], "independent": b.at[mid]})
    summary = pd.DataFrame(rows)
    C.save_csv("recompute_summary.csv", summary)
    C.save_csv("recompute_examples.csv", pd.DataFrame(examples))
    out = {"independent_matches": int(len(ind)), "production_played": int(len(prod)), "compared_matches": int(len(both)),
           "only_in_production": int(len(only_prod)), "only_in_independent": int(len(only_ind)),
           "only_in_production_ids": list(only_prod[:20]), "only_in_independent_ids": list(only_ind[:20]),
           "features_compared": len(cols), "cells_compared": int(len(both) * len(cols)),
           "cells_mismatch": int(sum(r["missing_mismatch"] + r["value_mismatch"] for r in rows)),
           "features_with_mismatch": [r["feature"] for r in rows if r["missing_mismatch"] + r["value_mismatch"] > 0]}
    C.save_json("recompute.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("recompute")
    run()
