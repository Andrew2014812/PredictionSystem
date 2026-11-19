"""Data checks after the daily update (v6.3): matches, results, statistics, prices and predictions.

Like ``invariants.check_history`` every check returns a list of examples (empty = fine) and nothing is repaired
here: a problem is written to the run's diagnostics (``local_data/api/diagnostics``) instead of being hidden.

Matches
* the same match stored twice (one id twice, or the same pairing on the same day under two ids);
* a pairing that appears with home and away swapped on the same day (one source turned the match round);
* a played match without a score, or an unplayed one with a score; a played match dated in the future;
* a result that a provider has (API-Football / OddsPapi live table) but the match table does not use.
Statistics
* impossible values: negative counts, more than 15 goals for a side, more than 35 corners in a match,
  shots on target above shots, corners present for one side only.
Prices
* a stored price that is not a price (≤ 1.00, missing or above 1 000; 1.01 is a real quote), a price collected
  after kick-off.
Predictions
* a recorded (live) prediction whose last version was generated after kick-off (finished live predictions are
  frozen);
* a settled prediction whose status does not follow from the stored result;
* pre-match features that may know the future: a team's "matches played this season" larger than the number of
  its earlier matches in the table (checked on the recent matches).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..markets.markets import settle_selection

MAX_EXAMPLES = 20
RECENT_DAYS = 21


def _ex(values) -> list:
    return list(values)[:MAX_EXAMPLES]


def check_matches(features: pd.DataFrame, today: pd.Timestamp | None = None) -> dict[str, list]:
    out: dict[str, list] = {}
    f = features
    today = pd.Timestamp(today or pd.Timestamp.now().normalize())
    dup_id = f.loc[f["match_id"].duplicated(), "match_id"]
    if len(dup_id):
        out["match id stored twice"] = _ex(dup_id)
    key = ["league", "date", "home_team", "away_team"]
    dup_pair = f.loc[f.duplicated(key, keep=False)].sort_values(key)
    if len(dup_pair):
        out["same pairing twice on one day"] = _ex(dup_pair["match_id"])
    pairs = set(zip(f["league"], f["date"], f["home_team"], f["away_team"]))
    swapped = [m for m, lg, d, h, a in zip(f["match_id"], f["league"], f["date"], f["home_team"], f["away_team"])
               if (lg, d, a, h) in pairs and h < a]
    if swapped:
        out["home and away swapped between two rows"] = _ex(swapped)
    played = f["played"].astype(bool)
    no_score = f.loc[played & (f["home_goals"].isna() | f["away_goals"].isna()), "match_id"]
    if len(no_score):
        out["played without a score"] = _ex(no_score)
    score_unplayed = f.loc[~played & f["home_goals"].notna(), "match_id"]
    if len(score_unplayed):
        out["score of an unplayed match"] = _ex(score_unplayed)
    future = f.loc[played & (pd.to_datetime(f["date"]) > today + pd.Timedelta(days=1)), "match_id"]
    if len(future):
        out["played match dated in the future"] = _ex(future)
    return out


def check_statistics(features: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    f = features.loc[features["played"].astype(bool)]
    counts = [c for c in ("home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot",
                          "home_corners", "away_corners", "home_yellows", "away_yellows", "home_reds", "away_reds")
              if c in f]
    neg = f.loc[(f[counts] < 0).any(axis=1), "match_id"]
    if len(neg):
        out["negative statistic"] = _ex(neg)
    many = f.loc[(f["home_goals"] > 15) | (f["away_goals"] > 15), "match_id"]
    if len(many):
        out["more than 15 goals for a side"] = _ex(many)
    if "total_corners" in f:
        corners = f.loc[f["total_corners"] > 35, "match_id"]
        if len(corners):
            out["more than 35 corners"] = _ex(corners)
    for side in ("home", "away"):
        if {f"{side}_sot", f"{side}_shots"} <= set(f.columns):
            bad = f.loc[f[f"{side}_sot"] > f[f"{side}_shots"], "match_id"]
            if len(bad):
                out[f"{side} shots on target above shots"] = _ex(bad)
    if {"home_corners", "away_corners"} <= set(f.columns):
        one = f.loc[f["home_corners"].isna() != f["away_corners"].isna(), "match_id"]
        if len(one):
            out["corners of one side only"] = _ex(one)
    return out


def check_unused_results(features: pd.DataFrame, live: pd.DataFrame | None) -> dict[str, list]:
    """A finished match in the providers' live table that the match table does not show as played."""
    if live is None or live.empty:
        return {}
    fin = live.loc[live["played"].astype(bool) & live["home_goals"].notna()]
    known = features.set_index("match_id")["played"].astype(bool)
    missing = [m for m in fin["match_id"] if m in known.index and not known[m]]
    return {"provider result not used": _ex(missing)} if missing else {}


def check_prices(odds: pd.DataFrame | None) -> dict[str, list]:
    if odds is None or odds.empty:
        return {}
    out: dict[str, list] = {}
    bad = odds.loc[(odds["odds"] <= 1.0) | (odds["odds"] > 1000) | odds["odds"].isna()]
    if len(bad):
        out["impossible price"] = _ex(bad["match_id"] + " " + bad["market"].astype(str))
    if "kickoff" in odds:
        late = odds.loc[pd.to_datetime(odds["collected_at"]) >= pd.to_datetime(odds["kickoff"])]
        if len(late):
            out["price collected after kick-off (stored, not used)"] = [len(late)]
    return out


def check_predictions(picks: pd.DataFrame, features: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    if picks.empty:
        return out
    known = features.set_index("match_id")
    live = picks.loc[picks["source"] == "live"]
    if {"generated_at", "match_id"} <= set(live.columns) and "kickoff" in known:
        # generated_at is UTC, kick-off UK time: compare in UTC
        from ..prediction.price_timing import uk_to_utc
        ko = uk_to_utc(live["match_id"].map(known["kickoff"])).set_axis(live.index)
        late = live.loc[pd.to_datetime(live["generated_at"]) > ko, "pick_id"]
        if len(late):
            out["recorded prediction generated after kick-off"] = _ex(late)
    done = picks.loc[picks["status"].isin(["WON", "LOST", "PUSH"]) & picks["home_goals"].notna()]
    wrong = []
    for r in done.itertuples(index=False):
        corners = r.total_corners if r.total_corners == r.total_corners else None
        expected = settle_selection(r.market, r.selection, r.home_goals, r.away_goals, corners)
        if expected is not None and expected != r.status:
            wrong.append(r.pick_id)
    if wrong:
        out["status does not follow from the result"] = _ex(wrong)
    return out


def check_feature_time(features: pd.DataFrame, days: int = RECENT_DAYS) -> dict[str, list]:
    """``h_season_games`` / ``a_season_games`` count only earlier matches of the team in that league season."""
    need = {"h_season_games", "a_season_games"}
    if not need <= set(features.columns):
        return {}
    f = features.sort_values(["date", "kickoff" if "kickoff" in features else "date"])
    last = f["date"].max()
    recent = f.loc[f["date"] >= last - pd.Timedelta(days=days)]
    played = f.loc[f["played"].astype(bool)]
    bad = []
    for r in recent.itertuples(index=False):
        season = played.loc[(played["league"] == r.league) & (played["season"] == r.season)
                            & (played["date"] < r.date)]
        for side, team in (("h", r.home_team), ("a", r.away_team)):
            earlier = int(((season["home_team"] == team) | (season["away_team"] == team)).sum())
            value = getattr(r, f"{side}_season_games")
            if value == value and value > earlier:
                bad.append(r.match_id)
    return {"feature counts more matches than were played before": _ex(sorted(set(bad)))} if bad else {}


def check_all(features: pd.DataFrame, picks: pd.DataFrame | None = None, odds: pd.DataFrame | None = None,
              live: pd.DataFrame | None = None, today: pd.Timestamp | None = None) -> dict[str, list]:
    out: dict[str, list] = {}
    out.update(check_matches(features, today))
    out.update(check_statistics(features))
    out.update(check_unused_results(features, live))
    out.update(check_prices(odds))
    if picks is not None:
        out.update(check_predictions(picks, features))
    out.update(check_feature_time(features))
    return out


__all__ = ["check_all", "check_matches", "check_statistics", "check_unused_results", "check_prices",
           "check_predictions", "check_feature_time", "np"]
