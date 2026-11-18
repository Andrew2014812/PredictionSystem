"""Team identification: an audit of the current names and a controlled misidentification on 2026/27.

Audit. (1) Clubs of 2026/27 the system has never seen ("outside") whose normalised name shares a distinctive word
(≥ 5 letters) with, or is very similar to (difflib ratio ≥ 0.8), a club of the same country already known — a
possible alias / misspelling; (2) clubs with far fewer matches than their league rivals (a phantom name lives a
day or two); (3) the three phantom match ids found by v6.2 must be absent; (4) share of matches identified by
the OddsPapi feed at closing (a failed name match costs the real price of every market but 1X2 / 2.5 / AH).

Controlled misidentification (new in v6.3; v6.2 measured it on Football-Data prices without any mitigation).
Every played 2026/27 match gets one misidentified club (home in even rows, away in odd rows): its features are
blanked exactly as the feature builder leaves a club it has never seen (the v6.2 ``newcomer`` blanking: season /
form / venue / trend features missing, counters 0, Elo = base of the tier, no head-to-head, derived features
recomputed with the builder's own formulas). The production models predict again; Main runs in both universes.
Measured: damage vs Baseline B and how much of it the low-data mitigations recover.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.alt_ml import mainroi as R
from experiments.v6_3.replay import engine as EN
from src.features.builder import FeatureBuilder
from src.leagues import get_league
from src.providers import aliases

from . import data as Q

DERIVED = ["diff_season_ppg", "diff_position", "diff_points", "diff_season_gd", "diff_venue_ppg", "diff_form5_ppg",
           "diff_form10_ppg", "diff_form10_gd", "diff_form10_shots", "diff_form10_sot", "diff_form10_corners",
           "diff_rest_days", "h_attack_vs_a_defence", "a_attack_vs_h_defence", "h_form_attack_vs_a_form_defence",
           "a_form_attack_vs_h_form_defence", "h_sot_vs_a_sot_conceded", "a_sot_vs_h_sot_conceded",
           "h_corners_vs_a_corners_conceded", "a_corners_vs_h_corners_conceded", "form_corners_sum"]
V62_PHANTOMS = ["B1-20261104-kortrijk-kv-mechelen", "E1-20261103-preston-sheffield-utd", "E2-20261104-oxford-utd-reading"]


def _cols(df, pattern):
    rx = re.compile(pattern)
    return [c for c in df.columns if rx.search(c)]


def blank_newcomer(x: pd.DataFrame, side: str) -> pd.DataFrame:
    """The features of ``side`` (h / a) as for a club the system has never seen (v6.2 sensitivity.blank)."""
    x = x.copy()
    s = side
    x[_cols(x, rf"^{s}_(season_|points$|position|venue_(?!form)|trend_)")] = np.nan
    x[f"{s}_season_games"] = 0.0
    x[f"{s}_venue_games"] = 0.0
    x[_cols(x, rf"^{s}_(form\d+_|venue_form5_|freq_|trend_)")] = np.nan
    for k in (5, 10, 15):
        x[f"{s}_form{k}_n"] = 0.0
    x[f"{s}_elo"] = x["league"].map(lambda lg: 1500.0 - 100.0 * (get_league(lg).tier - 1))
    x[f"{s}_rest_days"] = np.nan
    x[_cols(x, r"^h2h_")] = np.nan
    x["h2h_n"] = 0.0
    x["elo_diff"] = x["h_elo"] + 60.0 - x["a_elo"]
    x["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-(x["elo_diff"]) / 400.0))
    src = [c for c in x.columns if c.startswith(("h_", "a_"))]
    rows = x[src].to_dict("records")
    derived = pd.DataFrame([FeatureBuilder._derived_features({k: v for k, v in r.items() if v == v}) for r in rows],
                           index=x.index)
    x[DERIVED] = derived[DERIVED]
    return x


def perturbed_matches() -> tuple[pd.DataFrame, pd.Series]:
    m = R.matches()
    played = m.loc[m["played"].astype(bool)].reset_index(drop=True)
    side = pd.Series(np.where(np.arange(len(played)) % 2 == 0, "h", "a"), index=played["match_id"])
    parts = [blank_newcomer(played.loc[(side == s).to_numpy()], s) for s in ("h", "a")]
    return pd.concat(parts).sort_index(), side


def name_audit() -> dict:
    f = Q.features()
    ts = Q.team_seasons(f.loc[f["season"] >= 2021])
    cur = ts.loc[ts["season"] == C.CURRENT]
    known = ts.loc[ts["season"] < C.CURRENT]
    rows = []
    for r in cur.loc[cur["status"] == "outside"].itertuples():
        words = {w for w in aliases.normalise(r.team).split() if len(w) >= 5}
        for k in known.loc[known["country"] == r.country, "team"].unique():
            if k == r.team:
                continue
            nk = aliases.normalise(k)
            ratio = difflib.SequenceMatcher(None, aliases.normalise(r.team), nk).ratio()
            shared = words & set(nk.split())
            if ratio >= 0.8 or shared:
                rows.append({"league": r.league, "new_name": r.team, "known_name": k, "ratio": round(ratio, 3),
                             "shared": " ".join(sorted(shared))})
    sus = pd.DataFrame(rows)
    games = cur.groupby("league")["games"].median().rename("league_median")
    low = cur.join(games, on="league")
    low = low.loc[low["games"] < 0.5 * low["league_median"]]
    m = R.matches()
    closing = R.prices("close")
    op = set(closing.loc[closing["provider"] == "oddspapi", "match_id"])
    played = m.loc[m["played"].astype(bool)]
    cov = played.assign(op=played["match_id"].isin(op)).groupby("league")["op"].mean()
    return {"clubs_2026": int(len(cur)), "outside_clubs_2026": int((cur["status"] == "outside").sum()),
            "similar_name_pairs": sus.to_dict("records"), "clubs_with_few_matches": low[["league", "team", "games",
                                                                                         "league_median"]].to_dict("records"),
            "v62_phantoms_present": int(m["match_id"].isin(V62_PHANTOMS).sum()),
            "oddspapi_closing_share": float(played["match_id"].isin(op).mean()),
            "oddspapi_closing_share_by_league": cov.round(4).to_dict()}


def perturbed_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    x, side = perturbed_matches()
    out = EN.outputs_from_bundles(EN.production_bundles(), x)
    return out, x, side
