"""Prediction engine: model outputs -> distributions -> markets -> odds ->
main / risk recommendation, for a batch of matches."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config
from ..markets.distributions import (corners_distribution, empirical_corners_distribution,
                                     empirical_score_matrix, reconcile_with_1x2, score_matrix)
from ..markets.markets import corner_markets, expected_corners, goal_markets
from ..markets.odds import PricedSelection, fit_market_score_matrix, price
from ..markets.recommendation import main_prediction, risk_prediction
from ..modelling.training import ModelBundle

SNAPSHOT_ID_COLUMNS = ["match_id", "league", "country", "season", "date", "time", "kickoff",
                       "home_team", "away_team"]
ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25"]


class LeagueBaselines:
    """Recent empirical score / corner distributions of each league.

    Uses only played matches strictly before the requested date; they price
    the simulated odds of the "naive bookmaker".
    """

    def __init__(self, history: pd.DataFrame, window: int = config.LEAGUE_PRIOR_WINDOW):
        played = history.loc[history["played"].astype(bool)].sort_values("date")
        self.window = window
        self.data = {
            league: (g["date"].to_numpy(), g["home_goals"].to_numpy(float),
                     g["away_goals"].to_numpy(float), g["total_corners"].to_numpy(float))
            for league, g in played.groupby("league")
        }
        self._cache: dict = {}

    def at(self, league: str, day) -> tuple[np.ndarray | None, np.ndarray | None]:
        key = (league, day)
        if key in self._cache:
            return self._cache[key]
        if league not in self.data:
            return None, None
        dates, hg, ag, corners = self.data[league]
        end = int(np.searchsorted(dates, np.datetime64(day), side="left"))
        start = max(0, end - self.window)
        if end - start < 30:
            result = (None, None)
        else:
            matrix = empirical_score_matrix(hg[start:end], ag[start:end])
            c = corners[start:end]
            c = c[~np.isnan(c)]
            corners_pmf = empirical_corners_distribution(c) if len(c) >= 30 else None
            result = (matrix, corners_pmf)
        self._cache[key] = result
        return result


@dataclass
class PredictionBatch:
    snapshots: pd.DataFrame     # one row per match: probabilities & expected values
    markets: pd.DataFrame       # one row per priced selection
    picks: pd.DataFrame         # stored predictions (market picks, main, risk)


def _pick_row(match: dict, sel: PricedSelection, role: str, meta: dict, kind: str = "") -> dict:
    return {
        "pick_id": f"{match['match_id']}|{role}|{sel.market}|{sel.selection}",
        "match_id": match["match_id"], "league": match["league"], "season": match["season"],
        "date": match["date"], "time": match["time"], "home_team": match["home_team"],
        "away_team": match["away_team"], "role": role, "kind": kind,
        "market": sel.market, "group": sel.group, "selection": sel.selection,
        "probability": sel.probability, "odds": sel.odds, "odds_source": sel.odds_source,
        "fair_odds": sel.fair_odds, "ev": sel.ev,
        "status": "UPCOMING", "home_goals": np.nan, "away_goals": np.nan,
        "total_corners": np.nan, "profit": np.nan, "settled_at": pd.NaT, **meta,
    }


def predict_matches(matches: pd.DataFrame, bundles: dict[str, ModelBundle], baselines: LeagueBaselines,
                    version: str, source: str) -> PredictionBatch:
    """Predict every market for ``matches`` (rows of the feature table)."""
    if matches.empty or "result" not in bundles:
        empty = pd.DataFrame()
        return PredictionBatch(empty, empty, empty)

    generated_at = pd.Timestamp(datetime.now()).floor("s")
    p1x2 = bundles["result"].predict(matches)
    lam_home = bundles["home_goals"].predict(matches)
    lam_away = bundles["away_goals"].predict(matches)
    corners_bundle = bundles.get("corners")
    corner_mu = corners_bundle.predict(matches) if corners_bundle else np.full(len(matches), np.nan)
    corner_leagues = set(corners_bundle.extra.get("leagues", [])) if corners_bundle else set()
    nb_size = corners_bundle.extra.get("nb_size") if corners_bundle else None

    snapshots, market_rows, picks = [], [], []
    for i, match in enumerate(matches.to_dict("records")):
        matrix = reconcile_with_1x2(score_matrix(lam_home[i], lam_away[i]), p1x2[i])
        selections = goal_markets(matrix)
        corners_ok = match["league"] in corner_leagues and not np.isnan(corner_mu[i])
        corners_pmf = corners_distribution(corner_mu[i], nb_size) if corners_ok else None
        if corners_pmf is not None:
            selections += corner_markets(corners_pmf)

        base_matrix, base_corners = baselines.at(match["league"], match["date"])
        base_goal = ({(s.market, s.selection): s.probability for s in goal_markets(base_matrix, 0)}
                     if base_matrix is not None else {})
        base_corner = ({(s.market, s.selection): s.probability for s in corner_markets(base_corners)}
                       if base_corners is not None else {})
        priced = price(selections, match, fit_market_score_matrix(match), base_goal, base_matrix, base_corner)
        main, risk = main_prediction(priced), risk_prediction(priced)

        meta = {"model_version": version, "source": source, "generated_at": generated_at}
        snapshots.append({
            **{c: match[c] for c in SNAPSHOT_ID_COLUMNS + ODDS_COLUMNS},
            "p_home": p1x2[i][0], "p_draw": p1x2[i][1], "p_away": p1x2[i][2],
            "xg_home": float(lam_home[i]), "xg_away": float(lam_away[i]),
            "corners_available": bool(corners_ok),
            "exp_corners": expected_corners(corners_pmf) if corners_pmf is not None else np.nan,
            "corners_nb_size": nb_size if corners_ok else np.nan,
            "main_market": main.pick.market if main else None,
            "main_selection": main.pick.selection if main else None,
            "main_kind": main.kind if main else None,
            "risk_market": risk.pick.market if risk else None,
            "risk_selection": risk.pick.selection if risk else None,
            **meta,
        })
        market_rows += [{"match_id": match["match_id"], **s.to_dict()} for s in priced]

        by_market: dict[str, list[PricedSelection]] = {}
        for s in priced:
            by_market.setdefault(s.market, []).append(s)
        for market in config.HISTORY_MARKETS:
            if market in by_market:
                best = max(by_market[market], key=lambda s: s.probability)
                picks.append(_pick_row(match, best, "market", meta))
        if main:
            picks.append(_pick_row(match, main.pick, "main", meta, main.kind))
        if risk:
            picks.append(_pick_row(match, risk.pick, "risk", meta, risk.kind))

    markets_df = pd.DataFrame(market_rows)
    for col in ("probability", "odds", "fair_odds", "implied_probability", "ev"):
        markets_df[col] = markets_df[col].astype("float32")
    return PredictionBatch(pd.DataFrame(snapshots), markets_df, pd.DataFrame(picks))
