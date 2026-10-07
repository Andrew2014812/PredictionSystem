"""Prediction engine: model outputs -> distributions -> markets -> real odds
-> main / risk recommendation, for a batch of matches."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config
from ..markets.distributions import corners_distribution, reconcile_with_1x2, score_matrix
from ..markets.markets import corner_markets, expected_corners, goal_markets
from ..markets.odds import PricedSelection, price
from ..markets.recommendation import main_prediction, risk_prediction
from ..modelling.decision import decide
from ..modelling.tasks import RESULT_CLASSES
from ..modelling.training import ModelBundle

SNAPSHOT_ID_COLUMNS = ["match_id", "league", "country", "season", "date", "time", "kickoff",
                       "home_team", "away_team"]


@dataclass
class PredictionBatch:
    snapshots: pd.DataFrame     # one row per match: probabilities & expected values
    markets: pd.DataFrame       # one row per selection (probability + real odds if any)
    picks: pd.DataFrame         # stored predictions (market forecasts, main, risk)


def _pick_row(match: dict, sel: PricedSelection, role: str, meta: dict, kind: str = "") -> dict:
    return {
        "pick_id": f"{match['match_id']}|{role}|{sel.market}|{sel.selection}",
        "match_id": match["match_id"], "league": match["league"], "season": match["season"],
        "date": match["date"], "time": match["time"], "home_team": match["home_team"],
        "away_team": match["away_team"], "role": role, "kind": kind,
        "market": sel.market, "group": sel.group, "selection": sel.selection,
        "probability": sel.probability, "odds": sel.odds if sel.has_odds else np.nan,
        "bookmaker": sel.bookmaker, "provider": sel.provider, "real_odds": sel.has_odds,
        "ev": sel.ev if sel.has_odds else np.nan,
        "status": "UPCOMING", "home_goals": np.nan, "away_goals": np.nan,
        "total_corners": np.nan, "profit": np.nan, "settled_at": pd.NaT, **meta,
    }


def _market_forecasts(by_market: dict[str, list[PricedSelection]], outcome_1x2: str) -> list[PricedSelection]:
    """The single forecast of each history market for one match."""
    out = []
    for market in config.HISTORY_MARKETS:
        sels = by_market.get(market)
        if not sels:
            continue
        if market == "1X2":                         # draw-aware decision rule
            out.append(next(s for s in sels if s.selection == outcome_1x2))
        else:
            out.append(max(sels, key=lambda s: s.probability))
    # handicap: the most probable priced half / whole line selection in the main band
    hcp = [s for m, sels in by_market.items() if m.startswith("HANDICAP_") for s in sels
           if s.has_odds and config.MAIN_MIN_PROBABILITY <= s.probability <= config.MAIN_MAX_PROBABILITY]
    if hcp:
        out.append(max(hcp, key=lambda s: s.probability))
    return out


def predict_matches(matches: pd.DataFrame, bundles: dict[str, ModelBundle], odds: dict,
                    version: str, source: str) -> PredictionBatch:
    """Predict every market for ``matches`` (rows of the feature table).

    ``odds`` = match_id -> {(market, selection): real price} (providers.odds_lookup).
    """
    if matches.empty or "result" not in bundles:
        empty = pd.DataFrame()
        return PredictionBatch(empty, empty, empty)

    generated_at = pd.Timestamp(datetime.now()).floor("s")
    result = bundles["result"]
    p1x2 = result.predict(matches)
    outcomes = decide(p1x2, result.extra.get("draw_multiplier", 1.0))
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

        priced = price(selections, odds.get(match["match_id"]))
        main, risk = main_prediction(priced), risk_prediction(priced)
        outcome = RESULT_CLASSES[int(outcomes[i])]

        meta = {"model_version": version, "source": source, "generated_at": generated_at}
        snapshots.append({
            **{c: match[c] for c in SNAPSHOT_ID_COLUMNS},
            "p_home": p1x2[i][0], "p_draw": p1x2[i][1], "p_away": p1x2[i][2],
            "predicted_outcome": outcome,
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
        for sel in _market_forecasts(by_market, outcome):
            picks.append(_pick_row(match, sel, "market", meta))
        if main:
            picks.append(_pick_row(match, main.pick, "main", meta, main.kind))
        if risk:
            picks.append(_pick_row(match, risk.pick, "risk", meta, risk.kind))

    markets_df = pd.DataFrame(market_rows)
    for col in ("probability", "odds", "ev"):
        markets_df[col] = pd.to_numeric(markets_df[col], errors="coerce").astype("float32")
    return PredictionBatch(pd.DataFrame(snapshots), markets_df, pd.DataFrame(picks))
