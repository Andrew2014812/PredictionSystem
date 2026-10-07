"""Prediction history: MATCH -> many PREDICTIONS.

Files (Parquet, ``local_data/predictions/``):

* ``snapshots.parquet`` - one row per predicted match (probabilities, expected
  goals / corners, chosen main and risk selections, model version);
* ``markets.parquet``   - every priced selection of every predicted match;
* ``picks.parquet``     - the stored predictions that are settled and enter
  ROI statistics: the most likely selection of each history market
  (role="market") plus the main and risk predictions.

A prediction is frozen once its match date has passed: re-running the
pipeline only refreshes predictions of matches that are still upcoming.
``source`` tells how a prediction was produced:

* ``live``     - generated before kick-off;
* ``backfill`` - a played match of the current season predicted afterwards by
  a model whose training data ends before that match (still out-of-sample:
  features use only pre-match information);
* ``backtest`` - the test season, predicted by the model trained on the
  train + validation seasons.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .. import config
from ..markets.markets import selection_won
from ..storage import predictions_storage
from .engine import PredictionBatch

SNAPSHOTS = "snapshots.parquet"
MARKETS = "markets.parquet"
PICKS = "picks.parquet"

STATUSES = ("UPCOMING", "WON", "LOST", "VOID")


def load_snapshots() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(SNAPSHOTS)


def load_markets() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(MARKETS)


def load_picks() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(PICKS)


def drop_source(source: str) -> int:
    """Remove every stored prediction of one origin (e.g. "backfill" before a retrain)."""
    snapshots = load_snapshots()
    if snapshots.empty:
        return 0
    ids = set(snapshots.loc[snapshots["source"] == source, "match_id"])
    for name, df in ((SNAPSHOTS, snapshots), (MARKETS, load_markets()), (PICKS, load_picks())):
        if not df.empty:
            predictions_storage.write_parquet(name, df.loc[~df["match_id"].isin(ids)])
    return len(ids)


def _replace(existing: pd.DataFrame, new: pd.DataFrame, drop_ids: set[str]) -> pd.DataFrame:
    if not existing.empty:
        existing = existing.loc[~existing["match_id"].isin(drop_ids)]
    frames = [f for f in (existing, new) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else new


def save_batch(batch: PredictionBatch, replace_source: str | None = None) -> None:
    """Store a batch. Matches in the batch replace their previous predictions.

    With ``replace_source`` every stored prediction of that source is
    dropped first (used to regenerate the whole backtest).
    """
    snapshots, markets, picks = load_snapshots(), load_markets(), load_picks()
    if replace_source and not snapshots.empty:
        old = set(snapshots.loc[snapshots["source"] == replace_source, "match_id"])
        snapshots = snapshots.loc[~snapshots["match_id"].isin(old)]
        markets = markets.loc[~markets["match_id"].isin(old)] if not markets.empty else markets
        picks = picks.loc[~picks["match_id"].isin(old)] if not picks.empty else picks
    if batch.snapshots.empty:
        ids: set[str] = set()
    else:
        ids = set(batch.snapshots["match_id"])
    predictions_storage.write_parquet(SNAPSHOTS, _replace(snapshots, batch.snapshots, ids))
    predictions_storage.write_parquet(MARKETS, _replace(markets, batch.markets, ids))
    predictions_storage.write_parquet(PICKS, _replace(picks, batch.picks, ids))


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
def _rescheduled_result(pick: pd.Series, by_pair: dict) -> pd.Series | None:
    """Played row of the same pairing up to 60 days after the predicted date."""
    candidates = by_pair.get((pick["league"], pick["home_team"], pick["away_team"]))
    if candidates is None:
        return None
    later = candidates.loc[(candidates["date"] >= pick["date"])
                           & (candidates["date"] <= pick["date"] + pd.Timedelta(days=60))]
    return later.iloc[0] if not later.empty else None


def settle(results: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Settle UPCOMING picks against played matches; returns all picks.

    Unit stake per pick with a real price: a win returns ``odds - 1``, a loss
    ``-1``, a void 0. Picks without a real price are settled (won / lost) but
    have no profit.
    A pick without a result VOID_AFTER_DAYS after its date is voided
    (postponed or abandoned match).
    """
    picks = load_picks()
    if picks.empty:
        return picks
    today = pd.Timestamp(today or date.today())
    played = results.loc[results["played"].astype(bool),
                         ["match_id", "league", "home_team", "away_team", "date",
                          "home_goals", "away_goals", "total_corners"]]
    by_id = played.set_index("match_id")
    by_pair = {k: g.sort_values("date") for k, g in played.groupby(["league", "home_team", "away_team"])}

    pending = picks.loc[picks["status"] == "UPCOMING"]
    score_cols = ["home_goals", "away_goals", "total_corners"]
    scores = by_id.reindex(pending["match_id"])[score_cols].set_axis(pending.index)
    for idx in scores.index[scores["home_goals"].isna()]:
        res = _rescheduled_result(pending.loc[idx], by_pair)
        if res is not None:
            scores.loc[idx, score_cols] = res[score_cols].to_numpy(dtype=float)

    status = pd.Series("UPCOMING", index=pending.index, dtype=object)
    profit = pd.Series(np.nan, index=pending.index)
    has_result = scores["home_goals"].notna()
    overdue = pending["date"] + pd.Timedelta(days=config.VOID_AFTER_DAYS) < today
    status[~has_result & overdue] = "VOID"
    profit[~has_result & overdue] = 0.0
    for idx in scores.index[has_result]:
        pick, row = pending.loc[idx], scores.loc[idx]
        won = selection_won(pick["market"], pick["selection"], row["home_goals"], row["away_goals"],
                            row["total_corners"])
        odds = pick["odds"] if pick["odds"] == pick["odds"] and pick["odds"] else None
        if won is None:
            # e.g. corners not (yet) reported: an API result carries goals only,
            # the statistics arrive with Football-Data. Void only when overdue.
            if overdue[idx]:
                status[idx], profit[idx] = "VOID", 0.0
            continue
        status[idx] = "WON" if won else "LOST"
        # profit only with a real bookmaker price; otherwise the pick is settled
        # (hit / miss) but has no profit
        profit[idx] = (odds - 1 if won else -1.0) if odds else np.nan

    decided = status != "UPCOMING"
    picks.loc[pending.index[has_result], score_cols] = scores.loc[has_result, score_cols].to_numpy()
    picks.loc[decided[decided].index, "status"] = status[decided]
    picks.loc[decided[decided].index, "profit"] = profit[decided]
    picks.loc[decided[decided].index, "settled_at"] = today

    predictions_storage.write_parquet(PICKS, picks)
    return picks
