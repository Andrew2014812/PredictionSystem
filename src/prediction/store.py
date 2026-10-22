"""Prediction history: MATCH -> many PREDICTIONS.

Files (Parquet, ``local_data/predictions/``):

* ``snapshots.parquet`` - one row per predicted match (probabilities, expected
  goals / corners, chosen main and risk selections, model version);
* ``markets.parquet``   - every priced selection of every predicted match;
* ``picks.parquet``     - the stored predictions that are settled and enter
  ROI statistics: the most likely selection of each history market
  (role="market") plus the main and risk predictions.

A recorded pre-match prediction is frozen once its match has kicked off:
re-running the pipeline only refreshes matches that have not started. Every
pre-match refresh is a new VERSION (``version`` 1, 2, …); all versions are
kept in ``versions.parquet`` (picks of each version with its ``generated_at``)
while the three main tables hold the latest version before kick-off — the one
Prediction History and ROI use. A postponed match gets a new version for its
new kick-off.
``generated_at`` is stored as naive UTC; ``kickoff`` as naive UK local time.
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
from ..markets.markets import PUSH, WON, settle_selection
from ..storage import predictions_storage
from .engine import PredictionBatch

SNAPSHOTS = "snapshots.parquet"
MARKETS = "markets.parquet"
PICKS = "picks.parquet"
VERSIONS = "versions.parquet"      # archive: every pre-match version of the stored predictions

STATUSES = ("UPCOMING", "WON", "LOST", "PUSH", "VOID")   # PUSH = Asian handicap refund


def load_snapshots() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(SNAPSHOTS)


def load_markets() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(MARKETS)


def load_picks() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(PICKS)


def load_versions() -> pd.DataFrame:
    return predictions_storage.read_parquet_or_empty(VERSIONS)


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


def _locked_ids(snapshots: pd.DataFrame, now: pd.Timestamp, batch: pd.DataFrame | None = None) -> set[str]:
    """Recorded pre-match predictions whose match has started: never rewritten.

    A match that was postponed (the batch carries a NEW kick-off that is still
    in the future) is not locked: it gets a new pre-match version, the old one
    stays in the version archive."""
    if snapshots.empty or "kickoff" not in snapshots:
        return set()
    started = pd.to_datetime(snapshots["kickoff"]) <= now
    locked = set(snapshots.loc[started & (snapshots["source"] == "live"), "match_id"])
    if batch is not None and not batch.empty and "kickoff" in batch:
        moved = batch.loc[pd.to_datetime(batch["kickoff"]) > now, ["match_id", "kickoff"]]
        old = snapshots.set_index("match_id")["kickoff"]
        rescheduled = {m for m, k in zip(moved["match_id"], pd.to_datetime(moved["kickoff"]))
                       if m in locked and pd.Timestamp(old.get(m)) != k}
        locked -= rescheduled
    return locked


def _archive_versions(batch: PredictionBatch, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Append the pre-match ("live") picks of the batch to the version archive and
    return the batch snapshots with their version numbers."""
    snaps = batch.snapshots.copy()
    previous = (snapshots.set_index("match_id")["version"] if "version" in snapshots and not snapshots.empty
                else pd.Series(dtype=float))
    known = set(snapshots["match_id"]) if not snapshots.empty else set()
    prev = snaps["match_id"].map(previous)
    prev = prev.where(prev.notna(), snaps["match_id"].isin(known).astype(float))   # pre-v5 rows = version 1
    snaps["version"] = (prev.fillna(0) + 1).astype(int)
    live = snaps.loc[snaps["source"] == "live", ["match_id", "version", "kickoff", "generated_at"]]
    if not live.empty and not batch.picks.empty:
        rows = batch.picks.merge(live, on="match_id", suffixes=("", "_snap"))
        if not rows.empty:
            rows = rows.drop(columns=[c for c in rows.columns if c.endswith("_snap")])
            frames = [f for f in (load_versions(), rows) if not f.empty]
            predictions_storage.write_parquet(VERSIONS, pd.concat(frames, ignore_index=True))
    return snaps


def save_batch(batch: PredictionBatch, replace_source: str | None = None,
               now: pd.Timestamp | None = None) -> None:
    """Store a batch. Matches in the batch replace their previous predictions,
    except recorded pre-match ("live") predictions of matches that have already
    kicked off — those are frozen. A refreshed pre-match prediction keeps the
    time of its first version in ``first_generated_at``.

    With ``replace_source`` every stored prediction of that source is
    dropped first (used to regenerate the whole backtest).
    """
    from .pipeline import uk_now
    now = now if now is not None else uk_now()
    snapshots, markets, picks = load_snapshots(), load_markets(), load_picks()
    locked = _locked_ids(snapshots, now, batch.snapshots)
    if locked and not batch.snapshots.empty:
        keep = ~batch.snapshots["match_id"].isin(locked)
        batch = PredictionBatch(batch.snapshots.loc[keep],
                                batch.markets.loc[~batch.markets["match_id"].isin(locked)] if not batch.markets.empty else batch.markets,
                                batch.picks.loc[~batch.picks["match_id"].isin(locked)] if not batch.picks.empty else batch.picks)
    if not batch.snapshots.empty:
        first = (snapshots.set_index("match_id")["first_generated_at"] if "first_generated_at" in snapshots
                 else snapshots.set_index("match_id")["generated_at"]) if not snapshots.empty else pd.Series(dtype="datetime64[ns]")
        new = batch.snapshots.copy()
        new["first_generated_at"] = pd.to_datetime(new["match_id"].map(first)).fillna(pd.to_datetime(new["generated_at"]))
        batch = PredictionBatch(new, batch.markets, batch.picks)
        batch = PredictionBatch(_archive_versions(batch, snapshots), batch.markets, batch.picks)
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
        outcome = settle_selection(pick["market"], pick["selection"], row["home_goals"], row["away_goals"],
                                   row["total_corners"])
        odds = pick["odds"] if pick["odds"] == pick["odds"] and pick["odds"] else None
        if outcome is None:
            # e.g. corners not (yet) reported: an API result carries goals only,
            # the statistics arrive with Football-Data. Void only when overdue.
            if overdue[idx]:
                status[idx], profit[idx] = "VOID", 0.0
            continue
        status[idx] = outcome
        # profit only with a real bookmaker price; otherwise the pick is settled
        # (hit / miss) but has no profit. A refund (PUSH) returns the stake.
        if outcome == PUSH:
            profit[idx] = 0.0 if odds else np.nan
        else:
            profit[idx] = (odds - 1 if outcome == WON else -1.0) if odds else np.nan

    decided = status != "UPCOMING"
    picks.loc[pending.index[has_result], score_cols] = scores.loc[has_result, score_cols].to_numpy()
    picks.loc[decided[decided].index, "status"] = status[decided]
    picks.loc[decided[decided].index, "profit"] = profit[decided]
    picks.loc[decided[decided].index, "settled_at"] = today

    predictions_storage.write_parquet(PICKS, picks)
    return picks
