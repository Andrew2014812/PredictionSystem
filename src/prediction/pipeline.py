"""Generating and settling predictions with the active model version."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ..modelling.registry import active_version, load_bundles, version_metadata
from ..providers import real_odds_lookup
from .engine import PredictionBatch, predict_matches
from .store import load_snapshots, save_batch, settle

log = logging.getLogger(__name__)


UK = ZoneInfo("Europe/London")


def uk_now() -> pd.Timestamp:
    """Current time on the clock kick-off times are written in (UK local time)."""
    return pd.Timestamp(datetime.now(timezone.utc).astimezone(UK).replace(tzinfo=None))


def select_live_targets(features: pd.DataFrame, snapshots: pd.DataFrame, trained_to: pd.Timestamp,
                        now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(upcoming, backfill) rows to predict.

    * upcoming — not played and kicking off AFTER ``now`` (UK time). Only these
      are (re)predicted, so a prediction is never created or changed once the
      match has started;
    * backfill — played after the model's training data ends and never
      predicted before (an after-the-fact model evaluation, not a recorded
      pre-match prediction).
    """
    known = set(snapshots["match_id"]) if not snapshots.empty else set()
    played = features["played"].astype(bool)
    kickoff = pd.to_datetime(features["kickoff"]).fillna(pd.to_datetime(features["date"]))
    upcoming = features.loc[~played & (kickoff > now)]
    backfill = features.loc[played & (features["date"] > trained_to) & ~features["match_id"].isin(known)]
    return upcoming, backfill


def generate_live_predictions(features: pd.DataFrame, now: pd.Timestamp | None = None) -> int:
    version = active_version()
    bundles = load_bundles(version)
    meta = version_metadata(version)
    if not bundles or meta is None:
        log.warning("no trained models - run scripts/retrain_models.py first")
        return 0
    now = now if now is not None else uk_now()
    trained_to = pd.Timestamp(meta["trained_to"])
    upcoming, backfill = select_live_targets(features, load_snapshots(), trained_to, now)
    odds = real_odds_lookup(pd.concat([upcoming, backfill]))
    total = 0
    for frame, source in ((backfill, "backfill"), (upcoming, "live")):
        batch: PredictionBatch = predict_matches(frame, bundles, odds, version, source)
        if not batch.snapshots.empty:
            save_batch(batch)
            total += len(batch.snapshots)
        log.info("%s predictions: %d matches", source, len(frame))
    return total


def settle_predictions(features: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    return settle(features, today)
