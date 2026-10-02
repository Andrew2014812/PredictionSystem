"""Generating and settling predictions with the active model version."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from ..modelling.registry import active_version, load_bundles, version_metadata
from .engine import LeagueBaselines, PredictionBatch, predict_matches
from .store import load_snapshots, save_batch, settle

log = logging.getLogger(__name__)


def select_live_targets(features: pd.DataFrame, snapshots: pd.DataFrame, trained_to: pd.Timestamp,
                        today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(upcoming, backfill) rows to predict.

    * upcoming - not played and scheduled today or later (refreshed each run);
    * backfill - played after the model's training data ends and never
      predicted before.
    """
    known = set(snapshots["match_id"]) if not snapshots.empty else set()
    played = features["played"].astype(bool)
    upcoming = features.loc[~played & (features["date"] >= today)]
    backfill = features.loc[played & (features["date"] > trained_to) & ~features["match_id"].isin(known)]
    return upcoming, backfill


def generate_live_predictions(features: pd.DataFrame, today: date | None = None) -> int:
    version = active_version()
    bundles = load_bundles(version)
    meta = version_metadata(version)
    if not bundles or meta is None:
        log.warning("no trained models - run scripts/retrain_models.py first")
        return 0
    today = pd.Timestamp(today or date.today())
    trained_to = pd.Timestamp(meta["trained_to"])
    upcoming, backfill = select_live_targets(features, load_snapshots(), trained_to, today)
    baselines = LeagueBaselines(features)
    total = 0
    for frame, source in ((backfill, "backfill"), (upcoming, "live")):
        batch: PredictionBatch = predict_matches(frame, bundles, baselines, version, source)
        if not batch.snapshots.empty:
            save_batch(batch)
            total += len(batch.snapshots)
        log.info("%s predictions: %d matches", source, len(frame))
    return total


def settle_predictions(features: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    return settle(features, today)
