"""Building and loading the processed feature table."""
from __future__ import annotations

import logging

import pandas as pd

from ..data.dataset import build_matches
from ..storage import processed_storage
from .builder import build_features

log = logging.getLogger(__name__)

FEATURES_FILE = "features.parquet"


def rebuild_features() -> pd.DataFrame:
    features = build_features(build_matches())
    processed_storage.write_parquet(FEATURES_FILE, features)
    log.info("saved %s", processed_storage.path(FEATURES_FILE))
    return features


def load_features() -> pd.DataFrame:
    if not processed_storage.exists(FEATURES_FILE):
        return rebuild_features()
    return processed_storage.read_parquet(FEATURES_FILE)
