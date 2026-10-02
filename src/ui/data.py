"""Cached, read-only data access for the web interface.

Large tables are cached as resources (shared, not copied) and keyed by the
file modification time, so a pipeline run is picked up on the next rerun.
Callers must not mutate the returned frames.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from ..features.store import FEATURES_FILE
from ..modelling import registry
from ..prediction import store
from ..storage import models_storage, predictions_storage, processed_storage


def _mtime(path) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading match data…")
def _features(_mtime_key: float) -> pd.DataFrame:
    if not processed_storage.exists(FEATURES_FILE):
        return pd.DataFrame()
    df = processed_storage.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def features() -> pd.DataFrame:
    return _features(_mtime(processed_storage.path(FEATURES_FILE)))


@st.cache_resource(show_spinner=False)
def _prediction_table(name: str, _mtime_key: float) -> pd.DataFrame:
    df = predictions_storage.read_parquet_or_empty(name)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def snapshots() -> pd.DataFrame:
    return _prediction_table(store.SNAPSHOTS, _mtime(predictions_storage.path(store.SNAPSHOTS)))


def markets() -> pd.DataFrame:
    return _prediction_table(store.MARKETS, _mtime(predictions_storage.path(store.MARKETS)))


def picks() -> pd.DataFrame:
    return _prediction_table(store.PICKS, _mtime(predictions_storage.path(store.PICKS)))


@st.cache_resource(show_spinner=False)
def _markets_by_match(_mtime_key: float) -> dict[str, pd.DataFrame]:
    m = markets()
    return {mid: g for mid, g in m.groupby("match_id")} if not m.empty else {}


def match_markets(match_id: str) -> pd.DataFrame:
    by_match = _markets_by_match(_mtime(predictions_storage.path(store.MARKETS)))
    return by_match.get(match_id, pd.DataFrame())


@st.cache_resource(show_spinner=False)
def _card_markets(_mtime_key: float) -> pd.DataFrame:
    """1X2 / Over 2.5 / BTTS Yes probability, odds and source per match (for match cards)."""
    m = markets()
    if m.empty:
        return pd.DataFrame()
    wanted = {("1X2", "H"): "h", ("1X2", "D"): "d", ("1X2", "A"): "a",
              ("TOTAL_2.5", "OVER"): "o25", ("TOTAL_2.5", "UNDER"): "u25",
              ("BTTS", "YES"): "btts", ("BTTS", "NO"): "nobtts"}
    keys = pd.Series(list(zip(m["market"], m["selection"]))).map(wanted)
    sub = m.assign(key=keys.to_numpy()).dropna(subset=["key"])
    wide = sub.pivot_table(index="match_id", columns="key",
                           values=["probability", "odds"], aggfunc="first")
    wide.columns = [f"{k}_{'p' if v == 'probability' else 'odds'}" for v, k in wide.columns]
    src = sub.pivot_table(index="match_id", columns="key", values="odds_source", aggfunc="first")
    src.columns = [f"{c}_src" for c in src.columns]
    return wide.join(src)


def card_markets() -> pd.DataFrame:
    return _card_markets(_mtime(predictions_storage.path(store.MARKETS)))


@st.cache_resource(show_spinner=False)
def bundles(version: str | None):
    try:
        return registry.load_bundles(version)
    except Exception:          # incompatible / missing pickle
        return {}


@st.cache_data(show_spinner=False)
def _metadata(version: str | None, _mtime_key: float) -> dict | None:
    return registry.version_metadata(version)


def metadata(version: str | None = None) -> dict | None:
    version = version or registry.active_version()
    return _metadata(version, _mtime(models_storage.path(f"{version}/metadata.json")))


def model_versions() -> dict:
    return registry.list_versions()


def active_version() -> str | None:
    return registry.active_version()


def match_dates() -> np.ndarray:
    f = features()
    return np.sort(f["date"].dt.normalize().unique()) if not f.empty else np.array([])


def match_row(match_id: str) -> pd.Series | None:
    f = features()
    rows = f.loc[f["match_id"] == match_id]
    if not rows.empty:
        return rows.iloc[0]
    s = snapshots()
    rows = s.loc[s["match_id"] == match_id] if not s.empty else s
    return rows.iloc[0] if not rows.empty else None


def snapshot(match_id: str) -> pd.Series | None:
    s = snapshots()
    if s.empty:
        return None
    rows = s.loc[s["match_id"] == match_id]
    return rows.iloc[0] if not rows.empty else None
