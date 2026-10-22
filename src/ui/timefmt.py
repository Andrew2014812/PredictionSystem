"""Times in the visitor's time zone.

Kick-off times are stored as UK local time (the clock Football-Data and the providers use; naive
timestamps), quote times also as UK time, generation times as UTC. They are converted only for display,
once, into the browser's IANA time zone (``st.context.timezone``), with Europe/Kyiv as the fallback.
The calendar date is converted too: a 23:00 kick-off in London is 01:00 the next day in Kyiv.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

SOURCE_TZ = "Europe/London"
FALLBACK_TZ = "Europe/Kyiv"


def user_tz_name() -> str:
    try:
        name = st.context.timezone
    except Exception:
        name = None
    if name:
        try:
            ZoneInfo(name)
            return name
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return FALLBACK_TZ


def user_tz() -> ZoneInfo:
    return ZoneInfo(user_tz_name())


def today_local() -> date:
    return datetime.now(user_tz()).date()


def _localize(values: pd.Series, source: str) -> pd.Series:
    ts = pd.to_datetime(values)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(source, ambiguous="NaT", nonexistent="shift_forward")
    return ts.dt.tz_convert(user_tz_name())


def local_kickoff(df: pd.DataFrame) -> pd.Series:
    """Kick-off of each row in the visitor's zone (NaT when the time is unknown)."""
    if "kickoff" in df:
        kick = pd.to_datetime(df["kickoff"])
    elif "time" in df:
        kick = pd.to_datetime(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") + " " + df["time"].fillna("").astype(str),
                              errors="coerce")
    else:
        kick = pd.Series(pd.NaT, index=df.index)
    has_time = df["time"].notna() & (df["time"].astype(str).str.len() >= 4) if "time" in df else kick.notna()
    kick = kick.where(has_time)
    return _localize(kick, SOURCE_TZ)


def with_local_time(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ``local_date`` and ``local_time`` (strings HH:MM) next to the stored UK date / time."""
    out = df.copy()
    local = local_kickoff(out)
    out["local_date"] = local.dt.tz_localize(None).dt.normalize().fillna(pd.to_datetime(out["date"]).dt.normalize())
    out["local_time"] = local.dt.strftime("%H:%M").where(local.notna(), out.get("time"))
    return out


def local_stamp(value, source: str = SOURCE_TZ, fmt: str = "%d %b, %H:%M") -> str:
    """One timestamp (stored as ``source`` local time when naive) formatted in the visitor's zone."""
    if value is None or (isinstance(value, float) and value != value) or value is pd.NaT:
        return ""
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize(source, ambiguous="NaT", nonexistent="shift_forward")
    return ts.tz_convert(user_tz_name()).strftime(fmt)
