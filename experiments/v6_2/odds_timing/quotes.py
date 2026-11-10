"""The OddsPapi quote history with strictly known validity intervals (all times in UTC).

Every OddsPapi row is one bookmaker price of one selection with
    priced_at   when the bookmaker set the price (``changedAt`` / ``createdAt``);
    fetched_at  when FootPredict downloaded it;
    kickoff.
Two kinds of rows (src/providers/collect.py, scripts/oddspapi_history.py):

* **live fetch** (``fetched_at`` < kick-off, /odds): the price was current at ``fetched_at`` and had not changed
  since ``priced_at`` -> known valid on [priced_at, fetched_at];
* **archive** (``fetched_at`` >= kick-off, /historical-odds): the LAST snapshot strictly before kick-off, i.e. the
  closing price -> known valid on [priced_at, kick-off).

``price_at(t)`` returns a price only when t lies inside such a known interval, so a quote published after t is
never used for t, and a price that might have changed between two observations is not assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

KEY = ["match_id", "bookmaker", "market", "selection"]


def load(columns=None, markets=None) -> pd.DataFrame:
    cols = ["match_id", "provider", "bookmaker", "market", "line", "selection", "odds", "collected_at",
            "priced_at", "fetched_at", "kickoff"]
    df = pd.read_parquet(C.ODDS_FILE, columns=columns or cols)
    df = df.loc[df["provider"] == "oddspapi"]
    if markets is not None:
        df = df.loc[df["market"].isin(markets) | df["market"].str.split("_").str[0].isin(markets)]
    df = df.copy()
    for col in ("priced_at", "fetched_at", "kickoff"):
        df[col + "_utc"] = C.uk_to_utc(df[col]).to_numpy()
    df["archive"] = df["fetched_at"] >= df["kickoff"]
    df["valid_from"] = df["priced_at_utc"]
    df["valid_to"] = np.where(df["archive"], df["kickoff_utc"], df["fetched_at_utc"])
    df["valid_to"] = pd.to_datetime(df["valid_to"])
    df["lead_h"] = (df["kickoff_utc"] - df["priced_at_utc"]).dt.total_seconds() / 3600
    return df.reset_index(drop=True)


def closing(q: pd.DataFrame) -> pd.DataFrame:
    """The closing price of every key that has an archive row (the last snapshot before kick-off)."""
    a = q.loc[q["archive"]].sort_values("priced_at_utc")
    return a.drop_duplicates(KEY, keep="last")


def price_at(q: pd.DataFrame, when: pd.DataFrame, key=KEY) -> pd.DataFrame:
    """For every row of ``when`` (key columns + ``t`` in UTC) the price known to be valid at t (or NaN).

    When several observations cover t (overlapping intervals of the same price), the latest-published wins."""
    w = when[key + ["t"]].reset_index().rename(columns={"index": "_row"})
    m = w.merge(q[key + ["odds", "valid_from", "valid_to", "priced_at_utc", "archive"]], on=key, how="inner")
    m = m.loc[(m["valid_from"] <= m["t"]) & (m["t"] <= m["valid_to"])]
    m = m.sort_values("priced_at_utc").drop_duplicates("_row", keep="last")
    out = pd.DataFrame(index=when.index, columns=["odds_at", "priced_at_utc_at", "archive_at"])
    out.loc[m["_row"].to_numpy(), "odds_at"] = m["odds"].to_numpy()
    out.loc[m["_row"].to_numpy(), "priced_at_utc_at"] = m["priced_at_utc"].to_numpy()
    out.loc[m["_row"].to_numpy(), "archive_at"] = m["archive"].to_numpy()
    out["odds_at"] = pd.to_numeric(out["odds_at"])
    out["priced_at_utc_at"] = pd.to_datetime(out["priced_at_utc_at"])
    return out
