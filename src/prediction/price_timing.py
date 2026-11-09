"""When a stored price became available, compared with the prediction it is attached to.

A price shown with a prediction is not always one that could have been taken when the prediction was made.
Recorded (live) predictions get the price available at that moment; later runs may complete a missing
price from the OddsPapi archive — the last quote before kick-off, which can have been published after the
prediction. Retrospective (backfill) predictions are made after the match, so their price is the closing
one. Every Main / Risk price is labelled (``odds_timing``):

* ``at_generation`` — recorded with the prediction by the pipeline (the price on offer then);
* ``existed_at_generation`` — added later, but the quote had been published before the prediction;
* ``after_generation`` — added later and published after the prediction (a closing-type price);
* ``closing_backfill`` — a retrospective prediction's closing price;
* ``time_unknown`` — added later and the quote time cannot be found.

Times: ``generated_at`` is stored in UTC, quote times (``odds_priced_at``, ``collected_at``) and kick-offs in
UK time — they are compared in UTC. Football-Data publishes prices without a time (the file stores the match
date at 00:00), so its quote time is never shown and never compared.

Nothing here changes a prediction, its price or its result: the label only says which prices could have
been taken at prediction time (``decision_time_prices``), for the separate ROI shown on History.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AT_GENERATION = "at_generation"
EXISTED = "existed_at_generation"
AFTER = "after_generation"
CLOSING = "closing_backfill"
UNKNOWN = "time_unknown"
DECISION_TIME = (AT_GENERATION, EXISTED)
UNTIMED_PROVIDERS = {"football_data"}           # prices from a file without a quote time
ADDED_LATER = {"archive", "average_replaced"}
LABELS = {
    AT_GENERATION: "price at the time of the prediction",
    EXISTED: "on offer at prediction time, added later from the price archive",
    AFTER: "closing price published after the prediction (added later)",
    CLOSING: "closing price (prediction made after the match)",
    UNKNOWN: "added later from the price archive",
}


def uk_to_utc(values) -> pd.Series:
    """Naive UK-local timestamps → naive UTC (DST-aware)."""
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    local = s.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC").dt.tz_localize(None)


def recover_quote_times(picks: pd.DataFrame, odds: pd.DataFrame) -> pd.Series:
    """``odds_priced_at`` where it is missing for a price added later: the time of the stored quote with the
    same match, market, selection, bookmaker and price (the latest one before kick-off)."""
    times = pd.to_datetime(picks.get("odds_priced_at", pd.Series(pd.NaT, index=picks.index)), errors="coerce")
    need = times.isna() & picks["odds"].notna() & picks.get("odds_origin", pd.Series(index=picks.index)).isin(ADDED_LATER)
    if not need.any() or odds is None or odds.empty:
        return times
    q = odds.loc[odds["match_id"].isin(picks.loc[need, "match_id"]),
                 ["match_id", "market", "selection", "bookmaker", "odds", "collected_at", "kickoff"]].copy()
    q = q.loc[pd.to_datetime(q["kickoff"], errors="coerce").isna()
              | (pd.to_datetime(q["collected_at"]) < pd.to_datetime(q["kickoff"], errors="coerce"))]
    q["key_odds"] = q["odds"].round(3)
    last = (q.sort_values("collected_at").drop_duplicates(["match_id", "market", "selection", "bookmaker", "key_odds"],
                                                         keep="last")
            .set_index(["match_id", "market", "selection", "bookmaker", "key_odds"])["collected_at"])
    sub = picks.loc[need]
    keys = list(zip(sub["match_id"], sub["market"], sub["selection"], sub["bookmaker"], sub["odds"].round(3)))
    found = pd.Series([last.get(k, pd.NaT) for k in keys], index=sub.index)
    times.loc[need] = pd.to_datetime(found)
    return times


def classify(picks: pd.DataFrame, quote_times: pd.Series | None = None) -> pd.Series:
    """``odds_timing`` of every priced pick (empty for a pick without a price)."""
    out = pd.Series(None, index=picks.index, dtype=object)
    priced = picks["odds"].notna()
    if not priced.any():
        return out
    origin = picks.get("odds_origin", pd.Series(None, index=picks.index)).fillna("")
    source = picks.get("source", pd.Series("live", index=picks.index))
    quote = uk_to_utc(quote_times if quote_times is not None else picks.get("odds_priced_at")).set_axis(picks.index)
    generated = pd.to_datetime(picks["generated_at"], errors="coerce")
    later = priced & origin.isin(ADDED_LATER)
    out[priced & (source == "backfill")] = CLOSING
    live = priced & (source != "backfill")
    out[live & ~later] = AT_GENERATION
    known = later & quote.notna() & generated.notna()
    out[live & known & (quote <= generated)] = EXISTED
    out[live & known & (quote > generated)] = AFTER
    out[live & later & ~known] = UNKNOWN
    return out


def label_prices(picks: pd.DataFrame, odds: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Fill a missing quote time of prices added later and (re)compute ``odds_timing``; returns the table and
    the counts per label (for the run summary)."""
    if picks.empty:
        return picks, {}
    out = picks.copy()
    if odds is None:
        from ..providers.collect import load_api_odds
        odds = load_api_odds()
    times = recover_quote_times(out, odds)
    if "odds_priced_at" in out:
        filled = out["odds_priced_at"].isna() & times.notna()
        out.loc[filled, "odds_priced_at"] = times[filled]
    out["odds_timing"] = classify(out, times)
    counts = out.loc[out["role"].isin(["main", "risk"]), "odds_timing"].value_counts().to_dict()
    return out, {k: int(v) for k, v in counts.items()}


def decision_time_prices(picks: pd.DataFrame) -> pd.DataFrame:
    """Only the predictions whose price could have been taken when the prediction was made."""
    if picks.empty or "odds_timing" not in picks:
        return picks.iloc[0:0]
    return picks.loc[picks["odds_timing"].isin(DECISION_TIME)]


def shown_quote_time(provider, when) -> object:
    """The quote time to show next to a price: none for a file without quote times (Football-Data)."""
    if provider in UNTIMED_PROVIDERS:
        return np.nan
    return when
