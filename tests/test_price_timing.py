"""Price timing labels: which stored prices could have been taken when the prediction was made."""
import numpy as np
import pandas as pd

from src.prediction.price_timing import (AFTER, AT_GENERATION, CLOSING, EXISTED, UNKNOWN, classify,
                                         decision_time_prices, label_prices, shown_quote_time, uk_to_utc)


def _picks():
    return pd.DataFrame({
        "match_id": ["a", "b", "c", "d", "e", "f"], "role": ["main"] * 6,
        "source": ["live", "live", "live", "backfill", "live", "live"],
        "odds": [1.8, 1.9, 2.0, 1.7, 2.1, np.nan],
        "odds_origin": [None, "archive", "archive", None, "average_replaced", None],
        # generated in UTC; quotes in UK time (BST in October: UTC+1)
        "generated_at": pd.to_datetime(["2026-10-17 10:00"] * 6),
        "odds_priced_at": pd.to_datetime(["2026-10-17 10:30", "2026-10-17 10:30", "2026-10-17 11:30",
                                          "2026-10-17 14:00", None, None]),
        "market": ["1X2"] * 6, "selection": ["H"] * 6, "bookmaker": ["bet365"] * 6,
    })


def test_prices_are_labelled_by_when_they_existed():
    labels = classify(_picks()).tolist()
    # a: recorded with the prediction; b: 10:30 BST = 09:30 UTC, before 10:00 UTC; c: 11:30 BST = 10:30 UTC, after
    assert labels[:5] == [AT_GENERATION, EXISTED, AFTER, CLOSING, UNKNOWN]
    assert pd.isna(labels[5])                                                     # no price, no label


def test_uk_times_are_compared_in_utc_across_the_clock_change():
    utc = uk_to_utc(pd.to_datetime(["2026-07-01 12:00", "2026-12-01 12:00"]))
    assert utc.tolist() == [pd.Timestamp("2026-07-01 11:00"), pd.Timestamp("2026-12-01 12:00")]


def test_a_missing_quote_time_is_recovered_from_the_stored_quotes():
    picks = _picks()
    odds = pd.DataFrame({"match_id": ["e", "e"], "market": ["1X2"] * 2, "selection": ["H"] * 2,
                         "bookmaker": ["bet365"] * 2, "odds": [2.1, 2.1],
                         "collected_at": pd.to_datetime(["2026-10-17 09:00", "2026-10-17 12:00"]),
                         "kickoff": pd.to_datetime(["2026-10-17 15:00"] * 2)})
    out, counts = label_prices(picks, odds)
    assert out.loc[4, "odds_priced_at"] == pd.Timestamp("2026-10-17 12:00")      # latest before kick-off
    assert out.loc[4, "odds_timing"] == AFTER                                    # 11:00 UTC > 10:00 UTC
    assert counts[AFTER] == 2


def test_decision_time_prices_keep_only_prices_one_could_take():
    out, _ = label_prices(_picks(), pd.DataFrame())
    assert decision_time_prices(out)["match_id"].tolist() == ["a", "b"]


def test_football_data_prices_show_no_quote_time():
    assert np.isnan(shown_quote_time("football_data", pd.Timestamp("2026-10-17")))
    assert shown_quote_time("oddspapi", pd.Timestamp("2026-10-17 10:30")) == pd.Timestamp("2026-10-17 10:30")
