import numpy as np
import pandas as pd
import pytest

from src.analytics.performance import group_summary, summarize, timeline
from src.prediction import store


def _picks():
    return pd.DataFrame({
        "match_id": list("abcde"), "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                                                           "2026-01-09", "2026-01-10"]),
        "league": ["E0", "E0", "D1", "D1", "D1"], "group": ["TOTAL", "1X2", "TOTAL", "BTTS", "1X2"],
        "status": ["WON", "LOST", "WON", "UPCOMING", "VOID"], "odds": [1.8, 2.5, 2.0, 1.9, 2.0],
        "probability": [.6, .45, .55, .5, .5], "profit": [0.8, -1.0, 1.0, np.nan, 0.0],
    })


def test_overall_roi_and_counts():
    s = summarize(_picks())
    assert s["settled"] == 3 and s["won"] == 2 and s["lost"] == 1
    assert s["pending"] == 1 and s["void"] == 1
    assert s["profit"] == pytest.approx(0.8)
    assert s["roi"] == pytest.approx(0.8 / 3 * 100)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["longest_win_streak"] == 1 and s["longest_loss_streak"] == 1


def test_roi_by_league_and_period():
    g = group_summary(_picks(), "league").set_index("league")
    assert g.loc["E0", "roi"] == pytest.approx(-10.0)
    assert g.loc["D1", "roi"] == pytest.approx(100.0)
    t = timeline(_picks(), "Weekly")
    assert t["predictions"].sum() == 3
    assert t["cumulative_profit"].iloc[-1] == pytest.approx(0.8)


def test_settlement_profit(tmp_path, monkeypatch):
    from src.storage import LocalStorage
    monkeypatch.setattr(store, "predictions_storage", LocalStorage(tmp_path))
    picks = pd.DataFrame({
        "pick_id": ["1", "2", "3", "4"], "match_id": ["m1", "m1", "m1", "m2"],
        "league": ["E0"] * 4, "home_team": ["A"] * 3 + ["C"], "away_team": ["B"] * 3 + ["D"],
        "date": pd.to_datetime(["2026-01-01"] * 3 + ["2025-11-01"]),
        "market": ["TOTAL_2.5", "1X2", "CORNERS_9.5", "1X2"], "selection": ["OVER", "A", "OVER", "H"],
        "odds": [1.8, 3.0, 1.9, 2.0], "status": ["UPCOMING"] * 4, "profit": [np.nan] * 4,
        "home_goals": np.nan, "away_goals": np.nan, "total_corners": np.nan, "settled_at": pd.NaT,
    })
    store.predictions_storage.write_parquet(store.PICKS, picks)
    results = pd.DataFrame({"match_id": ["m1"], "league": ["E0"], "home_team": ["A"], "away_team": ["B"],
                            "date": pd.to_datetime(["2026-01-01"]), "played": [True],
                            "home_goals": [2.0], "away_goals": [1.0], "total_corners": [np.nan]})
    out = store.settle(results, today=pd.Timestamp("2026-01-05").date()).set_index("pick_id")
    assert (out.loc["1", "status"], out.loc["1", "profit"]) == ("WON", pytest.approx(0.8))
    assert (out.loc["2", "status"], out.loc["2", "profit"]) == ("LOST", -1.0)
    assert out.loc["3", "status"] == "UPCOMING"        # corners not reported yet: wait, do not void
    assert out.loc["4", "status"] == "VOID"            # never played, long overdue


def test_roi_uses_only_real_odds():
    picks = _picks()
    extra = pd.DataFrame({"match_id": ["f", "g"], "date": pd.to_datetime(["2026-01-11", "2026-01-12"]),
                          "league": ["E0", "E0"], "group": ["BTTS", "BTTS"], "status": ["WON", "LOST"],
                          "odds": [np.nan, np.nan], "probability": [.6, .6], "profit": [np.nan, np.nan]})
    s = summarize(pd.concat([picks, extra], ignore_index=True))
    assert s["settled"] == 5 and s["won"] == 3                 # hit rate counts every settled prediction
    assert s["bets"] == 3                                      # ... ROI only the ones with real odds
    assert s["roi"] == pytest.approx(0.8 / 3 * 100)
    assert np.isnan(summarize(extra)["roi"]) and np.isnan(summarize(extra)["profit"])


def test_void_is_not_staked():
    s = summarize(_picks())
    assert s["void"] == 1 and s["bets"] == 3


def test_best_and_worst_market_need_a_minimum_sample():
    from src.analytics.performance import best_and_worst
    rows = []
    for i in range(120):
        rows.append({"match_id": f"t{i}", "date": pd.Timestamp("2026-01-01"), "group": "TOTAL", "league": "E0",
                     "status": "WON" if i % 2 else "LOST", "odds": 2.2, "probability": .5,
                     "profit": 1.2 if i % 2 else -1.0})
        rows.append({"match_id": f"b{i}", "date": pd.Timestamp("2026-01-01"), "group": "BTTS", "league": "E0",
                     "status": "LOST", "odds": 1.9, "probability": .5, "profit": -1.0})
    rows.append({"match_id": "x", "date": pd.Timestamp("2026-01-01"), "group": "CORNERS", "league": "E0",
                 "status": "WON", "odds": 5.0, "probability": .5, "profit": 4.0})
    best, worst = best_and_worst(pd.DataFrame(rows))
    assert best["group"] == "TOTAL" and worst["group"] == "BTTS"    # CORNERS: too few bets


def test_date_presets():
    from src.analytics.performance import preset_range
    today = pd.Timestamp("2026-10-07")
    assert preset_range("Last 7 days", today, "2025-01-01") == (pd.Timestamp("2026-10-01"), today)
    assert preset_range("Last 30 days", today, "2025-01-01")[0] == pd.Timestamp("2026-09-08")
    assert preset_range("This month", today, "2025-01-01")[0] == pd.Timestamp("2026-10-01")
    assert preset_range("This year", today, "2025-01-01")[0] == pd.Timestamp("2026-01-01")
    assert preset_range("This season", today, "2025-01-01", season_start="2026-07-01")[0] == pd.Timestamp("2026-07-01")
    assert preset_range("All time", today, "2025-01-01")[0] == pd.Timestamp("2025-01-01")
    assert preset_range("Custom", today, "2025-01-01", ("2026-02-01", "2026-02-10")) == \
        (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"))
