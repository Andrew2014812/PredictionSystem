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
    assert out.loc["3", "status"] == "VOID"            # corners not reported
    assert out.loc["4", "status"] == "VOID"            # never played, long overdue
