import numpy as np
import pandas as pd

from src.prediction.history_repair import fill_archive_prices, restore_missing_risk


def _picks():
    base = {"match_id": "m1", "role": "main", "probability": 0.6, "model_version": "prod_1",
            "generated_at": pd.Timestamp("2026-10-01 08:00"), "bookmaker": None, "provider": None,
            "real_odds": False, "ev": np.nan, "profit": np.nan}
    return pd.DataFrame([
        {**base, "market": "1X2", "selection": "H", "source": "live", "odds": np.nan, "status": "WON"},
        {**base, "market": "TOTAL_2.5", "selection": "OVER", "source": "live", "odds": 1.9, "status": "LOST",
         "bookmaker": "Bet365", "provider": "the_odds_api", "real_odds": True, "ev": 0.14, "profit": -1.0},
        {**base, "market": "HANDICAP_-1", "selection": "H", "source": "live", "odds": np.nan, "status": "PUSH"},
        {**base, "market": "BTTS", "selection": "YES", "source": "backfill", "odds": np.nan, "status": "WON"},
    ])


def _prices():
    return pd.DataFrame({"match_id": ["m1"] * 4, "market": ["1X2", "TOTAL_2.5", "HANDICAP_-1", "BTTS"],
                         "selection": ["H", "OVER", "H", "YES"], "odds": [2.0, 2.2, 2.5, 1.8],
                         "bookmaker": ["Pinnacle"] * 4, "provider": ["oddspapi"] * 4})


def test_only_missing_recorded_prices_are_filled_and_marked():
    push = pd.Series([0.2], index=pd.MultiIndex.from_tuples([("m1", "HANDICAP_-1", "H")]))
    out, n = fill_archive_prices(_picks(), _prices(), push)
    assert n == 3
    first, kept, whole, backfill = (out.iloc[i] for i in range(4))
    assert first["odds"] == 2.0 and first["odds_origin"] == "archive" and first["provider"] == "oddspapi"
    assert first["profit"] == 1.0 and abs(first["ev"] - 0.2) < 1e-9
    assert first["probability"] == 0.6 and first["model_version"] == "prod_1"            # the pick is kept
    assert first["generated_at"] == pd.Timestamp("2026-10-01 08:00")
    assert kept["odds"] == 1.9 and kept["bookmaker"] == "Bet365" and kept["odds_origin"] is None   # never replaced
    assert whole["profit"] == 0.0 and abs(whole["ev"] - (0.6 * 2.5 - 0.8)) < 1e-9     # refund: stake back
    assert backfill["odds"] == 1.8 and backfill["odds_origin"] == "archive"            # retrospective rows too
    only_recorded, n = fill_archive_prices(_picks(), _prices(), push, sources=("live",))
    assert n == 2 and np.isnan(only_recorded.iloc[3]["odds"])


def test_restore_is_idempotent():
    once, _ = fill_archive_prices(_picks(), _prices(), pd.Series(dtype=float))
    twice, n = fill_archive_prices(once, _prices(), pd.Series(dtype=float))
    assert n == 0 and twice["odds"].equals(once["odds"])


def _markets():
    rows = [("1X2", "H", 0.55, 1.7), ("1X2", "D", 0.25, 3.6), ("1X2", "A", 0.20, 4.5),
            ("DOUBLE_CHANCE", "1X", 0.80, 1.2), ("DOUBLE_CHANCE", "X2", 0.45, 2.1),
            ("TOTAL_2.5", "OVER", 0.52, 1.9), ("TOTAL_2.5", "UNDER", 0.48, 1.95)]
    return pd.DataFrame([{"match_id": "m1", "market": m, "selection": s_, "probability": p, "odds": o,
                          "bookmaker": "bet365", "provider": "oddspapi", "push_probability": 0.0,
                          "priced_at": pd.Timestamp("2026-10-01 09:00")} for m, s_, p, o in rows])


def _main_only():
    return pd.DataFrame([{"pick_id": "m1|main|1X2|H", "match_id": "m1", "role": "main", "kind": "", "market": "1X2",
                          "group": "1X2", "selection": "H", "probability": 0.55, "odds": 1.7, "ev": -0.065,
                          "bookmaker": "bet365", "provider": "oddspapi", "real_odds": True, "status": "WON",
                          "profit": 0.7, "source": "live", "model_version": "prod_1",
                          "generated_at": pd.Timestamp("2026-10-01 08:00")}])


def test_a_missing_risk_is_rebuilt_from_the_stored_version_and_the_main_is_kept():
    now = pd.Timestamp("2026-10-23 15:00")
    out, n = restore_missing_risk(_main_only(), _markets(), now)
    assert n == 1 and len(out) == 2
    main, risk = out.iloc[0], out.iloc[1]
    assert (main["market"], main["selection"], main["probability"], main["profit"]) == ("1X2", "H", 0.55, 0.7)
    assert risk["role"] == "risk" and risk["origin"] == "risk_reconstructed" and risk["reconstructed_at"] == now
    assert (risk["market"], risk["selection"]) not in {("1X2", "H"), ("DOUBLE_CHANCE", "1X")}   # not the Main again
    assert risk["status"] == "UPCOMING" and np.isnan(risk["profit"])     # settled by the pipeline, not here
    assert risk["model_version"] == "prod_1" and risk["generated_at"] == pd.Timestamp("2026-10-01 08:00")
    assert abs(risk["ev"] - (risk["probability"] * risk["odds"] - 1)) < 1e-9
    again, n2 = restore_missing_risk(out, _markets(), now)
    assert n2 == 0 and len(again) == 2                                     # one Risk per match, never two
