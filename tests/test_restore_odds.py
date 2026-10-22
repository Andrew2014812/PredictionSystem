import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("restore_recorded_odds",
                                              Path(__file__).parent.parent / "scripts/restore_recorded_odds.py")
restore_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_mod)


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
    out, n = restore_mod.restore(_picks(), _prices(), push)
    assert n == 2
    first, kept, whole, backfill = (out.iloc[i] for i in range(4))
    assert first["odds"] == 2.0 and first["odds_origin"] == "archive" and first["provider"] == "oddspapi"
    assert first["profit"] == 1.0 and abs(first["ev"] - 0.2) < 1e-9
    assert first["probability"] == 0.6 and first["model_version"] == "prod_1"            # the pick is kept
    assert first["generated_at"] == pd.Timestamp("2026-10-01 08:00")
    assert kept["odds"] == 1.9 and kept["bookmaker"] == "Bet365" and kept["odds_origin"] is None   # never replaced
    assert whole["profit"] == 0.0 and abs(whole["ev"] - (0.6 * 2.5 - 0.8)) < 1e-9     # refund: stake back
    assert np.isnan(backfill["odds"])                                                  # recorded only


def test_restore_is_idempotent():
    once, _ = restore_mod.restore(_picks(), _prices(), pd.Series(dtype=float))
    twice, n = restore_mod.restore(once, _prices(), pd.Series(dtype=float))
    assert n == 0 and twice["odds"].equals(once["odds"])
