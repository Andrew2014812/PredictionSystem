"""v6.1 data fixes: verified provider ids (Barcelona / Espanyol) and one prediction per moved match."""
import json

import numpy as np
import pandas as pd

from src.analytics.performance import rescheduled_duplicates, summarize, unified_history
from src.providers.teams_registry import VERIFIED_IDS, TeamRegistry

LALIGA = {"Barcelona", "Espanol", "Real Madrid", "Santander", "Levante", "Ath Bilbao"}


def _registry(tmp_path, stored: dict) -> TeamRegistry:
    path = tmp_path / "provider_ids.json"
    path.write_text(json.dumps(stored))
    return TeamRegistry(path)


def test_the_wrongly_learned_espanyol_id_is_corrected(tmp_path):
    reg = _registry(tmp_path, {"oddspapi": {"2814": "Spain|Barcelona", "2829": "Spain|Real Madrid"}})
    assert reg.data["oddspapi"]["2814"] == "Spain|Espanol" and reg.data["oddspapi"]["2817"] == "Spain|Barcelona"
    assert ("oddspapi", "2814", "Spain|Barcelona", "Spain|Espanol") in reg.corrected
    assert reg.resolve("oddspapi", "SP1", "Spain", "FC Barcelona", 2817, LALIGA) == "Barcelona"
    assert reg.resolve("oddspapi", "SP1", "Spain", "Espanyol Barcelona", 2814, LALIGA) == "Espanol"
    assert reg.conflicts == []


def test_the_conflict_guard_still_refuses_a_name_that_contradicts_an_id(tmp_path):
    reg = _registry(tmp_path, {})
    # an Espanyol fixture carrying Barcelona's id gets no team (so no prices are attached)
    assert reg.resolve("oddspapi", "SP1", "Spain", "Espanyol Barcelona", 2817, LALIGA) is None
    assert reg.conflicts[-1][:3] == ("oddspapi", "2817", "Spain|Barcelona")
    # and a second id for a club that already has one is refused as well
    assert reg.resolve("oddspapi", "SP1", "Spain", "FC Barcelona", 99999, LALIGA) is None
    assert "already id 2817" in reg.conflicts[-1][3]


def test_verified_ids_are_one_id_per_club():
    for provider, ids in VERIFIED_IDS.items():
        assert len(set(ids.values())) == len(ids), provider


def _picks(rows):
    base = {"league": "F1", "home_team": "Paris SG", "away_team": "Lyon", "source": "live", "market": "1X2",
            "selection": "H", "probability": 0.6, "odds": 1.8, "home_goals": np.nan, "away_goals": np.nan}
    out = []
    for r in rows:
        for role in ("main", "risk"):
            row = {**base, **r, "role": role}
            if row["status"] in ("WON", "LOST"):
                row["profit"] = 0.8 if row["status"] == "WON" else -1.0
            out.append(row)
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"])
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    return df


def test_a_moved_match_counts_once_with_the_prediction_for_the_played_date():
    picks = _picks([
        # provisional date: predicted on the 23rd, settled later with the moved match's result
        {"match_id": "F1-20261025-paris-sg-lyon", "date": "2026-10-25", "generated_at": "2026-10-23 08:00",
         "status": "WON", "home_goals": 2, "away_goals": 1},
        {"match_id": "F1-20261026-paris-sg-lyon", "date": "2026-10-26", "generated_at": "2026-10-25 08:00",
         "status": "WON", "home_goals": 2, "away_goals": 1},
    ])
    moved = rescheduled_duplicates(picks)
    assert moved.to_dict("records") == [{"match_id": "F1-20261025-paris-sg-lyon",
                                         "kept_match_id": "F1-20261026-paris-sg-lyon"}]
    history = unified_history(picks)
    assert set(history["match_id"]) == {"F1-20261026-paris-sg-lyon"}
    assert summarize(history.loc[history["role"] == "main"])["bets"] == 1          # not two bets on one match


def test_a_recorded_prediction_wins_over_a_reconstruction_and_different_results_stay_apart():
    picks = _picks([
        {"match_id": "F1-20261003-paris-sg-lyon", "date": "2026-10-03", "generated_at": "2026-10-02",
         "status": "WON", "home_goals": 1, "away_goals": 0},
        {"match_id": "F1-20261004-paris-sg-lyon", "date": "2026-10-04", "generated_at": "2026-11-01",
         "status": "WON", "home_goals": 1, "away_goals": 0, "source": "backfill"},
    ])
    assert set(unified_history(picks)["match_id"]) == {"F1-20261003-paris-sg-lyon"}
    apart = _picks([
        {"match_id": "F1-20260810-paris-sg-lyon", "date": "2026-08-10", "generated_at": "2026-08-09",
         "status": "WON", "home_goals": 2, "away_goals": 0},
        {"match_id": "F1-20260901-paris-sg-lyon", "date": "2026-09-01", "generated_at": "2026-08-31",
         "status": "LOST", "home_goals": 0, "away_goals": 1},
    ])
    assert rescheduled_duplicates(apart).empty                     # two results: two matches (e.g. a cup tie)


def test_a_refund_is_neither_won_nor_lost_but_is_reported():
    picks = _picks([{"match_id": "I2-20261101-palermo-vicenza", "date": "2026-11-01",
                     "generated_at": "2026-10-31", "status": "PUSH", "home_goals": 3, "away_goals": 1}])
    s = summarize(picks.loc[picks["role"] == "risk"])
    assert (s["settled"], s["push"], s["bets"]) == (0, 1, 0)
