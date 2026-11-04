"""v6.1 data fixes: verified provider ids (Barcelona / Espanyol) and one prediction per moved match."""
import json

import pytest

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


# ---------------------------------------------------------------------------------------------- OddsPapi tournaments
TOURS = [
    {"tournamentId": 52, "tournamentName": "Süper Lig", "categoryName": "Türkiye", "categorySlug": "turkiye"},
    {"tournamentId": 98, "tournamentName": "1. Lig", "categoryName": "Türkiye", "categorySlug": "turkiye"},
    {"tournamentId": 207, "tournamentName": "League One", "categoryName": "Scotland", "categorySlug": "scotland"},
    {"tournamentId": 24, "tournamentName": "League One", "categoryName": "England", "categorySlug": "england"},
]


def test_a_turkish_league_is_found_under_the_turkiye_spelling():
    from src.leagues import get_league
    from src.providers.collect import match_tournament
    tid, candidates = match_tournament(get_league("T1"), TOURS)
    assert tid == 52 and {c["tournamentId"] for c in candidates} == {52, 98}
    assert match_tournament(get_league("SC2"), TOURS)[0] == 207            # not England's League One


def test_missing_tournaments_are_saved_and_asked_again_only_monthly(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    from src import config
    from src.providers import collect
    from src.storage import LocalStorage
    monkeypatch.setattr(collect, "api_storage", LocalStorage(tmp_path))
    monkeypatch.setattr(config, "ODDSPAPI_TOURNAMENTS", {})

    class FakeApi:
        calls = 0

        def tournaments(self, now, budget):
            FakeApi.calls += 1
            return TOURS
    now = datetime(2026, 11, 4, 3, 17)
    found = collect.oddspapi_tournaments(FakeApi(), now, 5)
    assert found["T1"] == 52 and found["SC2"] == 207 and "SC3" not in found
    report = collect.api_storage.read_json(collect.CANDIDATES_FILE)
    assert report["leagues"]["SC3"]["found"] is None and report["leagues"]["SC3"]["country_tournaments"]
    collect.oddspapi_tournaments(FakeApi(), now + timedelta(days=1), 5)          # SC3 still missing
    assert FakeApi.calls == 1                                                     # but no billable repeat
    collect.oddspapi_tournaments(FakeApi(), now + timedelta(days=31), 5)
    assert FakeApi.calls == 2


# ---------------------------------------------------------------------------------------------- spellings
def test_fixture_spellings_follow_the_results_files():
    from src.data.dataset import harmonise_fixture_names
    results = pd.DataFrame({"league": ["B1", "E1", "E1"], "season": [2026] * 3,
                            "home_team": ["Mechelen", "Sheffield United", "Sheffield Weds"],
                            "away_team": ["Genk", "Preston", "Leeds"]})
    fixtures = pd.DataFrame({"league": ["B1", "E1", "E1"], "date": pd.to_datetime(["2026-11-04", "2026-11-03",
                                                                                    "2026-11-07"]),
                             "home_team": ["Kortrijk", "Preston", "Sheffield Weds"],
                             "away_team": ["KV Mechelen", "Sheffield Utd", "Brand New FC"],
                             "match_id": ["x", "y", "z"]})
    out = harmonise_fixture_names(fixtures, results)
    assert out["away_team"].tolist() == ["Mechelen", "Sheffield United", "Brand New FC"]   # unknown: kept
    assert out["match_id"].tolist()[:2] == ["B1-20261104-kortrijk-mechelen", "E1-20261103-preston-sheffield-united"]


def test_a_club_booked_twice_on_one_day_counts_once_in_the_history():
    from src.analytics.performance import same_day_duplicates
    picks = _picks([
        {"match_id": "B1-20261018-mechelen-genk", "league": "B1", "home_team": "Mechelen", "away_team": "Genk",
         "date": "2026-10-18", "generated_at": "2026-10-17 10:00", "status": "WON", "home_goals": 1, "away_goals": 0},
        {"match_id": "B1-20261104-kortrijk-mechelen", "league": "B1", "home_team": "Kortrijk",
         "away_team": "Mechelen", "date": "2026-11-04", "generated_at": "2026-11-03 20:00", "status": "UPCOMING"},
        {"match_id": "B1-20261104-kortrijk-kv-mechelen", "league": "B1", "home_team": "Kortrijk",
         "away_team": "KV Mechelen", "date": "2026-11-04", "generated_at": "2026-11-02 20:00", "status": "UPCOMING"},
        {"match_id": "B1-20261018-kortrijk-gent", "league": "B1", "home_team": "Kortrijk", "away_team": "Gent",
         "date": "2026-10-18", "generated_at": "2026-10-17 10:00", "status": "LOST", "home_goals": 0, "away_goals": 1},
    ])
    dup = same_day_duplicates(picks)
    # the real club wins over an earlier prediction under the other spelling
    assert dup.to_dict("records") == [{"match_id": "B1-20261104-kortrijk-kv-mechelen",
                                       "kept_match_id": "B1-20261104-kortrijk-mechelen"}]
    assert "B1-20261104-kortrijk-kv-mechelen" not in set(unified_history(picks)["match_id"])


# ---------------------------------------------------------------------------------------------- diagnostics
def test_diagnostics_lines_say_what_a_source_delivered(tmp_path):
    from src.providers.diagnostics import RunDiagnostics
    from src.storage import LocalStorage
    d = RunDiagnostics()
    d.add("api_football", "requests", 4)
    d.http("api_football", 200)
    d.add("api_football", "returned", 126)
    d.add("api_football", "in_leagues", 92)
    d.add("api_football", "matched", 90)
    d.add("api_football", "finished", 21)
    d.add("api_football", "new_results", 6)
    d.skip("api_football", "unknown team name", 2)
    line = d.line("api_football")
    assert line.startswith("API-Football: 4 requests · 126 fixtures returned · 92 in our leagues · 90 matched")
    assert "21 finished" in line and "6 new results" in line and "2 unknown team name" in line
    d.error("api_football", "fixtures: {'access': 'Your account is suspended'}")
    assert "suspended" in d.line("api_football")
    assert d.line("oddspapi") == "OddsPapi: not used"
    report = d.save(LocalStorage(tmp_path))
    assert report["sources"]["api_football"]["counts"]["matched"] == 90
    assert (tmp_path / "diagnostics" / "history.csv").exists()


def test_past_days_without_results_are_asked_again_within_the_plan(monkeypatch):
    from datetime import date, datetime

    from src import config
    from src.providers.collect import overdue_days
    m = pd.DataFrame({"played": [False, False, True],
                      "kickoff": pd.to_datetime(["2026-11-01 15:00", "2026-11-03 20:00", "2026-11-02 15:00"])})
    now, today = datetime(2026, 11, 5, 1, 17), date(2026, 11, 5)
    monkeypatch.setattr(config, "API_FOOTBALL_PLAN_DAYS_BACK", 10)            # a paid plan
    assert overdue_days(m, now, today) == [date(2026, 11, 1), date(2026, 11, 3)]
    monkeypatch.setattr(config, "API_FOOTBALL_PLAN_DAYS_BACK", 1)             # the free plan
    assert overdue_days(m, now, today) == []                                  # yesterday is asked anyway


def test_a_market_average_price_is_replaced_by_a_real_quote_or_removed():
    from src.prediction.history_repair import AVERAGE, replace_average_prices
    picks = pd.DataFrame({
        "match_id": ["m1", "m2"], "source": ["live", "live"], "role": ["main", "risk"],
        "market": ["1X2", "TOTAL_2.5"], "selection": ["H", "OVER"], "probability": [0.6, 0.4],
        "odds": [2.10, 2.60], "bookmaker": [AVERAGE, AVERAGE], "provider": ["football_data"] * 2,
        "real_odds": [True, True], "ev": [0.26, 0.04], "status": ["WON", "LOST"], "profit": [1.10, -1.0],
        "odds_priced_at": pd.to_datetime(["2026-10-02", "2026-10-02"])})
    quotes = pd.DataFrame({"match_id": ["m1"], "market": ["1X2"], "selection": ["H"], "odds": [2.0],
                           "bookmaker": ["bet365"], "provider": ["oddspapi"],
                           "collected_at": pd.to_datetime(["2026-10-03 10:00"])})
    out, replaced, removed = replace_average_prices(picks, quotes, pd.Series(dtype=float))
    assert (replaced, removed) == (1, 1)
    a, b = out.iloc[0], out.iloc[1]
    assert (a["odds"], a["bookmaker"], a["odds_origin"], a["replaced_average_odds"]) == (2.0, "bet365", "average_replaced", 2.10)
    assert a["profit"] == pytest.approx(1.0) and a["selection"] == "H" and a["status"] == "WON"     # pick unchanged
    assert np.isnan(b["odds"]) and np.isnan(b["profit"]) and not b["real_odds"] and b["status"] == "LOST"
    again, r2, d2 = replace_average_prices(out, quotes, pd.Series(dtype=float))
    assert (r2, d2) == (0, 0)                                                   # safe to run every day


def test_history_invariants_report_problems_instead_of_hiding_them():
    from src.analytics.invariants import check_history
    picks = _picks([{"match_id": "F1-20261010-paris-sg-lyon", "date": "2026-10-10", "generated_at": "2026-10-09 10:00",
                     "status": "WON", "home_goals": 2, "away_goals": 1}])
    picks["pick_id"] = picks["match_id"] + "|" + picks["role"]
    picks["bookmaker"], picks["real_odds"] = "bet365", True
    picks["odds_priced_at"] = pd.Timestamp("2026-10-10 12:00")
    matches = pd.DataFrame({"match_id": ["F1-20261010-paris-sg-lyon"], "league": ["F1"], "home_team": ["Paris SG"],
                            "away_team": ["Lyon"], "kickoff": [pd.Timestamp("2026-10-10 20:00")],
                            "home_goals": [2.0], "away_goals": [1.0]})
    assert check_history(picks, matches) == {}
    bad = picks.copy()
    bad.loc[0, "bookmaker"] = "Market average"
    bad.loc[1, "odds_priced_at"] = pd.Timestamp("2026-10-10 21:00")                  # after kick-off
    issues = check_history(bad, matches.assign(home_goals=3.0))
    assert {"price without a real bookmaker", "price quoted after kick-off", "result differs from the match"} <= set(issues)
    assert check_history(picks.loc[picks["role"] == "main"], matches) == {"main without a risk": ["F1-20261010-paris-sg-lyon"]}


def test_the_odds_api_can_be_limited_to_named_leagues(monkeypatch):
    from src import config
    monkeypatch.setenv("ODDS_API_LEAGUES", "t1, sc2")
    assert config.odds_api_leagues() == {"T1", "SC2"}
    monkeypatch.setenv("ODDS_API_LEAGUES", "")
    assert config.odds_api_leagues() == set()                                  # default: nothing is bought
