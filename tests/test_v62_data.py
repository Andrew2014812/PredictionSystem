"""v6.2 data fixes: moved matches, withdrawn results, waiting matches and the run diagnostics."""
import numpy as np
import pandas as pd
import pytest

from src.prediction import store


def _pick(mid, date, status, goals=(np.nan, np.nan), role="main", source="live", gen="2026-10-30 16:13",
          market="TOTAL_2.5"):
    return {"pick_id": f"{mid}-{role}-{market}", "match_id": mid, "league": "F1", "home_team": "Nice",
            "away_team": "Rennes", "date": pd.Timestamp(date), "role": role, "source": source,
            "generated_at": pd.Timestamp(gen), "status": status, "home_goals": goals[0], "away_goals": goals[1],
            "market": market, "selection": "OVER", "odds": 1.8, "profit": np.nan, "total_corners": np.nan,
            "settled_at": pd.NaT}


# ---------------------------------------------------------------------------------------------- moved matches
def test_a_match_moved_to_an_earlier_day_keeps_the_played_date():
    """Nice – Rennes was listed for 31.10 and played on 30.10: the 31.10 prediction never gets a result."""
    from src.analytics.performance import rescheduled_duplicates
    picks = pd.DataFrame([_pick("F1-20261030-nice-rennes", "2026-10-30", "LOST", (5, 2)),
                          _pick("F1-20261031-nice-rennes", "2026-10-31", "UPCOMING")])
    moved = rescheduled_duplicates(picks)
    assert moved["match_id"].tolist() == ["F1-20261031-nice-rennes"]
    assert moved["kept_match_id"].tolist() == ["F1-20261030-nice-rennes"]


def test_a_postponed_match_still_keeps_the_later_date():
    from src.analytics.performance import rescheduled_duplicates
    picks = pd.DataFrame([_pick("F1-20261018-nice-rennes", "2026-10-18", "WON", (2, 1)),
                          _pick("F1-20261105-nice-rennes", "2026-11-05", "WON", (2, 1), gen="2026-11-05 07:00")])
    assert rescheduled_duplicates(picks)["kept_match_id"].tolist() == ["F1-20261105-nice-rennes"]


# ---------------------------------------------------------------------------------------------- withdrawn results
def _results(rows):
    base = {"league": "SP2", "home_team": "Tenerife", "away_team": "Andorra", "home_goals": np.nan,
            "away_goals": np.nan, "total_corners": np.nan}
    return pd.DataFrame([{**base, **r} for r in rows]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_a_result_the_source_withdrew_reopens_the_prediction(tmp_path, monkeypatch):
    """API-Football reported Tenerife – Andorra 2–2 for a day, then as not played; no source has it now."""
    from src.storage import LocalStorage
    monkeypatch.setattr(store, "predictions_storage", LocalStorage(tmp_path))
    pick = {**_pick("SP2-20261101-tenerife-andorra", "2026-11-01", "WON", (2, 2)), "league": "SP2",
            "home_team": "Tenerife", "away_team": "Andorra", "profit": 0.8, "settled_at": pd.Timestamp("2026-11-02")}
    store.predictions_storage.write_parquet(store.PICKS, pd.DataFrame([pick]))
    results = _results([{"match_id": "SP2-20261101-tenerife-andorra", "date": "2026-11-01", "played": False},
                        {"match_id": "SP2-20261102-tenerife-andorra", "date": "2026-11-02", "played": False}])
    out = store.settle(results, today=pd.Timestamp("2026-11-09").date())
    row = out.iloc[0]
    assert row["status"] == "UPCOMING" and np.isnan(row["profit"]) and pd.notna(row["reopened_at"])
    later = store.settle(results, today=pd.Timestamp("2026-11-30").date()).iloc[0]
    assert later["status"] == "VOID"                                  # still no result: voided in time


def test_a_result_that_still_exists_is_not_reopened(tmp_path, monkeypatch):
    from src.storage import LocalStorage
    monkeypatch.setattr(store, "predictions_storage", LocalStorage(tmp_path))
    pick = {**_pick("SP2-20261101-tenerife-andorra", "2026-11-01", "WON", (2, 2)), "league": "SP2",
            "home_team": "Tenerife", "away_team": "Andorra", "profit": 0.8}
    store.predictions_storage.write_parquet(store.PICKS, pd.DataFrame([pick]))
    results = _results([{"match_id": "SP2-20261101-tenerife-andorra", "date": "2026-11-01", "played": False},
                        {"match_id": "SP2-20261102-tenerife-andorra", "date": "2026-11-02", "played": True,
                         "home_goals": 2.0, "away_goals": 2.0}])
    out = store.settle(results, today=pd.Timestamp("2026-11-09").date())
    assert out.iloc[0]["status"] == "WON" and out.iloc[0]["profit"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------------------------- waiting matches
def test_waiting_matches_name_the_reason():
    from src.analytics.waiting import NO_RESULT, STATISTICS, waiting_matches
    picks = pd.DataFrame([
        {**_pick("E0-20261108-sunderland-chelsea", "2026-11-08", "UPCOMING", market="CORNERS_9.5"),
         "league": "E0", "home_team": "Sunderland", "away_team": "Chelsea"},
        {**_pick("SC1-20261024-morton-ayr", "2026-10-24", "UPCOMING"), "league": "SC1", "home_team": "Morton",
         "away_team": "Ayr"},
        {**_pick("SP2-20261106-castellon-eibar", "2026-11-06", "UPCOMING"), "league": "SP2",
         "home_team": "Castellon", "away_team": "Eibar"},
        {**_pick("F1-20261031-nice-rennes", "2026-10-31", "UPCOMING")},
    ])
    features = pd.DataFrame([
        {"match_id": "E0-20261108-sunderland-chelsea", "league": "E0", "home_team": "Sunderland",
         "away_team": "Chelsea", "date": pd.Timestamp("2026-11-08"), "played": True, "home_goals": 1.0},
        {"match_id": "SC1-20261024-morton-ayr", "league": "SC1", "home_team": "Morton", "away_team": "Ayr",
         "date": pd.Timestamp("2026-10-24"), "played": False, "home_goals": np.nan},
        {"match_id": "SP2-20261106-castellon-eibar", "league": "SP2", "home_team": "Castellon",
         "away_team": "Eibar", "date": pd.Timestamp("2026-11-06"), "played": False, "home_goals": np.nan},
        {"match_id": "F1-20261030-nice-rennes", "league": "F1", "home_team": "Nice", "away_team": "Rennes",
         "date": pd.Timestamp("2026-10-30"), "played": True, "home_goals": 5.0},
    ])
    not_played = pd.DataFrame([{"league": "SP2", "date": "2026-11-06", "home_team": "Castellon",
                                "away_team": "Eibar", "status": "PST", "seen_at": "2026-11-07T08:00:00"}])
    table = waiting_matches(picks, features, None, None, not_played, today=pd.Timestamp("2026-11-10"))
    reasons = dict(zip(table["match_id"], table["reason"]))
    assert reasons["E0-20261108-sunderland-chelsea"] == STATISTICS
    assert reasons["SC1-20261024-morton-ayr"] == NO_RESULT
    assert reasons["SP2-20261106-castellon-eibar"] == "postponed (API-Football PST)"
    assert "moved match" in reasons["F1-20261031-nice-rennes"]
    assert table.set_index("match_id").loc["SC1-20261024-morton-ayr", "void_after"] == "2026-11-14"


def test_postponed_statuses_are_kept_once_per_match(tmp_path, monkeypatch):
    from src.providers import collect
    from src.storage import LocalStorage
    monkeypatch.setattr(collect, "api_storage", LocalStorage(tmp_path))
    row = {"league": "SP2", "date": "2026-11-06", "home_team": "Castellon", "away_team": "Eibar", "status": "PST"}
    collect.save_not_played([{**row, "seen_at": "2026-11-06T08:00"}])
    collect.save_not_played([{**row, "seen_at": "2026-11-07T08:00"}])
    kept = collect.load_not_played()
    assert len(kept) == 1 and kept.iloc[0]["seen_at"] == "2026-11-07T08:00"


# ---------------------------------------------------------------------------------------------- diagnostics
def test_each_source_is_timed_apart_and_steps_are_listed():
    import time

    from src.providers.diagnostics import RunDiagnostics
    d = RunDiagnostics()
    with d.timer("api_football"):
        time.sleep(0.01)
        with d.timer("oddspapi"):
            time.sleep(0.02)
    with d.stage("features"):
        pass
    timing = d.timing()
    assert timing["oddspapi"] >= 0.0 and set(timing) >= {"api_football", "oddspapi", "features", "total"}
    assert d.lines()[-1].startswith("Timing: ")


def test_settled_predictions_are_counted_apart_from_matches():
    import importlib.util
    from pathlib import Path

    from src.providers.diagnostics import diag
    spec = importlib.util.spec_from_file_location("run_pipeline", Path(__file__).parent.parent / "scripts/run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    diag.reset()
    done = pd.DataFrame({"match_id": ["a", "a", "a", "b", "b", "c"],
                         "role": ["main", "risk", "market", "main", "risk", "main"],
                         "status": ["WON", "PUSH", "LOST", "LOST", "WON", "VOID"]})
    mod.record_settlement(done)
    c = diag.data["predictions"]["counts"]
    assert (c["settled_now"], c["settled_matches"], c["settled_main"], c["settled_risk"]) == (5, 3, 2, 2)
    assert (c["settled_market"], c["settled_push"], c["voided_now"]) == (1, 1, 1)
    assert "on 3 matches: Main 2, Risk 2, market forecasts 1, refunds 1" in diag.line("predictions")


def test_a_club_spelt_without_history_is_reported():
    """The fixtures file once wrote "KV Mechelen" for "Mechelen": a club with no history must be named."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_pipeline", Path(__file__).parent.parent / "scripts/run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    soon = pd.Timestamp.now().normalize() + pd.Timedelta(days=2)
    features = pd.DataFrame([
        {"league": "B1", "home_team": "Mechelen", "away_team": "Genk", "played": True, "date": pd.Timestamp("2026-10-31")},
        {"league": "B1", "home_team": "Kortrijk", "away_team": "Gent", "played": True, "date": pd.Timestamp("2026-10-30")},
        {"league": "B1", "home_team": "Kortrijk", "away_team": "KV Mechelen", "played": False, "date": soon},
    ])
    assert mod.clubs_without_history(features) == ["B1: KV Mechelen"]


# ---------------------------------------------------------------------------------------------- v6.2 study fixes
def test_another_providers_fixture_row_does_not_undo_a_result(tmp_path, monkeypatch):
    from src.providers import collect
    from src.storage import LocalStorage
    monkeypatch.setattr(collect, "live_storage", LocalStorage(tmp_path))
    base = {"match_id": "SP2-20261102-tenerife-andorra", "league": "SP2", "country": "Spain", "season": 2026,
            "date": "2026-11-02", "time": "13:00", "kickoff": "2026-11-02 13:00", "home_team": "Tenerife",
            "away_team": "Andorra"}
    collect._save_live([{**base, "played": True, "home_goals": 2, "away_goals": 2, "provider": "api_football"}])
    collect._save_live([{**base, "played": False, "provider": "oddspapi"}])          # a price feed's listing
    row = collect.load_live_matches().iloc[0]
    assert bool(row["played"]) and row["home_goals"] == 2
    collect._save_live([{**base, "played": False, "provider": "api_football"}])      # the source corrects itself
    assert not bool(collect.load_live_matches().iloc[0]["played"])


def test_current_season_result_names_are_harmonised():
    """05.11.2026: the current results files briefly wrote "KV Mechelen", "Sheffield Utd" and "Oxford Utd"."""
    from src.data.dataset import harmonise_names
    hist = pd.DataFrame([{"league": "B1", "season": 2025, "home_team": "Mechelen", "away_team": "Genk"},
                         {"league": "B1", "season": 2025, "home_team": "Kortrijk", "away_team": "Gent"},
                         {"league": "E1", "season": 2025, "home_team": "Sheffield United", "away_team": "Preston"},
                         {"league": "B1", "season": 2026, "home_team": "RAAL La Louviere", "away_team": "Genk"}])
    rows = pd.DataFrame([{"league": "B1", "season": 2026, "date": pd.Timestamp("2026-11-04"),
                          "home_team": "Kortrijk", "away_team": "KV Mechelen", "match_id": "x"},
                         {"league": "E1", "season": 2026, "date": pd.Timestamp("2026-11-03"),
                          "home_team": "Preston", "away_team": "Sheffield Utd", "match_id": "y"},
                         {"league": "B1", "season": 2026, "date": pd.Timestamp("2026-11-07"),
                          "home_team": "RAAL La Louviere", "away_team": "Gent", "match_id": "z"}])
    out = harmonise_names(rows, pd.concat([hist, rows]), min_matches=3)
    assert out["away_team"].tolist()[:2] == ["Mechelen", "Sheffield United"]
    assert out["match_id"].tolist()[:2] == ["B1-20261104-kortrijk-mechelen", "E1-20261103-preston-sheffield-united"]
    assert out["home_team"].iloc[2] == "RAAL La Louviere"                  # a newcomer keeps its name


def test_fc_edinburgh_is_edinburgh_city():
    from src.data.cleaning import normalise
    raw = pd.DataFrame({"Div": ["SC2"], "Date": ["05/11/2022"], "HomeTeam": ["FC Edinburgh"], "AwayTeam": ["Peterhead"],
                        "FTHG": [1], "FTAG": [0], "FTR": ["H"]})
    assert normalise(raw)["home_team"].iloc[0] == "Edinburgh City"


def test_api_football_spellings_resolve():
    from src.providers.aliases import resolve
    assert resolve("OFI", {"OFI Crete", "AEK"}) == "OFI Crete"
    assert resolve("Vicenza Virtus", {"Vicenza", "Padova"}) == "Vicenza"
    assert resolve("Real Sociedad II", {"Sociedad B", "Eibar"}) == "Sociedad B"


def test_an_unpriced_whole_line_is_judged_at_its_break_even_price():
    """A whole handicap line refunds on a level result: break-even (1 − P(refund)) / P(win), not 1 / P(win)."""
    from src.markets.odds import PricedSelection
    from src.markets.recommendation import effective_odds
    sel = PricedSelection(market="HANDICAP_-1", selection="H", probability=0.33,
                          push_probability=0.22)
    assert effective_odds(sel) == pytest.approx((1 - 0.22) / 0.33)
    assert sel.fair_odds == pytest.approx((1 - 0.22) / 0.33)
