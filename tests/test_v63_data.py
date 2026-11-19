"""v6.3 data checks after the daily update (src/analytics/data_checks.py) and the reproduction check of older
explanations (src/prediction/inputs.reproduces)."""
import numpy as np
import pandas as pd

from src.analytics import data_checks as DC
from src.prediction.inputs import reproduces


def _matches(**over):
    base = pd.DataFrame({
        "match_id": ["E0-a", "E0-b", "E0-c"], "league": ["E0"] * 3, "season": [2026] * 3,
        "date": pd.to_datetime(["2026-11-01", "2026-11-01", "2026-11-08"]),
        "kickoff": pd.to_datetime(["2026-11-01 15:00", "2026-11-01 15:00", "2026-11-08 15:00"]),
        "home_team": ["A", "C", "A"], "away_team": ["B", "D", "C"], "played": [True, True, False],
        "home_goals": [1.0, 2.0, np.nan], "away_goals": [0.0, 2.0, np.nan], "total_corners": [9.0, 11.0, np.nan],
        "home_shots": [10.0, 8.0, np.nan], "away_shots": [7.0, 9.0, np.nan], "home_sot": [4.0, 3.0, np.nan],
        "away_sot": [2.0, 4.0, np.nan], "home_corners": [5.0, 6.0, np.nan], "away_corners": [4.0, 5.0, np.nan],
    })
    for k, v in over.items():
        base[k] = v
    return base


def test_clean_data_pass_every_check():
    assert DC.check_matches(_matches(), pd.Timestamp("2026-11-10")) == {}
    assert DC.check_statistics(_matches()) == {}


def test_duplicates_swaps_and_score_states_are_reported():
    m = pd.concat([_matches(), _matches().iloc[[0]].assign(match_id="E0-a2"),
                   _matches().iloc[[1]].assign(match_id="E0-b-swapped", home_team="D", away_team="C")],
                  ignore_index=True)
    m.loc[2, "home_goals"] = 1.0                                       # a score on an unplayed match
    out = DC.check_matches(m, pd.Timestamp("2026-11-10"))
    assert set(out["same pairing twice on one day"]) == {"E0-a", "E0-a2"}
    assert out["home and away swapped between two rows"] == ["E0-b"]
    assert out["score of an unplayed match"] == ["E0-c"]
    future = _matches(date=pd.to_datetime(["2026-12-01", "2026-11-01", "2026-11-08"]))
    assert DC.check_matches(future, pd.Timestamp("2026-11-10"))["played match dated in the future"] == ["E0-a"]


def test_impossible_statistics_are_reported():
    m = _matches(away_sot=[9.0, 4.0, np.nan], total_corners=[40.0, 11.0, np.nan], home_corners=[5.0, np.nan, np.nan])
    out = DC.check_statistics(m)
    assert out["away shots on target above shots"] == ["E0-a"]
    assert out["more than 35 corners"] == ["E0-a"]
    assert out["corners of one side only"] == ["E0-b"]


def test_results_prices_and_predictions():
    m = _matches()
    live = pd.DataFrame({"match_id": ["E0-c"], "played": [True], "home_goals": [2.0]})
    assert DC.check_unused_results(m, live) == {"provider result not used": ["E0-c"]}
    odds = pd.DataFrame({"match_id": ["E0-a", "E0-a"], "market": ["1X2", "CORNERS_3.5"], "odds": [0.95, 1.01],
                         "collected_at": pd.to_datetime(["2026-11-01 10:00", "2026-11-01 16:00"]),
                         "kickoff": pd.to_datetime(["2026-11-01 15:00"] * 2)})
    out = DC.check_prices(odds)
    assert out["impossible price"] == ["E0-a 1X2"]                     # 1.01 is a real quote, not an error
    assert out["price collected after kick-off (stored, not used)"] == [1]
    picks = pd.DataFrame({"pick_id": ["p1", "p2"], "match_id": ["E0-a", "E0-b"], "source": ["live", "live"],
                          "generated_at": pd.to_datetime(["2026-11-01 09:00", "2026-11-01 16:30"]),
                          "market": ["1X2", "TOTAL_2.5"], "selection": ["H", "OVER"], "status": ["LOST", "WON"],
                          "home_goals": [1.0, 2.0], "away_goals": [0.0, 2.0], "total_corners": [9.0, 11.0]})
    out = DC.check_predictions(picks, m)
    assert out["recorded prediction generated after kick-off"] == ["p2"]   # 16:30 UTC > 15:00 UK (GMT)
    assert out["status does not follow from the result"] == ["p1"]         # 1-0 and home win: WON, not LOST


def test_feature_counts_cannot_know_later_matches():
    m = _matches()
    m["h_season_games"] = [0, 0, 1]
    m["a_season_games"] = [0, 0, 1]
    assert DC.check_feature_time(m) == {}
    m.loc[2, "a_season_games"] = 3                                  # C has played one match before E0-c
    assert DC.check_feature_time(m) == {"feature counts more matches than were played before": ["E0-c"]}


class _Fixed:
    def __init__(self, value, extra=None):
        self.value, self.extra = value, extra or {}

    def predict(self, df):
        return np.array([self.value])


def test_an_older_prediction_counts_as_reproduced_only_if_its_model_gives_the_stored_numbers():
    snap = pd.Series({"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "xg_home": 1.6, "xg_away": 1.1})
    bundles = {"result": _Fixed([0.501, 0.299, 0.2]), "home_goals": _Fixed(1.605), "away_goals": _Fixed(1.1)}
    row = pd.Series({"match_id": "x"})
    assert reproduces(row, snap, bundles)
    bundles["result"] = _Fixed([0.52, 0.28, 0.2])
    assert not reproduces(row, snap, bundles)
    assert not reproduces(None, snap, bundles) and not reproduces(row, snap, {})


def test_a_match_the_bookmakers_stopped_quoting_is_reported_as_probably_postponed():
    from src.analytics.waiting import waiting_matches
    feats = _matches()
    feats.loc[2, ["played", "home_goals"]] = [False, np.nan]
    picks = pd.DataFrame({"pick_id": ["p"], "match_id": ["E0-c"], "league": ["E0"], "date": [pd.Timestamp("2026-11-08")],
                          "home_team": ["A"], "away_team": ["C"], "market": ["1X2"], "status": ["UPCOMING"]})
    odds = pd.DataFrame({"match_id": ["E0-c"], "collected_at": [pd.Timestamp("2026-11-07 05:00")]})
    table = waiting_matches(picks, feats.drop(index=[0, 1]), today=pd.Timestamp("2026-11-12"), odds=odds)
    assert table["reason"].iloc[0] == "no result; bookmakers stopped quoting 34 h before kick-off (probably postponed)"
    recent = odds.assign(collected_at=pd.Timestamp("2026-11-08 09:00"))
    table = waiting_matches(picks, feats.drop(index=[0, 1]), today=pd.Timestamp("2026-11-12"), odds=recent)
    assert table["reason"].iloc[0] == "no result from any source yet"


def test_lineups_pilot_counts_line_ups_and_player_statistics_and_stays_in_budget():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "api_football_lineups_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    live = pd.DataFrame({"provider": ["api_football", "api_football", "oddspapi", "api_football"],
                         "played": [True, True, True, False], "date": pd.to_datetime(["2026-11-18"] * 4),
                         "provider_fixture_id": ["11", "12", "id99", "13"]})
    from datetime import date
    assert pilot.fixture_ids(live, date(2026, 11, 18)) == [11, 12]
    assert [len(b) for b in pilot.batches(list(range(45)))] == [20, 20, 5]
    player = {"statistics": [{"games": {"minutes": 90, "rating": "7.1", "position": "M"}}]}
    item = {"league": {"id": 39}, "lineups": [{"formation": "4-3-3", "startXI": [{}] * 11},
                                              {"formation": "4-4-2", "startXI": [{}] * 11}],
            "players": [{"players": [player]}, {"players": [player, {"statistics": [{"games": {}}]}]}]}
    s = pilot.summarise([item, {"league": {"id": 39}}])
    assert s["fixtures"] == 2 and s["with_lineups_both"] == 1 and s["with_players_both"] == 1
    assert (s["players"], s["players_with_rating"], s["players_with_minutes"]) == (3, 2, 2)
    assert s["start_xi_sizes"] == {"11": 2} and s["leagues"] == {"39": 2}
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/lineups-pilot.yml").read_text()
    assert "workflow_dispatch" in text and "schedule" not in text and "contents: read" in text
