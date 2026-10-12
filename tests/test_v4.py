"""Pre-match freezing, explanation facts, feature catalogue, datasets and Elo."""
import numpy as np
import pandas as pd

from src.analytics.performance import best_and_worst, select_dataset
from src.explain.facts import market_facts
from src.explain.narrative import influence_table
from src.features.builder import build_features
from src.features.catalog import catalogue, family_of
from src.features.state import EloRatings
from src.prediction import store
from src.prediction.engine import PredictionBatch
from src.prediction.pipeline import select_live_targets
from src.ui.components import missing


# -- pre-match only ----------------------------------------------------------
def _features(kickoffs, played):
    return pd.DataFrame({"match_id": [f"m{i}" for i in range(len(kickoffs))],
                         "date": [pd.Timestamp(k).normalize() for k in kickoffs],
                         "kickoff": pd.to_datetime(kickoffs), "played": played})


def test_only_matches_that_have_not_kicked_off_are_predicted_as_live():
    f = _features(["2026-10-12 12:00", "2026-10-12 17:30", "2026-10-13 20:00"], [False, False, False])
    upcoming, _ = select_live_targets(f, pd.DataFrame(), pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-12 15:00"))
    assert list(upcoming["match_id"]) == ["m1", "m2"]          # the 12:00 match has started


def test_started_match_keeps_its_recorded_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(store.predictions_storage, "root", tmp_path)

    def batch(prob, generated):
        snaps = pd.DataFrame({"match_id": ["m0", "m1"], "kickoff": pd.to_datetime(["2026-10-12 12:00",
                                                                                   "2026-10-12 20:00"]),
                              "source": "live", "p_home": prob, "generated_at": pd.Timestamp(generated)})
        picks = pd.DataFrame({"pick_id": ["m0|main", "m1|main"], "match_id": ["m0", "m1"], "role": "main",
                              "source": "live", "status": "UPCOMING", "probability": prob})
        return PredictionBatch(snaps, pd.DataFrame(), picks)

    store.save_batch(batch(0.5, "2026-10-12 09:00"), now=pd.Timestamp("2026-10-12 10:00"))
    store.save_batch(batch(0.7, "2026-10-12 14:00"), now=pd.Timestamp("2026-10-12 15:00"))
    snaps = store.load_snapshots().set_index("match_id")
    assert snaps.loc["m0", "p_home"] == 0.5                    # frozen at kick-off
    assert snaps.loc["m1", "p_home"] == 0.7                    # still upcoming: refreshed
    assert snaps.loc["m1", "first_generated_at"] == pd.Timestamp("2026-10-12 09:00")
    assert store.load_picks().set_index("match_id").loc["m0", "probability"] == 0.5


# -- datasets and ROI highlights ----------------------------------------------
def _picks(n, group, profit, source="live", league="E0"):
    return pd.DataFrame({"group": group, "league": league, "source": source, "status": "WON",
                         "odds": 2.0, "profit": profit, "date": pd.Timestamp("2026-10-01")}, index=range(n))


def test_recorded_and_evaluation_datasets_are_separate():
    picks = pd.concat([_picks(2, "1X2", 1.0, "live"), _picks(3, "1X2", 1.0, "backtest"),
                       _picks(4, "1X2", 1.0, "backfill")], ignore_index=True)
    assert len(select_dataset(picks, "recorded")) == 2
    assert len(select_dataset(picks, "evaluation")) == 7


def test_best_and_worst_need_a_minimum_sample_and_follow_the_given_rows():
    picks = pd.concat([_picks(40, "TOTAL", 1.0), _picks(40, "BTTS", -1.0), _picks(5, "CORNERS", 1.0)],
                      ignore_index=True)
    best, worst = best_and_worst(picks)
    assert best["group"] == "TOTAL" and worst["group"] == "BTTS"           # 5 corner bets are too few
    best, worst = best_and_worst(picks.loc[picks["group"] != "TOTAL"])
    assert best["group"] == "BTTS" and worst is None


# -- explanations --------------------------------------------------------------
def test_goal_market_facts_use_pre_match_numbers():
    row = pd.Series({"h_venue_gf": 2.1, "a_venue_gf": 1.3, "a_venue_ga": 1.8, "h_venue_ga": 0.9,
                     "h_freq_over25": 0.7, "a_freq_over25": 0.6, "league_over25_rate": 0.55})
    facts = market_facts("TOTAL_2.5", row, "Arsenal", "Leeds", limit=10)
    assert "Arsenal score 2.10 goals per game at home this season" in facts
    assert "Arsenal: 7 of the last 10 matches had over 2.5 goals" in facts
    assert "In this league 55% of matches go over 2.5 goals" in facts
    uk = market_facts("TOTAL_2.5", row, "Арсенал", "Лідс", lang="uk")
    assert uk and all("goals" not in f for f in uk)


def test_result_facts_skip_missing_values():
    row = pd.Series({"h_position": 3.0, "a_position": np.nan, "elo_diff": -42.0})
    facts = market_facts("1X2", row, "A", "B")
    assert facts == ["A are 3rd in the table", "Strength rating gap (Elo, incl. home advantage): 42 in favour of B"]


def test_influence_is_a_share_of_the_total():
    expl = pd.DataFrame({"feature": ["h_elo", "a_elo", "league_draw_rate"], "shap": [0.3, -0.1, 0.1],
                         "value": [1600, 1500, 0.25], "mean": [1500, 1500, 0.26]})
    factors = influence_table(expl, n=3)
    assert [round(f.relative, 2) for f in factors] == [0.6, 0.2, 0.2]
    assert factors[0].direction == "up" and factors[1].direction == "down"


def test_missing_catches_numpy_nan_of_every_width():
    assert missing(np.float32("nan")) and missing(np.float64("nan")) and missing(None) and missing(pd.NA)
    assert not missing(1.85) and not missing("1.85")


# -- features ----------------------------------------------------------------
def test_every_feature_family_has_a_readable_entry():
    entries = catalogue({"result": ["h_form5_gf", "a_form10_gf", "elo_diff", "h_h2h_win_rate"]}, "uk")
    names = {e["name"] for e in entries}
    assert "Забиті / пропущені голи (форма)" in names and "Рейтинг сили Elo" in names
    assert family_of("h_form10_adj_cf") is not None and family_of("league=E0") is not None


def test_elo_moves_towards_the_result_and_regresses_between_seasons():
    elo = EloRatings()
    home, away = ("E0", "A"), ("E0", "B")
    elo.update(home, away, 3, 0, 2024, 1)
    assert elo.get(home, 2024, 1) > 1500 > elo.get(away, 2024, 1)
    gain = elo.get(home, 2024, 1) - 1500
    assert abs(elo.get(home, 2025, 1) - 1500 - 0.8 * gain) < 1e-9
    assert EloRatings.base(2) == 1400


def test_elo_features_use_only_earlier_matches(make_matches):
    matches = make_matches([("05/08/2023", "A", "B", 3, 0), ("12/08/2023", "B", "A", 1, 1),
                            ("19/08/2023", "A", "B", None, None)])
    f = build_features(matches).sort_values("date").reset_index(drop=True)
    assert f.loc[0, "h_elo"] == f.loc[0, "a_elo"] == 1500         # nothing known before the first match
    assert f.loc[1, "a_elo"] > 1500 > f.loc[1, "h_elo"]           # A won the first match
    assert f.loc[2, "h_elo"] != f.loc[1, "a_elo"]                 # the draw was applied before match 3


def test_dixon_coles_rho_zero_is_poisson_and_fit_recovers_rho():
    from src.markets.distributions import fit_dixon_coles_rho, score_matrix
    assert np.allclose(score_matrix(1.4, 1.1, rho=0.0), score_matrix(1.4, 1.1))
    corrected = score_matrix(1.4, 1.1, rho=-0.1)
    assert abs(corrected.sum() - 1) < 1e-9 and corrected[0, 0] > score_matrix(1.4, 1.1)[0, 0]
    rng = np.random.default_rng(3)
    lh, la = rng.uniform(1.0, 2.0, 4000), rng.uniform(0.7, 1.5, 4000)
    cells = [rng.choice(121, p=score_matrix(a, b, rho=-0.1).ravel()) for a, b in zip(lh, la)]
    rho = fit_dixon_coles_rho(lh, la, np.array(cells) // 11, np.array(cells) % 11)
    assert -0.2 < rho < -0.03
