"""Pre-match freezing, explanation facts, feature catalogue, datasets and Elo."""
import numpy as np
import pandas as pd

from src.analytics.performance import best_and_worst, select_dataset
from src.explain.facts import market_facts
from src.explain.narrative import key_factors
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
def _history(results, venue):
    """Team history rows (newest first): results = [(goals for, goals against), ...]."""
    return pd.DataFrame({"venue": venue, "gf": [g for g, _ in results], "ga": [a for _, a in results],
                         "corners_for": 5.0, "corners_against": 4.0,
                         "result": ["W" if g > a else ("L" if g < a else "D") for g, a in results]})


def test_total_facts_count_the_exact_line_and_the_real_number_of_matches():
    home = _history([(2, 0), (1, 0), (3, 2), (0, 0), (1, 1)], "H")         # only 5 home matches
    away = _history([(1, 2), (0, 3), (2, 2)], "A")
    facts = market_facts("TOTAL_1.5", "OVER", "Arsenal", "Leeds", home, away, limit=10)
    assert "Arsenal: 3 of the last 5 home matches had over 1.5 goals" in facts
    assert "Leeds: 3 of the last 3 away matches had over 1.5 goals" in facts
    facts = market_facts("TOTAL_2.5", "UNDER", "Arsenal", "Leeds", home, away, limit=10)
    assert "Arsenal: 4 of the last 5 home matches had under 2.5 goals" in facts
    assert not any("10" in f for f in facts)                                 # never "x of 10" with 5 matches
    uk = market_facts("TOTAL_1.5", "OVER", "Арсенал", "Лідс", home, away, lang="uk")
    assert uk and all("goals" not in f for f in uk)


def test_handicap_facts_settle_the_line_and_count_refunds():
    home = _history([(2, 0), (2, 1), (3, 0), (1, 1)], "H")
    facts = market_facts("HANDICAP_-1", "H", "Arsenal", "Leeds", home, None, limit=10)
    assert facts[0] == "With -1 Arsenal would have won this bet in 2 and got a refund in 1 of the last 4 home matches"


def test_result_facts_skip_missing_values():
    row = pd.Series({"h_position": 3.0, "a_position": np.nan})
    facts = market_facts("1X2", "H", "A", "B", _history([(1, 0), (0, 1)], "H"), None, row)
    assert facts == ["A won 1 of the last 2 home matches", "A are 3rd in the table"]


def test_influence_is_a_share_of_the_total():
    expl = pd.DataFrame({"feature": ["h_form5_ppg", "a_form5_ppg", "league_draw_rate"], "shap": [0.3, -0.1, 0.1],
                         "value": [2.1, 1.5, 0.25], "mean": [1.4, 1.4, 0.26]})
    factors = key_factors(expl)
    assert [round(f.relative, 2) for f in factors] == [0.6, 0.2, 0.2]          # 2 for, 1 substantial against
    assert [f.direction for f in factors] == ["up", "up", "down"]
    assert factors[0].level == "Supports this prediction" and factors[2].level == "Works against this prediction"
    small = pd.DataFrame({"feature": ["h_form5_ppg", "a_form5_ppg"], "shap": [0.9, -0.05], "value": [1, 1],
                          "mean": [1, 1]})
    assert [f.direction for f in key_factors(small)] == ["up"]                 # a 5 % "against" is not shown
    elo = pd.DataFrame({"feature": ["elo_diff", "h_elo", "h_form5_ppg", "a_form10_opp_elo"],
                        "shap": [0.5, 0.2, 0.2, -0.1], "value": [1, 1, 1, 1], "mean": [1, 1, 1, 1]})
    shown = key_factors(elo)
    assert [f.feature for f in shown] == ["h_form5_ppg"] and round(shown[0].relative, 2) == 0.2   # Elo hidden


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


# -- v5: pre-match versions ---------------------------------------------------
def _live_batch(prob, generated, kickoff="2026-10-12 20:00"):
    snaps = pd.DataFrame({"match_id": ["m1"], "kickoff": pd.to_datetime([kickoff]), "source": "live",
                          "p_home": prob, "generated_at": pd.Timestamp(generated)})
    picks = pd.DataFrame({"pick_id": ["m1|main"], "match_id": ["m1"], "role": "main", "source": "live",
                          "status": "UPCOMING", "probability": prob})
    return PredictionBatch(snaps, pd.DataFrame(), picks)


def test_every_pre_match_refresh_is_archived_as_a_version(tmp_path, monkeypatch):
    monkeypatch.setattr(store.predictions_storage, "root", tmp_path)
    store.save_batch(_live_batch(0.50, "2026-10-12 09:00"), now=pd.Timestamp("2026-10-12 10:00"))
    store.save_batch(_live_batch(0.60, "2026-10-12 15:00"), now=pd.Timestamp("2026-10-12 16:00"))
    store.save_batch(_live_batch(0.70, "2026-10-12 21:00"), now=pd.Timestamp("2026-10-12 21:30"))   # after kick-off
    snap = store.load_snapshots().set_index("match_id").loc["m1"]
    assert snap["p_home"] == 0.60 and snap["version"] == 2               # the last version BEFORE kick-off
    versions = store.load_versions()
    assert list(versions["version"]) == [1, 2] and list(versions["probability"]) == [0.50, 0.60]


def test_a_postponed_match_gets_a_new_version(tmp_path, monkeypatch):
    monkeypatch.setattr(store.predictions_storage, "root", tmp_path)
    store.save_batch(_live_batch(0.50, "2026-10-12 09:00"), now=pd.Timestamp("2026-10-12 10:00"))
    # the original kick-off has passed, but the match was moved to a later date
    store.save_batch(_live_batch(0.55, "2026-10-13 09:00", kickoff="2026-10-20 20:00"),
                     now=pd.Timestamp("2026-10-13 10:00"))
    snap = store.load_snapshots().set_index("match_id").loc["m1"]
    assert snap["p_home"] == 0.55 and snap["version"] == 2
    assert pd.Timestamp(snap["kickoff"]) == pd.Timestamp("2026-10-20 20:00")


def test_prices_collected_after_kick_off_are_not_pre_match_prices():
    from src.providers import pre_match_only
    matches = pd.DataFrame({"match_id": ["m1"], "kickoff": pd.to_datetime(["2026-10-12 20:00"])})
    odds = pd.DataFrame({"match_id": ["m1", "m1"], "odds": [2.0, 2.4],
                         "collected_at": pd.to_datetime(["2026-10-12 18:00", "2026-10-12 20:30"])})
    assert list(pre_match_only(odds, matches)["odds"]) == [2.0]
