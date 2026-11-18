"""v6.3: readable factor values on the "Why …?" card (who is ahead, by how much, in what unit)."""
import pandas as pd

from src.explain import evidence as EVD
from src.explain.card import card_factors


def test_differences_name_the_leading_team_and_keep_the_magnitude():
    assert EVD.difference("elo_diff", -108.0, "Leganes", "Burgos") == "Burgos +108 Elo points"
    assert EVD.difference("elo_diff", 54.4, "Leganes", "Burgos") == "Leganes +54 Elo points"
    assert EVD.difference("diff_season_gd", -1.31, "Leganes", "Burgos") == "Burgos +1.31 goals/game"
    assert EVD.difference("diff_season_ppg", -0.92, "Leganes", "Burgos") == "Burgos +0.92 points/game"
    assert EVD.difference("diff_form10_shots", 2.04, "Leganes", "Burgos") == "Leganes +2.0 shots/game"
    assert EVD.difference("diff_points", 0.0, "Leganes", "Burgos") == "level"
    # a lower table position is better: home 3rd vs away 8th = -5 -> home 5 places higher
    assert EVD.difference("diff_position", -5.0, "Leganes", "Burgos") == "Leganes 5 places higher"
    assert EVD.difference("diff_position", 1.0, "Leganes", "Burgos") == "Burgos 1 place higher"


def test_expected_result_and_pairs_show_units():
    assert EVD.expected_result(0.3512, "Leganes") == "Leganes expected result: 0.35 / 1"
    assert EVD.pair("form10_gf", {"h": 1.45, "a": 1.2}, "Leganes", "Burgos") == "Leganes 1.45 · Burgos 1.20 goals/game"
    assert EVD.pair("form10_win_rate", {"h": 0.6, "a": 0.3}, "A", "B") == "A 60% · B 30%"
    assert EVD.pair("elo", {"h": 1612.4, "a": 1540.0}, "A", "B") == "A 1612 · B 1540 Elo points"
    assert EVD.pair("position", {"h": 3.0, "a": 12.0}, "A", "B") == "A 3rd · B 12th"
    assert EVD.pair("position", {"h": 21.0, "a": 1.0}, "A", "B", "uk") == "A 21 місце · B 1 місце"
    assert EVD.pair("trend_scoring", {"h": 0.31}, "A", "B") == "A +0.31 vs season average"
    assert EVD.single("h2h_avg_goals", 2.4) == "2.40 goals/match"
    assert EVD.single("h2h_draw_rate", 0.25) == "25%"
    assert EVD.difference("elo_diff", -108.0, "Леганес", "Бургос", "uk") == "Бургос +108 очок Ело"


def test_card_uses_the_readable_values_without_changing_numbers_or_direction():
    expl = pd.DataFrame({"feature": ["elo_diff", "elo_exp_home", "diff_season_gd", "h_form10_gf", "a_form10_gf"],
                         "contribution": [-0.04, -0.02, -0.03, 0.01, 0.005],
                         "value": [-108.0, 0.35, -1.31, 1.45, 1.2]})
    by = {f.key: f for f in card_factors(expl, "Leganes", "Burgos")}
    assert by["elo_diff"].evidence == "Burgos +108 Elo points" and by["elo_diff"].supports is False
    assert by["form10_gf"].evidence == "Leganes 1.45 · Burgos 1.20 goals/game" and by["form10_gf"].supports
    total = 0.04 + 0.02 + 0.03 + 0.015
    assert abs(by["elo_diff"].share - 0.04 / total) < 1e-9          # relative influence unchanged
