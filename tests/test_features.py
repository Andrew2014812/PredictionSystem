import numpy as np
import pandas as pd

from src.features.builder import build_features
from src.features.sets import GROUPS, model_features


def _fixture_set(make_matches):
    return make_matches([
        ("01/08/2025", "A", "B", 3, 0),
        ("01/08/2025", "C", "D", 1, 1),
        ("08/08/2025", "A", "C", 2, 2),   # same day as B-D: must not see each other
        ("08/08/2025", "B", "D", 0, 4),
        ("15/08/2025", "A", "D", 1, 0),
        ("22/08/2025", "B", "C", None, None),  # future fixture
    ])


def test_first_match_has_no_history(make_matches):
    f = build_features(_fixture_set(make_matches)).set_index("match_id")
    first = f.iloc[0]
    assert first["h_season_games"] == 0 and first["h_form5_n"] == 0 and first["h2h_n"] == 0
    assert np.isnan(first.get("h_season_ppg", np.nan))


def test_features_use_only_earlier_days(make_matches):
    f = build_features(_fixture_set(make_matches))
    a_vs_c = f.loc[(f["home_team"] == "A") & (f["away_team"] == "C")].iloc[0]
    # A won 3-0 on day 1; nothing from day 2 (incl. its own 2-2) may be visible
    assert a_vs_c["h_season_games"] == 1
    assert a_vs_c["h_season_gf"] == 3 and a_vs_c["h_season_ga"] == 0
    assert a_vs_c["h_points"] == 3 and a_vs_c["h_position"] == 1
    assert a_vs_c["h_rest_days"] == 7


def test_simultaneous_matches_do_not_leak(make_matches):
    f = build_features(_fixture_set(make_matches))
    b_vs_d = f.loc[(f["home_team"] == "B") & (f["away_team"] == "D")].iloc[0]
    # D's 4-0 win is on the same day; its season goals must still be 1 per game
    assert b_vs_d["a_season_gf"] == 1


def test_future_fixture_does_not_update_state(make_matches):
    matches = _fixture_set(make_matches)
    with_fixture = build_features(matches)
    without = build_features(matches.loc[matches["played"]])
    played_ids = without["match_id"]
    cols = [c for c in without.columns if c.startswith(("h_", "a_", "diff_", "h2h_", "league_"))]
    pd.testing.assert_frame_equal(
        with_fixture.set_index("match_id").loc[played_ids, cols].sort_index(),
        without.set_index("match_id")[cols].sort_index(), check_dtype=False)
    fixture = with_fixture.loc[~with_fixture["played"]].iloc[0]
    assert fixture["h_season_games"] == 2 and pd.isna(fixture["home_goals"])


def test_features_are_invariant_to_later_data(make_matches):
    """Appending later matches must not change features of earlier ones."""
    matches = _fixture_set(make_matches)
    early = build_features(matches.iloc[:4]).set_index("match_id")
    full = build_features(matches).set_index("match_id")
    cols = [c for c in early.columns if c.startswith(("h_", "a_", "diff_", "h2h_"))]
    pd.testing.assert_frame_equal(early[cols], full.loc[early.index, cols], check_dtype=False)


def test_odds_are_not_model_features():
    cols = ["market_prob_home", "odds_home", "h_season_ppg", "h_form5_gf", "h_venue_corners_for",
            "league_avg_corners", "h2h_draw_rate"]
    for model in ("result", "home_goals", "away_goals", "corners"):
        numeric, categorical = model_features(model, cols)
        assert not any(c.startswith(("odds_", "market_")) for c in numeric)
        assert categorical == ["league"]
    assert "h_venue_corners_for" in model_features("corners", cols)[0]
    assert "h_venue_corners_for" not in model_features("result", cols)[0]
    assert "market" in GROUPS
