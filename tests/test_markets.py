import numpy as np
import pytest

from src import config
from src.markets.distributions import (corners_distribution, outcome_probabilities, prob_over,
                                       reconcile_with_1x2, score_matrix)
from src.markets.markets import corner_markets, goal_markets, selection_won
from src.markets.odds import fit_market_score_matrix, price
from src.markets.recommendation import main_prediction, risk_prediction
from src.markets.odds import PricedSelection


def _by_key(sels):
    return {(s.market, s.selection): s.probability for s in sels}


def test_score_matrix_is_a_distribution():
    m = score_matrix(1.6, 1.1)
    assert m.shape == (config.MAX_GOALS + 1,) * 2
    assert m.sum() == pytest.approx(1.0) and (m >= 0).all()


def test_reconciled_1x2_matches_classifier_and_sums_to_one():
    target = np.array([0.5, 0.3, 0.2])
    m = reconcile_with_1x2(score_matrix(1.4, 1.2), target)
    assert m.sum() == pytest.approx(1.0)
    assert outcome_probabilities(m) == pytest.approx(target)


def test_market_probabilities_are_consistent():
    p = _by_key(goal_markets(reconcile_with_1x2(score_matrix(1.5, 1.0), np.array([.48, .27, .25]))))
    assert p[("1X2", "H")] + p[("1X2", "D")] + p[("1X2", "A")] == pytest.approx(1.0)
    for line in config.TOTAL_GOAL_LINES:
        assert p[(f"TOTAL_{line}", "OVER")] + p[(f"TOTAL_{line}", "UNDER")] == pytest.approx(1.0)
    assert p[("TOTAL_1.5", "OVER")] > p[("TOTAL_2.5", "OVER")] > p[("TOTAL_3.5", "OVER")]
    assert p[("BTTS", "YES")] + p[("BTTS", "NO")] == pytest.approx(1.0)
    assert p[("DOUBLE_CHANCE", "1X")] == pytest.approx(p[("1X2", "H")] + p[("1X2", "D")])
    assert p[("DOUBLE_CHANCE", "X2")] == pytest.approx(p[("1X2", "D")] + p[("1X2", "A")])
    assert p[("DOUBLE_CHANCE", "12")] == pytest.approx(p[("1X2", "H")] + p[("1X2", "A")])
    # handicap +0 would equal 1X2; -1 home win is "win by 2+" and smaller than a home win
    assert p[("HANDICAP_-1", "H")] < p[("1X2", "H")]
    for line in config.HANDICAP_LINES:
        total = sum(p[(f"HANDICAP_{line:+d}", s)] for s in "HDA")
        assert total == pytest.approx(1.0)


def test_exact_scores_are_top_cells_in_order():
    m = score_matrix(1.5, 1.0)
    exact = [s for s in goal_markets(m) if s.market == "EXACT_SCORE"]
    assert len(exact) == config.EXACT_SCORE_TOP_N
    probs = [s.probability for s in exact]
    assert probs == sorted(probs, reverse=True)
    h, a = (int(x) for x in exact[0].selection.split("-"))
    assert m[h, a] == pytest.approx(m.max())


def test_corner_distribution_and_lines():
    pmf = corners_distribution(10.2, 12.0)
    assert pmf.sum() == pytest.approx(1.0)
    p = _by_key(corner_markets(pmf))
    assert p[("CORNERS_9.5", "OVER")] + p[("CORNERS_9.5", "UNDER")] == pytest.approx(1.0)
    assert prob_over(pmf, 8.5) > prob_over(pmf, 10.5)


@pytest.mark.parametrize("market,selection,hg,ag,corners,expected", [
    ("1X2", "H", 2, 1, None, True), ("1X2", "D", 2, 1, None, False),
    ("DOUBLE_CHANCE", "X2", 1, 1, None, True), ("TOTAL_2.5", "OVER", 2, 1, None, True),
    ("TOTAL_2.5", "UNDER", 2, 1, None, False), ("BTTS", "YES", 2, 0, None, False),
    ("EXACT_SCORE", "2-1", 2, 1, None, True), ("HANDICAP_-1", "D", 2, 1, None, True),
    ("HANDICAP_-1", "H", 3, 1, None, True), ("HANDICAP_+1", "A", 0, 2, None, True),
    ("CORNERS_9.5", "OVER", 1, 1, 11, True), ("CORNERS_9.5", "OVER", 1, 1, None, None),
])
def test_settlement_rules(market, selection, hg, ag, corners, expected):
    assert selection_won(market, selection, hg, ag, corners) is expected


def test_odds_sources_are_separated():
    match = {"odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.8, "odds_over25": 1.9, "odds_under25": 1.95}
    sels = goal_markets(score_matrix(1.5, 1.1)) + corner_markets(corners_distribution(10, None))
    base = {(s.market, s.selection): s.probability for s in sels}
    priced = {(s.market, s.selection): s for s in
              price(sels, match, fit_market_score_matrix(match), base, score_matrix(1.4, 1.1), base)}
    assert priced[("1X2", "H")].odds_source == "market" and priced[("1X2", "H")].odds == 2.0
    assert priced[("TOTAL_2.5", "OVER")].odds_source == "market"
    assert priced[("BTTS", "YES")].odds_source == "derived"
    assert priced[("CORNERS_9.5", "OVER")].odds_source == "simulated"
    s = priced[("1X2", "H")]
    assert s.fair_odds == pytest.approx(1 / s.probability)
    assert s.ev == pytest.approx(s.probability * 2.0 - 1)


def test_without_any_odds_everything_is_simulated():
    sels = goal_markets(score_matrix(1.5, 1.1))
    base = {(s.market, s.selection): s.probability for s in sels}
    priced = price(sels, {}, None, base, score_matrix(1.4, 1.1), {})
    assert {s.odds_source for s in priced} == {"simulated"}


def _ps(market, selection, p, odds, source="market"):
    return PricedSelection(market, selection, p, odds, source)


def test_main_prediction_prefers_value_over_probability():
    sels = [_ps("DOUBLE_CHANCE", "1X", 0.80, 1.20),          # very likely, odds too low
            _ps("1X2", "H", 0.55, 2.00),                     # EV +10%
            _ps("TOTAL_2.5", "OVER", 0.60, 1.70)]            # EV +2%
    main = main_prediction(sels)
    assert main.kind == "value" and (main.pick.market, main.pick.selection) == ("1X2", "H")


def test_main_prediction_falls_back_to_confidence():
    sels = [_ps("1X2", "H", 0.50, 1.80), _ps("TOTAL_2.5", "OVER", 0.58, 1.60)]
    main = main_prediction(sels)
    assert main.kind == "confidence" and main.pick.market == "TOTAL_2.5"


def test_risk_prediction_rules():
    assert risk_prediction([_ps("1X2", "A", 0.30, 3.6)]).pick.selection == "A"
    assert risk_prediction([_ps("1X2", "A", 0.20, 6.0)]) is None        # probability too low
    assert risk_prediction([_ps("1X2", "A", 0.30, 3.2)]) is None        # EV below threshold
    assert risk_prediction([_ps("EXACT_SCORE", "2-1", 0.30, 9.0)]) is None
