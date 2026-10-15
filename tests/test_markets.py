import numpy as np
import pytest

from src import config
from src.markets.distributions import (corners_distribution, outcome_probabilities, prob_over,
                                       reconcile_with_1x2, score_matrix)
from src.markets.markets import corner_markets, goal_markets, selection_won
from src.markets.odds import PricedSelection, price
from src.markets.recommendation import main_prediction, odds_preference, risk_prediction


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
    # Asian -1: home "wins" only by 2+, a 1-goal win is a refund
    assert p[("HANDICAP_-1", "H")] < p[("1X2", "H")]
    sels = {(s.market, s.selection): s for s in goal_markets(reconcile_with_1x2(score_matrix(1.5, 1.0),
                                                                                np.array([.48, .27, .25])))}
    for line in config.HANDICAP_LINES:
        h, a = sels[(f"HANDICAP_{line:+g}", "H")], sels[(f"HANDICAP_{line:+g}", "A")]
        assert (f"HANDICAP_{line:+g}", "D") not in p                      # Asian: never a "draw" outcome
        assert h.probability + a.probability + h.push_probability == pytest.approx(1.0)
        assert h.push_probability == a.push_probability
        if line % 1:
            assert h.push_probability == 0                                # half line: no refund
        else:                                                             # European 3-way kept separately
            assert h.push_probability > 0
            assert sum(p[(f"EHANDICAP_{line:+g}", s)] for s in "HDA") == pytest.approx(1.0)
            assert p[(f"EHANDICAP_{line:+g}", "D")] == pytest.approx(h.push_probability)
    # half line -0.5 is exactly "home win"; +0.5 home is "home or draw"
    assert p[("HANDICAP_-0.5", "H")] == pytest.approx(p[("1X2", "H")])
    assert p[("HANDICAP_+0.5", "H")] == pytest.approx(p[("DOUBLE_CHANCE", "1X")])


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
    ("EXACT_SCORE", "2-1", 2, 1, None, True),
    # Asian whole lines: an exact margin is a refund (None from selection_won)
    ("HANDICAP_-1", "H", 3, 1, None, True), ("HANDICAP_-1", "H", 2, 1, None, None),
    ("HANDICAP_-1", "A", 2, 1, None, None), ("HANDICAP_-1", "A", 1, 1, None, True),
    ("HANDICAP_+1", "A", 0, 2, None, True), ("HANDICAP_+1", "H", 0, 1, None, None),
    ("HANDICAP_-2", "H", 2, 0, None, None), ("HANDICAP_-2", "H", 2, 1, None, False),
    # European 3-way: every score decides
    ("EHANDICAP_-1", "D", 2, 1, None, True), ("EHANDICAP_+1", "D", 0, 1, None, True),
    ("EHANDICAP_-2", "H", 3, 0, None, True), ("EHANDICAP_-1", "A", 1, 1, None, True),
    # half lines: two outcomes, no draw after the handicap
    ("HANDICAP_-0.5", "H", 1, 0, None, True), ("HANDICAP_-0.5", "A", 1, 1, None, True),
    ("HANDICAP_+0.5", "H", 1, 1, None, True), ("HANDICAP_-1.5", "H", 2, 1, None, False),
    ("HANDICAP_-1.5", "A", 2, 1, None, True), ("HANDICAP_+2.5", "H", 0, 2, None, True),
    ("CORNERS_9.5", "OVER", 1, 1, 11, True), ("CORNERS_9.5", "OVER", 1, 1, None, None),
])
def test_settlement_rules(market, selection, hg, ag, corners, expected):
    assert selection_won(market, selection, hg, ag, corners) is expected


def test_only_real_prices_are_attached():
    sels = goal_markets(score_matrix(1.5, 1.1)) + corner_markets(corners_distribution(10, None))
    real = {("1X2", "H"): {"odds": 2.0, "bookmaker": "Bet365", "provider": "football_data"},
            ("TOTAL_2.5", "OVER"): {"odds": 1.9, "bookmaker": "Bet365", "provider": "football_data"}}
    priced = {(s.market, s.selection): s for s in price(sels, real)}
    s = priced[("1X2", "H")]
    assert s.odds == 2.0 and s.bookmaker == "Bet365" and s.ev == pytest.approx(s.probability * 2.0 - 1)
    # nothing is derived or simulated: every other selection has no price and no EV
    others = [p for k, p in priced.items() if k not in real]
    assert others and all(p.odds is None and p.ev is None and not p.has_odds for p in others)


def test_no_odds_at_all():
    priced = price(goal_markets(score_matrix(1.5, 1.1)), None)
    assert not any(p.has_odds for p in priced)


def _ps(market, selection, p, odds=None):
    return PricedSelection(market, selection, p, odds, "Bet365" if odds else None)


def test_main_prediction_softly_prefers_odds_in_the_preferred_range():
    sels = [_ps("DOUBLE_CHANCE", "1X", 0.84, 1.12),          # very likely, but a 1.12 price
            _ps("1X2", "H", 0.55, 1.95)]                     # inside 1.40–2.50
    main = main_prediction(sels)
    assert main.kind == "priced" and (main.pick.market, main.pick.selection) == ("1X2", "H")
    # outside the range the weight decays smoothly instead of dropping to zero
    assert 0 < odds_preference(1.12) < odds_preference(1.37) < odds_preference(1.40) == 1.0
    assert odds_preference(2.50) == 1.0 > odds_preference(2.6) > odds_preference(3.5) > 0
    assert odds_preference(1.37) > 0.95 and odds_preference(2.6) > 0.95        # barely penalised


def test_main_prediction_does_not_chase_ev_without_probability():
    sels = [_ps("1X2", "H", 0.56, 1.80),                     # EV +0.8%, likely
            _ps("1X2", "A", 0.41, 2.60)]                     # EV +6.6%, much less likely
    assert main_prediction(sels).pick.selection == "H"


def test_main_prefers_a_real_price_over_a_similar_unpriced_selection():
    sels = [_ps("1X2", "H", 0.55, 1.85), _ps("BTTS", "YES", 0.57)]
    assert main_prediction(sels).pick.market == "1X2"


def test_main_prediction_exists_without_any_odds():
    sels = [_ps("1X2", "H", 0.48), _ps("1X2", "D", 0.27), _ps("1X2", "A", 0.25), _ps("TOTAL_2.5", "UNDER", 0.55)]
    main = main_prediction(sels)
    assert main is not None and main.pick.ev is None and main.kind == "model"
    assert main.pick.market == "TOTAL_2.5"


def test_main_prediction_ignores_near_certain_selections():
    sels = [_ps("TOTAL_1.5", "OVER", 0.92), _ps("1X2", "H", 0.38), _ps("1X2", "A", 0.32), _ps("1X2", "D", 0.30)]
    main = main_prediction(sels)
    assert main.pick.market == "1X2" and main.pick.selection == "H"      # falls back to most probable 1X2


def test_risk_prediction_exists_without_positive_ev():
    main = _ps("1X2", "H", 0.50, 2.0)
    sels = [main, _ps("1X2", "D", 0.27, 3.4), _ps("1X2", "A", 0.23, 4.0)]     # both EV negative
    risk = risk_prediction(sels, main)
    assert risk is not None and risk.pick.ev < 0
    assert risk.pick.selection == "D"                    # more likely of two similar prices


def test_risk_never_repeats_main_in_another_form():
    main = _ps("1X2", "H", 0.50, 2.0)
    sels = [main, _ps("HANDICAP_-0.5", "H", 0.50, 2.02), _ps("1X2", "D", 0.25, 3.6)]
    assert risk_prediction(sels, main).pick.market == "1X2"           # "home -0.5" is the same bet
    assert risk_prediction(sels, main).pick.selection == "D"


def test_risk_is_not_the_longest_price_on_the_board():
    main = _ps("1X2", "H", 0.55, 1.8)
    sels = [main, _ps("TOTAL_3.5", "OVER", 0.30, 3.3), _ps("EXACT_SCORE", "3-2", 0.03, 41.0)]
    assert risk_prediction(sels, main).pick.market == "TOTAL_3.5"


def test_risk_exists_without_any_real_price():
    main = _ps("1X2", "H", 0.50)
    risk = risk_prediction([main, _ps("1X2", "A", 0.25), _ps("BTTS", "NO", 0.45)], main)
    assert risk is not None and risk.kind == "model" and risk.pick.odds is None


@pytest.mark.parametrize("market,selection,hg,ag,expected", [
    ("HANDICAP_-1", "H", 2, 1, "PUSH"), ("HANDICAP_-1", "H", 3, 1, "WON"), ("HANDICAP_-1", "H", 1, 1, "LOST"),
    ("HANDICAP_+1", "A", 0, 1, "PUSH"), ("HANDICAP_+2", "H", 0, 2, "PUSH"), ("HANDICAP_+2", "H", 0, 1, "WON"),
    ("HANDICAP_-0.5", "H", 1, 1, "LOST"), ("HANDICAP_+1.5", "H", 0, 1, "WON"), ("HANDICAP_-2.5", "A", 2, 0, "WON"),
    ("EHANDICAP_-1", "H", 2, 1, "LOST"), ("EHANDICAP_-1", "D", 2, 1, "WON"),
])
def test_asian_and_european_handicap_settlement(market, selection, hg, ag, expected):
    from src.markets.markets import settle_selection
    assert settle_selection(market, selection, hg, ag) == expected


def test_asian_whole_line_ev_counts_the_refund():
    # p(win) 0.45, p(refund) 0.25, p(loss) 0.30 at odds 2.0 -> 0.45 * 1 - 0.30 = +0.15
    sel = PricedSelection("HANDICAP_-1", "H", 0.45, 2.0, push_probability=0.25)
    assert sel.ev == pytest.approx(0.15) and sel.lose_probability == pytest.approx(0.30)
    assert PricedSelection("1X2", "H", 0.45, 2.0).ev == pytest.approx(-0.10)
