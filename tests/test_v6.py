"""v6: one prediction history, the explanation card, the header search."""
import numpy as np
import pandas as pd

from src.analytics.performance import select_dataset, summarize, unified_history
from src.explain.card import card_factors, shares_are_meaningful


def _pick(match, role, source, status="WON", odds=2.0, profit=1.0):
    return {"match_id": match, "role": role, "source": source, "status": status, "odds": odds, "profit": profit,
            "date": pd.Timestamp("2026-09-01"), "market": "1X2", "selection": "H", "probability": 0.5,
            "group": "1X2", "league": "E0", "season": 2026}


def test_unified_history_prefers_recorded_and_never_counts_a_match_twice():
    picks = pd.DataFrame([
        _pick("m1", "main", "live"), _pick("m1", "risk", "live", "LOST", 4.0, -1.0),
        _pick("m1", "main", "backfill", "LOST", 1.9, -1.0),          # the same match rebuilt: ignored
        _pick("m2", "main", "backfill", "WON", 1.5, 0.5),           # no recorded prediction: rebuilt one used
        _pick("m3", "main", "backtest"),                            # test-season evaluation: not in the history
    ])
    h = unified_history(picks)
    assert sorted(zip(h["match_id"], h["role"], h["source"])) == [
        ("m1", "main", "live"), ("m1", "risk", "live"), ("m2", "main", "backfill")]
    assert select_dataset(picks, "history").equals(h)
    s = summarize(h.loc[h["role"] == "main"])
    assert s["bets"] == 2 and abs(s["roi"] - (1.0 + 0.5) / 2 * 100) < 1e-9     # one stake per match


def test_roi_never_counts_predictions_without_a_real_price():
    picks = pd.DataFrame([_pick("m1", "main", "live"), _pick("m2", "main", "live", "WON", np.nan, np.nan),
                          _pick("m3", "main", "live", "PUSH", 2.0, 0.0)])
    s = summarize(picks)
    assert s["settled"] == 2 and s["bets"] == 1 and s["roi"] == 100.0        # refund and unpriced: not staked
    assert s["hit_rate"] == 1.0


def test_card_pairs_home_and_away_values_and_shares_add_up_to_their_part():
    expl = pd.DataFrame({
        "feature": ["h_corners_for_form10", "a_corners_for_form10", "form_corners_sum", "league", "h_rest_days"],
        "shap": [0.30, 0.10, 0.25, -0.05, -0.30],
        "value": [6.1, 5.0, 11.4, "E0", 3],
    })
    f = card_factors(expl, "Stuttgart", "Gladbach")
    by = {x.key: x for x in f}
    pair = by["corners_for_form10"]
    assert pair.evidence == "Stuttgart: 6.10 · Gladbach: 5.00" and pair.supports
    assert abs(pair.share - 0.40 / 1.00) < 1e-9                    # share of the total |SHAP| = 1.00
    assert f[0].key == "corners_for_form10"                        # the biggest supporting factor first
    assert by["rest_days"].supports is False and by["rest_days"].share >= 0.10   # a big factor against is shown
    assert all(0 < x.share <= 1 for x in f)


def test_exact_score_explanations_show_no_shares():
    assert not shares_are_meaningful("EXACT_SCORE")
    assert shares_are_meaningful("CORNERS_9.5") and shares_are_meaningful("TOTAL_2.5")


def test_search_label_keeps_the_club_name_first():
    from src.ui.prefs import search_label
    assert search_label("Arsenal", "Premier League", None) == "Arsenal · Premier League"
    assert search_label("Man City", "Premier League", "Manchester City").startswith("Man City · ")


def test_draw_rule_names_a_draw_only_in_close_matches_and_keeps_probabilities():
    from src import config
    from src.modelling.decision import decide
    p = np.array([[0.36, 0.30, 0.34],      # close, P(D) high      -> draw
                  [0.50, 0.29, 0.21],      # P(D) high, not close  -> home
                  [0.37, 0.26, 0.37],      # close, P(D) too low   -> arg-max
                  [0.20, 0.30, 0.50]])
    before = p.copy()
    out = decide(p, 1.0, config.DRAW_THRESHOLD, config.DRAW_MAX_GAP)
    assert list(out) == [1, 0, 0, 2]
    assert np.array_equal(p, before)                                  # probabilities untouched
    assert list(decide(p)) == [0, 0, 0, 2]                           # without the rule: plain arg-max
