"""v6.1 "Why this prediction?" card: the explanation is computed on the market probability itself,
through the chain features → models → expected goals / distribution → market."""
import numpy as np
import pandas as pd
import pytest

from src.explain import chain
from src.explain.card import INFLUENCE_TIP, card_factors, chain_note, factor_table, shares_are_meaningful
from src.markets.distributions import corners_distribution, prob_over, reconcile_with_1x2, score_matrix
from src.markets.markets import goal_markets
from src.modelling.tasks import TASKS
from src.modelling.training import TaskTrainer, corner_leagues
from src.prediction.engine import predict_matches
from tests import test_prediction
from tests.test_prediction import PARAMS

synthetic = test_prediction.synthetic          # the module fixture of the synthetic league
CASES = [(1.62, 1.10, (0.48, 0.27, 0.25)), (0.85, 2.05, (0.20, 0.24, 0.56)), (2.9, 0.4, (0.81, 0.13, 0.06))]
SELECTIONS = [("1X2", "H"), ("1X2", "D"), ("DOUBLE_CHANCE", "X2"), ("TOTAL_1.5", "OVER"), ("TOTAL_2.5", "UNDER"),
              ("TOTAL_3.5", "UNDER"), ("BTTS", "YES"), ("BTTS", "NO"), ("HANDICAP_-0.5", "H"), ("HANDICAP_-1", "H"),
              ("HANDICAP_+1.5", "H"), ("HANDICAP_-1.5", "A"), ("HANDICAP_+2", "A"), ("EXACT_SCORE", "1-1")]


def _u(lh, la, p):
    return np.array([*np.log(p), np.log(lh), np.log(la)])


# ----------------------------------------------------------------------------- the outer function g
@pytest.mark.parametrize("lh,la,p", CASES)
@pytest.mark.parametrize("rho", [0.0, -0.08])
def test_vectorised_chain_equals_the_engine_markets(lh, la, p, rho):
    matrix = reconcile_with_1x2(score_matrix(lh, la, rho=rho), np.array(p))
    assert np.allclose(chain.score_matrices(_u(lh, la, p)[None], rho)[0], matrix, atol=1e-12)
    engine = {(s.market, s.selection): s.probability for s in goal_markets(matrix, top_scores=121)}
    for (market, selection), prob in engine.items():
        assert chain.market_probability(market, selection, _u(lh, la, p), rho)[0] == pytest.approx(prob, abs=1e-12)


def test_corner_probability_is_the_negative_binomial_tail():
    for mu, size in ((10.3, 71.0), (8.2, None)):
        over = prob_over(corners_distribution(mu, size), 9.5)
        assert chain.market_probability("CORNERS_9.5", "OVER", [[np.log(mu)]], nb_size=size)[0] == pytest.approx(over)
        assert chain.market_probability("CORNERS_9.5", "UNDER", [[np.log(mu)]], nb_size=size)[0] == pytest.approx(1 - over)


# ----------------------------------------------------------------------------- attribution
def _inputs(lh, la, p, base=(1.4, 1.1, (0.45, 0.27, 0.28)), n_features=6, seed=0, names=chain.GOAL_OUTPUTS):
    """Synthetic TreeSHAP: random per-feature values whose columns add up to u − ū (local accuracy)."""
    rng = np.random.default_rng(seed)
    u = _u(lh, la, p) if names == chain.GOAL_OUTPUTS else np.array([np.log(lh)])
    b = _u(*base) if names == chain.GOAL_OUTPUTS else np.array([np.log(base[0])])
    phi = rng.normal(0, 0.2, (n_features, len(names)))
    phi[-1] += (u - b) - phi.sum(0)
    feats = [f"f{i}" for i in range(n_features)]
    return chain.ChainInputs(tuple(names), u, b, pd.DataFrame(phi, index=feats, columns=list(names)),
                             pd.Series(np.arange(n_features, dtype=float), index=feats),
                             nb_size=71.0 if names == chain.CORNER_OUTPUTS else None)


@pytest.mark.parametrize("lh,la,p", CASES)
@pytest.mark.parametrize("market,selection", SELECTIONS)
def test_contributions_add_up_to_the_market_probability_change(lh, la, p, market, selection):
    inputs = _inputs(lh, la, p)
    ex = chain.attribute(inputs, market, selection)
    assert ex.probability == pytest.approx(chain.market_probability(market, selection, inputs.u)[0])
    assert ex.total == pytest.approx(ex.probability - ex.base_probability, abs=1e-7)      # completeness
    assert ex.output_effect.sum() == pytest.approx(ex.probability - ex.base_probability, abs=1e-7)


def test_corner_contributions_add_up_and_follow_the_expected_corners():
    inputs = _inputs(11.0, 0, None, base=(9.6,), names=chain.CORNER_OUTPUTS)
    over = chain.attribute(inputs, "CORNERS_9.5", "OVER")
    under = chain.attribute(inputs, "CORNERS_9.5", "UNDER")
    assert over.total == pytest.approx(over.probability - over.base_probability, abs=1e-6)
    assert over.total > 0 > under.total                                  # more corners expected than usual
    np.testing.assert_allclose(over.contributions.set_index("feature")["contribution"],
                               -under.contributions.set_index("feature")["contribution"], atol=1e-9)


def test_complementary_selections_get_opposite_contributions():
    inputs = _inputs(*CASES[0])
    for market, a, b in (("TOTAL_2.5", "OVER", "UNDER"), ("BTTS", "YES", "NO")):
        ca = chain.attribute(inputs, market, a).contributions.set_index("feature")["contribution"]
        cb = chain.attribute(inputs, market, b).contributions.set_index("feature")["contribution"]
        np.testing.assert_allclose(ca, -cb.reindex(ca.index), atol=1e-9)


def test_signs_follow_the_real_effect_on_the_market():
    """A feature that only raises the home goal rate helps Over and BTTS Yes and hurts Under / BTTS No;
    a feature that only raises the draw score lowers a total (draws are low-scoring)."""
    inputs = _inputs(*CASES[0], base=(1.4, 1.1, (0.45, 0.22, 0.33)))
    phi = pd.DataFrame(0.0, index=["home_attack", "draw_maker", "rest"], columns=list(chain.GOAL_OUTPUTS))
    du = inputs.u - inputs.base
    phi.loc["home_attack", "log_lambda_home"] = du[3]
    phi.loc["draw_maker", "z_D"] = du[1]
    phi.loc["rest"] = du - phi.sum(0)
    inputs.phi = phi
    c = lambda m, s: chain.attribute(inputs, m, s).contributions.set_index("feature")["contribution"]  # noqa: E731
    assert du[3] > 0 and du[1] > 0
    assert c("TOTAL_2.5", "OVER")["home_attack"] > 0 > c("TOTAL_2.5", "UNDER")["home_attack"]
    assert c("BTTS", "YES")["home_attack"] > 0 > c("BTTS", "NO")["home_attack"]
    assert c("TOTAL_2.5", "OVER")["draw_maker"] < 0 < c("TOTAL_2.5", "UNDER")["draw_maker"]
    assert c("1X2", "D")["draw_maker"] > 0


def test_models_that_do_not_drive_a_market_get_no_weight():
    inputs = _inputs(*CASES[1])
    for market, selection in (("1X2", "H"), ("DOUBLE_CHANCE", "1X"), ("HANDICAP_-0.5", "H"), ("HANDICAP_+0.5", "A")):
        ex = chain.attribute(inputs, market, selection)
        assert abs(ex.weights["log_lambda_home"]) < 1e-6 and abs(ex.weights["log_lambda_away"]) < 1e-6
        assert ex.contributions["via_log_lambda_home"].abs().max() < 1e-6
    # ...while a two-goal handicap needs both the result and the goal models
    ex = chain.attribute(inputs, "HANDICAP_-1.5", "A")
    assert abs(ex.weights["log_lambda_away"]) > 1e-3 and abs(ex.weights["z_A"]) > 1e-3


def test_the_exact_score_is_not_monotonic_in_the_expected_goals():
    """P(1-1) first rises and then falls with the home goal rate — why the card shows no shares for it."""
    lam = np.linspace(0.3, 3.5, 40)
    U = np.array([_u(lh, 1.2, (0.4, 0.3, 0.3)) for lh in lam])
    grad = chain.gradient("EXACT_SCORE", "1-1", U)[:, 3]
    assert grad.max() > 0 > grad.min()
    over = chain.gradient("TOTAL_2.5", "OVER", U)[:, 3]
    assert (over > 0).all()                                   # a total is monotonic
    assert not shares_are_meaningful("EXACT_SCORE")


# ----------------------------------------------------------------------------- the card
def test_card_factors_use_net_contributions_and_relative_influence():
    expl = pd.DataFrame({
        "feature": ["h_corners_for_form10", "a_corners_for_form10", "form_corners_sum", "league", "h_rest_days",
                    "a_rest_days", "elo_diff"],
        "contribution": [0.030, -0.010, 0.025, -0.005, -0.030, 0.0, 0.002],
        "value": [6.1, 5.0, 11.4, "E0", 3, 4, 120.0],
    })
    table = factor_table(expl)
    assert table["share"].sum() == pytest.approx(1.0)
    by = table.set_index("key")
    assert by.loc["corners_for_form10", "contribution"] == pytest.approx(0.020)          # net of the pair
    assert by.loc["corners_for_form10", "share"] == pytest.approx(0.020 / 0.082)
    f = card_factors(expl, "Stuttgart", "Gladbach")
    assert 3 <= len(f) <= 5
    # the factors FOR the selection by influence, then the strongest one against it (≥ 10 %)
    assert [x.key for x in f] == ["form_corners_sum", "corners_for_form10", "elo_diff", "rest_days"]
    assert [x.supports for x in f] == [True, True, True, False]
    assert f[1].evidence == "Stuttgart: 6.10 · Gladbach: 5.00"
    assert sum(x.share for x in f) <= 1.0 + 1e-9                                          # shares need not add to 100 %
    small = expl.assign(contribution=[0.030, -0.010, 0.025, -0.005, -0.003, 0.0, 0.002])
    assert all(x.supports for x in card_factors(small, "S", "G"))                        # a small "against" is not shown


def test_influence_tooltip_text():
    assert INFLUENCE_TIP == ("Only the most influential factors are shown. Percentages represent their share of "
                             "total model influence.")


def _snap(lh, la, p, mu=10.2, size=71.0):
    return {"p_home": p[0], "p_draw": p[1], "p_away": p[2], "xg_home": lh, "xg_away": la, "exp_corners": mu,
            "corners_nb_size": size}


@pytest.mark.parametrize("lh,la,p", CASES)
def test_chain_values_reproduce_the_stored_market_probabilities(lh, la, p):
    matrix = reconcile_with_1x2(score_matrix(lh, la), np.array(p))
    stored = {(s.market, s.selection): s for s in goal_markets(matrix)}
    for (market, selection), sel in stored.items():
        v = chain.chain_values(market, selection, _snap(lh, la, p))
        assert v.probability == pytest.approx(sel.probability, abs=1e-12)
        if market.startswith("HANDICAP_") and float(market.split("_")[1]).is_integer():
            assert v.push == pytest.approx(sel.push_probability, abs=1e-12)
    v = chain.chain_values("BTTS", "YES", _snap(lh, la, p))
    assert v.p_home_scores == pytest.approx(matrix[1:, :].sum()) and v.p_away_scores == pytest.approx(matrix[:, 1:].sum())
    top = chain.chain_values("EXACT_SCORE", "1-1", _snap(lh, la, p))
    assert len(top.alternatives) == 2 and "1-1" not in dict(top.alternatives)
    assert top.alternatives[0][1] >= top.alternatives[1][1]
    corners = chain.chain_values("CORNERS_9.5", "OVER", _snap(lh, la, p))
    assert corners.probability == pytest.approx(prob_over(corners_distribution(10.2, 71.0), 9.5))


def test_notes_read_like_football_and_follow_the_chain():
    lh, la, p = CASES[0]
    snap = _snap(lh, la, p)
    note = lambda m, s: chain_note(m, s, chain.chain_values(m, s, snap), "Como", "Roma", nb_size=71.0)  # noqa: E731
    assert note("TOTAL_2.5", "OVER") == ("Como 1.62 and Roma 1.10 expected goals: about 2.7 in total, just above "
                                         "the 2.5 line.")
    assert "under 1.5 needs a quieter game" in note("TOTAL_1.5", "UNDER")
    assert note("HANDICAP_-1", "H").startswith("Como -1 needs Como to win by 2 or more goals; a 1-goal win refunds")
    assert note("HANDICAP_-1.5", "A").startswith("Roma +1.5 wins unless Roma lose by 2 or more goals")
    assert note("HANDICAP_-1", "A").startswith("Roma +1 wins if Roma avoid defeat; a one-goal defeat refunds")
    assert note("HANDICAP_+2", "A").startswith("Roma -2 needs Roma to win by 3 or more goals")
    assert note("HANDICAP_-0.5", "H").startswith("Como -0.5 is simply a Como win")
    assert "Como a 7" in note("BTTS", "YES") or "Como a 8" in note("BTTS", "YES")
    assert "corners (usually between" in note("CORNERS_9.5", "OVER")
    score = note("EXACT_SCORE", "1-1")
    assert "1:1 (" in score and "close behind" in score
    for m, s in SELECTIONS + [("CORNERS_9.5", "UNDER")]:
        text = note(m, s)
        assert text and "nan" not in text.lower() and "reason" not in text.lower()


# ----------------------------------------------------------------------------- real models end to end
@pytest.fixture(scope="module")
def trained(synthetic):
    played = synthetic.loc[synthetic["played"]]
    out = {}
    for name, task in TASKS.items():
        b = TaskTrainer(task, synthetic.columns).fit(played, PARAMS, 30)
        if name == "corners":
            b.extra.update({"leagues": corner_leagues(synthetic), "nb_size": 40.0})
        out[name] = b
    return out


def test_explanations_of_real_models_match_the_engine(synthetic, trained):
    upcoming = synthetic.loc[~synthetic["played"]]
    batch = predict_matches(upcoming, trained, {}, "test", "live")
    markets = batch.markets.set_index(["match_id", "market", "selection"])["probability"]
    for _, snap in batch.snapshots.head(2).iterrows():
        row = upcoming.loc[upcoming["match_id"] == snap["match_id"]]
        for market, selection in SELECTIONS[:-1] + [("CORNERS_9.5", "OVER")]:
            ex = chain.explain_market(row, market, selection, trained, key=f"test|{snap['match_id']}")
            stored = float(markets[(snap["match_id"], market, selection)])
            assert ex.probability == pytest.approx(stored, abs=1e-5)                     # the stored probability
            assert ex.total == pytest.approx(ex.probability - ex.base_probability, abs=1e-5)
            values = chain.chain_values(market, selection, snap)
            assert values.probability == pytest.approx(stored, abs=1e-5)
            # every feature contribution is TreeSHAP of a model output times its path weight
            via = ex.contributions[[c for c in ex.contributions if c.startswith("via_")]].sum(axis=1)
            assert np.allclose(via, ex.contributions["contribution"])


def test_a_goal_total_mentions_the_1x2_forecast_only_when_it_drives_it():
    lh, la, p = CASES[0]
    inputs = _inputs(lh, la, p, base=(lh, la, (0.45, 0.22, 0.33)))      # only the 1X2 outputs differ from ū
    values = chain.chain_values("TOTAL_2.5", "UNDER", _snap(lh, la, p))
    ex = chain.attribute(inputs, "TOTAL_2.5", "UNDER")
    text = chain_note("TOTAL_2.5", "UNDER", values, "Como", "Roma", ex)
    assert "The 1X2 forecast (Como 48% · draw 27% · Roma 25%)" in text
    goals_only = chain.attribute(_inputs(lh, la, p, base=(1.2, 0.9, p)), "TOTAL_2.5", "UNDER")
    assert "1X2" not in chain_note("TOTAL_2.5", "UNDER", values, "Como", "Roma", goals_only)


def test_note_texts_exist_in_both_languages_with_the_same_placeholders():
    import re
    from src.explain.card import NOTE
    assert NOTE["en"].keys() == NOTE["uk"].keys()
    for key, text in NOTE["en"].items():
        assert set(re.findall(r"{\w+}", text)) == set(re.findall(r"{\w+}", NOTE["uk"][key])), key
