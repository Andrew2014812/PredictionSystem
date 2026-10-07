import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.modelling.decision import choose_draw_multiplier, decide
from src.modelling.evaluation import classifier_metrics


def _sample(n=3000, seed=1):
    """Draws are never the arg-max, but are often close (as in real football)."""
    rng = np.random.default_rng(seed)
    p_home = rng.uniform(0.30, 0.60, n)
    p_draw = rng.uniform(0.24, 0.31, n)
    p_away = 1 - p_home - p_draw
    proba = np.column_stack([p_home, p_draw, p_away])
    y = np.array([rng.choice(3, p=p) for p in proba])
    return proba, y


def test_plain_argmax_rarely_predicts_draws():
    proba, _ = _sample()
    assert np.mean(decide(proba, 1.0) == 1) < 0.05


def test_draw_rule_keeps_probabilities_and_log_loss():
    proba, y = _sample()
    choice = choose_draw_multiplier(proba, y, max_draw_share=0.27)
    before = classifier_metrics(y, proba)
    after = classifier_metrics(y, proba, decide(proba, choice["multiplier"]))
    assert after["log_loss"] == pytest.approx(before["log_loss"]) == pytest.approx(log_loss(y, proba))
    assert after["brier"] == pytest.approx(before["brier"])
    assert after["recall"]["D"] > before["recall"]["D"]
    assert after["macro_f1"] >= before["macro_f1"]


def test_draw_share_cap_is_respected():
    proba, y = _sample()
    cap = 0.20
    choice = choose_draw_multiplier(proba, y, max_draw_share=cap)
    assert np.mean(decide(proba, choice["multiplier"]) == 1) <= cap
    assert all(g["draw_share"] <= cap for g in choice["grid"] if g["multiplier"] == choice["multiplier"])


def test_multiplier_one_is_plain_argmax():
    proba, _ = _sample(200)
    assert (decide(proba, 1.0) == proba.argmax(1)).all()
