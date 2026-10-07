"""Small end-to-end model test on synthetic data (fast: few trees, no Hyperopt)."""
import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features
from src.modelling.tasks import TASKS
from src.modelling.training import TaskTrainer, corner_leagues
from src.prediction.engine import predict_matches
from tests.conftest import raw_row
from src.data.cleaning import normalise

PARAMS = {"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 1.0, "subsample": 1.0,
          "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0, "half_life_years": 0}


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    rows, day = [], pd.Timestamp("2023-08-05")
    for week in range(60):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            rows.append(raw_row("E0", day.strftime("%d/%m/%Y"), h, a, rng.poisson(1.5), rng.poisson(1.1),
                                HC=rng.poisson(5), AC=rng.poisson(4)))
        day += pd.Timedelta(days=7)
    order = rng.permutation(teams)
    for h, a in zip(order[::2], order[1::2]):           # upcoming round
        rows.append(raw_row("E0", day.strftime("%d/%m/%Y"), h, a))
    return build_features(normalise(pd.DataFrame(rows)))


def test_prediction_output(synthetic):
    played = synthetic.loc[synthetic["played"]]
    bundles = {}
    for name, task in TASKS.items():
        bundle = TaskTrainer(task, synthetic.columns).fit(played, PARAMS, 30)
        if name == "corners":
            bundle.extra.update({"leagues": corner_leagues(synthetic), "nb_size": None})
        bundles[name] = bundle
    upcoming = synthetic.loc[~synthetic["played"]]
    first = upcoming["match_id"].iloc[0]
    odds = {first: {("1X2", "H"): {"odds": 2.1, "bookmaker": "Bet365", "provider": "football_data"}}}
    batch = predict_matches(upcoming, bundles, odds, "test", "live")

    snaps = batch.snapshots
    assert len(snaps) == len(upcoming)
    total = snaps[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-5)
    assert (snaps["xg_home"] > 0).all() and (snaps["exp_corners"] > 0).all()

    m = batch.markets
    for mid, g in m.groupby("match_id"):
        p = g.set_index(["market", "selection"])["probability"]
        assert p.loc["1X2"].sum() == pytest.approx(1.0, abs=1e-5)
        assert p.loc["TOTAL_2.5"].sum() == pytest.approx(1.0, abs=1e-5)
        assert p.loc["BTTS"].sum() == pytest.approx(1.0, abs=1e-5)
        assert (g["probability"].between(0, 1)).all()
        priced = g.loc[g["odds"].notna()]
        if mid == first:
            assert list(zip(priced["market"], priced["selection"])) == [("1X2", "H")]
        else:
            assert priced.empty                        # no real price -> no odds, nothing simulated
    picks = batch.picks
    assert set(picks["role"]) >= {"market", "main"}
    assert picks.groupby("match_id")["role"].apply(lambda r: (r == "main").sum()).eq(1).all()
    assert (picks["status"] == "UPCOMING").all()
    assert not picks.loc[picks["match_id"] != first, "real_odds"].any()


def test_prediction_matrix_columns_match_training(synthetic):
    """Unknown league at prediction time must not change the matrix columns."""
    played = synthetic.loc[synthetic["played"]]
    bundle = TaskTrainer(TASKS["result"], synthetic.columns).fit(played, PARAMS, 10)
    other = synthetic.loc[~synthetic["played"]].assign(league="XX")
    X = bundle.matrix(other.drop(columns=["h_form5_gf"]))
    assert list(X.columns) == bundle.feature_names
    assert (X.filter(like="league=").to_numpy() == 0).all()
    assert X["h_form5_gf"].isna().all()
