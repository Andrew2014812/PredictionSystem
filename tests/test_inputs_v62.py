"""A stored prediction is explained with the inputs it was made from, even after the data changed."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.modelling.registry import active_version, load_bundles
from src.prediction import store
from src.prediction.engine import predict_matches
from src.prediction.inputs import explanation_row, load_inputs, stored_inputs


@pytest.fixture(scope="module")
def bundles():
    b = load_bundles(active_version())
    if not b:
        pytest.skip("no trained models")
    return b


@pytest.fixture(scope="module")
def matches():
    f = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    played = f.loc[f["played"].astype(bool) & (f["season"] == f["season"].max())]
    return played.tail(6).reset_index(drop=True)


def test_inputs_are_stored_and_reproduce_the_stored_probabilities(tmp_path, monkeypatch, bundles, matches):
    from src.prediction import inputs as inputs_mod
    from src.storage import LocalStorage
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(store, "predictions_storage", storage)
    monkeypatch.setattr(inputs_mod, "predictions_storage", storage)
    batch = predict_matches(matches, bundles, {}, active_version(), "live")
    store.save_batch(batch, now=pd.Timestamp("2000-01-01"))
    saved = load_inputs()
    assert len(saved) == len(matches) and saved["feature_set"].nunique() == 1
    snap = batch.snapshots.iloc[0]
    row = stored_inputs(snap["match_id"], snap["generated_at"], saved)
    p = bundles["result"].predict(pd.DataFrame([row]))[0]
    assert p == pytest.approx([snap["p_home"], snap["p_draw"], snap["p_away"]], abs=1e-6)
    assert bundles["home_goals"].predict(pd.DataFrame([row]))[0] == pytest.approx(snap["xg_home"], abs=1e-5)


def test_the_explanation_does_not_change_when_the_data_is_updated(tmp_path, monkeypatch, bundles, matches):
    from src.explain.chain import explain_market
    from src.prediction import inputs as inputs_mod
    from src.storage import LocalStorage
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(store, "predictions_storage", storage)
    monkeypatch.setattr(inputs_mod, "predictions_storage", storage)
    batch = predict_matches(matches, bundles, {}, active_version(), "live")
    store.save_batch(batch, now=pd.Timestamp("2000-01-01"))
    snap = batch.snapshots.iloc[0]
    current = matches.iloc[0].copy()
    at_generation = explain_market(pd.DataFrame([current]), "TOTAL_2.5", "OVER", bundles, key="a")

    later = current.copy()                                  # a month later the table holds other values
    for col in ("h_elo", "a_elo", "elo_diff", "diff_form5_ppg", "diff_form10_gd", "h_attack_vs_a_defence"):
        if col in later:
            later[col] = float(later[col]) + 0.7 if pd.notna(later[col]) else 1.0
    drifted = explain_market(pd.DataFrame([later]), "TOTAL_2.5", "OVER", bundles, key="b")
    row, reproduced = explanation_row(later, stored_inputs(snap["match_id"], snap["generated_at"], load_inputs()))
    again = explain_market(pd.DataFrame([row]), "TOTAL_2.5", "OVER", bundles, key="c")
    assert reproduced
    assert again.probability == pytest.approx(at_generation.probability, abs=1e-6)
    a = again.contributions.set_index("feature")["contribution"]
    g = at_generation.contributions.set_index("feature")["contribution"]
    assert np.allclose(a.reindex(g.index).to_numpy(), g.to_numpy(), atol=1e-6)
    assert drifted.probability != pytest.approx(at_generation.probability, abs=1e-6)   # what v6.1 showed


def test_older_predictions_say_they_are_explained_with_current_data():
    current = pd.Series({"match_id": "m", "h_elo": 1500.0})
    row, reproduced = explanation_row(current, None)
    assert row is current and not reproduced


def test_frozen_predictions_keep_their_inputs(tmp_path, monkeypatch, bundles, matches):
    """After kick-off a recorded prediction is frozen — and so are its inputs."""
    from src.prediction import inputs as inputs_mod
    from src.storage import LocalStorage
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(store, "predictions_storage", storage)
    monkeypatch.setattr(inputs_mod, "predictions_storage", storage)
    first = predict_matches(matches.head(2), bundles, {}, active_version(), "live")
    store.save_batch(first, now=pd.Timestamp("2000-01-01"))
    before = load_inputs().set_index("match_id")["generated_at"].to_dict()
    second = predict_matches(matches.head(2), bundles, {}, active_version(), "live")
    second.inputs["generated_at"] = second.inputs["generated_at"] + pd.Timedelta(hours=1)
    store.save_batch(second, now=pd.Timestamp("2100-01-01"))          # long after kick-off: locked
    assert load_inputs().set_index("match_id")["generated_at"].to_dict() == before
