"""Integration (v6.2): recorded predictions and their model inputs survive the regeneration of the
retrospective (backfill) predictions; a price completed later is labelled as published after the prediction
and is left out of the prediction-time ROI; the explanation of a recorded prediction keeps its inputs."""
import pandas as pd
import pytest

from src.analytics.performance import summarize, unified_history
from src.prediction import history_repair, inputs as inputs_mod, store
from src.prediction.engine import predict_matches
from src.prediction.inputs import load_inputs, stored_inputs
from src.prediction.price_timing import AFTER, AT_GENERATION, decision_time_prices, label_prices
from src.storage import LocalStorage
from tests import test_prediction
from tests.test_integration_v6 import _bundles

synthetic = test_prediction.synthetic


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    for mod in (store, history_repair, inputs_mod):
        monkeypatch.setattr(mod, "predictions_storage", storage)
    return storage


def test_recorded_predictions_survive_a_backfill_regeneration(synthetic, tmp_store):
    upcoming = synthetic.loc[~synthetic["played"]]
    played = synthetic.loc[synthetic["played"]].tail(6)
    bundles = _bundles(synthetic)
    live = predict_matches(upcoming, bundles, {}, "v1", "live")
    store.save_batch(live, now=pd.Timestamp("2000-01-01"))
    recorded = store.load_picks().sort_values("pick_id").reset_index(drop=True)
    recorded_inputs = load_inputs().sort_values("match_id").reset_index(drop=True)

    for _ in range(2):                                    # regenerated twice, as a history run does
        backfill = predict_matches(played, bundles, {}, "v1", "backfill")
        store.save_batch(backfill, replace_source="backfill", now=pd.Timestamp("2100-01-01"))
    picks = store.load_picks()
    live_now = picks.loc[picks["source"] == "live"].sort_values("pick_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(live_now[recorded.columns], recorded, check_dtype=False)
    assert (picks["source"] == "backfill").any()
    saved = load_inputs()
    pd.testing.assert_frame_equal(saved.loc[saved["source"] == "live"].sort_values("match_id").reset_index(drop=True),
                                  recorded_inputs, check_dtype=False)
    assert saved.loc[saved["source"] == "backfill", "match_id"].nunique() == len(played)   # no duplicates


def test_a_price_found_later_is_not_a_prediction_time_price(synthetic, tmp_store):
    upcoming = synthetic.loc[~synthetic["played"]]
    bundles = _bundles(synthetic)
    batch = predict_matches(upcoming, bundles, {}, "v1", "live")
    m0 = batch.picks.loc[batch.picks["role"] == "main"].iloc[0]
    priced = predict_matches(upcoming, bundles, {m0["match_id"]: {(m0["market"], m0["selection"]): {
        "odds": 1.9, "bookmaker": "Bet365", "provider": "oddspapi"}}}, "v1", "live")
    store.save_batch(priced, now=pd.Timestamp("2000-01-01"))
    picks = store.settle(upcoming.assign(played=True, home_goals=2.0, away_goals=1.0, total_corners=10.0))
    other = picks.loc[(picks["role"] == "main") & (picks["match_id"] != m0["match_id"])].iloc[0]
    late = pd.Timestamp(other["generated_at"]) + pd.Timedelta(hours=30)      # published after the prediction (UTC)
    archive = pd.DataFrame([{"match_id": other["match_id"], "market": other["market"], "selection": other["selection"],
                             "odds": 2.4, "bookmaker": "pinnacle", "provider": "oddspapi", "collected_at": late}])
    filled, n = history_repair.fill_archive_prices(picks, archive, pd.Series(dtype=float))
    labelled, counts = label_prices(filled, pd.DataFrame())
    main = labelled.loc[labelled["role"] == "main"].set_index("match_id")["odds_timing"]
    assert n == 1 and main[m0["match_id"]] == AT_GENERATION and main[other["match_id"]] == AFTER
    history = unified_history(labelled).query("role == 'main'")
    assert summarize(history)["bets"] == 2                          # both shown and settled
    assert summarize(decision_time_prices(history))["bets"] == 1    # only one could have been taken
    # the explanation inputs of the recorded prediction are still those it was made from
    snap = priced.snapshots.iloc[0]
    assert stored_inputs(snap["match_id"], snap["generated_at"]) is not None
