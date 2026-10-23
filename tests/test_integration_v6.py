"""Integration: history → features → models → prediction → real odds → settlement → profit / ROI →
analytics, and a price found later in the archive that completes a stored prediction."""
import numpy as np
import pandas as pd
import pytest

from src.analytics.performance import group_summary, summarize, unified_history
from src.modelling.tasks import TASKS
from src.modelling.training import TaskTrainer, corner_leagues
from src.prediction import history_repair, store
from src.prediction.engine import predict_matches
from src.storage import LocalStorage
from tests.test_prediction import PARAMS, synthetic  # noqa: F401  (module fixture)


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(store, "predictions_storage", storage)
    monkeypatch.setattr(history_repair, "predictions_storage", storage)
    return storage


def _bundles(frame):
    played = frame.loc[frame["played"]]
    out = {}
    for name, task in TASKS.items():
        b = TaskTrainer(task, frame.columns).fit(played, PARAMS, 30)
        if name == "corners":
            b.extra.update({"leagues": corner_leagues(frame), "nb_size": None})
        out[name] = b
    return out


def test_prediction_to_roi_and_a_later_archive_price(synthetic, tmp_store):  # noqa: F811
    upcoming = synthetic.loc[~synthetic["played"]]
    bundles = _bundles(synthetic)
    priced_match, unpriced_match = upcoming["match_id"].iloc[0], upcoming["match_id"].iloc[1]
    first = predict_matches(upcoming, bundles, {}, "v1", "live")
    main0 = first.picks.loc[(first.picks["role"] == "main") & (first.picks["match_id"] == priced_match)].iloc[0]
    odds = {priced_match: {(main0["market"], main0["selection"]): {"odds": 1.9, "bookmaker": "Bet365",
                                                                   "provider": "oddspapi"}}}
    batch = predict_matches(upcoming, bundles, odds, "v1", "live")
    store.save_batch(batch, now=pd.Timestamp("2000-01-01"))
    stored = store.load_picks()
    assert stored.groupby("match_id")["role"].apply(lambda r: (r == "risk").sum()).eq(1).all()   # one Risk each

    # the matches are played: home 2 - 1 away everywhere
    results = upcoming.assign(played=True, home_goals=2.0, away_goals=1.0, total_corners=10.0)
    picks = store.settle(results)
    main = picks.loc[(picks["role"] == "main") & (picks["match_id"] == priced_match)].iloc[0]
    assert main["status"] in ("WON", "LOST", "PUSH")
    assert main["profit"] == pytest.approx(0.9 if main["status"] == "WON" else (-1.0 if main["status"] == "LOST" else 0))
    other = picks.loc[(picks["role"] == "main") & (picks["match_id"] == unpriced_match)].iloc[0]
    assert np.isnan(other["odds"]) and np.isnan(other["profit"])          # no price → no profit, no ROI
    s = summarize(unified_history(picks).query("role == 'main'"))
    assert s["bets"] == 1 and s["settled"] >= 2
    by_league = group_summary(unified_history(picks).query("role == 'main'"), "league")
    assert int(by_league["bets"].sum()) == 1                             # analytics agrees with the history

    # later, the archive has a real pre-kick-off price for the unpriced Main
    archive = pd.DataFrame([{"match_id": unpriced_match, "market": other["market"], "selection": other["selection"],
                             "odds": 2.5, "bookmaker": "pinnacle", "provider": "oddspapi",
                             "collected_at": pd.Timestamp("2000-01-01")}])
    filled, n = history_repair.fill_archive_prices(picks, archive, pd.Series(dtype=float))
    after = filled.loc[(filled["role"] == "main") & (filled["match_id"] == unpriced_match)].iloc[0]
    assert n == 1 and after["odds"] == 2.5 and after["odds_origin"] == "archive"
    assert (after["market"], after["selection"], after["probability"], after["model_version"]) == (
        other["market"], other["selection"], other["probability"], other["model_version"])   # pick unchanged
    expected = 1.5 if after["status"] == "WON" else (-1.0 if after["status"] == "LOST" else 0.0)
    assert after["profit"] == pytest.approx(expected)
    s2 = summarize(unified_history(filled).query("role == 'main'"))
    assert s2["bets"] == 2 and s2["profit"] == pytest.approx(s["profit"] + expected)   # ROI recomputed
