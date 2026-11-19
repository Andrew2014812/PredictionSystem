"""v6.3 forward shadows: the frozen Main-rule variants (src/research/variants.py) are registered against unchanged
preregistrations, choose as preregistered and never leave the production configuration changed."""
import numpy as np
import pandas as pd

from src import config
from src.research import evaluate as E
from src.research import registry as R
from src.research import variants as V


def _ctx(rows):
    markets = pd.DataFrame(rows, columns=["match_id", "market", "selection", "probability", "odds", "bookmaker",
                                          "provider", "push_probability", "priced_at"])
    main = markets.sort_values("probability", ascending=False).iloc[0]
    picks = pd.DataFrame([{"match_id": "m", "role": "main", "market": main["market"], "selection": main["selection"],
                           "probability": main["probability"], "odds": main["odds"], "group": "TOTAL"}])
    empty = pd.DataFrame()
    return V.Context(now=pd.Timestamp("2026-11-20"), snapshots=empty, markets=markets, picks=picks, history=empty,
                     history_push=empty)


def test_every_v63_shadow_is_frozen_and_wired():
    v63 = [s for s in R.SHADOWS if s.id.startswith("v63")]
    assert len(v63) == 8 and sum(s.implemented for s in v63) == 4
    for s in v63:
        assert s.check() is None, s.key                          # the preregistration file is unchanged
        assert s.forward_rule["utc"] >= "2026-11-18"
        if s.implemented:
            assert s.id in V.COMPUTE and s.id in E.EVALUATORS
        else:
            assert s.not_implemented_reason


def test_reliability_variants_restore_the_production_configuration():
    before = dict(config.MARKET_RELIABILITY)
    ctx = _ctx([("m", "TOTAL_2.5", "UNDER", 0.62, 1.55, "bet365", "oddspapi", np.nan, pd.NaT),
                ("m", "HANDICAP_+0.5", "H", 0.64, 1.60, "bet365", "oddspapi", np.nan, pd.NaT)])
    from src.markets.recommendation import main_prediction
    assert main_prediction(V.selections(ctx.markets)).pick.market == "TOTAL_2.5"   # production: the total
    total = V.main_reliability_total(ctx, {})
    handicap = V.main_reliability_handicap(ctx, {})
    assert config.MARKET_RELIABILITY == before
    assert total["market"].iloc[0] == "HANDICAP_+0.5"              # TOTAL 0.95 tips the choice to the handicap
    assert handicap["market"].iloc[0] == "HANDICAP_+0.5"


def test_the_odds_floor_falls_back_to_the_statistical_main_without_a_bet():
    ctx = _ctx([("m", "TOTAL_3.5", "UNDER", 0.80, 1.20, "bet365", "oddspapi", np.nan, pd.NaT),
                ("m", "TOTAL_2.5", "UNDER", 0.55, 1.35, "bet365", "oddspapi", np.nan, pd.NaT)])
    row = V.main_floor_140(ctx, {}).iloc[0]
    assert np.isnan(row["odds"]) and row["payload"]["fallback"] and not row["payload"]["counted_as_bet"]
    ctx = _ctx([("m", "TOTAL_3.5", "UNDER", 0.80, 1.20, "bet365", "oddspapi", np.nan, pd.NaT),
                ("m", "TOTAL_2.5", "UNDER", 0.55, 1.75, "bet365", "oddspapi", np.nan, pd.NaT)])
    row = V.main_floor_140(ctx, {}).iloc[0]
    assert (row["market"], row["odds"]) == ("TOTAL_2.5", 1.75) and not row["payload"]["fallback"]
