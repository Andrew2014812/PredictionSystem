"""E7: local sensitivity of the market probability (method C).

    .venv/bin/python experiments/v6_1/explain/sensitivity.py

1. ∂P/∂(model output) at the match, in readable units: per +0.1 expected goals of each team, per
   +1 expected corner, per +1 percentage point moved into each 1X2 class (taken evenly from the other two);
2. propagated to the features: does a small change of a feature move the model outputs at all? Tree
   ensembles are piecewise constant, so the local derivative ∂u/∂x is zero almost everywhere and a
   gradient chain rule cannot attribute anything; finite steps of growing size show how sensitivity
   only appears when a step crosses a split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Data, chain, log, model_outputs, panel, raw_columns, save_json, setup_logging  # noqa: E402
from run_research import EXAMPLES  # noqa: E402

from src.markets.markets import market_group  # noqa: E402


def output_sensitivity(market: str, selection: str, snap: pd.Series, nb: float | None) -> dict:
    if market_group(market) == "CORNERS":
        mu = float(snap["exp_corners"])
        f = lambda m: chain.market_probability(market, selection, [[np.log(m)]], nb_size=nb)[0]  # noqa: E731
        return {"P": f(mu), "per_corner_pp": (f(mu + 0.5) - f(mu - 0.5)) * 100}
    p = np.array([snap["p_home"], snap["p_draw"], snap["p_away"]], float)
    lh, la = float(snap["xg_home"]), float(snap["xg_away"])

    def g(p_, lh_, la_):
        return chain.market_probability(market, selection, [[*np.log(p_), np.log(lh_), np.log(la_)]])[0]

    out = {"P": g(p, lh, la), "per_0.1_home_goals_pp": (g(p, lh + 0.05, la) - g(p, lh - 0.05, la)) * 100,
           "per_0.1_away_goals_pp": (g(p, lh, la + 0.05) - g(p, lh, la - 0.05)) * 100}
    for i, name in enumerate(("home", "draw", "away")):
        step = np.full(3, -0.005)
        step[i] = 0.01
        out[f"per_1pp_{name}_pp"] = (g(p + step / 2, lh, la) - g(p - step / 2, lh, la)) * 100
    return out


def feature_steps(row: pd.DataFrame, bundles: dict, sd: pd.Series) -> dict:
    """Share of features whose step of size ε·σ changes any model output, for growing ε."""
    cols = [c for c in raw_columns(bundles) if c != "league" and c in sd.index and pd.notna(row[c].iloc[0])
            and sd[c] > 0]
    base = model_outputs(row, bundles)
    base_vec = np.concatenate([base["goal"][0], base["corner"][0]])
    out = {}
    for eps in (1e-4, 1e-3, 1e-2, 0.1, 0.5):
        frame = pd.concat([row] * len(cols), ignore_index=True)
        for i, c in enumerate(cols):
            frame.loc[i, c] = float(row[c].iloc[0]) + eps * float(sd[c])
        U = model_outputs(frame, bundles)
        moved = np.abs(np.column_stack([U["goal"], U["corner"]]) - base_vec[None, :]).max(1) > 1e-7
        out[str(eps)] = float(moved.mean())
    return {"features": len(cols), "share_moving": out}


def main() -> None:
    setup_logging("sensitivity")
    data = Data()
    out = {"examples": {}, "feature_steps": []}
    for mid in EXAMPLES:
        snap = data.snaps.loc[data.snaps["match_id"] == mid].iloc[0]
        nb = None if pd.isna(snap["corners_nb_size"]) else float(snap["corners_nb_size"])
        items = []
        for m, s, role in panel(snap, data.match_markets(mid)):
            items.append({"market": m, "selection": s, "role": role, **output_sensitivity(m, s, snap, nb)})
        out["examples"][mid] = {"home": snap["home_team"], "away": snap["away_team"], "xg_home": float(snap["xg_home"]),
                                "xg_away": float(snap["xg_away"]), "p": [float(snap["p_home"]), float(snap["p_draw"]),
                                                                       float(snap["p_away"])],
                                "exp_corners": float(snap["exp_corners"]), "items": items}
    recent = data.features.loc[data.features["played"].astype(bool)
                               & (data.features["date"] >= data.features["date"].max() - pd.Timedelta(days=730))]
    pool = data.pool()
    sm = pool.loc[pool["model_version"] == "prod_20261012"].sample(20, random_state=3)
    bundles = data.bundles["prod_20261012"]
    sd = recent[[c for c in raw_columns(bundles) if c != "league"]].std(numeric_only=True)
    for snap in sm.itertuples(index=False):
        out["feature_steps"].append({"match_id": snap.match_id, **feature_steps(data.row(snap.match_id), bundles, sd)})
    steps = pd.DataFrame([r["share_moving"] for r in out["feature_steps"]])
    out["feature_steps_mean"] = steps.mean().to_dict()
    save_json("e7_sensitivity.json", out)
    log.info("E7 %s", out["feature_steps_mean"])


if __name__ == "__main__":
    main()
