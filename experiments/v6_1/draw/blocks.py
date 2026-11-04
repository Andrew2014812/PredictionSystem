"""Step 1 — production-faithful walk-forward blocks.

For every block season b (2022…2026) the four production models are refitted on the played seasons < b
with the tuned parameters and tree counts of the active model and season b is predicted through the
production engine (src/prediction/engine.predict_matches) with the real pre-match prices that FootPredict
knows for those matches (Football-Data Bet365; for 2026/27 also the API prices collected before kick-off).

Per block the cache keeps
    pred_<b>.parquet     one row per match: ids, result, pH/pD/pA, lambda_home/away, Bet365 1X2 odds,
                         production Main / Risk selection
    markets_<b>.parquet  every priced selection of every match (probability, odds, push probability)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.modelling.tasks import TASKS
from src.modelling.training import TaskTrainer, corner_leagues, negative_binomial_size
from src.prediction.engine import predict_matches
from src.providers import real_odds_lookup

from . import common as C

log = C.setup_logging("draw61.blocks")
ID_COLS = ["match_id", "league", "season", "date", "home_team", "away_team", "home_goals", "away_goals",
           "total_corners", "result", "odds_home", "odds_draw", "odds_away"]


def _bundles(train: pd.DataFrame, meta: dict) -> tuple[dict, dict]:
    bundles, secs = {}, {}
    cl = corner_leagues(train)
    for name, task in TASKS.items():
        rows = train.loc[train["league"].isin(cl)] if name == "corners" else train
        info = meta["tasks"][name]
        t0 = time.time()
        b = TaskTrainer(task, C.features().columns).fit(rows, info["params"], info["n_estimators"])
        secs[name] = time.time() - t0
        if name == "corners":
            fit = rows.loc[task.trainable(rows)]
            b.extra.update({"leagues": cl, "nb_size": negative_binomial_size(fit[task.target].to_numpy(float),
                                                                             b.predict(fit))})
        if name == "result":
            b.extra["draw_multiplier"] = 1.0
        bundles[name] = b
    return bundles, secs


def block(b: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    f_pred, f_mk, f_info = C.CACHE / f"pred_{b}.parquet", C.CACHE / f"markets_{b}.parquet", C.CACHE / f"block_{b}.json"
    if not refresh and f_pred.exists() and f_mk.exists():
        return pd.read_parquet(f_pred), pd.read_parquet(f_mk)
    t0 = time.time()
    df = C.features()
    train, ev = df.loc[df["season"] < b], df.loc[df["season"] == b].reset_index(drop=True)
    bundles, secs = _bundles(train, C.prod_meta())
    t1 = time.time()
    batch = predict_matches(ev, bundles, real_odds_lookup(ev), f"draw61_block_{b}", "study")
    snap = batch.snapshots.set_index("match_id")
    pred = ev[ID_COLS].copy()
    pred["y"] = C.y_index(pred["result"])
    pred["pH"], pred["pD"], pred["pA"] = (snap.loc[pred["match_id"], c].to_numpy() for c in ("p_home", "p_draw", "p_away"))
    pred["lh"], pred["la"] = snap.loc[pred["match_id"], "xg_home"].to_numpy(), snap.loc[pred["match_id"], "xg_away"].to_numpy()
    picks = batch.picks
    for role in ("main", "risk"):
        r = picks.loc[picks["role"] == role].set_index("match_id")
        for col in ("market", "selection", "probability", "odds", "ev"):
            pred[f"{role}_{col}"] = pred["match_id"].map(r[col])
    mk = batch.markets[["match_id", "market", "selection", "group", "probability", "odds", "ev",
                        "push_probability", "bookmaker", "provider"]].copy()
    pred.to_parquet(f_pred)
    mk.to_parquet(f_mk)
    info = {"block": b, "train_seasons": sorted(int(s) for s in train["season"].unique()), "train_rows": int(len(train)),
            "rows": int(len(ev)), "fit_seconds": secs, "engine_seconds": time.time() - t1,
            "total_seconds": time.time() - t0}
    f_info.write_text(json.dumps(info))
    log.info("block %s: %s", b, info)
    return pred, mk


def block_info(b: int) -> dict:
    path = C.CACHE / f"block_{b}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def all_blocks(refresh: bool = False) -> dict[int, pd.DataFrame]:
    return {b: block(b, refresh)[0] for b in C.BLOCKS}


def check_against_backtest() -> dict:
    """Block 2025 must reproduce the stored backtest predictions of the website (same model setup)."""
    from src.prediction import store
    snap = store.load_snapshots()
    bt = snap.loc[snap["source"] == "backtest"].set_index("match_id")
    pred = block(2025)[0].set_index("match_id")
    common = pred.index.intersection(bt.index)
    diff = np.abs(pred.loc[common, ["pH", "pD", "pA"]].to_numpy()
                  - bt.loc[common, ["p_home", "p_draw", "p_away"]].to_numpy().astype(float))
    same_main = (pred.loc[common, "main_market"].astype(str) + pred.loc[common, "main_selection"].astype(str)
                 == bt.loc[common, "main_market"].astype(str) + bt.loc[common, "main_selection"].astype(str))
    return {"matches": int(len(common)), "max_abs_prob_diff": float(diff.max()), "mean_abs_prob_diff": float(diff.mean()),
            "same_main_share": float(same_main.mean())}


if __name__ == "__main__":
    all_blocks()
    print(check_against_backtest())
