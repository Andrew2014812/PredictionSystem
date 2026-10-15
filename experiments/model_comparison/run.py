"""Run the model comparison (research only; never touches production models or data).

    python -m experiments.model_comparison.run                 # all folds (several hours)
    python -m experiments.model_comparison.run --quick         # tiny budgets, smoke test
    python -m experiments.model_comparison.run --folds 2025

Results: experiments/model_comparison/results/fold_<season>.json and predictions
fold_<season>.parquet; then ``python -m experiments.model_comparison.report``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.tasks import TASKS
from src.modelling.training import negative_binomial_size

from . import data as D
from . import metrics as M
from .models import (CatBoostCandidate, Ensemble, LightGBMCandidate, LogisticCandidate, NegBinGLMCandidate,
                     PoissonGLMCandidate, XGBoostCandidate, _score)

log = logging.getLogger("comparison")
RESULTS = Path(__file__).with_name("results")
SYSTEMS = ("XGBoost", "Improved XGBoost", "CatBoost", "LightGBM", "GLM", "Ensemble")
DIRECT = {"btts": None, "over15": 1.5, "over25": 2.5, "over35": 3.5, "corners_over95": 9.5}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tot = df["home_goals"] + df["away_goals"]
    known = df["home_goals"].notna()
    df["btts"] = np.where(known, ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(float), np.nan)
    for name, line in (("over15", 1.5), ("over25", 2.5), ("over35", 3.5)):
        df[name] = np.where(known, (tot > line).astype(float), np.nan)
    df["corners_over95"] = np.where(df["total_corners"].notna(), (df["total_corners"] > 9.5).astype(float), np.nan)
    return df


def candidates(task: str, kind: str, numeric: list[str], quick: bool) -> list:
    ev, ev_big = (2, 3) if quick else (8, 20)        # XGBoost trials vs 4 settings of CatBoost / LightGBM
    if kind == "binary":
        return [LightGBMCandidate(kind, numeric), LogisticCandidate(kind, numeric)]
    out = [XGBoostCandidate(kind, numeric, task, ev), XGBoostCandidate(kind, numeric, task, ev_big, "Improved XGBoost"),
           CatBoostCandidate(kind, numeric), LightGBMCandidate(kind, numeric)]
    out.append(LogisticCandidate(kind, numeric) if kind == "classifier" else PoissonGLMCandidate(kind, numeric))
    if task == "corners":
        out.append(NegBinGLMCandidate(kind, numeric))
    if quick:
        for c in out:
            if hasattr(c, "grid") and isinstance(c.grid, dict):
                c.grid = {k: v[:1] for k, v in c.grid.items()}
            elif hasattr(c, "grid"):
                c.grid = c.grid[:1]
    return out


def run_task(fold: D.Fold, task: str, kind: str, numeric: list[str], target: str, leagues, quick: bool):
    tr, va = D.task_rows(target, fold.tune_train, leagues), D.task_rows(target, fold.tune_valid, leagues)
    fit_rows, test = D.task_rows(target, fold.fit, leagues), D.task_rows(target, fold.test, leagues)
    y = (lambda d: TASKS[task].target_values(d).to_numpy()) if task in TASKS else (lambda d: d[target].to_numpy(float))
    y_tr, y_va, y_fit, y_te = y(tr), y(va), y(fit_rows), y(test)
    done, preds, info, fit_preds = [], {}, {}, {}
    for cand in candidates(task, kind, numeric, quick):
        started = time.time()
        try:
            cand.tune(tr, va, y_tr, y_va)
            cand.fit(fit_rows, y_fit)
            preds[cand.name] = cand.predict(test)
            if kind == "count":
                fit_preds[cand.name] = (y_fit, cand.predict(fit_rows))
        except Exception as exc:                       # an algorithm failing must not stop the study
            log.warning("%s %s failed: %s", task, cand.name, exc)
            continue
        info[cand.name] = {"params": cand.params, "validation_score": _score(kind, y_va, cand.valid_pred),
                           "seconds": round(time.time() - started, 1), "family": cand.family}
        done.append(cand)
        log.info("fold %s %s %-30s valid %.5f (%.0fs)", fold.label, task, cand.name,
                 info[cand.name]["validation_score"], info[cand.name]["seconds"])
    ranked = sorted(done, key=lambda c: info[c.name]["validation_score"])
    if len(ranked) >= 2:
        ens = Ensemble(kind, ranked[:2], y_va)
        preds[ens.name] = ens.predict(test)
        info[ens.name] = {"params": ens.params, "validation_score": _score(kind, y_va, ens.valid_pred),
                          "family": "Ensemble"}
        if kind == "count":
            fit_preds[ens.name] = (y_fit, ens.predict(fit_rows))
    run_task.fit_preds = fit_preds
    return test, y_te, preds, info


def evaluate(kind: str, y, preds: dict, base: float | None = None) -> dict:
    if kind == "classifier":
        return {n: M.classification(y.astype(int), p) for n, p in preds.items()}
    if kind == "count":
        return {n: M.counts(y, p) for n, p in preds.items()}
    return {n: M.binary(y, p, base) for n, p in preds.items()}


def system_member(info: dict, system: str) -> str | None:
    """Candidate name of a 'system' (algorithm family) inside one task."""
    for name, v in info.items():
        if system == "Ensemble" and v["family"] == "Ensemble":
            return name
        if system == "Improved XGBoost" and name == "Improved XGBoost":
            return name
        if system == "XGBoost" and name == "XGBoost":
            return name
        if system in ("CatBoost", "LightGBM") and name == system:
            return name
        if system == "GLM" and v["family"] == "GLM" and name != "Negative binomial regression":
            return name
    return None


def roi(test: pd.DataFrame, probs: np.ndarray, odds_cols, outcome: np.ndarray, value_only: bool = True) -> dict:
    """Unit stake on the selection with the highest EV (EV > 0 when value_only), real Bet365 prices only."""
    odds = test[list(odds_cols)].to_numpy(float)
    ok = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    ev = probs * odds - 1
    pick = ev.argmax(1)
    bet = ok & ((ev.max(1) > 0) if value_only else True)
    won = pick == outcome
    price = odds[np.arange(len(odds)), pick]
    profit = np.where(won, price - 1, -1.0)[bet]
    return {"bets": int(bet.sum()), "hit_rate": float(won[bet].mean()) if bet.any() else None,
            "profit": float(profit.sum()), "roi": float(profit.mean()) if bet.any() else None,
            "avg_odds": float(price[bet].mean()) if bet.any() else None}


def run_fold(features: pd.DataFrame, fold: D.Fold, quick: bool) -> dict:
    cl = D.corner_scope(features)
    cols = list(features.columns)
    res = {"season": fold.label, "rows": {"tune_train": len(fold.tune_train), "tune_valid": len(fold.tune_valid),
                                          "fit": len(fold.fit), "test": len(fold.test)}}
    out_preds = {}
    task_out = {}
    for task, kind in (("result", "classifier"), ("home_goals", "count"), ("away_goals", "count"),
                       ("corners", "count")):
        numeric = D.numeric_columns(task, cols)
        target = TASKS[task].target
        test, y, preds, info = run_task(fold, task, kind, numeric, target, cl if task == "corners" else None, quick)
        task_out[task] = (test, y, preds, info)
        if task == "corners":
            corner_fit_preds = dict(run_task.fit_preds)
        res[task] = {"n_features": len(numeric), "candidates": info, "metrics": evaluate(kind, y, preds)}
        for n, p in preds.items():
            arr = np.asarray(p)
            if arr.ndim == 2:
                for i, c in enumerate(M.CLASSES):
                    out_preds[(task, n, c)] = pd.Series(arr[:, i], index=test["match_id"].to_numpy())
            else:
                out_preds[(task, n, "")] = pd.Series(arr, index=test["match_id"].to_numpy())

    # markets derived from each system's score matrix (1X2 + both goal models of the same family)
    rt, ry, rp, ri = task_out["result"]
    _, _, hp, hi = task_out["home_goals"]
    _, _, ap, ai = task_out["away_goals"]
    hg, ag = rt["home_goals"].to_numpy(float), rt["away_goals"].to_numpy(float)
    base_rows = D.task_rows("home_goals", fold.fit)
    tot_b = base_rows["home_goals"] + base_rows["away_goals"]
    res_b = np.where(base_rows["home_goals"] > base_rows["away_goals"], 0,
                     np.where(base_rows["home_goals"] == base_rows["away_goals"], 1, 2))
    base = {"over15": float((tot_b > 1.5).mean()), "over25": float((tot_b > 2.5).mean()),
            "over35": float((tot_b > 3.5).mean()),
            "btts": float(((base_rows["home_goals"] > 0) & (base_rows["away_goals"] > 0)).mean()),
            "dc_1x": float((res_b != 2).mean()), "dc_x2": float((res_b != 0).mean()), "dc_12": float((res_b != 1).mean())}
    res["derived_markets"], derived_probs = {}, {}
    for system in SYSTEMS:
        r_name, h_name, a_name = system_member(ri, system), system_member(hi, system), system_member(ai, system)
        if not (r_name and h_name and a_name):
            continue
        d = M.derived_goal_markets(np.asarray(rp[r_name]), np.asarray(hp[h_name]), np.asarray(ap[a_name]), hg, ag)
        derived_probs[system] = d
        res["derived_markets"][system] = {"members": [r_name, h_name, a_name], **M.goal_market_metrics(d, hg, ag, base)}
        res["derived_markets"][system]["roi_1x2_value"] = roi(rt, np.asarray(rp[r_name]),
                                                              ("odds_home", "odds_draw", "odds_away"), ry)
        res["derived_markets"][system]["roi_1x2_most_likely"] = roi(rt, np.asarray(rp[r_name]),
                                                                    ("odds_home", "odds_draw", "odds_away"), ry, False)
        ou = np.column_stack([d["over25"], 1 - d["over25"]])
        res["derived_markets"][system]["roi_over_under_25_value"] = roi(
            rt, ou, ("odds_over25", "odds_under25"), np.where(hg + ag > 2.5, 0, 1))
        log.info("fold %s derived %s over25 ll %.4f", fold.label, system,
                 res["derived_markets"][system]["over25"]["log_loss"])

    # corners lines from each corners count model (one NB dispersion from the fit rows)
    ct, cy, cp, ci = task_out["corners"]
    cfit = D.task_rows("total_corners", fold.fit, cl)
    res["corners_lines"] = {}
    for name, mu in cp.items():
        y_fit, mu_fit = corner_fit_preds[name]
        size = negative_binomial_size(np.asarray(y_fit, float), np.asarray(mu_fit, float))
        probs = M.corner_line_probs(np.asarray(mu), size)
        res["corners_lines"][name] = M.binary(cy > 9.5, probs, float((cfit["total_corners"] > 9.5).mean()))

    # direct market models vs derived probabilities
    res["direct_markets"], direct_probs = {}, {}
    goal_numeric = sorted(set(D.numeric_columns("home_goals", cols)) | set(D.numeric_columns("away_goals", cols)))
    for target, line in DIRECT.items():
        is_corner = target.startswith("corners")
        numeric = D.numeric_columns("corners", cols) if is_corner else goal_numeric
        f2 = D.Fold(fold.season, add_targets(fold.tune_train), add_targets(fold.tune_valid), add_targets(fold.fit),
                    add_targets(fold.test))
        test, y, preds, info = run_task(f2, target, "binary", numeric, target, cl if is_corner else None, quick)
        rate = float(D.task_rows(target, f2.fit, cl if is_corner else None)[target].mean())
        res["direct_markets"][target] = {"candidates": info, "metrics": evaluate("binary", y, preds, rate)}
        direct_probs[target] = (test["match_id"].to_numpy(), preds)

    # consistency of direct totals / BTTS models (same rows: all played matches of the fold)
    res["consistency"] = consistency(direct_probs, rt, hg, ag)
    res["out_preds"] = out_preds
    return res


def consistency(direct_probs: dict, rt: pd.DataFrame, hg, ag) -> dict:
    """Logical checks of separately trained market models and a monotone repair."""
    out = {}
    ids = rt["match_id"].to_numpy()
    for model in ("LightGBM", "Logistic regression"):
        try:
            series = {t: pd.Series(np.asarray(direct_probs[t][1][model]), index=direct_probs[t][0]).reindex(ids)
                      for t in ("over15", "over25", "over35", "btts")}
        except KeyError:
            continue
        p15, p25, p35, pb = (series[t].to_numpy() for t in ("over15", "over25", "over35", "btts"))
        ok = np.isfinite(p15) & np.isfinite(p25) & np.isfinite(p35) & np.isfinite(pb)
        p15, p25, p35, pb = p15[ok], p25[ok], p35[ok], pb[ok]
        viol = {"over25_above_over15": float((p25 > p15).mean()), "over35_above_over25": float((p35 > p25).mean()),
                "btts_above_over15": float((pb > p15).mean())}
        # repair: pool-adjacent-violators across the three lines (keeps p15 >= p25 >= p35)
        rep = np.column_stack([p15, p25, p35]).copy()
        for r in rep:
            for _ in range(2):
                for k in range(2):
                    if r[k + 1] > r[k]:
                        r[k] = r[k + 1] = (r[k] + r[k + 1]) / 2
        tot = (hg + ag)[ok]
        after = {name: float(M.binary(tot > line, rep[:, k], 0.5)["log_loss"])
                 for k, (name, line) in enumerate((("over15", 1.5), ("over25", 2.5), ("over35", 3.5)))}
        before = {name: float(M.binary(tot > line, col, 0.5)["log_loss"])
                  for name, line, col in (("over15", 1.5, p15), ("over25", 2.5, p25), ("over35", 3.5, p35))}
        out[model] = {"violation_rate": viol, "log_loss_before_repair": before, "log_loss_after_repair": after}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*", default=[2023, 2024, 2025])
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    features = D.load()
    for fold in D.folds(features, args.folds):
        started = time.time()
        res = run_fold(features, fold, args.quick)
        preds = res.pop("out_preds")
        frame = pd.DataFrame({f"{t}|{n}|{c}": s for (t, n, c), s in preds.items()})
        frame.index.name = "match_id"
        suffix = "_quick" if args.quick else ""
        frame.to_parquet(RESULTS / f"fold_{fold.season}{suffix}.parquet")
        res["minutes"] = round((time.time() - started) / 60, 1)
        (RESULTS / f"fold_{fold.season}{suffix}.json").write_text(json.dumps(res, indent=1, default=float))
        log.info("fold %s done in %.1f min", fold.label, res["minutes"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("hyperopt", "src.modelling.training"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
