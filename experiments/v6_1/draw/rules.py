"""Step 3 (questions 1 and 5) — does FootPredict name draws better now, and is there a better decision rule?

Named outcome only (the probabilities, log loss and Brier never change):

    argmax                       the old way (v1–v5)
    current  t = 0.28, g = 0.20  v6: a draw when P(D) >= t and |P(H) - P(A)| <= g, else arg-max
    threshold grid               t = 0.24…0.34, g in {0.04 … 0.30, 1.0}
    multiplier grid              argmax(P(H), m P(D), P(A)), m = 1.05…1.80

Pre-defined criteria for replacing the current rule (fixed before the run):
    1. chosen on the pooled selection folds 2023/24 + 2024/25: highest macro F1 among rules whose predicted draw
       share does not exceed the draw share of the training seasons (no inflating the number of draws);
    2. macro F1 at least +0.005 above the current rule on EACH selection fold;
    3. accuracy at most 0.5 pp below the current rule on each selection fold;
    4. confirmed on 2026/27: macro F1 not below the current rule on block 2026 and on the recorded production
       history. Only then the rule may be adopted.

Evaluation sets: walk-forward blocks 2023–2026 (production setup refitted), the stored 2025/26 backtest that the
Model Analysis page shows (model backtest_2526) and the 2026/27 production history (recorded probabilities).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import config
from src.modelling.decision import decide

from . import blocks as B
from . import common as C
from . import history as Hm
from . import probs as PR

log = C.setup_logging("draw61.rules")
CURRENT = {"t": config.DRAW_THRESHOLD, "g": config.DRAW_MAX_GAP}
MIN_F1_GAIN, MAX_ACC_LOSS = 0.005, 0.005


def rule_grid() -> list[tuple[str, dict]]:
    out = [("argmax", {"kind": "argmax"})]
    for t, g in itertools.product(np.round(np.arange(0.24, 0.345, 0.01), 2),
                                  [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 1.0]):
        out.append((f"thr_t{t:.2f}_g{g:.2f}", {"kind": "threshold", "t": float(t), "g": float(g)}))
    for m in np.round(np.arange(1.05, 1.801, 0.05), 2):
        out.append((f"mult_{m:.2f}", {"kind": "multiplier", "m": float(m)}))
    return out


CURRENT_NAME = f"thr_t{CURRENT['t']:.2f}_g{CURRENT['g']:.2f}"


def apply(params: dict, p: np.ndarray) -> np.ndarray:
    if params["kind"] == "argmax":
        return decide(p)
    if params["kind"] == "threshold":
        return decide(p, 1.0, params["t"], params["g"])
    return decide(p, params["m"])


def current(p):
    return decide(p, 1.0, CURRENT["t"], CURRENT["g"])


def summary(y, p, pred, odds=None) -> dict:
    out = {**C.class_metrics(y, pred), **{k: v for k, v in C.prob_metrics(y, p).items()
                                         if k in ("log_loss", "brier", "mean_p_draw", "draw_log_loss", "draw_brier")}}
    if odds is not None:
        o = odds[np.arange(len(pred)), pred]
        ok = np.isfinite(o) & (o > 1)
        prof = np.where(pred == y, o - 1, -1.0)
        out["named_roi"] = C.roi_ci(prof[ok])
        d = ok & (pred == 1)
        out["named_draw_roi"] = C.roi_ci(prof[d])
        out["named_draw_avg_odds"] = float(o[d].mean()) if d.any() else None
    return out


def _sets() -> dict:
    """name -> (y, p, odds, league, role) for every evaluation set of the baseline probabilities."""
    sets = {}
    for b in C.FOLDS:
        fr = B.block(b)[0]
        sets[f"block_{b}"] = (fr["y"].to_numpy(), fr[["pH", "pD", "pA"]].to_numpy(),
                              fr[["odds_home", "odds_draw", "odds_away"]].to_numpy(float), fr["league"].to_numpy(),
                              f"{C.season_label(b)} (walk-forward, {C.ROLE[b]})")
    bt = Hm.backtest()
    sets["page_backtest_2025"] = (bt["y"].to_numpy(), Hm.probs(bt), bt[["odds_home", "odds_draw", "odds_away"]].to_numpy(float),
                                  bt["league"].to_numpy(), "2025/26 — модель backtest_2526 (страница Model Analysis)")
    h = Hm.history()
    sets["history_2026"] = (h["y"].to_numpy(), Hm.probs(h), h[["odds_home", "odds_draw", "odds_away"]].to_numpy(float),
                            h["league"].to_numpy(), "2026/27 — история продакшна (записанные вероятности)")
    return sets


def run() -> dict:
    df = C.features()
    cap = float((C.y_index(df.loc[df["season"] < C.SELECT[-1], "result"]) == 1).mean())
    sets = _sets()
    res = {"current_rule": CURRENT, "draw_share_cap": cap,
           "criteria": {"min_macro_f1_gain_each_selection_fold": MIN_F1_GAIN,
                        "max_accuracy_loss_each_selection_fold": MAX_ACC_LOSS,
                        "draw_share_cap": "draw share of the training seasons 2021/22–2023/24",
                        "confirmation": "macro F1 >= current on block 2026 and on the 2026/27 production history"}}

    # 1. old vs current per evaluation set (+ confusion matrices, named-outcome ROI)
    per_set = {}
    for name, (y, p, odds, lg, label) in sets.items():
        per_set[name] = {"label": label, "argmax": summary(y, p, decide(p), odds), "current": summary(y, p, current(p), odds)}
    h = Hm.history()
    rec = h["recorded_outcome"].map({"H": 0, "D": 1, "A": 2})
    okr = rec.notna().to_numpy()
    per_set["history_2026"]["recorded_named_outcome"] = summary(h["y"].to_numpy()[okr], Hm.probs(h)[okr], rec[okr].to_numpy(int))
    per_set["history_2026"]["recorded_matches_rule"] = {
        str(v): float((rec[g.index] == current(Hm.probs(g))).mean()) for v, g in h.groupby("model_version")}
    res["per_set"] = per_set

    # 2. page check: stored backtest vs the model metadata of the Model Analysis page
    meta = C.prod_meta()["tasks"]["result"]
    bt = Hm.backtest()
    res["page_check"] = {
        "page_period": "test season " + ", ".join(C.prod_meta()["periods"]["test"]) + " (model backtest_2526, trained 2021/22–2024/25)",
        "metadata_test_metrics_confusion": meta["test_metrics"]["confusion_matrix"],
        "metadata_draw_multiplier": meta.get("extra", {}).get("draw_multiplier"),
        "recomputed_argmax_confusion": per_set["page_backtest_2025"]["argmax"]["confusion"],
        "current_rule_confusion": per_set["page_backtest_2025"]["current"]["confusion"],
        "stored_predicted_outcome_confusion": C.class_metrics(
            bt["y"].to_numpy(), bt["recorded_outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy(int))["confusion"],
        "n": int(len(bt)),
        "argmax_matches_metadata": per_set["page_backtest_2025"]["argmax"]["confusion"] == meta["test_metrics"]["confusion_matrix"],
        "block_2025_vs_stored_backtest": B.check_against_backtest(),
    }

    # 3. by league (pooled walk-forward 2023–2025, and 2026/27 history)
    by_league = {}
    for tag, names in (("blocks_2023_2025", ["block_2023", "block_2024", "block_2025"]), ("history_2026", ["history_2026"])):
        y = np.concatenate([sets[n][0] for n in names])
        p = np.concatenate([sets[n][1] for n in names])
        o = np.concatenate([sets[n][2] for n in names])
        lg = np.concatenate([sets[n][3] for n in names])
        rows = {}
        for league in sorted(set(lg)):
            k = lg == league
            rows[league] = {"argmax": summary(y[k], p[k], decide(p[k]), o[k]), "current": summary(y[k], p[k], current(p[k]), o[k])}
        by_league[tag] = rows
    res["by_league"] = by_league

    # 4. rule search on the baseline and on the probability variants
    probs_res = C.load_json("probabilities.json")
    models = list(dict.fromkeys(["a_baseline", probs_res["selection"]["best"] or "a_baseline", "c3_draw_bias",
                                 "b_dc", "d_draw_xgb_blend", "e_stack", "p_parity_xgb", "e_book_blend"]))
    fr = {s: PR.load_probs(s) for s in C.FOLDS}
    search = {}
    for model in models:
        P = {s: PR.get(fr[s], model) for s in C.FOLDS}
        ok = {s: np.isfinite(P[s]).all(1) for s in C.FOLDS}
        Y = {s: fr[s]["y"].to_numpy() for s in C.FOLDS}
        grid = []
        for name, prm in rule_grid():
            row = {"rule": name, "params": prm}
            for s in C.FOLDS:
                pred = apply(prm, P[s][ok[s]])
                m = C.class_metrics(Y[s][ok[s]], pred)
                row[str(s)] = {k: m[k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "draw_recall",
                                                 "draw_precision", "predicted_draw_share")}
            ys = np.concatenate([Y[s][ok[s]] for s in C.SELECT])
            pred = np.concatenate([apply(prm, P[s][ok[s]]) for s in C.SELECT])
            m = C.class_metrics(ys, pred)
            row["pooled"] = {k: m[k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "predicted_draw_share")}
            grid.append(row)
        allowed = [r for r in grid if r["pooled"]["predicted_draw_share"] <= cap]
        best = max(allowed, key=lambda r: (r["pooled"]["macro_f1"], r["pooled"]["accuracy"]))
        cur = next(r for r in grid if r["rule"] == CURRENT_NAME)
        prod = {}                      # production reference on the same rows: baseline probabilities + current rule
        for s in C.FOLDS:
            m = C.class_metrics(Y[s][ok[s]], current(PR.get(fr[s], "a_baseline")[ok[s]]))
            prod[str(s)] = {k: m[k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "draw_recall",
                                              "draw_precision", "predicted_draw_share")}
        search[model] = {"best": best, "current": cur, "argmax": grid[0], "production_reference": prod,
                         "top10": sorted(allowed, key=lambda r: -r["pooled"]["macro_f1"])[:10]}
        if model == "a_baseline":
            search[model]["grid"] = grid
    res["rule_search"] = search

    # 5. honest per-fold choice (rule chosen on the INNER block of each fold, baseline probabilities)
    honest = {}
    for s in C.FOLDS:
        inner, outer = B.block(s - 1)[0], B.block(s)[0]
        capi = float((C.y_index(df.loc[df["season"] < s - 1, "result"]) == 1).mean())
        pi, yi = inner[["pH", "pD", "pA"]].to_numpy(), inner["y"].to_numpy()
        rows = []
        for name, prm in rule_grid():
            pred = apply(prm, pi)
            m = C.class_metrics(yi, pred)
            if m["predicted_draw_share"] <= capi:
                rows.append((m["macro_f1"], m["accuracy"], name, prm))
        f1, acc, name, prm = max(rows, key=lambda r: (r[0], r[1]))
        po = outer[["pH", "pD", "pA"]].to_numpy()
        honest[str(s)] = {"rule": name, "inner_macro_f1": f1, "cap": capi,
                          "outer": C.class_metrics(outer["y"].to_numpy(), apply(prm, po)),
                          "outer_current": C.class_metrics(outer["y"].to_numpy(), current(po))}
    res["honest_per_fold"] = honest

    # 6. verdict on replacing the current rule (baseline probabilities and the best probability variant)
    verdict = {}
    for model, sr in search.items():
        b, c = sr["best"], sr["production_reference"]
        gains = {str(s): b[str(s)]["macro_f1"] - c[str(s)]["macro_f1"] for s in C.FOLDS}
        acc = {str(s): b[str(s)]["accuracy"] - c[str(s)]["accuracy"] for s in C.FOLDS}
        sel_ok = all(gains[str(s)] >= MIN_F1_GAIN and acc[str(s)] >= -MAX_ACC_LOSS for s in C.SELECT)
        hist_gain = None
        if model == "a_baseline":
            hp = Hm.probs(h)
            hist_gain = (C.class_metrics(h["y"].to_numpy(), apply(b["params"], hp))["macro_f1"]
                         - C.class_metrics(h["y"].to_numpy(), current(hp))["macro_f1"])
        confirmed = gains[str(C.CHECK)] >= 0 and (hist_gain is None or hist_gain >= 0)
        verdict[model] = {"best_rule": b["rule"], "macro_f1_gain_vs_production": gains, "accuracy_change_vs_production": acc,
                          "history_2026_macro_f1_gain": hist_gain, "passes_selection_criteria": bool(sel_ok),
                          "confirmed_2026": bool(confirmed), "adopt": bool(sel_ok and confirmed and model == "a_baseline")}
    res["verdict"] = verdict
    C.save_json("rules.json", res)
    _csv(res)
    return res


def _csv(res: dict) -> None:
    rows = []
    for name, v in res["per_set"].items():
        for rule in ("argmax", "current", "recorded_named_outcome"):
            if rule not in v:
                continue
            cm = v[rule]["confusion"]
            for i, actual in enumerate(C.CLASSES):
                rows.append({"set": name, "label": v["label"], "rule": rule, "actual": actual,
                             "pred_H": cm[i][0], "pred_D": cm[i][1], "pred_A": cm[i][2]})
    pd.DataFrame(rows).to_csv(C.RESULTS / "confusion_matrices.csv", index=False)
    rows = []
    for name, v in res["per_set"].items():
        for rule in ("argmax", "current"):
            m = v[rule]
            rows.append({"set": name, "rule": rule, "n": m["n"], "actual_draw_share": m["actual_draw_share"],
                         "mean_p_draw": m["mean_p_draw"], "named_draw_share": m["predicted_draw_share"],
                         "draw_precision": m["draw_precision"], "draw_recall": m["draw_recall"], "draw_f1": m["draw_f1"],
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                         "log_loss": m["log_loss"], "brier": m["brier"], "named_roi": m["named_roi"]["roi"],
                         "named_draw_roi": m["named_draw_roi"]["roi"]})
    pd.DataFrame(rows).to_csv(C.RESULTS / "old_vs_current_by_set.csv", index=False)
    rows = []
    for tag, leagues in res["by_league"].items():
        for lg, v in leagues.items():
            for rule in ("argmax", "current"):
                m = v[rule]
                rows.append({"period": tag, "league": lg, "rule": rule, "n": m["n"], "actual_draw_share": m["actual_draw_share"],
                             "mean_p_draw": m["mean_p_draw"], "named_draw_share": m["predicted_draw_share"],
                             "draw_precision": m["draw_precision"], "draw_recall": m["draw_recall"],
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "log_loss": m["log_loss"]})
    pd.DataFrame(rows).to_csv(C.RESULTS / "old_vs_current_by_league.csv", index=False)


if __name__ == "__main__":
    run()
