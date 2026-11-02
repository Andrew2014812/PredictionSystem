"""Q1 — the draw problem of the 1X2 classifier: diagnosis, probability variants, decision rules.

Probability variants (all scored on the OUTER season of each fold; every fitted constant —
temperature, vector scaling, blend weight, Dixon-Coles rho — comes from the INNER validation
season, i.e. a model fitted on < s-1 predicting s-1):

    a   baseline            production XGBoost (multi:softprob, tuned params)
    b   draw weight         the same model refitted with a sample weight on draws (1.2 / 1.4 / balanced)
    c1  temperature         one temperature T
    c2  vector scaling      z_k' = a_k log p_k + b_k  (b_H = 0)
    c3  draw bias           log p_D + b only
    d   draw model blend    binary Draw-vs-not XGBoost blended into P(D), H/A renormalised
    e   Poisson blend       (1-w) XGBoost + w Poisson score-matrix 1X2 (goal models)
    e0  Poisson only        1X2 of the goal models' score matrix
    f   Dixon-Coles blend   as e with the low-score correction (rho fitted on inner validation)
    f0  Dixon-Coles only
    ec2 Poisson blend + vector scaling (fitted on inner validation)

Decision rules (named outcome only; probabilities unchanged):
    argmax | multiplier argmax(P_H, m P_D, P_A) | threshold: draw if P_D >= t and |P_H - P_A| <= g
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import f1_score, log_loss

from src import config
from src.markets.distributions import fit_dixon_coles_rho
from src.modelling.tasks import RESULT_CLASSES, TASKS

from . import common as C
from . import scores as S

log = C.setup_logging("draw")
DRAW_WEIGHTS = {"b_w1.2": 1.2, "b_w1.4": 1.4, "b_balanced": "balanced"}


# ---------------------------------------------------------------------------------------
# calibration maps
# ---------------------------------------------------------------------------------------
def temperature(p, T):
    q = np.power(np.clip(p, 1e-12, 1), 1 / T)
    return q / q.sum(1, keepdims=True)


def fit_temperature(p, y):
    return float(minimize_scalar(lambda t: log_loss(y, temperature(p, t), labels=[0, 1, 2]),
                                 bounds=(0.5, 2.5), method="bounded").x)


def vector_scale(p, a, b):
    z = np.log(np.clip(p, 1e-12, 1)) * a + b
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_vector(p, y):
    def nll(x):
        return log_loss(y, vector_scale(p, x[:3], np.r_[0.0, x[3:]]), labels=[0, 1, 2])
    x = minimize(nll, np.r_[1, 1, 1, 0, 0], method="L-BFGS-B").x
    return x[:3].tolist(), [0.0, *x[3:].tolist()]


def fit_draw_bias(p, y):
    def f(b):
        return log_loss(y, vector_scale(p, np.ones(3), np.array([0, b, 0])), labels=[0, 1, 2])
    return float(minimize_scalar(f, bounds=(-1, 1), method="bounded").x)


def blend_draw(p, q, w):
    pd_ = (1 - w) * p[:, 1] + w * q
    rest = np.clip(p[:, 0] + p[:, 2], 1e-12, None)
    return np.stack([p[:, 0] * (1 - pd_) / rest, pd_, p[:, 2] * (1 - pd_) / rest], 1)


def best_weight(fn, y, grid=np.round(np.arange(0, 1.0001, 0.05), 2)):
    scores = {float(w): log_loss(y, fn(w), labels=[0, 1, 2]) for w in grid}
    return min(scores, key=scores.get), scores


# ---------------------------------------------------------------------------------------
# decision rules
# ---------------------------------------------------------------------------------------
def rule_multiplier(p, m):
    return np.argmax(p * np.array([1.0, m, 1.0]), 1)


def rule_threshold(p, t, g):
    named = np.where(p[:, 0] >= p[:, 2], 0, 2)
    draw = (p[:, 1] >= t) & (np.abs(p[:, 0] - p[:, 2]) <= g)
    return np.where(draw, 1, np.where(p.argmax(1) == 1, 1, named))


def rule_grid():
    out = [("argmax", {}, lambda p: p.argmax(1))]
    for m in np.round(np.arange(1.05, 1.801, 0.05), 2):
        out.append((f"mult_{m:.2f}", {"multiplier": float(m)}, lambda p, m=m: rule_multiplier(p, m)))
    for t, g in itertools.product(np.round(np.arange(0.24, 0.345, 0.01), 2),
                                  [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 1.0]):
        out.append((f"thr_t{t:.2f}_g{g:.2f}", {"t": float(t), "g": float(g)},
                    lambda p, t=t, g=g: rule_threshold(p, t, g)))
    return out


def choose_rule(p, y, cap):
    """Highest macro F1 among rules whose predicted draw share does not exceed ``cap``."""
    rows = []
    for name, params, fn in rule_grid():
        pred = fn(p)
        rows.append({"rule": name, "params": params, "macro_f1": float(f1_score(y, pred, average="macro")),
                     "accuracy": float((pred == y).mean()), "draw_share": float((pred == 1).mean())})
    ok = [r for r in rows if r["draw_share"] <= cap] or rows[:1]
    return max(ok, key=lambda r: (r["macro_f1"], r["accuracy"])), rows


def apply_rule(name, p):
    return dict((n, fn) for n, _, fn in rule_grid())[name](p)


# ---------------------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------------------
def diagnose(fold_preds: dict) -> dict:
    df = C.features()
    task = TASKS["result"]
    out = {
        "result_classes": RESULT_CLASSES,
        "label_mapping": {c: i for i, c in enumerate(RESULT_CLASSES)},
        "objective": task.objective, "eval_metric": task.eval_metric,
        "draw_rule_enabled": config.DRAW_RULE,
        "production_draw_multiplier": C.prod_meta()["tasks"]["result"]["extra"].get("draw_multiplier"),
        "class_share_by_season": {int(s): {c: float((g["result"] == c).mean()) for c in RESULT_CLASSES}
                                  | {"n": int(len(g))} for s, g in df.groupby("season")},
    }
    fit = C.production_model("result").fit(C.before(df, 2023), C.target("result", C.before(df, 2023)))
    out["xgb_classes_order"] = [int(c) for c in fit.model.classes_]
    per_fold = {}
    for s, (inner, outer) in fold_preds.items():
        p = outer[["pH", "pD", "pA"]].to_numpy()
        y = outer["y"].to_numpy()
        # order sanity: the column read as P(H) must be highest on matches the home team won
        mean_by_true = {c: {k: float(p[y == i, j].mean()) for j, k in enumerate(CLASSES_)}
                        for i, c in enumerate(RESULT_CLASSES)}
        gap = np.abs(p[:, 0] - p[:, 2])
        bins = pd.cut(gap, [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0], include_lowest=True)
        by_gap = []
        for b, g in pd.DataFrame({"gap": bins, "y": y, "pD": p[:, 1], "pmax_other": np.maximum(p[:, 0], p[:, 2])}
                                 ).groupby("gap", observed=True):
            by_gap.append({"gap_bin": str(b), "n": int(len(g)), "actual_draw_share": float((g["y"] == 1).mean()),
                           "mean_p_draw": float(g["pD"].mean()), "mean_max_p_home_away": float(g["pmax_other"].mean())})
        odds = outer[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
        ok = np.all(odds > 1, 1)
        inv = 1 / odds[ok]
        book = inv / inv.sum(1, keepdims=True)
        mults = {}
        for m in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
            pred = rule_multiplier(p, m)
            mults[str(m)] = {"predicted_draw_share": float((pred == 1).mean()),
                             "accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}
        per_fold[str(s)] = {
            "n": int(len(y)), "actual_draw_share": float((y == 1).mean()), "mean_p_draw": float(p[:, 1].mean()),
            "share_draw_is_argmax": float((p.argmax(1) == 1).mean()),
            "p_draw_quantiles": {str(q): float(np.quantile(p[:, 1], q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99, 1.0)},
            "share_p_draw_ge_0.30": float((p[:, 1] >= 0.30).mean()),
            "share_p_draw_ge_0.33": float((p[:, 1] >= 0.333).mean()),
            "mean_probability_by_true_class": mean_by_true,
            "draw_reliability": C.reliability(y == 1, p[:, 1]),
            "draw_ece": C.ece(y == 1, p[:, 1]),
            "draw_share_by_home_away_gap": by_gap,
            "bookmaker": {"n": int(ok.sum()), "mean_p_draw": float(book[:, 1].mean()),
                          "share_draw_is_argmax": float((book.argmax(1) == 1).mean()),
                          "max_p_draw": float(book[:, 1].max())},
            "multiplier_effect": mults,
        }
    out["folds"] = per_fold
    return out


CLASSES_ = ("H", "D", "A")


# ---------------------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------------------
def add_abs_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    cols = [c for c in df.columns if c.startswith("diff_") or c == "elo_diff"]
    names = []
    for c in cols:
        df[f"abs_{c}"] = df[c].abs()
        names.append(f"abs_{c}")
    df["abs_elo_exp_gap"] = (df["elo_exp_home"] - 0.5).abs()
    names.append("abs_elo_exp_gap")
    return df, names


def run_fold(season: int, inner: pd.DataFrame, outer: pd.DataFrame) -> tuple[dict, dict, dict]:
    df = C.features()
    tr_out = C.before(df, season)
    y_in = inner["y"].to_numpy()
    p_out, p_in = outer[["pH", "pD", "pA"]].to_numpy(), inner[["pH", "pD", "pA"]].to_numpy()
    probs, params, secs = {"a_baseline": p_out}, {}, {}
    info = C.base_predictions(season)[2]
    base_sec = info["outer"]["seconds"]["result"]
    inner_sec = info["inner"]["seconds"]["result"]
    secs["a_baseline"] = base_sec

    # b: draw sample weights (refit on < s)
    y_tr = C.target("result", tr_out)
    for name, w in DRAW_WEIGHTS.items():
        if w == "balanced":
            freq = np.bincount(y_tr, minlength=3) / len(y_tr)
            sw = (1 / (3 * freq))[y_tr]
        else:
            sw = np.where(y_tr == 1, w, 1.0)
        m = C.production_model("result").fit(tr_out, y_tr, sample_weight=sw)
        probs[name] = m.predict(C.seasons(df, [season]))
        secs[name] = m.seconds
        params[name] = {"draw_weight": w}

    # c: calibration maps fitted on the inner validation season
    t0 = time.time()
    T = fit_temperature(p_in, y_in)
    probs["c1_temperature"] = temperature(p_out, T)
    params["c1_temperature"] = {"T": T}
    secs["c1_temperature"] = base_sec + inner_sec + time.time() - t0
    t0 = time.time()
    a, b = fit_vector(p_in, y_in)
    probs["c2_vector"] = vector_scale(p_out, np.array(a), np.array(b))
    params["c2_vector"] = {"a": a, "b": b}
    secs["c2_vector"] = base_sec + inner_sec + time.time() - t0
    t0 = time.time()
    bd = fit_draw_bias(p_in, y_in)
    probs["c3_draw_bias"] = vector_scale(p_out, np.ones(3), np.array([0, bd, 0]))
    params["c3_draw_bias"] = {"b_draw": bd}
    secs["c3_draw_bias"] = base_sec + inner_sec + time.time() - t0

    # d: binary draw model (+ |difference| features), trees by early stopping on the inner season
    t0 = time.time()
    full_abs, abs_cols = add_abs_features(df)
    numeric = C.task_numeric("result") + abs_cols
    rp, _ = C.prod_params("result")
    dm = C.XGBFit(numeric, rp, 1, "binary:logistic", "logloss", True)
    tr_in_a, va_in_a = C.before(full_abs, season - 1), C.seasons(full_abs, [season - 1])
    n_trees = dm.early_stopped_trees(tr_in_a, (tr_in_a["result"] == "D").astype(int), va_in_a,
                                     (va_in_a["result"] == "D").astype(int))
    dm.n = n_trees
    q_in = dm.fit(tr_in_a, (tr_in_a["result"] == "D").astype(int)).predict(va_in_a)
    w_d, grid_d = best_weight(lambda w: blend_draw(p_in, q_in, w), y_in)
    tr_out_a = C.before(full_abs, season)
    q_out = dm.fit(tr_out_a, (tr_out_a["result"] == "D").astype(int)).predict(C.seasons(full_abs, [season]))
    probs["d_draw_model_blend"] = blend_draw(p_out, q_out, w_d)
    params["d_draw_model_blend"] = {"n_trees": n_trees, "weight": w_d,
                                    "draw_model_log_loss_inner": float(log_loss(y_in == 1, q_in)),
                                    "extra_features": len(abs_cols)}
    secs["d_draw_model_blend"] = base_sec + time.time() - t0

    # e / f: score-matrix 1X2 from the goal models (independent Poisson / Dixon-Coles)
    t0 = time.time()
    pois_in = S.outcome_1x2(S.matrices(inner["lh"], inner["la"]))
    pois_out = S.outcome_1x2(S.matrices(outer["lh"], outer["la"]))
    w_e, _ = best_weight(lambda w: (1 - w) * p_in + w * pois_in, y_in)
    probs["e0_poisson_only"] = pois_out
    probs["e_poisson_blend"] = (1 - w_e) * p_out + w_e * pois_out
    params["e_poisson_blend"] = {"weight_poisson": w_e}
    goal_sec = info["outer"]["seconds"]["home_goals"] + info["outer"]["seconds"]["away_goals"]
    goal_sec_in = info["inner"]["seconds"]["home_goals"] + info["inner"]["seconds"]["away_goals"]
    secs["e0_poisson_only"] = goal_sec
    secs["e_poisson_blend"] = base_sec + inner_sec + goal_sec + goal_sec_in + time.time() - t0
    t0 = time.time()
    rho = fit_dixon_coles_rho(inner["lh"].to_numpy(), inner["la"].to_numpy(),
                              inner["home_goals"].to_numpy(), inner["away_goals"].to_numpy())
    dc_in = S.outcome_1x2(S.matrices(inner["lh"], inner["la"], rho))
    dc_out = S.outcome_1x2(S.matrices(outer["lh"], outer["la"], rho))
    w_f, _ = best_weight(lambda w: (1 - w) * p_in + w * dc_in, y_in)
    probs["f0_dixon_coles_only"] = dc_out
    probs["f_dixon_coles_blend"] = (1 - w_f) * p_out + w_f * dc_out
    params["f0_dixon_coles_only"] = {"rho": rho}
    params["f_dixon_coles_blend"] = {"rho": rho, "weight_dc": w_f}
    secs["f0_dixon_coles_only"] = goal_sec + goal_sec_in
    secs["f_dixon_coles_blend"] = secs["e_poisson_blend"] + time.time() - t0
    # e + vector scaling, both fitted on the inner season
    t0 = time.time()
    blend_in = (1 - w_e) * p_in + w_e * pois_in
    a2, b2 = fit_vector(blend_in, y_in)
    probs["ec2_poisson_blend_vector"] = vector_scale(probs["e_poisson_blend"], np.array(a2), np.array(b2))
    params["ec2_poisson_blend_vector"] = {"weight_poisson": w_e, "a": a2, "b": b2}
    secs["ec2_poisson_blend_vector"] = secs["e_poisson_blend"] + time.time() - t0

    # naive baselines (training seasons < s)
    yt = C.target("result", tr_out)
    freq = np.bincount(yt, minlength=3) / len(yt)
    probs["naive_constant"] = np.tile(freq, (len(outer), 1))
    lf = tr_out.assign(y=yt).groupby("league")["y"].apply(lambda s: np.bincount(s, minlength=3) / len(s))
    probs["naive_league"] = np.stack([lf.get(l, freq) for l in outer["league"]])
    secs["naive_constant"] = secs["naive_league"] = 0.0
    params["naive_constant"] = {"freq": freq.tolist()}
    return probs, params, secs


VARIANT_TITLES = {
    "a_baseline": "a) baseline XGBoost", "b_w1.2": "b) вес ничьих 1.2", "b_w1.4": "b) вес ничьих 1.4",
    "b_balanced": "b) balanced веса", "c1_temperature": "c1) температура", "c2_vector": "c2) vector scaling",
    "c3_draw_bias": "c3) сдвиг ничьей", "d_draw_model_blend": "d) бинарная модель ничьей",
    "e0_poisson_only": "e0) только Пуассон", "e_poisson_blend": "e) бленд с Пуассоном",
    "f0_dixon_coles_only": "f0) только Диксон–Коулс", "f_dixon_coles_blend": "f) бленд с Д–К",
    "ec2_poisson_blend_vector": "e+c2) бленд + vector", "naive_constant": "наивная константа",
    "naive_league": "наивная по лиге", "bookmaker": "букмекер (Bet365)",
}


def run() -> dict:
    t_start = time.time()
    fold_preds = {s: C.base_predictions(s)[:2] for s in C.FOLDS}
    result = {"diagnosis": diagnose(fold_preds), "folds": {}}
    all_probs = {}
    for s in C.FOLDS:
        inner, outer = fold_preds[s]
        probs, params, secs = run_fold(s, inner, outer)
        all_probs[s] = probs
        y = outer["y"].to_numpy()
        base_loss = C.multiclass_loss_per_match(y, probs["a_baseline"])
        fold = {"variants": {}, "params": params, "train_seconds": secs}
        for name, p in probs.items():
            met = C.cls_metrics(y, p)
            met["delta_log_loss_vs_baseline"] = C.paired_bootstrap(C.multiclass_loss_per_match(y, p), base_loss)
            met["reliability_draw"] = C.reliability(y == 1, p[:, 1])
            fold["variants"][name] = met
        odds = outer[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
        ok = np.all(odds > 1, 1)
        inv = 1 / odds[ok]
        book = inv / inv.sum(1, keepdims=True)
        fold["bookmaker_same_rows"] = {"n": int(ok.sum()), "bookmaker": C.cls_metrics(y[ok], book),
                                       **{n: {k: v for k, v in C.cls_metrics(y[ok], probs[n][ok]).items()
                                              if k in ("log_loss", "brier", "accuracy")}
                                          for n in probs}}
        fold["bookmaker_same_rows"]["bookmaker"]["reliability_draw"] = C.reliability(y[ok] == 1, book[:, 1])
        result["folds"][str(s)] = fold
        pd.DataFrame({f"{n}|{c}": p[:, i] for n, p in probs.items() for i, c in enumerate(CLASSES_)}
                     ).assign(match_id=outer["match_id"].to_numpy(), y=y).to_parquet(C.CACHE / f"draw_probs_{s}.parquet")
        log.info("fold %s done: %s", s, {n: round(v["log_loss"], 4) for n, v in fold["variants"].items()})

    # selection of the probability model on 2023 + 2024
    names = [n for n in all_probs[C.SELECT[0]] if not n.startswith("naive")]
    sel = {}
    for n in names:
        ll = [result["folds"][str(s)]["variants"][n]["log_loss"] for s in C.SELECT]
        br = [result["folds"][str(s)]["variants"][n]["brier"] for s in C.SELECT]
        bl = [result["folds"][str(s)]["variants"]["a_baseline"]["log_loss"] for s in C.SELECT]
        bb = [result["folds"][str(s)]["variants"]["a_baseline"]["brier"] for s in C.SELECT]
        ys = np.concatenate([fold_preds[s][1]["y"].to_numpy() for s in C.SELECT])
        ps = np.concatenate([all_probs[s][n] for s in C.SELECT])
        sel[n] = {"pooled_log_loss": float(log_loss(ys, ps, labels=[0, 1, 2])),
                  "worst_fold_log_loss_delta": float(max(a - b for a, b in zip(ll, bl))),
                  "better_log_loss_both": bool(all(a < b for a, b in zip(ll, bl))),
                  "better_brier_both": bool(all(a < b for a, b in zip(br, bb)))}
    ok = [n for n in names if n != "a_baseline" and sel[n]["better_log_loss_both"] and sel[n]["better_brier_both"]]
    chosen = min(ok, key=lambda n: sel[n]["pooled_log_loss"]) if ok else "a_baseline"
    result["probability_selection"] = {"table": sel, "eligible": ok, "chosen": chosen,
                                       "rule": "lower log loss AND Brier than baseline on both 2023 and 2024; "
                                               "then lowest pooled log loss"}

    # decision rules: (1) honest per fold — chosen on the INNER season predictions of the same
    # probability model; (2) final — chosen on pooled 2023+2024 OUTER predictions, 2025 reported
    df = C.features()
    rules = {}
    for model in dict.fromkeys(["a_baseline", chosen]):
        per_fold = {}
        for s in C.FOLDS:
            inner, outer = fold_preds[s]
            y_in, y = inner["y"].to_numpy(), outer["y"].to_numpy()
            cap = float((C.target("result", C.before(df, s - 1)) == 1).mean())
            p_in = inner_probs(model, s, inner, result["folds"][str(s)]["params"])
            if p_in is None:
                continue
            best, _ = choose_rule(p_in, y_in, cap)
            pred = apply_rule(best["rule"], all_probs[s][model])
            per_fold[str(s)] = {"rule": best["rule"], "inner_choice": best,
                                "metrics": C.cls_metrics(y, all_probs[s][model], pred)}
        ys = np.concatenate([fold_preds[s][1]["y"].to_numpy() for s in C.SELECT])
        ps = np.concatenate([all_probs[s][model] for s in C.SELECT])
        cap = float((C.target("result", C.before(df, C.SELECT[-1])) == 1).mean())
        best, grid = choose_rule(ps, ys, cap)
        final = {str(s): C.cls_metrics(fold_preds[s][1]["y"].to_numpy(), all_probs[s][model],
                                       apply_rule(best["rule"], all_probs[s][model])) for s in C.FOLDS}
        roi = {}
        for s in C.FOLDS:
            outer = fold_preds[s][1]
            for rname in ("argmax", best["rule"]):
                pred = apply_rule(rname, all_probs[s][model])
                odds = outer[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)[np.arange(len(pred)), pred]
                okk = np.isfinite(odds) & (odds > 1)
                prof = np.where(pred == outer["y"].to_numpy(), odds - 1, -1.0)[okk]
                roi.setdefault(rname, {})[str(s)] = {"bets": int(okk.sum()), "roi": float(prof.mean())}
        rules[model] = {"honest_per_fold": per_fold, "final_rule": best, "draw_share_cap": cap,
                        "final_rule_metrics": final, "roi_named_outcome_bet365": roi,
                        "grid_top10": sorted(grid, key=lambda r: -r["macro_f1"])[:10]}
    result["decision_rules"] = rules
    result["minutes"] = (time.time() - t_start) / 60
    C.save_json("draw.json", result)
    figures(result, all_probs, fold_preds)
    return result


def inner_probs(model: str, s: int, inner: pd.DataFrame, params: dict):
    """Inner-season probabilities of a variant (for choosing the decision rule honestly)."""
    p = inner[["pH", "pD", "pA"]].to_numpy()
    if model == "a_baseline":
        return p
    if model == "c1_temperature":
        return temperature(p, params[model]["T"])
    if model == "c2_vector":
        return vector_scale(p, np.array(params[model]["a"]), np.array(params[model]["b"]))
    if model == "c3_draw_bias":
        return vector_scale(p, np.ones(3), np.array([0, params[model]["b_draw"], 0]))
    pois = S.outcome_1x2(S.matrices(inner["lh"], inner["la"]))
    if model == "e_poisson_blend":
        w = params[model]["weight_poisson"]
        return (1 - w) * p + w * pois
    if model == "ec2_poisson_blend_vector":
        w = params[model]["weight_poisson"]
        return vector_scale((1 - w) * p + w * pois, np.array(params[model]["a"]), np.array(params[model]["b"]))
    if model == "f_dixon_coles_blend":
        dc = S.outcome_1x2(S.matrices(inner["lh"], inner["la"], params[model]["rho"]))
        w = params[model]["weight_dc"]
        return (1 - w) * p + w * dc
    return None   # variants refitted on < s have no inner-season predictions (rule chosen pooled only)


def figures(result, all_probs, fold_preds):
    plt = C.plot_style()
    chosen = result["probability_selection"]["chosen"]
    show = list(dict.fromkeys(["a_baseline", chosen, "b_balanced", "c2_vector", "e0_poisson_only"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, s in zip(axes, C.FOLDS):
        y = fold_preds[s][1]["y"].to_numpy()
        for k, n in enumerate(show):
            rel = C.reliability(y == 1, all_probs[s][n][:, 1], bins=20, min_n=30)
            ax.plot([r["predicted"] for r in rel], [r["observed"] for r in rel], "o-", ms=3,
                    color=C.PALETTE[k], label=VARIANT_TITLES[n])
        book = result["folds"][str(s)]["bookmaker_same_rows"]["bookmaker"]["reliability_draw"]
        ax.plot([r["predicted"] for r in book], [r["observed"] for r in book], "s--", ms=3, color="#333333",
                label="букмекер (Bet365)")
        ax.plot([0.05, 0.45], [0.05, 0.45], ":", color="grey")
        ax.set_xlim(0.05, 0.45), ax.set_ylim(0.05, 0.45)
        tag = "отбор" if s in C.SELECT else "только для информации"
        ax.set_title(f"Сезон {config.season_label(s)} ({tag})")
        ax.set_xlabel("Предсказанная P(ничья)")
    axes[0].set_ylabel("Фактическая доля ничьих")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Калибровка P(ничья): диаграмма надёжности (бины по 0.05, n ≥ 30)")
    fig.savefig(C.FIGURES / "draw_reliability.png"), plt.close(fig)

    # confusion matrices: baseline arg-max vs recommended (probability model + decision rule)
    rec = result["decision_rules"][chosen]["final_rule"]["rule"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for col, (label, seas) in enumerate((("2023/24 + 2024/25 (отбор)", C.SELECT),
                                         ("2025/26 (только для информации)", (C.REPORT,)))):
        y = np.concatenate([fold_preds[s][1]["y"].to_numpy() for s in seas])
        for row, (title, model, rule) in enumerate((("Базовая модель, argmax", "a_baseline", "argmax"),
                                                    (f"Рекомендация: {chosen}, {rec}", chosen, rec))):
            p = np.concatenate([all_probs[s][model] for s in seas])
            cm = C.cls_metrics(y, p, apply_rule(rule, p))["confusion"]
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=10,
                            color="white" if cm[i][j] > np.max(cm) * 0.6 else "black")
            ax.set_xticks(range(3), ["П1", "Х", "П2"]), ax.set_yticks(range(3), ["П1", "Х", "П2"])
            ax.set_xlabel("Предсказано"), ax.set_ylabel("Факт"), ax.grid(False)
            ax.set_title(f"{title}\n{label}", fontsize=9)
    fig.suptitle("Матрицы ошибок 1X2")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "draw_confusion.png"), plt.close(fig)

    # predicted draw share by variant
    names = [n for n in result["folds"]["2023"]["variants"] if not n.startswith("naive")]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(names))
    for k, s in enumerate(C.FOLDS):
        v = result["folds"][str(s)]["variants"]
        ax.bar(x + (k - 1) * 0.27, [v[n]["predicted_draw_share"] for n in names], 0.27, color=C.PALETTE[k],
               label=f"доля предсказанных ничьих (argmax), {config.season_label(s)}")
        ax.scatter(x + (k - 1) * 0.27, [v[n]["mean_probability"]["D"] for n in names], color="black", s=10,
                   zorder=3, label="средняя P(ничья)" if k == 0 else None)
    act = np.mean([result["folds"][str(s)]["variants"]["a_baseline"]["actual_draw_share"] for s in C.FOLDS])
    ax.axhline(act, color="#C44E52", ls="--", label=f"фактическая доля ничьих ≈ {act:.3f}")
    ax.set_xticks(x, [VARIANT_TITLES[n] for n in names], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Доля")
    ax.set_title("Предсказанная доля ничьих (argmax) и средняя P(ничья) по вариантам")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(C.FIGURES / "draw_share_by_variant.png"), plt.close(fig)


if __name__ == "__main__":
    run()
