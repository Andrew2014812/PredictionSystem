"""ИССЛЕДОВАНИЕ 8 — alternative ML algorithms. Analysis of every stage (light: no fitting here).

    experiments/v6_3/heavy.sh 14400 experiments/v6_3/alt_ml/logs/tune.log .venv/bin/python -m experiments.v6_3.alt_ml.tune
    experiments/v6_3/heavy.sh 5400  experiments/v6_3/alt_ml/logs/static.log .venv/bin/python -m experiments.v6_3.alt_ml.fit26 static
    experiments/v6_3/heavy.sh 7200  experiments/v6_3/alt_ml/logs/walk.log .venv/bin/python -m experiments.v6_3.alt_ml.fit26 walk
    .venv/bin/python -m experiments.v6_3.alt_ml.run
    .venv/bin/python -m experiments.v6_3.alt_ml.build_pdf

Compute budget (declared before the run): ≤ 3 configurations per algorithm and task on a fixed grid (no Bayesian
search; production's XGBoost had a Hyperopt search, so the comparison is, if anything, tilted towards XGBoost),
2 threads per model, stage 1 ≤ 2 h, static fits ≤ 1 h, walk-forward only for XGBoost and the two best
alternatives of 2025/26 (5 monthly cut-offs × 4 tasks).

Variants on 2026/27 (all reported, Holm over all of them per universe):
* static_<algo> — all four tasks from the algorithm (fitted on 2021/22–2025/26);
* ens_gbm — mean of XGBoost (production), LightGBM and CatBoost outputs; ens_all — mean of all seven;
  ens_xgb_best — mean of XGBoost and the alternative with the best 2025/26 validation;
* best_per_task — per task the algorithm / ensemble with the best 2025/26 validation score;
* wf_<algo> — monthly walk-forward refits (XGBoost with production parameters included, to separate the effect
  of refitting from the effect of the algorithm).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from scipy import stats

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV

from . import fit26 as F
from . import mainroi as R
from . import models as M

D = C.study_dirs("alt_ml")
RES = D["results"]
CLS = ["p_home", "p_draw", "p_away"]
TASK_COLS = {"result": CLS, "home_goals": ["lam_home"], "away_goals": ["lam_away"], "corners": ["corner_mu"]}
NB_SIZE = float(EN.production_bundles()["corners"].extra.get("nb_size") or 0) or None


# ------------------------------------------------------------------------------------------------ validation
def validation_outputs() -> dict:
    """2025/26 outputs of every algorithm's best configuration (and xgb:prod) as OUTPUT_COLS tables."""
    best = F.best_configs()
    vals = {t: pd.read_parquet(D["cache"] / f"valid_{t}.parquet") for t in M.TASK_NAMES}
    out = {}
    keys = {a: {t: v[3] for t, v in tasks.items()} for a, tasks in best.items()}
    keys["xgb"] = {t: "xgb:prod" for t in M.TASK_NAMES}            # the production configuration
    for algo, tk in keys.items():
        o = vals["result"][["match_id"]].copy()
        r = vals["result"].set_index("match_id")
        o["p_home"], o["p_draw"], o["p_away"] = (r[f"{tk['result']}|{c}"].to_numpy() for c in "HDA")
        for t, col in (("home_goals", "lam_home"), ("away_goals", "lam_away"), ("corners", "corner_mu")):
            v = vals[t].set_index("match_id")[tk[t]]
            o[col] = o["match_id"].map(v)
        out[algo] = o
    return out


def ensembles(outs: dict, order: list[str]) -> dict:
    """Ensembles defined before 2026/27: linear pools of the outputs (1X2 probabilities, rates)."""
    best_alt = order[0]
    defs = {"ens_gbm": ["xgb", "lgbm", "catboost"], "ens_all": list(outs), "ens_xgb_best": ["xgb", best_alt]}
    res = {}
    for name, members in defs.items():
        members = [m for m in members if m in outs]
        base = outs[members[0]][["match_id"]].copy()
        for col in EN.OUTPUT_COLS:
            base[col] = np.mean([outs[m].set_index("match_id")[col].reindex(base["match_id"]).to_numpy()
                                 for m in members], axis=0)
        res[name] = base
    return res, defs


# ------------------------------------------------------------------------------------------------ metrics
def per_match_losses(out: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    x = truth.join(out.set_index("match_id"), how="left")
    y = x["result"].map({"H": 0, "D": 1, "A": 2})
    P = x[CLS].to_numpy(float)
    ok = y.notna().to_numpy()
    yi = y.fillna(0).astype(int).to_numpy()
    onehot = np.eye(3)[yi]
    l = pd.DataFrame(index=x.index)
    l["ll"] = np.where(ok, -np.log(np.clip(P[np.arange(len(x)), yi], 1e-9, 1)), np.nan)
    l["brier"] = np.where(ok, ((P - onehot) ** 2).sum(1), np.nan)
    for t, col, ycol in (("home_goals", "lam_home", "home_goals"), ("away_goals", "lam_away", "away_goals"),
                         ("corners", "corner_mu", "total_corners")):
        yv, mu = x[ycol].to_numpy(float), x[col].to_numpy(float)
        okc = np.isfinite(yv) & np.isfinite(mu)
        mu_c = np.clip(np.nan_to_num(mu, nan=1.0), 1e-6, None)
        yv0 = np.nan_to_num(yv)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * (np.where(yv0 > 0, yv0 * np.log(yv0 / mu_c), 0.0) - (yv0 - mu_c))
        if t == "corners" and NB_SIZE:
            logp = stats.nbinom.logpmf(yv0, NB_SIZE, NB_SIZE / (NB_SIZE + mu_c))
        else:
            logp = stats.poisson.logpmf(yv0, mu_c)
        l[f"dev_{t}"] = np.where(okc, dev, np.nan)
        l[f"logscore_{t}"] = np.where(okc, -logp, np.nan)
        l[f"abs_{t}"] = np.where(okc, np.abs(yv0 - mu_c), np.nan)
        l[f"sq_{t}"] = np.where(okc, (yv0 - mu_c) ** 2, np.nan)
    return l


def task_metrics(out: pd.DataFrame, truth: pd.DataFrame) -> dict:
    x = truth.join(out.set_index("match_id"), how="left")
    l = per_match_losses(out, truth)
    y = x["result"].map({"H": 0, "D": 1, "A": 2})
    ok = y.notna()
    P = x.loc[ok, CLS].to_numpy(float)
    yy = y[ok].astype(int).to_numpy()
    pred = P.argmax(1)
    top = P.max(1)
    res = {"result": {"n": int(ok.sum()), "accuracy": float(accuracy_score(yy, pred)),
                      "macro_f1": float(f1_score(yy, pred, average="macro")),
                      "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
                      "log_loss": float(l["ll"].mean()), "brier": float(l["brier"].mean()),
                      "ece_top": R.ece(top, (pred == yy).astype(float)),
                      "draw_predicted_share": float((pred == 1).mean()),
                      "mean_p": {c: float(P[:, i].mean()) for i, c in enumerate("HDA")}}}
    for t, col, ycol in (("home_goals", "lam_home", "home_goals"), ("away_goals", "lam_away", "away_goals"),
                         ("corners", "corner_mu", "total_corners")):
        okc = x[ycol].notna() & x[col].notna()
        if not okc.any():
            continue
        yv, mu = x.loc[okc, ycol].to_numpy(float), x.loc[okc, col].to_numpy(float)
        res[t] = {"n": int(okc.sum()), "mae": float(np.abs(yv - mu).mean()),
                  "rmse": float(np.sqrt(((yv - mu) ** 2).mean())), "r2": float(r2_score(yv, mu)),
                  "poisson_deviance": float(l.loc[okc.to_numpy(), f"dev_{t}"].mean()),
                  "log_score": float(l.loc[okc.to_numpy(), f"logscore_{t}"].mean()),
                  "mean_pred": float(mu.mean()), "mean_true": float(yv.mean())}
    return res


def paired_losses(new: pd.DataFrame, base: pd.DataFrame) -> dict:
    out = {}
    for col in ("ll", "brier", "dev_home_goals", "dev_away_goals", "dev_corners", "abs_corners"):
        a, b = new[col].to_numpy(), base[col].reindex(new.index).to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        out[col] = C.paired_mean_diff(a[ok], b[ok])
    return out


# ------------------------------------------------------------------------------------------------ main ROI
def main_eval(outputs: pd.DataFrame, label: str, mains: dict, cals: dict) -> dict:
    res = {}
    for u in R.UNIVERSES:
        main, cands = R.run_main(outputs, u, label, keep_candidates=True)
        ev = R.evaluate(main, u)
        ev["monthly"] = R.monthly_diff(main, R.baseline_b(u)).to_dict("records")
        ev["league"] = R.league_diff(main, R.baseline_b(u)).to_dict("records")
        ev["top_wins"] = R.top_win_dependence(main)
        ev["odds_sensitivity"] = R.odds_sensitivity(main)
        ev["groups"] = main["group"].value_counts().to_dict()
        if u == "close":
            cal = R.market_calibration(cands.loc[cands["match_id"].isin(R.results().index)])
            ev["calibration"] = R.calibration_compare(cal, cals["base"])
        res[u] = ev
        mains[(label, u)] = main
    return res


def figures(val: pd.DataFrame, metrics26: dict, roi: dict, labels: list[str]) -> None:
    import matplotlib.pyplot as plt
    from scripts import report_kit as K
    K.plot_style()
    Fd = D["figures"]
    # 1. validation score relative to xgb:prod per task
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, t in zip(axs.ravel(), M.TASK_NAMES):
        v = val.loc[val["task"] == t].copy()
        ref = float(v.loc[v["key"] == "xgb:prod", "score"].iloc[0])
        v["rel"] = (v["score"] - ref) / ref * 100
        v = v.sort_values("rel")
        ax.barh(range(len(v)), v["rel"], color=[K.ACCENT if k.startswith("xgb") else (K.PALETTE[2] if k.startswith("ens") else K.MUTED)
                                                for k in v["key"]])
        ax.set_yticks(range(len(v)), v["key"], fontsize=7.5)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title({"result": "1X2 (log loss)", "home_goals": "голы хозяев (девиация)",
                      "away_goals": "голы гостей (девиация)", "corners": "угловые (девиация)"}[t], fontsize=9)
        ax.set_xlabel("% к XGBoost (prod)")
    K.save_chart(fig, Fd / "validation_2025.png")
    # 2. 2026/27 log loss / deviance differences vs production with CI
    fig, axs = plt.subplots(1, 4, figsize=(12, 0.32 * len(labels) + 1.6), sharey=True)
    for ax, (col, title, scale) in zip(axs, (("ll", "1X2 log loss", 1000), ("dev_home_goals", "голы хозяев, девиация", 1000),
                                             ("dev_away_goals", "голы гостей, девиация", 1000),
                                             ("dev_corners", "угловые, девиация", 1000))):
        d = [metrics26[k]["paired"][col] for k in labels]
        ax.errorbar([v["diff"] * scale for v in d], range(len(labels)),
                    xerr=[[(v["diff"] - v["ci95"][0]) * scale for v in d], [(v["ci95"][1] - v["diff"]) * scale for v in d]],
                    fmt="o", capsize=2, color=K.PALETTE[0])
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Δ к production, ·10⁻³")
    axs[0].set_yticks(range(len(labels)), labels, fontsize=7)
    K.save_chart(fig, Fd / "metrics_2026_paired.png")
    # 3. Main ROI difference vs Baseline B, both universes
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(labels) + 1.4))
    for j, u in enumerate(("close", "t09")):
        d = [roi[k][u]["vs_b"]["roi_diff"] for k in labels]
        y = np.arange(len(labels)) + (j - 0.5) * 0.3
        ax.errorbar([v["diff"] * 100 for v in d], y,
                    xerr=[[(v["diff"] - v["ci95"][0]) * 100 for v in d], [(v["ci95"][1] - v["diff"]) * 100 for v in d]],
                    fmt="o", capsize=2, color=K.PALETTE[j], label=u)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("ΔROI Main к Baseline B того же универсума, п.п. (парный бутстреп, 95 %)")
    ax.legend(fontsize=7)
    K.save_chart(fig, Fd / "main_roi_2026.png")
    # 4. monthly ROI of the walk-forward variants (close)
    wf = [k for k in labels if k.startswith("wf_")]
    if wf:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        mb = pd.DataFrame(roi[wf[0]]["close"]["monthly"])
        mb = mb.loc[mb["bets"] >= 20]
        ax.plot(mb["month"], mb["roi_b"] * 100, "k--", label="Baseline B")
        for j, k in enumerate(wf):
            mm = pd.DataFrame(roi[k]["close"]["monthly"])
            mm = mm.loc[mm["bets"] >= 20]
            ax.plot(mm["month"], mm["roi"] * 100, "o-", color=K.PALETTE[j], label=k)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_ylabel("ROI Main (close), %")
        ax.legend(fontsize=7)
        K.save_chart(fig, Fd / "walk_forward_monthly.png")


def decide(metrics26: dict, roi: dict) -> tuple[dict, dict]:
    """A: ΔROI > 0 in both universes with Holm p < 0.05 (close) and the 1X2 log loss not worse.
    B: at least one model metric (1X2 log loss, goal / corner deviance) significantly better (CI below 0), the
       1X2 log loss not significantly worse and ROI not lower in both universes. A swap of the algorithm adds
       production complexity (new dependency, model format, retraining of every task), so a small positive ROI
       without any model gain is NOT enough for B (the task asks whether the gain justifies the complexity).
    C: otherwise. wf_xgb is the refitting control (same algorithm), not a candidate."""
    out, why = {}, {}
    for k in roi:
        c, t = roi[k]["close"], roi[k]["t09"]
        dc, dt = c["vs_b"]["roi_diff"]["diff"], t["vs_b"]["roi_diff"]["diff"]
        p = metrics26[k]["paired"]
        better = [col for col in ("ll", "dev_home_goals", "dev_away_goals", "dev_corners")
                  if p[col]["ci95"][1] is not None and p[col]["ci95"][1] < 0]
        worse_ll = p["ll"]["ci95"][0] is not None and p["ll"]["ci95"][0] > 0
        if k == "wf_xgb":
            out[k], why[k] = "control", "refitting control (same algorithm as production)"
        elif dc > 0 and dt > 0 and (c.get("holm_p") or 1) < 0.05 and not worse_ll:
            out[k], why[k] = "A", "ROI better in both universes (Holm) and log loss not worse"
        elif worse_ll:
            out[k], why[k] = "C", "1X2 log loss significantly worse"
        elif better and not (dc < 0 and dt < 0):
            out[k], why[k] = "B", "better " + ", ".join(better) + "; ROI not lower in both universes"
        else:
            out[k], why[k] = "C", "no significant model gain; ROI difference not significant"
    return out, why


def preregister(dec: dict, best: dict, ens_defs: dict, best_task: dict) -> dict:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    pre = {}
    for k, d in dec.items():
        if d != "B":
            continue
        if k in ens_defs:
            members = ens_defs[k]
        elif k == "best_per_task":
            members = sorted({m for v in best_task.values() for m in (ens_defs.get(v) or [v])})
        else:
            members = [k.split("_", 1)[1]]
        pre[k] = {"hypothesis": f"Outputs of '{k}' give a lower loss than production on the task(s) where they were "
                                f"better in 2026/27 and a Main ROI not lower than production's on future matches.",
                  "rule": {"variant": k,
                           "configs": {a: ({t: "xgb:prod (production)" for t in M.TASK_NAMES} if a == "xgb" else
                                           {t: best[a][t][3] for t in M.TASK_NAMES}) for a in members if a in best},
                           "best_per_task": best_task if k == "best_per_task" else None,
                           "ensemble_members": ens_defs.get(k), "ensemble": "unweighted mean of outputs",
                           "refit": "monthly walk-forward" if k.startswith("wf_") else "fitted on all seasons before"},
                  "primary_metric": "paired loss difference vs production on the improved task(s) "
                                    "(1X2 log loss / Poisson deviance); secondary: paired Main ROI difference (close)",
                  "minimum_forward_sample": "1000 played matches (about 2.5 months of the 21 leagues)",
                  "frozen_at": now}
    return pre


def redecide() -> None:
    met = C.read_json(RES / "metrics_2026.json")
    mn = C.read_json(RES / "main_2026.json")
    sel = C.read_json(RES / "selection.json")
    mn["decisions"], mn["decision_reasons"] = decide(met, mn["variants"])
    C.write_json(RES / "main_2026.json", mn)
    C.write_json(RES / "preregistration.json", preregister(mn["decisions"], F.best_configs(), sel["ensembles"],
                                                           sel["best_per_task"]))


def main() -> None:
    started = time.time()
    log = C.setup_logging("alt_ml.run", D["logs"] / "run.log")
    tune = C.read_json(RES / "tune.json")
    best = F.best_configs()
    gain = F.validation_gain()
    order = sorted([a for a in gain if a != "xgb"], key=lambda a: -gain[a])
    # ---- stage 1: validation table, ensembles on 2025/26, best per task
    rows = []
    for t, v in tune.items():
        for key, c in v["configs"].items():
            rows.append({"task": t, "key": key, "algo": c["algo"], "score": c["score"], "n_iter": c["n_iter"],
                         "seconds": c["seconds"],
                         "best_of_algo": (key == "xgb:prod") if c["algo"] == "xgb" else best[c["algo"]][t][3] == key})
    vouts = validation_outputs()
    ens_v, ens_defs = ensembles(vouts, order)
    f = M.load_features()
    v25 = f.loc[(f["season"] == 2025) & f["played"].astype(bool)].set_index("match_id")
    truth25 = v25[["result", "home_goals", "away_goals", "total_corners"]]
    for name, o in ens_v.items():
        l = per_match_losses(o, truth25)
        for t, col in (("result", "ll"), ("home_goals", "dev_home_goals"), ("away_goals", "dev_away_goals"),
                       ("corners", "dev_corners")):
            rows.append({"task": t, "key": name, "algo": name, "score": float(l[col].mean()), "n_iter": None,
                         "seconds": 0.0, "best_of_algo": True})
    val = pd.DataFrame(rows)
    val.to_csv(RES / "validation_2025.csv", index=False)
    best_task = {}
    for t in M.TASK_NAMES:
        v = val.loc[(val["task"] == t) & val["best_of_algo"]].copy()
        v["algo2"] = v["algo"]
        best_task[t] = str(v.sort_values("score").iloc[0]["algo2"])
    log.info("best per task on 2025/26: %s; order of alternatives: %s", best_task, order)
    # Main ROI on 2025/26 (Football-Data prices) per algorithm — selection-season evidence
    sel_roi = {}
    f25 = f.loc[f["season"] == 2025]
    base25 = R.hist_main(f25, vouts["xgb"], "xgb_2025")
    for name, o in {**vouts, **ens_v}.items():
        mm = base25 if name == "xgb" else R.hist_main(f25, o, f"{name}_2025")
        s = EV.summary(mm)
        sel_roi[name] = {"summary": s, "roi_ci": EV.roi_with_ci(mm),
                         "vs_xgb": None if name == "xgb" else EV.compare(mm, base25)}
        log.info("2025/26 Main %s: ROI %.4f bets %d", name, s["roi"], s["bets"])
    # ---- stage 2: 2026/27 outputs
    m = R.matches()
    played = m.loc[m["played"].astype(bool)].set_index("match_id")
    truth = played[["result", "home_goals", "away_goals", "total_corners"]]
    outs = {"production": pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")}
    static = {"xgb": outs["production"]}
    for algo in best:
        p = D["cache"] / f"out26_{algo}.parquet"
        if algo != "xgb" and p.exists():
            static[algo] = pd.read_parquet(p)
            outs[f"static_{algo}"] = static[algo]
    ens26, _ = ensembles(static, order)
    outs.update(ens26)
    # best per task (chosen on 2025/26)
    bpt = outs["production"][["match_id"]].copy()
    src = {**static, **ens26}
    for t, a in best_task.items():
        o = src[a].set_index("match_id").reindex(bpt["match_id"])
        for col in TASK_COLS[t]:
            bpt[col] = o[col].to_numpy()
    outs["best_per_task"] = bpt
    for algo in F.walk_algos():
        p = D["cache"] / f"wf_{algo}.parquet"
        if p.exists():
            outs[f"wf_{algo}"] = pd.read_parquet(p).drop(columns=["refit_cutoff"], errors="ignore")
    # ---- metrics
    base_l = per_match_losses(outs["production"], truth)
    metrics26 = {}
    for k, o in outs.items():
        l = per_match_losses(o, truth)
        metrics26[k] = {"metrics": task_metrics(o, truth), "paired": paired_losses(l, base_l)}
        if k.startswith("wf_") and k != "wf_xgb" and "wf_xgb" in outs:
            metrics26[k]["paired_vs_wf_xgb"] = paired_losses(l, per_match_losses(outs["wf_xgb"], truth))
    # market odds as a reference for the 1X2 metrics
    q = R.market_probs("close").reset_index().rename(columns={"q_home": "p_home", "q_draw": "p_draw", "q_away": "p_away"})
    qo = q.assign(lam_home=np.nan, lam_away=np.nan, corner_mu=np.nan)
    tq = truth.loc[truth.index.isin(q["match_id"])]
    metrics26["_market_close"] = {"metrics": {"result": task_metrics(qo, tq)["result"]}}
    # ---- Main ROI
    mains, roi = {}, {}
    base_cands = R.candidates_b("close")
    cals = {"base": R.market_calibration(base_cands.loc[base_cands["match_id"].isin(R.results().index)])}
    labels = [k for k in outs if k != "production"]
    for k in labels:
        roi[k] = main_eval(outs[k], k, mains, cals)
        log.info("2026/27 %s: close ROI %.4f dROI %.4f p %.3f | t09 ROI %.4f dROI %.4f", k,
                 roi[k]["close"]["summary"]["roi"], roi[k]["close"]["vs_b"]["roi_diff"]["diff"],
                 roi[k]["close"]["vs_b"]["roi_diff"]["p"], roi[k]["t09"]["summary"]["roi"],
                 roi[k]["t09"]["vs_b"]["roi_diff"]["diff"])
    for u in R.UNIVERSES:
        h = C.holm({k: roi[k][u]["vs_b"]["roi_diff"]["p"] for k in labels})
        for k in labels:
            roi[k][u]["holm_p"] = h.get(k)
    # walk-forward alternatives vs walk-forward XGBoost (algorithm effect with refitting held equal)
    wf_vs = {}
    for k in labels:
        if k.startswith("wf_") and k != "wf_xgb" and ("wf_xgb", "close") in mains:
            wf_vs[k] = {u: EV.compare(mains[(k, u)], mains[("wf_xgb", u)]) for u in R.UNIVERSES}
    dec, why = decide(metrics26, roi)
    baseline_cal = {g: {"brier": float(c["brier"].mean()), "ece": R.ece(c["pc"].to_numpy(), c["y"].to_numpy()),
                        "n": int(len(c))} for g, c in cals["base"].groupby("group")}
    C.write_json(RES / "selection.json", {"best_configs": {a: {t: {"key": v[3], "score": v[2], "n_iter": v[1]}
                                                                  for t, v in tasks.items()} for a, tasks in best.items()},
                                          "validation_gain": gain, "order": order, "best_per_task": best_task,
                                          "ensembles": ens_defs, "walk_algos": F.walk_algos(), "roi_2025": sel_roi})
    C.write_json(RES / "metrics_2026.json", metrics26)
    C.write_json(RES / "main_2026.json", {"variants": roi, "decisions": dec, "decision_reasons": why,
                                          "wf_vs_wf_xgb": wf_vs,
                                          "baseline_calibration": baseline_cal,
                                          "baseline_b": {u: {"summary": EV.summary(R.baseline_b(u)),
                                                             "roi_ci": EV.roi_with_ci(R.baseline_b(u))} for u in R.UNIVERSES},
                                          "baseline_a": {"summary": EV.summary(R.baseline_a()),
                                                         "roi_ci": EV.roi_with_ci(R.baseline_a())},
                                          "n_variants": len(labels), "n_configs_stage1": int(len(val))})
    # ---- per-match tables (Baseline B and the best candidates), coverage
    res = R.results()
    cand_best = sorted([k for k in labels if not k.startswith("wf_")],
                       key=lambda k: metrics26[k]["paired"]["ll"]["diff"])[:1] + [k for k in labels if k.startswith("wf_") and k != "wf_xgb"][:1]
    cand_best = list(dict.fromkeys(cand_best + [k for k, d in dec.items() if d == "B"]))
    for u in R.UNIVERSES:
        EV.per_match_table(R.baseline_b(u), res, f"baseline_b_{u}", EN.PRODUCTION_VERSION).to_csv(
            RES / f"per_match_baseline_b_{u}.csv", index=False)
        for k in cand_best:
            EV.per_match_table(mains[(k, u)], res, f"{k}_{u}", f"alt_ml:{k}").to_csv(RES / f"per_match_{k}_{u}.csv", index=False)
    cov = []
    for u in R.UNIVERSES:
        for k, mm in [("baseline_b", R.baseline_b(u))] + [(k, mains[(k, u)]) for k in labels]:
            cov.append({"universe": u, "method": k, "season_matches": int(len(m)), "played": int(len(played)),
                        "processed": int(mm["match_id"].nunique()), "with_real_price": int(mm["odds"].notna().sum()),
                        "bets": int(mm["bet"].sum()), "push_or_void": int(mm["status"].isin(["PUSH", "VOID"]).sum()),
                        "no_price_not_a_bet": int((mm["odds"].isna() & mm["status"].isin(["WON", "LOST"])).sum()),
                        "unknown_result": int((mm["status"] == "UNKNOWN").sum()),
                        "no_corner_model_league": int((~m.loc[m["played"].astype(bool), "league"].isin(
                            EN.production_bundles()["corners"].extra["leagues"])).sum())})
    pd.DataFrame(cov).to_csv(RES / "coverage.csv", index=False)
    figures(val, metrics26, roi, labels)
    C.write_json(RES / "preregistration.json", preregister(dec, best, ens_defs, best_task))
    C.write_json(RES / "run_info.json", C.run_info("alt_ml", started, {
        "grid": M.GRID, "train_tuning": "2021/22-2024/25", "validation": "2025/26", "fit_2026": "2021/22-2025/26",
        "walk_forward": "monthly cut-offs, expanding window", "threads": C.N_JOBS},
        {"libraries_extra": {"lightgbm": __import__("lightgbm").__version__,
                             "catboost": __import__("catboost").__version__},
         "bootstrap": C.N_BOOT, "seed": C.SEED}))
    log.info("done in %.0fs", time.time() - started)


def figures_only() -> None:
    val = pd.read_csv(RES / "validation_2025.csv")
    met = C.read_json(RES / "metrics_2026.json")
    roi = C.read_json(RES / "main_2026.json")["variants"]
    figures(val, met, roi, list(roi))


if __name__ == "__main__":
    import sys
    {"redecide": redecide, "figures": figures_only}.get((sys.argv[1:] or ["main"])[0], main)()
