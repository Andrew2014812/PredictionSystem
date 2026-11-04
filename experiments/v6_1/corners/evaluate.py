"""Distributions, metrics, per-period / per-league quality and the adoption rule.

Reference = the current production setup: ``xgb_current`` with ``nb_mom`` (one NB size by the
method of moments on the inner season).

Adoption rule (fixed before looking at the results):
  * "convincingly better" on a selection fold (2023/24, 2024/25): mean line log loss over the
    product lines 8.5 / 9.5 / 10.5 lower with the paired-bootstrap 95 % CI of the per-match
    difference entirely below 0, AND full-distribution log score lower;
  * must hold on BOTH selection folds;
  * "not worse" on 2025/26 and 2026/27: both point estimates (line log loss, log score) <= reference;
  * and not worse at real 2026/27 corner prices (betting.py: ROI difference CI not entirely < 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from . import data as D
from . import dist as Q
from .fit import MODELS, load_fold
from .models import convex_weights, pdev

log = D.log
REF = ("xgb_current", "nb_mom")
TOTAL_DISTS = ["poisson", "nb_mom", "nb_mle", "nb_league", "nb1"]
SIDE_DISTS = ["conv_indep", "nb_joint"]
SIDE_MODELS = ("xgb_home_away", "team_strength")
ENSEMBLE_MEMBERS = ["xgb_current", "xgb_enriched", "xgb_tuned", "xgb_ts", "lgbm", "catboost", "poisson_glm",
                    "nb_glm", "team_strength", "xgb_home_away", "xgb_ts_offset"]
SIMPLE_MEMBERS = ["xgb_current", "poisson_glm", "team_strength"]
EXT_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
TITLES = {"xgb_current": "XGBoost (текущая)", "xgb_enriched": "XGBoost + новые признаки",
          "xgb_tuned": "XGBoost, Hyperopt", "xgb_ts": "XGBoost + сила команд", "lgbm": "LightGBM",
          "catboost": "CatBoost", "poisson_glm": "Poisson GLM", "nb_glm": "NB GLM",
          "team_strength": "Иерархическая (команды)", "xgb_ts_offset": "XGBoost от силы команд", "xgb_home_away": "XGBoost дома + в гостях",
          "ens_weighted": "Ансамбль (веса)", "ens_simple": "Ансамбль (равные)", "naive_league": "Среднее лиги"}


def _arr(pred: pd.DataFrame, part: str, ids, col="mu"):
    p = pred.loc[pred["part"] == part].set_index("match_id")
    return p.loc[ids, col].to_numpy(float) if col in p else None


def fold_means(season: int):
    """{model: {"inner": (mu, mh, ma), "outer": (...)}} incl. the two ensembles; and model infos."""
    preds, infos = load_fold(season)
    f = D.fold_rows(season)
    ids = {"inner": f["inner_valid"]["match_id"], "outer": f["test"]["match_id"]}
    means = {}
    for name, p in preds.items():
        means[name] = {part: (_arr(p, part, ids[part]), _arr(p, part, ids[part], "mu_home"),
                              _arr(p, part, ids[part], "mu_away")) for part in ids}
    members = [m for m in ENSEMBLE_MEMBERS if m in means]
    y_in = f["inner_valid"]["total_corners"].to_numpy(float)
    P_in = np.column_stack([means[m]["inner"][0] for m in members])
    P_out = np.column_stack([means[m]["outer"][0] for m in members])
    ok = np.isfinite(P_in).all(1)
    w = convex_weights(P_in[ok], y_in[ok])
    means["ens_weighted"] = {"inner": (P_in @ w, None, None), "outer": (P_out @ w, None, None)}
    infos["ens_weighted"] = {"weights": dict(zip(members, w.round(4).tolist()))}
    simple = [m for m in SIMPLE_MEMBERS if m in means]
    means["ens_simple"] = {part: (np.mean([means[m][part][0] for m in simple], 0), None, None) for part in ids}
    infos["ens_simple"] = {"members": simple}
    return means, infos, f


def point_metrics(y, mu) -> dict:
    res = y - mu
    q = pd.qcut(pd.Series(mu).rank(method="first"), 10, labels=False).to_numpy()
    return {"n": int(len(y)), "mae": float(np.abs(res).mean()), "rmse": float(np.sqrt(np.mean(res ** 2))),
            "r2": float(r2_score(y, mu)), "poisson_deviance": pdev(y, mu), "bias": float(mu.mean() - y.mean()),
            "mean_actual": float(y.mean()), "mean_predicted": float(mu.mean()), "sd_predicted": float(mu.std()),
            "corr": float(np.corrcoef(y, mu)[0, 1]),
            "deciles": [{"predicted": float(mu[q == b].mean()), "observed": float(y[q == b].mean())} for b in range(10)]}


def line_block(y, pmf, lines, base_rates) -> tuple[dict, np.ndarray]:
    out, per_match = {}, []
    for line in lines:
        p, o = Q.p_over(pmf, line), y > line
        ll = Q.binary_ll(o, p)
        if line in D.LINES:
            per_match.append(ll)
        b = base_rates[line]
        out[str(line)] = {"log_loss": float(ll.mean()), "brier": float(np.mean((p - o) ** 2)), "ece": Q.ece(o, p),
                          "observed": float(o.mean()), "mean_p": float(p.mean()),
                          "accuracy": float(((p > 0.5) == o).mean()),
                          "base_log_loss": float(Q.binary_ll(o, np.full(len(o), b)).mean()),
                          "reliability": Q.reliability(o, p)}
    return out, np.mean(per_match, 0)


def run() -> dict:
    out = {"folds": {}, "reference": list(REF)}
    p_store = []
    for s in D.FOLDS:
        means, infos, f = fold_means(s)
        va, te = f["inner_valid"], f["test"]
        y_in, y = va["total_corners"].to_numpy(float), te["total_corners"].to_numpy(float)
        train = f["outer_train"]
        base_rates = {l: float((train["total_corners"] > l).mean()) for l in EXT_LINES}
        lrate = {l: train.assign(o=train["total_corners"] > l).groupby("league")["o"].mean() for l in D.LINES}
        phase, league = Q.season_phase(te["date"]), te["league"].to_numpy()
        fold = {"n": int(len(te)), "n_inner": int(len(va)), "infos": infos, "point": {}, "dist": {},
                "baselines": {"base_rate": base_rates, "league_rate_log_loss": {
                    str(l): float(Q.binary_ll(y > l, te["league"].map(lrate[l]).fillna(base_rates[l]).to_numpy()).mean())
                    for l in D.LINES}}}
        per_match = {}
        for name, mm in means.items():
            mu_in, mh_in, ma_in = mm["inner"]
            mu, mh, ma = mm["outer"]
            fold["point"][name] = point_metrics(y, mu)
            dists = TOTAL_DISTS + (SIDE_DISTS if name in SIDE_MODELS else [])
            for dk in dists:
                d = Q.Dispersion(dk).fit(y_in, mu_in, league=va["league"].to_numpy(),
                                         yh=va["home_corners"].to_numpy(float), mh=mh_in,
                                         ya=va["away_corners"].to_numpy(float), ma=ma_in)
                pmf = d.pmf(mu, league=league, mh=mh, ma=ma)
                ls, cr = Q.log_score(pmf, y), Q.crps(pmf, y)
                lines, line_pm = line_block(y, pmf, EXT_LINES, base_rates)
                rec = {"info": d.info, "log_score": float(ls.mean()), "crps": float(cr.mean()),
                       "pit": Q.pit_summary(Q.pit(pmf, y)), "lines": lines,
                       "mean_line_log_loss": float(line_pm.mean()),
                       "mean_line_brier": float(np.mean([lines[str(l)]["brier"] for l in D.LINES])),
                       "by_phase": {}, "by_league": {}}
                for key, lab in (("by_phase", phase), ("by_league", league)):
                    for g in np.unique(lab):
                        m = lab == g
                        rec[key][str(g)] = {"n": int(m.sum()), "mean_line_log_loss": float(line_pm[m].mean()),
                                            "log_score": float(ls[m].mean())}
                # Platt recalibration of the three product lines (fitted on the inner season)
                if dk == "nb_mom":
                    pmf_in = d.pmf(mu_in, league=va["league"].to_numpy(), mh=mh_in, ma=ma_in)
                    cal = []
                    for line in D.LINES:
                        fcal = Q.platt(y_in > line, Q.p_over(pmf_in, line))
                        cal.append(Q.binary_ll(y > line, fcal(Q.p_over(pmf, line))))
                    rec["platt_mean_line_log_loss"] = float(np.mean(cal))
                fold["dist"][f"{name}|{dk}"] = rec
                per_match[(name, dk)] = (ls, line_pm, cr)
                if s == D.CHECK or s in D.SELECT:
                    p_store.append(pd.DataFrame({"season": s, "match_id": te["match_id"].to_numpy(), "model": name,
                                                 "dist": dk, "y": y, **{f"p{l}": Q.p_over(pmf, l) for l in EXT_LINES},
                                                 "mu": mu}))
        ref = per_match[REF]
        for key, (ls, lpm, cr) in per_match.items():
            rec = fold["dist"][f"{key[0]}|{key[1]}"]
            rec["delta_log_score"] = Q.bootstrap_mean(ls - ref[0])
            rec["delta_line_log_loss"] = Q.bootstrap_mean(lpm - ref[1])
            rec["delta_crps"] = Q.bootstrap_mean(cr - ref[2])
        out["folds"][str(s)] = fold
        log.info("fold %d evaluated (%d model x distribution pairs)", s, len(per_match))
    out["selection"] = select(out)
    D.save_result("metrics.json", out)
    pd.concat(p_store, ignore_index=True).to_parquet(D.CACHE / "line_probs.parquet")
    tables(out)
    return out


def select(out: dict) -> dict:
    keys = list(out["folds"][str(D.SELECT[0])]["dist"])
    table = {}
    for k in keys:
        rows = {s: out["folds"][str(s)]["dist"][k] for s in D.FOLDS}
        if k == "|".join(REF):
            continue
        conv = {s: rows[s]["delta_line_log_loss"]["ci95"][1] < 0 and rows[s]["delta_log_score"]["mean"] < 0
                for s in D.SELECT}
        weak = {s: rows[s]["delta_line_log_loss"]["mean"] < 0 and rows[s]["delta_log_score"]["mean"] < 0
                for s in D.SELECT}
        notworse = {s: rows[s]["delta_line_log_loss"]["mean"] <= 0 and rows[s]["delta_log_score"]["mean"] <= 0
                    for s in (D.INFO, D.CHECK)}
        table[k] = {"convincing_both": all(conv.values()), "point_better_both": all(weak.values()),
                    "not_worse_2025_2026": all(notworse.values()),
                    "pooled_select_line_ll": float(np.mean([rows[s]["mean_line_log_loss"] for s in D.SELECT])),
                    "delta_line_ll": {str(s): rows[s]["delta_line_log_loss"] for s in D.FOLDS},
                    "delta_log_score": {str(s): rows[s]["delta_log_score"]["mean"] for s in D.FOLDS}}
    eligible = [k for k, v in table.items() if v["convincing_both"] and v["not_worse_2025_2026"]]
    weak = [k for k, v in table.items() if v["point_better_both"]]
    best = sorted(table, key=lambda k: table[k]["pooled_select_line_ll"])[:8]
    return {"table": table, "eligible_before_betting": eligible, "point_better_both": weak,
            "best_pooled_selection": best, "rule": __doc__.split("Adoption rule")[1].strip()}


def tables(out: dict) -> None:
    """CSV tables: one row per model x distribution x fold."""
    rows = []
    for s, fold in out["folds"].items():
        for k, r in fold["dist"].items():
            name, dk = k.split("|")
            pm = fold["point"][name]
            rows.append({"season": int(s), "model": name, "dist": dk, "mae": pm["mae"], "rmse": pm["rmse"],
                         "r2": pm["r2"], "poisson_deviance": pm["poisson_deviance"], "bias": pm["bias"],
                         "log_score": r["log_score"], "crps": r["crps"], "mean_line_log_loss": r["mean_line_log_loss"],
                         **{f"ll_{l}": r["lines"][str(l)]["log_loss"] for l in D.LINES},
                         **{f"ece_{l}": r["lines"][str(l)]["ece"] for l in D.LINES},
                         "pit_chi2": r["pit"]["chi2"], "d_line_ll": r["delta_line_log_loss"]["mean"],
                         "d_line_ll_lo": r["delta_line_log_loss"]["ci95"][0],
                         "d_line_ll_hi": r["delta_line_log_loss"]["ci95"][1],
                         "d_log_score": r["delta_log_score"]["mean"]})
    pd.DataFrame(rows).to_csv(D.RESULTS / "models_by_fold.csv", index=False, float_format="%.5f")
