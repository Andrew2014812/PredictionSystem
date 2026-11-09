"""Prediction-uncertainty study: does knowing how reliable an estimate is help the Main?

Steps (results/*.json; every rule is fixed before it is applied and chosen on 2023/24 + 2024/25 only):

1. measures      size of every uncertainty measure per season; ensemble spread vs the calibration error
2. predictive    do the measures flag Mains that lose more often than stated? logistic regression
                 won ~ logit(p) + z(u) per season and pooled over the selection seasons (bootstrap CI, Holm),
                 AUC of u for a miss, over-estimation gap by quintile of u (bootstrap CI)
3. filters       "drop the most uncertain Mains": threshold = the (1 - q) quantile of u over the Mains of the
                 same season played on EARLIER days (warm-up: the first 300 Mains are never dropped);
                 kept-set gap, hit rate and ROI at real prices against all Mains, paired bootstrap by match;
                 promising only if the improvement holds in both selection seasons and the Holm-corrected
                 pooled p < 0.05
4. calibration   bootstrap 95 % band of the reliability curve of the Main per season
5. intervals     ensemble interval of the Main probability vs the observed hit rate of its probability bin
6. bagging       the ensemble mean as a prediction: Main gap and 1X2 log loss against the single model
7. stability     2026/27 recorded Mains: version changes / probability drift vs the gap
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import common as C
from . import data as D

MEASURES = {"u_seed": "разброс по сидам", "u_boot": "разброс бутстрепа", "u_sub": "разброс подвыборок",
            "u_1x2": "разброс 1X2 (бутстреп)", "u_missing": "доля пропусков", "u_va": "ширина Венна–Аберса",
            "conf_size": "размер конформного множества 1X2"}
DROP = (0.10, 0.20, 0.30)
WARMUP = 300
EDGES = np.arange(0.40, 0.90, 0.05)


def load_all() -> dict[int, pd.DataFrame]:
    return {s: D.build(s) for s in C.SEASONS}


def decided(m: pd.DataFrame) -> pd.DataFrame:
    return m.loc[m["y"].notna()].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ 1. measures
def describe(ms: dict) -> dict:
    out = {}
    for s, m in ms.items():
        d = decided(m)
        row = {"mains": int(len(m)), "decided": int(len(d)), "priced_bets": int(m["profit"].notna().sum()),
               "gap": C.gap_ci(d["pc"], d["y"]), "roi": C.mean_ci(m["profit"])}
        for col in MEASURES:
            v = m[col].to_numpy(float)
            v = v[np.isfinite(v)]
            row[col] = {"median": float(np.median(v)) if len(v) else None,
                        "p10": float(np.quantile(v, .1)) if len(v) else None,
                        "p90": float(np.quantile(v, .9)) if len(v) else None, "n": int(len(v))}
        row["members"] = {f: float(m[f"k_{f}"].median()) for f in C.FAMILIES}
        cols = [c for c in MEASURES if m[c].notna().sum() > 50]
        row["spearman"] = m[cols].rank().corr().round(3).to_dict()
        row["conf_size_share"] = {str(int(k)): float(v) for k, v in
                                  m["conf_size"].value_counts(normalize=True).sort_index().items()}
        out[str(s)] = row
    return out


# ------------------------------------------------------------------------------------------------ 2. predictive value
def _logit_fit(y, x_logit, z) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([x_logit, z])
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(X, y)
    return np.r_[lr.intercept_, lr.coef_[0]]


def beta_u(d: pd.DataFrame, col: str, n_boot: int = 500) -> dict:
    ok = d[col].notna().to_numpy()
    x = C.logit(d.loc[ok, "pc"].to_numpy())
    u = d.loc[ok, col].to_numpy(float)
    sd = u.std()
    if len(u) < 200 or sd == 0:
        return {"n": int(len(u))}
    z = (u - u.mean()) / sd
    y = d.loc[ok, "y"].to_numpy(float)
    est = _logit_fit(y, x, z)
    rng = np.random.default_rng(C.SEED)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        bs.append(_logit_fit(y[i], x[i], z[i])[2])
    bs = np.array(bs)
    p = float(min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))
    miss = 1 - y
    return {"n": int(len(y)), "beta_u": float(est[2]), "beta_u_ci95": [float(np.quantile(bs, .025)),
                                                                        float(np.quantile(bs, .975))],
            "p": p, "beta_logit_p": float(est[1]), "auc_u_for_miss": C.roc_auc(u, miss),
            "auc_p_for_miss": C.roc_auc(-d.loc[ok, "pc"].to_numpy(), miss)}


def quintiles(d: pd.DataFrame, col: str, k: int = 5) -> list[dict]:
    ok = d[col].notna()
    dd = d.loc[ok]
    if len(dd) < 200:
        return []
    q = pd.qcut(dd[col].rank(method="first"), k, labels=False)
    out = []
    for b in range(k):
        x = dd.loc[q == b]
        g = C.gap_ci(x["pc"], x["y"], n_boot=1000)
        out.append({"q": b + 1, "n": int(len(x)), "u_mean": float(x[col].mean()), "mean_p": g["mean_p"],
                    "hit_rate": g["hit_rate"], "gap": g["mean"], "gap_ci95": g["ci95"],
                    "roi": float(x["profit"].mean()) if x["profit"].notna().any() else None,
                    "bets": int(x["profit"].notna().sum())})
    return out


def predictive(ms: dict) -> dict:
    out = {"per_season": {}, "pooled_select": {}, "quintiles": {}}
    for s, m in ms.items():
        d = decided(m)
        res = {c: beta_u(d, c) for c in MEASURES}
        h = C.holm({c: r.get("p") for c, r in res.items()})
        for c in res:
            res[c]["p_holm_season"] = h.get(c)
        out["per_season"][str(s)] = res
        out["quintiles"][str(s)] = {c: quintiles(d, c) for c in ("u_boot", "u_seed", "u_va", "u_missing", "u_1x2")}
        dd = d.loc[d[["u_1x2"]].notna().all(1)]
        out.setdefault("frozen_scale", {})[str(s)] = {c: {"mean": float(dd[c].mean()), "sd": float(dd[c].std()),
                                                          "q80": float(dd[c].quantile(0.8))} for c in MEASURES}
    # pooled selection seasons: z-scored within season
    frames = []
    for s in C.SELECT:
        d = decided(ms[s]).copy()
        for c in MEASURES:
            v = d[c].astype(float)
            d[c] = (v - v.mean()) / (v.std() or 1)
        frames.append(d)
    pool = pd.concat(frames, ignore_index=True)
    res = {c: beta_u(pool, c) for c in MEASURES}
    h = C.holm({c: r.get("p") for c, r in res.items()})
    for c in res:
        res[c]["p_holm"] = h.get(c)
    out["pooled_select"] = res
    out["n_tests"] = len(MEASURES)
    return out


# ------------------------------------------------------------------------------------------------ 3. filters
def expanding_threshold(m: pd.DataFrame, col: str, drop: float) -> np.ndarray:
    """Per Main: the (1 - drop) quantile of ``col`` over Mains of EARLIER days of the season (NaN in warm-up)."""
    days = m["date"].dt.normalize().to_numpy()
    u = m[col].to_numpy(float)
    thr = np.full(len(m), np.nan)
    for day in np.unique(days):
        past = u[(days < day) & np.isfinite(u)]
        if len(past) >= WARMUP:
            thr[days == day] = np.quantile(past, 1 - drop)
    return thr


def filter_eval(m: pd.DataFrame, col: str, drop: float, boot: bool = True) -> dict:
    thr = expanding_threshold(m, col, drop)
    u = m[col].to_numpy(float)
    keep = ~(np.isfinite(thr) & np.isfinite(u) & (u > thr))
    y, pc, profit = m["y"].to_numpy(float), m["pc"].to_numpy(float), m["profit"].to_numpy(float)
    dec = np.isfinite(y)
    out = {"dropped_share": float(1 - keep.mean()), "kept": int(keep.sum()),
           "gap_all": float(np.mean(pc[dec] - y[dec])), "gap_kept": float(np.mean(pc[dec & keep] - y[dec & keep])),
           "gap_dropped": float(np.mean(pc[dec & ~keep] - y[dec & ~keep])) if (dec & ~keep).any() else None,
           "hit_all": float(y[dec].mean()), "hit_kept": float(y[dec & keep].mean()),
           "roi_all": float(np.nanmean(profit)), "roi_kept": float(np.nanmean(np.where(keep, profit, np.nan))),
           "roi_dropped": float(np.nanmean(np.where(~keep, profit, np.nan))) if (~keep & np.isfinite(profit)).any()
           else None}
    if not boot:
        return out
    prof_k = np.where(keep, profit, np.nan)
    out["roi_diff"] = C.paired_nanmean_diff(profit, prof_k)
    a = np.where(dec, pc - y, np.nan)
    b = np.where(dec & keep, pc - y, np.nan)
    diffs = []
    for idx in C.boot_idx(len(a)):
        with np.errstate(all="ignore"):
            diffs.append(np.abs(np.nanmean(b[idx], 1)) - np.abs(np.nanmean(a[idx], 1)))
    dd = np.concatenate(diffs)
    out["abs_gap_diff"] = {"diff": float(abs(np.nanmean(b)) - abs(np.nanmean(a))),
                           "ci95": [float(np.nanquantile(dd, .025)), float(np.nanquantile(dd, .975))],
                           "p": float(min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0))))}
    return out


def filters(ms: dict) -> dict:
    cols = [c for c in MEASURES if c != "conf_size"]
    out = {"variants": {}, "rule": "перспективен, если ΔROI > 0 (или Δ|gap| < 0) в обоих сезонах отбора и p Холма "
                                   "объединённого теста < 0.05"}
    pvals_roi, pvals_gap = {}, {}
    for col in cols:
        for q in DROP:
            key = f"{col}|{int(q * 100)}"
            res = {str(s): filter_eval(ms[s], col, q) for s in C.SEASONS}
            pool = pd.concat([ms[s].assign(_s=s) for s in C.SELECT], ignore_index=True)
            # pooled paired test over the two selection seasons (thresholds stay per season)
            parts = [ms[s] for s in C.SELECT]
            keep_all, prof_all, a_all, b_all = [], [], [], []
            for m in parts:
                thr = expanding_threshold(m, col, q)
                u = m[col].to_numpy(float)
                keep = ~(np.isfinite(thr) & np.isfinite(u) & (u > thr))
                prof = m["profit"].to_numpy(float)
                y, pc = m["y"].to_numpy(float), m["pc"].to_numpy(float)
                dec = np.isfinite(y)
                prof_all.append(prof)
                keep_all.append(np.where(keep, prof, np.nan))
                a_all.append(np.where(dec, pc - y, np.nan))
                b_all.append(np.where(dec & keep, pc - y, np.nan))
            prof, profk = np.concatenate(prof_all), np.concatenate(keep_all)
            roi_pool = C.paired_nanmean_diff(prof, profk)
            a, b = np.concatenate(a_all), np.concatenate(b_all)
            diffs = []
            for idx in C.boot_idx(len(a)):
                with np.errstate(all="ignore"):
                    diffs.append(np.abs(np.nanmean(b[idx], 1)) - np.abs(np.nanmean(a[idx], 1)))
            dd = np.concatenate(diffs)
            gap_pool = {"diff": float(abs(np.nanmean(b)) - abs(np.nanmean(a))),
                        "ci95": [float(np.nanquantile(dd, .025)), float(np.nanquantile(dd, .975))],
                        "p": float(min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0))))}
            del pool
            out["variants"][key] = {"measure": col, "drop": q, "seasons": res, "pooled_select_roi": roi_pool,
                                    "pooled_select_abs_gap": gap_pool}
            pvals_roi[key], pvals_gap[key] = roi_pool["p"], gap_pool["p"]
    h_roi, h_gap = C.holm(pvals_roi), C.holm(pvals_gap)
    best_roi = best_gap = None
    for key, v in out["variants"].items():
        v["pooled_select_roi"]["p_holm"] = h_roi.get(key)
        v["pooled_select_abs_gap"]["p_holm"] = h_gap.get(key)
        sel = [v["seasons"][str(s)] for s in C.SELECT]
        v["roi_up_both"] = all(r["roi_diff"]["diff"] > 0 for r in sel)
        v["gap_down_both"] = all(r["abs_gap_diff"]["diff"] < 0 for r in sel)
        v["promising_roi"] = bool(v["roi_up_both"] and h_roi.get(key, 1) < 0.05)
        v["promising_gap"] = bool(v["gap_down_both"] and h_gap.get(key, 1) < 0.05)
        score_roi = min(r["roi_diff"]["diff"] for r in sel)
        score_gap = max(r["abs_gap_diff"]["diff"] for r in sel)
        if best_roi is None or score_roi > out["variants"][best_roi]["_score_roi"]:
            best_roi = key
        if best_gap is None or score_gap < out["variants"][best_gap]["_score_gap"]:
            best_gap = key
        v["_score_roi"], v["_score_gap"] = score_roi, score_gap
    out["n_variants"] = len(out["variants"])
    out["best_roi"], out["best_gap"] = best_roi, best_gap
    out["any_promising"] = any(v["promising_roi"] or v["promising_gap"] for v in out["variants"].values())
    return out


# ------------------------------------------------------------------------------------------------ 4.-6.
def calibration_band(ms: dict) -> dict:
    out = {}
    for s, m in ms.items():
        d = decided(m)
        p, y = d["pc"].to_numpy(float), d["y"].to_numpy(float)
        idx = np.digitize(p, EDGES) - 1
        rows = []
        rng = np.random.default_rng(C.SEED)
        for b in range(len(EDGES) - 1):
            k = np.flatnonzero(idx == b)
            if len(k) < 30:
                continue
            bs = [y[rng.choice(k, len(k))].mean() for _ in range(1000)]
            rows.append({"lo": float(EDGES[b]), "hi": float(EDGES[b + 1]), "n": int(len(k)),
                         "predicted": float(p[k].mean()), "observed": float(y[k].mean()),
                         "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         "inside": bool(np.quantile(bs, .025) <= p[k].mean() <= np.quantile(bs, .975))})
        out[str(s)] = rows
    return out


def intervals(ms: dict) -> dict:
    """Ensemble interval [min, max] of the Main probability over the bootstrap members vs the systematic gap."""
    out = {}
    for s, m in ms.items():
        d = decided(m)
        width = (d["hi_boot"] - d["lo_boot"]).to_numpy(float)
        sd = d["u_boot"].to_numpy(float)
        # how often the observed hit rate of the Main's 0.05-bin lies inside the mean member interval of the bin
        idx = np.digitize(d["pc"], EDGES) - 1
        inside = []
        for b in range(len(EDGES) - 1):
            k = idx == b
            if k.sum() < 30:
                continue
            lo, hi = d.loc[k, "lo_boot"].mean(), d.loc[k, "hi_boot"].mean()
            obs = d.loc[k, "y"].mean()
            inside.append({"lo": float(EDGES[b]), "n": int(k.sum()), "member_lo": float(lo), "member_hi": float(hi),
                           "observed": float(obs), "inside": bool(lo <= obs <= hi)})
        gap = C.gap_ci(d["pc"], d["y"], n_boot=1000)
        out[str(s)] = {"median_sd": float(np.nanmedian(sd)), "median_width": float(np.nanmedian(width)),
                       "p90_width": float(np.nanquantile(width, .9)), "gap": gap["mean"], "gap_ci95": gap["ci95"],
                       "bins": inside, "share_bins_inside": float(np.mean([b["inside"] for b in inside]))
                       if inside else None,
                       "va_median_width": float(np.nanmedian(d["u_va"])),
                       "va_contains_hit_share": None}
    return out


def bagging(ms: dict) -> dict:
    out = {}
    for s, m in ms.items():
        d = decided(m)
        res = {}
        for fam in C.FAMILIES:
            pm = d[f"pm_{fam}"].to_numpy(float)
            pc_ens = pm / np.clip(1 - d["p_push"].fillna(0).to_numpy(float), 1e-6, None)
            ok = np.isfinite(pc_ens)
            g_single = C.gap_ci(d.loc[ok, "pc"], d.loc[ok, "y"], n_boot=1000)
            g_ens = C.gap_ci(pc_ens[ok], d.loc[ok, "y"], n_boot=1000)
            res[fam] = {"n": int(ok.sum()), "gap_single": g_single["mean"], "gap_ensemble": g_ens["mean"],
                        "gap_ensemble_ci95": g_ens["ci95"]}
        out[str(s)] = res
    # 1X2 log loss of the bootstrap-ensemble mean vs the single walk-forward model (history seasons)
    from . import ensemble as E
    for s in C.SEASONS:
        if s == C.CURRENT:
            continue
        o = D.outer_1x2(s).set_index("match_id")
        for fam in ("seed", "boot"):
            ens, _ = E.load(fam, s)
            if ens.empty:
                continue
            mean = ens.groupby("match_id")[["pH", "pD", "pA"]].mean().reindex(o.index)
            ok = mean.notna().all(1).to_numpy()
            y = o["y"].to_numpy(int)[ok]
            p1 = o[["pH", "pD", "pA"]].to_numpy(float)[ok]
            p2 = mean.to_numpy(float)[ok]
            l1 = -np.log(np.clip(p1[np.arange(len(y)), y], 1e-15, 1))
            l2 = -np.log(np.clip(p2[np.arange(len(y)), y], 1e-15, 1))
            out[str(s)][f"logloss_1x2_{fam}"] = {"single": float(l1.mean()), "ensemble": float(l2.mean()),
                                                 "diff": C.mean_ci(l2 - l1, n_boot=1000)}
    return out


def stability(ms: dict) -> dict:
    m = ms[C.CURRENT]
    d = decided(m)
    d = d.loc[d["source"] == "live"]
    if "n_versions" not in d:
        return {}
    out = {"recorded_decided": int(len(d)), "n_versions_median": float(d["n_versions"].median()),
           "main_changed_share": float(d["main_changed"].mean()),
           "p_drift_median": float(d["p_drift"].median())}
    groups = {}
    for name, mask in (("Main не менялся", d["main_changed"] == 0), ("Main менялся", d["main_changed"] == 1)):
        x = d.loc[mask]
        if len(x) >= 30:
            g = C.gap_ci(x["pc"], x["y"], n_boot=1000)
            groups[name] = {"n": int(len(x)), "gap": g["mean"], "gap_ci95": g["ci95"], "hit_rate": g["hit_rate"],
                            "mean_p": g["mean_p"], "roi": float(x["profit"].mean()) if x["profit"].notna().any()
                            else None}
    q = d["p_drift"].quantile([0, .5, 1]).to_numpy()
    for name, mask in (("дрейф ≤ медианы", d["p_drift"] <= q[1]), ("дрейф > медианы", d["p_drift"] > q[1])):
        x = d.loc[mask]
        if len(x) >= 30:
            g = C.gap_ci(x["pc"], x["y"], n_boot=1000)
            groups[name] = {"n": int(len(x)), "gap": g["mean"], "gap_ci95": g["ci95"], "hit_rate": g["hit_rate"],
                            "mean_p": g["mean_p"], "roi": float(x["profit"].mean()) if x["profit"].notna().any()
                            else None}
    out["groups"] = groups
    out["beta"] = {c: beta_u(d, c) for c in ("p_drift", "p_sd_versions", "n_versions", "main_changed")}
    return out


def decisions(pred: dict, filt: dict) -> dict:
    pooled = pred["pooled_select"]
    sig = [c for c, r in pooled.items() if r.get("p_holm") is not None and r["p_holm"] < 0.05]
    cur = pred["per_season"][str(C.CURRENT)]
    signal = [c for c, r in cur.items() if r.get("p_holm_season") is not None and r["p_holm_season"] < 0.05
              and r.get("beta_u", 0) < 0 and c not in sig]
    return {
        "current_signal": {"measures": signal,
                           "decision": "B (низкая априорная вероятность)" if signal else None,
                           "why": "сигнал найден только на уже просмотренном 2026/27 (Холм по мерам сезона), на "
                                  "сезонах отбора не подтверждён — проверяется только вперёд" if signal else
                                  "на 2026/27 нет значимых мер после поправки Холма"},
        "flag_on_site": {"significant_measures": sig,
                         "decision": "B" if sig else "C",
                         "why": "мера неопределённости значимо (Холм) предсказывает проигрыш сверх p на сезонах отбора"
                         if sig else "ни одна мера не предсказывает проигрыш сверх p после поправки Холма"},
        "filter_main": {"decision": "B" if filt["any_promising"] else "C",
                        "best_roi": filt["best_roi"], "best_gap": filt["best_gap"]},
    }


def main() -> None:
    C.setup_logging("unc-run")
    t0 = time.time()
    ms = load_all()
    C.save_json("measures.json", describe(ms))
    pred = predictive(ms)
    C.save_json("predictive.json", pred)
    filt = filters(ms)
    for v in filt["variants"].values():
        v.pop("_score_roi", None)
        v.pop("_score_gap", None)
    C.save_json("filters.json", filt)
    C.save_json("calibration_band.json", calibration_band(ms))
    C.save_json("intervals.json", intervals(ms))
    C.save_json("bagging.json", bagging(ms))
    C.save_json("stability.json", stability(ms))
    C.save_json("decisions.json", decisions(pred, filt))
    infos = {}
    from . import ensemble as E
    for fam in C.FAMILIES:
        for s in C.SEASONS:
            _, inf = E.load(fam, s)
            infos[f"{fam}|{s}"] = {"members": len(inf), "seconds": float(sum(i["seconds"] for i in inf)),
                                   "train_rows": [i["train_rows"] for i in inf]}
    C.save_json("run_info.json", {"environment": C.environment(), "data_version": C.data_version(),
                                  "families": C.FAMILIES, "sub_fraction": C.SUB_FRACTION, "warmup": WARMUP,
                                  "drop_shares": DROP, "ensembles": infos,
                                  "conformal": {str(s): json.loads((C.CACHE / f"mains_{s}.json").read_text())
                                                for s in C.SEASONS},
                                  "minutes": round((time.time() - t0) / 60, 2)})


if __name__ == "__main__":
    main()
