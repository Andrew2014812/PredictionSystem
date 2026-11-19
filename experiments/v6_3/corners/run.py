"""v6.3 corners research: every experiment after the fits (light; the fits are in fit.py, run via heavy.sh).

    .venv/bin/python -m experiments.v6_3.corners.data
    experiments/v6_3/heavy.sh 14400 experiments/v6_3/corners/logs/fit_all.log .venv/bin/python -m experiments.v6_3.corners.fit all
    experiments/v6_3/heavy.sh 1800 experiments/v6_3/corners/logs/fit_wf_extra.log .venv/bin/python -m experiments.v6_3.corners.fit wf --models xgb_mkt,lgbm_mkt,cat_mkt,xgb_ha
    .venv/bin/python -m experiments.v6_3.corners.run
    .venv/bin/python -m experiments.v6_3.corners.build_pdf

Writes results/*.json|csv, figures/*.png. Nothing about 2026/27 is used to choose a model, a distribution, a
calibration or ensemble weights: all of them are chosen on 2025/26 (models fitted on 2021/22–2024/25).
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.corners import betting as B
from experiments.v6_3.corners import data as D
from experiments.v6_3.corners import dists as Q
from experiments.v6_3.corners import evaluate as E
from experiments.v6_3.corners import models as M
from experiments.v6_3.replay import evaluate as EV

R = D.RESULTS
SEL_CUT, EV_CUT = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-07-01")
WF_CUTS = [pd.Timestamp(x) for x in ("2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01")]


# ---------------------------------------------------------------------------------------------- 1. data
def coverage(frame: pd.DataFrame) -> dict:
    f = frame.loc[frame["played"].astype(bool)]
    g = f.groupby(["league", "season"])
    t = g.agg(matches=("match_id", "size"), with_corners=("total_corners", "count"),
              mean=("total_corners", "mean"), var=("total_corners", "var"), home=("home_corners", "mean"),
              away=("away_corners", "mean"), shots=("home_shots", "count"), fouls=("home_fouls", "count")).reset_index()
    t["share"] = t["with_corners"] / t["matches"]
    t["disp"] = t["var"] / t["mean"]
    t.to_csv(R / "coverage_league_season.csv", index=False)
    s = f.groupby("season").agg(matches=("match_id", "size"), with_corners=("total_corners", "count"),
                                mean=("total_corners", "mean"), var=("total_corners", "var"),
                                home=("home_corners", "mean"), away=("away_corners", "mean"))
    s["corr_home_away"] = f.groupby("season").apply(lambda d: d[["home_corners", "away_corners"]].corr().iloc[0, 1])
    s["disp"] = s["var"] / s["mean"]
    lg = f.groupby("league").agg(matches=("match_id", "size"), mean=("total_corners", "mean"),
                                 var=("total_corners", "var"))
    lg["disp"] = lg["var"] / lg["mean"]
    return {"by_season": {str(k): v for k, v in s.to_dict(orient="index").items()},
            "by_league": lg.to_dict(orient="index"),
            "total_played": int(len(f)), "total_with_corners": int(f["total_corners"].notna().sum()),
            "leagues": int(f["league"].nunique()), "seasons": sorted(int(x) for x in f["season"].unique())}


def inventory() -> dict:
    """What statistics exist locally (no API is called)."""
    fd_cols = set()
    per_col = {}
    for path in sorted(glob.glob(str(C.ROOT / "local_data/results/*/*.csv"))):
        cols = pd.read_csv(path, nrows=0, encoding="latin1").columns
        fd_cols |= set(cols)
        season = Path(path).parent.name
        for c in ("HC", "HS", "HST", "HF", "HY", "HxG", "Referee"):
            per_col.setdefault(c, set())
            if c in cols:
                per_col[c].add(season)
    api = {}
    for path in glob.glob(str(C.ROOT / "local_data/api/api_football/*.json")):
        try:
            d = json.loads(Path(path).read_text())
        except Exception:
            api["unreadable"] = api.get("unreadable", 0) + 1
            continue
        ep = d.get("url", "").rsplit("/", 1)[-1]
        api[ep] = api.get(ep, 0) + 1
    live = pd.read_parquet(C.ROOT / "local_data/results_live/fixtures.parquet")
    odds = pd.read_parquet(C.DATA_FILES["odds"], columns=["market", "provider", "bookmaker"])
    corner_odds = odds.loc[odds["market"].str.startswith("CORNERS")].groupby(["provider", "bookmaker"]).size()
    wanted = {"blocked shots": ["blocked"], "crosses": ["cross"], "wide attacks / flank play": ["wide", "flank"],
              "full-backs / wingers": ["position", "lineup"], "possession / style": ["possession", "pass"]}
    missing = {k: not any(any(w.lower() in c.lower() for w in ws) for c in fd_cols) for k, ws in wanted.items()}
    return {"football_data_columns": sorted(fd_cols),
            "football_data_seasons_with": {k: sorted(v) for k, v in per_col.items()},
            "api_football_cached_endpoints": api,
            "results_live": {"rows": int(len(live)), "played": int(live["played"].sum()),
                             "with_corners": int(pd.to_numeric(live["home_corners"], errors="coerce").notna().sum()),
                             "providers": live["provider"].value_counts().to_dict()},
            "corner_odds_rows": {f"{a}/{b}": int(n) for (a, b), n in corner_odds.items()},
            "idea_without_history": missing}


def feature_signal(frame: pd.DataFrame, sets: dict) -> pd.DataFrame:
    """Correlation of each feature with total corners on 2021/22–2024/25 (before the selection season)."""
    tr = frame.loc[(frame["season"] < D.SELECT) & frame["total_corners"].notna()]
    y = tr["total_corners"].astype(float)
    rows = []
    for c in sets["stats_mkt"]:
        v = tr[c].astype(float)
        ok = v.notna()
        r = float(np.corrcoef(v[ok], y[ok])[0, 1]) if ok.sum() > 500 and v[ok].std() > 0 else np.nan
        rows.append({"feature": c, "group": "market" if c in sets["market"] else "new" if c in sets["new"] else "prod",
                     "coverage": float(ok.mean()), "corr": r})
    out = pd.DataFrame(rows).sort_values("corr", key=lambda s: -s.abs())
    out.to_csv(R / "feature_signal.csv", index=False)
    return out


# ---------------------------------------------------------------------------------------------- 2. selection
def crossfit_ensemble(sel: pd.DataFrame) -> np.ndarray:
    half = (sel["date"] < pd.Timestamp(f"{D.SELECT + 1}-01-01")).to_numpy()
    P = sel[[f"mu_{k}" for k in E.ENS_PARTS]].to_numpy(float)
    y = sel["total_corners"].to_numpy(float)
    out = np.zeros(len(sel))
    for fm in (half, ~half):
        w = M.convex_weights(P[fm], y[fm])
        out[~fm] = P[~fm] @ w
    return out


def mean_table(t: pd.DataFrame, models: list[str], ref: str = "xgb_prod") -> dict:
    y = t["total_corners"].to_numpy(float)
    out = {}
    for m in models:
        mu = t[f"mu_{m}"].to_numpy(float)
        res = E.point_metrics(y, mu)
        if m != ref:
            mu0 = t[f"mu_{ref}"].to_numpy(float)
            res["dev_diff"] = C.paired_mean_diff(E.unit_poisson_dev(y, mu), E.unit_poisson_dev(y, mu0))
            res["mae_diff"] = C.paired_mean_diff(np.abs(y - mu), np.abs(y - mu0))
        out[m] = res
    return out


def pover_of(pmf: np.ndarray) -> dict:
    return {line: Q.p_over(pmf, line) for line in D.LINES}


def main() -> None:
    started = time.time()
    log = C.setup_logging("corners.run", D.LOGS / "run.log")
    frame, sets = D.load_frame()
    r_prod = E.prod_r()
    out: dict = {"r_prod": r_prod}

    log.info("1. data")
    out["coverage"] = coverage(frame)
    C.write_json(R / "data_inventory.json", inventory())
    feature_signal(frame, sets)
    out["frame_info"] = C.read_json(R / "frame_info.json")

    log.info("2. selection season 2025/26")
    sel_all = E.stage_table("select", frame, SEL_CUT)
    sel = E.scored(sel_all)
    y_sel = sel["total_corners"].to_numpy(float)
    w_full = M.convex_weights(sel[[f"mu_{k}" for k in E.ENS_PARTS]].to_numpy(float), y_sel)
    weights = dict(zip(E.ENS_PARTS, w_full.tolist()))
    sel = E.add_ensembles(sel, weights)
    sel["mu_ens_w"] = crossfit_ensemble(sel)                   # honest (cross-fitted) score of the weighted blend
    models = ["xgb_prod", "xgb_stats", "xgb_mkt", "lgbm_mkt", "cat_mkt", "xgb_ha", "rating", "glm", "naive",
              "ens_w", "ens_eq"]
    out["select_mean"] = mean_table(sel, models)
    out["select_mean_holm"] = C.holm({k: v["dev_diff"]["p"] for k, v in out["select_mean"].items() if "dev_diff" in v})
    out["ensemble_weights"] = weights
    cand = [m for m in models if m != "naive"]
    best_model = min(cand, key=lambda m: out["select_mean"][m]["poisson_dev"])
    out["best_model"] = best_model
    log.info("best mean model on 2025/26: %s", best_model)

    dist_models = list(dict.fromkeys(["xgb_prod", best_model, "ens_w", "xgb_ha", "rating"]))
    sel_d = {}
    for m in dist_models:
        for d in E.dist_names(m):
            t0 = time.time()
            sel_d[f"{m}|{d}"] = E.crossfit_dist(sel, m, d, r_prod)
            log.info("crossfit %s|%s %s (%.0fs)", m, d, sel_d[f"{m}|{d}"], time.time() - t0)
    out["select_dist"] = sel_d
    best_combo = min(sel_d, key=lambda k: sel_d[k]["log_score_cf"])
    bm, bd = best_combo.split("|")
    out["best_combo"] = {"model": bm, "dist": bd}
    best_dist_prod = min([k for k in sel_d if k.startswith("xgb_prod|")], key=lambda k: sel_d[k]["log_score_cf"])
    out["best_dist_for_prod"] = best_dist_prod.split("|")[1]
    log.info("best combination on 2025/26: %s; best distribution for production mean: %s", best_combo, best_dist_prod)

    # final distribution parameters (whole 2025/26)
    params = {}
    for key in sel_d:
        m, d = key.split("|")
        a = E.dist_args(sel, m)
        params[key] = E.fit_dist(d, y_sel, r_prod=r_prod, **a)
    out["dist_params"] = params
    if "xgb_ha|conv_frank" in params:
        out["frank_kendall_tau"] = Q.kendall_tau_frank(params["xgb_ha|conv_frank"]["theta"])
    if "rating|conv_frank" in params:
        out["frank_kendall_tau_rating"] = Q.kendall_tau_frank(params["rating|conv_frank"]["theta"])

    # calibration of the lines, chosen on 2025/26 (cross-fitted)
    p_sel_best = pover_of(E.make_pmf(bd, params[best_combo], **E.dist_args(sel, bm)))
    out["select_calibration"] = {k: E.crossfit_calibration(sel, p_sel_best, k) for k in E.CALIB}
    cal_choice = min(out["select_calibration"], key=lambda k: out["select_calibration"][k]["ll_mean_cf"])
    cal_keep = out["select_calibration"][cal_choice]["ll_mean_cf"] < out["select_calibration"][cal_choice]["ll_mean_raw"]
    out["calibration_choice"] = {"method": cal_choice, "kept": bool(cal_keep)}

    # dispersion diagnostics on 2025/26: empirical variance vs mean in bins of μ (best model)
    mu_b = sel[f"mu_{bm}"].to_numpy(float)
    bins = np.quantile(mu_b, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, mu_b, side="right") - 1, 0, 9)
    out["dispersion_bins"] = [{"mu": float(mu_b[idx == b].mean()), "mean_y": float(y_sel[idx == b].mean()),
                               "var_y": float(y_sel[idx == b].var()), "n": int((idx == b).sum())} for b in range(10)]

    log.info("3. evaluation season 2026/27")
    ev_all = E.stage_table("eval", frame, EV_CUT)
    ev_all = E.add_ensembles(ev_all, weights)
    ev = E.scored(ev_all)
    y = ev["total_corners"].to_numpy(float)
    out["eval_mean"] = mean_table(ev, models + ["xgb_prod_refit"])
    out["eval_mean_holm"] = C.holm({k: v["dev_diff"]["p"] for k, v in out["eval_mean"].items() if "dev_diff" in v})
    out["eval_mean"]["season_mean_in_sample"] = {"mae": float(np.abs(y - y.mean()).mean())}

    pmf_prod = E.make_pmf("nb_prod", {"r": r_prod}, **E.dist_args(ev, "xgb_prod"))
    ls_prod = Q.log_score(pmf_prod, y)
    pov_prod = pover_of(pmf_prod)
    ev_d = {}
    for key, prm in params.items():
        m, d = key.split("|")
        pmf = E.make_pmf(d, prm, **E.dist_args(ev, m))
        res = E.dist_metrics(y, pmf)
        res["log_score_diff"] = C.paired_mean_diff(Q.log_score(pmf, y), ls_prod)
        res["lines_diff"] = E.paired_lines(y, pover_of(pmf), pov_prod)
        ev_d[key] = res
    out["eval_dist"] = ev_d
    out["eval_prod_check"] = {"mae": out["eval_mean"]["xgb_prod"]["mae"], "log_score": float(ls_prod.mean())}

    # calibration on 2026/27: static map (2025/26) and monthly expanding map
    p_ev_best = pover_of(E.make_pmf(bd, params[best_combo], **E.dist_args(ev_all, bm)))
    played = ev_all["played"].astype(bool).to_numpy() & ev_all["total_corners"].notna().to_numpy()
    cal_static = E.calibrate_lines(p_sel_best, y_sel, p_ev_best, cal_choice)
    cal_exp = E.expanding_calibration(sel, p_sel_best, ev_all, p_ev_best, cal_choice)
    out["eval_calibration"] = {}
    for nm, pv in (("raw", p_ev_best), ("static", cal_static), ("expanding", cal_exp)):
        pp = {k: v[played] for k, v in pv.items()}
        out["eval_calibration"][nm] = {"lines": E.lines_from_pover(y, pp), "vs_prod": E.paired_lines(y, pp, pov_prod)}

    # reliability (production vs best) for the figures
    out["reliability"] = {}
    for nm, pv in (("prod", pov_prod), ("best", {k: v[played] for k, v in p_ev_best.items()})):
        out["reliability"][nm] = {str(l): Q.reliability((y > l).astype(float), pv[l]) for l in D.LINES}
    out["pit"] = {"prod": np.histogram(Q.pit(pmf_prod, y), bins=10, range=(0, 1))[0].tolist(),
                  "best": np.histogram(Q.pit(E.make_pmf(bd, params[best_combo], **E.dist_args(ev, bm)), y),
                                       bins=10, range=(0, 1))[0].tolist()}

    # monthly paired log score (best combo vs production)
    pmf_best_ev = E.make_pmf(bd, params[best_combo], **E.dist_args(ev, bm))
    d_ls = Q.log_score(pmf_best_ev, y) - ls_prod
    month = ev["date"].dt.to_period("M").astype(str)
    out["eval_monthly"] = {mth: {"n": int((month == mth).sum()), "log_score_diff": float(d_ls[month == mth].mean()),
                                 "mae_prod": float(np.abs(y - ev["mu_xgb_prod"])[month == mth].mean()),
                                 "mae_best": float(np.abs(y - ev[f"mu_{bm}"])[month == mth].mean())}
                           for mth in sorted(month.unique())}

    log.info("4. monthly walk-forward")
    wf = walk_forward_table(frame, ev_all, weights)
    out["wf_models"] = [c[3:] for c in wf.columns if c.startswith("mu_") and not c.startswith(("mu_h_", "mu_a_"))
                        and c not in ("mu_prod", "mu_rating")]
    wfp = D.CACHE / "preds_wf.parquet"
    out["wf_fitted"] = sorted({c[3:] for c in pd.read_parquet(wfp).columns if c.startswith("mu_")
                               and not c.startswith(("mu_h_", "mu_a_"))}) if wfp.exists() else []
    wfs = E.scored(wf)
    out["wf_mean"] = mean_table(wfs, ["prod"] + out["wf_models"], ref="prod") if "mu_prod" in wfs else {}
    out["wf_dist"] = {}
    for m in out["wf_models"]:
        key = f"{m}|{bd}" if f"{m}|{bd}" in params else (f"{bm}|{bd}")
        if bd.startswith("conv") and not E.has_sides(m):
            continue
        pmf = E.make_pmf(bd, params.get(f"{m}|{bd}", params[best_combo]), **E.dist_args(wfs, m))
        yw = wfs["total_corners"].to_numpy(float)
        res = E.dist_metrics(yw, pmf)
        pmf0 = E.make_pmf("nb_prod", {"r": r_prod}, **E.dist_args(ev.set_index("match_id").loc[wfs["match_id"]].reset_index(), "xgb_prod"))
        res["log_score_diff"] = C.paired_mean_diff(Q.log_score(pmf, yw), Q.log_score(pmf0, yw))
        res["params_from"] = key
        out["wf_dist"][m] = res

    log.info("5. variants at real prices")
    variants = build_variants(ev_all, wf, params, best_combo, out, r_prod, cal_static if cal_keep else None)
    out["variants"] = {k: v["desc"] for k, v in variants.items()}
    out["variant_quality"] = variant_quality(ev_all, variants, pov_prod, ls_prod)
    chk = pd.DataFrame({"match_id": ev_all["match_id"].to_numpy()})
    pmf_chk = E.make_pmf("nb", params["xgb_prod|nb"], **E.dist_args(ev_all, "xgb_prod"))
    for l in D.LINES:
        chk[l] = Q.p_over(pmf_chk, l)
    pred = ev_all[["match_id", "date", "league", "home_team", "away_team", "total_corners"]].copy()
    for k, v in variants.items():
        if v["pover"] is not None:
            p = v["pover"].set_index("match_id")
            for l in D.LINES:
                pred[f"p{l}_{k}"] = pred["match_id"].map(p[l])
            if "mu" in v:
                pred[f"mu_{k}"] = pred["match_id"].map(v["mu"].set_index("match_id")["mu"])
    pred.to_csv(R / "corner_predictions_2026.csv", index=False)

    results = ev_all.set_index("match_id")["total_corners"]
    a = B.baseline_a()
    out["betting"], out["main"], out["market_quality"], out["self_check"] = {}, {}, {}, {}
    out["breakdown"], out["robustness"] = {}, {}
    for u in B.UNIVERSES:
        cands = B.load_cands(u)
        base = B.baseline_b(u)
        cv = {k: B.with_probs(cands, v["pover"]) for k, v in variants.items()}
        out["betting"][u] = B.corner_roi({k: c for k, c in cv.items() if k != "no_corners"}, "prod")
        out["market_quality"][u] = B.market_quality(cands, results, {k: v["pover"] for k, v in variants.items()
                                                                      if v["pover"] is not None})
        mains = {}
        for k, c in cv.items():
            mains[k] = B.main_table(c, k)
            log.info("%s %s: %s", u, k, EV.summary(B.played(mains[k])))
        same = mains["prod"].merge(base[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_b"))
        out["self_check"][f"{u}_prod_replacement_equals_baseline_b"] = float(
            ((same["market"] == same["market_b"]) & (same["selection"] == same["selection_b"])).mean())
        res_u = {k: B.main_summary(mm, base, a) for k, mm in mains.items()}
        hp = C.holm({k: r["vs_b"]["roi_diff"]["p"] for k, r in res_u.items() if k != "prod"})
        for k in res_u:
            res_u[k]["holm_p"] = hp.get(k)
        out["main"][u] = res_u
        chosen = out["chosen_variant"] = "best"
        mm = B.played(mains[chosen]).copy()
        mm["month"] = pd.to_datetime(mm["date"]).dt.to_period("M").astype(str)
        bb = B.played(base).copy()
        bb["month"] = pd.to_datetime(bb["date"]).dt.to_period("M").astype(str)
        out["breakdown"][u] = {"month": EV.breakdown(mm, "month").to_dict(orient="records"),
                               "month_b": EV.breakdown(bb, "month").to_dict(orient="records"),
                               "league": EV.breakdown(mm, "league").to_dict(orient="records"),
                               "league_b": EV.breakdown(bb, "league").to_dict(orient="records")}
        out["robustness"][u] = {k: B.robustness(mains[k], base) for k in mains}
        res_df = ev_all.set_index("match_id")[["home_goals", "away_goals", "total_corners"]] if "home_goals" in ev_all \
            else frame.set_index("match_id").loc[ev_all["match_id"], ["home_goals", "away_goals", "total_corners"]]
        EV.per_match_table(B.played(base), res_df, f"baseline_b_{u}", "prod_20261012").to_csv(
            R / f"per_match_baseline_b_{u}.csv", index=False)
        for k in ("best", "best_wf"):
            if k in mains:
                EV.per_match_table(B.played(mains[k]), res_df, f"corners_{k}_{u}", f"corners_{k}").to_csv(
                    R / f"per_match_corners_{k}_{u}.csv", index=False)
        if u == "close":       # engine route: production μ + global NB size fitted by ML on 2025/26
            out["self_check"]["engine_route"] = {"variant": "xgb_prod + NB r(ML) = %.1f" % params["xgb_prod|nb"]["r"],
                                                 **B.self_check_engine(
                pd.read_parquet(C.SHARED_CACHE / "matches.parquet"), variants["prod"]["mu"], params["xgb_prod|nb"]["r"],
                "closing", B.with_probs(cands, chk))}
        out["coverage_2026"] = out.get("coverage_2026", {})
        out["coverage_2026"][u] = coverage_2026(ev_all, cands, mains, base)
    pd.DataFrame([{"universe": u, **r} for u in B.UNIVERSES for r in out["coverage_2026"][u]]).to_csv(
        R / "coverage_2026.csv", index=False)

    out["decisions"] = decide(out)
    C.write_json(R / "results.json", out)
    prereg(out)
    C.write_json(R / "run_info.json", C.run_info("corners", started, {
        "select_season": "2025/26 (fit 2021/22-2024/25)", "eval_season": "2026/27 (fit 2021/22-2025/26)",
        "walk_forward_cutoffs": [str(c.date()) for c in WF_CUTS], "lines": list(D.LINES),
        "trees": C.read_json(D.CACHE / "trees.json"), "rating_params": out["frame_info"]["rating"]["best"],
        "production_nb_size": r_prod, "wf_models": out["wf_fitted"]},
        {"variants_tried": {"mean_models": len(models) + 1, "distributions": len(sel_d),
                            "calibrations": len(E.CALIB), "main_variants": len(variants) - 1}}))
    from experiments.v6_3.corners import figures
    figures.make(out)
    log.info("done in %.0fs", time.time() - started)


# ---------------------------------------------------------------------------------------------- walk-forward
def walk_forward_table(frame: pd.DataFrame, ev_all: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """2026/27 means from monthly refits: July from the eval-stage models (fitted before 2026-07-01), later months
    from the wf-stage models; GLM refitted monthly here; ratings are online anyway."""
    path = D.CACHE / "preds_wf.parquet"
    base_cols = ["match_id", "date", "season", "league", "played", "total_corners", "home_corners", "away_corners"]
    t = ev_all[base_cols].copy()
    if not path.exists():
        return t
    wf = pd.read_parquet(path)
    mus = [c for c in wf.columns if c.startswith("mu_")]
    j = t.merge(wf[["match_id"] + mus], on="match_id", how="left")
    for c in mus:
        src = c if c in ev_all else None
        if src:
            j[c] = j[c].fillna(j["match_id"].map(ev_all.set_index("match_id")[src]))
    j["mu_prod"] = ev_all["mu_xgb_prod"].to_numpy()          # the production bundle itself (reference)
    j["mu_rating"] = ev_all["mu_rating"].to_numpy()
    j["mu_h_rating"], j["mu_a_rating"] = ev_all["mu_h_rating"].to_numpy(), ev_all["mu_a_rating"].to_numpy()
    glm = ev_all["mu_glm"].to_numpy().copy()
    leagues = E.leagues()
    for i, cut in enumerate(WF_CUTS):
        end = WF_CUTS[i + 1] if i + 1 < len(WF_CUTS) else pd.Timestamp.max
        msk = ((j["date"] >= cut) & (j["date"] < end)).to_numpy()
        tgt = frame.set_index("match_id").loc[j.loc[msk, "match_id"]].reset_index()
        glm[msk] = M.glm_predictions(frame, leagues, cut, tgt)
    j["mu_glm"] = glm
    if all(f"mu_{k}" in j for k in E.ENS_PARTS):
        j["mu_ens_w"] = j[[f"mu_{k}" for k in E.ENS_PARTS]].to_numpy(float) @ np.array([weights[k] for k in E.ENS_PARTS])
    return j


# ---------------------------------------------------------------------------------------------- variants
def build_variants(ev_all, wf, params, best_combo, out, r_prod, cal_static) -> dict:
    bm, bd = best_combo.split("|")

    def pv(t, m, d, prm):
        pmf = E.make_pmf(d, prm, **E.dist_args(t, m))
        p = pd.DataFrame({"match_id": t["match_id"].to_numpy()})
        for l in D.LINES:
            p[l] = Q.p_over(pmf, l)
        p.attrs["pmf"] = pmf
        return p

    def mu(t, m):
        return pd.DataFrame({"match_id": t["match_id"].to_numpy(), "mu": t[f"mu_{m}"].to_numpy(float)})

    v = {"prod": {"desc": "Продакшн: XGBoost, NB r = %.1f (= Baseline B)" % r_prod,
                  "pover": pv(ev_all, "xgb_prod", "nb_prod", {"r": r_prod}), "mu": mu(ev_all, "xgb_prod"),
                  "engine_r": r_prod}}
    dp = out["best_dist_for_prod"]
    if dp != "nb_prod":
        v["prod_dist"] = {"desc": f"Продакшн μ + распределение {E.DLABELS[dp]} (2025/26)",
                          "pover": pv(ev_all, "xgb_prod", dp, params[f"xgb_prod|{dp}"]), "mu": mu(ev_all, "xgb_prod")}
        if dp in ("nb", "nb_mom"):
            v["prod_dist"]["engine_r"] = params[f"xgb_prod|{dp}"]["r"]
    v["best"] = {"desc": f"{E.LABELS.get(bm, bm)} + {E.DLABELS[bd]} (выбор на 2025/26)",
                 "pover": pv(ev_all, bm, bd, params[best_combo]), "mu": mu(ev_all, bm)}
    if bd in ("nb", "nb_mom") and "engine_r" not in v.get("prod_dist", {}):
        v["best"]["engine_r"] = params[best_combo]["r"]
    if cal_static is not None:
        p = pd.DataFrame({"match_id": ev_all["match_id"].to_numpy()})
        for l in D.LINES:
            p[l] = cal_static[l]
        v["best_cal"] = {"desc": "Лучший + калибровка линий (карта 2025/26)", "pover": p, "mu": mu(ev_all, bm)}
    if f"mu_{bm}" in wf and wf[f"mu_{bm}"].notna().all() and bm != "rating":
        v["best_wf"] = {"desc": f"{E.LABELS.get(bm, bm)}, ежемесячное переобучение + {E.DLABELS[bd]}",
                        "pover": pv(wf, bm, bd, params[best_combo]), "mu": mu(wf, bm)}
    if "ens_w" != bm and "ens_w|" + bd in params:
        v["ens"] = {"desc": f"Ансамбль (веса 2025/26) + {E.DLABELS[bd]}",
                    "pover": pv(ev_all, "ens_w", bd, params["ens_w|" + bd]), "mu": mu(ev_all, "ens_w")}
    if "rating|conv_frank" in params and bm != "rating":
        v["rating"] = {"desc": "Динамические рейтинги + свёртка с копулой Франка",
                       "pover": pv(ev_all, "rating", "conv_frank", params["rating|conv_frank"]),
                       "mu": mu(ev_all, "rating")}
    v["no_corners"] = {"desc": "Main без рынков угловых (контроль)", "pover": None}
    return v


def coverage_2026(ev_all, cands, mains, base) -> list[dict]:
    c = cands.loc[cands["group"] == "CORNERS"]
    priced = set(c.loc[c["odds"].notna(), "match_id"])
    played = ev_all["played"].astype(bool)
    low = (ev_all["h_x_nhist"].fillna(0) < 3) | (ev_all["a_x_nhist"].fillna(0) < 3)
    rows = [{"item": "matches of the season (corner leagues)", "n": int(len(ev_all))},
            {"item": "played", "n": int(played.sum())},
            {"item": "played with corner statistics", "n": int(ev_all.loc[played, "total_corners"].notna().sum())},
            {"item": "processed by every method (corner mean + distribution)", "n": int(len(ev_all))},
            {"item": "played with a real corner price (any line)", "n": int(ev_all.loc[played, "match_id"].isin(priced).sum())},
            {"item": "played with < 3 earlier matches of a team (weak statistics, still predicted)",
             "n": int((low & played).sum())}]
    for k, mm in mains.items():
        p = B.played(mm)
        rows.append({"item": f"Main {k}: settled / bets / no price (not a bet) / push",
                     "n": f"{len(p)} / {int(p['bet'].sum())} / {int(p['odds'].isna().sum())} / {int((p['status'] == 'PUSH').sum())}"})
    pb = B.played(base)
    rows.append({"item": "Baseline B: settled / bets / no price", "n": f"{len(pb)} / {int(pb['bet'].sum())} / {int(pb['odds'].isna().sum())}"})
    return rows


# ---------------------------------------------------------------------------------------------- decisions
def variant_quality(ev_all, variants, pov_prod, ls_prod) -> dict:
    """Paired distribution quality of each money variant vs production on the played 2026/27 matches."""
    played = (ev_all["played"].astype(bool) & ev_all["total_corners"].notna()).to_numpy()
    y = ev_all["total_corners"].to_numpy(float)[played]
    month = ev_all["date"].dt.to_period("M").astype(str).to_numpy()[played]
    out = {}
    for k, v in variants.items():
        if k == "prod" or v["pover"] is None:
            continue
        pv = {l: v["pover"][l].to_numpy(float)[played] for l in D.LINES}
        res = {"lines_diff": E.paired_lines(y, pv, pov_prod), "lines": E.lines_from_pover(y, pv)}
        pmf = v["pover"].attrs.get("pmf")
        if pmf is not None:
            ls = Q.log_score(pmf[played], y)
            res["log_score"] = float(ls.mean())
            res["log_score_diff"] = C.paired_mean_diff(ls, ls_prod)
            res["monthly_ls_diff"] = {m: float((ls - ls_prod)[month == m].mean()) for m in sorted(set(month))}
        out[k] = res
    return out


def decide(out: dict) -> dict:
    """A / B / C per variant.

    A — log score AND mean line log loss significantly better than production on 2026/27 (paired bootstrap),
        better in at least 3 of 4 full months, and Main not significantly worse in either universe;
    B — both point estimates better (or Main better in both universes at Holm p < 0.05) and Main not significantly
        worse — promising, frozen for a forward check;
    C — otherwise. ``no_corners`` is a control (removes the corner markets from Main)."""
    dec = {}
    for k in out["variants"]:
        if k == "prod":
            continue
        q = out["variant_quality"].get(k, {})
        lsd, lld = q.get("log_score_diff"), q.get("lines_diff", {}).get("mean")
        ls_sig = None if lsd is None else lsd["ci95"][1] < 0
        ll_sig = None if lld is None else lld["ci95"][1] < 0
        ls_pt = lsd is not None and lsd["diff"] < 0
        ll_pt = lld is not None and lld["diff"] < 0
        months = q.get("monthly_ls_diff", {})
        full = [m for m in months if m not in ("2026-07",)]
        stable = bool(full) and sum(months[m] < 0 for m in full) >= 0.75 * len(full)
        main_ok = all((out["main"][u][k]["holm_p"] or 1) < 0.05 and out["main"][u][k]["vs_b"]["roi_diff"]["diff"] > 0
                      for u in B.UNIVERSES)
        main_worse = any(out["main"][u][k]["vs_b"]["roi_diff"]["ci95"][1] < 0 for u in B.UNIVERSES)
        main_pt = all(out["main"][u][k]["vs_b"]["roi_diff"]["diff"] > 0 for u in B.UNIVERSES)
        if k == "no_corners":
            d = "B" if main_ok else "C"
        elif ls_sig and ll_sig and stable and not main_worse:
            d = "A"
        elif ((ls_pt or lsd is None) and ll_pt and not main_worse) or (main_ok and not main_worse):
            d = "B"
        else:
            d = "C"
        dec[k] = {"decision": d, "log_score_significant": ls_sig, "line_ll_significant": ll_sig,
                  "log_score_point_better": ls_pt if lsd is not None else None, "line_ll_point_better": ll_pt,
                  "months_stable": stable if months else None, "main_significant_both": main_ok,
                  "main_worse_ci": main_worse, "main_point_better_both": main_pt}
    return dec


def rule_of(out: dict, k: str) -> dict:
    bm, bd = out["best_combo"]["model"], out["best_combo"]["dist"]
    if k == "ens":
        return {"mean": "convex ensemble", "weights": out["ensemble_weights"], "distribution": bd,
                "params": out["dist_params"].get(f"ens_w|{bd}"), "fit": "2021/22-2025/26, weights and params on 2025/26"}
    if k == "rating":
        return {"mean": "dynamic ratings", "rating": out["frame_info"]["rating"]["best"], "distribution": "conv_frank",
                "params": out["dist_params"].get("rating|conv_frank")}
    if k == "prod_dist":
        return {"mean": "production XGBoost", "distribution": out["best_dist_for_prod"],
                "params": out["dist_params"].get(f"xgb_prod|{out['best_dist_for_prod']}")}
    return {"mean": bm, "distribution": bd, "params": out["dist_params"].get(f"{bm}|{bd}"),
            "refit": "monthly, on all matches before the 1st of the month" if k == "best_wf" else "once, on 2021/22-2025/26",
            "calibration": "none (not better on 2025/26)" if not out["calibration_choice"]["kept"] else out["calibration_choice"]}


def prereg(out: dict) -> None:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    entries = []
    for k, d in out["decisions"].items():
        if d["decision"] != "B":
            continue
        entries.append({"variant": k, "description": out["variants"][k],
                        "hypothesis": "The corner distribution of this variant gives a lower log score of total "
                                      "corners and a lower mean log loss of over/under 8.5/9.5/10.5 than production "
                                      "(XGBoost + NB r=%.1f), without lowering Main ROI." % out["r_prod"],
                        "rule": rule_of(out, k),
                        "primary_metric": "paired mean log score of total corners vs production (lower is better)",
                        "secondary": ["mean line log loss 8.5/9.5/10.5", "Main ROI paired vs Baseline B (close and t09)"],
                        "minimum_forward_sample": "1500 played matches with corner statistics (≈ 3 months)",
                        "frozen_at": now})
    C.write_json(R / "preregistration.json", entries)


if __name__ == "__main__":
    main()
