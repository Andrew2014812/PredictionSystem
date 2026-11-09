"""Collects the numbers of the thirteen v6.2 studies into results/summary.json.

    python -m experiments.v6_2.summary.collect

Every value is read from a study's own result file (results/*.json, results/*.csv). A few qualitative items that
exist only in a hand-written findings.md (the list of production defects, The Odds API share, the v6.1 price
coverage) are parsed from that file. Each value carries the file it came from ("src"), so the summary PDF can
cite it. Nothing is typed in by hand and nothing is recomputed from raw data.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V62 = ROOT / "experiments" / "v6_2"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "summary.json"

STUDIES = ["xg", "data_quality", "odds_timing", "bookmaker", "main_calibration", "risk", "market_models",
           "goal_distributions", "leagues", "temporal", "uncertainty", "forward", "data_sources"]
REPORTS = {"xg": "XG_Research.pdf", "data_quality": "Data_Quality_Research.pdf",
           "odds_timing": "Odds_Timing_Research.pdf", "bookmaker": "Bookmaker_Benchmark_Research.pdf",
           "main_calibration": "Main_Calibration_Research.pdf", "risk": "Risk_Selection_Research.pdf",
           "market_models": "Market_Models_Research.pdf", "goal_distributions": "Goal_Distributions_Research.pdf",
           "leagues": "League_Specialization_Research.pdf", "temporal": "Temporal_Features_Research.pdf",
           "uncertainty": "Prediction_Uncertainty_Research.pdf", "forward": "Forward_Validation_Research.pdf",
           "data_sources": "Data_Sources_Report.pdf"}
SEASONS = ["2023", "2024", "2025", "2026"]

USED: set[str] = set()


def rel(study: str, name: str) -> str:
    return f"experiments/v6_2/{study}/{name}"


def J(study: str, name: str):
    """A study's result JSON (results/<name>)."""
    path = V62 / study / "results" / name
    USED.add(rel(study, f"results/{name}"))
    return json.loads(path.read_text(encoding="utf-8"))


def CSV(study: str, name: str) -> list[dict]:
    path = V62 / study / "results" / name
    USED.add(rel(study, f"results/{name}"))
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def FINDINGS(study: str) -> str:
    USED.add(rel(study, "findings.md"))
    return (V62 / study / "findings.md").read_text(encoding="utf-8")


def V(v, src, ci=None, **extra) -> dict:
    """One value with its 95 % interval and the file it was read from."""
    out = {"v": None if v is None else float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v,
           "src": src}
    if ci is not None:
        out["ci"] = [float(ci[0]), float(ci[1])]
    out.update(extra)
    return out


def src(study: str, name: str) -> str:
    return rel(study, f"results/{name}")


def prereg_info(pre: dict, hyp: str, where: str) -> dict:
    """written_at, N and the criterion of a preregistered hypothesis that has no forward shadow."""
    crit = pre["hypotheses"][hyp]["forward_criterion"]
    n = re.search(r">= (\d+) settled", crit)
    return {"written_at": pre["written_at"], "min_n": int(n.group(1)) if n else None, "criterion": crit,
            "name": pre["hypotheses"][hyp].get("name") or pre["hypotheses"][hyp].get("variant"), "src": where}


# ---------------------------------------------------------------------------------------------- studies
def xg() -> dict:
    aud, met, main, corn, dec, pre = (J("xg", n) for n in ("audit.json", "metrics.json", "main.json", "corners.json",
                                                          "decisions.json", "preregistration.json"))
    s_met, s_main = src("xg", "metrics.json"), src("xg", "main.json")
    vs = met["vs_B0"]
    sig = [v for v, p in met["holm"]["goals_dev"].items() if p < 0.05 and vs[v]["goals_dev"]["mean"] < 0]
    sig += [v for v, p in met["holm"]["logloss_1x2"].items() if p < 0.05 and vs[v]["logloss_1x2"]["mean"] < 0]
    b0 = main["summary"]["B0"]
    return {
        "leagues": V(len(aud["xg_leagues"]), src("xg", "audit.json")),
        "no_xg_leagues": V(aud["no_xg_leagues"], src("xg", "audit.json")),
        "matches_with_xg": V(aud["overall"]["matches_with_xg"], src("xg", "audit.json")),
        "first_date": V(aud["overall"]["first_date"], src("xg", "audit.json")),
        "goals_over_xg": V(aud["overall"]["ratio"], src("xg", "audit.json")),
        "n_eval": V(met["n_matches"], s_met), "eval_period": V(met["eval_period"], s_met),
        "n_variants": V(met["n_variants"] - 1, s_met),
        "n_significant_better": V(len(set(sig)), s_met),
        "X5_goals": V(vs["X5"]["goals_dev"]["mean"], s_met, vs["X5"]["goals_dev"]["ci95"],
                      p=vs["X5"]["goals_dev"]["p"], p_holm=met["holm"]["goals_dev"]["X5"]),
        "X5_ll": V(vs["X5"]["logloss_1x2"]["mean"], s_met, vs["X5"]["logloss_1x2"]["ci95"]),
        "X5_n_required": V(vs["X5"]["goals_dev"]["n_required"], s_met),
        "T1_ll": V(vs["T1"]["logloss_1x2"]["mean"], s_met, vs["T1"]["logloss_1x2"]["ci95"]),
        "X1_goals": V(vs["X1"]["goals_dev"]["mean"], s_met, vs["X1"]["goals_dev"]["ci95"]),
        "X1_ll": V(vs["X1"]["logloss_1x2"]["mean"], s_met, vs["X1"]["logloss_1x2"]["ci95"]),
        "B0_gap": V(b0["bet_gap"], s_main, b0["bet_gap_ci95"]),
        "B0_roi": V(b0["roi"], s_main, b0["roi_ci95"], n=b0["bets"]),
        "X5_dgap": V(main["vs_B0"]["X5"]["gap"]["diff"], s_main, main["vs_B0"]["X5"]["gap"]["ci95"]),
        "X5_droi": V(main["vs_B0"]["X5"]["roi"]["diff"], s_main, main["vs_B0"]["X5"]["roi"]["ci95"]),
        "corners": V(corn["diff_xg_vs_B0"]["mean"], src("xg", "corners.json"), corn["diff_xg_vs_B0"]["ci95"]),
        "decisions": V({k: v["decision"] for k, v in dec.items()}, src("xg", "decisions.json")),
        "forward_min": V(pre["forward_test"]["minimum_matches"], src("xg", "preregistration.json")),
    }


def data_quality() -> dict:
    rc, sens, var, st = (J("data_quality", n) for n in ("recompute.json", "sensitivity.json", "variants.json",
                                                        "strata.json"))
    s_sens, s_var, s_st = (src("data_quality", n) for n in ("sensitivity.json", "variants.json", "strata.json"))
    nh, na = sens["variants"]["newcomer_h"], sens["variants"]["newcomer_a"]
    e50 = var["variants"]["elo_new_-50"]["all"]["all"]
    out_new = st["vs_established"]["outside|1-10"]
    text = FINDINGS("data_quality")
    defects = re.findall(r"^- \*\*(D\d)\.\*\* (.+)$", text, re.M)
    return {
        "cells": V(rc["cells_compared"], src("data_quality", "recompute.json")),
        "cells_mismatch": V(rc["cells_mismatch"], src("data_quality", "recompute.json")),
        "matches": V(rc["compared_matches"], src("data_quality", "recompute.json")),
        "features": V(rc["features_compared"], src("data_quality", "recompute.json")),
        "rebuild_cells_differing": V(var["reproduction"]["cells_differing"], s_var),
        "zero_home_ll": V(nh["d_log_loss"]["mean"], s_sens, nh["d_log_loss"]["ci95"], n=nh["matches"]),
        "zero_home_main_changed": V(nh["main_changed"], s_sens),
        "zero_away_ll": V(na["d_log_loss"]["mean"], s_sens, na["d_log_loss"]["ci95"]),
        "zero_away_main_changed": V(na["main_changed"], s_sens),
        "zero_home_droi": V(nh["d_main_roi"]["diff"], s_sens, nh["d_main_roi"]["ci95"]),
        "n_sens_variants": V(len(sens["variants"]), s_sens),
        "n_dq_variants": V(len(var["variants"]), s_var),
        "newcomer_ll": V(out_new["log_loss_diff"]["diff"], s_st, out_new["log_loss_diff"]["ci95"]),
        "newcomer_excess_ll": V(out_new["excess_log_loss_diff"]["diff"], s_st,
                                out_new["excess_log_loss_diff"]["ci95"]),
        "dq_spearman": V(st["dq_spearman_excess_ll"], s_st),
        "elo50_ll": V(e50["d_log_loss"]["mean"], s_var, e50["d_log_loss"]["ci95"], n=e50["matches"]),
        "elo50_droi": V(e50["d_main_roi"]["diff"], s_var, e50["d_main_roi"]["ci95"]),
        "elo50_meets_rule": V(var["meets_selection_rule"]["elo_new_-50"], s_var),
        "defects": V([f"{k}. {re.sub(r'[*`]', '', t)}" for k, t in defects], rel("data_quality", "findings.md")),
    }


def odds_timing() -> dict:
    aud, clv, dr = J("odds_timing", "audit.json"), J("odds_timing", "clv.json"), J("odds_timing", "drift.json")
    s_a, s_c, s_d = (src("odds_timing", n) for n in ("audit.json", "clv.json", "drift.json"))
    roi = {}
    for role in ("main", "risk"):
        for k in ("current_all_sources", "current_live", "live_A1_A2", "live_B", "backfill_C"):
            r = aud["roi"][role][k]
            roi[f"{role}|{k}"] = V(r["roi"], s_a, r["roi_ci95"], n=r["bets"])
    betas = [v["beta"][1] for k, v in dr["model_vs_drift_controlled"].items() if k.endswith("|1X2")]
    return {
        "counts": V(aud["counts"]["by_class_role"], s_a),
        "roi": roi,
        "B_after_h": V(aud["timing"]["quote_after_gen_h_B"]["median"], s_a),
        "clv_main_odds": V(clv["main|A1"]["clv_odds"]["mean"], s_c, clv["main|A1"]["clv_odds"]["ci95"]),
        "clv_main_fair": V(clv["main|A1"]["clv_fair"]["mean"], s_c, clv["main|A1"]["clv_fair"]["ci95"]),
        "clv_risk_fair": V(clv["risk|A1"]["clv_fair"]["mean"], s_c, clv["risk|A1"]["clv_fair"]["ci95"]),
        "pinnacle_beta_min": V(min(betas), s_d), "pinnacle_beta_max": V(max(betas), s_d),
        "close_minus_open_ll": V(dr["fd"]["all_seasons_b365|1X2"]["ll_close_minus_open"]["diff"], s_d),
    }


def bookmaker() -> dict:
    bm, enc, bet, dm, hy, pre = (J("bookmaker", n) for n in ("benchmark.json", "encompassing.json", "betting.json",
                                                            "demargin.json", "hybrids.json", "preregistration.json"))
    s_bm, s_enc, s_bet, s_dm, s_hy = (src("bookmaker", n) for n in ("benchmark.json", "encompassing.json",
                                                                   "betting.json", "demargin.json", "hybrids.json"))
    vs = {}
    for m in ("1X2", "OU25", "AH"):
        for s in SEASONS:
            d = bm["model_vs_market"].get(f"{m}|{s}|b365_open", {}).get("d_ll")
            if d:
                vs[f"{m}|{s}"] = V(d["diff"], s_bm, d["ci95"], n=d["n"])
    encs = {}
    for m in ("1X2", "OU25", "AH"):
        for s in ("select", "2026"):
            t = enc["tests"].get(f"{m}|{s}|b365_open")
            if t:
                encs[f"{m}|{s}"] = V(t["b_model"], s_enc, t["b_model_ci95"], p=t["p_lr"])
    value = {}
    for m in ("1X2", "OU25", "AH"):
        for s in SEASONS:
            b = bet.get(f"{m}|{s}|model|0.0")
            if b and b.get("roi") is not None:
                value[f"{m}|{s}"] = V(b["roi"], s_bet, b["roi_ci95"], n=b["bets"], ev=b["mean_ev"],
                                      clv=b["clv_fair"]["mean"])
    best = {}
    for k, v in hy["results"].items():
        nm = min((x for x in ("H1", "H2", "H3", "H4", "H5") if x in v), key=lambda x: v[x]["ll"])
        d = v[nm]["d_ll_vs_H0"]
        best[k] = V(d["diff"], s_hy, d["ci95"], name=nm, p=d["p_two_sided"], p_holm=v[nm].get("p_holm"))
    margins = {k: V(dm["by_key"][k]["margin"], s_dm) for k in
               ("1X2|2026|b365_open", "OU25|2026|b365_open", "AH|2026|b365_open", "1X2|2024|ps_close",
                "1X2|2024|bfe_close")}
    return {"vs_market": vs, "encompassing": encs, "value_bets": value, "best_hybrid": best, "margins": margins,
            "prereg_status": V(pre["status"], src("bookmaker", "preregistration.json"))}


def main_calibration() -> dict:
    var, sel, nc = (J("main_calibration", n) for n in ("variants.json", "selection.json", "ncand_summary.json"))
    s_v = src("main_calibration", "variants.json")
    v = var["variants"]
    out = {}
    for k in ("site", "bet"):
        for s in SEASONS:
            m = v[k][s]["A базовый"]
            out[f"base|{k}|{s}"] = V(m["gap"], s_v, m["gap_ci"], roi=m["roi"], roi_ci=m["roi_ci"], n=m["bets"])
    s26 = v["site"]["2026"]
    for key, nm in (("shown", "A как показано"), ("platt_hist", "показано + PS-Platt"),
                    ("ps_group", "показано + PS-Group"), ("platt_monthly", "показано + PS-Platt по месяцам")):
        m = s26[nm]
        out[key] = V(m["gap"], s_v, m["gap_ci"], roi=m["roi"], roi_ci=m["roi_ci"], n=m["bets"])
    pooled = var["pooled"]["bet"]["variants"]["PS-Platt"]
    out["pooled_platt_dabsgap"] = V(pooled["d_absgap"], s_v, pooled["d_absgap_ci"], p_holm=pooled["d_absgap_p_holm"])
    out["q2_gap_2024"] = V(v["bet"]["2024"]["Q2 Платт по рынкам"]["gap"], s_v)
    out["r5_gap_2024"] = V(v["bet"]["2024"]["R5 Келли"]["gap"], s_v)
    out["ncand_2024"] = V(nc["bet"]["2024"]["median"], src("main_calibration", "ncand_summary.json"))
    out["ncand_2026"] = V(nc["bet"]["2026"]["median"], src("main_calibration", "ncand_summary.json"))
    r = sel["roi"]
    out["best_roi"] = V(r["pooled_d_roi"], src("main_calibration", "selection.json"), r["pooled_d_roi_ci"],
                        name=r["chosen"], p_holm=r["pooled_p_holm"])
    out["n_variants"] = V(len(v["site"]["2023"]) - 1, s_v)
    return out


def risk() -> dict:
    r, sel, evr = J("risk", "risk.json"), J("risk", "selection.json"), J("risk", "ev_refunds.json")
    s_r = src("risk", "risk.json")
    out = {}
    for s in SEASONS:
        m = r["strategies"][s]["Текущий Risk (v6.1)"]
        e = r["diagnostics"][s]["ev_decomposition"]
        out[f"base|{s}"] = V(m["roi"], s_r, m["roi_ci"], n=m["bets"], odds=m["avg_odds"], gap=m["gap"],
                             ev_model=e["book_model_ev"], ev_market=e["book_market_ev"],
                             model_minus_market=e["model_minus_market"],
                             market_minus_realised_ci=e["market_minus_realised_ci"],
                             gap_chosen=r["diagnostics"][s]["selection"]["priced"]["gap_chosen"])
    sh = r["strategies"]["2026"]["Risk как показан"]
    out["shown"] = V(sh["roi"], s_r, sh["roi_ci"], n=sh["bets"], gap=sh["gap"])
    p = sel["pooled"]
    out["best_roi"] = V(p["d_roi"], src("risk", "selection.json"), p["d_roi_ci"], name=sel["chosen"],
                        p_holm=p["p_holm"])
    out["n_rules"] = V(len(sel["eligible"]), src("risk", "selection.json"))
    out["refund_changed"] = V({s: evr["fair_odds"]["risk"][s]["changed"] for s in SEASONS},
                              src("risk", "ev_refunds.json"))
    return out


def market_models() -> dict:
    cc = J("market_models", "comparisons.json")["comparisons"]
    coh, mr, ref, dec = (J("market_models", n) for n in ("coherence.json", "main_risk.json", "market_reference.json",
                                                        "decisions.json"))
    s_c = src("market_models", "comparisons.json")
    direct = [k for k in cc if not k.split("|")[1].startswith("S_")]
    t = cc["TOTAL|S_TOT"]
    sub = mr["variants"]["SUB_TOTAL"]
    refs = {}
    for s, mk in ref.items():
        for m, x in mk.items():
            refs[f"{m}|{s}"] = V(x["CUR"] - x["market"], src("market_models", "market_reference.json"), n=x["n"])
    decs = {k: d["decision"] for k, d in dec["decisions"].items()} if "decisions" in dec else \
        {k: d["decision"] for k, d in dec.items() if isinstance(d, dict) and "decision" in d}
    pre = J("market_models", "preregistration.json")
    return {
        "prereg": prereg_info(pre, "H1_TOTAL_S_TOT", src("market_models", "preregistration.json")),
        "n_direct": V(len(direct), s_c),
        "n_direct_worse": V(sum(1 for k in direct if cc[k]["select"]["group"]["ci95"][0] > 0), s_c),
        "n_direct_better": V(sum(1 for k in direct if cc[k]["select"]["group"]["ci95"][1] < 0), s_c),
        "es": V(cc["EXACT_SCORE|ES"]["select"]["group"]["diff"], s_c, cc["EXACT_SCORE|ES"]["select"]["group"]["ci95"]),
        "s_tot": V(t["select"]["group"]["diff"], s_c, t["select"]["group"]["ci95"], p_holm=t["holm_p_select"],
                   seasons={s: t[s]["group"]["diff"] for s in SEASONS}),
        "s_tot_droi_select": V(sub["select"]["main_vs_cur"]["diff"], src("market_models", "main_risk.json"),
                               sub["select"]["main_vs_cur"]["ci95"]),
        "s_tot_droi_2026": V(sub["2026"]["main_vs_cur"]["diff"], src("market_models", "main_risk.json"),
                             sub["2026"]["main_vs_cur"]["ci95"]),
        "coherence_mae": V(coh["2024"]["GD"]["home_win_vs_classifier_mae"], src("market_models", "coherence.json")),
        "coherence_max": V(coh["2024"]["GD"]["home_win_vs_classifier_max"], src("market_models", "coherence.json")),
        "model_minus_market": refs,
        "n_comparisons": V(J("market_models", "comparisons.json")["n_comparisons"], s_c),
        "decisions": V(decs, src("market_models", "decisions.json")),
    }


def goal_distributions() -> dict:
    comp = J("goal_distributions", "comparisons.json")
    mr, dec = J("goal_distributions", "main_risk.json"), J("goal_distributions", "decisions.json")
    s_c = src("goal_distributions", "comparisons.json")
    c = comp["comparisons"]
    hy = c["HY+R"]
    fam = {k: V(c[k]["select"]["SCORE"]["diff"], s_c, c[k]["select"]["SCORE"]["ci95"])
           for k in ("DC+R", "BP+R", "NB+R", "CMP+R", "FR+R", "FRCMP+R", "MH1+R", "P")}
    m = mr["variants"]["HY+R"]["select"]["main_vs_ref"]
    pre = J("goal_distributions", "preregistration.json")
    return {
        "prereg": prereg_info(pre, "H1_HY+R", src("goal_distributions", "preregistration.json")),
        "hy": V(hy["select"]["SCORE"]["diff"], s_c, hy["select"]["SCORE"]["ci95"], p_holm=hy["holm_p_score_select"],
                seasons={s: hy[s]["SCORE"]["diff"] for s in SEASONS}),
        "hy_droi": V(m["diff"], src("goal_distributions", "main_risk.json"), m["ci95"]),
        "families": fam,
        "n_variants": V(comp["n_variants_compared"], s_c),
        "decisions": V({k: d["decision"] for k, d in dec["decisions"].items()},
                       src("goal_distributions", "decisions.json")),
    }


def leagues() -> dict:
    perf, var, tr = J("leagues", "perf.json"), J("leagues", "variants.json"), J("leagues", "transfer.json")
    s_p, s_v = src("leagues", "perf.json"), src("leagues", "variants.json")
    L = perf["select"]["leagues"]
    seasons = {s: V(perf["seasons"][s]["d"]["diff"], s_p, perf["seasons"][s]["d"]["ci95"], n=perf["seasons"][s]["n"])
               for s in SEASONS}
    vv = {}
    for k, d in var["variants"].items():
        mm = d["select"]["d_" + d["metric"]]
        vv[k] = V(mm["diff"], s_v, mm["ci95"], name=d["name"], p_holm=d["select"]["p_holm"], decision=d["decision"],
                  droi=d["select"]["d_roi"]["diff"])
    better = [(k, x["d"]["diff"]) for k, x in L.items() if x["d"]["diff"] < 0]
    return {
        "seasons": seasons,
        "select": V(perf["select"]["d"]["diff"], s_p, perf["select"]["d"]["ci95"], n=perf["select"]["n"]),
        "n_leagues": V(len(L), s_p),
        "n_worse": V(sum(1 for x in L.values() if x["d"]["diff"] > 0), s_p),
        "n_worse_sig": V(sum(1 for x in L.values() if x["d"]["ci95"][0] > 0), s_p),
        "better_leagues": V([k for k, _ in better], s_p),
        "I2": V(perf["select"]["meta"]["I2"], s_p),
        "variants": vv,
        "n_variants": V(var["n_variants_tested"], s_v),
        "n_variants_worse_sig": V(sum(1 for x in vv.values() if x["ci"][0] > 0), s_v),
        "transfer_all": V(tr["pooled"]["all"]["pooled"]["diff"], src("leagues", "transfer.json"),
                          tr["pooled"]["all"]["pooled"]["ci95"]),
        "transfer_own": V(tr["pooled"]["own"]["pooled"]["diff"], src("leagues", "transfer.json"),
                          tr["pooled"]["own"]["pooled"]["ci95"]),
        "ref_roi": V(var["reference"]["B0"]["select"]["main"]["roi"]["mean"], s_v,
                     var["reference"]["B0"]["select"]["main"]["roi"]["ci95"]),
    }


def temporal() -> dict:
    tv, lk = J("temporal", "variants.json"), J("temporal", "leakage_test.json")
    s_v = src("temporal", "variants.json")
    vv = {}
    for k, d in tv["variants"].items():
        s = d["select"]
        vv[k] = V(s["d_ll"]["diff"], s_v, s["d_ll"]["ci95"], name=d["name"], p_holm=s["p_holm_ll"],
                  decision=d["decision"], droi=s["d_roi"]["diff"], droi_ci=s["d_roi"]["ci95"],
                  dev=s["d_dev"]["diff"], dev_ci=s["d_dev"]["ci95"])
    t0 = tv["reference"]["T0"]["select"]
    rep = {k: v for k, v in lk["reproduction"].items() if isinstance(v, dict)}
    return {
        "variants": vv,
        "n_variants": V(tv["n_variants_tested"], s_v),
        "t0_vs_market": V(t0["vs_market"]["diff"], s_v, t0["vs_market"]["ci95"], n=t0["vs_market"]["n"]),
        "t0_roi": V(t0["main"]["roi"]["mean"], s_v, t0["main"]["roi"]["ci95"]),
        "leak_features": V(len(rep), src("temporal", "leakage_test.json")),
        "leak_max_diff": V(max(x["max_abs_diff"] for x in rep.values()), src("temporal", "leakage_test.json")),
        "leak_all_passed": V(lk["all_passed"], src("temporal", "leakage_test.json")),
    }


def uncertainty() -> dict:
    ms, pred, filt, bag, dec, pre = (J("uncertainty", n) for n in ("measures.json", "predictive.json", "filters.json",
                                                                  "bagging.json", "decisions.json",
                                                                  "preregistration.json"))
    s_p, s_f = src("uncertainty", "predictive.json"), src("uncertainty", "filters.json")
    pool = pred["pooled_select"]
    aucs = [x["auc_u_for_miss"] for x in pool.values()]
    br = filt["best_roi"]
    b = filt["variants"][br]["pooled_select_roi"]
    u26 = pred["per_season"]["2026"]["u_1x2"]
    return {
        "n_measures": V(len(pool), s_p),
        "n_measures_sig": V(sum(1 for x in pool.values() if x.get("p_holm", 1) < 0.05), s_p),
        "auc_min": V(min(aucs), s_p), "auc_max": V(max(aucs), s_p),
        "u_seed_2024": V(ms["2024"]["u_seed"]["median"], src("uncertainty", "measures.json")),
        "u_boot_2024": V(ms["2024"]["u_boot"]["median"], src("uncertainty", "measures.json")),
        "gap_2026": V(ms["2026"]["gap"]["mean"], src("uncertainty", "measures.json"), ms["2026"]["gap"]["ci95"]),
        "n_filters": V(filt["n_variants"], s_f), "any_promising": V(filt["any_promising"], s_f),
        "best_filter": V(b["diff"], s_f, b["ci95"], name=br, p_holm=b.get("p_holm")),
        "bag_2026_single": V(bag["2026"]["seed"]["gap_single"], src("uncertainty", "bagging.json")),
        "bag_2026_boot": V(bag["2026"]["boot"]["gap_ensemble"], src("uncertainty", "bagging.json"),
                           bag["2026"]["boot"]["gap_ensemble_ci95"]),
        "u1x2_2026": V(u26["beta_u"], s_p, u26["beta_u_ci95"]),
        "u1x2_select": V(pool["u_1x2"]["beta_u"], s_p, pool["u_1x2"]["beta_u_ci95"], p_holm=pool["u_1x2"]["p_holm"]),
        "decisions": V({k: v["decision"] for k, v in dec.items() if isinstance(v, dict) and "decision" in v},
                       src("uncertainty", "decisions.json")),
    }


def forward() -> dict:
    reg, eta, sim, fid, cur = (J("forward", n) for n in ("registry.json", "eta.json", "simulation.json",
                                                        "fidelity.json", "current_run.json"))
    s_r, s_e = src("forward", "registry.json"), src("forward", "eta.json")
    shadows = []
    for s in reg["shadows"]:
        e = eta["shadows"].get(s["key"], {})
        shadows.append({"key": s["key"], "family": s["family"], "hypothesis": s["hypothesis"],
                        "description": s["description"], "criterion": s["criterion"], "metric": s["metric"],
                        "min_n": s["min_n"], "implemented": s["implemented"],
                        "reason": s["not_implemented_reason"], "frozen_at_utc": s["frozen_at_utc"],
                        "prereg": s["prereg"], "earliest": e.get("earliest_date"),
                        "per_week": e.get("cases_per_week"), "src": f"{s_r}; {s_e}"})
    w = fid["x5_weekly_w"]
    return {
        "shadows": V(shadows, s_r),
        "prereg_files": V(sorted(reg["preregistrations"]), s_r),
        "played_per_week": V(eta["played_per_week"], s_e),
        "sim_runs": V(sim["n_runs"], src("forward", "simulation.json")),
        "sim_matches": V(sim["live_matches"], src("forward", "simulation.json")),
        "sim_after_kickoff": V(sim["rows_recorded_at_or_after_kickoff"], src("forward", "simulation.json")),
        "sim_tamper": V(sim["tamper_detected"], src("forward", "simulation.json")),
        "sim_prod_unchanged": V(sim["production_unchanged"], src("forward", "simulation.json")),
        "x5_w_max_diff": V(max(abs(x["diff"]) for x in w), src("forward", "fidelity.json")),
        "x5_w_first_diff_week": V(next((x["week"] for x in w if abs(x["diff"]) > 1e-9), None),
                                  src("forward", "fidelity.json")),
        "platt_prereg_n": V(fid["ps_platt_current_fit"]["preregistered"]["fitted_on_settled_shown_mains"],
                            src("forward", "fidelity.json")),
        "platt_shadow_n": V(fid["ps_platt_current_fit"]["shadow"]["n"], src("forward", "fidelity.json")),
        "current_prod_unchanged": V(cur["production_unchanged"], src("forward", "current_run.json")),
    }


def data_sources() -> dict:
    op, af = J("data_sources", "oddspapi.json"), J("data_sources", "api_football.json")
    s_o, s_af = src("data_sources", "oddspapi.json"), src("data_sources", "api_football.json")
    overall = {r["role"]: r for r in CSV("data_sources", "bets_overall.csv")}
    by_src = {(r["role"], r["source"]): r for r in CSV("data_sources", "bets_by_source.csv")}
    groups = {(r["group"], r["role"]): r for r in CSV("data_sources", "bets_by_group.csv")}
    s_ov, s_bs, s_g = (src("data_sources", n) for n in ("bets_overall.csv", "bets_by_source.csv",
                                                       "bets_by_group.csv"))
    text = FINDINGS("data_sources")
    toa = re.search(r"The Odds API:\*\* (\d[\d ]*) Main/Risk \(([\d.]+) %\)", text)
    v61 = re.search(r"в v6\.1: ([\d.]+) % и ([\d.]+) %", text)
    out = {
        "op_played": V(op["played"], s_o), "op_with": V(op["with_oddspapi"], s_o),
        "op_share": V(op["share_oddspapi"], s_o),
        "op_catalogue": V(op["markets"]["catalogue_markets"], s_o),
        "op_stored": V(op["markets"]["markets_stored"], s_o),
        "op_used": V(op["markets"]["markets_used_by_predictions"], s_o),
        "op_billable": V({m["month"]: m["billable"] for m in op["costs"]["per_month"]}, s_o),
        "op_free_plan": V(op["costs"]["free_plan_per_month"], s_o),
        "af_categories": V(af["categories"], s_af),
        "af_lead_median_h": V(af["lead"]["median_h"], s_af), "af_lead_n": V(af["lead"]["n"], s_af),
        "af_delay_h": V(af["delay_after_kickoff"]["api_football"]["median_h"], s_af),
        "fd_delay_h": V(af["delay_after_kickoff"]["football_data"]["median_h"], s_af),
        "af_settleable": V(af["value"]["settleable_with_goals"], s_af),
        "af_main_risk": V(af["value"]["main_risk_predictions"], s_af),
        "toa_picks": V(int(toa.group(1).replace(" ", "")) if toa else None, rel("data_sources", "findings.md")),
        "toa_share": V(float(toa.group(2)) if toa else None, rel("data_sources", "findings.md")),
        "v61_cov_main": V(float(v61.group(1)) if v61 else None, rel("data_sources", "findings.md")),
        "v61_cov_risk": V(float(v61.group(2)) if v61 else None, rel("data_sources", "findings.md")),
    }
    for role in ("main", "risk"):
        r = overall[role]
        out[f"{role}_overall"] = V(float(r["roi"]) / 100, s_ov, n=int(r["bets"]), ev=float(r["mean_ev"]),
                                   coverage=float(r["coverage"]), avg_odds=float(r["avg_odds"]))
        for s in ("live", "backfill"):
            b = by_src[(role, s)]
            out[f"{role}_{s}"] = V(float(b["roi"]) / 100, s_bs, n=int(b["bets"]))
    out["groups"] = {f"{g}|{role}": V(float(r["roi"]) / 100, s_g, n=int(r["bets"]))
                     for (g, role), r in groups.items()}
    return out


def production() -> dict:
    """Texts below the WCAG threshold before / after the v6.2 design (reports/v6_2/data/contrast_*.json)."""
    out = {}
    for k in ("before", "after"):
        name = f"reports/v6_2/data/contrast_{k}.json"
        USED.add(name)
        pages = json.loads((ROOT / name).read_text(encoding="utf-8"))["pages"]
        for theme in ("dark", "light"):
            out[f"{k}|{theme}"] = V(sum(len(p["below"]) for n, p in pages.items() if n.endswith(theme)), name,
                                    pages=sum(1 for n in pages if n.endswith(theme)))
    return out


# ---------------------------------------------------------------------------------------------- cross-study
def cross(d: dict) -> dict:
    """Totals that compare studies (all computed from the values collected above)."""
    tried = {"xg": d["xg"]["n_variants"]["v"], "main_calibration": d["main_calibration"]["n_variants"]["v"],
             "risk": d["risk"]["n_rules"]["v"], "market_models": d["market_models"]["n_comparisons"]["v"],
             "goal_distributions": d["goal_distributions"]["n_variants"]["v"],
             "leagues": d["leagues"]["n_variants"]["v"], "temporal": d["temporal"]["n_variants"]["v"],
             "uncertainty": d["uncertainty"]["n_filters"]["v"] + d["uncertainty"]["n_measures"]["v"],
             "bookmaker": len(d["bookmaker"]["best_hybrid"]) * 5,
             "data_quality": d["data_quality"]["n_sens_variants"]["v"] + d["data_quality"]["n_dq_variants"]["v"]}
    eta = d["forward"]["played_per_week"]["v"]
    est = {}
    for k in ("goal_distributions", "market_models"):
        p = d[k]["prereg"]
        est[k] = (date.fromisoformat(p["written_at"][:10]) + timedelta(weeks=p["min_n"] / eta)).isoformat()
    return {"variants_tried": tried, "variants_total": sum(tried.values()),
            "eta_without_shadow": {"date": est, "rule": "дата записи предрегистрации + N / played_per_week "
                                                        "(experiments/v6_2/forward/results/eta.json)"}}


def collect() -> dict:
    USED.clear()
    data = {s: globals()[s]() for s in STUDIES}
    data["cross"] = cross(data)
    data["production"] = production()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = None
    data["meta"] = {"built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": sha,
                    "reports": REPORTS, "files_read": sorted(USED)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


if __name__ == "__main__":
    d = collect()
    print(f"{OUT.relative_to(ROOT)}: {len(d['meta']['files_read'])} files read")
