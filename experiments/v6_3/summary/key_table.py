"""The owner's key table: Current Production vs Best Validated Experimental Candidate.

    .venv/bin/python -m experiments.v6_3.summary.key_table     (after overview and combine)

The candidate is chosen by the criterion frozen in ``results/combinations_plan.json`` before the combinations were
evaluated — by evidence, not by ROI: among the preregistered B candidates of the 13 studies and the 12
combinations, (1) a positive ROI difference to Baseline B in both universes, (2) the smallest Holm-corrected p at
close (Holm within the candidate's own family), (3) the smallest worse raw p of the two universes.
Writes ``results/production_vs_best.json`` and ``results/production_vs_best.csv``.
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C

DIRS = C.study_dirs("summary")
RES = DIRS["results"]
V = C.V63
U = ("close", "t09")


def pool() -> list[dict]:
    """Every candidate in one format: name, source, both universes."""
    out = []
    b = pd.read_csv(RES / "b_candidates.csv")
    for r in b.to_dict("records"):
        rec = {"name": r["candidate"], "source": r["study"], "study_key": r["study_key"], "label": r["label"],
               "holm_family": r["holm_family"], "model_effect": r["model_effect"]}
        for u in U:
            rec[u] = {k: r.get(f"{k}_{u}") for k in ("roi", "hit_rate", "avg_odds", "bets", "profit", "d_roi", "d_lo",
                                                     "d_hi", "p", "p_holm", "roi_lo", "roi_hi")}
        out.append(rec)
    comb = C.read_json(RES / "combinations.json")
    t = pd.DataFrame(comb["table"])
    for vid, g in t.loc[t["role"] == "combination"].groupby("id"):
        g = g.set_index("universe")
        rec = {"name": vid, "source": "Комбинации (этап 6)", "study_key": "combination",
               "label": g.loc["close", "parts"], "holm_family": "12 комбинаций, отдельно по вселенным",
               "model_effect": ""}
        for u in U:
            r = g.loc[u]
            rec[u] = {"roi": r["roi"], "hit_rate": r["hit_rate"], "avg_odds": r["avg_odds"], "bets": r["bets"],
                      "profit": r["profit"], "d_roi": r["d_roi"], "d_lo": r["d_lo"], "d_hi": r["d_hi"], "p": r["p"],
                      "p_holm": r["p_holm"], "roi_lo": r["roi_lo"], "roi_hi": r["roi_hi"]}
        out.append(rec)
    return out


def order(cands: list[dict]) -> list[dict]:
    def hc(c):
        v = c["close"]["p_holm"]
        return 1.0 if v is None or v != v else float(v)

    def worse(c):
        return max(float(c[u]["p"]) for u in U)
    pos = [c for c in cands if all(float(c[u]["d_roi"]) > 0 for u in U)]
    rest = [c for c in cands if c not in pos]
    for c in pos:
        c["positive_both"] = True
    for c in rest:
        c["positive_both"] = False
    return sorted(pos, key=lambda c: (hc(c), worse(c))) + sorted(rest, key=lambda c: (hc(c), worse(c)))


def production_metrics() -> dict:
    alt = C.read_json(V / "alt_ml/results/metrics_2026.json")
    goals = C.read_json(V / "goals/results/results.json")["summary_2026"]["prod"]
    corners = C.read_json(V / "corners/results/results.json")["eval_mean"]["xgb_prod"]
    r = alt["production"]["metrics"]["result"]
    return {"1x2_log_loss": r["log_loss"], "1x2_accuracy": r["accuracy"], "1x2_brier": r["brier"],
            "market_1x2_log_loss_close": alt["_market_close"]["metrics"]["result"]["log_loss"],
            "goals_score_log_loss": goals["score_ll"], "goals_poisson_deviance": goals["dev_goals"],
            "goals_mae": goals["mae_goals"], "corners_mae": corners["mae"], "corners_r2": corners["r2"],
            "corners_poisson_deviance": corners["poisson_dev"], "n": r["n"]}


def candidate_metrics(c: dict, prod: dict) -> dict:
    """Model metrics of the chosen candidate from the owning study (production values where it changes nothing)."""
    out = dict(prod)
    src = "правило выбора Main не меняет модели — метрики как у продакшна"
    parts = c["label"].split("+") if c["study_key"] == "combination" else []
    if c["study_key"] == "quality":
        q = C.read_json(V / "quality/results/mitigations_2026.json")["variants"][c["name"]]["universes"]["close"]
        out["1x2_log_loss"] = prod["1x2_log_loss"] + q["d_log_loss"]["diff"]
        out["d_1x2_log_loss"] = q["d_log_loss"]
        out["d_goals_deviance"] = q["d_goal_dev"]
        out["d_corners_deviance"] = q["d_corner_dev"]
        out["1x2_accuracy"] = out["1x2_brier"] = None
        src = "смесь вероятностей 1X2 с рынком: меняется только 1X2 (Δ из quality/results/mitigations_2026.json)"
    elif "GOALS" in parts or "CORNERS" in parts or c["study_key"] in ("goals", "corners"):
        g = C.read_json(V / "goals/results/results.json")
        cn = C.read_json(V / "corners/results/results.json")["eval_mean"]["glm"]
        if "GOALS" in parts or c["study_key"] == "goals":
            out["goals_score_log_loss"] = g["summary_2026"]["stacked"]["score_ll"]
            out["goals_poisson_deviance"] = g["summary_2026"]["stacked"]["dev_goals"]
            out["goals_mae"] = g["summary_2026"]["stacked"]["mae_goals"]
        if "CORNERS" in parts or c["study_key"] == "corners":
            out["corners_mae"], out["corners_r2"], out["corners_poisson_deviance"] = cn["mae"], cn["r2"], cn["poisson_dev"]
        src = "голы — единая модель голов команды, угловые — Poisson GLM (результаты исследований 10 и 11)"
    elif c["study_key"] not in ("main_selection", "market_profitability", "combination"):
        src = "см. эффект на метрики модели в исследовании: " + str(c.get("model_effect", ""))
        out = {k: None for k in prod}
    out["source"] = src
    return out


def run() -> dict:
    cands = order(pool())
    best = cands[0]
    comb = C.read_json(RES / "combinations.json")
    prod = production_metrics()
    out = {"criterion": C.read_json(RES / "combinations_plan.json")["protocol"]["best_candidate_criterion"],
           "n_pool": len(cands), "n_positive_both": int(sum(c["positive_both"] for c in cands)),
           "best": best, "ranking": [{"name": c["name"], "source": c["source"], "label": c["label"],
                                      "positive_both": c["positive_both"], "p_holm_close": c["close"]["p_holm"],
                                      "p_close": c["close"]["p"], "p_t09": c["t09"]["p"],
                                      "d_close": c["close"]["d_roi"], "d_t09": c["t09"]["d_roi"]} for c in cands],
           "baseline_b": comb["baseline_b"], "baseline_a": comb["baseline_a"],
           "production_metrics": prod, "candidate_metrics": candidate_metrics(best, prod),
           "best_rule_candidate": next(c for c in cands if c["study_key"] in ("main_selection",
                                                                              "market_profitability")),
           "best_combination": next(c for c in cands if c["study_key"] == "combination")}
    rows = []
    bb, ba = comb["baseline_b"], comb["baseline_a"]
    rows.append({"method": "Baseline A (история сайта)", "universe": "site", **_fig(ba["summary"], ba["roi_ci"])})
    for u in U:
        rows.append({"method": "Baseline B (продакшн, replay)", "universe": u, **_fig(bb[u]["summary"], bb[u]["roi_ci"])})
        c = best[u]
        rows.append({"method": f"{best['name']} ({best['source']})", "universe": u, "roi": c["roi"],
                     "roi_lo": c["roi_lo"], "roi_hi": c["roi_hi"], "profit": c["profit"], "hit_rate": c["hit_rate"],
                     "avg_odds": c["avg_odds"], "bets": c["bets"], "d_roi": c["d_roi"], "d_lo": c["d_lo"],
                     "d_hi": c["d_hi"], "p": c["p"], "p_holm": c["p_holm"]})
    pd.DataFrame(rows).to_csv(RES / "production_vs_best.csv", index=False)
    C.write_json(RES / "production_vs_best.json", out)
    return out


def _fig(s: dict, ci: dict) -> dict:
    return {"roi": s["roi"], "roi_lo": ci["ci95"][0], "roi_hi": ci["ci95"][1], "profit": s["profit"],
            "hit_rate": s["hit_rate"], "avg_odds": s["avg_odds"], "bets": s["bets"]}


if __name__ == "__main__":
    o = run()
    print("best:", o["best"]["name"], o["best"]["source"])
    for r in o["ranking"][:6]:
        print(r)
