"""Overview of the 13 v6.3 studies, read from their result files (no number is typed by hand).

    .venv/bin/python -m experiments.v6_3.summary.overview

For every study: the question, the variants tried, the matches used, its preregistered B candidates with their
Main figures in both universes (ROI, hit rate, average odds, bets, ROI difference to Baseline B with the paired
95 % interval, raw and Holm p as the study computed them) and the model-metric effect, the decision. The "best
candidate" of a study is chosen among its B candidates by the evidence criterion of the plan (positive in both
universes -> smallest Holm p at close -> smallest worse raw p), never by the highest ROI.

Writes ``results/overview.json``, ``results/overview.csv`` (one row per study) and ``results/b_candidates.csv``
(one row per B candidate).
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C

DIRS = C.study_dirs("summary")
RES = DIRS["results"]
V = C.V63
U = ("close", "t09")


def j(path: str):
    return C.read_json(V / path)


def uni(summary: dict, vs_b: dict | None, holm_p=None, roi_ci=None) -> dict:
    """One universe of a candidate in the common format."""
    rd = (vs_b or {}).get("roi_diff", {})
    return {"roi": summary.get("roi"), "hit_rate": summary.get("hit_rate"), "avg_odds": summary.get("avg_odds"),
            "bets": summary.get("bets"), "profit": summary.get("profit"),
            "roi_ci": (roi_ci or {}).get("ci95") if isinstance(roi_ci, dict) else roi_ci,
            "d_roi": rd.get("diff"), "d_ci": rd.get("ci95"), "p": rd.get("p"), "p_holm": holm_p,
            "matches": (vs_b or {}).get("matches"),
            "d_hit": (vs_b or {}).get("hit_rate_diff", {}).get("diff")}


def cand(name: str, label: str, close: dict, t09: dict, model: str = "", **extra) -> dict:
    return {"name": name, "label": label, "close": close, "t09": t09, "model_effect": model, **extra}


def ms_rows(path: str, names: list[str], labels: dict) -> list[dict]:
    """main_selection / market_profitability candidate tables."""
    rows = j(path)["candidates"]
    out = []
    for n in names:
        by = {r["universe"]: r for r in rows if r["candidate"] == n}
        def one(r):
            return {"roi": r["roi"], "hit_rate": r["hit_rate"], "avg_odds": r["avg_odds"], "bets": r["bets"],
                    "profit": r["profit"], "roi_ci": r["roi_ci"], "d_roi": r["roi_diff"], "d_ci": r["diff_ci"],
                    "p": r["p"], "p_holm": r["p_holm"], "matches": r.get("common_matches"),
                    "d_hit": r.get("hit_diff")}
        out.append(cand(n, labels[n], one(by["close"]), one(by["t09"]), "—", changed_share=by["close"]["changed_share"]))
    return out


# ------------------------------------------------------------------------------------------------ studies
def main_selection() -> dict:
    d = j("main_selection/results/main_selection.json")
    labels = {"bt:reliability": "надёжность TOTAL 0.95", "bt:ev_weight": "вес EV 1.0",
              "floor_1.40": "порог 1.40 + «не ставка»"}
    return {"key": "main_selection", "title": "1. Main Selection 3.0", "pdf": "Main_Selection_Research.pdf",
            "question": "Какое правило выбора одного Main на матч даёт больший ROI?",
            "variants": f"{d['n_variants_grid']} правил в сетке, {d['n_candidates']} заранее выбранных кандидатов",
            "n_variants": d["n_variants_grid"],
            "matches": d["universe_sizes"]["close"]["matches"],
            "selection": f"2025/26 бэктест ({d['universe_sizes']['bt']['matches']} матчей) и расширяющееся окно",
            "b": ms_rows("main_selection/results/main_selection.json", list(labels), labels),
            "holm_family": f"{d['n_candidates']} кандидатов, отдельно по вселенным"}


def calibration() -> dict:
    d = j("calibration/results/calibration.json")
    pre = j("calibration/results/preregistration.json")
    names = [e["id"].replace("v63_calibration_", "").replace("@v1", "").replace("_", "|") for e in pre["entries"]]
    raw_ll = d["quality"]["raw|-|-"]["band"]["log_loss"]
    out = []
    for n in names:
        if n not in d["main"]["close"]:
            continue
        x = {u: d["main"][u][n] for u in U}
        ll = d["quality"][n]["band"]["log_loss"]
        out.append(cand(n, f"{n.split('|')[0]} по {'группам' if '|group|' in n else 'ключам'}",
                        uni(x["close"]["summary"], x["close"]["vs_b"], d["main"]["close"]["holm"].get(n),
                            x["close"]["roi_ci"]),
                        uni(x["t09"]["summary"], x["t09"]["vs_b"], d["main"]["t09"]["holm"].get(n), x["t09"]["roi_ci"]),
                        f"log loss кандидатов полосы Main {raw_ll:.5f} → {ll:.5f}"))
    return {"key": "calibration", "title": "2. Model Probability Calibration",
            "pdf": "Probability_Calibration_Research.pdf",
            "question": "Улучшает ли калибровка вероятностей точность и ROI Main?",
            "variants": f"{len(d['quality']) - 1} карт калибровки, {len(d['family'])} в проверке Main",
            "n_variants": len(d["quality"]) - 1, "matches": d["main"]["close"]["baseline_b"]["vs_b"]["matches"],
            "selection": f"2025/26 бэктест ({d['data']['backtest_matches']} матчей) + месяцы 2026/27 до матча",
            "b": out, "holm_family": f"{len(d['family'])} карт, отдельно по вселенным"}


def market_aware() -> dict:
    d = j("market_aware/results/market_aware.json")
    ph = d["holm"]["p_holm"]
    gap = d["demargin"]["close"]["1X2"]["model_minus_power"]
    wf = d["information"]["close"]["walk_forward"]["1X2"]["blend_minus_model"]
    x = {u: d["variants"][u]["blend_5050"] for u in U}
    b = [cand("blend_5050", "V6 смесь 50/50 модели и рынка",
              uni(x["close"]["summary"], x["close"]["vs_b"], ph.get("blend_5050|close"), x["close"]["roi_ci"]),
              uni(x["t09"]["summary"], x["t09"]["vs_b"], ph.get("blend_5050|t09"), x["t09"]["roi_ci"]),
              f"1X2: модель − рынок {gap['diff'] * 1e3:+.1f}·10⁻³; смесь − модель {wf['diff'] * 1e3:+.1f}·10⁻³ "
              f"(walk-forward, {wf['n']} матчей)")]
    return {"key": "market_aware", "title": "3. Market-Aware Main", "pdf": "Market_Aware_Main_Research.pdf",
            "question": "Помогает ли учёт рыночной вероятности при выборе Main?",
            "variants": f"{d['n_variants']} вариантов", "n_variants": d["n_variants"],
            "matches": x["close"]["vs_b"]["matches"], "selection": "веса по прошлым месяцам 2026/27; v6.2 2022–2025",
            "b": b, "holm_family": f"{d['holm']['n_tests']} тестов (варианты × вселенные)"}


def market_profitability() -> dict:
    d = j("market_profitability/results/market_profitability.json")
    labels = {"bt:boost": "надёжность HANDICAP ×1.15 (1.035)"}
    return {"key": "market_profitability", "title": "4. Profitability by Market",
            "pdf": "Market_Profitability_Research.pdf",
            "question": "Какие рынки прибыльны и стоит ли менять их приоритет?",
            "variants": f"{d['n_variants_grid']} правил в сетке, {d['n_candidates']} кандидатов",
            "n_variants": d["n_variants_grid"],
            "matches": d["baselines"]["close"]["summary"]["matches"],
            "selection": "2025/26 бэктест и расширяющееся окно",
            "b": ms_rows("market_profitability/results/market_profitability.json", list(labels), labels),
            "holm_family": f"{d['n_candidates']} кандидатов, отдельно по вселенным"}


def timing() -> dict:
    d = j("timing/results/timing.json")
    s = d["moments"]["summary"]
    same = d["same_picks"]["t09+close"]["close_minus_t09"]["roi_diff"]
    clv = d["clv"]
    return {"key": "timing", "title": "5. Timing & Closing Line Value", "pdf": "Timing_CLV_Research.pdf",
            "question": "Когда формировать Main и что даёт движение линии?",
            "variants": f"{len(s)} момента решения + живые версии; {d['holm']['n_tests']} тестов",
            "n_variants": len(s), "matches": s["close"]["matches"], "selection": "—", "b": [],
            "holm_family": f"{d['holm']['n_tests']} тестов",
            "note": (f"Main за 48 ч: ROI {s['early']['roi'] * 100:+.1f}% на {s['early']['bets']} ставках; T: "
                     f"{s['t09']['roi'] * 100:+.1f}% ({s['t09']['bets']}); закрытие {s['close']['roi'] * 100:+.1f}% "
                     f"({s['close']['bets']}). Те же Main T по цене закрытия: {same['diff'] * 100:+.2f} п.п. "
                     f"[{same['ci95'][0] * 100:+.2f}; {same['ci95'][1] * 100:+.2f}]. CLV Main T: ценовой "
                     f"{clv['clv_price']['mean'] * 100:+.2f}%, справедливый {clv['clv_fair']['mean'] * 100:+.2f}%."),
            "decision_note": "правил-кандидатов B нет; B — мониторинг CLV"}


def quality() -> dict:
    d = j("quality/results/mitigations_2026.json")
    pre = j("quality/results/preregistration.json")
    labels = {"mkt_season_0.75": "смесь 1X2 с рынком по нехватке данных сезона (w 0.75)",
              "mkt_both": "смесь 1X2 с рынком по нехватке данных (сезон + история)",
              "mkt_uniform": "равномерная смесь 1X2 с рынком (контроль)"}
    out = []
    for n in pre:
        v = d["variants"][n]["universes"]
        ll = v["close"]["d_log_loss"]
        out.append(cand(n, labels.get(n, n),
                        uni(v["close"]["summary"], v["close"]["vs_b"], v["close"].get("holm_p"), v["close"]["roi_ci"]),
                        uni(v["t09"]["summary"], v["t09"]["vs_b"], v["t09"].get("holm_p"), v["t09"]["roi_ci"]),
                        f"1X2 log loss {ll['diff'] * 1e3:+.2f}·10⁻³ [{ll['ci95'][0] * 1e3:+.2f}; "
                        f"{ll['ci95'][1] * 1e3:+.2f}]",
                        d_log_loss=ll, d_goal_dev=v["close"].get("d_goal_dev"),
                        d_corner_dev=v["close"].get("d_corner_dev")))
    return {"key": "quality", "title": "6. Quality vs Coverage", "pdf": "Quality_Coverage_Research.pdf",
            "question": "Где модель слабее из-за нехватки или качества данных и как это смягчить?",
            "variants": f"{d['n_variants_tried']} смягчений", "n_variants": d["n_variants_tried"],
            "matches": out[0]["close"]["matches"], "selection": "2024/25 + 2025/26 (вне выборки)", "b": out,
            "holm_family": f"{d['n_variants_on_2026']} вариантов, отдельно по вселенным"}


def recent_weighting() -> dict:
    d = j("recent_weighting/results/results.json")
    roi = pd.read_csv(V / "recent_weighting/results/main_roi.csv").set_index(["variant", "universe"])
    labels = {"wf:cur2": "ежемесячно, текущий сезон ×2 (B для 1X2)", "wf:hl0": "ежемесячно, без затухания (B для голов)"}
    out = []
    for n, lab in labels.items():
        sch = n.split(":")[1]
        pm = d["paired"][f"2026|wf|{sch}|vs_B"]

        def one(u):
            r = roi.loc[(n, u)]
            return {"roi": r["roi"], "hit_rate": r["hit_rate"], "avg_odds": r["avg_odds"], "bets": int(r["bets"]),
                    "profit": r["profit"], "roi_ci": [r["roi_lo"], r["roi_hi"]], "d_roi": r["d_roi"],
                    "d_ci": [r["d_lo"], r["d_hi"]], "p": r["p"], "p_holm": r["p_holm"], "matches": None,
                    "d_hit": None}
        out.append(cand(n, lab, one("close"), one("t09"),
                        f"Δ к B ·10⁻³: 1X2 {pm['result']['diff'] * 1e3:+.2f}, голы {pm['goals']['diff'] * 1e3:+.2f}, "
                        f"угловые {pm['corners']['diff'] * 1e3:+.2f}", paired=pm))
    nv = d["n_variants"]
    return {"key": "recent_weighting", "title": "7. Recent Season Weighting",
            "pdf": "Recent_Season_Weighting_Research.pdf",
            "question": "Помогает ли больший вес текущего сезона и частое переобучение?",
            "variants": f"{nv['monthly_schemes']} схем ежемесячно + {nv['static_schemes']} однократно + "
                        f"{len(nv['other'])} прочих; {nv['roi_variants']} вариантов с ROI",
            "n_variants": nv["roi_variants"], "matches": d["coverage"]["played"],
            "selection": "2025/26 (ежемесячное переобучение)", "b": out,
            "holm_family": f"{nv['roi_variants']} вариантов, отдельно по вселенным"}


def alt_ml() -> dict:
    d = j("alt_ml/results/main_2026.json")
    m = j("alt_ml/results/metrics_2026.json")
    labels = {"ens_xgb_best": "ансамбль XGBoost + лучший альтернативный", "best_per_task": "лучший по задаче"}
    out = []
    for n, lab in labels.items():
        x = d["variants"][n]
        pm = m[n]["paired"]
        out.append(cand(n, lab, uni(x["close"]["summary"], x["close"]["vs_b"], x["close"]["holm_p"], x["close"]["roi_ci"]),
                        uni(x["t09"]["summary"], x["t09"]["vs_b"], x["t09"]["holm_p"], x["t09"]["roi_ci"]),
                        f"·10⁻³: log loss 1X2 {pm['ll']['diff'] * 1e3:+.2f}, девиация голов хозяев "
                        f"{pm['dev_home_goals']['diff'] * 1e3:+.2f} [{pm['dev_home_goals']['ci95'][0] * 1e3:+.2f}; "
                        f"{pm['dev_home_goals']['ci95'][1] * 1e3:+.2f}], угловых {pm['dev_corners']['diff'] * 1e3:+.2f}",
                        paired=pm))
    return {"key": "alt_ml", "title": "8. Alternative ML Algorithms", "pdf": "Alternative_ML_Research.pdf",
            "question": "Лучше ли другие алгоритмы ML, чем XGBoost продакшна?",
            "variants": f"{d['n_variants']} вариантов (6 алгоритмов, ансамбли, ежемесячные)",
            "n_variants": d["n_variants"], "matches": out[0]["close"]["matches"], "selection": "2025/26",
            "b": out, "holm_family": f"{d['n_variants']} вариантов, отдельно по вселенным"}


def meta_selection() -> dict:
    d = j("meta_selection/results/meta_selection.json")
    pq = d["close"]["prob_quality"]
    x = {u: d[u]["meta_lr_ev"] for u in U}
    b = [cand("meta_lr_ev", "логистическая метамодель, max EV",
              uni(x["close"]["summary"], x["close"]["vs_b"], d["close"]["holm"]["meta_lr_ev"], x["close"]["roi_ci"]),
              uni(x["t09"]["summary"], x["t09"]["vs_b"], d["t09"]["holm"]["meta_lr_ev"], x["t09"]["roi_ci"]),
              f"log loss пула: продакшн {pq['raw']['log_loss']:.4f}, рынок {pq['market_q']['log_loss']:.4f}, "
              f"метамодель {pq['p_lr']['log_loss']:.4f}")]
    n = len(d["meta"]) + len(d["simple"])
    return {"key": "meta_selection", "title": "9. Meta-Selection Model", "pdf": "Meta_Selection_Research.pdf",
            "question": "Может ли метамодель выбирать Main лучше правила?",
            "variants": f"{n} вариантов ({len(d['meta'])} метамодели + {len(d['simple'])} простых)",
            "n_variants": n, "matches": x["close"]["vs_b"]["matches"],
            "selection": "пул 2025/26 + месяцы 2026/27 до матча", "b": b,
            "holm_family": f"{len(d['close']['holm'])} вариантов, отдельно по вселенным"}


def goals() -> dict:
    d = j("goals/results/results.json")
    roi = pd.read_csv(V / "goals/results/main_roi.csv").set_index(["variant", "universe"])
    pm = d["paired_2026"]["stacked"]

    def one(u):
        r = roi.loc[("stacked", u)]
        return {"roi": r["roi"], "hit_rate": r["hit_rate"], "avg_odds": r["avg_odds"], "bets": int(r["bets"]),
                "profit": r["profit"], "roi_ci": [r["roi_lo"], r["roi_hi"]], "d_roi": r["d_roi"],
                "d_ci": [r["d_lo"], r["d_hi"]], "p": r["p"], "p_holm": r["p_holm"], "matches": None, "d_hit": None}
    b = [cand("stacked", "единая модель голов команды", one("close"), one("t09"),
              f"·10⁻³: log loss счёта {pm['score_ll']['diff'] * 1e3:+.2f} [{pm['score_ll']['ci95'][0] * 1e3:+.2f}; "
              f"{pm['score_ll']['ci95'][1] * 1e3:+.2f}] (p Холма {d['holm_2026']['stacked']:.3f}), девиация голов "
              f"{pm['dev_goals']['diff'] * 1e3:+.2f}", paired=pm, prob_holm=d["holm_2026"]["stacked"])]
    return {"key": "goals", "title": "10. Goal Model Improvements", "pdf": "Goal_Model_Research.pdf",
            "question": "Можно ли точнее прогнозировать голы и рынки голов?",
            "variants": f"{d['n_variants']} вариантов", "n_variants": d["n_variants"],
            "matches": d["summary_2026"]["prod"]["n"], "selection": "2023/24–2025/26", "b": b,
            "holm_family": f"{d['n_variants']} вариантов"}


def corners() -> dict:
    d = j("corners/results/results.json")
    labels = {"best": "Poisson GLM + NB, r по лигам", "best_wf": "Poisson GLM ежемесячно + NB",
              "ens": "ансамбль (веса 2025/26) + NB"}
    mean_of = {"best": ("eval_mean", "glm"), "best_wf": ("wf_mean", "glm"), "ens": ("eval_mean", "ens_w")}
    prod = d["eval_mean"]["xgb_prod"]
    out = []
    for n, lab in labels.items():
        x = {u: d["main"][u][n] for u in U}
        sec, key = mean_of[n]
        mm = d[sec][key]
        eff = (f"MAE {prod['mae']:.3f} → {mm['mae']:.3f}, R² {prod['r2']:.3f} → {mm['r2']:.3f}, Δ девиации "
               f"{mm['dev_diff']['diff']:+.4f} [{mm['dev_diff']['ci95'][0]:+.4f}; {mm['dev_diff']['ci95'][1]:+.4f}]")
        out.append(cand(n, lab, uni(x["close"]["summary"], x["close"]["vs_b"], x["close"]["holm_p"], x["close"]["roi_ci"]),
                        uni(x["t09"]["summary"], x["t09"]["vs_b"], x["t09"]["holm_p"], x["t09"]["roi_ci"]), eff))
    return {"key": "corners", "title": "11. Corners Research", "pdf": "Corners_Research.pdf",
            "question": "Можно ли улучшить прогноз угловых?",
            "variants": f"{len(d['select_mean'])} моделей среднего, {len(d['select_dist'])} сочетаний с "
                        f"распределениями, {len(d['variants'])} вариантов Main",
            "n_variants": len(d["select_mean"]) + len(d["select_dist"]), "matches": prod["n"],
            "selection": "2025/26", "b": out, "holm_family": f"{len(d['variants']) - 1} вариантов, по вселенным"}


def squad() -> dict:
    e2 = j("squad/results/e2_season_2026.json")
    e1 = j("squad/results/e1_earlier_seasons.json")
    dec = j("squad/results/decisions.json")
    x = {u: e2["universes"][u]["variants"]["pre"]["covered"] for u in U}
    ll = e2["model"]["pre"]["vs_production"]["ll_1x2"]
    ll25 = e1["2025"]["variants"]["pre"]["covered"]["ll_1x2"]
    b = [cand("e2_pre", "предсезонная сила состава (стоимость игроков)",
              uni(x["close"]["summary"], x["close"]["vs_b"], x["close"]["vs_b"]["roi_diff"].get("p_holm"),
                  x["close"]["roi_ci"]),
              uni(x["t09"]["summary"], x["t09"]["vs_b"], x["t09"]["vs_b"]["roi_diff"].get("p_holm"), x["t09"]["roi_ci"]),
              f"1X2 log loss 2026/27 {ll['diff'] * 1e3:+.1f}·10⁻³ [{ll['ci95'][0] * 1e3:+.1f}; {ll['ci95'][1] * 1e3:+.1f}] "
              f"(p Холма {ll.get('p_holm')}); 2025/26 {ll25['diff'] * 1e3:+.1f}·10⁻³ (p Холма {ll25.get('p_holm')})",
              covered=e2["covered_played"])]
    return {"key": "squad", "title": "12. Squad Strength Research", "pdf": "Squad_Strength_Research.pdf",
            "question": "Даёт ли оценка силы состава информацию сверх модели?",
            "variants": f"{len(dec)} вариантов (E1 ранние сезоны, E2 2026/27)", "n_variants": len(dec),
            "matches": e2["covered_played"],
            "selection": f"2024/25, 2025/26 (покрыто {e1['2025']['matches_covered']} матчей 2025/26)",
            "b": b, "holm_family": e2["holm_family"]["what"] + f", {e2['holm_family']['tests']} тестов",
            "decisions": {k: v["decision"] for k, v in dec.items()}}


def consensus() -> dict:
    d = j("consensus/results/consensus.json")
    ph = d["holm"]["p_holm"]
    mw = d["model"]["close"]["walk_forward"]["1X2"]
    labels = {"cons_blend_wf": "C2 смесь модели с консенсусом книг", "cons_market_only": "C1 консенсус вместо p"}
    out = []
    for n, lab in labels.items():
        x = {u: d["variants"][u][n] for u in U}
        out.append(cand(n, lab, uni(x["close"]["summary"], x["close"]["vs_b"], ph.get(f"{n}|close"), x["close"].get("roi_ci")),
                        uni(x["t09"]["summary"], x["t09"]["vs_b"], ph.get(f"{n}|t09"), x["t09"].get("roi_ci")),
                        f"1X2 log loss (walk-forward, {mw['n']}): модель {mw['model']:.4f}, одна книга "
                        f"{mw['single']:.4f}, консенсус {mw['consensus']:.4f}, смесь с консенсусом "
                        f"{mw['blend_consensus']:.4f}"))
    return {"key": "consensus", "title": "13. Market Consensus", "pdf": "Market_Consensus_Research.pdf",
            "question": "Точнее ли консенсус нескольких букмекеров одной книги и помогает ли он Main?",
            "variants": f"{len(d['variants_ru'])} вариантов Main", "n_variants": len(d["variants_ru"]),
            "matches": out[0]["close"]["matches"], "selection": "FD 2021/22–2025/26; месяцы 2026/27 до матча",
            "b": out, "holm_family": f"{d['holm']['n_tests']} тестов (варианты × вселенные)"}


STUDIES = [main_selection, calibration, market_aware, market_profitability, timing, quality, recent_weighting,
           alt_ml, meta_selection, goals, corners, squad, consensus]


# ------------------------------------------------------------------------------------------------ criterion
def worse_p(c: dict) -> float:
    ps = [c[u]["p"] for u in U if c[u].get("p") is not None]
    return max(ps) if ps else 1.0


def holm_close(c: dict) -> float:
    v = c["close"].get("p_holm")
    return 1.0 if v is None or v != v else float(v)


def evidence_order(cands: list[dict]) -> list[dict]:
    """Positive ROI difference in both universes first, then the smallest Holm p at close, then the smaller worse
    raw p — the criterion frozen in ``combinations_plan.json``."""
    pos = [c for c in cands if all((c[u].get("d_roi") or -1) > 0 for u in U)]
    rest = [c for c in cands if c not in pos]
    key = lambda c: (holm_close(c), worse_p(c))  # noqa: E731
    return sorted(pos, key=key) + sorted(rest, key=key)


def run() -> dict:
    studies = [f() for f in STUDIES]
    rows, brows = [], []
    for s in studies:
        order = evidence_order(s["b"])
        s["best"] = order[0]["name"] if order else None
        s["best_positive_both"] = bool(order and all((order[0][u].get("d_roi") or -1) > 0 for u in U))
        s["decision"] = (f"A — нет; B — {len(s['b'])}; остальные — C" if s["b"]
                         else "A — нет; " + s.get("decision_note", "B — нет"))
        best = order[0] if order else None
        row = {"study": s["title"], "question": s["question"], "variants": s["variants"],
               "n_variants": s["n_variants"], "matches": s["matches"], "selection": s["selection"],
               "b_candidates": ", ".join(c["name"] for c in s["b"]), "best": best["name"] if best else "",
               "best_label": best["label"] if best else "", "model_effect": best["model_effect"] if best else
               s.get("note", ""), "decision": s["decision"], "holm_family": s["holm_family"], "pdf": s["pdf"]}
        for u in U:
            for k in ("roi", "hit_rate", "avg_odds", "bets", "d_roi", "p", "p_holm"):
                row[f"{k}_{u}"] = best[u].get(k) if best else None
            row[f"d_lo_{u}"] = best[u]["d_ci"][0] if best and best[u].get("d_ci") else None
            row[f"d_hi_{u}"] = best[u]["d_ci"][1] if best and best[u].get("d_ci") else None
        rows.append(row)
        for c in s["b"]:
            br = {"study": s["title"], "study_key": s["key"], "candidate": c["name"], "label": c["label"],
                  "model_effect": c["model_effect"], "holm_family": s["holm_family"]}
            for u in U:
                for k in ("roi", "hit_rate", "avg_odds", "bets", "profit", "d_roi", "p", "p_holm", "matches"):
                    br[f"{k}_{u}"] = c[u].get(k)
                br[f"d_lo_{u}"], br[f"d_hi_{u}"] = (c[u]["d_ci"] or [None, None]) if c[u].get("d_ci") else (None, None)
                br[f"roi_lo_{u}"], br[f"roi_hi_{u}"] = (c[u]["roi_ci"] or [None, None]) if c[u].get("roi_ci") else (None, None)
            brows.append(br)
    out = {"created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "git_sha": C.git_sha(),
           "studies": studies, "n_studies": len(studies),
           "criterion": "positive ROI difference vs Baseline B in both universes -> smallest Holm p at close -> "
                        "smallest max(raw p close, raw p t09)"}
    C.write_json(RES / "overview.json", out)
    pd.DataFrame(rows).to_csv(RES / "overview.csv", index=False)
    pd.DataFrame(brows).to_csv(RES / "b_candidates.csv", index=False)
    return out


if __name__ == "__main__":
    o = run()
    for s in o["studies"]:
        print(s["title"], "| best:", s["best"], "| B:", [c["name"] for c in s["b"]])
