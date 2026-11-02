"""Q6 — before / after summary across tasks (reads results/*.json of Q1–Q5).

Writes results/summary.json and results/summary_tables.md (markdown tables pasted into findings.md),
plus figures/summary_before_after.png.
"""
from __future__ import annotations

import numpy as np

from src import config

from . import common as C

log = C.setup_logging("summary")
F = [str(s) for s in C.FOLDS]

COMPLEXITY = {
    "a_baseline": "production, 1 model",
    "b_w1.2": "1 refit with draw sample weight", "b_w1.4": "1 refit with draw sample weight",
    "b_balanced": "1 refit with inverse-frequency weights",
    "c1_temperature": "+1 inner model, 1 parameter", "c2_vector": "+1 inner model, 5 parameters",
    "c3_draw_bias": "+1 inner model, 1 parameter",
    "d_draw_model_blend": "+ binary draw XGBoost (2 fits + early stopping), 1 weight",
    "e0_poisson_only": "goal models only (already in production)",
    "e_poisson_blend": "+ inner result & goal models, 1 weight",
    "f0_dixon_coles_only": "goal models + rho", "f_dixon_coles_blend": "+ inner models, rho + 1 weight",
    "ec2_poisson_blend_vector": "+ inner models, 1 weight + 5 parameters",
    "naive_constant": "no model", "naive_league": "no model",
    "xgb_production": "production, 1 model", "poisson_glm": "linear GLM, L2 tuned on inner season",
    "blend": "XGBoost + GLM, 1 weight", "naive_league_mean": "no model",
    "xgb_enriched": "+ ~40 rolling features, trees by early stopping",
    "xgb_clean": "corner-only subset + rolling features", "xgb_home_away": "2 models + NB convolution",
    "poisson_glm_enriched": "GLM + rolling features", "negbin_glm_clean": "NB GLM (statsmodels), slow",
}


def fmt(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f} %"


def draw_tables(r):
    sel = r["probability_selection"]["chosen"]
    lines = ["| Вариант | LL 2023 | LL 2024 | LL 2025* | Brier 2023 | Brier 2024 | Brier 2025* | ECE(D) 2024 | "
             "ср. P(D) 2024 | доля D argmax 2024 | лучше на обоих? |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in r["folds"]["2023"]["variants"]:
        v = [r["folds"][f]["variants"][n] for f in F]
        ok = r["probability_selection"]["table"].get(n, {})
        both = "—" if not ok else ("да" if ok["better_log_loss_both"] and ok["better_brier_both"] else "нет")
        lines.append(f"| {n}{' **(выбран)**' if n == sel else ''} | {fmt(v[0]['log_loss'])} | {fmt(v[1]['log_loss'])} | "
                     f"{fmt(v[2]['log_loss'])} | {fmt(v[0]['brier'])} | {fmt(v[1]['brier'])} | {fmt(v[2]['brier'])} | "
                     f"{fmt(v[1]['ece']['D'])} | {fmt(v[1]['mean_probability']['D'], 3)} | "
                     f"{pct(v[1]['predicted_draw_share'])} | {both} |")
    bm = ["| Сезон | n с ценами | LL букмекер | LL baseline | LL выбранный | Brier букмекер | Brier baseline |",
          "|---|---|---|---|---|---|---|"]
    for f in F:
        b = r["folds"][f]["bookmaker_same_rows"]
        bm.append(f"| {f} | {b['n']} | {fmt(b['bookmaker']['log_loss'])} | {fmt(b['a_baseline']['log_loss'])} | "
                  f"{fmt(b[sel]['log_loss'])} | {fmt(b['bookmaker']['brier'])} | {fmt(b['a_baseline']['brier'])} |")
    cls = ["| Вариант | Сезон | Accuracy | Macro F1 | Bal. acc | Draw prec | Draw recall | Home recall | Away recall | "
           "Доля D пред. | Доля D факт |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in r["folds"]["2023"]["variants"]:
        for f in F:
            v = r["folds"][f]["variants"][n]
            cls.append(f"| {n} | {f} | {pct(v['accuracy'])} | {fmt(v['macro_f1'], 3)} | {fmt(v['balanced_accuracy'], 3)} | "
                       f"{pct(v['draw_precision'])} | {pct(v['draw_recall'])} | {pct(v['home_recall'])} | "
                       f"{pct(v['away_recall'])} | {pct(v['predicted_draw_share'])} | {pct(v['actual_draw_share'])} |")
    rules = ["| Модель вероятностей | Правило | Сезон | Accuracy | Macro F1 | Bal. acc | Draw prec | Draw recall | "
             "Доля D пред. | ROI исхода Bet365 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for model, d in r["decision_rules"].items():
        name = d["final_rule"]["rule"]
        for f in F:
            for rname, m in (("argmax", r["folds"][f]["variants"][model]), (name, d["final_rule_metrics"][f])):
                roi = d["roi_named_outcome_bet365"][rname][f]["roi"]
                rules.append(f"| {model} | {rname} | {f} | {pct(m['accuracy'])} | {fmt(m['macro_f1'], 3)} | "
                             f"{fmt(m['balanced_accuracy'], 3)} | {pct(m['draw_precision'])} | {pct(m['draw_recall'])} | "
                             f"{pct(m['predicted_draw_share'])} | {pct(roi)} |")
    honest = ["| Модель | Сезон | правило (выбрано на внутр. сезоне) | Macro F1 | Draw recall | Draw prec | Доля D пред. | Accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    for model, d in r["decision_rules"].items():
        for f, h in d["honest_per_fold"].items():
            m = h["metrics"]
            honest.append(f"| {model} | {f} | {h['rule']} | {fmt(m['macro_f1'], 3)} | {pct(m['draw_recall'])} | "
                          f"{pct(m['draw_precision'])} | {pct(m['predicted_draw_share'])} | {pct(m['accuracy'])} |")
    return {"probability": "\n".join(lines), "bookmaker": "\n".join(bm), "classes": "\n".join(cls),
            "rules": "\n".join(rules), "rules_honest": "\n".join(honest)}


def goals_tables(r):
    out = []
    for task in ("home_goals", "away_goals"):
        t = ["| Вариант | Сезон | MAE | RMSE | Девиация | R² | ср. прогноз | ср. факт | Δдевиации [95 % ДИ] |",
             "|---|---|---|---|---|---|---|---|---|"]
        for n in r["folds"]["2023"][task]["variants"]:
            for f in F:
                v = r["folds"][f][task]["variants"][n]
                d = v["delta_deviance_vs_xgb"]
                t.append(f"| {n} | {f} | {fmt(v['mae'])} | {fmt(v['rmse'])} | {fmt(v['poisson_deviance'])} | "
                         f"{fmt(v['r2'])} | {fmt(v['mean_predicted'], 3)} | {fmt(v['mean_actual'], 3)} | "
                         f"{d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}] |")
        out.append(f"**{task}**\n\n" + "\n".join(t))
    return "\n\n".join(out)


def corners_tables(r):
    t = ["| Вариант | Сезон | RMSE | MAE | R² | Девиация | LL 8.5 | LL 9.5 | LL 10.5 | Brier 9.5 | r (NB) | Δдевиации [95 % ДИ] |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in r["folds"]["2023"]["variants"]:
        for f in F:
            v = r["folds"][f]["variants"][n]
            L = v["lines"]
            t.append(f"| {n} | {f} | {fmt(v['rmse'])} | {fmt(v['mae'])} | {fmt(v['r2'])} | {fmt(v['poisson_deviance'])} | "
                     f"{fmt(L['8.5']['log_loss'])} | {fmt(L['9.5']['log_loss'])} | {fmt(L['10.5']['log_loss'])} | "
                     f"{fmt(L['9.5']['brier'])} | {fmt(v['nb_size_inner'], 1)} | "
                     f"{v['delta_deviance_vs_production']['mean_diff']:+.4f} [{v['delta_deviance_vs_production']['ci95'][0]:+.4f}; "
                     f"{v['delta_deviance_vs_production']['ci95'][1]:+.4f}] |")
    base = ["| Сезон | Линия | LL базовой частоты | LL частоты лиги | LL прод. | Brier базовой | Brier прод. |",
            "|---|---|---|---|---|---|---|"]
    for f in F:
        for l in ("8.5", "9.5", "10.5"):
            v = r["folds"][f]["variants"]["xgb_production"]["lines"][l]
            base.append(f"| {f} | {l} | {fmt(v['baseline_log_loss'])} | {fmt(v['league_rate_log_loss'])} | "
                        f"{fmt(v['log_loss'])} | {fmt(v['baseline_brier'])} | {fmt(v['brier'])} |")
    return "\n".join(t), "\n".join(base)


def markets_tables(r):
    chk = ["| Проверка | " + " | ".join(f"макс. нарушение {f}" for f in F) + " | нарушений (все сезоны) |",
           "|---|" + "---|" * (len(F) + 1)]
    for k in r["folds"]["2023"]["consistency"]["checks"]:
        vals = [r["folds"][f]["consistency"]["checks"][k] for f in F]
        chk.append(f"| {k} | " + " | ".join(f"{v['max_abs_violation']:.1e}" for v in vals)
                   + f" | {sum(v['violations'] for v in vals)} |")
    q = ["| Рынок | LL 2023 | база 2023 | LL 2024 | база 2024 | LL 2025* | база 2025* | лучше базы на обоих |",
         "|---|---|---|---|---|---|---|---|"]
    for mk in r["folds"]["2023"]["quality"]:
        cells = []
        for f in F:
            x = r["folds"][f]["quality"][mk]
            x = x.get("reconciled", x)
            cells += [fmt(x.get("log_loss", x.get("model_log_loss"))), fmt(x.get("baseline_log_loss"))]
        q.append(f"| {mk} | " + " | ".join(cells) + f" | {'да' if r['verdict'][mk]['beats_base_rate_both'] else 'нет'} |")
    gap = ["| Сезон | ср.|ΔП1| | ср.|ΔХ| | ср.|ΔП2| | ср. P(Х) классиф. | ср. P(Х) Пуассон | расхождение argmax | LL 1X2 классиф. | LL 1X2 Пуассон |",
           "|---|---|---|---|---|---|---|---|---|"]
    for f in F:
        g = r["folds"][f]["gap_1x2_score_matrix_vs_classifier"]
        qq = r["folds"][f]["quality"]["1X2"]
        gap.append(f"| {f} | {fmt(g['mean_abs_gap']['H'])} | {fmt(g['mean_abs_gap']['D'])} | {fmt(g['mean_abs_gap']['A'])} | "
                   f"{fmt(g['mean_p_draw_classifier'])} | {fmt(g['mean_p_draw_poisson'])} | {pct(g['argmax_disagreement'])} | "
                   f"{fmt(qq['model_log_loss'])} | {fmt(qq['poisson_matrix_log_loss'])} |")
    return "\n".join(chk), "\n".join(q), "\n".join(gap)


def risk_tables(r):
    t = ["| Вариант | Сезон | Покрытие | С ценой | Hit rate | Ср. кэф | ROI [95 % ДИ] | Ср. EV | Risk на каждый Main |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in r["variants"]:
        for f in F:
            v = r["folds"][f][n]
            t.append(f"| {n}{' **(выбран)**' if n == r['selection']['chosen'] else ''} | {f} | {pct(v['coverage'])} | "
                     f"{pct(v['priced_share'])} | {pct(v['hit_rate'])} | {fmt(v['avg_odds'], 2)} | "
                     f"{pct(v['roi'])} [{pct(v['roi_ci95'][0])}; {pct(v['roi_ci95'][1])}] | {pct(v['avg_ev_priced'])} | "
                     f"{'да' if v['risk_for_every_main'] else 'нет'} |")
    return "\n".join(t)


def run() -> dict:
    d, g, c, m, k = (C.load_json(x) for x in ("draw.json", "goals.json", "corners.json", "markets.json", "risk.json"))
    rows = []
    for n, _ in d["folds"]["2023"]["variants"].items():
        rows.append({"task": "1X2", "variant": n, "complexity": COMPLEXITY.get(n),
                     **{f"log_loss_{f}": d["folds"][f]["variants"][n]["log_loss"] for f in F},
                     **{f"draw_recall_{f}": d["folds"][f]["variants"][n]["draw_recall"] for f in F},
                     "train_seconds_2024": d["folds"]["2024"]["train_seconds"].get(n),
                     "chosen": n == d["probability_selection"]["chosen"]})
    for task in ("home_goals", "away_goals"):
        for n in g["folds"]["2023"][task]["variants"]:
            rows.append({"task": task, "variant": n, "complexity": COMPLEXITY.get(n),
                         **{f"poisson_deviance_{f}": g["folds"][f][task]["variants"][n]["poisson_deviance"] for f in F},
                         **{f"rmse_{f}": g["folds"][f][task]["variants"][n]["rmse"] for f in F},
                         "train_seconds_2024": g["folds"]["2024"][task]["train_seconds"].get(n),
                         "chosen": n == g["decision"][task]["chosen"]})
    for n in c["folds"]["2023"]["variants"]:
        rows.append({"task": "corners", "variant": n, "complexity": COMPLEXITY.get(n),
                     **{f"rmse_{f}": c["folds"][f]["variants"][n]["rmse"] for f in F},
                     **{f"r2_{f}": c["folds"][f]["variants"][n]["r2"] for f in F},
                     **{f"mean_line_log_loss_{f}": c["folds"][f]["variants"][n]["mean_line_log_loss"] for f in F},
                     "train_seconds_2024": c["folds"]["2024"]["train_seconds"].get(n),
                     "chosen": n == c["selection"]["chosen"]})
    for n in k["variants"]:
        rows.append({"task": "risk", "variant": n,
                     **{f"roi_{f}": k["folds"][f][n]["roi"] for f in F},
                     **{f"coverage_{f}": k["folds"][f][n]["coverage"] for f in F},
                     "chosen": n == k["selection"]["chosen"]})
    tabs = {}
    if (C.RESULTS / "timing.json").exists():
        t = C.load_json("timing.json")
        tabs["timing"] = "\n".join(["| Модель (фолд 2024, обучение на < 2024, последовательный замер) | Время, с |",
                                     "|---|---|"] + [f"| {k} | {v:.1f} |" for k, v in t["seconds"].items()])
    tabs |= {"draw": draw_tables(d), "goals": goals_tables(g), "corners": corners_tables(c),
            "markets": markets_tables(m), "risk": risk_tables(k)}
    C.save_json("summary.json", {"rows": rows, "chosen": {
        "probability_model": d["probability_selection"]["chosen"],
        "decision_rule": d["decision_rules"][d["probability_selection"]["chosen"]]["final_rule"],
        "goals": {t: g["decision"][t]["chosen"] for t in ("home_goals", "away_goals")},
        "corners": c["selection"]["chosen"], "risk": k["selection"]["chosen"]}})
    md = []
    for sec, val in tabs.items():
        md.append(f"## {sec}\n")
        if isinstance(val, dict):
            for kk, vv in val.items():
                md.append(f"### {kk}\n\n{vv}\n")
        elif isinstance(val, tuple):
            for vv in val:
                md.append(vv + "\n")
        else:
            md.append(val + "\n")
    (C.RESULTS / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")
    figure(d, g, c, k)
    return {"rows": rows}


def best_alternative(values: dict, base: str, lower=True) -> str:
    """Alternative (not production, not naive) with the best pooled 2023+2024 value."""
    cand = {n: v for n, v in values.items() if n != base and not n.startswith("naive")}
    return (min if lower else max)(cand, key=cand.get)


def figure(d, g, c, k):
    plt = C.plot_style()
    sel = [str(s) for s in C.SELECT]
    dv = {n: np.mean([d["folds"][f]["variants"][n]["log_loss"] for f in sel]) for n in d["folds"]["2023"]["variants"]}
    gv = {n: np.mean([g["folds"][f]["away_goals"]["variants"][n]["poisson_deviance"] for f in sel])
          for n in g["folds"]["2023"]["away_goals"]["variants"]}
    cv = {n: np.mean([c["folds"][f]["variants"][n]["mean_line_log_loss"] for f in sel]) for n in c["folds"]["2023"]["variants"]}
    kv = {n: np.mean([k["folds"][f][n]["roi"] for f in sel]) for n in k["variants"]}
    panels = [
        ("1X2 log loss", lambda n, f: d["folds"][f]["variants"][n]["log_loss"], "a_baseline",
         best_alternative(dv, "a_baseline"), d["probability_selection"]["chosen"]),
        ("Голы гостей: девиация", lambda n, f: g["folds"][f]["away_goals"]["variants"][n]["poisson_deviance"],
         "xgb_production", best_alternative(gv, "xgb_production"), g["decision"]["away_goals"]["chosen"]),
        ("Угловые: ср. log loss линий", lambda n, f: c["folds"][f]["variants"][n]["mean_line_log_loss"],
         "xgb_production", best_alternative(cv, "xgb_production"), c["selection"]["chosen"]),
        ("Risk: ROI, %", lambda n, f: k["folds"][f][n]["roi"] * 100, "current (no EV rule)",
         best_alternative(kv, "current (no EV rule)", lower=False), k["selection"]["chosen"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    x = np.arange(len(F))
    for ax, (title, get, base, alt, rec) in zip(axes, panels):
        series = [("продакшн", base, C.PALETTE[7]), (f"лучший эксперимент: {alt}", alt, C.PALETTE[1]),
                  (f"рекомендация: {rec}", rec, C.PALETTE[0])]
        vals = []
        for j, (label, n, col) in enumerate(series):
            v = [get(n, f) for f in F]
            vals += v
            ax.bar(x + (j - 1) * 0.27, v, 0.27, color=col, label=label)
        lo, hi = min(vals), max(vals)
        if lo > 0:
            ax.set_ylim(lo - (hi - lo) * 1.2 - 1e-3, hi + (hi - lo) * 0.6 + 1e-3)
        ax.set_xticks(x, [config.season_label(int(f)) + ("*" if int(f) == C.REPORT else "") for f in F])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6.5, loc="lower left" if lo < 0 else "upper left")
    fig.suptitle("До / после по задачам (выбор только на 2023/24 + 2024/25; * — 2025/26 только для информации)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "summary_before_after.png"), plt.close(fig)


if __name__ == "__main__":
    run()
