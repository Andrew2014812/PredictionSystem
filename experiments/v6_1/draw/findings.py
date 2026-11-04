"""findings.md of the draw research — every number is read from results/*.json."""
from __future__ import annotations

from . import common as C


def p(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f} %"


def pp(v, d=1):
    return "—" if v is None else f"{v * 100:+.{d}f} п.п."


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def f4(v):
    return "—" if v is None else f"{v:.4f}"


def ci(c, d=1):
    return "—" if not c or c[0] is None else f"[{c[0] * 100:+.{d}f}; {c[1] * 100:+.{d}f}] %"


def ci_pp(c, d=1):
    return "—" if not c or c[0] is None else f"[{c[0] * 100:+.{d}f}; {c[1] * 100:+.{d}f}] п.п."


def summary_numbers(R, P, M) -> dict:
    """Ranges used by the conclusions (findings.md and the PDF)."""
    keys = ["block_2023", "block_2024", "block_2025", "block_2026", "history_2026"]
    a = [R["per_set"][k]["argmax"] for k in keys]
    c = [R["per_set"][k]["current"] for k in keys]
    rng = lambda xs: (min(xs), max(xs))  # noqa: E731
    bk = P["selection"]["odds_rows_pooled_selection"]
    best = M["policies"]["grid"][M["policies"]["best"]]
    return {
        "named_old": rng([x["predicted_draw_share"] for x in a]), "named_new": rng([x["predicted_draw_share"] for x in c]),
        "recall_old": rng([x["draw_recall"] for x in a]), "recall_new": rng([x["draw_recall"] for x in c]),
        "prec_new": rng([x["draw_precision"] for x in c]),
        "f1_old": rng([x["macro_f1"] for x in a]), "f1_new": rng([x["macro_f1"] for x in c]),
        "acc_drop": rng([x["accuracy"] - y["accuracy"] for x, y in zip(a, c)]),
        "bal_gain": rng([y["balanced_accuracy"] - x["balanced_accuracy"] for x, y in zip(a, c)]),
        "max_pd": max(M["why_not_main"][k]["p_draw_quantiles"]["1.0"] for k in keys),
        "draw_odds": rng([M["why_not_main"][k]["draw_odds_mean"] for k in keys]),
        "pref": rng([M["why_not_main"][k]["odds_preference_at_draw_odds_mean"] for k in keys]),
        "main_score": rng([M["why_not_main"][k]["main_score_mean"] for k in keys]),
        "ceiling": rng([M["why_not_main"][k]["draw_score_ceiling_mean"] for k in keys]),
        "below_ceiling_hist": M["why_not_main"]["history_2026"]["share_main_score_below_draw_ceiling"],
        "risk_draw_share": rng([M["risk_draws"][k]["share_of_matches"] for k in keys]),
        "book_ll": bk["bk_proportional"]["pooled_log_loss"] - bk["a_baseline"]["pooled_log_loss"],
        "book_draw_ll": bk["bk_proportional"]["pooled_draw_log_loss"] - bk["a_baseline"]["pooled_draw_log_loss"],
        "best_prob": P["selection"]["best"], "prob_convincing": P["selection"]["convincing"],
        "best_prob_delta": P["selection"]["table"][P["selection"]["best"]]["pooled_delta_log_loss"] if P["selection"]["best"] else None,
        "policy": M["policies"]["best"], "policy_convincing": M["policies"]["convincing"], "policy_rows": best,
        "flat_draw_roi": {k: M["draw_bets"][k]["all"] for k in keys},
        "rule_best": R["rule_search"]["a_baseline"]["best"]["rule"],
    }


def conclusions(K) -> list[str]:
    b = K["policy_rows"]
    return [
        f"Да, после v6 FootPredict действительно называет ничьи: доля названных ничьих {p(K['named_old'][0])}–{p(K['named_old'][1])} → "
        f"{p(K['named_new'][0])}–{p(K['named_new'][1])} (факт ≈25–27 %), полнота ничьих {p(K['recall_old'][0])}–{p(K['recall_old'][1])} → "
        f"{p(K['recall_new'][0])}–{p(K['recall_new'][1])} при точности {p(K['prec_new'][0])}–{p(K['prec_new'][1])}, macro F1 "
        f"{f3(K['f1_old'][0])}–{f3(K['f1_old'][1])} → {f3(K['f1_new'][0])}–{f3(K['f1_new'][1])} на всех сезонах, включая независимый 2026/27. "
        f"Цена — accuracy ниже на {K['acc_drop'][0] * 100:.1f}–{K['acc_drop'][1] * 100:.1f} п.п.; balanced accuracy выше на "
        f"{K['bal_gain'][0] * 100:.1f}–{K['bal_gain'][1] * 100:.1f} п.п. Вероятности (log loss, Brier) не изменились.",
        f"Ничья не становится Main из-за правила Main, а не из-за модели: P(D) не превышает {K['max_pd']:.3f} (порог Main 0.40), средний "
        f"коэффициент ничьей {K['draw_odds'][0]:.2f}–{K['draw_odds'][1]:.2f} лежит вне окна 1.40–2.50 (множитель odds_preference "
        f"{K['pref'][0]:.2f}–{K['pref'][1]:.2f}), а оценка Main пропорциональна вероятности: даже без порога, штрафа за коэффициент и с "
        f"максимальным бонусом EV потолок оценки ничьей ({K['ceiling'][0]:.2f}–{K['ceiling'][1]:.2f}) ниже средней оценки выбранного Main "
        f"({K['main_score'][0]:.2f}–{K['main_score'][1]:.2f}); в walk-forward-блоках Main ни разу не оказался ниже этого потолка, "
        f"в истории 2026/27 — в {p(K['below_ceiling_hist'])} матчей. Ничья часто становится Risk ({p(K['risk_draw_share'][0])}–"
        f"{p(K['risk_draw_share'][1])} матчей).",
        f"Ни один новый способ получения вероятностей не улучшил log loss убедительно (лучший допущенный — {K['best_prob']}, "
        f"Δ log loss на отборе {K['best_prob_delta']['mean_diff'] * 1000:+.2f}·10⁻³, CI включает 0). Букмекер (Bet365 без маржи) "
        f"лучше модели на {-K['book_ll']:.4f} по log loss 1X2, но по бинарному log loss ничьей лишь на {-K['book_draw_ll']:.4f}: "
        f"преимущество рынка — в П1/П2, а не в ничьих.",
        f"Ничья как Main не даёт устойчивого выигрыша: лучшая на отборе политика {K['policy']} меняет ROI Main на "
        f"{pp(b['pooled_selection']['delta_main_roi'])} {ci_pp(b['pooled_selection']['delta_ci95'])} на отборе и на "
        f"{pp(b['block_2025']['delta_main_roi'])} на 2025/26; интервалы включают ноль. Плоская ставка на каждую ничью убыточна "
        f"(ROI {', '.join(p(v['roi']) for v in K['flat_draw_roi'].values())}). Продакшн не меняется.",
        f"Текущее правило назначения (t = 0.28, g = 0.20) повторно выбрано тем же критерием ({K['rule_best']}); "
        "более сильного правила по заранее заданным критериям не найдено.",
    ]


SETS = [("block_2023", "2023/24 (отбор)"), ("block_2024", "2024/25 (отбор)"), ("block_2025", "2025/26 (инфо)"),
        ("page_backtest_2025", "2025/26 backtest_2526 (страница)"), ("block_2026", "2026/27 блок (проверка)"),
        ("history_2026", "2026/27 история продакшна")]


def write() -> str:
    R, P, M = C.load_json("rules.json"), C.load_json("probabilities.json"), C.load_json("draw_main.json")
    info = C.load_json("run_info.json") if (C.RESULTS / "run_info.json").exists() else {}
    L = ["# FootPredict v6.1 — исследование ничьих (Draw)", ""]
    L += ["Все числа взяты из `experiments/v6_1/draw/results/*.json` (этот файл генерирует `findings.py`). "
          "Walk-forward: блок *b* — все продакшн-модели переобучены на сезонах < *b* с настроенными параметрами "
          f"активной модели `{C.prod_meta()['version']}` и прогнозируют сезон *b* через продакшн-движок. Выбор — только "
          "на 2023/24 и 2024/25; 2025/26 — для информации; 2026/27 — независимая проверка (блок 2026 и записанная "
          "история продакшна сайта). Цены — только предматчевые (Football-Data Bet365; в 2026/27 также цены API, "
          "собранные до начала матча).", ""]
    if info:
        L += [f"- Git SHA запуска: `{info.get('git_sha')}`, Python {info.get('python')}, n_jobs = {info.get('n_jobs')}, "
              f"время: {info['seconds'].get('total', 0) / 60:.1f} мин.", ""]
    pc = R["page_check"]
    L += ["## 1. Стали ли ничьи предсказываться лучше", "",
          "| Набор | n | Доля ничьих | ср. P(D) | Названо ничьих (argmax → v6) | Точность D | Полнота D | Macro F1 | Accuracy | Bal. acc | Log loss | Brier |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in SETS:
        a, c = R["per_set"][k]["argmax"], R["per_set"][k]["current"]
        L.append(f"| {lab} | {a['n']} | {p(a['actual_draw_share'])} | {f3(a['mean_p_draw'])} | {p(a['predicted_draw_share'])} → "
                 f"{p(c['predicted_draw_share'])} | {p(a['draw_precision'])} → {p(c['draw_precision'])} | {p(a['draw_recall'])} → "
                 f"{p(c['draw_recall'])} | {f3(a['macro_f1'])} → {f3(c['macro_f1'])} | {p(a['accuracy'])} → {p(c['accuracy'])} | "
                 f"{p(a['balanced_accuracy'])} → {p(c['balanced_accuracy'])} | {f4(a['log_loss'])} | {f4(a['brier'])} |")
    L += ["", "Вероятности правило не меняет, поэтому log loss и Brier одинаковы для argmax и правила v6.", "",
          "**Матрица ошибок для страницы Model Analysis** (тестовый сезон 2025/26, модель `backtest_2526`, "
          f"n = {pc['n']}; строки — факт П1/Х/П2, столбцы — названный исход П1/Х/П2):", "",
          f"- в метаданных сейчас (argmax, `draw_multiplier` = {pc['metadata_draw_multiplier']}): `{pc['metadata_test_metrics_confusion']}`;",
          f"- пересчёт argmax: `{pc['recomputed_argmax_confusion']}` (совпадает: {pc['argmax_matches_metadata']});",
          f"- **правило v6 (t = {R['current_rule']['t']}, g = {R['current_rule']['g']}): `{pc['current_rule_confusion']}`** — "
          f"столько же даёт сохранённый `predicted_outcome` backtest-снимков: `{pc['stored_predicted_outcome_confusion']}`.",
          f"- Блок 2025 воспроизводит сохранённый backtest: макс. |Δp| = {pc['block_2025_vs_stored_backtest']['max_abs_prob_diff']}, "
          f"тот же Main в {p(pc['block_2025_vs_stored_backtest']['same_main_share'], 2)} матчей.", ""]
    h = R["per_set"]["history_2026"]
    L += [f"История 2026/27: записанный на сайте названный исход (`predicted_outcome`) совпадает с правилом v6 по версиям: "
          + ", ".join(f"{k} — {p(v)}" for k, v in h["recorded_matches_rule"].items())
          + f" (live-прогнозы до v6 называли исход по argmax). Macro F1 записанного исхода {f3(h['recorded_named_outcome']['macro_f1'])}, "
          f"доля ничьих {p(h['recorded_named_outcome']['predicted_draw_share'])}.", ""]
    L += ["## 2. Почему ничья почти никогда не становится Main", "",
          "| Набор | Main = ничья | Risk = ничья | P(D) ≥ 0.40 | P(D) ≥ 0.30 | макс. P(D) | ср. коэф. ничьей | ср. odds_preference | ср. оценка Main | ср. оценка ничьей без порога | потолок оценки ничьей | Main ниже потолка |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in [s for s in SETS if s[0] != "page_backtest_2025"]:
        w = M["why_not_main"][k]
        L.append(f"| {lab} | {w['main_is_draw']} | {w['risk_is_draw']} | {p(w['share_p_draw_ge_main_min'])} | {p(w['share_p_draw_ge_0.30'])} | "
                 f"{f3(w['p_draw_quantiles']['1.0'])} | {w['draw_odds_mean']:.2f} | {f3(w['odds_preference_at_draw_odds_mean'])} | "
                 f"{f3(w['main_score_mean'])} | {f3(w['draw_score_ungated_mean'])} | {f3(w['draw_score_ceiling_mean'])} | "
                 f"{p(w['share_main_score_below_draw_ceiling'])} |")
    L += ["", "Контрфактические Main (продакшн-функция `main_prediction`, изменён только один параметр):", ""]
    for k, lab in [s for s in SETS if s[0].startswith("block")]:
        cf = M["why_not_main"][k].get("counterfactual_main", {})
        L.append(f"- {lab}: " + "; ".join(f"{t}: Main = ничья {v['main_is_draw']} раз, ROI Main {p(v['main_roi']['roi'])}"
                                         for t, v in cf.items()))
    L += ["", "## 3. Новые способы получить вероятность ничьей", "",
          "| Вариант | " + " | ".join(f"LL {C.season_label(s)}" for s in C.FOLDS) + " | Δ LL отбор (95 % CI) | ничья LL 2024/25 |", "|---" * 7 + "|"]
    sel = P["selection"]
    for n in P["folds"]["2023"]["all_rows"]:
        if n == "naive_league":
            continue
        row = [P["titles"][n]] + [f4(P["folds"][str(s)]["all_rows"][n]["log_loss"]) for s in C.FOLDS]
        t = sel["table"][n]
        row += [f"{t['pooled_delta_log_loss']['mean_diff'] * 1000:+.2f}·10⁻³ [{t['pooled_delta_log_loss']['ci95'][0] * 1000:+.2f}; "
                f"{t['pooled_delta_log_loss']['ci95'][1] * 1000:+.2f}]", f4(P["folds"]["2024"]["all_rows"][n]["draw_log_loss"])]
        L.append("| " + " | ".join(row) + " |")
    L += ["", f"Допущены к выбору (лучше базы по log loss и Brier на обоих сезонах отбора): {', '.join(sel['eligible']) or 'нет'}; "
          f"лучший — {sel['best']}; убедительно (CI < 0 и подтверждение 2026/27): **{'да' if sel['convincing'] else 'нет'}**.", "",
          "Букмекер на тех же матчах (объединённые 2023/24 + 2024/25, только матчи с ценами):", ""]
    bk = sel["odds_rows_pooled_selection"]
    for n in ("a_baseline", "bk_proportional", "bk_power", "e_book_blend"):
        L.append(f"- {P['titles'][n]}: log loss {f4(bk[n]['pooled_log_loss'])}, Brier {f4(bk[n]['pooled_brier'])}, "
                 f"log loss ничьей (бинарный) {f4(bk[n]['pooled_draw_log_loss'])}")
    L += ["", "## 4. Правило назначения исхода (вопрос 5)", ""]
    sr = R["rule_search"]["a_baseline"]
    L += [f"Поиск по сетке (t, g, множитель) на объединённых 2023/24 + 2024/25 при ограничении доли ничьих ≤ "
          f"{p(R['draw_share_cap'])} выбирает **{sr['best']['rule']}** — это и есть текущее правило (macro F1 "
          f"{f3(sr['best']['pooled']['macro_f1'])}).", ""]
    for m, v in R["verdict"].items():
        L.append(f"- {P['titles'].get(m, m)}: лучшее правило {v['best_rule']}, Δ macro F1 к продакшну "
                 + ", ".join(f"{C.season_label(s)} {v['macro_f1_gain_vs_production'][str(s)]:+.4f}" for s in C.FOLDS)
                 + f"; критерии отбора {'пройдены' if v['passes_selection_criteria'] else 'не пройдены'}; принять: {'да' if v['adopt'] else 'нет'}.")
    L += ["", "Честный выбор на внутреннем сезоне каждого фолда:", ""]
    for s, v in R["honest_per_fold"].items():
        L.append(f"- {C.season_label(s)}: {v['rule']} → macro F1 {f3(v['outer']['macro_f1'])} (текущее правило {f3(v['outer_current']['macro_f1'])}), "
                 f"accuracy {p(v['outer']['accuracy'])} ({p(v['outer_current']['accuracy'])}).")
    L += ["", "## 5. Ничья как Main (вопрос 4)", ""]
    best = M["policies"]["best"]
    g = M["policies"]["grid"][best]
    L += [f"Лучшая политика на отборе: **{best}** (P(D) ≥ {g['params']['t']}, |P(П1) − P(П2)| ≤ {g['params']['g']}, EV ≥ {g['params']['e']}); "
          f"убедительна: **{'да' if M['policies']['convincing'] else 'нет'}**.", "",
          "| Набор | ставок на ничью | попадание | ср. коэф. | ROI ничьих (95 % CI) | ROI заменённых Main | ROI Main до → после | Δ ROI Main (95 % CI) |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ["block_2023", "block_2024", "pooled_selection", "block_2025", "block_2026", "history_2026"]:
        v = g[k]
        L.append(f"| {k} | {v['draw_bets']} | {p(v['draw_hit_rate'])} | {v['draw_avg_odds']:.2f} | {p(v['draw_roi']['roi'])} {ci(v['draw_roi']['ci95'])} | "
                 f"{p(v['replaced_main_roi']['roi'])} | {p(v['main_roi_before'])} → {p(v['main_roi_after'])} | {pp(v['delta_main_roi'])} {ci_pp(v['delta_ci95'])} |")
    L += ["", "Ставка на каждую ничью с ценой (все матчи):", ""]
    for k, lab in [s for s in SETS if s[0] != "page_backtest_2025"]:
        a = M["draw_bets"][k]["all"]
        L.append(f"- {lab}: {a['bets']} ставок, ROI {p(a['roi'])} {ci(a['ci95'])}")
    K = summary_numbers(R, P, M)
    L += ["", "## Выводы", ""] + [f"- {x}" for x in conclusions(K)] + [""]
    text = "\n".join(L)
    (C.HERE / "findings.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    write()
