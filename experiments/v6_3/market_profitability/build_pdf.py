"""Builds reports/v6_3/Market_Profitability_Research.pdf from results/ (no number is typed by hand).

    python -m experiments.v6_3.market_profitability.build_pdf
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.main_selection import report_util as U
from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, figure, formula, note, table

DIRS = C.study_dirs("market_profitability")
RES, FIG = DIRS["results"], DIRS["figures"]
OUT = C.ROOT / "reports" / "v6_3" / "Market_Profitability_Research.pdf"
GROUPS = ["1X2", "TOTAL", "HANDICAP", "DOUBLE_CHANCE", "BTTS", "CORNERS"]
SHORT = {"DOUBLE_CHANCE": "DC", "HANDICAP": "HCP", "CORNERS": "CORN", "ALL": "все"}
PROFILE_W = [w * K.cm for w in (1.5, 1.9, 1.3, 3.0, 1.5, 1.6, 1.7, 1.2, 1.4, 1.4)]

DECISIONS = [
    ("Принудительный приоритет одной группы (1X2, TOTAL, HANDICAP, DC, BTTS, CORNERS)", "C",
     "Приоритет тоталов (выбор 2025/26) ухудшает T; DC, BTTS и угловые ухудшают обе вселенные; приоритет фор "
     "положителен в 2026/27, но отрицателен в 2025/26."),
    ("Усиление надёжности HANDICAP ×1.15 (выбор 2025/26 и окна)", "B",
     "Положительно на close и T, выбрано и на 2025/26, и окном во всех месяцах; не проходит Холма; меняет "
     "большую долю Main — заморожено для форвард-проверки."),
    ("Усиление других групп ×1.15", "C", "Эффекты малы или отрицательны."),
    ("Исключение одной группы / только 1X2 / только 1X2+TOTAL+HANDICAP", "C",
     "Без тоталов — положительно на close, отрицательно в T; остальные ≈ 0 или хуже."),
    ("Сглаженная историческая доходность (эмпирический Байес)", "C",
     "Оценённое k очень велико (различия рынков в пределах шума) — правило совпадает с продакшном; при k = 200 "
     "небольшой плюс без значимости и без устойчивости по выборкам."),
    ("Нижняя граница исторического ROI (учёт числа наблюдений)", "C", "Знак эффекта различается между close и T."),
    ("Исключение значимо убыточных рынков", "C",
     "Ни одна группа/линия не хуже общего уровня значимо — правило ни разу не срабатывает."),
]


def profile_table(pr: pd.DataFrame, universe: str) -> list[list]:
    out = [["Группа", "Источник", "Ставок", "ROI [95 % ДИ]", "Попадание", "Ср. p (без возврата)",
            "Заявлено − факт, п.п.", "Ср. коэф.", "Ср. EV", "Маржа, п.п."]]
    for g in GROUPS + ["ALL"]:
        for src in ("cand", "main"):
            x = pr.loc[(pr["universe"] == universe) & (pr["group"] == g) & (pr["source"] == src)]
            if x.empty or not x["bets"].iloc[0]:
                continue
            r = x.iloc[0]
            out.append([SHORT.get(g, g), "кандидаты" if src == "cand" else "Main", int(r["bets"]),
                        f"{U.pc(r['roi'])} {U.ci([r['ci_lo'], r['ci_hi']])}", U.pc(r["hit_rate"]), U.pc(r["avg_p"]),
                        U.pp(r["gap"]), U.num(r["avg_odds"]), U.pp(r["mean_ev"]), U.pp(r["margin"])])
    return out


def build() -> None:
    res = C.read_json(RES / "market_profitability.json")
    det = C.read_json(RES / "candidates_detail.json")
    info = C.read_json(RES / "run_info.json")
    pre = C.read_json(RES / "preregistration.json")
    s1 = C.read_json(C.V63 / "main_selection" / "results" / "main_selection.json")
    base_json = C.read_json(C.SHARED_RESULTS / "baseline.json")
    grid = pd.read_csv(RES / "grid.csv")
    cov = pd.read_csv(RES / "coverage.csv")
    pr = pd.DataFrame(res["profile"])
    mix = pd.DataFrame(res["mix"]).drop_duplicates(["rule", "universe", "group"])
    cands = res["candidates"]
    crow = {(r["candidate"], r["universe"]): r for r in cands}
    base = res["baselines"]
    bc, bt9 = base["close"]["summary"], base["t09"]["summary"]
    covb = base_json["coverage"]
    boost = crow[("bt:boost", "close")]
    boost_t = crow[("bt:boost", "t09")]
    n_holm = sum(1 for r in cands if r.get("p_holm") is not None and r["p_holm"] < 0.05)
    n_grid_holm = int((grid["p_holm_grid"] < 0.05).sum())
    snaps = res["history_snapshots"]
    k_close = snaps.get("eb_cand_group_kEB_b3|close", {})
    k_noseed = snaps.get("eb_cand_group_kEB_b3_noseed|close", {})
    g_bt = grid.loc[grid["universe"] == "bt"].set_index("rule")
    cand_close = pr.loc[(pr["universe"] == "close") & (pr["source"] == "cand") & (pr["group"].isin(GROUPS))]
    best_g = cand_close.sort_values("roi", ascending=False).iloc[0]
    worst_g = cand_close.sort_values("roi").iloc[0]
    cand_gap_max = float(cand_close["gap"].abs().max())
    mg = pr.loc[(pr["universe"] == "close") & (pr["source"] == "main") & (pr["bets"] >= 100)
                & (pr["group"].isin(GROUPS))].sort_values("gap", ascending=False).iloc[0]
    main_gap_max, main_gap_grp = float(mg["gap"]), mg["group"]
    main_close = pr.loc[(pr["universe"] == "close") & (pr["source"] == "main")].set_index("group")

    rep = U.V63Report(OUT, "Profitability by Market", "FootPredict v6.3 — исследование 4",
                      {"Сезон проверки": f"2026/27, {covb['first_date']} — {covb['last_played']} "
                                         f"({covb['played']} сыгранных матчей)",
                       "История доходности": "только матчи, рассчитанные раньше прогнозируемого (окно по дням), "
                                             "затравка — бэктест 2025/26",
                       "Вариантов в сетке": str(res["n_variants_grid"]),
                       "Заранее выбранных кандидатов": str(res["n_candidates"])})
    rep.add(H1("1. Название исследования"),
            P("**Доходность прогнозов FootPredict по рынкам и правила выбора рынка для Main.** Какие рынки "
              "система прогнозирует эффективнее и стоит ли менять их приоритет при выборе одного Main на матч."))
    rep.add(H1("2. Постановка проблемы"),
            P(f"Main продакшна в 2026/27 на {U.pc(main_close.loc['TOTAL', 'bets'] / bc['bets'])} состоит из тоталов "
              f"и на {U.pc(main_close.loc['HANDICAP', 'bets'] / bc['bets'])} из азиатских фор (close). В прежних "
              "исследованиях часть рынков имела положительный ROI, часть — отрицательный, но это могло быть "
              "случайностью или следствием самого правила выбора: рынок, который правило выбирает редко и только в "
              "«лучших» ситуациях, выглядит доходнее. Нужно разделить качество прогноза рынка и эффект отбора и "
              "проверить, улучшает ли другой приоритет рынков общий Main."))
    rep.add(H1("3. Гипотезы"),
            bullets([
                "H1. Рынки различаются по доходности прогнозов модели при флэт-ставке на каждый допустимый исход "
                "(независимо от правила выбора).",
                "H2. Приоритет одной группы (1X2, тоталы, азиатские форы, двойной шанс, BTTS, угловые) или её "
                "исключение улучшает ROI Main.",
                "H3. Историческая доходность рынков, сглаженная по числу наблюдений (эмпирический Байес) и известная "
                "до матча, улучшает выбор рынка.",
                "H4. Ранжирование рынков с учётом числа наблюдений (нижняя граница ROI) или исключение значимо "
                "убыточных рынков улучшает Main.",
                "Ожидание: различия рынков малы относительно шума; правило, опирающееся на прошлый ROI рынков, "
                "почти не отличается от продакшна."]))
    rep.add(H1("4. Исходные данные"),
            P("Те же таблицы кандидатов, что и в исследовании 1 (`experiments/v6_3/main_selection/prepare.py`): "
              "2026/27 close (основная вселенная) и T (устойчивость), все исходы матча, модели `prod_20261012`; "
              "2025/26 — бэктест продакшна (`backtest_2526`), цены Football-Data Bet365 только для 1X2, тотала 2.5 "
              "и одной азиатской линии (BTTS, DC и угловые в 2025/26 без цен, поэтому их история начинается с "
              "2026/27)."),
            P(f"Таблица 1. Покрытие (матчей сезона {covb['season_matches']}, сыграно {covb['played']}).", "small"),
            table(U.coverage_rows(cov), widths=U.widths("cov")))
    rep.add(H1("5. Методика"),
            H2("5.1 Доходность рынков"),
            P("Для каждой группы — две доходности: (а) **все кандидаты**: флэт-ставка на каждый исход в диапазоне "
              "Main (0.40 ≤ p ≤ 0.85) с реальной ценой — качество прогноза рынка без влияния правила выбора; "
              "(б) **Main**: ставки Main продакшна в этой группе — с эффектом отбора. Интервалы — бутстреп по "
              "ставкам; «заявлено − факт» на шкале без возврата (p / (1 − P(push)))."),
            H2("5.2 История доходности до матча"),
            P("Для матча дня d используются только ставки, рассчитанные в дни < d той же вселенной, плюс весь "
              "2025/26 как затравка (вариант без затравки тоже проверен). Ячейки: группа, линия рынка или группа × "
              "диапазон коэффициента; источник — все кандидаты или Main продакшна. Сглаживание:"),
            formula("r̃_{j} = (n_{j} r_{j} + k · m) / (n_{j} + k),   k = s² / τ²,   "
                    "τ² = max(0, Var_{w}(r_{j}) − mean(s² / n_{j}))"),
            P("m — общий ROI истории, s² — дисперсия прибыли одной ставки; k оценивается заново каждый день "
              "(эмпирический Байес) или фиксируется (k = 200). Множитель к оценке правила продакшна:"),
            formula("score' = score · exp(β · (r̃_{j} − m)),   β ∈ {3, 10}"),
            P("Нижняя граница: r_{j} − 1.645·s/√n_{j} вместо r̃_{j} (ячейка без истории получает m − 0.3). "
              "Исключение: ячейка с n_{j} ≥ 100 и r_{j} + 1.96·s/√n_{j} < m не участвует в выборе."),
            H2("5.3 Выбор параметров и статистика"),
            P("Как в исследовании 1: член семейства выбирается на 2025/26 (bt) или расширяющимся окном по месяцам "
              "2026/27 (ew); полная сетка — отдельно, с Холмом по всем вариантам. Сравнение — парный бутстреп с "
              f"Baseline B той же вселенной ({info['params']['n_boot']} повторов)."))
    fam_rows = [["Семейство", "Члены"]] + [[f, ", ".join(m)] for f, m in res["families"].items()]
    rep.add(H1("6. Эксперименты"),
            P(f"{res['n_variants_grid']} вариантов в {len(res['families'])} семействах, {res['n_candidates']} заранее выбранных "
              "кандидатов (выбор на 2025/26 и окно по каждому семейству + три априорных правила истории)."),
            P("Таблица 2. Семейства.", "small"), table(fam_rows, widths=[3.0 * K.cm, 14 * K.cm]))
    rep.add(H1("7. Результаты"),
            H2("7.1 Какие рынки система прогнозирует эффективнее"),
            P("Таблица 3. 2026/27 close: доходность по группам рынков.", "small"),
            table(profile_table(pr, "close"), widths=PROFILE_W),
            P("Таблица 4. 2026/27 T.", "small"), table(profile_table(pr, "t09"), widths=PROFILE_W),
            P("Таблица 5. 2025/26 (только рынки с ценой Football-Data).", "small"), table(profile_table(pr, "bt"), widths=PROFILE_W),
            note(f"Все группы убыточны при флэт-ставке на каждого кандидата. Лучше всего — {best_g['group']} "
                 f"({U.pc(best_g['roi'])} на close), хуже всего — {worst_g['group']} ({U.pc(worst_g['roi'])}). "
                 f"У кандидатов |заявлено − факт| не больше {U.pp(cand_gap_max, 1, False)} п.п., а выбранный Main "
                 f"переоценён до {U.pp(main_gap_max)} п.п. (больше всего у {main_gap_grp}): эффект отбора, а не "
                 "качества рынка. Положительный ROI Main в "
                 "редких группах (1X2, DC, BTTS, угловые) — малые выборки отбора."))
    pri = [["Правило", "Ставок close", "ROI close", "ΔROI close [95 % ДИ]", "ROI T", "ΔROI T", "ROI 2025/26",
            "Изменено close"]]
    for fam in ("priority", "boost", "exclude"):
        for _, r in grid.loc[(grid["family"] == fam) & (grid["universe"] == "close")
                             & (grid["rule"] != "production")].iterrows():
            t = grid.loc[(grid["rule"] == r["rule"]) & (grid["universe"] == "t09")].iloc[0]
            pri.append([r["rule"], r["bets"], U.pc(r["roi"]), f"{U.pp(r['roi_diff'])} {U.ci([r['ci_lo'], r['ci_hi']])}",
                        U.pc(t["roi"]), U.pp(t["roi_diff"]), U.pc(g_bt.loc[r["rule"], "roi"]),
                        U.pc(r["changed_share"], 0)])
    rep.add(H2("7.2 Приоритет, усиление и исключение рынков"),
            P("Таблица 6. Простые правила рынков: влияние на общий Main.", "small"),
            table(pri, widths=[4.0 * K.cm, 1.5 * K.cm, 1.5 * K.cm, 3.2 * K.cm, 1.5 * K.cm, 1.4 * K.cm, 1.7 * K.cm,
                               1.7 * K.cm]))
    mx = [["Правило"] + GROUPS + ["ROI всего"]]
    for rule in ["production"] + [f"priority_{g}" for g in GROUPS] + ["boost_HANDICAP", "without_TOTAL"]:
        x = mix.loc[(mix["rule"] == rule) & (mix["universe"] == "close")].set_index("group")
        if x.empty:
            continue
        tot = grid.loc[(grid["rule"] == rule) & (grid["universe"] == "close"), "roi"].iloc[0]
        mx.append([rule] + [f"{U.pc(x.loc[g, 'share'], 0)} / {U.pc(x.loc[g, 'roi'], 0)}" if g in x.index else "—"
                            for g in GROUPS] + [U.pc(tot)])
    rep.add(P("Таблица 7. Доля ставок Main / ROI внутри группы при разных правилах (close).", "small"), table(mx))
    rep.add(H2("7.3 Историческая доходность рынков (без будущей информации)"),
            P(f"Оценённое k эмпирического Байеса (close, группы, все кандидаты, с затравкой) — в начале сезона "
              f"{U.num(k_close.get('k_first'), 0) if k_close.get('k_first') else '∞'} (различий групп не видно), "
              f"в конце {U.num(k_close.get('k_last'), 0) if k_close.get('k_last') else '∞'}; без затравки — "
              f"{U.num(k_noseed.get('k_last'), 0) if k_noseed.get('k_last') else '∞'}. Чем больше k, тем сильнее "
              "оценка рынка стянута к общему уровню: различия рынков почти целиком объясняются шумом."),
            P("Таблица 8. Правила по истории доходности (полная сетка).", "small"),
            table(U.grid_table(grid.loc[grid["family"].str.startswith("history")]), widths=U.widths("grid")))
    rep.add(H2("7.4 Заранее выбранные кандидаты"),
            P("Таблица 9. close.", "small"), table(U.candidates_table(cands, "close"), widths=U.widths("cand")),
            P("Таблица 10. T.", "small"), table(U.candidates_table(cands, "t09"), widths=U.widths("cand")),
            P("Таблица 11. Выбор окна по месяцам (close).", "small"), table(U.ew_table(res["ew_choices"], "close")),
            note(f"После Холма по {res['n_candidates']} кандидатам значимых: {n_holm}; по сетке: {n_grid_holm}. "
                 f"Самый последовательный кандидат — усиление азиатских фор (выбран и на 2025/26, и окном): close "
                 f"{U.pp(boost['roi_diff'])} п.п. {U.ci(boost['diff_ci'])}, T {U.pp(boost_t['roi_diff'])} п.п. "
                 f"{U.ci(boost_t['diff_ci'])}, меняет {U.pc(boost['changed_share'], 0)} Main."))
    sens = [n for n in ("bt:boost", "bt:history_eb", "eb_cand_group_kEB_b3", "lcb_cand_group_b3") if n in det]
    rep.add(P("Таблица 12. Чувствительность (close).", "small"),
            table(U.sensitivity_table(det, base, sens, "close"), widths=U.widths("sens")),
            P("Таблица 13. ΔROI по месяцам (close).", "small"), table(U.months_table(det, sens, "close")),
            P("Таблица 14. ΔROI по месяцам (T).", "small"), table(U.months_table(det, sens, "t09")),
            P("Таблица 15. Усиление фор по лигам (close).", "small"),
            table(U.league_table(det, base, "bt:boost", "close")))
    rep.add(H1("8. Графики"),
            *figure(FIG / "profile.png", "Рис. 1. ROI по группам рынков: флэт на всех кандидатов и Main продакшна."),
            *figure(FIG / "gaps.png", "Рис. 2. Переоценка вероятности (заявлено − факт): кандидаты откалиброваны, "
                                      "выбранный Main — нет."),
            *figure(FIG / "history_path.png", "Рис. 3. Сглаженный ROI групп на каждую дату 2026/27 (close), "
                                              "только по прошлым матчам."),
            *figure(FIG / "rules_forest.png", "Рис. 4. Приоритет, усиление и исключение рынков: разница с B."),
            *figure(FIG / "priority_mix.png", "Рис. 5. Рынки Main при принудительном приоритете группы."),
            *figure(FIG / "candidates_forest.png", "Рис. 6. Заранее выбранные кандидаты против Baseline B."))
    rel1 = {(r["candidate"], r["universe"]): r for r in s1["candidates"]}
    rep.add(H1("9. Статистический анализ"),
            bullets([
                f"Проверено {res['n_variants_grid']} вариантов и {res['n_candidates']} кандидатов; после Холма "
                f"значимых улучшений нет ({n_holm} / {n_grid_holm}).",
                "Различия доходности групп при флэт-ставке на кандидатов (табл. 3–5) сопоставимы с шириной "
                "интервалов; эмпирический Байес это подтверждает (большое k).",
                f"Усиление фор: {boost['months_better']}/{boost['months_n']} месяцев лучше B на close, "
                f"{boost_t['months_better']}/{boost_t['months_n']} в T. То же направление в исследовании 1 "
                f"(надёжность TOTAL 0.95: close {U.pp(rel1[('bt:reliability', 'close')]['roi_diff'])} п.п., T "
                f"{U.pp(rel1[('bt:reliability', 't09')]['roi_diff'])} п.п.) — это один и тот же эффект, а не два "
                "независимых подтверждения.",
                "Высокий ROI Main в редких группах (1X2, DC, BTTS, угловые) держится на десятках ставок и "
                "исчезает, как только правило заставляет выбирать эти рынки чаще (табл. 6–7): отбор, а не рынок.",
                "2026/27 уже изучался — это хронологическая проверка, а не независимое доказательство."]))
    rep.add(H1("10. Выводы"),
            bullets([
                f"Все рынки убыточны при флэт-ставке; наименее убыточны азиатские форы ({best_g['group']}), наиболее — "
                f"{worst_g['group']}. Модель не обыгрывает маржу ни на одном рынке.",
                "Принудительный приоритет или исключение рынков не улучшает Main устойчиво.",
                "Историческая доходность рынков, известная до матча и сглаженная по числу наблюдений, почти не "
                "отличается между рынками — правило на её основе совпадает с продакшном или даёт неустойчивый "
                "эффект.",
                "Единственное последовательное направление — чуть больший вес азиатских фор (меньший — тоталов); "
                "эффект не доказан и проверяется вперёд."]))
    dec = [["Вариант", "Решение", "Обоснование"]] + [list(x) for x in DECISIONS]
    pre_rows = [["ID", "Изменение", "Мин. форвард-выборка", "Заморожено"]] + [
        [p["id"], p["production_change"], p["minimum_forward_matches"], p["frozen_at"]] for p in pre]
    rep.add(H1("11. Рекомендации (A / B / C)"),
            table(dec, widths=[5.0 * K.cm, 2.0 * K.cm, 10.0 * K.cm]),
            P("**Продакшн не меняется.** Кандидат B заморожен (`results/preregistration.json`); проверять его "
              "вместе с кандидатом исследования 1 «TOTAL 0.95» нельзя как два независимых — выбрать одно."),
            table(pre_rows, widths=[4.2 * K.cm, 7.6 * K.cm, 2.2 * K.cm, 3 * K.cm]))
    rep.add(H1("12. Ограничения"),
            bullets([
                "2025/26 — только 1X2, тотал 2.5 и одна азиатская линия с ценой (Bet365, открытие), поэтому история "
                "BTTS, DC, угловых и других линий начинается с нуля в августе 2026; цены 2025/26 и 2026/27 close "
                "разного происхождения.",
                f"2026/27 ({covb['first_date']} — {covb['last_played']}) уже изучался; редкие рынки в Main имеют "
                "десятки ставок.",
                "Флэт-ROI кандидатов коррелирован внутри матча (несколько исходов одного матча); интервалы в "
                "таблицах 3–5 по ставкам, поэтому несколько уже реальных.",
                f"Вселенная T: {U.pc(1 - bt9['priced_share'], 0)} Main продакшна без цены к T; правила, выбирающие исходы с ценой, ставят на "
                "других матчах."]))
    rep.add(H1("Как воспроизвести"),
            *K.reproduce(["python -m experiments.v6_3.main_selection.prepare        # общие таблицы кандидатов",
                          "python -m experiments.v6_3.market_profitability.run",
                          "python -m experiments.v6_3.market_profitability.figures",
                          "python -m experiments.v6_3.market_profitability.preregister",
                          "python -m experiments.v6_3.market_profitability.build_pdf"]),
            P(f"Результаты: `experiments/v6_3/market_profitability/results/`. Git SHA {info['git_sha'][:10]}.",
              "small"))
    rep.build()
    print(f"{OUT} — {rep.check()} pages")
    findings(res, cands, pr, pre, base)


def findings(res, cands, pr, pre, base) -> None:
    bc = base["close"]["summary"]
    lines = ["# FootPredict v6.3 — Profitability by Market: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_3/Market_Profitability_Research.pdf`.", "",
             "## Доходность рынков (2026/27 close; флэт на всех кандидатов с ценой / Main продакшна)", "",
             "| Группа | ROI кандидатов [95 % ДИ] | ставок | ROI Main | ставок Main | заявлено − факт Main, п.п. |",
             "|---|---|---|---|---|---|"]
    for g in GROUPS:
        c = pr.loc[(pr["universe"] == "close") & (pr["group"] == g) & (pr["source"] == "cand")].iloc[0]
        m = pr.loc[(pr["universe"] == "close") & (pr["group"] == g) & (pr["source"] == "main")].iloc[0]
        lines.append(f"| {g} | {U.pc(c['roi'])} {U.ci([c['ci_lo'], c['ci_hi']])} | {int(c['bets'])} | "
                     f"{U.pc(m['roi'])} | {int(m['bets'])} | {U.pp(m['gap'])} |")
    lines += ["", f"Baseline B close: ROI {U.pc(bc['roi'])}, {bc['bets']} ставок. {res['n_variants_grid']} вариантов, "
              f"{res['n_candidates']} кандидатов.", "", "## Кандидаты (ΔROI к Baseline B, п.п.)", "",
              "| Кандидат | close ΔROI [95 % ДИ] | p Холма | T ΔROI | Изменено |", "|---|---|---|---|---|"]
    rows = {(r["candidate"], r["universe"]): r for r in cands}
    for name in dict.fromkeys(r["candidate"] for r in cands):
        c, t = rows[(name, "close")], rows[(name, "t09")]
        lines.append(f"| {name} | {U.pp(c['roi_diff'])} {U.ci(c['diff_ci'])} | {U.pv(c['p_holm'])} | "
                     f"{U.pp(t['roi_diff'])} | {U.pc(c['changed_share'], 0)} |")
    lines += ["", "## Решения", ""] + [f"- **{b}** — {a}: {c}" for a, b, c in DECISIONS]
    lines += ["", "Продакшн не меняется. Предрегистрация (`results/preregistration.json`):", ""]
    lines += [f"- `{p['id']}` — {p['production_change']}; минимум {p['minimum_forward_matches']} матчей после "
              f"{p['frozen_at']}." for p in pre]
    lines += ["", "## Воспроизведение", "", "```", "python -m experiments.v6_3.main_selection.prepare",
              "python -m experiments.v6_3.market_profitability.run",
              "python -m experiments.v6_3.market_profitability.figures",
              "python -m experiments.v6_3.market_profitability.preregister",
              "python -m experiments.v6_3.market_profitability.build_pdf", "```", ""]
    (RES.parent / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
