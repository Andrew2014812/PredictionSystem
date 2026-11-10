"""Builds reports/v6_2/Research_Summary.pdf: the conclusions of the thirteen v6.2 studies side by side.

    python -m experiments.v6_2.summary.build_pdf

The numbers are collected from the studies' result files by ``collect.py`` (written to results/summary.json) at
build time; this file only formats them and adds the text. Every table names the file its numbers come from.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, figure, note, table

from . import collect as CL

OUT = CL.ROOT / "reports" / "v6_2" / "Research_Summary.pdf"
FIG = CL.HERE / "figures"
# categorical colours of the two charts (checked for colour-blind separation and contrast on white)
SERIES = ["#0a8f6e", "#b97f0c", "#3d6fb6"]
MUTED, GRID = K.MUTED, K.GRID


# ---------------------------------------------------------------------------------------------- formatting
def _s(x, d):
    return f"{x:+.{d}f}"


def ll(m, d=1):
    """A log-loss difference in units of 10⁻³ with its interval: '+14.9 [+11.0; +18.5]'."""
    if m is None or m.get("v") is None:
        return "—"
    s = _s(m["v"] * 1e3, d)
    return s + (f" [{_s(m['ci'][0] * 1e3, d)}; {_s(m['ci'][1] * 1e3, d)}]" if m.get("ci") else "")


def pp(m, d=1, key="v", ci="ci"):
    """A fraction shown in percentage points with its interval: '+4.9 [+2.9; +6.9]'."""
    if m is None or m.get(key) is None:
        return "—"
    s = _s(m[key] * 100, d)
    c = m.get(ci)
    return s + (f" [{_s(c[0] * 100, d)}; {_s(c[1] * 100, d)}]" if c else "")


def pc(m, d=1):
    """ROI in percent with its interval: '-4.1 % [-7.1; -1.2]'."""
    if m is None or m.get("v") is None:
        return "—"
    c = m.get("ci")
    return f"{m['v'] * 100:+.{d}f} %" + (f" [{c[0] * 100:+.{d}f}; {c[1] * 100:+.{d}f}]" if c else "")


def n(v):
    v = v["v"] if isinstance(v, dict) else v
    return f"{int(round(v)):,}".replace(",", " ")


def x(v, d=3):
    v = v["v"] if isinstance(v, dict) else v
    return f"{v:.{d}f}"


def pv(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def dru(iso):
    return "—" if not iso else f"{iso[8:10]}.{iso[5:7]}.{iso[0:4]}"


def T(rows, widths, font=None):
    return [table(rows, widths=[w * cm for w in widths], font=font), Spacer(1, 6)]


def srcline(*paths):
    # "results/*.json" would be read as italics by the kit's markup: name the folder instead
    return P("Источник (`experiments/v6_2/…`): " + "; ".join(f"`{p.replace('*.json', '')}`" for p in paths), "small")


# ---------------------------------------------------------------------------------------------- charts
def chart_market(d) -> Path:
    vs = d["bookmaker"]["vs_market"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    names = {"1X2": "1X2", "OU25": "Тотал 2.5", "AH": "Азиатская фора"}
    marks = {"1X2": "o", "OU25": "s", "AH": "D"}
    seasons = CL.SEASONS
    for i, m in enumerate(("1X2", "OU25", "AH")):
        xs, ys, lo, hi = [], [], [], []
        for j, s in enumerate(seasons):
            v = vs.get(f"{m}|{s}")
            if v:
                xs.append(j + (i - 1) * 0.18)
                ys.append(v["v"] * 1e3)
                lo.append(v["v"] * 1e3 - v["ci"][0] * 1e3)
                hi.append(v["ci"][1] * 1e3 - v["v"] * 1e3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marks[m], color=SERIES[i], ms=6, lw=1.6, capsize=0,
                    label=names[m])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(["2023/24", "2024/25", "2025/26\n(для инф.)", "2026/27\n(не holdout)"])
    ax.set_xlim(-0.5, len(seasons) - 0.2)
    ax.set_ylabel("log loss модель − рынок, ×10⁻³")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    ax.text(-0.45, -1.5, "ниже нуля — модель точнее рынка", fontsize=7.5, color=MUTED, va="top")
    return K.save_chart(fig, FIG / "model_vs_market.png")


def roi_rows(d):
    """Best change of Main / Risk ROI that each study found (difference to production, pooled selection seasons
    unless said otherwise)."""
    t = d["temporal"]["variants"]["T9"]
    return [
        ("Main: лучший вариант калибровки/ранжирования", d["main_calibration"]["best_roi"], "main_calibration"),
        ("Risk: лучшее правило (стек с рынком)", d["risk"]["best_roi"], "risk"),
        ("Main: фильтр по неопределённости", d["uncertainty"]["best_filter"], "uncertainty"),
        ("Main: xG X5 (2026/27, вне отбора)", d["xg"]["X5_droi"], "xg"),
        ("Main: смесь XGB + Махер (голы)", d["goal_distributions"]["hy_droi"], "goal_distributions"),
        ("Main: стекинг тотала S_TOT", d["market_models"]["s_tot_droi_select"], "market_models"),
        ("Main: все временные признаки T9", {"v": t["droi"], "ci": t["droi_ci"]}, "temporal"),
        ("Main: Elo новичка −50", d["data_quality"]["elo50_droi"], "data_quality"),
    ]


def chart_roi(d) -> Path:
    rows = roi_rows(d)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (lab, m, _) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([m["ci"][0] * 100, m["ci"][1] * 100], [y, y], color=SERIES[0], lw=2, solid_capstyle="round")
        ax.plot(m["v"] * 100, y, "o", color=SERIES[0], ms=6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlabel("изменение ROI к продакшну, п.п. (95 % ДИ)")
    ax.grid(axis="y", visible=False)
    return K.save_chart(fig, FIG / "roi_changes.png")


# ---------------------------------------------------------------------------------------------- sections
def plain(rep, d):
    ds, ot, bk, mc, rk = d["data_sources"], d["odds_timing"], d["bookmaker"], d["main_calibration"], d["risk"]
    rep.add(H1("1. Простыми словами"),
            P(f"В v6.2 проведено 13 исследований. Они перебрали около {n(d['cross']['variants_total'])} вариантов "
              "моделей, признаков, правил выбора и калибровок. Итог честный и короткий:"),
            *bullets([
                f"**Модель не точнее букмекера.** По 1X2 она уступает цене Bet365 до начала матча в каждом сезоне "
                f"(2023/24: {ll(bk['vs_market']['1X2|2023'])}×10⁻³ log loss) и в {n(d['leagues']['n_worse'])} из "
                f"{n(d['leagues']['n_leagues'])} лиг. Это нормально для открытых данных: в цене уже есть всё, что "
                "знает модель, и больше.",
                f"**Поэтому ставки по прогнозам в минусе.** Main 2026/27: ROI {pc(ds['main_overall'])} на "
                f"{n(ds['main_overall']['n'])} ставках, Risk — {pc(ds['risk_overall'])} на {n(ds['risk_overall']['n'])}. "
                f"Маржа Bet365 в 1X2 — {x(bk['margins']['1X2|2026|b365_open']['v'] * 100, 1)} %; "
                "модель её не перекрывает.",
                f"**«Положительный EV» обманчив.** Из многих кандидатов система выбирает тот, где модель сильнее всего "
                f"расходится с рынком, — и чаще всего это ошибка модели. Поэтому показанная вероятность Main "
                f"завышена на {pp(mc['shown'])} п.п., а EV Risk по модели {x(rk['base|2026']['ev_model'] * 100, 1)} % "
                f"при EV по рынку {x(rk['base|2026']['ev_market'] * 100, 1)} %.",
                f"**Часть прежнего ROI была «задним числом».** {n(ot['counts']['v']['C']['main'])} из "
                f"{n(ot['roi']['main|current_all_sources']['n'])} ставок Main 2026/27 — ретроспективные прогнозы по "
                f"цене закрытия. По ценам, которые действительно были при прогнозе, ROI Main "
                f"{pc(ot['roi']['main|live_A1_A2'])} ({n(ot['roi']['main|live_A1_A2']['n'])} ставок) — не плюс, "
                "но и не доказанный минус. Теперь у каждой цены есть метка времени.",
                "**Данные в порядке.** Признаки пересчитаны независимо — 0 расхождений; найденные дефекты данных "
                "исправлены в v6.2.",
                "**Ничего из исследованных улучшений в продакшн не вошло.** Несколько идей (калибровка Main, "
                "смесь моделей голов, xG и др.) заморожены и проверяются на будущих матчах; первые выводы — "
                f"не раньше {dru(min(s['earliest'] for s in d['forward']['shadows']['v'] if s['earliest']))}.",
                "**Что это значит для владельца.** Сайт честно показывает прогнозы и их историю; как инструмент "
                "для прибыльных ставок система пока не годится, и v6.2 объясняет почему цифрами.",
            ]))


def overview(rep, d):
    xg, dq, ot, bk, mc, rk = d["xg"], d["data_quality"], d["odds_timing"], d["bookmaker"], d["main_calibration"], d["risk"]
    mm, gd, lg, tp, un, fw, ds = (d[k] for k in ("market_models", "goal_distributions", "leagues", "temporal",
                                                 "uncertainty", "forward", "data_sources"))
    impl = sum(1 for s in fw["shadows"]["v"] if s["implemented"])
    first = min(s["earliest"] for s in fw["shadows"]["v"] if s["earliest"])
    t9 = tp["variants"]["T9"]
    rows = [["Тема", "Главный вопрос", "Главный результат [95 % ДИ]", "Решение", "Отчёт"],
            ["xG", "Помогает ли xG/xGA (есть только в 2026/27)?",
             f"Лучший X5: девиация голов {ll(xg['X5_goals'], 1)}×10⁻³, log loss 1X2 {ll(xg['X5_ll'])}×10⁻³; "
             f"значимо лучше после Холма: {n(xg['n_significant_better'])} из {n(xg['n_variants'])}",
             "B: X5, T1; C: остальные", "XG_Research.pdf"],
            ["Качество данных", "Верны ли данные и признаки; вредит ли нехватка истории?",
             f"{n(dq['cells_mismatch'])} расхождений на {n(dq['cells'])} значениях; без истории хозяев log loss "
             f"{ll(dq['zero_home_ll'])}×10⁻³", "A: D1–D6 (сделано); B: Elo новичка −50; C: 3", "Data_Quality_Research.pdf"],
            ["Время цены", "Существовала ли цена при прогнозе; честен ли ROI истории?",
             f"ROI Main по ценам при прогнозе {pc(ot['roi']['main|live_A1_A2'])}; справедливый CLV Main "
             f"{pc(ot['clv_main_fair'])}", "A: метки времени (сделано); B: 2; C: 1", "Odds_Timing_Research.pdf"],
            ["Букмекер", "Точнее ли модель вероятности рынка без маржи?",
             f"1X2, модель − Bet365: {ll(bk['vs_market']['1X2|2024'])}×10⁻³ (2024/25); value-ставки 1X2 "
             f"{pc(bk['value_bets']['1X2|2024'])}", "A: показ рынка; B: пул H2; C: гибриды", "Bookmaker_Benchmark_Research.pdf"],
            ["Калибровка Main", "Можно ли убрать переоценку Main?",
             f"gap показанного Main 2026/27 {pp(mc['shown'])} п.п.; помесячная PS-Platt {pp(mc['platt_monthly'])} "
             f"п.п. (in-sample); лучший ΔROI {pp(mc['best_roi'])} п.п.", "B: H1, H2; C: прочее", "Main_Calibration_Research.pdf"],
            ["Risk 2.0", "Почему EV Risk > 0, а ROI < 0?",
             f"2023/24: EV модели {pp(rk['base|2023'], key='ev_model', ci=None)} %, EV рынка "
             f"{pp(rk['base|2023'], key='ev_market', ci=None)} %, ROI {pc(rk['base|2023'])}; лучший ΔROI "
             f"{pp(rk['best_roi'])} п.п.", "A: возврат на целой линии (сделано); B: 2; C", "Risk_Selection_Research.pdf"],
            ["Модели рынков", "Нужны ли отдельные модели для рынков?",
             f"из {n(mm['n_direct'])} прямых моделей значимо хуже {n(mm['n_direct_worse'])}, лучше "
             f"{n(mm['n_direct_better'])}; S_TOT {ll(mm['s_tot'], 2)}×10⁻³ (p Холма {pv(mm['s_tot']['p_holm'])})",
             "B: S_TOT; C: 17", "Market_Models_Research.pdf"],
            ["Распределения голов", "Лучше ли другое распределение счёта?",
             f"смесь XGB + Махер {ll(gd['hy'], 2)}×10⁻³ (p Холма {pv(gd['hy']['p_holm'])}), но ΔROI Main "
             f"{pp(gd['hy_droi'])} п.п.", "B: смесь; C: 20", "Goal_Distributions_Research.pdf"],
            ["Лиги", "Нужна ли специализация по лигам?",
             f"значимо хуже {n(lg['n_variants_worse_sig'])} из {n(lg['n_variants'])} вариантов; отдельная модель "
             f"на лигу {ll(lg['variants']['M2'], 2)}×10⁻³", "C: все 15", "League_Specialization_Research.pdf"],
            ["Временные признаки", "Дают ли новые признаки формы прирост?",
             f"все 8 семейств вместе (T9) {ll({'v': t9['v'], 'ci': t9['ci']}, 2)}×10⁻³; утечки нет",
             "C: все 12", "Temporal_Features_Research.pdf"],
            ["Неопределённость", "Предсказывает ли разброс ансамбля проигрыш Main?",
             f"AUC {x(un['auc_min'])}–{x(un['auc_max'])}; значимых мер {n(un['n_measures_sig'])} из "
             f"{n(un['n_measures'])}; лучший фильтр {pp(un['best_filter'])} п.п. (p Холма {pv(un['best_filter']['p_holm'])})",
             "A: отчёты; B: метка u_1x2; C", "Prediction_Uncertainty_Research.pdf"],
            ["Форвард", "Как честно проверить B-гипотезы?",
             f"{len(fw['shadows']['v'])} теней в реестре, {impl} записываются; форвард-результатов 0; первые выводы "
             f"не раньше {dru(first)}", "A: запись теней (сделано); B: все", "Forward_Validation_Research.pdf"],
            ["Источники данных", "Что даёт каждый источник; верна ли история ставок?",
             f"OddsPapi {n(ds['op_with'])} из {n(ds['op_played'])} матчей; API-Football раньше Football-Data на "
             f"{x(ds['af_lead_median_h'], 1)} ч (медиана); ROI Main {pc(ds['main_overall'], 2)}",
             "A: 5 (сделано/готово); B: xG; C: The Odds API", "Data_Sources_Report.pdf"]]
    rep.page_break()
    rep.add(H1("2. Все исследования на одной странице"),
            *T(rows, [2.8, 3.1, 6.4, 2.5, 2.2], font=8),
            P("Log loss — разность «вариант − эталон» (меньше — лучше), ×10⁻³; gap — заявленная вероятность минус "
              "доля выигрышей; п.п. — процентные пункты. Отчёты лежат в `reports/v6_2/`; числа — из "
              "`experiments/v6_2/<исследование>/results/` (полный список файлов — в разделе 19).", "small"))
    rep.page_break()


def q1(rep, d):
    bk, rk, ot, dq, tp, lg, un, ds = (d[k] for k in ("bookmaker", "risk", "odds_timing", "data_quality", "temporal",
                                                      "leagues", "uncertainty", "data_sources"))
    r0, r1 = tp["variants"]["R0"], tp["variants"]["R1"]
    rep.add(H1("3. Что нового удалось узнать о FootPredict?"),
            *bullets([
                f"**Отставание от рынка постоянное и равномерное.** Модель − Bet365 (открытие) по 1X2: "
                + ", ".join(f"{s[:4]}/{int(s[2:4]) + 1}: {ll(bk['vs_market'][f'1X2|{s}'])}" for s in CL.SEASONS)
                + f" ×10⁻³; хуже рынка в {n(lg['n_worse'])} из {n(lg['n_leagues'])} лиг (значимо — в "
                f"{n(lg['n_worse_sig'])}). Лиг, где модель надёжно лучше, нет.",
                f"**Убыток объясняется механикой выбора, а не ошибкой кода.** EV Risk по модели выше EV по цене "
                f"без маржи на " + ", ".join(pp(rk[f'base|{s}'], key='model_minus_market', ci=None) for s in CL.SEASONS)
                + " п.п. по сезонам, а реальный ROI совпадает с EV рынка (ДИ разности содержит 0). Это «проклятие "
                "победителя»: выбирается кандидат с наибольшей ошибкой модели в свою пользу.",
                f"**Переоценка Main выросла в 2026/27** ({pp(d['main_calibration']['shown'])} п.п. против "
                f"{pp(d['main_calibration']['base|site|2023'])} в 2023/24), потому что у правила стало больше "
                f"кандидатов с ценой: медиана {n(d['main_calibration']['ncand_2024'])} в 2024/25 → "
                f"{n(d['main_calibration']['ncand_2026'])} в 2026/27.",
                f"**История ставок смешивала разные цены.** Main 2026/27: A1 (записана при прогнозе) "
                f"{n(ot['counts']['v']['A1']['main'])}, A2 {n(ot['counts']['v']['A2']['main'])}, B (опубликована после "
                f"прогноза, медиана через {x(ot['B_after_h'], 1)} ч) {n(ot['counts']['v']['B']['main'])}, C (закрытие "
                f"для ретроспективы) {n(ot['counts']['v']['C']['main'])}.",
                f"**Конвейер признаков точен.** Независимый пересчёт: {n(dq['cells_mismatch'])} расхождений на "
                f"{n(dq['cells'])} значениях ({n(dq['matches'])} матчей × {n(dq['features'])} признаков); тест "
                f"утечки временных признаков пройден ({'да' if tp['leak_all_passed']['v'] else 'нет'}).",
                f"**Elo несёт почти весь временной сигнал.** Без оконной формы и Elo (R0) {ll(r0)}×10⁻³, с Elo, "
                f"но без формы (R1) {ll(r1, 2)}×10⁻³ — форма поверх Elo почти ничего не добавляет.",
                f"**Неуверенность модели не предупреждает о проигрыше.** AUC мер {x(un['auc_min'])}–{x(un['auc_max'])} "
                "(0.5 — угадывание).",
                f"**API-Football даёт результат намного раньше.** Медиана от начала матча до результата: "
                f"{x(ds['af_delay_h'], 1)} ч против {x(ds['fd_delay_h'], 1)} ч у Football-Data.",
            ]),
            srcline("bookmaker/results/benchmark.json", "leagues/results/perf.json", "risk/results/risk.json",
                    "main_calibration/results/*.json", "odds_timing/results/audit.json",
                    "data_quality/results/recompute.json", "temporal/results/*.json",
                    "uncertainty/results/predictive.json", "data_sources/results/api_football.json"))


def q2(rep, d):
    tp, lg, dq, bk, ds, xg = (d[k] for k in ("temporal", "leagues", "data_quality", "bookmaker", "data_sources", "xg"))
    rows = [["Данные", "Что показало исследование", "Число [95 % ДИ]", "Источник"],
            ["Цена рынка (Bet365, Pinnacle)", "точнее модели во всех сезонах; модель рядом с ценой почти ничего не "
             "добавляет (вес b₂ на отборе)", f"1X2: b₂ = {bk['encompassing']['1X2|select']['v']:+.3f} "
             f"[{bk['encompassing']['1X2|select']['ci'][0]:+.3f}; {bk['encompassing']['1X2|select']['ci'][1]:+.3f}]",
             "bookmaker"],
            ["Elo", "без Elo и формы качество падает; Elo заменяет оконную форму",
             f"R0 {ll(tp['variants']['R0'])}×10⁻³; R1 {ll(tp['variants']['R1'], 2)}×10⁻³", "temporal"],
            ["История команды", "обнуление признаков хозяев", f"{ll(dq['zero_home_ll'])}×10⁻³; Main меняется в "
             f"{x(dq['zero_home_main_changed']['v'] * 100, 0)} % матчей", "data_quality"],
            ["Данные всех лиг вместе", "модель, не видевшая лигу, против модели только на своей лиге",
             f"{ll(lg['transfer_all'])} против {ll(lg['transfer_own'])}×10⁻³", "leagues"],
            ["Реальные цены OddsPapi", "покрытие матчей и прогнозов ценой",
             f"{x(ds['op_share']['v'] * 100, 2)} % матчей; Main {x(ds['main_overall']['coverage'] * 100, 2)} % "
             f"(v6.1: {x(ds['v61_cov_main'], 1)} %)", "data_sources"],
            ["Результаты API-Football", "опережение Football-Data; прогнозы, рассчитываемые сразу",
             f"{x(ds['af_lead_median_h'], 1)} ч; {n(ds['af_settleable'])} из {n(ds['af_main_risk'])} Main/Risk",
             "data_sources"],
            ["xG", "только 2026/27, 18 лиг; прироста нет", f"X5 {ll(xg['X5_ll'])}×10⁻³ (1X2)", "xg"]]
    rep.add(H1("4. Какие данные оказались наиболее полезными?"),
            P("Полезность измерялась тем, насколько хуже становится прогноз без этих данных (или лучше — с ними). "
              "Самые ценные — то, что уже используется: цена рынка как эталон, Elo, история команды и общий пул "
              "всех лиг. Новые источники полезны для полноты и скорости, а не для точности."),
            *T(rows, [3.2, 6.0, 5.3, 2.5]),
            P("Источник: `results/` указанного исследования (`experiments/v6_2/<исследование>/results/`).", "small"))


def q3(rep, d):
    xg = d["xg"]
    rep.add(H1("5. Помогает ли xG?"),
            note(f"Нет, пока не доказано. Из {n(xg['n_variants'])} вариантов с xG ни один не лучше v6.1 после "
                 f"поправки Холма. Лучший (X5, смесь с рейтингом xG): девиация голов {ll(xg['X5_goals'], 1)}×10⁻³, "
                 f"log loss 1X2 {ll(xg['X5_ll'])}×10⁻³."),
            *bullets([
                f"xG есть только в 2026/27: {n(xg['leagues'])} лиг, {n(xg['matches_with_xg'])} матчей с "
                f"{dru(xg['first_date']['v'])}; в {', '.join(xg['no_xg_leagues']['v'])} колонок нет. Оценка — на "
                f"{n(xg['n_eval'])} матчах {dru(xg['eval_period']['v'][0])}–{dru(xg['eval_period']['v'][1])}.",
                f"Чистая модель по xG (X1) значимо хуже: голы {ll(xg['X1_goals'], 1)}×10⁻³, 1X2 {ll(xg['X1_ll'])}×10⁻³.",
                f"Для угловых xG вредит: {ll(xg['corners'], 1)}×10⁻³ девиации.",
                f"Main: X5 меняет переоценку на {pp(xg['X5_dgap'])} п.п. и ROI на {pp(xg['X5_droi'])} п.п. — "
                "не значимо.",
                f"Чтобы увидеть эффект X5 с мощностью 80 %, нужно около {n(xg['X5_n_required'])} матчей; "
                f"предрегистрированная форвард-проверка — на ≥ {n(xg['forward_min'])} матчах "
                f"(не раньше {dru(next(s['earliest'] for s in d['forward']['shadows']['v'] if s['key'] == 'xg_x5@v1'))}).",
                f"T1 (прокси xG по ударам на длинной истории): 1X2 {ll(xg['T1_ll'])}×10⁻³; тень не записывается "
                "(нужно переобучение моделей).",
            ]),
            srcline("xg/results/metrics.json", "xg/results/main.json", "xg/results/corners.json",
                    "xg/results/audit.json", "forward/results/eta.json"))


def q4(rep, d):
    gd, mm, xg, tp, lg, dq = (d[k] for k in ("goal_distributions", "market_models", "xg", "temporal", "leagues",
                                              "data_quality"))
    fam = gd["families"]
    rows = [["Модель / вариант", "Δ к продакшну ×10⁻³ [95 % ДИ]", "p Холма", "ΔROI Main, п.п.", "Решение"],
            ["Смесь XGB + Махер + согл. (счёт)", ll(gd["hy"], 2), pv(gd["hy"]["p_holm"]), pp(gd["hy_droi"]), "B"],
            ["Стекинг тотала S_TOT (тотал)", ll(mm["s_tot"], 2), pv(mm["s_tot"]["p_holm"]),
             pp(mm["s_tot_droi_select"]), "B"],
            ["xG X5 (голы, 2026/27)", ll(xg["X5_goals"], 2), pv(xg["X5_goals"]["p_holm"]), pp(xg["X5_droi"]), "B"],
            ["Elo новичка −50 (1X2)", ll(dq["elo50_ll"], 2), "—", pp(dq["elo50_droi"]), "B"],
            ["Бивариантный Пуассон + согл. (счёт)", ll(fam["BP+R"], 2), "—", "—", "C (≡ продакшн)"],
            ["Все временные признаки T9 (1X2)", ll({"v": tp["variants"]["T9"]["v"], "ci": tp["variants"]["T9"]["ci"]}, 2),
             pv(tp["variants"]["T9"]["p_holm"]),
             pp({"v": tp["variants"]["T9"]["droi"], "ci": tp["variants"]["T9"]["droi_ci"]}), "C"],
            ["Лучшая специализация по лигам M5 (1X2)", ll(lg["variants"]["M5"], 2), pv(lg["variants"]["M5"]["p_holm"]),
             f"{lg['variants']['M5']['droi'] * 100:+.1f}", "C"]]
    rep.add(H1("6. Какие модели улучшились?"),
            P("В продакшне — ни одна: модели v6.1 не менялись. В исследованиях вероятность улучшили лишь немного "
              "и только по одной метрике; значимо после поправки Холма — одна смесь средних голов, и та снижает "
              "ROI Main."),
            *T(rows, [5.0, 4.0, 1.7, 3.6, 2.7]),
            *bullets([
                f"Смесь XGB + Махер: по сезонам " + ", ".join(f"{k[:4]}: {v * 1e3:+.2f}" for k, v in gd['hy']['seasons'].items())
                + " ×10⁻³ — выигрыш исчезает к 2026/27.",
                f"S_TOT лучше во всех четырёх сезонах (" + ", ".join(f"{v * 1e3:+.2f}" for v in mm['s_tot']['seasons'].values())
                + " ×10⁻³), но эффект крошечный и не проходит Холма.",
                "Бивариантный Пуассон «значим», но его параметр λ₃ ≈ 0 — он совпадает с продакшном (решение C).",
            ]),
            srcline("goal_distributions/results/comparisons.json", "goal_distributions/results/main_risk.json",
                    "market_models/results/comparisons.json", "market_models/results/main_risk.json",
                    "xg/results/*.json", "data_quality/results/variants.json", "temporal/results/variants.json",
                    "leagues/results/variants.json"))


def q5(rep, d):
    bk, mm, ds = d["bookmaker"], d["market_models"], d["data_sources"]
    vs = bk["vs_market"]
    mk = {"1X2": "1X2", "OU25": "Тотал 2.5", "AH": "Аз. фора"}
    rows = [["Рынок"] + ["2023/24", "2024/25", "2025/26 (инф.)", "2026/27 (не holdout)"]]
    for m in ("1X2", "OU25", "AH"):
        rows.append([mk[m]] + [ll(vs.get(f"{m}|{s}")) for s in CL.SEASONS])
    mr = mm["model_minus_market"]
    g = ds["groups"]
    rows2 = [["Группа рынков", "Main: ставок", "Main: ROI", "Risk: ставок", "Risk: ROI"]]
    for grp, lab in (("1X2", "1X2"), ("TOTAL", "Тоталы"), ("HANDICAP", "Азиатские форы"),
                     ("DOUBLE_CHANCE", "Двойной шанс"), ("CORNERS", "Угловые"), ("BTTS", "Обе забьют")):
        a, b = g.get(f"{grp}|main"), g.get(f"{grp}|risk")
        rows2.append([lab, n(a["n"]) if a else "—", pc(a) if a else "—", n(b["n"]) if b else "—", pc(b) if b else "—"])
    rep.page_break()
    rep.add(H1("7. Какие рынки прогнозируются лучше?"),
            P("«Лучше» — ближе к вероятности рынка без маржи (рынок — самый строгий эталон). Ближе всего модель к "
              "рынку на азиатской форе (с 2024/25 разница не значима), дальше — тотал 2.5, хуже всего — 1X2."),
            *T(rows, [3.0, 3.5, 3.5, 3.5, 3.5]),
            *figure(chart_market(d), "Модель − рынок (Bet365 открытие, без маржи), log loss ×10⁻³ с 95 % ДИ. "
                                     "Источник: experiments/v6_2/bookmaker/results/benchmark.json", width=15 * cm),
            P(f"Для 2026/27 есть также BTTS и угловые 9.5 (без ДИ): модель − рынок "
              f"{mr['BTTS|2026']['v'] * 1e3:+.1f} и {mr['CORNERS_9.5|2026']['v'] * 1e3:+.1f}×10⁻³ против "
              f"{mr['1X2|2026']['v'] * 1e3:+.1f} у 1X2 и {mr['TOTAL_2.5|2026']['v'] * 1e3:+.1f} у тотала 2.5 "
              "(`market_models/results/market_reference.json`). Отдельные модели для рынков не лучше вывода из "
              f"общей матрицы счёта: из {n(mm['n_direct'])} прямых моделей значимо хуже {n(mm['n_direct_worse'])}."),
            P("ROI по группам рынков 2026/27 (история сайта, только для информации: малые группы, ДИ не считались):"),
            *T(rows2, [4.0, 2.6, 3.6, 2.6, 4.2]),
            srcline("bookmaker/results/benchmark.json", "market_models/results/market_reference.json",
                    "data_sources/results/bets_by_group.csv"))


def q6(rep, d):
    mc, xg, un = d["main_calibration"], d["xg"], d["uncertainty"]
    rows = [["Сезон", "gap Main (все Main), п.п.", "gap Main (с ценой), п.п.", "ROI Main с ценой"]]
    for s in CL.SEASONS:
        lab = {"2023": "2023/24", "2024": "2024/25", "2025": "2025/26 (инф.)", "2026": "2026/27 (пересчёт)"}[s]
        b = mc[f"base|bet|{s}"]
        rows.append([lab, pp(mc[f"base|site|{s}"]), pp(b), pp(b, key="roi", ci="roi_ci") + " %"])
    rows.append(["2026/27 как показано", pp(mc["shown"]), "—", pp(mc["shown"], key="roi", ci="roi_ci") + " %"])
    rows2 = [["Вариант (2026/27, показанный Main)", "gap, п.п. [95 % ДИ]", "Статус"],
             ["Продакшн (как показано)", pp(mc["shown"]), "действует"],
             ["PS-Platt по истории 2022–2026", pp(mc["platt_hist"]), "не переносится"],
             ["PS-Group (карты групп рынков)", pp(mc["ps_group"]), "B, тень"],
             ["PS-Platt по месяцам на Main продакшна", pp(mc["platt_monthly"]), "B, тень"],
             ["xG X5 (изменение gap к B0)", pp(xg["X5_dgap"]), "B, тень"],
             ["Среднее 8 бутстреп-моделей", pp(un["bag_2026_boot"]), "C"]]
    rep.add(H1("8. Уменьшилась ли переоценка Main?"),
            note(f"В продакшне — нет: показанный Main 2026/27 переоценён на {pp(mc['shown'])} п.п. Исследования "
                 f"нашли способ убрать её на уже просмотренных данных (помесячная PS-Platt {pp(mc['platt_monthly'])} п.п.), "
                 "но это гипотеза для форвард-проверки, а ROI она не меняет: выбор Main остаётся тем же."),
            *T(rows, [4.0, 4.4, 4.4, 4.2]),
            *T(rows2, [7.0, 5.0, 5.0]),
            *bullets([
                f"На сезонах 2023/24–2025/26 любая карта после выбора сводит gap к нулю (объединённо Δ|gap| "
                f"{pp(mc['pooled_platt_dabsgap'])} п.п., p Холма {pv(mc['pooled_platt_dabsgap']['p_holm'])}), "
                "но карты, обученные на истории, на 2026/27 не переносятся — иной режим цен.",
                f"Калибровка до выбора и ранжирование по EV/Келли переоценку усиливают (2024/25: Q2 "
                f"{pp(mc['q2_gap_2024'])}, R5 {pp(mc['r5_gap_2024'])} п.п.).",
                "Калибровка меняет только показанную вероятность: ROI во второй таблице одинаков для всех карт.",
            ]),
            srcline("main_calibration/results/variants.json", "xg/results/main.json", "uncertainty/results/bagging.json"))


def q7(rep, d):
    ds, ot, bk, mc, rk = d["data_sources"], d["odds_timing"], d["bookmaker"], d["main_calibration"], d["risk"]
    rows = [["Изменение", "ΔROI, п.п. [95 % ДИ]", "p Холма", "Исследование"]]
    for lab, m, st in roi_rows(d):
        rows.append([lab, pp(m), pv(m.get("p_holm")) if m.get("p_holm") is not None else "—", st])
    rows2 = [["Набор (2026/27)", "Main ROI [95 % ДИ]", "ставок", "Risk ROI [95 % ДИ]", "ставок"]]
    for k, lab in (("current_all_sources", "все цены (как на сайте)"), ("live_A1_A2", "только цены при прогнозе (A1+A2)"),
                   ("live_B", "цены после прогноза (B)"), ("backfill_C", "ретроспектива по закрытию (C)")):
        a, b = ot["roi"][f"main|{k}"], ot["roi"][f"risk|{k}"]
        rows2.append([lab, pc(a), n(a["n"]), pc(b), n(b["n"])])
    vb = bk["value_bets"]
    rep.add(H1("9. Улучшился ли ROI?"),
            note(f"Нет. ROI продакшна 2026/27: Main {pc(ds['main_overall'], 2)} ({n(ds['main_overall']['n'])} ставок, "
                 f"средний EV по модели {x(ds['main_overall']['ev'] * 100, 1)} %), Risk {pc(ds['risk_overall'], 2)} "
                 f"({n(ds['risk_overall']['n'])} ставок, EV {x(ds['risk_overall']['ev'] * 100, 1)} %). Ни один из "
                 "исследованных вариантов не улучшил ROI значимо после поправки на множественные сравнения."),
            *figure(chart_roi(d), "Лучшее изменение ROI в каждом исследовании (точка — оценка, линия — 95 % ДИ; "
                                  "объединённые сезоны отбора, xG — 2026/27). Источник: results/ исследований.",
                    width=15 * cm),
            *T(rows, [7.0, 4.2, 1.8, 4.0]),
            H2("Почему ROI не растёт"),
            *bullets([
                f"**Эффективность рынка.** Цена точнее модели (1X2 {ll(bk['vs_market']['1X2|2023'])}×10⁻³ в 2023/24); "
                "тест охвата не находит в модели информации сверх цены на сезонах отбора "
                f"(1X2: b₂ = {bk['encompassing']['1X2|select']['v']:+.3f}, p = {pv(bk['encompassing']['1X2|select']['p'])}).",
                f"**Маржа.** Bet365 до начала матча: 1X2 {x(bk['margins']['1X2|2026|b365_open']['v'] * 100, 1)} %, "
                f"тотал 2.5 {x(bk['margins']['OU25|2026|b365_open']['v'] * 100, 1)} %, фора "
                f"{x(bk['margins']['AH|2026|b365_open']['v'] * 100, 1)} % (2026/27). Справедливый CLV Main "
                f"{pc(ot['clv_main_fair'])}, Risk {pc(ot['clv_risk_fair'])} — ставки хуже честной цены закрытия.",
                f"**Проклятие победителя.** Value-ставки «везде, где EV модели > 0» по 1X2: ROI "
                + ", ".join(f"{s[:4]}: {pc(vb[f'1X2|{s}'])}" for s in ("2023", "2024", "2025"))
                + f" при среднем EV модели {x(vb['1X2|2023']['ev'] * 100, 1)} %. Чем больше кандидатов, тем сильнее "
                f"переоценка выбранного: Risk 2026/27 переоценён на {pp(rk['base|2026'], key='gap_chosen', ci=None)} п.п.",
                f"**Часть прежнего плюса/минуса — артефакт цены.** Ретроспективные прогнозы выбирались уже по цене "
                f"закрытия ({n(ot['counts']['v']['C']['main'])} Main). Честный ROI — по ценам при прогнозе "
                "(таблица ниже); его интервал широк и содержит ноль, но не говорит о прибыли.",
            ]),
            *T(rows2, [5.6, 4.0, 1.6, 4.0, 1.6]),
            srcline("data_sources/results/bets_overall.csv", "odds_timing/results/audit.json",
                    "odds_timing/results/clv.json", "bookmaker/results/*.json", "risk/results/risk.json"))


def q8(rep, d):
    bk, ot, lg, mm, ds = d["bookmaker"], d["odds_timing"], d["leagues"], d["market_models"], d["data_sources"]
    e = bk["encompassing"]
    h = bk["best_hybrid"]["OU25|2026"]
    rep.add(H1("10. В чём FootPredict превосходит букмекерские вероятности?"),
            note("В точности вероятностей — ни в чём надёжно. Есть лишь слабые признаки информации, которой нет в "
                 "цене, и все они найдены на уже просмотренном 2026/27 без поправки на множественность."),
            *bullets([
                f"**Направление движения линии.** При контроле других букмекеров цена Pinnacle смещается к модели "
                f"на b₁ = {x(ot['pinnacle_beta_min'])}–{x(ot['pinnacle_beta_max'])} доли разрыва (1X2, все сезоны "
                f"значимы). CLV Main к той же цене {pc(ot['clv_main_odds'], 2)} — модель обычно берёт цену, которая "
                f"потом падает; но справедливый CLV {pc(ot['clv_main_fair'])}: маржа больше этого преимущества.",
                f"**Тотал 2.5 в 2026/27.** Вес модели рядом с ценой b₂ = {e['OU25|2026']['v']:+.3f} "
                f"[{e['OU25|2026']['ci'][0]:+.3f}; {e['OU25|2026']['ci'][1]:+.3f}], p = {pv(e['OU25|2026']['p'])}; "
                f"пул модели и рынка {ll(h)}×10⁻³ (p = {pv(h['p'])}) — на сезонах отбора этого нет "
                f"(b₂ = {e['OU25|select']['v']:+.3f}, p = {pv(e['OU25|select']['p'])}). Это гипотеза B (тень bookmaker_h2_pool).",
                f"**Близость на мелких рынках.** BTTS и угловые 9.5 в 2026/27: модель − рынок "
                f"{mm['model_minus_market']['BTTS|2026']['v'] * 1e3:+.1f} и "
                f"{mm['model_minus_market']['CORNERS_9.5|2026']['v'] * 1e3:+.1f}×10⁻³ (без ДИ) — почти вровень, "
                "но не лучше.",
                f"**Лиги.** Точечно лучше рынка только {', '.join(lg['better_leagues']['v'])} (незначимо); "
                f"межлиговая неоднородность I² = {x(lg['I2']['v'] * 100, 0)} %, порядок лиг между сезонами не повторяется.",
                f"**Не в вероятностях, а в сервисе:** единая согласованная матрица на все {n(ds['op_used'])} рынков "
                f"(прямые модели расходились бы с 1X2 до {x(mm['coherence_max']['v'] * 100, 1)} п.п.), объяснение "
                "каждого прогноза, прозрачная история с метками времени цен, быстрые результаты API-Football.",
            ]),
            srcline("odds_timing/results/drift.json", "odds_timing/results/clv.json",
                    "bookmaker/results/encompassing.json", "bookmaker/results/hybrids.json",
                    "market_models/results/market_reference.json", "market_models/results/coherence.json",
                    "leagues/results/perf.json"))


def q9(rep, d):
    lg, tp, mm, gd, un, bk, mc, xg, dq = (d[k] for k in ("leagues", "temporal", "market_models", "goal_distributions",
                                                          "uncertainty", "bookmaker", "main_calibration", "xg",
                                                          "data_quality"))
    fam = gd["families"]
    rows = [["Направление", "Что не сработало", "Число [95 % ДИ]"],
            ["Специализация по лигам", f"{n(lg['n_variants_worse_sig'])} из {n(lg['n_variants'])} вариантов значимо "
             "хуже; отдельная модель на лигу", f"M2 {ll(lg['variants']['M2'], 2)}×10⁻³"],
            ["Временные признаки", "ни одно из 8 семейств не значимо; простые средние без Elo хуже",
             f"T9 {ll({'v': tp['variants']['T9']['v'], 'ci': tp['variants']['T9']['ci']}, 2)}×10⁻³"],
            ["Отдельные модели рынков", f"{n(mm['n_direct_worse'])} из {n(mm['n_direct'])} прямых моделей хуже "
             "матрицы", f"точный счёт {ll(mm['es'], 2)}×10⁻³"],
            ["Распределения голов", "Диксон–Коулс, отр. биномиальное, КМП, копулы — без выигрыша; рейтинги Махера "
             "одни — хуже", f"ДК {ll(fam['DC+R'], 2)}; Махер {ll(fam['MH1+R'], 2)}×10⁻³"],
            ["Неопределённость", f"ни одна из {n(un['n_measures'])} мер не значима; ни один из "
             f"{n(un['n_filters'])} фильтров не прошёл правило", f"AUC {x(un['auc_min'])}–{x(un['auc_max'])}"],
            ["Value betting по модели", "ставки на все исходы с EV модели > 0", f"1X2 2023/24 {pc(bk['value_bets']['1X2|2023'])}"],
            ["Калибровка до выбора, ранжирование по EV/Келли", "усиливают переоценку Main",
             f"Q2 {pp(mc['q2_gap_2024'])}, R5 {pp(mc['r5_gap_2024'])} п.п."],
            ["Чистая модель по xG", "хуже v6.1 по голам и 1X2", f"X1 1X2 {ll(xg['X1_ll'])}×10⁻³"],
            ["xG для угловых", "ухудшает прогноз", f"{ll(xg['corners'], 1)}×10⁻³"],
            ["Гибриды модели и рынка H1/H3/H4/H5", "не лучше цены после Холма", "p Холма 1.000 на отборе"],
            ["Отказ от Main при пустых признаках", "качество данных матча почти не связано с ошибкой",
             f"Спирмен {x(dq['dq_spearman'])}"]]
    rep.add(H1("11. Какие эксперименты не сработали?"),
            P("Отрицательные результаты — тоже результат: они закрывают популярные идеи для этой системы."),
            *T(rows, [4.2, 7.4, 5.4]),
            srcline("leagues/results/variants.json", "temporal/results/variants.json",
                    "market_models/results/comparisons.json", "goal_distributions/results/comparisons.json",
                    "uncertainty/results/*.json", "bookmaker/results/betting.json", "main_calibration/results/variants.json",
                    "xg/results/*.json", "data_quality/results/strata.json"))


# decisions: (study, category, what, status in production) — statuses are taken from the README sections
# "Data diagnostics and history fixes (v6.2)" / "Design v6.2…" and the code on this branch
DONE, PART, REC, SHADOW, NOSHADOW, NO = ("внедрено в v6.2", "частично внедрено", "только рекомендация",
                                         "форвард-тень пишется", "предрегистрация есть, тени нет", "не внедрять")


def decisions_rows(d):
    xgd = d["xg"]["decisions"]["v"]
    c_xg = sum(1 for v in xgd.values() if v == "C")
    c_mm = sum(1 for v in d["market_models"]["decisions"]["v"].values() if v == "C")
    c_gd = sum(1 for v in d["goal_distributions"]["decisions"]["v"].values() if v == "C")
    return [
        ["Качество данных", "A", "D1–D6 и переименование Edinburgh City (написания клубов, отозванный результат, "
         "перенос на ранний день, VOID без цены, алиасы)", DONE],
        ["Качество данных", "B", "Elo новичка −50; отметка присуждённых 3:0; флаг «мало данных»",
         REC + " (нет предрегистрации)"],
        ["Качество данных", "C", "Elo новичка −100; история National League; отказ от Main при пустых признаках", NO],
        ["Время цены", "A", "класс времени у каждой цены, время в UTC, подсказка у цены, у Football-Data время "
         "котировки не показывается", DONE],
        ["Время цены", "A", "официальный ROI истории — только цены, существовавшие при прогнозе (A1+A2)",
         PART + ": History показывает этот ROI рядом с общим"],
        ["Время цены", "B", "ретроспектива по цене номинального момента решения; ранняя публикация Main",
         REC + " (нет предрегистрации)"],
        ["Время цены", "C", "перевыбор Main по цене закрытия", NO],
        ["Букмекер", "A", "показывать вероятность рынка без маржи рядом с вероятностью модели", REC + " (не сделано)"],
        ["Букмекер", "B", "EV Main/Risk по пулу модели и цены H2", SHADOW],
        ["Букмекер", "C", "гибриды H1/H3/H4/H5, value betting по модели, специализация", NO],
        ["Калибровка Main", "B", "H1 PS-Platt по месяцам; H2 PS-Group (только показанная вероятность)", SHADOW],
        ["Калибровка Main", "C", "калибровка до выбора, ранжирование по EV/Келли/нижней границе, меньше рынков, "
         "поправка по лигам", NO],
        ["Risk 2.0", "A", "эффективный коэффициент целой линии без цены: (1 − P(возврат)) / P(выигрыш)", DONE],
        ["Risk 2.0", "B", "стек модели с рынком (пол EV ≥ 0); вероятность Risk = рынок без маржи", SHADOW],
        ["Risk 2.0", "C", "прочие ранжирования, сокращение ставок, показ EV модели как преимущества", NO],
        ["Модели рынков", "B", "стекинг тотала S_TOT", NOSHADOW],
        ["Модели рынков", "C", f"{c_mm} прочих вариантов (прямые модели, стекинг)", NO],
        ["Распределения голов", "B", "смесь XGB + Махер + согласование с 1X2", NOSHADOW],
        ["Распределения голов", "C", f"{c_gd} прочих; Диксон–Коулс оставить выключенным (ρ подбирать после "
         "согласования, если включать)", NO],
        ["xG", "B", "X5 (основной), T1 (вторичный)", "X5: " + SHADOW + "; T1: тень-заглушка"],
        ["xG", "C", f"{c_xg} прочих вариантов", NO],
        ["Лиги", "C", "все 15 вариантов специализации", NO],
        ["Временные признаки", "C", "все 12 вариантов", NO],
        ["Неопределённость", "A", "полосы неопределённости и интервалы метрик в отчётах", "в отчётах (не сайт)"],
        ["Неопределённость", "B", "метка «ненадёжный прогноз» по разбросу 1X2 (низкий приор)",
         "тень зарегистрирована, не пишется"],
        ["Неопределённость", "C", "метка по правилу отбора; фильтр Main; среднее ансамбля", NO],
        ["Форвард", "A", "запись замороженных теней в ежедневном прогоне (стадия shadow)", DONE],
        ["Источники данных", "A", "три алиаса API-Football; учёт платных запросов OddsPapi; цена перенесённого "
         "матча", DONE],
        ["Источники данных", "A", "внешний запуск ежедневного обновления в 03:17",
         PART + ": код готов, cron-сервис настраивает владелец"],
        ["Источники данных", "B", "xG — по итогам исследования xG", "см. xG"],
        ["Источники данных", "C", "The Odds API как резерв", NO + " (остаётся выключенным)"],
    ]


def q10_q11(rep, d):
    fw = d["forward"]
    done = [r for r in decisions_rows(d) if r[1] == "A"]
    rep.add(H1("12. Какие изменения следует внедрить?"),
            P("Категория A — готово к внедрению. Большая часть уже в коде ветки v6.2 (раздел README «Data diagnostics "
              "and history fixes (v6.2)»); это исправления данных и учёта, а не улучшения модели:"),
            *bullets([f"**{r[0]}:** {r[2]} — *{r[3]}*." for r in done]),
            P("Что осталось сделать из A и рядом с ним:"),
            *bullets([
                "показать вероятность рынка без маржи рядом с вероятностью модели (единственное несделанное A);",
                "сделать ROI по ценам при прогнозе (A1+A2) основным показателем истории, а общий — справочным;",
                "настроить внешний cron в 03:17 (инструкция в README, раздел «Daily update time (v6.2)»);",
                f"добавить тени для двух предрегистрированных B без записи (смесь голов, S_TOT) и реализовать "
                f"заглушки T1 и u_1x2 — иначе у них не будет форвард-данных; сейчас пишется "
                f"{sum(1 for s in fw['shadows']['v'] if s['implemented'])} из {len(fw['shadows']['v'])} теней реестра.",
            ]))
    rep.add(H1("13. Какие изменения пока не стоит внедрять?"),
            *bullets([
                "**Все гипотезы B** — до своего форвард-объёма (раздел 16): калибровка Main, стек Risk с рынком, "
                "вероятность Risk = рынок, пул H2, xG X5/T1, смесь голов, S_TOT, метка неопределённости, Elo новичка.",
                "**Ничего, что меняет выбор Main/Risk ради ROI:** ни один вариант не дал значимого прироста (раздел 9), "
                f"а смесь голов, лучшая по вероятности, снижает ROI Main на {pp(d['goal_distributions']['hy_droi'])} п.п.",
                "**Все решения C** (таблица раздела 15): специализация по лигам, новые временные признаки, отдельные "
                "модели рынков, value betting, фильтры по неопределённости, The Odds API как резерв.",
            ]))


def decisions_table(rep, d):
    rows = [["Исследование", "Кат.", "Что", "Статус в продакшне"]] + decisions_rows(d)
    rep.add(H1("15. Все решения A / B / C и что уже в продакшне"),
            P("«Внедрено в v6.2» — изменение есть в коде этой ветки и описано в README (разделы «Data diagnostics and "
              "history fixes (v6.2)», «Daily update time (v6.2)»). Остальное — только рекомендация исследования; "
              "исследования сами продакшн не меняют."),
            *T(rows, [3.4, 1.0, 7.8, 4.8], font=8))
    pr = d["production"]
    rep.add(H2("Другие изменения v6.2 (не результаты исследований)"),
            *bullets([
                f"Дизайн v6.2: тексты ниже порога контраста WCAG — тёмная тема {n(pr['before|dark'])} → "
                f"{n(pr['after|dark'])}, светлая {n(pr['before|light'])} → {n(pr['after|light'])} "
                f"({n(pr['after|dark']['pages'])} страниц на тему; `reports/v6_2/data/contrast_*.json`); откат "
                "`scripts/use_design.py v6.1`.",
                "Логотип открывает «Матчи» внутри приложения без перезагрузки; поля графиков подгоняются под подписи.",
                "Значок «Hosted with Streamlit» на Community Cloud убрать нельзя (страница хостинга рисуется до кода "
                "приложения) — нужен другой хостинг, решение за владельцем.",
                "Каждый прогноз хранит входы моделей (`inputs.parquet`): «Почему этот прогноз?» объясняет "
                "опубликованную вероятность; время каждого источника и шага, `waiting.csv` с причинами ожидания.",
            ]))


def forward_table(rep, d):
    fw = d["forward"]
    short = {"maincal_ps_platt_monthly@v1": "Main: PS-Platt по месяцам",
             "maincal_ps_group@v1": "Main: PS-Group",
             "risk_stacked@v1": "Risk: стек с рынком",
             "risk_market_probability@v1": "Risk: вероятность рынка",
             "bookmaker_h2_pool@v1": "Пул H2 (модель + Bet365)",
             "xg_x5@v1": "xG X5", "xg_t1@v1": "xG T1 (прокси по ударам)",
             "uncertainty_flag_u_1x2@v1": "Метка u_1x2"}
    rows = [["Id (реестр)", "Что", "Критерий (из реестра)", "N", "Запись", "Не раньше"]]
    for s in fw["shadows"]["v"]:
        rows.append([s["key"], short.get(s["key"], s["key"]), s["criterion"], n(s["min_n"]),
                     "да" if s["implemented"] else "нет", dru(s["earliest"]) if s["earliest"] else "— (тень не пишется)"])
    est = d["cross"]["eta_without_shadow"]["date"]
    for k, key, lab in (("goal_distributions", "HY+R", "Смесь XGB + Махер (счёт)"),
                        ("market_models", "S_TOT", "Стекинг тотала S_TOT")):
        p = d[k]["prereg"]
        rows.append([f"{k}: {key}", lab, "95 % ДИ Δ log loss < 0 и ДИ ΔROI Main не целиком ниже 0", n(p["min_n"]),
                     "нет", f"≈ {dru(est[k])}*"])
    rep.add(H1("16. Предрегистрированные форвард-гипотезы"),
            P(f"Гипотезы заморожены 09.11.2026 (файлы `results/preregistration.json` закоммичены до форвард-окна). "
              "Учитываются только матчи, начавшиеся после заморозки и записанные тенью до начала матча. Дата «не "
              "раньше» — оценка исследования «Форвард» по темпу сезона (`forward/results/eta.json`)."),
            *T(rows, [3.6, 2.5, 5.8, 1.1, 1.6, 2.4], font=8),
            P(f"* Тени нет: дата оценена в этом отчёте как дата записи предрегистрации + N / "
              f"{x(fw['played_per_week'], 1)} матчей в неделю (`eta.json`); без тени нужен отдельный пересчёт на "
              "будущих матчах с замороженными параметрами.", "small"),
            P(f"Проверка механики на реплее (мнимая заморозка, не доказательство): {n(fw['sim_runs'])} прогонов, "
              f"{n(fw['sim_matches'])} матчей, записей в момент начала или позже — {n(fw['sim_after_kickoff'])}, "
              f"подделка обнаружена — {'да' if fw['sim_tamper']['v'] else 'нет'}, файлы продакшна не изменились — "
              f"{'да' if fw['sim_prod_unchanged']['v'] else 'нет'}.", "small"))


def caveats(rep, d):
    tried = d["cross"]["variants_tried"]
    total = d["cross"]["variants_total"]
    rep.add(H1("17. Методологические оговорки"),
            *bullets([
                "**2026/27 — не нетронутый holdout.** Его просматривали исследования v6.1 и все исследования v6.2; "
                "все числа 2026/27 в отчёте — «для информации». Гипотезы, найденные на нём (метка u_1x2, пул H2 на "
                "тотале, помесячная PS-Platt), имеют низкую априорную вероятность.",
                "**2025/26** использовался в ранней разработке — тоже не независим. Отбор вариантов — только на "
                "2023/24 и 2024/25.",
                f"**Множественные сравнения.** Всего проверено около {n(total)} вариантов ("
                + ", ".join(f"{k} {n(v)}" for k, v in tried.items())
                + f"). При уровне 5 % без поправки ≈ {n(total * 0.05)} «значимых» результатов возникли бы случайно; "
                "поэтому внутри исследований применялась поправка Холма, а между исследованиями поправки нет — "
                "одиночный «значимый» результат следует считать гипотезой.",
                "**Форвард-данных ещё нет.** Ни одна гипотеза B не подтверждена и не отвергнута; первые выводы — "
                "не раньше дат раздела 16.",
                "**xG** есть только с 2026/27, поэтому его оценка — внутри одного сезона (недельное расширяющееся окно).",
                "**Цены прошлых сезонов** есть только у 1X2, тотала 2.5 и одной азиатской линии; ROI Main/Risk в "
                "отборе считается по части ставок. BTTS и угловые с ценой — только 2026/27.",
                "**Разные срезы данных.** Исследования запускались в разное время 09.11.2026 и на разных выборках "
                "(раздел 18): небольшие расхождения одних и тех же показателей ожидаемы.",
            ]))


def inconsistencies(rep, d):
    ds, ot, mc, xg, un, rk, bk, lg, fw, tp = (d[k] for k in ("data_sources", "odds_timing", "main_calibration", "xg",
                                                              "uncertainty", "risk", "bookmaker", "leagues", "forward",
                                                              "temporal"))
    rows = [["Показатель", "Значения в разных исследованиях", "Почему"],
            ["ROI Main 2026/27", f"data_sources {pc(ds['main_overall'], 2)} ({n(ds['main_overall']['n'])}); "
             f"odds_timing {pc({'v': ot['roi']['main|current_all_sources']['v']})} "
             f"({n(ot['roi']['main|current_all_sources']['n'])}); main_calibration "
             f"{pp(mc['shown'], key='roi', ci=None)} % ({n(mc['shown']['n'])}); xG B0 {pc({'v': xg['B0_roi']['v']})} "
             f"({n(xg['B0_roi']['n'])})", "разные срезы истории; xG — пересчёт B0 с 01.09"],
            ["Risk ROI 2026/27", f"data_sources {pc(ds['risk_overall'], 2)}; odds_timing "
             f"{pc({'v': ot['roi']['risk|current_all_sources']['v']})}; risk: как показан {pc({'v': rk['shown']['v']})}, "
             f"пересчёт {pc({'v': rk['base|2026']['v']})}", "срез; пересчёт выбора по текущим ценам"],
            ["Записанные (live) ставки Main", f"data_sources {n(ds['main_live']['n'])}; odds_timing "
             f"{n(ot['roi']['main|current_live']['n'])}", "срез истории"],
            ["gap Main 2026/27, п.п.", f"показанный {pp(mc['shown'], ci=None)}; пересчёт walk-forward "
             f"{pp(mc['base|site|2026'], ci=None)}; xG B0 {pp(xg['B0_gap'], ci=None)}; неопределённость "
             f"{pp(un['gap_2026'], ci=None)}", "показанный против пересчитанного; разные выборки"],
            ["Модель − рынок 1X2 2026/27, ×10⁻³", f"bookmaker {ll(bk['vs_market']['1X2|2026'])}; leagues "
             f"{ll(lg['seasons']['2026'])}", "разное число матчей (n = "
             f"{n(bk['vs_market']['1X2|2026']['n'])} и {n(lg['seasons']['2026']['n'])})"],
            ["Модель − рынок, отбор, ×10⁻³", f"leagues {ll(lg['select'])}; temporal T0 {ll(tp['t0_vs_market'])}",
             "сид 42 против среднего 3 сидов"],
            ["PS-Platt: число Main в подгонке", f"предрегистрация {n(fw['platt_prereg_n'])}, тень "
             f"{n(fw['platt_shadow_n'])}", "история изменилась после записи"],
            ["xG X5: недельный вес w", f"тень отличается от предрегистрации с недели {fw['x5_w_first_diff_week']['v']}, "
             f"максимум |Δw| = {x(fw['x5_w_max_diff'], 4)}", "тень берёт λ прошлых матчей из сохранённых прогнозов продакшна, исследование — пересчёт одной моделью"]]
    rep.add(H1("18. Расхождения между исследованиями"),
            *T(rows, [3.6, 9.0, 4.4], font=8),
            *bullets([
                "Две предрегистрации (смесь голов, S_TOT) не попали в форвард-реестр — для них не пишется тень.",
                "Тени T1 и u_1x2 зарегистрированы, но не реализованы (нужны отдельные модели).",
                "B-решения исследований качества данных и времени цены не предрегистрированы.",
                "Единственное A исследования «Букмекер» (показ вероятности рынка) в v6.2 не сделано.",
                f"S_TOT: ΔROI Main на отборе {pp(d['market_models']['s_tot_droi_select'])}, а на 2026/27 "
                f"{pp(d['market_models']['s_tot_droi_2026'])} п.п. — знак меняется, 2026/27 не независим.",
                "Смесь голов улучшает вероятность, но снижает ROI Main — цели «точность» и «ROI» здесь расходятся.",
            ]))


def future(rep, d):
    fw = d["forward"]
    rep.add(H1("14. Что исследовать в будущем?"),
            *bullets([
                "**Форвард-отчёт** по каждой тени ровно один раз при достижении N (раздел 16), с поправкой Холма "
                "внутри семейства; первая проверка — калибровка Main.",
                "**Чистый holdout 2027/28:** заморозить модели v6.2 до начала сезона и не смотреть на него до конца.",
                "**Более длинная история xG** из законных источников (сейчас только 2026/27); тогда — переобучение T1.",
                "**Смены тренера и составы:** в исследовании временных признаков не проверены — нет локального "
                "источника с датой до матча.",
                "**Время решения:** ретроспектива по цене номинального момента (нужны все снимки /historical-odds) "
                "и ранняя публикация Main — сейчас CLV по той же цене положителен "
                f"({pc(d['odds_timing']['clv_main_odds'], 2)}).",
                "**Эталон без маржи:** биржа Betfair (маржа закрытия 1X2 "
                f"{x(d['bookmaker']['margins']['1X2|2024|bfe_close']['v'] * 100, 1)} % против "
                f"{x(d['bookmaker']['margins']['1X2|2026|b365_open']['v'] * 100, 1)} % у Bet365) для честной оценки CLV.",
                "**Присуждённые результаты 3:0** и клубы из незагружаемых лиг (влияние не измерено).",
                f"**Сохранить ансамбль 8 бутстреп-моделей**, чтобы тень метки u_1x2 начала писаться (сейчас "
                f"{sum(1 for s in fw['shadows']['v'] if not s['implemented'])} тени не пишутся).",
            ]))


def files(rep, d):
    meta = d["meta"]
    rep.add(H1("19. Источники чисел и воспроизведение"),
            P(f"Все числа прочитаны при сборке из {len(meta['files_read'])} файлов результатов (сводка — "
              "`experiments/v6_2/summary/results/summary.json`, у каждого значения поле `src`). Файлы:"),
            P(", ".join(f"`{f.replace('experiments/v6_2/', '')}`" for f in meta["files_read"]), "small"),
            *K.reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                          "$PY -m experiments.v6_2.summary.collect     # results/summary.json",
                          "$PY -m experiments.v6_2.summary.build_pdf   # этот отчёт (вызывает collect)"]))


def build() -> Path:
    d = CL.collect()
    rep = K.Report(OUT, "Сводный отчёт исследований v6.2", "FootPredict v6.2 — сравнение выводов 13 исследований",
                   facts={"Исследований": "13", "Данные": "результаты исследований на 09.11.2026",
                          "Код": "experiments/v6_2/summary/"})
    plain(rep, d)
    overview(rep, d)
    for f in (q1, q2, q3, q4, q5, q6, q7, q8, q9, q10_q11, future, decisions_table, forward_table, caveats,
              inconsistencies, files):
        f(rep, d)
    rep.build()
    pages = rep.check()
    print(f"{OUT.relative_to(CL.ROOT)}: {pages} pages")
    return OUT


if __name__ == "__main__":
    build()
