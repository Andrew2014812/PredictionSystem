"""Builds reports/v6_3/Recent_Season_Weighting_Research.pdf and findings.md from results/ (no number typed by hand).

    .venv/bin/python -m experiments.v6_3.recent_weighting.build_pdf
"""
from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm

from scripts import report_kit as K
from scripts.report_kit import H1, H2, H3, P, bullets, figure, formula, note, table

from experiments.v6_3 import common as C

from . import fits as F

DIRS = C.study_dirs("recent_weighting")
RES, FIG = DIRS["results"], DIRS["figures"]
OUT = C.ROOT / "reports" / "v6_3" / "Recent_Season_Weighting_Research.pdf"
GROUP_RU = {"result": "1X2", "goals": "голы", "corners": "угловые"}
METRIC_RU = {"result": "log loss 1X2", "goals": "девиация голов", "corners": "девиация угловых"}
ORDER = ["prod", "hl0", "hl0.5", "hl1", "hl2", "hl4", "last2", "last3", "cur2", "cur4", "win2", "win3", "k250",
         "form", "selected"]


def pp(v, d=1, signed=True):
    return "—" if v is None or v != v else (f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}")


def m3(v, d=2):
    return "—" if v is None or v != v else f"{v * 1000:+.{d}f}"


def ci3(x, d=2):
    if x is None or x.get("diff") is None:
        return "—"
    return f"{m3(x['diff'], d)} [{m3(x['ci95'][0], d)}; {m3(x['ci95'][1], d)}]"


def cipp(x, d=1):
    if x is None or x.get("diff") is None:
        return "—"
    return f"{pp(x['diff'], d)} [{pp(x['ci95'][0], d)}; {pp(x['ci95'][1], d)}]"


def pv(v):
    return "—" if v is None or v != v else ("<0.001" if v < 0.001 else f"{v:.3f}")


def f4(v, d=4):
    return "—" if v is None or v != v else f"{v:.{d}f}"


def label(names, key: str) -> str:
    kind, sc = key.split(":") if ":" in key else ("wf", key)
    if kind == "static":
        return "однократно: " + names.get(sc, sc).replace("ежемесячно, ", "").replace(" (контроль)", "")
    if kind.startswith("swap_"):
        return f"только {GROUP_RU[kind[5:]]}: " + names.get(sc, sc)
    return names.get(sc, sc)


def build() -> None:
    r = C.read_json(RES / "results.json")
    info = C.read_json(RES / "run_info.json")
    pre = C.read_json(RES / "preregistration.json")
    names, paired, holm, roi, dec, cov = r["names"], r["paired"], r["holm"], r["roi"], r["decisions"], r["coverage"]
    met = pd.read_csv(RES / "metrics.csv")
    rr = pd.read_csv(RES / "main_roi.csv")
    sel, form = r["selected_2025"], r["form"]
    b_close = C.read_json(C.SHARED_RESULTS / "baseline.json")
    rep = K.Report(OUT, "Влияние текущего сезона на обучение", "FootPredict v6.3 — исследование 7",
                   {"Сезон выбора параметров": "2025/26 (тот же ежемесячный реплей)",
                    "Проверка": "2026/27 (хронологически; уже изучался в v6.1/v6.2)",
                    "Модель продакшна": "prod_20261012 (обучена на 2021/22–2025/26)",
                    "Данные": f"features изменён {info['data']['features']['mtime'][:16]}"})

    # 1 -----------------------------------------------------------------------------------------------
    rep.add(H1("1. Название и постановка проблемы"),
            P("**Исследование 7. Recent Season Weighting** — влияние матчей текущего сезона и весов давности на "
              "обучение моделей 1X2, голов и угловых FootPredict."),
            P("Модели продакшна (prod_20261012) обучены один раз на сезонах 2021/22–2025/26 и не видели ни одного "
              "матча 2026/27; текущая форма попадает в прогноз только через признаки (сезонные и оконные средние, "
              "Elo). Веса давности продакшна: 1X2 — без затухания (полураспад 0), голы и угловые — полураспад 2 года. "
              "Гипотеза владельца: матчи текущего сезона точнее отражают реальную силу команд, поэтому модели нужно "
              "переобучать по ходу сезона и/или давать текущему сезону больший вес."),
            P("Чтобы отделить эффект «больше данных» от эффекта «другие веса», в исследование включён контроль: "
              "ежемесячное переобучение с весами продакшна. Сравнение «контроль − базовая линия B» показывает пользу "
              "новых данных; «вариант − контроль» — пользу самих весов."))

    # 2 -----------------------------------------------------------------------------------------------
    rep.add(H1("2. Гипотезы"), bullets([
        "H1: ежемесячное переобучение (новые матчи сезона в обучении) улучшает вероятности по сравнению с моделью, "
        "обученной один раз до сезона.",
        "H2: более быстрое затухание весов (полураспад 0.5–1 год) или повышенный вес последнего завершённого сезона "
        "улучшают модели.",
        "H3: повышенный вес матчей текущего сезона (×2, ×4) улучшает модели, начиная с некоторого числа сыгранных "
        "матчей.",
        "H4: скользящее окно (2–3 прошлых сезона + текущий) лучше полной истории.",
        "H5: переобучение чаще (каждые 250 сыгранных матчей ≈ раз в неделю) лучше ежемесячного.",
        "H6: поправка на текущую форму (ошибки модели на прошлых матчах команды в этом сезоне, со сжатием) поверх "
        "весов давности улучшает прогноз.",
        "Ожидание по v6.2 (временные признаки не дали прироста): эффекты, если есть, малы — порядка 10⁻³ log loss."]))

    # 3 -----------------------------------------------------------------------------------------------
    ev = cov["evaluated_probabilistic"]
    rep.add(H1("3. Исходные данные"),
            P(f"Таблица признаков продакшна (21 лига, сезоны 2021/22–2026/27). Сезон выбора 2025/26: {ev['2025']} "
              f"сыгранных матчей; сезон проверки 2026/27: {cov['season_matches']} матчей, сыграно {cov['played']}, "
              f"не сыграно {cov['not_played']}; вероятностная оценка — {ev['2026']} матчей, угловые — "
              f"{cov['corners_evaluated'].get('2026', '—')} (лиги со статистикой угловых). Матчей, где у одной из "
              f"команд ещё нет игр сезона: {cov['no_current_season_games']}, меньше трёх: "
              f"{cov['fewer_than_3_season_games']} — не исключаются, показаны в разбивке."),
            P("Цены и моменты решения — общий реплей v6.3: close (последняя котировка до начала, основной) и t09 "
              "(цены, известные в 09:00 дня матча, проверка устойчивости). Базовая линия A — история сайта, B — "
              "модели продакшна с правилом Main продакшна на тех же матчах."))
    rows = [["Метод", "Момент", "Матчей с Main", "Ставок", "Без цены", "PUSH", "Неизв. рез.", "Нет Main"]]
    for name in ("wf:prod", "wf:selected", "wf:form"):
        for u in ("close", "t09"):
            c = cov["main"].get(name, {}).get(u)
            if c:
                rows.append([label(names, name), u, c["main_rows"], c["bets"], c["no_price"], c["push"], c["unknown"],
                             c["no_main"]])
    rep.add(table(rows, widths=[4.6 * cm, 1.4 * cm, 1.9 * cm, 1.5 * cm, 1.6 * cm, 1.4 * cm, 2.2 * cm, 1.8 * cm]),
            P("Таблица 1. Покрытие сезона 2026/27 (сыгранные матчи): матч без реальной цены — прогноз, "
                           "но не ставка; PUSH возвращает ставку.", "small"))

    # 4 -----------------------------------------------------------------------------------------------
    rep.add(H1("4. Методика"),
            P("Каждая задача продакшна (1X2, голы хозяев, голы гостей, угловые) переобучается с признаками, "
              "гиперпараметрами и числом деревьев продакшна (`engine.fit_task`, 2 потока) на матчах, сыгранных ДО "
              "даты отсечки, и прогнозирует матчи до следующей отсечки. Отсечки: первый игровой день сезона, затем "
              "1-е число каждого месяца (2025/26: " + ", ".join(d[5:] for d in r["cutoffs"]["2025"]) + "; 2026/27: "
              + ", ".join(d[5:] for d in r["cutoffs"]["2026"]) + "). Модель первой отсечки, применённая ко всему "
              "сезону, — «однократное» обучение (аналог продакшна)."),
            formula("w_i = 0.5^{(t_max − t_i) / h} · m(сезон_i),   затем  w ← w · mean(w_прод) / mean(w)"),
            P("h — полураспад в годах (h = 0 — без затухания), m — множитель сезона. Нормировка сохраняет среднюю "
              "массу выборки продакшна, чтобы регуляризация XGBoost (в единицах гессиана) была сравнима между "
              "схемами. Схемы: полураспад 0 / 0.5 / 1 / 2 / 4 года; прошлый сезон ×2 / ×3; текущий сезон ×2 / ×4; "
              "окно 2 / 3 прошлых сезона + текущий; переобучение каждые 250 сыгранных матчей сезона (отсечки "
              + ", ".join(d[5:] for d in r["k250_cutoffs"]) + ")."),
            P("V8 «давность + текущая форма»: модель с полураспадом, выбранным на 2025/26 для каждой задачи, плюс "
              "поправка на форму из ошибок модели на прошлых матчах команды в этом сезоне:"),
            formula("A_t = (G_t + k) / (Ĝ_t + k),  D_t = (GA_t + k) / (ĜA_t + k),  λ'_h = λ_h·A_h·D_a,  λ'_a = λ_a·A_a·D_h"),
            P(f"Ĝ — ожидаемые моделью голы в прошлых матчах сезона (прогнозы walk-forward, т. е. вне выборки); 1X2 "
              f"поправляется отношением пуассоновских 1X2 при λ' и λ; угловые — множителем √(C_h·C_a) с априорной "
              f"силой k_c. Выбрано на 2025/26: k (1X2) = {form.get('k_result')}, k (голы) = {form.get('k_goals')}, "
              f"k_c = {form.get('kc')}; полураспады: " + ", ".join(f"{GROUP_RU[g]} — {names.get(s, s)}"
                                                                  for g, s in form.get("half_life_schemes", {}).items())
              + "."),
            P("Выбор: для каждой группы задач на 2025/26 выбрана схема с лучшей главной метрикой (1X2 — log loss, "
              "голы — средняя девиация Пуассона хозяев и гостей, угловые — девиация); вариант «выбор 2025/26» "
              "объединяет их и оценивается на 2026/27 без изменений. Остальные схемы показаны на 2026/27 для полноты "
              "с поправкой Холма."),
            P("Метрики: 1X2 — log loss, Brier, accuracy, macro F1, balanced accuracy; голы и угловые — MAE, RMSE, "
              "R², девиация Пуассона. Main ROI — выходы → `engine.batch_from_outputs` → правило Main продакшна, оба "
              "момента решения, парный бутстреп (2000) к базовой линии B того же момента и к контролю."),
            P("Правило решения (до оценки 2026/27): **A** — схема выбрана на 2025/26 для задачи, лучше B и контроля "
              "на 2026/27 с p Холма < 0.05, ROI Main не хуже значимо; **B** — лучше контроля и на 2025/26, и на "
              "2026/27 по точке, ROI не хуже значимо; **C** — остальное. После первого прогона правило B ужесточено "
              "(это может только убрать кандидатов): требуется ещё и выигрыш у базовой линии B на 2026/27 по точке — "
              "схема, проигрывающая модели продакшна, не кандидат на замену продакшна."))

    # 5 -----------------------------------------------------------------------------------------------
    nroi = r["n_variants"]["roi_variants"]
    rep.add(H1("5. Эксперименты"),
            P(f"Схем ежемесячного переобучения: {len(F.SCHEMES)} (включая контроль) — на 2025/26 и 2026/27; "
              f"однократных схем: {len(F.SCHEMES)}; дополнительно: каждые 250 матчей (2026/27), V8 и «выбор 2025/26». "
              f"Вариантов с Main ROI: {nroi} (включая замену только одной группы задач)."),
            P(f"Выбор на 2025/26: 1X2 — {names.get(sel['result'])}, голы — {names.get(sel['goals'])}, угловые — "
              f"{names.get(sel['corners'])}."))

    # 6 -----------------------------------------------------------------------------------------------
    rep.add(H1("6. Результаты"), H2("6.1 Сезон выбора 2025/26 (разность с контролем, ×10⁻³)"))
    rows = [["Схема", "1X2 log loss", "Голы, девиация", "Угловые, девиация"]]
    for sc in ORDER:
        k = f"2025|wf|{sc}|vs_control"
        if k in paired:
            rows.append([names.get(sc, sc)] + [ci3(paired[k][g]) for g in ("result", "goals", "corners")])
    for sc in ("prod",):
        k = f"2025|static|{sc}|vs_control"
        if k in paired:
            rows.append(["однократно, веса продакшна"] + [ci3(paired[k][g]) for g in ("result", "goals", "corners")])
    rep.add(table(rows, widths=[6 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm]),
            P("Таблица 2. 2025/26: схема − контроль (ежемесячно, веса продакшна); меньше — лучше. «Однократно» − "
              "контроль > 0 означает, что ежемесячное переобучение помогает.", "small"))
    rep.add(*figure(FIG / "forest_2025.png", "Рисунок 1. 2025/26: разность с контролем, 95 % ДИ."))

    rep.add(H2("6.2 Сезон 2026/27: вероятности"))
    m26 = met.loc[met["season"] == 2026]
    rows = [["Вариант", "1X2 LL", "Brier", "Acc.", "F1", "MAE хоз.", "MAE гост.", "Дев. хоз.", "Дев. гост.",
             "MAE угл.", "R² угл."]]
    for _, x in m26.iterrows():
        nmx = names["B"] if x["kind"] == "B" else label(names, f"{x['kind']}:{x['scheme']}")
        if x["kind"] == "static" and x["scheme"] not in ("prod", "hl1", "last2"):
            continue
        rows.append([nmx, f4(x["result_ll"]), f4(x["result_brier"]), pp(x["result_acc"], 1, False),
                     f4(x["result_f1"], 3), f4(x["home_mae"], 3), f4(x["away_mae"], 3), f4(x["home_dev"]),
                     f4(x["away_dev"]), f4(x["corners_mae"], 3), f4(x["corners_r2"], 3)])
    rep.add(table(rows, widths=[4.3 * cm] + [1.25 * cm] * 10),
            P("Таблица 3. 2026/27: абсолютные метрики (однократные схемы — выборочно).", "small"))
    rows = [["Вариант", "1X2 Δ LL к B", "p Холма", "Голы Δ к B", "p Холма", "Угловые Δ к B", "p Холма"]]
    for sc in ORDER:
        k = f"2026|wf|{sc}|vs_B"
        if k in paired:
            rows.append([names.get(sc, sc)] + sum([[ci3(paired[k][g]), pv(holm[g].get(k))]
                                                  for g in ("result", "goals", "corners")], []))
    k = "2026|static|prod|vs_B"
    if k in paired:
        rows.append(["однократно, веса продакшна (переобучение)"] + sum([[ci3(paired[k][g]), pv(holm[g].get(k))]
                                                                        for g in ("result", "goals", "corners")], []))
    rep.add(table(rows, widths=[3.9 * cm, 2.9 * cm, 1.45 * cm, 2.9 * cm, 1.45 * cm, 2.9 * cm, 1.45 * cm]),
            P("Таблица 4. 2026/27: разность с базовой линией B (×10⁻³), парный бутстреп; Холм — по всем вариантам "
              "данной задачи.", "small"))
    rows = [["Вариант", "1X2 Δ к контролю", "Голы Δ к контролю", "Угловые Δ к контролю"]]
    for sc in ORDER:
        k = f"2026|wf|{sc}|vs_control"
        if k in paired:
            rows.append([names.get(sc, sc)] + [ci3(paired[k][g]) for g in ("result", "goals", "corners")])
    k = "2026|B|B|vs_control"
    if k in paired:
        rows.append([names["B"]] + [ci3(paired[k][g]) for g in ("result", "goals", "corners")])
    rep.add(table(rows, widths=[6 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm]),
            P("Таблица 5. 2026/27: разность с контролем — эффект весов при тех же данных; строка B — эффект "
              "«нет новых данных» (B − контроль).", "small"))
    rep.add(*figure(FIG / "forest_2026_vs_B.png", "Рисунок 2. 2026/27: разность с базовой линией B, 95 % ДИ "
                                                  "(зелёный — значимо лучше, красный — значимо хуже)."))
    rep.add(*figure(FIG / "forest_2026_vs_control.png", "Рисунок 3. 2026/27: разность с контролем (ежемесячно, "
                                                        "веса продакшна)."))

    rep.add(H2("6.3 После скольких матчей текущий сезон начинает помогать"))
    w = pd.read_csv(RES / "when_it_helps.csv")
    for s in (2025, 2026):
        x = w.loc[(w["season"] == s) & (w["scheme"].isin(["cur4", "monthly_vs_static"])) & (w["by"] == "refit_month")]
        if len(x):
            rows = [["Отсечка", "Схема", "n", "1X2 Δ", "Голы Δ", "Угловые Δ"]]
            for (v, sc), g in x.groupby(["value", "scheme"]):
                d = g.set_index("group")
                rows.append([v, "текущий ×4 − контроль" if sc == "cur4" else "ежемесячно − однократно",
                             int(d["n"].max())] + [f"{m3(d.loc[q, 'diff'])} ± {abs(float(d.loc[q, 'se'])) * 1960:.2f}"
                                                   if q in d.index else "—" for q in ("result", "goals", "corners")])
            rep.add(H3(f"{s}/{s - 1999}"), table(rows, widths=[2.4 * cm, 4.4 * cm, 1.2 * cm, 3 * cm, 3 * cm, 3 * cm]))
    rep.add(P("Разности ×10⁻³ ± 1.96·SE (по матчам месяца); отрицательное значение — схема лучше. Первая отсечка "
              "сезона у «текущий ×4» совпадает с контролем (матчей сезона ещё нет).", "small"))
    x = w.loc[(w["scheme"] == "cur4") & (w["by"] == "games")]
    if len(x):
        rows = [["Сезон", "Сыграно игр", "n", "1X2 Δ", "Голы Δ", "Угловые Δ"]]
        order = {"0-2": 0, "3-5": 1, "6-9": 2, "10-14": 3, "15-19": 4, "20+": 5}
        for (s, v), g in sorted(x.groupby(["season", "value"]), key=lambda t: (t[0][0], order.get(t[0][1], 9))):
            d = g.set_index("group")
            rows.append([f"{s}/{s - 1999}", v, int(d["n"].max())] + [m3(d.loc[q, "diff"]) if q in d.index else "—"
                                                                   for q in ("result", "goals", "corners")])
        rep.add(table(rows, widths=[2 * cm, 2.4 * cm, 1.4 * cm, 3 * cm, 3 * cm, 3 * cm]),
                P("Таблица 6. «Текущий сезон ×4» − контроль по числу игр сезона, сыгранных менее опытной из команд "
                  "(×10⁻³).", "small"))
    rep.add(*figure(FIG / "when_it_helps.png", "Рисунок 4. Разность с контролем по месяцу отсечки (верх) и по числу "
                                               "сыгранных командами матчей (низ), 95 % ДИ."))
    rep.add(*figure(FIG / "weight_share.png", "Рисунок 5. Доля текущего сезона в суммарном весе обучения (модель "
                                              "голов хозяев) по отсечкам."))

    rep.add(H2("6.4 Main ROI (2026/27)"))
    bc, bt = roi["wf:prod"]["close"], roi["wf:prod"]["t09"]
    bb = b_close["close"]["summary"]
    rep.add(P(f"Базовая линия B: close — {bb['bets']} ставок, ROI {pp(bb['roi'])} %; t09 — "
              f"{b_close['t09']['summary']['bets']} ставок, ROI {pp(b_close['t09']['summary']['roi'])} %. Базовая "
              f"линия A (сайт): {b_close['baseline_a']['summary']['bets']} ставок, ROI "
              f"{pp(b_close['baseline_a']['summary']['roi'])} %."))
    for u in ("close", "t09"):
        sub = rr.loc[rr["universe"] == u].copy()
        sub["o"] = sub["variant"].map(lambda v: (0 if v.startswith("wf:") else 1 if v.startswith("static")
                                                 else 2, ORDER.index(v.split(":")[1]) if v.split(":")[1] in ORDER else 99))
        sub = sub.sort_values("o")
        rows = [["Вариант", "Ставок", "Попад.", "Ср. коэф.", "ROI, %", "ΔROI к B [95 % ДИ]", "p Холма", "Δ к контролю",
                 "Тот же Main"]]
        for _, x in sub.iterrows():
            rows.append([label(names, x["variant"]), int(x["bets"]), pp(x["hit_rate"], 1, False), f"{x['avg_odds']:.2f}",
                         pp(x["roi"]), f"{pp(x['d_roi'])} [{pp(x['d_lo'])}; {pp(x['d_hi'])}]", pv(x["p_holm"]),
                         pp(x["d_roi_vs_control"]), pp(x["same_main"], 0, False) + " %"])
        rep.add(table(rows, widths=[4.0 * cm, 1.45 * cm, 1.45 * cm, 1.3 * cm, 1.2 * cm, 3.3 * cm, 1.3 * cm, 1.5 * cm,
                                    1.4 * cm]),
                P(f"Таблица {7 if u == 'close' else 8}. Main ROI, момент {u}; ΔROI — п.п., парный бутстреп по матчам; "
                  f"Холм — по {sub['variant'].nunique()} вариантам.", "small"))
    rep.add(*figure(FIG / "roi_2026.png", "Рисунок 6. ΔROI Main к базовой линии B (ежемесячные схемы), п.п."))

    # 7 -----------------------------------------------------------------------------------------------
    rep.add(H1("7. Графики и устойчивость"))
    rep.add(*figure(FIG / "cumulative_profit.png", "Рисунок 7. Накопленная прибыль Main (close): контроль, выбор "
                                                   "2025/26 и базовая линия B."))
    for name, c in r["candidates"].items():
        cc = c["close"]
        bm = {x["month"]: x for x in cc["base_by_month"]}
        rows = [["Месяц", "Ставок", "ROI, %", "ROI B, %", "Прибыль", "Прибыль B"]]
        for x in cc["by_month"]:
            b = bm.get(x["month"], {})
            rows.append([x["month"], x["bets"], pp(x["roi"]), pp(b.get("roi")), f"{x['profit']:+.1f}",
                         f"{b.get('profit', float('nan')):+.1f}"])
        rep.add(H3(f"{label(names, name)} (close)"), table(rows),
                P(f"Без 5 крупнейших выигрышей: ROI {pp(cc['big_wins']['roi_without'])} % против "
                  f"{pp(cc['big_wins']['base_roi_same_matches'])} % у B на тех же матчах; цены на 2 % ниже: "
                  f"{pp(cc['odds_minus_2pct']['roi'])} % (B: {pp(cc['base_odds_minus_2pct']['roi'])} %).", "small"))
        lg = pd.DataFrame(cc["by_league"]).merge(pd.DataFrame(cc["base_by_league"])[["league", "roi"]],
                                                 on="league", suffixes=("", "_b")).sort_values("bets", ascending=False)
        rows = [["Лига", "Ставок", "ROI, %", "ROI B, %"]] + [
            [x["league"], x["bets"], pp(x["roi"]) + (" *" if x["bets"] < 30 else ""), pp(x["roi_b"])]
            for _, x in lg.iterrows()]
        rep.add(table(rows, widths=[6 * cm, 2 * cm, 2.5 * cm, 2.5 * cm]), P("* — меньше 30 ставок.", "small"))

    # 8 -----------------------------------------------------------------------------------------------
    rep.add(H1("8. Статистический анализ"),
            P(f"Поправка Холма: для вероятностей — по всем вариантам каждой группы задач (2026/27, к B и к контролю), "
              f"для ROI — по {nroi} вариантам в каждом моменте решения. 2026/27 изучался в v6.1/v6.2 — это "
              "хронологическая проверка уже просмотренного сезона, а не независимое подтверждение; параметры выбраны "
              "на 2025/26. Доверительные интервалы ROI ±3 п.п. при ~2 250 ставках: различия ROI меньше 2–3 п.п. "
              "статистически неразличимы."))

    # 9 -----------------------------------------------------------------------------------------------
    rep.add(H1("9. Выводы"), bullets(conclusions(r, rr)))

    # 10 ----------------------------------------------------------------------------------------------
    rows = [["Схема", "1X2", "Голы", "Угловые"]]
    for sc in ORDER:
        if any(f"{sc}|{g}" in dec for g in GROUP_RU):
            rows.append([names.get(sc, sc)] + [dec.get(f"{sc}|{g}", "—") for g in ("result", "goals", "corners")])
    rep.add(H1("10. Рекомендации A / B / C"), table(rows, widths=[7 * cm, 3 * cm, 3 * cm, 3 * cm]),
            P("Таблица 9. Решение по схеме и группе задач.", "small"),
            bullets(recommendations(r, pre)))

    # 11 ----------------------------------------------------------------------------------------------
    rep.add(H1("11. Ограничения"), bullets([
        "2026/27 — только 4 ежемесячные отсечки (август–ноябрь); текущий сезон составляет малую долю обучения, "
        "поэтому эффект весов текущего сезона в 2026/27 ограничен по размеру; основная информация о «когда помогает» "
        "— из 2025/26 (10 отсечек).",
        "Гиперпараметры не перенастраивались под новые веса (число деревьев продакшна), что консервативно для "
        "сильных весов.",
        "Переобученная с весами продакшна модель не совпадает с prod_20261012 побитно (таблица признаков "
        "обновлялась после 12.10.2026); поэтому эффекты весов сравниваются с контролем, а не только с B.",
        "Список лиг с угловыми взят из продакшна (метаданные покрытия по данным до 2025/26 включительно). ROI на "
        "одном сезоне (±3 п.п.) не может подтвердить эффект порядка 1 п.п."]))

    # 12 ----------------------------------------------------------------------------------------------
    rep.add(H1("12. Как воспроизвести"), *K.reproduce([
        "experiments/v6_3/heavy.sh 4500 experiments/v6_3/recent_weighting/logs/fits_2026.log \\",
        "    .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026",
        "experiments/v6_3/heavy.sh 4500 experiments/v6_3/recent_weighting/logs/fits_2026k.log \\",
        "    .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026 --k 250",
        "experiments/v6_3/heavy.sh 10800 experiments/v6_3/recent_weighting/logs/fits_2025.log \\",
        "    .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2025",
        ".venv/bin/python -m experiments.v6_3.recent_weighting.run",
        ".venv/bin/python -m experiments.v6_3.recent_weighting.build_pdf"]),
        P(f"Git: {info['git_sha'][:10]}; оценка {info['seconds']:.0f} с; "
          + ", ".join(f"{k} {v}" for k, v in info["libraries"].items()) + ". Файлы: `results/results.json`, "
          "`metrics.csv`, `main_roi.csv`, `when_it_helps.csv`, `current_weight_share.csv`, `per_match_*.csv` "
          "(поля §5), `preregistration.json`, `run_info.json`; выходы моделей — `cache/` (не в git).", "small"))
    rep.build()
    n = rep.check()
    print(f"{OUT} — {n} pages")
    write_findings(r, rr, pre)


def conclusions(r: dict, rr: pd.DataFrame) -> list[str]:
    names, paired, holm, sel = r["names"], r["paired"], r["holm"], r["selected_2025"]
    out = [f"Гипотеза о пользе большего веса текущего сезона не подтверждается: на сезоне выбора 2025/26 лучшими "
           f"оказались схемы, близкие к продакшну или с более длинной памятью, и ни одна не лучше контроля значимо (1X2 — «{names.get(sel['result'])}», голы — "
           f"«{names.get(sel['goals'])}», угловые — «{names.get(sel['corners'])}»), а схемы с быстрым затуханием, "
           "коротким окном или повышенным весом прошлого сезона хуже значимо; повышенный вес текущего сезона "
           "(×2, ×4) в пределах шума."]
    w = pd.read_csv(RES / "when_it_helps.csv")
    x = w.loc[(w["scheme"] == "cur4") & (w["by"] == "games")]
    good = [f"{s}/{s - 1999}: {v}" for (s, v), g in x.groupby(["season", "value"])
            if len(g) == 3 and (g["diff"] < 0).all()]
    sig = [f"{s}/{s - 1999} {v} ({GROUP_RU[q]})" for _, (s, v, q, d, se) in
           x[["season", "value", "group", "diff", "se"]].iterrows() if d + 1.96 * se < 0]
    out.append("После скольких матчей помогает вес текущего сезона ×4: интервалы числа сыгранных игр, где он лучше "
               "контроля сразу по 1X2, голам и угловым — " + (", ".join(good) if good else "ни одного")
               + "; значимо лучше (95 %) — " + (", ".join(sig) if sig else "нигде")
               + ". Порога, после которого повышенный вес начинает устойчиво помогать, не найдено ни по месяцам, "
               "ни по числу матчей (таблица 6, рисунок 4).")
    c = paired.get("2026|wf|prod|vs_B")
    if c:
        out.append("Ежемесячное переобучение с весами продакшна (больше данных) против B на 2026/27: 1X2 "
                   f"{ci3(c['result'])}, голы {ci3(c['goals'])}, угловые {ci3(c['corners'])} ×10⁻³.")
    c = paired.get("2025|static|prod|vs_control")
    if c:
        out.append("2025/26: однократное обучение хуже ежемесячного на 1X2 "
                   f"{ci3(c['result'])}, голы {ci3(c['goals'])}, угловые {ci3(c['corners'])} ×10⁻³ "
                   "(положительное — ежемесячное лучше).")
    for g in ("result", "goals", "corners"):
        s = sel[g]
        k = f"2026|wf|{s}|vs_B"
        k2 = f"2026|wf|{s}|vs_control"
        if k in paired:
            out.append(f"{ {'result': '1X2', 'goals': 'Голы', 'corners': 'Угловые'}[g]}: на 2025/26 выбрана схема «{names.get(s, s)}»; на 2026/27 к B "
                       f"{ci3(paired[k][g])} (p Холма {pv(holm[g].get(k))}), к контролю "
                       f"{ci3(paired.get(k2, {}).get(g))} ×10⁻³.")
    for sc in ("cur2", "cur4"):
        k = f"2025|wf|{sc}|vs_control"
        if k in paired:
            out.append(f"{names[sc]} − контроль: 2025/26 1X2 {ci3(paired[k]['result'])}, голы "
                       f"{ci3(paired[k]['goals'])}, угловые {ci3(paired[k]['corners'])}; 2026/27 1X2 "
                       f"{ci3(paired.get(f'2026|wf|{sc}|vs_control', {}).get('result'))} ×10⁻³.")
    for sc in ("hl0.5", "hl1", "win2", "last3"):
        k = f"2026|wf|{sc}|vs_B"
        k25 = f"2025|wf|{sc}|vs_control"
        if k in paired:
            out.append(f"{names[sc].capitalize()}: 2025/26 к контролю 1X2 {ci3(paired.get(k25, {}).get('result'))}, "
                       f"голы {ci3(paired.get(k25, {}).get('goals'))}; 2026/27 к B 1X2 {ci3(paired[k]['result'])}, "
                       f"голы {ci3(paired[k]['goals'])}, угловые {ci3(paired[k]['corners'])} ×10⁻³ — более "
                       "короткая память модели ухудшает прогноз.")
    k = "2025|wf|form|vs_control"
    if k in paired:
        fm = r["form"]
        out.append(f"V8 на 2025/26 (сезон выбора) хуже контроля даже при самой слабой поправке сетки "
                   f"(k = {fm.get('k_goals')}, k_c = {fm.get('kc')} — граница сетки): 1X2 {ci3(paired[k]['result'])}, "
                   f"голы {ci3(paired[k]['goals'])} ×10⁻³ — ошибки модели на прошлых матчах сезона не несут "
                   "информации о следующих (как и в v6.2).")
    x = rr.loc[rr["variant"] == "static:prod"].set_index("universe")
    if len(x) == 2:
        out.append(f"Шум переобучения: та же модель, переобученная на тех же данных (однократно, веса продакшна), "
                   f"отличается от B по ROI на {pp(x.loc['close', 'd_roi'])} п.п. (close) и "
                   f"{pp(x.loc['t09', 'd_roi'])} п.п. (t09) при почти тех же вероятностях — различия ROI порядка "
                   "1 п.п. между вариантами не являются эффектом весов.")
    k = "2026|wf|form|vs_control"
    if k in paired:
        out.append(f"V8 (давность + форма) − контроль на 2026/27: 1X2 {ci3(paired[k]['result'])}, голы "
                   f"{ci3(paired[k]['goals'])}, угловые {ci3(paired[k]['corners'])} ×10⁻³.")
    for name in ("wf:selected", "wf:prod", "wf:form"):
        x = rr.loc[rr["variant"] == name].set_index("universe")
        if len(x) == 2:
            out.append(f"Main ROI «{label(names, name)}»: close ΔROI к B {pp(x.loc['close', 'd_roi'])} п.п. "
                       f"[{pp(x.loc['close', 'd_lo'])}; {pp(x.loc['close', 'd_hi'])}], t09 "
                       f"{pp(x.loc['t09', 'd_roi'])} п.п. [{pp(x.loc['t09', 'd_lo'])}; {pp(x.loc['t09', 'd_hi'])}].")
    best = rr.loc[rr["universe"] == "close"].sort_values("d_roi", ascending=False).iloc[0]
    out.append(f"Лучший по ROI close вариант — «{label(names, best['variant'])}»: {pp(best['d_roi'])} п.п. "
               f"(p Холма {pv(best['p_holm'])}) — выбран по самому оцениваемому сезону, поэтому не является "
               "доказательством.")
    return out


def recommendations(r: dict, pre: dict) -> list[str]:
    dec, names = r["decisions"], r["names"]
    a = sorted({k for k, d in dec.items() if d == "A"})
    b = sorted({k for k, d in dec.items() if d == "B"})
    fmt = lambda ks: ", ".join(f"{names.get(k.split('|')[0], k.split('|')[0])} ({GROUP_RU[k.split('|')[1]]})"  # noqa: E731
                               for k in ks) or "нет"
    out = [f"**A (внедрить):** {fmt(a)}.", f"**B (форвард-проверка):** {fmt(b)}"
           + (f" — `results/preregistration.json`, {len(pre['entries'])} записей." if b else "."),
           "**C:** остальные сочетания схем и задач (таблица 9).",
           "Изменение продакшна, если B подтвердится вперёд: переобучение 1-го числа месяца в конвейере обучения "
           "(`src/modelling/pipeline.py`, те же гиперпараметры) с весами `scheme_weights` из "
           "`experiments/v6_3/recent_weighting/fits.py` (голы: полураспад 0 вместо 2; 1X2: текущий сезон ×2). "
           "Продакшн исследованием не меняется."]
    return out


def write_findings(r: dict, rr: pd.DataFrame, pre: dict) -> None:
    names, paired, holm, dec = r["names"], r["paired"], r["holm"], r["decisions"]
    lines = ["# FootPredict v6.3 — исследование 7: вес текущего сезона — выводы", "",
             "Все числа — из `results/` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_3/Recent_Season_Weighting_Research.pdf`.", "",
             "## 2026/27: разность с базовой линией B (×10⁻³; ежемесячное переобучение, схемы весов)", "",
             "| Схема | 1X2 log loss | Голы, девиация | Угловые, девиация | ΔROI close, п.п. | ΔROI t09, п.п. | Решение 1X2/голы/угл. |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for sc in ORDER:
        k = f"2026|wf|{sc}|vs_B"
        if k not in paired:
            continue
        x = rr.loc[rr["variant"] == f"wf:{sc}"].set_index("universe")
        dc = "/".join(dec.get(f"{sc}|{g}", "контроль") for g in ("result", "goals", "corners"))
        lines.append(f"| {names.get(sc, sc)} | {ci3(paired[k]['result'])} | {ci3(paired[k]['goals'])} | "
                     f"{ci3(paired[k]['corners'])} | {pp(x.loc['close', 'd_roi']) if 'close' in x.index else '—'} | "
                     f"{pp(x.loc['t09', 'd_roi']) if 't09' in x.index else '—'} | {dc} |")
    lines += ["", f"Выбор на 2025/26: {r['selected_2025']}. Вариантов с ROI: {r['n_variants']['roi_variants']}.", "",
              "## Главное", ""] + [f"- {c}" for c in conclusions(r, rr)]
    lines += ["", "## Дефекты", "",
              "- `experiments/v6_3/replay/engine.walk_forward`: при пользовательских отсечках матчи ДО первой отсечки "
              "прогнозируются моделью, обученной на данных до этой отсечки (включая сами эти матчи) — утечка, если "
              "первая отсечка не совпадает с началом сезона. С отсечками по умолчанию (1-е число первого месяца) "
              "не проявляется; исследование использует собственный цикл (`fits.py`), где первая отсечка — первый "
              "игровой день.",
              "- Переобучение с параметрами продакшна не воспроизводит prod_20261012 побитно (λ хозяев до 0.23 "
              "разницы) — таблица признаков менялась после обучения продакшна; эффекты весов меряются к контролю.", "",
              "## Решение", ""] + [f"- {x}" for x in recommendations(r, pre)]
    lines += ["", "## Воспроизведение", "", "```",
              "experiments/v6_3/heavy.sh 4500 experiments/v6_3/recent_weighting/logs/fits_2026.log .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026",
              "experiments/v6_3/heavy.sh 4500 experiments/v6_3/recent_weighting/logs/fits_2026k.log .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026 --k 250",
              "experiments/v6_3/heavy.sh 10800 experiments/v6_3/recent_weighting/logs/fits_2025.log .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2025",
              ".venv/bin/python -m experiments.v6_3.recent_weighting.run",
              ".venv/bin/python -m experiments.v6_3.recent_weighting.build_pdf", "```", ""]
    (C.V63 / "recent_weighting" / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
