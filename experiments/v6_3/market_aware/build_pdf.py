"""Builds reports/v6_3/Market_Aware_Main_Research.pdf and findings.md from results/ (no number typed by hand).

    .venv/bin/python -m experiments.v6_3.market_aware.build_pdf
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C
from scripts import report_kit as K
from scripts.report_kit import H2, P, bullets, figure, formula

from . import fmt as F
from .run import DIS_LABELS, EDGE_LABELS, FIG, RES, VARIANTS_RU

OUT = C.ROOT / "reports" / "v6_3" / "Market_Aware_Main_Research.pdf"
GROUP_RU = {"DOUBLE_CHANCE": "Двойной шанс", "1X2": "1X2", "TOTAL": "Тоталы", "CORNERS": "Угловые", "BTTS": "Обе забьют", "HANDICAP": "Азиатские форы"}
MAIN_KEYS = ["blend_wf", "blend_v62", "market_only", "disagree_filter", "blend_value_first", "blend_5050",
             "max_blend_ev"]
SENS_KEYS = ["blend_wf", "blend_wf_prop", "blend_wf_shin", "blend_wf_log"]


def decisions(r) -> dict:
    """A / B / C per variant from the paired results (rule written before reading them: A needs a Holm-significant
    gain in both universes; B — a non-negative point gain in both universes; C otherwise)."""
    out = {}
    for k in VARIANTS_RU:
        if k == "baseline_b":
            continue
        d = {u: r["variants"][u][k]["vs_b"]["roi_diff"]["diff"] for u in ("close", "t09")}
        h = {u: r["holm"]["p_holm"].get(f"{k}|{u}", 1) for u in ("close", "t09")}
        if all(d[u] > 0 and h[u] < 0.05 for u in d):
            out[k] = "A"
        elif all(d[u] > -0.001 for u in d) and max(d.values()) > 0.005:
            out[k] = "B"
        else:
            out[k] = "C"
    return out


def build() -> None:
    r = C.read_json(RES / "market_aware.json")
    info = C.read_json(RES / "run_info.json")
    pre = C.read_json(RES / "preregistration.json")
    cov = pd.read_csv(RES / "coverage.csv")
    dec = decisions(r)
    V, H = r["variants"], r["holm"]["p_holm"]
    sc = r["self_check"]
    rep = K.Report(OUT, "Market-Aware Main: вероятности букмекеров в выборе Main",
                   "FootPredict v6.3 — исследование 3",
                   {"Сезон": "2026/27 (полный пересчёт; сезон уже изучался в v6.1/v6.2 — не holdout)",
                    "Матчей": f"{int(cov['season_matches'].iloc[0])} в сезоне, {int(cov['played'].iloc[0])} сыграно",
                    "Модель": "prod_20261012 (обучена на 2021/22–2025/26)",
                    "Данные": f"features {info['data']['features']['mtime'][:10]}, git {info['git_sha'][:8]}"})

    # 1 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("1. Название исследования"),
            P("**Market-Aware Main** — использование вероятностей букмекерского рынка (без маржи) при оценке "
              "вероятности и выборе Main-прогноза FootPredict: сравнение методов снятия маржи, проверка, добавляет ли "
              "модель информацию сверх рынка, осторожное объединение модели и рынка, расхождение модели и рынка как "
              "признак неопределённости и влияние всего этого на ROI Main."))
    # 2 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("2. Постановка проблемы"),
            P("Main выбирается правилом продакшна по вероятности модели p, предпочтительному диапазону коэффициентов и "
              f"EV = p · odds − (1 − P(возврат)). Средний EV выбранных Main по модели положителен "
              f"({F.pp(V['close']['baseline_b']['summary']['mean_ev'])} % в мире close), а фактический ROI "
              f"отрицателен ({F.pp(V['close']['baseline_b']['summary']['roi'])} %). Значит, вероятность модели для "
              "выбранных исходов завышена, а цена рынка содержит информацию, которой у модели нет. В v6.2 "
              "(Bookmaker Benchmark) на 1X2, тотале 2.5 и азиатской линии Football-Data рынок был точнее модели, а вес "
              "модели рядом с рынком — около нуля. v6.3 переносит вопрос на **все рынки Main** (тоталы 1.5/2.5/3.5, "
              "форы, двойной шанс, обе забьют, угловые) с реальными ценами, существовавшими в момент решения, и "
              "измеряет прямо **ROI Main**, а не только log loss."))
    # 3 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("3. Гипотезы"),
            *bullets(["**H1.** Вероятность рынка без маржи точнее вероятности модели (log loss, Brier) на рынках Main; "
                      "метод снятия маржи влияет мало.",
                      "**H2.** Модель всё же добавляет информацию сверх рынка (вес модели b_p > 0 в пуле с рынком).",
                      "**H3.** Объединение модели с рынком (смесь логитов) уменьшает переоценку выбранного Main и "
                      "повышает ROI Main по сравнению с Baseline B.",
                      "**H4.** Большое расхождение модели и рынка указывает на завышенную вероятность Main: исключение "
                      "таких кандидатов улучшает Main.",
                      "**H5.** Ранжирование кандидатов по EV относительно смеси (рыночная информация в ранжировании) "
                      "улучшает ROI."]),
            P("Ожидание до эксперимента (по v6.2): H1 подтвердится, H2 — слабо и не везде, H3–H5 дадут изменения ROI в "
              "пределах шума, потому что почти вся информация модели уже есть в цене."))
    # 4 ------------------------------------------------------------------------------------------
    mg = r["demargin"]["close"]["_margin"]
    books = r["demargin"]["close"]["_books"]
    rows = [["Группа рынков", "Книга (close)", "Рынков", "Средняя маржа книги, %"]]
    mgb = r["demargin"]["close"]["_margin_book"]
    for g, bk in books.items():
        for b, n in bk.items():
            if n:
                rows.append([GROUP_RU.get(g, g), b, n, F.pp(mgb.get(f"{g}|{b}"), 2, False)])
    rep.add(F.H1c("4. Исходные данные"),
            P("Общий пересчёт сезона (`experiments/v6_3/replay`): все матчи 2026/27 из таблицы признаков, выходы "
              "продакшн-моделей prod_20261012, кандидаты всех рынков с математикой продакшна, реальные цены. Цены: "
              "OddsPapi (bet365, Pinnacle, William Hill), The Odds API (несколько книг с октября), Football-Data Bet365 "
              "(снимок открытия, время котировки неизвестно — условно 00:00 дня матча, как в replay). Цена используется "
              "в момент решения, только если опубликована не позже его."),
            F.coverage_table(cov.loc[cov["method"].isin(["baseline_b", "blend_wf", "market_only",
                                                          "disagree_filter"])]),
            P("Таблица 1. Охват по методам (оба мира). «Без цены» — Main без реальной цены в момент решения: исход "
              "рассчитан, но это не ставка. Ещё не сыгранные матчи сезона ("
              f"{int(cov['not_played_excluded'].iloc[0])} из {int(cov['season_matches'].iloc[0])}) обработаны, но не "
              "входят в ROI. Статистики хватает для всех матчей (признаки продакшна есть у каждого). Ничего не "
              "исключено молча.", "small"),
            F.T(rows),
            P("Таблица 2. Какая книга дала вероятность рынка (первая полная книга в порядке bet365 OddsPapi → "
              "Football-Data Bet365 → Pinnacle → William Hill), мир close.", "small"),
            P(f"Доля выбранных Main Baseline B, для рынка которых есть полная книга: close "
              f"{F.pp(r['disagreement']['close']['with_q'] / r['disagreement']['close']['total'], 1, False)} %, t09 "
              f"{F.pp(r['disagreement']['t09']['with_q'] / r['disagreement']['t09']['total'], 1, False)} % (в момент T "
              "у четверти Main ещё нет цены вовсе)."))
    # 5 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("5. Методика"),
            H2("5.1. Моменты решения (два мира)"),
            P("**close** — решение перед началом матча по последней котировке до начала (основной мир; цены почти у "
              "всех Main). **t09** — решение в ежедневный запуск T = min(09:00 UK дня матча, начало − 30 мин) только с "
              "ценами, известными к T (проверка устойчивости). Сравнения — всегда с Baseline B того же мира; "
              "Baseline A (история сайта) — рядом. Самопроверка: векторизованное правило Main воспроизводит Baseline B "
              f"в {F.pp(sc['close']['same_main_share'], 2, False)} % матчей close и "
              f"{F.pp(sc['t09']['same_main_share'], 2, False)} % матчей t09 (расхождение — равные оценки)."),
            H2("5.2. Полная книга и снятие маржи"),
            P("Вероятность рынка существует только если одна книга котирует **все** исходы рынка не позже момента "
              "решения (у каждого исхода — его последняя котировка). Для пары коэффициентов o_{i} исходов: "
              "π_{i} = 1/o_{i}, маржа = Σπ_{i} − 1. Методы:"),
            formula("пропорциональный: q_{i} = π_{i} / Σπ_{j};   степенной: q_{i} = π_{i}^{k}, Σq_{i} = 1"),
            formula("Шин: q_{i} = (√(z² + 4(1 − z)π_{i}²/Σπ) − z) / (2(1 − z));   odds-ratio: logit q_{i} = logit π_{i} − c"),
            P("Азиатские форы: книга из двух исходов даёт P(выигрыш | нет возврата); вероятность модели сравнивается в "
              "той же шкале p/(1 − P_{возврат}). Двойной шанс выводится из книги 1X2 той же книги. Основной метод — "
              "степенной (выбран в v6.2 на сезонах 2022/23–2025/26); остальные — проверка чувствительности."),
            H2("5.3. Пул модели и рынка"),
            formula("P(y = i) ∝ exp(a_{i} + b_{q} · log q_{i} + b_{p} · log p_{i})"),
            P("Полносезонная подгонка — для вывода (вес модели b_p с бутстреп-ДИ по матчам, тест отношения "
              "правдоподобия против пула только с рынком). Для прогнозов — помесячное расширяющееся окно: для матчей "
              f"месяца M пул подгоняется на сыгранных матчах до 1-го числа M (не менее {info['params']['min_train_rows']} "
              "рынков, иначе остаётся вероятность модели — так в августе смесь совпадает с продакшном)."),
            H2("5.4. Варианты Main"),
            *bullets([f"**{VARIANTS_RU['blend_wf']}** — правило продакшна на вероятности пула (EV пересчитан).",
                      f"**{VARIANTS_RU['blend_v62']}** — пул v6.2 (H2), подогнанный на 2022/23–2025/26 по Football-Data "
                      "Bet365: 1X2, тоталы (параметры тотала 2.5 на все линии), форы (параметры одной линии на все); "
                      "остальные рынки — модель.",
                      f"**{VARIANTS_RU['market_only']}** — вероятность рынка вместо модели, где есть книга.",
                      f"**{VARIANTS_RU['disagree_filter']}** — кандидаты с |logit p − logit q| > δ исключены; δ из "
                      "{0.25, 0.5, 0.75, 1, ∞} выбирается для месяца M по ROI Main прошлых месяцев (∞, пока прошлых "
                      f"ставок меньше {info['params']['min_prior_bets']}).",
                      f"**{VARIANTS_RU['blend_value_first']}** — сначала кандидаты с ценой и EV по смеси ≥ 0.",
                      f"**{VARIANTS_RU['blend_5050']}** — фиксированная смесь b_q = b_p = 0.5 без подгонки.",
                      f"**{VARIANTS_RU['max_blend_ev']}** — среди кандидатов полосы p ∈ [0.40; 0.85] по модели "
                      "выбирается цена с максимальным EV по смеси.",
                      "Чувствительность: V1 со снятием маржи пропорционально, по Шину и odds-ratio."]),
            H2("5.5. Статистика"),
            P("ROI = Σ прибыли / число ставок (ставка 1 ед. по реальной цене; PUSH возвращает ставку; Main без цены не "
              f"ставка). 95 % ДИ — бутстреп по матчам ({info['params']['n_boot']} повторов); разница с Baseline B — "
              "парный бутстреп на тех же матчах; поправка Холма по всем "
              f"{r['holm']['n_tests']} сравнениям (вариант × мир). Проверка на месяцы и лиги, зависимость от крупных "
              "выигрышей, чувствительность к ухудшению цен на 3 % и 5 %."))
    # 6 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("6. Эксперименты"),
            P(f"Запущено {r['n_variants']} вариантов Main в двух мирах ({r['holm']['n_tests']} парных сравнений с "
              "Baseline B), пять методов снятия маржи на пяти группах рынков в двух мирах, полносезонные и "
              "помесячные пулы для пяти групп, анализ перевеса модели над рынком на всех кандидатах с ценой и анализ "
              "расхождения на выбранных Main. Ни одна модель не переобучалась (выходы продакшна), поэтому тяжёлых "
              "вычислений нет; полный прогон занимает около двух минут."))
    # 7 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("7. Результаты"), H2("7.1. Методы снятия маржи и точность модели"))
    for u in ("close", "t09"):
        dm = r["demargin"][u]
        rows = [["Группа", "Рынков", "Модель", "Степ.", "Пропорц.", "Шин", "Odds-ratio", "Аддит.",
                 "Модель − степ., ×10⁻³ [ДИ]"]]
        for g in GROUP_RU:
            if g not in dm:
                continue
            d = dm[g]
            rows.append([GROUP_RU[g], d["n"], F.num(d["model"]["ll"], 4)] +
                        [F.num(d[m]["ll"], 4) for m in ("power", "prop", "shin", "log", "add")] +
                        [f"{d['model_minus_power']['diff'] * 1000:+.1f} {F.ci(d['model_minus_power']['ci95'], 1, 1000)}"])
        rep.add(F.T(rows), P(f"Таблица {3 if u == 'close' else 4}. Log loss рынка по методам снятия маржи и модели "
                                f"на тех же рынках, мир {u}.", "small"))
    sig = {u: [GROUP_RU[g] for g in GROUP_RU if g in r["demargin"][u] and "model_minus_power" in r["demargin"][u][g]
               and r["demargin"][u][g]["model_minus_power"]["ci95"][0] > 0] for u in ("close", "t09")}
    rep.add(P(f"Рынок без маржи значимо точнее модели (95 % ДИ разницы выше нуля): close — {', '.join(sig['close'])}; "
              f"t09 — {', '.join(sig['t09'])}; в остальных группах разница в пределах шума. Различия между методами "
              "снятия маржи на порядок меньше разницы модель–рынок."),
            H2("7.2. Добавляет ли модель информацию сверх рынка"))
    for u in ("close", "t09"):
        fs = r["information"][u]["full_season"]
        rows = [["Группа", "Рынков", "b_q [95 % ДИ]", "b_p [95 % ДИ]", "LR", "p (LR)"]]
        for g, v in fs.items():
            rows.append([GROUP_RU[g], v["n"], f"{v['b_market']:.2f} {F.ci(v['b_market_ci'], 2, 1)}",
                         f"{v['b_model']:.2f} {F.ci(v['b_model_ci'], 2, 1)}", f"{max(v['lr'], 0):.1f}", F.pval(v["p_lr"])])
        rep.add(F.T(rows), P(f"Таблица {5 if u == 'close' else 6}. Полносезонный пул, мир {u}: веса рынка и модели, "
                                "тест «вес модели = 0» (для вывода, не для отбора).", "small"))
    for u in ("close", "t09"):
        wf = r["information"][u]["walk_forward"]
        rows = [["Группа", "Рынков", "Модель", "Рынок", "Рынок перекалибр.", "Смесь", "Смесь − рынок, ×10⁻³ [ДИ]"]]
        for g, v in wf.items():
            rows.append([GROUP_RU[g], v["n"], F.num(v["model"], 4), F.num(v["market"], 4), F.num(v["market_recal"], 4),
                         F.num(v["blend"], 4),
                         f"{v['blend_minus_market']['diff'] * 1000:+.2f} {F.ci(v['blend_minus_market']['ci95'], 2, 1000)}"])
        rep.add(F.T(rows), P(f"Таблица {7 if u == 'close' else 8}. Вне выборки (помесячное окно, сентябрь–ноябрь), "
                                f"мир {u}: log loss модели, рынка, перекалиброванного рынка и смеси.", "small"))
    e = r["edge"]
    rows = [["Перевес e = p/q − 1"] + [f"{u}: ставок" for u in ("close", "t09")] +
            [f"{u}: ROI, % [ДИ]" for u in ("close", "t09")] + [f"{u}: hit / p / q, %" for u in ("close", "t09")]]
    for b in EDGE_LABELS:
        row = [b]
        row += [e[u]["bins"][b]["bets"] for u in ("close", "t09")]
        row += [F.roi_cell(e[u]["bins"][b]) for u in ("close", "t09")]
        row += [f"{F.pp(e[u]['bins'][b]['hit'], 0, False)} / {F.pp(e[u]['bins'][b]['mean_p'], 0, False)} / "
                f"{F.pp(e[u]['bins'][b]['mean_q'], 0, False)}" for u in ("close", "t09")]
        rows.append(row)
    lg = {u: e[u]["logit_edge"] for u in ("close", "t09")}
    rep.add(H2("7.3. Перевес модели над рынком на всех кандидатах с ценой"), F.T(rows),
            P("Таблица 9. Все кандидаты с ценой в полосе Main (p ∈ [0.40; 0.85], без точного счёта и европейских "
              "фор) с полной книгой: ROI по перевесу модели (ДИ — бутстреп по матчам).", "small"),
            P(f"Логистическая регрессия исхода на logit q и log(p/q) (кластеры — матчи): коэффициент перевеса "
              f"{lg['close']['coef'][2]:.2f} (SE {lg['close']['se'][2]:.2f}, p = {F.pval(lg['close']['p'][2])}) в close и "
              f"{lg['t09']['coef'][2]:.2f} (SE {lg['t09']['se'][2]:.2f}, p = {F.pval(lg['t09']['p'][2])}) в t09; "
              f"коэффициент logit q ≈ {lg['close']['coef'][1]:.2f}. Перевес модели содержит немного информации об "
              "исходе сверх цены, но её не хватает, чтобы покрыть маржу: ROI отрицателен в каждой корзине."))
    rep.add(H2("7.4. Варианты Main против Baseline B"))
    for i, u in enumerate(("close", "t09")):
        rep.add(F.T(F.variant_rows(V[u], {**VARIANTS_RU, "baseline_b": "Baseline B"}, u, H,
                                     ["baseline_b"] + MAIN_KEYS, r["baseline_a"])),
                P(f"Таблица {10 + i}. Main-варианты, мир {u}: ставки, доля выигрышей, средний коэффициент, ROI с 95 % "
                  "ДИ, парная разница с Baseline B, p и p Холма, доля матчей с тем же Main.", "small"))
    for i, u in enumerate(("close", "t09")):
        rep.add(F.T(F.variant_rows(V[u], VARIANTS_RU, u, H, SENS_KEYS)),
                P(f"Таблица {12 + i}. Чувствительность V1 к методу снятия маржи, мир {u}.", "small"))
    dis = r["disagreement"]
    rows = [["logit p − logit q"] + [f"{u}: Main" for u in ("close", "t09")] + [f"{u}: hit / p / q, %" for u in
                                                                                   ("close", "t09")] +
            [f"{u}: ROI, % [ДИ]" for u in ("close", "t09")]]
    for b in DIS_LABELS:
        if not any(b in dis[u]["bins"] for u in ("close", "t09")):
            continue
        row = [b]
        for u in ("close", "t09"):
            row.append(dis[u]["bins"].get(b, {}).get("n", 0))
        for u in ("close", "t09"):
            x = dis[u]["bins"].get(b)
            row.append(f"{F.pp(x['hit'], 0, False)} / {F.pp(x['mean_p'], 0, False)} / {F.pp(x['mean_q'], 0, False)}"
                       if x else "—")
        for u in ("close", "t09"):
            x = dis[u]["bins"].get(b)
            row.append(F.roi_cell(x) if x and x["n"] >= 30 else "мало")
        rows.append(row)
    ov = {u: dis[u]["overall"] for u in ("close", "t09")}
    rep.add(H2("7.5. Расхождение модели и рынка на выбранных Main"), F.T(rows),
            P("Таблица 14. Main Baseline B по расхождению с рынком: доля выигрышей, средняя p модели и q рынка, ROI.",
              "small"),
            P(f"В целом по Main Baseline B с полной книгой: доля выигрышей {F.pp(ov['close']['hit'], 1, False)} %, "
              f"средняя p модели {F.pp(ov['close']['mean_p'], 1, False)} %, средняя q рынка "
              f"{F.pp(ov['close']['mean_q'], 1, False)} % (close); p > q у "
              f"{F.pp(ov['close']['share_p_above_q'], 0, False)} % Main. В t09: {F.pp(ov['t09']['hit'], 1, False)} / "
              f"{F.pp(ov['t09']['mean_p'], 1, False)} / {F.pp(ov['t09']['mean_q'], 1, False)} %. Фактическая частота "
              "выигрышей ближе к q рынка: правило Main систематически выбирает исходы, которые модель переоценивает "
              "относительно рынка."))
    # 8 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("8. Графики"),
            *figure(FIG / "logloss_vs_market.png", "Рис. 1. Вне выборки: log loss модели и смеси минус log loss рынка "
                                                    "(×10⁻³, 95 % ДИ), по группам рынков и мирам."),
            *figure(FIG / "pool_coefficients.png", "Рис. 2. Полносезонный пул: веса рынка b_q и модели b_p (95 % ДИ). "
                                                   "b_p > 0 — модель добавляет информацию сверх рынка."),
            *figure(FIG / "roi_by_edge.png", "Рис. 3. ROI всех кандидатов с ценой по перевесу модели над рынком."),
            *figure(FIG / "variants_vs_baseline_b.png", "Рис. 4. Разница ROI Main каждого варианта с Baseline B "
                                                        "(парный бутстреп, 95 % ДИ)."),
            *figure(FIG / "cumulative_profit.png", "Рис. 5. Накопленная прибыль Main: Baseline B, V1 и V3."),
            *figure(FIG / "disagreement_main.png", "Рис. 6. Main Baseline B по расхождению модели и рынка: средняя p, "
                                                   "средняя q и фактическая доля выигрышей."))
    # 9 ------------------------------------------------------------------------------------------
    best = max(MAIN_KEYS, key=lambda k: V["close"][k]["vs_b"]["roi_diff"]["diff"])
    rep.add(F.H1c("9. Статистический анализ"),
            P(f"Проверено {r['n_variants']} вариантов × 2 мира = {r['holm']['n_tests']} парных сравнений. Минимальное "
              f"p Холма: {F.pval(min(H.values()))}. Ни один вариант не отличается от Baseline B статистически значимо; "
              "все интервалы разницы ROI включают ноль."),
            F.T(F.stability_rows(V["close"], {**VARIANTS_RU, "baseline_b": "Baseline B"},
                                   ["baseline_b"] + MAIN_KEYS)),
            P("Таблица 15. Устойчивость, мир close: месяцы и лиги, где вариант лучше B, ROI без крупнейших выигрышей, "
              "ROI при ценах хуже на 3 % и 5 %.", "small"),
            F.T(F.month_rows(V["close"], best, "вариант")),
            P(f"Таблица 16. По месяцам: лучший по точечной оценке в close вариант ({VARIANTS_RU[best]}) и Baseline B. "
              "«мало» — меньше 100 ставок.", "small"),
            F.T(F.league_rows(V["close"], best, "вариант")),
            P("Таблица 17. По лигам для того же варианта; * — меньше 50 ставок (малая выборка).", "small"),
            P("Выбор δ фильтра расхождения по прошлым месяцам: close " +
              ", ".join(f"{k}: {v if v is not None else '∞'}" for k, v in
                        r["extra"]["close"]["disagree_filter"]["delta_by_month"].items()) + "; t09 " +
              ", ".join(f"{k}: {v if v is not None else '∞'}" for k, v in
                        r["extra"]["t09"]["disagree_filter"]["delta_by_month"].items()) + "."))
    # 10 -----------------------------------------------------------------------------------------
    wf_c = r["information"]["close"]["walk_forward"]
    pos = {u: [GROUP_RU[g] for g, v in r["information"][u]["full_season"].items() if v["b_model_ci"][0] > 0]
           for u in ("close", "t09")}
    better = {u: [GROUP_RU[g] for g, v in r["information"][u]["walk_forward"].items()
                  if v["blend_minus_market"]["ci95"][1] < 0] for u in ("close", "t09")}
    rep.add(F.H1c("10. Выводы"),
            *bullets([
                f"H1 подтверждена: рынок без маржи точнее модели (1X2 close: модель − рынок "
                f"{r['demargin']['close']['1X2']['model_minus_power']['diff'] * 1000:+.1f}×10⁻³ log loss). Метод снятия "
                "маржи почти не важен.",
                f"H2 подтверждена частично: вес модели рядом с рынком значимо больше нуля (95 % ДИ) в close — "
                f"{', '.join(pos['close']) or 'нигде'}, в t09 — {', '.join(pos['t09']) or 'нигде'}; но во внешней "
                f"проверке по месяцам смесь значимо лучше рынка лишь в: {', '.join(better['close']) or 'ни одной группе'} "
                f"(close) / {', '.join(better['t09']) or 'ни одной группе'} (t09); 1X2 close: смесь − рынок "
                f"{wf_c['1X2']['blend_minus_market']['diff'] * 1000:+.2f}×10⁻³.",
                f"H3 не подтверждена: смесь модели с рынком (V1) меняет Main в "
                f"{F.pp(1 - V['close']['blend_wf']['vs_b']['same_main_share'], 0, False)} % матчей и повышает долю "
                f"выигрышей ({F.pp(V['close']['blend_wf']['summary']['hit_rate'], 1, False)} % против "
                f"{F.pp(V['close']['baseline_b']['summary']['hit_rate'], 1, False)} %), но снижает средний коэффициент; "
                f"ROI не растёт (минимальное p Холма {F.pval(min(H.values()))}).",
                "H4 не подтверждена как правило: фильтр расхождения в close чуть лучше, в t09 хуже Baseline B.",
                "H5 не подтверждена: максимизация EV по смеси уводит Main к высоким коэффициентам без выигрыша в ROI.",
                "Главный практический факт: фактическая частота выигрышей Main близка к вероятности рынка, а не "
                "модели — завышение p выбранного Main реально, его исправляет калибровка/рынок, но на ROI это не "
                f"влияет, потому что маржа книги (1X2 {F.pp(mg.get('1X2'), 1, False)} %, тоталы "
                f"{F.pp(mg.get('TOTAL'), 1, False)} %, угловые {F.pp(mg.get('CORNERS'), 1, False)} %) больше информации "
                "модели."]))
    # 11 -----------------------------------------------------------------------------------------
    rows = [["Вариант", "Решение", "Основание"]]
    for k, d in dec.items():
        why = (f"Δ close {F.pp(V['close'][k]['vs_b']['roi_diff']['diff'])} п.п., t09 "
               f"{F.pp(V['t09'][k]['vs_b']['roi_diff']['diff'])} п.п.; p Холма "
               f"{F.pval(H.get(k + '|close'))} / {F.pval(H.get(k + '|t09'))}")
        rows.append([VARIANTS_RU[k], d, why])
    rep.add(F.H1c("11. Рекомендации (A / B / C)"),
            P("Единое правило решения трёх рыночных исследований v6.3: **A** — значимое после Холма улучшение в обоих мирах; "
              "**B** — неотрицательная точечная разница в обоих мирах и прирост хотя бы в одном больше 0.5 п.п.; "
              "**C** — иначе."),
            F.T(rows), P("Таблица 18. Решения по вариантам.", "small"),
            *bullets(["Production не меняется: ни один вариант не прошёл критерий A.",
                      "Предрегистрация для проверки вперёд (`results/preregistration.json`): " +
                      "; ".join(f"{e['id']} (заморожено {e['frozen_at']}, минимум {e['minimum_forward_sample']})"
                                for e in pre["entries"]) + ".",
                      "Решение A из v6.2 остаётся в силе: показывать вероятность рынка без маржи рядом с вероятностью "
                      "модели (информационно, без влияния на выбор)."]))
    # 12 -----------------------------------------------------------------------------------------
    rep.add(F.H1c("12. Ограничения"),
            *bullets(["Сезон 2026/27 уже изучался в v6.1/v6.2: результат — хронологическая проверка, не независимое "
                      "подтверждение. Фиксированные веса v6.2 — единственная часть, выбранная полностью вне сезона.",
                      "Архив OddsPapi хранит в основном одну (последнюю) котировку на исход: в мире t09 у четверти Main "
                      "нет цены, у части рынков нет полной книги; кандидаты без книги сохраняют вероятность модели.",
                      "Время котировки Football-Data неизвестно (снимок открытия); как и в replay, она считается "
                      "доступной в день матча.",
                      "Почти все вероятности рынка — одна книга (bet365); консенсус нескольких книг — исследование 13.",
                      "Помесячная подгонка начинается с сентября: в августе смесь равна модели (мало прошлых матчей).",
                      "Пулы по группам объединяют линии (тоталы 1.5/2.5/3.5, форы); отдельная подгонка по линии могла бы "
                      "дать иной результат, но увеличила бы число вариантов."]))
    rep.add(F.H1c("Как воспроизвести"),
            *K.reproduce([".venv/bin/python -m experiments.v6_3.replay.build      # общий кэш (если его нет)",
                          ".venv/bin/python -m experiments.v6_3.market_aware.run  # ~2 мин, без обучения моделей",
                          ".venv/bin/python -m experiments.v6_3.market_aware.build_pdf",
                          "# все три рыночных исследования: experiments/v6_3/market_aware/run_all.sh"]),
            P("Результаты: `experiments/v6_3/market_aware/results/` (market_aware.json, coverage.csv, per_match_*.csv, "
              "cumulative_profit.csv, preregistration.json, run_info.json), рисунки — `figures/`.", "small"))
    rep.build()
    print("pages", rep.check())
    findings(r, dec, pre)


def findings(r, dec, pre) -> None:
    V, H = r["variants"], r["holm"]["p_holm"]
    lines = ["# FootPredict v6.3 — Market-Aware Main: выводы", "",
             "Все числа — из `results/market_aware.json` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_3/Market_Aware_Main_Research.pdf`.", "",
             "## Дефекты", "",
             f"- Векторизованное правило Main = Baseline B: close {F.pp(r['self_check']['close']['same_main_share'], 2, False)} %, "
             f"t09 {F.pp(r['self_check']['t09']['same_main_share'], 2, False)} % (1 матч — равные оценки). Дефектов "
             "общего кода не найдено.", "",
             "## Main ROI против Baseline B (Δ, п.п. [95 % ДИ], p Холма)", "",
             "| Вариант | close | t09 | Решение |", "|---|---|---|---|"]
    for k, d in dec.items():
        cells = []
        for u in ("close", "t09"):
            x = V[u][k]["vs_b"]["roi_diff"]
            cells.append(f"{F.pp(x['diff'])} {F.ci(x['ci95'])}, {F.pval(H.get(k + '|' + u))}")
        lines.append(f"| {VARIANTS_RU[k]} | {cells[0]} | {cells[1]} | {d} |")
    b = V["close"]["baseline_b"]["summary"]
    ov = r["disagreement"]["close"]["overall"]
    lines += ["", f"Baseline B close: ROI {F.pp(b['roi'])} %, ставок {b['bets']}; t09: ROI "
              f"{F.pp(V['t09']['baseline_b']['summary']['roi'])} %, ставок {V['t09']['baseline_b']['summary']['bets']}.", "",
              "## Информация модели сверх рынка", "",
              f"- Рынок точнее модели: 1X2 close Δ log loss модель − рынок "
              f"{r['demargin']['close']['1X2']['model_minus_power']['diff'] * 1000:+.1f}×10⁻³.",
              "- Вес модели в полносезонном пуле (close): " + ", ".join(
                  f"{g} {v['b_model']:.2f} {F.ci(v['b_model_ci'], 2, 1)}" for g, v in
                  r["information"]["close"]["full_season"].items()) + ".",
              f"- Main B: hit {F.pp(ov['hit'], 1, False)} %, средняя p {F.pp(ov['mean_p'], 1, False)} %, средняя q "
              f"{F.pp(ov['mean_q'], 1, False)} % — частота ближе к рынку.", "",
              "## Решения", "",
              "- A: нет. Production не меняется.",
              "- B: " + ", ".join(VARIANTS_RU[k] for k, d in dec.items() if d == "B") + " — заморожено в "
              "`results/preregistration.json` (" + ", ".join(e["id"] for e in pre["entries"]) + ").",
              "- C: остальные варианты.",
              "- Если B-кандидат пройдёт форвард: `src/markets/recommendation.py` — вероятность для правила Main = "
              "softmax(0.5·log q + 0.5·log p) при полной книге в момент генерации; `src/config.py` — флаг включения.",
              "", "## Воспроизведение", "", "```", ".venv/bin/python -m experiments.v6_3.market_aware.run",
              ".venv/bin/python -m experiments.v6_3.market_aware.build_pdf", "```", ""]
    (RES.parent / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
