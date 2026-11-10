"""Builds reports/v6_2/Risk_Selection_Research.pdf and findings.md from results/ (no number is typed by hand).

    python -m experiments.v6_2.risk.build_pdf
"""
from __future__ import annotations

from reportlab.lib.units import cm

from experiments.v6_2.main_calibration.build_pdf import V62Report, pci, pp, pval
from scripts import report_kit as K
from scripts.report_kit import H1, H2, H3, P, bullets, figure, formula, note, table

from . import common as C

OUT = C.ROOT / "reports" / "v6_2" / "Risk_Selection_Research.pdf"
SEASONS = [str(s) for s in C.SEASONS]
SL = {str(s): C.season_label(s) for s in C.SEASONS}
SLX = {str(s): C.period_label(s) for s in C.SEASONS}
BASE = "Текущий Risk (v6.1)"


def ci_of(v, c, d=1):
    return pci({"d": v, "d_ci": c}, "d", d) if v is not None and c else (pp(v, d) if v is not None else "—")


def build() -> None:
    r = C.load_json("risk.json")
    sel = C.load_json("selection.json")
    evr = C.load_json("ev_refunds.json")
    pre = C.load_json("preregistration.json")
    info = C.load_json("run_info.json")
    st, dg = r["strategies"], r["diagnostics"]
    base = {s: st[s][BASE] for s in SEASONS}
    shown = st["2026"]["Risk как показан"]
    ed = {s: dg[s]["ev_decomposition"] for s in SEASONS}
    def rng(key):
        vals = [ed[x][key] for x in SEASONS]
        return f"от {pp(min(vals))} до {pp(max(vals))}"
    rep = V62Report(OUT, "Выбор Risk 2.0", "FootPredict v6.2 — исследование",
                    {"Сезоны отбора": "2023/24, 2024/25", "Для информации": "2025/26",
                     "Текущий сезон (уже просмотрен)": "2026/27 до " + info["data"]["picks"]["mtime"][:10],
                     "Продакшн-правило Risk": "не изменяется"})

    rep.add(H1("1. Цель и проблема"),
            P("Risk — альтернативный прогноз матча с коэффициентом около 3 и выше, который всё ещё правдоподобен. "
              "Правило v6.1 выбирает его из пула исходов (без европейских фор, без самого Main, без исходов, совпадающих "
              "с Main или противоположных ему, p ≥ 0.12) по надёжности рынка, p^{0.5} и мягкому окну коэффициентов 3–6. "
              "По v6.1 все стратегии Risk убыточны, а EV модели у выбранного Risk часто положителен."),
            P("**Цель:** объяснить, почему EV модели положителен, а ROI отрицателен (переоценка редких исходов, эффект "
              "фаворит–аутсайдер, маржа, эффект выбора из многих кандидатов, зависимость от коэффициента), проверить "
              "калибровку после выбора, рынки, лиги, альтернативные ранжирования и сокращение числа ставок, а также "
              "проверить корректность EV для рынков с возвратом (азиатские целые и четвертные линии)."),
            note(f"Главный результат. EV модели у выбранного Risk {rng('book_model_ev')} % по сезонам, "
                 f"а EV по цене рынка без маржи {rng('book_market_ev')} %, и именно он "
                 f"совпадает с фактом ({rng('book_realised')} %; разность рынок − факт "
                 f"везде с 95 % ДИ, содержащим 0). Весь «плюс» модели — переоценка выбранного исхода: кандидаты пула "
                 f"откалиброваны, выбранный переоценён на {pp(dg['2023']['selection']['priced']['gap_chosen'])} п.п. (2023/24) и "
                 f"{pp(dg['2026']['selection']['priced']['gap_chosen'])} п.п. (2026/27). Ни одно ранжирование не делает Risk "
                 f"прибыльным; ошибка в формуле EV с возвратом не найдена, найдена ошибка «справедливого» коэффициента целой "
                 f"линии без цены (ранжирование)."))

    info26 = r["info"]["2026"]
    rep.add(H1("2. Данные и протокол"),
            P("**2022/23–2025/26:** таблицы кандидатов v6.1 (`experiments/v6_1/risk`) — продакшн-функции "
              "(`goal_markets`, `corner_markets`, `main_prediction`) на walk-forward прогнозах (те же, что в исследовании "
              "Main; кэш проверен повторным обучением), реальные цены Bet365 из Football-Data (1X2, тотал 2.5, одна "
              "азиатская линия). **2026/27:** хранилище продакшна — последняя до матча версия вероятностей каждого матча "
              "единой истории, записанные цены и последняя котировка OddsPapi до начала матча для пропусков; Main — "
              f"показанный продакшном. 2026/27 уже просмотрен в v6.1 и **не является независимой проверкой**."),
            P(f"Векторная копия правила Risk сверена с `src.markets.recommendation.risk_prediction`: "
              f"{r['info']['2024']['check_vs_production']['agree']}/{r['info']['2024']['check_vs_production']['checked']} матчей "
              f"2024/25 и {info26['check_vs_production']['agree']}/{info26['check_vs_production']['checked']} матчей 2026/27 совпали. "
              f"Пересчитанный Risk 2026/27 совпадает с показанным в {pp(info26['recomputed_equals_shown'], 0, False)} % матчей "
              f"(остальное — цены, добавленные из архива после показа); показанный Risk приведён отдельно."),
            table([["Сезон", "Матчей", "Строк-кандидатов"]] +
                  [[SLX[s] if s in SLX else s, r["info"][s]["matches"], r["info"][s]["rows"]] for s in ["2022"] + SEASONS]),
            P("Калибровочные модели для сезона s обучаются только на сезонах до s (расширяющееся окно от 2022/23). "
              "Все правила сезона оцениваются на одних и тех же матчах; ставка — Risk с реальной ценой, выигрыш или "
              "проигрыш (возврат не входит в ROI, как в продакшне).", "small"))

    rep.add(H1("3. Гипотезы"),
            bullets(["H1. Модель систематически переоценивает редкие исходы (p 0.12–0.45).",
                     "H2. Эффект фаворит–аутсайдер: у длинных коэффициентов маржа и/или ошибка цены больше.",
                     "H3. Эффект выбора: выбранный из десятков кандидатов Risk переоценён, даже если пул откалиброван.",
                     "H4. Калибровка после выбора делает вероятность Risk честной.",
                     "H5. Другое ранжирование или сокращение числа ставок даёт неотрицательный ROI.",
                     "H6. EV для ставок с возвратом в продакшне считается неверно."]))

    rep.add(H1("4. Методы"),
            P("p — вероятность выигрыша по модели, p_{r} — вероятность возврата, p_{c} = p / (1 − p_{r}); o — реальный "
              "коэффициент; q — вероятность рынка без маржи (пропорциональное снятие маржи с полной книги: "
              "q_{i} = (1/o_{i}) / Σ_{j}(1/o_{j}); для целой линии это P(выигрыш | нет возврата), и "
              "P(выигрыш) = q·(1 − p_{r}))."),
            formula("EV модели = p·o − (1 − p_{r});   EV рынка = q·(1 − p_{r})·o − (1 − p_{r});   факт = прибыль на ставку"),
            formula("EV модели − факт = (EV модели − EV рынка) + (EV рынка − факт)"),
            P("Первое слагаемое — расхождение модели с рынком (по цене), второе — ошибка цены рынка без маржи плюс "
              "случайность; если рынок без маржи несмещён, второе слагаемое в среднем ≈ 0, а EV рынка ≈ −маржа."),
            bullets(["Эффект выбора: выбранный против невыбранных кандидатов пула той же вероятности (перевзвешивание "
                     "по интервалам p) и по квартилям размера пула.",
                     "Калибровка после выбора: PS-Platt, PS-Group, PS-Market (y на logit p_{c} и logit q_{c}), PS-OS (поправка "
                     "порядковой статистики) — по выбранным Risk истории; в 2026/27 также помесячно.",
                     "Ранжирования (кандидаты пула с ценой и коэффициентом 2.50–8.00, иначе продакшн-Risk): максимум EV по "
                     "сырой p, по калиброванной (Платт по типу исхода), по стеку с рынком "
                     "p_{s} = σ(b_{1}·logit p_{c} + b_{2}·logit q_{c} + b_{0}), максимум Келли, EV по нижней границе "
                     "p_{s} − √(p_{s}(1 − p_{s})/n), перевес над рынком p_{s} − q, максимум p_{s}, максимум q; окна "
                     "2.50–3.50 и 3.00–4.50; правило v6.1 «стек с рынком, пол EV ≥ 0» (с исправленной вероятностью рынка "
                     "для целых линий).",
                     "Меньше ставок: пороги ключа (фиксированы до запуска) и топ-1/3/5 матчей календарного дня по ключу "
                     "(без заглядывания вперёд).",
                     f"Статистика: бутстрап по матчам, 2000 повторов, парные разности к текущему Risk; объединённые сезоны "
                     f"отбора и поправка Холма по {r['pooled']['n_rules']} правилам."]),
            P("Правило отбора (записано в коде до запуска): " + sel["rule"], "small"))

    rows = [["Сезон", "Ставок", "Ср. коэф.", "Заявлено", "Факт", "gap, п.п.", "EV модели", "ROI, % [95 % ДИ]", "Просадка",
             "Серия проигр."]]
    for s in SEASONS:
        m = base[s]
        rows.append([SLX[s], m["bets"], f"{m['avg_odds']:.2f}", pp(m["mean_p"], 1, False), pp(m["hit"], 1, False),
                     pp(m["gap"]), pp(m["model_ev"]), pci(m, "roi"), f"{m['max_drawdown']:.0f}", m["longest_losing"]])
    m = shown
    rows.append(["2026/27 как показан", m["bets"], f"{m['avg_odds']:.2f}", pp(m["mean_p"], 1, False), pp(m["hit"], 1, False),
                 pp(m["gap"]), pp(m["model_ev"]), pci(m, "roi"), f"{m['max_drawdown']:.0f}", m["longest_losing"]])
    rep.add(H1("5. Результаты"), H2("5.1 Текущий Risk"), table(rows),
            P("Заявлено/факт — в %, просадка — в единицах ставки, серия — самая длинная серия проигрышей.", "small"))

    rows = [["Сезон", "Ставок с книгой", "Ср. p", "Ср. q", "EV модели", "EV рынка", "Факт [95 % ДИ]", "Модель − рынок",
             "Рынок − факт [95 % ДИ]"]]
    for s in SEASONS + ["2026s"]:
        e = dg["2026"]["ev_decomposition_shown"] if s == "2026s" else ed[s]
        rows.append(["2026/27 показан" if s == "2026s" else SLX[s], e["with_book"], pp(e["mean_p_book"], 1, False),
                     pp(e["mean_q"], 1, False), pp(e["book_model_ev"]), pp(e["book_market_ev"]),
                     ci_of(e["book_realised"], e["book_realised_ci"]), pp(e["model_minus_market"]),
                     ci_of(e["market_minus_realised"], e["market_minus_realised_ci"])])
    rep.add(H2("5.2 Почему EV модели положителен, а ROI отрицателен"),
            P("Разложение на ставках Risk с полной книгой (все величины — % на единицу ставки, возврат = 0)."),
            table(rows), *figure(C.FIGURES / "ev_decomposition.png", "Рис. 1. EV модели, EV по цене рынка и факт."),
            note("EV рынка (≈ минус маржа) предсказывает фактическую доходность; вся разница между положительным EV модели "
                 "и отрицательным ROI — расхождение модели с ценой, т. е. переоценка выбранного исхода моделью."))

    def band_rows(key):
        out = [["Интервал p", "Сезон", "n", "p модели", "Факт", "gap, п.п.", "z", "n с ценой", "q рынка", "Факт (с ценой)"]]
        for s in ("2024", "2026"):
            for x in dg[s][key]:
                out.append([x["band"], SL[s], x["n"], pp(x["mean_p"], 1, False), pp(x["hit"], 1, False), pp(x["gap"]),
                            f"{x['z']:+.1f}" if x.get("z") is not None else "—", x.get("n_q", "—"),
                            pp(x["mean_q"], 1, False) if x.get("mean_q") is not None else "—",
                            pp(x["hit_q"], 1, False) if x.get("hit_q") is not None else "—"])
        return out
    rep.add(H2("5.3 Калибровка редких исходов"),
            P("Все кандидаты пула Risk (p ≥ 0.12), вероятность без возврата; z = (факт − p)/ст. ош. Показаны 2024/25 и "
              "2026/27; остальные сезоны — на рис. 2 и в `results/risk.json`."),
            table(band_rows("rare_pool"), widths=[1.9 * cm, 1.5 * cm, 1.3 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.1 * cm,
                                                 1.7 * cm, 1.7 * cm, 2.3 * cm]),
            *figure(C.FIGURES / "rare_calibration.png", "Рис. 2. Калибровка кандидатов пула и выбранного Risk."),
            note("Кандидаты пула откалиброваны (|gap| около 1 п.п. и меньше во всех интервалах), выбранный Risk — нет, и "
                 "сильнее всего на верхнем краю (p 0.40–0.50). H1 в общем виде отвергнута; переоценка возникает при выборе."))

    rows = [["Сезон", "gap выбранного [95 % ДИ]", "gap невыбранных той же p", "EV выбранного", "EV невыбранных той же p"]]
    for s in SEASONS:
        x = dg[s]["selection"]["priced"]
        rows.append([SLX[s], ci_of(x["gap_chosen"], x["gap_chosen_ci"]), pp(x["gap_rest_reweighted"]), pp(x["ev_chosen"]),
                     pp(x["ev_rest_reweighted"])])
    rows2 = [["Сезон"] + [f"Квартиль {i + 1}: размер пула / gap" for i in range(4)]]
    for s in SEASONS:
        rows2.append([SL[s]] + [f"{q['range']}: {pp(q['gap'])} ± {pp(1.96 * q['se'], 1, False)}" for q in dg[s]["selection"]["by_pool_size"]])
    rep.add(H2("5.4 Эффект выбора из многих кандидатов"),
            P("Кандидаты пула с реальной ценой; «невыбранные той же p» перевзвешены к распределению вероятностей выбранных."),
            table(rows), H3("Переоценка выбранного Risk по размеру пула (кандидатов с p ≥ 0.12)"), table(rows2),
            note("Выбранный Risk переоценён, невыбранные кандидаты той же вероятности — нет; переоценка растёт с размером "
                 "пула. H3 подтверждена: выбор по p^{0.5} и окну коэффициентов сам отбирает исходы, где модель случайно "
                 "выше рынка."))

    fl = dg["pooled_history"]["favourite_longshot"]
    fl26 = dg["2026"]["favourite_longshot"]
    rows = [["Полоса коэф.", "n (история)", "ROI, % [95 % ДИ]", "q − факт", "модель − факт", "n (2026/27)",
             "ROI 26/27, %", "q − факт 26/27", "модель − факт 26/27"]]
    f26 = {x["band"]: x for x in fl26}
    for x in fl:
        y = f26.get(x["band"])
        rows.append([x["band"], x["n"], ci_of(x["roi"], x["roi_ci"]), pp(x["q_gap"], 2), pp(x["model_gap"], 2),
                     y["n"] if y else "—", pp(y["roi"]) if y else "—", pp(y["q_gap"], 2) if y else "—",
                     pp(y["model_gap"], 2) if y else "—"])
    mg = {x["group"]: x for x in dg["pooled_history"]["margins"]}
    mg26 = {x["group"]: x for x in dg["2026"]["margins"]}
    rows_m = [["Рынок", "Книг (история)", "Маржа, % (история)", "10–90 %", "Книг 26/27", "Маржа, % 26/27", "10–90 % 26/27"]]
    for g in sorted(set(mg) | set(mg26)):
        a, b = mg.get(g), mg26.get(g)
        rows_m.append([C.GROUP_RU.get(g, g), a["books"] if a else "—", pp(a["overround"], 1, False) if a else "—",
                       f"{pp(a['overround_p10'], 1, False)}–{pp(a['overround_p90'], 1, False)}" if a else "—",
                       b["books"] if b else "—", pp(b["overround"], 1, False) if b else "—",
                       f"{pp(b['overround_p10'], 1, False)}–{pp(b['overround_p90'], 1, False)}" if b else "—"])
    mb = dg["pooled_history"]["margin_by_band"]
    rep.add(H2("5.5 Эффект фаворит–аутсайдер и маржа"),
            P("Каждый исход с реальной ценой из полной книги (история 2022/23–2025/26 — Bet365 Football-Data: 1X2, тотал "
              "2.5, азиатская линия; 2026/27 — все рынки с записанными ценами). «q − факт» и «модель − факт» — в п.п."),
            table(rows, widths=[1.9 * cm, 1.6 * cm, 3.0 * cm, 1.5 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 2.2 * cm]),
            *figure(C.FIGURES / "favourite_longshot.png", "Рис. 3. Эффект фаворит–аутсайдер."),
            H3("Маржа букмекера (overround книги)"),
            table(rows_m, widths=[2.6 * cm, 2.2 * cm, 2.6 * cm, 2.4 * cm, 2.0 * cm, 2.6 * cm, 2.6 * cm]),
            P("Ожидаемая потеря при правильной цене без маржи (1 − q·o) по полосам коэффициента (история): "
              + "; ".join(f"{x['band']}: {pp(x['margin_at_q'], 1, False)} %" for x in mb)
              + ". Пропорциональное снятие маржи по построению распределяет её равномерно, поэтому дополнительная потеря "
                "у длинных коэффициентов — это смещение самой цены (q − факт > 0 при коэффициентах ≥ 4.5).", "small"),
            note("Эффект фаворит–аутсайдер есть: слепая ставка на исходы с коэффициентом ≥ 4.5 теряет заметно больше маржи, "
                 "вероятность рынка у них завышена, а модель завышает их ещё сильнее. Окно Risk 3–6 лежит как раз на границе "
                 "этого участка (H2 подтверждена)."))

    od = r["breakdowns"]["2023-2025"]["odds"]
    rows = [["Полоса коэф.", "Ставок", "Ср. коэф.", "p модели", "Факт", "gap, п.п.", "EV модели", "q", "ROI, % [95 % ДИ]"]]
    for x in od:
        rows.append([x["band"] + (" (мало)" if x["small"] else ""), x["n"], f"{x['avg_odds']:.2f}", pp(x["mean_p"], 1, False),
                     pp(x["hit"], 1, False), pp(x["gap"]), pp(x["model_ev"]), pp(x["q"], 1, False) if x["q"] is not None else "—",
                     ci_of(x["roi"], x["roi_ci"])])
    rep.add(H2("5.6 Зависимость от коэффициента (текущий Risk, 2023/24–2025/26)"), table(rows),
            *figure(C.FIGURES / "odds_dependence.png", "Рис. 4. EV модели и фактический ROI по полосам коэффициента."))

    def post_rows(key):
        out = [["Вероятность", "n", "Заявлено", "Факт", "gap, п.п. [95 % ДИ]", "log loss"]]
        for nm, x in r["post"][key].items():
            out.append([nm, x["n"], pp(x["mean_p"], 1, False), pp(x["hit"], 1, False), ci_of(x["gap"], x["gap_ci"]),
                        f"{x['log_loss']:.4f}"])
        return out
    rep.add(H2("5.7 Калибровка выбранного Risk после выбора"),
            P("Карты обучены на выбранных Risk сезонов до текущего; «рынок» — q без маржи (только Risk с полной книгой, "
              "поэтому n меньше)."))
    for key in SEASONS + ["2026_shown"]:
        rep.add(H3(SLX.get(key, "2026/27, показанный Risk")), table(post_rows(key), widths=[4.6 * cm, 1.4 * cm, 1.8 * cm,
                                                                                            1.6 * cm, 4.2 * cm, 2 * cm]))
    rep.add(*figure(C.FIGURES / "post_calibration.png", "Рис. 5. Калибровка Risk до и после калибровки после выбора."),
            note("Карты после выбора, обученные на истории, слегка перекорректируют сезоны отбора и недокорректируют "
                 f"2026/27. Вероятность рынка без маржи откалибрована во всех сезонах (|gap| ≤ "
                 f"{pp(max(abs(r['post'][k]['рынок (Шин→пропорц.)']['gap']) for k in SEASONS), 1, False)} п.п.) — честная "
                 f"вероятность Risk есть вероятность рынка (H4: да, но лучшая «калибровка» — рынок)."))

    grp = r["breakdowns"]
    rows = [["Группа", "Ставок 23–25", "ROI 23–25, %", "gap 23–25", "Ставок 26/27", "ROI 26/27, %", "gap 26/27"]]
    g1 = {x["key"]: x for x in grp["2023-2025"]["group"]}
    g2 = {x["key"]: x for x in grp["2026"]["group"]}
    for g in sorted(set(g1) | set(g2)):
        a, b = g1.get(g), g2.get(g)
        rows.append([C.GROUP_RU.get(g, g), a["bets"] if a else 0, pp(a["roi"]) if a and a["roi"] is not None else "—",
                     pp(a["gap"]) if a else "—", b["bets"] if b else 0, pp(b["roi"]) if b and b["roi"] is not None else "—",
                     pp(b["gap"]) if b else "—"])
    lg = sorted(grp["2023-2025"]["league"], key=lambda x: x["roi"] if x["roi"] is not None else 0)
    rows_l = [["Лига", "Ставок", "ROI, % [95 % ДИ]", "gap, п.п.", "Лига", "Ставок", "ROI, % [95 % ДИ]", "gap, п.п."]]
    half = (len(lg) + 1) // 2
    for i in range(half):
        a = lg[i]
        b = lg[i + half] if i + half < len(lg) else None
        rows_l.append([a["key"], a["bets"], ci_of(a["roi"], a["roi_ci"]), pp(a["gap"])] +
                      ([b["key"], b["bets"], ci_of(b["roi"], b["roi_ci"]), pp(b["gap"])] if b else ["", "", "", ""]))
    n_pos = sum(1 for x in lg if x["roi_ci"] and x["roi_ci"][0] > 0)
    rep.add(H2("5.8 Рынки и лиги"), P("Текущий Risk по группам рынков (2023/24–2025/26 вместе и 2026/27). В истории цены "
                                      "есть только у 1X2, тотала 2.5 и одной форы, поэтому ставки — почти только 1X2."),
            table(rows), H3("Лиги, 2023/24–2025/26"), table(rows_l),
            P(f"Лиг с нижней границей 95 % ДИ ROI выше 0: {n_pos} из {len(lg)} (без поправки на множественность); "
              "различия между лигами совместимы с шумом.", "small"))

    names = list(st["2023"])
    rows = [["Правило", "ROI 23/24", "ROI 24/25", "ROI 25/26", "ROI 26/27", "Коэф. 24/25", "EV 24/25", "ΔROI отбор [95 % ДИ]",
             "p Холма"]]
    for nm in names:
        pr = r["pooled"]["rules"].get(nm)
        rows.append([nm] + [pp(st[s][nm]["roi"]) if nm in st[s] else "—" for s in SEASONS] +
                    [f"{st['2024'][nm]['avg_odds']:.2f}", pp(st["2024"][nm]["model_ev"]),
                     ci_of(pr["d_roi"], pr["d_roi_ci"]) if pr else "—", pval(pr["p_holm"]) if pr else "—"])
    rows.append(["Risk как показан (2026/27)", "—", "—", "—", pp(shown["roi"]), "—", "—", "—", "—"])
    rep.add(H2("5.9 Альтернативные ранжирования"),
            table(rows, widths=[4.4 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 2.6 * cm, 1.2 * cm]),
            *figure(C.FIGURES / "strategies_roi.png", "Рис. 6. ROI правил Risk (95 % ДИ)."))

    rows = [["Правило | ограничение", "Ставок 23/24", "ROI 23/24", "Ставок 24/25", "ROI 24/25", "ROI 25/26", "ROI 26/27",
             "ROI 26/27 [95 % ДИ]"]]
    for nm in r["fewer"]["2023"]:
        cells = [r["fewer"][s][nm] for s in SEASONS]
        rows.append([nm, cells[0]["bets"], pp(cells[0]["roi"]), cells[1]["bets"], pp(cells[1]["roi"]), pp(cells[2]["roi"]),
                     pp(cells[3]["roi"]), ci_of(cells[3]["roi"], cells[3]["roi_ci"]) if cells[3]["bets"] > 1 else "—"])
    rep.add(H2("5.10 Меньше ставок"),
            P("Ставка только при значении ключа выше порога или только на 1/3/5 лучших матчей дня по ключу. Пустые "
              "ячейки — нет ставок."),
            table(rows, widths=[5.6 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 3.0 * cm]),
            *figure(C.FIGURES / "fewer_bets.png", "Рис. 7. ROI при сокращении числа ставок."),
            note("При сокращении числа ставок ROI не становится устойчиво положительным: знак меняется от сезона к сезону, "
                 "доверительные интервалы при сотнях ставок шириной в десятки п.п. Порог «EV стека ≥ 0» почти не срабатывает в "
                 "истории и срабатывает часто в 2026/27 — стек, обученный на рынках с ценой в истории (1X2, тотал 2.5, "
                 "фора), переносится на остальные рынки 2026/27 только общей картой. H5 отвергнута."))

    # ---------------------------------------------------------------- EV with refunds
    ex = evr["example"]
    stc = evr["stored"]
    rows = [["Линия (хозяева)", "Тип", "P(выигрыш)", "P(возврат)", "EV верный", "EV без учёта возврата / как соседняя линия"]]
    for x in ex["rows"]:
        if x["kind"] == "quarter":
            rows.append([f"{x['line']:+g}", "четвертная", " / ".join(pp(v, 1, False) for v in x["p_win_halves"]),
                         " / ".join(pp(v, 1, False) for v in x["p_refund_halves"]), pp(x["ev"]),
                         f"как {x['halves'][0]:+g}: {pp(x['ev_if_treated_as_lower'])}; как {x['halves'][1]:+g}: {pp(x['ev_if_treated_as_upper'])}"])
        else:
            rows.append([f"{x['line']:+g}", "целая" if x["kind"] == "whole" else "половинная", pp(x["p_win"], 1, False),
                         pp(x["p_refund"], 1, False), pp(x["ev"]), pp(x["ev_without_refund"])])
    fo = evr["fair_odds"]
    rows_f = [["Сезон", "Матчей", "Risk меняется", "из них целые линии без цены", "Эфф. коэф. в правиле",
               "Верный безубыточный коэф.", "Факт у старого выбора", "p без возврата"]]
    for s in SEASONS:
        x = fo["risk"][s]
        b = x["before"]
        rows_f.append([SL[s], x["matches"], x["changed"], x["changed_whole_line_before"],
                       f"{b['shown_eff_odds']:.2f}" if b["shown_eff_odds"] else "—",
                       f"{b['eff_odds']:.2f}" if b["eff_odds"] else "—",
                       pp(b["hit"], 1, False) if b["hit"] is not None else "—",
                       pp(b["mean_p_no_refund"], 1, False) if b["mean_p_no_refund"] is not None else "—"])
    main_changed = {s: fo["main"][s]["changed"] for s in SEASONS}
    sc = evr["scale"]
    q = evr["quarter"]["rows"]
    rows_q = [["Четвертная линия", "Строк", "Ср. |ошибка EV|, если взять соседнюю линию", "… если игнорировать возврат"]]
    for x in q:
        rows_q.append([f"{x['quarter_line']:+g}", x["n"], pp(x["mean_abs_err_as_lower"], 1, False),
                       pp(x["mean_abs_err_ignoring_refund"], 1, False)])
    rep.add(H2("5.11 EV для ставок с возвратом (азиатские линии)"),
            P("Ставка 1, коэффициент o. Половинная линия (±0.5, ±1.5): выигрыш o − 1 или проигрыш −1. Целая линия "
              "(±1, ±2): выигрыш, возврат (0) или проигрыш. Четвертная линия L (±0.25, ±0.75, …) — половина ставки на "
              "L − ¼ и половина на L + ¼ при одном коэффициенте; исходы: полный выигрыш W, выигрыш половины HW (вторая "
              "половина возвращена), проигрыш половины HL, полный проигрыш L."),
            formula("половинная: EV = p_{w}·o − 1;   целая: EV = p_{w}(o − 1) − p_{l} = p_{w}·o − (1 − p_{r})"),
            formula("четвертная: EV = p_{W}(o − 1) + p_{HW}(o − 1)/2 − p_{HL}/2 − p_{L} = ½·EV(L − ¼) + ½·EV(L + ¼)"),
            P(f"Пример (независимый Пуассон {ex['lambda_home']} : {ex['lambda_away']}, коэффициент {ex['odds']}):"),
            table(rows, widths=[2.2 * cm, 2.0 * cm, 2.4 * cm, 2.2 * cm, 1.8 * cm, 6.4 * cm]),
            H3("Сравнение с продакшном"),
            bullets([f"`src/markets/odds.py`: EV = p·o − (1 − p_{{r}}) — верно для половинных и целых линий. Пересчёт всех "
                     f"{stc['markets_rows_priced']} строк `markets.parquet` с ценой (из них {stc['markets_whole_line_priced']} — "
                     f"целые линии): максимальное расхождение {stc['markets_max_abs_diff']:.1g} (точность float32); "
                     f"{stc['picks_priced_with_push_known']} прогнозов `picks.parquet` с ценой: {stc['picks_max_abs_diff']:.1g}; "
                     f"ни один из {stc['picks_whole_line_priced']} прогнозов на целой линии не посчитан по формуле без возврата.",
                     f"Четвертные линии продакшн не использует: все три поставщика их отбрасывают (`src/providers/oddspapi.py`, "
                     f"`api_football.py`, `football_data.py`); в архиве {stc['archive_handicap_quotes']} котировок фор — "
                     f"{stc['archive_quarter_handicap_quotes']} четвертных, четвертных тоталов — "
                     f"{stc['archive_quarter_total_quotes']}. Ошибки EV четвертных линий в продакшне нет; таблица ниже "
                     f"показывает, какой была бы ошибка при их добавлении без формулы для разделённой ставки.",
                     "Соглашения согласованы, но их надо помнить при сравнении: EV — на поставленную единицу с учётом "
                     "возвратов, ROI и hit rate продакшна возвраты исключают (`src/analytics/performance.py`), поэтому "
                     "ожидаемый ROI целой линии равен EV/(1 − p_{r}), а сравнимая с hit rate вероятность — "
                     "p/(1 − p_{r}). В 2026/27 таких показанных прогнозов мало: Main на целой линии "
                     f"{sc['main']['whole_line_settled']}, Risk {sc['risk']['whole_line_settled']} (рассчитанных)."]),
            table(rows_q, widths=[2.6 * cm, 1.6 * cm, 6.6 * cm, 6.2 * cm]),
            P("Пары соседних линий продакшна на прогнозах 2026/27 (хозяева и гости). Цена условная: o = 0.95 / (средняя "
              "вероятность выигрыша двух половин + ½ средней вероятности возврата) — это иллюстрация масштаба ошибки "
              "(в % на ставку), а не оценка реальных цен.", "small"),
            H3("Найденная ошибка: «справедливый» коэффициент целой линии без цены"),
            P("`src/markets/recommendation.effective_odds` для исхода без реальной цены возвращает 1/p_{w}. Для целой "
              "линии безубыточный коэффициент равен (1 − p_{r})/p_{w} (EV = 0), т. е. правило видит более длинный "
              "коэффициент, чем есть. Для Risk (окно 3–6) это продвигает целые линии без цены в окно (таблица: "
              "коэффициент в правиле против верного). `PricedSelection.fair_odds` повторяет ту же формулу."),
            table(rows_f, widths=[1.5 * cm, 1.4 * cm, 1.6 * cm, 2.4 * cm, 2.3 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm]),
            P(f"Main меняется в {', '.join(f'{SL[s]}: {main_changed[s]}' for s in SEASONS)} матчах. В 2026/27 почти у всех "
              f"кандидатов есть реальная цена, поэтому ошибка там не срабатывает; в walk-forward истории (и в бэктесте "
              f"2025/26 продакшна) затронуто {', '.join(pp(fo['risk'][s]['changed'] / fo['risk'][s]['matches'], 1, False) + ' %' for s in SEASONS[:3])} "
              f"Risk — все без цены, т. е. на ROI это не влияет, но такой «Risk» на деле имеет более короткий коэффициент "
              f"и более высокую частоту выигрыша (таблица), чем предполагает окно Risk."),
            note("Исправление (не внесено, продакшн не меняется): в `effective_odds` и `fair_odds` использовать "
                 "`(1 - push_probability) / probability` вместо `1 / probability`. Последствия проверены выше: меняются "
                 "только выборы без реальной цены."))

    rep.add(H1("6. Отбор по зафиксированному правилу"),
            P(f"Допустимы по ограничениям (доля ставок с ценой ≥ 85 %, средний коэффициент ≥ 2.80 в обоих сезонах): "
              f"{len(sel['eligible'])} правил. Выбрано: **{sel['chosen']}** (min ROI сезонов отбора "
              f"{pp(sel['ranking'][0]['worst_roi'])} % против {pp(min(base[s]['roi'] for s in ('2023', '2024')))} % у текущего). "
              f"ΔROI по сезонам отбора {', '.join(pp(x) for x in sel['d_roi_each'])} п.п.; объединённо "
              f"{ci_of(sel['pooled']['d_roi'], sel['pooled']['d_roi_ci'])}, p Холма {pval(sel['pooled']['p_holm'])} → "
              f"{'перспективно' if sel['promising'] else 'не перспективно по критерию'}. На 2025/26 ΔROI "
              f"{pp(st['2025'][sel['chosen']]['d_roi'])}, на 2026/27 {pp(st['2026'][sel['chosen']]['d_roi'])} п.п. — знак тот "
              f"же во всех четырёх сезонах, но интервалы включают 0, а само правило убыточно."))

    rep.add(H1("7. Отрицательные результаты"),
            bullets(["Модель не переоценивает редкие исходы в целом (H1 отвергнута): переоценка возникает только у выбранного.",
                     "Ни одно ранжирование (EV, калиброванный EV, стек с рынком, Келли, нижняя граница, перевес над рынком, "
                     "окна коэффициентов) не даёт неотрицательного ROI ни в одном сезоне отбора; ранжирование по сырому EV "
                     f"в сезонах отбора хуже текущего (объединённо {ci_of(r['pooled']['rules']['Макс. EV (сырая p)']['d_roi'], r['pooled']['rules']['Макс. EV (сырая p)']['d_roi_ci'])} п.п.).",
                     "Сокращение числа ставок не даёт устойчивого результата.",
                     "Ошибки в формуле EV с возвратом нет (H6 отвергнута); найдена только ошибка эффективного коэффициента "
                     "целой линии без цены."]))
    rep.add(H1("8. Ограничения"),
            bullets(["В истории реальные цены только у 1X2, тотала 2.5 и одной азиатской линии — Risk-ставки истории почти "
                     "все на 1X2; рынки 2026/27 (тоталы, обе забьют, угловые, двойной шанс) проверяются только на "
                     "просмотренном сезоне.",
                     "Пропорциональное снятие маржи — упрощение; для полос с длинными коэффициентами оно само смещено (5.5).",
                     "Бутстрап по матчам не учитывает корреляцию внутри тура; ДИ слегка оптимистичны."]))
    h = pre["hypotheses"]
    rep.add(H1("9. Выводы и рекомендации"),
            table([["Решение", "Что", "Почему"],
                   ["A", "Исправить «справедливый» коэффициент целой линии без цены: (1 − p_{r})/p_{w}",
                    "Математическая ошибка ранжирования; затрагивает только выборы без цены"],
                   ["B", h["H1_rule"]["name"], "ΔROI > 0 во всех сезонах, но ДИ включают 0; правило всё равно убыточно — форвард-тест"],
                   ["B", h["H2_display"]["name"], "Лучше всех откалибрована во всех сезонах; форвард-тест отображения"],
                   ["C", "Ранжирование по EV / Келли / нижней границе / перевесу; меньше ставок; окна коэффициентов",
                    "Не улучшают ROI или нестабильны"],
                   ["C", "Показывать EV модели для Risk как преимущество",
                    f"EV модели выше EV рынка на {pp(min(ed[s]['model_minus_market'] for s in SEASONS))}…"
                    f"{pp(max(ed[s]['model_minus_market'] for s in SEASONS))} п.п.; честный EV ≈ −маржа"]],
                  widths=[2.0 * cm, 7.2 * cm, 7.8 * cm]),
            P("Решение A — исправление ошибки в коде, а не новая стратегия; по правилам исследования продакшн здесь не "
              "меняется, исправление описано для отдельного изменения."),
            H2("Предрегистрация"),
            P(f"`experiments/v6_2/risk/results/preregistration.json` (записан {pre['written_at']}): H1 — {h['H1_rule']['name']}; "
              f"H2 — {h['H2_display']['name']}; поправка Холма по двум гипотезам."),
            bullets(["H1: на ставках Risk матчей, сыгранных после коммита файла, парный бутстрап 95 % ДИ (по матчам, 2000 "
                     "повторов) разности ROI(H1) − ROI(текущий Risk) лежит выше 0; оценка один раз при ≥ 2000 рассчитанных "
                     "матчах с ценой у обоих выборов (или в конце сезона 2026/27: тогда результат неубедителен, если ДИ не "
                     "выше 0).",
                     "H2: на показанном Risk (рассчитан, полная книга) 95 % ДИ gap вероятности рынка содержит 0 и парный ДИ "
                     "|gap(рынок)| − |gap(модель)| ниже 0; оценка при ≥ 1000 рассчитанных Risk.",
                     "Параметры стека H1 заморожены в файле (коэффициенты по типам исходов, обучены на 2022/23–2025/26)."]))
    rep.add(H1("10. Воспроизведение"),
            K.reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                         "# таблицы кандидатов 2022-2025: experiments/v6_1/risk/cache/cand_*.parquet -> cache/",
                         "#   (или python -m experiments.v6_1.risk.data)",
                         "$PY -m experiments.v6_2.risk.data              # 2026/27 из хранилища",
                         "$PY -m experiments.v6_2.risk.run               # ~3 мин",
                         "$PY -m experiments.v6_2.risk.preregister",
                         "$PY -m experiments.v6_2.risk.build_pdf"]))
    rep.build()
    print("pages", rep.check(scale=1.0))
    findings(r, sel, evr, pre, base, shown, ed, dg)


def findings(r, sel, evr, pre, base, shown, ed, dg) -> None:
    fo = evr["fair_odds"]["risk"]
    lines = ["# FootPredict v6.2 — выбор Risk 2.0: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: `reports/v6_2/Risk_Selection_Research.pdf`. "
             "Продакшн-Risk не меняется.", "",
             "| Сезон | Ставок | Ср. коэф. | EV модели, % | EV рынка, % | ROI, % [95 % ДИ] | gap выбранного, п.п. |",
             "|---|---|---|---|---|---|---|"]
    for s in SEASONS:
        m, e = base[s], ed[s]
        lines.append(f"| {SLX[s]} | {m['bets']} | {m['avg_odds']:.2f} | {pp(e['book_model_ev'])} | {pp(e['book_market_ev'])} | "
                     f"{pci(m, 'roi')} | {pp(m['gap'])} |")
    lines += [f"| 2026/27 как показан | {shown['bets']} | {shown['avg_odds']:.2f} | — | — | {pci(shown, 'roi')} | {pp(shown['gap'])} |", "",
              "## Почему EV > 0, а ROI < 0", "",
              "- EV по цене рынка без маржи совпадает с фактом (рынок − факт: ДИ содержит 0 во всех сезонах); EV модели выше на "
              + ", ".join(pp(ed[s]['model_minus_market']) for s in SEASONS) + " п.п. по сезонам.",
              f"- Кандидаты пула откалиброваны; выбранный Risk переоценён ({pp(dg['2023']['selection']['priced']['gap_chosen'])} п.п. "
              f"2023/24, {pp(dg['2026']['selection']['priced']['gap_chosen'])} п.п. 2026/27) — эффект выбора; растёт с размером пула.",
              "- Фаворит–аутсайдер: исходы с коэффициентом ≥ 4.5 теряют больше маржи; модель завышает их сильнее рынка.", "",
              "## Правила", "",
              f"- Лучшее по зафиксированному правилу: {sel['chosen']}, ΔROI объединённо {ci_of(sel['pooled']['d_roi'], sel['pooled']['d_roi_ci'])}, "
              f"p Холма {pval(sel['pooled']['p_holm'])} — не перспективно по критерию; знак положителен во всех сезонах.",
              "- Ранжирования по EV/Келли/нижней границе/перевесу и сокращение числа ставок — без устойчивого улучшения.", "",
              "## EV с возвратом", "",
              f"- Формула продакшна для целых линий верна (макс. расхождение {evr['stored']['markets_max_abs_diff']:.1g}); четвертных линий нет "
              f"({evr['stored']['archive_quarter_handicap_quotes']} из {evr['stored']['archive_handicap_quotes']} котировок).",
              "- Ошибка: `effective_odds`/`fair_odds` = 1/p для целой линии без цены; верно (1 − p_refund)/p. Меняет Risk в "
              + ", ".join(f"{SL[s]}: {fo[s]['changed']}" for s in SEASONS) + " матчах (все — выборы без цены).", "",
              "## Решения", "",
              "- A: исправить эффективный коэффициент целой линии (отдельным изменением кода).",
              f"- B: {pre['hypotheses']['H1_rule']['name']}; {pre['hypotheses']['H2_display']['name']} — форвард-тесты (`results/preregistration.json`).",
              "- C: остальные ранжирования, сокращение ставок, показ EV модели как преимущества.", "",
              "## Воспроизведение", "", "```",
              "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
              "$PY -m experiments.v6_2.risk.data && $PY -m experiments.v6_2.risk.run",
              "$PY -m experiments.v6_2.risk.preregister && $PY -m experiments.v6_2.risk.build_pdf", "```", ""]
    (C.HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    C.setup_logging()
    build()
