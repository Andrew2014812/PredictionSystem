"""Builds reports/v6_3/Market_Consensus_Research.pdf and findings.md from results/ (no number typed by hand).

    .venv/bin/python -m experiments.v6_3.consensus.build_pdf
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.market_aware import fmt as F
from scripts import report_kit as K
from scripts.report_kit import H2, P, bullets, figure, formula

from .fd import SEASONS
from .run import FIG, GAP_LABELS, RES, RULES, VARIANTS_RU

OUT = C.ROOT / "reports" / "v6_3" / "Market_Consensus_Research.pdf"
GROUP_RU = {"1X2": "1X2", "TOTAL": "Тоталы", "CORNERS": "Угловые", "BTTS": "Обе забьют", "HANDICAP": "Азиатские форы",
            "DOUBLE_CHANCE": "Двойной шанс"}
KEYS = ["single_market_only", "cons_market_only", "single_blend_wf", "cons_blend_wf", "dispersion_filter",
        "best_price"]


def season(s):
    return f"{s}/{(int(s) + 1) % 100:02d}"


def decisions(r) -> dict:
    out = {}
    for k in KEYS:
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
    r = C.read_json(RES / "consensus.json")
    info = C.read_json(RES / "run_info.json")
    pre = C.read_json(RES / "preregistration.json")
    cov = pd.read_csv(RES / "coverage.csv")
    av = pd.read_csv(RES / "availability.csv")
    dec = decisions(r)
    V, H = r["variants"], r["holm"]["p_holm"]
    names = {**VARIANTS_RU, "baseline_b": "Baseline B"}
    rep = K.Report(OUT, "Market Consensus: несколько букмекеров одновременно", "FootPredict v6.3 — исследование 13",
                   {"Сезоны": "2021/22–2025/26 (выбор правила консенсуса, только рынок), 2026/27 (проверка, Main)",
                    "Матчей 2026/27": f"{int(cov['season_matches'].iloc[0])} в сезоне, {int(cov['played'].iloc[0])} сыграно",
                    "Модель": "prod_20261012", "Данные": f"git {info['git_sha'][:8]}"})
    rep.add(F.H1c("1. Название исследования"),
            P("**Market Consensus** — консенсусная вероятность нескольких букмекеров (среднее, медиана, взвешенный "
              "консенсус), разброс между книгами как признак неопределённости и влияние консенсуса на вероятность "
              "модели, EV, выбор Main и калибровку FootPredict."))
    rep.add(F.H1c("2. Постановка проблемы"),
            P("Исследование 3 использует одну книгу (почти всегда bet365). Одна книга несёт свой шум и свою маржу; "
              "консенсус нескольких книг теоретически точнее. Но консенсус честен только если котировки одновременны: "
              "смешивание цены bet365 утра со вчерашней ценой Pinnacle даёт ложную «точность». Поэтому каждая книга "
              "входит своей последней котировкой не позже момента решения, разрыв во времени между книгами "
              "записывается, а многокнижные снимки Football-Data (все книги одного момента) используются там, где "
              "хранимых котировок мало."))
    rep.add(F.H1c("3. Гипотезы"),
            *bullets(["**H1.** Консенсус нескольких книг точнее одной книги (log loss).",
                      "**H2.** Разброс между книгами — признак неопределённости: при большом разбросе консенсус хуже "
                      "калиброван (избыточный log loss растёт, наклон калибровки < 1).",
                      "**H3.** Объединение модели с консенсусом точнее объединения с одной книгой.",
                      "**H4.** Консенсус в правиле Main (вместо модели или в смеси с ней) повышает ROI Main.",
                      "**H5.** Исключение кандидатов с большим разбросом книг или лучшая цена среди книг повышают ROI."]))
    # 4 ------------------------------------------------------------------------------------------
    rows = [["Книга", "Момент", "1X2", "Тоталы", "Форы", "Угловые", "Обе забьют"]]
    for u in ("t09", "close"):
        for b, g in r["availability"][u].items():
            if max(g.values()) < 0.01:
                continue
            rows.append([b, u] + [F.pp(g.get(x), 1, False) + "%" for x in ("1X2", "TOTAL", "HANDICAP", "CORNERS", "BTTS")])
    fdav = r["availability"]["football_data_2026"]
    rows2 = [["Снимок Football-Data", "Книга", "1X2", "Тотал 2.5", "Аз. линия (половинная)"]]
    for k, g in sorted(fdav.items()):
        snap, b = k.split("|")
        rows2.append(["открытие" if snap == "open" else "закрытие", b] +
                     [F.pp(g.get(x), 1, False) + "%" for x in ("1X2", "TOTAL_2.5", "AH")])
    bpm = {u: r["availability"][f"{u}_books_per_market"] for u in ("t09", "close")}
    rep.add(F.H1c("4. Исходные данные"),
            P("Хранимые котировки сезона 2026/27 (`odds_long` общего пересчёта): OddsPapi bet365, Pinnacle, William "
              "Hill; The Odds API (несколько британских книг с октября); Football-Data Bet365 (объединена с bet365 "
              "OddsPapi как одна книга). Многокнижные снимки Football-Data (`local_data/results/<сезон>/*.csv`) — "
              "2021/22–2026/27, открытие и закрытие."),
            F.T(rows),
            P("Таблица 1. Фактическая доступность полных книг (доля сыгранных матчей 2026/27, где книга котирует все "
              "исходы рынка не позже момента): t09 — момент T, close — перед началом.", "small"),
            P("Полных книг на рынок (1X2, тоталы, форы): T — " + ", ".join(f"{k} книга: {v}" for k, v in bpm["t09"].items())
              + "; close — " + ", ".join(f"{k} книга: {v}" for k, v in bpm["close"].items()) + "."),
            F.T(rows2),
            P("Таблица 2. Книги снимков Football-Data в 2026/27 (доля сыгранных матчей). В 2026/27 Pinnacle в файлах "
              "уже нет; тотал и азиатская линия — только Bet365 и сводные Max / Avg / биржа.", "small"))
    piv = av.loc[av["moment"].isin(["t09", "close"]) & (
        ((av["book"] == "bet365") & (av["group"] == "1X2")) | ((av["book"] == "pinnacle") & (av["group"] == "HANDICAP"))
        | ((av["book"] == "williamhill") & (av["group"] == "1X2")))]
    piv = piv.assign(col=piv["book"] + " " + piv["group"] + " " + piv["moment"])
    pv = piv.pivot_table(index="league", columns="col", values="share", aggfunc="first").fillna(0)
    nfd = av.loc[(av["moment"] == "fd_open") & (av["group"] == "1X2") &
                 ~av["book"].isin(["max", "avg", "betfair_exchange"]) & (av["share"] >= 0.9)].groupby("league").size()
    cols = [c for c in ["bet365 1X2 t09", "bet365 1X2 close", "pinnacle HANDICAP t09", "pinnacle HANDICAP close",
                        "williamhill 1X2 t09", "williamhill 1X2 close"] if c in pv.columns]
    rows3 = [["Лига"] + [c.replace("HANDICAP", "форы") for c in cols] + ["Книг FD 1X2"]]
    for lg_, x in pv.iterrows():
        rows3.append([lg_] + [F.pp(x[c], 0, False) + "%" for c in cols] + [int(nfd.get(lg_, 0))])
    rep.add(F.T(rows3), P("Таблица 3. Доступность по лигам: доля матчей с полной книгой; последняя колонка — число "
                          "именованных книг снимка открытия Football-Data 1X2 с охватом ≥ 90 %.", "small"))
    # 5 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("5. Методика"),
            P("Вероятность каждой книги — степенное снятие маржи полной книги (как в исследовании 3). Консенсусы по "
              "именованным книгам (без Max / Avg и биржи):"),
            formula("среднее: q̄_{i} = (1/K) Σ_{k} q_{ki};   медиана: med_{k} q_{ki} (нормировка Σ = 1)"),
            formula("среднее логитов: q_{i} ∝ exp((1/K) Σ_{k} log q_{ki});   взвешенное: w_{k} ∝ 1 / маржа_{k}"),
            P("Плюс справочные: средняя (Avg) и максимальная (Max) цена Football-Data, Betfair Exchange. Правило "
              "консенсуса выбирается по log loss 2021/22–2025/26 (матчи с ≥ 3 книгами 1X2) и проверяется на "
              "2026/27. Разброс — среднее по исходам стандартного отклонения q между книгами."),
            P("**Время.** Хранимые котировки: у каждой книги — её последняя котировка не позже момента решения; "
              "разрыв = max − min времени котировок книг рынка. Снимки Football-Data одновременны внутри снимка; "
              "для мира t09 используется снимок открытия (доступен к T по тому же допущению, что и Football-Data "
              "Bet365 в replay), для мира close — снимок закрытия (последние цены до начала)."),
            P("**q_cons кандидата Main**: консенсус снимка Football-Data (1X2 и двойной шанс — медиана именованных "
              f"книг, правило «{r['rule_1x2']['consensus_used']}»; тотал 2.5 и азиатская линия — Avg Football-Data), "
              "иначе медиана ≥ 2 хранимых книг, иначе одна книга (как в исследовании 3)."),
            P("Смеси модели — помесячное расширяющееся окно (как в исследовании 3). Варианты Main: C1 — консенсус "
              "вместо p; C2 — смесь модели с консенсусом; S1/S2 — то же с одной книгой (контроль, = V3/V1 "
              "исследования 3); C3 — исключение кандидатов с разбросом выше δ (δ — квантили 50/75/90 % разброса прошлых "
              "месяцев или ∞, по ROI прошлых месяцев); C4 — правило продакшна с лучшей ценой среди именованных книг в "
              f"момент решения. Парный бутстреп по матчам ({info['params']['n_boot']} повторов), Холм по "
              f"{r['holm']['n_tests']} сравнениям."))
    rep.add(F.H1c("6. Эксперименты"),
            *bullets([f"{len(RULES)} правил консенсуса × 3 рынка × 2 снимка × {len(SEASONS)} сезонов Football-Data "
                      "(только рынок).",
                      "Разброс книг по квинтилям: избыточный log loss и наклон калибровки, все сезоны.",
                      "Консенсус хранимых котировок в момент решения: охват, разрыв по времени, чувствительность.",
                      "Модель + консенсус: log loss вне выборки (сентябрь–ноябрь), калибровка, EV Main.",
                      f"{len(KEYS)} вариантов Main × 2 мира, плюс 4 сравнения консенсус против одной книги."]))
    # 7 ------------------------------------------------------------------------------------------
    rep.add(F.H1c("7. Результаты"), H2("7.1. Правила консенсуса (Football-Data, только рынок)"))
    sel = r["fd_rules"]["selection"]
    rows = [["Рынок | снимок"] + [r["rules_ru"][x] for x in RULES if x not in ("pinnacle", "exchange")] + ["Лучшее"]]
    for k, v in sel.items():
        ll = v["rules_ll_2021_2025"]
        rows.append([k.replace("|", " ")] + [F.num(ll.get(x), 4) for x in RULES if x not in ("pinnacle", "exchange")] +
                    [r["rules_ru"][v["chosen"]]])
    rep.add(F.T(rows), P("Таблица 4. Log loss правил консенсуса на 2021/22–2025/26 (выбор правила).", "small"))
    rows = [["Рынок | снимок", "Матчей", "Медиана − Bet365", "Среднее − Bet365", "Avg FD − Bet365", "Биржа − Bet365"]]
    for k, v in r["fd_rules"]["by_key"].items():
        if not k.endswith("2026"):
            continue

        def d(x):
            y = v.get(x)
            if not y or not y.get("minus_bet365"):
                return "—"
            m = y["minus_bet365"]
            if abs(m["diff"]) < 1e-9 and max(abs(c) for c in m["ci95"]) < 1e-9:
                return "= Bet365"
            return f"{m['diff'] * 1000:+.2f} {F.ci(m['ci95'], 2, 1000)}"
        rows.append([k.replace("|2026", "").replace("|", " "), v["n"], d("median"), d("mean"), d("fd_avg"), d("exchange")])
    rep.add(F.T(rows), P("Таблица 5. Проверка на 2026/27: разница log loss консенсуса и Bet365, ×10⁻³ [95 % ДИ] "
                         "(отрицательная — консенсус точнее). Для тотала и линии в 2026/27 именованная книга одна, "
                         "поэтому медиана = Bet365.", "small"))
    rows = [["Снимок", "Квинтиль разброса", "Сезонов", "Разброс", "Избыточный log loss ×10⁻³ (сред.)",
             "Наклон калибровки (сред.)"]]
    for snap in ("open", "close"):
        for q in "12345":
            xs = [r["fd_rules"]["dispersion"][f"{snap}|{s}"][q] for s in SEASONS
                  if q in r["fd_rules"]["dispersion"].get(f"{snap}|{s}", {})]
            if not xs:
                continue
            rows.append(["открытие" if snap == "open" else "закрытие", q, len(xs),
                         f"{sum(x['disp_mean'] for x in xs) / len(xs):.4f}",
                         f"{sum(x['excess']['diff'] for x in xs) / len(xs) * 1000:+.1f}",
                         f"{sum(x['calib_slope'] for x in xs) / len(xs):.2f}"])
    rep.add(H2("7.2. Разброс между книгами как признак неопределённости"), F.T(rows),
            P("Таблица 6. 1X2, медианный консенсус, матчи с ≥ 3 книгами; среднее по сезонам 2021/22–2026/27. "
              "Избыточный log loss = log loss − энтропия консенсуса (0 у идеально калиброванного прогноза); наклон "
              "калибровки 1 — идеальная калибровка.", "small"))
    dt = r["decision_time"]
    rows = [["Мир", "Рынков с ≥ 2 книгами", "Матчей", "1X2", "Тоталы", "Форы", "Разрыв, ч: медиана", "90 %"]]
    for u in ("t09", "close"):
        x = dt[u]
        rows.append([u, x["markets_multi_book"], x["matches_multi_book"], x["by_group"].get("1X2", 0),
                     x["by_group"].get("TOTAL", 0), x["by_group"].get("HANDICAP", 0),
                     f"{x['gap_quantiles_h']['0.5']:.1f}", f"{x['gap_quantiles_h']['0.9']:.1f}"])
    rows2 = [["Мир | группа", "Рынков", "Все"] + GAP_LABELS]
    for u in ("t09", "close"):
        for g, v in r["gap_sensitivity"][u].items():
            def c(x):
                y = v.get(x)
                if not y or y.get("diff") is None:
                    return f"n={y.get('n', 0)}" if y else "—"
                return f"{y['diff'] * 1000:+.1f} (n={y['n']})"
            rows2.append([f"{u} | {GROUP_RU.get(g, g)}", v["n"], c("all")] + [c(lab) for lab in GAP_LABELS])
    rep.add(H2("7.3. Консенсус хранимых котировок в момент решения и разрыв во времени"), F.T(rows),
            P("Таблица 7. Сколько рынков имеют ≥ 2 полные книги в момент решения и насколько разнесены их котировки "
              "во времени.", "small"), F.T(rows2),
            P("Таблица 8. Медианный консенсус минус bet365, log loss ×10⁻³, по разрыву между книгами (группы с < 30 "
              "рынками показаны только числом).", "small"))
    for i, u in enumerate(("close", "t09")):
        wf = r["model"][u]["walk_forward"]
        rows = [["Группа", "Рынков", "Модель", "Одна книга", "Консенсус", "Смесь (книга)", "Смесь (консенсус)",
                 "Консенсус − книга ×10⁻³", "Смесь конс. − смесь книги ×10⁻³"]]
        for g, v in wf.items():
            rows.append([GROUP_RU.get(g, g), v["n"], F.num(v["model"], 4), F.num(v["single"], 4),
                         F.num(v["consensus"], 4), F.num(v["blend_single"], 4), F.num(v["blend_consensus"], 4),
                         f"{v['consensus_minus_single']['diff'] * 1000:+.2f} {F.ci(v['consensus_minus_single']['ci95'], 2, 1000)}",
                         f"{v['blend_cons_minus_blend_single']['diff'] * 1000:+.2f} "
                         f"{F.ci(v['blend_cons_minus_blend_single']['ci95'], 2, 1000)}"])
        if i == 0:
            rep.add(H2("7.4. Консенсус и вероятность модели"))
        rep.add(F.T(rows), P(f"Таблица {9 + i}. Вне выборки (сентябрь–ноябрь), мир {u}: log loss модели, одной "
                              "книги, консенсуса и смесей с моделью на одних и тех же рынках.", "small"))
    rows = [["Мир", "Вероятность", "ECE, п.п.", "Brier", "Исходов"]]
    for u in ("close", "t09"):
        for k, lab in (("model", "модель"), ("single", "одна книга"), ("consensus", "консенсус")):
            c = r["model"][u]["calibration"][k]
            rows.append([u, lab, f"{c['ece'] * 100:.2f}", f"{c['brier']:.4f}", c["n"]])
    rep.add(F.T(rows), P("Таблица 11. Калибровка по всем исходам кандидатов (бинарно) с обеими вероятностями рынка.",
                         "small"))
    rep.add(H2("7.5. Main с консенсусом"))
    for i, u in enumerate(("close", "t09")):
        rep.add(F.T(F.variant_rows(V[u], names, u, H, ["baseline_b"] + KEYS, r["baseline_a"])),
                P(f"Таблица {12 + i}. Варианты Main, мир {u}.", "small"))
    rows = [["Сравнение", "Мир", "Δ ROI, п.п. [95 % ДИ]", "p", "p Холма", "Δ hit, п.п.", "Тот же Main"]]
    for u in ("close", "t09"):
        for k, v in V[u]["cons_vs_single"].items():
            rows.append([("C1 − S1 (рынок вместо p)" if k == "market_only" else "C2 − S2 (смесь)"), u,
                         f"{F.pp(v['roi_diff']['diff'])} {F.ci(v['roi_diff']['ci95'])}", F.pval(v["roi_diff"]["p"]),
                         F.pval(H.get(f"cons_vs_single_{k}|{u}")), F.pp(v["hit_rate_diff"]["diff"]),
                         F.pp(v["same_main_share"], 0, False) + "%"])
    rep.add(F.T(rows), P("Таблица 14. Консенсус против одной книги в том же правиле (парно).", "small"))
    rows = [["Мир", "Ставок B", "С консенсусом", "EV модели, %", "EV одной книги, %", "EV консенсуса, %",
             "EV модели > 0", "EV конс. > 0", "ROI при EV конс. ≤ 0, %"]]
    for u in ("close", "t09"):
        e = r["extra"][u]["ev_baseline_picks"]
        rows.append([u, e["n"], e["with_consensus"], F.pp(e["mean_ev_model"], 2), F.pp(e["mean_ev_single"], 2),
                     F.pp(e["mean_ev_consensus"], 2), F.pp(e["share_pos_model"], 0, False) + "%",
                     F.pp(e["share_pos_consensus"], 1, False) + "%", F.roi_cell(e["roi_ev_cons_neg"])])
    rep.add(F.T(rows), P("Таблица 15. EV Main-ставок Baseline B по модели, одной книге и консенсусу. Источник q_cons "
                         "этих ставок — close: " + ", ".join(f"{k} {v}" for k, v in
                                                             r["extra"]["close"]["ev_baseline_picks"]["sources"].items())
                         + "; t09: " + ", ".join(f"{k} {v}" for k, v in
                                                 r["extra"]["t09"]["ev_baseline_picks"]["sources"].items()) + ".",
                         "small"))
    rows = [["Мир", "Разброс книг", "Ставок", "Hit, %", "Средняя p, %", "ROI, % [95 % ДИ]"]]
    for u in ("close", "t09"):
        for k, v in r["extra"][u].get("roi_by_dispersion", {}).items():
            rows.append([u, k, v["n"], F.pp(v["hit"], 1, False), F.pp(v["mean_p"], 1, False), F.roi_cell(v)])
    bu = {u: r["extra"][u]["best_price_uplift"] for u in ("close", "t09")}
    rep.add(F.T(rows), P("Таблица 16. Main Baseline B на рынках с разбросом нескольких книг, по терцилям разброса "
                         f"(малые выборки). Лучшая цена среди книг (C4) выше цены продакшна у "
                         f"{bu['close']['mains_with_better_price']} Main в close и {bu['t09']['mains_with_better_price']} "
                         f"в t09; средний log-прирост цены {bu['close']['mean_log_ratio']:+.4f} / "
                         f"{bu['t09']['mean_log_ratio']:+.4f}.", "small"))
    rep.add(F.H1c("8. Графики"),
            *figure(FIG / "fd_rules_1x2.png", "Рис. 1. 1X2 Football-Data: медиана книг и Avg минус Bet365 по сезонам "
                                              "(log loss ×10⁻³, 95 % ДИ)."),
            *figure(FIG / "dispersion_quintiles.png", "Рис. 2. Разброс книг: избыточный log loss и наклон калибровки "
                                                      "консенсуса по квинтилям (среднее по сезонам)."),
            *figure(FIG / "book_time_gap.png", "Рис. 3. Разрыв во времени между котировками книг одного рынка "
                                               "(хранимые котировки, рынки с ≥ 2 книгами)."),
            *figure(FIG / "calibration.png", "Рис. 4. Калибровка: модель, одна книга, консенсус."),
            *figure(FIG / "variants_vs_baseline_b.png", "Рис. 5. Разница ROI Main вариантов с Baseline B (95 % ДИ)."))
    rep.add(F.H1c("9. Статистический анализ"),
            P(f"{r['holm']['n_tests']} парных сравнений ROI (6 вариантов × 2 мира + 4 сравнения консенсус против одной "
              f"книги); минимальное p Холма {F.pval(min(H.values()))}."),
            F.T(F.stability_rows(V["close"], names, ["baseline_b"] + KEYS)),
            P("Таблица 17. Устойчивость, мир close.", "small"),
            F.T(F.month_rows(V["close"], "cons_blend_wf", "C2")),
            P("Таблица 18. C2 по месяцам (close); «мало» — меньше 100 ставок.", "small"),
            F.T(F.league_rows(V["close"], "cons_blend_wf", "C2")),
            P("Таблица 19. C2 по лигам (close); * — меньше 50 ставок.", "small"))
    w = r["model"]["close"]["walk_forward"]["1X2"]
    sel1 = sel.get("1X2|open", {})
    rep.add(F.H1c("10. Выводы"),
            *bullets([f"H1 подтверждена слабо: на 2021/22–2025/26 лучшее правило 1X2 (открытие) — "
                      f"«{r['rules_ru'].get(sel1.get('chosen'), '—')}», но выигрыш над Bet365 — доли тысячной log loss; "
                      "на 2026/27 то же (таблица 5). В момент решения хранимых книг мало, и они разнесены во времени.",
                      "H2 не подтверждена: при большом разбросе книг консенсус не хуже калиброван (наклон ≈ 1, "
                      "избыточный log loss не растёт) — разброс отражает разные маржи и моменты, а не неопределённость "
                      "исхода.",
                      f"H3 подтверждена частично: в 1X2 close смесь модели с консенсусом точнее смеси с одной книгой "
                      f"({w['blend_cons_minus_blend_single']['diff'] * 1000:+.2f}×10⁻³ "
                      f"{F.ci(w['blend_cons_minus_blend_single']['ci95'], 2, 1000)}), в других группах разницы нет.",
                      f"H4: варианты с консенсусом лучше своих одно-книжных аналогов (C2 − S2 close "
                      f"{F.pp(V['close']['cons_vs_single']['blend']['roi_diff']['diff'])} п.п., t09 "
                      f"{F.pp(V['t09']['cons_vs_single']['blend']['roi_diff']['diff'])} п.п.), но против Baseline B "
                      f"разница незначима (C2 close {F.pp(V['close']['cons_blend_wf']['vs_b']['roi_diff']['diff'])} п.п., "
                      f"t09 {F.pp(V['t09']['cons_blend_wf']['vs_b']['roi_diff']['diff'])} п.п.).",
                      "H5 не подтверждена: фильтр разброса и лучшая цена меняют ROI в пределах долей процента.",
                      "EV консенсуса почти у всех Main отрицателен (≈ −маржа): по рынку правило Main берёт ставки с "
                      "отрицательным ожиданием, положительный EV модели — следствие её переоценки."]))
    rows = [["Вариант", "Решение", "Основание"]]
    for k, d in dec.items():
        rows.append([VARIANTS_RU[k], d, f"Δ close {F.pp(V['close'][k]['vs_b']['roi_diff']['diff'])}, t09 "
                                        f"{F.pp(V['t09'][k]['vs_b']['roi_diff']['diff'])} п.п.; p Холма "
                                        f"{F.pval(H.get(k + '|close'))} / {F.pval(H.get(k + '|t09'))}"])
    rep.add(F.H1c("11. Рекомендации (A / B / C)"),
            P("Единое правило решения трёх рыночных исследований v6.3: **A** — значимое после Холма улучшение в обоих "
              "мирах; **B** — неотрицательная точечная разница в обоих мирах и прирост хотя бы в одном больше 0.5 п.п.; "
              "**C** — иначе."),
            F.T(rows), P("Таблица 20. Решения по вариантам.", "small"),
            *bullets(["Production не меняется.",
                      "Предрегистрация (`results/preregistration.json`): " + "; ".join(
                          f"{e['id']} (заморожено {e['frozen_at']}, минимум {e['minimum_forward_sample']})"
                          for e in pre["entries"]) + ".",
                      "Если консенсус понадобится в продакшне, нужен источник нескольких книг в момент генерации "
                      "(сейчас это только снимки Football-Data для 1X2 / тотала 2.5 / линии и редкие котировки "
                      "Pinnacle и William Hill)."]))
    rep.add(F.H1c("12. Ограничения"),
            *bullets(["Хранимые котировки — почти только bet365; Pinnacle — в основном форы, William Hill и The Odds API "
                      "— сотни матчей. Полноценный консенсус в момент T возможен только по снимкам Football-Data.",
                      "Время снимка открытия Football-Data неизвестно; допущение «доступен к T» то же, что в replay для "
                      "Football-Data Bet365.",
                      "В 2026/27 Football-Data не публикует Pinnacle и другие именованные книги для тотала и линии — там "
                      "консенсус = Avg Football-Data.",
                      "Медиана хранимых книг разнесена во времени (медианный разрыв — часы); выборки с малым разрывом "
                      "слишком малы для вывода.",
                      "2026/27 уже изучался ранее: хронологическая проверка, не holdout."]))
    rep.add(F.H1c("Как воспроизвести"),
            *K.reproduce([".venv/bin/python -m experiments.v6_3.replay.build      # общий кэш (если его нет)",
                          ".venv/bin/python -m experiments.v6_3.consensus.run     # ~2 мин, без обучения моделей",
                          ".venv/bin/python -m experiments.v6_3.consensus.build_pdf"]),
            P("Результаты: `experiments/v6_3/consensus/results/` (consensus.json, availability.csv, coverage.csv, "
              "per_match_*.csv, preregistration.json, run_info.json).", "small"))
    rep.build()
    print("pages", rep.check())
    findings(r, dec, pre)


def findings(r, dec, pre) -> None:
    V, H = r["variants"], r["holm"]["p_holm"]
    lines = ["# FootPredict v6.3 — Market Consensus: выводы", "",
             "Все числа — из `results/consensus.json` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_3/Market_Consensus_Research.pdf`.", "",
             "## Доступность", "",
             f"- Хранимые котировки (close): bet365 1X2 {F.pp(r['availability']['close']['bet365']['1X2'], 0, False)} % "
             f"матчей; Pinnacle форы {F.pp(r['availability']['close']['pinnacle']['HANDICAP'], 0, False)} %, 1X2 "
             f"{F.pp(r['availability']['close']['pinnacle']['1X2'], 1, False)} %; William Hill 1X2 "
             f"{F.pp(r['availability']['close']['williamhill']['1X2'], 0, False)} %. Многокнижный консенсус в момент T — "
             "только снимки Football-Data.", "",
             "## Main ROI против Baseline B (Δ, п.п. [95 % ДИ], p Холма)", "",
             "| Вариант | close | t09 | Решение |", "|---|---|---|---|"]
    for k, d in dec.items():
        cells = [f"{F.pp(V[u][k]['vs_b']['roi_diff']['diff'])} {F.ci(V[u][k]['vs_b']['roi_diff']['ci95'])}, "
                 f"{F.pval(H.get(k + '|' + u))}" for u in ("close", "t09")]
        lines.append(f"| {VARIANTS_RU[k]} | {cells[0]} | {cells[1]} | {d} |")
    lines += ["", "Консенсус против одной книги: " + "; ".join(
        f"{k} {u} {F.pp(V[u]['cons_vs_single'][k]['roi_diff']['diff'])} п.п. (p {F.pval(V[u]['cons_vs_single'][k]['roi_diff']['p'])}, "
        f"Холм {F.pval(H.get(f'cons_vs_single_{k}|{u}'))})" for u in ("close", "t09") for k in ("market_only", "blend")),
        "", "## Решения", "", "- A: нет. Production не меняется.",
        "- B: " + ", ".join(VARIANTS_RU[k] for k, d in dec.items() if d == "B") + " — `results/preregistration.json` ("
        + ", ".join(e["id"] for e in pre["entries"]) + ").",
        "- C: остальные; разброс книг не является признаком неопределённости исхода.",
        "- Если B пройдёт форвард: `src/markets/recommendation.py` — вероятность Main из пула модели с консенсусом "
        "книг; нужен источник нескольких книг в момент генерации (`src/providers`).", "",
        "## Воспроизведение", "", "```", ".venv/bin/python -m experiments.v6_3.consensus.run",
        ".venv/bin/python -m experiments.v6_3.consensus.build_pdf", "```", ""]
    (RES.parent / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
