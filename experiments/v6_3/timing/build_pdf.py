"""Builds reports/v6_3/Timing_CLV_Research.pdf and findings.md from results/ (no number typed by hand).

    .venv/bin/python -m experiments.v6_3.timing.build_pdf
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.market_aware import fmt as F
from scripts import report_kit as K
from scripts.report_kit import H2, P, bullets, figure, formula

from .run import FIG, RES

OUT = C.ROOT / "reports" / "v6_3" / "Timing_CLV_Research.pdf"
MOMENT_RU = {"early": "за 48 ч до начала", "t09": "момент T (ежедневный запуск)", "close": "закрытие"}
GROUP_RU = {"1X2": "1X2", "TOTAL": "Тоталы", "CORNERS": "Угловые", "BTTS": "Обе забьют", "HANDICAP": "Азиатские форы",
            "DOUBLE_CHANCE": "Двойной шанс", "all": "все"}


def sl(v):
    return f"{v['slope']:+.3f} {F.ci(v['ci95'], 3, 1)}" if v.get("slope") is not None else "—"


def build() -> None:
    r = C.read_json(RES / "timing.json")
    info = C.read_json(RES / "run_info.json")
    cov = pd.read_csv(RES / "coverage.csv")
    mo, sp, clv, mv, lv, src = (r["moments"], r["same_picks"], r["clv"], r["movement"], r["live_versions"],
                                r["sources"])
    ph = r["holm"]["p_holm"]
    a = r["archive"]["books"]
    rep = K.Report(OUT, "Timing & Closing Line Value: время цены и Main", "FootPredict v6.3 — исследование 5",
                   {"Сезон": "2026/27 (полный пересчёт; не holdout)",
                    "Матчей": f"{int(cov['season_matches'].iloc[0])} в сезоне, {int(cov['played'].iloc[0])} сыграно",
                    "Модель": "prod_20261012", "Данные": f"git {info['git_sha'][:8]}"})
    rep.add(F.H1c("1. Название исследования"),
            P("**Timing & Closing Line Value** — влияние момента получения коэффициента и момента формирования Main на "
              "точность и доходность FootPredict: ранние цены, цены в момент прогноза, более поздние котировки, "
              "закрытие, движение линии, CLV Main-ставок и источник цены."))
    rep.add(F.H1c("2. Постановка проблемы"),
            P("FootPredict обновляется раз в день (около 09:00 UK) и показывает Main с ценой, существовавшей в момент "
              "генерации. Две вещи важны для владельца: (1) честность — цена, появившаяся позже, не может считаться "
              "доступной раньше; (2) может ли сам момент формирования Main (раньше или позже) менять ROI. Закрытие — "
              "эталон эффективности рынка (CLV), но только как диагностика: в момент утреннего запуска цены закрытия "
              "ещё нет. В v6.2 (Odds Timing) это изучалось на истории сайта; v6.3 повторяет вопрос на едином "
              "пересчёте сезона (Baseline B) и добавляет живые версии прогнозов с октября."))
    rep.add(F.H1c("3. Гипотезы"),
            *bullets(["**H1.** Цена закрытия точнее цены открытия / момента T (рынок учится к началу матча).",
                      "**H2.** Модель предугадывает движение рынка: расхождение p − q_T положительно связано с "
                      "движением q_close − q_T (модель знает то, что рынок узнаёт позже).",
                      "**H3.** Для Main-исходов цена в момент T в среднем лучше закрытия (положительный ценовой CLV), "
                      "но справедливый CLV (после снятия маржи) отрицателен.",
                      "**H4.** Момент формирования Main (за 48 ч / в T / перед началом) меняет ROI.",
                      "**H5.** Источник цены (OddsPapi, The Odds API, Football-Data) меняет ROI одних и тех же Main."]))
    rows = [["Книга", "Исходов × книга", "Матчей", "1 котировка", "≥ 3 котировок", "Известна к T", "Медиана, ч до начала"]]
    for b, v in a.items():
        if v.get("share_1_quote") is None or v["selections"] < 100:
            continue
        rows.append([b, v["selections"], v["matches"], F.pp(v["share_1_quote"], 0, False) + "%",
                     F.pp(v["share_3plus"], 0, False) + "%", F.pp(v["share_quoted_by_T"], 0, False) + "%",
                     f"{v['lead_h_quantiles']['0.5']:.1f}"])
    rows.append(["Football-Data Bet365", a["fd_bet365"]["selections"], "—", "1 снимок", "—", "условно да",
                 "неизвестно"])
    rep.add(F.H1c("4. Исходные данные"),
            P("Общий пересчёт сезона (`experiments/v6_3/replay`): выходы prod_20261012, кандидаты всех рынков, три "
              "набора цен — ранний (последняя котировка не позже начала − 48 ч), в момент T и закрытие (последняя "
              "котировка до начала); живые версии прогнозов `local_data/predictions/versions.parquet` (только чтение) "
              "с 22.10.2026."),
            F.T(rows), P("Таблица 1. Глубина архива цен: сколько котировок хранится на исход, какая доля известна к "
                         "моменту T, медианное время котировки до начала матча. Книги с менее чем 100 исходами не "
                         "показаны.", "small"),
            P("**Главное ограничение данных.** Архив OddsPapi хранит в основном одну котировку на исход (последнюю "
              f"перед началом): у bet365 одна котировка у {F.pp(a['bet365']['share_1_quote'], 0, False)} % исходов, к "
              f"моменту T известна только {F.pp(a['bet365']['share_quoted_by_T'], 0, False)} %. Поэтому движение линии "
              "внутри OddsPapi видно лишь для исходов, перекотированных после T; полное движение открытие → закрытие "
              "есть в двух снимках Football-Data (все матчи, время открытия не публикуется: пятница/вторник)."),
            F.T([["Момент", "Сыграно", "Обработано", "Ставок (цена в момент)", "Без цены", "PUSH",
                  "Нет статистики"]] +
                [[MOMENT_RU[x["moment"]], x["played"], x["processed"], x["bets_real_price"], x["no_price_at_moment"],
                  x["push"], x["without_statistics"]] for _, x in cov.iterrows()]),
            P(f"Таблица 2. Охват Baseline B по моментам решения. Не сыгранные матчи сезона "
              f"({int(cov['not_played_excluded'].iloc[0])}) не входят в ROI; Main без цены — не ставка.", "small"))
    rep.add(F.H1c("5. Методика"),
            P("Цена используется в момент решения только если `collected_at ≤ момент` (время UK; время генерации живых "
              "версий переведено из UTC в UK). Моменты: **early** = начало − 48 ч; **T** = min(09:00 UK дня матча, "
              "начало − 30 мин); **close** = последняя котировка до начала (для решения в T — только диагностика)."),
            formula("ценовой CLV = odds_{T} / odds_{close} − 1;    справедливый CLV = odds_{T} · q_{close} − (1 − P_{возврат})"),
            P("q_close — вероятность из полной книги закрытия (степенной метод снятия маржи, первая полная книга: bet365 "
              "OddsPapi → Football-Data Bet365 → Pinnacle → William Hill)."),
            formula("q_{close} − q_{T} = α + β · (p − q_{T}) + ε"),
            P("β > 0 означает, что рынок движется в сторону модели. ДИ — кластерный бутстреп по матчам. Сравнение "
              "Main разных моментов — парный бутстреп ROI по матчам; поправка Холма по всем "
              f"{r['holm']['n_tests']} проверкам ROI исследования."))
    rep.add(F.H1c("6. Эксперименты"),
            *bullets(["Baseline B построен трижды движком продакшна (early, T, close) с ценами своего момента; "
                      f"самопроверка: Main T и close совпадают с общим Baseline B в "
                      f"{F.pp(mo['self_check_same_main']['t09'], 2, False)} % и "
                      f"{F.pp(mo['self_check_same_main']['close'], 2, False)} % матчей.",
                      "Одни и те же Main момента T оценены по ранней цене, цене T и цене закрытия.",
                      "CLV Main-ставок момента T: по источнику и группе рынков; ROI по знаку CLV.",
                      "Движение рынка и модель: книги OddsPapi, перекотированные после T; Football-Data Bet365 и "
                      "среднее рынка (Avg) открытие → закрытие.",
                      "Живые версии: первая и последняя записанная версия Main каждого матча (цена — только "
                      "существовавшая при генерации).",
                      "Источник цены: те же Main при ценах разных источников в момент T."]))
    rep.add(F.H1c("7. Результаты"), H2("7.1. Main, сформированный в разные моменты"))
    rows = [["Момент", "Ставок", "Hit", "Ср. кэф", "ROI, % [95 % ДИ]", "Ставок*", "ROI*, % [95 % ДИ]"]]
    for m in ("early", "t09", "close"):
        s, s2 = mo["summary"][m], mo["all_priced"][m]
        rows.append([MOMENT_RU[m], s["bets"], F.pp(s["hit_rate"], 1, False), F.num(s["avg_odds"], 2),
                     F.roi_cell(s["roi_ci"]), s2["bets"], F.roi_cell(s2["roi_ci"])])
    rows.append(["Baseline A (сайт)", r["baseline_a"]["summary"]["bets"],
                 F.pp(r["baseline_a"]["summary"]["hit_rate"], 1, False), F.num(r["baseline_a"]["summary"]["avg_odds"], 2),
                 F.roi_cell(r["baseline_a"]["roi_ci"]), "—", "—"])
    rep.add(F.T(rows), P(f"Таблица 3. Baseline B по моментам решения. * — только {mo['all_priced_matches']} матчей, где "
                         "у Main есть цена во всех трёх моментах (одинаковая выборка).", "small"))
    rows = [["Сравнение", "Матчей", "Δ ROI, п.п. [95 % ДИ]", "p", "p Холма", "Тот же Main"]]
    for k, v in mo["pairs"].items():
        rows.append([k.replace("_vs_", " − "), v["matches"], f"{F.pp(v['roi_diff']['diff'])} {F.ci(v['roi_diff']['ci95'])}",
                     F.pval(v["roi_diff"]["p"]), F.pval(ph.get(k)), F.pp(v["same_main_share"], 0, False) + "%"])
    for k in ("t09_vs_early", "close_vs_t09"):
        v = mo["all_priced"][k]
        rows.append([k.replace("_vs_", " − ") + " *", v["matches"],
                     f"{F.pp(v['roi_diff']['diff'])} {F.ci(v['roi_diff']['ci95'])}", F.pval(v["roi_diff"]["p"]),
                     F.pval(ph.get("all_priced|" + k)), F.pp(v["same_main_share"], 0, False) + "%"])
    rep.add(F.T(rows), P("Таблица 4. Парные сравнения моментов (ROI каждого — по его собственным ставкам на общих "
                         "матчах). * — матчи с ценой во всех трёх моментах.", "small"),
            P("Различия «все матчи» смешивают два эффекта: момент решения и то, какие Main вообще имеют цену (в T у "
              "четверти Main цены ещё нет, а у раннего момента — у большинства). Честное сравнение — на матчах с ценой "
              "во всех трёх моментах (*): там разница в пределах шума."))
    t = sp["t09+close"]
    rows = [["Набор", "Ставок", "ROI по цене T, %", "ROI по закрытию, %", "Δ закрытие − T, п.п. [ДИ]", "p",
             "Закрытие ниже / выше"]]
    rows.append(["Main T с ценой T и закрытия", t["picks"], F.pp(t["t09"]["roi"]), F.pp(t["close"]["roi"]),
                 f"{F.pp(t['close_minus_t09']['roi_diff']['diff'], 2)} {F.ci(t['close_minus_t09']['roi_diff']['ci95'], 2)}",
                 F.pval(t["close_minus_t09"]["roi_diff"]["p"]),
                 f"{F.pp(t['share_close_lower'], 0, False)}% / {F.pp(t['share_close_higher'], 0, False)}%"])
    t3 = sp["early+t09+close"]
    rows.append(["… и с ранней ценой", t3["picks"], F.pp(t3["t09"]["roi"]), F.pp(t3["close"]["roi"]),
                 f"{F.pp(t3['close_minus_t09']['roi_diff']['diff'], 2)} {F.ci(t3['close_minus_t09']['roi_diff']['ci95'], 2)}",
                 F.pval(t3["close_minus_t09"]["roi_diff"]["p"]),
                 f"{F.pp(t3['share_close_lower'], 0, False)}% / {F.pp(t3['share_close_higher'], 0, False)}%"])
    rep.add(H2("7.2. Одни и те же Main по разным ценам"), F.T(rows),
            P(f"Таблица 5. Main момента T, оценённые по цене T и по цене закрытия (диагностика). Для набора с ранней "
              f"ценой: ROI по ранней цене {F.pp(t3['early']['roi'])} %, разница T − ранняя "
              f"{F.pp(t3['t09_minus_early']['roi_diff']['diff'], 2)} п.п. "
              f"{F.ci(t3['t09_minus_early']['roi_diff']['ci95'], 2)}.", "small"))
    rows = [["Показатель", "Среднее", "95 % ДИ", "Доля > 0", "Ставок"]]
    for k, lab in (("clv_price", "ценовой CLV"), ("clv_fair", "справедливый CLV")):
        rows.append([lab, F.pp(clv[k]["mean"], 2) + "%", F.ci(clv[k]["ci95"], 2), F.pp(clv[k]["share_positive"], 0, False) + "%",
                     clv[k]["n"]])
    rep.add(H2("7.3. Closing Line Value Main-ставок момента T"), F.T(rows),
            P("Таблица 6. CLV ставок Main момента T (каждая ставка имеет цену закрытия и полную книгу закрытия).",
              "small"))
    rows = [["Срез", "Ставок", "Ценовой CLV, %", "Справедливый CLV, %", "ROI, %"]]
    for key in ("by_provider", "by_group"):
        for k, v in clv[key].items():
            rows.append([("источник: " if key == "by_provider" else "") + GROUP_RU.get(k, k), v["n"],
                         F.pp(v["clv_price"], 2), F.pp(v["clv_fair"], 2), F.pp(v["roi"])])
    for k in ("clv_fair>0", "clv_fair<=0"):
        v = clv[k]
        rows.append([f"справедливый CLV {'> 0' if k.endswith('>0') else '≤ 0'}", v["n"], "—", "—", F.roi_cell(v)])
    rep.add(F.T(rows), P("Таблица 7. CLV и ROI по источнику цены T, группе рынков и знаку справедливого CLV "
                         f"(корреляция CLV с прибылью ставки {clv['corr_clv_profit']:.3f}; знак CLV известен только "
                         "после закрытия — это диагностика, не правило отбора).", "small"))
    rows = [["Источник / группа", "Наблюдений", "β [95 % ДИ]", "p", "|Δq| среднее, п.п.", "Закрытие − открытие, ×10⁻³ log loss"]]
    for g, v in mv["oddspapi"].items():
        rows.append([f"OddsPapi, {GROUP_RU.get(g, g)}", v["n"], sl(v), F.pval(v.get("p")),
                     F.pp(v.get("mean_abs_move"), 2, False), "—"])
    for k, v in mv["football_data"].items():
        b, m = k.split("|")
        ll = v.get("ll_close_minus_open")
        rows.append([f"Football-Data {b}, {m}", v["n"], sl(v), F.pval(v.get("p")), F.pp(v.get("mean_abs_move"), 2, False),
                     f"{ll['diff'] * 1000:+.2f} {F.ci(ll['ci95'], 2, 1000)}" if ll else "—"])
    rep.add(H2("7.4. Движение линии и модель"), F.T(rows),
            P(f"Таблица 8. Наклон β движения рынка на расхождение модели с рынком. OddsPapi: книги, полные в T и в "
              f"закрытии ({mv['oddspapi_books_complete_at_T_and_close']}), из них перекотированы после T — "
              f"{mv['requoted_after_T']} (только они в таблице). Football-Data: открытие → закрытие, все матчи; "
              "для 1X2 — исходы H и A.", "small"))
    sp_ = lv["same_pick_price_change"]
    rows = [["Версия", "Ставок", "Hit", "Ср. кэф", "ROI, % [95 % ДИ]", "Медиана, ч до начала"],
            ["первая", lv["first"]["bets"], F.pp(lv["first"]["hit_rate"], 1, False), F.num(lv["first"]["avg_odds"], 2),
             F.roi_cell(lv["first"]["roi_ci"]), f"{lv['lead_h_first_median']:.1f}"],
            ["последняя", lv["last"]["bets"], F.pp(lv["last"]["hit_rate"], 1, False), F.num(lv["last"]["avg_odds"], 2),
             F.roi_cell(lv["last"]["roi_ci"]), f"{lv['lead_h_last_median']:.1f}"]]
    rep.add(H2("7.5. Живые версии прогнозов (с 22.10.2026)"), F.T(rows),
            P(f"Таблица 9. Матчи с ≥ 2 живыми версиями до начала: {lv['matches_2plus']} из "
              f"{lv['matches_with_versions']}, сыграно {lv['settled_pairs']}. Main совпадает в "
              f"{F.pp(lv['same_main_share'], 0, False)} % пар; последняя − первая: Δ ROI "
              f"{F.pp(lv['last_vs_first']['roi_diff']['diff'])} п.п. {F.ci(lv['last_vs_first']['roi_diff']['ci95'])} "
              f"(p = {F.pval(lv['last_vs_first']['roi_diff']['p'])}, Холм {F.pval(ph.get('live|last_vs_first'))}); на "
              f"{lv['both_priced']['n']} матчах с ценой в обеих версиях Δ "
              f"{F.pp(lv['both_priced']['last_vs_first']['roi_diff']['diff'])} п.п. "
              f"{F.ci(lv['both_priced']['last_vs_first']['roi_diff']['ci95'])}. Тот же Main: цена последней версии "
              f"ниже в {F.pp(sp_['share_last_lower'], 0, False)} %, выше в {F.pp(sp_['share_last_higher'], 0, False)} % "
              f"случаев (n = {sp_['n']}). Цена существовала при генерации у "
              f"{F.pp(lv['price_existed_share_first'], 1, False)} % первых и "
              f"{F.pp(lv['price_existed_share_last'], 1, False)} % последних версий.", "small"))
    rows = [["Источник цены Main в T", "Ставок", "Hit", "Ср. кэф", "ROI, % [95 % ДИ]"]]
    for k, v in src["main_by_provider"].items():
        if v["bets"]:
            rows.append([k, v["bets"], F.pp(v["hit_rate"], 1, False), F.num(v["avg_odds"], 2), F.roi_cell(v["roi_ci"])])
    rows2 = [["Пара источников (те же Main)", "Main", "log(o_a/o_b)", "ROI a, %", "ROI b, %", "Δ, п.п. [ДИ]"]]
    for k, v in src["pairs"].items():
        rows2.append([k, v["n"], f"{v['mean_log_ratio']:+.4f}", F.pp(v["roi_a"]), F.pp(v["roi_b"]),
                      f"{F.pp(v['diff']['diff'])} {F.ci(v['diff']['ci95'])}"])
    rep.add(H2("7.6. Источник цены"), F.T(rows),
            P("Таблица 10. ROI Main момента T по источнику цены (разные рынки: Football-Data даёт только 1X2, тотал "
              "2.5 и азиатскую линию, поэтому сравнение не парное).", "small"), F.T(rows2),
            P("Таблица 11. Одни и те же Main с ценой двух источников в T (только пары с ≥ 30 Main). Доступность "
              "источников для Main в T: " + ", ".join(f"{k} {v}" for k, v in src["availability_at_T"].items() if v) +
              ".", "small"))
    rep.add(F.H1c("8. Графики"),
            *figure(FIG / "quote_lead_time.png", "Рис. 1. Когда хранятся котировки OddsPapi: время до начала матча."),
            *figure(FIG / "roi_by_moment.png", "Рис. 2. ROI Main, сформированного в разные моменты (95 % ДИ)."),
            *figure(FIG / "clv_distribution.png", "Рис. 3. Распределение CLV Main-ставок момента T."),
            *figure(FIG / "movement_slopes.png", "Рис. 4. Наклон движения рынка на расхождение модели с рынком."))
    rows = [["Проверка", "p", "p Холма"]] + [[k, F.pval(v), F.pval(ph.get(k))] for k, v in r["holm"]["p"].items()]
    rep.add(F.H1c("9. Статистический анализ"), F.T(rows),
            P(f"Таблица 12. Все {r['holm']['n_tests']} проверок ROI исследования с поправкой Холма. Значима после "
              "поправки только разница цены одних и тех же Main (T против закрытия) — это свойство цен, а не правило "
              "выбора. Сравнения моментов решения на одинаковой выборке незначимы."))
    fd1 = mv["football_data"].get("bet365|1X2", {})
    rep.add(F.H1c("10. Выводы"),
            *bullets([f"H1 подтверждена: закрытие точнее открытия (Football-Data Bet365 1X2: "
                      f"{fd1.get('ll_close_minus_open', {}).get('diff', 0) * 1000:+.2f}×10⁻³ log loss).",
                      f"H2 подтверждена: рынок движется в сторону модели (Football-Data Bet365 1X2 β = {sl(fd1)}; "
                      f"OddsPapi все рынки β = {sl(mv['oddspapi']['all'])}). Модель содержит часть информации, которую "
                      "рынок учитывает позже, но эффект мал (несколько процентов расхождения).",
                      f"H3 подтверждена: ценовой CLV Main {F.pp(clv['clv_price']['mean'], 2)} % "
                      f"{F.ci(clv['clv_price']['ci95'], 2)}, справедливый CLV {F.pp(clv['clv_fair']['mean'], 2)} % — "
                      "Main берёт цену лучше закрытия, но не обыгрывает закрытие после маржи.",
                      f"H4 не подтверждена на одинаковой выборке (T − ранний "
                      f"{F.pp(mo['all_priced']['t09_vs_early']['roi_diff']['diff'])} п.п., закрытие − T "
                      f"{F.pp(mo['all_priced']['close_vs_t09']['roi_diff']['diff'])} п.п., оба незначимы); живые версии "
                      "(первая против последней) тоже без значимой разницы.",
                      "H5: различия источников — в основном разные рынки и моменты снимка; на одних и тех же Main "
                      "разница цен мала, кроме Football-Data против OddsPapi bet365 (разные моменты одной книги).",
                      f"Для одних и тех же Main цена T выгоднее закрытия: ROI по закрытию ниже на "
                      f"{F.pp(-t['close_minus_t09']['roi_diff']['diff'], 2, False)} п.п. — утренняя публикация с ценой "
                      "генерации не проигрывает более поздней."]))
    rep.add(F.H1c("11. Рекомендации (A / B / C)"),
            F.T([["Предложение", "Решение", "Основание"],
                 ["Изменить время ежедневного запуска / формировать Main ближе к началу", "C",
                  "нет значимой разницы ROI моментов на одинаковой выборке; живые версии тоже"],
                 ["Пересчитывать Main по цене закрытия", "C", "цена закрытия недоступна в момент прогноза; ROI ниже"],
                 ["Сохранять у цены прогноза класс времени и время котировки (решение v6.2)", "A (уже принято)",
                  "без этого CLV и честность цены не проверить"],
                 ["Регулярно считать CLV Main как мониторинг (не как правило)", "B",
                  "положительный ценовой CLV — признак того, что модель опережает рынок; нужен ряд наблюдений"]]),
            P("Таблица 13. Решения. Ежедневное расписание не меняется: доказанной необходимости нет. B-кандидатов на "
              "изменение правила Main нет, поэтому предрегистрация (`results/preregistration.json`) пуста; мониторинг "
              "CLV — не правило выбора и не требует предрегистрации.", "small"))
    rep.add(F.H1c("12. Ограничения"),
            *bullets(["Архив OddsPapi почти не хранит промежуточных котировок: движение внутри OddsPapi видно только "
                      f"для {mv['requoted_after_T']} книг, перекотированных после T.",
                      "Время снимка открытия Football-Data неизвестно (пятница/вторник); Football-Data Bet365 в момент T "
                      "считается доступным так же, как в replay.",
                      f"Ранний момент (за 48 ч) имеет цену только у {F.pp(mo['summary']['early']['priced_share'], 0, False)} % "
                      "Main — выборка раннего Main мала и смещена к рынкам с ранними котировками.",
                      "Живые версии есть только с 22.10.2026 (около месяца) — малая выборка.",
                      "The Odds API и William Hill покрывают мало матчей; сравнение источников ограничено.",
                      "Сезон 2026/27 уже изучался ранее — хронологическая проверка, не holdout."]))
    rep.add(F.H1c("Как воспроизвести"),
            *K.reproduce([".venv/bin/python -m experiments.v6_3.replay.build   # общий кэш (если его нет)",
                          ".venv/bin/python -m experiments.v6_3.timing.run     # ~1 мин, без обучения моделей",
                          ".venv/bin/python -m experiments.v6_3.timing.build_pdf"]),
            P("Результаты: `experiments/v6_3/timing/results/` (timing.json, coverage.csv, clv_main_t09.csv, "
              "per_match_baseline_b_{early,t09,close}.csv, preregistration.json, run_info.json).", "small"))
    rep.build()
    print("pages", rep.check())
    findings(r)


def findings(r) -> None:
    mo, sp, clv, mv, lv = r["moments"], r["same_picks"], r["clv"], r["movement"], r["live_versions"]
    t = sp["t09+close"]
    lines = ["# FootPredict v6.3 — Timing & CLV: выводы", "",
             "Все числа — из `results/timing.json` (файл пишет `build_pdf.py`). PDF: `reports/v6_3/Timing_CLV_Research.pdf`.",
             "", "## Данные", "",
             f"- OddsPapi bet365: одна котировка у {F.pp(r['archive']['books']['bet365']['share_1_quote'], 0, False)} % "
             f"исходов, к T известна {F.pp(r['archive']['books']['bet365']['share_quoted_by_T'], 0, False)} %.",
             f"- Живые версии: {lv['matches_2plus']} матчей с ≥ 2 версиями (с 22.10.2026).", "",
             "## Результаты", "",
             "| Момент Main | Ставок | ROI, % [95 % ДИ] | Ставок (одинак. выборка) | ROI, % |", "|---|---|---|---|---|"]
    for m in ("early", "t09", "close"):
        s, s2 = mo["summary"][m], mo["all_priced"][m]
        lines.append(f"| {MOMENT_RU[m]} | {s['bets']} | {F.roi_cell(s['roi_ci'])} | {s2['bets']} | {F.pp(s2['roi'])} |")
    lines += ["",
              f"- Те же Main T по цене закрытия: Δ ROI {F.pp(t['close_minus_t09']['roi_diff']['diff'], 2)} п.п. "
              f"{F.ci(t['close_minus_t09']['roi_diff']['ci95'], 2)}.",
              f"- CLV Main T: ценовой {F.pp(clv['clv_price']['mean'], 2)} %, справедливый {F.pp(clv['clv_fair']['mean'], 2)} %.",
              f"- Рынок движется к модели: FD Bet365 1X2 β {sl(mv['football_data']['bet365|1X2'])}, OddsPapi β "
              f"{sl(mv['oddspapi']['all'])}.",
              f"- Живые версии, последняя − первая: Δ ROI {F.pp(lv['last_vs_first']['roi_diff']['diff'])} п.п. "
              f"{F.ci(lv['last_vs_first']['roi_diff']['ci95'])}.", "",
              "## Решения", "",
              "- C: менять расписание / формировать Main ближе к началу; пересчёт Main по закрытию.",
              "- A (уже v6.2): класс времени цены у прогноза.",
              "- B: CLV Main как мониторинг (не правило). Production не меняется; B-кандидатов правила нет.", "",
              "## Воспроизведение", "", "```", ".venv/bin/python -m experiments.v6_3.timing.run",
              ".venv/bin/python -m experiments.v6_3.timing.build_pdf", "```", ""]
    (RES.parent / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
