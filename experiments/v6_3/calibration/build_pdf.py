"""Builds reports/v6_3/Probability_Calibration_Research.pdf from results/ (no number is typed by hand).

    python -m experiments.v6_3.calibration.build_pdf
"""
from __future__ import annotations

from reportlab.lib.units import cm

from scripts import report_kit as K
from scripts.report_kit import H1, H2, H3, P, bullets, figure, formula, note, table

from experiments.v6_3 import common as C

from . import data as D
from . import maps as MP
from .figures import vlabel
from .figures import vshort

OUT = C.ROOT / "reports" / "v6_3" / "Probability_Calibration_Research.pdf"
RES = D.DIRS["results"]
FIG = D.DIRS["figures"]
U_RU = {"close": "close", "t09": "t09"}


def pp(v, d=1, signed=True):
    return "—" if v is None else (f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}")


def ci(v, c, d=1, signed=True):
    if v is None:
        return "—"
    if not c or c[0] is None:
        return pp(v, d, signed)
    return f"{pp(v, d, signed)} [{pp(c[0], d, signed)}; {pp(c[1], d, signed)}]"


def pv(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def f3(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def bin_label(b: str) -> str:
    lo, hi = (float(x) for x in b.strip("(]").split(","))
    lo, hi = int(lo) + 1, int(hi)
    return str(lo) if lo == hi else f"{lo}–{hi}"


def decision(r_close: dict, r_t09: dict, holm_close) -> str:
    d1, d2 = r_close["vs_b"]["roi_diff"], r_t09["vs_b"]["roi_diff"]
    if holm_close is not None and holm_close < 0.05 and d1["diff"] > 0 and d2["diff"] > 0:
        return "A"
    if d1["diff"] > 0 and d2["diff"] > 0:
        return "B"
    return "C"


def build() -> None:
    r = C.read_json(RES / "calibration.json")
    info = C.read_json(RES / "run_info.json")
    pre = C.read_json(RES / "preregistration.json") if (RES / "preregistration.json").exists() else None
    prim = r["primary"]
    pm, ps, _ = prim.split("|")
    q = r["quality"]
    base = r["main"]["close"]["baseline_b"]
    base9 = r["main"]["t09"]["baseline_b"]
    ra = r["baseline_a"]
    dat = r["data"]
    fam = r["family"]
    rep = K.Report(OUT, "Калибровка вероятностей модели", "FootPredict v6.3 — исследование 2",
                   {"Сезон проверки": "2026/27 (хронологическая проверка уже изученного сезона)",
                    "Обучение карт": "2025/26 (backtest) и/или месяцы 2026/27 до прогнозируемого",
                    "Данные": f"{dat['season_matches']} матчей 2026/27, {dat['backtest_matches']} матчей 2025/26"})

    # ------------------------------------------------------------------------------------------- 1
    wc = {(x["rule_on"], x["k"]): x for x in r["winners_curse"]["table"]}
    k_all = max(r["winners_curse"]["k"])
    rp, rp9 = r["main"]["close"][prim], r["main"]["t09"][prim]
    rep.add(H1("1. Название и краткие итоги"),
            P("**Калибровка вероятностей прогнозной модели FootPredict по рынкам и её влияние на выбор Main "
              "(проклятие победителя при выборе из многих кандидатов).**"),
            note(f"Итог. Основная карта ({vlabel(prim)}; выбрана только на сезоне 2025/26) меняет log loss "
                 f"кандидатов Main 2026/27 на {pp(r['paired_log_loss_vs_raw'][prim]['band']['diff'] * 10, 2)} × 10⁻³ "
                 f"(95 % ДИ {pp(r['paired_log_loss_vs_raw'][prim]['band']['ci95'][0] * 10, 2)}; "
                 f"{pp(r['paired_log_loss_vs_raw'][prim]['band']['ci95'][1] * 10, 2)}; отрицательно — лучше), ECE кандидатов "
                 f"{pp(q['raw|-|-']['band']['ece'], 2, False)} → {pp(q[prim]['band']['ece'], 2, False)} п.п. "
                 f"Переоценка выбранного Main Baseline B (close) {ci(base['gap']['gap'], base['gap']['gap_ci'])} п.п.; "
                 f"при выборе на откалиброванных вероятностях — {ci(rp['gap']['gap'], rp['gap']['gap_ci'])} п.п. "
                 f"ROI Main на откалиброванных вероятностях: close {pp(rp['summary']['roi'])} % против "
                 f"{pp(base['summary']['roi'])} % (ΔROI {ci(rp['vs_b']['roi_diff']['diff'], rp['vs_b']['roi_diff']['ci95'])} п.п.), "
                 f"t09 {pp(rp9['summary']['roi'])} % против {pp(base9['summary']['roi'])} %. Решение: "
                 f"{decision(rp, rp9, r['main']['close']['holm'].get(prim))}."))

    # ------------------------------------------------------------------------------------------- 2
    rep.add(H1("2. Постановка проблемы"),
            P("EV каждого рынка и выбор Main опираются на вероятность модели: EV = p · odds − (1 − P(push)). "
              "Если вероятность систематически завышена в какой-то группе рынков, правило Main будет предпочитать "
              "эту группу, а показанный EV окажется положительным там, где реальная доходность отрицательна. "
              "Исследование v6.2 «Калибровка Main 2.0» показало, что вероятность уже выбранного Main завышена "
              "(«проклятие победителя») и что пост-калибровка показанной вероятности возвращает разрыв к нулю, "
              "но не меняет выбор; калибровка до выбора по историческим таблицам v6.1 не улучшила ROI."),
            P("v6.3 расширяет это исследование: (1) калибровка **всех** рынков (1X2, тоталы, обе забьют, двойной "
              "шанс, азиатские и европейские форы, угловые, точный счёт) на новом источнике — out-of-sample "
              "прогнозах backtest 2025/26 по каждой селекции; (2) новые методы — бета-калибровка и температурная "
              "калибровка, гистограммная калибровка, карты по группе и по исходу; (3) три источника обучения "
              "карт (заморожено на 2025/26, расширяющееся окно 2026/27, объединение); (4) Main, выбранный "
              "правилом продакшна на откалиброванных вероятностях, в общем replay v6.3 при двух моментах решения; "
              "(5) контролируемый эксперимент, в котором меняется только число кандидатов, — прямое измерение "
              "проклятия победителя."))

    # ------------------------------------------------------------------------------------------- 3
    rep.add(H1("3. Гипотезы"), *bullets([
        "H1. Вероятности модели откалиброваны неодинаково по группам рынков; в зоне кандидатов Main (p от 0.40 до "
        "0.85) есть систематическое смещение.",
        "H2. Карта, обученная только на прошлых матчах (2025/26 и/или ранние месяцы 2026/27), уменьшает log loss, "
        "Brier и ECE на 2026/27.",
        "H3. Переоценка выбранного Main растёт с числом кандидатов (проклятие победителя); калибровка до выбора "
        "убирает систематическую часть, но не эффект отбора.",
        "H4. Main, выбранный правилом продакшна на откалиброванных вероятностях, даёт более честный EV (меньше "
        "ложного положительного EV) и не хуже по ROI, чем Baseline B."]))

    # ------------------------------------------------------------------------------------------- 4
    rows = [["Набор", "Матчи", "Строки (селекции)", "Решённые строки", "Цены", "Модель"],
            ["2025/26 backtest (обучение карт)", dat["backtest_matches"], dat["backtest_rows"],
             dat["backtest_settled_rows"], f"{pp(dat['backtest_priced_share'], 1, False)} % строк (Bet365, Football-Data)",
             "backtest_2526, обучена до 2025-06-01"],
            ["2026/27 replay (проверка)", dat["season_matches"], dat["season_rows"], dat["season_settled_rows"],
             "close: последняя котировка; t09: цена на момент T", "prod_20261012, обучена до 2026-05-31"]]
    rep.add(H1("4. Исходные данные"),
            P("Источники: `local_data/predictions/markets.parquet` (backtest-прогнозы 2025/26 по каждой селекции), "
              "`picks.parquet` (счёт и угловые 2025/26), общий replay v6.3 `experiments/v6_3/replay/cache/` "
              "(кандидаты 2026/27 со ставками, ценами и результатами, Baseline A и B). Лиги: 21 чемпионат "
              "(Англия E0–E3, Шотландия SC0–SC3, Германия D1–D2, Испания SP1–SP2, Италия I1–I2, Франция F1–F2, "
              "Нидерланды, Бельгия, Португалия, Турция, Греция)."),
            table(rows, widths=[4.2 * cm, 1.5 * cm, 2.2 * cm, 2.2 * cm, 3.6 * cm, 3.3 * cm]),
            P(f"Месяцы 2026/27: {', '.join(dat['months_2026'])} (последние дни июля отнесены к августу). "
              "Охват и исключения по матчам — в разделе 9."))

    # ------------------------------------------------------------------------------------------- 5
    rep.add(H1("5. Методика"),
            H2("5.1. Шкала и исход"),
            P("Все карты работают на шкале без возврата pc = p / (1 − P(push)) (отличается от p только для целых "
              "азиатских линий), исход y = 1 при выигрыше, 0 при проигрыше; возвраты не участвуют в обучении и "
              "метриках. Откалиброванная вероятность возвращается на шкалу продакшна p′ = pc′ · (1 − P(push))."),
            H2("5.2. Методы калибровки"),
            formula("Platt: pc′ = σ(a + b · logit pc)"),
            formula("Температурная: pc′ = σ(logit pc / T)"),
            formula("Бета: pc′ = σ(c + a · ln pc − b · ln(1 − pc)),  a, b ≥ 0"),
            P("Бета-калибровка (Kull и соавт., 2017) реализована как логистическая регрессия на признаках ln pc и "
              "−ln(1 − pc); при отрицательном коэффициенте признак удаляется и регрессия переобучается (пакет "
              "`betacal` не используется). Изотоническая регрессия — монотонная ступенчатая функция (PAV), "
              "гистограммная — 20 бинов равной численности со сглаживанием Лапласа. Области: общая карта, по "
              "группе рынков, по группе и исходу (например 1X2|D, TOTAL|OVER); при менее чем 300 решённых "
              "строках уровень заменяется более общим."),
            H2("5.3. Источники обучения (без утечки будущего)"),
            *bullets(["**2025/26** — карта обучена один раз на backtest 2025/26 и заморожена на весь 2026/27;",
                      "**месяцы 2026/27** — для матчей месяца M карта обучена на матчах 2026/27, сыгранных до 1-го "
                      "числа M (август — без калибровки);",
                      "**2025/26 + месяцы** — объединение обоих (основной источник)."]),
            P("Основная карта (метод и область) выбрана **только на 2025/26**: walk-forward внутри backtest-сезона "
              "(карта на месяцах до M, оценка на месяце M, сентябрь–май), критерий — log loss кандидатов Main. "
              "Сезон 2026/27 в выборе не участвовал."),
            H2("5.4. Метрики"),
            formula("Brier = (1/n) Σ (p_{i} − y_{i})^{2};   LogLoss = −(1/n) Σ [y_{i} ln p_{i} + (1 − y_{i}) ln(1 − p_{i})]"),
            formula("ECE = Σ_{b} (n_{b}/n) · |mean p_{b} − mean y_{b}|,  10 равных интервалов"),
            P("Разрыв Main = средняя заявленная вероятность − доля выигрышей решённых Main. Разности метрик — "
              "парный бутстреп по матчам (2000 повторов); ROI — по правилам продакшна (ставка 1, PUSH / VOID "
              "возвращают ставку, Main без реальной цены — не ставка), парный бутстреп ΔROI к Baseline B того же "
              "момента решения, поправка Холма по всем вариантам Main."),
            H2("5.5. Main на откалиброванных вероятностях"),
            P("Правило продакшна (`src/markets/recommendation.py`) без изменений параметров, но с откалиброванной "
              "вероятностью: она определяет попадание в диапазон 0.40–0.85, множитель p и EV = p′ · odds − (1 − P(push)). "
              "Формула EV не меняется. Векторная реализация правила проверена на всех матчах обеих вселенных: "
              f"совпадение с `evaluate.main_by_rule` {pp(r['self_check']['close']['equal_to_main_by_rule'], 2, False)} % (close), "
              f"{pp(r['self_check']['t09']['equal_to_main_by_rule'], 2, False)} % (t09); с движком Baseline B "
              f"{pp(r['self_check']['close']['equal_to_engine_baseline_b'], 2, False)} % / "
              f"{pp(r['self_check']['t09']['equal_to_engine_baseline_b'], 2, False)} % (одно расхождение в close — "
              "D2-20260906-greuther-furth-heidenheim: точное равенство очков «Двойной шанс 1X» и «Фора +0.5 хозяев», "
              "разрешённое движком в другом порядке кандидатов)."),
            H2("5.6. Эксперимент «число кандидатов»"),
            P(f"Для каждого сыгранного матча 2026/27 из кандидатов Main случайно оставляются k (k = "
              f"{', '.join(str(k) for k in r['winners_curse']['k'][:-1])} и все), правило продакшна выбирает Main, "
              f"измеряется заявленная вероятность − доля выигрышей; {r['winners_curse']['reps']} случайных повторов "
              "на каждое k. Матчи и модель одни и те же, меняется только число вариантов выбора — поэтому рост "
              "разрыва с k измеряет проклятие победителя напрямую. Тот же выбор оценивается и откалиброванной "
              "вероятностью основной карты (систематическая часть), а также выбор правилом на откалиброванных "
              "вероятностях."))

    # ------------------------------------------------------------------------------------------- 6
    n_maps = len(q) - 1
    rep.add(H1("6. Эксперименты"), *bullets([
        f"Самопроверка векторного правила на {r['self_check']['close']['matches']} матчах × 2 вселенные.",
        f"Выбор основной карты на 2025/26: {len(r['selection']['table'])} вариантов «метод × область», "
        f"{len(r['selection']['months'])} месяцев walk-forward.",
        f"Качество вероятностей 2026/27: {n_maps} карт (5 методов × 3 области × 3 источника) + исходная, "
        "по 8 группам рынков и по кандидатам Main.",
        f"Main на откалиброванных вероятностях: {len(fam)} вариантов × 2 момента решения, парно с Baseline B "
        "(поправка Холма по этим вариантам).",
        f"Проклятие победителя: {len(r['winners_curse']['k'])} значений k × {r['winners_curse']['reps']} повторов × "
        "2 правила; наблюдаемый разрыв по числу кандидатов (2026/27 и 2025/26).",
        "Устойчивость основной карты: месяцы, лиги, крупные выигрыши, снижение коэффициентов на 2 % и 5 %."]))

    # ------------------------------------------------------------------------------------------- 7
    rep.add(H1("7. Результаты"), H2("7.1. Калибровка по группам рынков (2026/27)"))
    rows = [["Группа", "n", "p ср.", "Факт", "Разрыв", "LogLoss", "LogLoss карта", "Brier", "Brier карта", "ECE", "ECE карта"]]
    w1 = [2.5 * cm, 1.3 * cm, 1.1 * cm, 1.1 * cm, 1.6 * cm, 1.7 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.1 * cm, 1.3 * cm]
    for g in D.GROUPS_ALL + ["band"]:
        a, b = q["raw|-|-"][g], q[prim][g]
        rows.append([D.GROUP_RU.get(g, "Кандидаты Main"), a["n"], pp(a["mean_p"], 1, False), pp(a["hit"], 1, False),
                     pp(a["gap"]), f3(a["log_loss"]), f3(b["log_loss"]), f3(a["brier"]), f3(b["brier"]),
                     pp(a["ece"], 2, False), pp(b["ece"], 2, False)])
    rep.add(table(rows, widths=w1), P("Таблица 1. Исходная вероятность и основная карта (вероятности и доли в %, разрыв в "
                                      "п.п., ECE в п.п.). «Кандидаты Main» — группы, из которых выбирается Main, p от 0.40 до "
                                      "0.85. В группе, где в таблицу входят все исходы рынка, средняя p равна частоте по "
                                      "построению (сумма вероятностей исходов = 1), поэтому смещение видно только по ECE и "
                                      "по кандидатам Main; у двойного шанса бинарный log loss совпадает с 1X2 (дополнения).",
                                      "small"))
    bq = r["quality_backtest_raw"]
    rows = [["Группа", "n 2025/26", "Разрыв 2025/26", "ECE 2025/26", "LogLoss 2025/26"]]
    for g in D.GROUPS_ALL:
        rows.append([D.GROUP_RU[g], bq[g]["n"], pp(bq[g]["gap"]), pp(bq[g]["ece"], 2, False), f3(bq[g]["log_loss"])])
    rep.add(P("Для сравнения — калибровка backtest-модели на 2025/26 (без карт, все селекции):"), table(rows),
            P("Таблица 2. Исходная калибровка 2025/26 (источник обучения карт).", "small"))
    rows = [["Группа", "Δ LogLoss ×10³ [95 % ДИ]", "p"]]
    for g in D.MAIN_GROUPS + ["band"]:
        v = r["paired_log_loss_vs_raw"][prim][g]
        rows.append([D.GROUP_RU.get(g, "Кандидаты Main"), f"{v['diff'] * 1000:+.2f} [{v['ci95'][0] * 1000:+.2f}; "
                     f"{v['ci95'][1] * 1000:+.2f}]", pv(v["p"])])
    rep.add(table(rows, widths=[5 * cm, 6 * cm, 2.5 * cm]),
            P(f"Таблица 3. Основная карта минус исходная вероятность, log loss, парный бутстреп по матчам "
              f"(отрицательно — лучше). Карта: {vlabel(prim)}.", "small"))

    rep.add(H2("7.2. Сравнение методов"))
    sel = r["selection"]["table"]
    rows = [["Метод", "Область", "LogLoss 2025/26 (wf)", "Brier 2025/26 (wf)", "LogLoss 2026/27", "ECE 2026/27"]]
    for x in sel:
        name = "raw|-|-" if x["method"] == "raw" else f"{x['method']}|{x['scope']}|pooled"
        rows.append([MP.METHOD_RU[x["method"]], "—" if x["method"] == "raw" else MP.SCOPE_RU[x["scope"]],
                     f3(x["log_loss"], 5), f3(x["brier"], 5), f3(q[name]["band"]["log_loss"], 5),
                     pp(q[name]["band"]["ece"], 2, False)])
    rep.add(table(rows, widths=[3.4 * cm, 3.4 * cm, 2.7 * cm, 2.7 * cm, 2.5 * cm, 2.0 * cm]),
            P("Таблица 4. Кандидаты Main: walk-forward внутри 2025/26 (критерий выбора основной "
                           "карты, отсортировано) и проверка на 2026/27 (источник 2025/26 + месяцы).", "small"))

    rep.add(H2("7.3. Main на откалиброванных вероятностях"))
    for u in ("close", "t09"):
        mr = r["main"][u]
        rows = [["Вариант", "Ставки", "ROI, %", "Попад., %", "Коэф.", "ΔROI, п.п. [95 % ДИ]", "p Холма",
                 "Тот же Main", "EV<0", "Разр., п.п."]]
        for n in ["baseline_b"] + fam:
            x = mr[n]
            s = x["summary"]
            d = x["vs_b"]["roi_diff"]
            rows.append([vshort(n), s["bets"], pp(s["roi"]), pp(s["hit_rate"], 1, False), f"{s['avg_odds']:.2f}",
                         "—" if n == "baseline_b" else ci(d["diff"], d["ci95"]),
                         "—" if n == "baseline_b" else pv(mr["holm"].get(n)),
                         pp(x["vs_b"]["same_main_share"], 0, False) + " %",
                         pp(x["neg_ev_share_stated"], 1, False) + " %", pp(x["gap"]["gap"])])
        rep.add(table(rows, widths=[3.9 * cm, 1.4 * cm, 1.3 * cm, 1.5 * cm, 1.2 * cm, 2.9 * cm, 1.3 * cm, 1.3 * cm,
                                    1.2 * cm, 1.4 * cm]),
                P(f"Таблица {5 if u == 'close' else 6}. Вселенная {u}: Main по правилу продакшна на откалиброванных "
                  f"вероятностях, парно с Baseline B ({mr['baseline_b']['vs_b']['matches']} матчей). Вариант: метод "
                  "(темп. — температурная, изотон., гистогр.), область (общая / группа рынков / группа и исход), "
                  "источник обучения (25/26 — заморожена на 2025/26; мес. 26/27 — расширяющееся окно; 25/26+мес. — "
                  "оба). «EV<0» — "
                  "доля ставок с отрицательным EV по той вероятности, которой правило пользовалось. Baseline A "
                  f"(история сайта): {ra['summary']['bets']} ставок, ROI {pp(ra['summary']['roi'])} %, попадание "
                  f"{pp(ra['summary']['hit_rate'], 1, False)} %.", "small"))
    rows = [["Вариант", "EV<0 исходный EV, %", "EV<0 заявленный EV, %", "Ср. заявл. p, %", "Факт, %", "Группы Main (топ-3)"]]
    for n in ["baseline_b"] + fam:
        x = r["main"]["close"][n]
        top = ", ".join(f"{D.GROUP_RU[g]} {v}" for g, v in list(sorted(x["groups"].items(), key=lambda kv: -kv[1]))[:3])
        rows.append([vshort(n), pp(x["neg_ev_share_raw"], 1, False), pp(x["neg_ev_share_stated"], 1, False),
                     pp(x["gap"]["mean_p"], 1, False), pp(x["gap"]["hit"], 1, False), top])
    rep.add(table(rows, widths=[4.4 * cm, 2 * cm, 2 * cm, 1.8 * cm, 1.5 * cm, 5.3 * cm]),
            P("Таблица 7. Close: доля отрицательного EV (по исходной вероятности и по вероятности, на которой "
              "выбирал вариант), заявленная вероятность и фактическая частота, состав Main.", "small"))

    rep.add(H2("7.4. Проклятие победителя: число кандидатов"))
    rows = [["k", "Заявл. − факт (исх.), п.п.", "Откалибр. − факт (тот же выбор)", "Правило на калибр.: заявл. − факт",
             "Попад., %", "ROI исх., %", "ROI калибр., %", "Коэф."]]
    for k in r["winners_curse"]["k"]:
        a, b = wc[("raw", k)], wc[("calibrated", k)]
        rows.append(["все" if k == k_all else k,
                     pp(a["gap_stated"]) if k == k_all else f"{pp(a['gap_stated'])} ± {pp(a['gap_stated_sd'] or 0, 1, False)}",
                     f"{pp(a['gap_cal'])}", f"{pp(b['gap_stated'])}", pp(a["hit"], 1, False), pp(a["roi"]),
                     pp(b["roi"]), f"{a['avg_odds']:.2f}"])
    rep.add(table(rows, widths=[1.0 * cm, 2.8 * cm, 2.8 * cm, 2.9 * cm, 1.8 * cm, 1.7 * cm, 1.8 * cm, 1.3 * cm]),
            P(f"Таблица 8. Close, 2026/27: k случайных кандидатов Main на матч, среднее по "
                           f"{r['winners_curse']['reps']} повторам (± стандартное отклонение по повторам; «все» — один расчёт).", "small"))
    dec = r["b_gap_decomposition"]
    rep.add(P(f"Разложение разрыва Baseline B (close): заявленная вероятность − факт = "
              f"{ci(dec['stated']['gap'], dec['stated']['gap_ci'])} п.п.; если те же Main оценить основной картой "
              f"(обученной без этих матчей), разрыв {ci(dec['primary_map']['gap'], dec['primary_map']['gap_ci'])} п.п. "
              + ("Разница — систематическая (маргинальная) ошибка калибровки выбранных селекций, остаток — эффект "
                 "отбора, который маргинальная карта не видит." if dec["primary_map"]["gap"] < dec["stated"]["gap"] else
                 "Карта разрыв не уменьшает: маргинальной ошибки калибровки у выбранных селекций практически нет, "
                 "разрыв целиком — эффект отбора максимума (таблица 8: при k = 1 разрыва нет).")))
    rows = [["Кандидатов с ценой", "n", "Разрыв, п.п. [95 % ДИ]", "Заявл. p, %", "Факт, %"]]
    for x in r["ncand_observed"]["close_ncand_priced"]:
        rows.append([bin_label(x["bin"]), x["n"], ci(x["gap"], x["gap_ci"]), pp(x["mean_p"], 1, False), pp(x["hit"], 1, False)])
    rows.append(["2025/26 (backtest):", "", "", "", ""])
    for x in r["ncand_observed"]["backtest_ncand_priced"]:
        rows.append([bin_label(x["bin"]), x["n"], ci(x["gap"], x["gap_ci"]), pp(x["mean_p"], 1, False), pp(x["hit"], 1, False)])
    rep.add(table(rows, widths=[3.5 * cm, 1.5 * cm, 4.5 * cm, 2.5 * cm, 2.5 * cm]),
            P("Таблица 9. Наблюдаемый разрыв Main по числу кандидатов с реальной ценой: Baseline B 2026/27 "
              "(close) и правило продакшна на backtest 2025/26 (цены только 1X2, тотал 2.5, одна азиатская линия).",
              "small"))

    # ------------------------------------------------------------------------------------------- 8
    rep.page_break()
    rep.add(H1("8. Графики"),
            *figure(FIG / "reliability_groups.png", "Рис. 1. Диаграммы надёжности по группам рынков: 2025/26 и "
                    "2026/27 без калибровки, 2026/27 с основной картой (бины с n ≥ 30)."),
            *figure(FIG / "ece_groups.png", "Рис. 2. ECE по группам: исходная вероятность и основная карта при трёх "
                    "источниках обучения.", width=16 * cm),
            *figure(FIG / "quality_heatmap.png", "Рис. 3. Δ log loss кандидатов Main 2026/27 к исходной вероятности "
                    "(× 1000; зелёный — лучше) для каждого метода, области и источника.", width=13 * cm),
            *figure(FIG / "winners_curse.png", "Рис. 4. Переоценка выбранного Main и ROI в зависимости от числа "
                    "кандидатов k (слева: заявлено − факт; справа: ROI и средний коэффициент)."),
            *figure(FIG / "ncand_observed.png", "Рис. 5. Наблюдаемая переоценка Main по числу кандидатов с ценой."),
            *figure(FIG / "roi_forest.png", "Рис. 6. ΔROI вариантов Main на откалиброванных вероятностях к "
                    "Baseline B, 95 % ДИ парного бутстрепа (зелёный — основная карта)."),
            *figure(FIG / "monthly.png", "Рис. 7. ROI по месяцам: Baseline B и основная карта."))

    # ------------------------------------------------------------------------------------------- 9
    rep.add(H1("9. Статистический анализ"))
    for u in ("close", "t09"):
        rb = r["robustness"][u][prim]
        rbb = r["robustness"][u]["baseline_b"]
        rows = [["Месяц", "Ставки", "ROI, %", "ROI B, %", "Δ, п.п.", "Малая выборка"]]
        for x in rb["by_month"]:
            rows.append([x["month"], x["bets"], pp(x["roi"]), pp(x["roi_b"]), pp(x["roi_diff"]), "да" if x["small"] else ""])
        rep.add(H3(f"Устойчивость основной карты — {u}"), table(rows, widths=[2.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2.5 * cm]),
                P(f"ROI по месяцам ({u}), основная карта и Baseline B; ниже — по лигам.", "small"))
        lg = sorted(rb["by_league"], key=lambda x: x["league"])
        rows = [["Лига", "Ставки", "ROI, %", "Δ к B, п.п."] * 2]
        half = (len(lg) + 1) // 2
        for i in range(half):
            row = []
            for x in (lg[i], lg[i + half] if i + half < len(lg) else None):
                row += ["", "", "", ""] if x is None else [x["league"] + (" *" if x["small"] else ""), x["bets"],
                                                           pp(x["roi"]), pp(x["roi_diff"])]
            rows.append(row)
        rep.add(table(rows), P(f"Лиги ({u}): лучше Baseline B в {rb['leagues_better']} из {rb['leagues']}; месяцы: "
                               f"{rb['months_better']} из {rb['months']}. * — меньше 60 ставок.", "small"))
        rows = [["Показатель", "Основная карта", "Baseline B"]]
        for k in ("1", "5", "10", "20"):
            rows.append([f"ROI без {k} крупнейших выигрышей, %", pp(rb["roi_without_top_wins"][k]),
                         pp(rbb["roi_without_top_wins"][k])])
        for k, lab in (("0.02", "2 %"), ("0.05", "5 %")):
            rows.append([f"ROI при коэффициентах ниже на {lab}, %", pp(rb["roi_odds_haircut"][k]),
                         pp(rbb["roi_odds_haircut"][k])])
        rep.add(table(rows, widths=[7 * cm, 3.5 * cm, 3.5 * cm]))
    rep.add(P(f"Поправка Холма применена к {len(fam)} вариантам Main в каждой вселенной; всего вариантов "
              f"карт, оценённых по качеству вероятностей: {n_maps}. Сезон 2026/27 уже изучался в v6.1/v6.2, поэтому "
              "результат — хронологическая проверка, а не независимое подтверждение."))
    rows = [["Цена", "Метод", "Матчей", "Сыгр.", "Обраб.", "С ценой", "Мало стат.",
             "Ставки", "Без цены", "PUSH/ VOID", "Не сыгр."]]
    for u in ("close", "t09"):
        for x in r["coverage"][u]:
            rows.append([u, vshort(x["method"]), x["season_matches"], x["played"], x["processed"], x["real_price"],
                         x["few_statistics"], x["bets"], x["excluded_no_price"], x["excluded_push_void"],
                         x["not_played"]])
    rep.add(H3("Охват"), table(rows, widths=[1.2 * cm, 3.0 * cm, 1.5 * cm, 1.6 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm,
                                             1.5 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm]),
            P("Таблица охвата: «мало статистики» — у одной из команд меньше 3 матчей сезона (прогноз всё равно "
              "строится, матч не исключается); «без цены» — решённый Main без реальной цены (не ставка); не сыгранные "
              "матчи не входят в ROI. Ничего не исключено молча.", "small"))

    # ------------------------------------------------------------------------------------------- 10-12
    rep.add(H1("10. Выводы"), *bullets(conclusions(r)))
    rep.add(H1("11. Рекомендации"), *bullets(recommendations(r, pre)))
    rep.add(H1("12. Ограничения"), *bullets([
        "Карты обучены на прогнозах другой версии модели (backtest_2526, обучена до 2025-06-01), а применяются к "
        "prod_20261012; перенос калибровки между версиями не гарантирован — поэтому сравниваются три источника.",
        "В 2025/26 реальные цены есть только у 1X2, тотала 2.5 и одной азиатской линии (Bet365 Football-Data); "
        "калибровке цены не нужны, но наблюдаемый разрыв Main 2025/26 по числу кандидатов с ценой имеет другой "
        "диапазон, чем в 2026/27.",
        "Сезон 2026/27 неполный (до середины ноября, 4 месяца) и уже изучался; месячные оценки по лигам малы.",
        "Вселенная close использует последнюю котировку перед стартом, t09 — цену на момент T, а четверть Main t09 "
        "без цены и не является ставкой.",
        "Калибровка по селекциям не согласует вероятности внутри рынка (сумма исходов 1X2 после карты может "
        "отличаться от 1); для многоклассового log loss 1X2 они нормированы."]))
    rep.add(H1("Как воспроизвести"), *K.reproduce([
        "experiments/v6_3/heavy.sh 3600 experiments/v6_3/calibration/logs/run.log \\",
        "    .venv/bin/python -m experiments.v6_3.calibration.run",
        ".venv/bin/python -m experiments.v6_3.calibration.figures",
        ".venv/bin/python -m experiments.v6_3.calibration.preregister",
        ".venv/bin/python -m experiments.v6_3.calibration.build_pdf"]),
        P(f"Git: {info['git_sha']}; данные: picks {info['data']['picks']['mtime']}, markets "
          f"{info['data']['markets']['mtime']}; время расчёта {info['seconds']} с. Результаты: "
          "`experiments/v6_3/calibration/results/` (calibration.json, per_match_*.csv, preregistration.json, "
          "run_info.json).", "small"))
    rep.build()
    print("pages", rep.check())


def conclusions(r: dict) -> list[str]:
    from .conclusions import conclusions as cc
    return cc(r)


def recommendations(r: dict, pre) -> list[str]:
    from .conclusions import recommendations as rr
    return rr(r, pre)


if __name__ == "__main__":
    build()
