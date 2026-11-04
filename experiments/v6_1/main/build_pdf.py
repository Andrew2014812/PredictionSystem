"""Builds reports/v6_1/Main_Strategy_Research.pdf from the result files of experiments/v6_1/main (Russian).

    python -m experiments.v6_1.main.build_pdf

Every number in the report is read from results/*.json; nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.report_kit import (H1, H2, H3, P, Report, bullets, code, figure, formula, note, num, reproduce, share,
                                table)

from . import common as C

R = C.RESULTS
FIG = C.FIGURES
OUT = C.ROOT / "reports" / "v6_1" / "Main_Strategy_Research.pdf"


def J(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


ST, CAL, BIAS, MIX, BET, PR, MOD = (J(n) for n in ("strategies.json", "calibration.json", "selection_bias.json",
                                                    "market_mix.json", "betting.json", "prices_clv.json",
                                                    "models_cmp.json"))
SEL, IND, PRE, AUD, RUN, CHK = (J(n) for n in ("selection.json", "independent.json", "preregistration.json",
                                               "fd_price_audit.json", "run_info.json", "self_check.json"))
FOLDS = [str(s) for s in C.FOLDS]
SL = {s: C.period_label(int(s)) for s in FOLDS + [str(C.LIVE)]}

RU = {
    "A": "A — текущий Main", "B1 isotonic": "B1 изотоническая калибровка", "B2 Platt": "B2 калибровка Платта",
    "B3 selection-conditional": "B3 калибровка по выбранным", "C1 EV weight 1": "C1 вес EV 1",
    "C1 EV weight 2": "C1 вес EV 2", "C1 EV weight 4": "C1 вес EV 4", "C2 value first EV>=0": "C2 сначала EV≥0",
    "C2 value first EV>=0.03": "C2 сначала EV≥3%", "C3 max EV": "C3 максимум EV",
    "C4 calibrated + EV weight 2": "C4 калибр. p + вес EV 2", "C5 calibrated max EV": "C5 максимум EV (калибр.)",
    "D1 no totals": "D1 без тоталов", "D2 only 1X2": "D2 только 1X2",
    "D3 drop over-confident groups": "D3 без переоценённых групп", "D4 totals reliability 0.9": "D4 надёжность TOTAL 0.9",
    "E1 lower confidence bound": "E1 нижняя граница", "E2 candidate-count penalty": "E2 штраф за число кандидатов",
    "F1 blend Shin (inner weight)": "F1 смесь с рынком (Шин)", "F2 blend proportional (inner weight)": "F2 смесь (пропорц.)",
    "F3 blend 50/50": "F3 смесь 50/50", "F4 market only": "F4 только рынок", "F5 blend Pinnacle": "F5 смесь с Pinnacle",
    "G1 EV>0 (raw)": "G1 ставка при EV>0", "G2 EV>5% (raw)": "G2 ставка при EV>5%",
    "G3 EV>0 (calibrated)": "G3 EV>0 (калибр.)", "G4 EV>3% (calibrated)": "G4 EV>3% (калибр.)",
    "G5 EV>0 (blend)": "G5 EV>0 (смесь)", "G6 EV>2% (blend)": "G6 EV>2% (смесь)",
    "G7 EV>0 (meta-model)": "G7 EV>0 (мета-модель)", "G8 EV>3% (meta-model)": "G8 EV>3% (мета-модель)",
    "G9 agrees with market (|p-q|<3pp)": "G9 согласие с рынком", "G10 odds 1.60–2.50": "G10 коэф. 1.60–2.50",
    "G11 F1 Main, EV>0 (blend)": "G11 Main F1 + EV>0",
}


def ru(n):
    return RU.get(n, n)


def f(v, d=1, signed=True):
    return share(v, d, signed)


def ci(c, d=1):
    if not c or c[0] is None or c[0] != c[0]:
        return "—"
    return f"[{c[0] * 100:+.{d}f}; {c[1] * 100:+.{d}f}]"


def pp(v, d=1):
    return "—" if v is None or v != v else f"{v * 100:+.{d}f} п.п."


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def rng(vals, scale=100.0, d=1, signed=True, unit=""):
    v = [x * scale for x in vals if x is not None and x == x]
    if not v:
        return "—"
    fmt = f"{{:{'+' if signed else ''}.{d}f}}"
    a, b = min(v), max(v)
    return (fmt.format(a) if abs(a - b) < 10 ** -d else f"{fmt.format(a)}…{fmt.format(b)}") + unit


def facts() -> dict:
    """Ranges quoted in the text, computed from the result files."""
    K = {}
    sel = FOLDS
    K["gap_all"] = rng([CAL[m][s]["all_candidates"]["gap"] for m in ("production", "bettable") for s in sel], unit=" п.п.")
    K["ece_all"] = rng([CAL[m][s]["all_candidates"]["ece"] for m in ("production", "bettable") for s in sel], 1, 3, False)
    K["gap_main_prod"] = rng([CAL["production"][s]["overall"]["gap"] for s in sel], signed=False, unit=" п.п.")
    K["gap_main_bet"] = rng([CAL["bettable"][s]["overall"]["gap"] for s in sel], signed=False, unit=" п.п.")
    K["gap_main_prod_bets"] = rng([CAL["production"][s]["overall_bets"]["gap"] for s in sel], signed=False, unit=" п.п.")
    K["gap_mkt_picks"] = rng([CAL[m][s]["market_on_picks"]["gap"] for m in ("production", "bettable") for s in sel], unit=" п.п.")

    def grp(name, mode):
        return [x["gap"] for s in sel for x in CAL[mode][s]["by_group"] if x["key"] == name and x["n"] >= 100]
    K["gap_ah"] = rng(grp("HANDICAP", "production") + grp("HANDICAP", "bettable"), signed=False, d=0, unit=" п.п.")
    K["gap_1x2"] = rng(grp("1X2", "production") + grp("1X2", "bettable"), signed=False, unit=" п.п.")
    K["gap_tot"] = rng(grp("TOTAL", "production") + grp("TOTAL", "bettable"), signed=False, unit=" п.п.")
    ev = [x["realised_roi"] for s in sel for x in BIAS["bettable"][s]["ev_inflation_all_priced"] if x["n"] >= 500]
    K["ev_corridor"] = rng(ev, d=0, unit=" %")
    sim = [BIAS["bettable"][s]["simulation"] for s in sel]
    K["sim_gap"] = rng([x["expected_gap_chosen"] for x in sim], signed=False, unit=" п.п.")
    K["sim_ev"] = rng([x["expected_true_ev_chosen"] for x in sim], unit=" %")
    K["obs_roi_A"] = rng([ST["modes"]["bettable"]["folds"][s]["A"]["roi"] for s in sel], unit=" %")
    K["margin"] = rng([BIAS["bettable"][s]["margin_chosen"] for s in sel], signed=False, unit=" %")
    ws = [v for s in sel for v in ST["fitted"]["bettable"][s]["blend_weight_shin"].values()]
    K["w_blend"] = rng(ws, 1, 2, False)
    cs = [x["ci_disagreement"] for s in sel for x in BIAS["bettable"][s]["disagreement"].values() if "ci_disagreement" in x]
    K["c_all_zero"] = all(lo <= 0 <= hi for lo, hi in cs)
    K["c_range"] = rng([x["c_disagreement"] for s in sel for x in BIAS["bettable"][s]["disagreement"].values()
                        if "c_disagreement" in x], 1, 2)
    fb = ST["modes"]["bettable"]["folds"]
    cg = [fb[s][n]["calibration_main"]["gap"] for s in sel for n in fb[s] if n.startswith(("C1", "C2", "C3", "C4", "C5"))]
    K["gap_C"] = rng(cg, signed=False, unit=" п.п.")
    f1 = "F1 blend Shin (inner weight)"
    K["f1_diff"] = pp(ST["pooled_vs_A"][f1]["diff"])
    K["f1_roi"] = rng([fb[s][f1]["roi"] for s in FOLDS[:2]], unit=" %")
    K["f1_roi_info"] = f(fb[FOLDS[2]][f1]["roi"])
    K["a_roi_info"] = f(fb[FOLDS[2]]["A"]["roi"])
    K["e2_gap"] = rng([fb[s]["E2 candidate-count penalty"]["calibration_main"]["gap"] for s in sel], unit=" п.п.")
    K["b3_gap"] = rng([fb[s]["B3 selection-conditional"]["calibration_main"]["gap"] for s in sel], unit=" п.п.")
    g7 = "G7 EV>0 (meta-model)"
    K["g7_bets"] = " → ".join(str(fb[s][g7]["bets"]) for s in sel)
    K["share_tot_prod"] = rng([next(x["share"] for x in MIX["mix"]["production"][s]["group"] if x["key"] == "TOTAL") for s in sel],
                              d=0, signed=False, unit=" %")
    K["share_tot_bet"] = rng([next(x["share"] for x in MIX["mix"]["bettable"][s]["group"] if x["key"] == "TOTAL") for s in sel],
                             d=0, signed=False, unit=" %")
    spread = []
    for s in FOLDS[:2]:
        base = fb[s]["A"]["roi"]
        for variants in MIX["sensitivity"]["bettable"][s].values():
            spread += [v["roi"] - base for v in variants]
    K["sens_spread"] = rng(spread, unit=" п.п.")
    K["sens_max_roi"] = f(max(v["roi"] for s in FOLDS[:2] for vs in MIX["sensitivity"]["bettable"][s].values() for v in vs))
    pa = {s: PR[s]["A"]["prices"] for s in sel}
    K["best_gain"] = rng([pa[s]["лучшая Bet365/Pinnacle откр."]["roi"] - pa[s]["Bet365 откр. (продакшн)"]["roi"] for s in sel], unit=" п.п.")
    K["max_gain"] = rng([pa[s]["Max FD откр. (инф.)"]["roi"] - pa[s]["Bet365 откр. (продакшн)"]["roi"] for s in sel], unit=" п.п.")
    K["max_roi"] = rng([pa[s]["Max FD откр. (инф.)"]["roi"] for s in sel], unit=" %")
    K["clv_fair"] = rng([PR[s]["A"]["clv"]["mean_clv_fair"] for s in sel], unit=" %")
    K["ll_gap"] = rng([x["1X2 multiclass log loss model"] - x["1X2 multiclass log loss market"] for s in sel for x in MOD[s].values()],
                      1, 3, False) if MOD else "—"
    K["mod_roi"] = rng([x["main_bettable"]["roi"] for s in sel for x in MOD[s].values()], unit=" %") if MOD else "—"
    K["ci_halfwidth_A"] = rng([(fb[s]["A"]["roi_ci95"][1] - fb[s]["A"]["roi_ci95"][0]) / 2 for s in sel], signed=False, unit=" п.п.")
    K["ci_halfwidth_G7"] = rng([(fb[s][g7]["roi_ci95"][1] - fb[s][g7]["roi_ci95"][0]) / 2 for s in sel], signed=False, unit=" п.п.")
    K["pin_2025"] = share(next(r["Pinnacle 1X2"] for r in AUD["by_season"] if int(r["season"]) == 2025), 0)
    if IND:
        b = IND["stored_A"]
        K["live_settled"] = b["settled"]
        K["live_ci_half"] = pp((b["roi_ci95"][1] - b["roi_ci95"][0]) / 2)
    return K


K: dict = {}


# --------------------------------------------------------------------------------------------- sections
def s_goal(rep):
    rep.add(H1("1. Цель"),
            P("Main — одна рекомендация на матч, которую показывает сайт FootPredict. Он угадывает примерно две "
              "трети исходов, но при ставке 1 единица по реальной цене теряет деньги. Цель исследования — "
              "выяснить, **почему** Main убыточен, и проверить, существует ли честно валидированная стратегия "
              "выбора, которая даёт хотя бы небольшой положительный ROI (+1–2 %). Подгонка под положительное "
              "число не допускается: отрицательный результат — допустимый и полноценный исход."),
            P("Отдельно разделяются две задачи: (1) *прогноз Main* — всегда один на матч, показывается на сайте; "
              "(2) *решение о ставке* — ставить или пропустить по реальной цене. Сравниваются семь семейств "
              "стратегий (A–G) на одних и тех же матчах, с бутстреп-интервалами и проверкой стабильности по "
              "сезонам, а выбранная заранее стратегия проверяется на независимом сезоне 2026/27."))


def s_problem(rep):
    rep.add(H1("2. Исходная проблема"))
    if IND:
        b = IND["stored_A"]
        rep.add(P("Пересчёт по хранилищу продакшна (`unified_history(store.load_picks())`, сезон 2026/27, "
                  "Main как записано, реальные цены — записанные или восстановленные из архива OddsPapi):"),
                table([["Показатель", "Значение"],
                       ["Матчей с Main", b["matches"]], ["Рассчитано (выигрыш/проигрыш)", b["settled"]],
                       ["Из них с реальной ценой (ставки)", f"{b['bets']} ({share(b['priced_share'])})"],
                       ["Доля угаданных (все рассчитанные)", share(b["hit_rate"])],
                       ["Доля угаданных (ставки)", share(b["bet_hit_rate"])],
                       ["Средний коэффициент", num(b["avg_odds"], 3)],
                       ["Безубыточная доля угаданных (среднее 1/коэф.)", share(b["breakeven_hit"])],
                       ["Средняя заявленная вероятность (все / ставки)", f"{share(b['mean_prob'])} / {share(b['bet_mean_prob'])}"],
                       ["Средний заявленный EV ставок", f(b["stated_ev"])],
                       ["Прибыль, ед.", num(b["profit"], 1)],
                       ["ROI [95 % ДИ]", f"{f(b['roi'])} {ci(b['roi_ci95'])}"]], widths=None),
                note(f"Main заявляет в среднем {share(b['bet_mean_prob'])} на ставку, а угадывает "
                     f"{share(b['bet_hit_rate'])}; при среднем коэффициенте {num(b['avg_odds'], 2)} для "
                     f"безубыточности нужно {share(b['breakeven_hit'])}. Модель считает, что у ставок есть "
                     f"преимущество {f(b['stated_ev'])}, фактически ROI {f(b['roi'])}. Расхождение между "
                     "заявленной и фактической вероятностью и есть предмет исследования."))
    a = {s: ST["modes"]["bettable"]["folds"][s]["A"] for s in FOLDS}
    ap = {s: ST["modes"]["production"]["folds"][s]["A"] for s in FOLDS}
    rows = [["Сезон", "Вселенная", "Main", "Ставок", "Угадано", "Заявлено", "Ср. коэф.", "ROI [95 % ДИ]"]]
    for s in FOLDS:
        for name, x in (("все кандидаты", ap[s]), ("только с ценой", a[s])):
            rows.append([SL[s], name, x["settled"], x["bets"], share(x["bet_hit_rate"]), share(x["bet_mean_prob"]),
                         num(x["avg_odds"], 2), f"{f(x['roi'])} {ci(x['roi_ci95'])}"])
    rep.add(P("То же правило на исторических сезонах (walk-forward: модели обучены только на более ранних сезонах; "
              "цены — реальные Bet365 до матча из Football-Data). Колонки «Угадано» и «Заявлено» — по ставкам:"),
            table(rows),
            P("Убыток устойчив во всех сезонах, на обеих «вселенных» кандидатов и в независимом сезоне. Он не "
              "объясняется одной неудачной лигой или рынком (см. разделы 8.1 и 8.6)."))


def s_impl(rep):
    rep.add(H1("3. Текущая реализация"),
            P("Код: `src/markets/recommendation.py` (`main_prediction`, `rule_score`, `effective_odds`, "
              "`EXCLUDED_GROUPS`), константы — `src/config.py`. Вероятности всех рынков выводятся из одной матрицы "
              "счетов: независимый Пуассон по ожидаемым голам двух XGBoost-регрессоров (MAX_GOALS = 10), "
              "области «П1/Н/П2» которой масштабируются к вероятностям классификатора 1X2 "
              "(`reconcile_with_1x2`); угловые — отрицательное биномиальное по XGBoost-прогнозу суммы. "
              "Коэффициенты не являются признаками моделей (`USE_ODDS_FEATURES = False`)."),
            H2("Правило Main"),
            P("Кандидаты — все исходы, кроме точного счёта и европейской форы, с вероятностью в диапазоне "
              "[MAIN_MIN_PROBABILITY, MAIN_MAX_PROBABILITY] = [0.40; 0.85]. Для каждого считается оценка:"),
            formula("score = R(рынок) · p^{γ} · w(k) · (1 + λ · clip(EV, −0.3; 0.3))  — при реальной цене"),
            formula("score = R(рынок) · p^{γ} · w(1/p) · 0.8  — без реальной цены"),
            formula("EV = p · k − (1 − p_{push}),   w(k) = exp(−½ · (d / σ)^{2}),   d — расстояние ln k до [ln 1.40; ln 2.50]"),
            P("Выбирается кандидат с наибольшей оценкой (при равенстве — с ценой, затем более вероятный). Если в "
              "диапазоне нет ни одного кандидата — самый вероятный исход 1X2."),
            table([["Константа", "Значение", "Смысл"],
                   ["MAIN_MIN/MAX_PROBABILITY", "0.40 / 0.85", "диапазон вероятности кандидатов"],
                   ["MAIN_PREFERRED_ODDS", "1.40–2.50", "мягкое предпочтение коэффициента"],
                   ["MAIN_ODDS_SOFTNESS σ", "0.15", "ширина затухания вне окна (лог-шкала)"],
                   ["MAIN_EV_WEIGHT λ", "0.5", "бонус/штраф за EV при реальной цене"],
                   ["MAIN_UNPRICED_FACTOR", "0.8", "множитель кандидата без цены"],
                   ["MAIN_PROBABILITY_POWER γ", "1.0", "тяга к самому вероятному исходу"],
                   ["MAIN_VALUE_FIRST_MIN_EV", "None", "этап «сначала ценные» отключён"],
                   ["MARKET_RELIABILITY", "1X2 1.00, TOTAL 1.00, BTTS 0.95, DC 0.90, HANDICAP 0.90, CORNERS 0.85",
                    "доверие к рынку"]]),
            P("Воспроизводимость: векторизованная копия правила (`experiments/v6_1/main/strategies.py`) сверена с "
              "продакшн-функциями на случайных матчах каждого фолда (`results/self_check.json`): " +
              "; ".join(f"{SL[s]} — совпало {v['same_main']} из {v['matches']} Main, макс. расхождение вероятностей "
                        f"{v['max_probability_diff']:.1e}, цен {v['max_odds_diff']:.1e}" for s, v in (CHK or {}).items()) + "."))


def s_hyp(rep):
    rep.add(H1("4. Гипотезы"),
            *bullets([
                "**H1 (проклятие победителя).** Выбирая максимум оценки среди многих кандидатов, правило систематически "
                "выбирает те, где модель ошиблась вверх: у выбранного Main заявленная вероятность выше фактической, "
                "даже если модель в целом откалибрована.",
                "**H2 (маржа).** Букмекерская маржа на фаворитах (коэффициенты 1.3–2.0) составляет 2–5 %; без реального "
                "информационного преимущества ROI ≈ −маржа.",
                "**H3 (информация рынка).** Вероятность букмекера без маржи точнее модели; отклонение модели от рынка "
                "почти не содержит сигнала, поэтому EV модели — в основном шум.",
                "**H4 (рыночный микс).** Слишком частый выбор тоталов (или конкретных линий) ухудшает ROI.",
                "**H5 (цена).** Лучшая доступная одновременно цена или ставка до движения линии (CLV) может "
                "перекрыть убыток.",
                "**H6 (калибровка).** Калибровка по рынкам или поправка на неопределённость уменьшит переоценку и "
                "улучшит выбор.",
                "**H7 (выборочные ставки).** Main показывать всегда, а ставить только при условии (EV, согласие с "
                "рынком, мета-модель) — так можно отсечь убыточные ставки.",
            ]))


def s_data(rep):
    rows = [["Сезон", "Строк FD", "B365 1X2", "Pinnacle 1X2", "Max/Avg 1X2", "B365 O/U 2.5", "B365 AH", "Закр. B365 1X2",
             "Закр. Pinnacle 1X2"]]
    for r in AUD["by_season"]:
        rows.append([C.season_label(int(r["season"])), int(r["rows"]), share(r["B365 1X2"]), share(r["Pinnacle 1X2"]),
                     share(r["Max 1X2"]), share(r["B365 O/U2.5"]), share(r["B365 AH"]), share(r["B365 closing 1X2"]),
                     share(r["Pinnacle closing 1X2"])])
    live = g(IND, "live_table", default={})
    rep.add(H1("5. Данные и периоды"),
            table([["Период", "Роль", "Источник вероятностей", "Цены"],
                   ["2023/24, 2024/25", "выбор вариантов (фолды отбора)",
                    "walk-forward: продакшн-XGBoost (параметры `prod_20261012`), обучение на сезонах < s",
                    "Football-Data: Bet365 до матча (продакшн), Pinnacle, Max/Avg, закрытие"],
                   ["2025/26", "только для информации (смотрели при разработке)", "то же", f"то же (Pinnacle — {K['pin_2025']} матчей)"],
                   ["2026/27 (по 2026-11)", "независимая проверка", "хранилище продакшна: вероятности, вычисленные до матча",
                    "записанные продакшном / архив OddsPapi строго до начала; FD 2026/27 для V2"],
                   ["2022/23", "внутренний сезон фолда 2023/24", "модели, обученные на 2021/22", "Football-Data"]]),
            P("Каждая подгоняемая часть стратегии для сезона s (калибровка, вес смеси, мета-модель, λ, исключаемые "
              "группы) обучается на сезоне s−1, предсказанном моделями, обученными на сезонах < s−1. Для 2026/27 — на "
              "2025/26 (модели < 2025/26)."),
            H2("Какие цены реально есть"),
            P("Доля непустых значений столбцов в сырых файлах Football-Data (`results/fd_price_audit.json`, по лигам — "
              f"`fd_price_columns_by_league.csv`). Pinnacle исчезает в 2025/26 ({K['pin_2025']}) и отсутствует в 2026/27; "
              "1XBet и Betfair есть только в 2024/25 — поэтому «лучшая цена» сравнивается по паре Bet365/Pinnacle:"),
            table(rows),
            P("Football-Data покрывает три рынка: 1X2, тотал 2.5 и одну азиатскую линию матча (полу- и целые линии; "
              "четвертные не используются, как в продакшне). Остальные кандидаты Main (тоталы 1.5/3.5, обе забьют, "
              "двойной шанс, прочие линии форы, угловые) на исторических сезонах цены не имеют."),
            H2("Две «вселенные» кандидатов"),
            *bullets([
                "**Только с ценой (основная для сравнения ставок).** Матчи, где есть цена Bet365 на все исходы 1X2 и "
                "тотала 2.5; кандидаты — исходы с реальной ценой. Каждая стратегия выбирает один Main на матч из тех же "
                "кандидатов и рассчитывается по той же цене — одинаковые матчи для всех.",
                "**Все кандидаты (как на сайте).** Правило видит все рынки; ROI считается только по тем Main, у которых "
                "есть цена. Показано как вторичный вид: на исторических сезонах цена есть лишь у половины Main.",
            ]),
            P(f"2026/27: {live.get('matches', '—')} матчей единой истории, {live.get('rows', '—')} кандидатов; "
              f"происхождение цены кандидатов: " + ", ".join(f"{k} — {v}" for k, v in (live.get("price_origin") or {}).items())
              + f". Пересчитанное правило A на дополненной таблице совпадает с записанным Main в "
                f"{share(live.get('same_main_as_stored'))} матчей (остальные записаны, когда часть цен ещё не была "
                f"известна)."))


def s_methods(rep):
    crit = PRE["criteria"] if PRE else {}
    strat = (PRE or {}).get("strategy", {})
    rep.add(H1("6. Методы и параметры"),
            H2("Метрики"),
            formula("ROI = Σ прибыль / число ставок;  выигрыш: k − 1,  проигрыш: −1,  возврат и отмена — не ставка"),
            formula("Brier = mean((p − y)^{2}),   LogLoss = −mean(y·ln p + (1 − y)·ln(1 − p))"),
            formula("ECE = Σ_{b} (n_{b} / n) · |ȳ_{b} − p̄_{b}|,  10 равных бинов по p"),
            P("Для азиатской целой линии вероятность сравнивается с исходом на шкале «без возврата»: p / (1 − p_{push})."),
            H2("Удаление маржи"),
            formula("пропорционально: q_{i} = (1/k_{i}) / Σ_{j} (1/k_{j})"),
            formula("Шин: q_{i} = (√(z^{2} + 4(1 − z)·π_{i}^{2}/Σπ) − z) / (2(1 − z)),  π_{i} = 1/k_{i},  z: Σq_{i} = 1"),
            P("Для двойного шанса сумма вероятностей равна 2; для целой азиатской линии цены условны на отсутствие "
              "возврата, q переводится в безусловную через p_{push} модели."),
            H2("Подгоняемые преобразования (только на внутреннем сезоне)"),
            *bullets([
                "изотоническая и логистическая (Платт) калибровка по ключу «группа | сторона» (TOTAL|OVER, 1X2|D, …); "
                "ключи с < 200 наблюдений используют карту группы;",
                "калибровка «по выбранным» — изотоническая карта, обученная только на Main правила A внутреннего сезона;",
                "смесь p' = w·p + (1 − w)·q, w ∈ {0; 0.05; …; 1} по минимуму log loss на всех кандидатах с ценой, "
                "отдельно для 1X2, тоталов и форы;",
                "штраф за число кандидатов p' = p − λ·ln n, λ так, чтобы средняя переоценка выбранных стала ≈ 0;",
                "нижняя граница p' = p_{iso} − SE (стандартная ошибка доли угаданных ключа и бина 0.05);",
                "мета-модель — логистическая регрессия P(выигрыш) Main внутреннего сезона по признакам logit p, logit q, "
                "их разности, ln k, p_{push}, индикаторам группы, Under и линии (без числа кандидатов — оно не "
                "переносится между вселенными).",
            ]),
            H2("Доверительные интервалы и сравнение"),
            P(f"ROI — перцентильный бутстреп по ставкам, {C.N_BOOT} повторов, seed {C.SEED}. Разница стратегий — "
              "парный бутстреп по матчам (один матч — одна строка, у каждой стратегии своя ставка или пропуск). "
              "Модели XGBoost повторяют продакшн (параметры и число деревьев активной версии), не более "
              f"{C.N_JOBS} потоков."),
            H2("Предрегистрация"),
            P(f"Критерии внедрения записаны и закоммичены **до** оценки любой альтернативной стратегии "
              f"(`results/preregistration.json`, этап criteria: {crit.get('timestamp_utc', '—')} UTC). Выбранная стратегия "
              f"записана до проверки на 2026/27 (этап strategy: {strat.get('timestamp_utc', 'не записан')} UTC; "
              f"коммит: {g(IND, 'preregistration_commit', default='—')})."),
            *bullets(crit.get("adoption_rule", [])),
            P("Выбор варианта внутри семейства: " + crit.get("variant_choice", "—") + "."))


def s_experiments(rep):
    ran = [["№", "Эксперимент", "Выполнен", "Комментарий"]]
    m = ST["modes"]["bettable"]["folds"][FOLDS[0]]
    ran += [
        ["1", "Калибровка выбранных Main (группы, лиги, сезоны, коэф., вероятности, линии)", "да", "calibration.json; 2026/27 — independent.json"],
        ["2", "Проклятие победителя, число кандидатов, инфляция EV, корреляция рынков, симуляция", "да", "selection_bias.json"],
        ["3", f"Стратегии A–G ({len(m)} вариантов) на одних матчах", "да", "strategies.json, обе вселенные, 3 сезона"],
        ["4", "Main vs решение о ставке (пороги EV, коэф., вероятности)", "да", "betting.json"],
        ["5", "Информация букмекера: смесь, модель расхождения, мета-модель", "да", "selection_bias.json, strategies.json (F, G7/G8)"],
        ["6", "Рыночный микс и чувствительность констант", "да", "market_mix.json"],
        ["7", "Лучшая цена и CLV", "да", "prices_clv.json; на 2026/27 — только Bet365 закрытие (Pinnacle в FD нет)"],
        ["8", "Альтернативные модели (CatBoost, LightGBM, GLM, ансамбли, Пуассон)", "да" if MOD else "нет",
         "по сохранённым прогнозам experiments/model_comparison, без переобучения: анализ 3 показал, что модель не "
         "добавляет информации к рынку — проверено, исправит ли это другой алгоритм"],
        ["9", "Независимая проверка 2026/27 предрегистрированной стратегии", "да" if IND else "нет", "independent.json"],
        ["—", "Отдельная калибровка по каждой лиге", "нет", "в лиге лишь несколько сотен Main за сезон: карта шумнее эффекта; "
         "лиги показаны как разрез калибровки"],
        ["—", "Переобучение моделей с коэффициентами в признаках", "нет", "смена модели вне рамок задачи; "
         "смесь F показывает верхнюю границу пользы рыночной информации"],
        ["—", "Ставки на тоталы 1.5/3.5, обе забьют, угловые на истории", "нет", "в Football-Data нет цен; эти рынки "
         "проверены только на 2026/27"],
    ]
    rep.add(H1("7. Эксперименты"), P("Что было запущено (да/нет и почему):"),
            table(ran, widths=[0.6 * 28.35, 7.2 * 28.35, 1.5 * 28.35, 7.7 * 28.35]))


def calib_table(rows, title_key="Группа", top=None, fmt_key=None):
    out = [[title_key, "n", "Заявлено", "Угадано", "Разница", "Brier", "LogLoss", "ECE", "Ставок", "ROI"]]
    for r in (rows if top is None else sorted(rows, key=lambda r: -r["n"])[:top]):
        if r["n"] < 20:
            continue
        key = C.GROUP_RU.get(r["key"], r["key"]) if fmt_key is None else fmt_key(r["key"])
        out.append([key, r["n"], share(r["mean_prob"]), share(r["hit_rate"]), pp(r["gap"]), num(r["brier"], 4),
                    num(r["log_loss"], 4), num(r["ece"], 4), r["bets"], f(r["roi"]) if r["roi"] is not None else "—"])
    return table(out)


def s_res_calibration(rep):
    rep.add(H1("8. Результаты"), H2("8.1 Калибровка выбранных Main"))
    rows = [["Сезон", "Вселенная", "Что", "n", "Заявлено", "Угадано", "Разница", "Brier", "LogLoss", "ECE"]]
    for s in FOLDS:
        for mode, mname in (("production", "все"), ("bettable", "с ценой")):
            r = CAL[mode][s]
            for what, x in (("Main (A)", r["overall"]), ("все кандидаты 0.40–0.85", r["all_candidates"]),
                            ("рынок на тех же Main", r["market_on_picks"])):
                rows.append([SL[s], mname, what, x["n"], share(x["mean_prob"]), share(x["hit_rate"]), pp(x["gap"]),
                             num(x["brier"], 4), num(x["log_loss"], 4), num(x["ece"], 4)])
    rep.add(P("Сравнение: выбранный Main, все кандидаты того же диапазона вероятностей и вероятность букмекера "
              "без маржи (Шин, Bet365) на тех же Main:"), table(rows),
            note(f"Все кандидаты модели откалиброваны почти идеально (разница {K['gap_all']}, ECE {K['ece_all']}), а "
                 f"выбранный из них Main переоценён на {K['gap_main_prod']} (только с ценой — {K['gap_main_bet']}). Значит, переоценка возникает **при выборе**, а не в "
                 f"модели вообще. Вероятность букмекера на тех же Main почти откалибрована (разница {K['gap_mkt_picks']}) и даёт меньший "
                 "Brier/LogLoss, чем модель."),
            *figure(FIG / "calib_reliability_A.png", "Рис. 1. Надёжность выбранных Main (A) по сезонам; линия — идеальная калибровка."),
            *figure(FIG / "calib_gap_groups.png", "Рис. 2. Переоценка по группам: выбранный Main и все кандидаты."))
    for mode, mname in (("production", "все кандидаты"), ("bettable", "только с ценой")):
        for s in FOLDS[:2] + [FOLDS[2]]:
            r = CAL[mode][s]
            rep.add(H3(f"Группы рынков — {SL[s]}, {mname}"), calib_table(r["by_group"]))
    for s in FOLDS:
        r = CAL["production"][s]
        rep.add(H3(f"Линии и стороны — {SL[s]}, все кандидаты"), calib_table(r["by_line"], "Линия", top=12),
                calib_table(r["by_side"], "Сторона", top=10),
                H3(f"Диапазоны вероятности и коэффициента — {SL[s]}"), calib_table(r["by_prob_band"], "p"),
                calib_table([x for x in r["by_odds_band"] if x["key"] != "nan"], "Коэф. (Main с ценой)"),
                H3(f"Лиги (12 крупнейших по числу Main) — {SL[s]}"), calib_table(r["by_league"], "Лига", top=12))
    rep.add(
            *figure(FIG / "calib_reliability_groups.png", "Рис. 3. Надёжность Main по группам рынков (все кандидаты)."),
            P(f"Сильнее всего переоценены азиатские форы (на {K['gap_ah']}): линия форы выставлена рынком около 50/50, и "
              "форой Main становится только там, где модель сильно расходится с рынком. Тоталы переоценены меньше "
              f"({K['gap_tot']}; Under в большинстве сезонов сильнее Over), 1X2 — на {K['gap_1x2']}."))


def s_res_bias(rep):
    rep.add(H2("8.2 Проклятие победителя"))
    rows = [["Сезон", "Вселенная", "Выбранный: заявлено − факт", "Невыбранные", "Невыбранные (взвеш. по p выбранных)",
             "n выбранных", "n невыбранных"]]
    for s in FOLDS:
        for mode, mname in (("production", "все"), ("bettable", "с ценой")):
            c = BIAS[mode][s]["chosen_vs_rest"]
            rows.append([SL[s], mname, pp(c["gap_chosen"]), pp(c["gap_rest"]), pp(c["gap_rest_reweighted"]),
                         c["n_chosen"], c["n_rest"]])
    rep.add(table(rows), *figure(FIG / "bias_chosen_vs_rest.png",
                                 "Рис. 4. При одинаковой заявленной вероятности выбранный кандидат угадывает реже невыбранного."))
    rows = [["Сезон", "Вселенная", "Квинтиль числа кандидатов", "n", "Разница", "±1.96·SE", "ROI ставок"]]
    for s in FOLDS[:2]:
        for mode, mname in (("production", "все"), ("bettable", "с ценой")):
            for x in BIAS[mode][s]["by_ncand"]:
                rows.append([SL[s], mname, x.get("range", x["band"]), x["n"], pp(x["gap"]), pp(1.96 * x["se"]),
                             f(x["roi"]) if x["roi"] is not None else "—"])
    rep.add(H3("Зависимость от числа кандидатов"), table(rows),
            *figure(FIG / "bias_ncand.png", "Рис. 5. Переоценка выбранного Main по квинтилям числа кандидатов."))
    rows = [["Сезон", "Заявленный EV (диапазон)", "n", "Ср. заявленный EV", "Факт. ROI [95 % ДИ]", "Ср. коэф."]]
    for s in FOLDS:
        for x in BIAS["bettable"][s]["ev_inflation_all_priced"]:
            rows.append([SL[s], x["band"], x["n"], f(x["stated_ev"]), f"{f(x['realised_roi'])} {ci(x['ci95'])}",
                         num(x["avg_odds"], 2)])
    rep.add(H3("Инфляция EV"),
            P("Все кандидаты с ценой в диапазоне 0.40–0.85, сгруппированные по EV модели. Если бы EV был верен, "
              "фактический ROI рос бы вместе с ним:"), table(rows),
            *figure(FIG / "bias_ev_inflation.png", "Рис. 6. Заявленный EV против фактического ROI."),
            note(f"Фактический ROI почти не зависит от заявленного EV: от «EV −15 %» до «EV +30 %» ROI остаётся в "
                 f"коридоре {K['ev_corridor']} (бины с ≥ 500 кандидатами). Заявленный EV модели не различает хорошие и плохие ставки."))
    rows = [["Сезон", "SD шума logit (1X2/TOTAL/AH)", "Ожидаемая переоценка выбранного", "Наблюдаемая",
             "Ожидаемый истинный EV", "Наблюдаемый ROI A", "EV по рынку на Main", "Маржа на Main"]]
    for s in FOLDS:
        b = BIAS["bettable"][s]
        sm = b["simulation"]
        sd = sm["noise_sd_logit"]
        rows.append([SL[s], "/".join(f"{sd.get(k, float('nan')):.2f}" for k in ("1X2", "TOTAL", "HANDICAP")),
                     pp(sm["expected_gap_chosen"]), pp(BIAS["bettable"][s]["chosen_vs_rest"]["gap_chosen"]),
                     f(sm["expected_true_ev_chosen"]), f(ST["modes"]["bettable"]["folds"][s]["A"]["roi"]),
                     f(b["market_ev_chosen"]), share(b["margin_chosen"])])
    rep.add(H3("Симуляция: модель = рынок + шум"),
            P("Пусть истинная вероятность — вероятность букмекера без маржи, а модель — она же плюс независимый шум "
              "той величины, на которую модель реально отличается от рынка (верхняя оценка чистого шума). Правило A "
              "применяется к таким «моделям» (20 повторов):"), table(rows),
            note(f"Модель «чистый шум вокруг рынка» воспроизводит и переоценку выбранного Main ({K['sim_gap']} против "
                 f"наблюдаемых {K['gap_main_bet']}), и его ROI ({K['sim_ev']} против {K['obs_roi_A']}). Наблюдаемое поведение Main "
                 "полностью объясняется шумом выбора и маржой; признаков информационного преимущества нет."))
    rc = BIAS["production"][FOLDS[1]]["residual_correlation"]
    rep.add(H3("Корреляция между рынками"),
            P(f"Остатки (исход − p) разных рынков одного матча сильно связаны: средний модуль внедиагональной "
              f"корреляции {rc['mean_abs_offdiag']:.2f} ({SL[FOLDS[1]]}). Кандидаты — не независимые ставки: тоталы "
              "1.5/2.5/3.5 и «обе забьют» отражают одну ошибку прогноза голов, поэтому выбор между ними — выбор "
              "направления ошибки модели, а не диверсификация."),
            *figure(FIG / "bias_residual_corr.png", "Рис. 7. Корреляция остатков между рынками одного матча."))


def s_res_market(rep):
    rep.add(H2("8.3 Информация букмекера"))
    rows = [["Сезон", "Группа", "n", "c (сигнал расхождения)", "95 % ДИ", "b (рынок)", "ср. |p − q|"]]
    for s in FOLDS:
        for gname, x in BIAS["bettable"][s]["disagreement"].items():
            if "c_disagreement" in x:
                rows.append([SL[s], C.GROUP_RU.get(gname, "все") if gname != "ALL" else "все", x["n"],
                             num(x["c_disagreement"], 3), f"[{x['ci_disagreement'][0]:+.3f}; {x['ci_disagreement'][1]:+.3f}]",
                             num(x["b_market"], 3), share(x["mean_abs_disagreement"])])
    rep.add(P("Логистическая регрессия на всех кандидатах с ценой:"),
            formula("logit P(выигрыш) = a + b · logit q + c · (logit p − logit q)"),
            P("Если модель знает что-то сверх рынка, c > 0 (c = 1 — модель полностью права в расхождении; c = 0 — "
              "расхождение — шум):"), table(rows))
    rows = [["Сезон", "Группа", "LogLoss модель", "Рынок Шин", "Рынок пропорц.", "Смесь (вес с внутр. сезона)", "Pinnacle Шин"]]
    for s in FOLDS:
        for gname, x in BIAS["bettable"][s]["losses"].items():
            rows.append([SL[s], "все" if gname == "ALL" else C.GROUP_RU.get(gname, gname), num(x["model"]["log_loss"], 4),
                         num(x["market_shin"]["log_loss"], 4), num(x["market_prop"]["log_loss"], 4),
                         num(x["blend_inner"]["log_loss"], 4),
                         num(x["pinnacle_shin"]["log_loss"], 4) if "pinnacle_shin" in x else "—"])
    w = {s: ST["fitted"]["bettable"][s]["blend_weight_shin"] for s in FOLDS}
    rep.add(table(rows),
            P("Вес модели в смеси, подобранный на внутреннем сезоне (Шин): " +
              "; ".join(f"{SL[s]}: " + ", ".join(f"{C.GROUP_RU.get(k, k) if k != 'ALL' else 'все'} {v:.2f}" for k, v in w[s].items())
                        for s in FOLDS) + "."),
            note(f"Коэффициент c ({K['c_range']}) во всех сезонах и группах {'статистически неотличим от нуля' if K['c_all_zero'] else 'почти всегда неотличим от нуля'}, а вес модели в смеси — "
                 f"{K['w_blend']}. На рынках, где есть цена, модель не содержит информации сверх вероятности Bet365 без маржи. "
                 "Это корень проблемы: любое правило, которое вознаграждает расхождение модели с рынком (EV), "
                 "выбирает шум."))


def strat_table(mode, names=None, cols=("roi", "bets")):
    folds = ST["modes"][mode]["folds"]
    names = names or list(folds[FOLDS[0]].keys())
    rows = [["Стратегия"] + [f"ROI {SL[s]}" for s in FOLDS] + [f"Ставок {SL[s]}" for s in FOLDS] + ["Разница с A (2023+24) [95 % ДИ]"]]
    pooled = ST["pooled_vs_A"]
    for n in names:
        r = [ru(n)]
        for s in FOLDS:
            x = folds[s][n]
            r.append(f"{f(x['roi'])} {ci(x['roi_ci95'], 1)}" if x["roi"] is not None else "—")
        for s in FOLDS:
            r.append(folds[s][n]["bets"])
        if mode == "bettable" and n in pooled and pooled[n]["diff"] is not None:
            r.append(f"{pp(pooled[n]['diff'])} {ci(pooled[n]['ci95'])}")
        else:
            r.append("—" if n != "A" else "база")
        rows.append(r)
    return table(rows, font=6.3)


def s_res_strategies(rep):
    rep.add(H2("8.4 Стратегии A–G на одних матчах"),
            P("Основная вселенная — только кандидаты с реальной ценой Bet365 (одинаковые матчи и цены для всех). "
              "ROI с 95 % бутстреп-ДИ; последняя колонка — парная разница ROI со стратегией A на объединённых "
              "фолдах отбора."))
    fam = {}
    for n in ST["modes"]["bettable"]["folds"][FOLDS[0]]:
        fam.setdefault(ST["modes"]["bettable"]["folds"][FOLDS[0]][n]["family"], []).append(n)
    from .strategies_eval import FAMILIES
    for k, title in FAMILIES.items():
        names = (["A"] if k != "A" else []) + fam.get(k, [])
        rep.add(H3(f"{k}: {title}"), strat_table("bettable", names))
    rows = [["Стратегия", "Разница 2023/24", "Разница 2024/25", "ECE Main 2023/24", "ECE Main 2024/25", "Brier 2024/25",
             "Доля тоталов 2024/25"]]
    folds = ST["modes"]["bettable"]["folds"]
    for n in folds[FOLDS[0]]:
        if folds[FOLDS[0]][n]["family"] == "G":
            continue
        rows.append([ru(n)] + [pp((folds[s][n]["vs_A"] or {}).get("diff")) if n != "A" else "база" for s in FOLDS[:2]]
                    + [num(folds[s][n]["calibration_main"]["ece"], 4) for s in FOLDS[:2]]
                    + [num(folds[FOLDS[1]][n]["calibration_main"]["brier"], 4),
                       share(folds[FOLDS[1]][n]["share_groups"].get("TOTAL", 0))])
    rep.add(H3("Калибровка показанного Main у стратегий, меняющих Main"), table(rows, font=6.5),
            *figure(FIG / "strat_overview.png", "Рис. 8a. ROI всех вариантов (≥ 300 ставок) с 95 % ДИ по сезонам."),
            *figure(FIG / "strat_reliability.png", "Рис. 8b. Надёжность показанного Main у разных стратегий."),
            *figure(FIG / "strat_cumulative.png", "Рис. 8c. Накопленная прибыль стратегий по сезонам."))
    rep.add(H3("Вторичная вселенная: все кандидаты (как на сайте)"),
            P("ROI здесь считается по разным подмножествам матчей (только Main с ценой), поэтому сравнение менее "
              "строгое:"), strat_table("production", [n for n in ST["modes"]["production"]["folds"][FOLDS[0]]
                                                       if not n.startswith("G")]))
    sel = SEL or {}
    tab = ST["modes"]["bettable"]["selection"]["table"]
    rows = [["Семейство", "Лучший вариант (худший фолд)", "ROI 2023/24", "ROI 2024/25", "Худший", "Лучше A на обоих",
             "p05 парной разницы", "Калибровка не хуже"]]
    for k, n in ST["modes"]["bettable"]["selection"]["family_best"].items():
        t = tab[n]
        pv = ST["pooled_vs_A"].get(n, {})
        rows.append([k, ru(n), f(t["roi"][0]), f(t["roi"][1]), f(t["worst_roi"]), "да" if t["better_than_A_both"] else "нет",
                     pp(pv.get("p05")) if pv else "—", "да" if t["calibration_not_worse"] else "нет"])
    cm = 28.35
    rep.add(H3("Отбор (только 2023/24 и 2024/25)"),
            table(rows, font=6.6, widths=[1.3 * cm, 4.6 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm, 1.8 * cm, 2.2 * cm, 2.4 * cm]),
            note(f"По заранее заданному правилу (наибольший ROI худшего фолда среди вариантов с ≥ 300 ставками на "
                 f"каждом фолде) предрегистрирована стратегия **{ru(sel.get('primary', '—'))}**; правила 1–3 на фолдах "
                 f"выполнены: {'да' if g(sel, 'preregistered', 'primary', 'passes_fold_rules_1_3') else 'нет'}. "
                 f"Вторичные (лучшие в своих семействах): {', '.join(ru(x) for x in sel.get('secondary', []))}. "
                 f"Варианты, прошедшие правила 1–3: {', '.join(ru(x) for x in sel.get('variants_passing_rules_1_to_3', [])) or 'нет'}."),
            P("Что показали семейства:"),
            *bullets([
                "**B (калибровка).** Карта, обученная на всех кандидатах, почти ничего не меняет — кандидаты и так "
                "откалиброваны; переоценка выбранного остаётся. Калибровка «по выбранным» (B3) убирает переоценку "
                "показанного Main почти полностью, но ROI на 2024/25 не улучшает.",
                "**C (сильнее EV).** Больший вес EV и «максимум EV» сдвигают выбор к длинным ценам и увеличивают "
                f"переоценку показанного Main ({K['gap_C']}); ни один вариант не лучше A на обоих фолдах.",
                "**D (рынки).** Ни один вариант (без тоталов, только 1X2, без переоценённых групп, надёжность тотала 0.9) "
                "не лучше A на обоих фолдах: выигрыш на одном сезоне сменяется проигрышем на другом.",
                f"**E (неопределённость).** Штраф за число кандидатов делает Main честнее (переоценка {K['e2_gap']}), но ROI "
                "почти не меняется.",
                "**F (рынок).** Смесь с рынком при весе модели ≈ 0 превращает Main в «фаворита букмекера в окне "
                f"1.40–2.50»: калибровка лучшая, ROI лучше A на обоих фолдах отбора (парно {K['f1_diff']}), но остаётся "
                f"{K['f1_roi']} — это маржа; на 2025/26 (инф.) {K['f1_roi_info']} против {K['a_roi_info']} у A.",
                "**G (выборочные ставки).** Пороги EV по сырой или калиброванной вероятности не помогают; по "
                "смеси с рынком ставок почти нет (EV < 0 почти всегда). Мета-модель G7 — единственный вариант "
                f"около нуля, но число ставок падает от фолда к фолду ({K['g7_bets']}).",
            ]))


def s_res_betting(rep):
    rep.add(H2("8.5 Main и решение о ставке"),
            P("Main = A показывается всегда; решение о ставке принимается отдельно. Все кандидаты (как на сайте):"))
    for mode, title in (("production", "все кандидаты"), ("bettable", "только с ценой")):
        rows = [["Условие ставки"] + [f"{SL[s]}: ставок / ROI / ДИ" for s in FOLDS]]
        for label in BET[mode][FOLDS[0]]:
            r = [label]
            for s in FOLDS:
                x = BET[mode][s][label]
                r.append(f"{x['bets']} / {f(x['roi']) if x['roi'] is not None else '—'} / {ci(x['roi_ci95'])}")
            rows.append(r)
        rep.add(H3(f"Вселенная: {title}"), table(rows, font=6.4))
    rep.add(*figure(FIG / "betting_thresholds.png", "Рис. 8. ROI ставок при порогах EV поверх показанного Main."),
            note("Ни один порог на сырой или калиброванной вероятности не даёт положительного ROI на обоих фолдах "
                 "отбора: чем выше заявленный EV, тем длиннее коэффициент, но не выше доходность. Мета-модель "
                 "оставляет мало ставок, и их число нестабильно."))


def s_res_mix(rep):
    rep.add(H2("8.6 Рыночный микс и чувствительность"))
    for s in FOLDS[:2]:
        rows = [["Линия", "Main", "Доля", "Угадано", "Заявлено", "Ставок", "Ср. коэф.", "ROI [95 % ДИ]"]]
        for r in MIX["mix"]["production"][s]["line_key"][:10]:
            rows.append([r["key"], r["n"], share(r["share"]), share(r["hit"]), share(r["stated"]), r["bets"],
                         num(r["avg_odds"], 2), f"{f(r['roi'])} {ci(r['roi_ci95'])}" if r["roi"] is not None else "нет цены"])
        rep.add(H3(f"Что выбирает Main — {SL[s]}, все кандидаты"), table(rows))
    rows = [["Сторона", *[f"доля {SL[s]}" for s in FOLDS], *[f"ROI {SL[s]}" for s in FOLDS]]]
    keys = [r["key"] for r in MIX["mix"]["bettable"][FOLDS[0]]["side"]]
    for k in keys:
        cells = [k]
        for s in FOLDS:
            r = next((x for x in MIX["mix"]["bettable"][s]["side"] if x["key"] == k), None)
            cells.append(share(r["share"]) if r else "—")
        for s in FOLDS:
            r = next((x for x in MIX["mix"]["bettable"][s]["side"] if x["key"] == k), None)
            cells.append(f(r["roi"]) if r and r["roi"] is not None else "—")
        rows.append(cells)
    rep.add(H3("Under и Over, стороны 1X2 и форы (только с ценой)"), table(rows),
            *figure(FIG / "mix_lines_odds.png", "Рис. 9. Линии Main и ROI по диапазонам коэффициента."))
    for mode, title in (("bettable", "только с ценой"), ("production", "все кандидаты")):
        rows = [["Параметр", "Значение", *[f"ROI {SL[s]}" for s in FOLDS], "Доля тоталов 2024/25"]]
        sens = MIX["sensitivity"][mode]
        for pname, variants in sens[FOLDS[0]].items():
            for i, v in enumerate(variants):
                rows.append([pname if i == 0 else "", v["label"], *[f(sens[s][pname][i]["roi"]) for s in FOLDS],
                             share(sens[FOLDS[1]][pname][i]["share_total"])])
        rep.add(H3(f"Чувствительность констант — {title}"), table(rows, font=6.5))
    rep.add(*figure(FIG / "mix_sensitivity_bettable.png", "Рис. 10. Чувствительность ROI к константам правила (только с ценой)."),
            *figure(FIG / "mix_sensitivity_production.png", "Рис. 11. То же, все кандидаты."),
            note(f"Тоталы действительно составляют {K['share_tot_prod']} Main (все кандидаты) и {K['share_tot_bet']} (с ценой), причём Under "
                 "чаще Over. Но их ROI сопоставим с 1X2, а изменение любой константы (надёжность тотала 0.8–1.05, вес EV "
                 "0–2, степень вероятности 0.5–3, окно коэффициентов, мягкость, нижняя граница вероятности) "
                 f"сдвигает ROI относительно A на {K['sens_spread']} (2023/24, 2024/25, только с ценой); лучший ROI в сетке {K['sens_max_roi']}. Константы не являются причиной убытка."))


def s_res_prices(rep):
    rep.add(H2("8.7 Лучшая цена и CLV"))
    labels = list(PR[FOLDS[0]]["A"]["prices"].keys())
    for name in ("A", "F1 blend Shin (inner weight)", "G7 EV>0 (meta-model)"):
        if name not in PR[FOLDS[0]]:
            continue
        rows = [["Цена"] + [f"{SL[s]}: ставок / ср. коэф. / ROI" for s in FOLDS]]
        for l in labels:
            r = [l]
            for s in FOLDS:
                x = PR[s][name]["prices"].get(l)
                r.append(f"{x['bets']} / {num(x['avg_odds'], 3)} / {f(x['roi'])}" if x and x["roi"] is not None else "—")
            rows.append(r)
        rep.add(H3(f"Одни и те же ставки, разные цены — {ru(name)}"), table(rows, font=6.5))
    rows = [["Стратегия", "Сезон", "n", "Ср. CLV (коэф.)", "Доля CLV > 0", "Ср. CLV к закр. Pinnacle без маржи",
             "Доля > 0", "ROI тех же ставок", "corr(CLV, прибыль)"]]
    for name in PR[FOLDS[0]]:
        for s in FOLDS:
            c = PR[s].get(name, {}).get("clv", {})
            if not c or not c.get("n"):
                continue
            rows.append([ru(name), SL[s], c["n"], f(c["mean_clv_odds"]), share(c["share_positive_clv_odds"]),
                         f(c["mean_clv_fair"]), share(c["share_positive_clv_fair"]), f(c["roi_same_bets"]),
                         num(c["corr_clv_profit"], 3)])
    rep.add(H3("Closing Line Value"), table(rows, font=6.4),
            *figure(FIG / "prices_roi_by_price.png", "Рис. 12. ROI одних и тех же Main при разных ценах."),
            *figure(FIG / "prices_clv_quintiles.png", "Рис. 13. ROI по квинтилям CLV."),
            note(f"Лучшая из двух одновременно доступных цен (Bet365/Pinnacle) улучшает ROI на {K['best_gain']}, "
                 f"максимум Football-Data (по книгам, которые у игрока могут отсутствовать) — на {K['max_gain']} (ROI {K['max_roi']}); "
                 f"ни одна цена не делает Main прибыльным. Средний CLV отрицателен ({K['clv_fair']} к закрытию Pinnacle без "
                 "маржи): Main в среднем выбирает цены хуже закрытия, т.е. рынок движется против выбранных исходов."),
            P("Ограничения CLV: «открытые» цены Football-Data — снимок пятницы/вторника, а не момент прогноза; "
              "закрытие есть только для 1X2, тотала 2.5 и одной линии форы; закрывающая линия форы может отличаться "
              f"от открывающей (тогда цены закрытия для ставки нет); в 2025/26 Pinnacle есть у {K['pin_2025']} матчей, в 2026/27 "
              "в Football-Data его нет — там CLV считается к закрытию Bet365."))


def s_res_models(rep):
    rep.add(H2("8.8 Альтернативные модели вероятностей"))
    if not MOD:
        rep.add(P("Не выполнено: файл results/models_cmp.json отсутствует."))
        return
    rows = [["Модель", "Сезон", "LogLoss 1X2 модель", "рынок", "c расхождения [95 % ДИ]", "Вес модели в смеси",
             "ROI A (с ценой) [95 % ДИ]", "Переоценка Main"]]
    for s in FOLDS:
        for name, x in MOD[s].items():
            d = x.get("disagreement") or {}
            w = x.get("blend_weight_inner")
            rows.append([name, SL[s], num(x["1X2 multiclass log loss model"], 4), num(x["1X2 multiclass log loss market"], 4),
                         f"{num(d.get('c_disagreement'), 3)} [{d.get('ci_disagreement', [np.nan])[0]:+.3f}; {d.get('ci_disagreement', [0, np.nan])[1]:+.3f}]" if d.get("ci_disagreement") else "—",
                         ", ".join(f"{k} {v:.2f}" for k, v in w.items()) if w else "—",
                         f"{f(x['main_bettable']['roi'])} {ci(x['main_bettable']['roi_ci95'])}",
                         pp(x["main_bettable"]["bet_mean_prob"] - x["main_bettable"]["bet_hit_rate"])])
    rep.add(P("Прогнозы альтернативных алгоритмов взяты из сохранённых результатов walk-forward сравнения "
              "(`experiments/model_comparison/results/fold_<s>.parquet`, обучение на сезонах < s); подставлены в ту же "
              "цепочку «матрица счетов → рынки → правило A» на тех же матчах и ценах. Вес смеси — по предыдущему "
              "фолду той же модели (для 2023/24 его нет)."), table(rows, font=6.2),
            *figure(FIG / "models_cmp.png", "Рис. 14. Разрыв log loss с рынком и ROI Main для разных моделей."),
            note(f"Все алгоритмы уступают рынку по log loss 1X2 на {K['ll_gap']}, у всех коэффициент c ≈ 0 и ROI Main "
                 f"отрицателен ({K['mod_roi']}). Замена XGBoost на CatBoost/LightGBM/GLM/ансамбль проблему не решает: у признаков "
                 "(без коэффициентов) нет информации, которой не было бы в цене."))


def s_res_independent(rep):
    rep.add(H2("8.9 Независимая проверка: 2026/27"))
    if not IND:
        rep.add(P("Не выполнено: результаты independent.json отсутствуют."))
        return
    prim = IND["preregistered_primary"]
    v1, v2 = IND["V1"], IND["V2"]
    rep.add(P(f"Предрегистрация закоммичена до проверки: {IND['preregistration_commit']}. Подгоняемые части — на "
              f"2025/26. Основная проверка — V1 (цены продакшна, все рынки с ценой; {v1['_matches']} матчей), "
              f"дополнительно — V2 (те же матчи, только рынки Football-Data по ценам Bet365 FD; {v2['_matches']} матчей)."))
    names = ["A", prim] + [n for n in IND["secondary"] if n != prim]
    names += [n for n in v1 if not n.startswith("_") and n not in names]
    for key, V, title in (("V1", v1, "V1 — цены продакшна"), ("V2", v2, "V2 — Football-Data Bet365")):
        rows = [["Стратегия", "Ставок", "Угадано", "Заявлено", "Ср. коэф.", "ROI [95 % ДИ]", "Разница с A [95 % ДИ]",
                 "Переоценка Main"]]
        for n in names:
            if n not in V:
                continue
            x = V[n]
            lab = ru(n) + (" ★ предрег." if n == prim else (" (втор.)" if n in IND["secondary"] else ""))
            rows.append([lab, x["bets"], share(x["bet_hit_rate"]), share(x["bet_mean_prob"]), num(x["avg_odds"], 2),
                         f"{f(x['roi'])} {ci(x['roi_ci95'])}" if x["roi"] is not None else "—",
                         f"{pp(x['vs_A']['diff'])} {ci(x['vs_A']['ci95'])}" if x.get("vs_A") and x["vs_A"]["diff"] is not None else "база" if n == "A" else "—",
                         pp(x["calibration_main"]["gap"])])
        rep.add(H3(title), table(rows, font=6.3))
    st = v1.get("_A_stored_on_universe", {})
    r4 = IND["rule4"].get(prim, {})
    rep.add(P(f"Записанный продакшном Main на матчах V1: {st.get('bets', '—')} ставок, ROI {f(st.get('roi'))} "
              f"{ci(st.get('roi_ci95'))}."),
            note(f"Правило 4 для {ru(prim)}: разница с A на V1 {pp(r4.get('V1_diff_vs_A'))} "
                 f"(ROI {f(r4.get('V1_roi'))} {ci(r4.get('V1_roi_ci95'))}, ставок {r4.get('V1_bets')}); "
                 f"выполнено: {'да' if r4.get('passes_rule4') else 'нет'}. Итог по предрегистрированным правилам: "
                 f"{'внедрять' if IND['adopt'] else 'не внедрять'}."),
            *figure(FIG / "independent_cumulative.png", "Рис. 15. 2026/27: накопленная прибыль (V1)."),
            P("Остальные варианты в таблицах выше показаны **только для информации**: они не были выбраны до проверки, "
              "и выбирать по ним задним числом — тот же отбор шума, который разбирается в разделе 8.2. Характерный "
              "пример: " + exploratory_example()))
    b = IND["stored_A"]
    rep.add(H3("Калибровка записанного Main 2026/27"),
            *figure(FIG / "independent_reliability.png", "Рис. 16. Надёжность записанных Main 2026/27.", width=11 * 28.35),
            calib_table(b["by_group"]), calib_table(b["by_line"], "Линия", top=12),
            calib_table(b["by_prob_band"], "p"), calib_table([x for x in b["by_odds_band"] if x["key"] != "nan"], "Коэф."),
            calib_table(b["by_league"], "Лига", top=12), calib_table(b["by_source"], "Источник"),
            calib_table(b["by_odds_origin"], "Цена"), calib_table(b["by_provider"], "Поставщик"))
    clv = IND.get("clv_V2", {})
    if clv:
        rows = [["Стратегия", "n", "Ср. CLV (коэф.)", "Доля CLV > 0", "CLV к закр. Bet365 без маржи", "ROI", "ROI по закрытию"]]
        for n, c in clv.items():
            if c.get("n"):
                rows.append([ru(n), c["n"], f(c["mean_clv_odds"]), share(c["share_positive_clv_odds"]),
                             f(c["mean_clv_fair_b365close"]), f(c["roi"]), f(c["roi_at_closing"])])
        rep.add(H3("CLV на 2026/27 (V2, закрытие Bet365)"), table(rows))


def exploratory_example() -> str:
    fb = ST["modes"]["bettable"]["folds"]
    v1 = IND["V1"]
    best = max((n for n in v1 if not n.startswith("_") and v1[n]["bets"] >= 1000 and v1[n]["roi"] is not None),
               key=lambda n: v1[n]["roi"])
    return (f"на V1 лучший ROI среди вариантов с ≥ 1 000 ставками у {ru(best)} ({f(v1[best]['roi'])} "
            f"{ci(v1[best]['roi_ci95'])}), но на фолдах отбора у него {f(fb[FOLDS[0]][best]['roi'])} и "
            f"{f(fb[FOLDS[1]][best]['roi'])} при {f(fb[FOLDS[0]]['A']['roi'])} и {f(fb[FOLDS[1]]['A']['roi'])} у A; "
            f"на V2 — {f(IND['V2'][best]['roi'])}. Калибровка «по выбранным» (B3), обученная на трёх рынках Football-Data, "
            f"на всех рынках V1 даёт переоценку {pp(v1['B3 selection-conditional']['calibration_main']['gap'])}: карта не "
            "переносится на рынки, которых не было при обучении.")


def s_ci(rep):
    rows = [["Что", "Период", "Оценка", "95 % ДИ", "Ставок"]]
    for s in FOLDS:
        x = ST["modes"]["bettable"]["folds"][s]["A"]
        rows.append(["ROI A (с ценой)", SL[s], f(x["roi"]), ci(x["roi_ci95"]), x["bets"]])
    prim = (SEL or {}).get("primary")
    if prim:
        for s in FOLDS:
            x = ST["modes"]["bettable"]["folds"][s][prim]
            rows.append([f"ROI {ru(prim)}", SL[s], f(x["roi"]), ci(x["roi_ci95"]), x["bets"]])
        pv = ST["pooled_vs_A"][prim]
        rows.append([f"{ru(prim)} − A (парно)", "2023/24+2024/25", pp(pv["diff"]), ci(pv["ci95"]), "—"])
    f1 = "F1 blend Shin (inner weight)"
    pv = ST["pooled_vs_A"][f1]
    rows.append([f"{ru(f1)} − A (парно)", "2023/24+2024/25", pp(pv["diff"]), ci(pv["ci95"]), "—"])
    if IND:
        b = IND["stored_A"]
        rows.append(["ROI A как записано", SL[str(C.LIVE)], f(b["roi"]), ci(b["roi_ci95"]), b["bets"]])
        for n in ("A", prim):
            x = IND["V1"].get(n)
            if x:
                rows.append([f"ROI {ru(n)} (V1)", SL[str(C.LIVE)], f(x["roi"]), ci(x["roi_ci95"]), x["bets"]])
    rep.add(H1("9. Доверительные интервалы"), table(rows),
            P(f"Полуширина 95 % интервала ROI стратегии A (≈ 7 000 ставок) — {K['ci_halfwidth_A']}; у G7 (сотни ставок) — "
              f"{K['ci_halfwidth_G7']}. Разница в 1–2 п.п. между стратегиями на одном сезоне статистически не различима; "
              "поэтому требование «лучше на обоих фолдах» и парный бутстреп обязательны."))


def live_positive() -> str:
    if not IND:
        return ""
    pos = []
    for V in ("V1", "V2"):
        for n, x in IND[V].items():
            if n.startswith("_") or n.startswith("G") or x["roi"] is None or x["bets"] < 1000:
                continue
            if x["roi"] > 0:
                pos.append((V, n, x))
    if not pos:
        return "На 2026/27 положительных точечных оценок у них тоже нет."
    excl = all(x["roi_ci95"][0] is not None and x["roi_ci95"][0] > 0 for _, _, x in pos)
    return ("На 2026/27 у части вариантов точечная оценка выше нуля (" +
            ", ".join(f"{V} {ru(n)} {f(x['roi'])}" for V, n, x in pos) + ")" +
            (", и интервал выше нуля." if excl else ", но все их 95 % интервалы включают 0, а на фолдах отбора эти "
             "варианты были хуже A или не лучше на обоих сезонах."))


def s_negative(rep):
    rep.add(H1("10. Отрицательные результаты"),
            *bullets([
                "Ни одна стратегия выбора Main (B–F) не даёт положительного ROI ни на одном из сезонов 2023/24–2025/26. "
                + live_positive(),
                "Калибровка (изотоническая, Платт) не устраняет проклятие победителя: модель уже откалибрована на всех "
                "кандидатах; переоценка возникает только при выборе.",
                "Усиление EV (вес EV 1–4, «сначала ценные», максимум EV) ухудшает калибровку показанного Main и на "
                "фолдах отбора не улучшает ROI.",
                "Пороги EV на сырой и калиброванной вероятности не отделяют прибыльные ставки.",
                f"Смесь с рынком почти полностью заменяет модель рынком (вес модели {K['w_blend']}) — это лучше A, но "
                "убыточно на размер маржи.",
                "Альтернативные алгоритмы (CatBoost, LightGBM, GLM, ансамбли, чистый Пуассон) не приближаются к рынку.",
                "Лучшая одновременная цена уменьшает убыток, но не меняет знак; CLV выбранных Main отрицателен.",
            ]))


def s_conclusions(rep):
    prim = (SEL or {}).get("primary", "—")
    adopt = IND and IND.get("adopt")
    rep.add(H1("11. Выводы"),
            *bullets([
                "**Почему Main убыточен.** (1) Модель без коэффициентов в признаках не знает о матче ничего сверх "
                f"цены букмекера: её отклонение от рынка — шум (c {K['c_range']}, вес в смеси {K['w_blend']}). (2) Правило выбирает "
                "максимум из многих коррелированных кандидатов, поэтому систематически берёт исходы, где шум "
                f"направлен вверх: Main переоценён на {K['gap_main_prod']} (ставки с ценой — {K['gap_main_prod_bets']}), хотя модель в целом "
                f"откалибрована. (3) Без преимущества ставка проигрывает маржу букмекера — {K['margin']} на выбранных "
                f"ценах; вместе с отбором шума это и даёт ROI {K['obs_roi_A']}.",
                ("Высокая доля угаданных — следствие коротких коэффициентов, а не преимущества: на 2026/27 средний "
                 f"коэффициент {num(IND['stored_A']['avg_odds'], 2)}, безубыточная доля {share(IND['stored_A']['breakeven_hit'])}, "
                 f"угадано {share(IND['stored_A']['bet_hit_rate'])}.") if IND else "—",
                f"Тоталы выбираются часто ({K['share_tot_prod']} на истории), но не являются причиной: их ROI сопоставим с 1X2.",
                f"По заранее заданному правилу отбора предрегистрирована стратегия {ru(prim)}; на 2026/27 она "
                f"{'подтвердилась' if IND and IND['rule4'].get(prim, {}).get('passes_rule4') else 'не подтвердилась'} "
                f"по правилу 4; решение о внедрении: {'да' if adopt else 'нет'}.",
                "Положительного ROI, отличимого от нуля, не найдено ни на одном фолде и ни на 2026/27.",
            ]))


def s_limits(rep):
    rep.add(H1("12. Ограничения"),
            *bullets([
                "На исторических сезонах цены есть только у трёх рынков Football-Data; рынки, которые Main выбирает "
                "чаще всего в продакшне (тоталы 1.5/3.5, другие линии форы), проверены по ценам лишь на 2026/27.",
                "«Открытые» цены Football-Data — снимок за 1–3 дня до матча, а не цена в момент прогноза; цены 2026/27 "
                "смешаны (записанные при прогнозе и последние архивные котировки до начала) и разных букмекеров "
                "(bet365, Pinnacle через OddsPapi).",
                f"2026/27 — {K.get('live_settled', '—')} рассчитанных Main: полуширина ДИ ROI {K.get('live_ci_half', '—')}; для "
                "выборочных стратегий с сотнями ставок — в несколько раз шире.",
                "Мета-модель и смесь обучены на одном предыдущем сезоне; признаки мета-модели переносятся на рынки "
                "2026/27 (больше линий) с экстраполяцией.",
                "Модели не переобучались с коэффициентами в признаках; вывод «модель не добавляет информации» "
                "относится к текущему набору признаков.",
                "Ставка фиксированная (1 единица); стратегии размера ставки (Келли) не рассматривались.",
            ]))


def g7_reco() -> str:
    g7 = "G7 EV>0 (meta-model)"
    r = (IND or {}).get("rule4", {}).get(g7)
    if not r:
        return "Слой «ставить / пропустить» не проверен на 2026/27."
    return (f"**Слой «ставить / пропустить» не внедрять.** Лучший на фолдах вариант — мета-модель G7 (логистическая "
            f"регрессия P(выигрыш) по p, q и цене) — на 2026/27 (V1) дал ROI {f(r['V1_roi'])} {ci(r['V1_roi_ci95'])} на "
            f"{r['V1_bets']} ставках, на {abs(r['V1_diff_vs_A']) * 100:.1f} п.п. {'хуже' if r['V1_diff_vs_A'] < 0 else 'лучше'} A. Если к нему вернуться, показывать только как "
            "отдельный флажок «ставка: да/нет» под неизменным Main, с числом ставок и ДИ ROI, и включать лишь после "
            "новой предрегистрированной проверки на следующем сезоне.")


def display_transfer() -> str:
    if not IND:
        return "проверка не выполнена."
    v1 = IND["V1"]
    return (f"переоценка Main у A {pp(v1['A']['calibration_main']['gap'])}, у E2 "
            f"{pp(v1['E2 candidate-count penalty']['calibration_main']['gap'])}, у B3 "
            f"{pp(v1['B3 selection-conditional']['calibration_main']['gap'])} — карта B3, обученная на трёх рынках, "
            "не переносится.")


def s_reco(rep):
    adopt = IND and IND.get("adopt")
    rep.add(H1("13. Рекомендации"),
            *bullets([
                ("Внедрить предрегистрированную стратегию (выполнены все правила)." if adopt else
                 "**Продакшн не менять:** ни одна стратегия не прошла заранее заданные правила внедрения целиком "
                 "(включая подтверждение на 2026/27); `src/markets/recommendation.py` и константы MAIN_* / "
                 "MARKET_RELIABILITY остаются прежними."),
                "Честно показывать Main как **прогноз**, а не ставку: на карточке матча рядом с вероятностью модели "
                "выводить вероятность букмекера без маржи и подпись «модель не видит преимущества над рынком» "
                "при |p − q| < 3 п.п.; убрать формулировки, которые читаются как совет ставить.",
g7_reco(),
                "Показанную вероятность Main честнее корректировать на эффект выбора: на фолдах калибровка «по "
                f"выбранным» (B3) и штраф за число кандидатов (E2) убирают переоценку (B3 {K['b3_gap']}, E2 {K['e2_gap']}) "
                "без изменения ROI. На 2026/27 (все рынки) " + display_transfer() + " Это вопрос честности интерфейса, "
                "а не прибыли; внедрять — только после отдельной проверки на рынках продакшна.",
                "Следующий шаг исследования, если цель — преимущество: признаки, которых нет в цене (составы, "
                "травмы, ранние цены против поздних), или ставки до движения линии; без этого любой выбор на "
                "рынках с ценой даёт ROI ≈ −маржа.",
            ]))


def s_repro(rep):
    env = (RUN or {}).get("environment", {})
    steps = (RUN or {}).get("steps", {})
    libs = env.get("libraries", {})
    rep.add(H1("14. Команды воспроизведения"),
            *reproduce([
                "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                "$PY -m experiments.v6_1.main.run --base --candidates     # прогнозы, кандидаты, self-check",
                "$PY -m experiments.v6_1.main.run --strategies --calibration --bias --mix --betting --prices --models --figures",
                "$PY -m experiments.v6_1.main.run --select                # results/selection.json",
                "$PY -m experiments.v6_1.main.preregister --strategy      # затем git commit preregistration.json",
                "$PY -m experiments.v6_1.main.run --independent           # 2026/27 (только после коммита)",
                "$PY -m experiments.v6_1.main.build_pdf",
            ]),
            table([["Параметр", "Значение"], ["Git SHA прогона", env.get("git_sha", "—")],
                   ["Python", env.get("python", "—")],
                   ["Библиотеки", ", ".join(f"{k} {v}" for k, v in libs.items() if v)],
                   ["Потоки на модель", env.get("n_jobs", "—")], ["Seed / бутстреп", f"{env.get('seed', '—')} / {env.get('n_boot', '—')}"],
                   ["Время шагов, мин", ", ".join(f"{k} {v['minutes']}" for k, v in steps.items())]]),
            P("Результаты: JSON и CSV в `experiments/v6_1/main/results/`, рисунки PNG в `experiments/v6_1/main/figures/`, "
              "кэш прогнозов и логи (`cache/`, `logs/`) в git не входят и пересоздаются командами выше."))


def build() -> Path:
    K.update(facts())
    prim = (SEL or {}).get("primary", "—")
    rep = Report(OUT, "Исследование стратегии Main", "FootPredict v6.1",
                 facts={"Фолды отбора": "2023/24, 2024/25", "Для информации": "2025/26",
                        "Независимая проверка": "2026/27 (хранилище продакшна)",
                        "Предрегистрированная стратегия": ru(prim),
                        "Решение": ("внедрить" if IND and IND.get("adopt") else "продакшн без изменений")})
    for fn in (s_goal, s_problem, s_impl, s_hyp, s_data, s_methods, s_experiments, s_res_calibration, s_res_bias,
               s_res_market, s_res_strategies, s_res_betting, s_res_mix, s_res_prices, s_res_models,
               s_res_independent, s_ci, s_negative, s_conclusions, s_limits, s_reco, s_repro):
        fn(rep)
    rep.build()
    n = rep.check(scale=1.0)
    print(OUT, n, "pages")
    return OUT


if __name__ == "__main__":
    build()
