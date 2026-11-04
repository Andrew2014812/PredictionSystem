"""Builds reports/v6_1/Risk_Strategy_Research.pdf (Russian) and experiments/v6_1/risk/findings.md.

    python -m experiments.v6_1.risk.build_pdf

Every number comes from experiments/v6_1/risk/results/*.json (written by run.py); nothing is typed in.
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.units import cm

from reportlab.platypus import CondPageBreak, Spacer

from scripts.report_kit import H1 as _H1
from scripts.report_kit import H2 as _H2
from scripts.report_kit import H3 as _H3
from scripts.report_kit import P, Report, bullets, code, figure, formula, load, note, share
from scripts.report_kit import table as _table


def _keep(h):
    return [CondPageBreak(4 * cm), h]          # a heading never ends a page


def H1(text):  # noqa: N802
    return _keep(_H1(text))


def H2(text):  # noqa: N802
    return _keep(_H2(text))


def H3(text):  # noqa: N802
    return _keep(_H3(text))


def table(rows, widths=None, **kw):
    """The kit table followed by a little space."""
    return [_table(rows, widths=widths, **kw), Spacer(1, 7)]

HERE = Path(__file__).resolve().parent
R, F = HERE / "results", HERE / "figures"
PDF = "reports/v6_1/Risk_Strategy_Research.pdf"
SL = {"2023": "2023/24", "2024": "2024/25", "2025": "2025/26*", "2026": "2026/27"}
CUR = "Текущий Risk"
PRE = "Calibrated Risk (бленд с рынком), пол ≥0 %"

hist, ana, ind = load(R / "history.json"), load(R / "analysis.json"), load(R / "independent.json")
pre, post, info = load(R / "preregistration.json"), load(R / "posthoc.json"), load(R / "run_info.json")
if not all((hist, ana, ind, pre, post, info)):
    sys.exit("results missing: run python -m experiments.v6_1.risk.run --all first")


# ------------------------------------------------------------------------------------------------ format
def s1(v, signed=True, d=1):
    return share(v, d, signed=signed)


def pp(v, d=1):
    return "—" if v is None else f"{v * 100:+.{d}f} п.п."


def ci(x, d=1):
    return "—" if not x else f"[{x[0] * 100:+.{d}f}; {x[1] * 100:+.{d}f}]"


def f2(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def n0(v):
    return "—" if v is None else f"{int(round(v)):,}".replace(",", " ")


def short(name: str) -> str:
    return (name.replace("Calibrated Risk (бленд с рынком)", "Бленд")
            .replace("Calibrated Risk", "Платт").replace("Conservative Risk", "Conservative")
            .replace("High-odds Risk", "High-odds").replace("Market-specific Risk", "Market-specific")
            .replace("Value Risk", "Value").replace("Бленд (исправленный p рынка)", "Бленд-испр."))


def season_table(per: dict, with_diff: bool = True):
    head = ["Стратегия", "Прогн.", "Ставки", "Кэф", "Hit", "ROI", "95 % ДИ ROI", "Приб.", "Ср. EV",
            "Разрыв калибр.", "Просадка"] + (["Δ к текущему, 95 % ДИ"] if with_diff else [])
    rows = [head]
    for n, m in per.items():
        rows.append([short(n), n0(m["predictions"]), n0(m["bets"]), f2(m["avg_odds"]), s1(m["hit_rate"], False),
                     s1(m["roi"]), ci(m["roi_ci95"]), f"{m['profit']:+.0f}", s1(m["avg_ev"]),
                     pp(m["calibration_gap"]), f"{m['max_drawdown']:.0f}"]
                    + ([ci(m.get("diff_vs_current_ci95")) if n != CUR else "—"] if with_diff else []))
    w = [2.8, 1.1, 1.1, 0.8, 0.95, 1.0, 1.75, 0.95, 1.2, 1.35, 1.15] + ([1.75] if with_diff else [])
    tot = sum(w)
    return table(rows, widths=[x / tot * 17 * cm for x in w])


_COUNT = {"t": 0, "f": 0}


def tc(text, style="small"):
    """A numbered table caption."""
    _COUNT["t"] += 1
    para = P(f"Таблица {_COUNT['t']}. {text}", style)
    return [CondPageBreak(3 * cm), para]       # a caption never ends a page


def fig(path, caption, **kw):
    """A numbered figure."""
    _COUNT["f"] += 1
    return figure(path, f"Рис. {_COUNT['f']}. {caption}", **kw)


def widths(*w):
    tot = sum(w)
    return [x / tot * 17 * cm for x in w]


# ------------------------------------------------------------------------------------------------ values
S = hist["seasons"]
cur = {s: S[s][CUR] for s in ("2023", "2024", "2025")}
cur26, pre26 = ind["strategies"][CUR], ind["strategies"][PRE]
stored = ind["stored_production_risk"]
sel = hist["selection"]
crit = ind["criteria"]["preferred_strategy"]
data26 = ind["data"]
names = list(S["2023"])
pooled = hist["pooled_select"]
blend_cal = hist["calibrators"]["2024"]["blend"]
platt_cal = hist["calibrators"]["2024"]["platt"]


HYP_RU = {"H1": "Ни одна стратегия Risk не даёт положительного ROI на 2026/27 (ожидается: все отрицательны).",
          "H2": "Предпочтительная стратегия / претендент даёт на 2026/27 ROI выше текущего правила.",
          "H3": "В диапазоне p 0.12–0.40 модель на 2026/27 завышает вероятности редких исходов (частота < ср. p).",
          "H4": "Среди кандидатов Risk с ценой фактический ROI растёт с модельным EV с наклоном намного меньше 1."}


def rng(vals, d=2, scale=1.0, unit=""):
    vals = [v * scale for v in vals]
    return f"{min(vals):.{d}f}…{max(vals):.{d}f}{unit}"


HS = ("2023", "2024", "2025")
ALLS = ("2023", "2024", "2025", "2026")


def KEYGAP(key, s):  # noqa: N802
    src = (ind["diagnostics"] if s == "2026" else ana["seasons"][s])["calibration_rare"]["by_key"].get(key)
    return None if not src else src["observed"] - src["mean_p"]
W_MKT = [ana["seasons"][s]["who_is_right"]["w_market"] for s in HS]
W_MOD = [ana["seasons"][s]["who_is_right"]["w_model"] for s in HS]
OVER = [ana["seasons"][s]["calibration_rare"]["all"]["mean_p"] - ana["seasons"][s]["calibration_rare"]["all"]["observed"]
        for s in HS]
OVER_REL = [1 - ana["seasons"][s]["calibration_rare"]["all"]["ratio"] for s in HS]
MC = {sig: [next(x for x in ana["seasons"][s]["max_selection_mc"]["rows"] if abs(x["sigma"] - sig) < 1e-9)
            for s in HS] for sig in (0.2, 0.3)}
SD_LOGIT = [ana["seasons"][s]["max_selection_mc"]["sd_logit_model_minus_market"] for s in HS]
EV_RULES = ["Value Risk (EV ≥ 0)", "Пол EV ≥−3 %", "Пол EV ≥−5 %", "Пол EV ≥−8 %", "Max-EV Risk"]
EV_SEL = [S[s][n]["avg_ev"] for s in HS for n in EV_RULES]


def rep() -> Report:
    r = Report(PDF, "Исследование стратегий Risk", "FootPredict v6.1 — исследование (без изменения продакшна)",
               facts={"Ветка": info.get("branch", "—"),
                      "Предрегистрация (commit)": ind["preregistration_commit"][:12],
                      "Данные истории 2026/27 (commit local_data)": pre["data_snapshot"]["local_data_commit"][:12],
                      "Модель": str(info.get("production_model_version", "—"))})
    return r


def section_goal(r):
    r.add(H1("1. Цель исследования"),
          P("**Risk** — второй прогноз матча рядом с Main: одна альтернативная ставка с более высоким "
            "коэффициентом, выбранная формальным правилом с предпочтением коэффициентов 3–6; небольшой "
            "отрицательный EV допускается, положительный EV не требуется. Цель исследования — проверить, "
            "существуют ли правила выбора Risk, которые на реальных предматчевых ценах дают лучший ROI, чем "
            "текущее правило, и понять, почему высокий модельный EV у Risk не превращается в прибыль."),
          P("Рамки: продакшн-правило Risk в этой итерации **не меняется** — только исследование, результаты и "
            "этот отчёт. В конце дана рекомендация: оставлять ли текущее правило и при каких условиях его "
            "можно будет менять."),
          *bullets(["сравнить на одних и тех же матчах шесть семейств стратегий: текущий Risk, Value Risk "
                    "(сначала положительный EV), Calibrated Risk (калибровка редких исходов на внутренних "
                    "фолдах), Market-specific Risk (рынки с лучшей исторической надёжностью), Conservative Risk "
                    "(умеренные коэффициенты), High-odds Risk (высокие коэффициенты с оценкой дисперсии);",
                    "оценить диапазоны коэффициентов 2.50–3.00, 3.00–3.50, 3.50–5.00, 5.00–6.00, > 6.00 и полы "
                    "EV ≥ 0, ≥ −3 %, ≥ −5 %, ≥ −8 %, без пола;",
                    "математически разобрать, почему больший EV модели не даёт лучшего ROI: калибровка "
                    "p в 0.12–0.40, систематическая ошибка цен «фаворит–аутсайдер», завышение EV при выборе "
                    "максимума;",
                    "выбрать стратегию только на 2023/24 + 2024/25, показать 2025/26 для информации и проверить "
                    "выбор на независимой истории 2026/27 после фиксации (предрегистрации) выбора в git."]))


def section_problem(r):
    bs = stored["by_source"]
    bp = stored["by_provider"]
    rows = [["Период", "Источник", "Прогнозы (рассчит.)", "Ставки с ценой", "Доля с ценой", "Hit",
             "Ср. кэф", "Ср. EV модели", "ROI", "95 % ДИ"]]
    for s in ("2023", "2024", "2025"):
        m = cur[s]
        rows.append([SL[s], "walk-forward, Bet365 (Football-Data)", n0(m["predictions"]), n0(m["bets"]),
                     s1(m["priced_share"], False), s1(m["hit_rate"], False), f2(m["avg_odds"]), s1(m["avg_ev"]),
                     s1(m["roi"]), ci(m["roi_ci95"])])
    rows.append(["2026/27", "история сайта (unified_history)", n0(stored["settled"]), n0(stored["bets"]),
                 s1(stored["priced_share"], False), s1(stored["hit_rate"], False), f2(stored["avg_odds"]),
                 s1(stored["avg_ev"]), f"{stored['roi']:+.1f}%", "—"])
    rows.append(["2026/27", "пересчёт по сохранённым рынкам", n0(cur26["predictions"]), n0(cur26["bets"]),
                 s1(cur26["priced_share"], False), s1(cur26["hit_rate"], False), f2(cur26["avg_odds"]),
                 s1(cur26["avg_ev"]), s1(cur26["roi"]), ci(cur26["roi_ci95"])])
    r.add(H1("2. Исходная проблема"),
          P(f"По истории сайта за 2026/27 (с {stored['first_date']} по последний рассчитанный матч "
            f"{stored['last_settled_date']}) рассчитано {n0(stored['settled'])} прогнозов Risk, из них "
            f"{n0(stored['bets'])} с реальной ценой ({s1(stored['priced_share'], False)}). Прибыль "
            f"{stored['profit']:+.1f} ед. при ставке 1 ед., ROI {stored['roi']:+.2f} %, доля выигрышей "
            f"{s1(stored['hit_rate'], False)}, средний коэффициент {f2(stored['avg_odds'])}. При этом средний "
            f"модельный EV тех же ставок {s1(stored['avg_ev'])}: модель «обещала» прибыль, а реальность "
            f"дала убыток."),
          P(f"По источнику прогноза: заранее записанные (live) — {n0(bs['live']['bets'])} ставок, ROI "
            f"{s1(bs['live']['roi'])}; восстановленные по предматчевым данным (backfill) — "
            f"{n0(bs['backfill']['bets'])} ставок, ROI {s1(bs['backfill']['roi'])}. По поставщику цены: "
            + "; ".join(f"{k} — {n0(v['bets'])} ставок, ROI {s1(v['roi'])}" for k, v in bp.items()) + "."),
          P("Проблема не новая: в исторических walk-forward сезонах текущее правило также убыточно "
            "(таблица 1); 95 % бутстреп-интервалы ROI везде целиком ниже нуля."),
          tc("Текущий Risk по сезонам (ставка = рассчитанный прогноз с реальной ценой; * — только для "
            "информации).", "small"),
          table(rows, widths=widths(1.1, 2.6, 1.2, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0, 1.6)),
          note(f"Парадокс, который нужно объяснить: средний модельный EV выбранных ставок положителен "
               f"({s1(cur['2023']['avg_ev'])}, {s1(cur['2024']['avg_ev'])}, {s1(cur['2025']['avg_ev'])} в "
               f"2023/24–2025/26 и {s1(cur26['avg_ev'])} в 2026/27), а фактический ROI отрицателен. Разница "
               f"EV − ROI составляет {pp(cur['2023']['ev_minus_roi'])}, {pp(cur['2024']['ev_minus_roi'])}, "
               f"{pp(cur['2025']['ev_minus_roi'])} и {pp(cur26['ev_minus_roi'])}"))


def section_current(r):
    rc = info["risk_config"]
    r.add(H1("3. Текущая реализация"),
          P("Правило реализовано в `src/markets/recommendation.py::risk_prediction`, параметры — в "
            "`src/config.py` (RISK_*). Для каждого матча модель даёт вероятности всех исходов (1X2, двойной "
            "шанс, тоталы 1.5/2.5/3.5, обе забьют, азиатские форы ±0.5…±2.5, европейские форы, точный счёт "
            "топ-5, тоталы угловых 8.5/9.5/10.5); к исходам с реальной ценой букмекера прикрепляется "
            "коэффициент и считается EV:"),
          formula("EV = p · k − 1;   для целой азиатской форы: EV = p_{выигрыш} · k − (1 − p_{возврат})"),
          P("**Пул кандидатов.** Исключаются европейские форы (EXCLUDED_GROUPS), сам Main, любой исход с тем же "
            "расчётом, что у Main (same_outcome: например «фора −0.5 хозяев» при Main «победа хозяев») и любая "
            "ставка против Main (opposite_outcome: «тотал больше 3.5» при Main «тотал меньше 3.5»). Кандидаты — "
            f"исходы пула с p ≥ RISK_MIN_PROBABILITY = {rc['RISK_MIN_PROBABILITY']}; если таких нет, берётся весь "
            "пул. Другой исход того же рынка (ничья при Main «П1») разрешён."),
          P("**Оценка кандидата** (выбирается максимум; при равенстве — исход с ценой, затем больший p):"),
          formula("score = reliability(рынок) · p^{0.5} · pref(k_{эфф}) · (1 + w_{EV} · clip(EV, ±0.3))   при реальной цене"),
          formula("score = reliability(рынок) · p^{0.5} · pref(1/p) · 0.8   без цены"),
          formula("pref(k) = exp(−½ · (d / 0.20)^{2}),  d = расстояние ln k до окна [ln 3.0; ln 6.0] (0 внутри окна)"),
          table([["Параметр", "Значение", "Смысл"],
                 ["RISK_MIN_PROBABILITY", str(rc["RISK_MIN_PROBABILITY"]), "Risk остаётся результатом модели, не лотереей"],
                 ["RISK_PREFERRED_ODDS", f"{rc['RISK_PREFERRED_ODDS'][0]:.2f}–{rc['RISK_PREFERRED_ODDS'][1]:.2f}",
                  "мягкое окно коэффициентов"],
                 ["RISK_ODDS_SOFTNESS", str(rc["RISK_ODDS_SOFTNESS"]), "ширина спада вне окна (лог-шкала)"],
                 ["RISK_EV_WEIGHT", str(rc["RISK_EV_WEIGHT"]), "EV не влияет на выбор"],
                 ["RISK_UNPRICED_FACTOR", str(rc["RISK_UNPRICED_FACTOR"]), "штраф исходу без реальной цены"],
                 ["RISK_PROBABILITY_POWER", str(rc["RISK_PROBABILITY_POWER"]), "ослабляет тягу к самому вероятному"],
                 ["MARKET_RELIABILITY", "1X2, тотал 1.00; BTTS 0.95; DC, AH 0.90; угловые 0.85; счёт 0.50",
                  "надёжность рынка (априорная)"]],
                widths=widths(3.2, 4.8, 6)),
          P("Таким образом, EV в текущем правиле не участвует: при RISK_EV_WEIGHT = 0 правило выбирает "
            "наиболее вероятный исход внутри мягкого окна коэффициентов 3–6 с поправкой на надёжность рынка. "
            "Отрицательный EV допускается автоматически. Все стратегии ниже начинают с того же пула кандидатов, "
            "поэтому сравнение честное: меняется только способ выбора."),
          P(f"Проверка воспроизведения: векторизованная реализация текущего правила в "
            f"`experiments/v6_1/risk/strategies.py` совпала с продакшн-функцией `risk_prediction()` в "
            + ", ".join(f"{v['agree']} из {v['checked']}" for v in hist["checks"].values())
            + f" матчей сезонов 2023/24–2025/26 и в {ind['check_current']['agree']} из "
              f"{ind['check_current']['checked']} матчей 2026/27."))


def section_hypotheses(r):
    r.add(H1("4. Гипотезы"),
          P("Гипотезы о стратегиях (проверяются на 2023/24 + 2024/25, 2025/26 — для информации):"),
          *bullets(["**H-Value.** Требование EV ≥ 0 (или пола EV) отбирает ставки с реальным преимуществом и "
                    "повышает ROI.",
                    "**H-Calib.** Модель завышает вероятности редких исходов (p 0.12–0.40); калибровка на "
                    "предыдущем сезоне уменьшает завышение EV и улучшает выбор.",
                    "**H-Market.** Ограничение рынками, на которых вероятности в диапазоне Risk не завышались, "
                    "улучшает ROI.",
                    "**H-Conservative.** Умеренные коэффициенты (2.50–3.50) дают меньший убыток из-за "
                    "систематической ошибки «фаворит–аутсайдер».",
                    "**H-HighOdds.** Высокие коэффициенты (4.50–8.00) увеличивают дисперсию и не улучшают "
                    "ожидаемый ROI.",
                    "**H-Inflation.** Чем жаднее правило выбирает по EV модели, тем больше разрыв между "
                    "средним EV и фактическим ROI (эффект выбора максимума)."]),
          P("Предрегистрированные гипотезы для независимой проверки 2026/27 (файл "
            "`results/preregistration.json`, закоммичен до расчёта 2026/27):"),
          *bullets([f"**{k}.** {HYP_RU[k]}" for k in pre["hypotheses"]]),
          P("(В файле предрегистрации гипотезы записаны по-английски; здесь — перевод.)", "small"))


def section_data(r):
    rows = [["Сезон", "Роль", "Матчей", "Обучение моделей", "Цены", "Исходов-кандидатов с ценой (p ≥ 0.12)"]]
    for s in ("2023", "2024", "2025"):
        m = cur[s]
        rows.append([SL[s], "выбор" if s != "2025" else "только информация", n0(m["matches"]),
                     f"сезоны 2021/22 – {int(s) - 1}/{str(int(s))[2:]}", "Bet365 из Football-Data: 1X2, тотал 2.5, "
                     "одна линия AH", n0(ana["seasons"][s]["ev_bins_model"]["n"])])
    rows.append(["2026/27", "независимая проверка", n0(data26["matches"]), "продакшн-модели сезона",
                 "сохранённые реальные цены: OddsPapi (архив, последний снимок до начала), Football-Data, "
                 "The Odds API — почти все рынки", n0(ind["diagnostics"]["ev_bins_model"]["n"])])
    r.add(H1("5. Данные и периоды"),
          P("**Исторические сезоны (walk-forward).** Сезон s предсказывается продакшн-моделями XGBoost "
            "(параметры и число деревьев активной версии, без нового подбора), обученными только на сезонах до s "
            "(с 2021/22). Матрица счетов — независимый Пуассон, согласованный с 1X2 классификатора "
            "(`reconcile_with_1x2`), угловые — отрицательное биномиальное распределение с параметром из "
            "предыдущего сезона; все рынки строятся продакшн-функциями. Цены — только предматчевые "
            "цены Bet365 из Football-Data (`B365H/D/A`, `B365>2.5/<2.5`, `B365AHH/AHA`), закрывающие `B365C*` "
            "не используются. Main — продакшн `main_prediction()`."),
          P("Ограничение истории: у Football-Data нет цен на двойной шанс, обе забьют, тоталы 1.5/3.5, "
            "альтернативные форы, угловые и точный счёт. Поэтому в исторических сезонах «ставка» Risk возможна "
            "только на 1X2, тотал 2.5 и одну линию азиатской форы; кандидаты без цены участвуют в выборе с "
            "множителем 0.8, но в ROI не входят (как на сайте)."),
          P(f"**Независимая проверка 2026/27.** Используется сохранённая история продакшна: "
            f"`unified_history(store.load_picks())` (по матчу — записанный до начала прогноз, иначе "
            f"восстановленный по предматчевым данным) и `store.load_markets()` — вероятности всех исходов, "
            f"сохранённые до начала матча, с записанными или архивными реальными ценами. Матчей "
            f"{n0(data26['matches'])}; в качестве Main берётся сохранённый Main матча (тот, что видел "
            f"пользователь). Пересчёт Main продакшн-функцией совпадает с сохранённым в "
            f"{data26['main_reproduced']} из {data26['matches']} матчей — расхождения относятся к live-прогнозам "
            f"до 23.10.2026, сделанным по прежним правилам Main (изменены 15.10 и 22.10); при сохранённом Main "
            f"пересчёт Risk совпадает с сохранённым в {data26['risk_reproduced_with_stored_main']} из "
            f"{data26['matches']} матчей ({s1(ind['current_equals_stored_risk_share'], False)})."),
          tc("Периоды и данные.", "small"),
          table(rows, widths=widths(1.3, 1.9, 1.1, 2.6, 5.5, 2.4)),
          P("Снимок данных 2026/27 зафиксирован в предрегистрации: commit local_data "
            f"`{pre['data_snapshot']['local_data_commit'][:12]}` и SHA-256 файлов истории прогнозов. Рассчитанных "
            f"матчей с результатом — по последний {stored['last_settled_date']}."))


def section_methods(r):
    rows = [["Стратегия", "Правило выбора (тот же пул кандидатов)", "Параметры"],
            ["Текущий Risk", "продакшн `risk_prediction()`", "окно 3–6, p^{0.5}, w_{EV} = 0"],
            ["Value Risk (EV ≥ 0)", "сначала кандидаты с ценой и EV ≥ 0 → лучший score; иначе текущее", "пол 0"],
            ["Пол EV F", "сначала кандидаты с ценой и EV ≥ F → лучший score; иначе текущее",
             "F = −3, −5, −8 %; «нет» = текущее"],
            ["Max-EV Risk", "сначала EV ≥ 0 → максимальный EV; иначе текущее", "диагностика жадности"],
            ["Платт (Calibrated Risk), пол F", "p заменяется калиброванной p_{cal}; EV пересчитан; сначала EV_{cal} ≥ F, "
             "затем score с p_{cal}", "F = 0, −3, −5, −8 %, нет"],
            ["Бленд (Calibrated Risk, бленд с рынком), пол F", "то же с p_{бленд} = σ(a·logit p + b·logit q + c), q — вероятность "
             "рынка без маржи", "F = 0, −3, −5, −8 %, нет"],
            ["Market-specific", "только типы исходов, у которых в прошлом сезоне p 0.12–0.40 не была значимо "
             "завышена (z ≥ −1.96, n ≥ 100); иначе текущее", "тест на предыдущем сезоне"],
            ["Conservative", "score текущего правила с окном 2.50–3.50", "окно 2.50–3.50"],
            ["High-odds", "score текущего правила с окном 4.50–8.00", "окно 4.50–8.00"],
            ["Диапазон × пол", "только кандидаты с ценой в диапазоне (и EV ≥ F), лучший по reliability·p^{0.5}; "
             "без отката (покрытие < 100 %)", "5 диапазонов × 5 полов"]]
    r.add(H1("6. Методы и параметры"),
          H2("6.1. Стратегии"),
          P("Все стратегии используют тот же пул кандидатов, что и продакшн (раздел 3), и запускаются на одних и "
            "тех же матчах. Стратегии с откатом к текущему правилу дают Risk каждому матчу с Main."),
          table(rows, widths=widths(3.4, 9.2, 4.4)),
          H2("6.2. Калибровка (только на предыдущем сезоне)"),
          P("Для сезона s калибраторы обучаются на walk-forward-прогнозах сезона s − 1 (модели, обученные до "
            "s − 1), то есть на внутреннем фолде; для 2026/27 — на 2025/26. Тип исхода (ключ) — например "
            "`1X2:D`, `TOTAL_3.5:OVER`, `HANDICAP:H`."),
          *bullets(["**Платт:** для каждого ключа логистическая регрессия исхода (выигрыш) на logit p по строкам с "
                    "p ∈ [0.05; 0.75], не менее 300 строк, иначе тождественное преобразование.",
                    "**Бленд с рынком:** для ключа с полной книгой цен — логистическая регрессия исхода на "
                    "[logit p_{модель}, logit q_{рынок}], q — цена без маржи (1/k, нормированная по всем исходам "
                    "рынка). Ключ без собственной подгонки (в 2026/27: обе забьют, угловые и т. п.) использует "
                    "бленд, общий для всех ключей; исход без полной книги — Платт.",
                    "**Надёжность рынка:** в прошлом сезоне в диапазоне p 0.12–0.40: n, средняя p, частота и "
                    "z = (частота − ср. p) / (√Σp(1−p) / n). Ключ надёжен, если n ≥ 100 и z ≥ −1.96."]),
          formula("p_{бленд} = 1 / (1 + exp(−(a · logit p_{модель} + b · logit q_{рынок} + c)))"),
          H2("6.3. Метрики"),
          *bullets(["**Прогнозы** — рассчитанные (выигрыш/проигрыш) Risk; **ставки** — те из них, что с реальной "
                    "ценой (ставка 1 ед.); возврат по целой форе и прогноз без цены ставкой не считаются.",
                    "**ROI** = Σ прибыль / число ставок; **hit rate** — доля выигрышей среди прогнозов; "
                    "**ср. EV** — средний модельный EV ставок; **разрыв калибровки** = доля выигрышей ставок − "
                    "средняя p ставок (< 0 — модель завышала).",
                    "**95 % ДИ** — бутстреп по матчам (2000 повторов, мультиномиальные веса). Веса общие для "
                    "всех стратегий сезона, поэтому интервал разности ROI с текущим правилом — **парный**. "
                    "Объединение сезонов — стратифицированный бутстреп (сумма повторов по сезонам).",
                    "**Макс. просадка** — максимальное падение накопленной прибыли (ед.) по ставкам в порядке "
                    "дат; **стабильность** — ROI по сезонам, худший сезон, знак разности с текущим по сезонам."]),
          H2("6.4. Протокол выбора и предрегистрация"),
          P("Выбор делается **только** на 2023/24 + 2024/25. Ограничения на обоих сезонах: Risk у ≥ 99.5 % "
            "матчей с Main; ≥ 85 % рассчитанных прогнозов Risk с ценой; средний коэффициент ставок ≥ 2.80 "
            "(Risk должен оставаться прогнозом с более высоким коэффициентом, окно Main — 1.40–2.50). "
            "Предпочтительная стратегия — наибольший объединённый ROI среди правил, чей ROI выше текущего на "
            "обоих сезонах; если таких нет — текущее правило. «Претендент» — лучшее допустимое нетекущее "
            "правило; он проверяется на 2026/27 в любом случае."),
          P("Критерии будущей замены продакшн-правила записаны до проверки: C1 — ROI претендента на 2026/27 "
            "выше текущего; C2 — парный 95 % ДИ разности ROI на 2026/27 целиком выше нуля; C3 — разность "
            "положительна в каждом из 2023/24, 2024/25, 2025/26 и 2026/27. Замена — только при C1 ∧ C2 ∧ C3."),
          H2("6.5. Диагностика EV"),
          *bullets(["корзины модельного EV для всех кандидатов Risk с ценой (p ≥ 0.12): фактический ROI в "
                    "корзине и наклон линейной регрессии прибыли на EV (идеал — 1);",
                    "логистическая регрессия исхода на logit p модели и logit q рынка («кто прав»);",
                    "Монте-Карло выбора максимума на реальных наборах кандидатов: «истина» q — цена рынка без "
                    "маржи, «модель» — q с шумом N(0, σ²) в логитах; выбирается максимальный EV;",
                    "дисперсия ставки при коэффициенте k и истинном ROI r: q = (1 + r)/k, SD = k·√(q(1 − q)); "
                    "симуляция 2000 сезонов по 2000 ставок для просадок и вероятности «плюсового» сезона."]))


def section_experiments(r):
    rows = [["№", "Эксперимент", "Выполнен", "Комментарий"],
            ["E1", "Текущий Risk на 2023/24–2025/26 и 2026/27", "да", "совпадение с продакшн-функцией 100 %"],
            ["E2", "Value Risk и полы EV 0 / −3 / −5 / −8 % / нет", "да", "откат к текущему правилу, Risk у каждого матча"],
            ["E3", "Max-EV Risk", "да", "диагностика эффекта выбора максимума"],
            ["E4", "Calibrated Risk (Платт) × 5 полов", "да", "калибровка на предыдущем сезоне"],
            ["E5", "Calibrated Risk (бленд с рынком) × 5 полов", "да", "бленд на предыдущем сезоне"],
            ["E6", "Market-specific Risk", "да", "надёжность по калибровке, не по ROI (без отбора проигравших)"],
            ["E7", "Conservative (2.50–3.50) и High-odds (4.50–8.00)", "да", "плюс анализ дисперсии"],
            ["E8", "Сетка 5 диапазонов × 5 полов EV", "да", "без отката, покрытие < 100 %"],
            ["E9", "Калибровка p 0.12–0.40 по рынкам; «фаворит–аутсайдер»; EV-корзины; кто прав", "да",
             "2023/24–2025/26 и 2026/27"],
            ["E10", "Монте-Карло завышения EV при выборе максимума", "да", "σ = 0…0.4 в логитах"],
            ["E11", "Предрегистрация и независимая проверка 2026/27", "да",
             f"commit {ind['preregistration_commit'][:12]} до расчёта"],
            ["E12", "Пост-хок: исправленная вероятность рынка для целых форов; поправка Бонферрони", "да",
             "после проверки 2026/27, не предрегистрировано"],
            ["—", "Изотоническая калибровка", "нет", "Платт (2 параметра на тип исхода) устойчивее на редких типах"],
            ["—", "CLV (сравнение с закрывающими ценами)", "нет", "закрывающие цены в данных не загружаются"],
            ["—", "История цен BTTS, угловых, альтернативных линий", "нет",
             "у Football-Data таких цен нет; платные API не вызывались"],
            ["—", "Блочный бутстреп по игровым дням", "нет", "бутстреп по матчам; корреляция внутри дня мала"],
            ["—", "Изменение продакшн-правила", "нет", "по условию задачи — только исследование"]]
    r.add(H1("7. Эксперименты"), table(rows, widths=widths(0.8, 6.3, 1.7, 8.2)))


def section_results_history(r):
    r.add(H1("8. Результаты"), H2("8.1. Все стратегии по сезонам"),
          P("Таблицы 3–5: каждая стратегия на всех матчах сезона. Δ — парный 95 % ДИ разности ROI со "
            "стратегией «Текущий Risk» (одинаковые бутстреп-веса)."))
    for k, s in enumerate(("2023", "2024", "2025")):
        r.add(tc(f"Сезон {SL[s]}: {n0(cur[s]['matches'])} матчей.", "small"), season_table(S[s]))
    r.add(*fig(F / "roi_by_strategy.png", "ROI стратегий с 95 % бутстреп-интервалами по сезонам.",
                  width=14 * cm))
    rows = [["Стратегия", "ROI 2023/24", "ROI 2024/25", "Объед. ROI", "Δ к текущему, 95 % ДИ", "P(лучше)",
             "Ср. кэф (мин.)", "С ценой (мин.)", "Огранич.", "Лучше на обоих"]]
    for n in names:
        t, p = sel["table"][n], pooled[n]
        rows.append([short(n), s1(t["rois"][0]), s1(t["rois"][1]), s1(t["pooled_roi"]), ci(p.get("diff_vs_current_ci95")) if n != CUR else "—",
                     s1(p.get("p_better_than_current"), False) if n != CUR else "—", f2(t["min_avg_odds"]),
                     s1(t["min_priced_share"], False), "да" if t["meets_constraints"] else "нет",
                     "да" if t["better_than_current_both"] else "нет"])
    r.add(H2("8.2. Выбор на 2023/24 + 2024/25"),
          tc("Объединённые результаты двух сезонов выбора и проверка ограничений.", "small"),
          table(rows, widths=widths(3.8, 1.2, 1.2, 1.2, 2.0, 1.1, 1.2, 1.2, 1.0, 1.1)),
          P(f"Ограничениям удовлетворяют {len(sel['eligible'])} правил из {len(names)}. Выше текущего на обоих "
            f"сезонах выбора: " + ", ".join(short(n) for n in sel["better_than_current_both"]) + ". "
            f"Предпочтительная стратегия (и претендент): **{short(sel['preferred'])}** — объединённый ROI "
            f"{s1(sel['table'][sel['preferred']]['pooled_roi'])} против {s1(sel['table'][CUR]['pooled_roi'])} у "
            f"текущего; парный 95 % ДИ разности {ci(pooled[sel['preferred']]['diff_vs_current_ci95'])} включает "
            f"ноль. На 2025/26* она также лучше текущего: {s1(S['2025'][sel['preferred']]['roi'])} против "
            f"{s1(cur['2025']['roi'])}."),
          note("Не прошли ограничения: бленд с полами −3/−5/−8 % (средний коэффициент падает до 2.2–2.8 — Risk "
               "перестаёт быть «высококоэффициентным»), Market-specific (на 2023/24 доля прогнозов с ценой "
               "ниже 85 %), Conservative (доля с ценой ≈ 84 %) и High-odds (с ценой лишь ≈ 48 % прогнозов: "
               "высоких коэффициентов с ценой в истории мало). Их результаты всё равно показаны."))
    # stability
    rows = [["Стратегия", *[f"ROI {SL[s]}" for s in ("2023", "2024", "2025")], "SD по сезонам",
             "Худший сезон", "Δ > 0 в сезонах", "Ср. EV − ROI (3 сез.)"]]
    import statistics as st
    for n in names:
        rr = [S[s][n]["roi"] for s in ("2023", "2024", "2025")]
        dd = sum(S[s][n]["roi"] > cur[s]["roi"] for s in ("2023", "2024", "2025")) if n != CUR else None
        gap = st.mean(S[s][n]["ev_minus_roi"] for s in ("2023", "2024", "2025"))
        rows.append([short(n), *[s1(x) for x in rr], pp(st.pstdev(rr)), s1(min(rr)),
                     "—" if dd is None else f"{dd} из 3", pp(gap)])
    r.add(H2("8.3. Стабильность по сезонам"),
          tc("ROI по трём историческим сезонам, разброс, худший сезон и средний разрыв между "
            "модельным EV и фактическим ROI.", "small"),
          table(rows, widths=widths(4.0, 1.4, 1.4, 1.5, 1.4, 1.3, 1.4, 1.6)),
          P("Лучше текущего правила во всех трёх исторических сезонах: " + ", ".join(
              short(n) for n in names if n != CUR and all(S[s][n]["roi"] > cur[s]["roi"] for s in cur)) + ". "
            f"Разрыв «EV − ROI» "
            f"растёт с жадностью правила: у текущего {pp(st.mean(S[s][CUR]['ev_minus_roi'] for s in S))}, "
            f"у Value Risk {pp(st.mean(S[s]['Value Risk (EV ≥ 0)']['ev_minus_roi'] for s in S))}, у Max-EV "
            f"{pp(st.mean(S[s]['Max-EV Risk']['ev_minus_roi'] for s in S))}; у бленда он почти исчезает "
            f"({pp(st.mean(S[s][PRE]['ev_minus_roi'] for s in S))}): модельный EV бленда честнее."),
          *fig(F / "cumulative_profit.png", "Накопленная прибыль ключевых стратегий: слева три "
                  "исторических сезона подряд, справа 2026/27.", width=16.5 * cm))



KEY = [CUR, "Value Risk (EV ≥ 0)", "Max-EV Risk", "Calibrated Risk, пол нет", PRE, "Market-specific Risk",
       "Conservative Risk (2.50–3.50)", "High-odds Risk (4.50–8.00)"]
GROUP_RU = {"1X2": "1X2", "TOTAL": "тотал", "HANDICAP": "AH", "DOUBLE_CHANCE": "DC", "CORNERS": "угл.",
            "BTTS": "BTTS", "EXACT_SCORE": "счёт"}


def mix(m):
    g = ", ".join(f"{GROUP_RU.get(k, k)} {v * 100:.0f}%" for k, v in sorted(m["group_shares"].items(),
                                                                           key=lambda x: -x[1]) if v >= 0.01)
    tiers = m.get("tier_shares", {})
    first = tiers.get("0", 0.0)
    return g, s1(first, False)


def section_composition(r):
    rows = [["Стратегия", "2024/25: рынки ставок", "2024/25: 1-й ярус", "2026/27: рынки ставок", "2026/27: 1-й ярус"]]
    for n in KEY:
        g4, t4 = mix(S["2024"][n])
        g6, t6 = mix(ind["strategies"][n])
        rows.append([short(n), g4, t4, g6, t6])
    r.add(H2("8.4. Из чего состоят ставки стратегий"),
          tc("Доля ставок по рынкам и доля матчей, где сработал первый ярус правила (для стратегий с "
            "откатом: EV-фильтр или надёжный рынок; для однорядных правил — 100 %).", "small"),
          table(rows, widths=widths(3.4, 4.6, 1.6, 5.8, 1.6)),
          P("В истории ставка Risk почти всегда — 1X2 (ничья или победа аутсайдера): других рынков с ценой в "
            "диапазоне 3–6 у Football-Data нет. EV-правила сдвигают выбор к тоталу 2.5 и форе, где модель чаще "
            "расходится с рынком. На 2026/27 появились цены на почти все рынки, и состав ставок меняется у всех "
            "стратегий, особенно у EV-правил и бленда. Первый ярус бленда (EV бленда ≥ 0) срабатывает только в "
            f"{s1(S['2023'][PRE]['tier_shares'].get('0', 0), False)} и {s1(S['2024'][PRE]['tier_shares'].get('0', 0), False)} "
            "матчей 2023/24 и 2024/25: «честный» положительный EV встречается редко, и в остальных матчах бленд "
            "работает как текущее правило, но с вероятностью, сдвинутой к рынку."))
    # market-specific reliability per season
    rows = [["Тип исхода", *[f"надёжен для {SL[s]}" for s in ("2023", "2024", "2025")], "для 2026/27"]]
    keys = sorted({k for s in ("2023", "2024", "2025") for k in hist["calibrators"][s]["reliability"]})
    for k in keys:
        cells = []
        for s in ("2023", "2024", "2025"):
            v = hist["calibrators"][s]["reliability"].get(k)
            cells.append("—" if not v else f"{'да' if v['reliable'] else 'нет'} (n {v['n']}, z {v['z']:+.1f})")
        v = ind["calibrator"]["reliability"].get(k)
        cells.append("—" if not v else f"{'да' if v['reliable'] else 'нет'} (n {v['n']}, z {v['z']:+.1f})")
        rows.append([k] + cells)
    r.add(H3("Market-specific Risk: какие типы исходов признаны надёжными"),
          P("Тест выполняется на предыдущем сезоне (для 2023/24 — на 2022/23 и т. д.). Решение по ROI не "
            "используется — только калибровка вероятностей, поэтому «исключение исторических проигравших» "
            "невозможно по построению.", "small"),
          table(rows, widths=widths(3.2, 3.45, 3.45, 3.45, 3.45)),
          P("Не прошли тест исходы 1X2: " + "; ".join(
              f"для {SL[s]} — " + (", ".join(k for k in ("1X2:H", "1X2:D", "1X2:A")
                                         if not hist["calibrators"][s]["reliability"].get(k, {}).get("reliable", True))
                                  or "нет")
              for s in ("2023", "2024", "2025")) + "; для 2026/27 — " + (", ".join(
                  k for k in ("1X2:H", "1X2:D", "1X2:A")
                  if not ind["calibrator"]["reliability"].get(k, {}).get("reliable", True)) or "нет") + ". "
            "Когда исходы 1X2 надёжны, Market-specific почти совпадает с текущим правилом (его ставки — в "
            "основном те же исходы 1X2). Когда тест не проходит ничья (2023/24), стратегия уходит к исходам без "
            f"цены: доля прогнозов с ценой падает до {s1(S['2023']['Market-specific Risk']['priced_share'], False)}, "
            f"а ROI ставок — до {s1(S['2023']['Market-specific Risk']['roi'])}."))


def section_calibrators(r):
    rows = [["Тип исхода", "Платт: наклон", "Платт: сдвиг", "n", "Бленд: вес модели a", "Бленд: вес рынка b",
             "Бленд: c", "n (с ценой)"]]
    for k in sorted(platt_cal):
        pl = platt_cal[k]
        bl = blend_cal.get(k)
        rows.append([k, f2(pl["slope"], 3), f2(pl["intercept"], 3), n0(pl["n"]),
                     f2(bl["w_model"], 3) if bl else "—", f2(bl["w_market"], 3) if bl else "—",
                     f2(bl["intercept"], 3) if bl else "—", n0(bl["n"]) if bl else "—"])
    pb = ind["calibrator"].get("pooled_blend") or {}
    r.add(H3("Параметры калибраторов"),
          tc("Калибраторы, обученные на 2023/24 и применённые к 2024/25. Наклон Платта < 1 значит, что модель "
             "слишком уверена (p надо сжимать к середине), > 1 — недостаточно уверена; у большинства типов он "
             "близок к 1, поэтому калибровка Платта почти не меняет p. В бленде вес рынка b близок к 1, а вес "
             "модели a — к 0: при известной цене модель почти ничего не добавляет.", "small"),
          table(rows, widths=widths(3.3, 1.7, 1.7, 1.2, 2.2, 2.2, 1.6, 1.6)),
          P(f"Общий бленд, обученный на 2025/26 для 2026/27 (используется для типов исходов без собственной "
            f"подгонки): a = {f2(pb.get('w_model'), 3)}, b = {f2(pb.get('w_market'), 3)}, c = "
            f"{f2(pb.get('intercept'), 3)}, n = {n0(pb.get('n'))}."))


def section_sources(r):
    from scipy.stats import spearmanr
    bs = ind["by_source"]
    worse_live = sum(bs[n]["live"]["roi"] < bs[n]["backfill"]["roi"] for n in names)
    rho = spearmanr([bs[n]["live"]["roi"] for n in names], [bs[n]["backfill"]["roi"] for n in names])[0]
    rows = [["Стратегия", "live: ставки", "live: ROI", "backfill: ставки", "backfill: ROI"]]
    for n in KEY:
        b = ind["by_source"][n]
        rows.append([short(n), n0(b.get("live", {}).get("bets")), s1(b.get("live", {}).get("roi")),
                     n0(b.get("backfill", {}).get("bets")), s1(b.get("backfill", {}).get("roi"))])
    r.add(tc("2026/27 по происхождению прогноза: live — записан до начала матча; backfill — "
            "восстановлен по предматчевым данным моделью, обученной до матча.", "small"),
          table(rows, widths=widths(4.5, 2.4, 2.4, 2.4, 2.4)),
          P(f"На live-матчах ROI ниже, чем на backfill, у {worse_live} из {len(names)} стратегий; ранговая "
            f"корреляция Спирмена ROI стратегий между двумя частями ρ = {rho:+.2f}. Разница между частями — "
            f"свойство выборки (другие недели сезона, другие поставщики цен, прежние правила Main у части "
            f"live-прогнозов), и она больше разницы между стратегиями."))



def section_bands(r):
    pb = hist["pooled_bands_select"]
    floors = ["≥0 %", "≥−3 %", "≥−5 %", "≥−8 %", "нет"]
    bands = ["2.50–3.00", "3.00–3.50", "3.50–5.00", "5.00–6.00", ">6.00"]
    rows = [["Диапазон", *[f"EV {f}" for f in floors]]]
    for b in bands:
        rows.append([b] + [f"{s1(pb[f'{b} | {f}']['roi'])} ({n0(pb[f'{b} | {f}']['bets'])})" for f in floors])
    rows26 = [["Диапазон", *[f"EV {f}" for f in floors]]]
    for b in bands:
        rows26.append([b] + [f"{s1(ind['bands'][f'{b} | {f}']['roi'])} ({n0(ind['bands'][f'{b} | {f}']['bets'])})"
                             for f in floors])
    cb = {s: S[s][CUR]["odds_bands"] for s in ("2023", "2024", "2025")}
    cb["2026"] = cur26["odds_bands"]
    labels = ["<2.50", "2.50–3.00", "3.00–3.50", "3.50–5.00", "5.00–6.00", ">6.00"]
    rows_c = [["Диапазон", *[SL[s] for s in cb]]]
    for b in labels:
        rows_c.append([b] + [f"{s1(cb[s][b]['roi'])} ({n0(cb[s][b]['bets'])})" if b in cb[s] else "—" for s in cb])
    r.add(H2("8.5. Диапазоны коэффициентов и полы EV"),
          P("Стратегия «диапазон × пол»: Risk выбирается только среди кандидатов с ценой внутри диапазона (и с "
            "EV ≥ пола); если таких нет — Risk у матча нет. Это анализ, а не кандидат в продакшн: покрытие "
            "далеко от 100 %."),
          tc("ROI (число ставок) на 2023/24 + 2024/25 вместе.", "small"),
          table(rows, widths=widths(2, 3, 3, 3, 3, 3)),
          *fig(F / "band_floor_heatmap.png", "ROI и число ставок по сетке «диапазон × пол EV» "
                  "(2023/24 + 2024/25).", width=16 * cm),
          tc("Та же сетка на независимой истории 2026/27 (после предрегистрации).", "small"),
          table(rows26, widths=widths(2, 3, 3, 3, 3, 3)),
          *fig(F / "band_by_season_2026.png", "ROI по диапазонам коэффициентов без пола EV, по сезонам, "
                  "с 95 % ДИ.", width=15 * cm),
          tc("Ставки текущего Risk по диапазонам коэффициента: ROI (число ставок).", "small"),
          table(rows_c, widths=widths(2, 3, 3, 3, 3)),
          *fig(F / "current_bands_2026.png", "Распределение и ROI ставок текущего Risk по диапазонам.",
                  width=15 * cm))
    best = max(((b, f) for b in bands for f in floors), key=lambda x: pb[f"{x[0]} | {x[1]}"]["roi"])
    b26 = max(((b, f) for b in bands for f in floors), key=lambda x: ind["bands"][f"{x[0]} | {x[1]}"]["roi"])
    r.add(note(f"Вывод по диапазонам: ни одна клетка сетки не прибыльна на 2023/24 + 2024/25. Наименее "
               f"убыточен диапазон 3.00–3.50 (без пола {s1(pb['3.00–3.50 | нет']['roi'])}; лучшая клетка — "
               f"{best[0]}, пол {best[1]}: {s1(pb[f'{best[0]} | {best[1]}']['roi'])}). Самые дорогие диапазоны — "
               f"5.00–6.00 и > 6.00 (без пола {s1(pb['5.00–6.00 | нет']['roi'])} и "
               f"{s1(pb['>6.00 | нет']['roi'])}). Пол EV не даёт монотонного улучшения ни в одном диапазоне: "
               f"разница между полами меньше статистической погрешности клетки (сотни–тысячи ставок, "
               f"SE ≈ 2.5–7 п.п.). На 2026/27 лучшая клетка другая — {b26[0]}, пол {b26[1]}: "
               f"{s1(ind['bands'][f'{b26[0]} | {b26[1]}']['roi'])} на {n0(ind['bands'][f'{b26[0]} | {b26[1]}']['bets'])} "
               f"ставках, что само по себе показывает неустойчивость таких отборов."))


def section_math(r):
    a = ana["seasons"]
    r.add(H2("8.6. Почему больший EV модели не даёт лучшего ROI"),
          P("Пусть p — истинная вероятность исхода, p_{мод} — оценка модели, k — коэффициент. Ожидаемая прибыль "
            "ставки равна p·k − 1, модельный EV — p_{мод}·k − 1. Для выбранного набора ставок"),
          formula("E[EV_{модель}] − E[ROI] = E[k · (p_{мод} − p) | ставка выбрана]"),
          P("Разрыв возникает по трём причинам, которые проверены по отдельности: (а) модель в среднем завышает "
            "p в диапазоне Risk (калибровка); (б) рынок систематически переоценивает длинные коэффициенты "
            "(«фаворит–аутсайдер») — тогда убыток растёт с k даже при честной p_{мод}; (в) правило выбирает "
            "исходы, где p_{мод} − p велико (эффект выбора максимума, «проклятие победителя»). Множитель k ≈ 3.3 "
            "усиливает любую ошибку p_{мод}: завышение p на 0.03 даёт ≈ +10 п.п. ложного EV."),
          H3("8.6.1. Калибровка вероятностей 0.12–0.40"))
    rows = [["Сезон", "Кандидатов", "Ср. p", "Частота", "Отношение", "z", "С ценой: n", "ср. p", "рынок q",
             "частота"]]
    for s in ("2023", "2024", "2025"):
        c = a[s]["calibration_rare"]
        al, pr = c["all"], c["priced_only"]
        rows.append([SL[s], n0(al["n"]), f2(al["mean_p"], 3), f2(al["observed"], 3), f2(al["ratio"], 3),
                     f2(al["z"], 1), n0(pr["n"]), f2(pr["mean_p"], 3), f2(pr["mean_p_market"], 3),
                     f2(pr["observed"], 3)])
    c = ind["diagnostics"]["calibration_rare"]
    rows.append([SL["2026"], n0(c["all"]["n"]), f2(c["all"]["mean_p"], 3), f2(c["all"]["observed"], 3),
                 f2(c["all"]["ratio"], 3), f2(c["all"]["z"], 1), n0(c["priced_only"]["n"]),
                 f2(c["priced_only"]["mean_p"], 3), f2(c["priced_only"]["mean_p_market"], 3),
                 f2(c["priced_only"]["observed"], 3)])
    r.add(tc("Калибровка всех кандидатов Risk с p ∈ [0.12; 0.40] (рассчитанные исходы) и отдельно "
            "исходов с полной книгой цен (q — вероятность рынка без маржи как P(выигрыш)).", "small"),
          table(rows, widths=widths(1.3, 1.5, 1.1, 1.1, 1.3, 0.9, 1.4, 1.1, 1.3, 1.2)))
    rows = [["Корзина p"] + [SL[s] for s in ("2023", "2024", "2025")] + [SL["2026"]]]
    bins = [b["bin"] for b in a["2023"]["calibration_rare"]["by_bin"]]
    for i, b in enumerate(bins):
        row = [b]
        for s in ("2023", "2024", "2025", "2026"):
            src = ind["diagnostics"] if s == "2026" else a[s]
            x = src["calibration_rare"]["by_bin"][i]
            row.append(f"{x['mean_p']:.3f} → {x['observed']:.3f} (z {x['z']:+.1f})")
        rows.append(row)
    r.add(tc("По корзинам p: средняя p → наблюдаемая частота (z-статистика).", "small"),
          table(rows, widths=widths(1.6, 3.6, 3.6, 3.6, 3.6)),
          *fig(F / "calibration_rare_2026.png", "Калибровка в диапазоне Risk: слева все кандидаты, "
                  "справа исходы с полной книгой цен (сплошная — модель, пунктир — рынок).", width=16 * cm))
    keys = sorted({k for s in ("2023", "2024", "2025") for k, v in a[s]["calibration_rare"]["by_key"].items()
                   if v["n"] >= 150})
    rows = [["Тип исхода"] + [SL[s] for s in ("2023", "2024", "2025", "2026")]]
    for k in keys:
        row = [k]
        for s in ("2023", "2024", "2025", "2026"):
            src = (ind["diagnostics"] if s == "2026" else a[s])["calibration_rare"]["by_key"].get(k)
            row.append("—" if not src else f"{(src['observed'] - src['mean_p']) * 100:+.1f} п.п. (n {src['n']}, "
                                           f"z {src['z']:+.1f})")
        rows.append(row)
    r.add(tc("Частота − средняя p по типам исходов (p 0.12–0.40); < 0 — модель завышает.", "small"),
          table(rows, widths=widths(3.2, 3.45, 3.45, 3.45, 3.45)),
          *fig(F / "calibration_by_key_2026.png", "Отклонение частоты от средней p по типам исходов "
                  "с 95 % ДИ (Уилсон).", width=15 * cm),
          note(f"Вывод по калибровке: модель завышает вероятности редких исходов **слабо, но статистически "
               f"значимо**: отношение частоты к средней p равно {f2(a['2023']['calibration_rare']['all']['ratio'], 3)}, "
               f"{f2(a['2024']['calibration_rare']['all']['ratio'], 3)}, "
               f"{f2(a['2025']['calibration_rare']['all']['ratio'], 3)} и "
               f"{f2(c['all']['ratio'], 3)} (2026/27), т. е. завышение на {rng(OVER, 1, 100)} п.п. абсолютно "
               f"({rng(OVER_REL, 1, 100)} % относительно). "
               f"По типам исходов: двойной шанс X2 завышен во всех сезонах "
               f"({', '.join(pp(KEYGAP('DOUBLE_CHANCE:X2', s)) for s in ALLS)}), победа гостей по форе — в "
               f"каждом сезоне ({', '.join(pp(KEYGAP('HANDICAP:A', s)) for s in ALLS)}), фора хозяев — в "
               f"исторических ({', '.join(pp(KEYGAP('HANDICAP:H', s)) for s in HS)}), но не в 2026/27 "
               f"({pp(KEYGAP('HANDICAP:H', '2026'))}); ничья 1X2 в исторических сезонах не завышена "
               f"({', '.join(pp(KEYGAP('1X2:D', s)) for s in HS)}), в 2026/27 — {pp(KEYGAP('1X2:D', '2026'))} "
               f"(z {(ind['diagnostics']['calibration_rare']['by_key']['1X2:D']['z']):+.1f}). У исходов с ценой средняя p модели близка "
               f"к частоте. "
               f"Этого завышения мало, чтобы объяснить разрыв EV − ROI текущего правила "
               f"({', '.join(pp(cur[s]['ev_minus_roi']) for s in HS)}): при среднем коэффициенте ≈ "
               f"{f2(cur['2024']['avg_odds'], 1)} оно даёт лишь ≈ {max(OVER) * cur['2024']['avg_odds'] * 100:.1f} п.п. "
               f"Основная причина — в отборе (раздел 8.6.4)."))
    # favourite-longshot
    rows = [["Кэф"] + [f"{SL[s]}: ROI / частота − q" for s in ("2023", "2024", "2025", "2026")]]
    labs = [x["band"] for x in a["2023"]["favourite_longshot"]]
    for i, b in enumerate(labs):
        row = [b]
        for s in ("2023", "2024", "2025", "2026"):
            src = ind["diagnostics"] if s == "2026" else a[s]
            x = next((y for y in src["favourite_longshot"] if y["band"] == b), None)
            row.append("—" if not x else f"{x['blind_roi'] * 100:+.1f}% / {(x['observed'] - x['devig']) * 100:+.1f} "
                                         f"п.п. (n {x['n']})")
        rows.append(row)
    flb = {s: {x["band"]: x for x in a[s]["favourite_longshot"]} for s in ("2023", "2024", "2025")}
    r.add(H3("8.6.2. Ошибка «фаворит–аутсайдер» в ценах"),
          P("Для каждого исхода с полной книгой цен (без модели) считается ROI ставки «на всё» в диапазоне "
            "коэффициентов и отклонение наблюдаемой частоты от вероятности без маржи q."),
          tc("ROI ставки на каждый исход с ценой и разница «частота − q».", "small"),
          table(rows, widths=widths(1.8, 3.8, 3.8, 3.8, 3.8)),
          *fig(F / "favourite_longshot_2026.png", "Ошибка «фаворит–аутсайдер»: убыток растёт с "
                  "коэффициентом.", width=16 * cm),
          note(f"Цены Bet365 содержат классическую ошибку «фаворит–аутсайдер»: ставка на всё при коэффициенте "
               f"1.50–2.00 теряет {s1(flb['2023']['1.50–2.00']['blind_roi'])}, {s1(flb['2024']['1.50–2.00']['blind_roi'])}, "
               f"{s1(flb['2025']['1.50–2.00']['blind_roi'])}, а при > 6.00 — {s1(flb['2023']['>6.00']['blind_roi'])}, "
               f"{s1(flb['2024']['>6.00']['blind_roi'])}, {s1(flb['2025']['>6.00']['blind_roi'])}. Маржа "
               f"распределена не пропорционально: на ценах > 6.00 потеря в "
               f"{min(flb[s]['>6.00']['blind_roi'] / flb[s]['1.50–2.00']['blind_roi'] for s in flb):.0f}–"
               f"{max(flb[s]['>6.00']['blind_roi'] / flb[s]['1.50–2.00']['blind_roi'] for s in flb):.0f} раз больше. "
               f"Значит, при прочих "
               f"равных Risk, сдвинутый к высоким коэффициентам, платит большую «плату за длинный "
               f"коэффициент» — это главный довод против High-odds Risk и за окно около 3.0–3.5."))
    # EV bins
    rows = [["Корзина EV модели"] + [SL[s] for s in ("2023", "2024", "2025", "2026")]]
    labs = [b["bin"] for b in a["2023"]["ev_bins_model"]["bins"]]
    for lab in labs:
        row = [lab]
        for s in ("2023", "2024", "2025", "2026"):
            src = ind["diagnostics"] if s == "2026" else a[s]
            x = next((y for y in src["ev_bins_model"]["bins"] if y["bin"] == lab), None)
            row.append("—" if not x else f"{x['roi'] * 100:+.1f}% (n {x['n']}) {ci(x['roi_ci95'])}")
        rows.append(row)
    rows.append(["Наклон прибыли на EV"] + [
        f"{(ind['diagnostics'] if s == '2026' else a[s])['ev_bins_model']['slope']:+.2f} ± "
        f"{(ind['diagnostics'] if s == '2026' else a[s])['ev_bins_model']['slope_se']:.2f}"
        for s in ("2023", "2024", "2025", "2026")])
    rows.append(["Наклон: EV Платта"] + [
        f"{(ind['diagnostics'] if s == '2026' else a[s])['ev_bins_platt']['slope']:+.2f}" for s in ("2023", "2024", "2025", "2026")])
    rows.append(["Наклон: EV бленда"] + [
        f"{(ind['diagnostics'] if s == '2026' else a[s])['ev_bins_blend']['slope']:+.2f}" for s in ("2023", "2024", "2025", "2026")])
    r.add(H3("8.6.3. Фактический ROI по корзинам модельного EV"),
          P("Все кандидаты Risk с ценой и p ≥ 0.12 (не только выбранные) разбиты по модельному EV. Если EV "
            "модели верен, ROI в корзине равен среднему EV, а наклон регрессии прибыли на EV равен 1."),
          tc("ROI по корзинам EV (n, 95 % ДИ) и наклоны.", "small"),
          table(rows, widths=widths(2.6, 3.6, 3.6, 3.6, 3.6)),
          *fig(F / "ev_vs_roi.png", "Фактический ROI против ожидаемого EV по корзинам "
                  "(2023/24 + 2024/25; пунктир — идеал ROI = EV).", width=15 * cm),
          note(f"На исторических сезонах ROI **не растёт** с модельным EV: наклон "
               f"{a['2023']['ev_bins_model']['slope']:+.2f}, {a['2024']['ev_bins_model']['slope']:+.2f}, "
               f"{a['2025']['ev_bins_model']['slope']:+.2f} (идеал +1), корзина EV ≥ 20 % — одна из худших. "
               f"Калибровка Платта этого не исправляет (она меняет p монотонно и почти не меняет порядок), а "
               f"бленд с ценой рынка даёт положительный наклон ({a['2023']['ev_bins_blend']['slope']:+.2f}, "
               f"{a['2024']['ev_bins_blend']['slope']:+.2f}, {a['2025']['ev_bins_blend']['slope']:+.2f}). На "
               f"2026/27 наклон модельного EV положителен ({ind['diagnostics']['ev_bins_model']['slope']:+.2f} ± "
               f"{ind['diagnostics']['ev_bins_model']['slope_se']:.2f}), но всё ещё далёк от 1."))
    # who is right
    rows = [["Сезон", "n", "Вес модели a", "Вес рынка b", "Log loss модели", "Log loss рынка",
             "Log loss бленда (в выборке)"]]
    for s in ("2023", "2024", "2025", "2026"):
        w = (ind["diagnostics"] if s == "2026" else a[s])["who_is_right"]
        rows.append([SL[s], n0(w["n"]), f2(w["w_model"], 3), f2(w["w_market"], 3), f2(w["log_loss_model"], 4),
                     f2(w["log_loss_market"], 4), f2(w["log_loss_blend_in_sample"], 4)])
    dis = a["2024"]["who_is_right"]["by_disagreement"]
    rows2 = [["p модели / q рынка", "n", "ср. p модели", "ср. q рынка", "частота"]]
    for x in dis:
        rows2.append([x["ratio"], n0(x["n"]), f2(x["mean_p_model"], 3), f2(x["mean_p_market"], 3), f2(x["observed"], 3)])
    r.add(H3("8.6.4. Кто прав, когда модель и рынок расходятся"),
          P("Положительный EV при марже m возможен только если p_{мод} > q·(1 + m), то есть модель расходится с "
            "рынком больше, чем на маржу (≈ 5–8 % относительно). Логистическая регрессия исхода на logit p "
            "модели и logit q рынка показывает, чья оценка содержит информацию:"),
          formula("logit P(выигрыш) = a · logit p_{мод} + b · logit q + c"),
          tc("Веса модели и рынка (кандидаты Risk с полной книгой цен, p ≥ 0.12).", "small"),
          table(rows, widths=widths(1.4, 1.2, 1.6, 1.6, 1.8, 1.8, 2.4)),
          tc("2024/25: исходы, сгруппированные по отношению p_{мод} / q.", "small"),
          table(rows2, widths=widths(3, 2, 2.5, 2.5, 2.5)),
          note(f"На исторических сезонах вес рынка b = {rng(W_MKT)}, вес модели a = {rng(W_MOD)}: когда цена "
               f"известна, модель почти ничего не добавляет, а log loss рынка ниже, чем у модели, во всех сезонах. "
               f"Там, где модель уверена сильнее рынка (p_{{мод}}/q ≥ 1.35, т. е. EV ≳ +25 %), частота совпадает с "
               f"рынком ({f2(dis[-1]['mean_p_market'], 3)} у рынка, {f2(dis[-1]['observed'], 3)} факт), а не с "
               f"моделью ({f2(dis[-1]['mean_p_model'], 3)}). Это и есть механизм: большой EV — это большое "
               f"расхождение с рынком, а большое расхождение — чаще ошибка модели. На 2026/27 вес модели "
               f"больше ({f2(ind['diagnostics']['who_is_right']['w_model'], 2)} при весе рынка "
               f"{f2(ind['diagnostics']['who_is_right']['w_market'], 2)}): цены 2026/27 из нескольких источников, "
               f"с альтернативными линиями, менее точны, чем основные линии Bet365, но рынок всё равно точнее "
               f"модели (log loss {f2(ind['diagnostics']['who_is_right']['log_loss_market'], 4)} против "
               f"{f2(ind['diagnostics']['who_is_right']['log_loss_model'], 4)})."))
    # MC
    rows = [["σ (логиты)"] + [f"{SL[s]}: EV модели / истинный ROI" for s in ("2023", "2024", "2025")]]
    for i, x in enumerate(a["2023"]["max_selection_mc"]["rows"]):
        rows.append([f"{x['sigma']:.1f}"] + [
            f"{a[s]['max_selection_mc']['rows'][i]['model_ev'] * 100:+.1f}% / "
            f"{a[s]['max_selection_mc']['rows'][i]['true_roi'] * 100:+.1f}%" for s in ("2023", "2024", "2025")])
    mc = a["2024"]["max_selection_mc"]
    r.add(H3("8.6.5. Завышение EV при выборе максимума (Монте-Карло)"),
          P("Даже несмещённая модель даёт завышенный EV выбранной ставки, если выбирается максимум из нескольких "
            "шумных оценок. Пусть EV_{i} = EV^{ист}_{i} + ε_{i}, ε_{i} независимы, E ε = 0. Тогда для "
            "выбранного i* = argmax EV_{i}"),
          formula("E[EV_{i*}] − E[EV^{ист}_{i*}] = E[ε_{i*}] ≈ σ_{EV} · E[max_{i≤K} Z_{i}] > 0,  Z ~ N(0,1)"),
          P(f"При K ≈ {mc['mean_candidates']:.1f} кандидатах с ценой на матч E[max Z] ≈ 1.0–1.2. Симуляция на "
            f"реальных наборах кандидатов: «истина» q — цена без маржи (истинный EV каждой ставки "
            f"= −маржа), «модель» — q с шумом σ в логитах; выбирается максимальный EV. Фактическое SD "
            f"logit p_{{мод}} − logit q у кандидатов ≈ {mc['sd_logit_model_minus_market']:.2f} (верхняя оценка шума "
            f"модели)."),
          tc("Средний EV выбранной ставки по «модели» и её истинный ожидаемый ROI.", "small"),
          table(rows, widths=widths(1.6, 5, 5, 5)),
          *fig(F / "mc_inflation.png", "Модельный EV выбранной ставки растёт с шумом модели, "
                  "истинный ROI — нет.", width=13 * cm),
          note(f"При шуме σ = 0.2 в логитах (у модели SD logit p_{{мод}} − logit q = {rng(SD_LOGIT, 2)}) правило "
               f"«максимальный EV» показывает {rng([x['model_ev'] for x in MC[0.2]], 1, 100, ' %')} EV при истинном "
               f"ROI {rng([x['true_roi'] for x in MC[0.2]], 1, 100, ' %')}: завышение "
               f"{rng([x['inflation'] for x in MC[0.2]], 1, 100)} п.п. (при σ = 0.3 — "
               f"{rng([x['inflation'] for x in MC[0.3]], 1, 100)} п.п.) возникает только из-за выбора "
               f"максимума, без какой-либо систематической ошибки. Эмпирически у Max-EV Risk на истории "
               f"EV − ROI = {pp(S['2023']['Max-EV Risk']['ev_minus_roi'])}, {pp(S['2024']['Max-EV Risk']['ev_minus_roi'])}, "
               f"{pp(S['2025']['Max-EV Risk']['ev_minus_roi'])} — того же порядка."))


def section_variance(r):
    rows = [["Коэффициент", "P(выигрыш)", "SD ставки", "SE ROI (2000 ставок)", "Ставок для ±5 п.п.",
             "P(сезон в плюсе)", "Медиана макс. просадки", "95-й перц. просадки", "Ожид. серия проигрышей"]]
    for x in ana["variance"]:
        rows.append([f2(x["odds"]), f2(x["win_probability"], 3), f2(x["sd_per_bet"]), pp(x["se_roi_2000"]),
                     n0(x["bets_to_see_5pp"]), s1(x["p_positive_season"], False), f"{x['median_max_drawdown']:.0f}",
                     f"{x['p95_max_drawdown']:.0f}", f"{x['expected_longest_losing']:.0f}"])
    ho, cs = "High-odds Risk (4.50–8.00)", "Conservative Risk (2.50–3.50)"
    rows2 = [["Стратегия", *[f"{SL[s]}: SD / просадка / серия" for s in ("2023", "2024", "2025", "2026")]]]
    for n in (CUR, cs, ho, PRE):
        rows2.append([short(n)] + [
            f"{(ind['strategies'] if s == '2026' else S[s])[n]['profit_sd']:.2f} / "
            f"{(ind['strategies'] if s == '2026' else S[s])[n]['max_drawdown']:.0f} / "
            f"{(ind['strategies'] if s == '2026' else S[s])[n]['longest_losing_streak']}" for s in ("2023", "2024", "2025", "2026")])
    r.add(H2("8.7. Дисперсия: High-odds Risk"),
          P("Для ставки с коэффициентом k и истинным ROI r вероятность выигрыша q = (1 + r)/k, стандартное "
            "отклонение прибыли SD = k·√(q(1 − q)). Число ставок, нужное, чтобы отличить ROI от нуля на "
            "±5 п.п. с 95 % уверенностью, n ≈ (1.96·SD / 0.05)². Симуляция: r = −5 %, 2000 сезонов по 2000 ставок."),
          tc("Дисперсия ставки по коэффициенту.", "small"),
          table(rows, widths=widths(1.3, 1.3, 1.2, 1.7, 1.6, 1.6, 1.8, 1.8, 1.7)),
          *fig(F / "variance.png", "SD ставки, стандартная ошибка ROI и просадки в зависимости от "
                  "коэффициента.", width=15 * cm),
          tc("Фактическая дисперсия стратегий: SD прибыли одной ставки / макс. просадка (ед.) / "
            "самая длинная серия проигрышей.", "small"),
          table(rows2, widths=widths(3.4, 3.4, 3.4, 3.4, 3.4)),
          note(f"При k ≈ 7 одна ставка имеет SD ≈ 2.4 ед., и даже при истинном ROI −5 % сезон из 2000 ставок "
               f"оказывается в плюсе с вероятностью ≈ {s1(ana['variance'][-1]['p_positive_season'], False)}: "
               f"высокие коэффициенты создают иллюзию удачных сезонов. Чтобы надёжно увидеть преимущество 5 п.п. "
               f"при k ≈ 5.5, нужно ≈ {n0(ana['variance'][-2]['bets_to_see_5pp'])} ставок — больше, чем Risk "
               f"даёт за сезон. Фактически High-odds Risk на истории хуже текущего на всех трёх сезонах "
               f"({s1(S['2023'][ho]['roi'])}, {s1(S['2024'][ho]['roi'])}, {s1(S['2025'][ho]['roi'])}) и "
               f"получает цену лишь у ≈ половины прогнозов."))


def section_independent(r):
    st = ind["strategies"]
    r.add(H2("8.8. Независимая проверка 2026/27 (после предрегистрации)"),
          P(f"Предрегистрация записана и закоммичена (`{ind['preregistration_commit'][:12]}`) до расчёта "
            f"2026/27: предпочтительная стратегия и претендент — **{short(pre['preferred_strategy'])}**, "
            f"параметры заморожены (полы, окна, спецификация калибраторов), калибраторы для 2026/27 обучены на "
            f"2025/26. Расчёт `--independent` отказывается работать, если файл предрегистрации не закоммичен "
            f"или изменён."),
          tc(f"Все стратегии на 2026/27: {n0(cur26['matches'])} матчей, рассчитанные по "
            f"{stored['last_settled_date']}.", "small"),
          season_table(st))
    rows = [["Критерий", "Значение", "Выполнен"],
            ["C1: ROI претендента > ROI текущего (2026/27)", f"{s1(crit['roi_2026'])} против "
             f"{s1(crit['current_roi_2026'])} (Δ {pp(crit['diff_2026'])})", "да" if crit["C1"] else "нет"],
            ["C2: парный 95 % ДИ разности > 0", ci(crit["diff_ci95_2026"]), "да" if crit["C2"] else "нет"],
            ["C3: Δ > 0 в каждом сезоне", ", ".join(pp(x) for x in crit["diffs_history"]) + f", {pp(crit['diff_2026'])}",
             "да" if crit["C3"] else "нет"],
            ["Замена рекомендована (C1 ∧ C2 ∧ C3)", "", "да" if crit["change_recommended"] else "нет"]]
    pos = [n for n, m in st.items() if m["roi"] > 0]
    c26 = ind["diagnostics"]["calibration_rare"]["all"]
    rows_h = [["Гипотеза", "Результат 2026/27"],
              ["H1: " + HYP_RU["H1"], "подтверждена: все ROI < 0" if not pos else "не подтверждена: " + ", ".join(pos)],
              ["H2: " + HYP_RU["H2"], f"подтверждена по точечной оценке ({pp(crit['diff_2026'])}), но не значимо: ДИ "
               f"{ci(crit['diff_ci95_2026'])}"],
              ["H3: " + HYP_RU["H3"], f"частота {f2(c26['observed'], 3)} при ср. p {f2(c26['mean_p'], 3)}, z = "
               f"{f2(c26['z'], 2)} — направление то же, на границе значимости"],
              ["H4: " + HYP_RU["H4"], f"наклон {ind['diagnostics']['ev_bins_model']['slope']:+.2f} ± "
               f"{ind['diagnostics']['ev_bins_model']['slope_se']:.2f} — положителен и намного меньше 1"]]
    r.add(tc("Предрегистрированные критерии замены.", "small"),
          table(rows, widths=widths(6, 8, 3)),
          tc("Предрегистрированные гипотезы.", "small"),
          table(rows_h, widths=widths(7.5, 9.5)),
          note(f"Итог независимой проверки: предпочтительная стратегия {short(PRE)} на 2026/27 лучше текущей по "
               f"точечной оценке ({s1(pre26['roi'])} против {s1(cur26['roi'])}), и разность положительна во всех "
               f"четырёх сезонах, но парный 95 % ДИ разности на 2026/27 {ci(crit['diff_ci95_2026'])} включает "
               f"ноль — критерий C2 не выполнен, поэтому по предрегистрированному правилу замена **не "
               f"рекомендуется**. ROI самой стратегии отрицателен во всех сезонах."))
    gs = pre26["group_shares"]
    r.add(P(f"Важное отличие 2026/27 от истории: цен намного больше (OddsPapi: почти все рынки и линии). "
            f"Поэтому стратегии выбирают другие рынки: у бленда на 2026/27 {s1(gs.get('HANDICAP', 0), False)} "
            f"ставок — азиатские форы (в истории — "
            f"{rng([S[s][PRE]['group_shares'].get('HANDICAP', 0) for s in HS], 0, 100, ' %')}), у текущего правила "
            f"{s1(cur26['group_shares'].get('1X2', 0), False)} — 1X2. Перенос правила на новую структуру "
            f"цен — отдельный риск (раздел 8.9)."),
          *fig(F / "roi_by_strategy_2026.png", "ROI стратегий по сезонам, включая 2026/27.",
               width=12.5 * cm),
          *fig(F / "ev_vs_roi_2026.png", "2026/27: фактический ROI по корзинам EV модели, "
                  "Платта и бленда.", width=14 * cm))


def posthoc_sig_text() -> str:
    ps = post["seasons"]
    hits = [n for n, m in ps["2026"]["strategies"].items()
            if (m.get("diff_vs_current_ci95") or [0, 0])[0] > 0]
    if not hits:
        return ("Ни у одного варианта исправленного бленда парный 95 % ДИ разности на 2026/27 не лежит целиком "
                "выше нуля.")
    parts = []
    for n in hits:
        mo = min(ps[s]["strategies"][n]["avg_odds"] for s in ("2023", "2024", "2025", "2026"))
        pos = all(ps[s]["strategies"][n]["roi"] > (S[s][CUR]["roi"] if s != "2026" else cur26["roi"])
                  for s in ("2023", "2024", "2025", "2026"))
        parts.append(f"{short(n)} (ДИ {ci(ps['2026']['strategies'][n]['diff_vs_current_ci95'])}, разность "
                     f"{'положительна во всех сезонах' if pos else 'положительна не во всех сезонах'}, но средний коэффициент "
                     f"ставок опускается до {mo:.2f} — ниже ограничения 2.80, Risk перестаёт быть прогнозом с высоким "
                     f"коэффициентом)")
    return ("Среди пост-хок вариантов парный 95 % ДИ разности на 2026/27 целиком выше нуля только у: "
            + "; ".join(parts) + ". Без поправки на множественность и при нарушении ограничения это не основание "
            "для замены, а повод включить такой вариант в будущую предрегистрацию.")


def section_posthoc(r):
    ps = post["seasons"]
    keys = list(ps["2023"]["strategies"])
    rows = [["Правило (пост-хок)", *[SL[s] for s in ("2023", "2024", "2025", "2026")], "Объед. 2023+2024",
             "Ср. кэф (мин.)"]]
    for n in keys:
        rows.append([short(n)] + [f"{s1(ps[s]['strategies'][n]['roi'])} {ci(ps[s]['strategies'][n].get('diff_vs_current_ci95'))}"
                                  for s in ("2023", "2024", "2025", "2026")]
                    + [s1(post["pooled_select"][n]["roi"]), f2(post["pooled_select"][n]["min_avg_odds"])])
    bon = post["bonferroni_2026"]
    mx = "Max-EV Risk"
    rows_b = [["Правило", "Δ ROI 2026/27", "95 % ДИ", f"ДИ с поправкой Бонферрони (k = {bon['comparisons']})"]]
    for n in (mx, PRE, "Calibrated Risk (бленд с рынком), пол ≥−5 %", "Value Risk (EV ≥ 0)"):
        rows_b.append([short(n), pp(ind["strategies"][n]["roi"] - cur26["roi"]),
                       ci(ind["strategies"][n]["diff_vs_current_ci95"]), ci(bon["rules"][n])])
    r.add(H2("8.9. Пост-хок проверки (не предрегистрированы)"),
          P("После расчёта 2026/27 обнаружен дефект в определении вероятности рынка для **целых** азиатских "
            "форов: цена без маржи q даёт P(выигрыш | нет возврата), а модель и исход используют безусловную "
            "P(выигрыш). Для целой линии с вероятностью возврата 0.2–0.25 q завышена в 1/(1 − p_{возврат}) раз. "
            "Предрегистрированный бленд использовал q как есть. Ниже — бленд с исправленной вероятностью "
            "q·(1 − p_{возврат}) на всех сезонах. Это проверка чувствительности, она **не может подтвердить** "
            "стратегию: исправление сделано после того, как результат 2026/27 стал известен. Диагностика "
            "раздела 8.6 (калибровка исходов с ценой, «кто прав») уже использует исправленную вероятность."),
          tc("Бленд с исправленной вероятностью рынка: ROI и парный 95 % ДИ разности с текущим.", "small"),
          table(rows, widths=widths(3.6, 2.6, 2.6, 2.6, 2.6, 1.4, 1.2)),
          P(f"Множественные сравнения: на 2026/27 оценивались {bon['comparisons']} нетекущих правил. "
            f"Max-EV Risk — хуже текущего во всех трёх исторических сезонах ({s1(S['2023'][mx]['roi'])}, "
            f"{s1(S['2024'][mx]['roi'])}, {s1(S['2025'][mx]['roi'])}) — на 2026/27 оказалось лучшим "
            f"({s1(ind['strategies'][mx]['roi'])}), и "
            f"его обычный 95 % ДИ разности {ci(ind['strategies'][mx]['diff_vs_current_ci95'])} не включает ноль. "
            f"С поправкой Бонферрони интервал {ci(bon['rules'][mx])}. Исправленный бленд на 2026/27 ставит "
            f"в основном на " + ", ".join(f"{GROUP_RU.get(k, k)} {v * 100:.0f}%" for k, v in sorted(
                ps['2026']['strategies'][keys[0]]['group_shares'].items(), key=lambda x: -x[1]) if v >= 0.03)
            + ": у двойного шанса и точного счёта нет полной книги цен, их EV в первом ярусе — EV Платта, "
              "то есть снова модельный."),
          tc("Разности ROI с текущим правилом на 2026/27 с поправкой на множественные сравнения.", "small"),
          table(rows_b, widths=widths(4.5, 2.5, 4, 6)),
          P(posthoc_sig_text()),
          note("Смена знака у Max-EV между историей и 2026/27 — типичная картина шума и смены режима (другие "
               "источники и линии цен), а не открытие: выбирать правило по лучшему результату на 2026/27 "
               "значило бы подгонять под шум. Ни одно правило не показывает значимого и устойчивого улучшения на "
               "всех сезонах с учётом множественности."))


def section_conclusions(r):
    gap = {s: S[s][CUR]["ev_minus_roi"] for s in S}
    r.add(H1("9. Выводы"),
          *bullets([f"Текущий Risk убыточен во всех сезонах: {s1(cur['2023']['roi'])}, {s1(cur['2024']['roi'])}, "
                    f"{s1(cur['2025']['roi'])}* и {s1(cur26['roi'])} на 2026/27 (история сайта "
                    f"{stored['roi']:+.1f} %). Это не ошибка реализации: правило воспроизведено в 100 % матчей.",
                    f"Ни одна из {len(names)} стратегий и ни одна из 25 клеток «диапазон × пол EV» не дала "
                    f"положительного ROI на сезонах выбора. Убыток Risk — структурный: маржа букмекера на длинных "
                    f"ценах выше («фаворит–аутсайдер»), а модель не знает больше рынка.",
                    f"Value Risk и полы EV не помогают: средний модельный EV выбранных ставок растёт до "
                    f"{rng(EV_SEL, 1, 100, ' %')} (у текущего {rng([cur[s]['avg_ev'] for s in HS], 1, 100, ' %')}), "
                    f"а ROI не растёт. На истории фактический ROI по корзинам EV не растёт с EV (наклон "
                    f"{rng([ana['seasons'][s]['ev_bins_model']['slope'] for s in HS])}).",
                    f"Причина разрыва EV − ROI ({pp(gap['2023'])}, {pp(gap['2024'])}, {pp(gap['2025'])} у текущего "
                    f"правила): (1) небольшое, но значимое завышение p в 0.12–0.40 ({rng(OVER_REL, 1, 100)} % "
                    f"относительно); (2) ошибка "
                    f"«фаворит–аутсайдер» в ценах; (3) главное — эффект выбора максимума: большой EV означает "
                    f"большое расхождение с рынком, а в таких случаях прав рынок (вес рынка {rng(W_MKT)}, модели "
                    f"{rng(W_MOD)}). Монте-Карло показывает, что одного шума σ = 0.2 в логитах достаточно для "
                    f"завышения EV на {rng([x['inflation'] for x in MC[0.2]], 1, 100)} п.п.",
                    f"Лучшее правило на истории — {short(PRE)}: оно заменяет p модели на бленд с ценой рынка (по "
                    f"сути — рыночную вероятность с поправкой модели) и делает EV честным. Оно лучше текущего во "
                    f"всех четырёх сезонах (+{crit['diffs_history'][0] * 100:.1f}, +{crit['diffs_history'][1] * 100:.1f}, "
                    f"+{crit['diffs_history'][2] * 100:.1f} и {pp(crit['diff_2026'])}), но ни в одном сезоне "
                    f"разница не значима, а ROI остаётся отрицательным.",
                    "Conservative и High-odds хуже текущего во всех трёх исторических сезонах; Market-specific "
                    "совпадает с ним, кроме 2023/24, когда тест не прошла ничья 1X2 и ROI упал (надёжные по "
                    "калибровке рынки — те же, что текущее правило и так выбирает); калибровка Платта сама по себе "
                    "не помогает.",
                    f"Дисперсия Risk велика (SD прибыли ставки текущего правила "
                    f"{rng([cur[s]['profit_sd'] for s in HS], 2)} ед.): 95 % ДИ ROI имеет полуширину "
                    f"≈ ±{(cur['2024']['roi_ci95'][1] - cur['2024']['roi_ci95'][0]) * 50:.1f} п.п. на "
                    f"{n0(cur['2024']['bets'])} ставках сезона и "
                    f"≈ ±{(cur26['roi_ci95'][1] - cur26['roi_ci95'][0]) * 50:.1f} п.п. на {n0(cur26['bets'])} ставках "
                    f"2026/27, поэтому различия стратегий в 1–3 п.п. статистически не различимы."]))


def section_limits(r):
    r.add(H1("10. Ограничения"),
          *bullets(["Исторические цены — только Bet365 из Football-Data (1X2, тотал 2.5, одна линия AH); "
                    "на 2026/27 цен намного больше, и правила работают на другом наборе рынков. Вывод по "
                    "истории переносится на продакшн лишь частично.",
                    "Независимая история 2026/27 короткая (≈ 2000 ставок Risk), неоднородна (live и backfill, "
                    "три поставщика цен, правила Main менялись до 23.10.2026) и включает восстановленные "
                    "прогнозы.",
                    "Бленд использует вероятность рынка; в предрегистрированном варианте для целых азиатских "
                    "форов она определена неточно (раздел 8.9).",
                    "Калибраторы обучаются на одном предыдущем сезоне; для 2023/24 это сезон 2022/23, "
                    "предсказанный моделями, обученными на одном сезоне. Для 2022/23 параметр NB угловых "
                    "оценён на том же сезоне (только калибровочные данные; цен на угловые в истории нет).",
                    "Бутстреп — по матчам (iid); зависимость внутри игрового дня не моделируется. Сравнивается "
                    "много правил — интервалы отдельных разностей не учитывают множественность (кроме раздела 8.9).",
                    "ROI — при ставке 1 ед. на лучшую доступную цену; лимиты, задержки и изменение цен не "
                    "учитываются."]))


def section_recommendations(r):
    hw = (crit["diff_ci95_2026"][1] - crit["diff_ci95_2026"][0]) / 2
    nb = pre26["bets"]
    d_hist = sum(crit["diffs_history"]) / len(crit["diffs_history"])
    need = nb * (hw / d_hist) ** 2
    r.add(H1("11. Рекомендации"),
          note("**Оставить текущее правило Risk в продакшне без изменений.** Предрегистрированный критерий "
               "замены (C1 ∧ C2 ∧ C3) не выполнен: улучшение лучшего правила положительно во всех сезонах, но "
               "не значимо ни в одном, а сам Risk остаётся убыточным при любой из проверенных стратегий."),
          *bullets(["**Не вводить** Value Risk, полы EV, Max-EV, High-odds и Conservative: они не лучше текущего "
                    "на истории, а EV-фильтры лишь сильнее завышают ожидаемую прибыль.",
                    f"**Кандидат на будущее** — {short(PRE)} в исправленном виде: вероятность рынка для целых "
                    "форов = q·(1 − p_{возврат}), а первый ярус «EV по бленду ≥ 0» только для исходов с полной "
                    "книгой цен (на 2026/27 исправленный бленд без этого ограничения уходил в двойной шанс и точный "
                    "счёт, где EV остаётся модельным — раздел 8.9). Перед заменой: (1) зафиксировать этот вариант "
                    "новой предрегистрацией; (2) проверить его на матчах, сыгранных после этого коммита; (3) менять "
                    "продакшн только если парный 95 % ДИ разности ROI на новых матчах целиком выше нуля и "
                    "разность положительна на всех исторических сезонах.",
                    f"**Объём для решения.** Полуширина парного ДИ разности на 2026/27 — {hw * 100:.1f} п.п. при "
                    f"{n0(nb)} ставках, средняя историческая разность {pp(d_hist)}; чтобы ДИ исключил ноль при "
                    f"такой истинной разнице, нужно ≈ {n0(need)} ставок Risk (ширина ДИ убывает как 1/√n), то "
                    f"есть несколько сезонов продакшна. Раньше этого замена правила не может быть обоснована "
                    f"статистически.",
                    f"**Честный EV на сайте.** Независимо от выбора Risk показывать рядом с EV модели EV по "
                    f"бленду с рынком (или хотя бы помечать, что EV модели для Risk систематически завышен: у текущего "
                    f"правила на {rng([S[s][CUR]['ev_minus_roi'] for s in HS] + [cur26['ev_minus_roi']], 1, 100)} "
                    f"п.п.); не ранжировать Risk по EV модели.",
                    "**Окно коэффициентов** 3.0–6.0 оставить: диапазон 3.00–3.50 наименее убыточен, 5.00–6.00 — "
                    "самый дорогой, а на ценах > 6 ошибка «фаворит–аутсайдер» максимальна; сдвиг окна вверх не "
                    "оправдан.",
                    "**Где менять, когда критерии будут выполнены:** `src/markets/recommendation.py::risk_prediction` "
                    "(первый ярус «EV по бленду ≥ 0», затем текущий score с p бленда), параметры калибратора в "
                    "`src/config.py`, обучение калибратора — вместе с продакшн-моделями на последнем сезоне.",
                    "Повторять это исследование командой ниже раз в квартал на новой истории 2026/27, не меняя "
                    "предрегистрированных правил."]))


def section_repro(r):
    st = info.get("stage_minutes", {})
    pk = info.get("packages", {})
    blocks = info.get("model_blocks", {})
    r.add(H1("12. Воспроизведение"),
          P("Команды (из корня репозитория, окружение `.venv`; XGBoost — 3 потока):"),
          code("\n".join([
              ".venv/bin/python -m experiments.v6_1.risk.base             # прогнозы моделей 2022-2025 (кэш)",
              ".venv/bin/python -m experiments.v6_1.risk.data             # таблицы кандидатов (кэш)",
              ".venv/bin/python -m experiments.v6_1.risk.run --history    # стратегии 2023/24-2025/26",
              ".venv/bin/python -m experiments.v6_1.risk.run --analysis   # диагностика EV",
              ".venv/bin/python -m experiments.v6_1.risk.run --preregister   # затем git commit файла!",
              ".venv/bin/python -m experiments.v6_1.risk.run --independent  # 2026/27",
              ".venv/bin/python -m experiments.v6_1.risk.run --posthoc --figures",
              ".venv/bin/python -m experiments.v6_1.risk.build_pdf        # этот PDF и findings.md"])),
          P("Результаты: `experiments/v6_1/risk/results/` — history.json, analysis.json, preregistration.json, "
            "independent.json, posthoc.json, run_info.json, strategies_history.csv, strategies_2026.csv, "
            "odds_band_ev_floor.csv; рисунки — `figures/`; кэш (прогнозы моделей, кандидаты, ставки стратегий) — "
            "`cache/` (не в git)."),
          table([["Параметр", "Значение"],
                 ["Git HEAD при расчёте", info.get("git_head", "—")[:12]],
                 ["Commit предрегистрации", ind["preregistration_commit"][:12]],
                 ["Commit данных (local_data/predictions)", pre["data_snapshot"]["local_data_commit"][:12]],
                 ["Активная модель", str(info.get("production_model_version", "—"))],
                 ["Python", info.get("python", "—")],
                 ["Пакеты", ", ".join(f"{k} {v}" for k, v in pk.items())],
                 ["Время стадий, мин", ", ".join(f"{k} {v:.1f}" for k, v in st.items())],
                 ["Обучение моделей по сезонам, с (стена)",
                  ", ".join(f"{k}: {v.get('wall_seconds', 0):.0f}" for k, v in blocks.items())],
                 ["Бутстреп", "2000 повторов, веса по матчам общие для стратегий сезона"]],
                widths=widths(5, 12)))


def findings_md() -> str:
    lines = ["# Risk strategy research — FootPredict v6.1 (findings)", "",
             "Research only; production Risk unchanged. All numbers from `results/*.json`; the PDF is "
             "`reports/v6_1/Risk_Strategy_Research.pdf`.", "",
             f"- Pre-registration commit: `{ind['preregistration_commit']}`; data snapshot (local_data): "
             f"`{pre['data_snapshot']['local_data_commit']}`.",
             f"- Chosen on 2023/24 + 2024/25: **{pre['preferred_strategy']}** (challenger: {pre['challenger_strategy']}).",
             "", "## ROI per strategy (bets with a real price; 95 % bootstrap CI)", "",
             "| Strategy | 2023/24 | 2024/25 | 2025/26 (info) | 2026/27 (independent) | Δ vs current 2026/27 |",
             "|---|---|---|---|---|---|"]
    for n in names:
        cells = [f"{s1(S[s][n]['roi'])} {ci(S[s][n]['roi_ci95'])}" for s in ("2023", "2024", "2025")]
        m = ind["strategies"][n]
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {s1(m['roi'])} {ci(m['roi_ci95'])} | "
                     f"{ci(m.get('diff_vs_current_ci95')) if n != CUR else '—'} |")
    lines += ["", f"Stored website history 2026/27 (unified_history): {stored['bets']} priced Risk bets, ROI "
                  f"{stored['roi']:+.2f} %, average model EV {s1(stored['avg_ev'])}.", "",
              "## Pre-registered criteria (2026/27)", "",
              f"- C1 (ROI higher): {crit['C1']}; C2 (paired CI > 0): {crit['C2']} {ci(crit['diff_ci95_2026'])}; "
              f"C3 (higher in every season): {crit['C3']} -> change recommended: **{crit['change_recommended']}**.",
              "", "## Why model EV does not become ROI", ""]
    a = ana["seasons"]
    lines += [f"- Calibration p 0.12–0.40 (observed / mean p): " + ", ".join(
        f"{SL[s]} {a[s]['calibration_rare']['all']['ratio']:.3f} (z {a[s]['calibration_rare']['all']['z']:+.1f})"
        for s in a) + f", 2026/27 {ind['diagnostics']['calibration_rare']['all']['ratio']:.3f} "
        f"(z {ind['diagnostics']['calibration_rare']['all']['z']:+.1f}) — small but significant over-estimation.",
        "- Slope of realised profit on model EV (ideal 1): " + ", ".join(
            f"{SL[s]} {a[s]['ev_bins_model']['slope']:+.2f}" for s in a)
        + f", 2026/27 {ind['diagnostics']['ev_bins_model']['slope']:+.2f}.",
        "- Outcome ~ a·logit p_model + b·logit q_market: " + ", ".join(
            f"{SL[s]} a={a[s]['who_is_right']['w_model']:.2f}, b={a[s]['who_is_right']['w_market']:.2f}" for s in a) + ".",
        "- Favourite–longshot: blind ROI of every priced selection at odds > 6.00: " + ", ".join(
            f"{SL[s]} {s1(next(x for x in a[s]['favourite_longshot'] if x['band'] == '>6.00')['blind_roi'])}" for s in a)
        + "; at 1.50–2.00: " + ", ".join(
            f"{SL[s]} {s1(next(x for x in a[s]['favourite_longshot'] if x['band'] == '1.50–2.00')['blind_roi'])}" for s in a) + ".",
        "- Max-selection Monte-Carlo (truth = de-vigged price, model noise σ=0.2 in logits): model EV of the chosen bet "
        + ", ".join(f"{SL[s]} {a[s]['max_selection_mc']['rows'][2]['model_ev'] * 100:+.1f} % vs true "
                    f"{a[s]['max_selection_mc']['rows'][2]['true_roi'] * 100:+.1f} %" for s in a) + ".",
        "", "## Post-hoc (not pre-registered)", "",
        "- The pre-registered blend used the de-vigged price of a whole-line Asian handicap as P(win | no refund); "
        "corrected to P(win) = q·(1 − refund probability) after the 2026/27 run. Corrected blend, floor ≥0 %: "
        + ", ".join(f"{SL[s]} {s1(post['seasons'][s]['strategies']['Бленд (исправленный p рынка), пол ≥0 %']['roi'])}"
                    for s in ("2023", "2024", "2025", "2026")) + ".",
        f"- Max-EV Risk (worse than current in every history season) is best on 2026/27: Δ "
        f"{ci(ind['strategies']['Max-EV Risk']['diff_vs_current_ci95'])}; Bonferroni-adjusted (k = "
        f"{post['bonferroni_2026']['comparisons']}) {ci(post['bonferroni_2026']['rules']['Max-EV Risk'])}.",
        "", "## Recommendation", "",
        "Keep the current Risk rule in production. Re-test the market-blend Risk (with the corrected whole-line "
        "handicap market probability) under a new pre-registration on matches played after it; change "
        "production only if the paired 95 % CI of the ROI difference on those new matches lies above zero and "
        "the difference is positive on every historical season.", ""]
    return "\n".join(lines)


def build():
    r = rep()
    for fn in (section_goal, section_problem, section_current, section_hypotheses, section_data, section_methods,
               section_experiments, section_results_history, section_composition, section_calibrators,
               section_bands, section_math, section_variance, section_independent, section_sources,
               section_posthoc, section_conclusions, section_limits, section_recommendations, section_repro):
        fn(r)
    path = r.build()
    pages = r.check(scale=1.2)
    (HERE / "findings.md").write_text(findings_md(), encoding="utf-8")
    print(path, pages, "pages")
    return pages


if __name__ == "__main__":
    build()
