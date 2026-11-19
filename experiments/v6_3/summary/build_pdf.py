"""Builds reports/v6_3/FootPredict_v6.3_Research_Summary.pdf (Prompt 9, §32) from results/ of the summary and of the
13 studies — no number is typed by hand.

    .venv/bin/python -m experiments.v6_3.summary.build_pdf
"""
from __future__ import annotations

import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.units import cm

from experiments.v6_3 import common as C
from scripts import report_kit as K
from scripts.report_kit import H1, P, bullets, figure, formula, note, table

DIRS = C.study_dirs("summary")
RES, FIG = DIRS["results"], DIRS["figures"]
V = C.V63
OUT = C.ROOT / "reports" / "v6_3" / "FootPredict_v6.3_Research_Summary.pdf"
U = ("close", "t09")
BLUE, AMBER, INK, MUTED = "#3d6fb6", "#c88a12", K.INK, K.MUTED
BLOCKS_RU = {"REL": "MARKET_RELIABILITY['TOTAL'] 1.00 → 0.95 (исследование 1)",
             "EVW": "MAIN_EV_WEIGHT 0.5 → 1.0 (исследование 1)",
             "FLOOR": "реальная цена ≥ 1.40, иначе статистический Main «не ставка» (исследование 1)",
             "CAL": "температурная калибровка по группам рынков, карты до месяца матча (исследование 2)",
             "V6": "смесь логитов 50/50 модели и одной книги (исследование 3)",
             "C2": "смесь модели с консенсусом книг, веса по прошлым месяцам (исследование 13)",
             "GOALS": "λ хозяев / гостей единой модели голов команды (исследование 10)",
             "CORNERS": "вероятности линий угловых: Poisson GLM + NB, r по лигам (исследование 11)"}
NOT_COMBINED_RU = {
    "alt_ml ens_xgb_best / best_per_task": "λ голов хозяев пересекается с GOALS; не более двух модельных частей",
    "squad PRE": "покрывает только часть матчей (лиги с данными о составах)",
    "quality mkt_both / mkt_uniform / mkt_season_0.75": "смесь 1X2 с рынком пересекается с V6 / C2",
    "recent_weighting wf:cur2 / wf:hl0": "ежемесячное переобучение всех задач; выходы совместной схемы не кэшированы",
    "meta_selection meta_lr_ev": "заменяет само правило Main; параметры правила к ней неприменимы",
    "timing": "нет B-правила (только мониторинг CLV)"}
CRITERION_RU = ("кандидат выбирается среди предрегистрированных B-кандидатов исследований и 12 комбинаций по "
                "доказательности, а не по ROI: (1) разность ROI с Baseline B > 0 в обеих вселенных; (2) наименьший "
                "p Холма на close (Холм внутри семьи кандидата); (3) при равенстве — наименьший из худших сырых p "
                "двух вселенных. Если условию (1) не удовлетворяет никто, это указывается и берётся кандидат с "
                "наименьшим p Холма.")


# ------------------------------------------------------------------------------------------------ formatting
def isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and x != x)


def pp(x, d=1) -> str:
    return f"{x * 100:+.{d}f}" if isnum(x) else "—"


def pct(x, d=1) -> str:
    return f"{x * 100:.{d}f}%" if isnum(x) else "—"


def roi(x, d=1) -> str:
    return f"{x * 100:+.{d}f}%" if isnum(x) else "—"


def ci(lo, hi, d=1) -> str:
    return f"[{lo * 100:+.{d}f}; {hi * 100:+.{d}f}]" if isnum(lo) and isnum(hi) else ""


def pv(x) -> str:
    return f"{x:.3f}" if isnum(x) else "—"


def odds(x) -> str:
    return f"{x:.2f}" if isnum(x) else "—"


def n(x) -> str:
    return f"{int(x):,}".replace(",", " ") if isnum(x) else "—"


# ------------------------------------------------------------------------------------------------ figures
def forest(rows: list[dict], path, title: str) -> None:
    """ΔROI vs Baseline B with the paired 95 % interval, close (circles) and t09 (squares)."""
    K.plot_style()
    h = max(3.2, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, h))
    y = np.arange(len(rows))[::-1]
    for off, u, col, mk, lab in ((0.14, "close", BLUE, "o", "close (основная)"), (-0.14, "t09", AMBER, "s", "t09 (проверка)")):
        d = np.array([r[u]["d"] * 100 for r in rows])
        lo = np.array([r[u]["lo"] * 100 if isnum(r[u]["lo"]) else np.nan for r in rows])
        hi = np.array([r[u]["hi"] * 100 if isnum(r[u]["hi"]) else np.nan for r in rows])
        ax.hlines(y + off, lo, hi, color=col, lw=1.6, alpha=0.85)
        ax.plot(d, y + off, mk, color=col, ms=6, label=lab, markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.6)
    ax.set_xlabel("ΔROI к Baseline B той же вселенной, п.п. (95 % ДИ, парный бутстреп)")
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.legend(loc="lower right", fontsize=7.6, frameon=False)
    ax.grid(axis="y", visible=False)
    K.save_chart(fig, path)


def cumulative(path) -> None:
    K.plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=False)
    comb = C.read_json(RES / "combinations.json")
    best = comb["best_by_evidence"]
    for ax, u in zip(axes, U):
        for f, col, lab in ((RES / f"per_match_baseline_b_{u}.csv", MUTED, "Baseline B"),
                            (RES / f"per_match_{best}_{u}.csv", BLUE, f"{best} ({comb['details'][best]['parts'] and '+'.join(comb['details'][best]['parts'])})")):
            t = pd.read_csv(f)
            t = t.loc[t["status"].isin(["WON", "LOST"]) & t["odds"].notna()].sort_values(["date", "match_id"])
            ax.plot(np.arange(1, len(t) + 1), t["profit"].cumsum().to_numpy(), color=col, lw=2, label=lab)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"Вселенная {u}", fontsize=9.5, loc="left")
        ax.set_xlabel("ставки по дате")
        ax.set_ylabel("накопленная прибыль, ед. ставки")
        ax.legend(fontsize=7.6, frameon=False, loc="lower left")
    K.save_chart(fig, path)


# ------------------------------------------------------------------------------------------------ sections
def git_lines() -> list[tuple[str, str]]:
    try:
        out = subprocess.check_output(["git", "log", "--oneline", "origin/main..HEAD"], cwd=C.ROOT, text=True)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        sha, subj = line.split(" ", 1)
        rows.append((sha, subj))
    return rows


GROUPS = [
    (r"study$|starting audit|season replay|heavy v6\.3 research|research thread|walk-forward cut-offs|"
     r"research isolation", None),
    (r"^Document", "Документация"),
    (r"data checks|after the daily update|withdrawn before kick-off|waiting reasons", "Проверки данных после ежедневного обновления"),
    (r"factor values|table positions|Why card", "Читаемые значения факторов «Почему такой прогноз»"),
    (r"older predictions|reproduce them", "Проверка воспроизводимости старых объяснений"),
    (r"forward shadows", "Форвард-проверка: замороженные правила Main записываются ежедневно как теневые"),
    (r"theme|flags|Main result|Analytics|Teams|Model Analysis|screenshots|browser tests|contrast audit|"
     r"indicator variants", "Дизайн и интерфейс v6.3 (и браузерные тесты)"),
    (r"Document", "Документация"),
]


def implemented() -> list[list[str]]:
    rows = [["Группа", "Коммит", "Изменение (сообщение коммита)"]]
    for sha, subj in git_lines():
        grp = "Прочее"
        for pat, name in GROUPS:
            if re.search(pat, subj):
                grp = name
                break
        if grp is None:
            continue
        rows.append([grp, f"`{sha}`", subj])
    rows[1:] = sorted(rows[1:], key=lambda r: r[0])
    return rows


def rejected() -> list[list[str]]:
    j = lambda p: C.read_json(V / p)  # noqa: E731
    out = [["Исследование", "Отклонено (C)"]]
    ms = j("main_selection/results/main_selection.json")
    fam = [f for f in ms["families"] if f not in ("floor", "ev_weight", "reliability")]
    out.append(["1. Main Selection", "семейства: " + ", ".join(fam) + "; пороги 1.30 / 1.50 / 1.60; комбинация "
                "выборов 2025/26 (порог + EV + TOTAL)"])
    cal = j("calibration/results/preregistration.json")["decisions"]
    out.append(["2. Калибровка", ", ".join(k for k, v in cal.items() if v["decision"] == "C")])
    ma = j("market_aware/results/market_aware.json")["variants_ru"]
    out.append(["3. Market-Aware", "; ".join(v for k, v in ma.items() if k not in ("baseline_b", "blend_5050"))])
    mp = j("market_profitability/results/market_profitability.json")
    out.append(["4. Доходность рынков", "семейства: " + ", ".join(f for f in mp["families"] if f != "boost")
                + "; усиление других групп"])
    out.append(["5. Timing & CLV", "перенос формирования Main ближе к началу; пересчёт Main по цене закрытия"])
    q = j("quality/results/mitigations_2026.json")["decisions"]
    out.append(["6. Качество данных", ", ".join(k for k, v in q.items() if v == "C")])
    rw = j("recent_weighting/results/results.json")
    sch = sorted({k.split("|")[0] for k in rw["decisions"]})
    rej = [rw["names"].get(s, s) for s in sch if all(rw["decisions"].get(f"{s}|{t}") == "C"
                                                     for t in ("result", "goals", "corners"))]
    out.append(["7. Вес текущего сезона", "; ".join(rej)])
    am = j("alt_ml/results/main_2026.json")["decisions"]
    out.append(["8. Альтернативные ML", ", ".join(k for k, v in am.items() if v == "C")])
    me = j("meta_selection/results/meta_selection.json")
    out.append(["9. Метамодель", ", ".join(k for k in me["meta"] + me["simple"] if k != "meta_lr_ev")])
    g = j("goals/results/results.json")
    out.append(["10. Голы", "; ".join(g["names"][k] for k, v in g["decisions"].items() if v == "C")])
    cn = j("corners/results/results.json")
    out.append(["11. Угловые", "; ".join(cn["variants"][k] for k, v in cn["decisions"].items() if v["decision"] == "C")])
    sq = j("squad/results/decisions.json")
    out.append(["12. Сила состава", ", ".join(k for k, v in sq.items() if v["decision"] == "C")])
    co = j("consensus/results/consensus.json")["variants_ru"]
    out.append(["13. Консенсус", "; ".join(v for k, v in co.items() if k not in ("cons_blend_wf", "cons_market_only"))])
    return out


def build() -> None:
    ov = C.read_json(RES / "overview.json")
    comb = C.read_json(RES / "combinations.json")
    plan = C.read_json(RES / "combinations_plan.json")
    key = C.read_json(RES / "production_vs_best.json")
    bcand = pd.read_csv(RES / "b_candidates.csv")
    ctab = pd.DataFrame(comb["table"])
    studies = ov["studies"]
    st = {s["key"]: s for s in studies}
    bb, ba = comb["baseline_b"], comb["baseline_a"]
    alt = C.read_json(V / "alt_ml/results/metrics_2026.json")
    ma = C.read_json(V / "market_aware/results/market_aware.json")
    cal = C.read_json(V / "calibration/results/calibration.json")
    tim = C.read_json(V / "timing/results/timing.json")
    rw = C.read_json(V / "recent_weighting/results/results.json")
    rw_roi = pd.read_csv(V / "recent_weighting/results/main_roi.csv").set_index(["variant", "universe"])
    sq2 = C.read_json(V / "squad/results/e2_season_2026.json")
    sq1 = C.read_json(V / "squad/results/e1_earlier_seasons.json")
    sqd = C.read_json(V / "squad/results/decisions.json")
    sqs = C.read_json(V / "squad/results/sources.json")
    pm = key["production_metrics"]
    gap_mkt = ma["demargin"]["close"]["1X2"]["model_minus_power"]
    wc = cal["b_gap_decomposition"]["stated"]
    arch = tim["archive"]["books"]["bet365"]
    retrain = rw_roi.loc[("static:prod", "close")]
    best = key["best"]
    n_tests_total = sum(1 for _ in bcand.itertuples()) * 2
    mt = pd.read_parquet(C.SHARED_CACHE / "matches.parquet", columns=["match_id", "league", "played"])
    n_season, n_played, n_leagues = len(mt), int(mt["played"].astype(bool).sum()), int(mt["league"].nunique())
    priced_close = bb["close"]["summary"]["priced_share"]
    unpriced_t09 = 1 - bb["t09"]["summary"]["priced_share"]
    sq_cov = pd.read_csv(V / "squad/results/coverage_by_league_season.csv")
    sq_leagues = int(sq_cov.loc[sq_cov["pre_both"] > 0, "league"].nunique())
    width_close = float((bcand["d_hi_close"] - bcand["d_lo_close"]).median())
    abs_close = float(bcand["d_roi_close"].abs().median())
    ms_max = max(r["roi_diff"] for r in C.read_json(V / "main_selection/results/main_selection.json")["candidates"]
                 if r["universe"] == "close")
    gb = comb["details"]
    def gshare(groups, g):
        tot = sum(groups.values())
        return groups.get(g, 0) / tot if tot else None
    b_groups = pd.read_parquet(C.SHARED_CACHE / "baseline_b_close.parquet")
    b_groups = b_groups.loc[b_groups["bet"] == 1, "group"].value_counts().to_dict()

    # ---------------- figures
    rows = []
    for r in bcand.itertuples():
        rows.append({"label": f"{r.candidate} · {r.study.split('.')[0]}", "close": {"d": r.d_roi_close,
                     "lo": r.d_lo_close, "hi": r.d_hi_close}, "t09": {"d": r.d_roi_t09, "lo": r.d_lo_t09,
                                                                        "hi": r.d_hi_t09}})
    forest(rows, FIG / "b_candidates_forest.png", "B-кандидаты 13 исследований: ΔROI Main к Baseline B")
    crow = []
    for vid, g in ctab.groupby("id", sort=False):
        g = g.set_index("universe")
        crow.append({"label": f"{vid} · {g.loc['close', 'parts']}",
                     "close": {"d": g.loc["close", "d_roi"], "lo": g.loc["close", "d_lo"], "hi": g.loc["close", "d_hi"]},
                     "t09": {"d": g.loc["t09", "d_roi"], "lo": g.loc["t09", "d_lo"], "hi": g.loc["t09", "d_hi"]}})
    forest(crow, FIG / "combinations_forest.png", "Части (S_) и комбинации (K): ΔROI Main к Baseline B")
    cumulative(FIG / "best_combination_cumulative.png")

    rep = K.Report(OUT, "Сводный отчёт исследований v6.3", "FootPredict v6.3 — Research Summary",
                   facts={"Сезон проверки": "2026/27",
                          "Исследований": str(ov["n_studies"]),
                          "Источник чисел": "experiments/v6_3/*/results, experiments/v6_3/summary/results"})

    # 1
    rep.add(H1("1. Назначение и постановка задачи"),
            P("Версия v6.3 провела 13 исследований с одной целью: выяснить, дают ли лучшие вероятности, модели, "
              "данные и правило выбора более точный и более доходный Main — один прогноз на матч, по которому сайт "
              "считает ROI. Каждое исследование пересчитало весь сезон 2026/27 на своих данных, не меняя историю "
              "сайта. Этот документ сводит результаты: обзор 13 исследований, сравнение действующей системы с "
              "лучшими кандидатами, проверку комбинаций (этап 6), улучшения моделей 1X2, голов и угловых, итоги "
              "Squad Strength и Recent Season Weighting, статистическую надёжность, внедрённое и отклонённое, "
              "нерешённые проблемы и выводы для дипломного проекта."),
            note(f"Главный итог: ни одно исследование и ни одна комбинация не получили решения A. Действующая "
                 f"система (Baseline B: ROI close {roi(bb['close']['summary']['roi'])}, t09 "
                 f"{roi(bb['t09']['summary']['roi'])}) остаётся в production; {len(bcand)} B-кандидатов заморожены "
                 f"для проверки на будущих матчах."))
    # 2
    rep.add(H1("2. Исходные данные и методика"),
            P(f"Данные: сезон 2026/27 — {n(n_season)} матчей {n_leagues} лиги, из них сыграно {n(n_played)}; выбор параметров — на сезонах "
              "2021/22–2025/26 (в основном бэктест 2025/26) или на месяцах 2026/27 до прогнозируемого месяца "
              "(расширяющееся окно). Модели продакшна `prod_20261012` обучены на матчах до начала сезона."),
            *bullets([
                f"**Две вселенные решений.** close — решение перед началом матча по последней котировке (основная, "
                f"цена у {pct(priced_close, 2)} Main Baseline B); t09 — решение в момент ежедневного запуска "
                f"T = min(09:00 UK, начало − 30 мин) только по ценам, известным к T ({pct(unpriced_t09)} Main Baseline "
                f"B без цены — это не ставки).",
                "**Baseline A** — история сайта (live + backfill) так, как её считает Prediction History; "
                "**Baseline B** — модели и правило продакшна, переигранные на каждом матче в момент решения. Все "
                "парные сравнения — с Baseline B той же вселенной.",
                "**ROI** как в продакшне: ставка 1, WON/LOST с реальной ценой — ставка, PUSH/VOID возвращают "
                "ставку, прогноз без цены — не ставка. EV = p · odds − (1 − P(возврата)).",
                "**Неопределённость:** бутстреп по матчам (2 000 повторов) для ROI, парный бутстреп для разностей, "
                "поправка Холма по вариантам каждого исследования; число вариантов указано.",
                "**2026/27 не является независимой отложенной выборкой** — сезон уже изучался v6.1/v6.2; результат "
                "на нём — хронологическая проверка, а доказательством служит только форвард-проверка "
                "(предрегистрация)."]),
            formula("ROI = Σ прибыль / число ставок;   ΔROI = ROI(кандидат) − ROI(Baseline B) на тех же матчах"),
            P(f"Критерий выбора «лучшего проверенного кандидата» зафиксирован заранее в "
              f"`results/combinations_plan.json` ({plan['frozen_at']}), до расчёта комбинаций: {CRITERION_RU}"))

    # 3 overview
    rows = [["Исследование", "Вопрос", "Вариантов", "Матчи", "Решение"]]
    for s in studies:
        rows.append([s["title"], s["question"], s["variants"], n(s["matches"]), s["decision"]])
    rep.add(H1("3. Обзор 13 исследований"),
            P("Таблица 1 — что проверялось. Матчей — число сыгранных матчей 2026/27 в парном сравнении с Baseline B "
              "(Squad Strength — только матчи 11 лиг с данными о составах)."),
            table(rows, widths=[3.2 * cm, 4.9 * cm, 3.7 * cm, 1.6 * cm, 3.6 * cm]),
            P("Таблица 2 — лучший B-кандидат каждого исследования по критерию доказательности (не по "
              "наибольшему ROI). ΔROI — в процентных пунктах к Baseline B той же вселенной, ДИ 95 %; p Х — p после "
              "поправки Холма в семье исследования.", "small"))
    rows = [["Исследование", "Кандидат", "ROI close", "ΔROI close [ДИ]", "p Х", "ROI t09", "ΔROI t09 [ДИ]", "p Х",
             "Ставок close / t09"]]
    for s in studies:
        b = next((c for c in s["b"] if c["name"] == s["best"]), None)
        if b is None:
            rows.append([s["title"], "нет B-правила", "—", "—", "—", "—", "—", "—", "—"])
            continue
        c, t = b["close"], b["t09"]
        rows.append([s["title"], f"{b['name']}: {b['label']}", roi(c["roi"]),
                     f"{pp(c['d_roi'])} {ci(*(c['d_ci'] or [None, None]))}", pv(c["p_holm"]), roi(t["roi"]),
                     f"{pp(t['d_roi'])} {ci(*(t['d_ci'] or [None, None]))}", pv(t["p_holm"]),
                     f"{n(c['bets'])} / {n(t['bets'])}"])
    rep.add(table(rows, widths=[2.7 * cm, 2.7 * cm, 1.2 * cm, 2.4 * cm, 1.15 * cm, 1.2 * cm, 2.4 * cm, 1.15 * cm,
                                2.1 * cm]))
    rows = [["Исследование", "Лучший кандидат", "Hit close / t09", "Ср. коэф. close / t09", "Эффект на метрики модели"]]
    for s in studies:
        b = next((c for c in s["b"] if c["name"] == s["best"]), None)
        if b is None:
            rows.append([s["title"], "—", "—", "—", s.get("note", "")])
            continue
        rows.append([s["title"], b["name"], f"{pct(b['close']['hit_rate'])} / {pct(b['t09']['hit_rate'])}",
                     f"{odds(b['close']['avg_odds'])} / {odds(b['t09']['avg_odds'])}", b["model_effect"]])
    rep.add(P("Таблица 3 — тот же лучший кандидат: частота попаданий, средний коэффициент и влияние на метрики "
              "моделей (как его измерило исследование).", "small"),
            table(rows, widths=[3.0 * cm, 2.6 * cm, 2.3 * cm, 2.3 * cm, 6.8 * cm]))

    # 4 Main ROI comparison
    rows = [["Кандидат (исследование)", "ΔROI close [ДИ]", "p", "p Х", "ΔROI t09 [ДИ]", "p", "p Х", "Знак в обеих"]]
    for r in bcand.itertuples():
        both = "да" if r.d_roi_close > 0 and r.d_roi_t09 > 0 else "нет"
        rows.append([f"{r.candidate} ({r.study.split('.')[0]})", f"{pp(r.d_roi_close)} {ci(r.d_lo_close, r.d_hi_close)}",
                     pv(r.p_close), pv(r.p_holm_close), f"{pp(r.d_roi_t09)} {ci(r.d_lo_t09, r.d_hi_t09)}",
                     pv(r.p_t09), pv(r.p_holm_t09), both])
    n_pos = int(((bcand["d_roi_close"] > 0) & (bcand["d_roi_t09"] > 0)).sum())
    n_sig = int((bcand["p_holm_close"] < 0.05).sum() + (bcand["p_holm_t09"] < 0.05).sum())
    rep.add(H1("4. Сравнение Main ROI: все B-кандидаты"),
            P(f"Baseline B: close ROI {roi(bb['close']['summary']['roi'])} {ci(*bb['close']['roi_ci']['ci95'])} на "
              f"{n(bb['close']['summary']['bets'])} ставках, t09 {roi(bb['t09']['summary']['roi'])} "
              f"{ci(*bb['t09']['roi_ci']['ci95'])} на {n(bb['t09']['summary']['bets'])}; Baseline A "
              f"{roi(ba['summary']['roi'])} {ci(*ba['roi_ci']['ci95'])} на {n(ba['summary']['bets'])}."),
            P(f"Всего B-кандидатов: {len(bcand)}; положительны в обеих вселенных: {n_pos}; значимы после поправки "
              f"Холма хотя бы в одной вселенной: {n_sig} (из {n_tests_total} проверок)."),
            P("Таблица 4 — все B-кандидаты; p — сырой парный p, p Х — после поправки Холма в семье исследования.",
              "small"),
            table(rows, widths=[3.9 * cm, 2.9 * cm, 1.2 * cm, 1.2 * cm, 2.9 * cm, 1.2 * cm, 1.2 * cm, 1.5 * cm]),
            *figure(FIG / "b_candidates_forest.png",
                    "Рисунок 1 — ΔROI Main каждого B-кандидата к Baseline B; круг — close, квадрат — t09, линия — "
                    "95 % ДИ парного бутстрепа. Интервалы почти всех кандидатов пересекают ноль."))

    # 5 key table
    bc, b9 = bb["close"]["summary"], bb["t09"]["summary"]
    c, t = best["close"], best["t09"]
    cm_ = key["candidate_metrics"]

    def mm(k, d=4):
        return f"{pm[k]:.{d}f}" if isnum(pm.get(k)) else "—"

    def cmv(k, d=4):
        return f"{cm_[k]:.{d}f}" if isnum(cm_.get(k)) else "—"
    def gdev():
        d = cm_.get("d_goals_deviance")
        if d:
            return "без изменений" if abs(d["diff"]) < 1e-12 else f"Δ {d['diff']:+.4f}"
        return cmv("goals_poisson_deviance")
    rows = [["Показатель", "Baseline A (сайт)", "Продакшн B close", f"{best['name']} close", "Продакшн B t09",
             f"{best['name']} t09"],
            ["ROI [95 % ДИ]", f"{roi(ba['summary']['roi'])} {ci(*ba['roi_ci']['ci95'])}",
             f"{roi(bc['roi'])} {ci(*bb['close']['roi_ci']['ci95'])}", f"{roi(c['roi'])} {ci(c['roi_lo'], c['roi_hi'])}",
             f"{roi(b9['roi'])} {ci(*bb['t09']['roi_ci']['ci95'])}", f"{roi(t['roi'])} {ci(t['roi_lo'], t['roi_hi'])}"],
            ["Прибыль, ед.", f"{ba['summary']['profit']:+.1f}", f"{bc['profit']:+.1f}", f"{c['profit']:+.1f}",
             f"{b9['profit']:+.1f}", f"{t['profit']:+.1f}"],
            ["Hit rate", pct(ba["summary"]["hit_rate"]), pct(bc["hit_rate"]), pct(c["hit_rate"]), pct(b9["hit_rate"]),
             pct(t["hit_rate"])],
            ["Средний коэффициент", odds(ba["summary"]["avg_odds"]), odds(bc["avg_odds"]), odds(c["avg_odds"]),
             odds(b9["avg_odds"]), odds(t["avg_odds"])],
            ["Ставок", n(ba["summary"]["bets"]), n(bc["bets"]), n(c["bets"]), n(b9["bets"]), n(t["bets"])],
            ["ΔROI к B [ДИ], п.п.", "—", "0", f"{pp(c['d_roi'])} {ci(c['d_lo'], c['d_hi'])}", "0",
             f"{pp(t['d_roi'])} {ci(t['d_lo'], t['d_hi'])}"],
            ["p / p Холма", "—", "—", f"{pv(c['p'])} / {pv(c['p_holm'])}", "—", f"{pv(t['p'])} / {pv(t['p_holm'])}"],
            ["1X2 log loss", "—", mm("1x2_log_loss"), cmv("1x2_log_loss"), mm("1x2_log_loss"), cmv("1x2_log_loss")],
            ["Голы: девиация Пуассона", "—", mm("goals_poisson_deviance"),
             gdev(), mm("goals_poisson_deviance"), gdev()],
            ["Угловые: MAE / R²", "—", f"{mm('corners_mae', 3)} / {mm('corners_r2', 3)}",
             f"{cmv('corners_mae', 3)} / {cmv('corners_r2', 3)}", f"{mm('corners_mae', 3)} / {mm('corners_r2', 3)}",
             f"{cmv('corners_mae', 3)} / {cmv('corners_r2', 3)}"]]
    ll = cm_.get("d_1x2_log_loss")
    rep.add(H1("5. Current Production vs Best Validated Experimental Candidate"),
            P(f"Кандидат выбран по заранее зафиксированному критерию из {key['n_pool']} кандидатов (B-кандидаты "
              f"исследований и 12 комбинаций); положительны в обеих вселенных {key['n_positive_both']}. Лучшая "
              f"доказательность — **{best['name']}** ({best['source']}: {best['label']})."),
            table(rows, widths=[3.2 * cm, 2.6 * cm, 2.8 * cm, 2.9 * cm, 2.7 * cm, 2.8 * cm]),
            P("Таблица 5 — ключевая таблица. Метрики моделей — 2026/27, close (модели не зависят от вселенной); "
              f"{cm_['source']}. Для Baseline A метрики моделей не приводятся: история сайта составлена прогнозами "
              "разных версий моделей.", "small"))
    if ll:
        rep.add(P(f"Влияние кандидата на 1X2 log loss: {ll['diff'] * 1e3:+.2f}·10⁻³ [{ll['ci95'][0] * 1e3:+.2f}; "
                  f"{ll['ci95'][1] * 1e3:+.2f}] — значимо, но это около {abs(ll['diff']) / gap_mkt['diff'] * 100:.0f} % "
                  f"отставания модели от рынка ({gap_mkt['diff'] * 1e3:+.1f}·10⁻³)."))
    rk = [["#", "Кандидат", "Источник", "ΔROI close", "p Х close", "ΔROI t09", "p close / t09", "+ в обеих"]]
    for i, r in enumerate(key["ranking"][:8], 1):
        rk.append([str(i), f"{r['name']}: {r['label']}", r["source"], pp(r["d_close"]), pv(r["p_holm_close"]),
                   pp(r["d_t09"]), f"{pv(r['p_close'])} / {pv(r['p_t09'])}", "да" if r["positive_both"] else "нет"])
    hr = [r["p_holm_close"] for r in key["ranking"][:3]]
    rep.add(P(f"Таблица 6 — первые восемь кандидатов по критерию. Разница между первыми местами мала: p Холма "
              f"{pv(min(hr))}–{pv(max(hr))} у первых трёх — ни один кандидат не доказал преимущества на 2026/27.",
              "small"),
            table(rk, widths=[0.6 * cm, 4.6 * cm, 3.2 * cm, 1.6 * cm, 1.4 * cm, 1.6 * cm, 2.2 * cm, 1.4 * cm]),
            note(f"Честный вывод по таблице 5: лучший кандидат повышает ROI close на {pp(c['d_roi'])} п.п. "
                 f"{ci(c['d_lo'], c['d_hi'])}, но на t09 эффект {pp(t['d_roi'])} п.п. {ci(t['d_lo'], t['d_hi'])} и "
                 f"после поправки Холма (p {pv(c['p_holm'])}) не значим; ROI остаётся отрицательным "
                 f"({roi(c['roi'])}). Замена production не обоснована — кандидат идёт на форвард-проверку."))

    # 6 best Main strategy
    br, bk = key["best_rule_candidate"], key["best_combination"]
    rep.add(H1("6. Лучшая полученная стратегия Main"),
            P(f"Среди изменений самого правила Main (без новых моделей) лучшая доказательность у "
              f"**{br['name']}** ({br['label']}): close {pp(br['close']['d_roi'])} п.п. "
              f"{ci(br['close']['d_lo'], br['close']['d_hi'])}, t09 {pp(br['t09']['d_roi'])} п.п. "
              f"{ci(br['t09']['d_lo'], br['t09']['d_hi'])}, p Холма close {pv(br['close']['p_holm'])}. Среди комбинаций — "
              f"**{bk['name']}** ({bk['label']}): close {pp(bk['close']['d_roi'])} п.п. "
              f"{ci(bk['close']['d_lo'], bk['close']['d_hi'])} (ROI {roi(bk['close']['roi'])}, hit "
              f"{pct(bk['close']['hit_rate'])}, ср. коэф. {odds(bk['close']['avg_odds'])}, {n(bk['close']['bets'])} ставок), "
              f"t09 {pp(bk['t09']['d_roi'])} п.п. {ci(bk['t09']['d_lo'], bk['t09']['d_hi'])}; p Холма close "
              f"{pv(bk['close']['p_holm'])}."),
            P(f"Комбинация {bk['name']} сдвигает ставки Main от тоталов голов к азиатским форам: доля тоталов среди "
              f"ставок close {pct(gshare(b_groups, 'TOTAL'))} у Baseline B против "
              f"{pct(gshare(gb[bk['name']]['close']['groups_bets'], 'TOTAL'))}, доля фор "
              f"{pct(gshare(b_groups, 'HANDICAP'))} против {pct(gshare(gb[bk['name']]['close']['groups_bets'], 'HANDICAP'))}; "
              f"средний коэффициент {odds(bb['close']['summary']['avg_odds'])} → {odds(bk['close']['avg_odds'])}. "
              "Сдвиг виден в обеих вселенных, но интервалы t09 широкие, а после поправки Холма ни одна стратегия "
              "не значима. Это лучшая найденная гипотеза для форвард-проверки, а не доказанное улучшение; правила "
              "TOTAL 0.95, HANDICAP 1.035, EV 1.0 и порог 1.40 с v6.3 ежедневно записываются как теневые варианты."),
            *figure(FIG / "best_combination_cumulative.png",
                    f"Рисунок 2 — накопленная прибыль Baseline B и комбинации {comb['best_by_evidence']} по ставкам в "
                    f"порядке дат (ставка 1)."))

    # 7 combinations
    rows = [["ID", "Части", "ΔROI close [ДИ]", "p / p Х", "ΔROI t09 [ДИ]", "p / p Х", "Ставок t09",
             "Сохраняет / сумма"]]
    kg = {k["id"]: k for k in comb["keeps_gains"]}
    for vid, g in ctab.loc[ctab["role"] == "combination"].groupby("id", sort=False):
        g = g.set_index("universe")
        a, b = g.loc["close"], g.loc["t09"]
        rows.append([vid, a["parts"], f"{pp(a['d_roi'])} {ci(a['d_lo'], a['d_hi'])}", f"{pv(a['p'])} / {pv(a['p_holm'])}",
                     f"{pp(b['d_roi'])} {ci(b['d_lo'], b['d_hi'])}", f"{pv(b['p'])} / {pv(b['p_holm'])}", n(b["bets"]),
                     f"{'да' if kg[vid]['keeps_gains_both'] else 'нет'} / {'да' if kg[vid]['additive_both'] else 'нет'}"])
    srows = [["Часть", "Что меняет", "ΔROI close [ДИ]", "p", "ΔROI t09 [ДИ]", "p"]]
    for vid, g in ctab.loc[ctab["id"].str.startswith("S_")].groupby("id", sort=False):
        g = g.set_index("universe")
        a, b = g.loc["close"], g.loc["t09"]
        part = a["parts"]
        srows.append([part, BLOCKS_RU[part], f"{pp(a['d_roi'])} {ci(a['d_lo'], a['d_hi'])}", pv(a["p"]),
                      f"{pp(b['d_roi'])} {ci(b['d_lo'], b['d_hi'])}", pv(b["p"])])
    rp = pd.DataFrame(comb["reproduction"])
    cc = ctab.set_index(["id", "universe"])
    floor_t09 = ", ".join(f"{k["id"]} {pp(cc.loc[(k['id'], 't09'), 'd_roi'])}" for k in plan["combinations"]
                          if "FLOOR" in k["parts"])
    rel_close = cc.loc[("S_REL", "close"), "d_roi"]
    prob_rel = ", ".join(f"{k['id']} ({'+'.join(k['parts'])}) {pp(cc.loc[(k['id'], 'close'), 'd_roi'])}"
                         for k in plan["combinations"] if "REL" in k["parts"]
                         and any(x in k["parts"] for x in ("CAL", "V6", "C2")))
    n_keep = sum(k["keeps_gains_both"] for k in comb["keeps_gains"])
    n_add = sum(k["additive_both"] for k in comb["keeps_gains"])
    rep.add(H1("7. Совместимость кандидатов: комбинации (этап 6)"),
            P(f"Список из {plan['n_combinations']} комбинаций зафиксирован до их расчёта "
              f"(`results/combinations_plan.json`, {plan['frozen_at']}). Комбинировались B-кандидаты, действующие на "
              "выбор Main: параметры правила (REL — надёжность TOTAL 0.95; EVW — вес EV 1.0; FLOOR — порог 1.40 с "
              "«не ставкой»), вероятности (CAL — температурная калибровка по группам; V6 — смесь 50/50 с рынком; C2 — "
              "смесь с консенсусом книг) и выходы моделей из кэшей исследований (GOALS — единая модель голов, "
              "CORNERS — Poisson GLM угловых). REL и HANDICAP 1.035 двигают Main в одну сторону, поэтому в "
              "комбинациях только REL — меньшее изменение правила."),
            P(f"Проверка инструмента: векторное правило продакшна совпадает с Baseline B в "
              f"{pct(comb['self_check_rule_equals_baseline_b']['close'], 2)} матчей close и "
              f"{pct(comb['self_check_rule_equals_baseline_b']['t09'], 2)} t09 (одно расхождение — равенство оценок "
              f"двух тождественных исходов); на таблицах с моделью голов — с Main самого движка "
              f"{pct(comb['self_check_goals_rule_equals_engine']['close'], 2)} / "
              f"{pct(comb['self_check_goals_rule_equals_engine']['t09'], 2)}. Каждая часть воспроизводит ROI своего "
              f"исследования: максимальное расхождение {rp['abs_diff'].max():.1e} ({len(rp)} проверок)."),
            P("Таблица 7 — части по отдельности (на том же инструменте).", "small"),
            table(srows, widths=[1.9 * cm, 6.0 * cm, 2.9 * cm, 1.2 * cm, 2.9 * cm, 1.2 * cm]),
            P("Таблица 8 — комбинации. p Х — Холм по 12 комбинациям в каждой вселенной. «Сохраняет» — ΔROI "
              "комбинации не меньше лучшей из её частей в обеих вселенных; «сумма» — не меньше суммы частей.", "small"),
            table(rows, widths=[1.1 * cm, 3.3 * cm, 2.6 * cm, 1.6 * cm, 2.6 * cm, 1.6 * cm, 1.5 * cm, 1.9 * cm]),
            *figure(FIG / "combinations_forest.png",
                    "Рисунок 3 — ΔROI Main частей (S_) и комбинаций (K) к Baseline B; круг — close, квадрат — t09."),
            note(f"Комбинации в целом не сохраняют преимущество частей: сохраняют в обеих вселенных {n_keep} из "
                 f"{plan['n_combinations']}, аддитивных — {n_add}. Ни одна комбинация не значима после поправки Холма "
                 f"(минимальный p Холма close {pv(ctab.loc[ctab['role'] == 'combination', 'p_holm'].min())}). "
                 f"Комбинации с порогом 1.40 в t09: {floor_t09}. Эффект REL на close ({pp(rel_close)} п.п.) в паре с "
                 f"вероятностной частью: {prob_rel}. Два хороших по отдельности правила не дают автоматически лучший "
                 "результат вместе (§17)."))
    rep.add(P("Не комбинировались (указано в плане до расчёта): " + "; ".join(f"{k} — {NOT_COMBINED_RU[k]}"
                                                                          for k in plan["not_combined"])
              + ". Их совместная проверка оставлена форвард-проверке."))

    # 8 model improvements
    alt_p = alt["production"]["metrics"]
    rows = [["Задача", "Продакшн 2026/27", "Лучший вариант (исследование)", "Δ к продакшну [ДИ]", "Решение"]]
    gl = C.read_json(V / "goals/results/results.json")
    gst = gl["paired_2026"]["stacked"]
    cn = C.read_json(V / "corners/results/results.json")
    glm = cn["eval_mean"]["glm"]
    am = alt["ens_xgb_best"]["paired"]
    ensall = alt["ens_all"]["paired"]
    sqm = sq2["model"]["pre"]["vs_production"]["ll_1x2"]
    qv = C.read_json(V / "quality/results/mitigations_2026.json")["variants"]
    qll = qv["mkt_both"]["universes"]["close"]["d_log_loss"]
    rows += [
        ["1X2", f"log loss {alt_p['result']['log_loss']:.4f}, acc {pct(alt_p['result']['accuracy'])} (рынок "
                f"{alt['_market_close']['metrics']['result']['log_loss']:.4f})",
         "смесь с рынком mkt_both (6)", f"{qll['diff'] * 1e3:+.2f}·10⁻³ [{qll['ci95'][0] * 1e3:+.2f}; "
                                        f"{qll['ci95'][1] * 1e3:+.2f}]", "B"],
        ["1X2", "", "ансамбль всех алгоритмов ens_all (8)", f"{ensall['ll']['diff'] * 1e3:+.2f}·10⁻³ "
         f"[{ensall['ll']['ci95'][0] * 1e3:+.2f}; {ensall['ll']['ci95'][1] * 1e3:+.2f}]", "C"],
        ["1X2", "", f"сила состава PRE (12), {n(sqm['n'])} матчей", f"{sqm['diff'] * 1e3:+.1f}·10⁻³ "
         f"[{sqm['ci95'][0] * 1e3:+.1f}; {sqm['ci95'][1] * 1e3:+.1f}]", "B"],
        ["Голы", f"log loss счёта {gl['summary_2026']['prod']['score_ll']:.4f}, девиация "
                 f"{gl['summary_2026']['prod']['dev_goals']:.4f}", "единая модель голов команды (10)",
         f"счёт {gst['score_ll']['diff'] * 1e3:+.2f}·10⁻³ [{gst['score_ll']['ci95'][0] * 1e3:+.2f}; "
         f"{gst['score_ll']['ci95'][1] * 1e3:+.2f}], девиация {gst['dev_goals']['diff'] * 1e3:+.2f}·10⁻³", "B"],
        ["Голы хозяев", f"девиация {alt_p['home_goals']['poisson_deviance']:.4f}", "ens_xgb_best: XGB + GLM (8)",
         f"{am['dev_home_goals']['diff'] * 1e3:+.2f}·10⁻³ [{am['dev_home_goals']['ci95'][0] * 1e3:+.2f}; "
         f"{am['dev_home_goals']['ci95'][1] * 1e3:+.2f}]", "B"],
        ["Угловые", f"MAE {cn['eval_mean']['xgb_prod']['mae']:.3f}, R² {cn['eval_mean']['xgb_prod']['r2']:.3f}",
         "Poisson GLM + NB по лигам (11)", f"MAE {glm['mae']:.3f}, R² {glm['r2']:.3f}; девиация "
         f"{glm['dev_diff']['diff']:+.4f} [{glm['dev_diff']['ci95'][0]:+.4f}; {glm['dev_diff']['ci95'][1]:+.4f}]", "B"]]
    rep.add(H1("8. Улучшения моделей 1X2, голов и угловых"),
            P("Модели продакшна на 2026/27 и лучшие варианты исследований (парные разности на тех же матчах; "
              "отрицательная разность log loss / девиации — улучшение)."),
            table(rows, widths=[1.8 * cm, 4.2 * cm, 4.0 * cm, 5.0 * cm, 1.4 * cm]),
            P(f"Ни одно улучшение моделей не меняет Main ROI значимо. Самые заметные выигрыши — у вариантов, "
              f"использующих рынок (смесь 1X2 с ценой) или данные, недоступные в ежедневном конвейере (составы). "
              f"Модель 1X2 отстаёт от рынка на {gap_mkt['diff'] * 1e3:+.1f}·10⁻³ log loss "
              f"[{gap_mkt['ci95'][0] * 1e3:+.1f}; {gap_mkt['ci95'][1] * 1e3:+.1f}] — на порядок больше любого найденного "
              f"улучшения самой модели. Угловые: потолок предсказуемости низкий (R² около "
              f"{cn['eval_mean']['xgb_prod']['r2']:.2f}), выигрыш GLM — доли процента девиации."))

    # 9 squad
    e1 = sq1["2025"]["variants"]
    rows = [["Вариант", "Решение", "Данные", "Δ log loss 1X2 [ДИ], ·10⁻³", "p Холма", "ΔROI close / t09, п.п."]]
    for k, lab, data in (("dyn", "E1 DYN (динамический состав)", "2025/26"), ("pre", "E1 PRE (предсезонный)", "2025/26")):
        x = e1[k]["covered"]["ll_1x2"]
        rows.append([lab, sqd[f"e1_{k}"]["decision"], f"{data}, {n(x['n'])} матчей",
                     f"{x['diff'] * 1e3:+.1f} [{x['ci95'][0] * 1e3:+.1f}; {x['ci95'][1] * 1e3:+.1f}]", pv(x.get("p_holm")), "—"])
    for k in ("pre", "sa"):
        x = sq2["model"][k]["vs_production"]["ll_1x2"]
        r = {u: sq2["universes"][u]["variants"][k]["covered"]["vs_b"]["roi_diff"] for u in U}
        rows.append([f"E2 {k.upper()} (2026/27)", sqd[f"e2_{k}"]["decision"], f"2026/27, {n(x['n'])} матчей",
                     f"{x['diff'] * 1e3:+.1f} [{x['ci95'][0] * 1e3:+.1f}; {x['ci95'][1] * 1e3:+.1f}]", pv(x.get("p_holm")),
                     f"{pp(r['close']['diff'])} / {pp(r['t09']['diff'])}"])
    rep.add(H1("9. Результаты Squad Strength"),
            P(f"Сила состава оценивалась по открытому датасету Transfermarkt ({sq_leagues} из {n_leagues} лиг); "
              f"в 2026/27 покрыто сыгранных матчей: {n(sq2['covered_played'])} из {n(sq2['played'])}. Признаки состава "
              f"значимо улучшают log loss 1X2 на ранних сезонах и на 2026/27, но Main ROI не улучшают."),
            table(rows, widths=[3.6 * cm, 1.4 * cm, 3.0 * cm, 3.6 * cm, 1.4 * cm, 3.4 * cm]),
            P(f"Проверено источников: {len(sqs['sources'])}; поддерживаемого и юридически чистого "
              "источника составов для ежедневного конвейера нет (датасет заморожен, сайт Transfermarkt запрещает "
              "машинное использование; в сохранённых ответах API-Football нет ни одного ответа с составами). "
              "Внедрение невозможно до появления такого источника; пилот сбора через API-Football требует "
              "согласия владельца."))

    # 10 recent weighting
    rows = [["Схема (ежемесячное переобучение)", "Δ 1X2 ·10⁻³ [ДИ]", "Δ голы ·10⁻³", "Δ угловые ·10⁻³",
             "ΔROI close / t09", "Решение 1X2/голы/угл."]]
    for sch in ("prod", "hl0", "hl0.5", "hl1", "hl2", "hl4", "last2", "last3", "cur2", "cur4", "win2", "win3"):
        key_ = f"2026|wf|{sch}|vs_B"
        if key_ not in rw["paired"]:
            continue
        x = rw["paired"][key_]
        dec = "/".join(rw["decisions"].get(f"{sch}|{t}", "контроль") for t in ("result", "goals", "corners")) \
            if sch != "prod" else "контроль"
        rr = {u: rw_roi.loc[(f"wf:{sch}", u), "d_roi"] for u in U}
        rows.append([rw["names"][sch], f"{x['result']['diff'] * 1e3:+.2f} [{x['result']['ci95'][0] * 1e3:+.2f}; "
                     f"{x['result']['ci95'][1] * 1e3:+.2f}]", f"{x['goals']['diff'] * 1e3:+.2f}",
                     f"{x['corners']['diff'] * 1e3:+.2f}", f"{pp(rr['close'])} / {pp(rr['t09'])}", dec])
    rep.add(H1("10. Результаты Recent Season Weighting"),
            P(f"Проверено {rw['n_variants']['monthly_schemes']} схем весов с ежемесячным переобучением, "
              f"{rw['n_variants']['static_schemes']} однократных и {len(rw['n_variants']['other'])} прочих "
              f"({rw['n_variants']['roi_variants']} вариантов с Main ROI). Выбор схемы — на 2025/26."),
            table(rows, widths=[4.3 * cm, 3.1 * cm, 1.8 * cm, 1.9 * cm, 2.5 * cm, 2.6 * cm]),
            P("Гипотеза о пользе большего веса текущего сезона не подтверждается: короткая память (быстрое затухание, "
              "короткое окно, повышенный вес прошлого сезона) значимо ухудшает прогноз, а вес текущего сезона ×2 / ×4 "
              "в пределах шума; порог числа сыгранных матчей, после которого вес помогает, не найден. B — только "
              "«текущий сезон ×2» для 1X2 и «без затухания» для голов (форвард-проверка)."),
            P(f"Шум переобучения: та же модель, переобученная на тех же данных, отличается от Baseline B по ROI close "
              f"на {pp(retrain['d_roi'])} п.п. {ci(retrain['d_lo'], retrain['d_hi'])} — различия ROI порядка 1 п.п. "
              f"между вариантами переобучения не являются эффектом весов."))

    # 11 reliability
    n_var_total = sum(int(s["n_variants"]) for s in studies)
    rep.add(H1("11. Оценка статистической надёжности"),
            *bullets([
                f"**Множественные проверки.** Исследования перебрали в сумме около {n_var_total} вариантов (по "
                f"данным их файлов результатов) и {plan['n_combinations']} комбинаций. После поправки Холма ни один "
                f"кандидат Main ROI не значим ни в одной вселенной; минимальный p Холма close — "
                f"{pv(min(r['p_holm_close'] for r in key['ranking']))}.",
                f"**Размер эффекта против шума.** Медианная ширина 95 % ДИ разности ROI close у B-кандидатов — "
                f"{width_close * 100:.1f} п.п., медианный |ΔROI| — {abs_close * 100:.1f} п.п. Для подтверждения "
                f"нужны тысячи будущих ставок; минимальные форвард-выборки указаны в предрегистрациях.",
                f"**Две вселенные расходятся.** Многие правила положительны на close и отрицательны на t09 (порог "
                f"1.40: close {pp(st['main_selection']['b'][2]['close']['d_roi'])}, t09 "
                f"{pp(st['main_selection']['b'][2]['t09']['d_roi'])} п.п.); в t09 {pct(unpriced_t09)} Main Baseline B "
                f"без цены, а правило меняет и число ставок.",
                f"**Проклятие победителя.** Заявленная вероятность выбранного Main выше частоты попаданий на "
                f"{pp(wc['gap'])} п.п. {ci(*wc['gap_ci'])} — эффект отбора максимума среди многих кандидатов, а не "
                f"плохая калибровка отдельных вероятностей.",
                "**2026/27 уже изучался** (v6.1, v6.2): лучший по ROI вариант, выбранный на этом же сезоне, "
                "является оптимистичной оценкой. Поэтому все B заморожены (preregistration.json с моментом заморозки).",
                "**Зависимость от крупных выигрышей и коэффициентов** проверена в каждом исследовании (ROI без 5 "
                "крупнейших выигрышей, цена −2 %/−5 %); выводы от этого не меняются."]))

    # 12 implemented
    rep.add(H1("12. Внедрённые улучшения"),
            P("Ни одна исследовательская модель и ни одно новое правило Main не внедрены в production (решения A "
              "нет). В коде v6.3 (ветка `feature/system-v6.3`, ещё не объединена с `main` и не развёрнута) "
              "реализованы изменения, не связанные с моделями (по `git log origin/main..HEAD`, без коммитов самих "
              "исследований):"),
            table(implemented(), widths=[5.0 * cm, 1.6 * cm, 10.4 * cm]),
            P("Утверждать, что изменения работают на опубликованном сайте, можно только после слияния с `main` и "
              "развёртывания.", "small"))

    # 13 rejected
    rep.add(H1("13. Отклонённые методы"),
            P("Решение C по файлам результатов исследований (не показали преимущества, ухудшили результат или "
              "слишком сложны для полученной пользы):"),
            table(rejected(), widths=[3.4 * cm, 13.6 * cm]))

    # 14 problems
    rep.add(H1("14. Оставшиеся нерешённые проблемы"),
            *bullets([
                "**Нет поддерживаемого источника составов** — Squad Strength нельзя внедрить; история составов у "
                "проекта отсутствует.",
                f"**Архив цен хранит в основном одну котировку на исход** (bet365 OddsPapi: одна котировка у "
                f"{pct(arch['share_1_quote'])} исходов, к моменту T известна у {pct(arch['share_quoted_by_T'])}) — "
                f"исследования времени и движения линии ограничены; {pct(unpriced_t09)} Main в момент T без цены.",
                f"**Проклятие победителя ≈ {pp(wc['gap'])} п.п.** — заявленная вероятность Main систематически выше "
                f"частоты попаданий; поправки на отбор не улучшили ROI.",
                f"**Модель отстаёт от рынка** на {gap_mkt['diff'] * 1e3:.1f}·10⁻³ log loss 1X2 "
                f"({pm['1x2_log_loss']:.4f} против {pm['market_1x2_log_loss_close']:.4f} у цены закрытия без маржи) — "
                "смеси с рынком сокращают разрыв, но не дают устойчивого ROI.",
                f"**Main ROI отрицателен** у продакшна в обеих вселенных; наибольший ROI close среди всех "
                f"B-кандидатов и комбинаций — {roi(max(bcand['roi_close'].max(), ctab.loc[ctab['universe'] == 'close', 'roi'].max()))}; "
                "доходной стратегии с доказанным преимуществом не найдено.",
                "**Многокнижный консенсус в момент прогноза** доступен только по снимкам Football-Data — для C1/C2 "
                "нужен источник нескольких книг в ежедневном конвейере."]))

    # 15 conclusions
    rep.add(H1("15. Выводы для дипломного проекта"),
            *bullets([
                "Построена воспроизводимая экспериментальная инфраструктура: полный пересчёт сезона 2026/27 в двух "
                "моментах решения, две базовые линии, парный бутстреп, поправка Холма, предрегистрация кандидатов и "
                "их ежедневная теневая запись для форвард-проверки.",
                f"13 исследований и {plan['n_combinations']} комбинаций показали: действующая система близка к "
                f"пределу, достижимому на имеющихся данных; улучшения моделей реальны, но малы (единицы ·10⁻³ log "
                f"loss) и не переходят в значимый рост ROI.",
                f"Лучший по доказательности кандидат ({best['name']}) и лучшая комбинация правил ({bk['name']}) дают "
                f"+{abs(c['d_roi']) * 100:.1f} и +{abs(bk['close']['d_roi']) * 100:.1f} п.п. ROI на close без "
                "значимости после поправки на множественные проверки — это гипотезы, а не результат.",
                "Основное ограничение точности — информация, а не алгоритм: рынок знает больше модели (разрыв "
                "log loss), а данные о составах и многокнижные цены в момент прогноза недоступны.",
                "Методологический результат для записки: без парного сравнения, двух вселенных и поправки на "
                f"множественность легко «найти» правило с {pp(ms_max)} п.п. ROI на close (лучший из кандидатов "
                f"исследования 1), которое не переживает проверку."]))

    # 16 limitations
    rep.add(H1("16. Ограничения"),
            *bullets([
                f"Один сезон проверки (2026/27, {n(n_played)} матчей), уже изученный ранее; результаты — хронологическая "
                "проверка, не отложенная выборка.",
                "Холм в каждом исследовании считался по его собственной семье (разного размера); сравнение p Холма "
                "между исследованиями приблизительно.",
                "Комбинации проверены на тех же матчах, на которых выбраны части; их оценки оптимистичны.",
                "Модельные B-кандидаты, выходы которых не кэшированы (ансамбли ML, составы, схемы весов), не "
                "комбинировались.",
                f"В t09 {pct(unpriced_t09)} Main Baseline B без цены; правила меняют число ставок, поэтому ΔROI в t09 частично "
                "отражает смену состава ставок."]))
    rep.add(H1("17. Как воспроизвести"),
            *K.reproduce([".venv/bin/python -m experiments.v6_3.summary.plan        # план комбинаций (один раз)",
                          ".venv/bin/python -m experiments.v6_3.summary.overview",
                          ".venv/bin/python -m experiments.v6_3.summary.combine",
                          ".venv/bin/python -m experiments.v6_3.summary.key_table",
                          ".venv/bin/python -m experiments.v6_3.summary.build_pdf"]),
            P("Входы: результаты 13 исследований (`experiments/v6_3/<исследование>/results`), кэш replay "
              "(`experiments/v6_3/replay/cache`), кэши калибровки, голов, угловых и консенсуса. Выходы: "
              "`experiments/v6_3/summary/results/` (overview, b_candidates, combinations, production_vs_best, "
              "per_match_…), рисунки — `figures/`.", "small"))
    write_findings(ov, comb, plan, key, bcand, ctab, gap_mkt, wc, unpriced_t09)
    rep.story = keep_headings(rep.story)
    rep.build()
    print("pages:", rep.check())


def write_findings(ov, comb, plan, key, bcand, ctab, gap_mkt, wc, unpriced_t09) -> None:
    """findings.md of the summary, every number from the result files."""
    bb, ba, best = comb["baseline_b"], comb["baseline_a"], key["best"]
    bk, br = key["best_combination"], key["best_rule_candidate"]
    kg = comb["keeps_gains"]
    lines = ["# FootPredict v6.3 — сводка 13 исследований: выводы", "",
             "Все числа — из `results/*.json|csv` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_3/FootPredict_v6.3_Research_Summary.pdf`.", "",
             "## Главное", "",
             f"- Решения A нет ни в одном исследовании и ни в одной комбинации; production не меняется. "
             f"B-кандидатов: {len(bcand)}, положительны в обеих вселенных: "
             f"{int(((bcand['d_roi_close'] > 0) & (bcand['d_roi_t09'] > 0)).sum())}, значимы после Холма: "
             f"{int((bcand['p_holm_close'] < 0.05).sum() + (bcand['p_holm_t09'] < 0.05).sum())}.",
             f"- Baseline B: close ROI {roi(bb['close']['summary']['roi'])} {ci(*bb['close']['roi_ci']['ci95'])} "
             f"({n(bb['close']['summary']['bets'])} ставок), t09 {roi(bb['t09']['summary']['roi'])} "
             f"{ci(*bb['t09']['roi_ci']['ci95'])} ({n(bb['t09']['summary']['bets'])}); Baseline A "
             f"{roi(ba['summary']['roi'])} ({n(ba['summary']['bets'])}).",
             f"- Лучший проверенный кандидат (критерий доказательности, зафиксирован до комбинаций): **{best['name']}** "
             f"({best['source']}: {best['label']}) — close {pp(best['close']['d_roi'])} п.п. "
             f"{ci(best['close']['d_lo'], best['close']['d_hi'])}, p Холма {pv(best['close']['p_holm'])}; t09 "
             f"{pp(best['t09']['d_roi'])} п.п. {ci(best['t09']['d_lo'], best['t09']['d_hi'])}. ROI close "
             f"{roi(best['close']['roi'])} — остаётся отрицательным.",
             f"- Лучшее изменение правила: **{br['name']}** ({br['label']}): close {pp(br['close']['d_roi'])}, t09 "
             f"{pp(br['t09']['d_roi'])} п.п., p Холма {pv(br['close']['p_holm'])}. Лучшая комбинация: **{bk['name']}** "
             f"({bk['label']}): close {pp(bk['close']['d_roi'])} {ci(bk['close']['d_lo'], bk['close']['d_hi'])}, "
             f"t09 {pp(bk['t09']['d_roi'])} {ci(bk['t09']['d_lo'], bk['t09']['d_hi'])}, p Холма close "
             f"{pv(bk['close']['p_holm'])}.",
             f"- Комбинации ({plan['n_combinations']}, план заморожен {plan['frozen_at']}): сохраняют выигрыш частей в "
             f"обеих вселенных {sum(k['keeps_gains_both'] for k in kg)}, аддитивны {sum(k['additive_both'] for k in kg)}; "
             f"ни одна не значима после Холма. Инструмент воспроизводит ROI всех частей (расхождение "
             f"≤ {max(r['abs_diff'] for r in comb['reproduction']):.1e}); правило = Baseline B в "
             f"{pct(comb['self_check_rule_equals_baseline_b']['close'], 2)} / "
             f"{pct(comb['self_check_rule_equals_baseline_b']['t09'], 2)} матчей.",
             f"- Нерешённое: модель 1X2 хуже рынка на {gap_mkt['diff'] * 1e3:.1f}·10⁻³ log loss; проклятие победителя "
             f"{pp(wc['gap'])} п.п.; {pct(unpriced_t09)} Main в момент T без цены; нет источника составов.", "",
             "## Комбинации (ΔROI к Baseline B, п.п.)", "",
             "| ID | Части | close [ДИ] | p / p Холма | t09 [ДИ] | p / p Холма | Сохраняет |", "|---|---|---|---|---|---|---|"]
    kgd = {k["id"]: k for k in kg}
    for vid, g in ctab.loc[ctab["role"] == "combination"].groupby("id", sort=False):
        g = g.set_index("universe")
        a, b = g.loc["close"], g.loc["t09"]
        lines.append(f"| {vid} | {a['parts']} | {pp(a['d_roi'])} {ci(a['d_lo'], a['d_hi'])} | {pv(a['p'])} / "
                     f"{pv(a['p_holm'])} | {pp(b['d_roi'])} {ci(b['d_lo'], b['d_hi'])} | {pv(b['p'])} / {pv(b['p_holm'])} "
                     f"| {'да' if kgd[vid]['keeps_gains_both'] else 'нет'} |")
    lines += ["", "## Решения", "",
              "- A: нет. Изменения production не предлагаются.",
              f"- B (форвард-проверка, без изменений): все {len(bcand)} предрегистрированных кандидатов исследований с Main ROI (плюс squad e1_dyn / e1_pre — только метрики модели); "
              f"комбинации не регистрируются отдельно — их части уже заморожены, а лучшая комбинация "
              f"({bk['name']}) складывается из правил, которые с v6.3 ежедневно записываются как теневые.",
              "- C: всё, что отклонили исследования (раздел 13 PDF).", "",
              "## Файлы", "",
              "- `results/overview.csv|json`, `results/b_candidates.csv` — обзор 13 исследований и все B-кандидаты.",
              "- `results/combinations_plan.json` (план, заморожен до расчёта), `results/combinations.csv|json`, "
              "`results/combinations_coverage.csv`, `results/per_match_*` — комбинации.",
              "- `results/production_vs_best.csv|json` — ключевая таблица.", "",
              "## Воспроизведение", "", "```",
              ".venv/bin/python -m experiments.v6_3.summary.run",
              ".venv/bin/python -m experiments.v6_3.summary.build_pdf", "```", ""]
    (C.V63 / "summary" / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def keep_headings(story: list) -> list:
    """A section heading never stays alone at the bottom of a page: it is kept with the next two flowables."""
    from reportlab.platypus import KeepTogether, Paragraph
    out, i = [], 0
    while i < len(story):
        f = story[i]
        if isinstance(f, Paragraph) and f.style.name == "h1" and i + 1 < len(story):
            out.append(KeepTogether(story[i:i + 3]))
            i += 3
            continue
        out.append(f)
        i += 1
    return out


if __name__ == "__main__":
    build()
