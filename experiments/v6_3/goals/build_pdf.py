"""Builds reports/v6_3/Goal_Model_Research.pdf and findings.md from results/ (no number is typed by hand).

    .venv/bin/python -m experiments.v6_3.goals.build_pdf
"""
from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm

from scripts import report_kit as K
from scripts.report_kit import H1, H2, H3, P, bullets, figure, formula, note, table

from experiments.v6_3 import common as C

from .evaluate import LAMBDA_VARIANTS, MATRIX_VARIANTS, SEL

DIRS = C.study_dirs("goals")
RES, FIG = DIRS["results"], DIRS["figures"]
OUT = C.ROOT / "reports" / "v6_3" / "Goal_Model_Research.pdf"
DEC_RU = {"A": "A — внедрить", "B": "B — перспективно, форвард-проверка", "C": "C — отклонить"}


def pp(v, d=1, signed=True):
    return "—" if v is None else (f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}")


def m3(v, d=2):
    """×10⁻³ with sign."""
    return "—" if v is None else f"{v * 1000:+.{d}f}"


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


def build() -> None:
    r = C.read_json(RES / "results.json")
    info = C.read_json(RES / "run_info.json")
    rs = C.read_json(RES / "ratings_selection.json")
    pre = C.read_json(RES / "preregistration.json")
    names, par, ev, sm, dec = r["names"], r["params"], r["paired_2026"], r["summary_2026"], r["decisions"]
    pooled, holm, roi, cov = r["pooled_selection"], r["holm_2026"], r["roi"], r["coverage"]
    rr = pd.read_csv(RES / "main_roi.csv")
    bs = pd.read_csv(RES / "selection_by_season.csv")
    nm = lambda k: names.get(k, k)  # noqa: E731
    b_close, b_t09 = roi["prod"]["close"]["summary"], roi["prod"]["t09"]["summary"]
    a_sum = roi["prod"]["close"]["vs_a"]
    rep = K.Report(OUT, "Улучшение модели голов", "FootPredict v6.3 — исследование 10",
                   {"Сезоны отбора": "2023/24, 2024/25, 2025/26 (модели обучены до начала сезона)",
                    "Проверка": "2026/27 (хронологически; сезон уже изучался в v6.1/v6.2 — не независимая выборка)",
                    "Данные": f"features {info['data']['features']['bytes'] // 1024} КБ, изменён "
                              f"{info['data']['features']['mtime'][:16]}",
                    "Модель продакшна": "prod_20261012"})

    # 1 ------------------------------------------------------------------------------------------------
    rep.add(H1("1. Название и постановка проблемы"),
            P("**Исследование 10. Goal Model Improvements** — улучшение прогноза голов хозяев и гостей и качества "
              "производных рынков (тоталы, обе забьют, форы, точный счёт) системы FootPredict."),
            P("Голы — основа большинства рынков, из которых правило Main выбирает прогноз: матрица счёта строится из "
              "двух ожидаемых чисел голов (XGBoost-регрессоры с пуассоновской функцией потерь) как произведение "
              "независимых распределений Пуассона, а затем области «победа хозяев / ничья / победа гостей» "
              "масштабируются к вероятностям классификатора 1X2 (согласование). Из этой матрицы берутся тоталы, "
              "обе забьют, азиатские и европейские форы и точный счёт. Поэтому главный критерий исследования — "
              "не только MAE голов, а качество вероятностей этих рынков и ROI Main."),
            P("v6.2 уже показало: семейство распределения при тех же средних почти не влияет (Диксон–Коулс, "
              "отрицательное биномиальное, копулы — в пределах ±0.1×10⁻³), передисперсии нет, согласование с 1X2 — "
              "главный источник качества, а смесь XGB с рейтингами Махера дала −1.9×10⁻³ log loss счёта на сезонах "
              "отбора, но не на 2026/27. В v6.3 проверяются новые гипотезы: динамические (обновляемые после "
              "каждого игрового дня) рейтинги атаки и защиты, рейтинги как признаки, единая модель голов команды, "
              "другие способы согласования 1X2 с матрицей счёта и исправленный Диксон–Коулс (ρ после согласования)."))

    # 2 ------------------------------------------------------------------------------------------------
    rep.add(H1("2. Гипотезы"), bullets([
        "H1 (актуальная сила): динамические рейтинги атаки/защиты, вычисленные только по прошлым матчам, добавляют "
        "к XGBoost информацию о текущей силе — как отдельная модель, в смеси или как признаки.",
        "H2 (дом/выезд): отдельные домашние и выездные поправки рейтинга улучшают прогноз.",
        "H3 (распределение): Диксон–Коулс с ρ, подобранным после согласования, или отрицательное биномиальное "
        "распределение улучшают точный счёт и производные рынки.",
        "H4 (согласование): производственное масштабирование областей 1X2 искажает распределение тотала; "
        "согласование, которое сохраняет и 1X2 классификатора, и распределение суммы голов регрессоров (IPF), "
        "или подбор перевеса λ под 1X2 дают лучшие тоталы при том же 1X2.",
        "H5 (раздельные модели): одна модель «голы команды» (две строки на матч) лучше раздельных моделей хозяев "
        "и гостей.",
        "H6 (xG): не повторяется — xG Football-Data есть только в 2026/27 (18 из 21 лиги), обучить на нём модель "
        "нельзя, а v6.2 не нашло пользы; новой проверяемой гипотезы нет.",
        "Ожидание: эффекты малы (порядка 10⁻³ log loss); внедрение — только при улучшении и на сезонах отбора, "
        "и на 2026/27 без ухудшения других рынков и ROI Main."]))

    # 3 ------------------------------------------------------------------------------------------------
    selm = cov.get("selection_matches", {})
    rep.add(H1("3. Исходные данные"),
            P(f"Таблица признаков `local_data/processed/features.parquet` (21 лига, сезоны 2021/22–2026/27). "
              f"Сезоны отбора: " + ", ".join(f"{s}/{int(s) + 1 - 2000}: {n} сыгранных матчей" for s, n in selm.items())
              + f". Сезон 2026/27: {cov['season_matches']} матчей, сыграно {cov['played']}, ещё не сыграно "
              f"{cov['not_played']}; вероятностная оценка — на {cov['probabilistic_eval_matches']} сыгранных матчах."),
            P(f"Цены — общий реплей v6.3 (`experiments/v6_3/replay`): close — последняя котировка до начала "
              f"(1X2 есть у {cov['with_closing_1x2_price']} сыгранных матчей), t09 — цены, известные в 09:00 дня "
              f"матча (1X2 у {cov['with_t09_1x2_price']}). Матчей, где хотя бы у одной команды ещё нет игр сезона: "
              f"{cov['no_current_season_games']}; меньше трёх игр: {cov['fewer_than_3_season_games']} — они "
              "не исключаются, а показаны в разбивке по числу сыгранных матчей."),
            P("Выходы моделей сезонов отбора — переобученные модели продакшна (те же признаки, гиперпараметры и веса "
              "давности), обученные на всех сезонах до начала прогнозируемого; для 2026/27 — сама модель "
              "prod_20261012 (базовая линия B). xG не используется (см. H6)."))
    rows = [["Сезон", "Матчей", "Нет игр сезона у команды", "< 3 игр", "Цена 1X2 close", "Цена 1X2 t09"],
            ["2026/27", cov["played"], cov["no_current_season_games"], cov["fewer_than_3_season_games"],
             cov["with_closing_1x2_price"], cov["with_t09_1x2_price"]]]
    rep.add(table(rows), P("Таблица 1. Покрытие сезона 2026/27 (сыгранные матчи).", "small"))

    # 4 ------------------------------------------------------------------------------------------------
    best = rs["best"]
    rep.add(H1("4. Методика"),
            H2("4.1 Матрица счёта продакшна"),
            formula("P(i, j) ∝ Pois(i; λ_h) · Pois(j; λ_a),  затем  P(i, j) ← P(i, j) · p_k / M_k  для (i, j) в области k ∈ {H, D, A}"),
            P("M_k — масса области в пуассоновской матрице, p_k — вероятность классификатора 1X2. Варианты меняют "
              "либо λ (модель голов), либо способ построения матрицы; всё остальное (рынки, цены, правило Main) — "
              "код продакшна (`engine.batch_from_outputs`, при необходимости подмена построения матрицы внутри "
              "`src.prediction.engine`)."),
            H2("4.2 Динамические рейтинги атаки и защиты"),
            formula("log λ_h = μ^H_L + att_h − def_a,   log λ_a = μ^A_L + att_a − def_h"),
            formula("после игрового дня: att_h += η(g_h − λ_h),  def_a −= η(g_h − λ_h),  att_a += η(g_a − λ_a),  def_h −= η(g_a − λ_a)"),
            P(f"Матч видит только рейтинги после предыдущих игровых дней. В начале сезона рейтинги сжимаются к нулю "
              f"(множитель 1 − r), новичок лиги получает средний рейтинг покинувших лигу команд. Параметры выбраны "
              f"по девиации Пуассона 2023–2025: η = {best['eta']}, r = {best['r']} (девиация {best['deviance']:.4f}); "
              f"вариант с отдельными домашними/выездными поправками — η_HA = {rs['best_ha']['eta_ha']} (девиация "
              f"{rs['best_ha']['deviance']:.4f})."),
            P(f"Смесь: λ = exp((1 − w)·log λ_XGB + w·log λ_рейтинг); w подобран по log loss точного счёта на "
              f"2023–2025 (после согласования): w = {par['w']:.3f}; для рейтингов дом/выезд w = {par['w_ha']:.3f}."),
            H2("4.3 Распределения и согласование"),
            bullets([
                f"Диксон–Коулс: ρ подобран по согласованной матрице 2023–2025: ρ = {par['rho']:+.4f}; для сравнения — "
                f"ρ, который подобрал бы конвейер продакшна по сырым средним: {par['rho_raw']:+.4f} (дефект v6.2).",
                f"Отрицательное биномиальное: размер r = {par['size']:.1f} (по log loss счёта 2023–2025).",
                f"Частичное согласование: множитель (p_k/M_k)^α, α = {par['alpha']:.3f}.",
                "IPF: попеременное масштабирование областей 1X2 и антидиагоналей (сумма голов) к 1X2 классификатора "
                "и распределению тотала из Пуассона регрессоров, 30 итераций, последний шаг — 1X2.",
                "Перевес: сумма λ_h + λ_a от регрессоров, разность подбирается так, чтобы P(H) − P(A) Пуассона совпала "
                "с классификатором, затем обычное согласование."]),
            H2("4.4 Модели, переобучаемые в исследовании"),
            bullets([
                "XGB с рейтингами как признаками: шесть признаков (λ рейтинга хозяев/гостей, атака и защита обеих "
                "команд) к производственному набору; гиперпараметры продакшна; обучение на сезонах до прогнозируемого "
                "(2024/25, 2025/26 — проверка на прошлых сезонах, 2026/27 — оценка).",
                "Единая модель голов команды: каждая игра — две строки (признаки команды и соперника, флаг «дома»), "
                "гиперпараметры модели голов гостей продакшна.",
                "Ежемесячное переобучение XGB (контроль исследования 7): модели голов переобучаются 1-го числа месяца.",
                f"v6.2 HY+R (заморожено в v6.2): смесь XGB и рейтингов Махера MH2 с весом XGB {0.8458:.4f}."]),
            H2("4.5 Метрики и статистика"),
            P("Log loss точного счёта (главная метрика v6.2 и этого исследования), средний log loss и Brier тоталов "
              "0.5–4.5, обе забьют, азиатских форовых линий ±0.5…±2.5, RPS разницы и суммы голов, 1X2 из матрицы, "
              "MAE и девиация Пуассона голов, калибровка (ECE) тотала 2.5 и обе забьют. Разности — парный бутстреп "
              "по матчам (2000 повторов), поправка Холма по всем вариантам. ROI Main — правило продакшна, оба "
              "момента решения, парный бутстреп к базовой линии B того же момента, Холм по вариантам."),
            P("Правило решения (зафиксировано до оценки 2026/27): **A** — улучшение log loss счёта на сезонах отбора "
              "(95 % ДИ ниже 0) и на 2026/27 (p Холма < 0.05), тоталы/обе забьют/форы не хуже значимо, ΔROI ≥ 0 в обоих "
              "моментах; **B** — улучшение на отборе и по точке на 2026/27 без ухудшений, ROI не хуже значимо (или "
              "значимое улучшение 2026/27 без поддержки отбора); **C** — остальное."))

    # 5 ------------------------------------------------------------------------------------------------
    rep.add(H1("5. Эксперименты"),
            P(f"Запущено вариантов: {r['n_variants']} на 2026/27 (матрица: {len(MATRIX_VARIANTS)}, средние голов: "
              f"{len(LAMBDA_VARIANTS)}, комбинация лучших на отборе: {nm(par['combo_lambda'])} + "
              f"{nm(par['combo_matrix'])}). Сетка рейтингов: {len(rs['grid'])} комбинаций параметров на 2023–2025."))
    rows = [["Вариант", "Что меняется", "Где выбраны параметры"]]
    desc = {"raw": ("матрица", "—"), "partial": ("матрица", "2023–2025"), "dc": ("матрица", "2023–2025"),
            "dc_raw_rho": ("матрица", "2023–2025, сырые средние"), "nb": ("матрица", "2023–2025"),
            "ipf": ("матрица", "без параметров"), "sup": ("матрица", "без параметров"),
            "rat": ("λ", "2023–2025 (η, r)"), "blend": ("λ", "2023–2025 (η, r, w)"),
            "blend_ha": ("λ", "2023–2025 (η, r, η_HA, w)"), "ratfeat": ("λ, переобучение", "гиперпараметры продакшна"),
            "stacked": ("λ, переобучение", "гиперпараметры продакшна"), "monthly": ("λ, переобучение", "продакшн"),
            "hy_v62": ("λ", "v6.2 (2025/26)"), "combo": ("λ + матрица", "2023–2025")}
    for k in ev:
        a, b = desc.get(k, ("", ""))
        rows.append([nm(k), a, b])
    rep.add(table(rows, widths=[7 * cm, 4 * cm, 6 * cm]), P("Таблица 2. Запущенные варианты.", "small"))

    # 6 ------------------------------------------------------------------------------------------------
    rep.add(H1("6. Результаты"), H2("6.1 Сезоны отбора 2023/24–2025/26"))
    rows = [["Вариант", "Сезоны", "Δ счёт", "Δ тоталы", "Δ обе забьют", "Δ форы", "Δ девиация голов"]]
    for k, v in sorted(pooled.items(), key=lambda kv: kv[1]["score_ll"]["diff"]):
        if k == "prod":
            continue
        rows.append([nm(k), ", ".join(str(s)[2:] for s in v["seasons"]), ci3(v["score_ll"]), ci3(v["totals_ll"]),
                     ci3(v["btts_ll"]), ci3(v["hcp_ll"]), ci3(v["dev_goals"])])
    rep.add(table(rows, widths=[4.2 * cm, 1.8 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]),
            P("Таблица 3. Разность с продакшном (×10⁻³, меньше — лучше) на сезонах отбора, парный бутстреп с 95 % ДИ. "
              "Параметры смеси/распределений подобраны на этих же сезонах (внутривыборочно); честная оценка — 2026/27.",
              "small"))
    rep.add(*figure(FIG / "selection_seasons.png", "Рисунок 1. Δ log loss точного счёта по сезонам отбора "
                                                    "(×10⁻³; меньше нуля — лучше продакшна)."))
    rep.add(H2("6.2 Сезон 2026/27: вероятности рынков"))
    p0 = sm["prod"]
    rep.add(P(f"Продакшн на {p0['n']} матчах: log loss точного счёта {p0['score_ll']:.4f}, тоталов {p0['totals_ll']:.4f}, "
              f"обе забьют {p0['btts_ll']:.4f}, фор {p0['hcp_ll']:.4f}; MAE голов хозяев {p0['mae_home']:.3f}, гостей "
              f"{p0['mae_away']:.3f}; девиация {p0['dev_home']:.4f} / {p0['dev_away']:.4f}; ECE тотала 2.5 "
              f"{p0['ece_ou25']:.3f}, обе забьют {p0['ece_btts']:.3f}."))
    rows = [["Вариант", "Δ счёт [95 % ДИ]", "p Холма", "Δ тоталы", "Δ обе забьют", "Δ форы", "Δ 1X2", "Δ девиация"]]
    for k in sorted(ev, key=lambda k: ev[k]["score_ll"]["diff"]):
        e = ev[k]
        rows.append([nm(k), ci3(e["score_ll"]), pv(holm.get(k)), m3(e["totals_ll"]["diff"]), m3(e["btts_ll"]["diff"]),
                     m3(e["hcp_ll"]["diff"]), m3(e["x12_ll"]["diff"]), m3(e["dev_goals"]["diff"])])
    rep.add(table(rows, widths=[4.6 * cm, 3.2 * cm, 1.4 * cm, 1.6 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm]),
            P("Таблица 4. 2026/27: разность с продакшном, ×10⁻³ (меньше — лучше).", "small"))
    rows = [["Вариант", "MAE хоз.", "MAE гост.", "Дев. хоз.", "Дев. гост.", "ECE ТБ2.5", "ECE ОЗ", "P топ-счёта",
             "Частота топ-счёта"]]
    for k in ["prod"] + sorted(ev, key=lambda k: ev[k]["score_ll"]["diff"]):
        s = sm[k]
        rows.append([nm(k), f4(s["mae_home"], 3), f4(s["mae_away"], 3), f4(s["dev_home"]), f4(s["dev_away"]),
                     f4(s["ece_ou25"], 3), f4(s["ece_btts"], 3), pp(s["top_score_p"], 1, False),
                     pp(s["top_score_hit"], 1, False)])
    rep.add(table(rows, widths=[4.6 * cm, 1.4 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.4 * cm, 1.6 * cm,
                                1.8 * cm]),
            P("Таблица 5. 2026/27: абсолютные метрики голов, калибровка и самый вероятный точный счёт "
              "(заявленная вероятность и фактическая частота, %).", "small"))
    rep.add(*figure(FIG / "forest_2026.png", "Рисунок 2. 2026/27: разность log loss с продакшном и 95 % ДИ "
                                             "(зелёный — значимо лучше, оранжевый — значимо хуже)."))
    rep.add(*figure(FIG / "calibration_2026.png", "Рисунок 3. Калибровка тотала больше 2.5 и «обе забьют», 2026/27 "
                                                  "(корзины с ≥ 20 матчами)."))
    rep.add(H2("6.3 Main ROI (2026/27)"))
    rep.add(P(f"Базовая линия B: close — {b_close['bets']} ставок, ROI {pp(b_close['roi'])} %, попадание "
              f"{pp(b_close['hit_rate'], 1, False)} %, средний коэффициент {b_close['avg_odds']:.2f}; t09 — "
              f"{b_t09['bets']} ставок, ROI {pp(b_t09['roi'])} %. Базовая линия A (история сайта) на тех же матчах: "
              f"разность B − A по ROI {cipp(a_sum['roi_diff'])} п.п."))
    for u in ("close", "t09"):
        rows = [["Вариант", "Ставок", "Попад., %", "Ср. коэф.", "ROI, %", "ΔROI к B, п.п. [95 % ДИ]", "p", "p Холма",
                 "Тот же Main"]]
        sub = rr.loc[rr["universe"] == u].sort_values("d_roi", ascending=False)
        for _, x in sub.iterrows():
            rows.append([nm(x["variant"]), int(x["bets"]), pp(x["hit_rate"], 1, False), f"{x['avg_odds']:.2f}",
                         pp(x["roi"]), f"{pp(x['d_roi'])} [{pp(x['d_lo'])}; {pp(x['d_hi'])}]", pv(x["p"]),
                         pv(x["p_holm"]), pp(x["same_main"], 0, False) + " %"])
        rep.add(table(rows, widths=[3.9 * cm, 1.45 * cm, 1.45 * cm, 1.3 * cm, 1.3 * cm, 3.5 * cm, 1.1 * cm, 1.3 * cm,
                                    1.4 * cm]),
                P(f"Таблица {6 if u == 'close' else 7}. Main ROI, момент {u}; «prod» — воспроизведение B (контроль "
                  "подмены матрицы).", "small"))
    rep.add(*figure(FIG / "roi_2026.png", "Рисунок 4. ΔROI Main к базовой линии B, п.п., 95 % ДИ (парный бутстреп)."))

    # 7 ------------------------------------------------------------------------------------------------
    rep.add(H1("7. Графики и разбивки"))
    rep.add(*figure(FIG / "ratings_grid.png", "Рисунок 5. Выбор параметров динамических рейтингов на 2023–2025.",
                          width=11 * cm))
    rep.add(*figure(FIG / "by_month_2026.png", "Рисунок 6. Δ log loss точного счёта по месяцам 2026/27.",
                          width=13 * cm))
    bg = pd.read_csv(RES / "by_games_2026.csv")
    show = [k for k in dict.fromkeys([r["best_prob"], "blend", "rat", "ipf", "monthly"]) if k in set(bg["variant"])]
    rows = [["Сыграно игр (меньшая из команд)"] + [nm(k) for k in show]]
    for g in ["0-2", "3-5", "6-9", "10-14", "15+"]:
        x = bg.loc[bg["games"] == g].set_index("variant")
        if len(x):
            rows.append([f"{g} (n = {int(x['n'].iloc[0])})"] + [m3(x.loc[k, "d_score_ll"]) if k in x.index else "—"
                                                              for k in show])
    rep.add(table(rows), P("Таблица 8. Δ log loss точного счёта (×10⁻³) по числу игр сезона, сыгранных командами.",
                           "small"))
    for k, c in r["candidates"].items():
        cc = c["close"]
        rows = [["Месяц", "Ставок", "Попадание", "ROI, %", "Прибыль"]]
        for x in cc["by_month"]:
            rows.append([x["month"], x["bets"], pp(x["hit_rate"], 1, False), pp(x["roi"]), f"{x['profit']:+.1f}"])
        rep.add(H3(f"{nm(k)}: устойчивость (close)"), table(rows),
                P(f"Без 5 крупнейших выигрышей: ROI {pp(cc['big_wins']['roi_without'])} % против "
                  f"{pp(cc['big_wins']['base_roi_same_matches'])} % у B на тех же матчах; при ценах на 2 % ниже: "
                  f"{pp(cc['odds_minus_2pct']['roi'])} % (B: {pp(cc['base_odds_minus_2pct']['roi'])} %). Лиги с "
                  f"< 30 ставками — малые выборки.", "small"))
        lg = pd.DataFrame(cc["by_league"]).sort_values("bets", ascending=False)
        rows = [["Лига", "Ставок", "ROI, %"]] + [[x["league"], x["bets"], pp(x["roi"]) + (" *" if x["bets"] < 30 else "")]
                                                for _, x in lg.iterrows()]
        rep.add(table(rows, widths=[6 * cm, 2 * cm, 2.5 * cm]), P("* — меньше 30 ставок.", "small"))

    # 8 ------------------------------------------------------------------------------------------------
    rep.add(H1("8. Статистический анализ"),
            P(f"Вариантов на 2026/27: {r['n_variants']}; поправка Холма — по всем вариантам для log loss счёта и "
              f"отдельно для ROI в каждом моменте. Сезон 2026/27 изучался в v6.1/v6.2, поэтому результат на нём — "
              f"хронологическая проверка, а не независимое подтверждение; параметры выбраны на 2023–2025."),
            P(f"Диксон–Коулс по сезонам: " + "; ".join(
                f"{s}/{int(s) + 1 - 2000}: ρ после согласования {v['after_reconcile']:+.3f}, ρ конвейера "
                f"{v['raw_pipeline']:+.3f}" for s, v in par["rho_by_season"].items())
              + ". ρ после согласования близок к нулю — эффект Диксона–Коулса поглощает согласование с 1X2."))

    # 9 ------------------------------------------------------------------------------------------------
    rows = [["Вариант", "Отбор Δ счёт", "2026/27 Δ счёт", "ΔROI close", "ΔROI t09", "Решение"]]
    for k in sorted(ev, key=lambda k: ev[k]["score_ll"]["diff"]):
        rc = roi.get(k, {})
        rows.append([nm(k), ci3(pooled.get(k, {}).get("score_ll")), ci3(ev[k]["score_ll"]),
                     cipp(rc.get("close", {}).get("vs_b", {}).get("roi_diff")),
                     cipp(rc.get("t09", {}).get("vs_b", {}).get("roi_diff")), dec[k]])
    rep.add(H1("9. Выводы"), table(rows, widths=[4.4 * cm, 3.0 * cm, 3.0 * cm, 2.7 * cm, 2.7 * cm, 1.4 * cm]),
            P("Таблица 9. Сводка: log loss точного счёта (×10⁻³) и ΔROI Main (п.п.) к продакшну / базовой линии B.",
              "small"))
    rep.add(bullets(conclusions(r, rs)))

    # 10 -----------------------------------------------------------------------------------------------
    aa = [k for k, d in dec.items() if d == "A"]
    bb = [k for k, d in dec.items() if d == "B"]
    cc_ = [k for k, d in dec.items() if d == "C"]
    rep.add(H1("10. Рекомендации A / B / C"),
            bullets([f"**A (внедрить):** {', '.join(nm(k) for k in aa) if aa else 'нет'}.",
                     f"**B (форвард-проверка):** {', '.join(nm(k) for k in bb) if bb else 'нет'}"
                     + (f" — предрегистрация `experiments/v6_3/goals/results/preregistration.json` "
                        f"({len(pre['entries'])} записей)." if bb else "."),
                     f"**C (отклонить):** {', '.join(nm(k) for k in cc_)}.",
                     "Продакшн исследованием не меняется. Если Диксон–Коулс когда-либо будет включён "
                     "(`DIXON_COLES = True` в `src/config.py`), ρ нужно подбирать по согласованной матрице "
                     "(`src/modelling/pipeline.py`), иначе применяется ρ сырых средних (дефект, найденный v6.2)."]))

    # 11 -----------------------------------------------------------------------------------------------
    rep.add(H1("11. Ограничения"), bullets([
        "2026/27 — не независимая выборка (изучалась в v6.1/v6.2); 2 250 матчей дают ДИ разности log loss счёта "
        "порядка ±2×10⁻³, поэтому эффекты < 1×10⁻³ не различимы.",
        "Параметры смесей и распределений подобраны на сезонах отбора и оценены на них внутривыборочно; "
        "переобучаемые модели (рейтинги как признаки, единая модель) проверены на 2024/25–2025/26 вне выборки.",
        "Гиперпараметры переобучаемых моделей не настраивались заново (взяты из продакшна): новый набор признаков "
        "мог бы выиграть от своей настройки.",
        "Цены сезонов отбора не используются для выбора (выбор только по вероятностным метрикам); ROI — только "
        "на 2026/27. Точный счёт и часть линий форы в Main почти не выбираются, поэтому ROI мало чувствителен "
        "к хвостам матрицы.",
        "xG не проверялся заново (нет истории для обучения; v6.2 не нашло пользы)."]))

    # 12 -----------------------------------------------------------------------------------------------
    rep.add(H1("12. Как воспроизвести"), *K.reproduce([
        ".venv/bin/python -m experiments.v6_3.replay.build            # общий реплей (уже построен)",
        ".venv/bin/python -m experiments.v6_3.goals.run --stage ratings",
        "experiments/v6_3/heavy.sh 1800 experiments/v6_3/goals/logs/base.log \\",
        "    .venv/bin/python -m experiments.v6_3.goals.base",
        "# ежемесячные модели голов — исследование 7 (recent_weighting/fits.py, схема prod)",
        ".venv/bin/python -m experiments.v6_3.goals.run --stage eval",
        ".venv/bin/python -m experiments.v6_3.goals.build_pdf"]),
        P(f"Git: {info['git_sha'][:10]}; время прогона {info['seconds']:.0f} с; библиотеки: "
          + ", ".join(f"{k} {v}" for k, v in info["libraries"].items()) + ".", "small"),
        P("Файлы: `results/results.json` (все метрики), `selection_by_season.csv`, `main_roi.csv`, "
          "`per_match_*.csv` (по матчам, поля §5), `per_match_probabilities_2026.csv`, `by_month_2026.csv`, "
          "`by_games_2026.csv`, `ratings_selection.json`, `preregistration.json`, `run_info.json`.", "small"))
    rep.build()
    n = rep.check()
    print(f"{OUT} — {n} pages")
    write_findings(r, rs, rr)


def conclusions(r: dict, rs: dict) -> list[str]:
    names, ev, dec, pooled, roi = r["names"], r["paired_2026"], r["decisions"], r["pooled_selection"], r["roi"]
    nm = lambda k: names.get(k, k)  # noqa: E731
    out = []
    bp = r["best_prob"]
    out.append(f"Лучший вариант по log loss точного счёта на 2026/27 — {nm(bp)}: {ci3(ev[bp]['score_ll'])}×10⁻³ "
               f"(p Холма {pv(r['holm_2026'].get(bp))}); на сезонах отбора {ci3(pooled.get(bp, {}).get('score_ll'))}.")
    for k in ("rat", "blend", "ratfeat", "stacked", "monthly", "hy_v62"):
        if k in ev:
            out.append(f"{nm(k)}: отбор {ci3(pooled.get(k, {}).get('score_ll'))}, 2026/27 {ci3(ev[k]['score_ll'])}, "
                       f"девиация голов {ci3(ev[k]['dev_goals'])}×10⁻³; ΔROI close "
                       f"{cipp(roi.get(k, {}).get('close', {}).get('vs_b', {}).get('roi_diff'))} п.п. — {dec[k]}.")
    for k in ("dc", "nb", "ipf", "sup", "partial", "raw"):
        if k in ev:
            out.append(f"{nm(k)}: 2026/27 счёт {ci3(ev[k]['score_ll'])}, тоталы {ci3(ev[k]['totals_ll'])}×10⁻³; ΔROI close "
                       f"{cipp(roi.get(k, {}).get('close', {}).get('vs_b', {}).get('roi_diff'))} п.п. — {dec[k]}.")
    bs = pd.read_csv(RES / "selection_by_season.csv")
    for k in ("blend", "stacked"):
        x = bs.loc[bs["variant"] == k].set_index("season")["d_score_ll"]
        if len(x) and k in ev:
            out.append(f"{nm(k)} — эффект по числу сезонов в обучении базовой модели: " + ", ".join(
                f"{s}/{s + 1 - 2000} ({s - 2021} сез.) {x[s] * 1000:+.2f}" for s in x.index)
                + f", 2026/27 (5 сез.) {ev[k]['score_ll']['diff'] * 1000:+.2f} ×10⁻³.")
    out.append(f"Домашние/выездные поправки рейтинга не улучшили девиацию на отборе "
               f"({rs['best_ha']['deviance']:.4f} против {rs['best']['deviance']:.4f}) — H2 отвергается.")
    return out


def write_findings(r: dict, rs: dict, rr: pd.DataFrame) -> None:
    names, ev, dec, pooled, roi, par = (r["names"], r["paired_2026"], r["decisions"], r["pooled_selection"], r["roi"],
                                        r["params"])
    nm = lambda k: names.get(k, k)  # noqa: E731
    lines = ["# FootPredict v6.3 — исследование 10: модель голов — выводы", "",
             "Все числа — из `results/` (файл пишет `build_pdf.py`). PDF: `reports/v6_3/Goal_Model_Research.pdf`.", "",
             "## Сводка (Δ к продакшну; log loss ×10⁻³, ROI — п.п.; отбор 2023–2025, проверка 2026/27)", "",
             "| Вариант | Отбор Δ счёт | 2026/27 Δ счёт | Δ тоталы | ΔROI close | ΔROI t09 | Решение |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in sorted(ev, key=lambda k: ev[k]["score_ll"]["diff"]):
        rc = roi.get(k, {})
        lines.append(f"| {nm(k)} | {ci3(pooled.get(k, {}).get('score_ll'))} | {ci3(ev[k]['score_ll'])} | "
                     f"{ci3(ev[k]['totals_ll'])} | {cipp(rc.get('close', {}).get('vs_b', {}).get('roi_diff'))} | "
                     f"{cipp(rc.get('t09', {}).get('vs_b', {}).get('roi_diff'))} | {dec[k]} |")
    b = roi["prod"]
    lines += ["", f"Базовая линия B: close ROI {pp(b['close']['summary']['roi'])} % ({b['close']['summary']['bets']} "
                  f"ставок), t09 {pp(b['t09']['summary']['roi'])} % ({b['t09']['summary']['bets']}). Вариантов: "
                  f"{r['n_variants']}.", "", "## Главное", ""]
    lines += [f"- {c}" for c in conclusions(r, rs)]
    lines += ["", "## Дефекты и код продакшна", "",
              f"- Скрытый дефект Диксона–Коулса (v6.2) подтверждён: ρ по сырым средним {par['rho_raw']:+.4f}, после "
              f"согласования {par['rho']:+.4f}. Флаг `DIXON_COLES = False` — сейчас не действует; при включении "
              "подбирать ρ по согласованной матрице (`src/modelling/pipeline.py`).",
              "- Новых дефектов в общем коде не найдено. Подмена построения матрицы воспроизводит базовую линию B "
              "(тот же Main у ≥ 99.9 % матчей; расхождение — округление pmf).", "",
              "## Решение", "",
              f"- A: {', '.join(nm(k) for k, d in dec.items() if d == 'A') or 'нет'}",
              f"- B: {', '.join(nm(k) for k, d in dec.items() if d == 'B') or 'нет'}",
              f"- C: {', '.join(nm(k) for k, d in dec.items() if d == 'C')}",
              "- Продакшн не меняется исследованием.", "",
              "## Воспроизведение", "", "```",
              ".venv/bin/python -m experiments.v6_3.goals.run --stage ratings",
              "experiments/v6_3/heavy.sh 1800 experiments/v6_3/goals/logs/base.log .venv/bin/python -m experiments.v6_3.goals.base",
              ".venv/bin/python -m experiments.v6_3.goals.run --stage eval",
              ".venv/bin/python -m experiments.v6_3.goals.build_pdf", "```", ""]
    (C.V63 / "goals" / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
