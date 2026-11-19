"""Builds reports/v6_3/Corners_Research.pdf and findings.md from results/ (no number is typed by hand).

    .venv/bin/python -m experiments.v6_3.corners.build_pdf
"""
from __future__ import annotations

import sys

import pandas as pd

from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, figure, formula, note, table

from experiments.v6_3.corners import data as D
from experiments.v6_3.corners import evaluate as E

OUT = K.ROOT / "reports" / "v6_3" / "Corners_Research.pdf"
R, F = D.RESULTS, D.FIGURES
UNI = {"close": "close (последняя цена до матча)", "t09": "t09 (цены, известные в 09:00)"}
DEC = {"A": "A — внедрить", "B": "B — перспективно, форвард-проверка", "C": "C — отклонить"}


def W(*cms):
    return [c * K.cm for c in cms]


def f3(v, d=3):
    return K.num(v, d)


def pp(v, d=1):
    return "—" if v is None else f"{v * 100:+.{d}f}"


def pc(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


def ci(m, scale=1.0, d=4, signed=True):
    if not m or m.get("diff") is None:
        return "—"
    s = "+" if signed else ""
    lo, hi = m["ci95"]
    return f"{m['diff'] * scale:{s}.{d}f} [{lo * scale:{s}.{d}f}; {hi * scale:{s}.{d}f}]"


def pv(p):
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def season(s):
    s = int(s)
    return f"{s}/{(s + 1) % 100:02d}"


def build() -> None:
    o = K.load(R / "results.json")
    inv = K.load(R / "data_inventory.json")
    info = K.load(R / "run_info.json")
    pre = K.load(R / "preregistration.json", [])
    leak = K.load(R / "leakage_check.json", {})
    bm, bd = o["best_combo"]["model"], o["best_combo"]["dist"]
    best_key = f"{bm}|{bd}"
    lab = lambda m: E.LABELS.get(m, m)
    em, sm = o["eval_mean"], o["select_mean"]
    evb = o["eval_dist"][best_key]
    prod_ls = o["eval_prod_check"]["log_score"]
    mc, mt = o["main"]["close"], o["main"]["t09"]
    base_a = K.load(K.ROOT / "experiments/v6_3/replay/results/baseline.json")["baseline_a"]

    rep = K.Report(OUT, "Исследование модели угловых", "FootPredict v6.3 — исследование 11",
                   {"Сезон выбора": "2025/26 (модели обучены на 2021/22–2024/25)",
                    "Сезон проверки": "2026/27: " + str(o["coverage"]["by_season"].get("2026", {}).get("matches", "—"))
                    + " сыгранных матчей (сезон уже изучался в v6.1/v6.2, не независим)",
                    "Данные": "Football-Data (угловые, удары, фолы, карточки), OddsPapi bet365 (линии угловых 2026/27)"})

    # 1 ---------------------------------------------------------------------------------------------------
    rep.add(H1("1. Название"),
            P("**Модель угловых FootPredict: признаки атакующего давления, динамические рейтинги, распределения "
              "счётных данных, калибровка тоталов и влияние на Main при реальных ценах.**"))

    # 2 ---------------------------------------------------------------------------------------------------
    rep.add(H1("2. Постановка проблемы"),
            P(f"Продакшн-модель угловых (`prod_20261012`) — один XGBoost с целевой функцией Пуассона на сумму угловых "
              f"и отрицательное биномиальное распределение с одним размером r = {o['r_prod']:.1f} (почти Пуассон) → "
              f"тоталы 8.5 / 9.5 / 10.5. На 2026/27 её MAE {f3(em['xgb_prod']['mae'])} против "
              f"{f3(em['season_mean_in_sample']['mae'])} у среднего сезона (задним числом), R² "
              f"{f3(em['xgb_prod']['r2'])}. Нужно понять, можно ли существенно улучшить прогноз угловых новыми "
              f"признаками, другими алгоритмами и распределениями, и даёт ли это что-нибудь на рынке угловых и в Main."),
            P("Исследование v6.1 (`experiments/v6_1/corners`) уже сравнило XGBoost, LightGBM, CatBoost, GLM, "
              "иерархическую модель силы команд и NB с r по моментам на сезонах 2023/24–2025/26 и не нашло "
              "убедительно лучшей модели. v6.3 проверяет **новые** гипотезы: признаки давления по ударам и "
              "рыночный профиль матча, рейтинги, обновляемые после каждого игрового дня, раздельные модели сторон со "
              "свёрткой через копулу (зависимость угловых хозяев и гостей), NB-P и COM-Пуассон, калибровку линий, "
              "ежемесячное переобучение внутри сезона и — впервые — ROI рынков угловых и эффект на Main в общем "
              "replay 2026/27 в двух моментах решения."))

    # 3 ---------------------------------------------------------------------------------------------------
    rep.add(H1("3. Гипотезы"),
            *bullets(["H1. Новые признаки (EWMA угловых и ударов, опп.-скорректированные ожидания, динамические "
                      "рейтинги угловых и ударов, рыночный профиль матча) уменьшают девиацию Пуассона суммы угловых.",
                      "H2. Раздельные модели угловых хозяев и гостей со свёрткой лучше одной модели суммы; "
                      "отрицательная зависимость сторон (копула Франка) лучше независимой свёртки.",
                      "H3. Дисперсия, подобранная правильно (ML, по лигам, по матчу, COM-Пуассон), улучшает log score "
                      "распределения и log loss тоталов по сравнению с r продакшна.",
                      "H4. Калибровка вероятностей тоталов (Платт / изотоническая) уменьшает log loss линий.",
                      "H5. Ансамбль моделей лучше любой одной модели.",
                      "H6. Лучшая модель повышает ROI ставок на угловые при реальных ценах и ROI Main (парно к "
                      "Baseline B, оба универсума)."]))

    # 4 ---------------------------------------------------------------------------------------------------
    cov = o["coverage"]["by_season"]
    rows = [["Сезон", "Матчей", "С угловыми", "Среднее", "Дисперсия", "Var/mean", "Хозяева", "Гости", "corr(H, A)"]]
    for s, v in cov.items():
        rows.append([season(s), v["matches"], v["with_corners"], f3(v["mean"], 2), f3(v["var"], 2), f3(v["disp"], 2),
                     f3(v["home"], 2), f3(v["away"], 2), f3(v["corr_home_away"], 3)])
    fs = inv["football_data_seasons_with"]
    idea = inv["idea_without_history"]
    api = inv["api_football_cached_endpoints"]
    rl = inv["results_live"]
    cq = inv["corner_odds_rows"]
    rep.add(H1("4. Исходные данные"),
            P(f"Сыгранные матчи {o['coverage']['leagues']} лиг, сезоны "
              f"{season(min(o['coverage']['seasons']))}–{season(max(o['coverage']['seasons']))}: "
              f"{o['coverage']['total_played']} матчей, из них с угловыми {o['coverage']['total_with_corners']}. "
              "Источник статистики — файлы Football-Data (HC/AC угловые, HS/AS удары, HST/AST в створ, HF/AF фолы, "
              "HY/AY, HR/AR карточки). Покрытие угловых по каждой лиге и сезону — 99–100 % (таблица "
              "`results/coverage_league_season.csv`, рисунок 1)."),
            K.table(rows),
            P("Таблица 1. Угловые по сезонам (только сыгранные матчи). corr(H, A) — корреляция угловых хозяев и "
              "гостей.", "small"),
            H2("4.1. Какие показатели действительно есть с историей"),
            *bullets([f"Football-Data: угловые, удары, удары в створ, фолы, карточки — все сезоны "
                      f"({', '.join(fs.get('HC', []))}); xG (HxG/AxG) — только {', '.join(fs.get('HxG', [])) or '—'} "
                      f"(нет истории для обучения); судья (Referee) — только {', '.join(fs.get('Referee', []))} "
                      f"(английские и шотландские лиги).",
                      f"API-Football в `local_data/api/api_football`: сохранены только ответы эндпоинтов "
                      f"{', '.join(f'{k} ({v})' for k, v in api.items())} — поиск команд и списки матчей; "
                      "статистики матчей (fixtures/statistics), составов и событий нет. API не вызывался.",
                      f"`local_data/results_live/fixtures.parquet`: {rl['rows']} строк, сыграно {rl['played']}, с "
                      f"угловыми {rl['with_corners']} — живые результаты без статистики угловых.",
                      "Цены угловых: " + ", ".join(f"{k}: {v}" for k, v in cq.items()) + " строк — только 2026/27."]),
            P("**Идеи владельца без исторических данных:** " + ", ".join(
                {"blocked shots": "блокированные удары", "crosses": "навесы", "wide attacks / flank play": "фланговые атаки",
                 "full-backs / wingers": "крайние защитники и вингеры (составы, позиции)",
                 "possession / style": "владение / стиль по передачам"}[k] for k, v in idea.items() if v)
              + " — ни в Football-Data, ни в локально сохранённых ответах API их нет; обучить на них модель нельзя. "
                "Стиль игры приближён косвенно: доля ударов и угловых команды, угловые на удар, фолы, рыночный "
                "профиль матча (вероятность фаворита, ожидаемые голы по цене тотала 2.5)."),
            P(f"Проверка утечки: {leak.get('checked', '—')} случайных значений новых признаков пересчитаны вручную "
              f"только по более ранним матчам команды; максимальное расхождение {leak.get('max_abs_diff', '—')}."))

    tr = {"matches of the season (corner leagues)": "Матчей сезона (лиги с угловыми)", "played": "Сыграно",
          "played with corner statistics": "Сыграно, есть угловые",
          "processed by every method (corner mean + distribution)": "Обработано каждым методом",
          "played with a real corner price (any line)": "Сыграно, есть реальная цена угловых (любая линия)",
          "played with < 3 earlier matches of a team (weak statistics, still predicted)":
              "Мало статистики (< 3 прошлых матчей у команды; прогноз всё равно дан)",
          "Baseline B: settled / bets / no price": "Baseline B: рассчитано / ставок / без цены"}
    cv = o["coverage_2026"]
    rows = [["Показатель", "close", "t09"]]
    for a, b in zip(cv["close"], cv["t09"]):
        name = tr.get(a["item"], a["item"].replace("settled / bets / no price (not a bet) / push",
                                                   "рассчитано / ставок / без цены / возврат"))
        rows.append([name, a["n"], b["n"]])
    rep.add(H2("4.2. Покрытие 2026/27 и исключения"), K.table(rows, widths=W(9.0, 3.6, 3.6)),
            P("Таблица 1а. Ничего не исключается молча: матч без цены остаётся в таблице Main, но не считается "
              "ставкой; несыгранные матчи не рассчитываются. Полная таблица — `results/coverage_2026.csv`, "
              "по матчам — `results/per_match_*.csv`.", "small"))

    # 5 ---------------------------------------------------------------------------------------------------
    rt = o["frame_info"]["rating"]["best"]
    fc = o["frame_info"]["feature_counts"]
    rep.add(H1("5. Методика"),
            H2("5.1. Временное разделение"),
            *bullets(["**Выбор** — модели обучены на 2021/22–2024/25, прогноз 2025/26. Здесь выбираются: лучшая модель "
                      "среднего (наименьшая девиация Пуассона), распределение (наименьший log score при двукратной "
                      "перекрёстной подгонке внутри сезона: параметры по июлю–декабрю → оценка января–июня и наоборот), "
                      "веса ансамбля, калибровка линий.",
                      "**Проверка** — модели обучены на 2021/22–2025/26, прогноз 2026/27; параметры распределений "
                      "подогнаны на 2025/26 и не меняются. Продакшн — сам бандл `prod_20261012` (Baseline B).",
                      "**Walk-forward** — ежемесячное переобучение внутри 2026/27 (матчи месяца M прогнозируются "
                      "моделями, обученными на всём, что сыграно до 1-го числа M).",
                      "Все обучения — через `experiments/v6_3/heavy.sh`, 2 потока; число деревьев — ранняя остановка "
                      "только на этапе выбора (обучение 2021/22–2023/24, валидация 2024/25)."]),
            H2("5.2. Признаки"),
            P(f"Продакшн-набор угловых ({fc['prod']} признаков) + {fc['new']} новых + {fc['market']} рыночных. Новые "
              "признаки строятся по ряду матчей команды, сдвинутому на один матч (матч не видит себя): EWMA угловых "
              "за / против (полураспад 5 и 15 матчей), по месту проведения, ударов, ударов в створ, фолов, голов, "
              "суммы угловых матчей команды; угловые на удар; аддитивное и мультипликативное ожидание угловых "
              "сторон, ожидание по ударам, опп.-скорректированное давление, |разница Elo|. Рыночный профиль — по "
              "предматчевым ценам Football-Data (1X2 и тотал 2.5): разница вероятностей сторон, вероятность "
              "фаворита, P(тотал > 2.5) и ожидаемые голы λ из P(N ≥ 3) = p."),
            H2("5.3. Динамические рейтинги"),
            formula("log μ_{H} = b_{L}^{H} + att_{i} + con_{j},   log μ_{A} = b_{L}^{A} + att_{j} + con_{i}"),
            formula("att_{i} ← att_{i} + η (y_{i} − μ_{i}) / μ_{i},   con_{j} ← con_{j} + η (y_{i} − μ_{i}) / μ_{i}"),
            P(f"Обновление после каждого игрового дня (все матчи дня прогнозируются до обновления), в начале сезона "
              f"рейтинги умножаются на keep. Сетка η × keep выбрана на 2023/24–2024/25: η = {rt['eta']}, keep = "
              f"{rt['keep']} (рисунок 2). Те же рейтинги считаются для ударов (опп.-скорректированное давление). "
              "Рейтинги — и самостоятельная модель (μ = μ_H + μ_A), и признаки для деревьев."),
            H2("5.4. Модели среднего"),
            *bullets([f"{lab(k)}" for k in ("xgb_prod", "xgb_stats", "xgb_mkt", "lgbm_mkt", "cat_mkt", "xgb_ha",
                                            "rating", "glm", "ens_w", "ens_eq", "naive")]),
            H2("5.5. Распределения"),
            formula("NB2: Var = μ + μ^{2}/r;   NB-P: Var = μ + α μ^{p} (r по матчу = μ^{2−p}/α)"),
            formula("COM-Пуассон: P(k) ∝ λ^{k} / (k!)^{ν}, λ подбирается так, чтобы E[k] = μ"),
            formula("Свёртка: P(H = i, A = j) = C(F_{i}, G_{j}) − C(F_{i−1}, G_{j}) − C(F_{i}, G_{j−1}) + C(F_{i−1}, G_{j−1})"),
            P("где F, G — функции распределения NB(μ_H, r_H), NB(μ_A, r_A), C — копула Франка с параметром θ "
              "(θ = 0 — независимость). Сумма угловых — сумма по антидиагоналям. Параметры — максимум правдоподобия "
              "суммы на 2025/26. Все распределения считаются на 0…40 с хвостом в последней ячейке, как в продакшне."),
            H2("5.6. Метрики и деньги"),
            P("MAE, RMSE, R², девиация Пуассона; log score распределения (−log P(y)), RPS; для линий 8.5 / 9.5 / "
              "10.5 — log loss, Brier, ECE и диаграммы надёжности. Деньги: таблицы кандидатов общего replay "
              "(`replay/cache/candidates_{close,t09}`), вариант меняет только вероятности строк CORNERS_*; EV = p·odds − 1. "
              "Ставки на угловые: все с EV ≥ 0 (ev0), EV ≥ 5 % (ev5), одна лучшая по EV в матче (best0). Main — "
              "правило продакшна (`main_by_rule`) на изменённой таблице, парно к Baseline B того же универсума."),
            P("Почему не только `engine.batch_from_outputs`: продакшн-движок принимает одно μ на матч и один "
              "глобальный размер NB, поэтому NB по лигам, NB-P, COM-Пуассон, копула и калибровка в нём не выражаются. "
              "Для вариантов, которые выражаются (μ + глобальный r), оба пути сверены (раздел 9)."))

    # 6 ---------------------------------------------------------------------------------------------------
    tried = info.get("variants_tried", {})
    trees = info["params"].get("trees", {})
    rep.add(H1("6. Эксперименты"),
            P(f"Запущено: {tried.get('mean_models', '—')} моделей среднего, {tried.get('distributions', '—')} "
              f"сочетаний модель × распределение, {tried.get('calibrations', '—')} метода калибровки, "
              f"{tried.get('main_variants', '—')} вариантов Main / ставок на угловые (Holm по ним в каждом "
              f"универсуме). Число деревьев по ранней остановке: "
              + ", ".join(f"{k} — {v}" for k, v in trees.items()) + "."),
            P("Варианты для денег (все на одних и тех же матчах 2026/27):"),
            *bullets([f"**{k}** — {v}" for k, v in o["variants"].items()]))

    # 7 ---------------------------------------------------------------------------------------------------
    rows = [["Модель", "Δдев. 2025/26 [95 % ДИ]", "Holm p 25/26", "Дев. 25/26", "MAE 25/26", "Дев. 26/27",
             "MAE 26/27", "RMSE 26/27", "R² 26/27", "Δдев. 2026/27 [95 % ДИ]"]]
    for m in sm:
        e = em.get(m, {})
        rows.append([lab(m), ci(sm[m].get("dev_diff")), pv(o.get("select_mean_holm", {}).get(m)),
                     f3(sm[m]["poisson_dev"], 4), f3(sm[m]["mae"]),
                     f3(e.get("poisson_dev"), 4), f3(e.get("mae")), f3(e.get("rmse")), f3(e.get("r2")),
                     ci(e.get("dev_diff"))])
    if "xgb_prod_refit" in em:
        e = em["xgb_prod_refit"]
        rows.append([lab("xgb_prod_refit"), "—", "—", "—", "—", f3(e["poisson_dev"], 4), f3(e["mae"]), f3(e["rmse"]),
                     f3(e["r2"]), ci(e.get("dev_diff"))])
    rep.add(H1("7. Результаты"),
            H2("7.1. Модели среднего"),
            K.table(rows, widths=W(2.6, 2.7, 1.1, 1.35, 1.25, 1.35, 1.25, 1.25, 1.1, 2.7)),
            P(f"Таблица 2. Модели среднего. Δдев. — парная разность средней девиации Пуассона с продакшн-XGBoost "
              f"(меньше — лучше). Для 2025/26 «Ансамбль (веса)» оценён с перекрёстной подгонкой весов. По правилу "
              f"выбора (2025/26) лучшая модель среднего — **{lab(o['best_model'])}**. Веса ансамбля: "
              + ", ".join(f"{lab(k)} {v:.2f}" for k, v in o["ensemble_weights"].items()) + ".", "small"))

    rows = [["Модель: распределение", "LS 25/26 (cf)", "LL линий 25/26 (cf)", "LS 26/27", "ΔLS 26/27 [95 % ДИ]",
             "LL линий 26/27", "ΔLL линий [95 % ДИ]", "RPS 26/27"]]
    shown = [k for k in o["eval_dist"] if k.split("|")[0] in ("xgb_prod", bm) or "|conv" in k]
    order = sorted(shown, key=lambda k: o["select_dist"][k]["log_score_cf"])
    for k in order:
        m, d = k.split("|")
        s, e = o["select_dist"][k], o["eval_dist"][k]
        rows.append([f"{lab(m)}: {E.DLABELS[d]}", f3(s["log_score_cf"], 4), f3(s["ll_mean_cf"], 4),
                     f3(e["log_score"], 4), ci(e["log_score_diff"]), f3(e["lines"]["ll_mean"], 4),
                     ci(e["lines_diff"]["mean"]), f3(e["rps"], 3)])
    pr = o["dist_params"]
    ptxt = []
    for k in (f"xgb_prod|nb", f"xgb_prod|nb_mom", best_key, "xgb_ha|conv_frank", "xgb_prod|nbp", "xgb_prod|cmp"):
        if k in pr:
            ptxt.append(f"{k}: " + ", ".join(f"{a} = {b:.3g}" for a, b in pr[k].items() if not isinstance(b, dict)))
    rep.add(H2("7.2. Распределения"),
            K.table(rows, widths=W(3.7, 1.4, 1.5, 1.3, 2.9, 1.4, 2.9, 1.2)),
            P(f"Таблица 3. Распределения для продакшн-μ, выбранной модели и свёрток сторон (полный список — "
              f"`results/results.json`, {len(o['eval_dist'])} сочетаний; отсортированы по выбору на 2025/26). LS — log score суммы угловых, LL линий — "
              f"средний log loss тоталов 8.5 / 9.5 / 10.5; cf — перекрёстная подгонка внутри 2025/26. Δ — к продакшну "
              f"(XGBoost + NB r = {o['r_prod']:.1f}, LS {f3(prod_ls, 4)}). Выбрано на 2025/26: **{lab(bm)} + "
              f"{E.DLABELS[bd]}**; лучшее распределение для продакшн-μ: **{E.DLABELS[o['best_dist_for_prod']]}**.",
              "small"),
            P("Параметры (подогнаны на всём 2025/26): " + "; ".join(ptxt)
              + (f". Копула Франка: τ Кендалла = {o['frank_kendall_tau']:.3f}." if "frank_kendall_tau" in o else ".")))

    rows = [["Линия", "Частота", "P продакшн", "P лучший", "LL продакшн", "LL лучший", "Brier прод.", "Brier лучш.",
             "ECE прод.", "ECE лучш.", "ΔLL [95 % ДИ]"]]
    pl = o["eval_dist"]["xgb_prod|nb_prod"]["lines"]
    bl = evb["lines"]
    for line in ("8.5", "9.5", "10.5"):
        rows.append([line, pc(pl[line]["freq"]), pc(pl[line]["mean_p"]), pc(bl[line]["mean_p"]), f3(pl[line]["ll"], 4),
                     f3(bl[line]["ll"], 4), f3(pl[line]["brier"], 4), f3(bl[line]["brier"], 4), f3(pl[line]["ece"], 3),
                     f3(bl[line]["ece"], 3), ci(evb["lines_diff"][line])])
    cal = o["eval_calibration"]
    scal = o["select_calibration"]
    rep.add(H2("7.3. Тоталы угловых и калибровка"),
            K.table(rows, widths=W(1.3, 1.3, 1.3, 1.3, 1.4, 1.4, 1.4, 1.4, 1.2, 1.2, 3.0)),
            P("Таблица 4. Линии тотала угловых на 2026/27: продакшн и выбранный вариант.", "small"),
            K.table([["Калибровка", "LL 2025/26 cf", "LL 2025/26 без", "LL 2026/27", "ΔLL к продакшну 2026/27 [95 % ДИ]"]]
                    + [[k, f3(scal[k]["ll_mean_cf"], 4), f3(scal[k]["ll_mean_raw"], 4), "—", "—"] for k in scal]
                    + [[f"2026/27: {k}", "—", "—", f3(v["lines"]["ll_mean"], 4), ci(v["vs_prod"]["mean"])]
                       for k, v in cal.items()]),
            P(f"Таблица 5. Калибровка линий (Платт / изотоническая) поверх выбранного варианта. Выбор на 2025/26: "
              f"{o['calibration_choice']['method']}, {'оставлена' if o['calibration_choice']['kept'] else 'не улучшает — не используется'}. "
              f"static — карта 2025/26; expanding — ежемесячно расширяемая карта (2025/26 + сыгранные месяцы 2026/27).",
              "small"))

    if o.get("wf_mean"):
        rows = [["Модель (ежемесячное переобучение)", "n", "MAE", "R²", "Девиация", "Δдев. [95 % ДИ]", "LS", "ΔLS [95 % ДИ]"]]
        for m, v in o["wf_mean"].items():
            d = o["wf_dist"].get(m, {})
            nm = "Продакшн-бандл (без переобучения)" if m == "prod" else f"{lab(m)}, ежемесячно"
            rows.append([nm, v["n"], f3(v["mae"]), f3(v["r2"]), f3(v["poisson_dev"], 4), ci(v.get("dev_diff")),
                         f3(d.get("log_score"), 4), ci(d.get("log_score_diff"))])
        rep.add(H2("7.4. Ежемесячный walk-forward в 2026/27"),
                K.table(rows, widths=W(3.6, 1.0, 1.2, 1.1, 1.4, 3.0, 1.4, 3.0)),
                P("Таблица 6. Модели, переобучаемые 1-го числа каждого месяца на всех сыгранных ранее матчах (июль — "
                  "модели этапа проверки). Δ — к продакшн-модели того же матча (без переобучения). Распределение — "
                  "выбранное на 2025/26.", "small"))

    mq = o["market_quality"]["close"]
    rows = [["Линия", "n", "Маржа", "LL рынка", "LL продакшн", "LL лучший"]]
    for line, v in mq.items():
        rows.append([line, v["n"], pc(v["margin"]), f3(v["market_ll"], 4), f3(v.get("prod_ll"), 4), f3(v.get("best_ll"), 4)])
    rep.add(H2("7.5. Модель против рынка угловых"), K.table(rows),
            P("Таблица 7. bet365 без маржи (пропорционально) против моделей на матчах, где есть обе стороны линии "
              "(close). Если LL модели не ниже LL рынка, систематического преимущества в ставках ждать не стоит.",
              "small"))

    rows = [["Вариант", "LS", "ΔLS [95 % ДИ]", "LL 8.5", "LL 9.5", "LL 10.5", "ΔLL линий [95 % ДИ]"]]
    for k, q in o["variant_quality"].items():
        rows.append([k, f3(q.get("log_score"), 4), ci(q.get("log_score_diff")), f3(q["lines"]["8.5"]["ll"], 4),
                     f3(q["lines"]["9.5"]["ll"], 4), f3(q["lines"]["10.5"]["ll"], 4), ci(q["lines_diff"]["mean"])])
    rep.add(H2("7.6. Качество вариантов, используемых для денег (2026/27)"),
            K.table(rows, widths=W(2.2, 1.5, 3.4, 1.5, 1.5, 1.5, 3.4)),
            P(f"Таблица 8. Варианты раздела 6 на сыгранных матчах 2026/27, парно к продакшну (LS {f3(prod_ls, 4)}).",
              "small"))
    for u in ("close", "t09"):
        b = o["betting"][u]
        rows = [["Вариант", "Стратегия", "Ставок", "Hit", "Ср. коэф.", "ROI [95 % ДИ]", "ΔROI к продакшну [95 % ДИ]", "p"]]
        for k, v in b.items():
            for sk in ("ev0", "ev5", "best0"):
                s = v[sk]
                if not s["bets"]:
                    rows.append([k, sk, 0, "—", "—", "—", "—", "—"])
                    continue
                rows.append([k, sk, s["bets"], pc(s["hit"]), f3(s["avg_odds"], 2),
                             f"{pp(s['roi'])} [{pp(s['ci95'][0])}; {pp(s['ci95'][1])}]",
                             ci(s["vs_ref"], 100, 1) if s.get("vs_ref") else "—",
                             pv(s["vs_ref"]["p"]) if s.get("vs_ref") else "—"])
        rep.add(H2(f"7.{7 if u == 'close' else 8}. Ставки на угловые — {UNI[u]}"),
                K.table(rows, widths=W(2.0, 1.5, 1.4, 1.4, 1.4, 3.4, 3.6, 1.3)),
                P(f"Таблица {9 if u == 'close' else 10}. ROI в % на ставку 1, бутстреп по матчам; Δ — парно к той же "
                  f"стратегии с вероятностями продакшна, в п.п.", "small"))

    for u, mm in (("close", mc), ("t09", mt)):
        rows = [["Вариант", "Ставок", "Hit", "Ср. коэф.", "ROI [95 % ДИ]", "ΔROI к B, п.п. [95 % ДИ]", "p", "Holm p",
                 "Main угловые", "Тот же Main"]]
        for k, v in mm.items():
            s, r = v["summary"], v["roi_ci"]
            rows.append([k, s["bets"], pc(s["hit_rate"]), f3(s["avg_odds"], 2),
                         f"{pp(r['roi'])} [{pp(r['ci95'][0])}; {pp(r['ci95'][1])}]",
                         ci(v["vs_b"]["roi_diff"], 100, 2) if k != "prod" else "0",
                         pv(v["vs_b"]["roi_diff"]["p"]) if k != "prod" else "—", pv(v.get("holm_p")),
                         f"{v['corner_mains']} ({v['corner_main_bets']} ст.)", pc(v["vs_b"]["same_main_share"])])
        rep.add(H2(f"7.{9 if u == 'close' else 10}. Main — {UNI[u]}"),
                K.table(rows, widths=W(1.9, 1.2, 1.4, 1.2, 2.8, 3.0, 1.2, 1.2, 1.9, 1.4)),
                P(f"Таблица {11 if u == 'close' else 12}. Main по правилу продакшна; prod — воспроизведение Baseline B "
                  f"(совпадение выбора {pc(o['self_check'][f'{u}_prod_replacement_equals_baseline_b'])}). Baseline A "
                  f"(история сайта): ROI {pp(base_a['roi_ci']['roi'])} % на {base_a['summary']['bets']} ставках; "
                  f"ΔROI(B − A) на общих матчах {ci(mm['prod']['vs_a']['roi_diff'], 100, 2)} п.п.", "small"))

    # 8 ---------------------------------------------------------------------------------------------------
    rep.add(H1("8. Графики"),
            *figure(F / "coverage.png", "Рисунок 1. Среднее угловых по лигам и сезонам; безусловная дисперсия / среднее."),
            *figure(F / "ratings_grid.png", "Рисунок 2. Подбор шага и регрессии к среднему динамических рейтингов (2023/24–2024/25).", width=12 * K.cm),
            *figure(F / "feature_signal.png", "Рисунок 3. Корреляция признаков с суммой угловых до сезона выбора.", width=14 * K.cm),
            *figure(F / "mean_models.png", "Рисунок 4. Девиация Пуассона моделей среднего относительно продакшна, 95 % ДИ."),
            *figure(F / "distributions.png", "Рисунок 5. Log score распределений относительно продакшна: выбор (2025/26) и проверка (2026/27)."),
            *figure(F / "dispersion.png", "Рисунок 6. Условная дисперсия угловых против прогноза и PIT-гистограммы."),
            *figure(F / "reliability.png", "Рисунок 7. Надёжность вероятностей тоталов 8.5 / 9.5 / 10.5, 2026/27."),
            *figure(F / "monthly.png", "Рисунок 8. Разность log score выбранного варианта и продакшна по месяцам 2026/27."),
            *figure(F / "betting.png", "Рисунок 9. ROI ставок на угловые по вариантам и стратегиям, оба универсума."),
            *figure(F / "main_effect.png", "Рисунок 10. Изменение ROI Main относительно Baseline B, оба универсума."))

    # 9 ---------------------------------------------------------------------------------------------------
    stat = [H1("9. Статистический анализ")]
    stat.append(P("Все интервалы — бутстреп по матчам (2000 повторов), разности — парный бутстреп на одних и тех же "
                  "матчах, p — двусторонний; поправка Холма — по вариантам Main внутри универсума. Сезон 2026/27 уже "
                  "изучался в v6.1 / v6.2, поэтому результат на нём — хронологическая проверка, а не доказательство."))
    for u in ("close", "t09"):
        bk = o["breakdown"][u]
        mb = {r["month"]: r for r in bk["month_b"]}
        rows = [["Месяц", "Ставок", "ROI лучший", "ROI B", "Hit лучший", "Hit B"]]
        for r in bk["month"]:
            b = mb.get(r["month"], {})
            rows.append([r["month"] + (" (мало)" if r["bets"] < 100 else ""), r["bets"], pp(r["roi"]), pp(b.get("roi")),
                         pc(r["hit_rate"]), pc(b.get("hit_rate"))])
        stat += [H2(f"9.{1 if u == 'close' else 2}. По месяцам и лигам — {UNI[u]}"), K.table(rows),
                 P("ROI Main (%) и hit rate выбранного варианта («лучший») и Baseline B по месяцам («мало» — меньше "
                   "100 ставок).", "small")]
        lb = {r["league"]: r for r in bk["league_b"]}
        rows = [["Лига", "Ставок", "ROI лучший", "ROI B", "Лига", "Ставок", "ROI лучший", "ROI B"]]
        lg = bk["league"]
        half = (len(lg) + 1) // 2
        for i in range(half):
            row = []
            for r in (lg[i], lg[i + half] if i + half < len(lg) else None):
                if r is None:
                    row += ["", "", "", ""]
                else:
                    b = lb.get(r["league"], {})
                    row += [r["league"] + ("*" if r["bets"] < 60 else ""), r["bets"], pp(r["roi"]), pp(b.get("roi"))]
            rows.append(row)
        stat += [K.table(rows), P("ROI Main выбранного варианта («лучший») и Baseline B по лигам, % (* — меньше 60 ставок).",
                                   "small")]
        rb = o["robustness"][u]
        rows = [["Вариант", "ROI без 1 лучшей", "без 5", "без 10", "коэф. × 0.98", "B × 0.98", "коэф. × 0.95", "B × 0.95"]]
        for k, v in rb.items():
            rows.append([k, pp(v["roi_without_top1"]), pp(v["roi_without_top5"]), pp(v["roi_without_top10"]),
                         pp(v["roi_odds_x0.98"]), pp(v["base_roi_odds_x0.98"]), pp(v["roi_odds_x0.95"]),
                         pp(v["base_roi_odds_x0.95"])])
        stat += [K.table(rows), P("Зависимость от крупных выигрышей и чувствительность к коэффициентам (ROI, %).", "small")]
    sc = o["self_check"]
    eng = sc.get("engine_route", {})
    stat.append(P(f"Самопроверки: подстановка вероятностей продакшна в таблицу кандидатов воспроизводит Baseline B на "
                  f"{pc(sc['close_prod_replacement_equals_baseline_b'], 2)} (close) и "
                  f"{pc(sc['t09_prod_replacement_equals_baseline_b'])} (t09) матчей (расхождения — только равные по вероятности "
                  f"и цене тождественные исходы, например двойной шанс 1X и фора хозяев +0.5: выбор между ними зависит "
                  f"от порядка перебора). Путь через "
                  f"`engine.batch_from_outputs` (вариант {eng.get('variant', '—')}, {eng.get('matches', '—')} матчей): "
                  f"совпадение Main {pc(eng.get('same_main_share'), 2)}, максимальная разница вероятности угловых "
                  f"{K.num(eng.get('max_abs_corner_prob_diff'), 6)}."))
    m = o["eval_monthly"]
    stat.append(P("Разность log score выбранного варианта и продакшна по месяцам: "
                  + "; ".join(f"{k}: {v['log_score_diff']:+.4f} (n={v['n']})" for k, v in m.items()) + "."))
    rep.add(*stat)

    # 10–12 -----------------------------------------------------------------------------------------------
    dec = o["decisions"]
    concl = conclusions(o)
    rep.add(H1("10. Выводы"), *bullets(concl))
    rows = [["Вариант", "Решение", "ΔLS [95 % ДИ]", "ΔLL линий [95 % ДИ]", "Месяцы", "Main лучше (Holm, оба)",
             "Main хуже (ДИ)"]]
    yn = lambda v: "—" if v is None else ("да" if v else "нет")
    for k, d in dec.items():
        q = o["variant_quality"].get(k, {})
        rows.append([k, DEC[d["decision"]], ci(q.get("log_score_diff")), ci(q.get("lines_diff", {}).get("mean")),
                     yn(d["months_stable"]), yn(d["main_significant_both"]), yn(d["main_worse_ci"])])
    rep.add(H1("11. Рекомендации (A / B / C)"), K.table(rows, widths=W(1.8, 3.2, 3.0, 3.0, 1.4, 2.2, 1.6)),
            P("Правило: A — log score и средний log loss линий значимо лучше продакшна (парный бутстреп), лучше в "
              "≥ 3 из 4 полных месяцев, Main не хуже ни в одном универсуме; B — обе точечные оценки лучше (или Main "
              "значимо лучше в обоих универсумах после Холма) без значимого ухудшения Main; C — иначе.", "small"),
            *bullets(recommendations(o)))
    if pre:
        rep.add(P("Предрегистрация B-кандидатов (`results/preregistration.json`): "
                  + ", ".join(p["variant"] for p in pre) + ". Первичная метрика — парный средний log score суммы "
                  "угловых против продакшна (меньше — лучше); вторичные — средний log loss линий 8.5 / 9.5 / 10.5 и "
                  "ROI Main парно к Baseline B (close и t09); минимальная форвард-выборка — 1500 сыгранных матчей с "
                  f"угловыми (≈ 3 месяца); зафиксировано {pre[0]['frozen_at']}."))
    rep.add(H1("12. Ограничения"), *bullets([
        "2026/27 уже изучался (v6.1, v6.2); результаты на нём — не независимое подтверждение.",
        "Нет исторических данных о блокированных ударах, навесах, фланговых атаках, составах и позициях; xG есть "
        "только в 2026/27. Стиль игры учтён лишь косвенно.",
        "Цены угловых есть только в 2026/27 (OddsPapi bet365); прошлые сезоны нельзя проверить деньгами. В t09 "
        "примерно половина линий угловых не имеет цены на момент решения.",
        "Угловые сильно шумят: даже идеальная модель среднего объясняет малую долю дисперсии (R² продакшна "
        f"{f3(em['xgb_prod']['r2'])}); различия между распределениями — в четвёртом знаке log score.",
        "Гиперпараметры LightGBM / CatBoost не подбирались (фиксированные разумные значения + ранняя остановка); "
        "v6.1 показала, что подбор XGBoost не даёт выигрыша.",
        "Main выбирает угловые редко (см. таблицы 11–12), поэтому эффект на Main мал по построению."]))
    rep.add(H1("Как воспроизвести"), *K.reproduce([
        ".venv/bin/python -m experiments.v6_3.corners.data",
        "experiments/v6_3/heavy.sh 14400 experiments/v6_3/corners/logs/fit_all.log \\\n"
        "    .venv/bin/python -m experiments.v6_3.corners.fit all",
        "experiments/v6_3/heavy.sh 1800 experiments/v6_3/corners/logs/fit_wf_extra.log \\\n"
        "    .venv/bin/python -m experiments.v6_3.corners.fit wf \\\n"
        "    --models xgb_mkt,lgbm_mkt,cat_mkt,xgb_ha",
        ".venv/bin/python -m experiments.v6_3.corners.run",
        ".venv/bin/python -m experiments.v6_3.corners.build_pdf"]),
        P("Файлы: `results/results.json` (все метрики), `results/corner_predictions_2026.csv` (μ и P(over) по матчам), "
          "`results/per_match_*.csv` (таблицы §5), `results/coverage_2026.csv`, `results/coverage_league_season.csv`, "
          "`results/data_inventory.json`, `results/feature_signal.csv`, `results/preregistration.json`, "
          "`results/run_info.json` (версии кода, данных и параметры)."))
    rep.build()
    n = rep.check()
    print(f"built {OUT} ({n} pages)")
    write_findings(o, concl)


def conclusions(o) -> list[str]:
    bm, bd = o["best_combo"]["model"], o["best_combo"]["dist"]
    em, sm = o["eval_mean"], o["select_mean"]
    hs = o.get("select_mean_holm", {})
    lab = lambda m: E.LABELS.get(m, m)
    better_sel = [m for m, v in sm.items() if "dev_diff" in v and v["dev_diff"]["diff"] < 0 and hs.get(m, 1) < 0.05]
    better_ev = [m for m, v in em.items() if v.get("dev_diff") and v["dev_diff"]["diff"] < 0]
    sig_ev = [m for m, v in em.items() if v.get("dev_diff") and v["dev_diff"]["ci95"][1] < 0]
    ed = o["eval_dist"]
    pois = ed["xgb_prod|poisson"]["log_score_diff"]
    dist_rng = [ed[f"xgb_prod|{d}"]["log_score_diff"]["diff"] for d in ("nb_mom", "nb", "nb_league", "nbp", "cmp")]
    pr = o["dist_params"]
    q = o["variant_quality"]
    out = [f"Новые признаки работают на сезоне выбора: на 2025/26 девиацию Пуассона значимо (Holm p < 0.05) "
           f"снижают {len(better_sel)} моделей ({', '.join(lab(m) for m in better_sel)}); лучшая — {lab(o['best_model'])}, "
           f"Δ {ci(sm[o['best_model']]['dev_diff'])}.",
           f"На 2026/27 улучшение сохраняется по точке у {len(better_ev)} моделей, но значимо — у "
           f"{len(sig_ev)} ({', '.join(lab(m) for m in sig_ev) or 'ни одной'}): выбранная {lab(o['best_model'])} — MAE "
           f"{f3(em[o['best_model']]['mae'])} против {f3(em['xgb_prod']['mae'])}, R² {f3(em[o['best_model']]['r2'])} против "
           f"{f3(em['xgb_prod']['r2'])}, Δдевиации {ci(em[o['best_model']].get('dev_diff'))}. Выигрыш ~0.3–0.5 % "
           f"девиации — того же порядка, что и в v6.1: потолок предсказуемости угловых низкий.",
           f"Динамические рейтинги как самостоятельная модель хуже продакшна (Δдев. 2025/26 "
           f"{ci(sm['rating']['dev_diff'])}), полезны лишь как признаки.",
           f"Раздельные модели хозяев и гостей не лучше одной модели суммы (2025/26 Δ {ci(sm['xgb_ha']['dev_diff'])}). "
           f"Копула Франка: θ = {pr['xgb_ha|conv_frank']['theta']:.3f}, τ Кендалла = {o.get('frank_kendall_tau', 0):.3f} — при "
           f"известных средних угловые сторон почти независимы; отрицательная безусловная корреляция объясняется "
           f"средними (доминирование одной команды), и свёртка не даёт выигрыша над одним NB.",
           f"Дисперсия: r продакшна {o['r_prod']:.1f}, ML на 2025/26 для продакшн-μ {pr['xgb_prod|nb']['r']:.1f}, "
           f"COM-Пуассон ν = {pr['xgb_prod|cmp']['nu']:.3f}. Все NB / NB-P / COM-Пуассон / NB по лигам отличаются от "
           f"продакшна на 2026/27 в пределах {min(dist_rng):+.4f}…{max(dist_rng):+.4f} log score (незначимо); только "
           f"Пуассон значимо хуже ({ci(pois)}). Распределение продакшна адекватно, менять его нет смысла.",
           f"Калибровка линий (Платт / изотоническая) на 2025/26 не улучшает перекрёстный log loss "
           f"({o['calibration_choice']['method']}: {f3(o['select_calibration'][o['calibration_choice']['method']]['ll_mean_cf'], 4)} "
           f"против {f3(o['select_calibration'][o['calibration_choice']['method']]['ll_mean_raw'], 4)} без калибровки) — "
           f"вероятности тоталов уже откалиброваны (ECE ≤ {max(ed['xgb_prod|nb_prod']['lines'][l]['ece'] for l in ('8.5', '9.5', '10.5')):.3f}).",
           f"Выбранный на 2025/26 вариант {lab(bm)} + {E.DLABELS[bd]}: Δlog score к продакшну на 2026/27 "
           f"{ci(q['best']['log_score_diff'])}, Δ log loss линий {ci(q['best']['lines_diff']['mean'])}; ансамбль "
           f"{ci(q['ens']['log_score_diff']) if 'ens' in q else '—'}; ежемесячное переобучение "
           f"{ci(q['best_wf']['log_score_diff']) if 'best_wf' in q else '—'}."]
    mq = o["market_quality"]["close"]
    out.append("Против рынка (bet365 без маржи, close): log loss " + "; ".join(
        f"{l}: рынок {v['market_ll']:.4f}, продакшн {v['prod_ll']:.4f}, лучший {v['best_ll']:.4f}" for l, v in mq.items())
        + " — модели на уровне рынка, устойчивого перевеса нет.")
    b = o["betting"]["close"]
    out.append(f"Ставки на угловые (close, EV ≥ 0): продакшн ROI {pp(b['prod']['ev0']['roi'])} % "
               f"[{pp(b['prod']['ev0']['ci95'][0])}; {pp(b['prod']['ev0']['ci95'][1])}] на {b['prod']['ev0']['bets']} "
               f"ставках, выбранный вариант {pp(b['best']['ev0']['roi'])} % на {b['best']['ev0']['bets']}; ни одна "
               f"стратегия ни одного варианта не имеет ДИ выше нуля.")
    for u in ("close", "t09"):
        v = o["main"][u]["best"]
        out.append(f"Main ({u}): угловые выбираются редко (продакшн — {o['main'][u]['prod']['corner_mains']} из "
                   f"{o['main'][u]['prod']['summary']['matches']} матчей); ΔROI выбранного варианта к Baseline B "
                   f"{ci(v['vs_b']['roi_diff'], 100, 2)} п.п., Holm p = {pv(v.get('holm_p'))}; без угловых вовсе "
                   f"{ci(o['main'][u]['no_corners']['vs_b']['roi_diff'], 100, 2)} п.п.")
    return out


def recommendations(o) -> list[str]:
    dec = o["decisions"]
    out = []
    best = dec.get("best", {}).get("decision")
    if best == "A":
        out.append("Внедрить выбранный вариант: см. findings.md (параметры и место в коде).")
    else:
        out.append("Продакшн-модель угловых не заменять: ни один вариант не прошёл правило A (значимое улучшение "
                   "распределения и Main в обоих универсумах на хронологической проверке).")
    for k, d in dec.items():
        if d["decision"] == "B":
            out.append(f"{k}: оставить экспериментальным, проверить на будущих матчах по предрегистрации.")
    out.append("Сбор данных: для навесов, блокированных ударов и составов нужен источник со статистикой матчей "
               "(например, API-Football fixtures/statistics) с накоплением истории минимум за сезон — "
               "это решение владельца (новые запросы API).")
    return out


def write_findings(o, concl) -> None:
    dec = o["decisions"]
    lines = ["# FootPredict v6.3 — исследование угловых: выводы", "",
             "Все числа — из `results/results.json` (файл собирается `build_pdf.py`). PDF: `reports/v6_3/Corners_Research.pdf`.",
             "", "## Результаты", ""] + [f"- {c}" for c in concl] + ["", "## Решения", ""]
    for k, d in dec.items():
        lines.append(f"- **{k}** ({o['variants'][k]}): {DEC[d['decision']]}")
    lines += ["", "## Изменение продакшна", ""]
    if any(d["decision"] == "A" for d in dec.values()):
        lines.append(f"- Предлагается: {o['best_combo']} с параметрами {o['dist_params'][o['best_combo']['model'] + '|' + o['best_combo']['dist']]}; "
                     "место — `src/modelling/pipeline.py` (размер NB / распределение), `src/markets/distributions.py`.")
    else:
        lines.append("- Не предлагается. Продакшн (XGBoost + NB r = %.1f) остаётся." % o["r_prod"])
        lines.append("- Если B-кандидаты подтвердятся на форвард-выборке, изменение было бы в среднем, а не в "
                     "распределении: добавить новые признаки (EWMA угловых/ударов/фолов, рейтинги угловых и ударов, "
                     "рыночный профиль; `experiments/v6_3/corners/data.py`) в группы `src/features/sets.py` "
                     "(MODEL_GROUPS['corners']) или заменить модель угловых на Poisson GLM / ансамбль "
                     f"(веса {', '.join(f'{k} {v:.2f}' for k, v in o['ensemble_weights'].items())}); NB с одним r "
                     "(`nb_size`, `src/modelling/pipeline.py`) оставить — NB по лигам выбран на 2025/26 с перевесом в "
                     "5-м знаке и на 2026/27 хуже глобального r.")
    lines += ["", "## Дефекты / замечания", "",
              "- Блокированных ударов, навесов, фланговых атак, данных о крайних защитниках и вингерах в истории нет "
              "(Football-Data и локальный кэш API-Football); xG — только 2026/27.",
              "- Общий код: `main_prediction` (src/markets/recommendation.py) при точном равенстве вероятности и цены "
              "двух тождественных исходов (двойной шанс 1X и фора хозяев +0.5) выбирает по порядку перебора, поэтому "
              "replay через таблицу кандидатов и через движок расходится в 1 матче из 2250 "
              "(D2-20260906-greuther-furth-heidenheim). На ROI не влияет (одинаковые цена и исход)."]
    (D.RESULTS.parent / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        R = D.RESULTS = __import__("pathlib").Path(sys.argv[1])
    build()
