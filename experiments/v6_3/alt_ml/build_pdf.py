"""Builds reports/v6_3/Alternative_ML_Research.pdf and findings.md from results/ (no number typed by hand).

    .venv/bin/python -m experiments.v6_3.alt_ml.build_pdf
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C
from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, figure, formula, note, table

from . import models as M

D = C.study_dirs("alt_ml")
RES, FIG = D["results"], D["figures"]
OUT = C.ROOT / "reports" / "v6_3" / "Alternative_ML_Research.pdf"
TASK_RU = {"result": "1X2", "home_goals": "голы хозяев", "away_goals": "голы гостей", "corners": "угловые"}


def j(name):
    return C.read_json(RES / name)


def pp(v, d=2, signed=True):
    if v is None or v != v:
        return "—"
    return f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}"


def ci_pp(m, key="diff", d=2):
    if not m or m.get(key) is None:
        return "—"
    lo, hi = m["ci95"]
    return f"{pp(m[key], d)} [{pp(lo, d)}; {pp(hi, d)}]"


def ci_k(m, key="diff", d=2, scale=1000):
    if not m or m.get(key) is None:
        return "—"
    lo, hi = m["ci95"]
    return f"{m[key] * scale:+.{d}f} [{lo * scale:+.{d}f}; {hi * scale:+.{d}f}]"


def roi_txt(r):
    if not r or r.get("roi") is None:
        return "—"
    lo, hi = r["ci95"]
    return f"{r['roi'] * 100:+.1f} [{lo * 100:+.1f}; {hi * 100:+.1f}]"


def pval(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def W(*c):
    return [x * K.cm for x in c]


def build() -> None:
    tune, sel, met = j("tune.json"), j("selection.json"), j("metrics_2026.json")
    mn, info = j("main_2026.json"), j("run_info.json")
    val = pd.read_csv(RES / "validation_2025.csv")
    cov = pd.read_csv(RES / "coverage.csv")
    base = C.read_json(C.SHARED_RESULTS / "baseline.json")
    V, dec = mn["variants"], mn["decisions"]
    labels = list(V)
    secs = val.loc[~val["key"].str.startswith("ens")].groupby("algo")["seconds"].sum()
    total_min = float(secs.sum() / 60)
    prod = met["production"]["metrics"]
    rep = K.Report(OUT, "Альтернативные алгоритмы машинного обучения",
                   "FootPredict v6.3 — исследование 8 (Alternative ML Algorithms)",
                   {"Подбор": "обучение 2021/22–2024/25, валидация 2025/26",
                    "Проверка": "2026/27 (обучение до сезона и помесячный walk-forward); сезон уже изучен в v6.1/v6.2",
                    "Эталон": "XGBoost prod_20261012 (Baseline B)",
                    "Библиотеки": f"xgboost {info['libraries']['xgboost']}, lightgbm {info['libraries_extra']['lightgbm']}, "
                                  f"catboost {info['libraries_extra']['catboost']}, scikit-learn {info['libraries']['sklearn']}"})
    # 1-3
    rep.add(H1("1. Название исследования"),
            P("**Сравнение XGBoost с альтернативными алгоритмами машинного обучения для четырёх моделей FootPredict (1X2, "
              "голы хозяев, голы гостей, угловые) и влияние замены алгоритма на Main.** Исследование 8 технического "
              "задания v6.3."),
            H1("2. Постановка проблемы"),
            P("Все четыре модели FootPredict — XGBoost с параметрами, подобранными Hyperopt. Модель 1X2 проигрывает "
              f"рынку (log loss {prod['result']['log_loss']:.4f} против {met['_market_close']['metrics']['result']['log_loss']:.4f} "
              f"у цены closing без маржи), модель угловых почти не объясняет дисперсию (R² {prod['corners']['r2']:.3f}). "
              "Вопрос: даёт ли другой алгоритм (или ансамбль) лучшие вероятности и ожидания на тех же признаках, и "
              "оправдывает ли выигрыш усложнение production (новая зависимость, другой формат модели, другое время "
              "обучения)."),
            H1("3. Гипотезы"),
            bullets(["H1. Градиентный бустинг другой реализации (LightGBM, CatBoost, HistGradientBoosting) на тех же признаках "
                     "даёт меньший log loss / девиацию Пуассона, чем XGBoost production.",
                     "H2. Бэггинг деревьев (Random Forest, Extra Trees) или регуляризованные GLM лучше откалиброваны на "
                     "шумной цели (результат матча, угловые).",
                     "H3. Ансамбль разных алгоритмов уменьшает дисперсию прогноза и улучшает метрики.",
                     "H4. Улучшение метрик переходит в лучшую калибровку производных рынков и в ROI Main (парно к Baseline B)."]))
    # 4
    c0 = cov.loc[(cov["method"] == "baseline_b") & (cov["universe"] == "close")].iloc[0]
    rows = [["Период", "Назначение", "Строк (1X2)"],
            ["2021/22–2024/25", "обучение при подборе", str(tune["result"]["train_rows"])],
            ["2025/26", "валидация (выбор конфигурации, ансамблей, порядок алгоритмов)", str(tune["result"]["valid_rows"])],
            ["2021/22–2025/26", "обучение для 2026/27 (как production)", "—"],
            ["2026/27", f"проверка: {int(c0['played'])} сыгранных из {int(c0['season_matches'])}", str(int(c0["played"]))]]
    rep.add(H1("4. Исходные данные"),
            P("Таблица признаков production (`local_data/processed/features.parquet`), 21 лига, сезоны 2021/22–2026/27; "
              f"признаки каждой задачи — набор production (`TaskTrainer`, {tune['result']['features']} столбцов для 1X2 после "
              "one-hot лиги, {} для угловых); угловые — только в лигах модели угловых production. Цены и Baseline A/B — "
              "общий replay v6.3.".format(tune["corners"]["features"])),
            table(rows),
            P(f"Матчи 2026/27 без модели угловых (лига вне списка) получают рынки угловых без прогноза — как в production: "
              f"{int(c0['no_corner_model_league'])} матчей. Ни один сыгранный матч не исключён.", "small"))
    # 5
    grid_rows = [["Алгоритм", "Конфигурации (сетка)"]]
    for a, cfgs in M.GRID.items():
        grid_rows.append([M.ALGO_TITLE[a], "; ".join(c["cfg"] for c in cfgs)])
    rep.add(H1("5. Методика"),
            P("**Сопоставимость.** Все алгоритмы получают одинаковые входы: признаки и препроцессор production "
              "(`FeaturePreprocessor`), веса давности production для задачи (полураспад 2 года для голов и угловых, без "
              "весов для 1X2), те же лиги для угловых. Меняется только оценщик. GLM получают медианную импутацию с "
              "индикаторами пропусков и стандартизацию; деревья читают пропуски сами."),
            P("**Вычислительный бюджет (заявлен до запуска):** ≤ 3 конфигурации на алгоритм и задачу на фиксированной "
              "сетке (без байесовского поиска; XGBoost production подбирался Hyperopt — сравнение скорее в пользу XGBoost), "
              "2 потока на модель, этап подбора ≤ 2 ч, финальные обучения ≤ 1 ч, walk-forward — только XGBoost и два "
              f"лучших альтернативных алгоритма 2025/26. Фактически подбор занял {total_min:.0f} мин процессорного времени "
              "на 2 потоках."),
            table(grid_rows),
            P("Число итераций бустинга — ранняя остановка на 2025/26 (100 раундов терпения; HGB — шаг 25); то же число "
              "используется при обучении до 2026/27. XGBoost: prod — параметры и число деревьев production; prod_es — те же "
              "параметры с ранней остановкой на 2025/26."),
            P("**Метрики.** 1X2: accuracy, macro F1, balanced accuracy, log loss, Brier (сумма по трём классам), ECE "
              "наибольшей вероятности. Счётные цели: MAE, RMSE, R², средняя девиация Пуассона, логарифмическая оценка "
              "(Пуассон для голов, отрицательное биномиальное с размером production для угловых). Парные разности по "
              f"матчам к production — бутстреп ({C.N_BOOT} повторов)."),
            formula("D(y, μ) = 2·[y·ln(y/μ) − (y − μ)],   LogLoss = −ln p_{y},   Brier = Σ_{k}(p_{k} − 1[y = k])^{2}"),
            P("**Main.** Выходы каждого варианта проходят через `engine.batch_from_outputs` — математику рынков и правило "
              "Main production без изменений; универсумы close (основной) и t09; ROI парно к Baseline B того же "
              "универсума; Холм по всем вариантам в каждом универсуме. Калибровка производных рынков — Brier и ECE "
              "всех кандидатов по группам рынков (close)."),
            P("**Выбор без утечки.** Лучшая конфигурация алгоритма, состав ансамблей, «лучший алгоритм для задачи» и "
              "алгоритмы для walk-forward выбраны по 2025/26. Walk-forward: матчи месяца M прогнозируются моделями, "
              "обученными на всём сыгранном до 1-го числа M."))
    # 6
    rows = [["Группа вариантов", "Состав"]]
    rows.append(["static_<алгоритм>", "все 4 задачи — алгоритм, обучение 2021/22–2025/26"])
    for k, v in sel["ensembles"].items():
        rows.append([k, " + ".join(v)])
    rows.append(["best_per_task", "; ".join(f"{TASK_RU[t]}: {a}" for t, a in sel["best_per_task"].items())])
    rows.append(["wf_<алгоритм>", "помесячный walk-forward: " + ", ".join(sel["walk_algos"])])
    rep.add(H1("6. Эксперименты"),
            P(f"Этап 1: {mn['n_configs_stage1']} конфигураций и ансамблей × задачи на валидации 2025/26. Этап 2–3: "
              f"{mn['n_variants']} вариантов на 2026/27 в двух универсумах."), table(rows))
    # 7.1 validation
    rows = [["Алгоритм"] + [TASK_RU[t] for t in M.TASK_NAMES] + ["Ср. выигрыш"]]
    gain = sel["validation_gain"]
    for a in ["xgb"] + sel["order"]:
        r = [M.ALGO_TITLE[a]]
        for t in M.TASK_NAMES:
            b = sel["best_configs"][a][t] if a != "xgb" else {"score": tune[t]["configs"]["xgb:prod"]["score"],
                                                               "key": "xgb:prod"}
            r.append(f"{b['score']:.5f} ({b['key'].split(':')[1]})")
        r.append(f"{gain[a] * 100:+.2f}%" if a != "xgb" else "0 (эталон)")
        rows.append(r)
    for e in sel["ensembles"]:
        r = [e]
        for t in M.TASK_NAMES:
            v = val.loc[(val["task"] == t) & (val["key"] == e), "score"]
            r.append(f"{float(v.iloc[0]):.5f}")
        r.append("")
        rows.append(r)
    xr = [f"{tune[t]['configs']['xgb:prod']['score']:.5f}" for t in M.TASK_NAMES]
    rep.add(H1("7. Результаты"), H2("7.1 Подбор на 2025/26"),
            table(rows, widths=W(3.6, 2.75, 2.75, 2.75, 2.75, 2.0), font=7.2),
            P("Лучшая конфигурация каждого алгоритма (log loss для 1X2, средняя девиация Пуассона для счётных задач); "
              f"XGBoost с параметрами production: {', '.join(xr)}. «Ср. выигрыш» — среднее относительное улучшение к "
              "XGBoost prod по четырём задачам (оценка на той же валидации, где остановлен бустинг, — слегка "
              "оптимистична для бустингов с ранней остановкой).", "small"))
    sr = sel["roi_2025"]
    rows = [["Вариант", "ROI 2025/26, %", "Ставки", "Hit", "ΔROI к XGBoost, п.п.", "Тот же Main"]]
    for k, v in sr.items():
        vs = v["vs_xgb"]
        rows.append([k, roi_txt(v["roi_ci"]), str(v["summary"]["bets"]), K.share(v["summary"]["hit_rate"]),
                     ci_pp(vs["roi_diff"]) if vs else "—", K.share(vs["same_main_share"], 0) if vs else "—"])
    rep.add(H2("7.2 Main на сезоне подбора 2025/26 (цены Bet365 Football-Data)"), table(rows, font=7.8))
    # 7.3 metrics 2026/27
    rows = [["Вариант", "Acc", "Macro F1", "Bal. acc", "Log loss", "Brier", "ECE"]]
    for k in ["production"] + labels + ["_market_close"]:
        r = met[k]["metrics"]["result"]
        rows.append([k if k != "_market_close" else "рынок (closing)", K.share(r["accuracy"]), K.num(r["macro_f1"], 3),
                     K.num(r["balanced_accuracy"], 3), K.num(r["log_loss"], 4), K.num(r["brier"], 4), K.num(r["ece_top"], 3)])
    rep.add(H2("7.3 Метрики 2026/27"), table(rows, font=7.6), P("1X2, все сыгранные матчи 2026/27.", "small"))
    for t in ("home_goals", "away_goals", "corners"):
        rows = [["Вариант", "MAE", "RMSE", "R²", "Девиация", "Лог. оценка", "Ср. прогноз / факт"]]
        for k in ["production"] + labels:
            r = met[k]["metrics"][t]
            rows.append([k, K.num(r["mae"], 3), K.num(r["rmse"], 3), K.num(r["r2"], 3), K.num(r["poisson_deviance"], 4),
                         K.num(r["log_score"], 4), f"{r['mean_pred']:.2f} / {r['mean_true']:.2f}"])
        rep.add(table(rows, font=7.4), P(f"{TASK_RU[t].capitalize()}, 2026/27.", "small"))
    rows = [["Вариант", "Δ log loss", "Δ Brier", "Δ дев. хозяев", "Δ дев. гостей", "Δ дев. угловых"]]
    for k in labels:
        p = met[k]["paired"]
        rows.append([k, ci_k(p["ll"]), ci_k(p["brier"]), ci_k(p["dev_home_goals"]), ci_k(p["dev_away_goals"]),
                     ci_k(p["dev_corners"])])
    rep.add(H2("7.4 Парные разности с production (·10⁻³, 95 % ДИ)"),
            table(rows, widths=W(2.5, 2.9, 2.9, 2.9, 2.9, 2.9), font=6.8),
            P("Отрицательное значение — вариант лучше production на тех же матчах.", "small"))
    # 7.5 Main
    for u, title in (("close", "close (основной)"), ("t09", "t09 (проверка устойчивости)")):
        rows = [["Вариант", "ROI, %", "Ст.", "Hit", "Кф", "ΔROI к B, п.п.", "p", "Холм", "Тот же"]]
        for k in labels:
            e = V[k][u]
            s = e["summary"]
            rows.append([k, roi_txt(e["roi_ci"]), str(s["bets"]), K.share(s["hit_rate"]), K.num(s["avg_odds"], 2),
                         ci_pp(e["vs_b"]["roi_diff"]), pval(e["vs_b"]["roi_diff"]["p"]), pval(e.get("holm_p")),
                         K.share(e["vs_b"]["same_main_share"], 0)])
        b = mn["baseline_b"][u]
        rows.append(["Baseline B", roi_txt(b["roi_ci"]), str(b["summary"]["bets"]), K.share(b["summary"]["hit_rate"]),
                     K.num(b["summary"]["avg_odds"], 2), "0", "—", "—", "100%"])
        ba = mn["baseline_a"]
        rows.append(["Baseline A (сайт)", roi_txt(ba["roi_ci"]), str(ba["summary"]["bets"]),
                     K.share(ba["summary"]["hit_rate"]), K.num(ba["summary"]["avg_odds"], 2), "—", "—", "—", "—"])
        rep.add(H2(f"7.{5 if u == 'close' else 6} Main 2026/27 — {title}"),
                table(rows, widths=W(2.7, 2.8, 1.0, 1.3, 1.1, 3.4, 1.2, 1.2, 1.3), font=6.8))
    if mn["wf_vs_wf_xgb"]:
        rows = [["Вариант", "ΔROI к wf_xgb close", "p", "ΔROI к wf_xgb t09", "p"]]
        for k, v in mn["wf_vs_wf_xgb"].items():
            rows.append([k, ci_pp(v["close"]["roi_diff"]), pval(v["close"]["roi_diff"]["p"]), ci_pp(v["t09"]["roi_diff"]),
                         pval(v["t09"]["roi_diff"]["p"])])
        rep.add(table(rows, widths=W(3.0, 4.6, 1.4, 4.6, 1.4), font=7.4), P("Walk-forward альтернатив против walk-forward XGBoost: эффект алгоритма при "
                                         "одинаковом переобучении.", "small"))
    # 7.6 calibration of derived markets
    bc = mn["baseline_calibration"]
    groups = [g for g in ("1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "HANDICAP", "CORNERS") if g in bc]
    rows = [["Вариант"] + [{"DOUBLE_CHANCE": "DC"}.get(g, g) for g in groups]]
    rows.append(["production: Brier / ECE"] + [f"{bc[g]['brier']:.4f} / {bc[g]['ece']:.3f}" for g in groups])
    for k in labels:
        cal = V[k]["close"].get("calibration", {})
        def cell(d):
            if not d or d.get("diff") is None:
                return "—"
            sig = " ↓" if d["ci95"][1] < 0 else (" ↑" if d["ci95"][0] > 0 else "")
            return f"{d['diff'] * 1000:+.2f}{sig}"
        rows.append([k] + [cell(cal.get(g, {}).get("d_brier_paired")) for g in groups])
    rep.add(H2("7.7 Калибровка производных рынков (close)"),
            table(rows, widths=W(2.6, *([14.4 / len(groups)] * len(groups))), font=6.4),
            P("Первая строка — Brier и ECE всех кандидатов production по группам рынков; ниже — парная разность Brier "
              "(·10⁻³, среднее по матчу); ↓ — значимо лучше, ↑ — значимо хуже (95 % парный бутстреп).", "small"))
    # 8
    rep.page_break()
    rep.add(H1("8. Графики"),
            figure(FIG / "validation_2025.png", "Рис. 1. Валидация 2025/26: отклонение оценки от XGBoost prod, % (меньше — лучше)."),
            figure(FIG / "metrics_2026_paired.png", "Рис. 2. 2026/27: парные разности метрик с production (95 % ДИ)."),
            figure(FIG / "main_roi_2026.png", "Рис. 3. ΔROI Main к Baseline B в двух универсумах (95 % ДИ)."))
    if (FIG / "walk_forward_monthly.png").exists():
        rep.add(figure(FIG / "walk_forward_monthly.png", "Рис. 4. ROI Main (close) по месяцам: walk-forward варианты и Baseline B (месяцы с ≥ 20 ставками; "
                          "июль — один матч 31.07)."))
    # 9
    best_ll = min(labels, key=lambda k: met[k]["paired"]["ll"]["diff"])
    best_roi = max(labels, key=lambda k: V[k]["close"]["vs_b"]["roi_diff"]["diff"])
    e = V[best_roi]["close"]
    mon = pd.DataFrame(e["monthly"])
    rows = [["Месяц", "Ставки", f"ROI {best_roi}, %", "ROI B, %"]]
    for r in mon.itertuples():
        rows.append([r.month + (" *" if r.small else ""), str(r.bets), pp(r.roi), pp(r.roi_b)])
    lg = pd.DataFrame(e["league"]).sort_values("d_roi")
    rep.add(H1("9. Статистический анализ"),
            P(f"Лучший по log loss 1X2 вариант — {best_ll}: {ci_k(met[best_ll]['paired']['ll'])}·10⁻³. Лучший по ΔROI "
              f"close — {best_roi}: {ci_pp(e['vs_b']['roi_diff'])} п.п., p Холма {pval(e.get('holm_p'))}; в t09 "
              f"{ci_pp(V[best_roi]['t09']['vs_b']['roi_diff'])} п.п. (p Холма {pval(V[best_roi]['t09'].get('holm_p'))}). "
              f"Без 5 крупнейших выигрышей ROI {pp(e['top_wins'].get('roi_without_top5'))} % (с ними {pp(e['top_wins'].get('roi'))} %); "
              f"при ценах ×0.98 / ×0.95: {pp(e['odds_sensitivity'].get('0.98'))} / {pp(e['odds_sensitivity'].get('0.95'))} %."),
            table(rows, widths=W(3.0, 2.0, 3.5, 3.0), font=8),
            P("Лиги с наибольшей разницей ΔROI (close): " + "; ".join(
                f"{r.league} {pp(r.d_roi)}" + (" *" if r.small else "") for r in lg.head(3).itertuples()) + " / " +
              "; ".join(f"{r.league} {pp(r.d_roi)}" + (" *" if r.small else "") for r in lg.tail(3).itertuples()) +
              f". * — меньше 100 ставок. Вариантов на 2026/27: {mn['n_variants']}; Холм — по всем вариантам в каждом "
              "универсуме. 2026/27 изучался в v6.1/v6.2 и не является независимым отложенным сезоном.", "small"))
    # 10
    cal_d = [V[k]["close"].get("calibration", {}).get(g, {}).get("d_brier_paired") or {} for k in labels for g in groups]
    cal_d = [d for d in cal_d if d.get("diff") is not None]
    cal_max = max(abs(d["diff"]) for d in cal_d) * 1000 if cal_d else float("nan")
    cal_better = sum(1 for d in cal_d if d["ci95"][1] < 0)
    cal_worse = sum(1 for d in cal_d if d["ci95"][0] > 0)
    n_better_ll = sum(1 for k in labels if met[k]["paired"]["ll"]["ci95"][1] < 0)
    n_worse_ll = sum(1 for k in labels if met[k]["paired"]["ll"]["ci95"][0] > 0)
    n_sig_roi = sum(1 for k in labels if (V[k]["close"].get("holm_p") or 1) < 0.05)
    rep.add(H1("10. Выводы"),
            bullets([(f"Ни один вариант не дал значимого после поправки Холма изменения ROI Main (close; вариантов "
                      f"{len(labels)})." if n_sig_roi == 0 else
                      f"Значимое после поправки Холма изменение ROI Main (close): {n_sig_roi} из {len(labels)} вариантов."),
                     f"Log loss 1X2 значимо лучше production у {n_better_ll} вариантов и значимо хуже у {n_worse_ll}; "
                     f"лучший — {best_ll} ({ci_k(met[best_ll]['paired']['ll'])}·10⁻³), что на порядок меньше разрыва с "
                     f"рынком ({(prod['result']['log_loss'] - met['_market_close']['metrics']['result']['log_loss']) * 1000:.1f}·10⁻³).",
                     "Разница алгоритмов меньше, чем разница входной информации: на одних и тех же признаках бустинги, "
                     "бэггинг и GLM дают близкие вероятности; Main совпадает с Baseline B в большинстве матчей (столбец "
                     "«Тот же Main»).",
                     f"Помесячное переобучение того же XGBoost (wf_xgb, контроль) меняет log loss на "
                     f"{ci_k(met['wf_xgb']['paired']['ll'])}·10⁻³; walk-forward альтернатив против него: "
                     + "; ".join(f"{k} ΔROI close {ci_pp(v['close']['roi_diff'])} п.п." for k, v in mn["wf_vs_wf_xgb"].items())
                     + ".",
                     f"Лучший средний выигрыш на валидации 2025/26 — {M.ALGO_TITLE[sel['order'][0]]} "
                     f"({sel['validation_gain'][sel['order'][0]] * 100:+.2f} %), но он не перенёсся на 2026/27: "
                     f"log loss 1X2 static_{sel['order'][0]} {ci_k(met['static_' + sel['order'][0]]['paired']['ll'])}·10⁻³ — выбор по одному сезону "
                     "ненадёжен при разнице в десятые доли процента.",
                     f"Калибровка производных рынков (7.7): парная разность Brier с production по всем вариантам и группам "
                     f"в пределах ±{cal_max:.1f}·10⁻³; значимо лучше — {cal_better} пар (вариант × группа), значимо хуже — "
                     f"{cal_worse}.",
                     "Решения по вариантам — раздел 11."]))
    ru = {"dev_home_goals": "девиация голов хозяев", "dev_away_goals": "девиация голов гостей",
          "dev_corners": "девиация угловых", "ll": "log loss 1X2"}

    def why_ru(text):
        if text.startswith("better "):
            items = text[len("better "):].split(";")[0].split(", ")
            return "лучше: " + ", ".join(ru.get(i, i) for i in items) + "; ROI не ниже хотя бы в одном универсуме"
        return {"refitting control (same algorithm as production)": "контроль переобучения (тот же алгоритм)",
                "ROI better in both universes (Holm) and log loss not worse": "ROI лучше в обоих универсумах (Холм)",
                "1X2 log loss significantly worse": "log loss 1X2 значимо хуже",
                "no significant model gain; ROI difference not significant": "нет значимого выигрыша метрик; ΔROI незначим"
                }.get(text, text)

    why = mn.get("decision_reasons", {})
    rows = [["Вариант", "Решение", "Основание"]]
    for k in labels:
        rows.append([k, dec[k], f"{why_ru(why.get(k, ''))}. Δ log loss {ci_k(met[k]['paired']['ll'], d=1)}; ΔROI close "
                                f"{pp(V[k]['close']['vs_b']['roi_diff']['diff'])} / t09 {pp(V[k]['t09']['vs_b']['roi_diff']['diff'])} п.п."])
    rep.add(H1("11. Рекомендации (A / B / C)"), table(rows, widths=[3.0 * K.cm, 1.7 * K.cm, 12.3 * K.cm], font=7.0),
            P("Правило: A — ΔROI > 0 в обоих универсумах, p Холма < 0.05 (close), log loss 1X2 не хуже; B — хотя бы одна "
              "метрика модели значимо лучше (ДИ ниже 0), log loss 1X2 не хуже и ROI не ниже хотя бы в одном универсуме; "
              "C — иначе. Требование задания «оправдывает ли выигрыш усложнение» учтено так: замена XGBoost требует новой "
              "зависимости, нового формата модели и переобучения всех задач, поэтому небольшой положительный ΔROI без "
              "выигрыша метрик модели не даёт B (правило уточнено в этом смысле после первого просмотра таблицы — "
              "уточнение только ужесточает отбор). Значимость метрик не поправлена на множественность (4 метрики × "
              f"{len(labels)} вариантов), поэтому B — только форвард-проверка (`results/preregistration.json`), "
              "а не внедрение. Production не меняется.", "small"))
    rep.add(H1("12. Ограничения"),
            bullets(["Сетки малы (бюджет); XGBoost production подобран Hyperopt, альтернативы — нет.",
                     "Ранняя остановка и оценка на одной валидации (2025/26) — оценки этапа 1 слегка оптимистичны.",
                     "Размер NB для угловых, множитель ничьих и ρ Диксона–Коулза взяты из production для всех вариантов.",
                     "2026/27 уже изучен; ROI — хронологическая проверка, не доказательство; цены в t09 известны частично.",
                     "Walk-forward выполнен только для трёх алгоритмов (бюджет).",
                     "HistGradientBoosting для 1X2 перезапущен после первого прохода с окончательной сеткой (шаг "
                     "скорости обучения 0.02, шаг 25 итераций вместо 0.05 / 50: первый проход останавливался на первой "
                     "контрольной точке); результаты отчёта — по окончательной сетке для всех задач.",
                     "Ансамбли и «лучший алгоритм для задачи» выбраны на той же валидации, где остановлен бустинг "
                     "участников; лучшая сила регуляризации GLM для счётных задач — на краю сетки."]),
            H1("Как воспроизвести"),
            K.reproduce(["H=experiments/v6_3/heavy.sh; L=experiments/v6_3/alt_ml/logs",
                         "$H 14400 $L/tune.log   .venv/bin/python -m experiments.v6_3.alt_ml.tune",
                         "$H 5400  $L/static.log .venv/bin/python -m experiments.v6_3.alt_ml.fit26 static",
                         "$H 7200  $L/walk.log   .venv/bin/python -m experiments.v6_3.alt_ml.fit26 walk",
                         ".venv/bin/python -m experiments.v6_3.alt_ml.run",
                         ".venv/bin/python -m experiments.v6_3.alt_ml.build_pdf"]),
            P("Файлы: `results/tune.json`, `validation_2025.csv`, `selection.json`, `metrics_2026.json`, `main_2026.json`, "
              "`per_match_*.csv` (поля §5), `coverage.csv`, `preregistration.json`, `run_info.json`; рисунки — `figures/`.",
              "small"))
    rep.build()
    print("pages", rep.check(scale=1.0))
    findings(sel, met, mn, labels, best_ll, prod)


def findings(sel, met, mn, labels, best_ll, prod) -> None:
    V, dec = mn["variants"], mn["decisions"]
    lines = ["# FootPredict v6.3 — исследование 8: альтернативные алгоритмы ML — выводы", "",
             "Все числа — из `results/` (пишет `build_pdf.py`). PDF: `reports/v6_3/Alternative_ML_Research.pdf`.", "",
             "## Дефекты", "", "- Дефектов production не найдено.", "",
             "## Подбор (2025/26)", "",
             "- Порядок альтернатив по среднему относительному выигрышу к XGBoost prod: "
             + ", ".join(f"{a} {sel['validation_gain'][a] * 100:+.2f}%" for a in sel["order"]) + ".",
             "- Лучший по задаче: " + ", ".join(f"{t}: {a}" for t, a in sel["best_per_task"].items()) + ".", "",
             "## 2026/27 (парно к production / Baseline B)", "",
             "| Вариант | Δ log loss ·10⁻³ | Δ дев. хозяев ·10⁻³ | Δ дев. угловых ·10⁻³ | ΔROI close, п.п. | p Холма | ΔROI t09, п.п. | Решение |",
             "|---|---|---|---|---|---|---|---|"]
    for k in labels:
        c, t = V[k]["close"], V[k]["t09"]
        lines.append(f"| {k} | {ci_k(met[k]['paired']['ll'])} | {ci_k(met[k]['paired']['dev_home_goals'])} | "
                     f"{ci_k(met[k]['paired']['dev_corners'])} | "
                     f"{ci_pp(c['vs_b']['roi_diff'])} | {pval(c.get('holm_p'))} | {ci_pp(t['vs_b']['roi_diff'])} | {dec[k]} |")
    a_list = [k for k in labels if dec[k] == "A"]
    labels = [k for k in labels]
    b_list = [k for k in labels if dec[k] == "B"]
    lines += ["", "## Решение", "",
              f"- Лучший по log loss 1X2: {best_ll}. "
              + (f"Варианты A: {', '.join(a_list)} — кандидаты на внедрение после тестов." if a_list else
                 "Ни один вариант не получил A: ROI Main значимо не меняется (Холм), а выигрыш метрик мал "
                 "относительно разрыва с рынком; замена XGBoost потребовала бы новой зависимости и переобучения "
                 "всех задач — production не меняется."),
              f"- B (форвард-проверка): {', '.join(b_list) if b_list else 'нет'}.",
              "- Кандидаты B — только форвард-проверка (`results/preregistration.json`). Если форвард-проверка "
              "подтвердит выигрыш по голам хозяев, изменение production было бы таким: в `src/modelling/pipeline.py` "
              "обучать для задачи home_goals вторую модель — Poisson GLM (`PoissonRegressor`, alpha из "
              "`results/selection.json`, медианная импутация с индикаторами + стандартизация) на тех же признаках — и "
              "усреднять её λ с XGBoost; остальные задачи без изменений.", "",
              "## Ограничения", "", "- Малые сетки; XGBoost подобран Hyperopt.",
              "- 2026/27 — уже изученный сезон.", ""]
    (C.V63 / "alt_ml" / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
