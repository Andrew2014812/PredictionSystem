"""Builds reports/v6_2/Market_Models_Research.pdf and findings.md from results/ (no number is typed in)."""
from __future__ import annotations

import numpy as np

from scripts.report_kit import H1, H2, P, Report, bullets, figure, formula, note, reproduce, table

from . import common as C
from . import figures as F
from .figures import GROUP_RU

PDF = C.ROOT / "reports" / "v6_2" / "Market_Models_Research.pdf"
PY = "/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python"
CMDS = [f"PY={PY}", "$PY -m experiments.v6_2.market_models.data      # walk-forward base predictions (cached)",
        "$PY -m experiments.v6_2.market_models.direct    # 12 direct models x 5 folds (~40 min, n_jobs=2)",
        "$PY -m experiments.v6_2.market_models.run", "$PY -m experiments.v6_2.market_models.build_pdf"]
SEAS = ("2023", "2024", "2025", "2026")


def dci(c, k=1e3, d=2):
    if not c or c.get("diff") is None or c["diff"] != c["diff"]:
        return "—"
    return f"{c['diff'] * k:+.{d}f} [{c['ci95'][0] * k:+.{d}f}; {c['ci95'][1] * k:+.{d}f}]"


def pp(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}"


def f4(x, d=4):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def main() -> None:
    mm = C.load_json("market_metrics.json")
    comp = C.load_json("comparisons.json")
    cal = C.load_json("calibration.json")
    coh = C.load_json("coherence.json")
    cost = C.load_json("cost.json")
    lg = C.load_json("leagues.json")
    mr = C.load_json("main_risk.json")
    ref = C.load_json("market_reference.json")
    dec = C.load_json("decisions.json")
    st = C.load_json("stacking.json")
    info = C.load_json("run_info.json")
    F.build()
    names, met, cc = mm["names"], mm["metrics"], comp["comparisons"]
    best = lg["best_by_rule"]
    decs = dec["decisions"]
    a_list = [k for k, d in decs.items() if d["decision"] == "A"]
    b_list = [k for k, d in decs.items() if d["decision"] == "B"]
    n = mm["n"]

    def lab(key):
        g, v = key.split("|")
        return f"{GROUP_RU[g]}: {names[v]}"

    rep = Report(PDF, "Отдельные модели для рынков", facts={
        "Сезоны": "отбор 2023/24 + 2024/25; 2025/26 — для информации; 2026/27 — текущий, уже просмотренный",
        "Матчей": f"2023/24 {n['2023']}, 2024/25 {n['2024']}, 2025/26 {n['2025']}, 2026/27 {n['2026']}",
        "Сравнений": f"{comp['n_comparisons']} (рынок × вариант), поправка Холма"})

    rep.add(H1("1. Цель и вопрос"), P(
        "В FootPredict все рынки голов выводятся из двух регрессоров голов (матрица счёта, независимый Пуассон), "
        "согласованных с классификатором 1X2; угловые — из регрессора угловых через отрицательное биномиальное. "
        "Вопрос: дают ли отдельные модели, обученные прямо на исход рынка (обе забьют, тоталы, двойной шанс, "
        "форы, точный счёт, угловые), или их стекинг с текущими вероятностями лучшие вероятности — на тех же "
        "матчах, по log loss, Brier и калибровке, и лучшие Main/Risk — и сколько это стоит (число моделей, время)."))
    rep.add(H1("2. Гипотезы"), bullets([
        "H1: бинарный классификатор на исход рынка (BTTS, тотал на линии, 1X/X2, угловые на линии) точнее "
        "вывода из матрицы — он не связан допущениями Пуассона.",
        "H2: мультикласс по числу голов (0…6+) или разнице (≤−3…≥3) точнее и сохраняет согласованность линий.",
        "H3: стекинг (текущая вероятность + прямая модель) лучше обеих.",
        "H4: если вероятности лучше, Main/Risk на реальных ценах тоже лучше.",
        "H0: вывод всех рынков из матрицы не хуже (и дешевле, и согласован)."]))
    rep.add(H1("3. Данные"), P(
        "Таблица признаков продакшна (все сыгранные матчи 2021/22–2026/27, 21 лига, предматчевые признаки). "
        "Walk-forward: модели фолда s обучены на всех сезонах < s и прогнозируют сезон s; текущие вероятности — "
        "продакшн-модели с параметрами активной версии, обученные так же (кэш v6.1, расхождение пересчёта 0). "
        "Цены: Bet365 открытия Football-Data (1X2, тотал 2.5, одна азиатская линия матча) для 2023/24–2025/26; для "
        "2026/27 — записанные цены продакшна и архив OddsPapi до начала матча (почти все рынки)."))
    rep.add(H1("4. Методы"), H2("4.1 Модели"), bullets([
        "Текущий (CUR): матрица Пуассона из λ_h, λ_a, согласованная с 1X2; угловые NB(μ, r).",
        "BIN: XGBoost binary:logistic на исход: обе забьют, больше 1.5/2.5/3.5, 1X, X2 (P(12) = 2 − P(1X) − P(X2)), "
        "угловые больше 8.5/9.5/10.5 (только лиги с угловыми).",
        "TOT: multi:softprob, число голов 0…5, 6+ (7 классов) → все линии тотала из одного распределения.",
        "GD: multi:softprob, разница ≤−3, −2, …, 2, ≥3 (7 классов) → 1X2, двойной шанс и все азиатские линии "
        "(целые — с вероятностью возврата).",
        "ES: multi:softprob, счёт h, a ≤ 3 (16 классов) + «другой».",
        "Признаки: объединение наборов продакшн-моделей 1X2 и голов (+ лига); угловые — набор модели угловых. "
        "Параметры не настраивались: бинарные — параметры классификатора 1X2, мультиклассы — те же с шагом 0.03, "
        "угловые — параметры регрессора угловых; число деревьев — ранняя остановка в предыдущем фолде."]))
    rep.add(H2("4.2 Стекинг"),
            formula("S_BIN: p = σ(b₀ + b₁ logit p_CUR + b₂ logit p_BIN), логистическая регрессия на сезоне s−1"),
            formula("S_TOT, S_GD, S_ES: p_k ∝ p_CUR,k^a · p_DIR,k^b (логарифмический пул), (a, b) — ML на сезоне s−1"),
            P("Веса фолда s подбираются по прогнозам фолда s−1 (модели обучены на сезонах < s−1), т. е. без данных "
              "сезона s."))
    rep.add(H2("4.3 Метрики и проверка"),
            formula("Главная метрика группы: средний по линиям log loss на матч (целые форы — 3 исхода)"),
            P("Также Brier, ECE (10 бинов), кривые калибровки, RPS для тотала и разницы. Разности — парный бутстреп по "
              "матчам (2000 повторов для группы), p — нормальное приближение парной разности; Холм по всем "
              f"{comp['n_comparisons']} сравнениям. Main/Risk — продакшн-правило (векторная копия, совпадение с "
              "функциями продакшна 100 % на контрольной выборке), подмена только вероятностей одной группы рынков."),
            H2("4.4 Правило решения (зафиксировано до запуска)"), P(dec["rule"]))

    # --------------------------------------------------------------- results
    rep.add(H1("5. Качество вероятностей"))
    rows = [["Рынок: вариант", "LL отбор", "Δ ×10⁻³ [95 % ДИ]", "p Холма", "Δ 23/24", "Δ 24/25", "Δ 25/26",
             "Δ 26/27"]]
    for g, vs in mm["group_variants"].items():
        rows.append([f"{GROUP_RU[g]}: {names['CUR']}", f4(met[g]["CUR"]["select"]["group"]), "—", "—", "—", "—",
                     "—", "—"])
        for v in vs:
            c = cc[f"{g}|{v}"]
            rows.append([f"{GROUP_RU[g]}: {names[v]}", f4(met[g][v]["select"]["group"]), dci(c["select"]["group"]),
                         f"{c['holm_p_select']:.3f}"] + [f"{c[s]['group']['diff'] * 1e3:+.2f}" for s in SEAS])
    rep.add(P("Средний log loss группы (отбор = 2023/24+2024/25) и разность с текущим (вариант − текущий):"),
            table(rows), figure(C.FIGURES / "forest.png", "Рис. 1. Δ log loss к текущему, 2023/24+2024/25."),
            figure(C.FIGURES / "seasons.png", "Рис. 2. Лучший вариант каждой группы по сезонам (Δ ×10⁻³)."))

    rep.add(H2("5.1 По линиям"))
    rows = [["Линия", "Вариант", "Δ отбор ×10⁻³ [95 % ДИ]", "Δ 2025/26", "Δ 2026/27"]]
    for g in ("TOTAL", "BTTS", "DOUBLE_CHANCE", "HANDICAP", "CORNERS"):
        v = best[g]
        c = cc[f"{g}|{v}"]
        for k in mm["group_keys"][g]:
            rows.append([k, names[v], dci(c["select"].get(k)), dci(c["2025"].get(k)), dci(c["2026"].get(k))])
    rep.add(P("Лучший вариант группы (по правилу) по отдельным линиям:"), table(rows))

    rep.add(H2("5.2 Калибровка"))
    rows = [["Исход", "Вариант", "ECE отбор", "ECE 2026/27", "ср. p − частота отбор", "Brier отбор"]]
    for k in ("TOTAL_2.5", "BTTS", "DC_1X", "AH_-0.5", "CORNERS_9.5", "DRAW"):
        for v, x in cal[k].items():
            s, c26 = x["select"], x["2026"]
            rows.append([k, names[v], f4(s.get("ece")), f4(c26.get("ece")),
                         pp(s["mean_p"] - s["rate"], 2) if s.get("n") else "—", f4(s.get("brier"))])
    rep.add(table(rows), figure(C.FIGURES / "calibration.png", "Рис. 3. Калибровка, 2023/24+2024/25."))

    rep.add(H2("5.3 Согласованность рынков"))
    cs = coh
    rows = [["Проверка", "BIN 23/24", "BIN 24/25", "S_BIN 24/25", "BIN 26/27"]]
    for k, t in (("totals_not_monotone", "P(>1.5) ≥ P(>2.5) ≥ P(>3.5) нарушено"),
                 ("btts_above_over15", "P(обе) > P(>1.5)"),
                 ("corners_not_monotone", "угловые 8.5/9.5/10.5 не монотонны"),
                 ("dc_implied_draw_outside_0_1", "P(X) = P(1X)+P(X2)−1 вне [0; 1]"),
                 ("dc_implied_draw_vs_classifier_mae", "|P(X) из 1X/X2 − P(X) 1X2|, ср."),
                 ("btts_cur_above_over15_bin", "P(обе) матрицы > P(>1.5) BIN")):
        rows.append([t, pp(cs["2023"]["BIN"][k], 2) if "mae" not in k else f4(cs["2023"]["BIN"][k]),
                     pp(cs["2024"]["BIN"][k], 2) if "mae" not in k else f4(cs["2024"]["BIN"][k]),
                     pp(cs["2024"]["S_BIN"][k], 2) if "mae" not in k else f4(cs["2024"]["S_BIN"][k]),
                     pp(cs["2026"]["BIN"][k], 2) if "mae" not in k else f4(cs["2026"]["BIN"][k])])
    rep.add(P("Доли матчей (%) с нарушением логической связи рынков (у текущей матрицы — 0 по построению):"),
            table(rows))
    rows = [["Проверка", "23/24", "24/25", "25/26", "26/27"]]
    for v, k, t in (("GD", "home_win_vs_classifier_mae", "GD: |P(П1) − P(П1) классификатора|, ср."),
                    ("GD", "ah_m05_vs_1x2_home_mae", "GD: |P(фора −0.5) − P(П1) на сайте|, ср."),
                    ("GD", "home_win_vs_classifier_max", "GD: то же, максимум"),
                    ("TOT", "over25_vs_matrix_mae", "TOT: |P(>2.5) − P(>2.5) матрицы|, ср.")):
        rows.append([t] + [f4(cs[s][v][k]) for s in SEAS])
    rep.add(P("Расхождение мультиклассов с остальными рынками сайта, если подменить только одну группу:"),
            table(rows))

    rep.add(H2("5.4 Рядом с рынком"))
    rows = [["Сезон", "Рынок", "n", "Bet365 без маржи", "Текущий", "Лучший вариант"]]
    for s in SEAS:
        for mk, r in ref[s].items():
            g = {"1X2": "1X2", "TOTAL_2.5": "TOTAL", "BTTS": "BTTS", "CORNERS_9.5": "CORNERS"}[mk]
            rows.append([C.period_label(int(s)), mk, r["n"], f4(r["market"]), f4(r["CUR"]),
                         f"{f4(r.get(best[g]))} ({names[best[g]]})"])
    rep.add(P("Log loss на матчах с полной ценой (пропорциональное снятие маржи; Bet365 открытия, 2026/27 — "
              "записанные цены):"), table(rows))

    rep.add(H2("5.5 По лигам"))
    rows = [["Группа (вариант)", "Лиг", "лучше (ДИ < 0)", "хуже (ДИ > 0)"]]
    for g, x in lg["by_group"].items():
        L = x["leagues"]
        rows.append([f"{GROUP_RU[g]} ({names[x['variant']]})", len(L),
                     ", ".join(k for k, v in L.items() if v["ci95"][1] < 0) or "—",
                     ", ".join(k for k, v in L.items() if v["ci95"][0] > 0) or "—"])
    rep.add(P("Лучший вариант группы по лигам, 2023/24+2024/25 (≈ 600–800 матчей на лигу, без поправки на "
              "множественность):"), table(rows))

    rep.add(H1("6. Main и Risk"))
    cur = mr["variants"]["CUR"]
    rows = [["Подмена", "Main ROI", "ΔROI Main [95 % ДИ], п.п.", "Δ попад., п.п.", "ΔROI Risk [95 % ДИ]",
             "Изм. Main", "ΔROI Main 26/27"]]
    rows.append(["нет (текущий)", pp(cur["select"]["main"]["roi"]), "—", "—", "—", "—", "—"])
    for k, x in mr["variants"].items():
        if k == "CUR":
            continue
        a, h, r, a26 = (x["select"]["main_vs_cur"], x["select"]["main_hit_vs_cur"], x["select"]["risk_vs_cur"],
                        x["2026"]["main_vs_cur"])
        sub = ", ".join(f"{GROUP_RU[g]}→{names[v]}" for g, v in mr["substitutions"][k].items())
        rows.append([sub or k, pp(x["select"]["main"]["roi"]),
                     f"{pp(a['diff'])} [{pp(a['ci95'][0])}; {pp(a['ci95'][1])}]", pp(h["diff"]),
                     f"{pp(r['diff'])} [{pp(r['ci95'][0])}; {pp(r['ci95'][1])}]", a.get("changed_picks", "—"),
                     f"{pp(a26['diff'])} [{pp(a26['ci95'][0])}; {pp(a26['ci95'][1])}]"])
    rep.add(P("Отбор = 2023/24+2024/25; в этих сезонах цена есть только у 1X2, тотала 2.5 и одной азиатской линии "
              "(ROI считается по ставкам с ценой). 2026/27 — цены почти всех рынков."), table(rows),
            figure(C.FIGURES / "main_risk.png", "Рис. 4. ΔROI Main и Risk при подмене группы, 2023/24+2024/25."))

    rep.add(H1("7. Стоимость"))
    d26, p26 = cost["direct"]["2026"], cost["production"]["2026"]
    rows = [["Модель", "Деревьев", "Обучение, с", "Ранняя остановка, с", "Прогноз, мс/матч", "Размер, КБ"]]
    for k, x in d26.items():
        rows.append([k, x["trees"], f"{x['fit_seconds']:.1f}", f"{x['es_seconds']:.1f}",
                     f"{x['predict_ms_per_match']:.3f}", f"{x['pickle_bytes'] / 1024:.0f}"])
    tot_fit = sum(x["fit_seconds"] for x in d26.values())
    tot_lat = sum(x["predict_ms_per_match"] for x in d26.values())
    rows.append(["итого 12 прямых", sum(x["trees"] for x in d26.values()), f"{tot_fit:.0f}",
                 f"{sum(x['es_seconds'] for x in d26.values()):.0f}", f"{tot_lat:.3f}",
                 f"{sum(x['pickle_bytes'] for x in d26.values()) / 1024:.0f}"])
    rows.append(["продакшн: 4 модели", "—", f"{sum(p26.values()):.0f}", "—", "—", "—"])
    rep.add(P("Фолд 2026/27 (обучение на 2021/22–2025/26, n_jobs = 2; общая машина — время ориентировочное):"),
            table(rows))

    rep.add(H1("8. Решения"))
    rows = [["Рынок: вариант", "Решение", "Δ отбор ×10⁻³", "p Холма", "Лучше в обоих", "Не хуже позже"]]
    for k, d in decs.items():
        rows.append([lab(k), d["decision"], f"{d['diff_select'] * 1e3:+.2f}", f"{d['holm_p']:.3f}",
                     "да" if d["better_both_selection"] else "нет", "да" if d["not_worse_later"] else "нет"])
    rep.add(table(rows))
    concl = conclusions(cc, met, best, names, coh, mr, decs, cost)
    rep.add(H1("9. Выводы"), bullets(concl))
    rep.add(H1("10. Отрицательные результаты и ограничения"), bullets(limitations()))
    rep.add(H1("11. Рекомендации"), bullets(recommendations(a_list, b_list, lab)))
    stk = st[str(C.CURRENT)]
    rep.add(H2("Приложение: веса стекинга фолда 2026/27"),
            table([["Ключ", "b₀", "b₁ (текущий)", "b₂ (прямой)"]]
                  + [[k, f"{w[0]:+.3f}", f"{w[1]:+.3f}", f"{w[2]:+.3f}"] for k, w in stk["bin"].items()]
                  + [[f"пул {k}", "—", f"{w[0]:+.3f}", f"{w[1]:+.3f}"] for k, w in stk["pool"].items()]))
    rep.add(H1("12. Воспроизведение"), reproduce(CMDS),
            note(f"Git SHA запуска: {info['git_sha']}; время оценки {info['seconds']:.0f} с."))
    rep.build()
    print("pages", rep.check())
    write_findings(cc, met, best, names, coh, mr, decs, cost, a_list, b_list, lab)


def conclusions(cc, met, best, names, coh, mr, decs, cost) -> list[str]:
    out = []
    for g, v in best.items():
        c = cc[f"{g}|{v}"]
        out.append(f"{GROUP_RU[g]}: лучший вариант — {names[v]}, Δ log loss {dci(c['select']['group'])} ×10⁻³ "
                   f"(p Холма {c['holm_p_select']:.3f}), решение {decs[f'{g}|{v}']['decision']}.")
    return out


def limitations() -> list[str]:
    return ["2026/27 уже просматривался в v6.1 — не независимая проверка; 2025/26 — для информации.",
            "Гиперпараметры прямых моделей не настраивались (взяты у моделей продакшна); тщательная настройка могла "
            "бы немного улучшить прямые модели, но и стоила бы ещё 12 поисков.",
            "Цены прошлых сезонов есть только у 1X2, тотала 2.5 и одной азиатской линии — ROI Main/Risk в отборе "
            "опирается на часть ставок; BTTS и угловые с ценой — только 2026/27.",
            "Стекинг обучается на одном предыдущем сезоне (~7 000 матчей) — веса шумные.",
            "Мультикласс точного счёта ограничен счётами до 3:3 + «другой»."]


def recommendations(a_list, b_list, lab) -> list[str]:
    return ["A: " + (", ".join(lab(k) for k in a_list) if a_list else "нет — оставить вывод всех рынков из матрицы."),
            "B: " + (", ".join(lab(k) for k in b_list) + " — предрегистрация `results/preregistration.json`"
                     if b_list else "нет."),
            "C: остальные варианты (без доказанного выигрыша они добавляют модели и ломают согласованность рынков).",
            "Продакшн не меняется исследованием."]


def write_findings(cc, met, best, names, coh, mr, decs, cost, a_list, b_list, lab) -> None:
    lines = ["# FootPredict v6.2 — отдельные модели для рынков: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_2/Market_Models_Research.pdf`.", "",
             "## Вероятности (вариант − текущий, log loss ×10⁻³, 2023/24+2024/25)", "",
             "| Рынок: вариант | Δ [95 % ДИ] | p Холма | Δ 2025/26 | Δ 2026/27 | Решение |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, d in decs.items():
        c = cc[k]
        lines.append(f"| {lab(k)} | {dci(c['select']['group'])} | {c['holm_p_select']:.3f} | "
                     f"{c['2025']['group']['diff'] * 1e3:+.2f} | {c['2026']['group']['diff'] * 1e3:+.2f} | "
                     f"{d['decision']} |")
    lines += ["", "## Главное", ""] + [f"- {x}" for x in conclusions(cc, met, best, names, coh, mr, decs, cost)]
    lines += ["", "## Решение", ""] + [f"- {x}" for x in recommendations(a_list, b_list, lab)]
    lines += ["", "## Воспроизведение", "", "```"] + CMDS + ["```", ""]
    (C.HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
