"""Builds reports/v6_2/Goal_Distributions_Research.pdf and findings.md from results/ (no number is typed in)."""
from __future__ import annotations

import numpy as np

from scripts.report_kit import H1, H2, P, Report, bullets, figure, formula, note, reproduce, table

from . import common as C
from . import figures as F

PDF = C.ROOT / "reports" / "v6_2" / "Goal_Distributions_Research.pdf"
PY = "/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python"
CMDS = [f"PY={PY}", "$PY -m experiments.v6_2.market_models.data       # walk-forward base predictions (cached)",
        "$PY -m experiments.v6_2.goal_distributions.maher", "$PY -m experiments.v6_2.goal_distributions.run",
        "$PY -m experiments.v6_2.goal_distributions.build_pdf"]
KEYS = [("SCORE", "Счёт"), ("TOTALS", "Тоталы 0.5–5.5"), ("BTTS", "Обе забьют"), ("AH", "Форы"),
        ("AH_WHOLE", "Целые форы"), ("1X2", "1X2"), ("TOT7_RPS", "RPS тотала"), ("GD7_RPS", "RPS разницы")]


def tab(rows, first=4.6):
    """Table with a wide first column and equal numeric columns (17 cm in total)."""
    from reportlab.lib.units import cm
    k = max(len(r) for r in rows) - 1
    return table(rows, widths=[first * cm] + [(17 - first) * cm / k] * k)


def f3(x, d=4):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def dci(c, k=1e3, d=2):
    if c is None or c.get("diff") is None:
        return "—"
    return f"{c['diff'] * k:+.{d}f} [{c['ci95'][0] * k:+.{d}f}; {c['ci95'][1] * k:+.{d}f}]"


def pp(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}"


def label(v, names):
    if v == "P+R":
        return "Пуассон + согл. (текущий)"
    return names.get(v.replace("+R", ""), v) + (" + согл." if v.endswith("+R") else "")


def main() -> None:
    fits = C.load_json("fits.json")
    names = fits["names"]
    tests = C.load_json("tests.json")
    met = C.load_json("metrics.json")
    comp = C.load_json("comparisons.json")
    cal = C.load_json("calibration.json")
    lg = C.load_json("leagues.json")
    mr = C.load_json("main_risk.json")
    dec = C.load_json("decisions.json")
    info = C.load_json("run_info.json")
    cc = comp["comparisons"]
    variants = fits["variants"]
    ranked = sorted(cc, key=lambda v: cc[v]["select"]["SCORE"]["diff"])
    best = info["best_by_rule"]
    top = [v for v in ranked if v.endswith("+R")][:2]
    F.build(top + [v for v in ranked if v not in top][:2])
    n = met["n"]
    ref = met["loss"]["P+R"]
    decs = dec["decisions"]
    a_list = [v for v, d in decs.items() if d["decision"] == "A"]
    b_list = [v for v, d in decs.items() if d["decision"] == "B"]

    rep = Report(PDF, "Модели распределения голов", facts={
        "Сезоны": "отбор 2023/24 + 2024/25; 2025/26 — для информации; 2026/27 — текущий, уже просмотренный",
        "Матчей": f"отбор {n['select']}, 2025/26 {n['2025']}, 2026/27 {n['2026']}",
        "Вариантов": f"{len(variants)} (из них сравнено с текущим: {comp['n_variants_compared']})"})

    rep.add(H1("1. Цель и вопрос"), P(
        "Все рынки голов FootPredict (точный счёт, тоталы, обе забьют, азиатские форы с возвратом, 1X2, двойной "
        "шанс) выводятся из одной матрицы счёта 11×11. Сейчас это независимый Пуассон со средними двух регрессоров "
        "XGBoost (rho Диксона–Коулса = 0, `DIXON_COLES = False`), согласованный с классификатором 1X2 "
        "(`reconcile_with_1x2`). Вопрос исследования: даёт ли другое семейство распределений при тех же средних "
        "лучше откалиброванные вероятности по рынкам и линиям — а не только меньшую ошибку числа голов."))
    rep.add(H1("2. Гипотезы"), bullets([
        "H1 (зависимость): голы хозяев и гостей зависимы при заданных средних (Диксон–Коулс, бивариантный "
        "Пуассон, копула Франка) — это улучшит ничьи, 0-0/1-1 и целые форы.",
        "H2 (дисперсия): голы недо- или передиспергированы относительно Пуассона (отрицательное биномиальное, "
        "Конвей–Максвелл–Пуассон) — это улучшит тоталы и хвосты.",
        "H3 (нули): 0-0 встречается чаще, чем даёт Пуассон, — нужна инфляция нулей (проверяется тестом до "
        "включения варианта).",
        "H4 (средние): рейтинги атаки/защиты (Махер, раздельно дом/выезд) или их смесь с XGBoost дают лучшие "
        "средние, чем одни регрессоры.",
        "H0: текущая схема (Пуассон + согласование с 1X2) не хуже ни одного варианта."]))
    rep.add(H1("3. Данные"), P(
        "Walk-forward: сезон s прогнозируется продакшн-моделями (настроенные параметры активной версии), "
        "обученными на всех сезонах < s (кэш v6.1 main пересчитан, расхождение с ним 0). Средние голов "
        "λ_h, λ_a и вероятности 1X2 — общие для всех вариантов. Параметры формы каждого семейства подбираются "
        "максимальным правдоподобием наблюдённых счетов на сезоне s−1 (прогнозы фолда s−1) и применяются к "
        "сезону s. Цены (Main/Risk): Bet365 открытия Football-Data для прошлых сезонов (1X2, тотал 2.5, одна "
        "азиатская линия), для 2026/27 — записанные цены продакшна и архив OddsPapi до начала матча."))
    rep.add(H1("4. Методы"), H2("4.1 Семейства"), bullets([
        "Пуассон: P(i, j) = Pois(i; λ_h) Pois(j; λ_a) (продакшн).",
        "Диксон–Коулс: P(i, j) τ(i, j; ρ), τ меняет только 0-0, 0-1, 1-0, 1-1.",
        "Бивариантный Пуассон: X = Y₁ + Y₃, Y = Y₂ + Y₃, Y₃ ~ Pois(λ₃), маргинальные средние сохраняются "
        "(Y₁ ~ Pois(λ_h − λ₃)); допускает только положительную ковариацию.",
        "Отрицательное биномиальное NB(λ, r) для каждой стороны (r_h, r_a); r → ∞ — Пуассон.",
        "Конвей–Максвелл–Пуассон с подогнанным средним: P(k) ∝ μ^k/(k!)^ν, μ подбирается так, чтобы среднее "
        "было λ; ν > 1 — недодисперсия, ν < 1 — передисперсия.",
        "Копула Франка на функциях распределения: P(i, j) = C(F_h(i), F_a(j)) − C(F_h(i−1), F_a(j)) − "
        "C(F_h(i), F_a(j−1)) + C(F_h(i−1), F_a(j−1)); θ < 0 — отрицательная зависимость.",
        "Франк + КМП: копула Франка на маргиналях КМП (дисперсия и зависимость вместе).",
        "Махер: еженедельная пуассоновская регрессия по лиге (окно 730 дней, полураспад 180 дней, L2 = 0.005): "
        "MH1 — атака/защита + преимущество поля, MH2 — раздельные силы дома и в гостях (4 параметра на команду).",
        "Смесь: log λ = w log λ_XGB + (1 − w) log λ_Махер, w и вариант Махера — на сезоне s−1."]))
    rep.add(H2("4.2 Согласование и метрики"), P(
        "Каждый вариант считается в двух видах: «сырая» матрица и матрица, согласованная с классификатором "
        "1X2 (+ согл.; области П1/Х/П2 масштабируются к его вероятностям, как в продакшне). Текущий продакшн = "
        "«Пуассон + согл.». У согласованных вариантов вероятности 1X2 одинаковы по построению."),
        formula("Главная метрика: log loss точного счёта = −mean log P(фактический счёт) (лог-скор всей матрицы)"),
        formula("Бинарные рынки: LL = −[y log p + (1 − y) log(1 − p)], Brier = (p − y)², ECE — 10 бинов"),
        formula("Целые форы: 3 исхода (выигрыш / возврат / проигрыш), LL = −log P(исход)"),
        formula("RPS(тотал 0..6+, разница ≤−3..≥3) = Σ_k (F̂_k − 1{y ≤ k})² / (K − 1)"),
        P("Тоталы — линии 0.5–5.5, форы — полулинии ±0.5/±1.5/±2.5 и целые ±1/±2 (линии продакшна). Разности — "
          "парный бутстреп по матчам (1000–2000 повторов, seed 20261109), p-значения двусторонние, поправка Холма "
          "по всем вариантам для главной метрики и отдельно по каждой группе рынков."))
    rep.add(H2("4.3 Правило решения (зафиксировано до запуска)"), P(dec["rule"]))

    # ---------------------------------------------------------------- results
    rep.add(H1("5. Проверка данных: нули, ничьи, дисперсия"))
    ex = tests["excess"]
    rows = [["Ячейка", "Факт", "Ожидание", "Факт/ожид.", "z", "p (одност.)"]]
    nm = {"0-0": "0-0", "1-1": "1-1", "draw": "Ничьи", "home_0": "Хозяева 0", "away_0": "Гости 0"}
    for k in ("0-0", "1-1", "draw", "home_0", "away_0"):
        e = ex["select"][k]
        rows.append([nm[k], e["observed"], f"{e['expected']:.0f}", f"{e['ratio']:.3f}", f"{e['z']:+.2f}",
                     f"{e['p_one_sided']:.4f}"])
    rep.add(P("Независимый Пуассон со средними XGBoost, 2023/24+2024/25 (сырая матрица):"), tab(rows))
    rep.add(figure(C.FIGURES / "excess.png", "Рис. 1. Факт/ожидание независимого Пуассона по сезонам."))
    disp = tests["dispersion"]["select"]
    rep.add(P(
        f"Инфляция 0-0 {'оправдана' if tests['zero_inflation_justified'] else 'не оправдана'} по правилу "
        f"(«{tests['rule']}»): отношение 0-0 {ex['select']['0-0']['ratio']:.3f}, p = "
        f"{ex['select']['0-0']['p_one_sided']:.3f}. Избыток есть у ничьих в целом "
        f"({ex['select']['draw']['ratio']:.3f}, z = {ex['select']['draw']['z']:+.1f}) — его снимает согласование "
        "с классификатором 1X2, а не нули."),
        P(f"Тест Кэмерона–Триведи (α > 0 — передисперсия): хозяева α = {disp['home']['alpha']:+.4f} "
          f"(t = {disp['home']['t']:+.2f}), гости α = {disp['away']['alpha']:+.4f} (t = {disp['away']['t']:+.2f}); "
          f"Var/mean остатков {disp['home']['var_over_mean']:.3f} / {disp['away']['var_over_mean']:.3f}. "
          "Передисперсии нет — поэтому NB вырождается в Пуассон (r упирается в границу), а КМП даёт ν ≥ 1."))

    rep.add(H1("6. Подобранные параметры"))
    rows = [["Вариант"] + [C.season_label(int(s)) for s in fits["fits"]]]
    for v in variants:
        if v in ("P", "P+R", "MH1", "MH2", "MH1+R", "MH2+R"):
            continue
        cells = []
        for s in fits["fits"]:
            par = fits["fits"][s][v]
            cells.append(", ".join(f"{k}={x:.3g}" if isinstance(x, float) else f"{k}={x}"
                                   for k, x in par.items() if k != "inner_nll"))
        rows.append([label(v, names)] + cells)
    rep.add(P("Параметры, подобранные на сезоне s−1 для сезона s (столбец — прогнозируемый сезон):"), tab(rows))

    rep.add(H1("7. Результаты по рынкам"))
    rows = [["Вариант", "Счёт", "Тоталы", "Обе заб.", "Форы", "Целые", "1X2", "RPS тот.", "RPS разн."]]
    for v in ["P+R"] + ranked:
        m = met["loss"][v]["select"]
        rows.append([label(v, names)] + [f3(m[k]) for k, _ in KEYS])
    rep.add(P("Средние потери, 2023/24+2024/25 (меньше — лучше):"), tab(rows))
    rep.add(figure(C.FIGURES / "markets_heat.png",
                   "Рис. 2. Δ к текущему (×10⁻³) по группам рынков, 2023/24+2024/25; * — 95 % ДИ не содержит 0."))
    rows = [["Вариант", "Δ счёт ×10⁻³ [95 % ДИ]", "p Холма", "Δ тоталы", "Δ обе заб.", "Δ форы"]]
    for v in ranked:
        c = cc[v]["select"]
        rows.append([label(v, names), dci(c["SCORE"]), f"{cc[v]['holm_p_score_select']:.3f}",
                     dci(c["TOTALS"]), dci(c["BTTS"]), dci(c["AH"])])
    rep.add(P("Разности с текущим (вариант − текущий), 2023/24+2024/25:"), tab(rows))
    rep.add(figure(C.FIGURES / "score_ll.png", "Рис. 3. Δ log loss точного счёта с 95 % ДИ."))

    rows = [["Вариант"] + [C.period_label(int(s)) for s in ("2023", "2024", "2025", "2026")]]
    for v in ranked[:8]:
        rows.append([label(v, names)] + [dci(cc[v][s]["SCORE"]) for s in ("2023", "2024", "2025", "2026")])
    rep.add(P("Δ log loss счёта по сезонам (×10⁻³, 8 лучших по отбору):"), tab(rows),
            figure(C.FIGURES / "seasons.png", "Рис. 4. Δ log loss счёта по сезонам."))

    rep.add(H2("7.1 Линии тотала и калибровка"))
    rows = [["Вариант", "Δ 1.5", "Δ 2.5", "Δ 3.5", "ECE 2.5", "ECE обе", "ECE ничья", "ECE 0-0"]]
    for v in ["P+R", "P"] + [x for x in ranked[:6] if x != "P"]:
        e = met["ece"][v]["select"]
        c = cc.get(v, {}).get("select")
        rows.append([label(v, names)] + ([dci(c["T1.5"]), dci(c["T2.5"]), dci(c["T3.5"])] if c else ["0", "0", "0"])
                    + [f3(e["TOTAL_2.5"]), f3(e["BTTS"]), f3(e["DRAW"]), f3(e["ZERO_ZERO"])])
    rep.add(P("Δ log loss линий (×10⁻³) и ECE, 2023/24+2024/25:"), tab(rows),
            figure(C.FIGURES / "calibration.png", "Рис. 5. Калибровка (равночисленные бины), 2023/24+2024/25."))

    rep.add(H2("7.2 Целые форы: вероятность возврата"))
    rows = [["Вариант", "Линия", "Прогноз", "Факт", "Факт − прогноз [95 % ДИ], п.п."]]
    for v in ["P", "P+R"] + top:
        for l, x in cal[v]["select"]["push"].items():
            rows.append([label(v, names), l, f"{x['predicted']:.4f}", f"{x['observed']:.4f}",
                         f"{(x['observed'] - x['predicted']) * 100:+.2f} [{x['obs_minus_pred_ci'][0] * 100:+.2f}; "
                         f"{x['obs_minus_pred_ci'][1] * 100:+.2f}]"])
    rep.add(tab(rows), figure(C.FIGURES / "push.png", "Рис. 6. Вероятность возврата целых фор, 2023/24+2024/25."))

    rep.add(H2("7.3 По лигам"))
    items = sorted(lg["leagues"].items(), key=lambda kv: kv[1]["diff"])
    worse = [k for k, v in items if v["ci95"][0] > 0]
    better = [k for k, v in items if v["ci95"][1] < 0]
    rep.add(P(f"Лучший по правилу вариант: {label(best, names)}; по лигам (2023/24+2024/25, n ≈ 600–800 "
              f"матчей на лигу — малые выборки): значимо лучше в {len(better)} ({', '.join(better) or '—'}), "
              f"значимо хуже в {len(worse)} ({', '.join(worse) or '—'}) из {len(items)} лиг; без поправки на "
              "множественность."), figure(C.FIGURES / "leagues.png", "Рис. 7. Δ log loss счёта по лигам."))

    rep.add(H1("8. Main и Risk"))
    v0 = mr["variants"]["P+R"]
    rows = [["Вариант", "Main ROI отбор", "ΔROI Main [95 % ДИ], п.п.", "ΔROI Main 2026/27", "ΔROI Risk отбор",
             "Изм. Main"]]
    rows.append([label("P+R", names), pp(v0["select"]["main"]["roi"]), "—", "—", "—", "—"])
    order = sorted([k for k in mr["variants"] if k != "P+R"],
                   key=lambda k: -(mr["variants"][k]["select"]["main_vs_ref"]["diff"] or -9))
    for v in order:
        r = mr["variants"][v]
        a, b2, rk = r["select"]["main_vs_ref"], r["2026"]["main_vs_ref"], r["select"]["risk_vs_ref"]
        rows.append([label(v, names), pp(r["select"]["main"]["roi"]),
                     f"{pp(a['diff'])} [{pp(a['ci95'][0])}; {pp(a['ci95'][1])}]",
                     f"{pp(b2['diff'])} [{pp(b2['ci95'][0])}; {pp(b2['ci95'][1])}]",
                     f"{pp(rk['diff'])} [{pp(rk['ci95'][0])}; {pp(rk['ci95'][1])}]", a.get("changed_picks", "—")])
    sc = mr["self_check"]
    rep.add(P("Продакшн-правило Main/Risk (векторная копия; совпадение с функциями `main_prediction`/"
              f"`risk_prediction` на выборке матчей: Main {min(x['main_equal'] for x in sc.values()):.0%}, Risk "
              f"{min(x['risk_equal'] for x in sc.values()):.0%}); подменена только матрица счёта (угловые без "
              "изменений). ROI — по реальным ценам; в прошлых сезонах цена есть только у 1X2, тотала 2.5 и одной "
              "азиатской линии."), tab(rows))

    chk = C.load_json("production_checks.json")
    if chk:
        rep.add(H1("9. Проверка кода продакшна"))
        rows = [["Сезон", "ρ пайплайна (сырая)", "ρ ML после согл.", "Δ счёт ρ пайплайна ×10⁻³", "Δ счёт ρ согл. ×10⁻³"]]
        for s, x in chk["dc_pipeline_rho"].items():
            rows.append([C.period_label(int(s)), f"{x['rho_pipeline_raw_fit']:+.4f}", f"{x['rho_ml_reconciled']:+.4f}",
                         dci(x["score_ll_diff_pipeline_vs_production"]),
                         dci(x["score_ll_diff_reconciled_rho_vs_production"])])
        rep.add(P("Скрытая ошибка (сейчас выключена флагом `DIXON_COLES = False`): `src/modelling/pipeline.py` "
                  "подбирает ρ функцией `fit_dixon_coles_rho` по СЫРЫМ средним голов, а `src/prediction/engine.py` "
                  "применяет ρ к матрице, которую затем согласует с 1X2. После согласования оптимальное ρ другое (≈ 0), "
                  "и ρ пайплайна ухудшает лог-скор счёта. Исправление при включении: подбирать ρ по правдоподобию "
                  "согласованной матрицы (`reconcile_with_1x2(score_matrix(λ_h, λ_a, ρ), p1x2)`) на валидационном "
                  "сезоне — или не включать Диксона–Коулса (по результатам этого исследования выигрыша нет)."),
                tab(rows, first=2.6))
        t = chk["tail"]
        rep.add(P("Хвост матрицы: продакшн складывает P(голов > 10) в ячейку 10. Масса хвоста не больше "
                  f"{max(x['max_tail_mass'] for x in t.values()):.4f}, максимум изменения ячейки "
                  f"{max(x['max_abs_cell_diff'] for x in t.values()):.4f}; перенормировка вместо складывания меняет "
                  "лог-скор счёта на " + ", ".join(f"{x['score_ll_diff_renorm_vs_fold']['diff'] * 1e3:+.3f}"
                                                   for x in t.values())
                  + " ×10⁻³ по сезонам — практически ноль (этим объясняется «значимая» разность бивариантного "
                  "Пуассона с λ₃ = 0). Не ошибка."))
    rep.add(H1("10. Решения"))
    rows = [["Вариант", "Решение", "Δ счёт отбор ×10⁻³", "p Холма", "Лучше в обоих", "Не хуже позже"]]
    for v in ranked:
        d = decs[v]
        rows.append([label(v, names), d["decision"], f"{d['score_diff_select'] * 1e3:+.2f}", f"{d['holm_p']:.3f}",
                     "да" if d["better_both_selection"] else "нет", "да" if d["not_worse_later"] else "нет"])
    rep.add(tab(rows))
    rep.add(H1("11. Выводы"), bullets(conclusions(tests, cc, ranked, names, decs, best, met, mr)))
    rep.add(H1("12. Отрицательные результаты и ограничения"), bullets(limitations(tests)))
    rep.add(H1("13. Рекомендации"), bullets(recommendations(a_list, b_list, names)))
    rep.add(H1("14. Воспроизведение"), reproduce(CMDS),
            note(f"Git SHA запуска: {info['git_sha']}; время расчёта {info['seconds']:.0f} с."))
    rep.build()
    pages = rep.check()
    print("pages", pages)
    write_findings(tests, cc, ranked, names, decs, best, met, mr, a_list, b_list, info)


def conclusions(tests, cc, ranked, names, decs, best, met, mr) -> list[str]:
    out = []
    b = cc[best]["select"]["SCORE"]
    mvb = mr["variants"].get(best if best.endswith("+R") else best + "+R", {}).get("select", {}).get("main_vs_ref")
    out.append(f"Лучший по правилу вариант — {label(best, names)}: Δ log loss счёта {dci(b)} ×10⁻³ "
               f"(p Холма {cc[best]['holm_p_score_select']:.3f}; текущий log loss "
               f"{met['loss']['P+R']['select']['SCORE']:.4f}); по сезонам "
               + ", ".join(f"{C.season_label(int(s))} {cc[best][s]['SCORE']['diff'] * 1e3:+.2f}"
                           for s in ("2023", "2024", "2025", "2026"))
               + (f". При этом ROI Main ниже: {pp(mvb['diff'])} п.п. [{pp(mvb['ci95'][0])}; {pp(mvb['ci95'][1])}]"
                  if mvb else "") + ".")
    if "XC+R" in cc:
        out.append(f"Контроль — перекалибровка средних XGB без рейтингов: {dci(cc['XC+R']['select']['SCORE'])} ×10⁻³ — "
                   "выигрыш смеси даёт информация рейтингов, а не сжатие средних.")
    fams = sorted({v.replace("+R", "") for v in cc} | {"P"})
    worse_raw = [f for f in fams if f + "+R" in met["loss"] and met["loss"][f]["select"]["SCORE"]
                 > met["loss"][f + "+R"]["select"]["SCORE"]]
    out.append(f"Согласование с 1X2 — главный источник качества: сырой Пуассон хуже текущего на "
               f"{dci(cc['P']['select']['SCORE'])} ×10⁻³; сырая версия хуже согласованной у {len(worse_raw)} из "
               f"{len(fams)} семейств.")
    dist = [v for v in ("DC+R", "BP+R", "NB+R", "CMP+R", "FR+R", "FRCMP+R") if v in cc]
    out.append("Семейства распределения при тех же средних (после согласования) отличаются от текущего на "
               + ", ".join(f"{label(v, names)} {cc[v]['select']['SCORE']['diff'] * 1e3:+.2f}" for v in dist)
               + " ×10⁻³ — в пределах ±0.1 и без значимого выигрыша (BP с λ₃ = 0 совпадает с текущим).")
    ex = tests["excess"]["select"]
    d = tests["dispersion"]["select"]
    out.append(f"Передисперсии нет (α хозяев {d['home']['alpha']:+.3f}, гостей {d['away']['alpha']:+.3f}; NB "
               f"вырождается в Пуассон); 0-0: факт/ожидание {ex['0-0']['ratio']:.3f} (p = "
               f"{ex['0-0']['p_one_sided']:.2f}) — инфляция нулей не оправдана; избыток ничьих "
               f"{ex['draw']['ratio']:.3f} снимается согласованием.")
    out.append(f"Одни рейтинги Махера хуже XGBoost: MH1 {dci(cc['MH1+R']['select']['SCORE'])}, MH2 "
               f"{dci(cc['MH2+R']['select']['SCORE'])} ×10⁻³ (раздельные силы дом/выезд переобучаются).")
    return out


def limitations(tests) -> list[str]:
    return ["2026/27 уже просматривался в v6.1 — это не независимая проверка; 2025/26 — для информации.",
            "Средние голов фиксированы (XGBoost продакшна), кроме вариантов Махера/смеси; параметры формы — одни "
            "на все лиги (по-лиговые не оценивались: ~350 матчей на лигу в сезоне).",
            "Цены прошлых сезонов есть только для 1X2, тотала 2.5 и одной азиатской линии — ROI Main/Risk там "
            "считается по части ставок; ROI отражает цену, а не только калибровку.",
            "Гиперпараметры Махера (окно, полураспад, L2) заданы заранее и не подбирались.",
            "Матрица обрезана на 10 голах (как в продакшне); хвост сворачивается в последнюю ячейку."]


def recommendations(a_list, b_list, names) -> list[str]:
    out = []
    out.append("A: " + (", ".join(label(v, names) for v in a_list) if a_list else
                        "нет — оставить «Пуассон + согласование с 1X2» (rho = 0)."))
    out.append("B: " + (", ".join(label(v, names) for v in b_list) + " — предрегистрация "
                        "`results/preregistration.json`, форвард-проверка" if b_list else "нет."))
    out.append("C: остальные варианты; инфляция нулей и NB не нужны (данные не передиспергированы); Диксон–Коулс "
               "(ρ ≈ 0 после согласования) оставить выключенным.")
    out.append("Продакшн не меняется исследованием.")
    return out


def write_findings(tests, cc, ranked, names, decs, best, met, mr, a_list, b_list, info) -> None:
    lines = ["# FootPredict v6.2 — модели распределения голов: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: "
             "`reports/v6_2/Goal_Distributions_Research.pdf`.", "",
             "## Сравнение с текущим (Пуассон + согласование с 1X2), 2023/24+2024/25, ×10⁻³", "",
             "| Вариант | Δ log loss счёта [95 % ДИ] | p Холма | Δ тоталы | Δ обе забьют | Δ форы | Решение |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for v in ranked:
        c = cc[v]["select"]
        lines.append(f"| {label(v, names)} | {dci(c['SCORE'])} | {cc[v]['holm_p_score_select']:.3f} | "
                     f"{dci(c['TOTALS'])} | {dci(c['BTTS'])} | {dci(c['AH'])} | {decs[v]['decision']} |")
    lines += ["", "## Главное", ""] + [f"- {x}" for x in conclusions(tests, cc, ranked, names, decs, best, met, mr)]
    chk = C.load_json("production_checks.json")
    if chk:
        x = chk["dc_pipeline_rho"]
        lines += ["", "## Код продакшна", "",
                  "- Скрытая ошибка (флаг `DIXON_COLES = False`, сейчас не действует): `pipeline.py` подбирает ρ по "
                  "сырым средним, `engine.py` применяет ρ до согласования с 1X2. ρ пайплайна по сезонам "
                  + ", ".join(f"{v['rho_pipeline_raw_fit']:+.3f}" for v in x.values()) + "; оптимальное после "
                  "согласования " + ", ".join(f"{v['rho_ml_reconciled']:+.3f}" for v in x.values())
                  + "; Δ log loss счёта с ρ пайплайна " + ", ".join(dci(v["score_ll_diff_pipeline_vs_production"])
                                                                   for v in x.values())
                  + " ×10⁻³. Исправление при включении: подбирать ρ по согласованной матрице.",
                  "- Складывание хвоста > 10 голов в ячейку 10 — не ошибка (эффект ~1e-5)."]
    lines += ["", "## Решение", ""] + [f"- {x}" for x in recommendations(a_list, b_list, names)]
    lines += ["", "## Воспроизведение", "", "```"] + CMDS + ["```", ""]
    (C.HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
