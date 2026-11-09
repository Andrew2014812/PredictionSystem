"""Builds reports/v6_2/XG_Research.pdf from the result files of experiments/v6_2/xg (Russian) and writes
findings.md next to this file.

    python -m experiments.v6_2.xg.build_pdf

Every number in the report is read from results/*.json|csv; nothing is typed in by hand.
"""
from __future__ import annotations

import json

import pandas as pd

from scripts.report_kit import H1, H2, H3, P, Report, bullets, figure, formula, note, reproduce, table

from . import common as C

R, FIG = C.RESULTS, C.FIGURES
OUT = C.ROOT / "reports" / "v6_2" / "XG_Research.pdf"


def J(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


AUD, COV, SIG, INFO, MET, MAIN = (J(n) for n in ("audit.json", "features_coverage.json", "signal.json",
                                               "variants_info.json", "metrics.json", "main.json"))
BR, REL, CORN, SENS, DEC, PRE, RUN = (J(n) for n in ("breakdowns.json", "reliability.json", "corners.json",
                                                     "sensitivity.json", "decisions.json", "preregistration.json",
                                                     "run_info.json"))
LG = pd.read_csv(R / "audit_by_league.csv")
WK = pd.read_csv(R / "audit_by_week.csv")
LAB = MET["labels"]
VS = MET["vs_B0"]
SUM = MET["summary"]
ORDER = ["B0", "R0", "R0a", "X1", "G1", "X2", "X2a", "X3", "X3a", "X4", "X5", "M1", "M1a", "X2M", "X2Ma", "S0", "S1",
         "T0", "T1"]
ORDER = [v for v in ORDER if v in SUM]
KEY = ["R0a", "X1", "X2a", "X3a", "X5", "M1a", "X2Ma", "S1", "T1"]


# ------------------------------------------------------------------------------------------------ formatting
def W(*cms):
    return [c * 28.35 for c in cms]


class XGReport(Report):
    """The kit's Report with the running header of this series (v6.2) instead of the kit's default."""

    def build(self):
        from scripts.report_kit import _Doc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = _Doc(self.path, f"FootPredict — {self.title}", f"{self.subtitle} — {self.title}")
        doc.multiBuild(self._front() + self.story)
        return self.path


def f(v, d=4, signed=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if abs(v) < 0.5 * 10 ** -d:
        v = 0.0
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def pc(v, d=1, signed=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%"


def pp(v, d=1, signed=True):
    if v is None:
        return "—"
    if abs(v * 100) < 0.5 * 10 ** -d:
        v = 0.0
    return f"{v * 100:+.{d}f} п.п." if signed else f"{v * 100:.{d}f} п.п."


def ci(c, d=4, scale=1.0):
    if not c or c[0] is None:
        return "—"
    return f"[{c[0] * scale:+.{d}f}; {c[1] * scale:+.{d}f}]"


def dci(d, d_=4):
    return f"{f(d['mean'], d_, True)} {ci(d['ci95'], d_)}"


def nreq(v):
    if v is None:
        return "—"
    return f"{v:,.0f}".replace(",", " ")


def sig_list(metric, vs):
    H = MET["holm"][metric]
    worse = [v for v in ORDER[1:] if H.get(v, 1) < 0.05 and vs[v][metric]["mean"] > 0]
    return ", ".join(worse) or "нет"


def name(v):
    return f"{v} — {LAB.get(v, v)}"


# ------------------------------------------------------------------------------------------------ sections
def intro(rep: Report) -> None:
    ov = AUD["overall"]
    x5 = VS["X5"]
    rep.add(H1("1. Цель исследования"),
            P("Выяснить, может ли ожидаемое количество голов (xG) и пропущенных моментов (xGA) реально улучшить прогнозы "
              "FootPredict: голы хозяев и гостей, исход 1X2, тоталы, «обе забьют», форы, точный счёт, качество "
              "выбранного Main и доходность при реальных коэффициентах. Особый вопрос — уменьшает ли xG переоценку "
              "вероятности выбранного Main (эффект «проклятия победителя», найденный в исследовании v6.1)."),
            H1("2. Проблема и гипотезы"),
            P(f"xG появились в файлах Football-Data только в сезоне 2026/27 (в {len(AUD['xg_leagues'])} из "
              f"{len(AUD['xg_leagues']) + len(AUD['no_xg_leagues'])} лиг системы). Модели v6.1 обучены на сезонах "
              "2021/22–2025/26, где xG нет. Поэтому история xG короткая (с "
              f"{ov['first_date']} по {ov['last_date']}, {ov['matches_with_xg']} матчей), и xG нельзя просто добавить "
              "в признаки XGBoost: обучать нечему. Исследование проверяет, как использовать короткую историю честно."),
            *bullets([
                "**H1.** Предматчевые признаки xG (за последние матчи, с поправкой на соперника) уменьшают ошибку прогноза голов (отклонение Пуассона, RMSE, MAE).",
                "**H2.** Они улучшают вероятности 1X2 (log loss, Brier, калибровка) и производных рынков (тоталы, обе забьют, форы, точный счёт).",
                "**H3.** Они уменьшают переоценку выбранного Main и улучшают ROI при реальных ценах.",
                "**H4.** Внутри сезона xG информативнее голов (рейтинги по xG против тех же рейтингов по голам).",
                "**H5.** Информацию xG можно перенести в длинную историю через удары (прокси xG по ударам и ударам в створ).",
            ]),
            note(f"Короткий ответ: ни один вариант не дал значимого улучшения после поправки Холма. Лучший — X5 "
                 f"(смесь ставок голов v6.1 с рейтингом xG, один параметр): отклонение голов {dci(x5['goals_dev'])}, "
                 f"log loss 1X2 {dci(x5['logloss_1x2'])} на {MET['n_matches']} матчах. Решение — B (нужна форвард-проверка); "
                 "предрегистрация записана в `results/preregistration.json`."))


def data_section(rep: Report) -> None:
    an = AUD["anomalies"]
    ov = AUD["overall"]
    rep.add(H1("3. Данные и аудит xG"),
            H2("3.1. Источники"),
            *bullets([
                "`local_data/results/<сезон>/<лига>.csv` — файлы Football-Data; xG в колонках `HxG`, `AxG`.",
                "`local_data/processed/features.parquet` — все матчи с предматчевыми признаками v6.1 (без xG).",
                f"Модели v6.1 — набор `{C.PRODUCTION_VERSION}` (обучен на 2021/22–2025/26); его прогнозы на 2026/27 — база B0.",
                "Цены: `markets.parquet` (цены, записанные при прогнозе) и архив OddsPapi (`odds.parquet`, последняя котировка строго до начала матча). Среднее по рынку Football-Data не считается ценой букмекера и отброшено.",
            ]),
            H2("3.2. Есть ли xG в прошлых сезонах"),
            table([["Сезон", "Файлов", "С колонками HxG/AxG"]] +
                  [[r["season_code"][:2] + "/" + r["season_code"][2:], r["files"], r["with_xg"]] for r in AUD["seasons"]]),
            note("В файлах 2021/22–2025/26 колонок xG нет ни в одной лиге. Исторически xG использовать нельзя; "
                 "заполнять прошлые сезоны выдуманными xG исследование не стало."),
            H2("3.3. Заполненность по лигам"))
    rows = [["Лига", "Колонка xG", "Сыграно в системе", "Строк в файле", "С xG", "Доля", "Последний матч",
             "Последний с xG"]]
    for r in LG.itertuples():
        rows.append([r.league, "да" if r.has_xg_column else "нет", r.played_in_system, r.rows_in_file, r.with_xg,
                     pc(r.fill_rate, 0), r.last_played, r.last_xg if isinstance(r.last_xg, str) else "—"])
    rep.add(table(rows),
            P(f"В лигах {', '.join(AUD['no_xg_leagues'])} колонок xG нет вовсе. В остальных лигах xG есть у каждой "
              "строки файла; недостающие матчи — самые свежие, которые система уже получила от API, а Football-Data "
              f"ещё не опубликовала. Всего в лигах с xG сыграно {AUD['played_in_xg_leagues']} матчей, xG есть у "
              f"{AUD['played_with_xg']} ({pc(AUD['played_with_xg'] / AUD['played_in_xg_leagues'])})."),
            *figure(FIG / "audit_by_week.png", "Рис. 1. Сыгранные матчи лиг с xG и матчи с xG по неделям 2026/27."),
            H2("3.4. Аномалии"))
    lab = {"rows_in_files": "Строк в файлах 2026/27 (включённые лиги)", "rows_in_leagues_with_xg_column": "Строк в лигах с колонкой xG",
           "both_xg_present": "Оба значения xG есть", "only_one_side_present": "Есть только одно значение",
           "non_numeric_values": "Нечисловые значения", "negative": "Отрицательные xG",
           "above_5": "xG команды > 5", "above_8_impossible": "xG > 8 (невозможные)", "exact_zero": "xG = 0",
           "total_above_7": "Сумма xG матча > 7", "xg_but_no_goals": "xG есть, голов нет",
           "goals_but_no_xg_in_xg_leagues": "Голы есть, xG нет (лиги с xG)", "duplicate_match_ids": "Дубликаты матчей",
           "goal_minus_xg_above_4": "|голы − xG| > 4", "xg_without_shots": "xG есть, ударов нет",
           "xg_positive_with_zero_shots": "xG > 0 при 0 ударов", "zero_xg_with_goals": "xG = 0, но гол забит",
           "sot_exceeds_shots": "Ударов в створ больше, чем ударов", "goals_exceed_sot": "Голов больше, чем ударов в створ"}
    rows = [["Проверка", "Число"]] + [[lab.get(k, k), v] for k, v in an.items()]
    rows2 = [["Матч", "xG хоз.", "xG гост.", "Голы", "Удары", "В створ"]]
    for e in AUD["extreme_examples"][:6]:
        rows2.append([e["match_id"], f(e["xg_home"], 2), f(e["xg_away"], 2), f"{e['home_goals']:.0f}:{e['away_goals']:.0f}",
                      f"{e['home_shots']:.0f}–{e['away_shots']:.0f}", f"{e['home_sot']:.0f}–{e['away_sot']:.0f}"])
    rep.add(table(rows, widths=None),
            P("Невозможных значений (отрицательных, нечисловых, больше 8) нет. Крупные значения подтверждаются ударами "
              "и счётом (примеры ниже), поэтому они оставлены. «Голов больше, чем ударов в створ» объясняется автоголами. "
              "В очищенные данные входит матч, у которого оба xG в диапазоне 0–8 и известен счёт."),
            table(rows2),
            H2("3.5. Сопоставимость xG между лигами"))
    rows = [["Лига", "Матчей", "xG / матч", "Голы / матч", "Голы / xG [95 % ДИ]", "Корр.", "Наклон", "Псевдо-R²",
             "Доля хоз. xG", "Доля хоз. голов"]]
    for r in LG.dropna(subset=["ratio"]).sort_values("ratio").itertuples():
        rows.append([r.league, int(r.matches), f(r.xg_total, 2), f(r.goals_total, 2),
                     f"{r.ratio:.3f} [{r.ratio_lo:.2f}; {r.ratio_hi:.2f}]", f(r.corr, 2), f(r.slope, 2), f(r.pseudo_r2, 2),
                     pc(r.home_share_xg, 0), pc(r.home_share_goals, 0)])
    het = ov["ratio_heterogeneity"]
    rep.add(table(rows, widths=W(1.1, 1.5, 1.4, 1.5, 3.3, 1.2, 1.5, 1.8, 1.8, 1.9)),
            P(f"В целом голов на одну команду в матче {f(ov['mean_goals'], 3)} при xG {f(ov['mean_xg'], 3)} — отношение "
              f"{f(ov['ratio'], 3)}, корреляция по командо-матчам {f(ov['corr'], 3)}. Отношение голов к xG по лигам — от "
              f"{f(ov['ratio_spread_leagues'][0], 3)} до {f(ov['ratio_spread_leagues'][1], 3)}, но интервалы широкие: "
              f"проверка однородности (хи-квадрат {f(het['chi2'], 1)}, {het['df']} ст. св., p = {f(het['p'], 2)}) не "
              "отвергает одну общую шкалу. Значит, xG разных лиг на этой выборке сопоставимы; отличия лиг — в уровне "
              "результативности, а не в шкале xG."),
            *figure(FIG / "audit_ratio_by_league.png", "Рис. 2. Отношение голов к xG по лигам с 95 % бутстреп-интервалами."),
            *figure(FIG / "audit_goals_vs_xg.png", "Рис. 3. Средние голы команды в корзинах её xG (после матча).", width=11.5 * 28.35))
    sv, b0 = AUD["shots_vs_xg"], AUD["b0_vs_xg"]
    rep.add(H2("3.6. Что xG добавляет к ударам и к модели v6.1"),
            P(f"Линейная регрессия xG команды на удары и удары в створ объясняет R² = {f(sv['r2_xg_on_shots_sot'], 3)} "
              f"разброса xG ({sv['n_team_matches']} командо-матчей). Корреляция голов с xG {f(sv['corr_goals_xg'], 3)}, "
              f"с ударами в створ {f(sv['corr_goals_sot'], 3)}, с ударами {f(sv['corr_goals_shots'], 3)}. "
              "Удары в створ уже есть в признаках v6.1, поэтому значительная часть информации xG модели известна."),
            P(f"Предматчевая ставка голов v6.1 (λ) коррелирует с xG матча сильнее ({f(b0['corr_lambda_xg'], 3)}), чем с "
              f"голами ({f(b0['corr_lambda_goals'], 3)}): xG — менее шумная мишень. Это описательная (послематчевая) "
              "характеристика, а не признак модели."))


def features_section(rep: Report) -> None:
    bm = COV["by_month"]
    rep.add(H1("4. Предматчевые признаки xG"),
            P("Для матча дня d используются только матчи той же лиги, сыгранные в более ранние дни (d′ < d): ни xG "
              "самого матча, ни матчи того же дня в признаки не попадают. Матчи без опубликованного xG пропускаются."),
            *bullets([
                "`xg_n` — число матчей команды с xG до дня матча (начало сезона: 0, 1, 2, …);",
                "`xgf5`, `xga5`, `xgf10`, `xga10`, `xgd10` — средние xG за / против за 5 и 10 матчей и их разность;",
                "`gmx10`, `gamx10` — средние (голы − xG) и (пропущенные − xGA): реализация и «везение» вратаря;",
                "`xgf_v5`, `xga_v5` — то же только дома (для хозяев) или только в гостях (для гостей);",
                "`xgf_ewm`, `xga_ewm` — экспоненциально взвешенные средние (полураспад 4 матча);",
                "`xgf_s`, `xga_s`, `gmx_s` — сжатые к среднему лиги версии (начало сезона), см. формулы ниже;",
                "`att`, `dfn`, `exp_xg_h`, `exp_xg_a` — рейтинги атаки и обороны с поправкой на соперника и ожидаемые xG матча;",
                "`gatt`, `gdfn`, `exp_g_h`, `exp_g_a` — те же рейтинги по голам (контроль).",
            ]),
            H3("Сжатие к среднему лиги (начало сезона)"),
            formula("xgf_s = (n · m₁₀ + k · prior) / (n + k),   k = 3,   prior = средний xG команды в лиге до дня d"),
            formula("gmx_s = n · gmx₁₀ / (n + 5)"),
            H3("Рейтинги с поправкой на соперника"),
            formula("E[xG_{хоз}] = exp(μ + h + att_{хоз} − dfn_{гост}),   E[xG_{гост}] = exp(μ + att_{гост} − dfn_{хоз})"),
            formula("min Σ (E − y · ln E) + λ/2 · (Σ att² + Σ dfn²),   λ = 5"),
            P("Это квазиправдоподобие Пуассона (оно корректно для непрерывной неотрицательной величины, такой как xG). "
              "Рейтинги пересчитываются для каждого игрового дня каждой лиги по более ранним матчам; гребневый штраф λ "
              "тянет команду с малым числом матчей к среднему лиги (att = dfn = 0). Рейтинги определяются, когда в лиге "
              "есть хотя бы 6 более ранних матчей с xG. Все константы (k, λ, полураспад, минимум матчей) заданы до оценки."),
            H2("4.1. Начало сезона"))
    rows = [["Месяц", "Матчей", "Рейтинги есть", "Мин. xg_n (среднее)", "Обе ≥ 3 матчей", "Обе ≥ 5 матчей"]]
    for r in bm:
        rows.append([r["month"], r["matches"], pc(r["ratings"], 0), f(r["min_n_mean"], 1), pc(r["ge3"], 0), pc(r["ge5"], 0)])
    be = COV["band_eval"]
    rep.add(table(rows),
            P(f"В августе рейтинги есть только у {pc(bm[1]['ratings'], 0)} матчей, и почти ни у одной пары команд нет "
              "трёх матчей с xG. Поэтому август — только разогрев (история признаков и первые подгонки), а оценка "
              f"начинается с {MET['eval_period'][0]}. В оценочных матчах ({COV['eval_matches']}) рейтинги есть у "
              f"{pc(COV['eval_ratings_share'], 0)}, медиана минимального `xg_n` — {f(COV['eval_min_n_median'], 0)}; "
              f"у {be['1-2']} матчей у одной из команд всего 1–2 матча с xG."),
            *figure(FIG / "coverage_season_start.png", "Рис. 4. Сколько истории xG есть до матча, по месяцам."))


def sig_pairs():
    out = []
    for k, d in MET["pairs"].items():
        for m, lab in (("goals_dev", "голы"), ("logloss_1x2", "1X2")):
            lo, hi = d[m]["ci95"]
            if lo is not None and (lo > 0 or hi < 0):
                out.append(f"{k.replace('-', ' − ')} ({lab}, {d[m]['mean']:+.4f})")
    return ", ".join(out) or "—"


def best_ll():
    return min(ORDER[1:], key=lambda v: VS[v]["logloss_1x2"]["mean"])


def top_signal():
    return max(SIG["features"], key=lambda r: abs(r["resid"]["r"] or 0))


def signal_section(rep: Report) -> None:
    rows = [["Признак", "Тип", "Мишень", "Корр. с остатком v6.1 [95 % ДИ]", "Корр. с результатом"]]
    for r in SIG["features"]:
        rr = r["resid"]
        rows.append([r["label"], r["kind"], r["target"], f"{f(rr['r'], 3, True)} [{f(rr['lo'], 3, True)}; {f(rr['hi'], 3, True)}]",
                     f(r["raw"]["r"], 3, True)])
    h = SIG["xg_vs_goals"]
    rep.add(H1("5. Есть ли в xG сигнал, которого нет у v6.1"),
            P(f"Матчи оценки ({SIG['n']}). Остаток голов хозяев — голы − λ v6.1; остаток 1X2 — (1[H] − p_H) − (1[A] − p_A). "
              "Признак, полезный для v6.1, должен коррелировать с остатком, а не только с результатом."),
            table(rows, widths=[5.2 * 28.35, 1.4 * 28.35, 2.4 * 28.35, 5.0 * 28.35, 3.0 * 28.35]),
            *figure(FIG / "signal_residuals.png", "Рис. 5. Корреляция признаков с ошибкой v6.1 (зелёный — xG, жёлтый — признаки, которые v6.1 уже использует)."),
            P(f"Сильнее всего с остатком связан признак «{top_signal()['label']}» ({f(top_signal()['resid']['r'], 3, True)}), "
              f"но и признаки, которые v6.1 уже использует, имеют остаточную корреляцию того же порядка (до "
              f"{f(max(abs(r['resid']['r']) for r in SIG['features'] if r['kind'] != 'xG'), 3)}): в 2026/27 v6.1 немного "
              "недооценивает разброс силы команд, и xG-признак отчасти исправляет именно это."),
            H2("5.1. xG против голов на одинаковом окне"),
            P(f"На {h['n']} матчах, где у обеих команд не менее 5 матчей с xG, разница xGD за 10 матчей коррелирует с "
              f"разницей голов матча на {f(h['xgd10_diff']['r'], 3)}, а разница голов за те же 10 матчей — на "
              f"{f(h['gd10_diff']['r'], 3)}. Разность корреляций {f(h['diff_r'], 3, True)} {ci(h['diff_ci95'], 3)}; частная "
              f"корреляция xG при известной разнице голов {f(h['partial_xg_given_gd'], 3)}. xG немного информативнее "
              "голов на коротком окне (подтверждает H4 на уровне признаков), но разница на грани значимости."),
            P(f"Реализация (голы − xG) по смыслу должна возвращаться к среднему: её корреляция с остатком голов "
              f"{f(SIG['finishing_regression']['r'], 3, True)} {ci([SIG['finishing_regression']['lo'], SIG['finishing_regression']['hi']], 3)} — "
              "знак ожидаемый, но эффект не отличим от нуля."))


def methods_section(rep: Report) -> None:
    rows = [["Код", "Вариант", "Голы", "1X2", "Обучение"]]
    desc = {
        "B0": ("v6.1 как есть", "классификатор v6.1", "2021/22–2025/26"),
        "R0": ("GLM: a + b·ln λ_{B0}", "MNL со смещением ln p_{B0}", "2026/27, еженедельно"),
        "R0a": ("то же, все коэф. сжаты к B0", "то же, сжато к B0", "2026/27, еженедельно"),
        "X1": ("рейтинги xG, масштаб по GLM", "Пуассон из матрицы", "2026/27, еженедельно"),
        "G1": ("рейтинги по голам", "Пуассон из матрицы", "2026/27, еженедельно"),
        "X2": ("GLM: B0 + 4 признака xG (гребень)", "перенос изменения голов", "2026/27, еженедельно"),
        "X2a": ("X2, все коэф. сжаты к B0", "перенос", "2026/27, еженедельно"),
        "X3": ("GLM: B0 + ln E[xG]", "перенос", "2026/27, еженедельно"),
        "X3a": ("X3, сжато к B0", "перенос", "2026/27, еженедельно"),
        "X4": ("XGBoost по остатку (base margin ln λ_{B0})", "перенос", "2026/27, еженедельно"),
        "X5": ("(1 − w)·ln λ_{B0} + w·ln E[xG], один w", "перенос", "2026/27, еженедельно"),
        "M1": ("как B0", "MNL + 4 признака xG", "2026/27, еженедельно"),
        "M1a": ("как B0", "M1, сжато к B0", "2026/27, еженедельно"),
        "X2M": ("как X2", "как M1", "2026/27, еженедельно"),
        "X2Ma": ("как X2a", "как M1a", "2026/27, еженедельно"),
        "S0": ("XGBoost v6.1, только 2026/27", "XGBoost v6.1, только 2026/27", "2026/27, еженедельно"),
        "S1": ("S0 + 20 признаков xG", "S0 + признаки xG", "2026/27, еженедельно"),
        "T0": ("XGBoost v6.1 заново", "заново", "2021/22–2025/26"),
        "T1": ("T0 + прокси xG по ударам", "T0 + прокси", "2021/22–2025/26 (+ отображение по августу 2026)"),
    }
    for v in ORDER:
        d = desc[v]
        rows.append([v, LAB.get(v, v), d[0], d[1], d[2]])
    wk = INFO["in_season"]["weeks"]
    rows_w = [["Неделя с", "Обучающих матчей", "Матчей недели"]] + [[w["week"], w["train"], w["test"]] for w in wk]
    mp = INFO["proxy_mapping"]
    rep.add(H1("6. Методы"),
            H2("6.1. Варианты"),
            table(rows, widths=[1.1 * 28.35, 4.3 * 28.35, 4.6 * 28.35, 3.6 * 28.35, 3.4 * 28.35]),
            P(f"Всего вариантов: {MET['n_variants'] - 1} против базы (из них контроли без xG: R0, R0a, G1, S0, T0). "
              "Варианты R0a, X2a, X3a, M1a, X2Ma и X5 добавлены после первого прогона: в первой спецификации "
              "свободные (несжатые) коэффициенты перекалибровки были шумнее самой базы даже без xG. Они входят в "
              "число проверенных вариантов и в поправку Холма."),
            H2("6.2. Схема оценки: расширяющееся окно по неделям"),
            P(f"Модели внутри сезона переобучаются каждые {C.REFIT_DAYS} дней на всех матчах 2026/27, сыгранных до "
              "первого дня прогнозируемой недели, и прогнозируют только эту неделю. Длинные модели (B0, T0, T1) обучены "
              "один раз на 2021/22–2025/26. Все варианты сравниваются на одних и тех же матчах: "
              f"{MET['n_matches']} сыгранных матчей {len(MET['leagues'])} лиг с xG, "
              f"{MET['eval_period'][0]} – {MET['eval_period'][1]}."),
            table(rows_w),
            H2("6.3. Формулы"),
            formula("отклонение Пуассона: D = 2 · [y · ln(y/μ) − (y − μ)],   на матч: D_{хоз} + D_{гост}"),
            formula("log loss = −ln p_{y},   Brier = Σ_{k} (p_{k} − 1[y = k])²,   ECE = Σ_{b} (n_{b}/n) · |ō_{b} − p̄_{b}|"),
            formula("GLM со сжатием к B0: ln μ = ln λ_{B0} + a + (b − 1) · ln λ_{B0} + Σ c_{j} z_{j},   штраф 10/2 · (a² + (b−1)² + Σ c²)"),
            formula("MNL: η_{k} = ln p_{B0,k} + β_{k}ᵀ x,   k ∈ {H, A},   η_{D} = ln p_{B0,D},   p = softmax(η)"),
            formula("X5: ln μ = (1 − w) · ln λ_{B0} + w · ln E[xG],   w ∈ [0, 1] — максимум правдоподобия Пуассона"),
            formula("перенос на 1X2: p_{k} ∝ p_{B0,k} · P_{k}(μ_{хоз}, μ_{гост}) / P_{k}(λ_{хоз}, λ_{гост})"),
            formula(f"прокси xG (T1): xG ≈ {mp['a']:+.3f} + {mp['b_shots']:.3f} · удары + {mp['c_sot']:.3f} · в створ   "
                    f"(R² = {mp['r2']:.3f}, {mp['n']} командо-матчей августа)"),
            P("Признаки z для голов хозяев: ln(E[xG]/среднее лиги), ln(xgf_s/prior), ln(xga_s соперника/prior), gmx_s "
              "(для гостей — симметрично). Признаки x для 1X2: ln-отношения p_{H}/p_{D}, p_{A}/p_{D} базы и четыре "
              "разности xG (рейтинги, сжатые xG, суммарный уровень, реализация). Рынки строятся производственным "
              "путём: матрица счёта Пуассона, согласованная с вероятностями 1X2 (`reconcile_with_1x2`); угловые не "
              "меняются. Исход 1X2 называется правилом сайта: ничья, если P(D) ≥ 0.28 и |P(H) − P(A)| ≤ 0.20, иначе "
              "наибольшая вероятность (`src/modelling/decision.py`). Main выбирается производственным правилом "
              "(`src/markets/recommendation.py`) по вероятностям варианта и реальным ценам до начала матча."),
            H2("6.4. Статистика"),
            *bullets([
                f"95 % бутстреп-интервалы ({C.N_BOOT} повторов, ресэмплинг матчей), разности — парные по матчу.",
                "ROI = прибыль / число ставок с ценой; разность ROI бутстрепится как отношение сумм по тем же матчам.",
                f"Поправка Холма по {MET['n_variants'] - 1} вариантам для трёх основных метрик: отклонение голов, log loss 1X2, ROI Main.",
                "Размер выборки для мощности 80 % (двусторонний, α = 0.05): n = ((z_{0.975} + z_{0.8}) · σ_{d} / δ)², где δ — наблюдаемая разность, σ_{d} — разброс разности по матчам.",
                f"Параметры: n_jobs = {C.N_JOBS}, seed = {C.SEED}; все гиперпараметры заданы до оценки и на неделях оценки не подбирались.",
            ]))


def results_section(rep: Report) -> None:
    rep.add(H1("7. Результаты"), H2("7.1. Голы хозяев и гостей"))
    rows = [["Код", "Откл. (сумма)", "Разность с B0 [95 % ДИ]", "RMSE хоз.", "RMSE гост.", "MAE хоз.", "MAE гост.", "RMSE тотал"]]
    for v in ORDER:
        s = SUM[v]
        rows.append([v, f(s["goals_dev"]), dci(VS[v]["goals_dev"]) if v in VS else "база", f(s["rmse_h"]), f(s["rmse_a"]),
                     f(s["mae_h"]), f(s["mae_a"]), f(s["rmse_total"])])
    rep.add(table(rows, widths=W(1.2, 1.8, 4.6, 1.9, 1.9, 1.8, 1.9, 1.9)),
            *figure(FIG / "forest_main_metrics.png", "Рис. 6. Разность основных метрик с v6.1 (красный — значимо хуже, зелёный — значимо лучше)."),
            P(f"Точечная оценка лучше базы только у: {', '.join(v for v in ORDER[1:] if VS[v]['goals_dev']['mean'] < 0) or 'нет'} "
              f"(X5: {dci(VS['X5']['goals_dev'])}, интервал включает ноль). "
              f"Чистые рейтинговые модели заметно хуже базы (X1 {dci(VS['X1']['goals_dev'])}), но рейтинги по xG лучше "
              f"тех же рейтингов по голам (X1 − G1 {dci(MET['pairs']['X1-G1']['goals_dev'])}) — внутри сезона xG "
              "информативнее голов (H4), однако без длинной истории этого мало. XGBoost только на 2026/27 (S0, S1) и "
              "XGBoost по остатку (X4) переобучаются и значимо хуже базы."))
    rep.add(H2("7.2. Исход 1X2"))
    rows = [["Код", "Log loss", "Разность [95 % ДИ]", "Brier", "Точн. (правило)", "Точн. (argmax)", "Macro F1",
             "Разн. F1 [95 % ДИ]", "ECE"]]
    for v in ORDER:
        s = SUM[v]
        rows.append([v, f(s["logloss_1x2"]), dci(VS[v]["logloss_1x2"]) if v in VS else "база", f(s["brier_1x2"]),
                     pc(s["acc_rule"]), pc(s["acc_argmax"]), f(s["macro_f1"], 3),
                     (f"{VS[v]['macro_f1']['diff']:+.3f} {ci(VS[v]['macro_f1']['ci95'], 3)}" if v in VS else "—"),
                     f(s["ece_1x2"], 3)])
    rep.add(table(rows, widths=W(1.1, 1.4, 3.7, 1.3, 1.8, 1.8, 1.3, 3.4, 1.1)),
            P(f"По log loss 1X2 ни один вариант не лучше базы значимо; лучший по точечной оценке — {best_ll()} "
              f"({dci(VS[best_ll()]['logloss_1x2'])}). Коррекции 1X2 с признаками xG "
              f"(M1, M1a) хуже базы ({dci(VS['M1a']['logloss_1x2'])}). Точность и macro F1 у X5 и T1 чуть выше "
              f"(X5: F1 {VS['X5']['macro_f1']['diff']:+.3f} {ci(VS['X5']['macro_f1']['ci95'], 3)}), но интервалы включают ноль."))
    rep.add(H2("7.3. Рынки: тоталы, обе забьют, форы, точный счёт"))
    rows = [["Код", "Тотал 1.5", "Тотал 2.5", "Тотал 3.5", "Обе забьют", "Фора −2.5", "Фора −1.5", "Фора +1.5",
             "Фора +2.5", "Точный счёт", "Топ-5 счёт"]]
    for v in ORDER:
        s = SUM[v]
        rows.append([v, f(s["ll_over1.5"]), f(s["ll_over2.5"]), f(s["ll_over3.5"]), f(s["ll_btts"]), f(s["ll_ah-2.5"]),
                     f(s["ll_ah-1.5"]), f(s["ll_ah+1.5"]), f(s["ll_ah+2.5"]), f(s["ll_exact"]), pc(s["exact_top5"])])
    rep.add(P("Log loss бинарных рынков (форы — с точки зрения хозяев, половинные линии) и точного счёта; топ-5 — доля "
              "матчей, где счёт вошёл в пять самых вероятных."), table(rows, widths=W(1.1, *[1.59] * 10)))
    rows = [["Код", "Тотал 2.5 [95 % ДИ]", "Обе забьют [95 % ДИ]", "Фора −1.5 [95 % ДИ]", "Точный счёт [95 % ДИ]"]]
    for v in KEY:
        d = VS[v]
        rows.append([v, dci(d["ll_over2.5"]), dci(d["ll_btts"]), dci(d["ll_ah-1.5"]), dci(d["ll_exact"])])
    rep.add(P("Разности log loss с v6.1 для основных вариантов:"), table(rows, widths=W(1.2, 3.95, 3.95, 3.95, 3.95)),
            *figure(FIG / "forest_markets.png", "Рис. 7. Разности log loss тотала 2.5 и точного счёта с v6.1."),
            P(f"X5 улучшает тотал 2.5 и точный счёт по точечной оценке ({dci(VS['X5']['ll_over2.5'])}; "
              f"{dci(VS['X5']['ll_exact'])}), но не значимо. Значимых улучшений рынков нет ни у одного варианта."))
    rep.add(H2("7.4. Вклад именно xG: сравнение с контролем без xG"))
    lab = {"X2a-R0a": "X2a − R0a", "X3a-R0a": "X3a − R0a", "X5-B0": "X5 − B0", "M1a-R0a": "M1a − R0a",
           "X2Ma-R0a": "X2Ma − R0a", "X2-R0": "X2 − R0", "M1-R0": "M1 − R0", "X1-G1": "X1 − G1 (xG против голов)",
           "S1-S0": "S1 − S0", "T1-T0": "T1 − T0"}
    rows = [["Пара", "Откл. голов [95 % ДИ]", "Log loss 1X2 [95 % ДИ]", "Тотал 2.5", "Обе забьют"]]
    for k, d in MET["pairs"].items():
        rows.append([lab.get(k, k), dci(d["goals_dev"]), dci(d["logloss_1x2"]), f(d["ll_over2.5"]["mean"], 4, True),
                     f(d["ll_btts"]["mean"], 4, True)])
    rep.add(P("Перекалибровка на 2026/27 без xG (R0, R0a) сама по себе ухудшает базу: подгонка по нескольким сотням "
              "матчей шумная. Поэтому эффект xG надо мерить и против базы, и против такого же контроля без xG."),
            table(rows, widths=W(3.6, 4.1, 4.1, 2.6, 2.6)),
            P(f"Относительно своего контроля без xG разности малы: интервал исключает ноль только в сравнениях "
              f"{sig_pairs()}. Для M1a − R0a разность по голам отражает не xG, а то, что M1a оставляет голы базы, "
              "а R0a их перекалибрует. Рейтинги по xG значимо лучше рейтингов по голам (X1 − G1), что подтверждает H4."))


def stats_section(rep: Report) -> None:
    H = MET["holm"]
    rows = [["Код", "p откл.", "Холм", "p log loss", "Холм", "p ROI", "Холм", "n для откл.", "n для log loss", "n для ROI"]]
    for v in ORDER[1:]:
        d = VS[v]
        mr = MAIN["vs_B0"][v]
        rows.append([v, f(d["goals_dev"]["p"], 3), f(H["goals_dev"].get(v), 3), f(d["logloss_1x2"]["p"], 3),
                     f(H["logloss_1x2"].get(v), 3), f(mr["roi"]["p"], 3), f(H["roi_main"].get(v), 3),
                     nreq(d["goals_dev"]["n_required"]), nreq(d["logloss_1x2"]["n_required"]), nreq(mr["n_required_roi"])])
    rep.add(H2("7.5. Поправка Холма и размер выборки"),
            P(f"Двусторонние бутстреп-p-значения разности с базой, поправка Холма по {MET['n_variants'] - 1} вариантам. "
              "«n» — сколько матчей нужно, чтобы наблюдаемая разница (любого знака) стала значимой с мощностью 80 %."),
            table(rows, widths=W(1.2, 1.4, 1.4, 1.6, 1.4, 1.4, 1.4, 2.0, 2.2, 3.0)),
            *figure(FIG / "sample_size.png", "Рис. 8. Требуемое число матчей для наблюдаемых разностей (логарифмическая шкала)."),
            P(f"Значимы после поправки (p_Холм < 0.05) только ухудшения: по голам — {sig_list('goals_dev', VS)}, по 1X2 — "
              f"{sig_list('logloss_1x2', VS)}; значимых улучшений нет ни по одной основной метрике. Для эффекта X5 по отклонению голов нужно "
              f"около {nreq(VS['X5']['goals_dev']['n_required'])} матчей — в несколько раз больше, чем есть "
              f"({MET['n_matches']}); для разности ROI X5 — около {nreq(MAIN['vs_B0']['X5']['n_required_roi'])}."))


TW = W(1.7, 1.7, 4.4, 1.9, 1.6, 1.6, 2.1, 2.0)


def time_section(rep: Report) -> None:
    def rows_for(key):
        rows = [["Группа", "Матчей", "X5: голы [95 % ДИ]", "X5: log loss", "X3a: голы", "R0a: голы", "X5: ΔROI", "ROI B0"]]
        for r in BR[key]:
            rows.append([r["group"], r["n"], dci(r["X5_dgoals"]), f(r["X5_dll"]["mean"], 4, True), f(r["X3a_dgoals"]["mean"], 4, True),
                         f(r["R0a_dgoals"]["mean"], 4, True), pp(r["X5_droi"]), pc(r["B0_roi"])])
        return rows
    rep.add(H2("7.6. Время: месяцы и объём истории xG"),
            P("По месяцам оценки:"), table(rows_for("month"), widths=TW),
            P("По минимальному числу матчей с xG у двух команд (`min xg_n`):"), table(rows_for("band"), widths=TW),
            P(f"Если бы xG «включался» по мере накопления истории, выигрыш рос бы к ноябрю и в группе «8+». Этого не "
              f"видно: разность X5 по голам отрицательна в {sum(r['X5_dgoals']['mean'] < 0 for r in BR['month'])} из "
              f"{len(BR['month'])} месяцев и в {sum(r['X5_dgoals']['mean'] < 0 for r in BR['band'])} из {len(BR['band'])} "
              "полос истории без явного роста, а X3a и R0a колеблются вокруг нуля; все интервалы включают ноль."),
            *figure(FIG / "cumulative_difference.png", "Рис. 9. Накопленная разность метрик (вариант − v6.1) по матчам в порядке дат."),
            *figure(FIG / "coefficient_paths.png", "Рис. 10. Коэффициенты по неделям: вес w у X5 стабилен, коэффициенты X2a меняют знак."),
            P(f"Вес рейтинга xG в X5 после первой недели держится около {f(INFO['in_season']['coefs']['X5_w'][-1], 2)}; "
              "в X2a коэффициенты отдельных признаков xG меняют знак от недели к неделе (сильная коллинеарность рейтинга, "
              "сжатых средних и реализации) — поэтому многопараметрические коррекции проигрывают одному весу."))
    rows = [["Лига", "Матчей", "Откл. B0", "X5: голы [95 % ДИ]", "X5: log loss", "X3a: голы", "ROI B0", "X5: ΔROI"]]
    for r in BR["league"]:
        rows.append([r["group"], r["n"], f(r["B0_goals_dev"], 3), dci(r["X5_dgoals"]), f(r["X5_dll"]["mean"], 4, True),
                     f(r["X3a_dgoals"]["mean"], 4, True), pc(r["B0_roi"]), pp(r["X5_droi"])])
    better = sum(1 for r in BR["league"] if r["X5_dgoals"]["mean"] < 0)
    rep.add(H2("7.7. Лиги"),
            P(f"Выборки по лигам малы ({min(r['n'] for r in BR['league'])}–{max(r['n'] for r in BR['league'])} матчей), "
              "интервалы широкие; таблица — для поиска явных аномалий, не для выбора."),
            table(rows, widths=W(1.2, 1.7, 1.6, 4.4, 2.0, 1.8, 1.8, 2.5)),
            P(f"X5 лучше базы по отклонению голов в {better} из {len(BR['league'])} лигах; интервал исключает ноль в "
              f"{sum(1 for r in BR['league'] if r['X5_dgoals']['ci95'][0] > 0 or r['X5_dgoals']['ci95'][1] < 0)} лигах "
              "(при 18 сравнениях один-два таких случая ожидаются и случайно)."),
            *figure(FIG / "league_breakdown.png", "Рис. 11. Разности по лигам для X5 и X3a."))
    rep.add(H2("7.8. Калибровка"),
            P(f"ECE 1X2 (среднее по H, D, A): B0 {f(SUM['B0']['ece_1x2'], 3)}, X5 {f(SUM['X5']['ece_1x2'], 3)}, "
              f"X3a {f(SUM['X3a']['ece_1x2'], 3)}, X2Ma {f(SUM['X2Ma']['ece_1x2'], 3)}; тотал 2.5: B0 {f(SUM['B0']['ece_over2.5'], 3)}, "
              f"X5 {f(SUM['X5']['ece_over2.5'], 3)}; обе забьют: B0 {f(SUM['B0']['ece_btts'], 3)}, X5 {f(SUM['X5']['ece_btts'], 3)}."),
            *figure(FIG / "reliability.png", "Рис. 12. Диаграммы надёжности: 1X2 и выбранный Main."))


def main_section(rep: Report) -> None:
    S = MAIN["summary"]
    rows = [["Код", "Ставок", "Угад.", "Ср. p", "Переоценка [95 % ДИ]", "Ср. коэф.", "ROI [95 % ДИ]",
             "ΔROI с B0 [95 % ДИ]", "Δ переоц. [95 % ДИ]", "Тот же Main"]]
    for v in ORDER:
        s = S[v]
        d = MAIN["vs_B0"].get(v)
        rows.append([v, s["bets"], pc(s["bet_hit_rate"]), pc(s["bet_mean_p"]),
                     f"{pp(s['bet_gap'], 1, False)} {ci(s['bet_gap_ci95'], 1, 100)}", f(s["avg_odds"], 2),
                     f"{pc(s['roi'], 1, True)} {ci(s['roi_ci95'], 1, 100)}",
                     f"{pp(d['roi']['diff'])} {ci(d['roi']['ci95'], 1, 100)}" if d else "база",
                     f"{pp(d['gap']['diff'])} {ci(d['gap']['ci95'], 1, 100)}" if d else "база",
                     pc(d["same_main"], 0) if d else "100%"])
    b0 = S["B0"]
    x5 = MAIN["vs_B0"]["X5"]
    x3 = MAIN["vs_B0"]["X3a"]
    pr = MAIN["prices"]
    rep.add(H2("7.9. Выбранный Main: переоценка и ROI при реальных ценах"),
            P(f"Цены: {pr['rows']} котировок для {pr['matches']} матчей (записанные при прогнозе: {pr['origin'].get('recorded', 0)}, "
              f"из архива OddsPapi до начала матча: {pr['origin'].get('archive', 0)}; отброшено средних по рынку: "
              f"{pr['market_average_dropped']}). Main B0 совпадает с записанным в продакшне Main у "
              f"{pc(MAIN['b0_same_as_stored'], 0)} матчей ({pc(MAIN['b0_same_as_stored_prod_20261012'], 0)} для прогнозов "
              f"той же версии {C.PRODUCTION_VERSION}; остальное — более ранние версии моделей и обновлённые цены)."),
            table(rows, widths=W(1.1, 1.4, 1.3, 1.2, 2.4, 1.3, 2.3, 2.3, 2.3, 1.4)),
            *figure(FIG / "main_gap_roi.png", "Рис. 13. Переоценка выбранного Main и его ROI по вариантам."),
            P(f"Переоценка выбранного Main у базы на этих матчах — {pp(b0['bet_gap'], 1, False)} {ci(b0['bet_gap_ci95'], 1, 100)} "
              f"(заявлено {pc(b0['bet_mean_p'])}, угадано {pc(b0['bet_hit_rate'])}), ROI {pc(b0['roi'], 1, True)} "
              f"{ci(b0['roi_ci95'], 1, 100)}: эффект проклятия победителя, найденный в v6.1, на месте. xG его не "
              f"устраняет: у X5 переоценка меньше на {pp(-x5['gap']['diff'], 1, False)} {ci(x5['gap']['ci95'], 1, 100)}, "
              f"у X3a — на {pp(-x3['gap']['diff'], 1, False)} {ci(x3['gap']['ci95'], 1, 100)}; интервалы включают ноль, "
              f"и переоценка остаётся положительной у всех вариантов (наименьшая — {pp(min(s['bet_gap'] for s in S.values()), 1, False)})."),
            P(f"ROI Main у X5 выше на {pp(x5['roi']['diff'])} {ci(x5['roi']['ci95'], 1, 100)}, но так же ведёт себя и "
              f"контроль без xG R0 ({pp(MAIN['vs_B0']['R0']['roi']['diff'])} {ci(MAIN['vs_B0']['R0']['roi']['ci95'], 1, 100)}): "
              f"X5 меняет выбор Main в {pc(1 - x5['same_main'], 0)} матчей, R0 — в {pc(1 - MAIN['vs_B0']['R0']['same_main'], 0)}, "
              f"и разность ROI на {b0['bets']} ставках — в пределах шума выбора. ROI не может вырасти от лучших вероятностей, если они не лучше рыночных: у Bet365 "
              f"log loss 1X2 на тех же матчах {f(MET['bet365']['logloss_1x2'])} против {f(MET['bet365']['models_on_subset']['B0'])} "
              f"у базы и {f(MET['bet365']['models_on_subset']['X5'])} у X5 ({MET['bet365']['n_1x2']} матчей); по тоталу 2.5 — "
              f"{f(MET['bet365']['ll_over2.5'])} против {f(MET['bet365']['models_on_subset_over']['B0'])} и "
              f"{f(MET['bet365']['models_on_subset_over']['X5'])}. Разрыв базы с рынком по 1X2 — "
              f"{f(MET['bet365']['models_on_subset']['B0'] - MET['bet365']['logloss_1x2'], 4, True)}, а лучший вариант xG сдвигает log loss "
              "лишь на тысячные доли; к этому добавляется маржа букмекера. Поэтому даже если точность выбора немного растёт, "
              "ROI остаётся отрицательным: Main выбирается там, где модель расходится с рынком сильнее всего, и там же её ошибка больше."))
    c = CORN
    rep.add(H2("7.10. Угловые"),
            P(f"Проверено, несёт ли xG косвенный сигнал для угловых: недельная GLM-коррекция ожидаемых угловых v6.1 "
              f"признаками xG (суммарный уровень xG, разрыв сил, сжатые xG за) против такой же перекалибровки без xG. "
              f"{c['n']} матчей: отклонение B0 {f(c['dev_B0'])}, перекалибровка {f(c['dev_recal'])}, с xG {f(c['dev_xg'])}; "
              f"разность с перекалибровкой {dci(c['diff_xg_vs_recal'])}, с B0 {dci(c['diff_xg_vs_B0'])}. Корреляции "
              f"признаков с остатком угловых: {', '.join(f'{k} {v:+.3f}' for k, v in c['corr_with_residual'].items())}. "
              "Сигнала нет; xG для угловых не рекомендуется."))
    rows = [["Настройка", "Код", "Откл. голов [95 % ДИ]", "Log loss 1X2 [95 % ДИ]"]]
    names = {"lag4": "задержка xG 4 дня", "ridge3": "гребень 3", "ridge30": "гребень 30"}
    for k, s in SENS.items():
        for v in ("R0a", "X2a", "X3a", "X5", "M1a"):
            rows.append([names.get(k, k), v, dci(s[v]["goals_dev"]), dci(s[v]["logloss_1x2"])])
    rep.add(H2("7.11. Чувствительность"),
            P("Задержка публикации: Football-Data обновляет файлы не каждый день, поэтому свежий матч может прийти без xG. "
              "Вариант «задержка 4 дня» строит признаки только по матчам, сыгранным более чем за 4 дня до прогноза. "
              "Силу гребня меняли в 3 раза в обе стороны. Эти прогоны только проверяют устойчивость, на выбор не влияли."),
            table(rows, widths=W(3.6, 1.6, 5.9, 5.9)),
            P(f"X5 устойчив: при задержке 4 дня {dci(SENS['lag4']['X5']['goals_dev'])}; вес w один, и гребень на него не "
              "влияет. Многопараметрические коррекции меняются с силой гребня, но по отклонению голов остаются хуже базы; "
              f"разности log loss 1X2 у всех — не больше {max(abs(s[v]['logloss_1x2']['mean']) for s in SENS.values() for v in ('R0a', 'X2a', 'X3a', 'X5', 'M1a')):.4f} по модулю."))


def short_history(rep: Report) -> None:
    rep.add(H1("8. Проблема короткой истории"),
            *bullets([
                f"**Модели только на 2026/27.** XGBoost с настройками v6.1, обученный только на текущем сезоне (S0), хуже "
                f"базы ({dci(VS['S0']['goals_dev'])} по отклонению голов); добавление xG (S1) почти ничего не меняет "
                f"({dci(MET['pairs']['S1-S0']['goals_dev'])}). Длинная история важнее нового признака.",
                f"**xG как дополнительный сигнал поверх длинной истории.** Лучший способ — одна весовая смесь (X5): база "
                "не переобучается, на коротком отрезке оценивается один параметр. Коррекции с 4–6 параметрами (X2, X3, M1, X4) "
                "проигрывают из-за шума оценок.",
                "**Отдельные модели коррекции.** Даже коррекция без xG (R0, R0a) на сотнях матчей ухудшает базу; сжатие к базе "
                "(варианты «a») уменьшает вред, но не даёт выигрыша.",
                f"**Перенос в существующие признаки.** Прокси xG по ударам (T1) обучается на всей истории 2021/22–2025/26; "
                f"он почти совпадает с базой ({dci(VS['T1']['goals_dev'])} по голам, {dci(VS['T1']['logloss_1x2'])} по 1X2), "
                "потому что удары и удары в створ в модели уже есть: прокси — их линейная комбинация.",
                "**Повторная проверка T0.** Переобучение v6.1 с теми же параметрами на тех же сезонах воспроизводит базу "
                "в точности (все разности 0) — сравнения с B0 не искажены случайностью обучения.",
            ]),
            H2("8.1. Законные источники более длинной истории xG (не загружались)"),
            *bullets([
                "**Football-Data.co.uk** — xG начали публиковать с 2026/27; историю прошлых сезонов источник не даёт (проверено по локальным файлам).",
                "**StatsBomb Open Data** — бесплатные событийные данные с xG для отдельных турниров и сезонов по их пользовательскому соглашению; полного покрытия лиг FootPredict нет.",
                "**Understat** — открытые страницы с xG для нескольких топ-лиг за ряд сезонов; перед использованием нужно проверить условия сайта и разрешение на автоматический сбор.",
                "**FBref / Opta** — xG для крупных лиг; условия Sports Reference ограничивают массовую выгрузку, лицензия Opta — коммерческая.",
                "**API-Football** (ключ уже есть в проекте) — в статистике матчей бывает поле ожидаемых голов; глубину истории и доступность на бесплатном плане нужно проверить по документации, не тратя запросы.",
                "**Коммерческие поставщики** (Opta/Stats Perform, Wyscout, Sportmonks и др.) — полная история по лицензии, платно.",
                "**Накопление собственной истории** — каждый новый файл Football-Data добавляет xG; к концу 2026/27 будет полный сезон для 18 лиг.",
            ]))


def conclusions(rep: Report) -> None:
    rows = [["Код", "Решение", "Откл. голов", "Log loss 1X2", "ΔROI Main", "Причина"]]
    for v in ORDER[1:]:
        d = DEC[v]
        rows.append([v, d["decision"], f(d["goals_dev"], 4, True), f(d["logloss_1x2"], 4, True), pp(d["roi_diff"]), d["why"]])
    neg = [v for v in ORDER[1:] if DEC[v]["decision"] == "C"]
    rep.add(H1("9. Отрицательные результаты"),
            *bullets([
                f"Чистые модели на xG (X1) и короткая история (S0, S1) значимо хуже v6.1 по голам и 1X2.",
                f"XGBoost по остатку (X4) переобучается: {dci(VS['X4']['goals_dev'])} по отклонению голов.",
                f"Коррекции 1X2 признаками xG (M1, M1a) не улучшают log loss ({dci(VS['M1']['logloss_1x2'])}, {dci(VS['M1a']['logloss_1x2'])}).",
                "xG не устраняет переоценку выбранного Main и не делает ROI положительным.",
                f"Для угловых xG сигнала не несёт ({dci(CORN['diff_xg_vs_B0'])}).",
            ]),
            H1("10. Ограничения"),
            *bullets([
                f"Неполная первая половина сезона: {MET['n_matches']} матчей оценки; эффекты порядка тысячных долей отклонения не обнаруживаются (см. 7.5).",
                "2026/27 уже использовался исследованиями v6.1, поэтому это не нетронутая отложенная выборка; варианты «a» и X5 добавлены после первого прогона.",
                f"xG только в {len(AUD['xg_leagues'])} лигах; в {', '.join(AUD['no_xg_leagues'])} все варианты совпадают с базой.",
                "Методика расчёта xG поставщика неизвестна и может меняться; сопоставимость проверена только по отношению голов к xG.",
                "Момент публикации xG в файлах неизвестен; задержка учтена только как проверка чувствительности (4 дня).",
                "Цены — записанные при прогнозе и архив OddsPapi; это не всегда лучшая цена рынка.",
            ]),
            H1("11. Выводы"),
            *bullets([
                f"Аудит: xG есть только в 2026/27, в {len(AUD['xg_leagues'])} лигах, без невозможных значений; шкала xG между лигами сопоставима.",
                "Признаки xG содержат небольшой сигнал, которого нет у v6.1 (корреляция рейтинга xG с остатком голов), и внутри сезона xG информативнее голов.",
                f"Использовать этот сигнал при нынешнем объёме данных почти не удаётся: лучший вариант X5 улучшает отклонение голов на {f(-VS['X5']['goals_dev']['mean'], 4)} (незначимо), а 1X2 не улучшает.",
                "Переоценка выбранного Main и отрицательный ROI остаются: их причина — выбор по ошибке модели против рынка, а не нехватка xG.",
            ]),
            H1("12. Рекомендации (A / B / C)"),
            P("A — готово к внедрению; B — перспективно, нужна форвард-проверка; C — не рекомендуется. Правила решения "
              "(`study.decisions`): A требует значимого улучшения после Холма и форвард-подтверждения, поэтому в этом "
              "исследовании невозможно; B — точечная оценка лучше базы и своего контроля без значимого ухудшения основных метрик; "
              "иначе C."),
            table(rows, widths=W(1.1, 2.0, 1.9, 1.9, 1.9, 8.2)),
            note(f"Продакшн не меняется. Вариант X5 (основной) и T1 (вторичный) заморожены в "
                 f"`experiments/v6_2/xg/results/preregistration.json` (заморожено {PRE['frozen_at']}, данные по {PRE['data_until']}). "
                 f"Форвард-критерий: не менее {PRE['forward_test']['minimum_matches']} матчей после заморозки; разность "
                 "отклонения голов X5 − B0 < 0 с односторонним бутстреп-p < 0.05, log loss 1X2 не хуже чем на +0.002 "
                 "(верхняя граница 95 % ДИ), ROI Main не хуже чем на −2 п.п. (нижняя граница), переоценка Main не больше, чем у B0. "
                 f"Вариантов C: {len(neg)}."),
            P("Практический шаг без риска: продолжать накапливать xG (каждое обновление файлов Football-Data), и после "
              "окончания 2026/27 повторить исследование с полным сезоном — тогда xG можно будет добавить в обучение "
              "XGBoost как признак хотя бы одного полного сезона."))


def reproduce_section(rep: Report) -> None:
    env = RUN.get("environment", {})
    rep.add(H1("13. Воспроизведение"),
            *reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                        "$PY -m experiments.v6_2.xg.run --all --refresh     # аудит, признаки, варианты, оценка, графики",
                        "$PY -m experiments.v6_2.xg.preregister             # results/preregistration.json (+ коммит)",
                        "$PY -m experiments.v6_2.xg.build_pdf               # этот отчёт и findings.md"]),
            P(f"Git SHA прогона: `{env.get('git_sha', '—')}`; Python {env.get('python', '—')}, xgboost "
              f"{env.get('libraries', {}).get('xgboost', '—')}, pandas {env.get('libraries', {}).get('pandas', '—')}. "
              "Время шагов, параметры и версии входных файлов (строки, время изменения) — в `results/run_info.json`."))


def findings_md() -> str:
    x5, t1 = VS["X5"], VS["T1"]
    m5 = MAIN["vs_B0"]["X5"]
    b0 = MAIN["summary"]["B0"]
    lines = ["# FootPredict v6.2 — исследование xG: выводы", "",
             "Все числа — из `results/*.json` (файл собирается `build_pdf.py`). PDF: `reports/v6_2/XG_Research.pdf`.", "",
             "## Данные", "",
             f"- xG только в 2026/27: {len(AUD['xg_leagues'])} лиг, {AUD['overall']['matches_with_xg']} матчей "
             f"({AUD['overall']['first_date']} – {AUD['overall']['last_date']}); в {', '.join(AUD['no_xg_leagues'])} колонок нет; "
             "в 2021/22–2025/26 xG нет ни в одной лиге.",
             f"- Заполненность сыгранных матчей лиг с xG: {pc(AUD['played_with_xg'] / AUD['played_in_xg_leagues'])} "
             "(не хватает только самых свежих матчей); невозможных значений нет.",
             f"- Голы / xG = {f(AUD['overall']['ratio'], 3)}; однородность по лигам не отвергается "
             f"(p = {f(AUD['overall']['ratio_heterogeneity']['p'], 2)}).", "",
             "## Оценка", "",
             f"- {MET['n_matches']} матчей {MET['eval_period'][0]} – {MET['eval_period'][1]}, {MET['n_variants'] - 1} вариантов, "
             "еженедельное расширяющееся окно, парный бутстреп, Холм.", "",
             "| Вариант | Откл. голов − B0 [95 % ДИ] | Log loss 1X2 − B0 [95 % ДИ] | ΔROI Main | Решение |", "| --- | --- | --- | --- | --- |"]
    for v in ORDER[1:]:
        lines.append(f"| {name(v)} | {dci(VS[v]['goals_dev'])} | {dci(VS[v]['logloss_1x2'])} | "
                     f"{pp(MAIN['vs_B0'][v]['roi']['diff'])} | {DEC[v]['decision']} |")
    lines += ["", "## Главное", "",
              f"- Значимых улучшений после поправки Холма нет. Лучший — X5: голы {dci(x5['goals_dev'])}, 1X2 {dci(x5['logloss_1x2'])}; "
              f"нужно около {nreq(x5['goals_dev']['n_required'])} матчей для мощности 80 %.",
              f"- T1 (прокси xG по ударам на длинной истории): 1X2 {dci(t1['logloss_1x2'])}.",
              f"- Переоценка выбранного Main у B0 {pp(b0['bet_gap'], 1, False)}, ROI {pc(b0['roi'], 1, True)}; X5 меняет "
              f"переоценку на {pp(m5['gap']['diff'])} {ci(m5['gap']['ci95'], 1, 100)}, ROI на {pp(m5['roi']['diff'])} "
              f"{ci(m5['roi']['ci95'], 1, 100)} — не значимо.",
              f"- Угловые: xG ухудшает прогноз ({dci(CORN['diff_xg_vs_B0'])}).", "",
              "## Решение", "",
              f"- B: X5 (основной), T1 (вторичный) — предрегистрация `results/preregistration.json`, форвард-проверка "
              f"на ≥ {PRE['forward_test']['minimum_matches']} матчах после заморозки. Остальные варианты — C. Продакшн не меняется.", ""]
    return "\n".join(lines)


def build() -> None:
    rep = XGReport(OUT, "Исследование xG и xGA", "FootPredict v6.2",
                 facts={"Период оценки": f"{MET['eval_period'][0]} – {MET['eval_period'][1]} (2026/27)",
                        "Матчей в оценке": str(MET["n_matches"]), "Вариантов": str(MET["n_variants"] - 1),
                        "База": f"v6.1, {C.PRODUCTION_VERSION}"})
    intro(rep)
    data_section(rep)
    features_section(rep)
    signal_section(rep)
    methods_section(rep)
    results_section(rep)
    stats_section(rep)
    time_section(rep)
    main_section(rep)
    short_history(rep)
    conclusions(rep)
    reproduce_section(rep)
    rep.build()
    n = rep.check()
    (C.HERE / "findings.md").write_text(findings_md(), encoding="utf-8")
    print(f"{OUT} — {n} pages")


if __name__ == "__main__":
    build()
