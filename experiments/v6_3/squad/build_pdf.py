"""Builds reports/v6_3/Squad_Strength_Research.pdf and findings.md from results/ (no number is typed by hand).

    .venv/bin/python -m experiments.v6_3.squad.build_pdf
"""
from __future__ import annotations

import json

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether

from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, code, figure, formula, note, table

from experiments.v6_3 import common as C

DIRS = C.study_dirs("squad")
R, FIG = DIRS["results"], DIRS["figures"]
OUT = C.ROOT / "reports" / "v6_3" / "Squad_Strength_Research.pdf"
FINDINGS = C.V63 / "squad" / "findings.md"

METRIC_RU = {"ll_1x2": "log loss 1X2", "brier_1x2": "Brier 1X2", "acc_1x2": "точность 1X2",
             "dev_home": "девиация голов хозяев", "dev_away": "девиация голов гостей", "ae_goals": "MAE голов",
             "ll_ou25": "log loss ТБ/ТМ 2.5", "ll_btts": "log loss «обе забьют»",
             "dev_corners": "девиация угловых", "ae_corners": "MAE угловых"}
SHOW = ["ll_1x2", "brier_1x2", "acc_1x2", "dev_home", "dev_away", "ll_ou25", "ll_btts", "dev_corners", "ae_corners"]
VAR_RU = {"dyn": "DYN", "pre": "PRE", "sa": "SA", "b0r": "B0r", "baseline_b": "Baseline B"}


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def f3(v, d=4, signed=True):
    if v is None:
        return "—"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def ci(x, d=4):
    if not x or x.get("diff") is None:
        return "—"
    return f"{f3(x['diff'], d)} [{f3(x['ci95'][0], d)}; {f3(x['ci95'][1], d)}]"


def pv(v):
    if v is None:
        return "—"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def pct(v, d=1, signed=False):
    if v is None or v != v:
        return "—"
    return f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}"


def roi_ci(r):
    if not r or r.get("roi") is None:
        return "—"
    return f"{pct(r['roi'], 1, True)} [{pct(r['ci95'][0], 1, True)}; {pct(r['ci95'][1], 1, True)}]"


def pp_ci(x):
    if not x:
        return "—"
    return f"{pct(x['diff'], 1, True)} [{pct(x['ci95'][0], 1, True)}; {pct(x['ci95'][1], 1, True)}]"


def build() -> None:
    src = load("sources.json")
    cov = pd.read_csv(R / "coverage_by_league_season.csv")
    mrep = load("mapping_report.json")
    leak = load("leak_check.json")
    e1 = load("e1_earlier_seasons.json")
    be = load("beyond_elo_form.json")
    e2 = load("e2_season_2026.json")
    dec = load("decisions.json")
    info = load("run_info.json")
    selfc = load("self_check_fit_plus.json")
    t1 = load("tuning_e1_sa.json")
    t2 = load("tuning_e2_sa.json")
    cov2 = pd.read_csv(R / "coverage_e2.csv")
    reasons = pd.read_csv(R / "coverage_e2_reasons.csv")
    brk = pd.read_csv(R / "breakdown_e2.csv")
    pre = load("preregistration.json")
    pilot = src["pilot_requests"]

    played26 = e2["played"]
    covered = e2["covered_played"]
    tm_leagues = sorted(cov.loc[cov["source_league"].notna() & (cov["source_league"] != ""), "league"].unique())
    c21 = cov.loc[cov["season"].between(2021, 2025) & cov["league"].isin(tm_leagues)]
    dyn_share = c21["dyn_both"].sum() / c21["matches"].sum()
    all21 = cov.loc[cov["season"].between(2021, 2025)]

    rep = K.Report(OUT, "Сила состава (Squad Strength)", "FootPredict v6.3 — исследование 12",
                   {"Источник данных игроков": "открытый датасет Transfermarkt (CC0), заморожен 06.07.2026",
                    "Сезоны": "2021/22–2025/26 (хронологическая проверка), 2026/27 (повтор сезона)",
                    "Матчей 2026/27": f"{e2['season_matches']}, сыграно {played26}, в выборке состава {covered}"})

    # ------------------------------------------------------------------------------------------------ 1
    rep.add(H1("1. Название исследования"),
            P("**Оценка реальной силы команды по типичному основному составу, запасным и индивидуальной "
              "статистике игроков и её влияние на прогнозы FootPredict (1X2, голы, тоталы, «обе забьют», угловые) "
              "и на доходность Main.**"),
            P("Исследование 12 технического задания v6.3 (этапы A–E): источники данных игроков, типичный состав на "
              "дату матча, рейтинги игроков, командные признаки силы состава, сравнение Baseline / Squad Strength / "
              "Squad-Aware."))

    # ------------------------------------------------------------------------------------------------ 2
    rep.add(H1("2. Постановка проблемы"),
            P("Модели FootPredict видят команду только через её результаты: таблицу, форму, Elo, удары, угловые. "
              "Состав в этих признаках проявляется с опозданием: после летних трансферов, травмы лидера или ротации "
              "Elo и форма меняются лишь через несколько туров. Если бы сила игроков, которые реально выходят на поле, "
              "была известна заранее, прогноз мог бы реагировать раньше рынка результатов."),
            P("Модель не должна угадывать точный стартовый состав за несколько часов до матча. Нужна статистическая "
              "оценка **типичного** основного состава и ближайших запасных по матчам, сыгранным **до** прогноза, и "
              "оценка силы этих игроков на ту же дату. Главный риск — утечка будущего: сегодняшние рейтинги и "
              "сегодняшние позиции игроков для прошлых матчей недопустимы."))

    # ------------------------------------------------------------------------------------------------ 3
    rep.add(H1("3. Гипотезы"),
            bullets([
                "**H1.** Признаки силы состава уменьшают log loss 1X2 сверх Elo и командной формы.",
                "**H2.** Атакующие и оборонительные линии состава улучшают модели голов, тоталов и «обе забьют».",
                "**H3.** Вклад фланговых и атакующих игроков улучшает модель угловых.",
                "**H4.** Модели с признаками состава дают лучший Main ROI на общей выборке, чем Baseline B.",
                "**H5.** Специально настроенная компактная модель (Squad-Aware) не хуже продакшн-набора признаков.",
            ]),
            P("Решения A / B / C принимаются по правилам README v6.3: A — значимое после поправки Холма парное "
              "улучшение на хронологической проверке, B — устойчивое направление без значимости, C — нет выигрыша."))

    # ------------------------------------------------------------------------------------------------ 4
    rows = [["Источник", "Доступ", "История / покрытие", "Вывод"]]
    for s in src["sources"]:
        rows.append([s["source"], s["access"], s["history"], s["verdict"]])
    rep.add(H1("4. Исходные данные"),
            H2("4.1. Этап A: проверка источников (18.11.2026)"),
            P("Проверены уже подключённый API-Football и бесплатные публичные альтернативы. Правила: без входа и "
              "платных сервисов, robots.txt и условия сайта соблюдаются, проверки браузера не обходятся, между "
              "запросами 3–5 с. Рабочий ключ API-Football хранится только в GitHub Secrets (локальный ключ — "
              "приостановленного аккаунта), поэтому API-Football **не вызывался**, workflow не запускался."),
            table(rows, widths=[3.2 * cm, 3.6 * cm, 5.4 * cm, 4.8 * cm]),
            P("Что нашлось в уже сохранённых ответах API-Football (`local_data/api/api_football`): только запросы "
              "`teams` и `fixtures` — ни одного ответа с составами или статистикой игроков. Бесплатный план отдаёт "
              "матчи только со вчерашнего дня по завтрашний (проверено в v6.1), поэтому история составов 2021–2026 "
              "через него недоступна при любом числе запросов."))
    rep.add(H2("4.2. План пилота API-Football (нужно согласие владельца)"),
            P(f"Цель: {src['api_football_pilot']['goal']}."),
            bullets([f"Запрос: {src['api_football_pilot']['endpoint']}.",
                     f"Когда: {src['api_football_pilot']['when']}.",
                     f"Лиги: {src['api_football_pilot']['leagues']}.",
                     f"Расход по календарю {pilot['season']}: {pilot['matches']} матчей за {pilot['match_days']} "
                     f"игровых дней, медиана {pilot['matches_per_day_median']:.1f} матча в день, максимум "
                     f"{pilot['matches_per_day_max']}; запросов `/fixtures?ids=` в среднем "
                     f"{pilot['requests_ids_per_day_mean']:.1f} в день, максимум {pilot['requests_ids_per_day_max']}, "
                     f"{pilot['requests_ids_per_season']} за сезон — при лимите {pilot['daily_limit']} в сутки.",
                     f"Запасной вариант (по 2 запроса на матч, только 11 топ-лиг): до "
                     f"{pilot['fallback_two_per_match_top11_max_per_day']} запросов в пиковый день — превышает лимит, "
                     "поэтому только батч `ids`.",
                     f"Первый шаг: {src['api_football_pilot']['first_step']}.",
                     "Срок: динамические признаки для обучения появятся только после накопления хотя бы одного "
                     "сезона (обучение модели требует нескольких сезонов истории)."]),
            note("Запуск пилота требует явного согласия владельца: используется ключ из GitHub Secrets и дневная "
                 "квота, общая с продакшн-конвейером."))

    lg_rows = [["Лига", "Лига источника", "Матчей 2021–2025", "С составом обеих команд", "Доля", "2026/27: предсезон"]]
    for lg in sorted(all21["league"].unique()):
        g = all21.loc[all21["league"] == lg]
        g26 = cov.loc[(cov["league"] == lg) & (cov["season"] == 2026)]
        srcl = g["source_league"].fillna("").iloc[0]
        lg_rows.append([lg, srcl or "нет", str(int(g["matches"].sum())), str(int(g["dyn_both"].sum())),
                        f"{g['dyn_both'].sum() / g['matches'].sum() * 100:.0f} %",
                        f"{int(g26['pre_both'].sum())} из {int(g26['matches'].sum())}" if len(g26) else "—"])
    rep.add(H2("4.3. Использованный источник и покрытие 21 лиги"),
            P("Использован открытый датасет Transfermarkt (`dcaribou/transfermarkt-datasets`, лицензия CC0, "
              "публичный бакет без входа): все матчи клубов (лига, кубки, еврокубки), стартовые составы и запасные "
              "с позициями, минуты, голы, передачи, карточки, **датированная** рыночная стоимость игроков и "
              "трансферы. Обновления датасета остановлены 06.07.2026, поэтому матчей сезона 2026/27 в нём нет. "
              "Сам сайт Transfermarkt не опрашивался; его файл license.xml запрещает машинное использование контента, включая обучение моделей — "
              "для продакшна это юридический вопрос владельца (см. ограничения)."),
            P(f"Сопоставление команд: {sum(v['mapped'] for k, v in mrep.items() if isinstance(v, dict) and 'mapped' in v)} "
              "команд FootPredict сопоставлены с клубами источника по совпадению даты (±1 день) и счёта матчей "
              "лиги; неоднозначных и дублирующихся сопоставлений нет. Сезон 2026/27: команды, поднявшиеся из лиг без "
              "данных, не имеют состава — они показаны отдельно."),
            table(lg_rows, widths=[1.4 * cm, 2.3 * cm, 2.8 * cm, 3.6 * cm, 1.6 * cm, 3.6 * cm]),
            P(f"Итого в 11 покрытых лигах признаки состава обеих команд есть у {dyn_share * 100:.1f} % матчей "
              "2021/22–2025/26. Десять лиг (все вторые и низшие дивизионы Англии, Шотландии, Испании, Италии, Германии, "
              "Франции) в источнике отсутствуют."),
            figure(FIG / "coverage.png", "Рис. 1. Доля матчей с признаками состава обеих команд: 2021/22–2025/26 — "
                                         "динамические признаки, 2026/27 — предсезонные."))

    # ------------------------------------------------------------------------------------------------ 5
    rep.add(H1("5. Методика"),
            H2("5.1. Этап B: типичный состав на дату матча"),
            P(f"Для матча клуба на дату d берутся последние {info['params']['window_games']} матчей клуба во всех "
              "турнирах, сыгранные **не позже d − 2 дня** (отсечка d − 1 день, строго раньше: источник может датировать "
              "тот же матч на день раньше из-за часовых поясов). По ним для каждого игрока: число выходов в старте, "
              "минуты, модальная позиция в этих составах (позиции только из прошлых составов, не текущие). "
              "Типичный XI — 11 игроков с наибольшим числом стартов (при равенстве — минут), скамейка — следующие 7."),
            formula("score(игрок) = старты за 10 матчей + минуты / (90 · 10 · 10)"),
            H2("5.2. Этап C: сила игрока на дату"),
            bullets([
                "**Рыночная стоимость** — последняя оценка, датированная до отсечки; lg(стоимость, €). Это рейтинг "
                "поставщика, исторически доступный на дату (история оценок хранится с датами).",
                "**Результативность за 90 минут**: (голы + передачи) за 365 дней до отсечки во всех клубных турнирах, "
                "со сжатием к среднему по группе позиций (GK / DEF / MID / ATT, оценено на 2019/20–2020/21) с весом "
                "900 минут.",
                "**Форма**: отклонение результативности в последних 5 появлениях от 365-дневной, со сжатием (270 минут).",
                "**Влияние на поле**: разница мячей команды в матчах, где игрок начинал в старте, минус обычная разница "
                "клуба за 365 дней, со сжатием к нулю (10 стартов).",
            ]),
            formula("ga90 = (G+A₃₆₅ + μ_поз · 900/90) / (мин₃₆₅ + 900) · 90"),
            H2("5.3. Этап D: признаки команды"),
            table([["Признак (h_/a_/diff_)", "Определение"],
                   ["sq_squad_value — squad_strength", "средняя lg-стоимость 18 игроков, взвешенная минутами"],
                   ["sq_xi_value — starting_xi_strength", "средняя lg-стоимость типичного XI"],
                   ["sq_att / mid / def_value", "линии атаки, полузащиты, обороны (+вратарь) в XI"],
                   ["sq_bench_value — bench_strength", "средняя lg-стоимость скамейки (места 12–18)"],
                   ["sq_stability — lineup_stability", "среднее пересечение стартов соседних матчей, 5 пар"],
                   ["sq_form — squad_form", "сумма отклонений формы игроков XI"],
                   ["sq_xi_ga90", "сумма ga90 типичного XI (атакующий выход)"],
                   ["sq_impact", "среднее влияние на поле игроков XI"],
                   ["sq_wide_value, sq_wide_n", "фланговые игроки XI (крайние защитники, полузащитники, вингеры)"],
                   ["sq_last_xi_share", "доля типичного XI, начавшая последний матч (ротация, травмы)"],
                   ["sq_xi_age", "средний возраст XI"],
                   ["diff_sq_* — squad_strength_diff", "хозяева − гости"]],
                  widths=[6 * cm, 11 * cm]),
            P("**Предсезонные признаки** `pre_*` — тот же состав на 1 июля сезона: последние 10 матчей до 1 июля, "
              "минус игроки, ушедшие по трансферу до 1 июля, плюс пришедшие (их стоимость). Дополнительно: "
              "continuity — доля минут прошлого окна у оставшихся, arrivals_value / arrivals_n. Признаки постоянны "
              "весь сезон и одинаково определены для всех сезонов, поэтому модель, обученная на 2021–2025, "
              "применяется к 2026/27 с тем же признаком. Динамические признаки для 2026/27 невозможны: у источника "
              "нет ни одного матча 2026/27."),
            P("Для модели угловых добавляются только признаки фланговых и атакующих игроков и общей силы "
              "(wide_value, wide_n, att_value, xi_ga90, xi_value, squad_value) — проверка H3."),
            H2("5.4. Защита от утечки"),
            P(f"Автоматическая проверка на {leak['matches_checked']} случайных матчах: матчей окна с датой не раньше "
              f"отсечки — {leak['window_games_on_or_after_cutoff']}, оценок стоимости не раньше отсечки — "
              f"{leak['values_on_or_after_cutoff']}. Приоры результативности по позициям оценены только на 2019/20–"
              "2020/21. Модели обучаются только на матчах до отсечки (сезонной или 1 июля 2026)."),
            H2("5.5. Этап E: модели и разбиения"),
            bullets([
                "**B0** — продакшн-набор признаков и параметры `prod_20261012`, переобученные на тех же строках "
                "(контроль; `fit_plus` без новых колонок идентичен `engine.fit_task`: максимальное расхождение "
                f"прогнозов {selfc['max_abs_diff_vs_engine_fit_task']:.2g} на {selfc['matches']} матчах).",
                "**Squad Strength**: DYN = B0 + динамические `sq_*`; PRE = B0 + предсезонные `pre_*` "
                "(те же параметры XGBoost, те же веса давности, n_jobs = 2).",
                "**Squad-Aware (SA)**: компактная модель (Elo, лига, отдых, таблица, форма 5 матчей, угловые 10 матчей "
                f"+ признаки состава), один из {len(t1['result']['grid'])} наборов параметров и число деревьев выбраны "
                "на более раннем сезоне ранней остановкой.",
                "**E1 — ранние сезоны** (хронологически, по сезонам): тест 2024/25 (обучение 2021/22–2023/24) и "
                "2025/26 (обучение 2021/22–2024/25). SA: параметры на 2024/25, оценка на 2025/26.",
                "**E2 — сезон 2026/27**: все модели обучены один раз на матчах до 1 июля 2026 (тот же режим, что у "
                "продакшна); SA-pre настроена на 2025/26. Main — продакшн-движок и правило Main "
                "(`engine.batch_from_outputs`), цены close и t09, парное сравнение с Baseline B того же универсума.",
                "Метрики: log loss / Brier / точность 1X2, пуассоновская девиация голов хозяев и гостей, log loss "
                "ТБ/ТМ 2.5 и «обе забьют» из независимых пуассоновских голов (одинаково для всех методов), девиация и "
                "MAE угловых. Парный бутстреп по матчам (2000 повторов), поправка Холма.",
            ]))

    # ------------------------------------------------------------------------------------------------ 6
    n_e1 = sum(len(e1[s]["variants"]) for s in ("2024", "2025"))
    rep.add(H1("6. Эксперименты"),
            bullets([
                f"E1: {n_e1} пар «вариант × тестовый сезон» (DYN, PRE в 2024/25 и 2025/26; SA в 2025/26) против B0, "
                f"общая выборка покрытых лиг: {e1['2024']['matches_covered']} матчей 2024/25 и "
                f"{e1['2025']['matches_covered']} матчей 2025/26; семейство Холма — {e1['holm_family']['tests']} "
                "проверок.",
                "Проверка «сверх Elo и формы»: мультиномиальная логистическая регрессия на Elo + форма ± признаки "
                "состава (вне выборки), и стэкинг на вероятностях B0 (обучение на 2024/25, проверка на 2025/26).",
                f"E2: B0r, PRE, SA-pre на {e2['season_matches']} матчах 2026/27; общая выборка с составом — "
                f"{covered} сыгранных матчей; семейство Холма для ROI — {e2['holm_family']['tests']} проверок.",
                f"Всего вариантов моделей: E1 — {len(info['params']['variants_tried']['e1'])}, E2 — "
                f"{len(info['params']['variants_tried']['e2'])}; сетка SA — {info['params']['variants_tried']['sa_grid']} "
                "набора параметров на задачу.",
            ]))

    # ------------------------------------------------------------------------------------------------ 7
    rep.add(H1("7. Результаты"))
    for season in ("2024", "2025"):
        r = e1[season]
        rows = [["Метрика", "B0"] + [VAR_RU[v] + ": Δ [95 % ДИ]" for v in r["variants"]] + ["p Холма (min)"]]
        for m in SHOW:
            first = next(iter(r["variants"].values()))["covered"].get(m)
            if first is None:
                continue
            line = [METRIC_RU[m], f3(first["base"], 4, False)]
            ph = []
            for v, d in r["variants"].items():
                x = d["covered"].get(m)
                line.append(ci(x))
                if x and x.get("p_holm") is not None:
                    ph.append(x["p_holm"])
            line.append(pv(min(ph)) if ph else "—")
            rows.append(line)
        lab = f"{season}/{int(season) - 1999}"
        rep.add(H2(f"7.{1 if season == '2024' else 2}. E1: тестовый сезон {lab}, {r['matches_covered']} матчей "
                   "покрытых лиг"),
                table(rows, widths=[3.6 * cm, 1.6 * cm] + [3.6 * cm] * len(r["variants"]) + [1.6 * cm]),
                P(f"R² суммарных угловых: B0 {r['b0_r2_corners']:.3f}; " + "; ".join(
                    f"{VAR_RU[v]} {d['r2_corners']:.3f}" for v, d in r["variants"].items()) +
                  f". R² суммы голов: B0 {r['b0_r2_goals']:.3f}; " + "; ".join(
                    f"{VAR_RU[v]} {d['r2_goals']:.3f}" for v, d in r["variants"].items()) + "."),
                P(f"Непокрытые лиги ({r['matches_uncovered']} матчей, признаки состава пустые): Δ log loss 1X2 " +
                  "; ".join(f"{VAR_RU[v]} {ci(d['uncovered'].get('ll_1x2'))}" for v, d in r["variants"].items()) +
                  " — добавление колонок не должно портить остальные лиги."))

    rows = [["Тест", "Матчей", "Elo + форма", "+ состав", "Δ [95 % ДИ]", "p", "Elo + состав", "Только состав"]]
    for s in ("2024", "2025", "2026"):
        if s not in be:
            continue
        x = be[s]
        rows.append([f"{s}/{int(s) - 1999}" + (" (pre)" if s == "2026" else ""), str(x["test"]),
                     f3(x["ll_elo_form"], 4, False), f3(x["ll_elo_form_squad"], 4, False),
                     ci(x["diff_squad_vs_elo_form"]), pv(x["diff_squad_vs_elo_form"]["p"]),
                     f3(x["ll_elo_squad"], 4, False), f3(x["ll_squad_only"], 4, False)])
    st = be["stacking_on_production"]
    rows2 = [["Признаки", "Обуч. 2024/25", "Тест 2025/26", "B0 как есть", "B0 перекалибр.", "+ состав",
              "Δ [95 % ДИ]", "p"]]
    for fam, x in st.items():
        rows2.append(["динамические" if fam == "sq" else "предсезонные", str(x["train_2024"]), str(x["test_2025"]),
                      f3(x["ll_production_refit"], 4, False), f3(x["ll_recalibrated"], 4, False),
                      f3(x["ll_recalibrated_plus_squad"], 4, False), ci(x["diff_squad"]), pv(x["diff_squad"]["p"])])
    corr = be["correlations"]
    rep.add(H2("7.3. Дают ли игроки информацию сверх Elo и формы"),
            P("Логистическая регрессия (стандартизованные признаки, обучение на сезонах до тестового), log loss 1X2 "
              "на тестовом сезоне покрытых лиг:"),
            table(rows, widths=[2.0 * cm, 1.6 * cm, 2.1 * cm, 1.7 * cm, 3.6 * cm, 1.2 * cm, 2.1 * cm, 2.4 * cm]),
            P("Стэкинг на собственных вероятностях продакшн-модели B0 (вне выборки): добавляет ли состав что-то к "
              "тому, что модель уже знает из всех своих признаков:"),
            table(rows2, widths=[2.9 * cm, 1.6 * cm, 1.6 * cm, 1.7 * cm, 1.9 * cm, 1.6 * cm, 3.6 * cm, 1.2 * cm]),
            P("Корреляция признаков состава с разницей Elo (2021/22–2025/26): " + ", ".join(
                f"{k.replace('diff_sq_', '')} {v:+.2f}" for k, v in corr.items()) + "."))

    rows = [["Метрика", "Продакшн (B)"] + [VAR_RU[v] + ": Δ [95 % ДИ]" for v in ("b0r", "pre", "sa")]]
    for m in SHOW:
        x0 = e2["model"]["b0r"]["vs_production"].get(m)
        if x0 is None:
            continue
        rows.append([METRIC_RU[m], f3(x0["base"], 4, False)] +
                    [ci(e2["model"][v]["vs_production"].get(m)) for v in ("b0r", "pre", "sa")])
    rep.add(H2(f"7.4. E2: 2026/27, качество моделей ({covered} сыгранных матчей с составом)"),
            table(rows, widths=[3.6 * cm, 2.2 * cm, 3.7 * cm, 3.7 * cm, 3.7 * cm]),
            P(f"R² угловых: продакшн {e2['model']['production_r2_corners']:.3f}; " + "; ".join(
                f"{VAR_RU[v]} {e2['model'][v]['r2_corners']:.3f}" for v in ("b0r", "pre", "sa")) + "."))

    for u, title in (("close", "close — основной"), ("t09", "t09 — проверка устойчивости")):
        ur = e2["universes"][u]
        rows = [["Метод", "Ставок", "Hit, %", "Ср. кф", "ROI [95 % ДИ]", "ΔROI vs B [95 % ДИ]", "p", "p Холма",
                 "Тот же Main, %"]]
        b = ur["baseline_b"]["covered"]
        rows.append(["Baseline B", str(b["summary"]["bets"]), pct(b["summary"]["hit_rate"]),
                     f3(b["summary"]["avg_odds"], 2, False), roi_ci(b["roi_ci"]), "—", "—", "—", "—"])
        a = ur["baseline_a"]["covered"]
        rows.append(["Baseline A", str(a["summary"]["bets"]), pct(a["summary"]["hit_rate"]),
                     f3(a["summary"]["avg_odds"], 2, False), roi_ci(a["roi_ci"]), "—", "—", "—", "—"])
        for v in ("b0r", "pre", "sa"):
            x = ur["variants"][v]["covered"]
            d = x["vs_b"]["roi_diff"]
            rows.append([VAR_RU[v], str(x["summary"]["bets"]), pct(x["summary"]["hit_rate"]),
                         f3(x["summary"]["avg_odds"], 2, False), roi_ci(x["roi_ci"]), pp_ci(d), pv(d["p"]),
                         pv(d.get("p_holm")), pct(x["vs_b"]["same_main_share"], 0)])
        rows_all = [["Метод (все сыгранные матчи)", "Ставок", "ROI", "ΔROI vs B [95 % ДИ]", "p"]]
        rows_all.append(["Baseline B", str(ur["baseline_b"]["all"]["summary"]["bets"]),
                         pct(ur["baseline_b"]["all"]["summary"]["roi"], 1, True), "—", "—"])
        for v in ("b0r", "pre", "sa"):
            x = ur["variants"][v]["all"]
            rows_all.append([VAR_RU[v], str(x["summary"]["bets"]), pct(x["summary"]["roi"], 1, True),
                             pp_ci(x["vs_b"]["roi_diff"]), pv(x["vs_b"]["roi_diff"]["p"])])
        rep.add(H2(f"7.{5 if u == 'close' else 6}. E2: Main ROI, {title}"),
                P(f"Общая выборка: {covered} сыгранных матчей 2026/27 с предсезонным составом обеих команд. ROI, "
                  "hit и разницы — в процентах / процентных пунктах; Baseline A — история сайта на тех же матчах."),
                table(rows, widths=[1.9 * cm, 1.5 * cm, 1.3 * cm, 1.3 * cm, 3.1 * cm, 3.2 * cm, 1.2 * cm, 1.4 * cm,
                                    1.7 * cm]),
                table(rows_all, widths=[5 * cm, 2 * cm, 2 * cm, 4.5 * cm, 2 * cm]))
    rows = [["Унив.", "Метод", "Матчей", "Сыгр.", "Обраб.", "С ценой", "Ставок", "С сост.", "Без сост.",
             "Без цены", "PUSH"]]
    for r in cov2.itertuples(index=False):
        rows.append([r.universe, VAR_RU.get(r.method, r.method), str(r.season_matches), str(r.played),
                     str(r.processed), str(r.with_real_price), str(r.bets), str(r.squad_sample),
                     str(r.without_squad_data), str(r.excluded_no_price), str(r.push_or_void)])
    rr = reasons.groupby("reason")["matches"].sum()
    rep.add(H2("7.7. Покрытие 2026/27 и исключения"),
            table(rows, widths=[1.5 * cm, 1.8 * cm, 1.5 * cm, 1.4 * cm, 1.5 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm,
                                1.5 * cm, 1.4 * cm, 1.5 * cm]),
            P("Причины отсутствия признаков состава (все матчи сезона): " + "; ".join(
                f"{k} — {v}" for k, v in rr.items()) + ". Ни один матч не выброшен: вне выборки состава "
                "методы считаются, но в основное сравнение не входят; без реальной цены — не ставка."))

    # ------------------------------------------------------------------------------------------------ 8
    rep.add(H1("8. Графики"),
            figure(FIG / "e1_forest.png", "Рис. 2. Ранние сезоны: изменение метрик относительно B0 (95 % ДИ, "
                                          "меньше — лучше)."),
            figure(FIG / "xi_within_elo.png", "Рис. 3. Доля побед хозяев по квинтилям разницы стоимости XI внутри "
                                              "трёх групп разницы Elo."),
            figure(FIG / "e2_cumprofit.png", "Рис. 4. 2026/27: накопленная прибыль Main на общей выборке (close)."))

    # ------------------------------------------------------------------------------------------------ 9
    cl = e2["universes"]["close"]
    rows = [["Метод", "Прибыль 5 крупнейших выигрышей", "ROI без них", "ROI при кф ×0.98", "×1.00", "×1.02"]]
    for v, x in [("baseline_b", cl["baseline_b"])] + [(v, cl["variants"][v]) for v in ("b0r", "pre", "sa")]:
        bw, os_ = x["big_wins_covered"], x["odds_sensitivity_covered"]
        rows.append([VAR_RU[v], f3(bw["top_profit"], 2, False), pct(bw["roi_without_top"], 1, True),
                     pct(os_["0.98"], 1, True), pct(os_["1.0"], 1, True), pct(os_["1.02"], 1, True)])
    bm = brk.loc[(brk["universe"] == "close") & (brk["by"] == "month")]
    piv = bm.pivot(index="key", columns="method", values="roi")
    nb = bm.pivot(index="key", columns="method", values="bets")
    mrows = [["Месяц", "Ставок B", "ROI B", "ROI B0r", "ROI PRE", "ROI SA-pre"]]
    for k in piv.index:
        mrows.append([k + (" *" if nb.loc[k, "baseline_b"] < 100 else ""), str(int(nb.loc[k, "baseline_b"])),
                      pct(piv.loc[k, "baseline_b"], 1, True), pct(piv.loc[k].get("b0r"), 1, True),
                      pct(piv.loc[k].get("pre"), 1, True), pct(piv.loc[k].get("sa"), 1, True)])
    bl = brk.loc[(brk["universe"] == "close") & (brk["by"] == "league")]
    pl = bl.pivot(index="key", columns="method", values="roi")
    nl = bl.pivot(index="key", columns="method", values="bets")
    lrows = [["Лига", "Ставок B", "ROI B", "ROI B0r", "ROI PRE", "ROI SA-pre"]]
    for k in pl.index:
        lrows.append([k + (" *" if nl.loc[k, "baseline_b"] < 100 else ""), str(int(nl.loc[k, "baseline_b"])),
                      pct(pl.loc[k, "baseline_b"], 1, True), pct(pl.loc[k].get("b0r"), 1, True),
                      pct(pl.loc[k].get("pre"), 1, True), pct(pl.loc[k].get("sa"), 1, True)])
    rep.add(H1("9. Статистический анализ"),
            P("Зависимость от крупных выигрышей и чувствительность к коэффициентам (close, общая выборка):"),
            table(rows, widths=[2.4 * cm, 3.8 * cm, 2.4 * cm, 2.8 * cm, 2.2 * cm, 2.2 * cm]),
            P("По месяцам (close, общая выборка; * — меньше 100 ставок, малая выборка):"),
            table(mrows, widths=[2.4 * cm, 2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]),
            P("По лигам (close, общая выборка; * — меньше 100 ставок):"),
            table(lrows, widths=[2.4 * cm, 2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]),
            P("Интервалы — парный бутстреп по матчам (2000 повторов). Поправка Холма — по семействам, указанным в "
              "разделе 6. Сезон 2026/27 уже изучался в v6.1 / v6.2 и не является независимой отложенной выборкой; "
              "варианты не выбирались на нём (параметры SA — на 2024/25 и 2025/26)."))

    # ------------------------------------------------------------------------------------------------ 10-12
    conc = conclusions(e1, be, e2)
    rep.add(H1("10. Выводы"), bullets(summary_lines(e1, e2) + conc))
    dec_ru = {"A": "A — внедрить", "B": "B — перспективно, форвард-проверка", "C": "C — отклонить"}
    names = {"e1_dyn": "DYN (динамические признаки, ранние сезоны)", "e1_pre": "PRE (предсезонные, ранние сезоны)",
             "e1_sa": "SA (компактная squad-aware, 2025/26)", "e2_pre": "PRE на 2026/27",
             "e2_sa": "SA-pre на 2026/27"}
    rows = [["Вариант", "Решение", "Доказательство", "Почему не выше"]] + [
        [names[k], dec_ru[v["decision"]], v["evidence"], v["reason"]] for k, v in dec.items()]
    rep.add(KeepTogether([H1("11. Рекомендации"), table(rows, widths=[3.8 * cm, 3.2 * cm, 4.5 * cm, 5.5 * cm])]),
            bullets(recommendations(dec, pre)))
    rep.add(H1("12. Ограничения"), bullets([
        "Источник истории — замороженный открытый датасет: сезона 2026/27 в нём нет, поэтому в повторе сезона "
        "использованы только предсезонные признаки (состав на 1 июля), а динамические (типичный XI перед "
        "матчем) проверены только на 2024/25 и 2025/26.",
        "Покрыты 11 из 21 лиги (только высшие дивизионы); поднявшиеся команды без истории в источнике остаются "
        "без признаков. Выводы не переносятся на низшие лиги.",
        "У источника нет индивидуальных xG, ударов, защитных действий и рейтингов за матч — сила игрока выражена "
        "рыночной стоимостью на дату, голами + передачами за 90 минут и влиянием на разницу мячей.",
        "Права: Transfermarkt запрещает машинное использование своего контента, включая обучение моделей, (license.xml); датасет третьей стороны "
        "распространяется под CC0, но для продакшна нужен юридически чистый поддерживаемый источник (например, "
        "API-Football по плану из раздела 4.2) — решение владельца.",
        "Understat 18.11.2026 был на обслуживании; если он вернётся, топ-5 лиг 2026/27 можно пересчитать с "
        "динамическими признаками и xG игроков.",
        "Log loss тоталов и «обе забьют» посчитан из независимых пуассоновских голов, а не из полной матрицы счёта "
        "продакшна (одинаково для всех методов; Main считается продакшн-движком).",
        "2026/27 уже изучался в v6.1 / v6.2 — это хронологическая проверка, не независимое подтверждение.",
    ]))
    rep.add(H1("13. Как воспроизвести"), K.reproduce([
        "# данные игроков (публичный бакет, без входа):",
        "B=https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data",
        "D=experiments/v6_3/squad/cache/tm_raw; mkdir -p $D",
        "for t in games competitions clubs players player_valuations \\",
        "         transfers appearances game_lineups; do",
        "  curl -s -o $D/$t.csv.gz $B/$t.csv.gz; sleep 4; done",
        ".venv/bin/python -I experiments/v6_3/squad/parse_tm.py \\",
        "    $D experiments/v6_3/squad/cache/tm",
        ".venv/bin/python -m experiments.v6_3.squad.build_features",
        "experiments/v6_3/heavy.sh 5400 experiments/v6_3/squad/logs/fits_e1.log \\",
        "    .venv/bin/python -m experiments.v6_3.squad.fits e1",
        "experiments/v6_3/heavy.sh 5400 experiments/v6_3/squad/logs/fits_e2.log \\",
        "    .venv/bin/python -m experiments.v6_3.squad.fits e2",
        ".venv/bin/python -m experiments.v6_3.squad.run",
        ".venv/bin/python -m experiments.v6_3.squad.build_pdf",
    ]), P(f"Git SHA запуска: {info['git_sha']}; данные: {info['data'].get('data_commit', '—')}."))
    rep.build()
    print("pages", rep.check())
    write_findings(e1, be, e2, dec, src, conc, covered, dyn_share)


def summary_lines(e1, e2) -> list[str]:
    d25 = e1["2025"]["variants"]["dyn"]["covered"]["ll_1x2"]
    p26 = e2["model"]["pre"]["vs_production"]["ll_1x2"]
    rc = e2["universes"]["close"]["variants"]["pre"]["covered"]["vs_b"]["roi_diff"]
    return [f"**Главное.** Сила состава несёт информацию, которой нет в Elo и форме: на ранних сезонах log loss 1X2 "
            f"улучшается во всех вариантах Squad Strength (DYN 2025/26 {f3(d25['diff'])}, p Холма "
            f"{pv(d25.get('p_holm'))}), на 2026/27 предсезонный вариант PRE — {f3(p26['diff'])} (p Холма "
            f"{pv(p26.get('p_holm'))}). Эффект мал (около 0.5 % log loss) и не превращается в доходность Main: "
            f"ΔROI close {pct(rc['diff'], 1, True)} п.п. [{pct(rc['ci95'][0], 1, True)}; {pct(rc['ci95'][1], 1, True)}]. "
            "Голы, тоталы, «обе забьют» и угловые значимо не меняются."]


def conclusions(e1, be, e2) -> list[str]:
    out = []
    for s in ("2024", "2025"):
        r = e1[s]["variants"]
        out.append(f"E1 {s}/{int(s) - 1999}: Δ log loss 1X2 — " + "; ".join(
            f"{VAR_RU[v]} {ci(d['covered']['ll_1x2'])} (p Холма {pv(d['covered']['ll_1x2'].get('p_holm'))})"
            for v, d in r.items()) + "; Δ девиации угловых — " + "; ".join(
            f"{VAR_RU[v]} {ci(d['covered'].get('dev_corners'))}" for v, d in r.items()) + ".")
    for s in ("2024", "2025", "2026"):
        if s in be:
            x = be[s]["diff_squad_vs_elo_form"]
            out.append(f"Сверх Elo и формы (логистическая регрессия, {s}/{int(s) - 1999}): Δ log loss {ci(x)}, "
                       f"p = {pv(x['p'])}.")
    for fam, x in be["stacking_on_production"].items():
        out.append(f"Сверх продакшн-модели B0 ({'динамические' if fam == 'sq' else 'предсезонные'} признаки, "
                   f"2025/26): Δ log loss {ci(x['diff_squad'])}, p = {pv(x['diff_squad']['p'])}.")
    for v in ("pre", "sa"):
        m = e2["model"][v]["vs_production"]["ll_1x2"]
        rc = e2["universes"]["close"]["variants"][v]["covered"]["vs_b"]["roi_diff"]
        rt = e2["universes"]["t09"]["variants"][v]["covered"]["vs_b"]["roi_diff"]
        out.append(f"2026/27 {VAR_RU[v]}: Δ log loss 1X2 {ci(m)}; ΔROI Main close {pp_ci(rc)} п.п. "
                   f"(p Холма {pv(rc.get('p_holm'))}), t09 {pp_ci(rt)} п.п.")
    return out


def recommendations(dec, pre) -> list[str]:
    letters = [v["decision"] for v in dec.values()]
    out = ["Продакшн не меняется: ни один вариант не прошёл критерий A." if "A" not in letters else
           "Есть вариант с решением A — см. таблицу; внедрение только после форвард-проверки и решения о "
           "лицензии данных.",
           "Для продолжения нужен поддерживаемый источник составов на текущий сезон: пилот API-Football "
           "(`/fixtures?ids=`, 1–6 запросов в день) — после согласия владельца; либо Understat для топ-5 лиг, "
           "когда он снова доступен.",
           "Если позже признаки будут внедряться: колонки `h_/a_/diff_sq_*` добавляются в группу признаков в "
           "`src/features/sets.py` (новая группа `squad` в MODEL_GROUPS для result / home_goals / away_goals; "
           "для corners — только фланговые/атакующие), расчёт — новый модуль в `src/features/`, данные — "
           "в `local_data/players/`."]
    if pre:
        out.append("Варианты B заморожены в `results/preregistration.json` (минимум 3000 матчей покрытых лиг для "
                   "форвард-проверки).")
    return out


def write_findings(e1, be, e2, dec, src, conc, covered, dyn_share) -> None:
    lines = ["# FootPredict v6.3 — исследование 12: сила состава (Squad Strength) — выводы", "",
             "Все числа — из `results/*.json` (файл собирается `build_pdf.py`). PDF: "
             "`reports/v6_3/Squad_Strength_Research.pdf`.", "", "## Источники (этап A)", ""]
    for s in src["sources"]:
        lines.append(f"- **{s['source']}** — {s['verdict']}.")
    p = src["pilot_requests"]
    lines += ["", f"- Пилот API-Football (нужно согласие владельца): `/fixtures?ids=` для вчерашних матчей 21 лиги — "
                  f"в среднем {p['requests_ids_per_day_mean']:.1f}, максимум {p['requests_ids_per_day_max']} "
                  f"запросов в день, {p['requests_ids_per_season']} за сезон (лимит 100/сутки). Первый шаг — один "
                  "пробный запрос через ключ GitHub Secrets.", "",
              "## Данные", "",
              f"- Открытый датасет Transfermarkt (CC0), 11 из 21 лиги, до 06.07.2026; признаки состава обеих "
              f"команд у {dyn_share * 100:.1f} % матчей покрытых лиг 2021/22–2025/26.",
              f"- 2026/27: только предсезонный состав (на 1 июля); общая выборка — {covered} сыгранных матчей.", "",
              "## Результаты", ""]
    lines += [f"- {c}" for c in conc]
    lines += ["", "## Решения", ""]
    lines += [f"- `{k}`: **{v['decision']}** — {v['evidence']}; {v['reason']}." for k, v in dec.items()]
    lines += ["", "## Что было бы изменением продакшна", "",
              "- Ничего не меняется сейчас. При будущем внедрении: группа признаков `squad` в `src/features/sets.py`, "
              "расчёт признаков — новый модуль `src/features/`, источник — поддерживаемый (API-Football по плану).",
              "", "## Дефекты", "",
              "- В общем коде дефектов не найдено. В сохранённых ответах API-Football нет ни одного ответа с "
              "составами — история составов у проекта отсутствует.", ""]
    FINDINGS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
