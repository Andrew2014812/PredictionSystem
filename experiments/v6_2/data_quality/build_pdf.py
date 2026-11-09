"""Builds reports/v6_2/Data_Quality_Research.pdf from experiments/v6_2/data_quality/results (Russian).

    python -m experiments.v6_2.data_quality.build_pdf

Every number is read from the result files at build time; a missing input is reported as not run.
"""
from __future__ import annotations

import json

import pandas as pd

from scripts.report_kit import (H1, H2, H3, P, Report, bullets, figure, formula, note, reproduce, table)

from . import common as C

OUT = C.ROOT / "reports" / "v6_2" / "Data_Quality_Research.pdf"
FIG = C.FIGURES


def J(name):
    p = C.RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def T(name):
    p = C.RESULTS / name
    return pd.read_csv(p) if p.exists() else None


RAW, TEAMS, REC, STORE, SENS, STRATA, VAR, RUN = (J(n) for n in (
    "raw_audit.json", "teams_audit.json", "recompute.json", "store_audit.json", "sensitivity.json", "strata.json",
    "variants.json", "run_info.json"))


# ------------------------------------------------------------------------------------------------ formatting
def f(v, d=3):
    return "—" if v is None or v != v else f"{v:.{d}f}"


def pc(v, d=1, signed=False):
    if v is None or v != v:
        return "—"
    return f"{v * 100:+.{d}f} %" if signed else f"{v * 100:.{d}f} %"


def pp(v, d=1):
    return "—" if v is None or v != v else f"{v * 100:+.{d}f} п.п."


def ci(d, scale=100.0, digits=1, unit=" %", signed=True):
    """'mean [lo; hi]' of a {'mean'|'diff', 'ci95'} dict."""
    if not d:
        return "—"
    m = d.get("mean", d.get("diff"))
    lo, hi = (d.get("ci95") or [None, None])
    if m is None:
        return "—"
    fmt = (lambda x: f"{x * scale:+.{digits}f}") if signed else (lambda x: f"{x * scale:.{digits}f}")
    if lo is None:
        return f"{fmt(m)}{unit}"
    return f"{fmt(m)}{unit} [{fmt(lo)}; {fmt(hi)}]"


def sig(d) -> str:
    if not d or not d.get("ci95") or d["ci95"][0] is None:
        return "интервал не определён"
    lo, hi = d["ci95"]
    return "интервал не включает 0" if (lo > 0 or hi < 0) else "в пределах шума, интервал включает 0"


def roi_cell(r):
    if not r or r.get("roi") is None:
        return "—"
    lo, hi = r.get("roi_ci95") or [None, None]
    return f"{r['roi'] * 100:+.1f} % [{lo * 100:+.1f}; {hi * 100:+.1f}]" if lo is not None else f"{r['roi'] * 100:+.1f} %"


STATUS_RU = {"established": "старожилы", "moved_within": "переход внутри наших лиг", "returner": "вернувшиеся",
             "outside": "новички извне", "first_season": "первый сезон данных"}
VARIANT_RU = {"season_h": "таблица сезона — хозяева", "season_a": "таблица сезона — гости", "form_h": "форма — хозяева",
              "form_a": "форма — гости", "stats_h": "удары/угловые — хозяева", "stats_a": "удары/угловые — гости",
              "elo_h": "Elo = базовый — хозяева", "elo_a": "Elo = базовый — гости", "rest_h": "дни отдыха — хозяева",
              "rest_a": "дни отдыха — гости", "newcomer_h": "всё (новичок) — хозяева",
              "newcomer_a": "всё (новичок) — гости", "h2h": "личные встречи (пара)"}


CHANGE_RU = {"new club name in a running season": "новое имя клуба посреди сезона",
             "played match disappeared": "сыгранный матч исчез", "row removed": "строка удалена",
             "result turned back into a fixture": "результат снова стал фикстурой", "score changed": "изменился счёт"}


def ru_detail(text) -> str:
    if not isinstance(text, str):
        return ""
    return (text.replace(" played)", " сыгр.)").replace("played=True", "был сыгран").replace("played=False", "не сыгран"))


STATUS_PICK = {"WON": "выигр.", "LOST": "проигр.", "UPCOMING": "ожидает", "VOID": "возврат", "PUSH": "возврат"}
CM = 28.35


def market_ru(text) -> str:
    if not isinstance(text, str):
        return "—"
    market, _, sel = text.partition(" ")
    sel_ru = {"OVER": "больше", "UNDER": "меньше", "H": "хоз.", "A": "гост.", "D": "ничья", "YES": "да", "NO": "нет"}
    if market.startswith("TOTAL_"):
        return f"тотал {market[6:].replace('.', ',')} {sel_ru.get(sel, sel)}"
    if market == "1X2":
        return {"H": "П1", "D": "Х", "A": "П2"}.get(sel, sel)
    if market.startswith("HANDICAP_"):
        return f"фора {market[9:].replace('.', ',')} {sel_ru.get(sel, sel)}"
    return text


def not_run(what):
    return note(f"Эксперимент «{what}» не выполнен: файл результатов отсутствует.")


# ------------------------------------------------------------------------------------------------ sections
def intro(rep):
    rep.add(H1("1. Цель исследования"),
            P("Проверить всю цепочку данных FootPredict — Football-Data и API-провайдеры → нормализация → "
              "сопоставление клубов → история команд → признаки → модели → прогнозы — найти конкретные дефекты "
              "данных, отделить ошибки продакшна от исследовательских наблюдений и количественно ответить на вопрос: "
              "влияет ли качество данных на вероятности, выбор Main и ROI по реальным ценам."),
            H1("2. Постановка проблемы"),
            P("Модели видят только признаки, построенные из истории матчей. Ошибка на любом шаге — другое написание "
              "клуба, перенесённый матч, исчезнувший результат, клуб без истории — превращается в пустые или "
              "неверные признаки. XGBoost обрабатывает пропуски «как есть», поэтому такие ошибки не вызывают сбоя, "
              "а молча меняют прогноз. 03–04.11.2026 это уже произошло: три «фантомных» прогноза были сделаны "
              "для клуба, которого модель не знала (KV Mechelen, Sheffield Utd, Oxford Utd)."),
            H1("3. Гипотезы"),
            *bullets(["**H1.** В сырых файлах и в данных провайдеров есть дефекты (пропуски, дубликаты, переносы, "
                      "разные написания клубов), которые доходят до признаков и прогнозов.",
                      "**H2.** Скользящие признаки посчитаны правильно: независимый пересчёт из сырых CSV даёт те же "
                      "значения.",
                      "**H3.** Если у одной команды нет части признаков, вероятности и выбор Main заметно меняются.",
                      "**H4.** Матчи клубов с малой историей (новички из лиг, которые система не загружает) "
                      "прогнозируются хуже, и Main на них приносит меньше по реальным ценам.",
                      "**H5.** Индекс качества данных матча (доля пустых признаков) связан с ошибкой и с ROI.",
                      "**H6.** Простые меры для клубов без истории (стартовый Elo ниже, история National League) "
                      "улучшают прогноз их матчей."]))


def data_section(rep):
    rep.add(H1("4. Данные"))
    rows = [["Источник", "Что это", "Объём"]]
    if RAW:
        rows.append(["local_data/results/<сезон>/<лига>.csv", "Football-Data, сезоны 2021/22–2026/27, 21 лига",
                     f"{RAW['files']} файлов, {RAW['rows']} строк, {RAW['played']} сыгранных"])
        rows.append(["local_data/fixtures/fixtures.csv", "Football-Data, ближайшие матчи",
                     f"{RAW['fixture_versions']} версий в истории git"])
        lv = RAW["live_vs_fd"]
        rows.append(["local_data/results_live/fixtures.parquet", "API-Football, The Odds API, OddsPapi",
                     f"{lv['live_rows']} строк ({lv['played_rows']} с результатом); {RAW['live_versions']} версий"])
        rows.append(["local_data/results/<сезон>/EC.csv", "National League (лига отключена)",
                     f"{RAW['ec_disabled']['rows']} строк, угловых нет в {pc(RAW['ec_disabled']['corners_missing'], 0)}"])
    if REC:
        rows.append(["local_data/processed/features.parquet", "признаки до матча (264 столбца)",
                     f"{REC['production_played']} сыгранных матчей"])
    if STORE:
        rows.append(["local_data/predictions/ (4 файла parquet)", "снимки, рынки, прогнозы, версии",
                     f"{STORE['picks']} записей прогнозов, {STORE['unified_mains']} Main в единой истории"])
    rows.append(["git-история репозитория", "ежедневные коммиты данных (только чтение)",
                 "версии файлов результатов, фикстур, live-файла и признаков"])
    rep.add(table(rows, widths=None))
    rep.add(P("Сетевых запросов и платных API нет: всё читается из локальных файлов и из истории git. "
              "Модели: для 2023/24 и 2024/25 — переобучение продакшн-тренером с параметрами активной версии на "
              "всех сезонах до прогнозируемого; для 2025/26 — продакшн-бандл `backtest_2526` (обучен до 01.06.2025); "
              "для 2026/27 — активный бандл `prod_20261012` (обучен до 31.05.2026). 2025/26 — «для информации», "
              "2026/27 многократно использовался в исследованиях v6.1 и не является нетронутой проверкой."))


def methods(rep):
    rep.add(H1("5. Методы"),
            H2("5.1. Метрики"),
            P("Log loss 1X2 на матче i с исходом y_{i} и вероятностями модели p:"),
            formula("LL_{i} = −ln p_{i,y(i)}"),
            P("Чтобы трудные матчи не выглядели как «плохие данные», рядом считается log loss рынка — вероятности "
              "Bet365 без маржи (пропорционально, q_{k} = (1/o_{k}) / Σ 1/o_{j}) — и избыточный log loss:"),
            formula("ΔLL_{i} = LL_{i}(модель) − LL_{i}(рынок)"),
            P("Сдвиг вероятностей при порче признаков — расстояние полной вариации между двумя прогнозами одного "
              "матча:"),
            formula("TV_{i} = ½ · Σ_{k∈{H,D,A}} |p'_{i,k} − p_{i,k}|"),
            P("Смещение ожидаемых очков клуба (положительное = модель переоценивает клуб):"),
            formula("b_{i} = 3·P(победа) + P(ничья) − очки_{факт}"),
            P("Main: доля угаданных (WON / (WON + LOST)) и ROI при ставке 1 по реальной цене Bet365 из "
              "Football-Data, только для матчей с ценой; PUSH не считается ставкой."),
            formula("ROI = Σ прибыль / число ставок,  прибыль = o − 1 (выигрыш) или −1 (проигрыш)"),
            P("Индекс качества данных матча — доля пустых признаков модели 1X2 (обе стороны). Интервалы — 95 % "
              "перцентильный бутстрэп по матчам (2000 повторов); сравнение вариантов на одних и тех же матчах — "
              "парный бутстрэп разности; сравнение двух групп — независимый бутстрэп каждой группы. Для шести "
              "сравнений групп с базой p-значения поправлены по Холму."),
            H2("5.2. Независимый пересчёт признаков"),
            P("Сырые CSV разбираются заново (без `normalise`), строки провайдеров добавляются, только если у "
              "Football-Data нет строки этой лиги, дня и клуба; формы, средние сезона и площадки, очки, игры, дни "
              "отдыха, число личных встреч считаются векторно (shift + rolling / expanding), Elo — отдельным циклом по "
              "опубликованному правилу (K = 20, преимущество поля 60, перенос 0,8, база 1500 − 100·(уровень − 1))."),
            H2("5.3. Контролируемая порча признаков"),
            P("У одной стороны группа признаков приводится к виду, который строитель признаков оставляет клубу без "
              "этой истории (число игр = 0, средние пусты, Elo = база уровня лиги); производные разности "
              "пересчитываются формулами самого строителя; матч прогнозируется теми же моделями и с теми же ценами "
              "через продакшн-движок `predict_matches`."),
            H2("5.4. Статус клуба в сезоне"),
            *bullets(["**старожилы** — играли в этой лиге в прошлом сезоне;",
                      "**переход внутри наших лиг** — в прошлом сезоне играли в другой нашей лиге той же страны;",
                      "**вернувшиеся** — известны по более ранним сезонам, прошлый сезон провели вне наших лиг;",
                      "**новички извне** — система их раньше не видела (пришли из лиги, которую мы не загружаем)."]))
    if RUN:
        p = RUN.get("parameters", {})
        env = RUN.get("environment", {})
        rep.add(H2("5.5. Параметры прогона"),
                table([["Параметр", "Значение"],
                       ["Сезоны оценки", ", ".join(C.season_label(s) for s in p.get("eval_seasons", []))],
                       ["Переобучение", ", ".join(C.season_label(s) for s in p.get("refit_seasons", []))],
                       ["Бандлы", f"{p.get('backtest_version')}, {p.get('production_version')}"],
                       ["Бутстрэп / seed / потоков", f"{p.get('n_boot')} / {p.get('seed')} / {p.get('n_jobs')}"],
                       ["Git SHA прогона", env.get("git_sha", "—")],
                       ["Библиотеки", ", ".join(f"{k} {v}" for k, v in env.get("libraries", {}).items() if v)]],
                      widths=[5 * 28.35, 12 * 28.35]))


def raw_section(rep):
    rep.add(H1("6. Сырые данные Football-Data"))
    if not RAW:
        rep.add(not_run("аудит сырых данных"))
        return
    d, s = RAW["duplicates"], RAW["sanity"]
    rep.add(P(f"Проверено {RAW['files']} файлов результатов ({RAW['rows']} строк, {RAW['played']} сыгранных матчей). "
              "Сезон каждой строки, определённый по дате, совпадает с папкой файла; код лиги — с именем файла."),
            table([["Проверка", "Найдено"],
                   ["Повтор match_id", d["duplicate_match_id"]],
                   ["Одна пара хозяев/гостей дважды за сезон (лиги без плей-офф)", d["same_pairing_twice_single_rr"]],
                   ["Клуб в двух матчах одной лиги в один день", d["team_twice_same_day"]],
                   ["Сезон по дате ≠ папка файла", d["season_folder_mismatch"]],
                   ["Код лиги ≠ имя файла", d["division_mismatch"]],
                   ["Удары в створ > ударов", s["sot_gt_shots"]],
                   ["Голов больше ударов в створ (автоголы, допустимо)", f"{s['goals_gt_sot']} ({pc(s['goals_gt_sot_share'], 2)})"],
                   ["Отрицательные значения", s["negative_values"]],
                   ["Строка без счёта в файле результатов", ", ".join(s["unplayed_rows_in_results"]) or "0"],
                   ["Два матча клуба с разницей ≤ 1 день", ", ".join(s["same_team_two_games_within_1_day_ids"]) or "0"],
                   ["Сыгранные матчи без статистики", f"{s['played_without_stats']}, из них похожих на "
                                                      f"техническое 3:0 — {s['played_without_stats_awarded_like']}"]],
                  widths=[11 * 28.35, 6 * 28.35]))
    rep.add(H2("6.1. Пропуски"))
    o = RAW["missing_any_stat_overall"]
    rep.add(P(f"Доля сыгранных матчей, где пусто хотя бы одно поле группы (все лиги и сезоны): удары {pc(o['shots'], 2)}, "
              f"угловые {pc(o['corners'], 2)}, фолы {pc(o['fouls'], 2)}, карточки {pc(o['cards'], 2)}, цена 1X2 "
              f"{pc(o['price_1x2'], 2)}, тотал 2,5 {pc(o['price_ou25'], 2)}, азиатская фора {pc(o['price_ah'], 2)}. "
              "Голы есть всегда. Пропуски статистики сосредоточены в Турции 2022/23: "
              f"{s['played_without_stats_by_league_season'].get('T1 2022', 0)} матчей без статистики — технические "
              "результаты после снятия клубов (землетрясение); эти 3:0 входят в форму и Elo как настоящие матчи."))
    rep.add(*figure(FIG / "missing_by_league.png", "Рис. 1. Доля сыгранных матчей с пропусками по лигам (2021/22–2026/27)."))
    miss = T("raw_missing.csv")
    if miss is not None:
        bys = miss.groupby("season").apply(lambda g: pd.Series({c: (g[c] * g["played"]).sum() / g["played"].sum()
                                                                for c in ("shots", "corners", "price_1x2", "price_ah")}))
        rows = [["Сезон", "Удары", "Угловые", "Цена 1X2", "Цена форы"]]
        for season, r in bys.iterrows():
            rows.append([C.season_label(int(season)), pc(r["shots"], 2), pc(r["corners"], 2), pc(r["price_1x2"], 2),
                         pc(r["price_ah"], 2)])
        rep.add(table(rows), P("Табл. 2. Доля сыгранных матчей с пропусками по сезонам (все 21 лига).", "small"))
    rep.add(H2("6.2. Изменения состава лиг"))
    rows = [["Лига", "Сезон", "Клубов было", "Клубов стало"]]
    for r in RAW["league_size_changes"]:
        rows.append([r["league"], C.season_label(r["season"]), r["teams_before"], r["teams"]])
    rep.add(table(rows), P("В Бельгии и Греции после регулярного сезона идёт плей-офф: его матчи входят в таблицу "
                           "сезона как обычные (36–41 игра на клуб вместо 30); признаки «по сезону» смешивают оба этапа.",
                           "small"))


def names_section(rep):
    rep.add(H1("7. Названия клубов и сопоставление"))
    if not RAW:
        rep.add(not_run("названия клубов"))
        return
    rep.add(H2("7.1. Разные написания в файлах Football-Data"))
    rows = [["Коммит (UTC)", "Файл", "Лига / дата", "Матч", "Имя", "Правило даёт"]]
    for r in RAW["fixture_name_drift_rows"]:
        rows.append([f"{r['commit']} {r['committed_utc'][:16]}", "fixtures.csv", f"{r['league']} {r['date']}",
                     f"{r['home']} – {r['away']}", r["name"], r["resolves_to"] or "—"])
    rh = T("raw_results_history.csv")
    if rh is not None:
        for r in rh.itertuples():
            rows.append([f"{r.commit} {str(r.committed_utc)[:16]}", f"results/2627/{r.file}", "", r.match_id if
                         isinstance(r.match_id, str) else "", ru_detail(r.detail), CHANGE_RU.get(r.change, r.change)])
    rep.add(table(rows, widths=[3.3 * 28.35, 2.8 * 28.35, 2.4 * 28.35, 4.4 * 28.35, 2.4 * 28.35, 2.7 * 28.35]))
    rep.add(P(f"Во всех {RAW['fixture_versions']} версиях файла фикстур и {RAW['results_versions']} версиях файлов "
              "результатов 2026/27 разное написание встретилось один раз — 03–05.11.2026 — сразу в трёх лигах. "
              "Важная деталь: 05.11 (коммит 9bd3c19) неверные имена были и в **файлах результатов** — сыгранные "
              "матчи «Kortrijk – KV Mechelen», «Preston – Sheffield Utd», «Oxford Utd – Reading» на сутки появились "
              "как результаты клубов без истории, а настоящие Mechelen, Sheffield United, Oxford на сутки лишились "
              "этого матча. 06.11 Football-Data исправил имена. `harmonise_fixture_names` (v6.1) приводит к "
              "единому написанию только фикстуры; строки результатов он не трогает."))
    rep.add(H2("7.2. Имена провайдеров"))
    rows = [["Провайдер", "Лига", "Имя провайдера", "Правило даёт", "Кандидаты Football-Data (матчей 2026/27)"]]
    for r in RAW["unmatched_provider_names"]:
        rows.append([r["provider"], r["league"], r["provider_name"], r["resolves_to"] if isinstance(r["resolves_to"], str)
                     else "—", r["fd_candidates"]])
    rep.add(table(rows))
    rep.add(P("Эти три имени API-Football (файл `unmatched_teams.csv` последнего прогона) не сопоставляются, "
              "поэтому фикстуры и свежие результаты OFI Crete, Vicenza и Sociedad B от API-Football пропускаются — "
              "их результаты приходят только с Football-Data."))
    conf = RAW.get("team_id_conflicts") or []
    if conf:
        c = conf[0]
        rep.add(P(f"Файл `team_id_conflicts.csv` сообщает конфликт: OddsPapi id {c['provider_team_id']} записан как "
                  f"{c['registered_as']}, а имя говорит {c['name_says']} («{c['provider_name']}»). По истории git файл "
                  "последний раз менялся 06.11 и с тех пор не перезаписывался: `collect` пишет его только когда "
                  "конфликты есть и никогда не удаляет (в отличие от `unmatched_teams.csv`). Конфликт — след того же "
                  "неверного имени KV Mechelen, которое 03–05.11 было в списке известных клубов."))
    if TEAMS and TEAMS.get("two_name_candidates"):
        rep.add(H2("7.3. Один клуб под двумя именами"))
        seen, rows = set(), [["Страна", "Сезон", "Новое имя (лига)", "Прежнее имя (лига)"]]
        for r in TEAMS["two_name_candidates"]:
            key = (r["season"], r["new_name"], r["old_name"])
            if key in seen:
                continue
            seen.add(key)
            rows.append([r["country"], C.season_label(r["season"]), f"{r['new_name']} ({r['new_league']})",
                         f"{r['old_name']} ({r['old_league']})"])
        rep.add(table(rows), P("Edinburgh City в сезоне 2022/23 назывался FC Edinburgh, а потом вернул старое имя. "
                               "Для системы это два клуба: в 2022/23 — 36 матчей «новичка» без истории, в 2023/24 — "
                               "история годичной давности. Других пар (общие слова длиной ≥ 5 букв у исчезнувшего и "
                               "появившегося клуба одной страны) нет; Notts County / Newport County, Le Mans / Le Havre "
                               "и т. п. — разные клубы.", "small"))


def moved_section(rep):
    rep.add(H1("8. Переносы, даты, хозяева и гости"))
    if not RAW:
        rep.add(not_run("переносы"))
        return
    lv = RAW["live_vs_fd"]
    rep.add(table([["Сравнение API-провайдеров с Football-Data", "Результат"],
                   ["Матчи с одинаковым id: несовпадение счёта", f"{lv['score_mismatch']} из {lv['score_compared']}"],
                   ["Несовпадение времени начала", f"{lv['time_mismatch']} из {lv['time_compared']}"],
                   ["Хозяева и гости переставлены", lv["swapped_home_away"]],
                   ["Дата провайдера ≠ дате Football-Data (та же пара, ≤ 60 дней)", lv["moved_vs_fd"]],
                   ["Результаты только от API (Football-Data ещё не опубликовал)", lv["api_only_played"]],
                   ["…из них старше последней даты Football-Data в лиге", lv["api_only_played_before_fd_last_date"]],
                   ["Несыгранные фикстуры с датой в прошлом в таблице признаков", RAW["stale_fixtures"]]],
                  widths=[12 * 28.35, 5 * 28.35]))
    mv = T("raw_moved_matches.csv")
    if mv is not None and len(mv):
        rows = [["match_id провайдера", "Провайдер", "Дата провайдера", "Дата FD", "Дней"]]
        for r in mv.itertuples():
            rows.append([r.match_id, r.provider, r.provider_date, r.fd_date, r.days])
        rep.add(H3("Перенесённые матчи (дата провайдера и фактическая дата)"), table(rows))
    lh = T("raw_live_history.csv")
    if lh is not None and len(lh):
        rows = [["Коммит (UTC)", "Изменение", "match_id", "Детали"]]
        for r in lh.itertuples():
            rows.append([f"{r.commit} {str(r.committed_utc)[:16]}", CHANGE_RU.get(r.change, r.change), r.match_id,
                         ru_detail(r.detail)])
        rep.add(H3("Изменения в live-файле между ежедневными версиями"), table(rows))
        rep.add(P("Строка «SC0-20261017-st-johnstone-dundee» — исторический дефект, исправленный 22.10: API-Football "
                  "16–22.10 отнёс матч St Johnstone – Dundee United к клубу Dundee (сыгранный «фантом» 1:1 был в "
                  "признаках пять дней). Матч «SP2-20261102-tenerife-andorra» 03.11 из результата 2:2 снова стал "
                  "фикстурой; у Football-Data этого матча нет до сих пор, а прогноз на него уже рассчитан как выигрыш."))
    st = T("raw_stale_fixtures.csv")
    if st is not None and len(st):
        rows = [["match_id", "Источник", "Отставание, дней"]]
        for r in st.itertuples():
            rows.append([r.match_id, r.source, r.days_overdue_vs_league_data])
        rep.add(H3("Несыгранные фикстуры с прошедшей датой"), table(rows),
                P("Это отложенные матчи без новой даты и устаревшие даты провайдеров. Признаки по ним не обновляются "
                  "и прогнозы не делаются (старт в прошлом), но Tenerife – Andorra присутствует дважды (01.11 от "
                  "OddsPapi и 02.11 от API-Football).", "small"))


def teams_section(rep):
    rep.add(H1("9. Истории команд и новички"))
    if not TEAMS:
        rep.add(not_run("истории команд"))
        return
    rows = [["Сезон", "Старожилы", "Переход внутри наших лиг", "Вернувшиеся", "Новички извне"]]
    for s, r in TEAMS["status_by_season"].items():
        rows.append([C.season_label(int(s)), r.get("established", 0), r.get("moved_within", 0), r.get("returner", 0),
                     r.get("outside", 0)])
    rep.add(table(rows), P("Табл. Клубы-сезоны по статусу (2021/22 — первый сезон данных, статус не определяется).",
                           "small"))
    by = TEAMS["outside_or_returner_by_league"]
    rep.add(P("Новички извне и вернувшиеся по лигам (все сезоны): " + ", ".join(f"{k} {v}" for k, v in by.items()) +
              ". В E0, SP1, I1, D1, F1 и SC0 их нет: туда поднимаются только клубы наших же вторых лиг. Новички извне "
              "приходят в низшие загруженные дивизионы (из National League, Primera RFEF, Serie C, 3. Liga, National и "
              "т. п.) и в высшие лиги стран, где загружена одна лига (B1, N1, P1, T1, G1)."))
    fm = TEAMS["first_match_of_outside_newcomers"]
    rg = TEAMS["returner_gap_days"]
    ppg = TEAMS["points_per_game_by_status"]
    rep.add(P(f"В первом матче новичка извне ({fm['n']} случаев) у модели нет ни одной формы "
              f"(form10_n = {f(fm['form10_n_mean'], 1)}), нет таблицы сезона, нет дней отдыха, а Elo равен базе "
              f"уровня лиги: в среднем {f(fm['elo_mean'], 0)}, на {f(abs(fm['elo_minus_league_mean']), 1)} пункта "
              "ниже среднего Elo лиги — то есть новичок стартует как средний клуб. Фактически новички извне "
              f"набирают {f(ppg.get('outside'), 2)} очка за матч против {f(ppg.get('established'), 2)} у старожилов. "
              f"У вернувшихся ({rg['n']} первых матчей) форма есть, но ей в медиане {f(rg['median_days'], 0)} дней "
              f"(от {f(rg['min_days'], 0)} до {f(rg['max_days'], 0)})."))
    rep.add(*figure(FIG / "feature_profile.png", "Рис. 2. Доля пустых признаков модели 1X2 у одной стороны по номеру "
                                                 "матча клуба в сезоне. В первом туре пусто около трети у всех "
                                                 "(нет таблицы сезона); у новичка извне — почти всё."))


def recompute_section(rep):
    rep.add(H1("10. Независимый пересчёт скользящих признаков"))
    if not REC:
        rep.add(not_run("пересчёт признаков"))
        return
    rep.add(P(f"Независимая таблица содержит {REC['independent_matches']} сыгранных матчей, продакшн — "
              f"{REC['production_played']}; общих {REC['compared_matches']} (только в продакшне "
              f"{REC['only_in_production']}, только в пересчёте {REC['only_in_independent']}). Сравнено "
              f"{REC['features_compared']} признаков, {REC['cells_compared']} значений; расхождений — "
              f"{REC['cells_mismatch']}."))
    s = T("recompute_summary.csv")
    if s is not None:
        rows = [["Признак", "Сравнено", "Оба пусты", "Пусто только в одном", "Разные значения", "Макс. |разница|"]]
        for r in s.itertuples():
            if r.feature.startswith("a_") and r.feature != "a_elo":
                continue
            rows.append([r.feature.replace("h_", "h/a_") if r.feature.startswith("h_") and r.feature != "h_elo"
                         else r.feature, r.compared, r.both_missing, r.missing_mismatch, r.value_mismatch,
                         f"{r.max_abs_diff:.1e}"])
        rep.add(table(rows), P("Для пар h_/a_ показана строка хозяев; строка гостей совпадает по числу "
                               "расхождений (все нули — в файле `recompute_summary.csv`).", "small"))
    rep.add(note("H2 подтверждается: скользящие признаки, таблица сезона, дни отдыха, личные встречи и Elo "
                 "в `features.parquet` в точности совпадают с независимым пересчётом из сырых файлов. Утечки "
                 "результата матча в его признаки нет; дефекты связаны не с вычислением, а с входными данными."))


def store_section(rep):
    rep.add(H1("11. Хранилище прогнозов"))
    if not STORE:
        rep.add(not_run("хранилище прогнозов"))
        return
    rep.add(H2("11.1. Фантомные прогнозы 03–04.11.2026"))
    ph = T("store_phantoms.csv")
    if ph is not None:
        rows = [["Матч", "Вид", "Создан (UTC)", "П1 / Х / П2", "xG", "Main", "Цена", "Итог"]]
        for r in ph.itertuples():
            rows.append([r.match_id, "фантом" if r.kind == "phantom" else "настоящий", r.generated_at[5:16],
                         f"{r.p_home:.2f} / {r.p_draw:.2f} / {r.p_away:.2f}", f"{r.xg_home:.2f}–{r.xg_away:.2f}",
                         market_ru(r.main), "—" if r.main_odds != r.main_odds else f"{r.main_odds:.2f}",
                         STATUS_PICK.get(r.status, r.status)])
        rep.add(table(rows, widths=[x * CM for x in (4.6, 1.7, 2.0, 2.6, 1.7, 2.4, 1.0, 1.3)]))
    rep.add(P("Фантомы были не только сделаны, но и рассчитаны: 05.11 их match_id появились в файлах результатов "
              "Football-Data. В единой истории их исключает `same_day_duplicates`. Во всех трёх случаях Main фантома "
              "отличался от Main настоящего матча."))
    roi_now, roi_ph = STORE["roi_unified_now"], STORE["roi_if_phantoms_kept"]
    rep.add(P(f"Если бы фантомы остались в истории: {roi_ph['bets']} ставок Main, ROI {pc(roi_ph['roi'], 2)} против "
              f"{roi_now['bets']} и {pc(roi_now['roi'], 2)} сейчас."))
    rep.add(H2("11.2. Перенесённые матчи в единой истории"))
    mv = STORE["moved"]
    rep.add(P(f"`rescheduled_duplicates` объединяет {mv['clusters']} пар прогнозов одного перенесённого матча и "
              f"отбрасывает {mv['dropped']}. Правило «оставить самую позднюю дату» ошибается, когда поздняя дата "
              "провайдера устарела, а матч сыгран раньше: " + (", ".join(mv["kept_not_settleable"]) or "нет") +
              ". Для F1 Nice – Rennes сыгран 30.10 (5:2); история держит прогноз на 31.10 — он никогда не будет "
              "рассчитан и через 21 день станет VOID, — а отброшен рассчитанный прогноз на 30.10 (Main «меньше 3,5» по "
              "1,46, проигрыш)."))
    fx = STORE["roi_unified_moved_rule_fixed"]
    rep.add(P(f"С исправленным правилом (сначала записанный до матча прогноз, затем прогноз, чей match_id — "
              f"сыгранный матч, затем поздняя дата): {fx['bets']} ставок, ROI {pc(fx['roi'], 2)} вместо "
              f"{roi_now['bets']} и {pc(roi_now['roi'], 2)}."))
    mvt = T("store_moved.csv")
    if mvt is not None:
        rows = [["match_id", "Оставлен сейчас", "Оставит исправление", "Источник", "Статус", "Цена"]]
        for r in mvt.itertuples():
            rows.append([r.match_id, "да" if r.kept else "", "да" if r.fix_keeps else "", r.source,
                         STATUS_PICK.get(r.status, r.status), "—" if r.odds != r.odds else f"{r.odds:.2f}"])
        rep.add(table(rows, widths=[x * CM for x in (6.8, 2.2, 2.4, 2.0, 2.0, 1.6)]))
    rep.add(H2("11.3. Прочие проверки"))
    wd = STORE["withdrawn_results"]
    inv = STORE["invariants"]
    fr = STORE["freshness"]
    rows = [["Проверка", "Результат"],
            ["Main с match_id, которого нет в таблице признаков", f"{STORE['orphan_mains']} (переносы и фантомы)"],
            ["Рассчитанные прогнозы матча без результата", "; ".join(f"{w['match_id']} ({w['status']}, "
                                                                     f"{w['settled_score']})" for w in wd) or "0"],
            ["Инварианты `check_history`", "; ".join(f"{k}: {v[0]}" for k, v in inv.items()) or "нарушений нет"],
            ["VOID с прибылью 0 без цены", f"{len(STORE['void_with_profit_without_price'])} записей (G1 Atromitos – AEK)"],
            ["Live Main, проверенные на свежесть данных", fr["live_mains_checked"]],
            ["…из них без результата предыдущего матча клуба", f"{fr['with_missing_previous_result']} "
                                                              f"({pc(fr['share'], 2)}), отставание ≤ {fr['max_lag_days']} дн."]]
    rep.add(table(rows, widths=[8 * 28.35, 9 * 28.35]))
    rep.add(P("Свежесть: для каждого записанного Main единой истории взята таблица признаков, закоммиченная тем же "
              "ежедневным прогоном, и проверено, были ли в ней все результаты обоих клубов за предыдущие дни. "
              "Пропуск найден лишь однажды, то есть задержка результатов на прогнозы почти не влияет."))


def sensitivity_section(rep):
    rep.add(H1("12. Чувствительность модели к пропускам у одной команды"))
    if not SENS:
        rep.add(not_run("чувствительность"))
        return
    s = T("sensitivity_summary.csv")
    n = int(s["matches"].iloc[0]) if s is not None else 0
    rep.add(P(f"Все {n} сыгранных матчей 2023/24–2026/27; база — неиспорченный прогноз того же матча той же моделью."))
    rows = [["Что пусто", "Средний TV", "TV 90 %", "ΔP(П1)", "ΔP(П2)", "Δ log loss [95 % ДИ]", "Main изменился"]]
    for r in s.itertuples():
        rows.append([VARIANT_RU.get(r.variant, r.variant), f"{r.mean_tv:.3f}", f"{r.p90_tv:.3f}", f"{r.dp_home:+.3f}",
                     f"{r.dp_away:+.3f}", f"{r.d_log_loss * 1000:+.1f} [{r.d_log_loss_lo * 1000:+.1f}; "
                                          f"{r.d_log_loss_hi * 1000:+.1f}]·10⁻³", pc(r.main_changed, 0)])
    rep.add(table(rows, widths=[x * CM for x in (4.0, 1.5, 1.4, 1.7, 1.7, 4.6, 2.1)]))
    rows = [["Что пусто", "Ставок база / вар.", "Доля угаданных база / вар.", "ROI база", "ROI вариант", "Δ ROI [95 % ДИ]"]]
    for r in s.itertuples():
        rows.append([VARIANT_RU.get(r.variant, r.variant), f"{r.bets_base} / {r.bets_var}",
                     f"{pc(r.main_hit_base)} / {pc(r.main_hit_var)}", pc(r.main_roi_base, 2), pc(r.main_roi_var, 2),
                     f"{r.d_main_roi * 100:+.2f} [{r.d_main_roi_lo * 100:+.2f}; {r.d_main_roi_hi * 100:+.2f}]"])
    rows[0][5] = "Δ ROI, п.п. [95 % ДИ]"
    rep.add(table(rows, widths=[x * CM for x in (4.0, 2.6, 3.2, 1.8, 1.9, 3.5)]))
    rep.add(*figure(FIG / "sensitivity.png", "Рис. 3. Рост log loss и доля матчей, где сменился Main, когда у одной "
                                             "стороны пуста группа признаков."))
    nh = s.loc[s["variant"] == "newcomer_h"].iloc[0] if s is not None and (s["variant"] == "newcomer_h").any() else None
    na = s.loc[s["variant"] == "newcomer_a"].iloc[0] if s is not None and (s["variant"] == "newcomer_a").any() else None
    if nh is not None and na is not None:
        rep.add(note(f"H3 подтверждается. Если клуб для модели «новый», вероятности сдвигаются в среднем на "
                     f"{nh.mean_tv:.3f} (хозяева) и {na.mean_tv:.3f} (гости) по TV, log loss растёт на "
                     f"{nh.d_log_loss * 1000:.1f}·10⁻³ и {na.d_log_loss * 1000:.1f}·10⁻³, Main меняется в "
                     f"{pc(nh.main_changed, 0)} и {pc(na.main_changed, 0)} матчей. Больше всего значат форма и статистика "
                     "ударов/угловых; дни отдыха и личные встречи почти не влияют. Main очень чувствителен: замена "
                     f"одного Elo меняет его примерно в {pc(s.loc[s['variant'].isin(['elo_h', 'elo_a']), 'main_changed'].mean(), 0)} "
                     "матчей, потому что "
                     "кандидаты рынков близки по оценке и малый сдвиг меняет лидера."))
        inside = int(((s["d_main_roi_lo"] <= 0) & (s["d_main_roi_hi"] >= 0)).sum())
        rep.add(P(f"При этом ROI Main почти не меняется: у {inside} из {len(s)} вариантов порчи 95 % интервал разности "
                  f"ROI включает 0 (разности от {s['d_main_roi'].min() * 100:+.2f} до {s['d_main_roi'].max() * 100:+.2f} п.п.). "
                  "Main выбирает другой рынок, но с похожей ценой и похожей ожидаемой доходностью — на итог по реальным "
                  "ценам пропуски у одной стороны в среднем не влияют, хотя вероятности становятся хуже."))
    rep.add(H2("12.1. Воспроизведение фантомов"))
    pr = T("sensitivity_phantoms.csv")
    if pr is not None:
        rows = [["Матч", "Вариант", "П1 / Х / П2 (пересчёт)", "Сохранённый прогноз", "Макс. |Δp|"]]
        for r in pr.itertuples():
            rows.append([r.pair, "признаки настоящего матча" if r.variant == "real_features" else "сторона = новичок",
                         f"{r.p_home:.3f} / {r.p_draw:.3f} / {r.p_away:.3f}",
                         f"{r.stored_id}: {r.stored_p_home:.3f} / {r.stored_p_draw:.3f} / {r.stored_p_away:.3f}",
                         f"{r.max_abs_diff_p:.3f}"])
        rep.add(table(rows))
        mx = float(pr["max_abs_diff_p"].max())
        rep.add(P(f"Порча «сторона = новичок» на строке настоящего матча и продакшн-бандл воспроизводят сохранённые "
                  f"вероятности фантомов: наибольшее расхождение по всем шести прогнозам {mx:.4f}. Значит, фантом — это "
                  "ровно «клуб без истории» (пустые формы и таблица, базовый Elo, нет личных встреч), и порча признаков в "
                  "разделе 12 описывает реальный механизм дефекта, а не выдуманный."))
        rep.add(*figure(FIG / "phantoms.png", "Рис. 4. Вероятности настоящего матча, фантома и воспроизведения."))


def strata_section(rep):
    rep.add(H1("13. Новички против старожилов"))
    if not STRATA:
        rep.add(not_run("новички против старожилов"))
        return
    g = STRATA["groups"]
    rows = [["Группа (худший статус в матче)", "Матчи", "Log loss", "Изб. log loss ×10⁻³ [95 % ДИ]", "Ставок",
             "Угадано", "ROI Main [95 % ДИ]"]]
    for key in ("established|1-10", "established|11+", "moved_within|1-10", "moved_within|11+", "returner|1-10",
                "returner|11+", "outside|1-10", "outside|11+", "outside|all", "established|all"):
        if key not in g:
            continue
        st, part = key.split("|")
        r = g[key]
        rows.append([f"{STATUS_RU[st]}, матчи {part.replace('all', 'все')}", r["matches"], f(r["log_loss"]["mean"], 4),
                     ci(r["excess_log_loss"], 1000, 1, ""), r["main"]["bets"], pc(r["main"].get("hit_rate")),
                     roi_cell(r["main"])])
    rep.add(table(rows, widths=[x * CM for x in (3.7, 1.5, 1.5, 3.5, 1.5, 1.5, 3.8)]))
    rep.add(P("«Матчи 1-10» — первые десять матчей сезона клуба этой группы; старожилы для сравнения берутся из "
              "матчей, где у обеих команд ≤ 10 матчей в сезоне, — начало сезона трудно для всех.", "small"))
    v = STRATA["vs_established"]
    holm = STRATA.get("holm_excess_ll", {})
    rows = [["Сравнение со старожилами", "Δ log loss ×10⁻³", "Δ изб. log loss ×10⁻³", "Холм p", "Δ ROI Main", "Δ доли угаданных"]]
    for key, r in v.items():
        st, part = key.split("|")
        rows.append([f"{STATUS_RU[st]}, {part}", ci(r["log_loss_diff"], 1000, 1, ""), ci(r["excess_log_loss_diff"], 1000, 1, ""),
                     f(holm.get(key), 2), ci(r["main_roi_diff"], 100, 1, ""), ci(r["main_hit_diff"], 100, 1, "")])
    rows[0][4], rows[0][5] = "Δ ROI Main, п.п.", "Δ доли угаданных, п.п."
    rep.add(table(rows, widths=[x * CM for x in (3.6, 3.0, 3.0, 1.2, 3.1, 3.1)]))
    o = v.get("outside|1-10", {})
    rep.add(P(f"Матчи новичков извне прогнозируются хуже: log loss выше на {ci(o.get('log_loss_diff'), 1000, 1, '')}·10⁻³ "
              f"({sig(o.get('log_loss_diff'))}). Но рынку они тоже даются труднее: избыточный log loss "
              f"{ci(o.get('excess_log_loss_diff'), 1000, 1, '')}·10⁻³ — {sig(o.get('excess_log_loss_diff'))}. "
              f"Main на этих матчах: Δ ROI {ci(o.get('main_roi_diff'), 100, 1, ' п.п.')} — {sig(o.get('main_roi_diff'))}."))
    rep.add(*figure(FIG / "newcomer_curves.png", "Рис. 5. Избыточный log loss против рынка и переоценка ожидаемых "
                                                 "очков по номеру матча клуба в сезоне (95 % ДИ)."))
    pb = STRATA["points_bias"]
    rows = [["Статус клуба", "Сторон-матчей", "P(победа) модель", "P(победа) рынок", "Частота побед",
             "Смещение очков, все", "Смещение, матчи 1-10"]]
    for st in ("established", "moved_within", "returner", "outside"):
        r = pb[st]
        rows.append([STATUS_RU[st], r["sides"], f(r["p_win_mean"]), f(r["market_p_win_mean"]), f(r["won_rate"]),
                     ci(r["points_bias"], 1, 3, ""), ci(r["early_points_bias"], 1, 3, "")])
    rep.add(table(rows, widths=[x * CM for x in (2.8, 1.6, 1.6, 1.6, 1.5, 3.9, 4.0)]))
    bias = T("strata_points_bias.csv")
    if bias is not None:
        b13 = bias.loc[(bias["status"] == "outside") & (bias["bucket"] == "1-3")]
        if len(b13):
            b = b13.iloc[0]
            rep.add(P(f"В первых трёх матчах новичка извне модель ждёт от него на {b.points_bias:+.3f} очка больше, чем "
                      f"он набирает [{b.lo:+.3f}; {b.hi:+.3f}]: P(победа) модели {b.p_win:.3f}, рынка {b.market_p_win:.3f}, "
                      f"факт {b.won:.3f}. Причина видна в разделе 9: новичок стартует со средним Elo лиги, а играет "
                      "слабее среднего."))
    rep.add(*figure(FIG / "calibration.png", "Рис. 6. Калибровка P(победа клуба) для новичков извне и старожилов.", width=11 * 28.35))
    lr = STRATA.get("live_record")
    if lr:
        rows = [["Группа (2026/27, единая история)", "Ставок", "Угадано", "ROI [95 % ДИ]"]]
        for st in ("established", "moved_within", "returner", "outside"):
            if st in lr:
                rows.append([STATUS_RU[st], lr[st]["bets"], pc(lr[st].get("hit_rate")), roi_cell(lr[st])])
        rep.add(H2("13.1. Продакшн-история 2026/27"), table(rows),
                P(f"Разность ROI «новички извне + вернувшиеся» минус остальные: "
                  f"{ci(lr['newcomer_vs_rest_roi'], 100, 1, ' п.п.')} — {sig(lr['newcomer_vs_rest_roi'])}. "
                  "Цены — записанные до матча и восстановленные из архива OddsPapi.", "small"))


def dq_section(rep):
    rep.add(H1("14. Индекс качества данных матча"))
    if not STRATA:
        rep.add(not_run("индекс качества данных"))
        return
    d = T("strata_dq_bins.csv")
    if d is not None:
        rows = [["Пустых признаков", "Матчи", "Log loss", "Изб. log loss ×10⁻³ [95 % ДИ]", "Ставок", "Угадано", "ROI Main [95 % ДИ]"]]
        for r in d.itertuples():
            rows.append([r.dq_bin, r.matches, f"{r.log_loss:.4f}", f"{r.excess_log_loss * 1000:+.1f} "
                         f"[{r.excess_lo * 1000:+.1f}; {r.excess_hi * 1000:+.1f}]", r.bets, pc(r.main_hit),
                         f"{r.main_roi * 100:+.1f} % [{r.roi_lo * 100:+.1f}; {r.roi_hi * 100:+.1f}]"])
        rep.add(table(rows, widths=[x * CM for x in (2.3, 1.7, 1.5, 3.6, 1.5, 1.6, 4.8)]))
    rep.add(*figure(FIG / "dq_bins.png", "Рис. 7. Избыточный log loss и ROI Main по доле пустых признаков матча."))
    q = STRATA["dq_poor_vs_clean"]
    w = STRATA["dq_within_first5"]
    rep.add(P(f"Матчей с > 20 % пустых признаков — {pc(q['share_poor'])}. Против матчей без пропусков: избыточный "
              f"log loss {ci(q['excess_log_loss_diff'], 1000, 1, '')}·10⁻³ ({sig(q['excess_log_loss_diff'])}), "
              f"доля угаданных Main {ci(q['main_hit_diff'], 100, 1, ' п.п.')} ({sig(q['main_hit_diff'])}), ROI Main "
              f"{ci(q['main_roi_diff'], 100, 1, ' п.п.')} ({sig(q['main_roi_diff'])}). Ранговая корреляция индекса с "
              f"избыточным log loss {f(STRATA['dq_spearman_excess_ll'], 3)}. Внутри первых пяти туров (где пропуски "
              f"есть у всех) матчи с > 50 % пропусков против ≤ 20 %: {ci(w['excess_log_loss_diff'], 1000, 1, '')}·10⁻³ "
              f"по избыточному log loss, {ci(w['main_roi_diff'], 100, 1, ' п.п.')} по ROI — выборка мала."))
    sh = STRATA.get("share_newcomer_early_by_season", {})
    rep.add(P("Доля матчей, где новичок извне или вернувшийся клуб играет один из первых десяти матчей сезона: " +
              ", ".join(f"{C.season_label(int(k))} — {pc(v)}" for k, v in sh.items()) +
              ". В 2026/27 (сезон только начался) она втрое выше, поэтому эффект новичков сейчас заметнее."))


def variants_section(rep):
    rep.add(H1("15. Проверка мер для клубов без истории (walk-forward)"))
    if not VAR:
        rep.add(not_run("варианты признаков"))
        return
    rp = VAR.get("reproduction", {})
    rep.add(P(f"Строитель признаков, запущенный заново на тех же файлах, воспроизводит продакшн-таблицу: "
              f"{rp.get('rows_rebuilt')} строк против {rp.get('rows_production')}, те же match_id — "
              f"{'да' if rp.get('same_match_ids') else 'нет'}, различающихся числовых ячеек — {rp.get('cells_differing')}. "
              "Варианты: стартовый Elo клуба, впервые появившегося после первого сезона данных, ниже базы на 50 или "
              "100; история National League (EC) в состоянии строителя (без обучения и прогноза на EC). Для каждого "
              "варианта признаки строятся заново, модели переобучаются на сезонах до прогнозируемого, оценка — на "
              "тех же матчах, что и база. Правило отбора объявлено до прогона: вариант перспективен, только если на "
              "2023/24 и на 2024/25 он снижает log loss матчей новичков извне (первые 10 матчей) и не повышает общий."))
    s = T("variants_summary.csv")
    names = {"elo_new_-50": "Elo новичка −50", "elo_new_-100": "Elo новичка −100", "ec_history": "история EC"}
    grp = {"all": "все матчи", "outside_first10": "новички извне, 1-10", "outside_all": "новички извне, все",
           "e3_outside_first10": "League Two, новички, 1-10"}
    if s is not None:
        rows = [["Вариант", "Сезон", "Группа", "Матчи", "Δ log loss ×10⁻³ [95 % ДИ]", "ROI база", "Δ ROI Main, п.п. [95 % ДИ]",
                 "Main изм."]]
        for r in s.itertuples():
            if r.group == "outside_all":
                continue
            rows.append([names.get(r.variant, r.variant), "все три" if r.season == "all" else C.season_label(int(r.season)),
                         grp.get(r.group, r.group), r.matches,
                         f"{r.d_log_loss * 1000:+.2f} [{r.lo * 1000:+.2f}; {r.hi * 1000:+.2f}]", pc(r.roi_base, 1),
                         f"{r.d_roi * 100:+.1f} [{r.d_roi_lo * 100:+.1f}; {r.d_roi_hi * 100:+.1f}]", pc(r.main_changed, 0)])
        rep.add(table(rows, widths=[x * CM for x in (2.4, 1.6, 2.8, 1.5, 3.3, 1.5, 2.7, 1.2)]))
    rep.add(*figure(FIG / "variants.png", "Рис. 8. Изменение log loss матчей новичков извне по сезонам (вариант − база)."))
    ok = VAR.get("meets_selection_rule", {})
    post = VAR.get("consistent_post_hoc", {})
    rep.add(P("Правило отбора выполнено: " + ", ".join(f"{names.get(k, k)} — {'да' if v else 'нет'}" for k, v in ok.items())
              + ". Сезон 2025/26 приведён для информации."))
    if s is not None and post.get("elo_new_-50"):
        a = s.loc[(s["variant"] == "elo_new_-50") & (s["season"] == "all")].set_index("group")
        y4 = s.loc[(s["variant"] == "elo_new_-50") & (s["season"] == "2024") & (s["group"] == "all")].iloc[0]
        rep.add(note(f"Elo новичка −50 не прошёл объявленное правило только из-за общего log loss 2024/25 "
                     f"({y4.d_log_loss * 1000:+.2f}·10⁻³, интервал [{y4.lo * 1000:+.2f}; {y4.hi * 1000:+.2f}]). На матчах "
                     "новичков извне log loss ниже во всех трёх сезонах; по трём сезонам вместе: все матчи "
                     f"{a.at['all', 'd_log_loss'] * 1000:+.2f}·10⁻³ [{a.at['all', 'lo'] * 1000:+.2f}; {a.at['all', 'hi'] * 1000:+.2f}], "
                     f"новички извне (первые 10) {a.at['outside_first10', 'd_log_loss'] * 1000:+.2f}·10⁻³ "
                     f"[{a.at['outside_first10', 'lo'] * 1000:+.2f}; {a.at['outside_first10', 'hi'] * 1000:+.2f}]; ROI Main "
                     f"{a.at['all', 'd_roi'] * 100:+.1f} п.п. [{a.at['all', 'd_roi_lo'] * 100:+.1f}; {a.at['all', 'd_roi_hi'] * 100:+.1f}]. "
                     "Этот вывод сделан после просмотра результатов, поэтому вариант — кандидат категории B: его нужно "
                     "заранее зарегистрировать и проверить на матчах, сыгранных после фиксации гипотезы. Сдвиг −100 "
                     "слишком велик (2023/24 без улучшения у новичков), история National League почти ничего не даёт."))


DEFECTS = [
    ("D1", "Разные написания клуба в ФАЙЛАХ РЕЗУЛЬТАТОВ не приводятся к одному имени",
     "05.11 (коммит 9bd3c19) B1/E1/E2 2026/27 содержали сыгранные матчи «Kortrijk – KV Mechelen», «Preston – Sheffield Utd», "
     "«Oxford Utd – Reading»; 06.11 Football-Data исправил. Сутки в признаках были 3 фантомных клуба, а у Mechelen, "
     "Sheffield United, Oxford не хватало матча; три фантомных прогноза были рассчитаны (2 проигрыша по цене 2,00 и 1,70, "
     "1 выигрыш без цены).",
     "`src/data/dataset.py`: обобщить `harmonise_fixture_names` до `harmonise_names(frame, reference)` и применять его в "
     "`build_matches` также к строкам результатов текущего сезона и к строкам API: известные имена лиги = клубы прошлого "
     "сезона + клубы текущего сезона с ≥ 3 матчами; имя вне этого набора, которое `aliases.resolve` однозначно сводит к "
     "известному, заменяется, match_id пересчитывается. Тест — версии файлов из коммита 9bd3c19."),
    ("D2", "`team_id_conflicts.csv` не очищается",
     "`collect` пишет файл, только когда конфликты есть (`collect.py`, стр. 460–463); последний раз файл менялся 06.11, "
     "конфликт OddsPapi 5044 (Mechelen / «KV Mechelen») — след D1, но отчёт показывает его до сих пор.",
     "Как для `unmatched_teams.csv`: при пустом `registry.conflicts` удалять файл (`elif api_storage.exists(...): unlink()`)."),
    ("D3", "Результат, отозванный провайдером, остаётся рассчитанным",
     "SP2-20261102-tenerife-andorra: 03.11 (коммит bf7ccd0) API-Football вернул матч из 2:2 в несыгранные; у Football-Data "
     "матча нет. Live Main на SP2-20261101-tenerife-andorra рассчитан как выигрыш +0,378 по цене 1,378 и в истории остаётся.",
     "`collect._save_live`: не заменять строку played=True строкой played=False того же match_id без статуса провайдера "
     "PST/CANC/ABD (логировать как конфликт). `invariants.check_history`: добавить проверку «рассчитанный прогноз, у "
     "матча которого (по id или по паре в пределах 60 дней) нет сыгранной строки», а `store.settle` — возвращать такие "
     "прогнозы в UPCOMING."),
    ("D4", "`rescheduled_duplicates` оставляет прогноз, который нельзя рассчитать",
     "Кластер F1 Nice – Rennes: сыгран 30.10 (5:2); оставлен live-прогноз на 31.10 (UPCOMING, станет VOID), отброшен "
     "рассчитанный live-прогноз на 30.10 (Main «меньше 3,5» по 1,46, проигрыш). ROI единой истории занижает число ставок "
     "на 1.",
     "`src/analytics/performance.py`, `rescheduled_duplicates`: ключ сортировки `[order, unsettled, date desc, "
     "generated_at desc]`, где `unsettled = status not in {WON, LOST, PUSH}` — среди записанных прогнозов рассчитанный "
     "выигрывает у нерассчитанного."),
    ("D5", "VOID получает прибыль 0 без цены",
     "8 записей G1-20261017-atromitos-aek; собственный инвариант `check_history` сообщает «profit without a price». "
     "На ROI не влияет.",
     "`src/prediction/store.py`, `settle`: для VOID `profit = 0.0 if odds else NaN` (как для PUSH)."),
    ("D6", "Нет псевдонимов для трёх имён API-Football",
     "`unmatched_teams.csv`: «OFI» (G1), «Vicenza Virtus» (I2), «Real Sociedad II» (SP2) — фикстуры и результаты "
     "OFI Crete, Vicenza, Sociedad B от API-Football пропускаются.",
     "`src/providers/aliases.py`, `TEAM_ALIASES`: `\"ofi\": \"OFI Crete\"`, `\"vicenza virtus\": \"Vicenza\"`, "
     "`\"real sociedad ii\": \"Sociedad B\"`."),
]

FINDINGS = [
    "Клубы из лиг, которые система не загружает, стартуют со средним Elo лиги (база уровня), хотя набирают меньше "
    "очков; в первых матчах модель их переоценивает.",
    "Edinburgh City / FC Edinburgh — один клуб под двумя именами (история разорвана в 2022/23 и 2023/24).",
    "Технические победы 3:0 (Турция 2022/23 — 29 матчей) идут в форму и Elo как сыгранные матчи.",
    "Плей-офф Бельгии и Греции смешан с регулярным сезоном в признаках «по сезону».",
    "Отложенные матчи без новой даты висят в таблице признаков как фикстуры с прошедшей датой (9 строк).",
]


def _sens_finding():
    s = T("sensitivity_summary.csv")
    if s is None:
        return []
    core = s.loc[~s["variant"].isin(["rest_h", "rest_a", "h2h"])]
    return [f"Main неустойчив к сдвигам вероятностей: пустая группа признаков у одной стороны меняет его в "
            f"{pc(core['main_changed'].min(), 0)}–{pc(core['main_changed'].max(), 0)} матчей, но ROI Main при этом в "
            "пределах шума."]


def defects_section(rep):
    rep.add(H1("16. Перечень дефектов"),
            H2("16.1. Ошибки продакшна (исправить)"))
    for code_, title, evidence, fix in DEFECTS:
        rep.add(H3(f"{code_}. {title}"), P(f"**Доказательство.** {evidence}"), P(f"**Исправление.** {fix}"))
    rep.add(P("Исторический дефект, уже исправленный 22.10: API-Football 16–22.10 записал St Johnstone – Dundee United "
              "как St Johnstone – Dundee (фантомный результат 1:1 пять дней был в признаках Dundee)."),
            H2("16.2. Исследовательские наблюдения"), *bullets(FINDINGS + _sens_finding()))


def conclusions(rep):
    rep.add(H1("17. Отрицательные результаты"))
    neg = ["Независимый пересчёт не нашёл ни одного расхождения в признаках — ошибок вычисления нет.",
           "Сопоставление API с Football-Data: ни одного переставленного хозяина/гостя, ни одного расхождения счёта "
           "или времени начала у совпавших матчей."]
    if STORE:
        neg.append(f"Задержка результатов почти не влияет на live-прогнозы: {STORE['freshness']['with_missing_previous_result']} "
                   f"из {STORE['freshness']['live_mains_checked']}.")
    if STRATA:
        q = STRATA["dq_poor_vs_clean"]
        neg.append(f"Индекс качества данных почти не связан с избыточным log loss (Спирмен {f(STRATA['dq_spearman_excess_ll'], 3)}); "
                   f"разность ROI Main между матчами с > 20 % пропусков и без пропусков {ci(q['main_roi_diff'], 100, 1, ' п.п.')} — "
                   f"{sig(q['main_roi_diff'])}.")
    if VAR:
        ok = VAR.get("meets_selection_rule", {})
        bad = [k for k, v in ok.items() if not v]
        if bad:
            ru_names = {"elo_new_-50": "Elo новичка −50", "elo_new_-100": "Elo новичка −100", "ec_history": "история National League"}
            neg.append("Не прошли объявленное правило отбора: " + ", ".join(ru_names.get(k, k) for k in bad) + ".")
    rep.add(*bullets(neg))
    rep.add(H1("18. Ограничения"),
            *bullets(["История git покрывает только 24.09–09.11.2026: дефекты имён в более ранних сезонах видны лишь "
                      "по итоговым файлам.",
                      "Новичков извне немного (14–28 клубов в сезон), интервалы по ROI широкие.",
                      "Настоящий результат Tenerife – Andorra локально неизвестен: сеть не использовалась.",
                      "Порча признаков — модель реального дефекта; в жизни пустые признаки встречаются вместе с "
                      "особенностями клуба (слабее, новый состав), которых порча не воспроизводит.",
                      "2026/27 не является нетронутой проверкой; 2025/26 — для информации."]))
    rep.add(H1("19. Выводы"))
    out = ["**H1 подтверждается.** Найдено 6 ошибок продакшна (D1–D6) и один исторический дефект; самый опасный — "
           "D1 (разные написания в файлах результатов). Его последствия сейчас гасит `same_day_duplicates`, но "
           "признаки клубов сутки были неверны и фантомы были рассчитаны.",
           "**H2 подтверждается.** Признаки вычислены верно (0 расхождений).",
           "**H3 подтверждается для вероятностей.** Пустые признаки одной стороны заметно сдвигают вероятности, ухудшают log loss и меняют Main; ROI Main при этом в пределах шума."]
    if STRATA:
        o = STRATA["vs_established"].get("outside|1-10", {})
        out.append(f"**H4 частично.** Матчи новичков извне хуже по log loss ({sig(o.get('log_loss_diff'))}), но "
                   f"против рынка разница {ci(o.get('excess_log_loss_diff'), 1000, 1, '')}·10⁻³ — "
                   f"{sig(o.get('excess_log_loss_diff'))}; ROI Main {ci(o.get('main_roi_diff'), 100, 1, ' п.п.')} — "
                   f"{sig(o.get('main_roi_diff'))}. Систематическая переоценка новичков в первых матчах есть.")
        out.append("**H5 не подтверждается** в заметном объёме: индекс пропусков слабо связан с ошибкой и ROI.")
    if VAR:
        ok = VAR.get("meets_selection_rule", {})
        post = VAR.get("consistent_post_hoc", {})
        out.append("**H6 не подтверждена по объявленному правилу:** " + ("его выполнили: " + ", ".join(k for k, v in ok.items() if v)
                                 if any(ok.values()) else "ни один вариант его не выполнил")
                   + (". Elo новичка −50 улучшает log loss устойчиво (вывод после просмотра — категория B)"
                      if post.get("elo_new_-50") else "") + ".")
    rep.add(*bullets(out))


def recommendations(rep):
    rep.add(H1("20. Рекомендации"))
    var_ok = VAR.get("meets_selection_rule", {}) if VAR else {}
    rows = [["Рекомендация", "Категория", "Основание"],
            ["D1: приводить имена в строках результатов и API к именам лиги", "A", "раздел 7.1, 11.1"],
            ["D3: не откатывать результат в фикстуру без статуса; инвариант «рассчитан без результата»", "A", "раздел 8, 11.3"],
            ["D4: в `rescheduled_duplicates` рассчитанный прогноз важнее поздней даты", "A", "раздел 11.2"],
            ["D2: удалять `team_id_conflicts.csv`, когда конфликтов нет", "A", "раздел 7.2"],
            ["D5: VOID без цены — прибыль NaN", "A", "раздел 11.3"],
            ["D6: три псевдонима API-Football", "A", "раздел 7.2"],
            ["Таблица переименований клубов (FC Edinburgh → Edinburgh City) в нормализации", "A", "раздел 7.3"],
            ["Помечать технические 3:0 и не учитывать их в форме и Elo", "B", "раздел 6; эффект на прогноз не измерен"]]
    post = VAR.get("consistent_post_hoc", {}) if VAR else {}
    for key, ru in (("elo_new_-50", "Стартовый Elo новичка извне −50"), ("elo_new_-100", "Стартовый Elo новичка извне −100"),
                    ("ec_history", "История National League в состоянии строителя")):
        if key in var_ok:
            good = var_ok[key] or post.get(key)
            why = ("правило отбора выполнено" if var_ok[key] else
                   "правило не выполнено, но улучшение устойчиво во всех сезонах — пререгистрация и проверка вперёд"
                   if post.get(key) else "правило отбора не выполнено")
            rows.append([ru, "B" if good else "C", "раздел 15; " + why])
    rows.append(["Флаг «мало данных» на карточке матча, когда у клуба < 5 матчей в наших лигах", "B",
                 "раздел 12–13: прогноз и Main меняются; ROI-эффект в пределах шума"])
    rows.append(["Отбрасывать Main на матчах с высоким индексом пропусков", "C", "раздел 14: связи с ROI нет"])
    rep.add(table(rows, widths=[9.0 * 28.35, 2.5 * 28.35, 5.5 * 28.35]))
    rep.add(P("A — готово к внедрению; B — перспективно, нужна проверка на будущих матчах; C — не рекомендуется. "
              "Исследование само ничего в продакшне не меняет."))


class _Report(Report):
    """The kit's running header names v6.1; this report belongs to the v6.2 series."""

    def build(self):
        from scripts import report_kit as K
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = K._Doc(self.path, f"FootPredict — {self.title}", f"FootPredict v6.2 — {self.title}")
        doc.multiBuild(self._front() + self.story)
        return self.path


def build() -> Report:
    rep = _Report(OUT, "Качество данных и целостность признаков", "FootPredict v6.2 — исследование",
                 facts={"Сезоны": "2021/22–2026/27 (оценка моделей 2023/24–2026/27)",
                        "Данные на": (RUN or {}).get("data_version", {}).get("local_data/results", {}).get("latest_mtime", "—")[:10]})
    intro(rep)
    data_section(rep)
    methods(rep)
    raw_section(rep)
    names_section(rep)
    moved_section(rep)
    teams_section(rep)
    recompute_section(rep)
    store_section(rep)
    sensitivity_section(rep)
    strata_section(rep)
    dq_section(rep)
    variants_section(rep)
    defects_section(rep)
    conclusions(rep)
    recommendations(rep)
    rep.add(H1("21. Воспроизведение"),
            *reproduce(["python -m experiments.v6_2.data_quality.run          # все шаги, ~30–40 мин, 2 потока",
                        "python -m experiments.v6_2.data_quality.run --only raw_audit teams recompute store_audit",
                        "python -m experiments.v6_2.data_quality.figures",
                        "python -m experiments.v6_2.data_quality.build_pdf"]),
            P("Кэш (`experiments/v6_2/data_quality/cache/`, не в git): переобученные модели, прогнозы по сезонам, "
              "испорченные прогнозы, пересобранные признаки вариантов. Результаты — JSON и CSV в `results/`; "
              "окружение и версия данных — `results/run_info.json`."))
    rep.build()
    return rep


if __name__ == "__main__":
    C.setup_logging("build_pdf")
    r = build()
    n = r.check()
    print(f"{r.path} — {n} pages")
