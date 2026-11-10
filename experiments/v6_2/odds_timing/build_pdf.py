"""Builds reports/v6_2/Odds_Timing_Research.pdf from results/*.json and figures/*.png (Russian).

    python -m experiments.v6_2.odds_timing.build_pdf

Every number is read from the result files of this folder; nothing is typed in by hand.
"""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from scripts.report_kit import (H1, H2, P, Report, bullets, figure, formula, note, reproduce, table)

from . import common as C

OUT = C.ROOT / "reports" / "v6_2" / "Odds_Timing_Research.pdf"
J = C.load_json
AUD, DR, ST, CLV, RUN = J("audit.json"), J("drift.json"), J("stability.json"), J("clv.json"), J("run_info.json")
FIG = C.FIGURES


class Report62(Report):
    """The series kit with the v6.2 running title."""

    def build(self):
        from scripts.report_kit import _Doc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = _Doc(self.path, f"FootPredict — {self.title}", f"FootPredict v6.2 — {self.title}")
        doc.multiBuild(self._front() + self.story)
        return self.path


def T(rows, widths=None):
    """A table (widths in cm, summing to 17) followed by a small gap."""
    return [table(rows, widths=[w * cm for w in widths] if widths else None), Spacer(1, 6)]


def f1(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def pc(v, d=1, signed=False):
    """fraction -> percent text"""
    if v is None:
        return "—"
    return f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%"


def ci(c, d=1, mult=100):
    if not c or c[0] is None:
        return "—"
    return f"[{c[0] * mult:+.{d}f}; {c[1] * mult:+.{d}f}]"


def roi_cell(r):
    if not r or not r.get("bets"):
        return "—"
    return f"{pc(r['roi'], 1, True)} {ci(r['roi_ci95'])}"


def n(v):
    return "—" if v is None else f"{int(v)}"


CLS_RU = {"A1": "A1 — цена записана при генерации", "A2": "A2 — добавлена из архива, опубликована до генерации",
          "B": "B — добавлена из архива, опубликована после генерации", "C": "C — закрытие (backfill)"}


# ------------------------------------------------------------------------------------------------ sections
def front(rep: Report):
    c = AUD["counts"]
    t = AUD["timing"]
    rm, rr = AUD["roi"]["main"], AUD["roi"]["risk"]
    rep.add(H1("Краткое содержание"))
    rep.add(note(
        f"Из {c['picks_priced']} сохранённых Main/Risk 2026/27 с ценой {c['by_class'].get('A1', 0)} имеют цену, записанную "
        f"в момент генерации (A1), {c['by_class'].get('A2', 0)} — цену из архива, опубликованную до генерации (A2), "
        f"{c['by_class'].get('B', 0)} — цену из архива, опубликованную ПОСЛЕ генерации (B), {c['by_class'].get('C', 0)} — "
        f"цену закрытия ретроспективного (backfill) прогноза (C). Ни одна цена класса A1 не опубликована после генерации "
        f"(проверка в UTC: {t['A1_quote_after_generation']} случаев, и все они — условное время Football-Data 00:00)."))
    rep.add(bullets([
        f"ROI live-Main как показано сейчас (все цены): {roi_cell(rm['current_live'])}; только цены, существовавшие при "
        f"генерации (A1+A2): {roi_cell(rm['live_A1_A2'])}; цены класса B: {roi_cell(rm['live_B'])}.",
        f"ROI live-Risk: все цены {roi_cell(rr['current_live'])}; A1+A2 {roi_cell(rr['live_A1_A2'])}; B {roi_cell(rr['live_B'])}.",
        f"Цены класса B опубликованы в медиане через {f1(t['quote_after_gen_h_B']['median'])} ч после генерации и за "
        f"{f1(t['quote_lead_h_B']['median'], 2)} ч до начала матча — это фактически цены закрытия.",
        f"Backfill-прогнозы сгенерированы после матча; их цена — закрытие (медиана публикации за "
        f"{f1(t['quote_lead_h_C']['median'], 2)} ч до начала), и сам выбор Main/Risk сделан по этим ценам.",
        f"Модель предсказывает направление движения линии: при контроле других букмекеров цена Pinnacle смещается к "
        f"модели на b₁ ≈ {f1(min(v['beta'][1] for k, v in DR['model_vs_drift_controlled'].items() if '1X2' in k), 3)}…"
        f"{f1(max(v['beta'][1] for k, v in DR['model_vs_drift_controlled'].items() if '1X2' in k), 3)} разрыва (1X2, все сезоны "
        f"значимы), но этого мало, чтобы перекрыть маржу: справедливый CLV Main A1 = "
        f"{pc(CLV['main|A1']['clv_fair']['mean'], 1, True)} {ci(CLV['main|A1']['clv_fair']['ci95'])}.",
    ]))


def goal(rep: Report):
    rep.add(H1("1. Цель исследования"))
    rep.add(P("Определить, КОГДА FootPredict получает цену для своего решения (Main — самый надёжный рынок матча, Risk — "
              "альтернатива с коэффициентом около 3 и выше), как цена меняется от ранней котировки к закрытию, как от этого "
              "зависят EV, выбор Main, риск, связь с результатом и CLV (closing line value), и выполнить аудит, который "
              "запросил владелец продукта: для каждой сохранённой Main/Risk 2026/27 с ценой определить, существовала ли цена "
              "в момент генерации, и как меняется ROI, если использовать только такие цены."))
    rep.add(H1("2. Проблема"))
    rep.add(P("Прогноз FootPredict пересчитывается при каждом запуске конвейера до начала матча; хранится последняя версия. "
              "Цены приходят из нескольких источников с разным временем: Football-Data (снимок пятницы/вторника и закрытие), "
              "The Odds API и OddsPapi (живые котировки с 22.10.2026 и архив /historical-odds — последняя котировка перед "
              "матчем). Модуль `src/prediction/history_repair.py` дополняет прогнозы без цены последней котировкой архива до "
              "начала матча — она может быть опубликована после того, как прогноз был сделан. Тогда ROI истории частично "
              "посчитан по ценам, которых в момент решения не было. Кроме того, в таблицах два разных часовых пояса: "
              "`generated_at` — UTC, а `kickoff`, `odds_priced_at`, `collected_at`, `priced_at`, `fetched_at` — местное время "
              "Великобритании (до 25.10.2026 это UTC+1)."))
    rep.add(H1("3. Гипотезы"))
    rep.add(bullets([
        "H1. Цены заметно движутся между ранней котировкой и закрытием, и закрытие точнее предсказывает результат.",
        "H2. Часть цен истории 2026/27 опубликована после генерации прогноза; без них ROI истории другой.",
        "H3. Модель несёт информацию, которую рынок учитывает позже (цена движется к модели) → положительный CLV.",
        "H4. Выбор Main/Risk зависит от момента, когда взята цена; более поздняя (закрывающая) цена меняет выбор и ROI.",
    ]))


def data(rep: Report):
    rep.add(H1("4. Данные"))
    dv = RUN["data_version"] if RUN else {}
    rows = [["Файл", "Строк", "Изменён"]]
    for k, v in dv.items():
        if v and ("parquet" in k):
            rows.append([f"`{k}`", n(v.get("rows")), v.get("mtime", "—")])
    rep.add(T(rows, None))
    rep.add(P("Football-Data: `local_data/results/<сезон>/*.csv`, сезоны 2021/22–2026/27, Bet365, Pinnacle (до 2025/26), "
              "среднее рынка, Betfair Exchange (с 2024/25) — снимок открытия и закрытия. Прогнозы модели для прошлых сезонов — "
              "walk-forward кэш v6.1 (модели обучены только на сезонах до прогнозируемого, параметры продакшна), для 2026/27 — "
              "сохранённые прогнозы продакшна (единая история сайта)."))
    rep.add(P("Поля времени OddsPapi: `priced_at` — когда букмекер выставил цену, `fetched_at` — когда FootPredict её "
              "скачал. Строка живого сбора (`fetched_at` < начала матча) известна как действующая на интервале "
              "[`priced_at`, `fetched_at`]; строка архива (`fetched_at` ≥ начала) — это последняя котировка до начала, она "
              "действовала на [`priced_at`, начало). Цена в момент t считается известной, только если t лежит внутри такого "
              "интервала; котировка, опубликованная после t, для момента t никогда не используется."))


def methods(rep: Report):
    rep.add(H1("5. Методы"))
    rep.add(P("Снятие маржи. Для полного рынка из k исходов с ценами o_{i} вероятности Шина:"))
    rep.add(formula("q_{i} = (√(z² + 4(1−z)·π_{i}²/Σπ) − z) / (2(1−z)),  π_{i} = 1/o_{i},  Σ q_{i} = 1"))
    rep.add(P("Дрейф: Δq = q_{закр} − q_{откр}. Log loss: −log q_{y}. CLV двух видов:"))
    rep.add(formula("CLV_{цена} = o_{взятая} / o_{закрытие того же букмекера} − 1,   CLV_{спр} = o_{взятая} · q_{закр} − 1"))
    rep.add(P("q_{закр} — вероятность закрытия без маржи: Betfair Exchange из Football-Data (1X2, тотал 2.5; маржа около 1 %), "
              "иначе полный рынок закрытия OddsPapi одного букмекера (Pinnacle, затем Bet365); для матчей без архива — последняя "
              "живая котировка не позже 3 ч до начала. CLV_{спр} — ожидаемая доходность ставки, если рынок закрытия верен."))
    rep.add(P("Связь модели с движением линии (контроль шума открытия по другим букмекерам того же снимка):"))
    rep.add(formula("Δq_{PS} = a + b_{1}(p − q_{PS,откр}) + b_{2}(q_{B365,откр} − q_{PS,откр}) + b_{3}(q_{сред,откр} − q_{PS,откр})"))
    rep.add(P("Классы цены прогноза (аудит): A1 — цена записана конвейером при генерации; A2 — добавлена позже из архива, "
              "но опубликована не позже генерации; B — добавлена из архива и опубликована после генерации (до начала); "
              "C — цена backfill-прогноза, сделанного после матча (закрытие). Все сравнения времени — в UTC."))
    rep.add(P("Неопределённость: 95 % бутстреп-интервалы (2000 повторов, где не указано иное; кластер = матч, если у матча "
              "несколько строк); разности — парный бутстреп на тех же матчах; ROI = средняя прибыль на ставку 1."))


def params(rep: Report):
    rep.add(H1("6. Параметры"))
    p = RUN["parameters"] if RUN else {}
    env = RUN["environment"] if RUN else {}
    rows = [["Параметр", "Значение"]] + [[k, str(v)] for k, v in p.items()]
    rows += [["seed", str(env.get("seed"))], ["n_jobs", str(env.get("n_jobs"))], ["git SHA прогона", env.get("git_sha", "—")[:12]]]
    rep.add(T(rows, None))


def audit_section(rep: Report):
    rep.add(H1("7. Аудит времени цен Main/Risk 2026/27"))
    c, t = AUD["counts"], AUD["timing"]
    rep.add(P(f"Универсум — единая история сайта (live, если прогноз был записан до начала, иначе backfill): "
              f"{c['picks_total']} Main/Risk, из них с ценой {c['picks_priced']}."))
    rows = [["Класс", "Main", "Risk", "Всего"]]
    for k in ("A1", "A2", "B", "C"):
        br = c["by_class_role"].get(k, {})
        rows.append([CLS_RU[k], n(br.get("main", 0)), n(br.get("risk", 0)), n(c["by_class"].get(k, 0))])
    rep.add(T(rows, None))
    rep.add(H2("7.1 По источнику, провайдеру и рынку"))
    rows = [["Срез", "A1", "A2", "B", "C"]]
    for src, v in c["by_source_class"].items():
        rows.append([f"источник: {src}"] + [n(v.get(k, 0)) for k in ("A1", "A2", "B", "C")])
    for prov, v in c["by_provider_class"].items():
        rows.append([f"провайдер: {prov}"] + [n(v.get(k, 0)) for k in ("A1", "A2", "B", "C")])
    for grp, v in c["by_group_class"].items():
        rows.append([f"рынок: {C.GROUP_RU.get(grp, grp)}"] + [n(v.get(k, 0)) for k in ("A1", "A2", "B", "C")])
    for org, v in c["by_origin_class"].items():
        rows.append([f"odds_origin: {org}"] + [n(v.get(k, 0)) for k in ("A1", "A2", "B", "C")])
    rep.add(T(rows, [7, 2.5, 2.5, 2.5, 2.5]))
    rows = [["Букмекер", "A1", "A2", "B", "C"]]
    for b, v in c["by_bookmaker_class"].items():
        rows.append([b] + [n(v.get(k, 0)) for k in ("A1", "A2", "B", "C")])
    rep.add(T(rows, None))
    vc = AUD["A1_version_check"]
    rep.add(P(f"Проверка A1 по архиву версий (`versions.parquet`, пишется в момент генерации): цена совпадает у "
              f"{vc['verified_equal']} прогнозов, отличается у {vc['differs']}, у {vc['not_in_version_archive']} версия "
              f"старше архива версий (до v5) — для них класс A1 следует из кода конвейера тех версий, но проверить нельзя. "
              f"Время котировки известно у {t['A1_with_quote_time']} A1; опубликованных после генерации (в UTC) — "
              f"{t['A1_quote_after_generation']}, и это условное время Football-Data (см. раздел 9). "
              f"Для цен B ни одной другой котировки, действовавшей в момент генерации, в архиве нет "
              f"({AUD.get('B_any_quote_valid_at_generation', 0)} найдено)."))
    rep.add(H2("7.2 Время генерации и время цены"))
    g = t["live_generation_lead_h"]
    rep.add(P(f"Хранимая (последняя до начала) версия live-прогноза сохранена в медиане за {f1(g['median'])} ч до начала "
              f"(квартили {f1(g['q25'])}–{f1(g['q75'])} ч; ≤ 1 ч — {pc(g['share_le_1h'])}, ≤ 6 ч — {pc(g['share_le_6h'])}, "
              f"≤ 24 ч — {pc(g['share_le_24h'])}). Первая версия появляется в медиане за "
              f"{f1(t['live_first_version_lead_h']['median'])} ч; версий на матч — в медиане "
              f"{f1(t['live_versions_per_match']['median'], 0)}."))
    rows = [["Класс", "Котировка − генерация, ч (медиана [кв.])", "До начала, ч (медиана [кв.])"]]
    for k in ("A1", "A2", "B", "C"):
        a, b = t[f"quote_after_gen_h_{k}"], t[f"quote_lead_h_{k}"]
        rows.append([k, f"{f1(a['median'])} [{f1(a['q25'])}; {f1(a['q75'])}]",
                     f"{f1(b['median'], 2)} [{f1(b['q25'], 2)}; {f1(b['q75'], 2)}]"])
    rep.add(T(rows, None))
    rep.add(figure(FIG / "lead_times.png", "Рис. 1. Слева — за сколько часов до начала сохранена хранимая версия live-прогноза; "
                                           "справа — за сколько часов до начала опубликована использованная цена (A1, B, C)."))
    rep.add(H2("7.3 ROI только по ценам, существовавшим при генерации"))
    rows = [["Набор", "Main: ставок", "Main: ROI [95 % ДИ]", "Risk: ставок", "Risk: ROI [95 % ДИ]"]]
    labels = [("current_all_sources", "все источники (как на сайте)"), ("current_live", "live, все цены"),
              ("live_A1", "live, только A1"), ("live_A1_A2", "live, A1 + A2 (были при генерации)"),
              ("live_A2", "live, A2"), ("live_B", "live, B (после генерации)"), ("backfill_C", "backfill, C (закрытие)")]
    for k, lab in labels:
        m, r = AUD["roi"]["main"][k], AUD["roi"]["risk"][k]
        rows.append([lab, n(m["bets"]), roi_cell(m), n(r["bets"]), roi_cell(r)])
    rep.add(T(rows, [5.4, 1.6, 3.9, 1.6, 4.5]))
    dm, dr = AUD["roi"]["main"]["live_diff_genonly_minus_current"], AUD["roi"]["risk"]["live_diff_genonly_minus_current"]
    rep.add(P(f"Если цены B считать недоступными (ставки нет), ROI live-Main меняется на {pc(dm['diff'], 1, True)} "
              f"(95 % ДИ {ci(dm['ci95'])} п.п.), live-Risk — на {pc(dr['diff'], 1, True)} ({ci(dr['ci95'])} п.п.). "
              f"Разница статистически не значима, но направление для Main понятно: прогнозы, которые при генерации не имели "
              f"цены, выбирались с понижающим множителем «без цены» и в основном на коротких коэффициентах "
              f"(средний коэффициент B-Main {f1(AUD['roi']['main']['live_B']['avg_odds'], 2)} против "
              f"{f1(AUD['roi']['main']['live_A1']['avg_odds'], 2)} у A1); их средний EV по добавленной цене "
              f"{pc(AUD['ev']['B|main']['mean_ev'], 1, True)} против {pc(AUD['ev']['A1|main']['mean_ev'], 1, True)} у A1."))
    rep.add(figure(FIG / "audit_classes.png", "Рис. 2. Классы цен и ROI по классам (95 % ДИ)."))
    rep.add(H2("7.4 Что такое «момент решения» для backfill"))
    bf = t["backfill_closing_published_before"]
    rep.add(P(f"Backfill-прогноз создан после матча моделью, обученной до начала сезона, по признакам только из данных до "
              f"матча — это вневыборочная оценка, но не решение, которое можно было исполнить. Его цена — последняя котировка "
              f"до начала (закрытие), и выбор Main/Risk сделан с учётом этой цены (EV входит в правило выбора), поэтому "
              f"средний EV backfill-прогнозов выше: Main {pc(AUD['ev']['C|main']['mean_ev'], 1, True)}, Risk "
              f"{pc(AUD['ev']['C|risk']['mean_ev'], 1, True)} (live A1: {pc(AUD['ev']['A1|main']['mean_ev'], 1, True)} и "
              f"{pc(AUD['ev']['A1|risk']['mean_ev'], 1, True)})."))
    rep.add(P(f"Номинальный момент решения можно задать как время до начала, типичное для live: медиана live — "
              f"{f1(bf['median_live_lead_h'])} ч. Закрывающая котировка backfill была уже опубликована за ≥ 24 ч до начала "
              f"лишь у {pc(bf['share_ge_24h'])}, за ≥ {f1(bf['median_live_lead_h'])} ч — у {pc(bf['share_ge_median_live_lead'])}, "
              f"за ≥ 1 ч — у {pc(bf['share_ge_1h'])}. Значит, почти все цены backfill в номинальный момент решения ещё не "
              f"существовали; ROI backfill — это ROI ставки по закрытию, а не воспроизведение решений продакшна."))


def drift_section(rep: Report):
    rep.add(H1("8. Как движется цена"))
    rep.add(H2("8.1 Football-Data: открытие → закрытие, 2021/22–2026/27"))
    rows = [["Сезон", "Рынок", "Букм.", "Матчей", "|Δq|, п.п.", "|Δq|>5 п.п.", "Маржа откр./закр.", "LL закр.−откр. ×1000 [ДИ]"]]
    for k, v in DR["fd"]["by_season"].items():
        s, m, b = k.split("|")
        if b not in ("b365", "ps", "bfe"):
            continue
        d = v["ll_close_minus_open"]
        rows.append([C.season_label(int(s)), {"TOTAL_2.5": "Т2.5"}.get(m, m), {"b365": "B365", "ps": "PS", "bfe": "BFE"}[b],
                     n(v["matches"]), f1(v["mean_abs_dq_pp"], 2), pc(v["share_abs_gt_5pp"]),
                     f"{pc(v['margin_open'])} / {pc(v['margin_close'])}",
                     f"{d['diff'] * 1000:+.2f} [{d['ci95'][0] * 1000:+.2f}; {d['ci95'][1] * 1000:+.2f}]"])
    rep.add(T(rows, [1.6, 1.2, 1.2, 1.4, 1.5, 1.7, 2.6, 5.8]))
    a = DR["fd"]["all_seasons_b365|1X2"]
    rep.add(P(f"Bet365 1X2 за все сезоны: средний |Δq| = {f1(a['mean_abs_dq_pp'], 2)} п.п., фаворит меняется в "
              f"{pc(a['favourite_changes'])} матчей, закрытие лучше открытия по log loss на "
              f"{a['ll_close_minus_open']['diff'] * 1000:+.2f}×10⁻³ ({ci(a['ll_close_minus_open']['ci95'], 2, 1000)}×10⁻³). "
              f"Маржа Bet365 одинакова на открытии и закрытии, у Pinnacle и особенно у биржи Betfair она к закрытию падает "
              f"(у биржи открытие неликвидно — его не стоит брать как эталон)."))
    rep.add(figure(FIG / "fd_drift.png", "Рис. 3. Средний модуль дрейфа вероятности Bet365 и выигрыш закрытия по log loss (95 % ДИ)."))
    rows = [["Вероятность на открытии", "Строк", "Средний Δq, п.п. [95 % ДИ]"]]
    for k, v in DR["fd"]["by_bucket"].items():
        rows.append([k, n(v["n"]), f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}; {v['ci95'][1]:+.3f}]"])
    rep.add(P("Направление дрейфа Bet365 1X2 по уровню вероятности (Δq в п.п.; сдвиги малы):"))
    rep.add(T(rows, None))

    rep.add(H2("8.2 Модель и движение линии"))
    rows = [["Сезон", "Рынок", "Букм.", "Строк", "Наклон b [95 % ДИ]", "corr", "Цена идёт к модели при |p−q|>3 п.п."]]
    for k, v in DR["model_vs_drift"].items():
        s, m, b = k.split("|")
        rows.append([C.season_label(int(s)) + (" (инф.)" if s in ("2025", "2026") else ""), {"TOTAL_2.5": "Т2.5"}.get(m, m),
                     {"b365": "B365", "ps": "PS", "bfe": "BFE"}[b], n(v["n"]),
                     f"{v['slope']:.3f} [{v['ci95'][0]:.3f}; {v['ci95'][1]:.3f}]", f1(v["corr"], 3),
                     pc(v["share_moves_towards_model_if_gap_gt_3pp"])])
    rep.add(T(rows, [2.2, 1.2, 1.2, 1.5, 4.0, 1.3, 5.6]))
    rep.add(P("Большие наклоны для биржи Betfair объясняются шумом её неликвидного открытия: любая разумная оценка "
              "«предсказывает» возврат такой цены. Поэтому главный тест — регрессия с контролем других букмекеров того же "
              "снимка (b₁ — вклад модели):"))
    rows = [["Сезон", "Рынок", "Строк", "b₁ модель [95 % ДИ]", "b₂ B365−PS [95 % ДИ]", "b₃ сред.−PS [95 % ДИ]"]]
    for k, v in DR["model_vs_drift_controlled"].items():
        s, m = k.split("|")
        b, c_ = v["beta"], v["ci95"]
        rows.append([C.season_label(int(s)) + (" (инф.)" if s == "2025" else ""), {"TOTAL_2.5": "Т2.5"}.get(m, m), n(v["n"]),
                     f"{b[1]:.3f} [{c_[1][0]:.3f}; {c_[1][1]:.3f}]", f"{b[2]:.3f} [{c_[2][0]:.3f}; {c_[2][1]:.3f}]",
                     f"{b[3]:.3f} [{c_[3][0]:.3f}; {c_[3][1]:.3f}]"])
    rep.add(T(rows, [2.2, 1.2, 1.5, 4.0, 4.0, 4.1]))
    rep.add(figure(FIG / "model_drift.png", "Рис. 4. Доля разрыва «модель − Pinnacle на открытии», которую цена Pinnacle проходит к закрытию."))

    rep.add(H2("8.3 OddsPapi 2026/27: цена по горизонтам"))
    lh = DR["live_horizons"]
    rep.add(P(f"Живой сбор OddsPapi (22.10–09.11.2026) наблюдал цены {lh['matches']} матчей в разные моменты; эталон — "
              f"последнее наблюдение не позже 3 ч до начала (медиана {f1(lh['reference_lead_h']['median'], 2)} ч). "
              f"Наблюдений за 3–24 ч до начала в сборе нет (запуски конвейера раз в сутки утром)."))
    rows = [["Рынок", "Когда", "Строк", "Матчей", "|Δq|, п.п. [95 % ДИ]", "|Δq|>2 п.п.", "Цена не менялась"]]
    for k, v in lh["by_group_bucket"].items():
        g, b = k.split("|")
        if v["matches"] < 20:
            continue
        m = v["mean_abs_dq_pp"]
        rows.append([C.GROUP_RU.get(g, g), b, n(v["n"]), n(v["matches"]), f"{m['mean']:.2f} [{m['ci95'][0]:.2f}; {m['ci95'][1]:.2f}]",
                     pc(v["share_abs_gt_2pp"]), pc(v["share_unchanged"])])
    rep.add(T(rows, [3.0, 1.6, 1.3, 1.4, 3.6, 2.2, 3.9]))
    ll = lh.get("ll_1x2_by_bucket") or {}
    if ll:
        rep.add(P("1X2: log loss ранней цены минус поздней (те же матчи): " + "; ".join(
            f"{b}: {v['diff'] * 1000:+.1f}×10⁻³ [{v['ci95'][0] * 1000:+.1f}; {v['ci95'][1] * 1000:+.1f}], n={v['n']}"
            for b, v in ll.items()) + "."))
    rep.add(figure(FIG / "live_horizons.png", "Рис. 5. Отличие ранней цены OddsPapi от цены за ≤ 3 ч до начала (95 % ДИ, кластер = матч)."))
    ca = DR["closing_age"]
    rows = [["Рынок", "Цен закрытия", "Медиана возраста, ч", "≥ 1 ч", "≥ 6 ч", "≥ 24 ч"]]
    for g in ("1X2", "TOTAL", "HANDICAP", "BTTS", "DOUBLE_CHANCE", "CORNERS", "ALL"):
        if g not in ca:
            continue
        v = ca[g]
        sh = v["share_published_at_least_h_before"]
        rows.append([C.GROUP_RU.get(g, "все") if g != "ALL" else "все", n(v["n"]), f1(v["median_lead_h"], 2),
                     pc(sh["1"]), pc(sh["6"]), pc(sh["24"])])
    rep.add(P("Возраст закрывающей цены архива — за сколько часов до начала она уже была выставлена (с этого момента и "
              "до начала цена не менялась):"))
    rep.add(T(rows, None))
    rep.add(figure(FIG / "closing_age.png", "Рис. 6. Доля закрывающих цен OddsPapi, уже опубликованных за h часов до начала."))


def stability_section(rep: Report):
    rep.add(H1("9. Стабильность выбора Main/Risk"))
    v = ST["versions"]
    rows = [["Версия за … до начала", "Main: версий", "Main: другой выбор", "Risk: другой выбор"]]
    from .stability import LEAD_RU
    for b in LEAD_RU:
        m, r = v["by_role_bucket"].get(f"main|{b}"), v["by_role_bucket"].get(f"risk|{b}")
        if m:
            rows.append([b, n(m["n"]), pc(m["changed"]), pc(r["changed"]) if r else "—"])
    rep.add(T(rows, None))
    fv, fr = v["first_vs_final"], v["first_vs_final_roi"]
    rep.add(P(f"Между соседними версиями одной модели Main меняется в {pc(v['consecutive_same_model']['main']['flip'])} "
              f"случаев, Risk — в {pc(v['consecutive_same_model']['risk']['flip'])}. Первая опубликованная версия "
              f"(медиана {f1(fv['main']['median_lead_first_h'])} ч до начала) отличается от финальной у "
              f"{pc(fv['main']['changed'])} Main и {pc(fv['risk']['changed'])} Risk. Читатель, поставивший по первой версии "
              f"по её цене: Main {roi_cell(fr['main']['first'])} ({fr['main']['first']['bets']} ставок), по финальной — "
              f"{roi_cell(fr['main']['final'])}; на одних матчах финальная − первая = "
              f"{pc(fr['main']['diff_final_minus_first_same_matches']['diff'], 1, True)} "
              f"{ci(fr['main']['diff_final_minus_first_same_matches']['ci95'])}; Risk: "
              f"{pc(fr['risk']['diff_final_minus_first_same_matches']['diff'], 1, True)} "
              f"{ci(fr['risk']['diff_final_minus_first_same_matches']['ci95'])}."))
    rep.add(H2("9.1 История: правило Main по цене открытия и по цене закрытия (Bet365, Football-Data)"))
    rows = [["Сезон", "Матчей", "Другой Main", "Выбор/цена откр.", "Выбор откр./цена закр.", "Выбор/цена закр.",
             "Закр.−откр. (выбор), п.п."]]
    for s, r in ST["history"].items():
        d = r["diff_close_choice_minus_open_choice"]
        rows.append([C.season_label(int(s)) + (" (инф.)" if s == "2025" else ""), n(r["matches"]), pc(r["share_different_main"]),
                     roi_cell(r["open_choice@open"]), roi_cell(r["open_choice@close"]), roi_cell(r["close_choice@close"]),
                     f"{d['diff'] * 100:+.1f} {ci(d['ci95'])}"])
    rep.add(T(rows, [1.9, 1.3, 1.6, 3.0, 3.0, 3.0, 3.2]))
    rows = [["Сезон", "EV откр.", "EV тех же по закр.", "EV выбора по закр.", "CLV спр. к PS закр. (откр.)"]]
    for s, r in ST["history"].items():
        rows.append([C.season_label(int(s)), pc(r["open_choice@open"]["mean_ev"], 1, True),
                     pc(r["open_choice@close"]["mean_ev"], 1, True), pc(r["close_choice@close"]["mean_ev"], 1, True),
                     pc(r["open_choice@open"]["mean_clv_fair_ps"], 1, True)])
    rep.add(T(rows, None))
    rep.add(figure(FIG / "stability.png", "Рис. 7. Слева — доля версий, выбор которых отличается от финального; справа — ROI Main "
                                          "при цене открытия и закрытия Bet365 (95 % ДИ)."))
    lr = ST["live_rerun"]
    rep.add(H2("9.2 2026/27: текущее правило при ценах генерации и при ценах закрытия"))
    rep.add(P(f"Матчи live с архивом закрытия: {lr['matches']}. При генерации цена была у {pc(lr['gen_priced_share'])} "
              f"выборов, в архиве закрытия — у {pc(lr['close_priced_share'])}. Сохранённые выборы делались правилами, "
              f"действовавшими тогда (правила менялись в октябре 2026), поэтому сравнение идёт по текущему правилу во всех "
              f"трёх «мирах»: цены генерации; закрытие только для тех же выборов, что имели цену (чистый эффект времени); "
              f"закрытие для всех выборов (время + покрытие)."))
    rows = [["Роль", "Другой выбор (то же покрытие)", "Другой выбор (полное)", "Цены генерации", "Те же выборы по закрытию",
             "Выбор по закр. (то же покр.)", "Выбор по закр. (полное)"]]
    for role in ("main", "risk"):
        r = lr[role]
        rows.append(["Main" if role == "main" else "Risk", pc(r["changed_gen_vs_close_same"]), pc(r["changed_gen_vs_close_full"]),
                     roi_cell(r["gen@gen_price"]), roi_cell(r["gen_choice@close_price"]), roi_cell(r["close_same@close"]),
                     roi_cell(r["close_full@close"])])
    rep.add(T(rows, [1.2, 2.0, 2.0, 3.0, 3.0, 2.9, 2.9]))
    rm, rr = lr["main"], lr["risk"]
    rep.add(P(f"Те же выборы по цене закрытия вместо цены генерации: Main {pc(rm['diff_gen_choice_close_minus_gen_price']['diff'], 1, True)} "
              f"{ci(rm['diff_gen_choice_close_minus_gen_price']['ci95'])}, Risk "
              f"{pc(rr['diff_gen_choice_close_minus_gen_price']['diff'], 1, True)} {ci(rr['diff_gen_choice_close_minus_gen_price']['ci95'])} "
              f"(п.п. ROI, парно). Цены Main к закрытию сокращаются (ранняя цена выгоднее), цены Risk — растут. "
              f"Выбор по закрытию при том же покрытии: Main {pc(rm['diff_close_same_minus_gen']['diff'], 1, True)} "
              f"{ci(rm['diff_close_same_minus_gen']['ci95'])}, Risk {pc(rr['diff_close_same_minus_gen']['diff'], 1, True)} "
              f"{ci(rr['diff_close_same_minus_gen']['ci95'])}."))


def clv_section(rep: Report):
    rep.add(H1("10. CLV Main и Risk"))
    rows = [["Роль|класс", "Прогнозов", "CLV цена: n", "CLV цена [ДИ]", "CLV спр.: n", "CLV спр. [ДИ]", "Доля CLV спр.>0"]]
    for k in ("main|A1", "main|A2", "main|B", "main|C", "risk|A1", "risk|B", "risk|C"):
        v = CLV.get(k)
        if not v:
            continue
        a, b = v["clv_odds"], v["clv_fair"]
        rows.append([k.replace("main", "Main").replace("risk", "Risk"), n(v["n"]), n(a["n"]),
                     f"{pc(a['mean'], 2, True)} {ci(a['ci95'], 2)}" if a["mean"] is not None else "—", n(b["n"]),
                     f"{pc(b['mean'], 1, True)} {ci(b['ci95'])}" if b["mean"] is not None else "—",
                     pc(v.get("clv_fair_share_pos"))])
    rep.add(T(rows, [1.8, 1.6, 1.6, 3.4, 1.6, 3.4, 3.6]))
    rows = [["Срез (A1+A2)", "Прогнозов", "CLV цена [ДИ]", "CLV спр. [ДИ]"]]
    for k, v in CLV.items():
        if "|A|" not in k:
            continue
        a, b = v["clv_odds"], v["clv_fair"]
        rows.append([k.replace("|A|", " ").replace("provider=", "провайдер ").replace("group=", "рынок ")
                     .replace("main", "Main").replace("risk", "Risk"), n(v["n"]),
                     f"{pc(a['mean'], 2, True)} {ci(a['ci95'], 2)}" if a.get("mean") is not None else "—",
                     f"{pc(b['mean'], 1, True)} {ci(b['ci95'])}" if b.get("mean") is not None else "—"])
    rep.add(T(rows, [6, 2, 4.5, 4.5]))
    m, r = CLV["main|A1"], CLV["risk|A1"]
    rep.add(P(f"Связь CLV с результатом (A1): корреляция CLV_спр и прибыли — Main {f1(m.get('corr_clv_fair_profit'), 3)}, "
              f"Risk {f1(r.get('corr_clv_fair_profit'), 3)}. ROI при CLV_спр > 0: Main {roi_cell(m.get('roi_clv_fair_pos'))} "
              f"({m['roi_clv_fair_pos']['bets']} ставок), при CLV_спр ≤ 0: {roi_cell(m.get('roi_clv_fair_neg'))}; Risk: "
              f"{roi_cell(r.get('roi_clv_fair_pos'))} и {roi_cell(r.get('roi_clv_fair_neg'))}. Для цен B и C CLV к цене того же "
              f"букмекера равен нулю по построению (это и есть закрытие), а CLV_спр — минус доля маржи закрытия."))
    rep.add(figure(FIG / "clv.png", "Рис. 8. CLV к цене закрытия того же букмекера и к справедливой цене закрытия."))


def bugs(rep: Report):
    t = AUD["timing"]
    c = AUD["counts"]
    rep.add(H1("11. Найденные проблемы продакшна"))
    fd = t["football_data_placeholder"]
    rep.add(P("**1. Условное время котировки Football-Data.** `src/providers/football_data.py` пишет `collected_at = r.date` "
              "(дата матча, 00:00). Это значение попадает в `odds_priced_at` прогноза и показывается на сайте как время "
              "котировки (`src/ui/components.py: price_origin`). Доказательство: у всех "
              f"{fd['n_with_time']} A1-прогнозов с ценой Football-Data и записанным временем оно равно 00:00, и у "
              f"{fd['shown_time_after_generation']} из них «время котировки» позже генерации — хотя цена была в данных в момент "
              "генерации. Исправление: хранить для Football-Data `priced_at = NaT` (время снимка неизвестно) или время загрузки "
              "файла и показывать «Football-Data, снимок без времени»; в проверках времени такие цены считать известными на "
              "момент загрузки файла."))
    org = c["by_origin_class"]
    rep.add(P("**2. Цены, опубликованные после генерации, выглядят как цены прогноза.** `fill_archive_prices` и "
              "`replace_average_prices` (`src/prediction/history_repair.py`) ставят последнюю котировку до начала, не сравнивая "
              "её время с `generated_at`; EV и прибыль пересчитываются по ней, и она входит в ROI записанной истории. "
              f"Из {c['by_class'].get('A2', 0) + c['by_class'].get('B', 0)} добавленных цен {c['by_class'].get('B', 0)} "
              f"опубликованы после генерации (медиана {f1(t['quote_after_gen_h_B']['median'])} ч после). Сайт помечает "
              f"только `odds_origin = \"archive\"`; {org.get('average_replaced', {}).get('B', 0)} цен `average_replaced` "
              "(все опубликованы после генерации) не помечены никак. У "
              f"{sum(c['by_quote_time_from'].get('matched_in_odds_parquet', {}).values())} добавленных цен `odds_priced_at` пуст (заполнены до "
              "появления колонки) — время восстановлено сопоставлением с `odds.parquet`. Исправление: при заполнении сравнивать "
              "время котировки с `generated_at` в UTC и писать класс времени цены (см. рекомендации), заполнять "
              "`odds_priced_at` всегда; показывать метку для всех добавленных цен; официальный ROI записанной истории считать "
              "только по ценам, существовавшим при генерации."))
    rep.add(P("**3. Backfill выбирает Main/Risk по цене закрытия.** `generate_live_predictions` (`src/prediction/pipeline.py`) "
              "строит `real_odds_lookup` и для backfill-матчей, которые уже сыграны: в него попадают все котировки до начала, "
              "т. е. закрытие. Правило выбора использует EV, поэтому выбор backfill зависит от цены, которой в типичный момент "
              f"решения ещё не было (≥ 24 ч до начала была опубликована лишь {pc(t['backfill_closing_published_before']['share_ge_24h'])} "
              "этих цен). Исправление: для backfill брать цену, действовавшую в номинальный момент решения (например, медианное "
              "время live-генерации до начала), по интервалам действия котировок; для этого `parse_odds` при разборе "
              "/historical-odds должен сохранять не только последний снимок до начала, а все снимки (сейчас `_pick_quote` "
              "оставляет только последний)."))
    rep.add(P(f"**4. Два часовых пояса в одной таблице.** `generated_at` — UTC, `odds_priced_at`/`kickoff` — время Великобритании. "
              f"Наивное сравнение без перевода помечает {t['A1_raw_uk_vs_utc_after_generation']} A1-цен как опубликованные после "
              f"генерации (в UTC таких {t['A1_quote_after_generation']}, и все — пункт 1). Ошибкой в коде это пока не является "
              "(продакшн нигде не сравнивает эти поля), но любая будущая проверка «цена до генерации» на сырых колонках будет "
              "неверной. Исправление: хранить все моменты в UTC (или с часовым поясом) либо добавить `generated_at_uk`."))


def negatives(rep: Report):
    rep.add(H1("12. Отрицательные результаты"))
    h = ST["history"]
    rep.add(bullets([
        "Выбор Main по цене закрытия не даёт значимого выигрыша ни в одном сезоне Football-Data (табл. 9.1), "
        "ни в 2026/27 (табл. 9.2).",
        f"Модель двигает линию лишь на несколько процентов разрыва; справедливый CLV Main A1 "
        f"{pc(CLV['main|A1']['clv_fair']['mean'], 1, True)}, Risk A1 {pc(CLV['risk|A1']['clv_fair']['mean'], 1, True)} — "
        "ставки по-прежнему хуже справедливой цены закрытия.",
        "Исключение цен B меняет ROI live-Main на несколько п.п., но различие статистически не значимо.",
        "Наблюдений OddsPapi за 3–24 ч до начала нет (расписание сбора), поэтому кривую дрейфа внутри последних суток "
        "по 2026/27 построить нельзя; Football-Data даёт только два снимка.",
        f"Ранняя (первая) версия прогноза не лучше финальной: разность ROI Main "
        f"{pc(ST['versions']['first_vs_final_roi']['main']['diff_final_minus_first_same_matches']['diff'], 1, True)} в пользу "
        f"финальной, незначимо.",
    ]))
    _ = h


def limits(rep: Report):
    rep.add(H1("13. Ограничения"))
    rep.add(bullets([
        "Архив OddsPapi хранит только последнюю котировку до начала: цену в произвольный момент по архиву можно "
        "установить, лишь если закрывающая котировка была выставлена раньше этого момента.",
        "Живой сбор OddsPapi начался 22.10.2026; для матчей после этой даты архива закрытия нет — используется последняя "
        "живая котировка не позже 3 ч до начала (помечено в данных).",
        "Football-Data «открытие» — снимок пятницы/вторника, не момент генерации прогноза.",
        f"{AUD['A1_version_check']['not_in_version_archive']} A1-прогнозов старше архива версий: класс A1 следует из кода, но цену нельзя сверить с моментом записи.",
        "2026/27 — не нетронутая отложенная выборка (использовалась в v6.1); все её цифры — для информации.",
        "Выборки по классам A2 и B невелики; доверительные интервалы широкие.",
    ]))


def conclusions(rep: Report):
    rep.add(H1("14. Выводы"))
    a = DR["fd"]["all_seasons_b365|1X2"]
    rep.add(bullets([
        f"H1 подтверждена: вероятности Bet365 между открытием и закрытием меняются в среднем на {f1(a['mean_abs_dq_pp'], 1)} п.п., "
        "закрытие точнее открытия во всех сезонах для 1X2 и тотала 2.5.",
        f"H2 подтверждена: {AUD['counts']['by_class'].get('B', 0)} цен live-истории опубликованы после генерации; "
        f"ещё {AUD['counts']['by_class'].get('C', 0)} цен backfill — закрытие. ROI только по ценам, существовавшим при генерации, приведён в разделе 7.3.",
        "H3 подтверждена частично: модель статистически значимо предсказывает направление движения линии, но эффект "
        "мал и не перекрывает маржу (справедливый CLV отрицателен).",
        "H4: выбор Main заметно зависит от того, какие цены есть (покрытие), но собственно время цены при том же "
        "покрытии меняет Main примерно в каждом десятом матче без значимого эффекта на ROI.",
    ]))


def recommendations(rep: Report):
    rep.add(H1("15. Рекомендации (A / B / C)"))
    rows = [["Решение", "Кат.", "Обоснование"],
            ["Хранить для каждой цены прогноза класс времени: `at_generation` (A1), `existed_at_generation` (A2), "
             "`after_generation` (B), `closing_backfill` (C) и время котировки в UTC; показывать на сайте метку для A2/B/C",
             "A", "Разделы 7, 11; изменение учёта, не модели"],
            ["Официальный ROI записанной истории — только A1+A2; B и C показывать отдельно как «оценка по цене закрытия»",
             "A", "Раздел 7.3: цены B недоступны в момент решения"],
            ["Исправить время котировки Football-Data (NaT вместо 00:00 даты матча)", "A", "Раздел 11, п. 1"],
            ["Backfill: брать цену номинального момента решения; сохранять все снимки /historical-odds", "B",
             "Раздел 11, п. 3; нужна загрузка истории"],
            ["Ставить/публиковать Main раньше ради выгодной цены", "B",
             "Цена Main к закрытию сокращается (CLV к той же цене > 0), но справедливый CLV < 0; нужна проверка вперёд"],
            ["Перевыбирать Main по цене закрытия", "C", "Разделы 9.1–9.2: выигрыша нет"],
            ["Хранить моменты времени в UTC (или с поясом)", "A", "Раздел 11, п. 4"]]
    rep.add(T(rows, [10, 1.2, 5.8]))
    rep.add(P("Метки для сайта (рус./укр. тексты — в интерфейсе): A1 — «цена на момент прогноза»; A2 — «цена была в линии "
              "на момент прогноза, добавлена позже»; B — «цена закрытия, опубликована после прогноза — не входит в ROI "
              "записанных прогнозов»; C — «цена закрытия (ретроспективный прогноз)»."))


def run():
    rep = Report62(OUT, "Время цены и CLV", "FootPredict v6.2 — исследование времени коэффициентов",
                 facts={"Данные": "Football-Data 2021/22–2026/27, OddsPapi 2026/27, прогнозы продакшна 2026/27",
                        "Код": "experiments/v6_2/odds_timing/"})
    front(rep)
    goal(rep)
    data(rep)
    methods(rep)
    params(rep)
    audit_section(rep)
    drift_section(rep)
    stability_section(rep)
    clv_section(rep)
    bugs(rep)
    negatives(rep)
    limits(rep)
    conclusions(rep)
    recommendations(rep)
    rep.add(H1("16. Воспроизведение"))
    rep.add(reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                       "# walk-forward кэш v6.1 (experiments/v6_1/main/cache): $PY -m experiments.v6_1.main.run --base --candidates",
                       "$PY -m experiments.v6_2.odds_timing.run",
                       "$PY -m experiments.v6_2.odds_timing.build_pdf"]))
    rep.build()
    pages = rep.check()
    write_findings()
    print("pages", pages)



def findings() -> str:
    """findings.md: the short plain-text summary (same numbers as the PDF)."""
    c, t = AUD["counts"], AUD["timing"]
    rm, rr = AUD["roi"]["main"], AUD["roi"]["risk"]
    ctr = DR["model_vs_drift_controlled"]
    lines = [
        "# FootPredict v6.2 — время цены и CLV: выводы",
        "",
        "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: `reports/v6_2/Odds_Timing_Research.pdf`.",
        "",
        "## Аудит цен Main/Risk 2026/27 (история сайта)",
        "",
        "| Класс | Main | Risk | Всего |",
        "| --- | --- | --- | --- |",
    ]
    for k in ("A1", "A2", "B", "C"):
        br = c["by_class_role"].get(k, {})
        lines.append(f"| {CLS_RU[k]} | {br.get('main', 0)} | {br.get('risk', 0)} | {c['by_class'].get(k, 0)} |")
    lines += ["", "По провайдеру (A1/A2/B/C): " + "; ".join(
        f"{p}: {v.get('A1', 0)}/{v.get('A2', 0)}/{v.get('B', 0)}/{v.get('C', 0)}" for p, v in c["by_provider_class"].items()),
        "", "| Набор | Main ROI [95 % ДИ] | Risk ROI [95 % ДИ] |", "| --- | --- | --- |"]
    for k, lab in (("current_all_sources", "все источники (как на сайте)"), ("current_live", "live, все цены"),
                   ("live_A1_A2", "live, только цены при генерации (A1+A2)"), ("live_A1", "live, A1"),
                   ("live_B", "live, B"), ("backfill_C", "backfill, C")):
        lines.append(f"| {lab} | {roi_cell(rm[k])} ({rm[k]['bets']}) | {roi_cell(rr[k])} ({rr[k]['bets']}) |")
    lines += ["",
              f"- Цены B опубликованы в медиане через {f1(t['quote_after_gen_h_B']['median'])} ч после генерации; "
              f"A1 после генерации (UTC): {t['A1_quote_after_generation']} (все — условное время Football-Data 00:00).",
              f"- Backfill: решение после матча, цена = закрытие; за ≥ 24 ч до начала опубликовано "
              f"{pc(t['backfill_closing_published_before']['share_ge_24h'])} этих цен.",
              "",
              "## Дрейф и CLV",
              "",
              f"- Bet365 1X2 открытие→закрытие: |Δq| {f1(DR['fd']['all_seasons_b365|1X2']['mean_abs_dq_pp'], 2)} п.п.; "
              f"закрытие точнее (Δ log loss {DR['fd']['all_seasons_b365|1X2']['ll_close_minus_open']['diff'] * 1000:+.2f}×10⁻³).",
              "- Модель → движение Pinnacle (b₁ с контролем других книг): " + "; ".join(
                  f"{k}: {v['beta'][1]:.3f} [{v['ci95'][1][0]:.3f}; {v['ci95'][1][1]:.3f}]" for k, v in ctr.items()),
              f"- CLV Main A1: к той же цене {pc(CLV['main|A1']['clv_odds']['mean'], 2, True)} "
              f"{ci(CLV['main|A1']['clv_odds']['ci95'], 2)}, справедливый {pc(CLV['main|A1']['clv_fair']['mean'], 1, True)} "
              f"{ci(CLV['main|A1']['clv_fair']['ci95'])}; Risk A1: {pc(CLV['risk|A1']['clv_odds']['mean'], 2, True)} "
              f"{ci(CLV['risk|A1']['clv_odds']['ci95'], 2)}, {pc(CLV['risk|A1']['clv_fair']['mean'], 1, True)} "
              f"{ci(CLV['risk|A1']['clv_fair']['ci95'])}.",
              "",
              "## Решения",
              "",
              "- A: класс времени цены у каждой цены прогноза + время котировки в UTC; метки на сайте для A2/B/C.",
              "- A: официальный ROI записанной истории — только цены, существовавшие при генерации (A1+A2).",
              "- A: время котировки Football-Data = неизвестно (не 00:00 даты матча); моменты времени — в UTC.",
              "- B: backfill по цене номинального момента решения (нужны все снимки /historical-odds); ранняя публикация Main.",
              "- C: перевыбор Main по цене закрытия.",
              "",
              "## Воспроизведение",
              "",
              "```",
              "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
              "$PY -m experiments.v6_2.odds_timing.run && $PY -m experiments.v6_2.odds_timing.build_pdf",
              "```", ""]
    return "\n".join(lines)


def write_findings():
    (C.HERE / "findings.md").write_text(findings(), encoding="utf-8")


if __name__ == "__main__":
    run()
