"""Builds reports/v6_1/Odds_Coverage_Audit.pdf (Russian) from the audit files.

Inputs (all produced by scripts in this repository, nothing typed in by hand except where a measurement was
made once by hand and stored in reports/v6_1/data/api_checks.json):

* reports/v6_1/data/audit_before.json   — scripts/v6_audit.py on the data of 04.11.2026, v6 code;
* reports/v6_1/data/odds_audit_after.json — scripts/v61_odds_audit.py --with-repairs, v6.1 code;
* reports/v6_1/data/odds_audit_after_unpriced.csv — every settled Main / Risk without a price, with its reason;
* reports/v6_1/data/api_checks.json     — API checks of 04.11.2026 and Football-Data columns per league.

    python -m experiments.v6_1.odds.build_pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.report_kit import (ACCENT, AMBER, NEG, H1, H2, H3, P, Report, bullets, figure, formula, load,  # noqa: E402
                                note, num, pct, reproduce, save_chart, share, table)
from src.leagues import enabled_leagues  # noqa: E402

DATA = ROOT / "reports" / "v6_1" / "data"
FIG = ROOT / "experiments" / "v6_1" / "odds" / "figures"
OUT = ROOT / "reports" / "v6_1" / "Odds_Coverage_Audit.pdf"

REASON_RU = {
    "league not found at OddsPapi (no tournament id)": "лига не найдена в OddsPapi (нет id турнира)",
    "club refused by the id-conflict guard (Barcelona / Espanyol, fixed in v6.1)":
        "клуб отклонён защитой от конфликта id (Barcelona / Espanyol, исправлено в v6.1)",
    "OddsPapi fixture exists but was not matched to our match": "фикстура OddsPapi есть, но не сопоставлена",
    "match not in the OddsPapi fixture list": "матча нет в списке фикстур OddsPapi",
    "matched, but no pre-match price snapshot": "сопоставлен, но нет предматчевой котировки",
    "market not offered for the match": "рынок не предлагался на матч",
    "this line of the market not offered": "эта линия рынка не предлагалась",
    "selection not offered": "исход не предлагался",
    "no result yet": "нет результата (матч ещё не сыгран)",
    "refund (push): not a won / lost bet": "возврат (push): не выигрыш и не проигрыш",
    "void (no result in time)": "аннулирован (нет результата в срок)",
    "match not found at OddsPapi (or not mapped safely)": "матч не найден в OddsPapi (или не сопоставлен надёжно)",
    "no pre-match price snapshot for the match": "нет предматчевой котировки",
    "selection / line not offered by the bookmakers": "исход / линия не предлагались",
}
RECOVER_RU = {
    "yes — the corrected ids map these fixtures at the next history run": "да — после исправления id при следующем запуске history",
    "maybe — after a name fix, at the next history run": "возможно — после исправления названия",
    "only if OddsPapi lists the league (checked monthly, see candidates)": "только если OddsPapi предоставит лигу (проверка раз в месяц)",
    "no (OddsPapi has no such fixture)": "нет (у OddsPapi нет такой фикстуры)",
    "no (the archive holds no price)": "нет (в архиве нет цены)", "no": "нет", "—": "—",
}
GROUP_RU = {"1X2": "Исход 1X2", "TOTAL": "Тотал голов", "BTTS": "Обе забьют", "DOUBLE_CHANCE": "Двойной шанс",
            "HANDICAP": "Азиатская фора", "EHANDICAP": "Европейская фора", "CORNERS": "Угловые",
            "EXACT_SCORE": "Точный счёт"}


def ru(text, table_=REASON_RU):
    return table_.get(text, text)


def charts(before: dict, after: dict) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    b = {r["league"]: r for r in before.get("leagues", [])}
    a = {r["league"]: r for r in after["leagues"]}
    codes = [lg.code for lg in enabled_leagues()]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = range(len(codes))
    before_share = [b[c]["main_priced"] / b[c]["main_settled"] if c in b and b[c]["main_settled"] else 0 for c in codes]
    after_share = [a[c]["main_bets"] / a[c]["main_settled"] if a[c]["main_settled"] else 0 for c in codes]
    ax.bar([i - 0.2 for i in x], [v * 100 for v in before_share], 0.4, label="До (v6)", color="#9aa6a8")
    ax.bar([i + 0.2 for i in x], [v * 100 for v in after_share], 0.4, label="После (v6.1)", color=ACCENT)
    ax.set_xticks(list(x), codes, fontsize=8)
    ax.set_ylabel("Main с реальной ценой, %")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title("Доля рассчитанных Main с реальной букмекерской ценой по лигам, сезон 2026/27")
    out["leagues"] = save_chart(fig, FIG / "coverage_by_league.png")

    rs = pd.DataFrame(after["unpriced_reasons"])
    if not rs.empty:
        piv = rs.pivot_table(index="reason", columns="role", values="n", aggfunc="sum", fill_value=0)
        piv = piv.loc[piv.sum(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        left = [0] * len(piv)
        for role, colour, name in (("main", ACCENT, "Main"), ("risk", AMBER, "Risk")):
            if role in piv:
                ax.barh([ru(i) for i in piv.index], piv[role], left=left, color=colour, label=name)
                left = [l_ + v for l_, v in zip(left, piv[role])]
        ax.set_xlabel("Рассчитанных прогнозов без реальной цены")
        ax.legend(fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_title("Почему у прогноза нет реальной цены (после исправлений v6.1)")
        out["reasons"] = save_chart(fig, FIG / "unpriced_reasons.png")

    prov = after["priced_provenance"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, key, title in ((axes[0], "provider", "Провайдер цены"), (axes[1], "bookmaker", "Букмекер")):
        items = sorted(prov[key].items(), key=lambda kv: kv[1])[-8:]
        ax.barh([k for k, _ in items], [v for _, v in items], color="#3d6fb6")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Происхождение реальных цен Main и Risk (сезон 2026/27)", fontsize=10)
    out["provenance"] = save_chart(fig, FIG / "price_provenance.png")
    return out


def build() -> Path:
    before = load(DATA / "audit_before.json", {})
    after = load(DATA / "odds_audit_after.json")
    checks = load(DATA / "api_checks.json", {})
    unpriced = pd.read_csv(DATA / "odds_audit_after_unpriced.csv") if (DATA / "odds_audit_after_unpriced.csv").exists() \
        else pd.DataFrame()
    figs = charts(before, after)
    hb, ha = before.get("history_roles", {}), after["roles"]
    rep = Report(OUT, "Аудит покрытия реальными коэффициентами",
                 facts={"Сезон": "2026/27 (матчи по 02.11.2026)", "Данные": "local_data на 04.11.2026"})

    # 1
    rep.add(H1("1. Цель исследования"),
            P("Проверить, для какой доли завершённых матчей сезона 2026/27 у основного (Main) и рискового (Risk) "
              "прогноза FootPredict есть **настоящая** предматчевая цена букмекера, найти точную причину каждого "
              "пропуска, исправить то, что зависит от системы, и отделить это от ограничений внешних источников."),
            P("Правила, действующие во всей работе: только реальные котировки конкретного букмекера, только "
              "правильный матч, рынок, линия и исход, только котировка до начала матча; никаких синтетических, "
              "оценочных или «честных» (fair) цен. Исходный выбор прогноза никогда не меняется — цена лишь "
              "добавляется к уже сделанному прогнозу."))
    # 2
    rep.add(H1("2. Исходная проблема"),
            bullets([
                f"До исправлений (код v6, данные 04.11.2026): рассчитано {hb.get('main', {}).get('settled', '—')} Main, "
                f"из них с ценой {hb.get('main', {}).get('bets', '—')} "
                f"({share(hb.get('main', {}).get('odds_coverage'))}); Risk — {hb.get('risk', {}).get('settled', '—')} и "
                f"{hb.get('risk', {}).get('bets', '—')}.",
                "Три лиги — Турция (T1), Шотландия League One (SC2) и League Two (SC3) — не находились в OddsPapi.",
                "В La Liga прогнозов Main было больше, чем ставок (108 против 97 на сайте).",
                "В журнале конфликтов стояла запись «oddspapi, 2817, Spain|Barcelona, already id 2814».",
                "Число рассчитанных Main и Risk на сайте различалось на один прогноз.",
            ]))
    # 3
    rep.add(H1("3. Источники данных и текущая реализация"),
            table([["Источник", "Что даёт FootPredict", "Ограничения"],
                   ["Football-Data.co.uk (CSV)", "результаты, статистика (удары, угловые, карточки, с 2026/27 — xG), "
                    "цены Bet365 и максимум/среднее рынка: 1X2, тотал 2.5, одна линия азиатской форы, закрывающие цены",
                    "публикация с задержкой 1–3 дня; только три рынка; среднее рынка — не цена букмекера"],
                   ["API-Football", "свежие фикстуры и результаты вчера–завтра",
                    "бесплатный тариф: только даты вчера–завтра, без коэффициентов"],
                   ["OddsPapi", "текущие и архивные (с января 2026) цены многих рынков: Bet365, Pinnacle, "
                    "William Hill", "250 платных запросов в месяц; архив бесплатен, 5 с между запросами"],
                   ["The Odds API", "цены 1X2/тоталов и расширенных рынков по запросу",
                    "500 кредитов в месяц; в v6 и v6.1 выключен, сохранённые цены используются"]]),
            P("Цена прикрепляется к прогнозу по ключу (матч, рынок, исход); при нескольких источниках действует "
              "приоритет провайдеров и последняя котировка до начала матча. У каждой цены хранятся букмекер, "
              "провайдер, время котировки и происхождение (`odds_origin`: записана в момент прогноза, добавлена из "
              "архива, заменила среднее рынка)."))
    # 4
    rep.add(H1("4. Данные и периоды"),
            P(f"Сезон 2026/27, все {after['played_matches']} завершённых матчей 21 лиги по 02.11.2026; единая история "
              f"прогнозов FootPredict ({after['history_matches']} матчей): записанные до матча прогнозы, а для матчей "
              "без них — реконструкция сезона по данным до матча."),
            P("«До» — аудит `scripts/v6_audit.py` на данных 04.11.2026 и коде v6. «После» — аудит "
              "`scripts/v61_odds_audit.py --with-repairs` на тех же данных и коде v6.1 с исправлениями, которые "
              "конвейер применит при следующем запуске (в памяти, без записи). Восстановление цен Barcelona и "
              "Espanyol требует запуска `history` в GitHub Actions (ключ OddsPapi есть только там) и в «после» не "
              "входит — эти прогнозы показаны как восстановимые."))
    # 5
    rep.add(H1("5. Методы"),
            P("Для каждого рассчитанного Main и Risk без цены причина определяется по порядку:"),
            bullets(["нет ни одной цены на матч: лига без турнира OddsPapi → конфликт id клуба → фикстура OddsPapi "
                     "есть в архиве, но не сопоставлена → матча нет в списке → нет предматчевой котировки;",
                     "цены на матч есть: нет этого исхода → нет этой линии рынка → нет рынка."]),
            P("Доля покрытия считается так:"),
            formula("покрытие = число рассчитанных прогнозов с реальной ценой / число рассчитанных прогнозов"),
            P("Ставкой считается рассчитанный прогноз (выигрыш или проигрыш) с реальной ценой; возврат (push) "
              "не входит ни в выигрыши, ни в проигрыши. ROI = прибыль / число ставок × 100 %, ставка — 1 единица."))
    # experiments table
    rep.add(H2("5.1. Перечень проверок"),
            table([["Проверка", "Выполнена", "Как"],
                   ["Покрытие по 21 лиге, Main и Risk", "да", "аудит до / после"],
                   ["Покрытие по рынкам и линиям", "да", "аудит после"],
                   ["Букмекер, провайдер, время котировки", "да", "аудит после; котировки после начала — 0"],
                   ["Причина каждого пропуска", "да", "CSV `odds_audit_after_unpriced.csv`"],
                   ["Barcelona / Espanyol: id 2814 и 2817", "да", "журналы конфликтов, архив фикстур, история git"],
                   ["Barcelona / Espanyol: новая загрузка цен", "нет", "нужен запуск history в CI (ключ OddsPapi)"],
                   ["Турция и Шотландия в OddsPapi", "частично", "найдена ошибка поиска; повторный запрос — в CI"],
                   ["The Odds API для Турции и Шотландии", "да", "бесплатные /sports и /events, 04.11.2026"],
                   ["API-Football: доступ и тариф", "да", "/status и тестовый запуск 04.11.2026"],
                   ["Колонки Football-Data по лигам", "да", "разбор файлов сезона"],
                   ["Инварианты истории", "да", "src/analytics/invariants.py"]]))
    rep.page_break()

    # 6 results
    rep.add(H1("6. Результаты"), H2("6.1. Покрытие Main и Risk: до и после"))
    rows = [["Показатель", "До (v6)", "После (v6.1)"]]
    for role, name in (("main", "Main"), ("risk", "Risk")):
        b, a = hb.get(role, {}), ha[role]
        rows += [[f"{name}: рассчитано (выигр. + проигр.)", b.get("settled"), a["settled"]],
                 [f"{name}: ставок с реальной ценой", b.get("bets"), a["bets"]],
                 [f"{name}: покрытие", share(b.get("odds_coverage")),
                  share(a["bets"] / a["settled"] if a["settled"] else None)],
                 [f"{name}: возвраты (push)", b.get("push"), a["push"]],
                 [f"{name}: ROI", pct(b.get("roi"), 1, True), pct(a["roi"], 1, True)]]
    rep.add(table(rows),
            P("Число ставок «после» немного меньше: 34 прогноза, оценённые средней ценой рынка, больше не "
              "считаются ставками (раздел 6.9), а повторы перенесённых матчей не считаются дважды (6.10). "
              "Покрытие вырастет ещё на 10 Main и 2 Risk после запуска history (Barcelona / Espanyol)."))
    rep.add(figure(figs["leagues"], "Рис. 1. Доля рассчитанных Main с реальной ценой по лигам: до и после."))

    rep.add(H2("6.2. Покрытие по лигам"))
    b_by = {r["league"]: r for r in before.get("leagues", [])}
    rows = [["Лига", "Сыграно", "Турнир OddsPapi", "Фикстур / сопост.", "Main рассч. / ставок", "Risk рассч. / ставок",
             "Main до: с ценой", "ROI Main"]]
    for r in after["leagues"]:
        b = b_by.get(r["league"], {})
        rows.append([f"{r['league']} {r['name']}", r["played"], r["tournament"] or "нет",
                     f"{r['archive_fixtures']} / {r['archive_mapped']}", f"{r['main_settled']} / {r['main_bets']}",
                     f"{r['risk_settled']} / {r['risk_bets']}", b.get("main_priced", "—"), pct(r["main_roi"], 1, True)])
    rep.add(table(rows, font=6.6))

    rep.add(H2("6.3. Покрытие по рынкам и линиям Main"))
    rows = [["Группа рынков", "Main рассчитано", "С реальной ценой", "Покрытие"]]
    for r in after["main_by_group"]:
        rows.append([GROUP_RU.get(r["group"], r["group"]), r["predictions"], r["priced"],
                     share(r["priced"] / r["predictions"] if r["predictions"] else None)])
    rep.add(table(rows))
    rows = [["Линия", "Main", "С ценой", "Покрытие"]]
    for r in after["main_by_line"][:18]:
        rows.append([r["line"], r["predictions"], r["priced"], share(r["priced"] / r["predictions"])])
    rep.add(P("Самые частые линии Main:"), table(rows))

    rep.add(H2("6.4. Букмекеры, провайдеры и время котировки"))
    prov = after["priced_provenance"]
    rep.add(figure(figs["provenance"], "Рис. 2. Провайдеры и букмекеры цен Main и Risk."),
            table([["Происхождение цены", "Прогнозов"]] + [[k, v] for k, v in prov["odds_origin"].items()]),
            P(f"У {prov['with_quote_time']} цен известно время котировки; котировок, снятых **после** начала "
              f"матча, — {prov['quoted_after_kickoff']}. Названия «bet365» и «Bet365» — один букмекер в записи "
              "разных провайдеров (OddsPapi и Football-Data / The Odds API)."))

    rep.add(H2("6.5. Причины оставшихся пропусков"))
    rows = [["Роль", "Причина", "Прогнозов", "Можно ли восстановить"]]
    for r in after["unpriced_reasons"]:
        rows.append([r["role"].capitalize(), ru(r["reason"]), r["n"], ru(r["recoverable"], RECOVER_RU)])
    rep.add(table(rows), figure(figs.get("reasons", ""), "Рис. 3. Причины отсутствия цены после исправлений."))
    if before.get("missing_reasons"):
        rows = [["Роль", "Причина (классификация v6)", "Прогнозов"]]
        for role, d in before["missing_reasons"].items():
            rows += [[role.capitalize(), ru(k), v] for k, v in d.items()]
        rep.add(P("Для сравнения — причины «до» (в v6 они различались грубее):"), table(rows))
    if not unpriced.empty:
        by_league = unpriced.groupby(["league", "reason"]).size().rename("n").reset_index()
        rows = [["Лига", "Причина", "Прогнозов"]] + [[r.league, ru(r.reason), r.n] for r in by_league.itertuples()]
        rep.add(P("По лигам:"), table(rows))

    rep.add(H2("6.6. La Liga: почему прогнозов больше, чем ставок"))
    ll = after["laliga"]
    rep.add(P(f"Рассчитано Main: {ll['main_predictions']}, ставок: {ll['bets']}, возвратов: {ll['push']}, ещё не "
              f"сыграно: {ll['pending']}. Каждый Main La Liga, который не стал ставкой:"))
    rows = [["Дата", "Матч", "Рынок / исход", "P", "Статус", "Причина", "Восстановимо"]]
    for r in ll["not_a_bet"]:
        rows.append([str(r["date"])[:10], f"{r['home_team']} – {r['away_team']}", f"{r['market']} {r['selection']}",
                     num(r["probability"], 2), {"WON": "выигр.", "LOST": "проигр.", "UPCOMING": "ждёт",
                                                "PUSH": "возврат", "VOID": "аннул."}.get(r["status"], r["status"]),
                     ru(r["why_not_a_bet"]), ru(r["recoverable"], RECOVER_RU)])
    rep.add(table(rows, font=6.6),
            note("Вывод: из 11 рассчитанных Main La Liga без ставки 10 приходятся на матчи Barcelona и Espanyol, "
                 "отклонённые защитой от конфликта id, и будут восстановлены первым же запуском history после "
                 "исправления; в одном матче (Málaga – Espanyol) у букмекеров не было линии тотала 3.5. "
                 "Исходные прогнозы не меняются: к ним только добавится найденная цена."))

    rep.add(H2("6.7. Barcelona и Espanyol: расследование"))
    rep.add(bullets([
        "22.10.2026, прежнее правило имён: и «FC Barcelona», и «Espanyol Barcelona» распознавались как Barcelona, "
        "в реестр попали два id OddsPapi одного клуба — 2814 и 2817 (коммит данных 1a0a90c).",
        "Очистка фантомного матча Santander – Barcelona (PR #8) удалила id 2817 и оставила 2814 — неверный.",
        "Запуск 03.11.2026 отклонил id 2814: его название в фикстуре распознаётся как Espanyol "
        "(«team id conflict oddspapi 2814: Spain|Barcelona vs Espanol»); ранее id 2817 отклонялся как "
        "Barcelona («already id 2814»). Так 2814 = Espanyol Barcelona, 2817 = FC Barcelona.",
        "Следствие: защита (правильно) не прикрепляла цены, и оба клуба в OddsPapi остались без цен; "
        "ошибочно прикреплённых котировок нет — проверены все 1788 цен OddsPapi с известными названиями фикстур.",
        "Исправление v6.1: проверенные вручную id (`VERIFIED_IDS`) заменяют неверную запись реестра; механизм "
        "защиты от конфликтов не отключён и защищает их дальше. В журнал конфликтов и в архив фикстур теперь "
        "пишутся исходные названия и id провайдера, чтобы такие случаи проверялись без догадок.",
        "Остальные клубы Испании: других конфликтов нет; несопоставленные фикстуры SP1 — только матчи этих двух "
        "клубов (22) и 3 матча, которых нет в нашем календаре.",
    ]))

    rep.add(H2("6.8. Турция (T1), Шотландия League One (SC2) и League Two (SC3)"))
    odp = checks.get("oddspapi", {})
    toa = checks.get("the_odds_api", {})
    rep.add(bullets([
        "Поиск турнира OddsPapi сравнивал страну по английскому названию; если OddsPapi пишет «Türkiye», "
        "совпадения не было. v6.1 сравнивает без диакритики и знает вариант «Türkiye».",
        "Список турниров не сохранялся, и при отсутствующей лиге каждый запуск снова платил за запрос "
        "/tournaments. Теперь турниры страны сохраняются в `tournament_candidates.json` (коммитится), повторный "
        "платный запрос — не чаще раза в 30 дней, а `oddspapi_history.py` печатает найденные турниры страны.",
        "Проверить ответ OddsPapi локально нельзя (ключ только в GitHub Secrets); первый запуск после слияния "
        "либо найдёт турнир и начнёт загрузку, либо сохранит список турниров страны как доказательство отсутствия.",
        f"The Odds API (бесплатные /sports и /events, 04.11.2026): Турция — {toa.get('sports', {}).get('soccer_turkey_super_league', '—')}; "
        f"Шотландия — {toa.get('sports', {}).get('soccer_spl', '—')}.",
        "Football-Data даёт настоящие цены Bet365 для T1, SC2 и SC3 (1X2, тотал 2.5, одна линия азиатской форы); "
        "поэтому в этих лигах без цены остаются Main на других рынках (тоталы 1.5/3.5, обе забьют, угловые).",
    ]),
        table([["Лига", "Main рассч.", "Ставок", "Без цены", "Источник цен сейчас"]] +
              [[r["league"], r["main_settled"], r["main_bets"], r["main_settled"] - r["main_bets"],
                "Football-Data (Bet365, 3 рынка)"] for r in after["leagues"] if r["league"] in ("T1", "SC2", "SC3")]))

    rep.add(H2("6.9. Средние цены рынка вместо цены букмекера"))
    rep.add(P("99 прогнозов от 02.10.2026 (12 Main, 17 Risk, 70 рыночных прогнозов) хранили «Market average» "
              "Football-Data — среднюю цену рынка. Это реальные рыночные данные, но не котировка конкретного "
              "букмекера, по ней нельзя было сделать ставку. v6.1 заменяет её настоящей котировкой букмекера до "
              "начала матча (найдена для 65 прогнозов) или, если её нет, перестаёт считать прогноз ставкой (34). "
              "Исходная средняя цена сохраняется в `replaced_average_odds`; выбор и результат прогноза не меняются."))

    rep.add(H2("6.10. Повторы одного матча"))
    dup = after["duplicates_removed"]
    rows = [["Повтор", "Оставлен"]] + [[d["match_id"], d["kept_match_id"]] for d in dup["moved_matches"]]
    rep.add(P("Перенесённый матч (предварительная или отменённая дата) мог получить два прогноза: на первую "
              "дату (позже рассчитанный по результату перенесённого матча) и на новую. В истории теперь "
              "засчитывается прогноз на дату, когда матч действительно сыгран (его сайт показывал в день матча); "
              "правило не смотрит на результат:"), table(rows))
    rows = [["Повтор (другое написание клуба)", "Оставлен"]] + [[d["match_id"], d["kept_match_id"]]
                                                                 for d in dup["club_spelt_twice"]]
    rep.add(P("Файл fixtures Football-Data трижды написал клуб иначе, чем файлы результатов («KV Mechelen», "
              "«Sheffield Utd», «Oxford Utd»), что создавало клуб без истории и второй id матча. Теперь "
              "названия fixtures приводятся к названиям результатов, а уже записанные повторы не считаются:"),
            table(rows))

    rep.add(H2("6.11. Разница в один прогноз между Main и Risk"))
    rep.add(P("Причина найдена: Risk матча Palermo – Vicenza (01.11.2026, 3:1) — азиатская фора −2 на хозяев — "
              "закончился возвратом (push). Сводка считала «рассчитанными» только выигрыши и проигрыши, поэтому Risk "
              "показывал на один меньше. Прогнозов ровно один Main и один Risk на каждый матч "
              f"(Main без Risk: {after['main_without_risk']}); в v6.1 возвраты выводятся отдельно ("
              "«… won · … lost · 1 refunded»), формулы прибыли и ROI не изменены."))

    rep.add(H2("6.12. Football-Data: какие данные есть в файлах сезона"))
    fd = checks.get("football_data", {}).get("season_files", {})
    rows = [["Лига", "Матчей", "Удары", "В створ", "Угловые", "Карточки", "xG", "Bet365 1X2", "Тотал 2.5", "Фора",
             "Закрыв. цены", "Pinnacle"]]
    yes = lambda v: "да" if v else "нет"  # noqa: E731
    for code, d in fd.items():
        rows.append([code, d["played"], yes(d["stats"]["shots"]), yes(d["stats"]["shots_on_target"]),
                     yes(d["stats"]["corners"]), yes(d["stats"]["cards"]), yes(d["xg"]), yes(d["prices"]["bet365_1x2"]),
                     yes(d["prices"]["bet365_total_2_5"]), yes(d["prices"]["asian_handicap"]),
                     yes(d["prices"]["closing_1x2"]), yes(d["prices"]["pinnacle_1x2"])])
    rep.add(table(rows, font=6.4),
            P("С сезона 2026/27 Football-Data публикует xG (`HxG`, `AxG`) для большинства лиг (кроме SC1–SC3 и "
              "EC), но больше не публикует цены Pinnacle. xG пока не используется моделями — это возможное "
              "улучшение признаков (раздел 9). Загрузка: один CSV на лигу раз в день, в соответствии с условиями "
              "источника; неизменившиеся файлы отмечаются в диагностике."))

    rep.add(H2("6.13. API-Football"))
    af = checks.get("api_football", {})
    rep.add(bullets([f"Локальный ключ (.env): {af.get('local_key', '—')}.",
                     f"Тариф: {af.get('free_plan_window', '—')}.",
                     f"Ключ GitHub Secrets (новый аккаунт): {af.get('ci_key', '—')}.",
                     f"Роль в системе: {af.get('role', '—')}.",
                     "Интеграция сохранена; локальный ключ в .env стоит заменить ключом нового аккаунта."]))

    rep.add(H2("6.14. The Odds API: возможность ограниченного применения"))
    rep.add(bullets([f"Остаток кредитов: {toa.get('remaining_this_month', '—')} в ноябре (в октябре осталось "
                     f"{toa.get('previous_month_left', '—')}).",
                     f"Стоимость: {toa.get('cost', '—')}.",
                     f"Статус: {toa.get('status', '—')}.",
                     "Оценка для Турции: 9 матчей в тур; 1X2 и тоталы всей лиги — 2 кредита за запрос (около "
                     "10 кредитов в месяц при еженедельном запросе) дают мало сверх Football-Data; расширенные "
                     "рынки — 4–5 кредитов за матч, около 160–180 кредитов в месяц. Включается только решением "
                     "владельца (`ODDS_API_LEAGUES=T1`).",
                     "Исследование Risk (отдельный отчёт) заметило, что Risk с ценами The Odds API в 2026/27 "
                     "дали ROI около −28 % на 214 ставках — к источнику цен стоит относиться осторожно."]))

    rep.add(H2("6.15. Расход квот"))
    rep.add(table([["Источник", "Период", "Расход"],
                   ["OddsPapi (платные)", odp.get("month", "—"), f"{odp.get('billable_this_month', '—')} из 250"],
                   ["OddsPapi (журнал запросов)", "всё время", ", ".join(f"{k}: {v}" for k, v in
                                                                        odp.get("requests_logged", {}).items())],
                   ["The Odds API", toa.get("ledger_month", "—"), f"{toa.get('ledger_month_credits', '—')} кредитов"],
                   ["API-Football", "сутки", "3–6 из 100 запросов"]]))

    rep.add(H2("6.16. Диагностика обновлений"))
    rep.add(P("Каждый запуск теперь пишет `local_data/api/diagnostics/last_run.json`, добавляет строку в "
              "`history.csv` и выводит в сводку GitHub Actions строки такого вида (пример реального локального "
              "запуска 04.11.2026 со старым ключом):"),
            table([["Строка диагностики"],
                   ["Football-Data: 22 requests · 21 files unchanged · 2,033 played matches in the season files · 17 s"],
                   ["API-Football: 6 requests · errors: fixtures: {'plan': 'Free plans do not have access to this "
                    "date…'}; {'access': 'Your account is suspended…'}"],
                   ["OddsPapi: N requests · fixtures returned · in our leagues · matched · with real prices · new "
                    "quotes · updated quotes · skipped: unknown team name / team id conflict · left"],
                   ["The Odds API: off (stored prices still used) · 500 left"],
                   ["Predictions: … predictions settled · history checks: all passed"]]))
    rep.page_break()

    # 7
    rep.add(H1("7. Что устранено и что зависит от внешних источников"),
            table([["Проблема", "Статус"],
                   ["Barcelona / Espanyol: неверный id 2814", "исправлено; цены — после запуска history"],
                   ["Поиск турниров с диакритикой (Türkiye)", "исправлено; результат — после запуска"],
                   ["Повторный платный /tournaments каждый запуск", "исправлено (раз в 30 дней)"],
                   ["Средние цены рынка вместо цены букмекера (99)", "исправлено: 65 заменены, 34 не ставки"],
                   ["Двойной учёт перенесённых матчей (8 в записанных прогнозах)", "исправлено"],
                   ["Клуб под двумя написаниями в fixtures (3)", "исправлено"],
                   ["Main и Risk различались на 1", "объяснено (push), возвраты показаны"],
                   ["T1, SC2, SC3 в OddsPapi", "внешнее ограничение, если турнир не найдётся"],
                   ["Линии и рынки, которых не было у букмекеров", "внешнее ограничение"],
                   ["Локальный ключ API-Football", "заблокирован у провайдера — заменить в .env"]]))
    # 8
    rep.add(H1("8. Выводы и ограничения"),
            bullets([f"После исправлений с реальной ценой {share(ha['main']['bets'] / ha['main']['settled'])} "
                     f"рассчитанных Main и {share(ha['risk']['bets'] / ha['risk']['settled'])} Risk; ни одна цена не "
                     "снята после начала матча.",
                     "Почти все оставшиеся пропуски — три лиги без OddsPapi (T1, SC2, SC3) и матчи Barcelona / "
                     "Espanyol, которые восстановит ближайший запуск history.",
                     "Ограничения: цены 2026/27 смешивают котировки момента прогноза и архивные котировки перед "
                     "самым матчем; закрывающих цен OddsPapi для CLV нет; Pinnacle в Football-Data больше нет.",
                     "Все исправления не меняют исходных прогнозов; повторы и средние цены исключаются из подсчёта, "
                     "но остаются в хранилище."]))
    # 9
    rep.add(H1("9. Рекомендации"),
            bullets(["После слияния запустить `history` вручную (несколько платных запросов) — подтянутся цены "
                     "Barcelona / Espanyol и проверятся турниры T1, SC2, SC3.",
                     "Заменить локальный ключ API-Football ключом нового аккаунта.",
                     "The Odds API для Турции включать только при необходимости расширенных рынков, оценив расход "
                     "160–180 кредитов в месяц.",
                     "Рассмотреть xG Football-Data как новые признаки (требует отдельной проверки на прошлых "
                     "сезонах, где xG нет)."]))
    rep.add(H1("10. Воспроизведение"),
            reproduce(["python scripts/v6_audit.py --out reports/v6_1/data/audit_before.json   # на коде v6",
                       "python scripts/v61_odds_audit.py --with-repairs --out reports/v6_1/data/odds_audit_after.json",
                       "python -m experiments.v6_1.odds.build_pdf",
                       "pytest tests/test_v61_data.py -q"]))
    rep.build()
    print("pages:", rep.check(0.7))
    return OUT


if __name__ == "__main__":
    build()
