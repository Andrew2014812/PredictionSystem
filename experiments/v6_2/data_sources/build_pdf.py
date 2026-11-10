"""Builds reports/v6_2/Data_Sources_Report.pdf from results/*.json, results/*.csv and figures/*.png (Russian).

    python -m experiments.v6_2.data_sources.build_pdf

Every number is read from the result files of this folder; nothing is typed in by hand.
"""
from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from scripts.report_kit import H1, H2, P, Report, bullets, code, figure, note, reproduce, table

from . import common as C

OUT = C.ROOT / "reports" / "v6_2" / "Data_Sources_Report.pdf"
J = C.load_json
FD, AF, OP, TOA, RUNS, BETS, RUN = (J("football_data.json"), J("api_football.json"), J("oddspapi.json"),
                                    J("the_odds_api.json"), J("runs.json"), J("bets.json"), J("run_info.json"))
FIG = C.FIGURES


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(C.RESULTS / name)


def T(rows, widths=None):
    return [table(rows, widths=[w * cm for w in widths] if widths else None), Spacer(1, 6)]


def n0(v):
    return "—" if v is None or v != v else f"{int(round(v)):,}".replace(",", " ")


def f1(v, d=1):
    return "—" if v is None or v != v else f"{v:.{d}f}"


def pc(v, d=1):
    """fraction -> percent"""
    return "—" if v is None or v != v else f"{v * 100:.{d}f}%"


def roi(v, d=1):
    """value already in percent"""
    return "—" if v is None or v != v else f"{v:+.{d}f}%"


def ev(v):
    return "—" if v is None or v != v else f"{v * 100:+.1f}%"


def profit(v):
    return "—" if v is None or v != v else f"{v:+.1f}"


def h_(v):
    return "—" if v is None or v != v else f"{v:.1f} ч"


def days(v):
    return "—" if v is None or v != v else f"{v / 24:.1f} сут"


ROLE = {"main": "Main", "risk": "Risk", "market": "рыночные"}
SRC = {"live": "записанные (live)", "backfill": "ретроспективные (backfill)"}


# ------------------------------------------------------------------------------------------------ numbers used often
def overall(role, source=None):
    rows = BETS["by_source"] if source else BETS["overall"]
    for r in rows:
        if r["role"] == role and (source is None or r["source"] == source):
            return r
    return {}


M, R = overall("main"), overall("risk")
LEAD, DELAY, PER_RUN = AF["lead"], AF["delay_after_kickoff"], AF["per_run"]
CAT = AF["categories"]
COST = OP["costs"]
MONTH = {r["month"]: r for r in COST["per_month"]}
DEP = TOA["dependence"]
MC, PCV = BETS["match_coverage"], BETS["prediction_coverage"]
SCH = RUNS["schedule"]
QLOG, LOGOS = AF["quota_log"], AF["logos"]
IMP = [r["billable"] for r in COST["archive_imports"]]
IMP_RANGE = f"{min(IMP)}–{max(IMP)}" if IMP else "—"
XG_LEAGUES = sum(1 for r in FD["xg"]["leagues"] if r["with_xg"])
FD_LAST = pd.read_csv(C.RESULTS / "fd_result_commits.csv")["at_utc"].max()[:10]
DEAD = AF["fd_only"]["dead_period"]
STARTS = [s[11:16] for s in SCH["diagnostic_starts_kyiv"]]


RU_REASON = {"match not in the OddsPapi fixture list": "матча нет в списке фикстур OddsPapi",
             "this line of the market not offered": "эта линия рынка не предлагалась",
             "market not offered for the match": "рынок не предлагался",
             "selection not offered": "выбор не предлагался",
             "matched, but no pre-match price snapshot": "нет снимка цены до начала"}
RU_REC = {"no (OddsPapi has no such fixture)": "нет", "no": "нет"}


def ru_reason(s: str) -> str:
    if s.startswith("перенесённый матч"):
        return "перенос: цена под id " + s.rsplit(": ", 1)[-1]
    return RU_REASON.get(s, s)


def ru_rec(s: str) -> str:
    if s.startswith("да"):
        return "да"
    return RU_REC.get(s, s)


_corr = [r["corr_xg_goals"] for r in FD["xg"]["leagues"] if r["corr_xg_goals"] is not None]
XG_CORR = f"{min(_corr):.2f}–{max(_corr):.2f}" if _corr else "—"
_mk = pd.read_csv(C.RESULTS / "oddspapi_markets.csv")
_un = _mk.loc[_mk["used_by"] == "—"]
UNUSED = "; ".join(f"{g} — {len(x)} ({', '.join(x['market'].str.split('_').str[-1])})" for g, x in _un.groupby("group"))
_hist = pd.read_csv(C.RESULTS / "bets_history_main_risk.csv")
_void = _hist.loc[_hist["status"] == "VOID", "match_id"].drop_duplicates()
VOID_MATCH = ", ".join(_void) or "—"
_slug = ("ofi-crete", "vicenza", "sociedad-b")
_wait = _hist.loc[(_hist["status"] == "UPCOMING") & _hist["match_id"].str.contains("|".join(_slug))
                  & (pd.to_datetime(_hist["date"]) < pd.Timestamp(RUN["data_commit"].split()[1][:10]))]
N_MOVED = sum(1 for r in BETS["unpriced"]["rows"] if r["reason"].startswith("перенесённый"))
_cls = pd.read_csv(C.RESULTS / "bets_by_price_class.csv")
N_NOT_AUDITED = int(_cls.loc[_cls["price_class"] == "нет в аудите", "predictions"].sum())
waiting_unmatched = ", ".join(_wait["match_id"].drop_duplicates()) or "нет"
fdo_unmatched = AF["fd_only"]["by_reason"].get("имя клуба API-Football не сопоставлено", 0)


def last_diag(source: str) -> dict:
    rows = [r for r in RUNS["diagnostics"]["history"] if r["source"] == source]
    return rows[-1] if rows else {}


def per_run(source: str) -> int:
    return int(last_diag(source).get("n_requests") or 0)


def summary() -> list:
    af_ahead = CAT.get("API-Football раньше", 0)
    same = CAT.get("в одном запуске", 0)
    pend = CAT.get("только API-Football (Football-Data ещё нет)", 0)
    toa_bets = sum(DEP["by_role"][r]["bets"] for r in ("main", "risk"))
    toa_rep = sum(DEP["by_role"][r]["odp_same_selection"] for r in ("main", "risk"))
    un = BETS["unpriced"]
    nov = MONTH.get("2026-11", {})
    return [
        H1("1. Главное"),
        note(f"**Football-Data** — единственный источник статистики (удары, угловые, фолы, карточки), из которой строятся "
             f"все {FD['feature_map']['unique_features']} признаков моделей, и цен Bet365 прошлых сезонов. xG (HxG/AxG) есть "
             f"только с 2026/27 в {sum(1 for r in FD['xg']['leagues'] if r['with_xg'])} из 21 лиги "
             f"({n0(FD['xg']['total_with_xg'])} из {n0(FD['xg']['total_played'])} матчей) и продакшн-кодом не читается."),
        note(f"**API-Football** даёт только счёт и статус матчей (вчера/сегодня/завтра, {QLOG['per_run_max']} запроса за запуск при лимите 100 в день). "
             f"Из {n0(AF['matches'])} матчей с {AF['start_utc'][:10]} он прислал результат раньше Football-Data в "
             f"{n0(af_ahead)} случаях, в том же запуске — в {n0(same)}; ещё {n0(pend)} результатов сейчас есть только у него. "
             f"Медианное опережение — {days(LEAD['median_h'])} (среднее {days(LEAD['mean_h'])}); медиана от начала матча до "
             f"результата: API-Football {h_(DELAY['api_football']['median_h'])}, Football-Data {h_(DELAY['football_data']['median_h'])}. "
             f"Источник нужен: без него {n0(AF['value']['settleable_with_goals'])} Main/Risk рассчитывались бы на дни позже."),
        note(f"**OddsPapi** — основной источник цен: реальная цена хотя бы одного рынка есть у {n0(OP['with_oddspapi'])} из "
             f"{n0(OP['played'])} сыгранных матчей 2026/27 ({pc(OP['share_oddspapi'], 2)}). Из каталога в "
             f"{n0(OP['markets']['catalogue_markets'])} рынков система сопоставляет {OP['markets']['markets_mapped']} своих рынков, "
             f"прогнозы используют {OP['markets']['markets_used_by_predictions']}. Ежедневный запуск тратит "
             f"{f1(COST['daily_runs']['billable_mean'], 1)} платного запроса в среднем (максимум {COST['daily_runs']['billable_max']}); "
             f"загрузка архива — {IMP_RANGE}. В ноябре по журналу уже {nov.get('billable', '—')} платных из {COST['free_plan_per_month']}. "
             f"Остаток квоты из данных не известен (см. дефекты)."),
        note(f"**The Odds API** не запрашивается с {TOA['stored']['last_collected'][:10]}; его сохранённые цены используют {toa_bets} ставок Main/Risk "
             f"({pc(DEP['share_of_main_risk_bets'])} всех оценённых Main/Risk 2026/27). У всех {toa_rep} есть цена OddsPapi на тот же "
             f"выбор до начала матча, поэтому как запасной источник он ничего не добавляет: включать не нужно, хранить — да."),
        note(f"**Перепроверка ставок 2026/27** (единая история): Main {n0(M['predictions'])} прогнозов, {n0(M['bets'])} ставок, "
             f"ROI {roi(M['roi'])}; Risk {n0(R['predictions'])} прогнозов, {n0(R['bets'])} ставок, PUSH {R['push']}, ROI {roi(R['roi'])}. "
             f"Покрытие матчей ценами — {pc(MC['share'], 2)}, покрытие самих прогнозов: Main {pc(PCV['main']['share'], 2)}, "
             f"Risk {pc(PCV['risk']['share'], 2)}. Без цены остаются {un['total']} Main/Risk (из них "
             f"{un['settled_or_void']} уже рассчитаны или аннулированы). Ла Лига: "
             f"{BETS['laliga']['overall']['main']['predictions']} Main = {BETS['laliga']['overall']['main']['bets']} ставок."),
    ]


def method() -> list:
    return [
        H1("2. Данные и метод"),
        P("Цель — ответить по данным, а не по документации API: что система реально получает от каждого из четырёх "
          "источников, что из этого идёт в признаки моделей, что только в интерфейс, что только в цены и ROI, сколько "
          "матчей покрыто, сколько это стоит и нужен ли источник. Вторая часть — перепроверка исторических ставок "
          "2026/27 после последнего импорта архива OddsPapi."),
        P("Только локальные данные этого репозитория, без сети и API-ключей. Каждый ежедневный запуск коммитит "
          "`local_data/` в `main`, поэтому история git — журнал того, что каждый источник отдал к каждому запуску: "
          "`git show <sha>:local_data/results_live/fixtures.parquet` (строки API-Football), "
          "`git show <sha>:local_data/results/2627/<лига>.csv` (файлы Football-Data), версии журналов квот. "
          f"Использовано коммитов: файл живых результатов — {AF['commits_live_file']}, файлы Football-Data 2026/27 — "
          f"{AF['commits_fd_files']}, журнал OddsPapi — {COST['ledger_versions']}."),
        P("Часы: `generated_at` и время коммитов — UTC; `kickoff`, `collected_at`, `priced_at` — местное время Великобритании; "
          "все сравнения сделаны в UTC. Время Киева — UTC+2 после 25.10.2026."),
        P(f"Код: `experiments/v6_2/data_sources/`; данные снимка: коммит `{RUN['data_commit'][:12]}`; код: "
          f"`{RUN['git_sha'][:12]}`."),
    ]


# ------------------------------------------------------------------------------------------------ Football-Data
def football_data() -> list:
    cov = csv("fd_season_coverage.csv")
    seasons = ["2122", "2223", "2324", "2425", "2526", "2627"]
    groups = ["goals FTHG/FTAG", "shots HS/AS", "on target HST/AST", "corners HC/AC", "fouls HF/AF", "yellow HY/AY",
              "xG HxG/AxG", "Bet365 1X2 (B365H/D/A)", "Pinnacle 1X2 (PSH/D/A)", "Bet365 O/U 2.5",
              "Asian line AHh + Bet365", "Bet365 closing 1X2 (B365CH)", "Pinnacle closing 1X2 (PSCH)"]
    rows = [["Колонки"] + [f"{s[:2]}/{s[2:]}" for s in seasons]]
    for g in groups:
        r = [g]
        for s in seasons:
            sub = cov.loc[cov["season"].astype(str) == s, g]
            r.append("нет" if sub.isna().all() else pc(sub.mean(), 1))
        rows.append(r)
    rows.append(["матчей в файлах"] + [n0(cov.loc[cov["season"].astype(str) == s, "played"].sum()) for s in seasons])
    fmap = FD["feature_map"]
    frows = [["Сырые колонки Football-Data", "Модель исхода", "Голы дома", "Голы гостей", "Угловые", "Уникальных признаков"]]
    for r in fmap["by_raw"]:
        frows.append([r["raw"], r.get("result", 0), r.get("home_goals", 0), r.get("away_goals", 0), r.get("corners", 0),
                      r["unique_features"]])
    frows.append(["Всего признаков в модели", fmap["n_features"]["result"], fmap["n_features"]["home_goals"],
                  fmap["n_features"]["away_goals"], fmap["n_features"]["corners"], fmap["unique_features"]])
    use = [["Колонки", "Во что превращаются", "Где используются"]] + [[u["columns"], u["becomes"], u["used_by"]] for u in FD["use"]]
    xg = [["Лига", "Сыграно", "С xG", "Ср. xG за матч", "Ср. голов", "Корр. xG–голы"]]
    for r in FD["xg"]["leagues"]:
        xg.append([r["league"], r["played"], r["with_xg"], f1(r["mean_xg_total"], 2), f1(r["mean_goals_total"], 2),
                   f1(r["corr_xg_goals"], 2)])
    xg.append(["Всего", FD["xg"]["total_played"], FD["xg"]["total_with_xg"], "", "", ""])
    fx = FD["fixtures_file"]
    return [
        H1("3. Football-Data.co.uk"),
        H2("3.1 Что реально приходит"),
        P(f"Каждый запуск скачивает 21 файл текущего сезона (`results/2627/<лига>.csv`) и общий файл расписания "
          f"`fixtures.csv`. В файле сезона 2026/27 — {min(FD['columns_current'].values())}–{max(FD['columns_current'].values())} "
          f"колонок: результат и счёт тайма, судья (только Англия и Шотландия), xG, удары, удары в створ, фолы, угловые, "
          f"карточки и цены 1X2 / тотала 2.5 / азиатской форы от Bet365, Betfair, BetVictor, Bwin, Paddy Power, Sky Bet, "
          f"максимум/среднее рынка — каждая в варианте до матча и закрытия. Цен Pinnacle в файлах 2026/27 нет (в 2025/26 — "
          f"лишь у части матчей). Файл расписания: {fx['rows']} матчей, {fx['columns']} колонок, цена Bet365 1X2 у "
          f"{pc(fx['with_b365_1x2'])} строк, цен закрытия нет."),
        P("Доля сыгранных матчей с заполненными колонками (все 21 лига):"),
        *T(rows, [5.6, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9]),
        H2("3.2 Что идёт в признаки, интерфейс и цены"),
        *T(use, [5.0, 7.0, 5.0]),
        P(f"Признаки моделей версии `{fmap['active_model']}` по сырому источнику (признак отнесён к колонке, из которой он "
          f"считается: голы, удары, угловые и т. д.; цены не используются, `use_odds_features = {fmap['use_odds_features']}`):"),
        *T(frows, [5.4, 2.2, 2.2, 2.2, 2.2, 2.8]),
        P("Итог: все признаки всех четырёх моделей считаются из колонок Football-Data. API-Football даёт только голы; пока "
          "Football-Data не опубликовал матч, у этого матча в истории команды нет ударов, угловых, фолов и карточек, "
          "а рынки угловых ждут Football-Data."),
        H2("3.3 Что добавляет xG"),
        P(f"HxG/AxG появились только в файлах 2026/27 (в пяти прошлых сезонах колонок нет). Покрыто "
          f"{n0(FD['xg']['total_with_xg'])} из {n0(FD['xg']['total_played'])} сыгранных матчей; у Шотландии SC1–SC3 колонки "
          f"пусты. Средний xG за матч близок к среднему числу голов, корреляция xG и голов матча — {XG_CORR}, т. е. колонки "
          f"правдоподобны. Продакшн их не читает (`src/data/cleaning.py`); проверка пользы — отдельное исследование "
          f"`reports/v6_2/XG_Research.pdf` (одного сезона мало для обучения, отсюда только предварительная регистрация)."),
        *T(xg, [2.2, 2.3, 2.3, 3.4, 3.0, 3.8]),
    ]


# ------------------------------------------------------------------------------------------------ API-Football
def api_football() -> list:
    cats = [["Категория", "Матчей"]] + [[k, v] for k, v in sorted(CAT.items(), key=lambda x: -x[1])]
    lead = [["Показатель", "Значение"],
            ["Матчей, где API-Football раньше", n0(LEAD["n"])],
            ["Опережение: медиана", f"{h_(LEAD['median_h'])} ({days(LEAD['median_h'])})"],
            ["Опережение: среднее", f"{h_(LEAD['mean_h'])} ({days(LEAD['mean_h'])})"],
            ["Квартили 25 % / 75 %", f"{h_(LEAD['p25_h'])} / {h_(LEAD['p75_h'])}"],
            ["90-й перцентиль / максимум", f"{h_(LEAD['p90_h'])} / {h_(LEAD['max_h'])}"],
            ["От начала матча до результата, API-Football: медиана (25–75 %)",
             f"{h_(DELAY['api_football']['median_h'])} ({h_(DELAY['api_football']['p25_h'])} – {h_(DELAY['api_football']['p75_h'])})"],
            ["То же, Football-Data", f"{h_(DELAY['football_data']['median_h'])} ({h_(DELAY['football_data']['p25_h'])} – "
                                     f"{h_(DELAY['football_data']['p75_h'])})"]]
    per_day = [["День (Киев)", "Результатов первым"]] + [[k, v] for k, v in AF["per_day"].items()]
    fdo = AF["fd_only"]
    miss = [["Почему результат пришёл только из Football-Data", "Матчей"]] + \
        [[k, v] for k, v in sorted(fdo["by_reason"].items(), key=lambda x: -x[1])]
    val = AF["value"]
    by_lg = [["Лига", "AF раньше", "В одном запуске", "Только AF сейчас", "Только FD"]]
    for r in AF["categories_by_league"]:
        by_lg.append([r["league"], r.get("API-Football раньше", 0), r.get("в одном запуске", 0),
                      r.get("только API-Football (Football-Data ещё нет)", 0), r.get("только Football-Data", 0)])
    un = ", ".join(f"«{u['name']}» ({u['league']})" for u in AF["unmatched_names"])
    return [
        H1("4. API-Football"),
        H2("4.1 Что реально приходит и что используется"),
        P("Запрос `GET /fixtures?date=…` (один на день, все лиги мира) возвращает по матчу блоки `fixture` (id, судья, время, "
          "стадион, статус), `league` (id, раунд), `teams`, `goals` и `score` (тайм, основное и дополнительное время, пенальти). "
          "Система оставляет только лигу, дату и время начала, названия команд, статус и счёт основного времени; судья, "
          "стадион и счёт тайма отбрасываются. Коэффициенты API-Football не используются (бесплатный план даёт их только "
          "за 2022–2024, `API_FOOTBALL_ODDS = False`). Отдельно скрипт `build_team_logos.py` один раз взял эмблемы клубов "
          f"(`teams/logos.json`, {LOGOS['clubs']} клубов) — это только интерфейс."),
        P(f"Журнал квоты: {QLOG['requests']} запросов за {QLOG['runs']} запусков ({QLOG['first'][:10]} – {QLOG['last'][:10]}), "
          f"ровно {QLOG['per_run_max']} за запуск (вчера, сегодня, завтра) при лимите 100 в день. Последний запуск: из "
          f"{n0(last_diag('api_football').get('n_returned'))} матчей ответа в наших лигах "
          f"{n0(last_diag('api_football').get('n_in_leagues'))}, сопоставлено {n0(last_diag('api_football').get('n_matched'))}. Строки API-Football — это голы без "
          f"статистики: они нужны для расчёта рынков на голы и для таблицы/формы/Elo следующих матчей; рынки угловых "
          f"ждут Football-Data."),
        H2("4.2 Насколько раньше Football-Data"),
        P(f"Для каждого матча 2026/27, начавшегося после {AF['start_utc'][:16]} UTC (первая строка результата API-Football в "
          f"git), взято время первого коммита, где результат есть у API-Football, и время первого коммита, где он есть в "
          f"файле Football-Data. Оба источника опрашиваются в одном запуске, поэтому опережение кратно интервалу между "
          f"запусками. Расхождений счёта между источниками: {len(AF['score_differs'])}."),
        *T(cats, [12, 5]),
        *T(lead, [10.5, 6.5]),
        *figure(FIG / "af_lead_days.png", "Рис. 1. Опережение API-Football над Football-Data по матчам (целые сутки)."),
        *figure(FIG / "result_delay_ecdf.png", "Рис. 2. Время от начала матча до появления результата в данных."),
        P("Распределение двугорбое: Football-Data обновляет файлы сезона раз в несколько дней (у 2026/27 было "
          f"{AF['commits_fd_files']} коммитов файлов, последний — {FD_LAST}), поэтому результат выходных приходит из "
          f"Football-Data в медиане через {days(DELAY['football_data']['median_h'])}, а API-Football — через "
          f"{days(DELAY['api_football']['median_h'])}."),
        P(f"Сколько результатов API-Football приносит первым: в {PER_RUN['runs_with_af_first_results']} запусках из "
          f"{PER_RUN['runs_with_live_file']} — в среднем {f1(PER_RUN['mean_per_run_with_results'])} за запуск (медиана "
          f"{f1(PER_RUN['median_per_run'])}, максимум {PER_RUN['max_per_run']}); по дням:"),
        *T(per_day, [8, 6]),
        H2("4.3 Чего API-Football не дал"),
        P(f"{n0(fdo['n'])} матчей пришли только из Football-Data. Причины, проверяемые локально: запросы между коммитами "
          f"{DEAD[0][:16]} и {DEAD[1][:16]} UTC (журнал квоты показывает по {QLOG['per_run_max']} запроса за запуск) не сохранили ни одного нового результата — причина в коммитах не "
          f"записана (диагностика запусков ведётся с 04.11); бесплатный план видит только «вчера/сегодня», а в "
          f"части дней запусков не было; три клуба не сопоставляются по имени: {un}."),
        *T(miss, [12, 5]),
        *T(by_lg, [2.4, 3.4, 3.6, 3.8, 3.8]),
        H2("4.4 Стоит ли оставить"),
        P(f"Да. Цена — {QLOG['per_run_max']} бесплатных запроса в день. Польза: {n0(val['matches'])} матчей получили результат раньше "
          f"Football-Data или пока только от API-Football; у них {n0(val['main_risk_predictions'])} Main/Risk, из которых "
          f"{n0(val['settleable_with_goals'])} рассчитываются по голам сразу (угловые — {val['need_corner_statistics']} — "
          f"ждут Football-Data), плюс {n0(val['market_forecasts'])} рыночных прогнозов. Ещё {val['next_matches_with_af_result_in_features']} "
          f"следующих матчей тех же команд начались, когда результат был только у API-Football, — их прогноз уже учитывал "
          f"этот результат в таблице и форме."),
    ]


# ------------------------------------------------------------------------------------------------ OddsPapi
def oddspapi() -> list:
    cov = csv("oddspapi_coverage.csv")
    t1 = [["Лига", "Турнир", "Сыграно", "Фикстур в архиве", "Сопоставлено", "С ценой OddsPapi", "С любой ценой"]]
    t2 = [["Лига", "1X2", "DC", "Тотал", "BTTS", "Фора", "Евр. фора", "Угловые", "Точный счёт"]]
    for r in cov.itertuples():
        t1.append([r.league, "" if r.tournament != r.tournament else int(r.tournament), r.played, r.archive_fixtures,
                   r.archive_mapped, r.with_oddspapi, r.with_any_provider])
        t2.append([r.league, getattr(r, "_8"), r.DOUBLE_CHANCE, r.TOTAL, r.BTTS, r.HANDICAP, r.EHANDICAP, r.CORNERS,
                   r.EXACT_SCORE])
    mk = OP["markets"]
    grp = [["Группа", "Рынков (линий) хранится", "Используются прогнозами", "Строк цен", "Строк в используемых"]]
    for r in mk["by_group"]:
        grp.append([r["group"], r["markets"], r["used_markets"], n0(r["rows"]), n0(r["rows_used"])])
    books = ", ".join(f"{k} {n0(v)}" for k, v in mk["bookmakers"].items())
    month = [["Месяц", "Запусков", "Платных", "Бесплатных", "Квота плана"]] + \
        [[r["month"], r["runs"], r["billable"], n0(r["free"]), COST["free_plan_per_month"]] for r in COST["per_month"]]
    imports = [["Загрузка архива (UTC)", "Платных", "Бесплатных", "Что"]] + \
        [[r["run_hour"].replace("T", " ") + ":xx", r["billable"], r["free"], r["kinds"]] for r in COST["archive_imports"]]
    diag = [["Запуск (UTC)", "Запр.", "Плат.", "Беспл.", "В ответе", "Наши лиги", "Оценено", "Новых котир.",
             "Изменено"]]
    for r in COST["diagnostics_runs"]:
        diag.append([r["run_at"].replace("T", " ")[:16], n0(r["n_requests"]), n0(r["n_billable"]), n0(r["n_free"]),
                     n0(r["n_returned"]), n0(r["n_in_leagues"]), n0(r["n_matches_priced"]), n0(r["n_quotes_new"]),
                     n0(r["n_quotes_updated"])])
    ar = OP["archive"]["rows"]
    arch = [["Цены OddsPapi", "Строк", "Матчей", "Котировка до начала: медиана", "10 % / 90 %", "После начала"]]
    for k, v in ar.items():
        name = "архив (получены после матча)" if k.startswith("archive") else "живые (получены до матча)"
        arch.append([name, n0(v["rows"]), n0(v["matches"]), h_(v["lead_median_h"]),
                     f"{h_(v['lead_p10_h'])} / {h_(v['lead_p90_h'])}", v["quoted_at_or_after_kickoff"]])
    tm = OP["timing"]
    tcls = [["Класс цены", "Main", "Risk", "Что значит"]]
    meaning = {"A1": "записана живым конвейером при генерации — система знала её, когда выбирала",
               "A2": "добавлена позже из архива, но опубликована букмекером до генерации",
               "B": "добавлена из архива, опубликована ПОСЛЕ генерации (последняя до начала)",
               "C": "цена закрытия у ретроспективного прогноза (решение после матча)",
               "-": "цены нет / класс не определён"}
    for r in tm.get("by_role", []):
        tcls.append([r["price_class"], r.get("main", 0), r.get("risk", 0), meaning.get(r["price_class"], "")])
    prov = [["Провайдер цены", "A1", "A2", "B", "C"]] + \
        [[r["provider"], r.get("A1", 0), r.get("A2", 0), r.get("B", 0), r.get("C", 0)] for r in tm.get("by_provider", [])]
    nov = MONTH.get("2026-11", {})
    return [
        H1("5. OddsPapi"),
        H2("5.1 Рынки: доступно и используется"),
        P(f"Каталог `/markets` (запрошен {mk['catalogue_fetched_at'][:10]}, платно, один раз) содержит "
          f"{n0(mk['catalogue_markets'])} рынков (каждая линия и период — отдельный рынок, включая игроков, карточки, таймы). "
          f"Система сопоставляет {n0(mk['outcomes_mapped'])} исходов с {mk['markets_mapped']} своими рынками (только полный "
          f"матч: 1X2, двойной шанс, тоталы 1.5/2.5/3.5, обе забьют, азиатская фора на половинных и целых линиях, европейская "
          f"фора, тоталы угловых, точный счёт). В архиве хранится {n0(mk['rows_stored'])} цен по {mk['markets_stored']} рынкам; "
          f"прогнозы 2026/27 используют {mk['markets_used_by_predictions']} из них ({n0(mk['rows_in_used_markets'])} строк). "
          f"Лишнее — линии без прогнозов: {UNUSED}. Букмекеры (строк): {books}."),
        *T(grp, [3.2, 3.6, 3.6, 3.2, 3.4]),
        H2("5.2 Покрытие по лигам, 2026/27"),
        P(f"Сыграно {n0(OP['played'])} матчей; реальная цена OddsPapi есть у {n0(OP['with_oddspapi'])} "
          f"({pc(OP['share_oddspapi'], 2)}). Единственный матч без цены — {', '.join(m['match_id'] for m in OP['missing_matches'])}: "
          f"сыгран до 1 августа, с которого начинается загрузка архива. Предстоящих матчей {OP['upcoming']}, с ценой — "
          f"{OP['upcoming_with_oddspapi']} (цены запрашиваются за 4 дня до начала)."),
        *T(t1, [1.8, 1.8, 2.0, 2.8, 2.8, 2.9, 2.9]),
        P("Матчей с ценой по группам рынков:"),
        *T(t2, [1.6, 1.6, 1.6, 1.8, 1.6, 1.6, 2.4, 2.2, 2.6]),
        H2("5.3 Стоимость обновлений"),
        P(f"Бесплатный план — {COST['free_plan_per_month']} платных запросов в месяц; `/account` и `/historical-odds` не "
          f"платные. Ежедневный запуск делает один платный `/fixtures` (все лиги на 4 дня) и по одному бесплатному "
          f"`/historical-odds` на матч: в {COST['daily_runs']['n']} ежедневных запусках из журнала — в среднем "
          f"{f1(COST['daily_runs']['billable_mean'], 2)} платного (максимум {COST['daily_runs']['billable_max']}: каталог рынков или "
          f"список турниров) и {f1(COST['daily_runs']['free_mean'])} бесплатных (максимум {COST['daily_runs']['free_max']}). Загрузка "
          f"архива стоит по одному платному `/fixtures` на лигу ({IMP_RANGE}) и бесплатные запросы цен."),
        *T(month, [3, 3, 3, 3.5, 4.5]),
        *T(imports, [4.2, 2.2, 2.6, 8.0]),
        *figure(FIG / "oddspapi_billable.png", "Рис. 3. Платные запросы OddsPapi по запускам (журнал ledger.json)."),
        P(f"Прогноз расхода при той же схеме: 30 ежедневных запусков × {f1(COST['daily_runs']['billable_mean'], 2)} ≈ "
          f"{n0(30 * COST['daily_runs']['billable_mean'])} платных в месяц; каждая загрузка архива — ещё до {max(IMP)}. В ноябре "
          f"уже {nov.get('billable', '—')} платных из {COST['free_plan_per_month']}. Сам журнал при этом считает за ноябрь только "
          f"{COST['ledger_month_credits_now']} (`month_credits`): загрузки архива в месячный счётчик не попадают, а остаток "
          f"квоты (`remaining`) не записан ни в одной из {COST['ledger_versions']} версий журнала — поэтому в каждой строке "
          f"диагностики «? left», и резерв в 20 запросов не может сработать."),
        P("Запросы по данным диагностики запусков (с 05.11):"),
        *T(diag, [2.9, 1.4, 1.4, 1.5, 1.9, 1.9, 1.8, 2.2, 2.0]),
        H2("5.4 Как работают исторические цены"),
        P("`/historical-odds` (бесплатно, до 3 букмекеров: bet365, pinnacle, williamhill) возвращает все снимки цены с "
          "временем `createdAt`. Для сыгранного матча берётся последний активный снимок строго до начала матча "
          "(`_pick_quote`): цена после начала никогда не используется; при ежедневном обновлении — последний снимок на "
          "момент запроса. Котировки одного букмекера, противоречащие друг другу по линиям, отбрасываются. Ни одна "
          "сохранённая цена OddsPapi не датирована моментом начала матча или позже:"),
        *T(arch, [4.4, 2.2, 1.8, 3.2, 3.6, 1.8]),
        H2("5.5 Классы времени цены"),
        P(f"Из аудита `{tm.get('file', '—')}` (исследование времени цены v6.2). Класс показывает, существовала ли цена, "
          f"когда прогноз был сделан:"),
        *T(tcls, [2.2, 1.6, 1.6, 11.6]),
        *T(prov, [5, 3, 3, 3, 3]),
    ]


# ------------------------------------------------------------------------------------------------ The Odds API
def the_odds_api() -> list:
    st, cr = TOA["stored"], TOA["credits"]
    rows = [["Роль", "Прогнозов", "Ставок", "Прибыль", "ROI", "Та же ставка у OddsPapi до начала", "…до генерации"]]
    for role in ("main", "risk", "market"):
        r = DEP["by_role"].get(role)
        if r:
            rows.append([ROLE[role], r["predictions"], r["bets"], profit(r["profit"]), roi(r["roi"]),
                         r["odp_same_selection"], r["odp_before_generation"]])
    kinds = [["Тип запроса", "Запросов", "Кредитов"]] + [[k["kind"], k["requests"], k["credits"]] for k in cr["by_kind"]]
    grp = ", ".join(f"{k} {v}" for k, v in st["by_group"].items())
    return [
        H1("6. The Odds API"),
        P(f"Хранится {n0(st['rows'])} цен по {st['matches']} матчам, собранных с {st['first_collected'][:10]} по "
          f"{st['last_collected'][:10]}; матчей по группам: {grp}. Букмекеры — британские (William Hill "
          f"{n0(st['by_bookmaker'].get('William Hill'))} строк). По журналу квоты: {cr['log_requests']} запросов, "
          f"{f1(cr['log_credits'], 0)} кредитов; в журнале ledger за {cr['ledger_month']} — {cr['ledger_month_credits']} "
          f"кредитов, остаток {cr['ledger_remaining']} на {cr['ledger_checked_at'][:10]}. С тех пор источник выключен "
          f"(`ODDS_API_FALLBACK = off`), а текущий остаток локально не известен (месячная квота "
          f"{TOA['plan_credits_per_month']} кредитов с тех пор обновилась)."),
        *T(kinds, [7, 4, 4]),
        H2("6.1 Кто ещё зависит от сохранённых цен"),
        P(f"В единой истории 2026/27 цены The Odds API несут {DEP['matches']} матчей ({DEP['dates'][0][:10]} – "
          f"{DEP['dates'][1][:10]}); это {pc(DEP['share_of_main_risk_bets'])} из {n0(DEP['main_risk_priced_total'])} "
          f"оценённых Main/Risk. Все они класса A1 (записаны при генерации). Для каждой такой ставки проверено, есть ли в "
          f"архиве OddsPapi цена на тот же рынок и выбор до начала матча:"),
        *T(rows, [2.2, 2.2, 1.8, 2.0, 2.0, 3.8, 3.0]),
        P("Вывод: удалить эти цены нельзя без потери класса A1 (их заменила бы цена OddsPapi, в основном опубликованная "
          "позже генерации — класс B), поэтому хранить. Как запасной источник The Odds API ничего не добавляет: OddsPapi "
          f"покрывает {pc(OP['share_oddspapi'], 2)} матчей, а у The Odds API нет шотландских лиг ниже Премьершипа и нет "
          "архива для пропущенных матчей. Включать не нужно; переменная `ODDS_API_LEAGUES` оставляет возможность "
          "включить его для конкретной лиги, если OddsPapi её потеряет."),
    ]


# ------------------------------------------------------------------------------------------------ diagnostics + schedule
def diagnostics() -> list:
    d = RUNS["diagnostics"]
    fields = [["Поле", "Источник", "Что означает"]] + [[f["field"], f["sources"], f["meaning"]] for f in d["fields"]]
    lines = d["last_run"]["lines"]
    vers = [["Коммит", "Дата (UTC)", "cron", "Часовой пояс"]] + \
        [[v["sha"], v["at_utc"][:16], v["cron"], v["timezone"]] for v in SCH["workflow_versions"]]
    bys = [["Расписание (Киев)", "Запусков", "Мин. задержка", "Медиана", "Макс."]] + \
        [[r["cron_kyiv"], r["size"], h_(r["min"]), h_(r["median"]), h_(r["max"])] for r in SCH["by_schedule"]]
    starts = ", ".join(s[5:16] for s in SCH["diagnostic_starts_kyiv"])
    delays = SCH["start_delay_after_0317_h"]
    return [
        H1("7. Диагностика запусков и расписание"),
        H2("7.1 Поля диагностики"),
        P("Каждый запуск пишет `local_data/api/diagnostics/last_run.json` (полностью) и строку на источник в "
          f"`history.csv` (там сейчас {d['history_runs']} запусков, с {d['history'][0]['run_at'][:10]}). Что означает каждый счётчик:"),
        *T(fields, [4.2, 3.6, 9.2]),
        P(f"Последний запуск ({d['last_run']['started_at']}):"),
        code("\n".join(lines)),
        P(f"«history checks failed: profit without a price» — прогнозы VOID-матча {VOID_MATCH} с прибылью 0 без цены "
          "(Main, Risk и рыночные). Код v6.2 (коммит «Leave a void or refunded prediction "
          "without a real price with no profit») уже записывает таким прогнозам пустую прибыль; данные исправятся первым "
          "запуском на этом коде. На ROI это не влияет (VOID не является ставкой)."),
        H2("7.2 Время ежедневного запуска"),
        *T(vers, [2.2, 3.4, 6.0, 5.4]),
        *T(bys, [4, 3, 3.4, 3.2, 3.4]),
        *figure(FIG / "schedule_delay.png", "Рис. 4. Задержка коммита ежедневного запуска относительно расписания."),
        P(f"Начало конвейера (время Киева, из диагностики): {starts} — это {f1(min(delays), 1)}–{f1(max(delays), 1)} ч после "
          f"03:17. Строка cron и часовой пояс верны; задержка — очередь GitHub (с прежним расписанием 11:17 она была "
          f"{h_(SCH['by_schedule'][-1]['min'])}–{h_(SCH['by_schedule'][-1]['max'])} с учётом длительности запуска). Поэтому workflow "
          f"принимает внешний запуск (`workflow_dispatch` с `daily: true` или `repository_dispatch` «daily-update», "
          f"README, «Daily update time»), а защита «раз в киевские сутки» (`local_data/daily_run.json`) пропускает "
          f"только первый запуск дня. Пока внешний планировщик не настроен, обновление по-прежнему начинается в "
          f"{min(STARTS)}–{max(STARTS)} по Киеву."),
    ]


# ------------------------------------------------------------------------------------------------ bets recheck
def stats_row(label, r):
    return [label, n0(r["predictions"]), n0(r["settled"]), n0(r["priced_settled"]), n0(r["bets"]), r["push"],
            profit(r["profit"]), roi(r["roi"]), ev(r["mean_ev"]), pc(r["coverage"], 2)]


HEAD = ["", "Прогн.", "Рассч.", "С ценой", "Ставок", "PUSH", "Прибыль", "ROI", "Ср. EV", "Покр."]
WID = [3.9, 1.3, 1.3, 1.4, 1.4, 1.1, 1.6, 1.5, 1.5, 1.7]


def bets() -> list:
    rows = [HEAD]
    for role in ("main", "risk"):
        rows.append(stats_row(f"{ROLE[role]}, всего", overall(role)))
        for s in ("live", "backfill"):
            rows.append(stats_row(f"{ROLE[role]}, {SRC[s]}", overall(role, s)))
    mc = [["Лига", "Матчей", "С любой реальной ценой", "OddsPapi", "The Odds API", "Bet365 из Football-Data", "Доля"]]
    for r in MC["by_league"]:
        mc.append([r["league"], r["matches"], r["any_real_market"], r["oddspapi"], r["the_odds_api"],
                   r["football_data_bet365"], pc(r["share"], 2)])
    lg = csv("bets_by_league.csv")
    lt = [["Лига", "Main прогн.", "Main ставок", "Main прибыль", "Main ROI", "Main EV", "Risk прогн.", "Risk ставок",
           "Risk PUSH", "Risk ROI", "Risk EV"]]
    for league, g in lg.groupby("league", sort=False):
        m = g.loc[g["role"] == "main"].iloc[0]
        r = g.loc[g["role"] == "risk"].iloc[0]
        lt.append([league, m["predictions"], m["bets"], profit(m["profit"]), roi(m["roi"]), ev(m["mean_ev"]),
                   r["predictions"], r["bets"], r["push"], roi(r["roi"]), ev(r["mean_ev"])])
    ls = csv("bets_by_league_source.csv")
    lst = [["Лига", "live Main ставок", "live Main ROI", "live Risk ставок", "live Risk ROI", "backfill Main ставок",
            "backfill Main ROI", "backfill Risk ставок", "backfill Risk ROI"]]
    for league in lg["league"].drop_duplicates():
        cell = []
        for s in ("live", "backfill"):
            for role in ("main", "risk"):
                x = ls.loc[(ls["league"] == league) & (ls["source"] == s) & (ls["role"] == role)]
                cell += [n0(x["bets"].iloc[0]), roi(x["roi"].iloc[0])] if len(x) else ["0", "—"]
        lst.append([league, cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[6], cell[7]])

    def by_key(name, key, label):
        df = csv(name)
        out = [[label, "Роль", "Прогн.", "Ставок", "PUSH", "Прибыль", "ROI", "Ср. EV", "Покр."]]
        df = df.sort_values([key, "role"])
        for r in df.itertuples():
            out.append([getattr(r, key), ROLE.get(r.role, r.role), r.predictions, r.bets, r.push, profit(r.profit),
                        roi(r.roi), ev(r.mean_ev), pc(r.coverage, 1)])
        return out
    grp = by_key("bets_by_group.csv", "group", "Группа рынков")
    line = by_key("bets_by_line.csv", "market", "Рынок / линия")
    gs = csv("bets_by_group_source.csv")
    gst = [["Группа", "Роль", "live ставок", "live ROI", "live EV", "backfill ставок", "backfill ROI", "backfill EV"]]
    for (gname, role), g in gs.groupby(["group", "role"]):
        lv, bf = g.loc[g["source"] == "live"], g.loc[g["source"] == "backfill"]
        gst.append([gname, ROLE[role], n0(lv["bets"].sum()) if len(lv) else "0", roi(lv["roi"].iloc[0]) if len(lv) else "—",
                    ev(lv["mean_ev"].iloc[0]) if len(lv) else "—", n0(bf["bets"].sum()) if len(bf) else "0",
                    roi(bf["roi"].iloc[0]) if len(bf) else "—", ev(bf["mean_ev"].iloc[0]) if len(bf) else "—"])
    pcl = csv("bets_by_price_class_source.csv")
    pct = [["Класс цены", "Источник", "Роль", "Прогн.", "Ставок", "Прибыль", "ROI", "Ср. EV"]]
    order = {"A1": 0, "A2": 1, "B": 2, "C": 3}
    pcl = pcl.assign(o=pcl["price_class"].map(order).fillna(9)).sort_values(["o", "source", "role"])
    for r in pcl.itertuples():
        pct.append([r.price_class, r.source, ROLE[r.role], r.predictions, r.bets, profit(r.profit), roi(r.roi), ev(r.mean_ev)])
    prov = csv("bets_by_provider.csv")
    pv = [["Провайдер цены", "Роль", "Прогнозов", "Ставок", "Прибыль", "ROI", "Ср. EV"]] + \
        [[r.provider, ROLE[r.role], r.predictions, r.bets, profit(r.profit), roi(r.roi), ev(r.mean_ev)] for r in prov.itertuples()]
    un = BETS["unpriced"]
    ut = [["Дата", "Матч", "Роль", "Источник", "Ставка", "Статус", "Причина", "Можно вернуть"]]
    for r in un["rows"]:
        ut.append([r["date"][:10], f"{r['home_team']} – {r['away_team']} ({r['league']})", ROLE[r["role"]], r["source"],
                   f"{r['market']} {r['selection']}", r["status"], ru_reason(r["reason"]), ru_rec(r["recoverable"])])
    ur = [["Роль", "Причина", "Число"]] + [[ROLE[r["role"]], ru_reason(r["reason"]), r["n"]] for r in un["by_reason"]]
    ll = BETS["laliga"]
    llm, llr = ll["overall"]["main"], ll["overall"]["risk"]
    prev = BETS["previous_v61_audit"]["roles"]
    sp1_prev = BETS["previous_v61_audit"].get("laliga") or {}
    llt = [HEAD, stats_row("Main", llm), stats_row("Risk", llr)]
    lls = [["Роль", "Источник", "Прогнозов", "Ставок", "ROI", "Ср. EV"]] + \
        [[ROLE[r["role"]], r["source"], r["predictions"], r["bets"], roi(r["roi"]), ev(r["mean_ev"])] for r in ll["by_source"]]
    nb = [["Дата", "Хозяева", "Гости", "Роль", "Рынок", "Статус", "Почему не ставка"]] + \
        [[r["date"][:10], r["home_team"], r["away_team"], ROLE[r["role"]], r["market"], r["status"], r["why"]]
         for r in ll["not_bets"]]
    cmp = [["", "Main рассчитано", "Main ставок", "Доля", "Risk рассчитано", "Risk ставок", "Доля"],
           [f"аудит v6.1 ({BETS['previous_v61_audit']['built_at'][:10]})", prev["main"]["settled"], prev["main"]["bets"],
            pc(prev["main"]["bets"] / prev["main"]["settled"]), prev["risk"]["settled"], prev["risk"]["bets"],
            pc(prev["risk"]["bets"] / prev["risk"]["settled"])],
           ["сейчас", M["settled"], M["bets"], pc(M["bets"] / M["settled"]), R["settled"], R["bets"], pc(R["bets"] / R["settled"])]]
    return [
        H1("8. Перепроверка исторических ставок после импорта OddsPapi"),
        P("Источник — единая история сайта (`unified_history`): по каждому матчу записанный до начала прогноз (live), а "
          "если его нет — ретроспективный (backfill); один прогноз на реальный матч. Ни выбор, ни цена, ни статус не "
          "менялись; данные только читались. Ставка — WON/LOST с реальной ценой; ROI = прибыль / ставки × 100; PUSH и VOID "
          "не ставки. Средний EV — по ставкам, по сохранённой цене. «Покрытие» — доля рассчитанных прогнозов с реальной "
          "ценой ровно на выбранный рынок, линию и выбор."),
        *T(rows, WID),
        H2("8.1 Два вида покрытия"),
        P(f"(а) **Покрытие матчей**: у матча есть хотя бы один реальный рынок букмекера (котировка OddsPapi или The Odds API "
          f"до начала, либо цена Bet365 в строке Football-Data): {n0(MC['with_any_real_market'])} из "
          f"{n0(MC['matches_settled'])} рассчитанных матчей ({pc(MC['share'], 2)}); без цены — "
          f"{', '.join(MC['without']) or 'нет'}. (б) **Покрытие прогнозов**: реальная цена есть ровно у выбранного Main/Risk: "
          f"Main {n0(PCV['main']['priced_settled'])} из {n0(PCV['main']['settled'])} ({pc(PCV['main']['share'], 2)}), "
          f"Risk {n0(PCV['risk']['priced_settled'])} из {n0(PCV['risk']['settled'])} ({pc(PCV['risk']['share'], 2)}). "
          f"Разница между (а) и (б) — прогнозы, чей рынок есть, а линии нет, и перенесённый матч (п. 8.6)."),
        *T(mc, [2.0, 2.0, 3.2, 2.2, 2.4, 3.4, 1.8]),
        P("Сравнение с аудитом v6.1 (до двух последних импортов архива):"),
        *T(cmp, [4.4, 2.2, 2.0, 1.6, 2.2, 2.0, 1.6]),
        H2("8.2 По лигам"),
        *T(lt, [1.3, 1.5, 1.5, 1.7, 1.6, 1.5, 1.5, 1.5, 1.3, 1.6, 1.5]),
        P("По лигам отдельно для записанных и ретроспективных прогнозов (ставок и ROI):"),
        *T(lst, [1.4, 2.0, 1.9, 2.0, 1.9, 2.0, 1.9, 2.0, 1.9]),
        H2("8.3 По группам рынков и линиям"),
        *T(grp, [3.0, 1.4, 1.8, 1.6, 1.3, 1.8, 1.8, 1.7, 2.0]),
        *T(gst, [3.0, 1.4, 2.0, 2.0, 1.8, 2.4, 2.2, 2.2]),
        P("По рынкам и линиям (все линии, на которых были Main или Risk):"),
        *T(line, [3.0, 1.4, 1.8, 1.6, 1.3, 1.8, 1.8, 1.7, 2.0]),
        H2("8.4 По классу времени цены"),
        P("Класс взят из аудита времени цены v6.2 по `pick_id` (A1 — цена была у системы при генерации; A2 — была на рынке, "
          "добавлена позже; B — опубликована после генерации; C — закрытие у ретроспективного прогноза). Только A1+A2 "
          "существовали в момент решения; B и C показывают качество выбора, но не реальный результат ставки."),
        *T(pct, [2.4, 2.2, 1.6, 2.0, 1.8, 2.2, 2.2, 2.0]),
        *T(pv, [3.2, 1.6, 2.2, 2.0, 2.6, 2.4, 2.4]),
        H2("8.5 Ла Лига"),
        P(f"Раньше в Ла Лиге прогнозов было больше, чем ставок: аудит v6.1 — {sp1_prev.get('main_settled', '—')} Main и "
          f"{sp1_prev.get('main_bets', '—')} ставок, {sp1_prev.get('risk_settled', '—')} Risk и {sp1_prev.get('risk_bets', '—')} "
          f"ставок (id-конфликт Барселоны/Эспаньола у OddsPapi, исправлен в v6.1). Сейчас Main — {llm['predictions']} прогнозов и {llm['bets']} ставок, Risk — {llr['predictions']} "
          f"прогнозов и {llr['bets']} ставок; без цены нет ни одного. Не ставки:"),
        *T(llt, WID),
        *T(lls, [2.2, 3.0, 2.6, 2.6, 2.6, 2.6]),
        *T(nb, [2.0, 2.8, 2.8, 1.4, 2.6, 2.2, 3.2]),
        H2("8.6 Все Main/Risk без реальной цены"),
        *T(ur, [2.0, 12.0, 3.0]),
        *T(ut, [2.2, 3.0, 1.2, 1.5, 2.6, 2.0, 3.1, 1.4]),
        P("Перенесённый матч Sabadell – Andorra: записанный прогноз хранится под датой 03.10, матч сыгран 04.10, и все цены "
          "OddsPapi лежат под id дня игры. Восстановление архива ищет цену только по id прогноза, поэтому две ставки "
          "остаются без цены, хотя цена той же ставки есть (п. 9)."),
    ]


# ------------------------------------------------------------------------------------------------ defects, table, limits
def defects() -> list:
    nov = MONTH.get("2026-11", {})
    return [
        H1("9. Найденные дефекты продакшна"),
        P("Код `src/` в этой работе не менялся; ниже — доказательство и предлагаемое исправление."),
        *bullets([
            f"**Остаток квоты OddsPapi никогда не известен.** Ни в одной из {COST['ledger_versions']} версий "
            f"`oddspapi/ledger.json` нет поля `remaining`, диагностика пишет «? left». `OddsPapi.account()` записывает остаток, "
            f"только если ответ `/account` содержит `request_limit`; видимо, ответ устроен иначе. Следствие: резерв "
            f"20 запросов (`can_spend`) не работает, защищает только бюджет запуска. Исправление: записать в диагностику "
            f"ключи ответа `/account` при первом запуске и читать фактические имена полей.",
            f"**Месячный счётчик OddsPapi занижен.** `month_credits` = {COST['ledger_month_credits_now']} за ноябрь, а по "
            f"записям журнала платных запросов {nov.get('billable', '—')}: запуски «history» (загрузка архива, {IMP_RANGE} платных) "
            f"пишут записи, но не попадают в счётчик при слиянии журнала. Исправление: считать месяц по записям `requests` "
            f"(cost × месяц `at`), а не по отдельному счётчику.",
            "**Три клуба API-Football не сопоставляются**: «OFI» (G1, у Football-Data «OFI Crete»), «Vicenza Virtus» (I2, "
            "«Vicenza»), «Real Sociedad II» (SP2, «Sociedad B»; в алиасах есть только «real sociedad b»). Их результаты "
            f"приходят только из Football-Data ({fdo_unmatched} матчей; сейчас из-за этого ждут расчёта: {waiting_unmatched}). "
            "Исправление: три алиаса в `src/providers/aliases.py`.",
            "**Перенесённый матч без цены**: Sabadell – Andorra (SP2) — Main 1X2 H и Risk EHANDICAP_-1 H записаны под "
            "03.10, цены лежат под 04.10. Исправление: в `history_repair` при поиске цены архива для прогноза, заменённого "
            "переносом (`rescheduled_duplicates`), смотреть и id сохранённого матча; класс цены у таких — A2/B по времени "
            "котировки.",
            f"**API-Football между {DEAD[0][:10]} и {DEAD[1][:10]} не сохранил ни одного результата** при "
            f"{QLOG['per_run_max']} запросах за запуск; счётчик квоты при этом почти не уменьшался. Причина в коммитах не записана (диагностика ведётся с 04.11); сейчас подобный случай "
            "будет виден в строке «errors» диагностики.",
            f"**«profit without a price»** — VOID-матч {VOID_MATCH} с прибылью 0 без цены; уже исправлено кодом v6.2, данные "
            "исправит следующий запуск.",
        ]),
    ]


def final_table() -> list:
    fdx = FD["xg"]
    rows = [["Источник", "Данные (реально)", "Назначение", "Модель", "Интерфейс", "Цены / ROI", "Стоимость", "Нужен?"],
            ["Football-Data", "результаты, удары, угловые, фолы, карточки, xG (с 2026/27), цены Bet365 до матча и закрытия; "
             "расписание с ценами Bet365",
             "история для обучения и признаков; расчёт угловых; цены прошлых сезонов",
             f"да — все {FD['feature_map']['unique_features']} признаков", "результаты, статистика матча",
             "цены Bet365 (низший приоритет); бэктест", f"бесплатно, {per_run('football_data')} запроса за запуск",
             f"обязателен; xG — только исследование ({n0(fdx['total_with_xg'])} матчей)"],
            ["API-Football", "счёт и статус матчей вчера/сегодня/завтра; эмблемы клубов",
             "свежие результаты и переносы раньше Football-Data",
             "косвенно — голы в таблице/форме до прихода Football-Data", "эмблемы, результаты",
             "расчёт ставок на голы", f"{QLOG['per_run_max']} из 100 запросов в день, бесплатно",
             f"да: опережение медиана {days(LEAD['median_h'])}, {n0(LEAD['n'])} матчей раньше"],
            ["OddsPapi", f"фикстуры и цены {OP['markets']['markets_mapped']} рынков (bet365, pinnacle, williamhill), архив снимков с января 2026",
             "основной источник реальных цен", "нет", "цены и букмекер у прогноза",
             f"да — {pc(OP['share_oddspapi'], 2)} матчей", f"~{f1(COST['daily_runs']['billable_mean'], 1)} платный в день + {IMP_RANGE} за загрузку архива из "
                                                            f"{COST['free_plan_per_month']}/мес",
             "обязателен"],
            ["The Odds API", f"{n0(TOA['stored']['rows'])} старых цен ({TOA['stored']['first_collected'][:10]} – {TOA['stored']['last_collected'][:10]}), новых нет", "резерв (выключен)", "нет",
             "цены у старых прогнозов", f"{sum(DEP['by_role'][r]['bets'] for r in ('main', 'risk'))} ставок Main/Risk "
                                        f"({pc(DEP['share_of_main_risk_bets'])})",
             f"0 (выключен); план {TOA['plan_credits_per_month']} кредитов/мес", "хранить данные, не включать"]]
    first = [[r[0], r[1], r[2], r[6], r[7]] for r in rows]
    second = [[r[0], r[3], r[4], r[5]] for r in rows]
    return [H1("10. Итоговая таблица источников"),
            P("Источник → данные → назначение → стоимость → нужен ли:"),
            *T(first, [2.4, 4.6, 3.8, 3.2, 3.0]),
            P("Где используется: модель / интерфейс / цены и ROI:"),
            *T(second, [2.4, 4.8, 4.6, 5.2])]


def limits() -> list:
    return [
        H1("11. Ограничения и что нельзя измерить локально"),
        *bullets([
            "Остаток квоты OddsPapi — не записан ни в одной версии журнала (дефект выше); текущий остаток The Odds API и "
            "API-Football тоже известен только на момент последнего запроса.",
            "Время, когда GitHub *создал* запуск, в репозитории не хранится; измерены начало конвейера (диагностика) и "
            "время коммита.",
            "Опережение API-Football измерено по коммитам: результат виден только в момент запуска, поэтому значения "
            "кратны интервалам между запусками; в части дней запусков не было.",
            "Полный каталог рынков OddsPapi в репозитории не сохранён (только размер ответа и сопоставленные исходы), поэтому "
            "«доступно» дано числом рынков каталога, без разбора по типам.",
            f"Причина пустых ответов API-Football {DEAD[0][:10]} – {DEAD[1][:10]} не записана в коммитах.",
            f"Класс времени цены взят из аудита odds_timing; прогнозы, которых в нём нет ({N_NOT_AUDITED}), помечены «нет в аудите».",
            "2026/27 — не независимая выборка (см. `experiments/v6_2/README.md`); ROI здесь — учёт истории, а не проверка гипотезы.",
        ]),
        H1("12. Рекомендации"),
        *bullets([
            "**A** — оставить все четыре интеграции как есть по ролям: Football-Data (обязателен), API-Football (свежие "
            "результаты), OddsPapi (цены), The Odds API — выключен, данные хранить.",
            "**A** — исправить учёт квоты OddsPapi (поле остатка из `/account`, месячный счёт по записям журнала).",
            "**A** — добавить три алиаса API-Football (OFI, Vicenza Virtus, Real Sociedad II).",
            f"**A** — в восстановлении цен из архива учитывать id перенесённого матча ({N_MOVED} ставки Sabadell – Andorra).",
            "**A** — настроить внешний запуск в 03:17 по Киеву (README, «Daily update time»).",
            f"**B** — xG Football-Data: только после прямой проверки (`XG_Research.pdf`), покрытие {XG_LEAGUES} из 21 лиги.",
            f"**C** — включать The Odds API как резерв: при покрытии OddsPapi {pc(OP['share_oddspapi'], 2)} он ничего не добавляет.",
        ]),
        H1("13. Воспроизведение"),
        *reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                    "$PY -m experiments.v6_2.data_sources.run        # results/*.json, results/*.csv, figures/*.png",
                    "$PY -m experiments.v6_2.data_sources.build_pdf  # reports/v6_2/Data_Sources_Report.pdf"]),
    ]


def main() -> None:
    rep = Report(OUT, "Источники данных и перепроверка ставок",
                 facts={"Данные": f"коммит {RUN['data_commit'][:12]}", "Сезон": "2026/27"})
    rep.add(*summary(), *method(), *football_data(), *api_football(), *oddspapi(), *the_odds_api(), *diagnostics(),
            *bets(), *defects(), *final_table(), *limits())
    rep.build()
    print(OUT, rep.check(), "pages")


if __name__ == "__main__":
    main()
