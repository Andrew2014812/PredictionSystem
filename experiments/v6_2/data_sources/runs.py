"""Run diagnostics (what each count of a run means, the values of the committed runs) and the daily schedule.

The schedule is measured from the repository itself: the cron line of every committed version of
``.github/workflows/update-data.yml`` and the start / commit time of every daily data run (diagnostics
``started_at`` since 05.11.2026, commit times of the "(daily)" data commits before that). GitHub's own "created"
time of a run is not in the repository; the start of the pipeline is a few minutes after it.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import common as C

WORKFLOW = ".github/workflows/update-data.yml"

# every counter a source can write (src/providers/*, src/data/download.py, scripts/run_pipeline.py)
FIELDS = [
    ("requests", "все", "HTTP-запросы источника в этом запуске (без ответов из кэша)"),
    ("cached", "OddsPapi", "ответы, взятые из локального кэша (запроса не было)"),
    ("billable / free", "OddsPapi", "платные (расходуют месячную квоту 250) и бесплатные (/account, /historical-odds) запросы"),
    ("http", "все", "HTTP-статусы ответов (200, 404 = «нет коэффициентов», 429, 5xx, network)"),
    ("errors", "все", "сообщения об ошибках API (например, тело ответа с ошибкой при HTTP 200)"),
    ("files / files_unchanged", "Football-Data", "файлы сезона, изменившиеся / не изменившиеся с прошлого запуска"),
    ("season_played", "Football-Data", "сыгранных матчей во всех файлах текущего сезона"),
    ("new_results", "Football-Data, API-Football", "результаты, которых раньше не было в данных (FD — новые строки файлов; AF — матч не был отмечен сыгранным)"),
    ("returned", "API-Football, OddsPapi", "матчей в ответе провайдера (все лиги мира)"),
    ("in_leagues", "API-Football, OddsPapi", "из них в 21 лиге FootPredict"),
    ("matched", "API-Football, OddsPapi", "из них сопоставлены с нашими командами (лига, дата, обе команды)"),
    ("skipped", "API-Football, OddsPapi", "пропущенные с причиной: не наша лига, неизвестное имя, статус PST/CANC, уже начался, конфликт id"),
    ("new_fixtures", "API-Football", "матчи, которых ещё не было в данных"),
    ("finished", "API-Football", "завершённые матчи (FT/AET/PEN) среди сопоставленных"),
    ("matches_asked / matches_priced", "OddsPapi", "матчей, у которых запрошены цены / получена хоть одна реальная цена"),
    ("quotes_new / quotes_updated", "OddsPapi, The Odds API", "новые котировки (выбор ещё не был оценён этим букмекером) / изменившиеся цены"),
    ("credits", "The Odds API", "потраченные кредиты (0, если источник выключен)"),
    ("left", "все API", "остаток квоты: API-Football — из заголовка ответа; OddsPapi — из /account («?» = неизвестно); The Odds API — из журнала"),
    ("leagues_updated", "все", "лиги, данные которых этот источник реально изменил"),
    ("seconds", "все", "время источника (запросы и разбор)"),
    ("settled_now / voided_now", "Predictions", "прогнозы, рассчитанные / аннулированные в этом запуске"),
    ("history_checks", "Predictions", "инварианты истории (src/analytics/invariants.py): один Main и один Risk на матч, цены до начала и т. д."),
    ("waiting_for_a_result", "Predictions", "прошедшие матчи, ещё ждущие результата"),
]


def schedule() -> dict:
    cm = C.commits(WORKFLOW)
    versions = []
    for r in cm.itertuples():
        text = (C.blob(r.sha, WORKFLOW) or b"").decode("utf-8", "replace")
        cron = re.findall(r'cron:\s*"([^"]+)"', text)
        tz = re.findall(r"timezone:\s*\"?([A-Za-z/_]+)", text)
        versions.append({"sha": r.sha[:7], "at_utc": str(r.at_utc), "subject": r.subject, "cron": "; ".join(cron),
                         "timezone": tz[0] if tz else "UTC",
                         "dispatch": "repository_dispatch" in text})
    vers = pd.DataFrame(versions)
    # keep only the versions whose schedule changed
    vers = vers.loc[(vers[["cron", "timezone"]] != vers[["cron", "timezone"]].shift()).any(axis=1)]

    data = C.commits("local_data/daily_run.json") if C.git("log", "--format=%H", "-1", "--", "local_data/daily_run.json").strip() else pd.DataFrame()
    daily = C.commits("local_data/predictions/picks.parquet")
    daily = daily.loc[daily["subject"].str.contains(r"\(daily\)")].copy()
    daily["kyiv"] = C.utc_to_kyiv(daily["at_utc"]).to_numpy()
    diag = pd.read_csv(C.API / "diagnostics" / "history.csv", usecols=["run_at", "source"])
    starts = pd.to_datetime(diag["run_at"].drop_duplicates())
    starts_kyiv = C.utc_to_kyiv(starts)

    def due(ts_utc: pd.Timestamp) -> pd.Timestamp:
        """Scheduled time (UTC) of the daily run the commit belongs to, from the workflow version in force.
        Two UTC cron lines one hour apart (summer / winter) aim at ONE Kyiv time: the Kyiv time both lines
        produce (UTC+2 or UTC+3); the other line's run is skipped by the workflow's hour check."""
        active = vers.loc[pd.to_datetime(vers["at_utc"]) <= ts_utc]
        v = active.iloc[-1] if len(active) else vers.iloc[0]
        entries = [tuple(map(int, re.match(r"(\d+) (\d+)", e.strip()).groups())) for e in v["cron"].split(";")]
        zone = v["timezone"]
        if zone == "UTC" and len(entries) > 1:
            sets = [{((h + o) % 24, mi) for o in (2, 3)} for mi, h in entries]
            target = set.intersection(*sets)
        else:
            target = None
        best = None
        for mi, h in entries:
            local = pd.Timestamp(ts_utc).tz_localize("UTC").tz_convert(zone)
            for back in range(0, 3):
                sched = (local.normalize() - pd.Timedelta(days=back)).replace(hour=h, minute=mi)
                if sched > local:
                    continue
                k = sched.tz_convert(C.KYIV)
                if target and (k.hour, k.minute) not in target:
                    continue
                best = sched if best is None or sched > best else best
                break
        return best.tz_convert("UTC").tz_localize(None)
    daily["due_utc"] = [due(t) for t in daily["at_utc"]]
    daily["delay_h"] = (daily["at_utc"] - daily["due_utc"]).dt.total_seconds() / 3600
    daily["cron_kyiv"] = C.utc_to_kyiv(daily["due_utc"]).dt.strftime("%H:%M").to_numpy()
    C.save_csv("daily_runs.csv", daily[["sha", "at_utc", "kyiv", "due_utc", "cron_kyiv", "delay_h", "subject"]])
    by_cron = daily.groupby("cron_kyiv")["delay_h"].agg(["size", "min", "median", "max"]).reset_index()
    return {"workflow_versions": vers.to_dict("records"), "daily_commits": int(len(daily)),
            "by_schedule": by_cron.to_dict("records"),
            "diagnostic_starts_kyiv": [str(t) for t in starts_kyiv],
            "start_delay_after_0317_h": [round((t - t.normalize().replace(hour=3, minute=17)).total_seconds() / 3600, 2)
                                         for t in starts_kyiv],
            "daily_run_json": json.loads((C.DATA / "daily_run.json").read_text()),
            "runs": daily.assign(at_utc=daily["at_utc"].astype(str), kyiv=daily["kyiv"].astype(str),
                                 due_utc=daily["due_utc"].astype(str))[["sha", "at_utc", "kyiv", "cron_kyiv", "delay_h"]]
            .to_dict("records")}


def diagnostics() -> dict:
    last = json.loads((C.API / "diagnostics" / "last_run.json").read_text())
    hist = pd.read_csv(C.API / "diagnostics" / "history.csv")
    return {"fields": [{"field": a, "sources": b, "meaning": c} for a, b, c in FIELDS],
            "last_run": {"started_at": last["started_at"], "finished_at": last["finished_at"], "lines": last["lines"],
                         "timing": last.get("timing_seconds")},
            "history_runs": int(hist["run_at"].nunique()),
            "history": hist.drop(columns=["line"]).to_dict("records"),
            "history_lines": hist[["run_at", "source", "line"]].to_dict("records")}


def run() -> dict:
    out = {"diagnostics": diagnostics(), "schedule": schedule()}
    C.save_json("runs.json", out)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["schedule"]["workflow_versions"], indent=1))
    print(json.dumps(r["schedule"]["by_schedule"], indent=1))
    print(r["schedule"]["diagnostic_starts_kyiv"])
    for x in r["schedule"]["runs"]:
        print(x)
