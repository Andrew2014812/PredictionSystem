"""The data-update workflow: daily at 11:00 Kyiv (DST-proof), manual modes, no concurrent runs."""
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "update-data.yml"


def _wf():
    data = yaml.safe_load(WF.read_text())
    return data, data.get("on") or data.get(True)          # YAML 1.1 reads "on" as True


def test_daily_schedule_and_manual_modes_exist():
    data, on = _wf()
    assert {c["cron"] for c in on["schedule"]} == {"0 8 * * *", "0 9 * * *"}
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["mode"]["options"] == ["results", "odds", "full", "history"]
    assert len(inputs) <= 10                                          # GitHub's limit for manual inputs
    assert {"league", "day", "match_id", "force", "credit_budget"} <= set(inputs)
    assert data["concurrency"]["group"] == "update-data" and data["concurrency"]["cancel-in-progress"] is False
    steps = data["jobs"]["update"]["steps"]
    assert any(s.get("with", {}).get("ref") == "main" for s in steps)
    env = next(s for s in steps if s.get("name") == "Run pipeline")["env"]
    assert env["ODDSPAPI_KEY"] == "${{ secrets.ODDSPAPI_KEY }}"            # the key only from Secrets


def _guard_script() -> str:
    data, _ = _wf()
    return next(s for s in data["jobs"]["update"]["steps"] if s.get("id") == "time")["run"]


def _run_guard(schedule: str, offset: str, tmp_path) -> bool:
    script = (_guard_script().replace("${{ github.event_name }}", "schedule")
              .replace("${{ github.event.schedule }}", schedule)
              .replace("$(TZ=Europe/Kyiv date +%z)", offset))
    out, summary = tmp_path / "out", tmp_path / "summary"
    out.write_text("")
    subprocess.run(["bash", "-e", "-c", script], check=True,
                   env={"GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary), "PATH": "/usr/bin:/bin"})
    return "run=true" in out.read_text()


def test_exactly_one_daily_run_in_summer_and_in_winter(tmp_path):
    for offset in ("+0300", "+0200"):
        runs = [_run_guard(s, offset, tmp_path) for s in ("0 8 * * *", "0 9 * * *")]
        assert runs.count(True) == 1, (offset, runs)


def test_the_chosen_schedule_is_11_00_in_kyiv_all_year():
    kyiv = ZoneInfo("Europe/Kyiv")
    day = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for _ in range(366):
        chosen = []
        for hour, cron in ((8, "0 8 * * *"), (9, "0 9 * * *")):
            run_at = day.replace(hour=hour)
            offset = run_at.astimezone(kyiv).strftime("%z")          # what the guard sees when it runs
            if cron == ("0 8 * * *" if offset == "+0300" else "0 9 * * *"):
                chosen.append(run_at)
        assert len(chosen) == 1 and chosen[0].astimezone(kyiv).hour == 11, day   # incl. both DST switch days
        day += timedelta(days=1)


def test_history_mode_downloads_then_fills_recorded_prices_only_by_hand():
    data, _ = _wf()
    steps = data["jobs"]["update"]["steps"]
    hist = next(s for s in steps if s.get("name", "").startswith("Download the price archive"))
    assert hist["if"] == "inputs.mode == 'history'"                     # never on the daily schedule
    run = hist["run"]
    assert run.index("oddspapi_history.py") < run.index("restore_recorded_odds.py --go") < run.index(
        "regenerate_predictions.py")
    assert next(s for s in steps if s.get("name") == "Run pipeline")["if"].endswith("inputs.mode != 'history'")
