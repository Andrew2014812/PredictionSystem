"""The data-update workflow: daily at 03:17 Kyiv (one run per Kyiv date), manual modes, no concurrent runs."""
import json
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
    assert on["schedule"] == [{"cron": "17 3 * * *", "timezone": "Europe/Kyiv"}]     # 03:17 Kyiv, summer and winter
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["mode"]["options"] == ["results", "odds", "full", "history"]
    assert len(inputs) <= 10                                          # GitHub's limit for manual inputs
    assert {"league", "day", "match_id", "force", "credit_budget"} <= set(inputs)
    assert data["concurrency"]["group"] == "update-data" and data["concurrency"]["cancel-in-progress"] is False
    steps = data["jobs"]["update"]["steps"]
    assert any(s.get("with", {}).get("ref") == "main" for s in steps)
    env = next(s for s in steps if s.get("name") == "Run pipeline")["env"]
    assert env["ODDSPAPI_KEY"] == "${{ secrets.ODDSPAPI_KEY }}"            # the key only from Secrets
    assert env["ODDS_API_FALLBACK"].endswith("|| 'off' }}")                # no The Odds API credits by default


def _guard_script() -> str:
    data, _ = _wf()
    return next(s for s in data["jobs"]["update"]["steps"] if s.get("id") == "time")["run"]


def _run_guard(event: str, today: str, last: str | None, tmp_path) -> tuple[bool, str]:
    script = (_guard_script().replace("${{ github.event_name }}", event)
              .replace("$(TZ=Europe/Kyiv date +%F)", today))
    work = tmp_path / today / str(last)
    (work / "local_data").mkdir(parents=True, exist_ok=True)
    if last is not None:
        (work / "local_data" / "daily_run.json").write_text(json.dumps({"kyiv_date": last}))
    out, summary = work / "out", work / "summary"
    out.write_text("")
    subprocess.run(["bash", "-e", "-c", script], check=True, cwd=work,
                   env={"GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary), "PATH": "/usr/bin:/bin"})
    text = out.read_text()
    return "run=true" in text, text


def test_one_scheduled_run_per_kyiv_date(tmp_path):
    assert _run_guard("schedule", "2026-11-05", "2026-11-04", tmp_path)[0]          # a new day runs
    assert _run_guard("schedule", "2026-11-05", None, tmp_path)[0]                  # first run ever
    assert not _run_guard("schedule", "2026-10-25", "2026-10-25", tmp_path)[0]      # repeated 03:17 (autumn)
    assert _run_guard("workflow_dispatch", "2026-11-05", "2026-11-05", tmp_path)[0]  # manual runs always run
    assert "kyiv_date=2026-11-05" in _run_guard("schedule", "2026-11-05", "2026-11-04", tmp_path)[1]


def test_kyiv_0317_exists_or_moves_forward_on_every_day_of_the_year():
    """03:17 Kyiv is a real time on every day except the spring switch, when GitHub moves the run to the next
    valid time (04:00); the guard then still sees the right Kyiv date. Autumn: 03:17 happens twice."""
    kyiv = ZoneInfo("Europe/Kyiv")
    day, missing, repeated = datetime(2026, 1, 1), [], []
    for _ in range(730):
        local = day.replace(hour=3, minute=17, tzinfo=kyiv)
        back = local.astimezone(timezone.utc).astimezone(kyiv)
        if back.hour != 3:
            missing.append(day.date())
        elif local.replace(fold=1).utcoffset() != local.replace(fold=0).utcoffset():
            repeated.append(day.date())
        day += timedelta(days=1)
    assert [str(d) for d in missing] == ["2026-03-29", "2027-03-28"]
    assert [str(d) for d in repeated] == ["2026-10-25", "2027-10-31"]


def test_the_daily_run_records_its_kyiv_date_with_the_data():
    data, _ = _wf()
    steps = data["jobs"]["update"]["steps"]
    names = [s.get("name", s.get("uses", "")) for s in steps]
    assert names.index("actions/checkout@v4") < names.index("Once a day")     # the guard reads the newest main
    commit = next(s for s in steps if s.get("name") == "Commit updated data")["run"]
    assert "local_data/daily_run.json" in commit and commit.index("daily_run.json") < commit.index("git add local_data")


def test_history_mode_downloads_then_fills_recorded_prices_only_by_hand():
    data, _ = _wf()
    steps = data["jobs"]["update"]["steps"]
    names = [s.get("name", "") for s in steps]
    download, keep, fill = (steps[names.index(n)] for n in (
        "Download the price archive", "Keep the downloaded archive", "Fill recorded prices on the newest data and commit"))
    assert download["if"] == fill["if"] == "inputs.mode == 'history'"     # never on the daily schedule
    assert "always()" in keep["if"]                                        # the archive survives a failure
    assert "oddspapi_history.py" in download["run"] and 'merge_history_archive.py" save' in download["run"]
    run = fill["run"]
    # newest main first, then the archive, then the prices and the backtest
    assert (run.index("git reset -q --hard origin/main") < run.index('merge_history_archive.py" merge')
            < run.index("restore_recorded_odds.py --go") < run.index("regenerate_predictions.py"))
    assert next(s for s in steps if s.get("name") == "Run pipeline")["if"].endswith("inputs.mode != 'history'")
    assert next(s for s in steps if s.get("name") == "Commit updated data")["if"].endswith("inputs.mode != 'history'")
