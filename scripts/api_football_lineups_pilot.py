"""Pilot: can API-Football (the plan FootPredict uses) give line-ups and player statistics of finished matches?

v6.3 Squad Strength study (reports/v6_3/Squad_Strength_Research.pdf): a player-strength feature needs a maintained
line-up source; the free plan serves fixtures only from yesterday to tomorrow, so a history can only be collected
day by day. This pilot asks for YESTERDAY's finished matches of the 21 leagues, by fixture id in batches of 20
(``GET /fixtures?ids=a-b-c``, the answer carries line-ups and player statistics), and reports what came back.

    API_FOOTBALL_KEY=... python scripts/api_football_lineups_pilot.py --out pilot_out [--day 2026-11-18]

Budget: at most ``--max-requests`` (default 6) of the 100 daily requests; the fixture ids come from the stored live
table (``local_data/results_live/fixtures.parquet``), so no request is spent on finding them. Nothing is written to
``local_data``: the raw answers and ``summary.json`` go to ``--out`` (the workflow keeps them as an artifact).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

BATCH = 20


def fixture_ids(live: pd.DataFrame, day: date) -> list[int]:
    """API-Football ids of the finished matches of ``day`` in the stored live table."""
    rows = live.loc[(live["provider"] == "api_football") & live["played"].astype(bool)
                    & (pd.to_datetime(live["date"]).dt.date == day)]
    ids = pd.to_numeric(rows["provider_fixture_id"], errors="coerce").dropna().astype(int)
    return sorted(set(ids))


def batches(ids: list[int], size: int = BATCH) -> list[list[int]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def summarise(items: list[dict]) -> dict:
    """Coverage of line-ups and player statistics in ``/fixtures?ids=`` answers."""
    out = {"fixtures": len(items), "with_lineups_both": 0, "with_formation_both": 0, "with_players_both": 0,
           "players": 0, "players_with_minutes": 0, "players_with_rating": 0, "players_with_position": 0,
           "start_xi_sizes": {}, "leagues": {}}
    for it in items:
        league = str((it.get("league") or {}).get("id"))
        out["leagues"][league] = out["leagues"].get(league, 0) + 1
        lineups = it.get("lineups") or []
        if len(lineups) == 2 and all(lu.get("startXI") for lu in lineups):
            out["with_lineups_both"] += 1
        if len(lineups) == 2 and all(lu.get("formation") for lu in lineups):
            out["with_formation_both"] += 1
        for lu in lineups:
            n = str(len(lu.get("startXI") or []))
            out["start_xi_sizes"][n] = out["start_xi_sizes"].get(n, 0) + 1
        teams = it.get("players") or []
        if len(teams) == 2 and all(t.get("players") for t in teams):
            out["with_players_both"] += 1
        for t in teams:
            for p in t.get("players") or []:
                stats = (p.get("statistics") or [{}])[0]
                games = stats.get("games") or {}
                out["players"] += 1
                out["players_with_minutes"] += games.get("minutes") is not None
                out["players_with_rating"] += games.get("rating") not in (None, "")
                out["players_with_position"] += bool(games.get("position"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--day", default="")
    ap.add_argument("--max-requests", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    day = date.fromisoformat(args.day) if args.day else \
        datetime.now(ZoneInfo("Europe/London")).date() - timedelta(days=1)
    live = pd.read_parquet(config.LIVE_DIR / "fixtures.parquet")
    ids = fixture_ids(live, day)
    report = {"day": str(day), "fixture_ids": len(ids), "requests": 0, "errors": []}
    from src.providers.api_football import ApiFootball
    api = ApiFootball()
    if not api.available:
        sys.exit("API_FOOTBALL_KEY is not set")
    items: list[dict] = []
    for chunk in batches(ids)[:max(0, args.max_requests)]:
        body = api._get("fixtures", {"ids": "-".join(map(str, chunk))}, max_age=0)
        report["requests"] += 1
        if body is None:
            report["errors"].append(f"no answer for {len(chunk)} fixtures")
            continue
        (out / f"fixtures_{chunk[0]}.json").write_text(json.dumps(body), encoding="utf-8")
        items += body.get("response") or []
    report["not_asked"] = max(0, len(ids) - BATCH * args.max_requests)
    report["coverage"] = summarise(items)
    (out / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
