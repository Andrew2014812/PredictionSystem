"""Download OddsPapi historical (pre-match) prices of played matches and report the coverage.

    python scripts/oddspapi_history.py                          # dry run for season 2026/27: estimate only
    python scripts/oddspapi_history.py --go                     # download season 2026/27 (resumable)
    python scripts/oddspapi_history.py --since 2026-01-01 --until 2026-06-30 --go
    python scripts/oddspapi_history.py --go --leagues E0 SP1 --max-matches 200
    python scripts/oddspapi_history.py --report                 # coverage report only (no request)

Cost: one billable /fixtures request per league (stored in local_data/api/oddspapi/history_progress.json,
so a resumed run does not pay again), then one FREE /historical-odds call per match (5-second cooldown,
~600 matches an hour). Matches already answered are skipped. A provider fixture is used only when it maps
to a PLAYED match of FootPredict's data (same league, UK date and both teams); the stored price of a
selection is the last snapshot strictly before kick-off — never a later one.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.cleaning import make_match_id  # noqa: E402
from src.data.dataset import build_matches  # noqa: E402
from src.leagues import enabled_leagues, get_league  # noqa: E402
from src.markets.markets import market_group  # noqa: E402
from src.providers.collect import (TeamResolver, _oddspapi_records, _save_odds, load_api_odds,  # noqa: E402
                                   oddspapi_outcomes, oddspapi_tournaments)
from src.providers.oddspapi import OddsPapi, fixture_row, parse_odds  # noqa: E402
from src.providers.teams_registry import TeamRegistry  # noqa: E402

UK = ZoneInfo("Europe/London")
PROGRESS = config.API_DIR / "oddspapi" / "history_progress.json"
COVERAGE = config.API_DIR / "oddspapi" / "history_coverage.json"
GROUPS = ("1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "HANDICAP", "CORNERS", "EXACT_SCORE")


def period(args) -> tuple[datetime, datetime]:
    if args.since:
        start = datetime.fromisoformat(args.since)
    else:
        start = datetime(config.current_season(), 8, 1)                # season 2026/27 by default
    end = datetime.fromisoformat(args.until) if args.until else datetime.now() - timedelta(days=1)
    return start.replace(tzinfo=timezone.utc), end.replace(hour=23, minute=59, tzinfo=timezone.utc)


def our_played(start, end, codes) -> pd.DataFrame:
    m = build_matches()
    return m.loc[m["played"].astype(bool) & (m["date"] >= pd.Timestamp(start.date()))
                 & (m["date"] <= pd.Timestamp(end.date())) & m["league"].isin(codes)]


def load_progress() -> dict:
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {"fixtures": {}}


def save_progress(data: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(data, indent=1), encoding="utf-8")


def coverage(played: pd.DataFrame, progress: dict, start, end) -> dict:
    """Per league: our played matches, OddsPapi fixtures, mapped matches, matches with prices per market."""
    odds = load_api_odds()
    odds = odds.loc[(odds["provider"] == "oddspapi") & odds["match_id"].isin(played["match_id"])] \
        if not odds.empty else odds
    groups = odds.assign(group=odds["market"].map(market_group)).groupby("match_id")["group"].agg(set) \
        if not odds.empty else pd.Series(dtype=object)
    rows = []
    for code, g in played.groupby("league"):
        fx = progress["fixtures"].get(code, {})
        mapped = {f["match_id"] for f in fx.get("items", []) if f.get("match_id")}
        priced = [groups.get(mid, set()) for mid in g["match_id"]]
        rows.append({"league": code, "matches": len(g), "oddspapi_fixtures": len(fx.get("items", [])),
                     "mapped": len(mapped & set(g["match_id"])), "with_odds": sum(1 for p in priced if p),
                     **{grp: sum(1 for p in priced if grp in p) for grp in GROUPS}})
    df = pd.DataFrame(rows)
    total = df.drop(columns="league").sum().to_dict() if not df.empty else {}
    return {"period": [str(start.date()), str(end.date())], "built_at": datetime.now().isoformat(timespec="seconds"),
            "total": total, "leagues": rows}


def print_coverage(report: dict) -> None:
    df = pd.DataFrame(report["leagues"])
    if df.empty:
        print("no matches")
        return
    df.loc[len(df)] = {"league": "TOTAL", **report["total"]}
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None, help="default: 1 August of the current season")
    parser.add_argument("--until", default=None, help="default: yesterday")
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--go", action="store_true", help="really download (otherwise only estimate)")
    parser.add_argument("--report", action="store_true", help="only print / save the coverage report")
    args = parser.parse_args()
    start, end = period(args)
    leagues = [lg for lg in enabled_leagues() if not args.leagues or lg.code in args.leagues]
    played = our_played(start, end, [lg.code for lg in leagues])
    progress = load_progress()
    if args.report:
        report = coverage(played, progress, start, end)
        COVERAGE.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print_coverage(report)
        return

    odp = OddsPapi()
    if not odp.available:
        sys.exit("ODDSPAPI_KEY is not set (.env locally, GitHub Secrets for Actions)")
    now = datetime.now(UK).replace(tzinfo=None)
    account = odp.account(now) or {}
    asked = {k.split("|", 1)[1] for k in odp.ledger.data.get("requests", {}) if k.startswith("historical|")}
    paid = sum(1 for lg in leagues if lg.code not in progress["fixtures"])
    todo = len(played) if not progress["fixtures"] else len(played) - len(asked)
    print(f"period {start.date()} – {end.date()}; leagues {len(leagues)}; our played matches {len(played)}")
    print(f"billable /fixtures requests: {paid} (+ market map / tournaments once if not stored); "
          f"free /historical-odds requests: about {max(todo, 0)} (~{max(todo, 0) * 5.6 / 3600:.1f} h)")
    print(f"OddsPapi account: limit {account.get('request_limit', '?')}, used {account.get('request_count', '?')}")
    if not args.go:
        print("dry run: nothing downloaded (add --go)")
        return

    resolver, registry = TeamResolver(build_matches()), TeamRegistry()
    tours = oddspapi_tournaments(odp, now, config.ODDSPAPI_RUN_BUDGET)
    omap = oddspapi_outcomes(odp, now, config.ODDSPAPI_RUN_BUDGET)
    ours = set(played["match_id"])
    done = priced = 0
    started = time.time()
    for lg in leagues:
        if odp.quota_exhausted or (args.max_matches is not None and done >= args.max_matches):
            break
        tid = tours.get(lg.code)
        if not tid:
            print(f"{lg.code}: no OddsPapi tournament id")
            continue
        entry = progress["fixtures"].get(lg.code)
        if entry is None or entry.get("period") != [str(start.date()), str(end.date())]:
            fixtures = odp.fixtures(now, config.ODDSPAPI_RUN_BUDGET + len(leagues), start, end, tournament_id=tid)
            items = []
            country = get_league(lg.code).country
            for fx in fixtures:
                row = fixture_row(fx, lg.code)
                if row is None:
                    continue
                home = registry.resolve("oddspapi", lg.code, country, row["home_name"], row["home_id"],
                                        resolver.known.get(lg.code))
                away = registry.resolve("oddspapi", lg.code, country, row["away_name"], row["away_id"],
                                        resolver.known.get(lg.code))
                mid = make_match_id(lg.code, pd.Timestamp(row["date"]), home, away) if home and away else None
                items.append({"fixture_id": row["provider_fixture_id"], "kickoff_utc": row["kickoff_utc"].isoformat(),
                              "home": row["home_name"], "away": row["away_name"],
                              "match_id": mid if mid in ours else None})
            entry = progress["fixtures"][lg.code] = {"period": [str(start.date()), str(end.date())], "items": items}
            save_progress(progress)
            registry.save()
        records = []
        for item in entry["items"]:
            if args.max_matches is not None and done >= args.max_matches:
                break
            if not item["match_id"] or odp.ledger.requested_within(f"historical|{item['fixture_id']}", 24 * 365, now):
                continue
            body = odp.historical_odds(item["fixture_id"], now)
            if odp.quota_exhausted:
                break
            done += 1
            kickoff = datetime.fromisoformat(item["kickoff_utc"])
            rows = parse_odds(body or {}, omap, kickoff)
            if rows:
                priced += 1
                records += _oddspapi_records(item["match_id"], item["fixture_id"], rows, now, kickoff)
            if done % 50 == 0:
                _save_odds(records)
                records = []
                odp.ledger.save()
                print(f"  … {done} matches, {priced} priced, {(time.time() - started) / 60:.0f} min")
        saved = _save_odds(records)
        odp.ledger.save()
        mapped = sum(1 for i in entry["items"] if i["match_id"])
        print(f"{lg.code}: {len(entry['items'])} fixtures, {mapped} mapped to played matches, saved {saved} prices")
    report = coverage(played, load_progress(), start, end)
    COVERAGE.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print_coverage(report)
    print(f"historical requests: {done}, matches with prices: {priced}, rate-limit waits: {odp.rate_limited}; "
          f"unmatched team names: {len(registry.unmatched)}"
          + ("; MONTHLY LIMIT REACHED" if odp.quota_exhausted else ""))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    main()
