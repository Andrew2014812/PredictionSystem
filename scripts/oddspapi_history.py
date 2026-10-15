"""Download OddsPapi historical prices of played 2026 matches (pre-match closing prices).

    python scripts/oddspapi_history.py --since 2026-01-01                 # dry run: estimate only
    python scripts/oddspapi_history.py --since 2026-01-01 --go --leagues E0 SP1
    python scripts/oddspapi_history.py --since 2026-01-01 --go --max-matches 200

Cost: one billable /fixtures request per league (per run, cached), then one FREE /historical-odds
call per match (5-second cooldown, so ~700 matches an hour). Matches already downloaded are skipped
(ledger key "historical|<fixture id>"), so the script can be stopped and resumed. The stored price
of a selection is the last snapshot before kick-off — never a later one.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.dataset import build_matches  # noqa: E402
from src.leagues import enabled_leagues, get_league  # noqa: E402
from src.providers.collect import (TeamResolver, _canonical, _oddspapi_records, _save_odds,  # noqa: E402
                                   oddspapi_tournaments)
from src.providers.oddspapi import OddsPapi, fixture_row, outcome_map, parse_odds  # noqa: E402
from src.providers.teams_registry import TeamRegistry  # noqa: E402

UK = ZoneInfo("Europe/London")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-01-01")
    parser.add_argument("--until", default=None, help="default: yesterday")
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--go", action="store_true", help="really download (otherwise only estimate)")
    args = parser.parse_args()
    odp = OddsPapi()
    if not odp.available:
        sys.exit("ODDSPAPI_KEY is not set (.env locally, GitHub Secrets for Actions)")
    now = datetime.now(UK).replace(tzinfo=None)
    start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.until) if args.until else datetime.now(timezone.utc) - timedelta(days=1)
           ).replace(tzinfo=timezone.utc)
    leagues = [lg for lg in enabled_leagues() if not args.leagues or lg.code in args.leagues]
    print(f"leagues: {len(leagues)}; fixtures requests (billable): {len(leagues)}; account: {odp.account(now)}")
    if not args.go:
        played = build_matches()
        played = played.loc[played["played"].astype(bool) & (played["date"] >= pd.Timestamp(args.since))
                            & played["league"].isin([lg.code for lg in leagues])]
        print(f"played matches since {args.since}: {len(played)} -> about {len(played) * 5 / 3600:.1f} h of free "
              f"historical requests; nothing downloaded (add --go)")
        return

    resolver = TeamResolver(build_matches())
    registry = TeamRegistry()
    tours = oddspapi_tournaments(odp, now, config.ODDSPAPI_RUN_BUDGET)
    omap = outcome_map(odp.markets(now, config.ODDSPAPI_RUN_BUDGET))
    done = priced = 0
    for lg in leagues:
        tid = tours.get(lg.code)
        if not tid:
            print(f"{lg.code}: no OddsPapi tournament id")
            continue
        fixtures = odp.fixtures(now, len(leagues) + config.ODDSPAPI_MONTHLY_RESERVE, start, end, tournament_id=tid)
        records = []
        for fx in fixtures:
            if args.max_matches is not None and done >= args.max_matches:
                break
            row = fixture_row(fx, lg.code)
            if row is None or row["kickoff_utc"] >= datetime.now(timezone.utc):
                continue
            country = get_league(lg.code).country
            home = registry.resolve("oddspapi", lg.code, country, row["home_name"], row["home_id"],
                                    resolver.known.get(lg.code))
            away = registry.resolve("oddspapi", lg.code, country, row["away_name"], row["away_id"],
                                    resolver.known.get(lg.code))
            if not home or not away or odp.ledger.requested_within(f"historical|{row['provider_fixture_id']}",
                                                                   24 * 365, now):
                continue
            body = odp.historical_odds(row["provider_fixture_id"], now)
            done += 1
            rows = parse_odds(body or {}, omap, row["kickoff_utc"])
            if rows:
                priced += 1
                match = _canonical({**row, "played": True}, home, away)
                records += _oddspapi_records(match["match_id"], row["provider_fixture_id"], rows, now)
        print(f"{lg.code}: {len(fixtures)} fixtures, saved {_save_odds(records)} prices")
        odp.ledger.save()
        registry.save()
    print(f"historical requests: {done}, matches with prices: {priced}; unmatched teams: {len(registry.unmatched)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
