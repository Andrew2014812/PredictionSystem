"""Show what OddsPapi returns for one match next to FootPredict's mapping (diagnostics, free endpoint).

    python scripts/oddspapi_inspect.py --fixture id1000001234567890
    python scripts/oddspapi_inspect.py --fixture <id> --market corner        # filter by market name
    python scripts/oddspapi_inspect.py --list "Bayern"                       # fixture ids of upcoming matches

Uses /historical-odds (free) and the stored market catalogue; prints every quote of every bookmaker with
the catalogue's market name, line and outcome name, and the FootPredict market / selection it maps to
(or "-" if it is not used), plus the quotes dropped as contradictory.
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.providers.collect import oddspapi_outcomes, oddspapi_tournaments  # noqa: E402
from src.providers.oddspapi import OddsPapi, _pick_quote, inconsistent_quotes  # noqa: E402

UK = ZoneInfo("Europe/London")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture")
    parser.add_argument("--market", default="")
    parser.add_argument("--list", default=None, help="team name to find upcoming fixture ids")
    args = parser.parse_args()
    odp = OddsPapi()
    if not odp.available:
        sys.exit("ODDSPAPI_KEY is not set")
    now = datetime.now(UK).replace(tzinfo=None)
    if args.list:
        tours = {v: k for k, v in oddspapi_tournaments(odp, now, config.ODDSPAPI_RUN_BUDGET).items()}
        start = datetime.now(UK)
        for fx in odp.fixtures(now, config.ODDSPAPI_RUN_BUDGET, start, start + timedelta(days=7)):
            names = f"{fx.get('participant1Name')} – {fx.get('participant2Name')}"
            if args.list.lower() in names.lower() and int(fx.get("tournamentId") or 0) in tours:
                print(fx.get("fixtureId"), fx.get("startTime"), tours[int(fx["tournamentId"])], names)
        return
    catalogue = {str(m["marketId"]): m for m in odp.markets(now, config.ODDSPAPI_RUN_BUDGET)}
    names = {str(o["outcomeId"]): o.get("outcomeName") for m in catalogue.values() for o in m.get("outcomes", [])}
    omap = oddspapi_outcomes(odp, now, config.ODDSPAPI_RUN_BUDGET)
    body = odp.historical_odds(args.fixture, now) or {}
    odp.ledger.save()
    for slug, book in (body.get("bookmakers") or body.get("bookmakerOdds") or {}).items():
        print(f"== {slug}")
        quotes = {}
        for mid, market in (book.get("markets") or {}).items():
            meta = catalogue.get(str(mid), {})
            if args.market.lower() not in str(meta.get("marketName", "")).lower():
                continue
            for oid, outcome in (market.get("outcomes") or {}).items():
                quote = _pick_quote((outcome.get("players") or {}).get("0"), None)
                mapped = omap.get(str(oid))
                if quote and mapped:
                    quotes[(mapped[0], mapped[2])] = (mapped[1], quote[0], quote[1])
                print(f"  {mid:>7} {str(meta.get('marketName')):<32} line={meta.get('handicap')!s:<6} "
                      f"{str(names.get(str(oid))):<14} price={quote[0] if quote else '-':<6} "
                      f"-> {mapped[0] + ' ' + mapped[2] if mapped else '-'}")
        bad = inconsistent_quotes(quotes)
        if bad:
            print("  contradictory, not used:", sorted(bad))


if __name__ == "__main__":
    main()
