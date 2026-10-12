"""Report provider team names that do not map to a Football-Data team.

Reads the raw provider responses cached under local_data/api/ (no API calls)
and writes local_data/api/unmatched_teams.csv.

    python scripts/check_team_aliases.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.dataset import build_matches  # noqa: E402
from src.leagues import by_api_football_id, by_odds_api_key  # noqa: E402
from src.providers.collect import TeamResolver  # noqa: E402


def provider_names() -> set[tuple[str, str, str]]:
    names = set()
    af_leagues, toa_leagues = by_api_football_id(), by_odds_api_key()
    for path in (config.API_DIR / "api_football").glob("*.json"):
        body = json.loads(path.read_text(encoding="utf-8")).get("body") or {}
        for item in body.get("response", []) if isinstance(body, dict) else []:
            lg = af_leagues.get(item.get("league", {}).get("id"))
            if lg:
                for side in ("home", "away"):
                    names.add(("api_football", lg.code, item["teams"][side]["name"]))
    for path in (config.API_DIR / "the_odds_api").glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        body = entry.get("body")
        events = body if isinstance(body, list) else [body] if isinstance(body, dict) else []
        for event in events:
            lg = toa_leagues.get(event.get("sport_key"))
            if lg:
                names |= {("the_odds_api", lg.code, event["home_team"]), ("the_odds_api", lg.code, event["away_team"])}
    return names


if __name__ == "__main__":
    resolve = TeamResolver(build_matches())
    names = provider_names()
    for provider, league, name in sorted(names):
        resolve(provider, league, name)
    out = pd.DataFrame(sorted(resolve.unmatched), columns=["provider", "league", "name"])
    print(f"{len(names)} provider names checked, {len(out)} unmatched")
    print(out.to_string(index=False))
