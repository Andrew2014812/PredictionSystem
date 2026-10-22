"""Build the club-badge URL map ``local_data/teams/logos.json`` (run once, then rarely).

Source: API-Football team data (``/teams?league=<id>&season=<year>``; the
free plan allows seasons 2022–2024, so recently promoted clubs can be added
by name with ``--by-name``). Team names are matched to Football-Data names with
the same alias rules as fixtures (no fuzzy guessing; ambiguous names are
skipped). Only the URL map is stored and committed — never the images. The
browser loads each badge from the provider's media server (those calls do not
use the API quota) and the interface falls back to a neutral icon.

Badges are the clubs' trademarks; they are shown unmodified, small, with credit.

    python scripts/build_team_logos.py                    # league lists (21 requests, cached)
    python scripts/build_team_logos.py --by-name 30       # + up to 30 lookups by name
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from src import config  # noqa: E402
from src.features.store import load_features  # noqa: E402
from src.leagues import enabled_leagues  # noqa: E402
from src.providers.aliases import resolve  # noqa: E402
from src.providers.cache import ApiCache  # noqa: E402

OUT = config.DATA_DIR / "teams" / "logos.json"
SEASON = 2024                     # newest season the free plan serves
CACHE = ApiCache("api_football")


def team_key(country: str, team: str) -> str:
    return f"{country}|{team}"


def call(path: str, params: dict, key: str) -> list:
    url = f"{config.API_FOOTBALL_URL}/{path}"
    cached = CACHE.get(url, params, 90 * 24 * 3600)
    if cached is not None:
        return cached.get("response", [])
    time.sleep(7)                    # free plan: 10 requests / minute
    r = requests.get(url, params=params, headers={"x-apisports-key": key}, timeout=config.HTTP_TIMEOUT)
    body = r.json()
    if body.get("errors"):
        print("  API error:", body["errors"])
        if "requests" in str(body["errors"]).lower():
            sys.exit("daily request limit reached — run again tomorrow (results so far are cached)")
        return []
    CACHE.put(url, params, body)
    return body.get("response", [])


def search_terms(team: str) -> list[str]:
    """Full alias names (longest first), then the stored name, then its longest word (API minimum 3 letters)."""
    from src.providers.aliases import TEAM_ALIASES
    aliases = sorted((a for a, t in TEAM_ALIASES.items() if t == team), key=len, reverse=True)
    words = sorted((w for w in team.replace("'", " ").split() if len(w) >= 4), key=len, reverse=True)
    terms = [*aliases[:1], team, *words[:1]]
    return [t for i, t in enumerate(terms) if len(t) >= 3 and t.lower() not in [x.lower() for x in terms[:i]]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-name", type=int, default=0, help="max lookups by team name for unmatched clubs")
    args = parser.parse_args()
    key = config.api_key("API_FOOTBALL_KEY")
    if not key:
        sys.exit("API_FOOTBALL_KEY is not set")
    features = load_features()
    recent = features.loc[features["season"] >= config.current_season() - 1]
    logos = json.loads(OUT.read_text()) if OUT.exists() else {}

    missing: list[tuple] = []
    for lg in enabled_leagues():
        known = set(recent.loc[recent["league"] == lg.code, "home_team"]) | \
            set(recent.loc[recent["league"] == lg.code, "away_team"])
        todo = {t for t in known if team_key(lg.country, t) not in logos}
        if not todo or not lg.api_football_id:
            continue
        for item in call("teams", {"league": lg.api_football_id, "season": SEASON}, key):
            name = resolve(item["team"]["name"], known)
            if name in todo and item["team"].get("logo"):
                logos[team_key(lg.country, name)] = {"url": item["team"]["logo"], "source": "api-football",
                                                     "provider_id": item["team"]["id"]}
                todo.discard(name)
        missing += [(lg, t) for t in sorted(todo)]
    print(f"matched from league lists: {len(logos)}; still missing: {len(missing)}")

    for lg, team in missing[:args.by_name]:
        # the API does not accept "country" together with "search": filter the country in the answer.
        # Football-Data abbreviates ("Bristol Rvs"), so the full alias name is searched first.
        for term in search_terms(team):
            hit = None
            for item in call("teams", {"search": term[:30]}, key):
                if item["team"].get("country") != lg.country or not item["team"].get("logo"):
                    continue
                if resolve(item["team"]["name"], {team}) == team:
                    hit = item
                    break
            if hit:
                logos[team_key(lg.country, team)] = {"url": hit["team"]["logo"], "source": "api-football",
                                                     "provider_id": hit["team"]["id"]}
                break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(sorted(logos.items())), indent=1, ensure_ascii=False), encoding="utf-8")
    left = [f"{lg.code}: {t}" for lg, t in missing if team_key(lg.country, t) not in logos]
    print(f"logos: {len(logos)}; without a logo: {len(left)}")
    for line in left:
        print("  ", line)


if __name__ == "__main__":
    main()
