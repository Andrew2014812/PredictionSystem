"""API-Football (api-sports.io, v3): fresh fixtures, results and Bet365 odds.

Endpoints used (header ``x-apisports-key``):

* ``GET /fixtures?date=YYYY-MM-DD&timezone=Europe/London`` — every fixture of
  a day (one request for all leagues), with status and final score;
* ``GET /odds?league=<id>&season=<year>&bookmaker=<id>&page=<n>`` — pre-match
  odds of the league's upcoming fixtures (paginated).

Only leagues of the registry are kept. Team names are mapped to
Football-Data names through ``aliases.resolve``; unknown names are skipped
and reported, never guessed.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .. import config
from ..leagues import by_api_football_id
from ..markets.markets import handicap_market, is_half_line
from .cache import ApiCache, fetch_json

log = logging.getLogger(__name__)

PROVIDER = "api_football"
UK = ZoneInfo("Europe/London")
FINISHED = {"FT", "AET", "PEN"}
NOT_PLAYED = {"PST", "CANC", "ABD", "AWD", "WO"}


class ApiFootball:
    def __init__(self, key: str | None = None, cache: ApiCache | None = None):
        self.key = key or config.api_key("API_FOOTBALL_KEY")
        self.cache = cache or ApiCache(PROVIDER)

    @property
    def available(self) -> bool:
        return bool(self.key)

    def quota_left(self) -> int:
        return config.API_FOOTBALL_DAILY_LIMIT - self.cache.requests_today()

    def _get(self, path: str, params: dict, max_age: float):
        if self.quota_left() <= 0:
            log.warning("API-Football daily quota used up; skipping %s", path)
            return None
        body = fetch_json(self.cache, f"{config.API_FOOTBALL_URL}/{path}", params,
                          {"x-apisports-key": self.key}, max_age,
                          remaining_header="x-ratelimit-requests-remaining")
        if body and body.get("errors"):
            log.warning("API-Football error for %s: %s", path, body["errors"])
            return None
        return body

    def fixtures(self, day: date, max_age: float = 1800) -> list[dict]:
        body = self._get("fixtures", {"date": day.isoformat(), "timezone": "Europe/London"}, max_age)
        return (body or {}).get("response", [])

    def odds(self, league_id: int, season: int, max_age: float = 3 * 3600) -> list[dict]:
        out, page, total = [], 1, 1
        while page <= total:
            body = self._get("odds", {"league": league_id, "season": season,
                                      "bookmaker": config.API_FOOTBALL_BOOKMAKER, "page": page}, max_age)
            if not body:
                break
            out += body.get("response", [])
            total = int(body.get("paging", {}).get("total", 1) or 1)
            page += 1
        return out


# ---------------------------------------------------------------------------
# parsing (pure functions, unit tested)
# ---------------------------------------------------------------------------
def parse_fixture(item: dict) -> dict | None:
    """Fixture / result in the system's terms (team names still provider names)."""
    league = by_api_football_id().get(item.get("league", {}).get("id"))
    if league is None:
        return None
    fx = item["fixture"]
    kickoff = datetime.fromisoformat(fx["date"].replace("Z", "+00:00")).astimezone(UK)
    status = fx.get("status", {}).get("short", "")
    goals = item.get("goals", {})
    finished = status in FINISHED
    return {
        "provider_fixture_id": str(fx["id"]), "league": league.code, "date": kickoff.date().isoformat(),
        "time": kickoff.strftime("%H:%M"), "home_name": item["teams"]["home"]["name"],
        "away_name": item["teams"]["away"]["name"], "status": status,
        "cancelled": status in NOT_PLAYED, "played": finished,
        # full-time result after 90 minutes (extra time does not count in league football)
        "home_goals": (item.get("score", {}).get("fulltime") or {}).get("home", goals.get("home"))
        if finished else None,
        "away_goals": (item.get("score", {}).get("fulltime") or {}).get("away", goals.get("away"))
        if finished else None,
    }


_NUM = r"([+-]?\d+(?:\.\d+)?)"


def parse_bet(name: str, values: list[dict]) -> list[tuple[str, float | None, str, float]]:
    """(market, line, selection, odds) for one API-Football bet type."""
    out = []

    def add(market, line, selection, odd):
        try:
            price = float(odd)
        except (TypeError, ValueError):
            return
        if price > 1.0:
            out.append((market, line, selection, price))

    for v in values:
        value, odd = str(v.get("value", "")).strip(), v.get("odd")
        if name == "Match Winner":
            sel = {"Home": "H", "Draw": "D", "Away": "A"}.get(value)
            if sel:
                add("1X2", None, sel, odd)
        elif name == "Goals Over/Under":
            m = re.fullmatch(r"(Over|Under) " + _NUM, value)
            if m and float(m[2]) in config.TOTAL_GOAL_LINES:
                add(f"TOTAL_{float(m[2])}", float(m[2]), m[1].upper(), odd)
        elif name == "Both Teams Score":
            if value in ("Yes", "No"):
                add("BTTS", None, value.upper(), odd)
        elif name == "Double Chance":
            sel = {"Home/Draw": "1X", "Draw/Away": "X2", "Home/Away": "12"}.get(value)
            if sel:
                add("DOUBLE_CHANCE", None, sel, odd)
        elif name == "Exact Score":
            m = re.fullmatch(r"(\d+):(\d+)", value)
            if m:
                add("EXACT_SCORE", None, f"{int(m[1])}-{int(m[2])}", odd)
        elif name in ("Handicap Result", "Asian Handicap"):
            # "Home -1" / "Draw -1" / "Away +1": the number is the handicap of
            # the named side; the market line is the home team's handicap.
            m = re.fullmatch(r"(Home|Draw|Away) " + _NUM, value)
            if not m:
                continue
            side, number = m[1], float(m[2])
            line = -number if side == "Away" else number
            half = is_half_line(line)
            if name == "Asian Handicap" and not half:
                continue          # whole / quarter Asian lines settle differently
            if name == "Handicap Result" and half:
                continue
            if line not in config.HANDICAP_LINES:
                continue
            add(handicap_market(line), line, {"Home": "H", "Draw": "D", "Away": "A"}[side], odd)
        elif name == "Corners Over Under":
            m = re.fullmatch(r"(Over|Under) " + _NUM, value)
            if m:
                add(f"CORNERS_{float(m[2])}", float(m[2]), m[1].upper(), odd)
    return out


def parse_odds(item: dict) -> tuple[str, list[tuple]]:
    """(provider_fixture_id, [(market, line, selection, odds, bookmaker), ...])."""
    fixture_id = str(item.get("fixture", {}).get("id"))
    rows = []
    for bookmaker in item.get("bookmakers", []):
        for bet in bookmaker.get("bets", []):
            rows += [(*r, bookmaker.get("name", "")) for r in parse_bet(bet.get("name", ""), bet.get("values", []))]
        if rows:
            break                 # one bookmaker per fixture (the requested one)
    return fixture_id, rows
