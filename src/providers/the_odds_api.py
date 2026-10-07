"""The Odds API (v4): additional real bookmaker odds and fresh scores.

Endpoints used (query parameter ``apiKey``):

* ``GET /sports/{sport}/odds?regions=uk&markets=h2h,totals,spreads&oddsFormat=decimal&bookmakers=...``
  — cost = number of markets × regions (bookmakers count as one region);
* ``GET /sports/{sport}/scores?daysFrom=3`` — completed matches, cost 2.

Featured soccer markets map to: h2h -> 1X2, totals -> total goals,
spreads -> handicap (half lines only; whole / quarter lines are Asian).
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import config
from ..leagues import by_odds_api_key
from ..markets.markets import handicap_market, is_half_line
from .cache import ApiCache, fetch_json

log = logging.getLogger(__name__)

PROVIDER = "the_odds_api"
UK = ZoneInfo("Europe/London")
MARKETS = ("h2h", "totals", "spreads")


class TheOddsApi:
    def __init__(self, key: str | None = None, cache: ApiCache | None = None):
        self.key = key or config.api_key("ODDS_API_KEY")
        self.cache = cache or ApiCache(PROVIDER)
        self.credits_used = 0

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, params: dict, cost: int, max_age: float):
        if self.credits_used + cost > config.ODDS_API_MAX_CREDITS_PER_RUN:
            log.warning("The Odds API credit cap for this run reached; skipping %s", path)
            return None
        cached = self.cache.get(f"{config.ODDS_API_URL}/{path}", params, max_age)
        if cached is not None:
            return cached
        self.credits_used += cost
        return fetch_json(self.cache, f"{config.ODDS_API_URL}/{path}", {**params, "apiKey": self.key}, {},
                          max_age, cost_header="x-requests-last", remaining_header="x-requests-remaining")

    def odds(self, sport_key: str, max_age: float = 3 * 3600) -> list[dict]:
        params = {"markets": ",".join(MARKETS), "oddsFormat": "decimal",
                  "bookmakers": ",".join(config.ODDS_API_BOOKMAKERS)}
        return self._get(f"sports/{sport_key}/odds", params, len(MARKETS), max_age) or []

    def scores(self, sport_key: str, days_from: int = 3, max_age: float = 1800) -> list[dict]:
        return self._get(f"sports/{sport_key}/scores", {"daysFrom": days_from}, 2, max_age) or []


def _kickoff(event: dict) -> datetime:
    return datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).astimezone(UK)


def parse_event(event: dict, sport_key: str) -> dict | None:
    league = by_odds_api_key().get(sport_key)
    if league is None:
        return None
    ko = _kickoff(event)
    return {"provider_fixture_id": event["id"], "league": league.code, "date": ko.date().isoformat(),
            "time": ko.strftime("%H:%M"), "home_name": event["home_team"], "away_name": event["away_team"]}


def parse_event_odds(event: dict) -> list[tuple]:
    """[(market, line, selection, odds, bookmaker)] — per market, the first bookmaker by priority."""
    home, away = event["home_team"], event["away_team"]
    books = {b["key"]: b for b in event.get("bookmakers", [])}
    order = [k for k in config.ODDS_API_BOOKMAKERS if k in books] + [k for k in books
                                                                     if k not in config.ODDS_API_BOOKMAKERS]
    rows = []
    for market_key in MARKETS:
        for book_key in order:
            market = next((m for m in books[book_key].get("markets", []) if m["key"] == market_key), None)
            if market is None:
                continue
            parsed = []
            for o in market.get("outcomes", []):
                price, name, point = o.get("price"), o.get("name"), o.get("point")
                if market_key == "h2h":
                    sel = "H" if name == home else ("A" if name == away else ("D" if name == "Draw" else None))
                    if sel:
                        parsed.append(("1X2", None, sel, price))
                elif market_key == "totals" and point is not None and float(point) in config.TOTAL_GOAL_LINES:
                    parsed.append((f"TOTAL_{float(point)}", float(point), name.upper(), price))
                elif market_key == "spreads" and point is not None:
                    line = float(point) if name == home else -float(point)
                    if is_half_line(line) and line in config.HANDICAP_LINES:
                        parsed.append((handicap_market(line), line, "H" if name == home else "A", price))
            if parsed:
                title = books[book_key].get("title", book_key)
                rows += [(m, ln, s, float(p), title) for m, ln, s, p in parsed if p and float(p) > 1.0]
                break
    return rows


def parse_score(event: dict) -> tuple[int, int] | None:
    if not event.get("completed") or not event.get("scores"):
        return None
    scores = {s["name"]: s["score"] for s in event["scores"]}
    try:
        return int(scores[event["home_team"]]), int(scores[event["away_team"]])
    except (KeyError, ValueError, TypeError):
        return None
