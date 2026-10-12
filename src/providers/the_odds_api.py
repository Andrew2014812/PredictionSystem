"""The Odds API (v4): fixtures and real bookmaker odds.

Endpoints used (query parameter ``apiKey``), verified on the free plan:

* ``GET /sports/{sport}/events`` — upcoming fixtures, 0 credits;
* ``GET /sports/{sport}/odds?markets=h2h,totals&bookmakers=...`` — featured
  markets for a whole league, 1 credit per market;
* ``GET /sports/{sport}/events/{eventId}/odds?regions=uk&markets=...`` —
  additional markets for ONE match, 1 credit per market returned:
  ``btts``, ``double_chance``, ``alternate_totals`` (goal lines 0.5–5.5),
  ``alternate_totals_corners`` (7.5–11.5), ``correct_score``,
  ``alternate_spreads`` (handicap; only half lines are used).

Credits are tracked in :class:`~src.providers.ledger.CreditLedger`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .. import config
from ..leagues import by_odds_api_key
from ..markets.markets import handicap_market, is_half_line
from .cache import ApiCache
from .ledger import CreditLedger

log = logging.getLogger(__name__)

PROVIDER = "the_odds_api"
UK = ZoneInfo("Europe/London")
FEATURED_MARKETS = ("h2h", "totals", "spreads")
EXTENDED_MARKETS = ("btts", "double_chance", "alternate_totals", "alternate_totals_corners",
                    "correct_score", "alternate_spreads")
CORNER_LINES = tuple(x + 0.5 for x in range(5, 15))      # 5.5 … 14.5


class TheOddsApi:
    def __init__(self, key: str | None = None, cache: ApiCache | None = None, ledger: CreditLedger | None = None):
        self.key = key or config.api_key("ODDS_API_KEY")
        self.cache = cache or ApiCache(PROVIDER)
        self.ledger = ledger or CreditLedger()

    @property
    def available(self) -> bool:
        return bool(self.key)

    @property
    def credits_used(self) -> int:
        return self.ledger.spent_this_run

    def _request(self, path: str, params: dict) -> tuple[object | None, dict]:
        url = f"{config.ODDS_API_URL}/{path}"
        try:
            response = requests.get(url, params={**params, "apiKey": self.key}, timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("The Odds API request failed: %s", exc)
            return None, {}
        headers = {k: response.headers.get(k) for k in ("x-requests-last", "x-requests-used", "x-requests-remaining")}
        self.cache.log_request(url, float(headers["x-requests-last"] or 0), headers["x-requests-remaining"])
        if response.status_code != 200:
            log.warning("The Odds API HTTP %s for %s: %s", response.status_code, path, response.text[:200])
            return None, headers
        body = response.json()
        self.cache.put(url, params, body)
        return body, headers

    def events(self, sport_key: str) -> list[dict]:
        """Upcoming fixtures of a league (free)."""
        cached = self.cache.get(f"{config.ODDS_API_URL}/sports/{sport_key}/events", {}, 3 * 3600)
        if cached is not None:
            return cached
        body, _ = self._request(f"sports/{sport_key}/events", {})
        return body or []

    def _paid(self, key: str, path: str, params: dict, expected_cost: int, now: datetime, run_budget: int,
              count_items) -> object | None:
        if not self.ledger.can_spend(expected_cost, run_budget):
            log.info("The Odds API budget reached; skipping %s", key)
            return None
        body, headers = self._request(path, params)
        cost = int(float(headers.get("x-requests-last") or (expected_cost if body is not None else 0)))
        self.ledger.record(key, cost, count_items(body) if body is not None else 0, now,
                           headers.get("x-requests-used"), headers.get("x-requests-remaining"))
        return body

    def league_odds(self, sport_key: str, now: datetime, run_budget: int) -> list[dict]:
        markets = config.ODDS_API_MARKETS
        params = {"markets": ",".join(markets), "oddsFormat": "decimal",
                  "bookmakers": ",".join(config.ODDS_API_BOOKMAKERS)}
        body = self._paid(f"odds|{sport_key}", f"sports/{sport_key}/odds", params, len(markets), now, run_budget,
                          lambda b: sum(1 for e in b if e.get("bookmakers")))
        return body or []

    def event_odds(self, sport_key: str, event_id: str, now: datetime, run_budget: int) -> dict | None:
        markets = config.ODDS_API_EXTENDED_MARKETS
        params = {"regions": "uk", "markets": ",".join(markets), "oddsFormat": "decimal"}
        return self._paid(f"event|{event_id}", f"sports/{sport_key}/events/{event_id}/odds", params, len(markets),
                          now, run_budget, lambda b: len({m["key"] for bk in b.get("bookmakers", [])
                                                           for m in bk.get("markets", [])}))


def _kickoff(event: dict) -> datetime:
    return datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).astimezone(UK)


def parse_event(event: dict, sport_key: str) -> dict | None:
    league = by_odds_api_key().get(sport_key)
    if league is None:
        return None
    ko = _kickoff(event)
    return {"provider_fixture_id": event["id"], "league": league.code, "date": ko.date().isoformat(),
            "time": ko.strftime("%H:%M"), "home_name": event["home_team"], "away_name": event["away_team"]}


def _parse_market(key: str, outcomes: list[dict], home: str, away: str) -> list[tuple]:
    """(market, line, selection, price) for one market of one bookmaker."""
    out = []
    for o in outcomes:
        price, name, point = o.get("price"), str(o.get("name", "")), o.get("point")
        if not price or float(price) <= 1.0:
            continue
        if key == "h2h":
            sel = "H" if name == home else ("A" if name == away else ("D" if name == "Draw" else None))
            if sel:
                out.append(("1X2", None, sel, price))
        elif key in ("totals", "alternate_totals") and point is not None and float(point) in config.TOTAL_GOAL_LINES:
            out.append((f"TOTAL_{float(point)}", float(point), name.upper(), price))
        elif key == "alternate_totals_corners" and point is not None and float(point) in CORNER_LINES:
            out.append((f"CORNERS_{float(point)}", float(point), name.upper(), price))
        elif key == "btts" and name in ("Yes", "No"):
            out.append(("BTTS", None, name.upper(), price))
        elif key == "double_chance":
            sel = {f"{home} or Draw": "1X", f"{away} or Draw": "X2", f"{home} or {away}": "12",
                   f"Draw or {home}": "1X", f"Draw or {away}": "X2", f"{away} or {home}": "12"}.get(name)
            if sel:
                out.append(("DOUBLE_CHANCE", None, sel, price))
        elif key == "correct_score":
            parts = dict(p.rsplit(":", 1) for p in name.split("|") if ":" in p)
            try:
                hg, ag = int(parts[home]), int(parts[away])
            except (KeyError, ValueError):
                continue
            out.append(("EXACT_SCORE", None, f"{hg}-{ag}", price))
        elif key in ("spreads", "alternate_spreads") and point is not None and name in (home, away):
            # the point is the named team's handicap; the system's line is the home team's
            line = float(point) if name == home else -float(point)
            if is_half_line(line) and line in config.HANDICAP_LINES:
                out.append((handicap_market(line), line, "H" if name == home else "A", price))
    return out


def parse_event_odds(event: dict) -> list[tuple]:
    """[(market, line, selection, odds, bookmaker)] — for every selection, the price of
    the first bookmaker in ``config.ODDS_API_BOOKMAKERS`` order that quotes it
    (exchanges are skipped). Featured and alternate markets can quote the same
    line; the first price found wins, so each selection appears once."""
    home, away = event["home_team"], event["away_team"]
    books = {b["key"]: b for b in event.get("bookmakers", []) if b["key"] not in config.ODDS_API_EXCLUDED_BOOKMAKERS}
    order = [k for k in config.ODDS_API_BOOKMAKERS if k in books] + sorted(k for k in books
                                                                          if k not in config.ODDS_API_BOOKMAKERS)
    seen, rows = set(), []
    for book_key in order:
        title = books[book_key].get("title", book_key)
        for market in books[book_key].get("markets", []):
            for m, ln, sel, price in _parse_market(market["key"], market.get("outcomes", []), home, away):
                if (m, sel) not in seen:
                    seen.add((m, sel))
                    rows.append((m, ln, sel, float(price), title))
    return rows


def parse_score(event: dict) -> tuple[int, int] | None:
    if not event.get("completed") or not event.get("scores"):
        return None
    scores = {s["name"]: s["score"] for s in event["scores"]}
    try:
        return int(scores[event["home_team"]]), int(scores[event["away_team"]])
    except (KeyError, ValueError, TypeError):
        return None
