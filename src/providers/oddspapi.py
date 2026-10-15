"""OddsPapi v4 (https://oddspapi.io/en/docs): real bookmaker odds of many football markets.

Endpoints (query parameter ``apiKey``):

* ``GET /account``                         free — request_limit / request_count;
* ``GET /markets``                         billable, fetched once and cached (the market catalogue);
* ``GET /tournaments?sportId=10``          billable, fetched once (league ids);
* ``GET /fixtures?sportId=10&from&to``     billable — all fixtures of the next days (< 10 days);
* ``GET /fixtures?tournamentId&from&to``   billable — one league (historical download);
* ``GET /odds?fixtureId&bookmakers``       billable — current prices of one match;
* ``GET /historical-odds?fixtureId&bookmakers`` FREE (max 3 bookmakers, 5 s cooldown) — every price
  snapshot with its ``createdAt``; the archive starts in January 2026.

Every billable request is recorded in a committed ledger (the CI cache does not persist), also when
it returned nothing, so the same match is not asked again within ODDSPAPI_REFRESH_HOURS, every run
respects a budget and a monthly reserve is never spent. A 4xx/5xx answer counts against the quota
(per the docs) and is recorded as such; "no market" is not an error.

Markets are identified by numeric ids; one id per line. The catalogue (``/markets``) gives the line
(``handicap``) and the outcome names, so the mapping to FootPredict's markets is built from it, not
hard-coded.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from .. import config
from ..markets.markets import european_handicap_market, handicap_market, is_half_line
from .cache import ApiCache
from .ledger import CreditLedger

log = logging.getLogger(__name__)

PROVIDER = "oddspapi"
UK = ZoneInfo("Europe/London")
COOLDOWN = {"odds": 0.5, "fixture": 0.5, "fixtures": 2.0, "settlements": 2.0, "historical-odds": 5.0}
FREE = {"account", "historical-odds"}
_NON_FULL_TIME = re.compile(r"half|1st|2nd|first|second|period|minute|interval|\b1h\b|\b2h\b", re.I)
_OTHER_STATS = re.compile(r"card|booking|shot|offside|foul|throw|goal ?kick|player|penalt|half ?time", re.I)


class OddsPapi:
    def __init__(self, key: str | None = None, cache: ApiCache | None = None, ledger: CreditLedger | None = None):
        self.key = key or config.api_key("ODDSPAPI_KEY")
        self.cache = cache or ApiCache(PROVIDER)
        self.ledger = ledger or CreditLedger(config.API_DIR / PROVIDER / "ledger.json", config.ODDSPAPI_MONTHLY_RESERVE)
        self._last_call: dict[str, float] = {}

    @property
    def available(self) -> bool:
        return bool(self.key)

    # -- transport ---------------------------------------------------------------------------------
    def _wait(self, endpoint: str) -> None:
        gap = COOLDOWN.get(endpoint, 1.0)
        last = self._last_call.get(endpoint)
        if last is not None and time.time() - last < gap:
            time.sleep(gap - (time.time() - last) + 0.05)
        self._last_call[endpoint] = time.time()

    def _get(self, endpoint: str, params: dict, ledger_key: str | None, now: datetime, run_budget: int,
             max_age: float | None = None):
        """GET with cache, cooldown and quota accounting. Returns (body or None, status)."""
        url = f"{config.ODDSPAPI_URL}/{endpoint}"
        if max_age is not None:
            cached = self.cache.get(url, params, max_age)
            if cached is not None:
                return cached, "cached"
        billable = endpoint not in FREE
        if billable and not self.ledger.can_spend(1, run_budget):
            log.info("OddsPapi budget reached; skipping %s %s", endpoint, params)
            return None, "budget"
        self._wait(endpoint)
        try:
            r = requests.get(url, params={**params, "apiKey": self.key}, timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:          # network failure: not counted by the API
            log.warning("OddsPapi %s failed: %s", endpoint, exc)
            return None, "network"
        status = r.status_code
        if status == 429 and "REQUEST_LIMIT" in r.text:
            self.ledger.data["remaining"] = 0             # quota exhausted (also blocks the free endpoints)
            return None, "quota"
        counted = billable and not (status in (401, 403) or status == 429)
        body = None
        if status == 200:
            body = r.json()
            self.cache.put(url, params, body)
        elif status == 404:
            body, status = None, "not_found"                # e.g. no odds for this match: an answer, not an error
        else:
            log.warning("OddsPapi HTTP %s for %s: %s", status, endpoint, r.text[:200])
        if counted:
            self.ledger.record(ledger_key or f"{endpoint}|{sorted(params.items())}", 1,
                               _count_items(body), now, remaining=self._remaining_after(1), status=status)
        return body, status

    def _remaining_after(self, cost: int):
        rem = self.ledger.remaining
        return None if rem is None else rem - cost

    # -- endpoints -------------------------------------------------------------------------------
    def account(self, now: datetime) -> dict | None:
        body, _ = self._get("account", {}, None, now, 0)
        if isinstance(body, dict) and "request_limit" in body:
            used = int(body.get("request_count") or 0)
            self.ledger.data.update(remaining=int(body["request_limit"]) - used, used=used,
                                    checked_at=now.isoformat(timespec="seconds"))
        return body

    def markets(self, now: datetime, run_budget: int) -> list[dict]:
        body, _ = self._get("markets", {}, "markets", now, run_budget, max_age=180 * 24 * 3600)
        return body if isinstance(body, list) else (body or {}).get("markets", []) if isinstance(body, dict) else []

    def tournaments(self, now: datetime, run_budget: int) -> list[dict]:
        body, _ = self._get("tournaments", {"sportId": config.ODDSPAPI_SPORT_ID}, "tournaments", now, run_budget,
                            max_age=60 * 24 * 3600)
        return body if isinstance(body, list) else []

    def fixtures(self, now: datetime, run_budget: int, start: datetime, end: datetime,
                 tournament_id: int | None = None) -> list[dict]:
        params = {"from": _iso(start), "to": _iso(end)}
        if tournament_id:
            params["tournamentId"] = tournament_id
        else:
            params["sportId"] = config.ODDSPAPI_SPORT_ID
        key = f"fixtures|{tournament_id or 'all'}|{params['from'][:10]}|{params['to'][:10]}"
        body, _ = self._get("fixtures", params, key, now, run_budget, max_age=6 * 3600)
        return body if isinstance(body, list) else []

    def odds(self, fixture_id: str, now: datetime, run_budget: int) -> dict | None:
        params = {"fixtureId": fixture_id, "bookmakers": ",".join(config.ODDSPAPI_BOOKMAKERS)}
        body, _ = self._get("odds", params, f"odds|{fixture_id}", now, run_budget)
        return body if isinstance(body, dict) else None

    def historical_odds(self, fixture_id: str, now: datetime) -> dict | None:
        params = {"fixtureId": fixture_id, "bookmakers": ",".join(config.ODDSPAPI_BOOKMAKERS[:3])}
        body, status = self._get("historical-odds", params, None, now, 0)
        self.ledger.data.setdefault("requests", {})[f"historical|{fixture_id}"] = {
            "at": now.isoformat(timespec="seconds"), "cost": 0, "items": _count_items(body), "status": str(status)}
        return body if isinstance(body, dict) else None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt.tzinfo else dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_items(body) -> int:
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict):
        books = body.get("bookmakerOdds") or body.get("bookmakers") or {}
        return sum(len(b.get("markets", {})) for b in books.values()) if isinstance(books, dict) else 0
    return 0


# -- market catalogue -> FootPredict markets -------------------------------------------------------
def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip().lower()


def _side(name: str) -> str | None:
    n = _norm(name)
    if n in ("1", "home", "team 1", "participant 1", "w1") or n.startswith("1 "):
        return "H"
    if n in ("x", "draw", "tie"):
        return "D"
    if n in ("2", "away", "team 2", "participant 2", "w2") or n.startswith("2 "):
        return "A"
    return None


def outcome_map(catalogue: list[dict]) -> dict[str, tuple[str, float | None, str]]:
    """outcomeId -> (market, line, selection) for every catalogue market FootPredict uses.

    Only full-time, non-player markets; goal totals on TOTAL_GOAL_LINES, corner totals on any
    half line, Asian handicap on HANDICAP_LINES (half and whole lines, quarter lines skipped),
    European handicap on whole lines, correct score, 1X2, double chance and BTTS."""
    out: dict[str, tuple[str, float | None, str]] = {}
    for m in catalogue:
        name = _norm(m.get("marketName"))
        mtype = _norm(m.get("marketType"))
        if m.get("playerProp") or _NON_FULL_TIME.search(name) or _NON_FULL_TIME.search(_norm(m.get("period"))):
            continue
        line = m.get("handicap")
        line = float(line) if line not in (None, "") else None
        outcomes = m.get("outcomes", [])
        corners = "corner" in name
        if not corners and _OTHER_STATS.search(name):
            continue
        is_team_total = re.search(r"\b(home|away|team ?1|team ?2)\b", name) and ("total" in name or "over" in name)

        def put(o, market, ln, sel):
            out[str(o["outcomeId"])] = (market, ln, sel)

        if corners:
            if line is not None and not is_team_total and is_half_line(line) and ("total" in name or "over" in name):
                for o in outcomes:
                    sel = _norm(o.get("outcomeName"))
                    if sel.startswith("over") or sel.startswith("under"):
                        put(o, f"CORNERS_{line}", line, "OVER" if sel.startswith("over") else "UNDER")
            continue
        if "double chance" in name or mtype == "doublechance":
            for o in outcomes:
                sel = re.sub(r"[^0-9x]", "", _norm(o.get("outcomeName")).replace("draw", "x"))
                sel = {"1x": "1X", "x1": "1X", "x2": "X2", "2x": "X2", "12": "12", "21": "12"}.get(sel)
                if sel:
                    put(o, "DOUBLE_CHANCE", None, sel)
        elif "both teams" in name or "btts" in name or mtype == "bothteamsscore":
            for o in outcomes:
                sel = _norm(o.get("outcomeName"))
                if sel in ("yes", "no"):
                    put(o, "BTTS", None, sel.upper())
        elif "correct score" in name or mtype == "correctscore":
            for o in outcomes:
                sc = re.fullmatch(r"(\d+)\s*[:\-]\s*(\d+)", _norm(o.get("outcomeName")))
                if sc:
                    put(o, "EXACT_SCORE", None, f"{int(sc[1])}-{int(sc[2])}")
        elif "asian" in name or mtype in ("spreads", "asianhandicap", "asian handicap"):
            if line is None or line not in config.HANDICAP_LINES:
                continue                                           # also drops quarter lines
            sides = [o for o in outcomes if _side(o.get("outcomeName")) in ("H", "A")] or outcomes[:2]
            for k, o in enumerate(sides[:2]):
                put(o, handicap_market(line), line, _side(o.get("outcomeName")) or "HA"[k])
        elif "handicap" in name and len(outcomes) == 3:            # European 3-way
            if line is None or line % 1 or line not in config.HANDICAP_LINES:
                continue
            for k, o in enumerate(outcomes):
                put(o, european_handicap_market(line), line, _side(o.get("outcomeName")) or "HDA"[k])
        elif ("total" in name or "over/under" in name or mtype == "totals") and not is_team_total:
            if line is None or line not in config.TOTAL_GOAL_LINES:
                continue
            for o in outcomes:
                sel = _norm(o.get("outcomeName"))
                if sel.startswith("over") or sel.startswith("under"):
                    put(o, f"TOTAL_{line}", line, "OVER" if sel.startswith("over") else "UNDER")
        elif len(outcomes) == 3 and (name in ("1x2", "full time result", "match result", "match winner", "3way")
                                     or mtype == "1x2"):
            for k, o in enumerate(outcomes):
                put(o, "1X2", None, _side(o.get("outcomeName")) or "HDA"[k])
    return out


# -- odds payloads ---------------------------------------------------------------------------------
def _bookmaker_order(books: dict) -> list[str]:
    usable = [b for b in books if b not in config.ODDSPAPI_EXCLUDED_BOOKMAKERS and not b.endswith("-ex")]
    preferred = [b for b in config.ODDSPAPI_BOOKMAKERS if b in usable]
    return preferred + sorted(b for b in usable if b not in preferred)


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_odds(body: dict, outcomes: dict, kickoff_utc: datetime | None = None) -> list[tuple]:
    """[(market, line, selection, price, bookmaker, priced_at)] — one price per selection, from the first
    bookmaker in priority order that quotes it. Works for both payloads:

    * ``/odds``: ``bookmakerOdds.{slug}.markets.{id}.outcomes.{id}.players."0".{price, active, changedAt}``;
    * ``/historical-odds``: ``bookmakers.{slug}.markets.{id}.outcomes.{id}.players."0": [{createdAt, price,
      active}, ...]`` — the last active snapshot BEFORE kick-off is used (the pre-match closing price);
      a snapshot at or after kick-off is never used.
    """
    books = body.get("bookmakerOdds") or body.get("bookmakers") or {}
    seen, rows = set(), []
    for slug in _bookmaker_order(books):
        book = books[slug] or {}
        if book.get("bookmakerIsActive") is False or book.get("suspended"):
            continue
        for market in (book.get("markets") or {}).values():
            if market.get("marketActive") is False:
                continue
            for outcome_id, outcome in (market.get("outcomes") or {}).items():
                key = outcomes.get(str(outcome_id))
                if key is None or (key[0], key[2]) in seen:
                    continue
                player = (outcome.get("players") or {}).get("0")
                quote = _pick_quote(player, kickoff_utc)
                if quote is None:
                    continue
                price, priced_at = quote
                seen.add((key[0], key[2]))
                rows.append((key[0], key[1], key[2], price, slug, priced_at))
    return rows


def _pick_quote(player, kickoff_utc):
    if isinstance(player, list):                                     # historical snapshots
        best = None
        for snap in player:
            when = _parse_time(snap.get("createdAt"))
            if snap.get("active") is False or not snap.get("price") or float(snap["price"]) <= 1.0:
                continue
            if kickoff_utc is not None and (when is None or when >= kickoff_utc):
                continue
            if best is None or (when and when > best[1]):
                best = (float(snap["price"]), when)
        return best
    if isinstance(player, dict):
        if player.get("active") is False or not player.get("price") or float(player["price"]) <= 1.0:
            return None
        return float(player["price"]), _parse_time(player.get("changedAt"))
    return None


def fixture_kickoff(fx: dict) -> datetime | None:
    return _parse_time(fx.get("startTime"))


def fixture_row(fx: dict, league: str) -> dict | None:
    ko = fixture_kickoff(fx)
    if ko is None:
        return None
    local = ko.astimezone(UK)
    return {"provider_fixture_id": str(fx.get("fixtureId")), "league": league, "date": local.date().isoformat(),
            "time": local.strftime("%H:%M"), "home_name": fx.get("participant1Name"),
            "away_name": fx.get("participant2Name"), "kickoff_utc": ko,
            "home_id": fx.get("participant1Id"), "away_id": fx.get("participant2Id"),
            "status": fx.get("statusId")}


def window(now_uk: datetime, days: int) -> tuple[datetime, datetime]:
    start = now_uk.replace(tzinfo=UK)
    return start, start + timedelta(days=days)
