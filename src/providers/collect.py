"""Collecting fresh fixtures, results and real odds from the API providers.

Run by the pipeline only (never by the web interface). Output files:

* ``local_data/results_live/fixtures.parquet`` — fixtures and finished
  results from the APIs, in the canonical match schema (goals only);
* ``local_data/odds/odds.parquet`` — archive of real bookmaker prices (unified
  schema). Every collection is kept; prices are only ever collected before
  kick-off, so a later quote never replaces a price after the match started;
* ``local_data/api/unmatched_teams.csv`` — provider team names that could not
  be mapped (add them to ``aliases.TEAM_ALIASES``).

Football-Data stays the historical source; when it publishes a match the
full Football-Data row (with shots, corners, cards) replaces the API row.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .. import config
from ..data.cleaning import CANONICAL_COLUMNS, make_match_id
from ..leagues import enabled_leagues, get_league
from ..storage import LocalStorage
from . import aliases
from .api_football import ApiFootball, parse_fixture, parse_odds
from .schema import ODDS_COLUMNS, empty_odds, odds_frame
from .oddspapi import OddsPapi, fixture_row, outcome_map
from .oddspapi import parse_odds as parse_oddspapi
from .teams_registry import TeamRegistry
from .the_odds_api import TheOddsApi, parse_event, parse_event_odds

log = logging.getLogger(__name__)

live_storage = LocalStorage(config.LIVE_DIR)
odds_storage = LocalStorage(config.ODDS_DIR)
api_storage = LocalStorage(config.API_DIR)
LIVE_FILE = "fixtures.parquet"
ODDS_FILE = "odds.parquet"


class TeamResolver:
    """Provider team name -> Football-Data name, per league; collects misses."""

    def __init__(self, matches: pd.DataFrame):
        recent = matches.loc[matches["season"] >= matches["season"].max() - 1]
        self.known = {lg: set(g["home_team"]) | set(g["away_team"]) for lg, g in recent.groupby("league")}
        self.unmatched: set[tuple[str, str, str]] = set()

    def __call__(self, provider: str, league: str, name: str) -> str | None:
        resolved = aliases.resolve(name, self.known.get(league))
        if resolved is None:
            self.unmatched.add((provider, league, name))
        return resolved


def _canonical(row: dict, home: str, away: str) -> dict:
    day = pd.Timestamp(row["date"])
    lg = get_league(row["league"])
    played = bool(row.get("played"))
    out = {c: None for c in CANONICAL_COLUMNS}
    out.update({
        "match_id": make_match_id(row["league"], day, home, away), "league": row["league"],
        "country": lg.country, "season": config.season_of(day, lg.calendar), "date": day,
        "time": row["time"], "kickoff": pd.Timestamp(f"{row['date']} {row['time']}"),
        "home_team": home, "away_team": away, "played": played,
        "home_goals": float(row["home_goals"]) if played else None,
        "away_goals": float(row["away_goals"]) if played else None,
    })
    if played:
        hg, ag = out["home_goals"], out["away_goals"]
        out["result"] = "H" if hg > ag else ("A" if hg < ag else "D")
    return out


def load_live_matches() -> pd.DataFrame:
    if not live_storage.exists(LIVE_FILE):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = live_storage.read_parquet(LIVE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    return df


def load_api_odds() -> pd.DataFrame:
    if not odds_storage.exists(ODDS_FILE):
        return empty_odds()
    return odds_storage.read_parquet(ODDS_FILE)


def _save_live(rows: list[dict]) -> int:
    if not rows:
        return 0
    new = pd.DataFrame(rows).reindex(columns=CANONICAL_COLUMNS + ["provider", "provider_fixture_id"])
    old = load_live_matches()
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    # newest information about a match wins (a fixture becomes a result)
    merged = merged.drop_duplicates("match_id", keep="last")
    for col in ("home_goals", "away_goals"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["played"] = merged["played"].astype(bool)
    live_storage.write_parquet(LIVE_FILE, merged)
    return len(new)


def _save_odds(records: list[dict]) -> int:
    new = odds_frame(records)
    if new.empty:
        return 0
    old = load_api_odds()
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    # keep the whole pre-match history; only drop a quote recorded twice in one collection
    merged = merged.sort_values("collected_at").drop_duplicates(
        ["match_id", "provider", "bookmaker", "market", "selection", "collected_at"], keep="last")
    odds_storage.write_parquet(ODDS_FILE, merged[ODDS_COLUMNS])
    return len(new)


def _odds_records(match_id: str, provider: str, fixture_id: str, rows, now) -> list[dict]:
    return [{"match_id": match_id, "provider": provider, "provider_fixture_id": fixture_id,
             "bookmaker": bookmaker, "market": market, "line": line, "selection": selection,
             "odds": price, "collected_at": now}
            for market, line, selection, price, bookmaker in rows]


MODES = ("results", "odds", "full")


def _extended_priority(candidate: tuple) -> tuple:
    """Top divisions first, then the earliest kick-off."""
    _, _, _, league, kickoff = candidate
    return (get_league(league).tier not in config.ODDS_API_EXTENDED_TIERS, kickoff)


# Names under which the leagues appear in OddsPapi's tournament list (matched with the country).
ODDSPAPI_NAMES = {
    "E0": ("premier league",), "E1": ("championship",), "E2": ("league one",), "E3": ("league two",),
    "SC0": ("premiership",), "SC1": ("championship",), "SC2": ("league one",), "SC3": ("league two",),
    "SP1": ("laliga", "la liga"), "SP2": ("laliga 2", "la liga 2", "segunda", "laliga hypermotion"),
    "I1": ("serie a",), "I2": ("serie b",), "D1": ("bundesliga",), "D2": ("2. bundesliga", "2 bundesliga"),
    "F1": ("ligue 1",), "F2": ("ligue 2",), "N1": ("eredivisie",), "B1": ("pro league", "jupiler pro league",
                                                                        "first division a"),
    "P1": ("liga portugal", "primeira liga", "liga portugal betclic"), "T1": ("super lig", "süper lig", "trendyol super lig"),
    "G1": ("super league", "super league 1", "stoiximan super league"),
}


def oddspapi_tournaments(odp: OddsPapi, now, budget: int) -> dict[str, int]:
    """League code -> OddsPapi tournament id: configured ids, then a stored map, then one
    /tournaments request (cached) matched on country + league name; ambiguous names are skipped."""
    stored = api_storage.read_json("oddspapi/tournaments.json") if api_storage.exists("oddspapi/tournaments.json") else {}
    found = {**stored, **config.ODDSPAPI_TOURNAMENTS}
    missing = [lg for lg in enabled_leagues() if lg.code not in found]
    if missing:
        tours = odp.tournaments(now, budget)
        for lg in missing:
            names = ODDSPAPI_NAMES.get(lg.code, (lg.name.lower(),))
            hits = [t for t in tours
                    if lg.country.lower() in f"{t.get('categoryName', '')} {t.get('categorySlug', '')}".lower()
                    and str(t.get("tournamentName", "")).strip().lower() in names]
            if len(hits) == 1:
                found[lg.code] = int(hits[0]["tournamentId"])
        api_storage.write_json("oddspapi/tournaments.json", {k: v for k, v in found.items()
                                                             if k not in config.ODDSPAPI_TOURNAMENTS})
    return found


def _oddspapi_records(match_id: str, fixture_id: str, rows, now, kickoff_utc) -> list[dict]:
    """One row per real price: when the bookmaker quoted it (``priced_at``), when FootPredict fetched it
    (``fetched_at``) and the kick-off; ``collected_at`` (the pre-match check) is the quote time."""
    uk = ZoneInfo("Europe/London")
    kickoff = kickoff_utc.astimezone(uk).replace(tzinfo=None) if kickoff_utc else None
    out = []
    for market, line, selection, price, bookmaker, priced_at in rows:
        when = priced_at.astimezone(uk).replace(tzinfo=None) if priced_at else now
        out.append({"match_id": match_id, "provider": PROVIDER_ODP, "provider_fixture_id": fixture_id,
                    "bookmaker": bookmaker, "market": market, "line": line, "selection": selection,
                    "odds": price, "collected_at": min(when, now), "priced_at": min(when, now),
                    "fetched_at": now, "kickoff": kickoff})
    return out


def oddspapi_outcomes(odp: OddsPapi, now, budget: int) -> dict:
    """outcomeId -> (market, line, selection), built from /markets once and committed (the raw API cache
    does not survive CI runs, so without this every run would pay for the catalogue again)."""
    name = "oddspapi/outcome_map.json"
    if api_storage.exists(name):
        stored = api_storage.read_json(name)
        if stored.get("outcomes"):
            return {k: tuple(v) for k, v in stored["outcomes"].items()}
    outcomes = outcome_map(odp.markets(now, budget))
    if outcomes:
        api_storage.write_json(name, {"built_at": now.isoformat(timespec="seconds"),
                                      "outcomes": {k: list(v) for k, v in outcomes.items()}})
    return outcomes


def collect_oddspapi(resolve_names, registry: TeamRegistry, now, run_budget: int, targets=None,
                     force: bool = False) -> tuple[list[dict], list[dict], dict, dict]:
    """Fixtures of the next ODDSPAPI_WINDOW_DAYS (one request for all leagues) and the latest
    pre-match prices of each of our matches, earliest kick-off first. ``targets`` =
    {"match_id"|"league"|"date": value} limits the matches; ``force`` ignores the refresh window.
    Prices are saved every 25 matches, so an interrupted run keeps what it fetched.
    Returns (fixture rows, price rows not yet saved, summary, match_id -> priced markets)."""
    odp = OddsPapi()
    live_rows, records, summary, covered = [], [], {}, {}
    if not odp.available:
        return live_rows, records, {"oddspapi": "no key"}, covered
    started = time.time()
    odp.account(now)
    tours = oddspapi_tournaments(odp, now, run_budget)
    by_tour = {v: k for k, v in tours.items()}
    omap = oddspapi_outcomes(odp, now, run_budget)
    start = pd.Timestamp(now).tz_localize("Europe/London").to_pydatetime()
    end = start + timedelta(days=config.ODDSPAPI_WINDOW_DAYS)
    todo = []
    for fx in odp.fixtures(now, run_budget, start, end):
        league = by_tour.get(int(fx.get("tournamentId") or 0))
        row = fixture_row(fx, league) if league else None
        if row is None or row["kickoff_utc"] <= start or row.get("status") not in (None, 0):
            continue                                                   # pre-match only
        country = get_league(league).country
        home = registry.resolve(PROVIDER_ODP, league, country, row["home_name"], row["home_id"], resolve_names.known.get(league))
        away = registry.resolve(PROVIDER_ODP, league, country, row["away_name"], row["away_id"], resolve_names.known.get(league))
        if not home or not away:
            continue
        canonical = _canonical({**row, "played": False}, home, away)
        live_rows.append({**canonical, "provider": PROVIDER_ODP, "provider_fixture_id": row["provider_fixture_id"]})
        if not targets or _targeted(canonical, targets):
            todo.append((row, canonical))
    priced = asked = 0
    for row, canonical in sorted(todo, key=lambda rc: rc[0]["kickoff_utc"]):
        if odp.quota_exhausted:
            break
        fid = row["provider_fixture_id"]
        key = f"historical|{fid}" if config.ODDSPAPI_ODDS_SOURCE == "historical" else f"odds|{fid}"
        if not force and odp.ledger.requested_within(key, config.ODDSPAPI_REFRESH_HOURS, now):
            continue
        asked += 1
        body = (odp.historical_odds(fid, now) if config.ODDSPAPI_ODDS_SOURCE == "historical"
                else odp.odds(fid, now, run_budget))
        rows = parse_oddspapi(body or {}, omap, row["kickoff_utc"])
        if rows:
            priced += 1
            records += _oddspapi_records(canonical["match_id"], fid, rows, now, row["kickoff_utc"])
            covered[canonical["match_id"]] = {r[0] for r in rows}
        if asked % 25 == 0:                                            # intermediate save
            _save_odds(records)
            records = []
            odp.ledger.save()
    odp.ledger.prune(now)
    odp.ledger.save()
    summary["oddspapi"] = (f"{odp.ledger.spent_this_run} billable requests this run, "
                           f"{odp.ledger.remaining if odp.ledger.remaining is not None else '?'} left; "
                           f"{asked} matches asked, {priced} priced, {odp.rate_limited} rate-limit waits, "
                           f"{(time.time() - started) / 60:.1f} min"
                           + ("; MONTHLY LIMIT REACHED" if odp.quota_exhausted else ""))
    return live_rows, records, summary, covered


def _targeted(row: dict, targets: dict) -> bool:
    return all(str(row.get(k)) == str(v) if k != "date" else str(pd.Timestamp(row["date"]).date()) == str(v)
               for k, v in targets.items())


def collect(matches: pd.DataFrame, today: date | None = None, mode: str = "full",
            run_budget: int | None = None, targets: dict | None = None, force: bool = False) -> dict:
    """Fetch fixtures / results / real odds from the configured providers.

    ``mode`` "results": free requests only — API-Football fixtures for yesterday,
    today and tomorrow (fresh scores) and The Odds API ``/events`` (fixtures of
    the coming week). "odds": real prices from The Odds API for matches starting
    within ODDS_API_WINDOW_HOURS. "full": both.

    Credits (odds only) are spent within ``run_budget`` and the monthly reserve;
    a league or match already asked within ODDS_API_REFRESH_HOURS is not asked
    again, whether or not that request returned prices (CreditLedger).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    today = today or date.today()
    now = datetime.now(ZoneInfo("Europe/London")).replace(tzinfo=None)   # kick-off times are UK time
    window_end = pd.Timestamp(now) + pd.Timedelta(hours=config.ODDS_API_WINDOW_HOURS)
    run_budget = config.ODDS_API_RUN_BUDGET if run_budget is None else run_budget
    resolve = TeamResolver(matches)
    summary = {"mode": mode, "api_football": "not used", "the_odds_api": "not used"}
    live_rows: list[dict] = []
    odds_records: list[dict] = []

    af = ApiFootball()
    if af.available and mode in ("results", "full"):
        days = [today + timedelta(days=d) for d in range(-config.API_FIXTURE_DAYS_BACK,
                                                          config.API_FIXTURE_DAYS_AHEAD + 1)]
        fixture_ids: dict[str, str] = {}
        upcoming_leagues: set[str] = set()
        for day in days:
            for item in af.fixtures(day, max_age=900):
                fx = parse_fixture(item)
                if fx is None or fx["cancelled"]:
                    continue
                home = resolve(PROVIDER_AF, fx["league"], fx["home_name"])
                away = resolve(PROVIDER_AF, fx["league"], fx["away_name"])
                if not home or not away:
                    continue
                row = _canonical(fx, home, away)
                row.update(provider=PROVIDER_AF, provider_fixture_id=fx["provider_fixture_id"])
                live_rows.append(row)
                fixture_ids[fx["provider_fixture_id"]] = row["match_id"]
                if not row["played"] and row["kickoff"] > pd.Timestamp(now):
                    upcoming_leagues.add(row["league"])
        if config.API_FOOTBALL_ODDS:                  # paid plans only (current-season odds)
            season = config.current_season(today)
            for lg in enabled_leagues():
                if not lg.api_football_id or lg.code not in upcoming_leagues:
                    continue
                for item in af.odds(lg.api_football_id, season):
                    fixture_id, rows = parse_odds(item)
                    if fixture_id in fixture_ids:
                        odds_records += _odds_records(fixture_ids[fixture_id], PROVIDER_AF, fixture_id, rows, now)
        summary["api_football"] = f"{af.cache.requests_today()} requests today"

    registry = TeamRegistry()
    odp_covered: dict = {}
    if mode in ("odds", "full"):
        rows, records, part, odp_covered = collect_oddspapi(resolve, registry, now, config.ODDSPAPI_RUN_BUDGET,
                                                            targets, force)
        live_rows += rows
        odds_records += records
        summary.update(part)
    registry.save()
    for item in registry.unmatched:
        resolve.unmatched.add(item)
    if registry.conflicts:                     # a provider id claimed by two clubs: nothing was attached
        api_storage.write_csv("team_id_conflicts.csv", pd.DataFrame(
            registry.conflicts, columns=["provider", "provider_team_id", "registered_as", "name_says"]))
        summary["team_id_conflicts"] = len(registry.conflicts)

    toa = TheOddsApi()
    fallback = config.odds_api_fallback()
    if toa.available and config.ODDS_API_ENABLED and fallback != "off":
        stored = load_api_odds()
        priced = (stored.groupby("match_id")["market"].agg(set).to_dict() if not stored.empty else {})
        if fallback == "missing":                          # never buy what another provider already priced
            for match_id, markets in odp_covered.items():
                priced.setdefault(match_id, set()).update(markets)
        else:
            priced = (stored.loc[stored["provider"] == PROVIDER_TOA]
                      .groupby("match_id")["market"].agg(set).to_dict() if not stored.empty else {})
        extended_candidates = []
        for lg in enabled_leagues():
            if not lg.odds_api_key:
                continue
            soon = []
            for event in toa.events(lg.odds_api_key):            # free
                ev = parse_event(event, lg.odds_api_key)
                if ev is None:
                    continue
                home = resolve(PROVIDER_TOA, ev["league"], ev["home_name"])
                away = resolve(PROVIDER_TOA, ev["league"], ev["away_name"])
                if not home or not away:
                    continue
                row = _canonical({**ev, "played": False}, home, away)
                if row["kickoff"] <= pd.Timestamp(now):
                    continue                                       # pre-match only
                live_rows.append({**row, "provider": PROVIDER_TOA, "provider_fixture_id": ev["provider_fixture_id"]})
                if row["kickoff"] <= window_end:
                    soon.append((ev["provider_fixture_id"], row["match_id"], row["kickoff"]))
            if mode == "results" or not soon:
                continue
            missing = [s for s in soon if not {"1X2", "TOTAL_2.5"} <= priced.get(s[1], set())]
            if missing and not toa.ledger.requested_within(f"odds|{lg.odds_api_key}",
                                                           config.ODDS_API_REFRESH_HOURS, now):
                by_id = {e["id"]: e for e in toa.league_odds(lg.odds_api_key, now, run_budget)}
                for fixture_id, match_id, _ in soon:
                    if fixture_id in by_id:
                        odds_records += _odds_records(match_id, PROVIDER_TOA, fixture_id,
                                                      parse_event_odds(by_id[fixture_id]), now)
            for fixture_id, match_id, kickoff in soon:
                have = priced.get(match_id, set())
                extended_known = "BTTS" in have and "DOUBLE_CHANCE" in have and any(m.startswith("CORNERS_") for m in have)
                if not extended_known and \
                        not toa.ledger.requested_within(f"event|{fixture_id}", 24 * 30, now):
                    extended_candidates.append((lg.odds_api_key, fixture_id, match_id, lg.code, kickoff))
        if mode in ("odds", "full") and config.ODDS_API_EXTENDED_MARKETS:
            for sport_key, fixture_id, match_id, _, _ in sorted(extended_candidates, key=_extended_priority):
                if not toa.ledger.can_spend(len(config.ODDS_API_EXTENDED_MARKETS), run_budget):
                    break
                body = toa.event_odds(sport_key, fixture_id, now, run_budget)
                if body:
                    odds_records += _odds_records(match_id, PROVIDER_TOA, fixture_id, parse_event_odds(body), now)
        toa.ledger.prune(now)
        toa.ledger.save()
        summary["the_odds_api"] = (f"fallback={fallback}: {toa.credits_used} credits this run, "
                                   f"{toa.ledger.remaining if toa.ledger.remaining is not None else '?'} left")

    summary["live_rows"] = _save_live(live_rows)
    summary["odds_rows"] = _save_odds(odds_records)
    if resolve.unmatched:
        api_storage.write_csv("unmatched_teams.csv",
                              pd.DataFrame(sorted(resolve.unmatched), columns=["provider", "league", "name"]))
        summary["unmatched_teams"] = len(resolve.unmatched)
    elif api_storage.exists("unmatched_teams.csv"):
        api_storage.path("unmatched_teams.csv").unlink()
    log.info("providers: %s", summary)
    return summary


PROVIDER_AF = "api_football"
PROVIDER_TOA = "the_odds_api"
PROVIDER_ODP = "oddspapi"
