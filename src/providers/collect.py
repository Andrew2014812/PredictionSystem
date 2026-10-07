"""Collecting fresh fixtures, results and real odds from the API providers.

Run by the pipeline only (never by the web interface). Output files:

* ``local_data/results_live/fixtures.parquet`` — fixtures and finished
  results from the APIs, in the canonical match schema (goals only);
* ``local_data/odds/odds.parquet`` — real bookmaker prices (unified schema),
  collected before kick-off;
* ``local_data/api/unmatched_teams.csv`` — provider team names that could not
  be mapped (add them to ``aliases.TEAM_ALIASES``).

Football-Data stays the historical source; when it publishes a match the
full Football-Data row (with shots, corners, cards) replaces the API row.
"""
from __future__ import annotations

import logging
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
    merged = merged.sort_values("collected_at").drop_duplicates(
        ["match_id", "provider", "bookmaker", "market", "selection"], keep="last")
    odds_storage.write_parquet(ODDS_FILE, merged[ODDS_COLUMNS])
    return len(new)


def _odds_records(match_id: str, provider: str, fixture_id: str, rows, now) -> list[dict]:
    return [{"match_id": match_id, "provider": provider, "provider_fixture_id": fixture_id,
             "bookmaker": bookmaker, "market": market, "line": line, "selection": selection,
             "odds": price, "collected_at": now}
            for market, line, selection, price, bookmaker in rows]


def collect(matches: pd.DataFrame, today: date | None = None) -> dict:
    """Fetch fixtures / results / odds from every configured provider.

    1. API-Football ``/fixtures`` for yesterday, today and tomorrow (all leagues,
       one request per day): fresh final scores and near fixtures.
    2. The Odds API ``/events`` per league (free): fixtures of the coming week.
    3. The Odds API ``/odds`` (h2h + totals) for leagues with a match in the next
       ODDS_API_WINDOW_HOURS, at most once per ODDS_API_REFRESH_HOURS.
    Odds are stored only for matches that have not started.
    """
    today = today or date.today()
    now = datetime.now(ZoneInfo("Europe/London")).replace(tzinfo=None)   # kick-off times are UK time
    window_end = pd.Timestamp(now) + pd.Timedelta(hours=config.ODDS_API_WINDOW_HOURS)
    resolve = TeamResolver(matches)
    summary = {"api_football": "no key", "the_odds_api": "no key"}
    live_rows: list[dict] = []
    odds_records: list[dict] = []

    af = ApiFootball()
    if af.available:
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

    toa = TheOddsApi()
    if toa.available:
        for lg in enabled_leagues():
            if not lg.odds_api_key:
                continue
            events = {e["id"]: e for e in toa.events(lg.odds_api_key)}       # free
            soon = []
            for event in events.values():
                ev = parse_event(event, lg.odds_api_key)
                if ev is None:
                    continue
                home = resolve(PROVIDER_TOA, ev["league"], ev["home_name"])
                away = resolve(PROVIDER_TOA, ev["league"], ev["away_name"])
                if not home or not away:
                    continue
                row = _canonical({**ev, "played": False}, home, away)
                if row["kickoff"] <= pd.Timestamp(now):
                    continue
                live_rows.append({**row, "provider": PROVIDER_TOA, "provider_fixture_id": ev["provider_fixture_id"]})
                if row["kickoff"] <= window_end:
                    soon.append((ev["provider_fixture_id"], row["match_id"]))
            if not soon:
                continue                                                      # no credits for idle leagues
            priced = {e["id"]: e for e in toa.odds(lg.odds_api_key)}
            for fixture_id, match_id in soon:
                if fixture_id in priced:
                    odds_records += _odds_records(match_id, PROVIDER_TOA, fixture_id,
                                                  parse_event_odds(priced[fixture_id]), now)
        summary["the_odds_api"] = f"{toa.credits_used} credits this run"

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
