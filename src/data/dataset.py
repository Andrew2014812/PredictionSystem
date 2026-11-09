"""Building the unified match table from local raw files.

Historical results (``local_data/results/<season>/<league>.csv``) and future
fixtures (``local_data/fixtures/fixtures.csv``) are read separately,
normalised to one schema and merged. The ``played`` flag is what separates
them downstream: only played matches update team state or become training
targets.
"""
from __future__ import annotations

import logging

import pandas as pd

from .. import config
from ..leagues import league_codes
from ..storage import fixtures_storage, results_storage
from .cleaning import (CANONICAL_COLUMNS, deduplicate, drop_double_bookings, drop_stale_fixtures, make_match_id,
                       normalise)

log = logging.getLogger(__name__)


def _read_raw_csv(path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def load_results(seasons: list[int] | None = None, leagues: list[str] | None = None) -> pd.DataFrame:
    """Normalised historical results for the given seasons and leagues."""
    leagues = set(leagues or league_codes())
    frames = []
    for path in results_storage.list("*/*.csv"):
        season_dir, league = path.parent.name, path.stem
        if league not in leagues:
            continue
        if seasons is not None and season_dir not in {config.season_code(s) for s in seasons}:
            continue
        frame = normalise(_read_raw_csv(path))
        frames.append(frame.loc[frame["played"]])
    if not frames:
        return normalise(pd.DataFrame())
    return pd.concat(frames, ignore_index=True)


def load_fixtures(leagues: list[str] | None = None) -> pd.DataFrame:
    """Normalised upcoming fixtures (rows without a result)."""
    name = config.FOOTBALL_DATA_FIXTURES_FILE
    if not fixtures_storage.exists(name):
        return normalise(pd.DataFrame())
    fixtures = normalise(_read_raw_csv(fixtures_storage.path(name)))
    fixtures = fixtures.loc[fixtures["league"].isin(leagues or league_codes()) & ~fixtures["played"]]
    return fixtures.reset_index(drop=True)


def load_live(leagues: list[str] | None = None) -> pd.DataFrame:
    """Fresh fixtures / results collected from the API providers (goals only)."""
    from ..providers.collect import load_live_matches
    live = load_live_matches().reindex(columns=CANONICAL_COLUMNS)
    return live.loc[live["league"].isin(leagues or league_codes())].reset_index(drop=True)


def _known_clubs(results: pd.DataFrame, min_matches: int = 3) -> tuple[dict[str, set], set]:
    """Per league: clubs of the previous season and clubs with at least ``min_matches`` matches this season;
    and every such club of any league (a club promoted from another of our leagues is known, not a
    misspelling)."""
    if results.empty:
        return {}, set()
    current = results["season"].max()
    prev = results.loc[results["season"] == current - 1]
    cur = results.loc[results["season"] == current]
    counts = pd.concat([cur[["league", "home_team"]].rename(columns={"home_team": "team"}),
                        cur[["league", "away_team"]].rename(columns={"away_team": "team"})]).value_counts()
    known: dict[str, set] = {}
    for lg, g in prev.groupby("league"):
        known.setdefault(lg, set()).update(set(g["home_team"]) | set(g["away_team"]))
    for (lg, team), n in counts.items():
        if n >= min_matches:
            known.setdefault(lg, set()).add(team)
    return known, set().union(*known.values()) if known else set()


def harmonise_names(frame: pd.DataFrame, results: pd.DataFrame, min_matches: int = 1) -> pd.DataFrame:
    """Football-Data sometimes spells a club differently for a few days ("KV Mechelen" for "Mechelen",
    "Sheffield Utd", "Oxford Utd" — in the fixtures file and, on 05.11.2026, in the current results files),
    which makes a club without history and a second id for one match. A name unknown to the league is
    mapped to a known club when the unambiguous rule for provider names (aliases.resolve) finds one; a club
    known in any league, and a new club without an unambiguous match, keep their name. Known = previous
    season, or at least ``min_matches`` matches this season (3 when the current results rows themselves are
    checked, so a one-day misspelling does not count as a club)."""
    if frame.empty or results.empty:
        return frame
    from ..providers import aliases
    known, everywhere = _known_clubs(results, min_matches)
    out = frame.copy()
    renamed = {}
    for col in ("home_team", "away_team"):
        def fix(row):
            names = known.get(row["league"])
            if not names or row[col] in names or row[col] in everywhere:
                return row[col]
            team = aliases.resolve(row[col], names)
            if team and team != row[col]:
                renamed[(row["league"], row[col])] = team
                return team
            return row[col]
        out[col] = out.apply(fix, axis=1)
    if renamed:
        log.info("club names harmonised with the results files: %s", renamed)
        out["match_id"] = [make_match_id(lg, d, h, a) for lg, d, h, a in
                           zip(out["league"], out["date"], out["home_team"], out["away_team"])]
    return out


def harmonise_fixture_names(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Kept for callers of v6.1: the fixtures file harmonised with the results files."""
    return harmonise_names(fixtures, results)


def build_matches(leagues: list[str] | None = None) -> pd.DataFrame:
    """All known matches (played + upcoming), deduplicated and time-ordered.

    Sources: Football-Data results, Football-Data fixtures and the API
    providers' fixtures / results. Deduplication keeps the most complete row,
    so a Football-Data result (with statistics) replaces an API result of the
    same match as soon as it is published.
    """
    results = load_results(leagues=leagues)
    if not results.empty:                       # the current season's rows may carry a passing misspelling
        current = results["season"] == results["season"].max()
        results = pd.concat([results.loc[~current], harmonise_names(results.loc[current], results, min_matches=3)],
                            ignore_index=True)
    fixtures = harmonise_names(load_fixtures(leagues), results)
    live = load_live(leagues)
    frames = [f.assign(_source=src) for f, src in ((results, "fd"), (fixtures, "fd"), (live, "api")) if not f.empty]
    matches = pd.concat(frames, ignore_index=True)
    before = len(matches)
    matches = deduplicate(matches)
    matches = drop_stale_fixtures(matches)
    matches = drop_double_bookings(matches).drop(columns="_source")
    log.info("matches: %d rows (%d duplicates/stale removed), %d upcoming",
             len(matches), before - len(matches), int((~matches["played"]).sum()))
    return matches.sort_values(["date", "kickoff", "league", "home_team"]).reset_index(drop=True)
