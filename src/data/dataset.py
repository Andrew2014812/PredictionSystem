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


def harmonise_fixture_names(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Football-Data's fixtures file sometimes spells a club unlike its results files ("KV Mechelen" vs
    "Mechelen", "Sheffield Utd" vs "Sheffield United"), which would make a new club without history and a
    second id for the same match. A fixture name unknown to the league's recent results is mapped to the
    results name with the same unambiguous rule as provider names (aliases.resolve); otherwise it is kept."""
    if fixtures.empty or results.empty:
        return fixtures
    from ..providers import aliases
    recent = results.loc[results["season"] >= results["season"].max() - 1]
    known = {lg: set(g["home_team"]) | set(g["away_team"]) for lg, g in recent.groupby("league")}
    out = fixtures.copy()
    renamed = {}
    for col in ("home_team", "away_team"):
        def fix(row):
            names = known.get(row["league"])
            if not names or row[col] in names:
                return row[col]
            team = aliases.resolve(row[col], names)
            if team:
                renamed[(row["league"], row[col])] = team
            return team or row[col]
        out[col] = out.apply(fix, axis=1)
    if renamed:
        log.info("fixture names harmonised with the results files: %s", renamed)
        out["match_id"] = [make_match_id(lg, d, h, a) for lg, d, h, a in
                           zip(out["league"], out["date"], out["home_team"], out["away_team"])]
    return out


def build_matches(leagues: list[str] | None = None) -> pd.DataFrame:
    """All known matches (played + upcoming), deduplicated and time-ordered.

    Sources: Football-Data results, Football-Data fixtures and the API
    providers' fixtures / results. Deduplication keeps the most complete row,
    so a Football-Data result (with statistics) replaces an API result of the
    same match as soon as it is published.
    """
    results = load_results(leagues=leagues)
    fixtures = harmonise_fixture_names(load_fixtures(leagues), results)
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
