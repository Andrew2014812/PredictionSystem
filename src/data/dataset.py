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
from .cleaning import deduplicate, drop_stale_fixtures, normalise

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


def build_matches(leagues: list[str] | None = None) -> pd.DataFrame:
    """All known matches (played + upcoming), deduplicated and time-ordered."""
    results = load_results(leagues=leagues)
    fixtures = load_fixtures(leagues)
    matches = pd.concat([results, fixtures], ignore_index=True)
    before = len(matches)
    matches = deduplicate(matches)
    matches = drop_stale_fixtures(matches)
    log.info("matches: %d rows (%d duplicates/stale removed), %d upcoming",
             len(matches), before - len(matches), int((~matches["played"]).sum()))
    return matches.sort_values(["date", "kickoff", "league", "home_team"]).reset_index(drop=True)
