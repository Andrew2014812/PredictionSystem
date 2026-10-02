"""Running state of teams, league tables, head-to-heads and league averages.

Every structure here only ever *reads* history that has already been
added. The feature builder reads state for all matches of a day first and
adds that day's results afterwards, so nothing about a match (or about a
simultaneous match of the same round) can leak into its own features.
"""
from __future__ import annotations

import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

# Per-match record of one team, from that team's perspective.
FIELDS = (
    "gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against",
    "corners_for", "corners_against", "fouls", "yellows", "reds",
    "points", "win", "draw", "loss",
    "btts", "over15", "over25", "over35", "clean_sheet", "failed_to_score",
)
IDX = {name: i for i, name in enumerate(FIELDS)}


def team_record(gf, ga, shots_for, shots_against, sot_for, sot_against,
                corners_for, corners_against, fouls, yellows, reds) -> np.ndarray:
    total = gf + ga
    win, draw = float(gf > ga), float(gf == ga)
    return np.array([
        gf, ga, shots_for, shots_against, sot_for, sot_against,
        corners_for, corners_against, fouls, yellows, reds,
        3 * win + draw, win, draw, float(gf < ga),
        float(gf > 0 and ga > 0), float(total > 1.5), float(total > 2.5), float(total > 3.5),
        float(ga == 0), float(gf == 0),
    ], dtype="float64")


def nanmean_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.full(len(FIELDS), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.vstack(rows), axis=0)


@dataclass
class Accumulator:
    """NaN-aware running sum, so missing statistics do not count as zeros."""
    total: np.ndarray = field(default_factory=lambda: np.zeros(len(FIELDS)))
    count: np.ndarray = field(default_factory=lambda: np.zeros(len(FIELDS)))
    games: int = 0

    def add(self, record: np.ndarray) -> None:
        present = ~np.isnan(record)
        self.total[present] += record[present]
        self.count[present] += 1
        self.games += 1

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.total / self.count, np.nan)

    def sum(self, name: str) -> float:
        return float(self.total[IDX[name]])


class TeamHistory:
    """Last N matches of every team across seasons and divisions."""

    def __init__(self, max_window: int):
        self.recent: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=max_window))
        self.last_date: dict[tuple, object] = {}

    def window(self, team: tuple, size: int) -> list[np.ndarray]:
        recent = self.recent.get(team)
        return list(recent)[-size:] if recent else []

    def add(self, team: tuple, day, record: np.ndarray) -> None:
        self.recent[team].append(record)
        self.last_date[team] = day


class SeasonTables:
    """League table plus overall / home / away season-to-date statistics."""

    def __init__(self):
        self.overall: dict[tuple, dict[str, Accumulator]] = defaultdict(lambda: defaultdict(Accumulator))
        self.home: dict[tuple, dict[str, Accumulator]] = defaultdict(lambda: defaultdict(Accumulator))
        self.away: dict[tuple, dict[str, Accumulator]] = defaultdict(lambda: defaultdict(Accumulator))

    def add(self, key: tuple, home: str, away: str, home_rec: np.ndarray, away_rec: np.ndarray) -> None:
        self.overall[key][home].add(home_rec)
        self.overall[key][away].add(away_rec)
        self.home[key][home].add(home_rec)
        self.away[key][away].add(away_rec)

    def standings(self, key: tuple) -> dict[str, int]:
        """Positions sorted by points, goal difference, goals scored, name."""
        table = self.overall.get(key, {})
        ranked = sorted(
            table,
            key=lambda t: (-table[t].sum("points"),
                           -(table[t].sum("gf") - table[t].sum("ga")),
                           -table[t].sum("gf"),
                           t),
        )
        return {team: pos + 1 for pos, team in enumerate(ranked)}

    def table(self, key: tuple) -> dict[str, Accumulator]:
        return self.overall.get(key, {})


class HeadToHead:
    """Recent meetings of every pair of teams (either venue)."""

    def __init__(self, size: int):
        self.meetings: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=size))

    @staticmethod
    def key(country: str, a: str, b: str) -> tuple:
        return (country, *sorted((a, b)))

    def get(self, country: str, a: str, b: str) -> list[tuple]:
        return list(self.meetings.get(self.key(country, a, b), ()))

    def add(self, country: str, home: str, away: str, hg: float, ag: float) -> None:
        self.meetings[self.key(country, home, away)].append((home, away, hg, ag))


class LeaguePrior:
    """Running averages of the last N matches of each league."""

    def __init__(self, size: int):
        self.matches: dict[str, deque] = defaultdict(lambda: deque(maxlen=size))

    def add(self, league: str, hg: float, ag: float, corners: float) -> None:
        self.matches[league].append((hg, ag, corners))

    def summary(self, league: str) -> dict[str, float]:
        rows = self.matches.get(league)
        if not rows:
            return {}
        arr = np.array(rows, dtype="float64")
        hg, ag, corners = arr[:, 0], arr[:, 1], arr[:, 2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "league_avg_home_goals": hg.mean(),
                "league_avg_away_goals": ag.mean(),
                "league_avg_corners": np.nanmean(corners),
                "league_home_win_rate": (hg > ag).mean(),
                "league_draw_rate": (hg == ag).mean(),
                "league_over25_rate": (hg + ag > 2.5).mean(),
                "league_btts_rate": ((hg > 0) & (ag > 0)).mean(),
            }
