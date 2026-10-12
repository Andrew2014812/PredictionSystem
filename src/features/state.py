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
    # opponent-adjusted: the opponent's pre-match strength and how the team did against it
    "opp_elo", "elo_surprise", "adj_gf", "adj_ga", "adj_cf", "adj_ca",
)
IDX = {name: i for i, name in enumerate(FIELDS)}


def team_record(gf, ga, shots_for, shots_against, sot_for, sot_against,
                corners_for, corners_against, fouls, yellows, reds, context: dict | None = None) -> np.ndarray:
    """One team's view of a played match.

    ``context`` holds the opponent's PRE-match strength (Elo, the team's own
    expected score, and the opponent's season averages before the match),
    so the adjusted values compare the performance with what the opponent
    usually allows — using only information available before kick-off.
    """
    c = context or {}
    nan = float("nan")
    total = gf + ga
    win, draw = float(gf > ga), float(gf == ga)
    score = win + 0.5 * draw
    return np.array([
        gf, ga, shots_for, shots_against, sot_for, sot_against,
        corners_for, corners_against, fouls, yellows, reds,
        3 * win + draw, win, draw, float(gf < ga),
        float(gf > 0 and ga > 0), float(total > 1.5), float(total > 2.5), float(total > 3.5),
        float(ga == 0), float(gf == 0),
        c.get("opp_elo", nan), score - c.get("expected", nan),
        gf - c.get("opp_ga", nan), ga - c.get("opp_gf", nan),
        corners_for - c.get("opp_ca", nan), corners_against - c.get("opp_cf", nan),
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

    def add(self, league: str, hg: float, ag: float, home_corners: float, away_corners: float) -> None:
        self.matches[league].append((hg, ag, home_corners, away_corners))

    def summary(self, league: str) -> dict[str, float]:
        rows = self.matches.get(league)
        if not rows:
            return {}
        arr = np.array(rows, dtype="float64")
        hg, ag, hc, ac = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        corners = hc + ac
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "league_avg_home_goals": hg.mean(),
                "league_avg_away_goals": ag.mean(),
                "league_avg_corners": np.nanmean(corners),
                "league_avg_home_corners": np.nanmean(hc),
                "league_avg_away_corners": np.nanmean(ac),
                "league_home_win_rate": (hg > ag).mean(),
                "league_draw_rate": (hg == ag).mean(),
                "league_over25_rate": (hg + ag > 2.5).mean(),
                "league_btts_rate": ((hg > 0) & (ag > 0)).mean(),
            }


class EloRatings:
    """Club Elo ratings updated after every played match (no future information).

    Standard football Elo: expected home score
    E = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400)), update K·G·(S − E) with the
    goal-difference multiplier G (1, 1.5, (11 + |gd|) / 8). A new club starts
    from a tier-dependent value; at the start of a new season every rating is
    pulled part of the way back towards its tier's base value.
    """

    def __init__(self, k: float = 20.0, home_advantage: float = 60.0, carry: float = 0.8):
        self.k, self.hfa, self.carry = k, home_advantage, carry
        self.ratings: dict[tuple, float] = {}
        self.season: dict[tuple, int] = {}

    @staticmethod
    def base(tier: int) -> float:
        return 1500.0 - 100.0 * (tier - 1)

    def get(self, team: tuple, season: int, tier: int) -> float:
        if team not in self.ratings:
            self.ratings[team], self.season[team] = self.base(tier), season
        elif self.season[team] != season:
            r = self.ratings[team]
            self.ratings[team] = self.carry * r + (1 - self.carry) * self.base(tier)
            self.season[team] = season
        return self.ratings[team]

    def expected_home(self, r_home: float, r_away: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(r_home + self.hfa - r_away) / 400.0))

    def update(self, home: tuple, away: tuple, hg: float, ag: float, season: int, tier: int) -> None:
        rh, ra = self.get(home, season, tier), self.get(away, season, tier)
        expected = self.expected_home(rh, ra)
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        gd = abs(hg - ag)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = self.k * g * (score - expected)
        self.ratings[home], self.ratings[away] = rh + delta, ra - delta
