"""Readable values of the factors on the "Why …?" card: who is ahead, by how much and in what unit.

A model feature is a number whose meaning depends on its definition (src/features/builder.py):

* ``diff_*`` and ``elo_diff`` are HOME minus AWAY. ``-108`` for the team-strength difference means the away team
  is 108 Elo points stronger (home advantage included), so the card says "Burgos +108 Elo points";
* ``elo_exp_home`` is the home team's expected result from the strength ratings (0 = certain defeat, 1 = certain
  win, a draw counts ½): "Leganes expected result: 0.35 / 1";
* a home / away pair (``h_form10_gf`` + ``a_form10_gf``) shows both teams with one unit: "Leganes 1.45 · Burgos
  1.20 goals/game".

Only the wording changes: the value is the feature value the model saw (rounded for display), and the sign of the
factor's influence on the probability is decided elsewhere (``card.card_factors``).
"""
from __future__ import annotations

import math
import re

UNITS = {
    "en": {
        "elo": "Elo points", "goals_g": "goals/game", "gd_g": "goal difference/game", "pts_g": "points/game",
        "shots_g": "shots/game", "sot_g": "shots on target/game", "corners_g": "corners/game",
        "cards_g": "per game", "fouls_g": "fouls/game", "pts": "points", "days": "days of rest", "matches": "matches",
        "meetings": "meetings", "goals_m": "goals/match", "corners_m": "corners/match", "places_higher": "places higher",
        "place_higher": "place higher", "level": "level", "expected": "expected result",
        "vs_usual": "vs opponents' usual", "vs_season": "vs season average", "above_exp": "points/game above expectation",
        "of": "of", "pos": "{n}."
    },
    "uk": {
        "elo": "очок Ело", "goals_g": "голу за гру", "gd_g": "різниці м'ячів за гру", "pts_g": "очка за гру",
        "shots_g": "удару за гру", "sot_g": "удару в площину за гру", "corners_g": "кутового за гру",
        "cards_g": "за гру", "fouls_g": "фолу за гру", "pts": "очок", "days": "днів відпочинку", "matches": "матчів",
        "meetings": "зустрічей", "goals_m": "голу за матч", "corners_m": "кутового за матч",
        "places_higher": "позицій вище", "place_higher": "позицію вище", "level": "порівну",
        "expected": "очікуваний результат", "vs_usual": "відносно звичного для суперників",
        "vs_season": "відносно сезону", "above_exp": "очка за гру понад очікування", "of": "з", "pos": "{n}."
    },
}

# home-minus-away differences: feature -> (unit key, decimals)
DIFF = {
    "elo_diff": ("elo", 0), "diff_season_ppg": ("pts_g", 2), "diff_venue_ppg": ("pts_g", 2),
    "diff_form5_ppg": ("pts_g", 2), "diff_form10_ppg": ("pts_g", 2), "diff_season_gd": ("goals_g", 2),
    "diff_form10_gd": ("goals_g", 2), "diff_form10_shots": ("shots_g", 1), "diff_form10_sot": ("sot_g", 1),
    "diff_form10_corners": ("corners_g", 1), "diff_points": ("pts", 0), "diff_rest_days": ("days", 0),
}

# one team's statistic (the part after h_ / a_): pattern -> (unit key or None, kind, decimals)
SIDE = [
    (r"^elo$|_opp_elo$", "elo", "num", 0),
    (r"rate$|share$|^freq_", None, "pct", 0),
    (r"^position$", None, "pos", 0),
    (r"^position_norm$", None, "pct", 0),
    (r"^points$", "pts", "num", 0),
    (r"rest_days$", "days", "num", 0),
    (r"_n$|games$", "matches", "num", 0),
    (r"adj_g[fa]$|adj_c[fa]$", "vs_usual", "signed", 2),
    (r"elo_surprise$", "above_exp", "signed", 2),
    (r"^trend_", "vs_season", "signed", 2),
    (r"ppg$", "pts_g", "num", 2),
    (r"_gd$", "gd_g", "signed", 2),
    (r"(gf|ga)$|attack_vs|_defence$", "goals_g", "num", 2),
    (r"sot_(for|against)$|^sot_vs", "sot_g", "num", 1),
    (r"shots_(for|against)$", "shots_g", "num", 1),
    (r"corner_diff$", "corners_g", "signed", 1),
    (r"corners_(for|against)$|^corners_vs", "corners_g", "num", 1),
    (r"fouls$", "fouls_g", "num", 1),
    (r"(yellows|reds)$", "cards_g", "num", 2),
]

# features of the match, not of one team
MATCH = [
    (r"^h2h_n$", "meetings", "num", 0),
    (r"rate$", None, "pct", 0),
    (r"^h2h_(avg_goals|home_team_goals)$|^league_avg_(home_|away_)?goals$", "goals_m", "num", 2),
    (r"^league_avg_.*corners$|^form_corners_sum$", "corners_m", "num", 1),
]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(v: float, d: int) -> str:
    return f"{v:.{d}f}"


def _value(v: float, unit: str | None, kind: str, d: int, u: dict) -> str:
    if kind == "pct":
        return f"{v * 100:.0f}%" if abs(v) <= 1.0 else f"{v:.0f}%"
    if kind == "pos":
        return u["pos"].format(n=int(round(v)))
    text = f"{v:+.{d}f}" if kind == "signed" else _num(v, d)
    return text


def _rule(rest: str, rules) -> tuple | None:
    for pattern, unit, kind, d in rules:
        if re.search(pattern, rest):
            return unit, kind, d
    return None


def difference(feature: str, value, home: str, away: str, lang: str = "en") -> str:
    """'Burgos +108 Elo points' for a home-minus-away difference (``level`` when it is zero)."""
    u = UNITS.get(lang, UNITS["en"])
    if _missing(value):
        return "—"
    v = float(value)
    if feature == "diff_position":           # a lower position number is the better one
        n = int(round(abs(v)))
        if n == 0:
            return u["level"]
        leader = home if v < 0 else away
        return f"{leader} {n} {u['place_higher'] if n == 1 else u['places_higher']}"
    unit, d = DIFF[feature]
    if round(abs(v), d) == 0:
        return u["level"]
    leader = home if v > 0 else away
    return f"{leader} +{_num(abs(v), d)} {u[unit]}"


def is_difference(feature: str) -> bool:
    return feature in DIFF or feature == "diff_position"


def expected_result(value, home: str, lang: str = "en") -> str:
    u = UNITS.get(lang, UNITS["en"])
    if _missing(value):
        return "—"
    return f"{home} {u['expected']}: {float(value):.2f} / 1"


def pair(rest: str, values: dict, home: str, away: str, lang: str = "en") -> str:
    """'Leganes 1.45 · Burgos 1.20 goals/game' for a statistic of both teams (``values``: side -> value)."""
    u = UNITS.get(lang, UNITS["en"])
    unit, kind, d = _rule(rest, SIDE) or (None, "num", 2)
    parts = []
    for side, team in (("h", home), ("a", away)):
        if side in values:
            v = values[side]
            parts.append(f"{team} {'—' if _missing(v) else _value(float(v), unit, kind, d, u)}")
    text = " · ".join(parts)
    return f"{text} {u[unit]}" if unit else text


def single(feature: str, value, lang: str = "en") -> str:
    """A match-level feature (head-to-head, league averages)."""
    u = UNITS.get(lang, UNITS["en"])
    if _missing(value):
        return "—"
    if isinstance(value, str):
        return value
    unit, kind, d = _rule(feature, MATCH) or (None, "num", 2)
    text = _value(float(value), unit, kind, d, u)
    return f"{text} {u[unit]}" if unit else text
