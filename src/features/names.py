"""Human-readable names of internal feature columns.

Names are composed from the tokens of the column name, so new features
following the naming scheme get a readable label automatically. Explicit
entries in ``OVERRIDES`` take precedence.
"""
from __future__ import annotations

import re

from ..leagues import league_label

SIDES = {"h": "Home team", "a": "Away team"}

STAT_LABELS = {
    "gf": "goals scored per game",
    "ga": "goals conceded per game",
    "gd": "goal difference per game",
    "ppg": "points per game",
    "shots_for": "shots per game",
    "shots_against": "shots conceded per game",
    "sot_for": "shots on target per game",
    "sot_against": "shots on target conceded per game",
    "corners_for": "corners per game",
    "corners_against": "corners conceded per game",
    "fouls": "fouls per game",
    "yellows": "yellow cards per game",
    "reds": "red cards per game",
    "win_rate": "win rate",
    "draw_rate": "draw rate",
    "loss_rate": "loss rate",
    "n": "matches available",
    "games": "matches played",
}

FREQ_LABELS = {
    "btts": "both-teams-scored rate",
    "over15": "over 1.5 goals rate",
    "over25": "over 2.5 goals rate",
    "over35": "over 3.5 goals rate",
    "clean_sheet": "clean sheet rate",
    "failed_to_score": "failed-to-score rate",
}

TREND_LABELS = {
    "scoring": "scoring trend (last 5 vs season)",
    "conceding": "conceding trend (last 5 vs season)",
    "shots": "shooting trend (last 5 vs season)",
    "corners": "corners trend (last 5 vs season)",
}

OVERRIDES = {
    "league": "League",
    "h_points": "Home team · league points",
    "a_points": "Away team · league points",
    "h_position": "Home team · league position",
    "a_position": "Away team · league position",
    "h_position_norm": "Home team · relative table position",
    "a_position_norm": "Away team · relative table position",
    "h_season_gd": "Home team · goal difference per game (season)",
    "a_season_gd": "Away team · goal difference per game (season)",
    "h_rest_days": "Home team · days of rest",
    "a_rest_days": "Away team · days of rest",
    "diff_rest_days": "Rest advantage of the home team (days)",
    "diff_season_ppg": "Points-per-game gap (season)",
    "diff_position": "League position gap",
    "diff_points": "League points gap",
    "diff_season_gd": "Goal difference gap (season)",
    "diff_venue_ppg": "Home form at home vs away form away",
    "diff_form5_ppg": "Recent form gap (last 5)",
    "diff_form10_ppg": "Recent form gap (last 10)",
    "diff_form10_gd": "Recent goal difference gap (last 10)",
    "diff_form10_shots": "Shots gap (last 10)",
    "diff_form10_sot": "Shots on target gap (last 10)",
    "diff_form10_corners": "Corners gap (last 10)",
    "h_attack_vs_a_defence": "Home attack vs away defence",
    "a_attack_vs_h_defence": "Away attack vs home defence",
    "h_form_attack_vs_a_form_defence": "Home recent attack vs away recent defence",
    "a_form_attack_vs_h_form_defence": "Away recent attack vs home recent defence",
    "h_sot_vs_a_sot_conceded": "Home shots on target vs away shots conceded",
    "a_sot_vs_h_sot_conceded": "Away shots on target vs home shots conceded",
    "h_corners_vs_a_corners_conceded": "Home corners vs away corners conceded",
    "a_corners_vs_h_corners_conceded": "Away corners vs home corners conceded",
    "form_corners_sum": "Combined corner volume of both teams (last 10)",
    "h2h_n": "Head-to-head meetings available",
    "h2h_home_win_rate": "Head-to-head win rate of the home team",
    "h2h_draw_rate": "Head-to-head draw rate",
    "h2h_away_win_rate": "Head-to-head win rate of the away team",
    "h2h_avg_goals": "Head-to-head goals per match",
    "h2h_home_team_goals": "Head-to-head goals of the home team",
    "h2h_btts_rate": "Head-to-head both-teams-scored rate",
    "h2h_over25_rate": "Head-to-head over 2.5 rate",
    "league_avg_home_goals": "League average home goals",
    "league_avg_away_goals": "League average away goals",
    "league_avg_corners": "League average corners",
    "league_home_win_rate": "League home win rate",
    "league_draw_rate": "League draw rate",
    "league_over25_rate": "League over 2.5 rate",
    "league_btts_rate": "League both-teams-scored rate",
    "market_prob_home": "Bookmaker probability: home win",
    "market_prob_draw": "Bookmaker probability: draw",
    "market_prob_away": "Bookmaker probability: away win",
    "market_prob_over25": "Bookmaker probability: over 2.5",
}

_TEAM = re.compile(r"^(?P<side>[ha])_(?P<rest>.+)$")


def _venue(side: str) -> str:
    return "home matches" if side == "h" else "away matches"


def feature_label(name: str) -> str:
    if name in OVERRIDES:
        return OVERRIDES[name]
    if name.startswith("league_") and "=" in name:          # one-hot column
        return f"League: {league_label(name.split('=', 1)[1])}"
    m = _TEAM.match(name)
    if not m:
        return name.replace("_", " ").capitalize()
    side, rest = m["side"], m["rest"]
    team = SIDES[side]
    if rest.startswith("season_"):
        stat = rest[len("season_"):]
        return f"{team} · {STAT_LABELS.get(stat, stat)} (season)"
    if rest.startswith("venue_"):
        stat = rest[len("venue_"):]
        return f"{team} · {STAT_LABELS.get(stat, stat)} ({_venue(side)})"
    form = re.match(r"form(\d+)_(.+)", rest)
    if form:
        return f"{team} · {STAT_LABELS.get(form[2], form[2])} (last {form[1]})"
    if rest.startswith("freq_"):
        stat = rest[len("freq_"):]
        return f"{team} · {FREQ_LABELS.get(stat, stat)} (last 10)"
    if rest.startswith("trend_"):
        stat = rest[len("trend_"):]
        return f"{team} · {TREND_LABELS.get(stat, stat)}"
    return f"{team} · {rest.replace('_', ' ')}"


def feature_labels(names) -> dict[str, str]:
    return {n: feature_label(n) for n in names}


# ---------------------------------------------------------------------------
# Short topics for the plain-language explanation ("Why this prediction?")
# ---------------------------------------------------------------------------
_TEAM_TOPICS = [
    (r"_(form\d+|season)_(gf|sot_for|shots_for)$|_freq_(over|btts)|_trend_(scoring|shots)$", "attacking output"),
    (r"_(form\d+|season)_(ga|sot_against|shots_against)$|_freq_clean_sheet|_trend_conceding$",
     "defensive record"),
    (r"_freq_failed_to_score$", "failures to score"),
    (r"_(form\d+)_(ppg|win_rate|draw_rate|loss_rate|gd|n)$", "recent results"),
    (r"_(points|season_ppg|season_gd|position|position_norm|season_games)$", "league standing"),
    (r"_venue_(ppg|games)$", "{venue} record"),
    (r"_venue_(gf|shots_for|sot_for)$", "{venue} attack"),
    (r"_venue_ga$", "{venue} defence"),
    (r"corners", "corner volume"),
    (r"_rest_days$", "rest before the match"),
    (r"_(fouls|yellows|reds)$", "discipline"),
]

_GLOBAL_TOPICS = [
    (r"^h_(attack_vs|form_attack_vs|sot_vs)", "home attack against the away defence"),
    (r"^a_(attack_vs|form_attack_vs|sot_vs)", "away attack against the home defence"),
    (r"corners", "corner volume of both teams"),
    (r"^diff_(position|points|season_ppg|season_gd)$", "gap in league standing"),
    (r"^diff_(form|venue)", "gap in recent form"),
    (r"^diff_rest", "difference in rest days"),
    (r"^h2h_", "head-to-head record"),
    (r"^league_(avg_home_goals|avg_away_goals|over25|btts)", "scoring level of the league"),
    (r"^league_", "typical results in this league"),
    (r"^league$", "league profile"),
    (r"^market_", "bookmaker expectations"),
]


def feature_topic(name: str) -> str:
    """Short noun phrase, e.g. "the home team's recent results"."""
    m = _TEAM.match(name)
    if m and not re.match(r"^[ha]_(attack_vs|form_attack_vs|sot_vs|corners_vs)", name):
        team = "the home team's" if m["side"] == "h" else "the away team's"
        venue = "home" if m["side"] == "h" else "away"
        for pattern, topic in _TEAM_TOPICS:
            if re.search(pattern, name):
                return f"{team} {topic.format(venue=venue)}"
        return f"{team} statistics"
    for pattern, topic in _GLOBAL_TOPICS:
        if re.search(pattern, name):
            return f"the {topic}"
    return feature_label(name).lower()
