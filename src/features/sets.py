"""Which features each model uses.

Features are grouped by meaning; every model picks the groups that are
relevant to its target. Groups are regular expressions over the column
names produced by ``features.builder``.
"""
from __future__ import annotations

import re

import pandas as pd

from .. import config

CATEGORICAL_FEATURES = ["league"]

GROUPS: dict[str, list[str]] = {
    "table": [r"^[ha]_(season_games|points|season_ppg|season_gd|position|position_norm)$"],
    "season_attack": [r"^[ha]_season_(gf|ga|shots_for|shots_against|sot_for|sot_against)$"],
    "season_corners": [r"^[ha]_season_corners_(for|against)$"],
    "discipline": [r"^[ha]_season_(fouls|yellows|reds)$"],
    "venue_result": [r"^[ha]_venue_(games|ppg)$"],
    "venue_goals": [r"^[ha]_venue_(gf|ga|shots_for|sot_for)$"],
    "venue_corners": [r"^[ha]_venue_corners_(for|against)$"],
    "form_result": [r"^[ha]_form\d+_(n|ppg|win_rate|draw_rate|loss_rate|gd)$"],
    "form_goals": [r"^[ha]_form\d+_(gf|ga|shots_for|shots_against|sot_for|sot_against)$"],
    "form_corners": [r"^[ha]_form\d+_corners_(for|against)$"],
    "frequencies": [r"^[ha]_freq_"],
    "trends_goals": [r"^[ha]_trend_(scoring|conceding|shots)$"],
    "trends_corners": [r"^[ha]_trend_corners(_conceded)?$"],
    "venue_form_goals": [r"^[ha]_venue_form5_(gf|ga|shots_for)$"],
    "venue_form_corners": [r"^[ha]_venue_form5_corners_(for|against)$"],
    "corner_pressure": [r"^[ha]_form10_(corner_diff|corner_share|shot_share)$"],
    "rest": [r"^[ha]_rest_days$", r"^diff_rest_days$"],
    "h2h_result": [r"^h2h_(n|home_win_rate|draw_rate|away_win_rate)$"],
    "h2h_goals": [r"^h2h_(n|avg_goals|home_team_goals|btts_rate|over25_rate)$"],
    "diff_result": [r"^diff_(season_ppg|position|points|season_gd|venue_ppg|form5_ppg|form10_ppg|form10_gd)$"],
    "diff_attack": [r"^diff_form10_(shots|sot)$", r"^[ha]_(attack_vs|form_attack_vs|sot_vs)_"],
    "diff_corners": [r"^diff_form10_corners$", r"^[ha]_corners_vs_", r"^form_corners_sum$"],
    "league_result": [r"^league_(home_win_rate|draw_rate)$"],
    "league_goals": [r"^league_(avg_home_goals|avg_away_goals|over25_rate|btts_rate)$"],
    "league_corners": [r"^league_avg_(home_|away_)?corners$"],
    # Bookmaker information; only included when USE_ODDS_FEATURES is on.
    "market": [r"^market_prob_"],
}

MODEL_GROUPS: dict[str, list[str]] = {
    "result": ["table", "season_attack", "venue_result", "venue_goals", "form_result", "form_goals",
               "trends_goals", "rest", "h2h_result", "h2h_goals", "diff_result", "diff_attack",
               "league_result", "league_goals", "discipline"],
    "home_goals": ["table", "season_attack", "venue_goals", "form_result", "form_goals", "frequencies",
                   "trends_goals", "rest", "h2h_goals", "diff_result", "diff_attack", "league_goals",
                   "venue_form_goals", "corner_pressure"],
    "away_goals": ["table", "season_attack", "venue_goals", "form_result", "form_goals", "frequencies",
                   "trends_goals", "rest", "h2h_goals", "diff_result", "diff_attack", "league_goals",
                   "venue_form_goals", "corner_pressure"],
    "corners": ["table", "season_attack", "season_corners", "venue_corners", "form_corners",
                "trends_corners", "rest", "diff_corners", "league_corners", "league_goals",
                "venue_form_corners", "corner_pressure", "form_goals", "venue_form_goals"],
}


def numeric_features(model: str, columns: list[str] | pd.Index) -> list[str]:
    groups = list(MODEL_GROUPS[model])
    if config.USE_ODDS_FEATURES:
        groups.append("market")
    patterns = [re.compile(p) for g in groups for p in GROUPS[g]]
    return [c for c in columns if any(p.search(c) for p in patterns)]


def model_features(model: str, columns: list[str] | pd.Index) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature columns of ``model``."""
    return numeric_features(model, columns), list(CATEGORICAL_FEATURES)
