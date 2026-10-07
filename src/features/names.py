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
    "corner_diff": "corner difference per game",
    "corner_share": "share of corners in its matches",
    "shot_share": "share of shots in its matches",
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
    "corners_conceded": "corners conceded trend (last 5 vs season)",
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
    "league_avg_home_corners": "League average home corners",
    "league_avg_away_corners": "League average away corners",
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

# Ukrainian vocabulary ------------------------------------------------------
SIDES_UK = {"h": "Господарі", "a": "Гості"}
STAT_LABELS_UK = {
    "gf": "забиті голи за матч", "ga": "пропущені голи за матч", "gd": "різниця голів за матч",
    "ppg": "очки за матч", "shots_for": "удари за матч", "shots_against": "удари суперника за матч",
    "sot_for": "удари в площину воріт за матч", "sot_against": "удари суперника в площину за матч",
    "corners_for": "кутові за матч", "corners_against": "кутові суперника за матч",
    "fouls": "фоли за матч", "yellows": "жовті картки за матч", "reds": "червоні картки за матч",
    "win_rate": "частка перемог", "draw_rate": "частка нічиїх", "loss_rate": "частка поразок",
    "n": "доступні матчі", "corner_diff": "різниця кутових за матч",
    "corner_share": "частка кутових у матчах", "shot_share": "частка ударів у матчах",
    "games": "зіграні матчі",
}
FREQ_LABELS_UK = {
    "btts": "частка матчів «обидві заб'ють»", "over15": "частка тоталу більше 1.5",
    "over25": "частка тоталу більше 2.5", "over35": "частка тоталу більше 3.5",
    "clean_sheet": "частка «сухих» матчів", "failed_to_score": "частка матчів без забитих",
}
TREND_LABELS_UK = {
    "scoring": "тренд забитих (останні 5 проти сезону)", "conceding": "тренд пропущених (останні 5 проти сезону)",
    "shots": "тренд ударів (останні 5 проти сезону)", "corners": "тренд кутових (останні 5 проти сезону)",
    "corners_conceded": "тренд кутових суперника (останні 5 проти сезону)",
}
OVERRIDES_UK = {
    "league": "Ліга",
    "h_points": "Господарі · очки в таблиці", "a_points": "Гості · очки в таблиці",
    "h_position": "Господарі · місце в таблиці", "a_position": "Гості · місце в таблиці",
    "h_position_norm": "Господарі · відносне місце в таблиці", "a_position_norm": "Гості · відносне місце в таблиці",
    "h_season_gd": "Господарі · різниця голів за матч (сезон)", "a_season_gd": "Гості · різниця голів за матч (сезон)",
    "h_rest_days": "Господарі · днів відпочинку", "a_rest_days": "Гості · днів відпочинку",
    "diff_rest_days": "Перевага господарів у відпочинку (дні)",
    "diff_season_ppg": "Різниця очок за матч (сезон)", "diff_position": "Різниця місць у таблиці",
    "diff_points": "Різниця очок у таблиці", "diff_season_gd": "Різниця голів (сезон)",
    "diff_venue_ppg": "Форма господарів удома проти форми гостей на виїзді",
    "diff_form5_ppg": "Різниця поточної форми (останні 5)", "diff_form10_ppg": "Різниця поточної форми (останні 10)",
    "diff_form10_gd": "Різниця голів (останні 10)", "diff_form10_shots": "Різниця ударів (останні 10)",
    "diff_form10_sot": "Різниця ударів у площину (останні 10)", "diff_form10_corners": "Різниця кутових (останні 10)",
    "h_attack_vs_a_defence": "Атака господарів проти оборони гостей",
    "a_attack_vs_h_defence": "Атака гостей проти оборони господарів",
    "h_form_attack_vs_a_form_defence": "Поточна атака господарів проти поточної оборони гостей",
    "a_form_attack_vs_h_form_defence": "Поточна атака гостей проти поточної оборони господарів",
    "h_sot_vs_a_sot_conceded": "Удари господарів у площину проти дозволених гостями",
    "a_sot_vs_h_sot_conceded": "Удари гостей у площину проти дозволених господарями",
    "h_corners_vs_a_corners_conceded": "Кутові господарів проти дозволених гостями",
    "a_corners_vs_h_corners_conceded": "Кутові гостей проти дозволених господарями",
    "form_corners_sum": "Сумарна кількість кутових обох команд (останні 10)",
    "h2h_n": "Кількість особистих зустрічей", "h2h_home_win_rate": "Частка перемог господарів в особистих зустрічах",
    "h2h_draw_rate": "Частка нічиїх в особистих зустрічах", "h2h_away_win_rate": "Частка перемог гостей в особистих зустрічах",
    "h2h_avg_goals": "Голи за матч в особистих зустрічах", "h2h_home_team_goals": "Голи господарів в особистих зустрічах",
    "h2h_btts_rate": "«Обидві заб'ють» в особистих зустрічах", "h2h_over25_rate": "Тотал більше 2.5 в особистих зустрічах",
    "league_avg_home_goals": "Середні голи господарів у лізі", "league_avg_away_goals": "Середні голи гостей у лізі",
    "league_avg_corners": "Середні кутові в лізі", "league_avg_home_corners": "Середні кутові господарів у лізі",
    "league_avg_away_corners": "Середні кутові гостей у лізі", "league_home_win_rate": "Частка перемог господарів у лізі",
    "league_draw_rate": "Частка нічиїх у лізі", "league_over25_rate": "Частка тоталу більше 2.5 у лізі",
    "league_btts_rate": "Частка «обидві заб'ють» у лізі",
    "market_prob_home": "Ймовірність букмекера: перемога господарів",
    "market_prob_draw": "Ймовірність букмекера: нічия", "market_prob_away": "Ймовірність букмекера: перемога гостей",
    "market_prob_over25": "Ймовірність букмекера: тотал більше 2.5",
}

_VOCAB = {
    "en": {"sides": SIDES, "stats": STAT_LABELS, "freq": FREQ_LABELS, "trend": TREND_LABELS,
           "overrides": OVERRIDES, "season": "season", "last": "last", "home_m": "home matches",
           "away_m": "away matches", "league": "League"},
    "uk": {"sides": SIDES_UK, "stats": STAT_LABELS_UK, "freq": FREQ_LABELS_UK, "trend": TREND_LABELS_UK,
           "overrides": OVERRIDES_UK, "season": "сезон", "last": "останні", "home_m": "домашні матчі",
           "away_m": "виїзні матчі", "league": "Ліга"},
}


def feature_label(name: str, lang: str = "en") -> str:
    v = _VOCAB.get(lang, _VOCAB["en"])
    if name in v["overrides"]:
        return v["overrides"][name]
    if name.startswith("league=") or (name.startswith("league_") and "=" in name):   # one-hot column
        return f"{v['league']}: {league_label(name.split('=', 1)[1])}"
    m = _TEAM.match(name)
    if not m:
        return name.replace("_", " ").capitalize()
    side, rest = m["side"], m["rest"]
    team, stats = v["sides"][side], v["stats"]
    venue = v["home_m"] if side == "h" else v["away_m"]
    if rest.startswith("season_"):
        stat = rest[len("season_"):]
        return f"{team} · {stats.get(stat, stat)} ({v['season']})"
    vform = re.match(r"venue_form(\d+)_(.+)", rest)
    if vform:
        return f"{team} · {stats.get(vform[2], vform[2])} ({v['last']} {vform[1]}, {venue})"
    if rest.startswith("venue_"):
        stat = rest[len("venue_"):]
        return f"{team} · {stats.get(stat, stat)} ({venue})"
    form = re.match(r"form(\d+)_(.+)", rest)
    if form:
        return f"{team} · {stats.get(form[2], form[2])} ({v['last']} {form[1]})"
    if rest.startswith("freq_"):
        stat = rest[len("freq_"):]
        return f"{team} · {v['freq'].get(stat, stat)} ({v['last']} 10)"
    if rest.startswith("trend_"):
        stat = rest[len("trend_"):]
        return f"{team} · {v['trend'].get(stat, stat)}"
    return f"{team} · {rest.replace('_', ' ')}"


def feature_labels(names, lang: str = "en") -> dict[str, str]:
    return {n: feature_label(n, lang) for n in names}


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


_TEAM_TOPICS_UK = {
    "attacking output": "атакувальна результативність {team}", "defensive record": "надійність оборони {team}",
    "failures to score": "матчі {team} без забитих", "recent results": "останні результати {team}",
    "league standing": "становище {team} у таблиці", "{venue} record": "результати {team} {venue}",
    "{venue} attack": "атака {team} {venue}", "{venue} defence": "оборона {team} {venue}",
    "corner volume": "кількість кутових {team}", "rest before the match": "відпочинок {team} перед матчем",
    "discipline": "дисципліна {team}",
}
_GLOBAL_TOPICS_UK = {
    "home attack against the away defence": "атака господарів проти оборони гостей",
    "away attack against the home defence": "атака гостей проти оборони господарів",
    "corner volume of both teams": "кількість кутових обох команд",
    "gap in league standing": "різниця в турнірному становищі", "gap in recent form": "різниця в поточній формі",
    "difference in rest days": "різниця у відпочинку", "head-to-head record": "історія особистих зустрічей",
    "scoring level of the league": "результативність ліги", "typical results in this league": "типові результати ліги",
    "league profile": "профіль ліги", "bookmaker expectations": "очікування букмекерів",
}


def feature_topic(name: str, lang: str = "en") -> str:
    """Short noun phrase, e.g. "the home team's recent results"."""
    uk = lang == "uk"
    m = _TEAM.match(name)
    if m and not re.match(r"^[ha]_(attack_vs|form_attack_vs|sot_vs|corners_vs)", name):
        home = m["side"] == "h"
        for pattern, topic in _TEAM_TOPICS:
            if re.search(pattern, name):
                if uk:
                    venue = "удома" if home else "на виїзді"
                    return _TEAM_TOPICS_UK[topic].format(venue=venue, team="господарів" if home else "гостей")
                team = "the home team's" if home else "the away team's"
                return f"{team} {topic.format(venue='home' if home else 'away')}"
        return ("статистика " + ("господарів" if home else "гостей")) if uk else \
            f"{'the home team' if home else 'the away team'}'s statistics"
    for pattern, topic in _GLOBAL_TOPICS:
        if re.search(pattern, name):
            return _GLOBAL_TOPICS_UK[topic] if uk else f"the {topic}"
    return feature_label(name, lang).lower()
