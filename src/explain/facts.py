"""Plain football facts behind a prediction, chosen for its market.

These are real pre-match statistics read from the feature row (the same
numbers the models saw), shown BEFORE the model's factors, so a reader who
knows football but not machine learning sees the evidence first.
"""
from __future__ import annotations

import math

from ..markets.markets import market_group

TEXT = {
    "en": {
        "home_attack": "{team} score {v} goals per game at home this season",
        "away_attack": "{team} score {v} goals per game away this season",
        "home_defence": "{team} concede {v} goals per game at home",
        "away_defence": "{team} concede {v} goals per game away",
        "over25": "{team}: {v} of the last 10 matches had over 2.5 goals",
        "btts": "{team}: both teams scored in {v} of the last 10 matches",
        "league_over25": "In this league {v} of matches go over 2.5 goals",
        "position": "{team} are {v} in the table",
        "ppg5": "{team} took {v} points per game over the last 5 matches",
        "home_record": "{team} average {v} points per game at home",
        "away_record": "{team} average {v} points per game away",
        "elo": "Strength rating gap (Elo, incl. home advantage): {v} in favour of {team}",
        "corners_for": "{team} win {v} corners per game (last 10)",
        "corners_against": "{team} concede {v} corners per game (last 10)",
        "league_corners": "This league averages {v} corners per match",
        "ordinal": lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}",
    },
    "uk": {
        "home_attack": "{team} забивають {v} гола за матч удома в цьому сезоні",
        "away_attack": "{team} забивають {v} гола за матч на виїзді в цьому сезоні",
        "home_defence": "{team} пропускають {v} гола за матч удома",
        "away_defence": "{team} пропускають {v} гола за матч на виїзді",
        "over25": "{team}: у {v} з останніх 10 матчів було більше 2.5 голів",
        "btts": "{team}: обидві команди забивали в {v} з останніх 10 матчів",
        "league_over25": "У цій лізі {v} матчів закінчуються тоталом більше 2.5",
        "position": "{team} — {v} місце в таблиці",
        "ppg5": "{team} набирали {v} очка за матч в останніх 5 іграх",
        "home_record": "{team} в середньому набирають {v} очка за матч удома",
        "away_record": "{team} в середньому набирають {v} очка за матч на виїзді",
        "elo": "Різниця рейтингів сили (Elo, з урахуванням поля): {v} на користь {team}",
        "corners_for": "{team} подають {v} кутових за матч (останні 10)",
        "corners_against": "{team} дозволяють {v} кутових за матч (останні 10)",
        "league_corners": "У цій лізі в середньому {v} кутових за матч",
        "ordinal": lambda n: f"{n}-е",
    },
}


def _ok(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def market_facts(market: str, row, home: str, away: str, lang: str = "en", limit: int = 5) -> list[str]:
    """Up to ``limit`` factual statements relevant to the market of the prediction."""
    tx = TEXT.get(lang, TEXT["en"])
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    group = market_group(market)
    facts: list[str] = []

    def add(key, team, value, fmt="{:.2f}"):
        if _ok(value):
            facts.append(tx[key].format(team=team, v=fmt.format(float(value)) if fmt else value))

    if group in ("TOTAL", "BTTS", "EXACT_SCORE"):
        add("home_attack", home, get("h_venue_gf"))
        add("away_attack", away, get("a_venue_gf"))
        add("away_defence", away, get("a_venue_ga"))
        add("home_defence", home, get("h_venue_ga"))
        key = "btts" if group == "BTTS" else "over25"
        stat = "freq_btts" if group == "BTTS" else "freq_over25"
        for side, team in (("h", home), ("a", away)):
            v = get(f"{side}_{stat}")
            if _ok(v):
                facts.append(tx[key].format(team=team, v=round(float(v) * 10)))
        v = get("league_over25_rate")
        if group == "TOTAL" and _ok(v):
            facts.append(tx["league_over25"].format(v=f"{float(v) * 100:.0f}%"))
    elif group == "CORNERS":
        add("corners_for", home, get("h_form10_corners_for"), "{:.1f}")
        add("corners_for", away, get("a_form10_corners_for"), "{:.1f}")
        add("corners_against", home, get("h_form10_corners_against"), "{:.1f}")
        add("corners_against", away, get("a_form10_corners_against"), "{:.1f}")
        add("league_corners", "", get("league_avg_corners"), "{:.1f}")
    else:                                   # 1X2, double chance, handicap
        for side, team in (("h", home), ("a", away)):
            pos = get(f"{side}_position")
            if _ok(pos):
                facts.append(tx["position"].format(team=team, v=tx["ordinal"](int(pos))))
        add("ppg5", home, get("h_form5_ppg"))
        add("ppg5", away, get("a_form5_ppg"))
        add("home_record", home, get("h_venue_ppg"))
        add("away_record", away, get("a_venue_ppg"))
        diff = get("elo_diff")
        if _ok(diff):
            facts.append(tx["elo"].format(v=f"{abs(float(diff)):.0f}", team=home if float(diff) >= 0 else away))
    return facts[:limit]
