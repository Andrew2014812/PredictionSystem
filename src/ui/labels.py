"""Localised names of markets, selections, statuses and countries."""
from __future__ import annotations

from ..leagues import get_league
from ..markets.markets import market_group, market_line
from .i18n import t

MARKET_GROUP_TITLES = {
    "1X2": "Match result", "DOUBLE_CHANCE": "Double chance", "TOTAL": "Total goals",
    "BTTS": "Both teams to score", "EXACT_SCORE": "Exact score", "HANDICAP": "Handicap",
    "CORNERS": "Total corners",
}


def group_title(group: str) -> str:
    return t(MARKET_GROUP_TITLES.get(group, group))


def market_title(market: str) -> str:
    group, line = market_group(market), market_line(market)
    title = group_title(group)
    if group in ("TOTAL", "CORNERS") and line is not None:
        return f"{title} {line:g}"
    if group == "HANDICAP" and line is not None:
        return f"{title} {line:+g}"
    return title


def selection_label(market: str, selection: str, home: str | None = None, away: str | None = None) -> str:
    home = home or t("Home")
    away = away or t("Away")
    group, line = market_group(market), market_line(market)
    if group == "1X2":
        return {"H": t("{team} win", team=home), "D": t("Draw"), "A": t("{team} win", team=away)}[selection]
    if group == "DOUBLE_CHANCE":
        return {"1X": t("{team} or draw", team=home), "X2": t("{team} or draw", team=away),
                "12": t("{a} or {b}", a=home, b=away)}[selection]
    if group == "TOTAL":
        return t("Over {line} goals", line=f"{line:g}") if selection == "OVER" else t("Under {line} goals", line=f"{line:g}")
    if group == "BTTS":
        return t("Both teams to score: Yes") if selection == "YES" else t("Both teams to score: No")
    if group == "EXACT_SCORE":
        return t("Exact score {score}", score=selection.replace("-", ":"))
    if group == "HANDICAP":
        return {"H": f"{home} ({line:+g})", "D": f"{t('Draw')} ({line:+g})", "A": f"{away} ({-line:+g})"}[selection]
    if group == "CORNERS":
        return (t("Over {line} corners", line=f"{line:g}") if selection == "OVER"
                else t("Under {line} corners", line=f"{line:g}"))
    return f"{market} {selection}"


def status_label(status: str) -> str:
    return t({"WON": "Won", "LOST": "Lost", "UPCOMING": "Upcoming", "VOID": "Void"}.get(status, status))


def country_label(country: str) -> str:
    return t(country)


# Ukrainian football names of the leagues (display only; the registry keeps the English names)
LEAGUE_NAMES_UK = {
    "E0": "Прем'єр-ліга", "E1": "Чемпіоншип", "E2": "Ліга 1", "E3": "Ліга 2", "EC": "Національна ліга",
    "SC0": "Прем'єршип", "SC1": "Чемпіоншип", "SC2": "Ліга 1", "SC3": "Ліга 2",
    "SP1": "Ла Ліга", "SP2": "Сегунда", "I1": "Серія A", "I2": "Серія B",
    "D1": "Бундесліга", "D2": "Друга Бундесліга", "F1": "Ліга 1", "F2": "Ліга 2",
    "N1": "Ередивізі", "B1": "Про Ліга", "P1": "Прімейра-ліга", "T1": "Суперліга", "G1": "Суперліга",
}


def league_name(code: str) -> str:
    from .i18n import current_language
    lg = get_league(code)
    return LEAGUE_NAMES_UK.get(code, lg.name) if current_language() == "uk" else lg.name


def league_label(code: str, with_country: bool = False) -> str:
    lg = get_league(code)
    return f"{country_label(lg.country)} · {league_name(code)}" if with_country else league_name(code)
