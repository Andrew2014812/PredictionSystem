"""League registry.

The single place that knows which Football-Data divisions exist. Adding a
division means adding one ``League`` line below; downloading, feature
engineering, training, prediction and the UI all read this registry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    code: str          # Football-Data division code (file name)
    name: str
    country: str
    tier: int
    enabled: bool = True


LEAGUES: tuple[League, ...] = (
    League("E0", "Premier League", "England", 1),
    League("E1", "Championship", "England", 2),
    League("E2", "League One", "England", 3),
    League("E3", "League Two", "England", 4),
    League("EC", "National League", "England", 5),
    League("SC0", "Premiership", "Scotland", 1),
    League("SC1", "Championship", "Scotland", 2),
    League("SC2", "League One", "Scotland", 3),
    League("SC3", "League Two", "Scotland", 4),
    League("SP1", "La Liga", "Spain", 1),
    League("SP2", "Segunda División", "Spain", 2),
    League("I1", "Serie A", "Italy", 1),
    League("I2", "Serie B", "Italy", 2),
    League("D1", "Bundesliga", "Germany", 1),
    League("D2", "2. Bundesliga", "Germany", 2),
    League("F1", "Ligue 1", "France", 1),
    League("F2", "Ligue 2", "France", 2),
    League("N1", "Eredivisie", "Netherlands", 1),
    League("B1", "Pro League", "Belgium", 1),
    League("P1", "Primeira Liga", "Portugal", 1),
    League("T1", "Süper Lig", "Turkey", 1),
    League("G1", "Super League", "Greece", 1),
)

_BY_CODE = {lg.code: lg for lg in LEAGUES}


def enabled_leagues() -> list[League]:
    return [lg for lg in LEAGUES if lg.enabled]


def league_codes() -> list[str]:
    return [lg.code for lg in enabled_leagues()]


def get_league(code: str) -> League:
    """Registry entry for ``code``; unknown codes get a generic entry."""
    return _BY_CODE.get(code) or League(code, code, "Other", 9)


def league_country(code: str) -> str:
    return get_league(code).country


def league_label(code: str, with_country: bool = False) -> str:
    lg = get_league(code)
    return f"{lg.country} · {lg.name}" if with_country else lg.name


def leagues_by_country() -> dict[str, list[League]]:
    """Enabled leagues grouped by country, in registry order."""
    grouped: dict[str, list[League]] = {}
    for lg in enabled_leagues():
        grouped.setdefault(lg.country, []).append(lg)
    for leagues in grouped.values():
        leagues.sort(key=lambda lg: lg.tier)
    return grouped
