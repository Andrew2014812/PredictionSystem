"""League registry.

The single place that knows which divisions exist. Adding a division means
adding one ``League`` line below; downloading, feature engineering,
training, prediction, the API providers and the UI all read this registry.

Only leagues with FULL support are enabled: several seasons of results with
shots, shots on target, corners and cards (so every model and market works)
plus fixtures and real odds. A league that cannot meet this is disabled
with the reason in ``disabled_reason`` — there is no partial support.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    code: str          # Football-Data division code (file name)
    name: str
    country: str
    tier: int
    api_football_id: int | None = None      # API-Football league id
    odds_api_key: str | None = None         # The Odds API sport key
    calendar: str = "european"              # "european" (Jul-Jun) | "calendar_year"
    enabled: bool = True
    disabled_reason: str = ""


LEAGUES: tuple[League, ...] = (
    League("E0", "Premier League", "England", 1, 39, "soccer_epl"),
    League("E1", "Championship", "England", 2, 40, "soccer_efl_champ"),
    League("E2", "League One", "England", 3, 41, "soccer_england_league1"),
    League("E3", "League Two", "England", 4, 42, "soccer_england_league2"),
    League("EC", "National League", "England", 5, 43, None, enabled=False,
           disabled_reason="no corner statistics in Football-Data (≈5% of matches)"),
    League("SC0", "Premiership", "Scotland", 1, 179, "soccer_spl"),
    League("SC1", "Championship", "Scotland", 2, 180, None),
    League("SC2", "League One", "Scotland", 3, 183, None),
    League("SC3", "League Two", "Scotland", 4, 184, None),
    League("SP1", "La Liga", "Spain", 1, 140, "soccer_spain_la_liga"),
    League("SP2", "Segunda División", "Spain", 2, 141, "soccer_spain_segunda_division"),
    League("I1", "Serie A", "Italy", 1, 135, "soccer_italy_serie_a"),
    League("I2", "Serie B", "Italy", 2, 136, "soccer_italy_serie_b"),
    League("D1", "Bundesliga", "Germany", 1, 78, "soccer_germany_bundesliga"),
    League("D2", "2. Bundesliga", "Germany", 2, 79, "soccer_germany_bundesliga2"),
    League("F1", "Ligue 1", "France", 1, 61, "soccer_france_ligue_one"),
    League("F2", "Ligue 2", "France", 2, 62, "soccer_france_ligue_two"),
    League("N1", "Eredivisie", "Netherlands", 1, 88, "soccer_netherlands_eredivisie"),
    League("B1", "Pro League", "Belgium", 1, 144, "soccer_belgium_first_div"),
    League("P1", "Primeira Liga", "Portugal", 1, 94, "soccer_portugal_primeira_liga"),
    League("T1", "Süper Lig", "Turkey", 1, 203, "soccer_turkey_super_league"),
    League("G1", "Super League", "Greece", 1, 197, "soccer_greece_super_league"),
)

_BY_CODE = {lg.code: lg for lg in LEAGUES}


def enabled_leagues() -> list[League]:
    return [lg for lg in LEAGUES if lg.enabled]


def league_codes() -> list[str]:
    return [lg.code for lg in enabled_leagues()]


def by_api_football_id() -> dict[int, League]:
    return {lg.api_football_id: lg for lg in enabled_leagues() if lg.api_football_id}


def by_odds_api_key() -> dict[str, League]:
    return {lg.odds_api_key: lg for lg in enabled_leagues() if lg.odds_api_key}


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
