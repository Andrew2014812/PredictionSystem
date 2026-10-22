"""Plain football facts behind a prediction, chosen for its exact market and line.

Facts are counted on the teams' real pre-match history — the home team's
recent HOME matches and the away team's recent AWAY matches — so a fact about
"over 1.5" counts exactly that line, and "x of the last k" always uses the
number of matches that really exist (k may be below 10 early in a season).
They are shown BEFORE the model's factors, so a reader who knows football but
not machine learning sees the evidence first.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..markets.markets import market_group, market_line, settle_selection

WINDOW = 10

TEXT = {
    "en": {
        "venue": {"H": "home", "A": "away"},
        "goals_for": "{team} score {v} goals per game in their last {k} {venue} matches",
        "goals_against": "{team} concede {v} goals per game in their last {k} {venue} matches",
        "total": "{team}: {x} of the last {k} {venue} matches had {cond}",
        "over": "over {line} goals", "under": "under {line} goals",
        "btts_yes": "{team}: both teams scored in {x} of the last {k} {venue} matches",
        "btts_no": "{team}: at least one side failed to score in {x} of the last {k} {venue} matches",
        "failed": "{team} failed to score in {x} of the last {k} {venue} matches",
        "clean": "{team} kept a clean sheet in {x} of the last {k} {venue} matches",
        "league_total": "In this league {v} of matches go over {line} goals",
        "league_btts": "In this league both teams score in {v} of matches",
        "corners_avg": "{team}'s last {k} {venue} matches averaged {v} corners in total",
        "corners_line": "{team}: {x} of the last {k} {venue} matches had {cond}",
        "c_over": "over {line} corners", "c_under": "under {line} corners",
        "won": "{team} won {x} of the last {k} {venue} matches",
        "drew": "{team} drew {x} of the last {k} {venue} matches",
        "lost": "{team} lost {x} of the last {k} {venue} matches",
        "unbeaten": "{team} were unbeaten in {x} of the last {k} {venue} matches",
        "not_drawn": "{x} of {team}'s last {k} {venue} matches had a winner",
        "position": "{team} are {v} in the table",
        "handicap": "With {line} {team} would have won this bet in {x} of the last {k} {venue} matches",
        "handicap_push": "With {line} {team} would have won this bet in {x} and got a refund in {y} of the last {k} {venue} matches",
        "score_seen": "{team}: the score {score} came up in {x} of the last {k} {venue} matches",
        "ordinal": lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}",
    },
    "uk": {
        "venue": {"H": "домашніх", "A": "виїзних"},
        "goals_for": "{team} забивають {v} гола за матч в останніх {k} {venue} матчах",
        "goals_against": "{team} пропускають {v} гола за матч в останніх {k} {venue} матчах",
        "total": "{team}: у {x} з останніх {k} {venue} матчів — {cond}",
        "over": "більше {line} голів", "under": "менше {line} голів",
        "btts_yes": "{team}: обидві команди забивали в {x} з останніх {k} {venue} матчів",
        "btts_no": "{team}: хоча б одна команда не забила в {x} з останніх {k} {venue} матчів",
        "failed": "{team} не забивали в {x} з останніх {k} {venue} матчів",
        "clean": "{team} зберегли ворота «сухими» в {x} з останніх {k} {venue} матчів",
        "league_total": "У цій лізі {v} матчів закінчуються тоталом більше {line}",
        "league_btts": "У цій лізі обидві команди забивають у {v} матчів",
        "corners_avg": "В останніх {k} {venue} матчах {team} в середньому було {v} кутових сумарно",
        "corners_line": "{team}: у {x} з останніх {k} {venue} матчів — {cond}",
        "c_over": "більше {line} кутових", "c_under": "менше {line} кутових",
        "won": "{team} виграли {x} з останніх {k} {venue} матчів",
        "drew": "{team} зіграли внічию {x} з останніх {k} {venue} матчів",
        "lost": "{team} програли {x} з останніх {k} {venue} матчів",
        "unbeaten": "{team} не програли в {x} з останніх {k} {venue} матчів",
        "not_drawn": "У {x} з останніх {k} {venue} матчів {team} був переможець",
        "position": "{team} — {v} місце в таблиці",
        "handicap": "З форою {line} ставка на {team} зіграла б у {x} з останніх {k} {venue} матчів",
        "handicap_push": "З форою {line} ставка на {team} зіграла б у {x} і повернулася б у {y} з останніх {k} {venue} матчів",
        "score_seen": "{team}: рахунок {score} був у {x} з останніх {k} {venue} матчів",
        "ordinal": lambda n: f"{n}-е",
    },
}


def _ok(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _recent(history: pd.DataFrame | None, venue: str) -> pd.DataFrame:
    """The last WINDOW played matches of a team at one venue (newest first)."""
    if history is None or history.empty:
        return pd.DataFrame()
    rows = history.loc[(history["venue"] == venue) & history["gf"].notna()]
    return rows.head(WINDOW)


def _as_match(rows: pd.DataFrame, venue: str) -> tuple[np.ndarray, np.ndarray]:
    """(home goals, away goals) of each match, whichever side the team was on."""
    gf, ga = rows["gf"].to_numpy(float), rows["ga"].to_numpy(float)
    return (gf, ga) if venue == "H" else (ga, gf)


def market_facts(market: str, selection: str, home: str, away: str, home_history: pd.DataFrame | None,
                 away_history: pd.DataFrame | None, row=None, lang: str = "en", limit: int = 4) -> list[str]:
    """Up to ``limit`` factual statements about this exact market and selection."""
    tx = TEXT.get(lang, TEXT["en"])
    get = row.get if row is not None and hasattr(row, "get") else (lambda k, d=None: d)
    group, line = market_group(market), market_line(market)
    sides = [(home, "H", _recent(home_history, "H")), (away, "A", _recent(away_history, "A"))]
    facts: list[str] = []

    def say(key, team, venue, **kw):
        facts.append(tx[key].format(team=team, venue=tx["venue"][venue], **kw))

    for team, venue, rows in sides:
        if rows.empty:
            continue
        k = len(rows)
        hg, ag = _as_match(rows, venue)
        if group == "TOTAL":
            hit = (hg + ag > line) if selection == "OVER" else (hg + ag < line)
            cond = tx["over" if selection == "OVER" else "under"].format(line=f"{line:g}")
            say("total", team, venue, x=int(hit.sum()), k=k, cond=cond)
        elif group == "BTTS":
            both = (hg > 0) & (ag > 0)
            if selection == "YES":
                say("btts_yes", team, venue, x=int(both.sum()), k=k)
            else:
                say("btts_no", team, venue, x=int((~both).sum()), k=k)
        elif group == "CORNERS":
            total = (rows["corners_for"] + rows["corners_against"]).dropna()
            if len(total):
                hit = (total > line) if selection == "OVER" else (total < line)
                cond = tx["c_over" if selection == "OVER" else "c_under"].format(line=f"{line:g}")
                say("corners_line", team, venue, x=int(hit.sum()), k=len(total), cond=cond)
        elif group in ("1X2", "DOUBLE_CHANCE"):
            res = rows["result"]
            wins, draws, losses = int((res == "W").sum()), int((res == "D").sum()), int((res == "L").sum())
            mine = (selection in ("H", "1X", "12") and venue == "H") or (selection in ("A", "X2", "12") and venue == "A")
            if selection == "D":
                say("drew", team, venue, x=draws, k=k)
            elif group == "1X2":
                say("won" if mine else "lost", team, venue, x=wins if mine else losses, k=k)
            elif selection == "12":
                say("not_drawn", team, venue, x=wins + losses, k=k)
            elif mine:
                say("unbeaten", team, venue, x=wins + draws, k=k)
            else:
                say("lost", team, venue, x=losses, k=k)
        elif group in ("HANDICAP", "EHANDICAP"):
            if selection == "D" or (selection == "H") != (venue == "H"):
                continue                      # the backed team's own matches only
            outcomes = [settle_selection(market, selection, h, a) for h, a in zip(hg, ag)]
            shown = f"{line:+g}" if selection == "H" else f"{-line:+g}"
            pushes = outcomes.count("PUSH")
            say("handicap_push" if pushes else "handicap", team, venue, line=shown,
                x=outcomes.count("WON"), y=pushes, k=k)
        elif group == "EXACT_SCORE":
            h, a = (int(x) for x in selection.split("-"))
            say("score_seen", team, venue, score=f"{h}:{a}", x=int(((hg == h) & (ag == a)).sum()), k=k)

    # goal levels behind goal markets
    if group in ("TOTAL", "BTTS", "EXACT_SCORE"):
        for team, venue, rows in sides:
            if not rows.empty:
                say("goals_for", team, venue, v=f"{rows['gf'].mean():.2f}", k=len(rows))
        if group == "TOTAL" and line == 2.5 and _ok(get("league_over25_rate")):
            facts.append(tx["league_total"].format(v=f"{float(get('league_over25_rate')) * 100:.0f}%", line="2.5"))
        if group == "BTTS" and _ok(get("league_btts_rate")):
            facts.append(tx["league_btts"].format(v=f"{float(get('league_btts_rate')) * 100:.0f}%"))
    if group == "CORNERS":
        for team, venue, rows in sides:
            total = (rows["corners_for"] + rows["corners_against"]).dropna() if not rows.empty else pd.Series()
            if len(total):
                say("corners_avg", team, venue, v=f"{total.mean():.1f}", k=len(total))
    if group in ("1X2", "DOUBLE_CHANCE", "HANDICAP", "EHANDICAP"):
        for side, team in (("h", home), ("a", away)):
            pos = get(f"{side}_position")
            if _ok(pos):
                facts.append(tx["position"].format(team=team, v=tx["ordinal"](int(pos))))
    return facts[:limit]
