"""Leagues: table, upcoming matches, recent results and prediction performance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.analytics.performance import group_summary, summarize
from src.analytics.teams import league_table
from src.leagues import enabled_leagues, get_league, league_label
from src.markets.markets import MARKET_TITLES
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, form_chips, guard, kpi_row, match_url, num,
                               page_header, pct, plotly, section, signed, team_url, tone)
from src.ui.theme import COLORS


def _table_html(table: pd.DataFrame, country: str) -> str:
    head = ("<tr><th>#</th><th>Team</th><th class='num'>P</th><th class='num'>W</th><th class='num'>D</th>"
            "<th class='num'>L</th><th class='num'>GF</th><th class='num'>GA</th><th class='num'>GD</th>"
            "<th class='num'>Pts</th><th>Form</th></tr>")
    rows = "".join(
        f"<tr><td>{r.Pos}</td><td><a href='{team_url(r.Team, country)}' target='_self' style='color:#e2e8f0'>"
        f"{esc(r.Team)}</a></td><td class='num'>{r.P}</td><td class='num'>{r.W}</td><td class='num'>{r.D}</td>"
        f"<td class='num'>{r.L}</td><td class='num'>{r.GF}</td><td class='num'>{r.GA}</td>"
        f"<td class='num'>{r.GD:+d}</td><td class='num'><b>{r.Pts}</b></td><td>{form_chips(r.Form)}</td></tr>"
        for r in table.itertuples())
    return f"<div class='card'><table class='mkt'>{head}{rows}</table></div>"


def _match_rows(matches: pd.DataFrame, played: bool) -> str:
    out = []
    for r in matches.itertuples():
        centre = f"{int(r.home_goals)}–{int(r.away_goals)}" if played else (r.time or "—")
        out.append(f"<a class='rm' style='grid-template-columns:80px 1fr 56px 1fr;color:#e2e8f0' "
                   f"href='{match_url(r.match_id)}' target='_self'><span class='d'>{date_text(r.date, '%a %d %b')}</span>"
                   f"<span style='text-align:right'>{esc(r.home_team)}</span><span class='s' style='text-align:center'>"
                   f"{centre}</span><span>{esc(r.away_team)}</span></a>")
    return "<div class='card'>" + "".join(out) + "</div>"


@guard
def render() -> None:
    page_header("Leagues", "Standings, fixtures, results and how predictions performed")
    features = data.features()
    if features.empty:
        empty_state("No data", "Run python scripts/run_pipeline.py")
        return
    codes = [lg.code for lg in enabled_leagues() if lg.code in set(features["league"])]
    default = st.query_params.get("league", "E0")
    c = st.columns([2, 1, 3])
    league = c[0].selectbox("League", codes, index=codes.index(default) if default in codes else 0,
                            format_func=lambda x: league_label(x, True))
    seasons = sorted(features.loc[features["league"] == league, "season"].unique(), reverse=True)
    season = c[1].selectbox("Season", seasons, format_func=config.season_label)
    lg = get_league(league)
    lf = features.loc[(features["league"] == league) & (features["season"] == season)]

    played = lf.loc[lf["played"].astype(bool)]
    goals = played["home_goals"] + played["away_goals"]
    kpi_row([
        ("Matches played", f"{len(played)}", f"{int((~lf['played'].astype(bool)).sum())} upcoming fixtures"),
        ("Goals / match", num(goals.mean()), f"home {num(played['home_goals'].mean())} · away {num(played['away_goals'].mean())}"),
        ("Home / Draw / Away", f"{pct((played['result'] == 'H').mean())} · {pct((played['result'] == 'D').mean())} · "
                               f"{pct((played['result'] == 'A').mean())}", "result distribution"),
        ("Over 2.5 · BTTS", f"{pct((goals > 2.5).mean())} · {pct(((played['home_goals'] > 0) & (played['away_goals'] > 0)).mean())}", ""),
        ("Corners / match", num(played["total_corners"].mean(), 1), "" if played["total_corners"].notna().any()
         else "no corner data"),
    ])

    left, right = st.columns([3, 2], gap="medium")
    with left:
        section("League table", config.season_label(season))
        table = league_table(features, league, season)
        if table.empty:
            empty_state("No matches played yet this season")
        else:
            st.html(_table_html(table, lg.country))
    with right:
        upcoming = lf.loc[~lf["played"].astype(bool)].sort_values(["date", "time"]).head(12)
        section("Upcoming matches")
        if upcoming.empty:
            st.caption("No published fixtures.")
        else:
            st.html(_match_rows(upcoming, False))
        section("Recent results")
        recent = played.sort_values(["date", "time"], ascending=False).head(12)
        if recent.empty:
            st.caption("No results yet.")
        else:
            st.html(_match_rows(recent, True))

    section("Prediction performance in this league")
    picks = data.picks()
    lp = picks.loc[picks["league"] == league] if not picks.empty else picks
    if lp.empty:
        st.caption("No predictions for this league yet.")
        return
    s = summarize(lp.loc[lp["role"] == "market"])
    m = summarize(lp.loc[lp["role"] == "main"])
    kpi_row([
        ("Market predictions", f"{s['settled']:,}", f"hit rate {pct(s['hit_rate'], 1)}"),
        ("Market ROI", signed(s["roi"], 1, "%"), f"profit {signed(s['profit'])} u", tone(s["roi"])),
        ("Main predictions", f"{m['settled']:,}", f"hit rate {pct(m['hit_rate'], 1)}"),
        ("Main ROI", signed(m["roi"], 1, "%"), f"profit {signed(m['profit'])} u", tone(m["roi"])),
    ])
    by_market = group_summary(lp.loc[lp["role"] == "market"], "group")
    if not by_market.empty:
        fig = go.Figure()
        fig.add_bar(x=by_market["group"].map(MARKET_TITLES), y=by_market["hit_rate"], name="Hit rate",
                    marker_color=COLORS["home"], text=[pct(v) for v in by_market["hit_rate"]], textposition="outside")
        fig.update_layout(title="Hit rate by market", yaxis_tickformat=".0%", yaxis_range=[0, 1])
        plotly(fig, height=300)


render()
