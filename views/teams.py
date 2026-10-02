"""Teams: search a team and see its form, statistics and trends."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.analytics.teams import all_teams, form_summary, league_table, team_matches
from src.leagues import get_league
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, form_chips, guard, kpi_row, match_url, num,
                               page_header, pct, plotly, section)
from src.ui.theme import COLORS


@st.cache_data(show_spinner=False)
def _teams_index(_key: int) -> pd.DataFrame:
    return all_teams(data.features())


def _stat_grid(s: dict) -> str:
    if not s.get("games"):
        return "<span class='muted small'>No matches.</span>"
    items = [("Matches", s["games"]), ("W-D-L", f"{s['wins']}-{s['draws']}-{s['losses']}"),
             ("Points / game", num(s["ppg"])), ("Goals", num(s["gf"])), ("Conceded", num(s["ga"])),
             ("Shots", num(s["shots_for"], 1)), ("On target", num(s["sot_for"], 1)),
             ("Corners", num(s["corners_for"], 1)), ("Corners against", num(s["corners_against"], 1)),
             ("Over 2.5", pct(s["over25_rate"])), ("BTTS", pct(s["btts_rate"])),
             ("Clean sheets", pct(s["clean_sheet_rate"]))]
    return ("<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:.6rem 1rem'>"
            + "".join(f"<div><div class='muted small'>{k}</div><b style='font-size:1.1rem'>{v}</b></div>"
                      for k, v in items) + "</div>")


@guard
def render() -> None:
    page_header("Teams", "Search a team to see its form and statistics")
    features = data.features()
    if features.empty:
        empty_state("No data")
        return
    teams = _teams_index(len(features))
    keys = [f"{r.team}|{r.country}" for r in teams.itertuples()]
    labels = {f"{r.team}|{r.country}": f"{r.team} — {r.league_name} ({r.country})" for r in teams.itertuples()}
    param = st.query_params.get("team")
    current = f"{param}|{st.query_params.get('country', '')}" if param else None
    if current not in labels and param:
        current = next((k for k in keys if k.split("|")[0] == param), None)
    choice = st.selectbox("Search team", keys, index=keys.index(current) if current in keys else None,
                          format_func=labels.get, placeholder="Type a team name, e.g. Arsenal")
    if not choice:
        empty_state("Choose a team", "Start typing a club name in the search box above.")
        return
    if choice != current:
        team, country = choice.split("|")
        st.query_params.update({"team": team, "country": country})
    team, country = choice.split("|")
    tm = team_matches(features, team, country)
    info = teams.loc[(teams["team"] == team) & (teams["country"] == country)].iloc[0]
    lg = get_league(info["league"])
    season = int(features.loc[(features["league"] == info["league"]), "season"].max())
    table = league_table(features, info["league"], season)
    row = table.loc[table["Team"] == team] if not table.empty else table

    st.html(f"<div class='match-header'><div class='mh-meta'><span>{esc(lg.country)} · {esc(lg.name)}</span>"
            f"<span>Season {config.season_label(season)}</span></div><div class='mh-team' style='margin-top:.5rem'>"
            f"{esc(team)}</div><div style='margin-top:.5rem'>{form_chips(''.join(tm['result'].head(5)))}</div></div>")
    if not row.empty:
        r = row.iloc[0]
        kpi_row([("Position", f"{r['Pos']}", f"of {len(table)}"), ("Points", f"{r['Pts']}", f"{r['P']} played"),
                 ("Record", f"{r['W']}-{r['D']}-{r['L']}", "W-D-L"),
                 ("Goals", f"{r['GF']}:{r['GA']}", f"GD {r['GD']:+d}")])

    section("Form")
    n = st.segmented_control("Window", [5, 10, 15], default=10, format_func=lambda x: f"Last {x}",
                             key="team_n") or 10
    last = tm.head(n)
    st.html(f"<div class='card'>{_stat_grid(form_summary(last))}</div>")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section("Home")
        st.html(f"<div class='card'>{_stat_grid(form_summary(tm.loc[tm['venue'] == 'H'].head(n)))}</div>")
    with c2:
        section("Away")
        st.html(f"<div class='card'>{_stat_grid(form_summary(tm.loc[tm['venue'] == 'A'].head(n)))}</div>")

    section("Trends", "5-match rolling averages over the last 30 matches")
    chron = tm.head(30).iloc[::-1]
    if len(chron) >= 5:
        roll = chron[["gf", "ga", "shots_for", "sot_for", "corners_for", "points"]].rolling(5, min_periods=3).mean()
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_scatter(x=chron["date"], y=roll["gf"], name="Scored", line=dict(color=COLORS["positive"]))
            fig.add_scatter(x=chron["date"], y=roll["ga"], name="Conceded", line=dict(color=COLORS["negative"]))
            fig.update_layout(title="Goals per game", legend=dict(orientation="h", y=1.12))
            plotly(fig, height=280)
        with c4:
            fig = go.Figure()
            fig.add_scatter(x=chron["date"], y=roll["shots_for"], name="Shots", line=dict(color=COLORS["home"]))
            fig.add_scatter(x=chron["date"], y=roll["sot_for"], name="On target", line=dict(color=COLORS["derived"]))
            fig.add_scatter(x=chron["date"], y=roll["corners_for"], name="Corners", line=dict(color=COLORS["away"]))
            fig.update_layout(title="Shots and corners per game", legend=dict(orientation="h", y=1.12))
            plotly(fig, height=280)
    else:
        st.caption("Not enough matches for trend charts.")

    section("Matches")
    upcoming = features.loc[~features["played"].astype(bool) & (features["country"] == country)
                            & ((features["home_team"] == team) | (features["away_team"] == team))]
    rows = [f"<a class='rm' style='grid-template-columns:26px 80px 22px 1fr 54px;color:#e2e8f0' "
            f"href='{match_url(m.match_id)}' target='_self'><span class='chip' style='background:#334155'>·</span>"
            f"<span class='d'>{date_text(m.date, '%d %b %y')}</span><span class='v'>{'H' if m.home_team == team else 'A'}</span>"
            f"<span>{esc(m.away_team if m.home_team == team else m.home_team)}</span><span class='s'>{m.time or ''}</span></a>"
            for m in upcoming.sort_values("date").itertuples()]
    rows += [f"<a class='rm' style='color:#e2e8f0' href='{match_url(m.match_id)}' target='_self'>"
             f"<span class='chip {m.result}'>{m.result}</span><span class='d'>{date_text(m.date, '%d %b %y')}</span>"
             f"<span class='v'>{m.venue}</span><span>{esc(m.opponent)}</span><span class='s'>{int(m.gf)}–{int(m.ga)}</span></a>"
             for m in tm.head(max(n, 15)).itertuples()]
    st.html("<div class='card'>" + "".join(rows) + "</div>")


render()
