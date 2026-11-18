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
                               page_header, pct, plotly, section, team_badge)
from src.ui.i18n import current_language, t
from src.ui.labels import country_label, league_name
from src.ui.flags import flag
from src.ui.theme import colors, design_name
from src.ui.timefmt import with_local_time

PERIODS = {"5": "Last 5", "10": "Last 10", "15": "Last 15", "season": "Current season", "all": "All available"}
MAX_LIST = 60


@st.cache_data(show_spinner=False)
def _teams_index(_key: int) -> pd.DataFrame:
    return all_teams(data.features())


def select_period(tm: pd.DataFrame, period: str, season: int) -> pd.DataFrame:
    """Team matches of a period (newest first)."""
    if period.isdigit():
        return tm.head(int(period))
    if period == "season":
        return tm.loc[tm["season"] == season]
    return tm


def _stat_grid(s: dict) -> str:
    """Grouped team statistics: results · attack · defence · corners."""
    if not s.get("games"):
        return f"<span class='muted small'>{t('No matches.')}</span>"
    groups = [
        (t("Results"), [(t("Matches"), s["games"]), (t("W-D-L"), f"{s['wins']}-{s['draws']}-{s['losses']}"),
                        (t("Points / game"), num(s["ppg"]))]),
        (t("Attack"), [(t("Goals / game"), num(s["gf"])), (t("Shots / game"), num(s["shots_for"], 1)),
                       (t("Shots on target / game"), num(s["sot_for"], 1)),
                       (t("Over 2.5") + " · BTTS", f"{pct(s['over25_rate'])} · {pct(s['btts_rate'])}")]),
        (t("Defence"), [(t("Conceded / game"), num(s["ga"])), (t("Clean sheets"), pct(s["clean_sheet_rate"])),
                        (t("Shots conceded / game"), num(s["shots_against"], 1))]),
        (t("Corners"), [(t("Corners / game"), num(s["corners_for"], 1)),
                        (t("Corners against"), num(s["corners_against"], 1))]),
    ]
    cells = []
    for title, items in groups:
        body = "".join(f"<div class='st-row'><span class='muted small'>{k}</span><b>{v}</b></div>" for k, v in items)
        cells.append(f"<div class='st-group'><div class='why-h'>{title}</div>{body}</div>")
    return "<div class='st-grid'>" + "".join(cells) + "</div>"


HOVER = "%{x|%d.%m.%Y}<br>%{fullData.name}: %{y:.2f}<extra></extra>"     # values: at most two decimals


def trend_layout(fig: go.Figure, title: str, legend: bool = True) -> go.Figure:
    """Title on top, the legend in one row just under it (inside the chart's own space, so no entry is cut
    at the bottom edge), two-decimal hover values."""
    fig.update_traces(hovertemplate=HOVER)
    # Plotly sizes the legend with the fallback font before the web font has loaded and then clips the wider
    # text ("Concedec"): a fixed entry width with room for the longest name
    longest = max((len(tr.name or "") for tr in fig.data), default=0)
    fig.update_layout(title=dict(text=title, y=0.97, yanchor="top"), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                                  bgcolor="rgba(0,0,0,0)", entrywidth=int(longest * 8.2 + 48),
                                  entrywidthmode="pixels"),
                      margin=dict(t=74 if legend else 52, b=36, l=40, r=12),
                      xaxis=dict(tickformat="%d.%m", nticks=6), yaxis=dict(tickformat=".1f"))
    return fig


def _trend_charts(matches: pd.DataFrame) -> None:
    chron = matches.iloc[::-1]
    if len(chron) < 3:
        st.caption(t("Not enough matches for trend charts."))
        return
    window = 3 if len(chron) <= 10 else 5
    roll = chron[["gf", "ga", "corners_for", "corners_against", "points"]].rolling(window, min_periods=2).mean()
    c = colors()
    cols = st.columns(3)
    with cols[0]:
        fig = go.Figure()
        fig.add_scatter(x=chron["date"], y=roll["gf"], name=t("Scored"), line=dict(color=c["pos"]))
        fig.add_scatter(x=chron["date"], y=roll["ga"], name=t("Conceded"), line=dict(color=c["neg"]))
        plotly(trend_layout(fig, t("Goals per game")), height=310)
    with cols[1]:
        fig = go.Figure()
        fig.add_scatter(x=chron["date"], y=roll["corners_for"], name=t("Corners"), line=dict(color=c["away"]))
        fig.add_scatter(x=chron["date"], y=roll["corners_against"], name=t("Corners against"),
                        line=dict(color=c["faint"], dash="dot"))
        plotly(trend_layout(fig, t("Corners per game")), height=310)
    with cols[2]:
        fig = go.Figure(go.Scatter(x=chron["date"], y=roll["points"], line=dict(color=c["home"]), name=t("Points")))
        fig = trend_layout(fig, t("Points per game"), legend=False)
        fig.update_layout(yaxis_range=[0, 3])
        plotly(fig, height=310)
    st.caption(t("{n}-match average: each point averages that match and the {k} before it, so one unusual "
                 "game does not dominate the line.", n=window, k=window - 1))


def _ordinal(n: int) -> str:
    if current_language() != "en":
        return f"{n}"
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _team_header(team: str, country: str, lg, season: int, tm: pd.DataFrame, row: pd.DataFrame, n_teams: int) -> None:
    """v6.3 header: badge, name, flag · league · season, last five results and the table position."""
    if design_name() != "v6.3":                        # earlier designs keep their header (rollback)
        st.html(f"<div class='match-header'><div class='mh-meta'><span>{esc(country_label(lg.country))} · "
                f"{esc(league_name(lg.code))}</span><span>{t('Season')} {config.season_label(season)}</span></div>"
                f"<div class='mh-team' style='margin-top:.5rem;display:flex;align-items:center;gap:.6rem'>"
                f"{team_badge(country, team, 40)}{esc(team)}</div><div style='margin-top:.5rem'>"
                f"{form_chips(''.join(tm['result'].head(5)))}</div></div>")
        return
    pos = ""
    if not row.empty:
        p = int(row.iloc[0]["Pos"])
        pos = (f"<div class='tpos'><div class='l'>{t('League position')}</div><div class='v'>{_ordinal(p)}</div>"
               f"<div class='l'>{t('of {n}', n=n_teams)}</div></div>")
    st.html(f"<div class='tm-head'>{team_badge(country, team, 64)}<div><div class='nm'>{esc(team)}</div>"
            f"<div class='meta'>{flag(lg.country, country_label(lg.country), 18)}"
            f"<span>{esc(league_name(lg.code))} · {esc(country_label(lg.country))}</span><span>·</span>"
            f"<span>{t('Season')} {config.season_label(season)}</span></div>"
            f"<div style='margin-top:.45rem'>{form_chips(''.join(tm['result'].head(5)))}</div></div>{pos}</div>")


@guard
def render() -> None:
    page_header(t("Teams"))
    features = data.features()
    if features.empty:
        empty_state(t("No data available."))
        return
    teams = _teams_index(len(features))
    keys = [f"{r.team}|{r.country}" for r in teams.itertuples()]
    labels = {f"{r.team}|{r.country}": f"{r.team} — {league_name(r.league)} ({country_label(r.country)})"
              for r in teams.itertuples()}
    param = st.query_params.get("team")
    current = f"{param}|{st.query_params.get('country', '')}" if param else None
    if current not in labels and param:
        current = next((k for k in keys if k.split("|")[0] == param), None)
    choice = st.container(key="team_search").selectbox(t("Search team"), keys, index=keys.index(current) if current in keys else None,
                          format_func=labels.get, placeholder=t("Type a team name, e.g. Arsenal"))
    if not choice:
        empty_state(t("Choose a team"), t("Start typing a club name in the search box above."))
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

    _team_header(team, country, lg, season, tm, row, len(table))
    if not row.empty:
        r = row.iloc[0]
        position = [] if design_name() == "v6.3" else [(t("League position"), f"{r['Pos']}", t("of {n}", n=len(table)))]
        kpi_row(position + [(t("Points"), f"{r['Pts']}", t("{n} played", n=r["P"])),
                 (t("W-D-L"), f"{r['W']}-{r['D']}-{r['L']}", ""),
                 (t("Goals"), f"{r['GF']}:{r['GA']}", f"{t('GD')} {r['GD']:+d}")])

    period = st.segmented_control(t("Period"), list(PERIODS), default="10", format_func=lambda k: t(PERIODS[k]),
                                  key="team_period") or "10"
    chosen = select_period(tm, period, season)
    section(t("Form"), t(PERIODS[period]))
    st.html(f"<div class='card'>{_stat_grid(form_summary(chosen))}</div>")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section(t("Home"))
        st.html(f"<div class='card'>{_stat_grid(form_summary(chosen.loc[chosen['venue'] == 'H']))}</div>")
    with c2:
        section(t("Away"))
        st.html(f"<div class='card'>{_stat_grid(form_summary(chosen.loc[chosen['venue'] == 'A']))}</div>")

    section(t("Trends"))
    _trend_charts(chosen)

    section(t("Matches"))
    upcoming = features.loc[~features["played"].astype(bool) & (features["country"] == country)
                            & ((features["home_team"] == team) | (features["away_team"] == team))]
    rows = [f"<a class='rm' href='{match_url(m.match_id)}' target='_self'><span class='chip' "
            f"style='background:var(--track)'>·</span><span class='d'>{date_text(m.date, '%d.%m.%y')}</span>"
            f"<span class='v'>{t('H') if m.home_team == team else t('A')}</span>"
            f"<span>{esc(m.away_team if m.home_team == team else m.home_team)}</span><span class='s'>{m.time or ''}</span></a>"
            for m in with_local_time(upcoming.sort_values("date")).assign(
                date=lambda d: d["local_date"], time=lambda d: d["local_time"]).itertuples()]
    pages = max(1, (len(chosen) - 1) // MAX_LIST + 1)
    page = st.number_input(t("Page"), 1, pages, 1, key="team_page") if pages > 1 else 1
    listed = chosen.iloc[(page - 1) * MAX_LIST: page * MAX_LIST]
    rows += [f"<a class='rm' href='{match_url(m.match_id)}' target='_self'>{form_chips(m.result)}"
             f"<span class='d'>{date_text(m.date, '%d.%m.%y')}</span><span class='v'>{t('H') if m.venue == 'H' else t('A')}</span>"
             f"<span>{esc(m.opponent)}</span><span class='s'>{int(m.gf)}:{int(m.ga)}</span></a>"
             for m in listed.itertuples()]
    st.html("<div class='card'>" + "".join(rows) + "</div>")
    if pages > 1:
        st.caption(t("Page {p} of {n} · {k} matches", p=page, n=pages, k=len(chosen)))


render()
