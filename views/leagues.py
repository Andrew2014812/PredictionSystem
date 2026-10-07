"""Leagues: table, upcoming matches, recent results and main-prediction performance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.analytics.performance import group_summary, summarize
from src.analytics.teams import league_table
from src.leagues import enabled_leagues, get_league
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, form_chips, guard, html_table, kpi_row, match_url,
                               num, page_header, pct, section, signed, team_url, tone)
from src.ui.i18n import t
from src.ui.labels import group_title, league_label


def _table_html(table: pd.DataFrame, country: str) -> str:
    head = (f"<tr><th>#</th><th>{t('Team')}</th><th class='num'>{t('P')}</th><th class='num'>{t('W')}</th>"
            f"<th class='num'>{t('D')}</th><th class='num'>{t('L')}</th><th class='num'>{t('GF')}</th>"
            f"<th class='num'>{t('GA')}</th><th class='num'>{t('GD')}</th><th class='num'>{t('Pts')}</th>"
            f"<th>{t('Form')}</th></tr>")
    rows = "".join(
        f"<tr><td>{r.Pos}</td><td><a href='{team_url(r.Team, country)}' target='_self'>{esc(r.Team)}</a></td>"
        f"<td class='num'>{r.P}</td><td class='num'>{r.W}</td><td class='num'>{r.D}</td><td class='num'>{r.L}</td>"
        f"<td class='num'>{r.GF}</td><td class='num'>{r.GA}</td><td class='num'>{r.GD:+d}</td>"
        f"<td class='num'><b>{r.Pts}</b></td><td>{form_chips(r.Form)}</td></tr>" for r in table.itertuples())
    return f"<div class='tablewrap'><table class='mkt'>{head}{rows}</table></div>"


def _match_rows(matches: pd.DataFrame, played: bool) -> str:
    out = []
    for r in matches.itertuples():
        centre = f"{int(r.home_goals)}–{int(r.away_goals)}" if played else (r.time or "—")
        out.append(f"<a class='rm' style='grid-template-columns:80px 1fr 56px 1fr' href='{match_url(r.match_id)}' "
                   f"target='_self'><span class='d'>{date_text(r.date, '%d.%m')}</span>"
                   f"<span style='text-align:right'>{esc(r.home_team)}</span><span class='s' style='text-align:center'>"
                   f"{centre}</span><span>{esc(r.away_team)}</span></a>")
    return "<div class='card'>" + "".join(out) + "</div>"


@guard
def render() -> None:
    page_header(t("Leagues"))
    features = data.features()
    if features.empty:
        empty_state(t("No data available."))
        return
    codes = [lg.code for lg in enabled_leagues() if lg.code in set(features["league"])]
    default = st.query_params.get("league", "E0")
    c = st.columns([2, 1, 3])
    league = c[0].selectbox(t("League"), codes, index=codes.index(default) if default in codes else 0,
                            format_func=lambda x: league_label(x, True))
    seasons = sorted(features.loc[features["league"] == league, "season"].unique(), reverse=True)
    season = c[1].selectbox(t("Season"), seasons, format_func=config.season_label)
    lg = get_league(league)
    lf = features.loc[(features["league"] == league) & (features["season"] == season)]

    played = lf.loc[lf["played"].astype(bool)]
    goals = played["home_goals"] + played["away_goals"]
    kpi_row([
        (t("Matches played"), f"{len(played)}", ""),
        (t("Goals / match"), num(goals.mean()), ""),
        (t("Home · Draw · Away"), f"{pct((played['result'] == 'H').mean())} · {pct((played['result'] == 'D').mean())} · "
                                  f"{pct((played['result'] == 'A').mean())}", ""),
        (t("Over 2.5 · BTTS"), f"{pct((goals > 2.5).mean())} · "
                               f"{pct(((played['home_goals'] > 0) & (played['away_goals'] > 0)).mean())}", ""),
        (t("Corners / match"), num(played["total_corners"].mean(), 1), ""),
    ])

    left, right = st.columns([3, 2], gap="medium")
    with left:
        section(t("League table"), config.season_label(season))
        table = league_table(features, league, season)
        if table.empty:
            empty_state(t("No matches played yet this season."))
        else:
            st.html(_table_html(table, lg.country))
    with right:
        upcoming = lf.loc[~lf["played"].astype(bool)].sort_values(["date", "time"]).head(12)
        section(t("Upcoming matches"))
        if upcoming.empty:
            st.caption(t("No upcoming fixtures published yet."))
        else:
            st.html(_match_rows(upcoming, False))
        section(t("Recent results"))
        recent = played.sort_values(["date", "time"], ascending=False).head(12)
        if recent.empty:
            st.caption(t("No results yet."))
        else:
            st.html(_match_rows(recent, True))

    section(t("FootPredict in this league"))
    picks = data.picks()
    lp = picks.loc[picks["league"] == league] if not picks.empty else picks
    main = summarize(lp.loc[lp["role"] == "main"]) if not lp.empty else None
    if not main or not main["settled"]:
        st.caption(t("No completed predictions for this league yet."))
        return
    kpi_row([
        (t("Matches predicted"), f"{main['settled']:,}", ""),
        (t("Main hit rate"), pct(main["hit_rate"], 1), ""),
        (t("Main profit"), f"{signed(main['profit'])} u" if main["bets"] else "—",
         t("{n} predictions with odds", n=main["bets"]), tone(main["profit"])),
        (t("Main ROI"), signed(main["roi"], 1, "%"), "", tone(main["roi"])),
    ])
    section(t("Market performance"))
    g = group_summary(lp.loc[lp["role"] == "market"], "group")
    if not g.empty:
        view = pd.DataFrame({t("Market"): g["group"].map(group_title), t("Predictions"): g["predictions"],
                             t("Hit rate"): g["hit_rate"], t("Profit"): g["profit"], t("ROI"): g["roi"]})
        st.html(html_table(view, formats={
            t("Hit rate"): lambda v: pct(v, 1), t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>",
            t("ROI"): lambda v: f"<span class='{tone(v)}'>{signed(v, 1, '%')}</span>"},
            align_right=(t("Predictions"), t("Hit rate"), t("Profit"), t("ROI"))))


render()
