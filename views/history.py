"""Prediction history: results of stored predictions with filters."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config
from src.analytics.performance import DATE_PRESETS, PERIODS, preset_range, summarize, timeline
from src.leagues import league_codes
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, guard, html_table, kpi_row, match_url, num,
                               page_header, pct, signed, status_badge, tone)
from src.ui.i18n import t
from src.ui.labels import MARKET_GROUP_TITLES, group_title, league_label, market_title, selection_label

ROLES = {"main": "Main predictions", "risk": "Risk predictions", "market": "Market forecasts", "all": "All"}
PAGE_SIZE = 50


def filter_picks(picks: pd.DataFrame, start, end, leagues, groups, role, statuses, sources=None,
                 odds_only: bool = False) -> pd.DataFrame:
    df = picks.loc[(picks["date"] >= pd.Timestamp(start)) & (picks["date"] <= pd.Timestamp(end))]
    if leagues:
        df = df.loc[df["league"].isin(leagues)]
    if groups:
        df = df.loc[df["group"].isin(groups)]
    if role and role != "all":
        df = df.loc[df["role"] == role]
    if statuses:
        df = df.loc[df["status"].isin(statuses)]
    if sources:
        df = df.loc[df["source"].isin(sources)]
    if odds_only:
        df = df.loc[df["odds"].notna()]
    return df


def default_statuses(show_upcoming: bool, advanced: list[str] | None = None) -> list[str]:
    """Completed predictions only, unless upcoming are requested or advanced statuses chosen."""
    if advanced:
        return advanced
    return ["WON", "LOST"] + (["UPCOMING"] if show_upcoming else [])


@guard
def render() -> None:
    page_header(t("Prediction History"), t("Prediction results and performance history."))
    picks = data.picks()
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return

    first = picks["date"].min()
    season_start = pd.Timestamp(year=config.current_season(), month=7, day=1)
    with st.container(border=True):
        c = st.columns([1.5, 1.5, 1.6, 1.6, 1.3])
        preset = c[0].selectbox(t("Period"), DATE_PRESETS, index=DATE_PRESETS.index("Last 30 days"),
                                format_func=t, key="h_preset")
        custom = None
        if preset == "Custom":
            custom = c[0].date_input(t("Date range"), value=(first.date(), date.today()), format="DD.MM.YYYY",
                                     key="h_custom")
        role = c[1].selectbox(t("Prediction type"), list(ROLES), format_func=lambda r: t(ROLES[r]), key="h_role")
        leagues = c[2].multiselect(t("League"), league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder=t("All leagues"), key="h_leagues")
        groups = c[3].multiselect(t("Market"), list(MARKET_GROUP_TITLES), format_func=group_title,
                                  placeholder=t("All markets"), key="h_groups")
        show_upcoming = c[4].toggle(t("Show upcoming"), key="h_upcoming")
        with st.expander(t("Advanced filters"), icon=":material/tune:"):
            a = st.columns(3)
            statuses = a[0].multiselect(t("Status"), ["WON", "LOST", "UPCOMING", "VOID"],
                                        format_func=lambda s: t({"WON": "Won", "LOST": "Lost",
                                                                 "UPCOMING": "Upcoming", "VOID": "Void"}[s]),
                                        key="h_status", placeholder=t("Won and lost"))
            sources = a[1].multiselect(t("Prediction origin"), ["live", "backfill", "backtest"],
                                       format_func=lambda s: t({"live": "Live (before kick-off)",
                                                                "backfill": "Backfill (current season)",
                                                                "backtest": "Backtest (test season)"}[s]),
                                       key="h_sources", placeholder=t("All"))
            odds_only = a[2].toggle(t("Only predictions with odds"), key="h_odds_only")

    start, end = preset_range(preset, date.today(), first, custom, season_start)
    df = filter_picks(picks, start, end, leagues, groups, role, default_statuses(show_upcoming, statuses),
                      sources, odds_only)
    s = summarize(df)
    kpi_row([
        (t("Predictions"), f"{s['settled']:,}", t("{a} won · {b} lost", a=s["won"], b=s["lost"])),
        (t("Hit rate"), pct(s["hit_rate"], 1), ""),
        (t("Profit"), f"{signed(s['profit'])} u" if s["bets"] else "—", t("{n} predictions with odds", n=s["bets"]),
         tone(s["profit"])),
        (t("ROI"), signed(s["roi"], 1, "%"), t("profit / amount staked"), tone(s["roi"])),
        (t("Average odds"), num(s["avg_odds"]), ""),
    ])
    note = t("Profit and ROI are calculated with a fixed stake of 1 unit per prediction, "
             "only for predictions with real bookmaker odds.")
    st.html(f'<div class="note">{note}</div>')

    if df.empty:
        empty_state(t("No completed predictions for this period."))
        return

    group_by = st.segmented_control(t("Group by"), list(PERIODS), default="Weekly", format_func=t,
                                    key="h_group_by") or "Weekly"
    tl = timeline(df, group_by)
    if not tl.empty:
        with st.expander(t("Summary by period"), icon=":material/calendar_month:", expanded=False):
            view = pd.DataFrame({
                t("Period"): tl["period"].dt.strftime("%d.%m.%Y"), t("Predictions"): tl["predictions"],
                t("Hit rate"): tl["hit_rate"], t("Profit"): tl["profit"], t("ROI"): tl["roi"],
            }).iloc[::-1]
            st.html(html_table(view, formats={t("Hit rate"): lambda v: pct(v, 1),
                                              t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>",
                                              t("ROI"): lambda v: f"<span class='{tone(v)}'>{signed(v, 1, '%')}</span>"},
                               align_right=(t("Predictions"), t("Hit rate"), t("Profit"), t("ROI"))))

    table = df.sort_values(["date", "time", "match_id"], ascending=[False, False, True])
    pages = max(1, (len(table) - 1) // PAGE_SIZE + 1)
    page = st.number_input(t("Page"), 1, pages, 1, key="h_page") if pages > 1 else 1
    chunk = table.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    roles = {"main": t("Main"), "risk": t("Risk"), "market": t("Forecast")}
    view = pd.DataFrame({
        t("Date"): [date_text(d, "%d.%m.%Y") for d in chunk["date"]],
        t("League"): [league_label(x) for x in chunk["league"]],
        t("Match"): [f"<a href='{match_url(m)}' target='_self'>{esc(h)} – {esc(a)}</a>"
                     for m, h, a in zip(chunk["match_id"], chunk["home_team"], chunk["away_team"])],
        t("Type"): chunk["role"].map(roles),
        t("Market"): chunk["market"].map(market_title),
        t("Prediction"): [selection_label(m, s_, h, a) for m, s_, h, a in
                          zip(chunk["market"], chunk["selection"], chunk["home_team"], chunk["away_team"])],
        t("Probability"): chunk["probability"],
        t("Odds"): chunk["odds"],
        t("Score"): [f"{int(h)}–{int(a)}" if h == h else "" for h, a in zip(chunk["home_goals"], chunk["away_goals"])],
        t("Result"): chunk["status"],
        t("Profit"): chunk["profit"],
    })
    st.html(html_table(view, formats={
        t("Match"): lambda v: v, t("Probability"): lambda v: pct(v, 1), t("Odds"): num,
        t("Result"): status_badge, t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>"},
        align_right=(t("Probability"), t("Odds"), t("Profit"))))
    st.caption(t("Page {p} of {n} · {k} predictions", p=page, n=pages, k=f"{len(table):,}"))


render()
