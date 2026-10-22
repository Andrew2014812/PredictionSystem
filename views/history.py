"""Prediction history: results of stored predictions with filters."""
from __future__ import annotations


import pandas as pd
import streamlit as st

from src import config
from src.analytics.performance import ORIGIN_LABELS, PERIODS, period_label, select_dataset, summarize, timeline
from src.leagues import league_codes
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, guard, html_table, kpi_row, match_url, missing, num,
                               odds_text, page_header, pct, period_filter, section, signed, status_badge, tone)
from src.ui.i18n import t
from src.ui.timefmt import with_local_time
from src.ui.labels import MARKET_GROUP_TITLES, group_title, league_label, market_title, selection_label

ROLES = {"main": "Main predictions", "risk": "Risk predictions", "market": "Market forecasts"}
DATA_VIEWS = {"recorded": "Recorded", "backtest": "Backtest {season}", "both": "Both"}
DATA_HELP = ("Recorded = predictions saved before kick-off (the real track record). Backtest = the same "
             "production model run afterwards on this season's earlier matches, from pre-match data and real "
             "archived prices — an evaluation, not published bets. Their results are never added up.")
PAGE_SIZE = 50
SUMMARY_ROWS = 12


def filter_picks(picks: pd.DataFrame, start, end, leagues, groups, role, statuses=("WON", "LOST"),
                 sources=None, odds_only: bool = False) -> pd.DataFrame:
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


def default_statuses() -> list[str]:
    """History shows finished predictions (upcoming matches live on the Matches page); an Asian
    handicap refund is shown as "Refund" but does not count in the hit rate or ROI."""
    return ["WON", "LOST", "PUSH"]


@guard
def render() -> None:
    page_header(t("Prediction History"), t("How FootPredict's predictions ended."))
    all_picks = data.picks()
    season = config.current_season()
    origin = st.segmented_control(t("Data"), list(DATA_VIEWS), default="recorded", key="h_data",
                                  format_func=lambda k: t(DATA_VIEWS[k]).format(season=config.season_label(season)),
                                  help=t(DATA_HELP)) or "recorded"
    recorded = select_dataset(all_picks, "recorded")
    backtest = select_dataset(all_picks, "season_backtest")
    backtest = backtest.loc[backtest["season"] == season] if not backtest.empty else backtest
    picks = {"recorded": recorded, "backtest": backtest,
             "both": pd.concat([recorded, backtest], ignore_index=True)}[origin]
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return
    with st.container(border=True):
        c = st.columns([1.5, 1.5, 1.7, 1.7])
        start, end = period_filter(c[0], "h", picks["date"].min())
        role = c[1].selectbox(t("Prediction type"), list(ROLES), format_func=lambda r: t(ROLES[r]), key="h_role",
                              help=t("Main = the one prediction FootPredict recommends per match. Risk = an optional "
                                     "high-odds pick. Market forecasts = the model's most likely outcome in each "
                                     "market; they are estimates, not recommended bets."))
        leagues = c[2].multiselect(t("League"), league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder=t("All leagues"), key="h_leagues")
        groups = c[3].multiselect(t("Market"), list(MARKET_GROUP_TITLES), format_func=group_title,
                                  placeholder=t("All markets"), key="h_groups")

    df = filter_picks(picks, start, end, leagues, groups, role, default_statuses())
    # Results of different origin are never added up: with "both", one KPI row per origin.
    parts = [("live", "Recorded before kick-off"), ("backfill", "Backtest")] if origin == "both" else [(None, None)]
    for source, title in parts:
        part = df if source is None else df.loc[df["source"] == source]
        if title:
            st.html(f"<div class='why-h'>{t(title)}</div>")
        s = summarize(part)
        kpi_row([
            (t("Predictions"), f"{s['settled']:,}", t("{a} won · {b} lost", a=s["won"], b=s["lost"])),
            (t("Hit rate"), pct(s["hit_rate"], 1), ""),
            (t("Profit"), f"{signed(s['profit'])} {t('u')}" if s["bets"] else "—",
             t("{n} predictions with odds", n=s["bets"]), tone(s["profit"])),
            (t("ROI"), signed(s["roi"], 1, "%"), t("profit / amount staked"), tone(s["roi"])),
            (t("Average odds"), num(s["avg_odds"]), ""),
        ])
    note = t("Profit and ROI are calculated with a fixed stake of 1 unit per prediction, "
             "only for predictions with real bookmaker odds.")
    st.html(f'<div class="note">{note}</div>')

    if df.empty:
        empty_state(t("No completed predictions for this period."))
        return

    section(t("Summary by period"))
    group_by = st.segmented_control(t("Group results by"), list(PERIODS), default="Weekly", format_func=t,
                                    key="h_group_by", help=t("Changes how the summary below is grouped in time; "
                                                             "it does not filter the predictions.")) or "Weekly"
    frames = []
    for source in (["live", "backfill"] if origin == "both" else [None]):
        tl = timeline(df if source is None else df.loc[df["source"] == source], group_by)
        if not tl.empty:
            frames.append(tl.assign(origin=t(ORIGIN_LABELS[source]) if source else ""))
    if frames:
        tl = pd.concat(frames).sort_values(["period", "origin"])
        with st.container():
            view = pd.DataFrame({
                t("Period"): [period_label(p_, group_by) for p_ in tl["period"]],
                **({t("Data"): tl["origin"]} if origin == "both" else {}),
                t("Predictions"): tl["predictions"],
                t("Hit rate"): tl["hit_rate"], t("Profit"): tl["profit"], t("ROI"): tl["roi"],
            }).iloc[::-1]
            if len(view) > SUMMARY_ROWS and not st.toggle(t("Show all {n} periods", n=len(view)), key="h_all_periods"):
                view = view.head(SUMMARY_ROWS)
            st.html(html_table(view, formats={t("Hit rate"): lambda v: pct(v, 1),
                                              t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>",
                                              t("ROI"): lambda v: f"<span class='{tone(v)}'>{signed(v, 1, '%')}</span>"},
                               align_right=(t("Predictions"), t("Hit rate"), t("Profit"), t("ROI"))))

    table = df.sort_values(["date", "time", "match_id"], ascending=[False, False, True])
    pages = max(1, (len(table) - 1) // PAGE_SIZE + 1)
    page = st.number_input(t("Page"), 1, pages, 1, key="h_page") if pages > 1 else 1
    chunk = with_local_time(table.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE])   # visitor's date
    roles = {"main": t("Main"), "risk": t("Risk"), "market": t("Forecast")}
    view = pd.DataFrame({
        t("Date"): [date_text(d, "%d.%m.%Y") for d in chunk["local_date"]],
        t("League"): [league_label(x) for x in chunk["league"]],
        t("Match"): [f"<a href='{match_url(m)}' target='_self'>{esc(h)} – {esc(a)}</a>"
                     for m, h, a in zip(chunk["match_id"], chunk["home_team"], chunk["away_team"])],
        t("Type"): chunk["role"].map(roles),
        t("Data"): [f"<span class='origin {s_}'>{t(ORIGIN_LABELS.get(s_, s_))}</span>" for s_ in chunk["source"]],
        t("Market"): chunk["market"].map(market_title),
        t("Prediction"): [selection_label(m, s_, h, a) for m, s_, h, a in
                          zip(chunk["market"], chunk["selection"], chunk["home_team"], chunk["away_team"])],
        t("Probability"): chunk["probability"],
        t("Odds"): chunk["odds"],
        t("Score"): [f"{int(h)}:{int(a)}" if h == h else "" for h, a in zip(chunk["home_goals"], chunk["away_goals"])],
        t("Result"): chunk["status"],
        t("Profit"): chunk["profit"],
    })
    st.html(html_table(view, formats={
        t("Match"): lambda v: v, t("Data"): lambda v: v, t("Probability"): lambda v: pct(v, 1), t("Odds"): odds_text,
        t("Result"): status_badge,
        t("Profit"): lambda v: "—" if missing(v) else f"<span class='{tone(v)}'>{signed(v)}</span>"},
        align_right=(t("Probability"), t("Odds"), t("Profit"))))
    st.caption(t("Page {p} of {n} · {k} predictions", p=page, n=pages, k=f"{len(table):,}"))


render()
