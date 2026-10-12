"""Prediction history: results of stored predictions with filters."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config
from src.analytics.performance import (DATASETS, DATE_PRESETS, PERIODS, preset_range, select_dataset, summarize,
                                       timeline)
from src.leagues import league_codes
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, guard, html_table, kpi_row, match_url, num,
                               missing, odds_text, page_header, pct, section, signed, status_badge,
                               tone)

DATASET_HELP = {
    "recorded": "Predictions saved before kick-off and settled after the match.",
    "evaluation": "How the model would have done on past matches (test season and the current season so far), "
                  "computed afterwards from pre-match data. Not predictions published before those matches.",
}
from src.ui.i18n import t
from src.ui.labels import MARKET_GROUP_TITLES, group_title, league_label, market_title, selection_label

ROLES = {"main": "Main predictions", "risk": "Risk predictions", "market": "Market forecasts"}
DATASET_LABELS = {"recorded": "Recorded predictions", "evaluation": "Historical evaluation"}
PAGE_SIZE = 50


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
    """History shows finished predictions only (upcoming matches live on the Matches page)."""
    return ["WON", "LOST"]


@guard
def render() -> None:
    page_header(t("Prediction History"), t("Prediction results and performance history."))
    picks = data.picks()
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return

    dataset = st.segmented_control(t("Data"), list(DATASETS), default="recorded", key="h_dataset",
                                   format_func=lambda k: t(DATASET_LABELS[k])) or "recorded"
    st.html(f'<div class="note">{t(DATASET_HELP[dataset])}</div>')
    picks = select_dataset(picks, dataset)
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return
    first = picks["date"].min()
    season_start = pd.Timestamp(year=config.current_season(), month=7, day=1)
    with st.container(border=True):
        c = st.columns([1.5, 1.5, 1.7, 1.7])
        preset = c[0].selectbox(t("Period"), DATE_PRESETS, index=DATE_PRESETS.index("All time"),
                                format_func=t, key="h_preset")
        custom = None
        if preset == "Custom":
            custom = c[0].date_input(t("Date range"), value=(first.date(), date.today()), format="DD.MM.YYYY",
                                     key="h_custom")
        role = c[1].selectbox(t("Prediction type"), list(ROLES), format_func=lambda r: t(ROLES[r]), key="h_role",
                              help=t("Main = the one prediction FootPredict recommends per match. Risk = an optional "
                                     "high-odds pick. Market forecasts = the model's most likely outcome in each "
                                     "market; they are estimates, not recommended bets."))
        leagues = c[2].multiselect(t("League"), league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder=t("All leagues"), key="h_leagues")
        groups = c[3].multiselect(t("Market"), list(MARKET_GROUP_TITLES), format_func=group_title,
                                  placeholder=t("All markets"), key="h_groups")

    start, end = preset_range(preset, date.today(), first, custom, season_start)
    df = filter_picks(picks, start, end, leagues, groups, role, default_statuses())
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

    section(t("Summary by period"))
    group_by = st.segmented_control(t("Group results by"), list(PERIODS), default="Weekly", format_func=t,
                                    key="h_group_by", help=t("Changes how the summary below is grouped in time; "
                                                             "it does not filter the predictions.")) or "Weekly"
    tl = timeline(df, group_by)
    if not tl.empty:
        with st.container():
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
        t("Match"): lambda v: v, t("Probability"): lambda v: pct(v, 1), t("Odds"): odds_text,
        t("Result"): status_badge,
        t("Profit"): lambda v: "—" if missing(v) else f"<span class='{tone(v)}'>{signed(v)}</span>"},
        align_right=(t("Probability"), t("Odds"), t("Profit"))))
    st.caption(t("Page {p} of {n} · {k} predictions", p=page, n=pages, k=f"{len(table):,}"))


render()
