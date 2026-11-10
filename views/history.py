"""Prediction history: results of stored predictions with filters."""
from __future__ import annotations


import pandas as pd
import streamlit as st

from src.analytics.performance import PERIODS, period_label, select_dataset, summarize, timeline
from src.leagues import league_codes
from src.ui import data
from src.ui.components import (PROFIT_HELP, PUSH_HELP, outcome_counts, date_text, empty_state, esc, guard, html_table, kpi_row, match_url, missing, num,
                               odds_text, page_header, pct, period_filter, section, signed, status_badge, tone)
from src.ui.i18n import t
from src.ui.timefmt import with_local_time
from src.ui.labels import MARKET_GROUP_TITLES, group_title, league_label, market_title, selection_label

ROLES = {"main": "Main predictions", "risk": "Risk predictions", "market": "Market forecasts"}
PAGE_SIZE = 50
SUMMARY_ROWS = 12



def _decision_time_note(df: pd.DataFrame, s: dict) -> None:
    """ROI counting only prices that existed when the prediction was made (recorded predictions); the rest
    are closing prices added afterwards or retrospective predictions — shown, but not a price one could
    have taken at prediction time."""
    from src.prediction.price_timing import decision_time_prices
    if "odds_timing" not in df or not s["bets"]:
        return
    d = summarize(decision_time_prices(df))
    if not d["bets"] or d["bets"] == s["bets"]:
        return
    st.caption(t("Counting only prices that were on offer when the prediction was made: ROI {roi} on {n} bets. "
                 "The other {m} bets use closing prices published afterwards or belong to predictions made "
                 "after the match.", roi=signed(d["roi"], 1, "%"), n=f"{d['bets']:,}", m=f"{s['bets'] - d['bets']:,}"))

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
    picks = select_dataset(data.picks(), "history")
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return
    with st.container(border=True):
        c = st.columns([1.5, 1.5, 1.7, 1.7, 0.6], vertical_alignment="bottom")
        start, end = period_filter(c[0], "h", picks["date"].min())
        role = c[1].selectbox(t("Prediction type"), list(ROLES), format_func=lambda r: t(ROLES[r]), key="h_role",
                              help=t("Main = the one prediction FootPredict recommends per match. Risk = an optional "
                                     "high-odds pick. Market forecasts = the model's most likely outcome in each "
                                     "market; they are estimates, not recommended bets."))
        leagues = c[2].multiselect(t("League"), league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder=t("All leagues"), key="h_leagues")
        query = c[3].text_input(t("Team or match"), key="h_team", placeholder=t("e.g. Arsenal"))
        with c[4].popover("", icon=":material/tune:", help=t("More filters")):
            groups = st.multiselect(t("Market"), list(MARKET_GROUP_TITLES), format_func=group_title,
                                    placeholder=t("All markets"), key="h_groups")

    df = filter_picks(picks, start, end, leagues, groups, role, default_statuses())
    q = (query or "").strip().lower()
    if q:
        df = df.loc[df["home_team"].str.lower().str.contains(q, regex=False)
                    | df["away_team"].str.lower().str.contains(q, regex=False)]
    s = summarize(df)
    kpi_row([
        (t("Predictions"), f"{s['settled']:,}", outcome_counts(s), "", t(PUSH_HELP)),
        (t("Hit rate"), pct(s["hit_rate"], 1), ""),
        (t("Profit · units"), signed(s["profit"]) if s["bets"] else "—",
         t("{n} bets", n=s["bets"]), tone(s["profit"]), t(PROFIT_HELP)),
        (t("ROI"), signed(s["roi"], 1, "%"), t("profit / amount staked"), tone(s["roi"])),
        (t("Average odds"), num(s["avg_odds"]), ""),
    ])
    _decision_time_note(df, s)
    note = t("Profit and ROI use a fixed stake of 1 unit and real bookmaker odds only; a prediction without "
             "a real price counts in the hit rate but not in profit or ROI.")
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
        view = pd.DataFrame({
            t("Period"): [period_label(p_, group_by) for p_ in tl["period"]],
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
        t("Match"): lambda v: v, t("Probability"): lambda v: pct(v, 1), t("Odds"): odds_text,
        t("Result"): status_badge,
        t("Profit"): lambda v: "—" if missing(v) else f"<span class='{tone(v)}'>{signed(v)}</span>"},
        align_right=(t("Probability"), t("Odds"), t("Profit"))))
    st.caption(t("Page {p} of {n} · {k} predictions", p=page, n=pages, k=f"{len(table):,}"))


render()
