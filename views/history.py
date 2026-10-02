"""Prediction history with filters and per-period aggregation."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics.performance import PERIODS, summarize, timeline
from src.leagues import league_codes, league_label
from src.markets.markets import MARKET_TITLES, market_title, selection_label
from src.markets.odds import SOURCE_LABELS
from src.ui import data
from src.ui.components import empty_state, guard, kpi_row, num, page_header, pct, signed, tone

ROLES = {"All": None, "Market predictions": "market", "Main prediction": "main", "Risk prediction": "risk"}
SOURCES = {"live": "Live (before kick-off)", "backfill": "Backfill (current season)",
           "backtest": "Backtest (test season)"}


def filter_picks(picks: pd.DataFrame, start, end, leagues, groups, role, statuses, odds_sources,
                 sources) -> pd.DataFrame:
    df = picks.loc[(picks["date"] >= pd.Timestamp(start)) & (picks["date"] <= pd.Timestamp(end))]
    if leagues:
        df = df.loc[df["league"].isin(leagues)]
    if groups:
        df = df.loc[df["group"].isin(groups)]
    if role:
        df = df.loc[df["role"] == role]
    if statuses:
        df = df.loc[df["status"].isin(statuses)]
    if odds_sources:
        df = df.loc[df["odds_source"].isin(odds_sources)]
    if sources:
        df = df.loc[df["source"].isin(sources)]
    return df


@guard
def render() -> None:
    page_header("Prediction History", "Every stored prediction, its odds, result and profit (unit stake)")
    picks = data.picks()
    if picks.empty:
        empty_state("No predictions stored yet", "Run scripts/retrain_models.py and scripts/run_pipeline.py.")
        return

    min_d, max_d = picks["date"].min().date(), picks["date"].max().date()
    with st.container(border=True):
        c = st.columns([1.6, 1.6, 1.6, 1.3, 1.3])
        default_start = max(min_d, date.today() - timedelta(days=60))
        period = c[0].date_input("Date range", value=(default_start, max_d), min_value=min_d, max_value=max_d,
                                 format="DD.MM.YYYY")
        leagues = c[1].multiselect("League", league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder="All leagues")
        groups = c[2].multiselect("Market", list(MARKET_TITLES), format_func=MARKET_TITLES.get,
                                  placeholder="All markets")
        role = c[3].selectbox("Prediction type", list(ROLES))
        statuses = c[4].multiselect("Result", ["WON", "LOST", "UPCOMING", "VOID"], placeholder="All")
        c2 = st.columns([1.6, 1.6, 1.6, 2.6])
        odds_sources = c2[0].multiselect("Odds source", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get,
                                         placeholder="All odds")
        sources = c2[1].multiselect("Prediction origin", list(SOURCES), format_func=SOURCES.get,
                                    placeholder="All")
        agg = c2[2].segmented_control("Period", list(PERIODS), default="Weekly") or "Weekly"

    start, end = (period if isinstance(period, tuple) and len(period) == 2 else (min_d, max_d))
    df = filter_picks(picks, start, end, leagues, groups, ROLES[role], statuses, odds_sources, sources)
    s = summarize(df)
    kpi_row([
        ("Predictions", f"{s['predictions']:,}", f"{s['pending']} upcoming · {s['void']} void"),
        ("Hit rate", pct(s["hit_rate"], 1), f"{s['won']} won · {s['lost']} lost"),
        ("Profit", f"{signed(s['profit'])} u", "unit stake per prediction", tone(s["profit"])),
        ("ROI", signed(s["roi"], 1, "%"), "profit / staked", tone(s["roi"])),
        ("Average odds", num(s["avg_odds"]), f"avg probability {pct(s['avg_probability'])}"),
    ])
    if not df.empty and df["odds_source"].isin(["derived", "simulated"]).any():
        st.caption("Profit with derived or simulated odds is a model exercise, not a real betting result — "
                   "use the Odds source filter to see bookmaker-priced predictions only.")

    tl = timeline(df, agg)
    if not tl.empty:
        with st.expander(f"{agg} summary", icon=":material/calendar_month:"):
            show = tl.sort_values("period", ascending=False)
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "period": st.column_config.DateColumn("Period", format="DD.MM.YYYY"),
                "predictions": "Predictions", "won": "Won",
                "profit": st.column_config.NumberColumn("Profit", format="%+.2f"),
                "avg_odds": st.column_config.NumberColumn("Avg odds", format="%.2f"),
                "hit_rate": st.column_config.ProgressColumn("Hit rate", format="percent", min_value=0, max_value=1),
                "roi": st.column_config.NumberColumn("ROI %", format="%+.1f"),
                "cumulative_profit": st.column_config.NumberColumn("Cumulative profit", format="%+.2f"),
                "cumulative_roi": st.column_config.NumberColumn("Cumulative ROI %", format="%+.1f"),
            })

    if df.empty:
        empty_state("No predictions match the filters")
        return
    table = df.sort_values(["date", "time", "match_id"], ascending=[False, False, True]).head(5000)
    view = pd.DataFrame({
        "Date": table["date"],
        "League": table["league"].map(lambda x: league_label(x)),
        "Match": table["home_team"] + " – " + table["away_team"],
        "Type": table["role"].map({"market": "Market", "main": "Main", "risk": "Risk"}),
        "Market": table["market"].map(market_title),
        "Prediction": [selection_label(m, s_, h, a) for m, s_, h, a in
                       zip(table["market"], table["selection"], table["home_team"], table["away_team"])],
        "Probability": table["probability"],
        "Odds": table["odds"],
        "Odds source": table["odds_source"].map({"market": "Bookmaker", "derived": "Derived",
                                                 "simulated": "Simulated"}),
        "Score": [f"{int(h)}–{int(a)}" if h == h else "" for h, a in zip(table["home_goals"], table["away_goals"])],
        "Status": table["status"],
        "Profit": table["profit"],
        "ROI %": table["profit"] * 100,
        "Link": "match?match_id=" + table["match_id"],
    })
    st.dataframe(view, hide_index=True, width="stretch", height=560, column_config={
        "Date": st.column_config.DateColumn(format="DD.MM.YYYY"),
        "Probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Odds": st.column_config.NumberColumn(format="%.2f"),
        "Profit": st.column_config.NumberColumn(format="%+.2f"),
        "ROI %": st.column_config.NumberColumn(format="%+.0f"),
        "Link": st.column_config.LinkColumn("Details", display_text="open"),
    })
    if len(df) > 5000:
        st.caption(f"Showing the latest 5,000 of {len(df):,} predictions — narrow the filters to see more.")


render()
