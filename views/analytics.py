"""Analytics: overall, per-league, per-market and per-period performance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.performance import PERIODS, cumulative_by_bet, group_summary, summarize, timeline
from src.leagues import league_codes, league_label
from src.markets.markets import MARKET_TITLES
from src.markets.odds import SOURCE_LABELS
from src.ui import data
from src.ui.components import empty_state, guard, kpi_row, num, page_header, pct, plotly, section, signed, tone
from src.ui.theme import COLORS

ROLE_LABELS = {"market": "Market predictions", "main": "Main prediction", "risk": "Risk prediction"}
ODDS_FILTER = {"All odds": None, "Bookmaker odds only": ["market"],
               "Bookmaker + derived": ["market", "derived"], "Simulated only": ["simulated"]}


def _table(df: pd.DataFrame, label_col: str, label_name: str) -> None:
    if df.empty:
        st.info("No settled predictions.")
        return
    st.dataframe(df[[label_col, "predictions", "won", "lost", "hit_rate", "avg_odds", "profit", "roi"]],
                 hide_index=True, width="stretch", column_config={
                     label_col: label_name, "predictions": "Predictions", "won": "Won", "lost": "Lost",
                     "hit_rate": st.column_config.ProgressColumn("Hit rate", format="percent", min_value=0, max_value=1),
                     "avg_odds": st.column_config.NumberColumn("Avg odds", format="%.2f"),
                     "profit": st.column_config.NumberColumn("Profit", format="%+.2f"),
                     "roi": st.column_config.NumberColumn("ROI %", format="%+.1f"),
                 })


@guard
def render() -> None:
    page_header("Analytics", "How the stored predictions performed (unit stake per prediction)")
    picks = data.picks()
    if picks.empty:
        empty_state("No predictions stored yet")
        return
    with st.container(border=True):
        c = st.columns([1.5, 1.5, 1.6, 1.4, 1.5])
        leagues = c[0].multiselect("League", league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder="All leagues")
        groups = c[1].multiselect("Market", list(MARKET_TITLES), format_func=MARKET_TITLES.get,
                                  placeholder="All markets")
        min_d, max_d = picks["date"].min().date(), picks["date"].max().date()
        period = c[2].date_input("Date period", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                                 format="DD.MM.YYYY")
        odds_choice = c[3].selectbox("Odds", list(ODDS_FILTER))
        origin = c[4].selectbox("Origin", ["All", "Backtest (test season)", "Current season (live + backfill)"])

    df = picks
    start, end = period if isinstance(period, tuple) and len(period) == 2 else (min_d, max_d)
    df = df.loc[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if leagues:
        df = df.loc[df["league"].isin(leagues)]
    if groups:
        df = df.loc[df["group"].isin(groups)]
    if ODDS_FILTER[odds_choice]:
        df = df.loc[df["odds_source"].isin(ODDS_FILTER[odds_choice])]
    if origin.startswith("Backtest"):
        df = df.loc[df["source"] == "backtest"]
    elif origin.startswith("Current"):
        df = df.loc[df["source"].isin(["live", "backfill"])]

    section("Overall performance")
    s = summarize(df)
    kpi_row([
        ("Predictions", f"{s['settled']:,}", f"settled · {s['pending']} upcoming"),
        ("Hit rate", pct(s["hit_rate"], 1), f"{s['won']:,} won · {s['lost']:,} lost"),
        ("Profit", f"{signed(s['profit'])} u", "", tone(s["profit"])),
        ("ROI", signed(s["roi"], 1, "%"), "overall", tone(s["roi"])),
        ("Average odds", num(s["avg_odds"]), ""),
        ("Streaks", f"{s['longest_win_streak']} / {s['longest_loss_streak']}", "longest win / loss"),
    ])

    section("By prediction type")
    roles = group_summary(df, "role")
    if not roles.empty:
        roles["role"] = roles["role"].map(ROLE_LABELS)
    _table(roles, "role", "Type")

    cols = st.columns(2, gap="medium")
    with cols[0]:
        section("League performance")
        by_league = group_summary(df, "league")
        if not by_league.empty:
            by_league["league"] = by_league["league"].map(lambda x: league_label(x, True))
        _table(by_league, "league", "League")
    with cols[1]:
        section("Market performance")
        by_market = group_summary(df.loc[df["role"] == "market"], "group")
        risk = df.loc[df["role"] == "risk"]
        if not risk.empty:
            risk_row = group_summary(risk.assign(group="RISK"), "group")
            by_market = pd.concat([by_market, risk_row], ignore_index=True)
        if not by_market.empty:
            by_market["group"] = by_market["group"].map(lambda g: MARKET_TITLES.get(g, "Risk prediction"))
        _table(by_market, "group", "Market")
        st.caption("Market rows use the most likely selection of each market for every match; "
                   "the risk row uses the optional high-odds prediction.")

    section("By period")
    agg = st.segmented_control("Period", list(PERIODS), default="Monthly", key="an_period") or "Monthly"
    tl = timeline(df, agg)
    if not tl.empty:
        st.dataframe(tl.sort_values("period", ascending=False), hide_index=True, width="stretch", column_config={
            "period": st.column_config.DateColumn("Period", format="DD.MM.YYYY"),
            "predictions": "Predictions", "won": "Won",
            "profit": st.column_config.NumberColumn("Profit", format="%+.2f"),
            "avg_odds": st.column_config.NumberColumn("Avg odds", format="%.2f"),
            "hit_rate": st.column_config.ProgressColumn("Hit rate", format="percent", min_value=0, max_value=1),
            "roi": st.column_config.NumberColumn("ROI %", format="%+.1f"),
            "cumulative_profit": st.column_config.NumberColumn("Cumulative profit", format="%+.2f"),
            "cumulative_roi": st.column_config.NumberColumn("Cumulative ROI %", format="%+.1f"),
        })

    with st.expander("Advanced Analytics", icon=":material/query_stats:"):
        bets = cumulative_by_bet(df)
        if bets.empty:
            st.info("No settled predictions for the charts.")
            return
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=bets["bet"], y=bets["cumulative_profit"], mode="lines",
                                       line=dict(color=COLORS["positive"])))
            fig.update_layout(title="Cumulative profit (units)", xaxis_title="Prediction #")
            plotly(fig)
        with c2:
            fig = go.Figure(go.Scatter(x=bets["bet"], y=bets["rolling_roi"], mode="lines",
                                       line=dict(color=COLORS["home"])))
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
            fig.update_layout(title="ROI over time (rolling 200, %)", xaxis_title="Prediction #")
            plotly(fig)
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure(go.Scatter(x=bets["bet"], y=bets["rolling_hit_rate"], mode="lines",
                                       line=dict(color=COLORS["away"])))
            fig.update_layout(title="Hit rate over time (rolling 200)", yaxis_tickformat=".0%",
                              xaxis_title="Prediction #")
            plotly(fig)
        with c4:
            fig = go.Figure(go.Bar(x=tl["period"], y=tl["predictions"], marker_color=COLORS["derived"]))
            fig.update_layout(title=f"Prediction volume ({agg.lower()})")
            plotly(fig)
        c5, c6 = st.columns(2)
        lg = group_summary(df, "league")
        with c5:
            if not lg.empty:
                lg = lg.sort_values("roi")
                fig = go.Figure(go.Bar(x=lg["roi"], y=lg["league"].map(league_label), orientation="h",
                                       marker_color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in lg["roi"]]))
                fig.update_layout(title="ROI by league (%)")
                plotly(fig, height=max(320, 22 * len(lg)))
        with c6:
            mk = group_summary(df, "group")
            if not mk.empty:
                fig = go.Figure()
                fig.add_bar(x=mk["group"].map(MARKET_TITLES), y=mk["hit_rate"], name="Hit rate",
                            marker_color=COLORS["home"])
                fig.update_layout(title="Hit rate by market", yaxis_tickformat=".0%")
                plotly(fig)
        st.caption(f"Odds sources: {', '.join(f'{k} = {v}' for k, v in SOURCE_LABELS.items())}.")


render()
