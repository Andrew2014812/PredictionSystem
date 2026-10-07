"""Analytics: FootPredict performance — Main predictions by default."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.analytics.performance import (DATE_PRESETS, MIN_MARKET_SAMPLE, PERIODS, best_and_worst,
                                       cumulative_by_date, group_summary, preset_range, summarize, timeline)
from src.leagues import league_codes
from src.ui import data
from src.ui.components import (empty_state, esc, guard, html_table, kpi_row, num, page_header, pct, plotly,
                               section, signed, tone)
from src.ui.i18n import t
from src.ui.labels import group_title, league_label
from src.ui.theme import colors

# prediction set -> (role filter, market group filter)
PREDICTION_SETS = {
    "main": ("Main predictions", "main", None),
    "risk": ("Risk predictions", "risk", None),
    "market": ("All market forecasts", "market", None),
    "1X2": ("Match result (1X2)", "market", "1X2"),
    "TOTAL": ("Total goals", "market", "TOTAL"),
    "BTTS": ("Both teams to score", "market", "BTTS"),
    "CORNERS": ("Total corners", "market", "CORNERS"),
    "HANDICAP": ("Handicap", "market", "HANDICAP"),
    "DOUBLE_CHANCE": ("Double chance", "market", "DOUBLE_CHANCE"),
    "EXACT_SCORE": ("Exact score", "market", "EXACT_SCORE"),
}


def select_set(picks: pd.DataFrame, key: str) -> pd.DataFrame:
    _, role, group = PREDICTION_SETS[key]
    df = picks.loc[picks["role"] == role]
    return df.loc[df["group"] == group] if group else df


def _table(df: pd.DataFrame, label_col: str, label_name: str, label_fn) -> None:
    if df.empty:
        st.caption(t("No completed predictions for this period."))
        return
    view = pd.DataFrame({
        label_name: df[label_col].map(label_fn), t("Predictions"): df["predictions"], t("Hit rate"): df["hit_rate"],
        t("Profit"): df["profit"], t("ROI"): df["roi"], t("Average odds"): df["avg_odds"],
    })
    st.html(html_table(view, formats={
        t("Hit rate"): lambda v: pct(v, 1), t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>",
        t("ROI"): lambda v: f"<span class='{tone(v)}'>{signed(v, 1, '%')}</span>", t("Average odds"): num},
        align_right=(t("Predictions"), t("Hit rate"), t("Profit"), t("ROI"), t("Average odds"))))


def _market_highlight(picks: pd.DataFrame) -> None:
    best, worst = best_and_worst(picks.loc[picks["role"] == "market"], "group")
    cols = st.columns(2)
    for col, item, title in ((cols[0], best, t("Best market")), (cols[1], worst, t("Weakest market"))):
        if item is None:
            col.html(f'<div class="kpi"><div class="label">{title}</div><div class="sub">'
                     f'{t("Not enough data (at least {n} predictions with odds).", n=MIN_MARKET_SAMPLE)}</div></div>')
            continue
        col.html(f'<div class="kpi"><div class="label">{title}</div>'
                 f'<div class="value">{esc(group_title(item["group"]))}</div>'
                 f'<div class="sub">ROI <span class="{tone(item["roi"])}">{signed(item["roi"], 1, "%")}</span> · '
                 f'{t("Hit rate")} {pct(item["hit_rate"], 1)} · {t("{n} predictions with odds", n=int(item["bets"]))}'
                 f'</div></div>')


@guard
def render() -> None:
    page_header(t("Analytics"))
    picks = data.picks()
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return
    first = picks["date"].min()
    with st.container(border=True):
        c = st.columns([1.6, 1.4, 1.6, 1.4])
        key = c[0].selectbox(t("Prediction set"), list(PREDICTION_SETS), key="a_set",
                             format_func=lambda k: t(PREDICTION_SETS[k][0]))
        preset = c[1].selectbox(t("Period"), DATE_PRESETS, index=DATE_PRESETS.index("All time"),
                                format_func=t, key="a_preset")
        custom = None
        if preset == "Custom":
            custom = c[1].date_input(t("Date range"), value=(first.date(), date.today()), format="DD.MM.YYYY",
                                     key="a_custom")
        leagues = c[2].multiselect(t("League"), league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder=t("All leagues"), key="a_leagues")
        group_by = c[3].selectbox(t("Group by"), list(PERIODS), index=2, format_func=t, key="a_group_by")

    season_start = pd.Timestamp(year=config.current_season(), month=7, day=1)
    start, end = preset_range(preset, date.today(), first, custom, season_start)
    scope = picks.loc[(picks["date"] >= start) & (picks["date"] <= end)]
    if leagues:
        scope = scope.loc[scope["league"].isin(leagues)]
    df = select_set(scope, key)

    title = t(PREDICTION_SETS[key][0])
    section(t("FootPredict performance") if key == "main" else title)
    s = summarize(df)
    kpi_row([
        (t("Main predictions") if key == "main" else t("Predictions"), f"{s['settled']:,}",
         t("{a} won · {b} lost", a=s["won"], b=s["lost"])),
        (t("Main hit rate") if key == "main" else t("Hit rate"), pct(s["hit_rate"], 1), ""),
        (t("Main profit") if key == "main" else t("Profit"), f"{signed(s['profit'])} u" if s["bets"] else "—",
         t("{n} predictions with odds", n=s["bets"]), tone(s["profit"])),
        (t("Main ROI") if key == "main" else t("ROI"), signed(s["roi"], 1, "%"), t("profit / amount staked"),
         tone(s["roi"])),
        (t("Average odds"), num(s["avg_odds"]), ""),
    ])
    note = t("Fixed stake of 1 unit per prediction; profit and ROI use real bookmaker odds only. "
             "One main prediction per match.") if key == "main" else \
        t("Fixed stake of 1 unit per prediction; profit and ROI use real bookmaker odds only.")
    st.html(f'<div class="note">{note}</div>')

    _market_highlight(scope)

    cols = st.columns(2, gap="medium")
    with cols[0]:
        section(t("Performance by league"))
        _table(group_summary(df, "league"), "league", t("League"), lambda x: league_label(x, True))
    with cols[1]:
        section(t("Performance by market"))
        _table(group_summary(df, "group"), "group", t("Market"), group_title)

    section(t("By period"))
    tl = timeline(df, group_by)
    if tl.empty:
        st.caption(t("No completed predictions for this period."))
    else:
        view = pd.DataFrame({t("Period"): tl["period"].dt.strftime("%d.%m.%Y"), t("Predictions"): tl["predictions"],
                             t("Hit rate"): tl["hit_rate"], t("Profit"): tl["profit"], t("ROI"): tl["roi"],
                             t("Cumulative profit"): tl["cumulative_profit"]}).iloc[::-1]
        st.html(html_table(view, formats={
            t("Hit rate"): lambda v: pct(v, 1), t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>",
            t("ROI"): lambda v: f"<span class='{tone(v)}'>{signed(v, 1, '%')}</span>",
            t("Cumulative profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>"},
            align_right=(t("Predictions"), t("Hit rate"), t("Profit"), t("ROI"), t("Cumulative profit"))))

    with st.expander(t("Advanced analytics"), icon=":material/query_stats:"):
        bets = cumulative_by_date(df)
        if bets.empty:
            st.caption(t("No predictions with odds in this selection."))
            return
        c = colors()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["cumulative_profit"], mode="lines",
                                       line=dict(color=c["accent"])))
            fig.update_layout(title=t("Cumulative profit (units)"))
            plotly(fig)
        with c2:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["rolling_roi"], mode="lines", line=dict(color=c["home"])))
            fig.add_hline(y=0, line_dash="dot", line_color=c["faint"])
            fig.update_layout(title=t("Rolling ROI, last 200 predictions (%)"))
            plotly(fig)
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["rolling_hit_rate"], mode="lines",
                                       line=dict(color=c["away"])))
            fig.update_layout(title=t("Rolling hit rate, last 200 predictions"), yaxis_tickformat=".0%")
            plotly(fig)
        with c4:
            fig = go.Figure(go.Bar(x=tl["period"], y=tl["predictions"], marker_color=c["home"]))
            fig.update_layout(title=t("Main prediction volume over time") if key == "main"
                              else t("Prediction volume over time"))
            plotly(fig)
        c5, c6 = st.columns(2)
        with c5:
            lg = group_summary(df, "league").dropna(subset=["roi"])
            if not lg.empty:
                lg = lg.sort_values("roi")
                fig = go.Figure(go.Bar(x=lg["roi"], y=[league_label(x) for x in lg["league"]], orientation="h",
                                       marker_color=[c["pos"] if v > 0 else c["neg"] for v in lg["roi"]]))
                fig.update_layout(title=t("ROI by league (%)"))
                plotly(fig, height=max(320, 22 * len(lg)))
        with c6:
            mk = group_summary(df, "group").dropna(subset=["roi"])
            if not mk.empty:
                fig = go.Figure(go.Bar(x=[group_title(g) for g in mk["group"]], y=mk["roi"],
                                       marker_color=[c["pos"] if v > 0 else c["neg"] for v in mk["roi"]]))
                fig.update_layout(title=t("ROI by market (%)"))
                plotly(fig)


render()
