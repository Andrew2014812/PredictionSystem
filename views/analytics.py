"""Analytics: FootPredict performance — Main predictions by default."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.performance import (MIN_MARKET_SAMPLE, PERIODS, best_and_worst, cumulative_by_date,
                                       group_summary, select_dataset, summarize, timeline)
from src.leagues import league_codes
from src.ui import data
from src.ui.components import (empty_state, esc, guard, html_table, kpi_row, num, page_header, pct, period_filter,
                               plotly, section, signed, tone)
from src.ui.i18n import t
from src.ui.labels import group_title, league_label
from src.ui.theme import colors

SUMMARY_ROWS = 12

# prediction set -> (role filter, market group filter)
PREDICTION_SETS = {
    "main": ("Main predictions", "main", None),
    "risk": ("Risk predictions", "risk", None),
    "market": ("All market forecasts", "market", None),
    "1X2": ("Match result (1X2)", "market", "1X2"),
    "TOTAL": ("Total goals", "market", "TOTAL"),
    "BTTS": ("Both teams to score", "market", "BTTS"),
    "CORNERS": ("Total corners", "market", "CORNERS"),
    "HANDICAP": ("Asian handicap", "market", "HANDICAP"),
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


def highlight_dimension(set_key: str) -> str:
    """Markets are compared inside a prediction set; inside one market, leagues are."""
    return "league" if PREDICTION_SETS[set_key][2] else "group"


def _market_highlight(df: pd.DataFrame, set_key: str) -> None:
    """Highest / lowest ROI market (or league, for a single-market set) WITHIN the active filters."""
    dim = highlight_dimension(set_key)
    best, worst = best_and_worst(df, dim)
    label = (lambda v: league_label(v, True)) if dim == "league" else group_title
    titles = ((t("Highest ROI league"), t("Lowest ROI league")) if dim == "league"
              else (t("Highest ROI market"), t("Lowest ROI market")))
    cols = st.columns(2)
    for col, item, title in ((cols[0], best, titles[0]), (cols[1], worst, titles[1])):
        if item is None:
            col.html(f'<div class="kpi"><div class="label">{title}</div><div class="value">—</div></div>')
            continue
        small = (f' <span class="chip-note" title="{esc(t("Fewer than {n} predictions with odds: treat with care.", n=MIN_MARKET_SAMPLE))}">'
                 f'{t("Small sample")}</span>' if item.get("small_sample") else "")
        col.html(f'<div class="kpi"><div class="label">{title}{small}</div>'
                 f'<div class="value">{esc(label(item[dim]))}</div>'
                 f'<div class="sub">ROI <span class="{tone(item["roi"])}">{signed(item["roi"], 1, "%")}</span> · '
                 f'{t("Hit rate")} {pct(item["hit_rate"], 1)} · {t("{n} predictions with odds", n=int(item["bets"]))}'
                 f'</div></div>')


def _bar(fig_x, fig_y, colors_, title, horizontal=False, text=None, height=None, pct_axis=False) -> None:
    fig = go.Figure(go.Bar(x=fig_x, y=fig_y, orientation="h" if horizontal else "v", marker_color=colors_,
                           text=text, textposition="outside" if text is not None else None, cliponaxis=False))
    fig.update_layout(title=title)
    if pct_axis:
        fig.update_layout(**({"xaxis_tickformat": ".0%"} if horizontal else {"yaxis_tickformat": ".0%"}))
    plotly(fig, height=height)


def _charts(df: pd.DataFrame) -> None:
    """The six charts: cumulative profit, rolling ROI, rolling hit rate, ROI by league,
    ROI by market and hit rate by market (with sample sizes)."""
    section(t("Charts"))
    c = colors()
    bets = cumulative_by_date(df)
    c1, c2, c3 = st.columns(3, gap="medium")
    if bets.empty:
        st.caption(t("No predictions with odds in this selection."))
    else:
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
        with c3:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["rolling_hit_rate"], mode="lines",
                                       line=dict(color=c["away"])))
            fig.update_layout(title=t("Rolling hit rate, last 200 predictions"), yaxis_tickformat=".0%")
            plotly(fig)
    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        lg = group_summary(df, "league").dropna(subset=["roi"]).sort_values("roi")
        if not lg.empty:
            _bar(lg["roi"], [league_label(x) for x in lg["league"]],
                 [c["pos"] if v > 0 else c["neg"] for v in lg["roi"]], t("ROI by league (%)"), horizontal=True,
                 height=max(320, 22 * len(lg)))
    with c5:
        mk = group_summary(df, "group").dropna(subset=["roi"])
        if not mk.empty:
            _bar([group_title(g) for g in mk["group"]], mk["roi"],
                 [c["pos"] if v > 0 else c["neg"] for v in mk["roi"]], t("ROI by market (%)"),
                 text=[f"n={int(n)}" for n in mk["bets"]])
    with c6:
        hr = group_summary(df, "group")
        if not hr.empty:
            _bar([group_title(g) for g in hr["group"]], hr["hit_rate"], c["home"], t("Hit rate by market"),
                 text=[f"n={int(n)}" for n in hr["predictions"]], pct_axis=True)
    st.caption(t("n = number of settled predictions (for ROI: with real odds). Small samples swing a lot."))


@guard
def render() -> None:
    page_header(t("Analytics"))
    # recorded pre-match predictions only; the historical model evaluation is on Model Analysis
    picks = select_dataset(data.picks(), "recorded")
    if picks.empty:
        empty_state(t("No completed predictions yet."))
        return
    with st.container(border=True):
        c = st.columns([1.6, 1.6, 1.8])
        key = c[0].selectbox(t("Prediction set"), list(PREDICTION_SETS), key="a_set",
                             format_func=lambda k: t(PREDICTION_SETS[k][0]))
        start, end = period_filter(c[1], "a", picks["date"].min())
        leagues = c[2].multiselect(t("League"), league_codes(), format_func=lambda x: league_label(x, True),
                                   placeholder=t("All leagues"), key="a_leagues")
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

    _market_highlight(df, key)

    cols = st.columns(2, gap="medium")
    with cols[0]:
        section(t("Performance by league"))
        _table(group_summary(df, "league"), "league", t("League"), lambda x: league_label(x, True))
    with cols[1]:
        section(t("Performance by market"))
        _table(group_summary(df, "group"), "group", t("Market"), group_title)

    _charts(df)

    head = st.columns([3, 2])
    with head[0]:
        section(t("Performance over time"))
    group_by = head[1].segmented_control(t("Group results by"), list(PERIODS), default="Monthly", format_func=t,
                                         key="a_group_by", label_visibility="collapsed") or "Monthly"
    tl = timeline(df, group_by)
    if tl.empty:
        st.caption(t("No completed predictions for this period."))
        return
    view = pd.DataFrame({t("Period"): tl["period"].dt.strftime("%d.%m.%Y"), t("Predictions"): tl["predictions"],
                         t("Hit rate"): tl["hit_rate"], t("Profit"): tl["profit"], t("ROI"): tl["roi"],
                         t("Cumulative profit"): tl["cumulative_profit"]}).iloc[::-1]
    if len(view) > SUMMARY_ROWS and not st.toggle(t("Show all {n} periods", n=len(view)), key="a_all_periods"):
        view = view.head(SUMMARY_ROWS)
    st.html(html_table(view, formats={
        t("Hit rate"): lambda v: pct(v, 1), t("Profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>",
        t("ROI"): lambda v: f"<span class='{tone(v)}'>{signed(v, 1, '%')}</span>",
        t("Cumulative profit"): lambda v: f"<span class='{tone(v)}'>{signed(v)}</span>"},
        align_right=(t("Predictions"), t("Hit rate"), t("Profit"), t("ROI"), t("Cumulative profit"))))


render()
