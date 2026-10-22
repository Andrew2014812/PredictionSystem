"""Analytics: FootPredict performance — Main predictions by default."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
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


WINDOWS = {"20": 20, "50": 50, "100": 100, "All": None}


def _hbar(df: pd.DataFrame, label_fn, value: str, count: str, title: str, pct_axis: bool,
          colorize: bool) -> None:
    """Horizontal bars sorted best on top, the sample size next to each bar; small samples muted."""
    c = colors()
    d = df.dropna(subset=[value]).sort_values(value)
    if d.empty:
        st.caption(t("No data for this chart yet."))
        return
    small = d[count] < MIN_MARKET_SAMPLE
    base = [c["pos"] if v > 0 else c["neg"] for v in d[value]] if colorize else [c["home"]] * len(d)
    fig = go.Figure(go.Bar(
        x=d[value], y=[label_fn(v) for v in d.iloc[:, 0]], orientation="h", marker_color=base,
        marker_opacity=[0.35 if s_ else 1.0 for s_ in small],
        text=[f"{(v * 100 if pct_axis else v):.0f}{'%' if pct_axis else ''} · n={int(n)}"
              for v, n in zip(d[value], d[count])],
        textposition="outside", cliponaxis=False))
    fig.update_layout(title=title, xaxis_tickformat=".0%" if pct_axis else None, yaxis=dict(automargin=True),
                      margin=dict(r=90))
    plotly(fig, height=max(300, 34 * len(d) + 90))


def _charts(df: pd.DataFrame) -> None:
    """The six charts, two per row: cumulative profit, ROI and hit rate over the last N bets, ROI by
    league, ROI by market and hit rate by market (sample sizes shown, small samples muted)."""
    head = st.columns([3, 2])
    with head[0]:
        section(t("Charts"))
    choice = head[1].segmented_control(t("Window"), list(WINDOWS), default="50", key="a_window",
                                       format_func=lambda k: t("All") if k == "All" else t("Last {n}", n=k),
                                       help=t("ROI and hit rate over the last N bets; All = cumulative since "
                                              "the first bet.")) or "50"
    window = WINDOWS[choice]
    over = t("cumulative") if window is None else t("last {n} bets", n=window)
    c = colors()
    bets = cumulative_by_date(df, window)
    row1 = st.columns(2, gap="medium")
    row2 = st.columns(2, gap="medium")
    row3 = st.columns(2, gap="medium")
    if bets.empty:
        row1[0].caption(t("No predictions with odds in this selection."))
    else:
        with row1[0]:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["cumulative_profit"], mode="lines",
                                       line=dict(color=c["accent"], width=2.5)))
            fig.add_hline(y=0, line_dash="dot", line_color=c["faint"])
            fig.update_layout(title=t("Cumulative profit (units)"))
            plotly(fig, height=330)
        with row1[1]:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["rolling_roi"], mode="lines",
                                       line=dict(color=c["home"], width=2.5)))
            fig.add_hline(y=0, line_dash="dot", line_color=c["faint"])
            fig.update_layout(title=t("ROI (%), {over}", over=over))
            plotly(fig, height=330)
        with row2[0]:
            fig = go.Figure(go.Scatter(x=bets["date"], y=bets["rolling_hit_rate"], mode="lines",
                                       line=dict(color=c["away"], width=2.5)))
            fig.update_layout(title=t("Hit rate, {over}", over=over), yaxis_tickformat=".0%")
            plotly(fig, height=330)
    with row2[1]:
        lg = group_summary(df, "league")
        if not lg.empty:
            _hbar(lg[["league", "roi", "bets"]].assign(roi=lg["roi"] / 100), lambda x: league_label(x),
                  "roi", "bets", t("ROI by league (n = bets with real odds)"), True, True)
    mk = group_summary(df, "group")
    with row3[0]:
        if not mk.empty:
            _hbar(mk[["group", "roi", "bets"]].assign(roi=mk["roi"] / 100), group_title, "roi", "bets",
                  t("ROI by market (n = bets with real odds)"), True, True)
    with row3[1]:
        if not mk.empty:
            _hbar(mk[["group", "hit_rate", "predictions"]], group_title, "hit_rate", "predictions",
                  t("Hit rate by market (n = settled predictions)"), True, False)
    st.caption(t("Faded bars: fewer than {n} predictions — too few to judge a market or league.",
                 n=MIN_MARKET_SAMPLE))


@guard
def render() -> None:
    page_header(t("Analytics"))
    origin = st.segmented_control(t("Data"), ["recorded", "backtest"], default="recorded", key="a_data",
                                  format_func=lambda k: t("Recorded") if k == "recorded"
                                  else t("Backtest {season}").format(season=config.season_label(config.current_season())),
                                  help=t("Recorded = predictions saved before kick-off. Backtest = the production "
                                         "model run afterwards on this season's earlier matches with pre-match data "
                                         "and real archived prices; it is an evaluation, not published bets.")) \
        or "recorded"
    picks = select_dataset(data.picks(), "recorded" if origin == "recorded" else "season_backtest")
    if origin == "backtest" and not picks.empty:
        picks = picks.loc[picks["season"] == config.current_season()]
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
        (t("Main profit") if key == "main" else t("Profit"), f"{signed(s['profit'])} {t('u')}" if s["bets"] else "—",
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
