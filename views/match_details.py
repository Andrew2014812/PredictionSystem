"""Match details: recommendations, all markets, explanation and team statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.analytics.teams import form_summary, h2h_summary, head_to_head, team_matches
from src.explain.narrative import build_narrative
from src.explain.shap_explainer import explain_selection
from src.leagues import get_league
from src.markets.distributions import corners_distribution, prob_over
from src.markets.markets import market_title, selection_label
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, form_chips, guard, market_table,
                               missing, num, page_header, pct, plotly, recommendation_card, section,
                               signed, status_badge, team_url)
from src.ui.theme import COLORS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _choose_match() -> str | None:
    match_id = st.query_params.get("match_id")
    if match_id:
        return match_id
    snaps = data.snapshots()
    page_header("Match Details", "Select a match")
    if snaps.empty:
        empty_state("No predictions yet", "Run the pipeline to generate predictions.")
        return None
    recent = snaps.sort_values("date", ascending=False).head(400)
    labels = {r.match_id: f"{date_text(r.date, '%d %b %Y')} · {r.home_team} – {r.away_team} ({r.league})"
              for r in recent.itertuples()}
    choice = st.selectbox("Match", list(labels), format_func=labels.get, index=None,
                          placeholder="Search a match…")
    if choice:
        st.query_params["match_id"] = choice
        st.rerun()
    return None


def _header(row: pd.Series) -> None:
    lg = get_league(row["league"])
    played = bool(row.get("played")) and not missing(row.get("home_goals"))
    centre = (f'{int(row["home_goals"])} : {int(row["away_goals"])}' if played
              else f'<span class="vs">{esc(row.get("time") or "—")}</span>')
    status = "Full time" if played else "Upcoming"
    st.html(
        f'<div class="match-header"><div class="mh-meta"><span>{esc(lg.country)} · {esc(lg.name)}</span>'
        f'<span>{date_text(row["date"])}</span><span>{esc(row.get("time") or "—")}</span>'
        f'<span>{status}</span><span>Season {config.season_label(int(row["season"]))}</span></div>'
        f'<div class="mh-row"><div class="mh-team"><a href="{team_url(row["home_team"], row["country"])}" '
        f'target="_self" style="color:inherit">{esc(row["home_team"])}</a><span class="sub">Home</span></div>'
        f'<div class="mh-score">{centre}</div>'
        f'<div class="mh-team away"><a href="{team_url(row["away_team"], row["country"])}" target="_self" '
        f'style="color:inherit">{esc(row["away_team"])}</a><span class="sub">Away</span></div></div></div>')


def _market_rows(markets: pd.DataFrame, market: str) -> pd.DataFrame:
    return markets.loc[markets["market"] == market] if not markets.empty else markets


def _pick(markets: pd.DataFrame, market, selection) -> pd.Series | None:
    if missing(market) or markets.empty:
        return None
    rows = markets.loc[(markets["market"] == market) & (markets["selection"] == selection)]
    return rows.iloc[0] if not rows.empty else None


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _result_block(row: pd.Series, match_id: str) -> None:
    picks = data.picks()
    mine = picks.loc[picks["match_id"] == match_id] if not picks.empty else picks
    if mine.empty:
        return
    section("Match result & settled predictions")
    rows = []
    for _, p in mine.iterrows():
        role = {"main": "Main", "risk": "Risk", "market": "Market"}[p["role"]]
        profit = "" if missing(p["profit"]) else f'<span class="{"pos" if p["profit"] > 0 else "neg"}">{signed(p["profit"])}</span>'
        rows.append(f"<tr><td>{role}</td><td>{esc(market_title(p['market']))}</td>"
                    f"<td>{esc(selection_label(p['market'], p['selection'], row['home_team'], row['away_team']))}</td>"
                    f"<td class='num'>{pct(p['probability'], 1)}</td><td class='num'>{num(p['odds'])}</td>"
                    f"<td>{status_badge(p['status'])}</td><td class='num'>{profit}</td></tr>")
    st.html("<div class='card'><table class='mkt'><tr><th>Type</th><th>Market</th><th>Prediction</th>"
            "<th class='num'>Prob.</th><th class='num'>Odds</th><th>Status</th><th class='num'>Profit</th></tr>"
            + "".join(rows) + "</table></div>")


def _recommendations(snap: pd.Series, markets: pd.DataFrame, home: str, away: str) -> None:
    main = _pick(markets, snap.get("main_market"), snap.get("main_selection"))
    risk = _pick(markets, snap.get("risk_market"), snap.get("risk_selection"))
    cols = st.columns(2, gap="medium")
    with cols[0]:
        if main is not None:
            st.html(recommendation_card("main", main, home, away, snap.get("main_kind") or ""))
        else:
            st.html('<div class="rec main"><div class="tag">Main prediction</div>'
                    '<div class="pick">No eligible selection</div><div class="note">No market had enough '
                    'probability within the allowed odds range.</div></div>')
    with cols[1]:
        if risk is not None:
            st.html(recommendation_card("risk", risk, home, away))
        else:
            st.html('<div class="rec risk"><div class="tag">Risk prediction</div>'
                    '<div class="pick" style="font-size:1.05rem">No risky selection qualifies</div>'
                    '<div class="note">Requires odds ≥ 3.0, probability ≥ 22% and EV ≥ +5%.</div></div>')


def _explanation(row: pd.Series, snap: pd.Series, markets: pd.DataFrame, home: str, away: str) -> None:
    section("Why this prediction?")
    bundles = data.bundles(snap["model_version"])
    if not bundles:
        st.info("The model that produced this prediction is not available, so it cannot be explained.")
        return
    options: dict[tuple, str] = {}
    if not missing(snap.get("main_market")):
        options[(snap["main_market"], snap["main_selection"])] = "Main prediction"
    for market in ("1X2", "TOTAL_2.5", "BTTS", f"CORNERS_{config.DEFAULT_CORNER_LINE}", "DOUBLE_CHANCE"):
        rows = _market_rows(markets, market)
        if not rows.empty:
            best = rows.loc[rows["probability"].idxmax()]
            options.setdefault((best["market"], best["selection"]), market_title(market))
    if not missing(snap.get("risk_market")):
        options.setdefault((snap["risk_market"], snap["risk_selection"]), "Risk prediction")
    keys = list(options)
    choice = st.selectbox("Explain", keys, label_visibility="collapsed",
                          format_func=lambda k: f"{options[k]}: {selection_label(k[0], k[1], home, away)}")
    market, selection = choice
    frame = pd.DataFrame([row])
    explanation = explain_selection(frame, market, selection, bundles)
    if explanation is None or explanation.empty:
        st.info("No explanation available for this market.")
        return
    sel_text = selection_label(market, selection, home, away)
    narrative = build_narrative(explanation, sel_text)

    def factor_html(f):
        arrow = "↑" if f.direction == "up" else "↓"
        return (f'<div class="factor"><span class="arrow {f.direction}">{arrow}</span>'
                f'<span>{esc(f.label)} <span class="val">{esc(f.value_text)}</span></span></div>')

    st.html(f'<div class="why"><p>{esc(narrative.sentence)}</p>'
            + ('<div class="muted small" style="margin-bottom:4px">Main factors in favour</div>'
               + "".join(factor_html(f) for f in narrative.supporting) if narrative.supporting else "")
            + ('<div class="muted small" style="margin:8px 0 4px">Factors against</div>'
               + "".join(factor_html(f) for f in narrative.opposing) if narrative.opposing else "")
            + "</div>")

    with st.expander("Detailed SHAP explanation", icon=":material/bar_chart:"):
        st.caption("Each bar is the contribution of one feature to this prediction (SHAP value, in the model's "
                   "log-scale). Green bars push towards the selection, red bars against it.")
        size = st.segmented_control("Show", ["Top 10", "Top 20", "All"], default="Top 10",
                                    key="shap_n", label_visibility="collapsed") or "Top 10"
        n = {"Top 10": 10, "Top 20": 20}.get(size, len(explanation))
        top = explanation.head(n).iloc[::-1]
        colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in top["shap"]]
        values = [v if isinstance(v, str) else (num(v) if not missing(v) else "n/a") for v in top["value"]]
        fig = go.Figure(go.Bar(x=top["shap"], y=top["label"], orientation="h", marker_color=colors,
                               customdata=values,
                               hovertemplate="%{y}<br>value: %{customdata}<br>SHAP: %{x:+.3f}<extra></extra>"))
        fig.update_layout(xaxis_title="SHAP contribution", yaxis=dict(automargin=True))
        plotly(fig, height=max(280, 26 * len(top) + 60))


def _markets_section(snap: pd.Series, markets: pd.DataFrame, home: str, away: str) -> None:
    section("Markets", "probabilities from the models · fair odds = 1 / probability")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.html('<div class="card"><b>Match result (1X2)</b>'
                + market_table(_market_rows(markets, "1X2"), home, away) + "<br><b>Double chance</b>"
                + market_table(_market_rows(markets, "DOUBLE_CHANCE"), home, away) + "</div>")
    with c2:
        xg_h, xg_a = snap["xg_home"], snap["xg_away"]
        totals = pd.concat([_market_rows(markets, f"TOTAL_{line}") for line in config.TOTAL_GOAL_LINES])
        st.html('<div class="card"><b>Goals</b><div style="display:flex;gap:1.4rem;margin:.5rem 0 .6rem">'
                f'<div><div class="muted small">{esc(home)}</div><div style="font-size:1.4rem;font-weight:800">{num(xg_h)}</div></div>'
                f'<div><div class="muted small">{esc(away)}</div><div style="font-size:1.4rem;font-weight:800">{num(xg_a)}</div></div>'
                f'<div><div class="muted small">Total</div><div style="font-size:1.4rem;font-weight:800">{num(xg_h + xg_a)}</div></div>'
                '</div><span class="muted small">Expected goals (Poisson rates of the goal models)</span><br><br>'
                '<b>Total goals</b>' + market_table(totals, home, away, highlight_best=False) + "</div>")
    with c3:
        exact = _market_rows(markets, "EXACT_SCORE").sort_values("probability", ascending=False)
        st.html('<div class="card"><b>Both teams to score</b>'
                + market_table(_market_rows(markets, "BTTS"), home, away)
                + f"<br><b>Exact score</b> <span class='muted small'>top {config.EXACT_SCORE_TOP_N}</span>"
                + market_table(exact, home, away) + "</div>")

    c4, c5 = st.columns(2, gap="medium")
    with c4:
        with st.container(border=True):
            st.markdown("**Total corners**")
            if not bool(snap.get("corners_available")):
                st.info("Corner prediction unavailable: this league has too little corner statistics "
                        "in the historical data.")
            else:
                mu, size = float(snap["exp_corners"]), snap.get("corners_nb_size")
                size = None if missing(size) else float(size)
                st.markdown(f"Expected total corners: **{mu:.1f}**")
                line = st.select_slider("Corner line", options=[x + 0.5 for x in range(5, 16)],
                                        value=config.DEFAULT_CORNER_LINE)
                pmf = corners_distribution(mu, size)
                over = prob_over(pmf, line)
                stored = _market_rows(markets, f"CORNERS_{line}")
                if not stored.empty:
                    st.html(market_table(stored, home, away))
                else:
                    st.html(f"<table class='mkt'><tr><th>Selection</th><th class='num'>Prob.</th>"
                            f"<th class='num'>Fair</th></tr><tr><td>Over {line:g} corners</td>"
                            f"<td class='num'>{pct(over, 1)}</td><td class='num'>{num(1 / over if over else None)}</td></tr>"
                            f"<tr><td>Under {line:g} corners</td><td class='num'>{pct(1 - over, 1)}</td>"
                            f"<td class='num'>{num(1 / (1 - over) if over < 1 else None)}</td></tr></table>")
                    st.caption("No odds for this line — fair odds only.")
                k = np.arange(len(pmf))
                fig = go.Figure(go.Bar(x=k[:26], y=pmf[:26],
                                       marker_color=[COLORS["positive"] if x > line else COLORS["muted"] for x in k[:26]]))
                fig.update_layout(xaxis_title="Total corners", yaxis_tickformat=".0%", bargap=0.15)
                plotly(fig, height=200)
    with c5:
        hcp = pd.concat([_market_rows(markets, f"HANDICAP_{line:+d}") for line in config.HANDICAP_LINES])
        st.html('<div class="card"><b>Handicap</b> <span class="muted small">European 3-way handicap: '
                'the line is added to the home team\'s goals; no refunds</span>'
                + market_table(hcp, home, away, highlight_best=False, bars=True) + "</div>")


def _comparison(row: pd.Series, home_tm: pd.DataFrame, away_tm: pd.DataFrame) -> None:
    metrics = [
        ("Position", "h_position", "a_position", "low", 0),
        ("Points", "h_points", "a_points", "high", 0),
        ("Points / game", "h_season_ppg", "a_season_ppg", "high", 2),
        ("Form (last 5) PPG", "h_form5_ppg", "a_form5_ppg", "high", 2),
        ("Goals / game", "h_season_gf", "a_season_gf", "high", 2),
        ("Conceded / game", "h_season_ga", "a_season_ga", "low", 2),
        ("Shots / game", "h_season_shots_for", "a_season_shots_for", "high", 1),
        ("On target / game", "h_season_sot_for", "a_season_sot_for", "high", 1),
        ("Corners / game", "h_season_corners_for", "a_season_corners_for", "high", 1),
        ("Yellow cards / game", "h_season_yellows", "a_season_yellows", "low", 1),
    ]
    out = []
    for label, hk, ak, better, digits in metrics:
        hv, av = row.get(hk), row.get(ak)
        if missing(hv) and missing(av):
            continue
        hv_f = 0.0 if missing(hv) else float(hv)
        av_f = 0.0 if missing(av) else float(av)
        total = abs(hv_f) + abs(av_f) or 1.0
        h_better = hv_f < av_f if better == "low" else hv_f > av_f
        a_better = av_f < hv_f if better == "low" else av_f > hv_f
        out.append(
            f'<div class="cmp-row"><div class="hv {"better" if h_better else "worse"}">{num(hv, digits)}</div>'
            f'<div class="cmp-bar h"><span style="width:{abs(hv_f) / total * 100:.0f}%;background:{COLORS["home"]}"></span></div>'
            f'<div class="lbl">{label}</div>'
            f'<div class="cmp-bar"><span style="width:{abs(av_f) / total * 100:.0f}%;background:{COLORS["away"]}"></span></div>'
            f'<div class="av {"better" if a_better else "worse"}">{num(av, digits)}</div></div>')
    form_row = ('<div style="display:flex;justify-content:space-between;align-items:center;padding-top:8px">'
                f'<div>{form_chips(form_summary(home_tm.head(5)).get("form", ""))}</div>'
                '<div class="muted small">Last 5 (newest first)</div>'
                f'<div>{form_chips(form_summary(away_tm.head(5)).get("form", ""))}</div></div>')
    if not out:
        st.info("No season statistics yet for these teams (start of the season).")
        return
    st.html('<div class="card"><div class="cmp-row" style="font-weight:700"><div class="hv">Home</div><div></div>'
            '<div class="lbl">State before kick-off</div><div></div><div class="av">Away</div></div>'
            + "".join(out) + form_row + "</div>")


def _match_list(tm: pd.DataFrame) -> str:
    rows = []
    for r in tm.itertuples():
        rows.append(f'<div class="rm"><span class="chip {r.result}">{r.result}</span>'
                    f'<span class="d">{date_text(r.date, "%d %b %y")}</span><span class="v">{r.venue}</span>'
                    f'<span>{esc(r.opponent)}</span><span class="s">{int(r.gf)}–{int(r.ga)}</span></div>')
    return "".join(rows)


def _form_stats(s: dict) -> str:
    if not s.get("games"):
        return '<span class="muted small">No matches.</span>'
    items = [("W-D-L", f'{s["wins"]}-{s["draws"]}-{s["losses"]}'), ("PPG", num(s["ppg"])),
             ("Goals", num(s["gf"])), ("Conceded", num(s["ga"])), ("Shots", num(s["shots_for"], 1)),
             ("On target", num(s["sot_for"], 1)), ("Corners", num(s["corners_for"], 1)),
             ("Over 2.5", pct(s["over25_rate"])), ("BTTS", pct(s["btts_rate"]))]
    return ('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem .8rem;margin:.4rem 0 .6rem">'
            + "".join(f'<div><div class="muted small">{k}</div><b>{v}</b></div>' for k, v in items) + "</div>")


def _recent(home: str, away: str, home_tm: pd.DataFrame, away_tm: pd.DataFrame) -> None:
    n = st.segmented_control("Last matches", [5, 10, 15], default=5, key="recent_n",
                             format_func=lambda x: f"Last {x}") or 5
    cols = st.columns(2, gap="medium")
    for col, team, tm, color in ((cols[0], home, home_tm, COLORS["home"]), (cols[1], away, away_tm, COLORS["away"])):
        with col:
            last = tm.head(n)
            st.html(f'<div class="card"><b>{esc(team)}</b>{_form_stats(form_summary(last))}'
                    + (_match_list(last) if not last.empty else "") + "</div>")
            if len(last) >= 3:
                chron = last.iloc[::-1]
                fig = go.Figure()
                fig.add_scatter(x=chron["date"], y=chron["gf"], name="Scored", mode="lines+markers",
                                line=dict(color=color))
                fig.add_scatter(x=chron["date"], y=chron["ga"], name="Conceded", mode="lines+markers",
                                line=dict(color=COLORS["muted"], dash="dot"))
                fig.update_layout(legend=dict(orientation="h", y=1.15), yaxis_title="Goals")
                plotly(fig, height=210)


def _home_away(row: pd.Series, home: str, away: str, home_tm: pd.DataFrame, away_tm: pd.DataFrame) -> None:
    season = row["season"]
    h = home_tm.loc[(home_tm["venue"] == "H") & (home_tm["season"] == season)]
    a = away_tm.loc[(away_tm["venue"] == "A") & (away_tm["season"] == season)]
    cols = st.columns(2, gap="medium")
    cols[0].html(f'<div class="card"><b>{esc(home)} at home</b> <span class="muted small">this season</span>'
                 f'{_form_stats(form_summary(h))}<div class="muted small">Last home results</div>'
                 f'{form_chips(form_summary(h.head(5)).get("form", ""))}</div>')
    cols[1].html(f'<div class="card"><b>{esc(away)} away</b> <span class="muted small">this season</span>'
                 f'{_form_stats(form_summary(a))}<div class="muted small">Last away results</div>'
                 f'{form_chips(form_summary(a.head(5)).get("form", ""))}</div>')


def _h2h(row: pd.Series, features: pd.DataFrame, home: str, away: str) -> None:
    meetings = head_to_head(features, home, away, row["country"], before=row["date"])
    if meetings.empty:
        st.info("These teams have not met in the available data.")
        return
    s = h2h_summary(meetings, home)
    st.html(
        '<div class="card"><div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.6rem;margin-bottom:.6rem">'
        + "".join(f'<div><div class="muted small">{k}</div><b style="font-size:1.2rem">{v}</b></div>' for k, v in [
            ("Meetings", s["games"]), (f"{home} wins", s["wins"]), ("Draws", s["draws"]),
            (f"{away} wins", s["losses"]), ("Goals / match", num(s["avg_goals"], 1)),
            ("BTTS · Over 2.5", f'{pct(s["btts_rate"])} · {pct(s["over25_rate"])}')])
        + "</div>"
        + "".join(f'<div class="rm" style="grid-template-columns:90px 1fr 60px 1fr"><span class="d">'
                  f'{date_text(m.date, "%d %b %Y")}</span><span style="text-align:right">{esc(m.home_team)}</span>'
                  f'<span class="s" style="text-align:center">{int(m.home_goals)}–{int(m.away_goals)}</span>'
                  f'<span>{esc(m.away_team)}</span></div>' for m in meetings.itertuples())
        + "</div>")
    if s["games"] < 3:
        st.caption("Few meetings — head-to-head statistics are not very informative here.")


def _rest(row: pd.Series, home: str, away: str) -> None:
    hr, ar = row.get("h_rest_days"), row.get("a_rest_days")
    diff = None if missing(hr) or missing(ar) else hr - ar
    cols = st.columns(3)
    cols[0].metric(f"{home} rest", "—" if missing(hr) else f"{int(hr)} days")
    cols[1].metric(f"{away} rest", "—" if missing(ar) else f"{int(ar)} days")
    cols[2].metric("Rest difference", "—" if diff is None else f"{diff:+.0f} days",
                   help="Positive = the home team had more rest.")
    st.caption(f"Days since each team's previous league match in the data (capped at {config.REST_DAYS_CAP}; "
               "cup and European matches are not part of the data source).")


# ---------------------------------------------------------------------------
@guard
def render() -> None:
    match_id = _choose_match()
    if not match_id:
        return
    row = data.match_row(match_id)
    if row is None:
        page_header("Match Details")
        empty_state("Match not found", "It may have been removed from the data source.")
        st.page_link("views/matches.py", label="Back to matches", icon=":material/arrow_back:")
        return
    page_header("Match Details")
    st.page_link("views/matches.py", label="Back to matches", icon=":material/arrow_back:",
                 query_params={"date": str(pd.Timestamp(row["date"]).date())})
    home, away = row["home_team"], row["away_team"]
    _header(row)

    snap = data.snapshot(match_id)
    features = data.features()
    feat_rows = features.loc[features["match_id"] == match_id]
    feat_row = feat_rows.iloc[0] if not feat_rows.empty else row

    if bool(row.get("played")):
        _result_block(row, match_id)

    if snap is None:
        st.info("There is no prediction for this match. Predictions exist for upcoming matches, the current "
                "season and the test season; earlier seasons were used to train the models.")
    else:
        markets = data.match_markets(match_id)
        _recommendations(snap, markets, home, away)
        if not feat_rows.empty:
            _explanation(feat_row, snap, markets, home, away)
        _markets_section(snap, markets, home, away)
        src = {"live": "generated before kick-off", "backfill": "generated after the match by a model "
               "trained only on earlier seasons (out-of-sample)", "backtest": "test-season backtest model"}
        st.caption(f"Model version {snap['model_version']} · {src.get(snap['source'], snap['source'])} · "
                   f"generated {date_text(snap['generated_at'], '%d %b %Y %H:%M')}")

    section("Teams before the match")
    home_tm = team_matches(features, home, row["country"], before=row["date"])
    away_tm = team_matches(features, away, row["country"], before=row["date"])
    tabs = st.tabs(["Team comparison", "Recent matches", "Home / Away", "Head-to-head", "Rest days"])
    with tabs[0]:
        _comparison(feat_row, home_tm, away_tm)
    with tabs[1]:
        _recent(home, away, home_tm, away_tm)
    with tabs[2]:
        _home_away(row, home, away, home_tm, away_tm)
    with tabs[3]:
        _h2h(row, features, home, away)
    with tabs[4]:
        _rest(feat_row, home, away)


render()
