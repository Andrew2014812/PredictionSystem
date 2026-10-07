"""Match details: main / risk prediction, markets, teams and the explanation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.analytics.teams import form_summary, h2h_summary, head_to_head, team_matches
from src.explain.narrative import build_narrative, influence_table
from src.explain.shap_explainer import explain_selection
from src.leagues import get_league
from src.markets.distributions import corners_distribution, prob_over
from src.markets.markets import is_half_line, market_line
from src.ui import data
from src.ui.components import (date_text, empty_state, esc, form_chips, guard, market_table, missing, num,
                               page_header, pct, plotly, recommendation_card, section, signed,
                               status_badge, team_url, tip)
from src.ui.i18n import current_language, t
from src.ui.labels import country_label, market_title, selection_label
from src.ui.prefs import pref_params
from src.ui.theme import colors


def _choose_match() -> str | None:
    match_id = st.query_params.get("match_id")
    if match_id:
        return match_id
    page_header(t("Match Details"))
    snaps = data.snapshots()
    if snaps.empty:
        empty_state(t("No predictions yet."))
        return None
    recent = snaps.sort_values("date", ascending=False).head(400)
    labels = {r.match_id: f"{date_text(r.date)} · {r.home_team} – {r.away_team}" for r in recent.itertuples()}
    choice = st.selectbox(t("Match"), list(labels), format_func=labels.get, index=None,
                          placeholder=t("Search a match…"))
    if choice:
        st.query_params["match_id"] = choice
        st.rerun()
    return None


def _header(row: pd.Series) -> None:
    lg = get_league(row["league"])
    played = bool(row.get("played")) and not missing(row.get("home_goals"))
    centre = (f'{int(row["home_goals"])} : {int(row["away_goals"])}' if played
              else f'<span class="vs">{esc(row.get("time") or "—")}</span>')
    st.html(
        f'<div class="match-header"><div class="mh-meta"><span>{esc(country_label(lg.country))} · {esc(lg.name)}</span>'
        f'<span>{date_text(row["date"])}</span><span>{esc(row.get("time") or "—")}</span>'
        f'<span>{t("Full time") if played else t("Upcoming")}</span></div>'
        f'<div class="mh-row"><div class="mh-team"><a href="{team_url(row["home_team"], row["country"])}" '
        f'target="_self">{esc(row["home_team"])}</a><span class="sub">{t("Home")}</span></div>'
        f'<div class="mh-score">{centre}</div>'
        f'<div class="mh-team away"><a href="{team_url(row["away_team"], row["country"])}" target="_self">'
        f'{esc(row["away_team"])}</a><span class="sub">{t("Away")}</span></div></div></div>')


def _rows(markets: pd.DataFrame, market: str) -> pd.DataFrame:
    return markets.loc[markets["market"] == market] if not markets.empty else markets


def _pick(markets: pd.DataFrame, market, selection) -> pd.Series | None:
    if missing(market) or markets.empty:
        return None
    rows = markets.loc[(markets["market"] == market) & (markets["selection"] == selection)]
    return rows.iloc[0] if not rows.empty else None


# ---------------------------------------------------------------------------
def _result_block(row: pd.Series, match_id: str) -> None:
    picks = data.picks()
    mine = picks.loc[(picks["match_id"] == match_id) & (picks["status"] != "UPCOMING")] if not picks.empty else picks
    if mine.empty:
        return
    section(t("Prediction results"))
    order = {"main": 0, "risk": 1, "market": 2}
    mine = mine.assign(_o=mine["role"].map(order)).sort_values("_o")
    rows = []
    for _, p in mine.iterrows():
        role = {"main": t("Main prediction"), "risk": t("Risk prediction"), "market": t("Forecast")}[p["role"]]
        profit = ("" if missing(p["profit"]) else
                  f'<span class="{"pos" if p["profit"] > 0 else "neg"}">{signed(p["profit"])}</span>')
        rows.append(f"<tr><td>{esc(role)}</td><td>{esc(market_title(p['market']))}</td>"
                    f"<td>{esc(selection_label(p['market'], p['selection'], row['home_team'], row['away_team']))}</td>"
                    f"<td class='num'>{pct(p['probability'], 1)}</td><td class='num'>{num(p['odds'])}</td>"
                    f"<td>{status_badge(p['status'])}</td><td class='num'>{profit}</td></tr>")
    st.html(f"<div class='tablewrap'><table class='mkt'><tr><th></th><th>{t('Market')}</th><th>{t('Prediction')}</th>"
            f"<th class='num'>{t('Probability')}</th><th class='num'>{t('Odds')}</th><th>{t('Result')}</th>"
            f"<th class='num'>{t('Profit')}</th></tr>" + "".join(rows) + "</table></div>")


def _recommendations(snap: pd.Series, markets: pd.DataFrame, home: str, away: str):
    main = _pick(markets, snap.get("main_market"), snap.get("main_selection"))
    risk = _pick(markets, snap.get("risk_market"), snap.get("risk_selection"))
    if main is None:
        return None, None
    if risk is None:
        st.html(recommendation_card("main", main, home, away))
    else:
        cols = st.columns(2, gap="medium")
        cols[0].html(recommendation_card("main", main, home, away))
        cols[1].html(recommendation_card("risk", risk, home, away))
    return main, risk


def _explain(feat_row: pd.Series, bundles, market: str, selection: str, home: str, away: str,
             role: str = "main") -> pd.DataFrame | None:
    lang = current_language()
    explanation = explain_selection(pd.DataFrame([feat_row]), market, selection, bundles)
    if explanation is None or explanation.empty:
        st.info(t("No explanation is available for this prediction."))
        return None
    sel_text = selection_label(market, selection, home, away)
    narrative = build_narrative(explanation, sel_text, lang)
    c = colors()
    rows = []
    for f in influence_table(explanation, 6, lang):
        color = c["pos"] if f.direction == "up" else c["neg"]
        rows.append(f'<div class="infl"><span>{"↑" if f.direction == "up" else "↓"} {esc(f.label)}</span>'
                    f'<div class="bar"><span style="width:{f.relative * 100:.0f}%;background:{color}"></span></div>'
                    f'<span class="lvl">{esc(f.level)}</span></div>')
    st.html(f'<div class="why {"risk" if role == "risk" else ""}"><p>{esc(narrative.sentence)}</p>{"".join(rows)}</div>')
    return explanation


def _explanation(feat_row: pd.Series, snap: pd.Series, markets: pd.DataFrame, main, risk, home, away) -> None:
    section(t("Why this prediction?"))
    bundles = data.bundles(snap["model_version"])
    if not bundles or main is None:
        st.info(t("No explanation is available for this prediction."))
        return
    explanation = _explain(feat_row, bundles, main["market"], main["selection"], home, away)
    if risk is not None and st.toggle(t("Explain risk prediction"), key="explain_risk"):
        _explain(feat_row, bundles, risk["market"], risk["selection"], home, away, role="risk")

    with st.expander(t("Advanced: explain another market · technical SHAP view"), icon=":material/tune:"):
        options = {}
        for market in ("1X2", "TOTAL_2.5", "BTTS", f"CORNERS_{config.DEFAULT_CORNER_LINE}", "DOUBLE_CHANCE"):
            rows = _rows(markets, market)
            if not rows.empty:
                best = rows.loc[rows["probability"].idxmax()]
                options[(best["market"], best["selection"])] = market_title(market)
        if options:
            key = st.selectbox(t("Market"), list(options), key="explain_other",
                               format_func=lambda k: f"{options[k]}: {selection_label(k[0], k[1], home, away)}")
            other = explain_selection(pd.DataFrame([feat_row]), key[0], key[1], bundles)
        else:
            other = explanation
        if other is None or other.empty:
            return
        st.caption(t("SHAP values are in the model's internal scale (log-odds or log of the expected count). "
                     "Green pushes towards the selection, red against it."))
        size = st.segmented_control(t("Show"), ["10", "20", t("All")], default="10", key="shap_n",
                                    label_visibility="collapsed") or "10"
        n = int(size) if size.isdigit() else len(other)
        lang = current_language()
        top = other.head(n).iloc[::-1]
        c = colors()
        fig = go.Figure(go.Bar(x=top["shap"], y=[f.label for f in influence_table(other.head(n), n, lang)][::-1],
                               orientation="h",
                               marker_color=[c["pos"] if v > 0 else c["neg"] for v in top["shap"]],
                               hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>"))
        fig.update_layout(xaxis_title="SHAP", yaxis=dict(automargin=True))
        plotly(fig, height=max(280, 26 * len(top) + 60))


def _markets_section(snap: pd.Series, markets: pd.DataFrame, home: str, away: str) -> None:
    section(t("Markets"))
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.html(f'<div class="card"><b>{t("Match result")}</b>' + market_table(_rows(markets, "1X2"), home, away)
                + f'<br><b>{t("Double chance")}</b>'
                + market_table(_rows(markets, "DOUBLE_CHANCE"), home, away) + "</div>")
    with c2:
        xg_h, xg_a = snap["xg_home"], snap["xg_away"]
        totals = pd.concat([_rows(markets, f"TOTAL_{line}") for line in config.TOTAL_GOAL_LINES])
        st.html(f'<div class="card"><b>{t("Expected goals")}</b>'
                '<div style="display:flex;gap:1.4rem;margin:.5rem 0 .8rem">'
                f'<div><div class="muted small">{esc(home)}</div><div style="font-size:1.4rem;font-weight:800">{num(xg_h)}</div></div>'
                f'<div><div class="muted small">{esc(away)}</div><div style="font-size:1.4rem;font-weight:800">{num(xg_a)}</div></div>'
                f'<div><div class="muted small">{t("Total")}</div><div style="font-size:1.4rem;font-weight:800">{num(xg_h + xg_a)}</div></div>'
                f'</div><b>{t("Total goals")}</b>' + market_table(totals, home, away, highlight_best=False) + "</div>")
    with c3:
        exact = _rows(markets, "EXACT_SCORE").sort_values("probability", ascending=False)
        st.html(f'<div class="card"><b>{t("Both teams to score")}</b>' + market_table(_rows(markets, "BTTS"), home, away)
                + f"<br><b>{t('Exact score')}</b>" + market_table(exact, home, away) + "</div>")

    c4, c5 = st.columns(2, gap="medium")
    with c4, st.container(border=True):
        st.markdown(f"**{t('Total corners')}**")
        mu, size = float(snap["exp_corners"]), snap.get("corners_nb_size")
        size = None if missing(size) else float(size)
        st.markdown(t("Expected total corners: **{n}**", n=f"{mu:.1f}"))
        line = st.select_slider(t("Corner line"), options=[x + 0.5 for x in range(5, 16)],
                                value=config.DEFAULT_CORNER_LINE)
        stored = _rows(markets, f"CORNERS_{line}")
        if stored.empty:
            over = prob_over(corners_distribution(mu, size), line)
            stored = pd.DataFrame([{"market": f"CORNERS_{line}", "selection": "OVER", "probability": over,
                                    "odds": np.nan, "ev": np.nan},
                                   {"market": f"CORNERS_{line}", "selection": "UNDER", "probability": 1 - over,
                                    "odds": np.nan, "ev": np.nan}])
        st.html(market_table(stored.reset_index(drop=True), home, away))
    with c5:
        hcp = pd.concat([_rows(markets, f"HANDICAP_{line:+g}") for line in config.HANDICAP_LINES]) \
            if not markets.empty else markets
        whole = hcp.loc[[not is_half_line(market_line(m)) for m in hcp["market"]]] if not hcp.empty else hcp
        half = hcp.loc[[is_half_line(market_line(m)) for m in hcp["market"]]] if not hcp.empty else hcp
        help_whole = t("European handicap: the handicap is added to the home team's goals, then the result "
                       "(home / draw / away) is decided.")
        help_half = t("Half-line handicap: the handicap is added to the home team's goals; a draw is impossible, "
                      "so there are only two outcomes.")
        st.html(f'<div class="card"><b>{t("European handicap")}</b>{tip(help_whole)}'
                + market_table(whole, home, away, highlight_best=False)
                + f'<br><b>{t("Handicap (half lines)")}</b>{tip(help_half)}'
                + market_table(half, home, away, highlight_best=False) + "</div>")


def _comparison(row: pd.Series, home_tm: pd.DataFrame, away_tm: pd.DataFrame) -> None:
    metrics = [
        ("League position", "h_position", "a_position", "low", 0),
        ("Points", "h_points", "a_points", "high", 0),
        ("Points / game", "h_season_ppg", "a_season_ppg", "high", 2),
        ("Points / game (last 5)", "h_form5_ppg", "a_form5_ppg", "high", 2),
        ("Goals / game", "h_season_gf", "a_season_gf", "high", 2),
        ("Conceded / game", "h_season_ga", "a_season_ga", "low", 2),
        ("Shots / game", "h_season_shots_for", "a_season_shots_for", "high", 1),
        ("Shots on target / game", "h_season_sot_for", "a_season_sot_for", "high", 1),
        ("Corners / game", "h_season_corners_for", "a_season_corners_for", "high", 1),
        ("Yellow cards / game", "h_season_yellows", "a_season_yellows", "low", 1),
    ]
    c = colors()
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
            f'<div class="cmp-bar h"><span style="width:{abs(hv_f) / total * 100:.0f}%;background:{c["home"]}"></span></div>'
            f'<div class="lbl">{t(label)}</div>'
            f'<div class="cmp-bar"><span style="width:{abs(av_f) / total * 100:.0f}%;background:{c["away"]}"></span></div>'
            f'<div class="av {"better" if a_better else "worse"}">{num(av, digits)}</div></div>')
    if not out:
        st.info(t("No season statistics yet for these teams."))
        return
    form_row = ('<div style="display:flex;justify-content:space-between;align-items:center;padding-top:8px">'
                f'<div>{form_chips(form_summary(home_tm.head(5)).get("form", ""))}</div>'
                f'<div class="muted small">{t("Last 5 matches")}</div>'
                f'<div>{form_chips(form_summary(away_tm.head(5)).get("form", ""))}</div></div>')
    st.html(f'<div class="card"><div class="cmp-row" style="font-weight:700"><div class="hv">{t("Home")}</div><div></div>'
            f'<div class="lbl">{t("Before kick-off")}</div><div></div><div class="av">{t("Away")}</div></div>'
            + "".join(out) + form_row + "</div>")


def _match_list(tm: pd.DataFrame) -> str:
    return "".join(
        f'<div class="rm">{form_chips(r.result)}<span class="d">{date_text(r.date, "%d.%m.%y")}</span>'
        f'<span class="v">{t("H") if r.venue == "H" else t("A")}</span><span>{esc(r.opponent)}</span>'
        f'<span class="s">{int(r.gf)}–{int(r.ga)}</span></div>' for r in tm.itertuples())


def _form_stats(s: dict) -> str:
    if not s.get("games"):
        return f'<span class="muted small">{t("No matches.")}</span>'
    items = [(t("W-D-L"), f'{s["wins"]}-{s["draws"]}-{s["losses"]}'), (t("Points / game"), num(s["ppg"])),
             (t("Goals"), num(s["gf"])), (t("Conceded"), num(s["ga"])), (t("Shots"), num(s["shots_for"], 1)),
             (t("On target"), num(s["sot_for"], 1)), (t("Corners"), num(s["corners_for"], 1)),
             (t("Over 2.5"), pct(s["over25_rate"])), ("BTTS", pct(s["btts_rate"]))]
    return ('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem .8rem;margin:.4rem 0 .6rem">'
            + "".join(f'<div><div class="muted small">{k}</div><b>{v}</b></div>' for k, v in items) + "</div>")


def _recent(home: str, away: str, home_tm: pd.DataFrame, away_tm: pd.DataFrame) -> None:
    n = st.segmented_control(t("Last matches"), [5, 10, 15], default=5, key="recent_n",
                             format_func=lambda x: t("Last {n}", n=x), label_visibility="collapsed") or 5
    c = colors()
    cols = st.columns(2, gap="medium")
    for col, team, tm, color in ((cols[0], home, home_tm, c["home"]), (cols[1], away, away_tm, c["away"])):
        with col:
            last = tm.head(n)
            st.html(f'<div class="card"><b>{esc(team)}</b>{_form_stats(form_summary(last))}'
                    + (_match_list(last) if not last.empty else "") + "</div>")
            if len(last) >= 3:
                chron = last.iloc[::-1]
                fig = go.Figure()
                fig.add_scatter(x=chron["date"], y=chron["gf"], name=t("Scored"), mode="lines+markers",
                                line=dict(color=color))
                fig.add_scatter(x=chron["date"], y=chron["ga"], name=t("Conceded"), mode="lines+markers",
                                line=dict(color=c["faint"], dash="dot"))
                fig.update_layout(legend=dict(orientation="h", y=1.15), yaxis_title=t("Goals"))
                plotly(fig, height=210)


def _home_away(row: pd.Series, home: str, away: str, home_tm: pd.DataFrame, away_tm: pd.DataFrame) -> None:
    season = row["season"]
    h = home_tm.loc[(home_tm["venue"] == "H") & (home_tm["season"] == season)]
    a = away_tm.loc[(away_tm["venue"] == "A") & (away_tm["season"] == season)]
    cols = st.columns(2, gap="medium")
    cols[0].html(f'<div class="card"><b>{esc(home)} · {t("at home")}</b> <span class="muted small">{t("this season")}</span>'
                 f'{_form_stats(form_summary(h))}{form_chips(form_summary(h.head(5)).get("form", ""))}</div>')
    cols[1].html(f'<div class="card"><b>{esc(away)} · {t("away")}</b> <span class="muted small">{t("this season")}</span>'
                 f'{_form_stats(form_summary(a))}{form_chips(form_summary(a.head(5)).get("form", ""))}</div>')


def _h2h(row: pd.Series, features: pd.DataFrame, home: str, away: str) -> None:
    meetings = head_to_head(features, home, away, row["country"], before=row["date"])
    if meetings.empty:
        st.info(t("These teams have not met in the available data."))
        return
    s = h2h_summary(meetings, home)
    st.html(
        '<div class="card"><div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.6rem;margin-bottom:.6rem">'
        + "".join(f'<div><div class="muted small">{k}</div><b style="font-size:1.2rem">{v}</b></div>' for k, v in [
            (t("Meetings"), s["games"]), (t("{team} wins", team=home), s["wins"]), (t("Draws"), s["draws"]),
            (t("{team} wins", team=away), s["losses"]), (t("Goals / match"), num(s["avg_goals"], 1)),
            ("BTTS · " + t("Over 2.5"), f'{pct(s["btts_rate"])} · {pct(s["over25_rate"])}')])
        + "</div>"
        + "".join(f'<div class="rm" style="grid-template-columns:90px 1fr 60px 1fr"><span class="d">'
                  f'{date_text(m.date)}</span><span style="text-align:right">{esc(m.home_team)}</span>'
                  f'<span class="s" style="text-align:center">{int(m.home_goals)}–{int(m.away_goals)}</span>'
                  f'<span>{esc(m.away_team)}</span></div>' for m in meetings.itertuples())
        + "</div>")


def _rest(row: pd.Series, home: str, away: str) -> None:
    hr, ar = row.get("h_rest_days"), row.get("a_rest_days")
    diff = None if missing(hr) or missing(ar) else hr - ar
    cols = st.columns(3)
    cols[0].metric(t("{team} rest", team=home), "—" if missing(hr) else t("{n} days", n=int(hr)))
    cols[1].metric(t("{team} rest", team=away), "—" if missing(ar) else t("{n} days", n=int(ar)))
    cols[2].metric(t("Rest difference"), "—" if diff is None else t("{n} days", n=f"{diff:+.0f}"),
                   help=t("Positive = the home team had more rest."))


# ---------------------------------------------------------------------------
@guard
def render() -> None:
    match_id = _choose_match()
    if not match_id:
        return
    row = data.match_row(match_id)
    if row is None:
        page_header(t("Match Details"))
        empty_state(t("Match not found."))
        st.page_link("views/matches.py", label=t("Back to matches"), icon=":material/arrow_back:")
        return
    st.page_link("views/matches.py", label=t("Back to matches"), icon=":material/arrow_back:",
                 query_params={"date": str(pd.Timestamp(row["date"]).date()), **pref_params()})
    home, away = row["home_team"], row["away_team"]
    _header(row)

    snap = data.snapshot(match_id)
    features = data.features()
    feat_rows = features.loc[features["match_id"] == match_id]
    feat_row = feat_rows.iloc[0] if not feat_rows.empty else row

    if bool(row.get("played")):
        _result_block(row, match_id)
    if snap is None:
        st.info(t("There is no prediction for this match."))
    else:
        markets = data.match_markets(match_id)
        main, risk = _recommendations(snap, markets, home, away)
        _markets_section(snap, markets, home, away)

    section(t("Teams before the match"))
    home_tm = team_matches(features, home, row["country"], before=row["date"])
    away_tm = team_matches(features, away, row["country"], before=row["date"])
    tabs = st.tabs([t("Team comparison"), t("Recent form"), t("Home / Away"), t("Head-to-head"), t("Rest days")])
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

    if snap is not None and not feat_rows.empty:
        _explanation(feat_row, snap, markets, main, risk, home, away)


render()
