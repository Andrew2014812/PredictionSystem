"""Matches: leagues on the left, matches of the selected date in the centre."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.leagues import get_league, leagues_by_country
from src.ui import data
from src.ui.components import (empty_state, esc, guard, league_title, match_url, odds_text,
                               page_header, pct, prob_bar, signed, source_badge)
from src.analytics.performance import summarize


def _param_date() -> date:
    raw = st.query_params.get("date")
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    today = date.today()
    dates = data.match_dates()
    if len(dates) == 0:
        return today
    days = pd.to_datetime(dates).date
    if today in set(days):
        return today
    future = [d for d in days if d > today]
    return future[0] if future else days[-1]


def _set(**params) -> None:
    for k, v in params.items():
        st.query_params[k] = v


def _date_strip(selected: date) -> None:
    today = date.today()
    window = [selected + timedelta(days=i) for i in range(-3, 4)]
    cols = st.columns([0.6] + [1] * len(window) + [0.6, 1.1, 1.6], vertical_alignment="center")
    cols[0].button("‹", key="prev", on_click=_set, kwargs={"date": str(selected - timedelta(days=1))},
                   width="stretch", help="Previous day")
    for col, day in zip(cols[1:], window):
        label = "Today" if day == today else day.strftime("%a %d")
        col.button(label, key=f"d{day}", type="primary" if day == selected else "secondary",
                   on_click=_set, kwargs={"date": str(day)}, width="stretch")
    cols[len(window) + 1].button("›", key="next", on_click=_set,
                                 kwargs={"date": str(selected + timedelta(days=1))},
                                 width="stretch", help="Next day")
    cols[len(window) + 2].button("Today", key="today", on_click=_set, kwargs={"date": str(today)},
                                 width="stretch", icon=":material/today:")
    picked = cols[len(window) + 3].date_input("Jump to date", value=selected, label_visibility="collapsed",
                                              format="DD.MM.YYYY")
    if picked != selected:
        _set(date=str(picked))
        st.rerun()


def _league_list(selected: str, counts: dict[str, int]) -> None:
    st.button(f"All leagues  ·  {sum(counts.values())}", key="lg_all", width="stretch",
              type="primary" if selected == "all" else "secondary", on_click=_set, kwargs={"league": "all"})
    for country, leagues in leagues_by_country().items():
        st.html(f'<div class="country-label">{esc(country)}</div>')
        for lg in leagues:
            n = counts.get(lg.code, 0)
            st.button(lg.name + (f"  ·  {n}" if n else ""), key=f"lg_{lg.code}",
                      width="stretch", type="primary" if selected == lg.code else "tertiary",
                      on_click=_set, kwargs={"league": lg.code})


def _day_matches(day: date) -> pd.DataFrame:
    f = data.features()
    ts = pd.Timestamp(day)
    cols = ["match_id", "league", "country", "date", "time", "home_team", "away_team", "played",
            "home_goals", "away_goals"]
    rows = f.loc[f["date"] == ts, cols] if not f.empty else pd.DataFrame(columns=cols)
    snaps = data.snapshots()
    if not snaps.empty:
        extra = snaps.loc[(snaps["date"] == ts) & ~snaps["match_id"].isin(rows["match_id"]),
                          ["match_id", "league", "country", "date", "time", "home_team", "away_team"]]
        if not extra.empty:
            extra = extra.assign(played=False, home_goals=float("nan"), away_goals=float("nan"))
            rows = pd.concat([rows, extra], ignore_index=True)
        rows = rows.merge(snaps[["match_id", "p_home", "p_draw", "p_away", "source"]], on="match_id", how="left")
    cards = data.card_markets()
    if not cards.empty:
        rows = rows.merge(cards, left_on="match_id", right_index=True, how="left")
    return rows.sort_values(["league", "time", "home_team"])


def _card(r: pd.Series) -> str:
    played = bool(r["played"]) and r["home_goals"] == r["home_goals"]
    status = '<span class="st ft">FT</span>' if played else '<span class="st">UPCOMING</span>'
    time = esc(r["time"] or "—")
    hg = int(r["home_goals"]) if played else ""
    ag = int(r["away_goals"]) if played else ""
    h_cls = a_cls = ""
    if played:
        h_cls, a_cls = ("win", "lose") if hg > ag else (("lose", "win") if hg < ag else ("", ""))
    teams = (f'<div class="mc-teams"><div class="team {h_cls}">{esc(r["home_team"])}<b>{hg}</b></div>'
             f'<div class="team {a_cls}">{esc(r["away_team"])}<b>{ag}</b></div></div>')

    has_pred = "p_home" in r and r["p_home"] == r["p_home"]
    if has_pred:
        probs = prob_bar(r["p_home"], r["p_draw"], r["p_away"],
                         (r.get("h_odds"), r.get("d_odds"), r.get("a_odds")),
                         (r.get("h_src"), r.get("d_src"), r.get("a_src")))
    else:
        probs = '<span class="muted small">No prediction (training period)</span>'

    def side(cap, p_key, odds_key, src_key, won=None):
        p = r.get(p_key)
        if p is None or p != p:
            return '<div class="mc-block"></div>'
        mark = ""
        if won is not None:
            mark = ' <span class="hit">✔</span>' if won else ' <span class="miss">✘</span>'
        return (f'<div class="mc-block"><div class="cap">{cap}</div><div class="val">{pct(p)}{mark}'
                f'<span class="odd">{odds_text(r.get(odds_key), r.get(src_key))}</span> '
                f'{source_badge(r.get(src_key))}</div></div>')

    over_won = btts_won = None
    if played and has_pred:
        over_won = (hg + ag > 2.5) == (r.get("o25_p", 0) >= 0.5)
        btts_won = (hg > 0 and ag > 0) == (r.get("btts_p", 0) >= 0.5)
    over_cap = "Over 2.5" if r.get("o25_p", 0) >= 0.5 else "Under 2.5"
    over_key = ("o25_p", "o25_odds", "o25_src") if r.get("o25_p", 0) >= 0.5 else ("u25_p", "u25_odds", "u25_src")
    btts_cap = "BTTS Yes" if r.get("btts_p", 0) >= 0.5 else "BTTS No"
    btts_key = ("btts_p", "btts_odds", "btts_src") if r.get("btts_p", 0) >= 0.5 else (
        "nobtts_p", "nobtts_odds", "nobtts_src")
    return (f'<a class="match-card" href="{match_url(r["match_id"])}" target="_self">'
            f'<div class="mc-time">{time}{status}</div>{teams}'
            f'<div class="mc-block"><div class="cap">1 · X · 2</div>{probs}</div>'
            f'{side(over_cap, *over_key, won=over_won) if has_pred else "<div></div>"}'
            f'{side(btts_cap, *btts_key, won=btts_won) if has_pred else "<div></div>"}</a>')


def _summary_panel(day_rows: pd.DataFrame, day: date) -> None:
    played = int(day_rows["played"].sum()) if not day_rows.empty else 0
    predicted = int(day_rows["p_home"].notna().sum()) if "p_home" in day_rows else 0
    st.html(f'<div class="card"><div class="cap muted small">{day.strftime("%A, %d %B %Y")}</div>'
            f'<div style="font-size:1.6rem;font-weight:800">{len(day_rows)} matches</div>'
            f'<div class="muted small">{played} finished · {len(day_rows) - played} upcoming · '
            f'{predicted} predicted</div></div>')
    picks = data.picks()
    if picks.empty or day_rows.empty:
        return
    day_picks = picks.loc[picks["match_id"].isin(day_rows["match_id"]) & (picks["role"] == "market")]
    s = summarize(day_picks)
    if s["settled"]:
        st.html(f'<div class="card"><div class="cap muted small">Market predictions this day</div>'
                f'<div style="font-size:1.3rem;font-weight:800">{pct(s["hit_rate"])} hit rate</div>'
                f'<div class="muted small">{s["won"]} won · {s["lost"]} lost · profit '
                f'<span class="{"pos" if s["profit"] > 0 else "neg"}">{signed(s["profit"])}</span> units</div></div>')
    st.html('<div class="card small muted">Probabilities come from the models; odds marked '
            '<span class="badge market">BOOK</span> are real bookmaker averages, '
            '<span class="badge derived">DERIVED</span> are computed from bookmaker 1X2 / totals prices, '
            '<span class="badge simulated">SIM</span> are simulated. Click a match for details.</div>')


@guard
def render() -> None:
    page_header("Matches", "Fixtures, results and model probabilities by date")
    selected_day = _param_date()
    selected_league = st.query_params.get("league", "all")
    _date_strip(selected_day)

    day_rows = _day_matches(selected_day)
    counts = day_rows.groupby("league").size().to_dict() if not day_rows.empty else {}
    left, centre, right = st.columns([1.15, 4.2, 1.35], gap="medium")
    with left, st.container(key="league_nav"):
        _league_list(selected_league, counts)
    shown = day_rows if selected_league == "all" else day_rows.loc[day_rows["league"] == selected_league]
    with centre:
        if shown.empty:
            league_txt = "" if selected_league == "all" else f" in {get_league(selected_league).name}"
            empty_state(f"No matches on {selected_day.strftime('%d %b %Y')}{league_txt}",
                        "Pick another date or league. Future fixtures appear once Football-Data publishes them.")
        for league, group in shown.groupby("league", sort=False):
            lg = get_league(league)
            st.html(f'<div class="league-head">{league_title(league)}'
                    f'<span class="country">{esc(lg.country)}</span></div>'
                    + "".join(_card(r) for _, r in group.iterrows()))
    with right:
        _summary_panel(shown, selected_day)


render()
