"""Matches: leagues on the left, matches of the selected date in the centre."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics.teams import all_teams
from src.leagues import get_league, leagues_by_country
from src.modelling.tasks import RESULT_CLASSES
from src.ui import data
from src.ui.components import (empty_state, esc, guard, match_url, odds_text, page_header, pct, prob_bar,
                               team_url)
from src.ui.i18n import t
from src.ui.labels import country_label


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
    days = sorted(set(pd.to_datetime(dates).date))
    if today in days:
        return today
    future = [d for d in days if d > today]
    return future[0] if future else days[-1]


def _set(**params) -> None:
    for k, v in params.items():
        st.query_params[k] = v


def _date_strip(selected: date) -> None:
    today = date.today()
    window = [selected + timedelta(days=i) for i in range(-3, 4)]
    cols = st.columns([0.55] + [1] * len(window) + [0.55, 1.1, 1.9], vertical_alignment="center")
    cols[0].button("‹", key="prev", on_click=_set, kwargs={"date": str(selected - timedelta(days=1))},
                   width="stretch", help=t("Previous day"))
    for col, day in zip(cols[1:], window):
        label = t("Today") if day == today else f"{t(day.strftime('%a'))} {day.day}"
        col.button(label, key=f"d{day}", type="primary" if day == selected else "secondary",
                   on_click=_set, kwargs={"date": str(day)}, width="stretch")
    cols[len(window) + 1].button("›", key="next", on_click=_set,
                                 kwargs={"date": str(selected + timedelta(days=1))},
                                 width="stretch", help=t("Next day"))
    cols[len(window) + 2].button(t("Today"), key="today", on_click=_set, kwargs={"date": str(today)},
                                 width="stretch", icon=":material/today:")
    picked = cols[len(window) + 3].date_input(t("Date"), value=selected, label_visibility="collapsed",
                                              format="DD.MM.YYYY")
    if picked != selected:
        _set(date=str(picked))
        st.rerun()


def _league_list(selected: str) -> None:
    st.button(t("All leagues"), key="lg_all", width="stretch",
              type="primary" if selected == "all" else "secondary", on_click=_set, kwargs={"league": "all"})
    for country, leagues in leagues_by_country().items():
        st.html(f'<div class="country-label">{esc(country_label(country))}</div>')
        for lg in leagues:
            st.button(lg.name, key=f"lg_{lg.code}", width="stretch",
                      type="primary" if selected == lg.code else "tertiary",
                      on_click=_set, kwargs={"league": lg.code})


@st.cache_data(show_spinner=False)
def _team_index(_key: int) -> pd.DataFrame:
    return all_teams(data.features())


def _team_search() -> None:
    query = st.text_input(t("Search team"), placeholder=t("Search team..."), label_visibility="collapsed",
                          key="team_search")
    if not query or len(query.strip()) < 2:
        return
    teams = _team_index(len(data.features()))
    hits = teams.loc[teams["team"].str.contains(query.strip(), case=False, regex=False)].head(8)
    if hits.empty:
        st.caption(t("No team found."))
        return
    st.html("<div class='card search-hit'>" + "".join(
        f"<div style='padding:3px 0'><a href='{team_url(r.team, r.country)}' target='_self'>{esc(r.team)}</a>"
        f" <span class='muted small'>{esc(r.league_name)} · {esc(country_label(r.country))}</span></div>"
        for r in hits.itertuples()) + "</div>")


def _day_matches(day: date) -> pd.DataFrame:
    f = data.features()
    ts = pd.Timestamp(day)
    cols = ["match_id", "league", "country", "date", "time", "home_team", "away_team", "played",
            "home_goals", "away_goals"]
    rows = f.loc[f["date"] == ts, cols] if not f.empty else pd.DataFrame(columns=cols)
    enabled = {lg.code for leagues in leagues_by_country().values() for lg in leagues}
    rows = rows.loc[rows["league"].isin(enabled)]
    snaps = data.snapshots()
    if not snaps.empty:
        keep = ["match_id", "p_home", "p_draw", "p_away"] + (["predicted_outcome"] if "predicted_outcome" in snaps
                                                             else [])
        rows = rows.merge(snaps[keep], on="match_id", how="left")
    cards = data.card_markets()
    if not cards.empty:
        rows = rows.merge(cards, left_on="match_id", right_index=True, how="left")
    return rows.sort_values(["league", "time", "home_team"])


def _card(r: pd.Series) -> str:
    played = bool(r["played"]) and r["home_goals"] == r["home_goals"]
    status = f'<span class="st ft">{t("FT")}</span>' if played else ""
    hg = int(r["home_goals"]) if played else ""
    ag = int(r["away_goals"]) if played else ""
    h_cls = a_cls = ""
    if played:
        h_cls, a_cls = ("win", "lose") if hg > ag else (("lose", "win") if hg < ag else ("", ""))
    teams = (f'<div class="mc-teams"><div class="team {h_cls}">{esc(r["home_team"])}<b>{hg}</b></div>'
             f'<div class="team {a_cls}">{esc(r["away_team"])}<b>{ag}</b></div></div>')
    has_pred = "p_home" in r and r["p_home"] == r["p_home"]
    if not has_pred:
        return (f'<a class="match-card" href="{match_url(r["match_id"])}" target="_self">'
                f'<div class="mc-time">{esc(r["time"] or "—")}{status}</div>{teams}<div></div><div></div></a>')
    outcome = r.get("predicted_outcome")
    best = RESULT_CLASSES.index(outcome) if isinstance(outcome, str) else None
    probs = prob_bar(r["p_home"], r["p_draw"], r["p_away"], (r.get("h_odds"), r.get("d_odds"), r.get("a_odds")),
                     best=best)
    over = r.get("o25_p", float("nan"))
    is_over = over == over and over >= 0.5
    total_p = over if is_over else r.get("u25_p")
    total_odds = r.get("o25_odds") if is_over else r.get("u25_odds")
    mark = ""
    if played and over == over:
        mark = (' <span class="hit">✔</span>' if (hg + ag > 2.5) == is_over else ' <span class="miss">✘</span>')
    total = (f'<div class="mc-block"><div class="cap">{t("Over 2.5") if is_over else t("Under 2.5")}</div>'
             f'<div class="val">{pct(total_p)}{mark}'
             + (f'<span class="odd">{odds_text(total_odds)}</span>' if total_odds == total_odds and total_odds else "")
             + "</div></div>")
    return (f'<a class="match-card" href="{match_url(r["match_id"])}" target="_self">'
            f'<div class="mc-time">{esc(r["time"] or "—")}{status}</div>{teams}'
            f'<div class="mc-block">{probs}</div>{total}</a>')


def _summary_panel(day_rows: pd.DataFrame, day: date) -> None:
    played = int(day_rows["played"].sum()) if not day_rows.empty else 0
    upcoming = len(day_rows) - played
    st.html(f'<div class="card"><div class="muted small">{esc(day.strftime("%d.%m.%Y"))}</div>'
            f'<div style="font-size:1.5rem;font-weight:800">{t("{n} matches", n=len(day_rows))}</div>'
            f'<div class="muted small">{t("{a} finished · {b} upcoming", a=played, b=upcoming)}</div></div>')


@guard
def render() -> None:
    page_header(t("Matches"))
    selected_day = _param_date()
    selected_league = st.query_params.get("league", "all")
    _date_strip(selected_day)

    day_rows = _day_matches(selected_day)
    left, centre, right = st.columns([1.15, 4.3, 1.3], gap="medium")
    with left, st.container(key="league_nav"):
        _league_list(selected_league)
    shown = day_rows if selected_league == "all" else day_rows.loc[day_rows["league"] == selected_league]
    with centre:
        if shown.empty:
            empty_state(t("No matches available for this date."), t("Pick another date or league."))
        for league, group in shown.groupby("league", sort=False):
            lg = get_league(league)
            st.html(f'<div class="league-head">{esc(lg.name)}'
                    f'<span class="country">{esc(country_label(lg.country))}</span></div>'
                    + "".join(_card(r) for _, r in group.iterrows()))
    with right:
        _team_search()
        _summary_panel(shown, selected_day)


render()
