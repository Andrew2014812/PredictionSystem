"""Reusable formatting helpers and HTML components of the interface."""
from __future__ import annotations

import functools
import html
import logging
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

from ..leagues import get_league
from ..markets.markets import market_title, selection_label
from ..markets.odds import SOURCE_LABELS
from .theme import COLORS, inject_css

log = logging.getLogger(__name__)

SOURCE_SHORT = {"market": "BOOK", "derived": "DERIVED", "simulated": "SIM"}


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA or x is pd.NaT


def pct(x, digits: int = 0) -> str:
    return "—" if missing(x) else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 2) -> str:
    return "—" if missing(x) else f"{x:.{digits}f}"


def signed(x, digits: int = 2, suffix: str = "") -> str:
    return "—" if missing(x) else f"{x:+.{digits}f}{suffix}"


def odds_text(odds, source) -> str:
    if missing(odds):
        return "—"
    return f"{odds:.2f}" if source == "market" else f"≈{odds:.2f}"


def esc(text) -> str:
    return html.escape(str(text))


def date_text(day, fmt: str = "%a %d %b %Y") -> str:
    return "—" if missing(day) else pd.Timestamp(day).strftime(fmt)


def match_url(match_id: str) -> str:
    return f"match?match_id={quote(match_id)}"


def team_url(team: str, country: str) -> str:
    return f"teams?team={quote(team)}&country={quote(country)}"


def selection_text(market: str, selection: str, home: str, away: str) -> str:
    return selection_label(market, selection, home, away)


# ---------------------------------------------------------------------------
# page scaffolding
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    inject_css()
    st.html(f'<div class="page-title">{esc(title)}</div>'
            + (f'<div class="page-subtitle">{esc(subtitle)}</div>' if subtitle else ""))


def section(title: str, hint: str = "") -> None:
    st.html(f'<div class="section-title">{esc(title)}'
            + (f'<span class="hint">{esc(hint)}</span>' if hint else "") + "</div>")


def empty_state(title: str, text: str = "") -> None:
    st.html(f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>')


def guard(page_fn):
    """Run a page and show a friendly message instead of a traceback."""
    @functools.wraps(page_fn)
    def wrapper(*args, **kwargs):
        # Streamlit's st.stop / st.rerun / st.switch_page raise BaseException
        # subclasses, so they pass through untouched.
        try:
            return page_fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - last line of defence for the UI
            log.exception("page failed")
            st.error("Something went wrong while building this page. "
                     "The data may be incomplete — try running `python scripts/run_pipeline.py`.",
                     icon=":material/error:")
    return wrapper


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value {tone}">{value}</div><div class="sub">{esc(sub)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.html(kpi(*item))


def tone(x) -> str:
    return "" if missing(x) else ("pos" if x > 0 else ("neg" if x < 0 else ""))


# ---------------------------------------------------------------------------
# small HTML pieces
# ---------------------------------------------------------------------------
def source_badge(source) -> str:
    if missing(source) or not source:
        return ""
    return f'<span class="badge {source}" title="{esc(SOURCE_LABELS.get(source, source))}">{SOURCE_SHORT.get(source, source)}</span>'


def status_badge(status) -> str:
    return f'<span class="badge {status}">{esc(status)}</span>'


def form_chips(form: str) -> str:
    return "".join(f'<span class="chip {c}">{c}</span>' for c in form) or '<span class="muted">—</span>'


def prob_bar(ph, pd_, pa, odds=(None, None, None), sources=(None, None, None),
             labels=("1", "X", "2")) -> str:
    if missing(ph):
        return '<span class="muted small">no prediction</span>'
    best = int(np.argmax([ph, pd_, pa]))
    bar = (f'<div class="pbar"><span style="width:{ph * 100:.1f}%;background:{COLORS["home"]}"></span>'
           f'<span style="width:{pd_ * 100:.1f}%;background:{COLORS["draw"]}"></span>'
           f'<span style="width:{pa * 100:.1f}%;background:{COLORS["away"]}"></span></div>')
    parts = []
    for i, (lbl, p, o, s) in enumerate(zip(labels, (ph, pd_, pa), odds, sources)):
        odd = f'<span class="o">{odds_text(o, s)}</span>' if not missing(o) else ""
        parts.append(f'<span class="{"best" if i == best else ""}">{lbl} <b>{p * 100:.0f}%</b>{odd}</span>')
    return bar + f'<div class="plabels">{"".join(parts)}</div>'


def mini_bar(p: float, color: str = COLORS["positive"]) -> str:
    width = 0 if missing(p) else max(0, min(100, p * 100))
    return f'<div class="minibar"><span style="width:{width:.0f}%;background:{color}"></span></div>'


def league_title(code: str) -> str:
    lg = get_league(code)
    return lg.name


def market_table(rows: pd.DataFrame, home: str, away: str, highlight_best: bool = True,
                 show_ev: bool = True, bars: bool = False) -> str:
    """HTML table of priced selections (probability, fair odds, odds, source, EV)."""
    if rows.empty:
        return '<span class="muted small">Not available.</span>'
    best = rows["probability"].idxmax() if highlight_best else None
    head = ("<tr><th>Selection</th>" + ("<th></th>" if bars else "") + "<th class='num'>Prob.</th><th class='num'>Fair</th>"
            "<th class='num'>Odds</th><th></th>" + ("<th class='num'>EV</th>" if show_ev else "") + "</tr>")
    body = []
    for idx, r in rows.iterrows():
        ev = r.get("ev")
        ev_cell = (f"<td class='num {tone(ev)}'>{signed(ev * 100 if not missing(ev) else ev, 1, '%')}</td>"
                   if show_ev else "")
        body.append(
            f"<tr class='{'top' if idx == best else ''}'>"
            f"<td>{esc(selection_label(r['market'], r['selection'], home, away))}</td>"
            + (f"<td style='width:90px'>{mini_bar(r['probability'])}</td>" if bars else "") +
            f"<td class='num'>{pct(r['probability'], 1)}</td>"
            f"<td class='num'>{num(r['fair_odds'])}</td>"
            f"<td class='num'>{odds_text(r['odds'], r['odds_source'])}</td>"
            f"<td>{source_badge(r['odds_source'])}</td>{ev_cell}</tr>")
    return f"<div style='overflow-x:auto'><table class='mkt'>{head}{''.join(body)}</table></div>"


def recommendation_card(role: str, rec: pd.Series | dict, home: str, away: str, kind: str = "") -> str:
    title = "Main prediction" if role == "main" else "Risk prediction"
    label = selection_label(rec["market"], rec["selection"], home, away)
    ev = rec.get("ev")
    note = {
        ("main", "value"): "Chosen by the value rule: highest reliability-weighted expected value among "
                           "selections with enough probability and moderate odds.",
        ("main", "confidence"): "No selection passed the minimum EV, so the most probable eligible "
                                "selection is shown. Not a value bet.",
        ("risk", "value"): "Higher-odds alternative (odds ≥ 3.0) with positive expected value. "
                           "High variance — most such picks lose.",
    }.get((role, kind or "value"), "")
    return (
        f'<div class="rec {role}"><div class="tag">{title}</div>'
        f'<div class="pick">{esc(label)}</div><div class="mkt">{esc(market_title(rec["market"]))} '
        f'{source_badge(rec.get("odds_source"))}</div>'
        f'<div class="grid">'
        f'<div><div class="l">Probability</div><div class="v">{pct(rec["probability"], 1)}</div></div>'
        f'<div><div class="l">{"Odds" if rec.get("odds_source") == "market" else "Est. odds"}</div>'
        f'<div class="v">{num(rec.get("odds"))}</div></div>'
        f'<div><div class="l">Fair odds</div><div class="v">{num(rec.get("fair_odds"))}</div></div>'
        f'<div><div class="l">EV</div><div class="v {tone(ev)}">'
        f'{signed(ev * 100 if not missing(ev) else ev, 1, "%")}</div></div>'
        f'</div><div class="note">{esc(note)}</div></div>')


def plotly(fig, height: int = 320) -> None:
    from .theme import PLOTLY_LAYOUT
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
