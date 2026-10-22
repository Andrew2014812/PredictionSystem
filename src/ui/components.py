"""Reusable formatting helpers and HTML components of the interface."""
from __future__ import annotations

import functools
import html
import logging

import numpy as np
import pandas as pd
import streamlit as st

from .i18n import t
from .labels import market_title, selection_label, status_label
from .prefs import href
from .theme import colors, plotly_layout

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def missing(x) -> bool:
    """True for None, NaN of any float type (incl. numpy.float32), pd.NA, NaT and inf."""
    if x is None or x is pd.NA or x is pd.NaT:
        return True
    if isinstance(x, (float, np.floating)):
        return not np.isfinite(x)
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def pct(x, digits: int = 0) -> str:
    return "—" if missing(x) else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 2) -> str:
    return "—" if missing(x) else f"{x:.{digits}f}"


def signed(x, digits: int = 2, suffix: str = "") -> str:
    return "—" if missing(x) else f"{x:+.{digits}f}{suffix}"


def odds_text(odds) -> str:
    return "—" if missing(odds) or not odds else f"{odds:.2f}"


def esc(text) -> str:
    return html.escape(str(text))


def date_text(day, fmt: str = "%d.%m.%Y") -> str:
    return "—" if missing(day) else pd.Timestamp(day).strftime(fmt)


def match_url(match_id: str) -> str:
    return href("match", match_id=match_id)


def team_url(team: str, country: str) -> str:
    return href("teams", team=team, country=country)


def tone(x) -> str:
    return "" if missing(x) else ("pos" if x > 0 else ("neg" if x < 0 else ""))


def tip(text: str) -> str:
    """Small (i) icon with a hover tooltip."""
    return f'<span class="tip" title="{esc(text)}">i</span>'


EV_HELP = ("Expected Value estimates the model's theoretical advantage at the current odds: "
           "EV = model probability × odds − 1. It is not a guaranteed profit.")


# ---------------------------------------------------------------------------
# page scaffolding
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
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
            st.error(t("This page could not be displayed right now. Please try again later."),
                     icon=":material/error:")
    return wrapper


def kpi(label: str, value: str, sub: str = "", tone_cls: str = "", help_text: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{esc(label)}{tip(help_text) if help_text else ""}</div>'
            f'<div class="value {tone_cls}">{value}</div><div class="sub">{esc(sub)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.html(kpi(*item))


# ---------------------------------------------------------------------------
# small HTML pieces
# ---------------------------------------------------------------------------
def status_badge(status) -> str:
    return f'<span class="status {status}">{esc(status_label(status))}</span>'


def form_chips(form: str) -> str:
    letters = {"W": t("W"), "D": t("D"), "L": t("L")}
    return "".join(f'<span class="chip {c}">{letters[c]}</span>' for c in form) or '<span class="muted">—</span>'


def prob_bar(ph, pd_, pa, odds=(None, None, None), labels=None, best: int | None = None) -> str:
    """Three-column 1X2 block: Home / Draw / Away with probability and real odds."""
    if missing(ph):
        return f'<span class="muted small">{t("No prediction")}</span>'
    c = colors()
    labels = labels or (t("Home"), t("Draw"), t("Away"))
    best = int(np.argmax([ph, pd_, pa])) if best is None else best
    bar = (f'<div class="pbar"><span style="width:{ph * 100:.1f}%;background:{c["home"]}"></span>'
           f'<span style="width:{pd_ * 100:.1f}%;background:{c["draw"]}"></span>'
           f'<span style="width:{pa * 100:.1f}%;background:{c["away"]}"></span></div>')
    parts = []
    for i, (lbl, p, o) in enumerate(zip(labels, (ph, pd_, pa), odds)):
        odd = f'<span class="o">{odds_text(o)}</span>' if not missing(o) and o else ""
        parts.append(f'<div class="{"best" if i == best else ""}">{esc(lbl)} <b>{p * 100:.0f}%</b>{odd}</div>')
    return bar + f'<div class="p3">{"".join(parts)}</div>'


def mini_bar(p: float, color: str | None = None) -> str:
    width = 0 if missing(p) else max(0, min(100, p * 100))
    color = color or colors()["accent"]
    return f'<div class="minibar"><span style="width:{width:.0f}%;background:{color}"></span></div>'


def market_table(rows: pd.DataFrame, home: str, away: str, highlight_best: bool = True, bars: bool = False) -> str:
    """Selections with model probability, real odds (if any) and EV."""
    if rows.empty:
        return f'<span class="muted small">{t("Not available for this match.")}</span>'
    has_odds = rows["odds"].notna().any()
    best = rows["probability"].idxmax() if highlight_best else None
    head = (f"<tr><th>{t('Selection')}</th>" + ("<th></th>" if bars else "")
            + f"<th class='num'>{t('Probability')}</th>"
            + (f"<th class='num'>{t('Odds')}</th><th class='num'>EV{tip(t(EV_HELP))}</th>" if has_odds else "")
            + "</tr>")
    body = []
    for idx, r in rows.iterrows():
        ev = r.get("ev")
        odds_cells = (f"<td class='num'>{odds_text(r['odds'])}</td>"
                      f"<td class='num {tone(ev)}'>{signed(ev * 100 if not missing(ev) else ev, 1, '%')}</td>"
                      if has_odds else "")
        body.append(f"<tr class='{'top' if idx == best else ''}'>"
                    f"<td>{esc(selection_label(r['market'], r['selection'], home, away))}</td>"
                    + (f"<td style='width:90px'>{mini_bar(r['probability'])}</td>" if bars else "")
                    + f"<td class='num'>{pct(r['probability'], 1)}</td>{odds_cells}</tr>")
    return f"<div style='overflow-x:auto'><table class='mkt'>{head}{''.join(body)}</table></div>"


PROVIDER_NAMES = {"oddspapi": "OddsPapi", "the_odds_api": "The Odds API", "api_football": "API-Football",
                  "football_data": "Football-Data"}


def price_origin(row) -> str:
    """"Bet365 · OddsPapi · 17 Oct, 15:42" (the visitor's time) for the small tooltip next to a price."""
    from .timefmt import local_stamp
    parts = [str(row.get("bookmaker") or "").strip(), PROVIDER_NAMES.get(row.get("provider"), row.get("provider"))]
    when = row.get("priced_at") if not missing(row.get("priced_at")) else row.get("odds_priced_at")
    parts.append(local_stamp(when) if not missing(when) else "")
    return " · ".join(p for p in parts if p) or t("Real bookmaker price")


def recommendation_card(role: str, rec: pd.Series | dict, home: str, away: str,
                        settled: pd.Series | dict | None = None) -> str:
    """Main (green) or Risk (orange) card: market, selection, probability, real odds and EV;
    for a finished match also the result (Won / Lost / Refund / Void) and the profit."""
    title = t("Main prediction") if role == "main" else t("Risk prediction")
    label = selection_label(rec["market"], rec["selection"], home, away)
    source = settled if settled is not None else rec
    odds, ev = source.get("odds"), source.get("ev")
    priced = not missing(odds) and bool(odds)
    cells = [f'<div><div class="l">{t("Probability")}</div><div class="v">{pct(rec["probability"], 1)}</div></div>']
    push = rec.get("push_probability")
    if not missing(push) and push and push > 0.001:
        cells.append(f'<div><div class="l">{t("Refund")}</div><div class="v">{pct(push, 1)}</div></div>')
    if priced:
        cells.append(f'<div><div class="l">{t("Odds")}{tip(price_origin(source))}</div>'
                     f'<div class="v">{odds_text(odds)}</div></div>')
        if not missing(ev):
            cells.append(f'<div><div class="l">{t("Expected Value")}{tip(t(EV_HELP))}</div>'
                         f'<div class="v {tone(ev)}">{signed(ev * 100, 1, "%")}</div></div>')
    else:
        cells.append(f'<div><div class="l">{t("Odds")}</div><div class="v na">{t("No real odds")}</div></div>')
    status = ""
    if settled is not None:
        st_value = settled.get("status")
        cells.append(f'<div><div class="l">{t("Result")}</div><div class="v">{status_badge(st_value)}</div></div>')
        profit = settled.get("profit")
        if not missing(profit):
            cells.append(f'<div><div class="l">{t("Profit")}</div><div class="v {tone(profit)}">{signed(profit)}</div></div>')
        status = f" {st_value}"
    note = (f'<div class="note">{t("Higher odds, higher risk: most such predictions lose, a few pay well.")}</div>'
            if role == "risk" and settled is None else "")
    return (f'<div class="rec {role}{status}"><div class="tag">{title}</div>'
            f'<div class="pick">{esc(label)}</div><div class="mkt">{esc(market_title(rec["market"]))}</div>'
            f'<div class="grid">{"".join(cells)}</div>{note}</div>')


def html_table(df: pd.DataFrame, formats: dict | None = None, align_right: tuple = (),
               link_col: str | None = None) -> str:
    """Theme-aware HTML table. ``formats`` maps column -> callable(value) -> html."""
    if df.empty:
        return ""
    formats = formats or {}
    head = "".join(f"<th class='{'num' if c in align_right else ''}'>{esc(c)}</th>" for c in df.columns
                   if c != link_col)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            if c == link_col:
                continue
            v = r[c]
            text = formats[c](v) if c in formats else ("—" if missing(v) else esc(v))
            cells.append(f"<td class='{'num' if c in align_right else ''}'>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tablewrap'><table class='mkt'><tr>{head}</tr>{''.join(rows)}</table></div>"


def bar_html(p: float, color: str) -> str:
    return f'<div class="minibar"><span style="width:{max(0, min(100, p * 100)):.0f}%;background:{color}"></span></div>'


def plotly(fig, height: int = 320) -> None:
    fig.update_layout(**plotly_layout(), height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def period_filter(col, key: str, first_date, default: str = "All time"):
    """Period selector with presets, a specific month / year and custom dates; returns (start, end)."""
    from datetime import date

    from .. import config
    from ..analytics.performance import DATE_PRESETS, preset_range
    preset = col.selectbox(t("Period"), DATE_PRESETS, index=DATE_PRESETS.index(default), format_func=t,
                           key=f"{key}_preset")
    first = pd.Timestamp(first_date)
    today = date.today()
    custom = None
    if preset == "Specific month":
        years = list(range(today.year, min(first.year, today.year) - 1, -1))
        c1, c2 = col.columns(2)
        month = c1.selectbox(t("Month"), range(1, 13), index=today.month - 1, format_func=lambda m: t(MONTHS[m - 1]),
                             key=f"{key}_month")
        year = c2.selectbox(t("Year"), years, key=f"{key}_year")
        custom = (year, month)
    elif preset == "Custom dates":
        custom = col.date_input(t("Date range"), value=(first.date(), today), format="DD.MM.YYYY",
                                key=f"{key}_custom")
    season_start = pd.Timestamp(year=config.current_season(), month=7, day=1)
    return preset_range(preset, today, first, custom, season_start)


# Neutral shield shown when no badge is known or the image cannot be loaded.
SHIELD_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
              "<path d='M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5z' fill='%2394a3b8' opacity='.55'/></svg>")


def team_badge(country: str, team: str, size: int = 24) -> str:
    """Small club badge (lazy-loaded from the provider's media server) or a neutral shield."""
    from . import data
    url = data.team_logo(country, team) or SHIELD_SVG
    return (f'<img class="badge" src="{esc(url)}" width="{size}" height="{size}" alt="" loading="lazy" '
            f'referrerpolicy="no-referrer" onerror="this.onerror=null;this.src=&quot;{SHIELD_SVG}&quot;">')
