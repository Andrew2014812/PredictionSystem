"""Per-session preferences (language, theme) and the global top bar.

Preferences live in ``st.session_state`` and are mirrored in the URL query
(``?lang=uk&theme=light``) so they survive full page loads from links.
"""
from __future__ import annotations

import base64
import functools
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from .theme import inject_css

THEMES = ("dark", "light")


SHOW_LANGUAGE_SWITCH = False          # the interface is English only for now


def init_prefs() -> None:
    qp = st.query_params
    if SHOW_LANGUAGE_SWITCH and qp.get("lang") in LANGUAGES:
        st.session_state["lang"] = qp["lang"]
    elif not SHOW_LANGUAGE_SWITCH:
        st.session_state["lang"] = DEFAULT_LANGUAGE     # old ?lang=uk links simply open in English
    if qp.get("theme") in THEMES:
        st.session_state["theme"] = qp["theme"]
    st.session_state.setdefault("lang", DEFAULT_LANGUAGE)
    st.session_state.setdefault("theme", "dark")


def pref_params() -> dict:
    """Non-default preferences, to carry over in links."""
    out = {}
    if st.session_state.get("lang", DEFAULT_LANGUAGE) != DEFAULT_LANGUAGE:
        out["lang"] = st.session_state["lang"]
    if st.session_state.get("theme", "dark") != "dark":
        out["theme"] = st.session_state["theme"]
    return out


def href(path: str, **params) -> str:
    """Relative link that keeps the language and theme."""
    query = {**params, **pref_params()}
    return f"{path}?{urlencode(query)}" if query else path


def _set_pref(key: str, widget_key: str) -> None:
    value = st.session_state.get(widget_key)
    if value is None:
        return
    st.session_state[key] = value
    if value in (DEFAULT_LANGUAGE, "dark"):
        st.query_params.pop(key, None)
    else:
        st.query_params[key] = value


# Small football used as the logo (inline SVG: crisp at any size, follows the theme colours).
ASSETS = Path(__file__).resolve().parents[2] / "assets"


@functools.lru_cache(maxsize=None)
def logo_data_uri(theme: str) -> str:
    """The FP mark (white F on the dark theme, navy F on the light theme) as a data URI: st.html strips
    inline <svg>, an <img> with an SVG data URI stays crisp at any size."""
    svg = (ASSETS / f"footpredict-mark-{theme}.svg").read_text(encoding="utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def app_root() -> str:
    """Path of the app root (the default page, Matches) as the browser sees it — "/" locally,
    "/~/+/" inside Streamlit Cloud's frame. The default page lives at the root, not at "/matches"
    (that address made Streamlit show "Page not found")."""
    try:
        path = urlparse(st.context.url or "").path or "/"
    except Exception:                                   # no browser context (tests)
        return "./"
    pages = ("match", "history", "leagues", "teams", "analytics", "models", "matches")
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1] in pages:
        parts = parts[:-1]
    return "/" + "/".join(parts) + ("/" if parts else "")


# Focus the search field as soon as the popover opens (the widget itself filters while typing).
_FOCUS_JS = """<script>
const doc = window.parent.document;
new MutationObserver(() => {
  const input = doc.querySelector('[data-testid="stPopoverBody"] .st-key-global_search input');
  if (input && !input.dataset.fpFocused) { input.dataset.fpFocused = "1"; input.focus(); }
}).observe(doc.body, {childList: true, subtree: true});
</script>"""


@st.cache_data(show_spinner=False)
def _team_options(_key: int) -> dict[str, str]:
    from ..analytics.teams import all_teams
    from . import data
    from .labels import league_name
    from ..providers.aliases import TEAM_ALIASES
    full: dict[str, str] = {}
    for alias, team in TEAM_ALIASES.items():          # "Man City" is also found as "Manchester City"
        if len(alias) > len(full.get(team, "")) and alias.replace(" ", "") != team.lower().replace(" ", ""):
            full[team] = alias.title()
    teams = all_teams(data.features())
    recent = teams.loc[teams["date"] >= teams["date"].max() - pd.Timedelta(days=550)]
    return {f"{r.team}|{r.country}": f"{r.team}{f' ({full[r.team]})' if r.team in full else ''} · "
                                     f"{league_name(r.league)}" for r in recent.itertuples()}


def _remember_team() -> None:
    choice = st.session_state.get("global_search")
    if choice:
        st.session_state["_go_team"] = choice
        st.session_state["global_search"] = None          # an empty field next time


def _team_search() -> None:
    """Magnifier in the top bar: a searchable list of all current clubs (local data, no API call);
    typing filters instantly, arrows + Enter or a click choose, Esc closes; opens the team page."""
    from . import data
    target = st.session_state.pop("_go_team", None)
    if target:
        team, country = target.split("|", 1)
        st.switch_page("views/teams.py", query_params={"team": team, "country": country, **pref_params()})
    options = _team_options(len(data.features()))
    with st.popover("", icon=":material/search:", help=t("Search team"), key="fp_search"):
        st.selectbox(t("Search team"), list(options), index=None, format_func=options.get, key="global_search",
                     placeholder=t("Start typing a club, e.g. Man"), label_visibility="collapsed",
                     on_change=_remember_team)
    components.html(_FOCUS_JS, height=0)


def top_bar() -> None:
    """Brand on the left, language and theme switches on the right."""
    inject_css()
    query = urlencode(pref_params())
    target = app_root() + (f"?{query}" if query else "")
    st.html(f'<div class="fp-brand"><a href="{target}" target="_self" aria-label="FootPredict — {t("Matches")}">'
            f'<img class="mark" src="{logo_data_uri(st.session_state.get("theme", "dark"))}" width="40" height="29" '
            f'alt=""><span>Foot<span class="accent">Predict</span></span></a></div>')
    with st.container(key="fp_prefs", horizontal=True):
        _team_search()
        if SHOW_LANGUAGE_SWITCH:                       # English only for now; translations stay in the code
            st.session_state["w_lang"] = st.session_state["lang"]
            st.segmented_control("Language", list(LANGUAGES), format_func=LANGUAGES.get, key="w_lang",
                                 label_visibility="collapsed", on_change=_set_pref, args=("lang", "w_lang"))
        st.session_state["w_theme"] = st.session_state["theme"]
        st.segmented_control("Theme", list(THEMES), key="w_theme", label_visibility="collapsed",
                             format_func=lambda x: ":material/dark_mode:" if x == "dark" else ":material/light_mode:",
                             help=t("Dark / light theme"), on_change=_set_pref, args=("theme", "w_theme"))
