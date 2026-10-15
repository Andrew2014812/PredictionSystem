"""Per-session preferences (language, theme) and the global top bar.

Preferences live in ``st.session_state`` and are mirrored in the URL query
(``?lang=uk&theme=light``) so they survive full page loads from links.
"""
from __future__ import annotations

from urllib.parse import urlencode

import streamlit as st

from .i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from .theme import inject_css

THEMES = ("dark", "light")


def init_prefs() -> None:
    qp = st.query_params
    if qp.get("lang") in LANGUAGES:
        st.session_state["lang"] = qp["lang"]
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
BALL_SVG = ('<svg class="ball" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
            '<circle cx="12" cy="12" r="10.4" fill="var(--surface)" stroke="currentColor" stroke-width="1.5"/>'
            '<path d="M12 7.3l3.5 2.5-1.3 4.1H9.8L8.5 9.8z" fill="currentColor"/>'
            '<path d="M12 7.3V3.4M15.5 9.8l3.6-1.3M14.2 13.9l2.3 3.2M9.8 13.9l-2.3 3.2M8.5 9.8 4.9 8.5" '
            'stroke="currentColor" stroke-width="1.3" fill="none" stroke-linecap="round"/></svg>')


def top_bar() -> None:
    """Brand on the left, language and theme switches on the right."""
    inject_css()
    st.html(f'<div class="fp-brand"><a href="{href("matches")}" target="_self" aria-label="FootPredict — '
            f'{t("Matches")}">{BALL_SVG}<span>Foot<span class="accent">Predict</span></span></a></div>')
    with st.container(key="fp_prefs", horizontal=True):
        st.session_state["w_lang"] = st.session_state["lang"]
        st.segmented_control("Language", list(LANGUAGES), format_func=LANGUAGES.get, key="w_lang",
                             label_visibility="collapsed", on_change=_set_pref, args=("lang", "w_lang"))
        st.session_state["w_theme"] = st.session_state["theme"]
        st.segmented_control("Theme", list(THEMES), key="w_theme", label_visibility="collapsed",
                             format_func=lambda x: ":material/dark_mode:" if x == "dark" else ":material/light_mode:",
                             help=t("Dark / light theme"), on_change=_set_pref, args=("theme", "w_theme"))
