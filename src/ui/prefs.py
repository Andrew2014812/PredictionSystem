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


def top_bar() -> None:
    """Brand on the left, language and theme switches on the right."""
    inject_css()
    st.html('<div class="fp-brand"><span class="ball">⚽</span><span>Foot<span class="accent">Predict</span></span></div>')
    with st.container(key="fp_prefs", horizontal=True):
        st.session_state["w_lang"] = st.session_state["lang"]
        st.segmented_control("Language", list(LANGUAGES), format_func=LANGUAGES.get, key="w_lang",
                             label_visibility="collapsed", on_change=_set_pref, args=("lang", "w_lang"))
        st.session_state["w_theme"] = st.session_state["theme"]
        st.segmented_control("Theme", list(THEMES), key="w_theme", label_visibility="collapsed",
                             format_func=lambda x: ":material/dark_mode:" if x == "dark" else ":material/light_mode:",
                             help=t("Dark / light theme"), on_change=_set_pref, args=("theme", "w_theme"))
