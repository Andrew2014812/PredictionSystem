"""Translation layer (EN default, UA).

Texts are written in English in the code and wrapped in ``t()``:

    t("Matches")                       -> "Матчі" when the language is UA
    t("{n} matches", n=5)              -> formatted after translation

English is the source language, so a missing translation falls back to the
English text; ``tests/test_i18n.py`` checks that every ``t("...")`` literal
used in the interface has a Ukrainian translation.
"""
from __future__ import annotations

import streamlit as st

from .i18n_uk import UK

LANGUAGES = {"en": "EN", "uk": "UA"}
DEFAULT_LANGUAGE = "en"


def current_language() -> str:
    lang = st.session_state.get("lang", DEFAULT_LANGUAGE)
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE


def translate(text: str, lang: str, **values) -> str:
    out = UK.get(text, text) if lang == "uk" else text
    return out.format(**values) if values else out


def t(text: str, **values) -> str:
    return translate(text, current_language(), **values)
