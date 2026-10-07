"""FootPredict — web interface entry point.

    streamlit run app.py
"""
import logging

import streamlit as st

from src.ui.data import data_status
from src.ui.i18n import t
from src.ui.prefs import init_prefs, top_bar

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="FootPredict", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
init_prefs()

PAGES = [
    st.Page("views/matches.py", title=t("Matches"), icon=":material/sports_soccer:", url_path="matches",
            default=True),
    st.Page("views/match_details.py", title=t("Match Details"), icon=":material/insights:", url_path="match",
            visibility="hidden"),
    st.Page("views/history.py", title=t("Prediction History"), icon=":material/history:", url_path="history"),
    st.Page("views/leagues.py", title=t("Leagues"), icon=":material/emoji_events:", url_path="leagues"),
    st.Page("views/teams.py", title=t("Teams"), icon=":material/groups:", url_path="teams"),
    st.Page("views/analytics.py", title=t("Analytics"), icon=":material/monitoring:", url_path="analytics"),
    st.Page("views/model_analysis.py", title=t("Model Analysis"), icon=":material/model_training:",
            url_path="models"),
]

page = st.navigation(PAGES, position="top")
top_bar()
page.run()

status = data_status()
if status.get("updated_at"):
    stamp = status["updated_at"].replace("T", " ")[:16]
    st.html(f'<div class="note" data-testid="fp-data-status" style="text-align:center;margin-top:2rem">'
            f'{t("Data updated")}: <span class="fp-stamp">{stamp}</span></div>')
