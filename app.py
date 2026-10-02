"""Web interface entry point.

    streamlit run app.py
"""
import logging

import streamlit as st

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="Football Prediction System", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")

PAGES = [
    st.Page("views/matches.py", title="Matches", icon=":material/sports_soccer:",
            url_path="matches", default=True),
    st.Page("views/match_details.py", title="Match Details", icon=":material/insights:",
            url_path="match", visibility="hidden"),
    st.Page("views/history.py", title="Prediction History", icon=":material/history:", url_path="history"),
    st.Page("views/leagues.py", title="Leagues", icon=":material/emoji_events:", url_path="leagues"),
    st.Page("views/teams.py", title="Teams", icon=":material/groups:", url_path="teams"),
    st.Page("views/analytics.py", title="Analytics", icon=":material/monitoring:", url_path="analytics"),
    st.Page("views/model_analysis.py", title="Model Analysis", icon=":material/model_training:",
            url_path="models"),
]

st.navigation(PAGES, position="top").run()
