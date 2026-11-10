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
    """The football mark (a ball with motion lines) as a data URI: st.html strips inline <svg>, an <img>
    with an SVG data URI stays crisp at any size."""
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


# The browser tab is always called "FootPredict" (Streamlit would show the page name).
# The interface is English, but Streamlit's date picker follows the browser language (a Russian browser
# showed "ноябрь" and Russian weekdays). Streamlit hands the picker ``navigator.language`` each time the page
# draws, so this script — the first element of every page, run in the page itself (st.html) — reports
# British English (day-first, like the DD.MM.YYYY dates) before the date field arrives; the
# "languagechange" events also update a picker that was drawn earlier. Our own constant, no user input.
_LANG_JS = """<script id="fp-lang">
(function () {
  const app = window;
  try {
    if (app.navigator.language !== "en-GB") {
      Object.defineProperty(app.navigator, "language", {get: () => "en-GB", configurable: true});
      Object.defineProperty(app.navigator, "languages", {get: () => ["en-GB", "en"], configurable: true});
    }
    const ping = () => app.dispatchEvent(new Event("languagechange"));
    ping();
    if (!app.__fpLangWatch) {
      app.__fpLangWatch = true;
      new app.MutationObserver((records) => {
        for (const r of records) for (const node of r.addedNodes) {
          if (node.nodeType === 1 && (node.matches?.('[data-testid="stDateInput"], [role="dialog"], [role="grid"]')
              || node.querySelector?.('[data-testid="stDateInput"], [role="grid"]'))) {
            for (const ms of [40, 200, 600]) app.setTimeout(ping, ms);
            return;
          }
        }
      }).observe(app.document.body, {childList: true, subtree: true});
    }
  } catch (e) {}
})();
</script>"""


HIDE_CLOUD_BADGES = True     # Streamlit Community Cloud's own corner badges (see _TITLE_JS)

# The tab title is "FootPredict" in the app page and, on Streamlit Community Cloud, also in the hosting page
# around it (Cloud would otherwise show "FootPredict · Streamlit"). That page is served from the app's own
# address, so the app may style it: the "Hosted with Streamlit" badge and the creator's avatar in the lower
# right corner are hidden there. If Cloud ever serves that page from another origin, the browser blocks the
# access and nothing happens (no error); the badges then stay, which is the only visible effect.
_TITLE_JS = """<script>
const docs = [window.parent.document];
try { if (window.top !== window.parent && window.top.document) docs.push(window.top.document); } catch (e) {}
for (const doc of docs) {
  const fix = () => { if (doc.title !== "FootPredict") doc.title = "FootPredict"; };
  fix();
  new MutationObserver(fix).observe(doc.querySelector("title") || doc.head, {childList: true, subtree: true,
                                                                           characterData: true});
}
const outer = docs[docs.length - 1];          // ("top" is a reserved global name in browsers)
if (HIDE && docs.length > 1 && !outer.getElementById("fp-cloud-chrome")) {
  const style = outer.createElement("style");
  style.id = "fp-cloud-chrome";
  style.textContent = '[class*="viewerBadge"], [class*="_profileContainer"], [class*="profilePreview"] ' +
                      '{ display: none !important; }';
  outer.head.appendChild(style);
}
</script>"""


def search_label(team: str, league: str, alias: str | None) -> str:
    """"Arsenal · Premier League"; a longer known name goes last ("Man City · Premier League · Manchester City")
    so it is found but never outranks a club whose own name starts with what was typed."""
    return f"{team} · {league}" + (f" · {alias}" if alias else "")


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
    recent = teams.loc[teams["date"] >= teams["date"].max() - pd.Timedelta(days=550)].sort_values("team")
    return {f"{r.team}|{r.country}": search_label(r.team, league_name(r.league), full.get(r.team))
            for r in recent.itertuples()}


def _remember_team() -> None:
    choice = st.session_state.get("global_search")
    if choice:
        st.session_state["_go_team"] = choice
        st.session_state["global_search"] = None          # an empty field next time


def _team_search() -> None:
    """Search field in the top bar: a list of all current clubs (local data, no API call) that filters while
    typing — exact and starting matches first —, arrows + Enter or a click choose, Esc closes; opens the
    team page."""
    from . import data
    target = st.session_state.pop("_go_team", None)
    if target:
        team, country = target.split("|", 1)
        st.switch_page("views/teams.py", query_params={"team": team, "country": country, **pref_params()})
    options = _team_options(len(data.features()))
    st.selectbox(t("Find a club"), list(options), index=None, format_func=options.get, key="global_search",
                 placeholder=t("Search..."), label_visibility="collapsed", filter_mode="fuzzy",
                 on_change=_remember_team)
    components.html(_TITLE_JS.replace("HIDE", "true" if HIDE_CLOUD_BADGES else "false"), height=0)


def _go_home() -> None:
    st.session_state["_go_home"] = True


def top_bar() -> None:
    """Brand on the left, language and theme switches on the right."""
    if st.session_state.pop("_go_home", False):
        # the logo: Matches with its default view (today, all leagues), inside the app — no page reload,
        # theme and language kept; also from Matches itself after another day or league was chosen
        st.switch_page("views/matches.py", query_params=pref_params())
    st.html(_LANG_JS, unsafe_allow_javascript=True)           # first: before any date field is drawn
    inject_css()
    # The logo is drawn as before; a transparent button over it goes to Matches inside the app (no page
    # reload, so Streamlit's start-up screen is not shown again; the theme and language stay).
    st.html(f'<div class="fp-brand" aria-hidden="true"><a tabindex="-1">'
            f'<img class="mark" src="{logo_data_uri(st.session_state.get("theme", "dark"))}" width="34" height="29" '
            f'alt=""><span>Foot<span class="accent">Predict</span></span></a></div>')
    with st.container(key="fp_home"):
        st.button(f"FootPredict — {t('Matches')}", key="fp_home_btn", type="tertiary", on_click=_go_home)
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
