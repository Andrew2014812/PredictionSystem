"""Theme of the web interface: one stylesheet driven by CSS variables, in versioned designs.

A design = colour palettes (dark, light), two font families and the heavy / title font weights. The
stylesheet only uses ``var(--...)``, so a design is switched without touching the CSS:

* ``"v6.1"`` (default): graphite-green dark theme, warmer light theme, IBM Plex Sans for headings and
  navigation, Source Sans 3 for text and tables, lighter headings (600–700 instead of 800);
* ``"v6"``: the FootPredict v6 look (navy dark theme, Inter, weight 800), kept unchanged for a rollback.

The design is chosen by ``UI_DESIGN`` in src/config.py, or by the environment variable / Streamlit secret
``FP_DESIGN`` (no switch on the site). Rollback to v6: ``python scripts/use_design.py v6`` — it sets
``UI_DESIGN`` and restores the v6 widget colours in ``.streamlit/config.toml`` (see README).
In the v6.1 design the built-in Streamlit widgets are re-coloured from the palette in both themes; in the
v6 design only the light theme is, the dark one uses config.toml, exactly as in v6.
"""
from __future__ import annotations

import os

import streamlit as st

from .. import config as app_config

PALETTES_V6 = {
    "dark": {
        "bg": "#0b1120", "surface": "#131c2e", "surface-2": "#16213a", "border": "#1f2a40",
        "border-soft": "#172036", "text": "#e2e8f0", "text-strong": "#ffffff", "muted": "#94a3b8",
        "faint": "#7d8ba1", "accent": "#22c55e", "accent-soft": "rgba(34,197,94,.14)",
        "risk": "#f59e0b", "risk-soft": "rgba(245,158,11,.14)", "home": "#3b82f6", "draw": "#94a3b8",
        "away": "#f97316", "pos": "#22c55e", "neg": "#ef4444", "track": "#1f2a40",
        "bar-pos": "rgba(34,197,94,.78)", "bar-neg": "rgba(239,68,68,.62)", "bar-hit": "#5f9fc4",
        "search-bg": "rgba(148,163,184,.10)",
        "header-grad": "linear-gradient(135deg, #131c2e 0%, #0f2a1f 100%)", "input": "#131c2e",
    },
    "light": {
        "bg": "#e9eef4", "surface": "#ffffff", "surface-2": "#f1f4f8", "border": "#cfd8e3",
        "border-soft": "#e1e7ef", "text": "#1b2636", "text-strong": "#0f172a", "muted": "#3f4c5e",
        "faint": "#4d5b6e", "accent": "#16a34a", "accent-soft": "rgba(22,163,74,.10)",
        "risk": "#d97706", "risk-soft": "rgba(217,119,6,.10)", "home": "#2563eb", "draw": "#94a3b8",
        "away": "#ea580c", "pos": "#16a34a", "neg": "#dc2626", "track": "#e2e8f0",
        "bar-pos": "rgba(22,163,74,.78)", "bar-neg": "rgba(220,38,38,.60)", "bar-hit": "#3b7ea1",
        "search-bg": "rgba(15,23,42,.05)",
        "header-grad": "linear-gradient(135deg, #ffffff 0%, #ecfdf3 100%)", "input": "#ffffff",
    },
}

PALETTES_V61 = {
    "dark": {
        "bg": "#101820", "surface": "#1b292d", "surface-2": "#203135", "border": "#2b3c40",
        "border-soft": "#213236", "text": "#e1e7e5", "text-strong": "#f5f8f7", "muted": "#9fb1ab",
        "faint": "#889a95", "accent": "#269d7d", "accent-soft": "rgba(38,157,125,.16)",
        "risk": "#e2a336", "risk-soft": "rgba(226,163,54,.14)", "home": "#5b93d3", "draw": "#9aa6a8",
        "away": "#df7d45", "pos": "#33b88b", "neg": "#e2615c", "track": "#26373b",
        "bar-pos": "rgba(51,184,139,.74)", "bar-neg": "rgba(226,97,92,.62)", "bar-hit": "#6aa0c2",
        "search-bg": "rgba(159,177,171,.10)",
        "header-grad": "linear-gradient(135deg, #1b292d 0%, #16302a 100%)", "input": "#1b292d",
    },
    "light": {
        "bg": "#f5f6f3", "surface": "#ffffff", "surface-2": "#eef1ed", "border": "#d5dbd6",
        "border-soft": "#e5e9e4", "text": "#1f2a2c", "text-strong": "#111a1c", "muted": "#4a5855",
        "faint": "#56645f", "accent": "#167c66", "accent-soft": "rgba(22,124,102,.10)",
        "risk": "#b8761a", "risk-soft": "rgba(184,118,26,.10)", "home": "#2f6db5", "draw": "#8b979a",
        "away": "#c6602a", "pos": "#167c66", "neg": "#c0392b", "track": "#e2e7e2",
        "bar-pos": "rgba(22,124,102,.74)", "bar-neg": "rgba(192,57,43,.60)", "bar-hit": "#3b7ea1",
        "search-bg": "rgba(17,26,28,.05)",
        "header-grad": "linear-gradient(135deg, #ffffff 0%, #edf5f1 100%)", "input": "#ffffff",
    },
}

# v6.2: a more neutral (less blue-graphite) background, fewer and calmer greens, and secondary text that
# reaches WCAG 4.5:1 on every surface. ``accent`` is the readable green for text, ``accent-fill`` the darker
# green behind white text (selected buttons); one green cannot do both on a dark surface.
PALETTES_V62 = {
    "dark": {
        "bg": "#131618", "surface": "#1b1f22", "surface-2": "#22272a", "border": "#353c40",
        "border-soft": "#2a3033", "text": "#e6e9ea", "text-strong": "#fafbfb", "muted": "#b2babd",
        "faint": "#9ea6aa", "accent": "#45b08f", "accent-fill": "#1f7660", "accent-soft": "rgba(69,176,143,.11)",
        "risk": "#e8a941", "risk-soft": "rgba(232,169,65,.12)", "home": "#6ea3df", "draw": "#a2abae",
        "away": "#e98a56", "pos": "#47c196", "neg": "#f07a74", "track": "#2b3134",
        "bar-pos": "rgba(71,193,150,.72)", "bar-neg": "rgba(240,122,116,.66)", "bar-hit": "#79a9c8",
        "search-bg": "rgba(178,186,189,.09)",
        "header-grad": "linear-gradient(135deg, #1b1f22 0%, #1d2325 100%)", "input": "#1b1f22",
        "chip-text": "#0b1120",
    },
    "light": {
        "bg": "#f3f4f2", "surface": "#ffffff", "surface-2": "#ecedeb", "border": "#cfd3d0",
        "border-soft": "#e2e5e2", "text": "#1b2123", "text-strong": "#0e1315", "muted": "#444d50",
        "faint": "#525b5f", "accent": "#12705b", "accent-fill": "#12705b", "accent-soft": "rgba(18,112,91,.08)",
        "risk": "#94600f", "risk-soft": "rgba(148,96,15,.09)", "home": "#2a63a6", "draw": "#7a8689",
        "away": "#a9501d", "pos": "#12705b", "neg": "#b3342a", "track": "#e0e4e0",
        "bar-pos": "rgba(18,112,91,.72)", "bar-neg": "rgba(179,52,42,.62)", "bar-hit": "#356f91",
        "search-bg": "rgba(14,19,21,.05)",
        "header-grad": "linear-gradient(135deg, #ffffff 0%, #f1f3f1 100%)", "input": "#ffffff",
        "chip-text": "#ffffff",
    },
}

# v6.2 refinements on top of the shared CSS (only in the v6.2 design, so v6.1 stays exactly as it was).
V62_CSS = """
/* secondary text is opaque: Streamlit draws captions at 60 % opacity */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stCaption { opacity: 1 !important;
  color: var(--muted) !important; }
/* selected buttons: darker green behind white text; green text only where it is readable */
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-segmented_controlActive"],
[data-testid="stBaseButton-pills"][aria-checked="true"] {
  background: var(--accent-fill) !important; border-color: var(--accent-fill) !important; color: #ffffff !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--accent-soft) !important;
  border-color: var(--accent) !important; }
/* clubs in tables are names, not links in green: strong text, green only on hover */
table.mkt a, .tablewrap a, .lt a { color: var(--text-strong) !important; font-weight: 600; text-decoration: none;
  border-bottom: 1px solid var(--border); }
table.mkt a:hover, .tablewrap a:hover { color: var(--accent) !important; border-bottom-color: var(--accent); }
table.mkt { font-size: .86rem; }
table.mkt th { color: var(--muted); font-size: .7rem; letter-spacing: .06em; padding: 6px 6px 5px; }
table.mkt td { padding: 5px 6px; }
table.mkt tbody tr:hover td { background: var(--surface-2); }
/* editorial headings: a rule under the page title and section titles, fewer rounded corners */
.page-title { padding-bottom: .35rem; border-bottom: 2px solid var(--text-strong); display: inline-block; }
.section-title { padding-bottom: .3rem; border-bottom: 1px solid var(--border); }
.card, .kpi, .tablewrap, .why, .empty { border-radius: 8px; }
.rec, .outcome, .match-header { border-radius: 10px; }
.mc { border-radius: 8px; }
.kpi { border-left: 3px solid var(--border); }
.kpi .label { color: var(--muted); font-size: .74rem; }
.kpi .value { letter-spacing: -0.01em; }
.chip { color: var(--chip-text); }
.chip.D { background: var(--surface-2); color: var(--text-strong); box-shadow: inset 0 0 0 1px var(--border); }
"""

DESIGNS = {
    "v6": {"palettes": PALETTES_V6, "body": "'Inter', sans-serif", "head": "'Inter', sans-serif",
           "fonts": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
           "w_heavy": 800, "w_title": 800, "size": "1rem", "widgets_dark": False},
    "v6.1": {"palettes": PALETTES_V61, "body": "'Source Sans 3', 'Source Sans Pro', sans-serif",
             "head": "'IBM Plex Sans', sans-serif",
             "fonts": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700"
                      "&family=Source+Sans+3:wght@400;500;600;700&display=swap",
             "w_heavy": 700, "w_title": 600, "size": "1.04rem", "widgets_dark": True},
    "v6.2": {"palettes": PALETTES_V62, "body": "'Source Sans 3', 'Source Sans Pro', sans-serif",
             "head": "'IBM Plex Sans', sans-serif",
             "fonts": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700"
                      "&family=Source+Sans+3:wght@400;500;600;700&display=swap",
             "w_heavy": 700, "w_title": 600, "size": "1.04rem", "widgets_dark": True, "extra_css": V62_CSS},
}
DEFAULT_DESIGN = "v6.2"


def design_name() -> str:
    """The active design: FP_DESIGN (environment or Streamlit secret), else config.UI_DESIGN."""
    name = os.environ.get("FP_DESIGN")
    if not name:
        try:
            name = st.secrets.get("FP_DESIGN")
        except Exception:                                     # noqa: BLE001 - no secrets file locally
            name = None
    name = name or getattr(app_config, "UI_DESIGN", DEFAULT_DESIGN)
    return name if name in DESIGNS else DEFAULT_DESIGN


def design() -> dict:
    return DESIGNS[design_name()]


# the active design's palettes (kept as a name: tests and older code read PALETTES)
PALETTES = PALETTES_V62

def current_theme() -> str:
    return st.session_state.get("theme", "dark")


def colors() -> dict:
    """Palette of the active theme of the active design (for Plotly and inline styles)."""
    return design()["palettes"][current_theme()]


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a: str, b: str, share: float) -> str:
    """The colour ``share`` of the way from ``a`` to ``b`` (hex), as a heatmap draws it."""
    share = min(max(float(share), 0.0), 1.0)
    return "#" + "".join(f"{round(x + (y - x) * share):02x}" for x, y in zip(_rgb(a), _rgb(b)))


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two hex colours."""
    def lum(c):
        f = [v / 255 for v in _rgb(c)]
        f = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in f]
        return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]
    x, y = sorted((lum(a), lum(b)), reverse=True)
    return (x + 0.05) / (y + 0.05)


def readable_on(background: str) -> str:
    """White or near-black, whichever reads better on ``background``."""
    return max(("#ffffff", "#0e1315"), key=lambda fg: contrast(fg, background))


def plotly_layout() -> dict:
    c = colors()
    return dict(
        template="plotly_dark" if current_theme() == "dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=design()["body"].replace("'", ""), size=12, color=c["muted"]),
        title_font_color=c["text-strong"], legend_font_color=c["text"],
        margin=dict(l=10, r=16, t=48, b=10),
        hoverlabel=dict(font_family=design()["body"].replace("'", "")),
        # every axis makes room for its own labels (long league and market names)
        xaxis=dict(gridcolor=c["border-soft"], zerolinecolor=c["border"], automargin=True,
                   tickfont=dict(color=c["muted"])),
        yaxis=dict(gridcolor=c["border-soft"], zerolinecolor=c["border"], automargin=True,
                   tickfont=dict(color=c["muted"])),
    )


CSS = """

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
  font-family: var(--font-body) !important;
}
html { font-size: var(--font-size); }
h1, h2, h3, h4, .page-title, .fp-brand, [data-testid="stTopNavLink"] p, .mh-team, .why .why-t, .rec .pick,
.outcome .pick, .kpi .value, .section-title, [data-testid="stExpander"] summary p {
  font-family: var(--font-head) !important; }
.kpi .value, .rec .grid .v, .outcome .grid .v, .mc-odds, td, .rm .s { font-variant-numeric: tabular-nums; }
/* Streamlit's own controls are not part of FootPredict: menu, deploy / fork buttons, run indicator,
   decoration bar and footer are hidden (the run indicator stays in the page for the tests, only unseen) */
[data-testid="stElementContainer"]:has(#fp-lang),
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"], footer, #MainMenu,
[data-testid="manage-app-button"], [data-testid="stBottomBlockContainer"] .stCaption:empty {
  display: none !important; visibility: hidden !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--bg) !important; }
.block-container { padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1480px; }
.stApp, .stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p { color: var(--text); }
a { text-decoration: none; }

/* brand + top navigation ------------------------------------------------------ */
[data-testid="stHeader"] { border-bottom: 1px solid var(--border); }
[data-testid="stToolbar"] .rc-overflow { padding-left: 200px; padding-right: 285px; box-sizing: border-box; }
.fp-brand { position: fixed; top: 0; left: 0; height: 3.75rem; z-index: 999991; display: flex;
  align-items: center; padding-left: 1.4rem; pointer-events: none;
  font-weight: var(--w-heavy); font-size: 1.22rem; letter-spacing: -0.03em; color: var(--text-strong); }
.fp-brand a { display: flex; align-items: center; gap: .45rem; color: var(--text-strong) !important;
  pointer-events: auto; text-decoration: none; }
.fp-brand .mark { flex: none; display: block; }
.fp-brand a { pointer-events: none; }
/* the clickable logo: an in-app button laid over the brand (whole logo, no reload) */
.st-key-fp_home { position: fixed !important; top: 0; left: 0; width: 13.4rem !important; height: 3.75rem;
  z-index: 999993; margin: 0 !important; padding: 0 !important; gap: 0 !important; }
.st-key-fp_home [data-testid="stElementContainer"], .st-key-fp_home .stButton,
.st-key-fp_home button { display: block !important; width: 100% !important; height: 3.75rem !important; margin: 0 !important;
  padding: 0 !important; opacity: 0 !important; cursor: pointer; }
.st-key-fp_home:focus-within { outline: 2px solid var(--accent); outline-offset: -6px; border-radius: 8px; }
.fp-brand .accent { color: var(--accent); }
.st-key-fp_prefs { position: fixed; top: .55rem; right: 1.2rem; z-index: 999992; width: auto !important; }
.st-key-fp_prefs [data-testid="stHorizontalBlock"] { gap: .4rem; }
.st-key-fp_prefs [data-testid="stElementContainer"]:has(iframe), .st-key-fp_prefs .stIFrame { display: none !important; }
.st-key-fp_prefs, .st-key-fp_prefs [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: center; }
.st-key-fp_prefs button { min-height: 2rem !important; padding: .1rem .6rem !important; }
/* laptops at 125 % zoom: a slimmer top menu so it never runs under the search field */
@media (max-width: 1250px) {
  [data-testid="stTopNavLink"] { padding: .25rem .4rem !important; }
  [data-testid="stTopNavLink"] p, [data-testid="stTopNavLink"] span { font-size: .8rem !important; }
  [data-testid="stTopNavLink"] > span:first-child { display: none !important; }
  .fp-brand a > span { display: none; }             /* the ball alone; the name returns on wider screens */
  .st-key-fp_home { width: 4.6rem !important; }
  [data-testid="stToolbar"] .rc-overflow { padding-left: 70px; }
}
/* header search: a compact translucent field with a magnifier, a little wider while typing */
.st-key-global_search { width: 145px !important; transition: width .18s ease; }
.st-key-global_search:focus-within { width: 250px !important; }
.st-key-global_search [data-baseweb="select"] > div { min-height: 2rem !important; border-radius: 999px !important;
  background: var(--search-bg) !important; border: 1px solid var(--border) !important; padding-left: 1.7rem !important;
  background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.4' stroke-linecap='round'%3E%3Ccircle cx='10.5' cy='10.5' r='6.5'/%3E%3Cpath d='m20 20-4.6-4.6'/%3E%3C/svg%3E") !important;
  background-repeat: no-repeat !important; background-position: .6rem center !important; background-size: .95rem !important; }
.st-key-global_search [data-baseweb="select"] input, .st-key-global_search [data-baseweb="select"] div { font-size: .85rem; }
.st-key-global_search [data-baseweb="select"] svg { display: none; }

/* headings ------------------------------------------------------------------- */
.page-title { font-size: 1.7rem; font-weight: var(--w-title); letter-spacing: -0.03em; margin: 0 0 .9rem 0;
  color: var(--text-strong); }
.page-subtitle { color: var(--muted); font-size: .92rem; margin: -.6rem 0 1.1rem 0; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 1.5rem 0 .6rem 0; color: var(--text-strong);
  display: flex; align-items: center; gap: .5rem; }
.section-title .hint { color: var(--faint); font-weight: 500; font-size: .8rem; }
.muted { color: var(--faint); }
.small { font-size: .8rem; }
.note { color: var(--faint); font-size: .8rem; margin: .3rem 0 .8rem 0; }
.pos { color: var(--pos) !important; } .neg { color: var(--neg) !important; }

/* cards ------------------------------------------------------------------- */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem;
  color: var(--text); }
.card + .card { margin-top: .6rem; }
.card b, .card strong { color: var(--text-strong); }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: .8rem 1rem; height: 100%; }
.kpi .label { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
img.badge { object-fit: contain; vertical-align: middle; flex: none; }
.origin { font-size: .7rem; font-weight: 700; padding: 1px 6px; border-radius: 6px; border: 1px solid var(--border); color: var(--muted); }
.origin.live { color: var(--accent); border-color: var(--accent); }
.chip-note { display: inline-block; margin-left: .4rem; padding: 0 .4rem; border-radius: 6px; font-size: .62rem; font-weight: 700; letter-spacing: .02em; text-transform: none; color: var(--risk); border: 1px solid var(--risk); vertical-align: middle; }
.kpi .value { font-size: 1.45rem; font-weight: var(--w-heavy); margin-top: .15rem; color: var(--text-strong); }
.kpi .sub { color: var(--faint); font-size: .78rem; }
.kpi .sub:empty::before { content: " "; }   /* same height with or without a sub line */

/* match list --------------------------------------------------------------- */
.league-head { display: flex; align-items: center; gap: .55rem; margin: 1.1rem 0 .45rem 0;
  color: var(--text); font-weight: 700; font-size: .9rem; }
.league-head .country { color: var(--faint); font-weight: 500; }
a.match-card { display: grid; grid-template-columns: 56px minmax(135px, 1.3fr) minmax(230px, 2.2fr) minmax(92px, .9fr);
  gap: clamp(.6rem, 1.2vw, 1.2rem); align-items: center; background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: .75rem 1.1rem; margin-bottom: .45rem; color: var(--text) !important; transition: border-color .12s ease; }
a.match-card:hover { border-color: var(--accent); background: var(--surface-2); }
.mc-time { text-align: center; font-weight: 700; font-size: .95rem; color: var(--text-strong); }
.mc-time .st { display: block; font-size: .66rem; font-weight: 600; color: var(--faint); margin-top: 2px; }
.mc-time .st.ft { color: var(--accent); }
.mc-teams .team { display: flex; justify-content: space-between; font-weight: 600; font-size: .93rem; padding: 2px 0;
  color: var(--text); }
.mc-teams .team b { font-weight: var(--w-heavy); min-width: 18px; text-align: right; }
.mc-teams .team.win { color: var(--text-strong); } .mc-teams .team.lose { color: var(--faint); }
.mc-block .cap { color: var(--faint); font-size: .66rem; text-transform: uppercase; letter-spacing: .06em;
  font-weight: 600; margin-bottom: 4px; }
.mc-block .val { font-weight: 700; font-size: .95rem; color: var(--text-strong); }
.mc-block .odd { color: var(--text); font-size: .9rem; font-weight: 700; margin-left: .4rem; }
.hit { color: var(--pos); } .miss { color: var(--neg); }

/* Main (60 %) and Risk (40 %) side by side, equal height */
.recs { display: grid; grid-template-columns: 3fr 2fr; gap: 1rem; align-items: stretch; margin-bottom: .6rem; }
.recs > .rec { height: 100%; box-sizing: border-box; margin: 0 !important; }
/* the "Why …?" card */
.why .why-t { font-size: 1.05rem; font-weight: var(--w-heavy); color: var(--text-strong); margin-bottom: .15rem; }
.why .why-p { color: var(--muted); font-size: .9rem; margin-bottom: .7rem; }
.why .why-p b { color: var(--accent); font-size: 1.05rem; }
.why.risk .why-p b { color: var(--risk); }
.why .fx-h { display: flex; justify-content: space-between; color: var(--faint); font-size: .7rem;
  text-transform: uppercase; letter-spacing: .06em; font-weight: 700; margin: .2rem 0 .35rem; }
.why .fx { display: grid; grid-template-columns: 1fr 150px 46px; gap: .8rem; align-items: center;
  padding: .42rem 0; border-bottom: 1px solid var(--border-soft); }
.why .fx-n { color: var(--text); font-weight: 600; font-size: .9rem; }
.why .fx-e { display: block; color: var(--muted); font-weight: 500; font-size: .8rem; margin-top: 1px; }
.why .fx .bar { height: 7px; background: var(--track); border-radius: 4px; overflow: hidden; }
.why .fx .bar span { display: block; height: 100%; border-radius: 4px; }
.why .fx-s { text-align: right; font-weight: 700; color: var(--text-strong); font-size: .88rem; }
.why .against { margin-left: .45rem; font-size: .68rem; font-weight: 700; color: var(--neg); text-transform: uppercase; }
.why .why-r { margin-top: .7rem; } .why .why-r > span { color: var(--faint); font-size: .7rem; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 700; }
.why .why-r ul { margin: .25rem 0 0 1.1rem; padding: 0; color: var(--text); font-size: .86rem; }
.why .why-sum { margin: .7rem 0 0; color: var(--text); font-size: .9rem; }
.why .why-note { margin: .6rem 0 0; color: var(--muted); font-size: .82rem; font-style: italic; }

/* probability bar --------------------------------------------------------- */
.pbar { display: flex; gap: 2px; height: 7px; border-radius: 6px; overflow: hidden; background: transparent; }
.pbar span { display: block; height: 100%; }
.pbar span:first-child { border-radius: 6px 0 0 6px; } .pbar span:last-child { border-radius: 0 6px 6px 0; }
.p3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin-top: 6px; }
.p3 div { font-size: .76rem; color: var(--muted); white-space: nowrap; }
.p3 div:nth-child(2) { text-align: center; } .p3 div:nth-child(3) { text-align: right; }
.p3 b { color: var(--text-strong); font-size: .86rem; font-weight: 700; }
.p3 .best b { color: var(--accent); }
.p3 .o { display: block; color: var(--text); font-size: .88rem; font-weight: 700; margin-top: 1px; }   /* real odds: readable, still secondary */

/* chips --------------------------------------------------------------------- */
.chip { display: inline-flex; width: 22px; height: 22px; border-radius: 6px; align-items: center; justify-content: center;
  font-size: .72rem; font-weight: var(--w-heavy); color: #0b1120; margin-right: 3px; }
.chip.W { background: var(--pos); } .chip.D { background: var(--draw); } .chip.L { background: var(--neg); }
.status { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .68rem; font-weight: 700;
  border: 1px solid currentColor; }
.status.WON { color: var(--pos); } .status.LOST { color: var(--neg); }
.status.UPCOMING, .status.VOID { color: var(--faint); }

/* match header ------------------------------------------------------------ */
.match-header { background: var(--header-grad); border: 1px solid var(--border); border-radius: 18px;
  padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.mh-meta { color: var(--muted); font-size: .85rem; display: flex; gap: 1.1rem; flex-wrap: wrap; }
.mh-row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin-top: .8rem; }
.mh-team { font-size: 1.6rem; font-weight: var(--w-heavy); letter-spacing: -0.02em; color: var(--text-strong); }
.mh-team a { color: inherit !important; }
.mh-team.away { text-align: right; }
.mh-team img.badge { display: block; margin-bottom: .35rem; } .mh-team.away img.badge { margin-left: auto; }
.mc-teams .team .nm { display: flex; align-items: center; gap: .4rem; min-width: 0; }
.mh-team .sub { display: block; font-size: .8rem; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.mh-score { font-size: 2.3rem; font-weight: var(--w-heavy); padding: 0 1.5rem; text-align: center; color: var(--text-strong); }
.mh-score .vs { font-size: 1rem; color: var(--faint); font-weight: 600; }

/* recommendation cards ------------------------------------------------------ */
.rec { border-radius: 16px; padding: 1.1rem 1.25rem; border: 1px solid var(--border); height: 100%; background: var(--surface); }
.rec.main { background: linear-gradient(135deg, var(--accent-soft), var(--surface) 65%); border-color: var(--accent); }
.rec.risk { background: linear-gradient(135deg, var(--risk-soft), var(--surface) 65%); border-color: var(--risk); }
.rec .tag { font-size: .7rem; font-weight: var(--w-heavy); letter-spacing: .1em; text-transform: uppercase; }
.rec.main .tag { color: var(--accent); } .rec.risk .tag { color: var(--risk); }
.rec .pick { font-size: 1.35rem; font-weight: var(--w-heavy); margin: .3rem 0 .1rem 0; color: var(--text-strong); }
.rec .mkt { color: var(--muted); font-size: .82rem; }
.rec { container-type: inline-size; }
/* one row of metrics in both cards: one column per metric, values on one line; a narrow card (Risk at
   125 % zoom) shows short labels instead of wrapping */
.rec .grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .5rem; margin-top: .8rem;
  align-items: end; }
.rec .grid .l, .rec .grid .v { white-space: nowrap; }
.rec .grid .l .short { display: none; }
@container (max-width: 640px) {
  .rec .grid .l .full { display: none; }
  .rec .grid .l .short { display: inline; }
}
@container (max-width: 470px) {
  .rec div.grid { gap: .3rem; }
  .rec .grid div.l { letter-spacing: .02em; font-size: .64rem; }
  .rec .grid div.v { font-size: .92rem; }
  .rec .grid div.v .status { padding: 1px 5px; font-size: .64rem; }
}
.rec .grid .v.na { font-weight: 500; font-size: .85rem; color: var(--muted); }
.rec .grid .l { color: var(--faint); font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.rec .grid .v { font-weight: var(--w-heavy); font-size: 1.05rem; color: var(--text-strong); }
.tip { display: inline-flex; width: 15px; height: 15px; border-radius: 50%; border: 1px solid var(--faint);
  color: var(--faint); font-size: .62rem; align-items: center; justify-content: center; cursor: help;
  margin-left: 4px; font-weight: 700; text-transform: none; letter-spacing: 0; }

/* tables ------------------------------------------------------------------- */
table.mkt { width: 100%; border-collapse: collapse; font-size: .84rem; color: var(--text); }
table.mkt th { text-align: left; color: var(--faint); font-weight: 600; font-size: .68rem; text-transform: uppercase;
  letter-spacing: .05em; padding: 5px 6px; border-bottom: 1px solid var(--border); white-space: nowrap; }
table.mkt td { padding: 6px; border-bottom: 1px solid var(--border-soft); }
table.mkt tr:last-child td { border-bottom: none; }
table.mkt td.num, table.mkt th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.mkt tr.top td { color: var(--text-strong); font-weight: 700; }
table.mkt a { color: var(--accent) !important; font-weight: 600; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .3rem .7rem; }
.minibar { height: 6px; border-radius: 4px; background: var(--track); overflow: hidden; min-width: 60px; }
.minibar span { display: block; height: 100%; background: var(--accent); }

/* grouped team statistics --------------------------------------------------- */
.st-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem 1.4rem; }
.st-group .st-row { display: flex; justify-content: space-between; gap: .6rem; padding: 3px 0;
  border-bottom: 1px solid var(--border-soft); }
.st-group .st-row:last-child { border-bottom: none; }
.st-group b { color: var(--text-strong); }

/* finished-match outcome cards ------------------------------------------------- */
.outcome { border-radius: 16px; padding: 1rem 1.2rem; border: 1px solid var(--border); background: var(--surface); }
.outcome.WON { border-color: var(--pos); background: linear-gradient(135deg, var(--accent-soft), var(--surface) 70%); }
.outcome.LOST { border-color: var(--neg); background: linear-gradient(135deg, rgba(239,68,68,.12), var(--surface) 70%); }
.outcome .tag { font-size: .7rem; font-weight: var(--w-heavy); letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.outcome .pick { font-size: 1.2rem; font-weight: var(--w-heavy); margin: .25rem 0 .1rem; color: var(--text-strong); }
.outcome .mkt { color: var(--muted); font-size: .82rem; }
.outcome .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; margin-top: .7rem; }
.outcome .grid .l { color: var(--faint); font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.outcome .grid .v { font-weight: var(--w-heavy); font-size: 1.02rem; color: var(--text-strong); }
.outcome .grid .v.na { font-weight: 500; font-size: .82rem; color: var(--muted); }
.why-h { font-size: .74rem; font-weight: var(--w-heavy); letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  margin: .2rem 0 .4rem; }
ul.facts { margin: 0 0 .9rem 1.1rem; padding: 0; color: var(--text); font-size: .9rem; line-height: 1.55; }
.odd { color: var(--muted); font-size: .78rem; margin-left: .25rem; }

/* team comparison --------------------------------------------------------- */
.cmp-row { display: grid; grid-template-columns: 70px 1fr 170px 1fr 70px; align-items: center; gap: .6rem; padding: 5px 0;
  border-bottom: 1px solid var(--border-soft); font-size: .86rem; color: var(--text); }
.cmp-row:last-child { border-bottom: none; }
.cmp-row .lbl { text-align: center; color: var(--muted); font-size: .78rem; }
.cmp-row .hv { text-align: right; font-weight: 700; } .cmp-row .av { font-weight: 700; }
.cmp-bar { height: 7px; background: var(--track); border-radius: 4px; overflow: hidden; display: flex; }
.cmp-bar.h { justify-content: flex-end; }
.cmp-bar span { display: block; height: 100%; }
.cmp-row .better { color: var(--text-strong); } .cmp-row .worse { color: var(--faint); font-weight: 500; }

/* why this prediction ------------------------------------------------------ */
.why { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: 12px; padding: 1rem 1.2rem; color: var(--text); }
.why.risk { border-left-color: var(--risk); }
.why p { margin: 0 0 .7rem 0; font-size: .95rem; line-height: 1.5; color: var(--text-strong); }
.infl { display: grid; grid-template-columns: 1fr 160px 220px; gap: .8rem; align-items: center; padding: 4px 0;
  font-size: .87rem; }
.infl .bar { height: 8px; background: var(--track); border-radius: 4px; overflow: hidden; }
.infl .bar span { display: block; height: 100%; }
.infl .lvl { font-size: .76rem; color: var(--muted); }

/* lists ------------------------------------------------------------------- */
.rm { display: grid; grid-template-columns: 26px 74px 22px 1fr 54px; gap: .5rem; align-items: center; font-size: .84rem;
  padding: 4px 0; border-bottom: 1px solid var(--border-soft); color: var(--text) !important; }
.rm:last-child { border-bottom: none; }
.rm .d { color: var(--faint); font-size: .76rem; } .rm .v { color: var(--faint); font-size: .72rem; font-weight: 700; }
.rm .s { font-weight: var(--w-heavy); text-align: right; font-variant-numeric: tabular-nums; color: var(--text-strong); }
.country-label { color: var(--faint); font-size: .7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; margin: .8rem 0 .2rem .2rem; }
.st-key-league_nav { gap: .15rem; }
.st-key-league_nav button { justify-content: flex-start !important; min-height: 2rem; padding: .25rem .7rem; }
.st-key-league_nav button > div { justify-content: flex-start !important; width: 100%; }
.st-key-league_nav button p { font-size: .86rem; }
.stButton > button { border-radius: 10px; font-weight: 600; }
.empty { text-align: center; padding: 2.4rem 1rem; color: var(--faint); border: 1px dashed var(--border); border-radius: 14px; }
.empty b { color: var(--text); display: block; font-size: 1rem; margin-bottom: .3rem; }
.mx { display: grid; grid-template-columns: 34px 1fr; gap: .6rem; padding: .55rem 0; border-bottom: 1px solid var(--border-soft); }
.mx:last-child { border-bottom: none; }
.mx .ic { font-size: 1.3rem; }
.mx b { display: block; font-size: .9rem; color: var(--text-strong); } .mx span { color: var(--muted); font-size: .84rem; }
.search-hit a { color: var(--accent) !important; font-weight: 600; }
"""

# Built-in Streamlit widgets are dark (config.toml); re-colour them for the light theme.
LIGHT_WIDGETS = """
[data-testid="stHeader"], header { background: var(--bg) !important; }
[data-testid="stHeader"] a, [data-testid="stHeader"] span, [data-testid="stHeader"] p { color: var(--text) !important; }
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stText"] { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
input::placeholder, textarea::placeholder { color: var(--faint) !important; opacity: 1 !important; }
[role="listbox"], [role="listbox"] li, [role="option"] { background: var(--surface) !important; color: var(--text) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--surface-2) !important; }
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within { border-color: var(--accent) !important; }
button:disabled, [data-baseweb="select"][aria-disabled="true"] { opacity: .55 !important; }
[data-testid="stWidgetLabel"] p, label p { color: var(--text) !important; }
[data-testid="stTooltipHoverTarget"] svg, [data-testid="stTooltipIcon"] svg { color: var(--muted) !important; }
button[kind], [data-testid^="stBaseButton"] { background: var(--surface) !important; color: var(--text) !important;
  border-color: var(--border) !important; }
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-segmented_controlActive"] {
  background: var(--accent) !important; color: #ffffff !important; border-color: var(--accent) !important; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-segmented_controlActive"] p { color: #ffffff !important; }
[data-testid="stBaseButton-tertiary"] { background: transparent !important; border-color: transparent !important; }
[data-testid="stButtonGroup"] button { background: var(--surface) !important; color: var(--text) !important;
  border-color: var(--border) !important; }
[data-testid="stButtonGroup"] button p, [data-testid="stButtonGroup"] button span { color: var(--text) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--accent-soft) !important;
  border-color: var(--accent) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] p,
[data-testid="stButtonGroup"] button[aria-checked="true"] span { color: var(--accent) !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
input, textarea { background: var(--input) !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-baseweb="select"] div, [data-baseweb="select"] input { background-color: var(--input) !important;
  color: var(--text) !important; }
[data-baseweb="select"] > div { border-color: var(--border) !important; }
[data-testid="stNumberInput"] button, [data-testid="stNumberInputContainer"] { background: var(--surface) !important;
  color: var(--text) !important; border-color: var(--border) !important; }
[data-testid="stNumberInput"] button svg { fill: var(--text) !important; }
[data-testid="stDateInput"] div { background-color: var(--input) !important; }
[data-baseweb="tag"] { background: var(--surface-2) !important; color: var(--text) !important; }
[data-baseweb="tag"] span { color: var(--text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"], [data-baseweb="calendar"],
[data-baseweb="calendar"] div, [data-baseweb="popover"] > div { background: var(--surface) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [aria-selected="true"][role="option"] { background: var(--surface-2) !important; }
[data-baseweb="select"] svg, [data-baseweb="input"] svg { fill: var(--muted) !important; color: var(--muted) !important; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stDateInput"] [role="group"] { background-color: var(--input) !important; border-color: var(--border) !important; }
[data-testid="stDateInput"] [role="spinbutton"], [data-testid="stDateInput"] [role="group"] span { color: var(--text) !important; }
[data-testid="stDateInput"] [role="group"] span:not([role]) { color: var(--muted) !important; }
[data-testid="stDateInput"] [role="spinbutton"][aria-valuetext="Empty"] { color: var(--faint) !important; }
[data-testid="stDateInput"] [role="spinbutton"]:focus { background: var(--accent-soft) !important; color: var(--text-strong) !important; }
[data-testid="stDateInput"] button svg { fill: var(--muted) !important; color: var(--muted) !important; }
[aria-disabled="true"], [disabled] { color: var(--faint) !important; }
[data-testid="stSelectbox"] [role="group"]:focus-within,
[data-testid="stMultiSelect"] [role="group"]:focus-within { border-color: var(--accent) !important; }
[data-testid="stSelectbox"] [role="group"] svg, [data-testid="stMultiSelect"] [role="group"] svg {
  fill: var(--muted) !important; color: var(--muted) !important; }
[data-testid="stExpander"] details { background: var(--surface) !important; border-color: var(--border) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
  color: var(--text) !important; }
[data-baseweb="tab-list"] button p, [data-baseweb="tab"] p { color: var(--muted) !important; }
[data-baseweb="tab-list"] button[aria-selected="true"] p { color: var(--text-strong) !important; }
[data-baseweb="tab-border"] { background: var(--border) !important; }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"] { border-color: var(--border) !important; }
[data-testid="stSlider"] div, [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--text) !important; }
[data-testid="stAlertContainer"] { background: var(--surface-2) !important; color: var(--text) !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p { color: var(--text-strong) !important; }
[data-testid="stTooltipIcon"] svg { stroke: var(--faint) !important; }
[data-testid="stCheckbox"] label span, [data-testid="stToggle"] label span { color: var(--text) !important; }
[data-testid="stPageLink"] a, [data-testid="stPageLink"] p, [data-testid="stPageLink"] span { color: var(--text) !important; }
[data-testid="stSpinner"] { color: var(--text) !important; }
"""


def inject_css() -> None:
    d = design()
    palette = d["palettes"][current_theme()]
    variables = ":root{" + "".join(f"--{k}:{v};" for k, v in palette.items()) + \
        f"--font-body:{d['body']};--font-head:{d['head']};--w-heavy:{d['w_heavy']};--w-title:{d['w_title']};" \
        f"--font-size:{d['size']};}}"
    widgets = current_theme() == "light" or d["widgets_dark"]
    extra = LIGHT_WIDGETS if widgets else ""
    st.html(f"<style>@import url('{d['fonts']}');{variables}{CSS}{extra}{d.get('extra_css', '')}</style>")
