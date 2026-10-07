"""Theme of the web interface: one stylesheet driven by CSS variables.

``PALETTES`` holds the dark (default) and light colour sets; the stylesheet
only uses ``var(--...)``. For the light theme a second block re-colours the
built-in Streamlit widgets, which are rendered dark by the app config.
"""
from __future__ import annotations

import streamlit as st

PALETTES = {
    "dark": {
        "bg": "#0b1120", "surface": "#131c2e", "surface-2": "#16213a", "border": "#1f2a40",
        "border-soft": "#172036", "text": "#e2e8f0", "text-strong": "#ffffff", "muted": "#94a3b8",
        "faint": "#64748b", "accent": "#22c55e", "accent-soft": "rgba(34,197,94,.14)",
        "risk": "#f59e0b", "risk-soft": "rgba(245,158,11,.14)", "home": "#3b82f6", "draw": "#94a3b8",
        "away": "#f97316", "pos": "#22c55e", "neg": "#ef4444", "track": "#1f2a40",
        "header-grad": "linear-gradient(135deg, #131c2e 0%, #0f2a1f 100%)", "input": "#131c2e",
    },
    "light": {
        "bg": "#f5f7fb", "surface": "#ffffff", "surface-2": "#eef2f8", "border": "#dfe5ef",
        "border-soft": "#e9edf4", "text": "#1e293b", "text-strong": "#0f172a", "muted": "#475569",
        "faint": "#64748b", "accent": "#16a34a", "accent-soft": "rgba(22,163,74,.10)",
        "risk": "#d97706", "risk-soft": "rgba(217,119,6,.10)", "home": "#2563eb", "draw": "#94a3b8",
        "away": "#ea580c", "pos": "#16a34a", "neg": "#dc2626", "track": "#e2e8f0",
        "header-grad": "linear-gradient(135deg, #ffffff 0%, #ecfdf3 100%)", "input": "#ffffff",
    },
}


def current_theme() -> str:
    return st.session_state.get("theme", "dark")


def colors() -> dict:
    """Palette of the active theme (for Plotly and inline styles)."""
    return PALETTES[current_theme()]


def plotly_layout() -> dict:
    c = colors()
    return dict(
        template="plotly_dark" if current_theme() == "dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=c["muted"]),
        margin=dict(l=10, r=10, t=36, b=10),
        hoverlabel=dict(font_family="Inter, sans-serif"),
        xaxis=dict(gridcolor=c["border-soft"], zerolinecolor=c["border"]),
        yaxis=dict(gridcolor=c["border-soft"], zerolinecolor=c["border"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
  font-family: 'Inter', sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--bg) !important; }
.block-container { padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1480px; }
.stApp, .stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p { color: var(--text); }
a { text-decoration: none; }

/* brand + top navigation ------------------------------------------------------ */
[data-testid="stHeader"] { border-bottom: 1px solid var(--border); }
[data-testid="stToolbar"] .rc-overflow { padding-left: 190px; padding-right: 190px; box-sizing: border-box; }
.fp-brand { position: fixed; top: 0; left: 0; height: 3.75rem; z-index: 999991; display: flex;
  align-items: center; gap: .45rem; padding-left: 1.4rem; pointer-events: none;
  font-weight: 800; font-size: 1.22rem; letter-spacing: -0.03em; color: var(--text-strong); }
.fp-brand .ball { font-size: 1.25rem; }
.fp-brand .accent { color: var(--accent); }
.st-key-fp_prefs { position: fixed; top: .55rem; right: 1.2rem; z-index: 999992; width: auto !important; }
.st-key-fp_prefs [data-testid="stHorizontalBlock"] { gap: .4rem; }
.st-key-fp_prefs button { min-height: 2rem !important; padding: .1rem .6rem !important; }

/* headings ------------------------------------------------------------------- */
.page-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 .9rem 0;
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
.kpi .value { font-size: 1.45rem; font-weight: 800; margin-top: .15rem; color: var(--text-strong); }
.kpi .sub { color: var(--faint); font-size: .78rem; }

/* match list --------------------------------------------------------------- */
.league-head { display: flex; align-items: center; gap: .55rem; margin: 1.1rem 0 .45rem 0;
  color: var(--text); font-weight: 700; font-size: .9rem; }
.league-head .country { color: var(--faint); font-weight: 500; }
a.match-card { display: grid; grid-template-columns: 62px minmax(170px, 1.3fr) minmax(300px, 2.2fr) minmax(120px, .9fr);
  gap: 1.2rem; align-items: center; background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: .75rem 1.1rem; margin-bottom: .45rem; color: var(--text) !important; transition: border-color .12s ease; }
a.match-card:hover { border-color: var(--accent); background: var(--surface-2); }
.mc-time { text-align: center; font-weight: 700; font-size: .95rem; color: var(--text-strong); }
.mc-time .st { display: block; font-size: .66rem; font-weight: 600; color: var(--faint); margin-top: 2px; }
.mc-time .st.ft { color: var(--accent); }
.mc-teams .team { display: flex; justify-content: space-between; font-weight: 600; font-size: .93rem; padding: 1px 0;
  color: var(--text); }
.mc-teams .team b { font-weight: 800; min-width: 18px; text-align: right; }
.mc-teams .team.win { color: var(--text-strong); } .mc-teams .team.lose { color: var(--faint); }
.mc-block .cap { color: var(--faint); font-size: .66rem; text-transform: uppercase; letter-spacing: .06em;
  font-weight: 600; margin-bottom: 4px; }
.mc-block .val { font-weight: 700; font-size: .95rem; color: var(--text-strong); }
.mc-block .odd { color: var(--muted); font-size: .8rem; font-weight: 500; margin-left: .35rem; }
.hit { color: var(--pos); } .miss { color: var(--neg); }

/* probability bar --------------------------------------------------------- */
.pbar { display: flex; height: 7px; border-radius: 6px; overflow: hidden; background: var(--track); }
.pbar span { display: block; height: 100%; }
.p3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin-top: 6px; }
.p3 div { font-size: .76rem; color: var(--muted); }
.p3 div:nth-child(2) { text-align: center; } .p3 div:nth-child(3) { text-align: right; }
.p3 b { color: var(--text-strong); font-size: .86rem; font-weight: 700; }
.p3 .best b { color: var(--accent); }
.p3 .o { display: block; color: var(--faint); font-size: .72rem; }

/* chips --------------------------------------------------------------------- */
.chip { display: inline-flex; width: 22px; height: 22px; border-radius: 6px; align-items: center; justify-content: center;
  font-size: .72rem; font-weight: 800; color: #0b1120; margin-right: 3px; }
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
.mh-team { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text-strong); }
.mh-team a { color: inherit !important; }
.mh-team.away { text-align: right; }
.mh-team .sub { display: block; font-size: .8rem; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.mh-score { font-size: 2.3rem; font-weight: 800; padding: 0 1.5rem; text-align: center; color: var(--text-strong); }
.mh-score .vs { font-size: 1rem; color: var(--faint); font-weight: 600; }

/* recommendation cards ------------------------------------------------------ */
.rec { border-radius: 16px; padding: 1.1rem 1.25rem; border: 1px solid var(--border); height: 100%; background: var(--surface); }
.rec.main { background: linear-gradient(135deg, var(--accent-soft), var(--surface) 65%); border-color: var(--accent); }
.rec.risk { background: linear-gradient(135deg, var(--risk-soft), var(--surface) 65%); border-color: var(--risk); }
.rec .tag { font-size: .7rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
.rec.main .tag { color: var(--accent); } .rec.risk .tag { color: var(--risk); }
.rec .pick { font-size: 1.35rem; font-weight: 800; margin: .3rem 0 .1rem 0; color: var(--text-strong); }
.rec .mkt { color: var(--muted); font-size: .82rem; }
.rec .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin-top: .8rem; }
.rec .grid .l { color: var(--faint); font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.rec .grid .v { font-weight: 800; font-size: 1.05rem; color: var(--text-strong); }
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
.infl { display: grid; grid-template-columns: 1fr 180px 150px; gap: .8rem; align-items: center; padding: 4px 0;
  font-size: .87rem; }
.infl .bar { height: 8px; background: var(--track); border-radius: 4px; overflow: hidden; }
.infl .bar span { display: block; height: 100%; }
.infl .lvl { font-size: .76rem; color: var(--muted); }

/* lists ------------------------------------------------------------------- */
.rm { display: grid; grid-template-columns: 26px 74px 22px 1fr 54px; gap: .5rem; align-items: center; font-size: .84rem;
  padding: 4px 0; border-bottom: 1px solid var(--border-soft); color: var(--text) !important; }
.rm:last-child { border-bottom: none; }
.rm .d { color: var(--faint); font-size: .76rem; } .rm .v { color: var(--faint); font-size: .72rem; font-weight: 700; }
.rm .s { font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; color: var(--text-strong); }
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
[data-testid="stCaptionContainer"] { color: var(--faint) !important; }
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
[data-testid="stExpander"] details { background: var(--surface) !important; border-color: var(--border) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
  color: var(--text) !important; }
[data-baseweb="tab-list"] button p, [data-baseweb="tab"] p { color: var(--muted) !important; }
[data-baseweb="tab-list"] button[aria-selected="true"] p { color: var(--text-strong) !important; }
[data-baseweb="tab-border"] { background: var(--border) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; }
[data-testid="stSlider"] div, [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--text) !important; }
[data-testid="stAlertContainer"] { background: var(--surface-2) !important; color: var(--text) !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p { color: var(--text-strong) !important; }
[data-testid="stTooltipIcon"] svg { stroke: var(--faint) !important; }
[data-testid="stCheckbox"] label span, [data-testid="stToggle"] label span { color: var(--text) !important; }
[data-testid="stPageLink"] a, [data-testid="stPageLink"] p, [data-testid="stPageLink"] span { color: var(--text) !important; }
[data-testid="stSpinner"] { color: var(--text) !important; }
"""


def inject_css() -> None:
    palette = PALETTES[current_theme()]
    variables = ":root{" + "".join(f"--{k}:{v};" for k, v in palette.items()) + "}"
    extra = LIGHT_WIDGETS if current_theme() == "light" else ""
    st.html(f"<style>{variables}{CSS}{extra}</style>")
