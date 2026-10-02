"""Global CSS of the web interface."""
from __future__ import annotations

import streamlit as st

COLORS = {
    "home": "#3b82f6",
    "draw": "#94a3b8",
    "away": "#f97316",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "muted": "#64748b",
    "card": "#131c2e",
    "border": "#1f2a40",
    "market": "#22c55e",
    "derived": "#38bdf8",
    "simulated": "#f59e0b",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
    margin=dict(l=10, r=10, t=36, b=10),
    hoverlabel=dict(font_family="Inter, sans-serif"),
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
  font-family: 'Inter', sans-serif !important;
}
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1480px; }
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { font-weight: 800 !important; }
a { text-decoration: none; }

.page-title { font-size: 1.75rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 .15rem 0; }
.page-subtitle { color: #94a3b8; font-size: .92rem; margin-bottom: 1.2rem; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 1.4rem 0 .6rem 0; color: #e2e8f0;
  display: flex; align-items: center; gap: .5rem; }
.section-title .hint { color: #64748b; font-weight: 500; font-size: .8rem; }
.muted { color: #64748b; }
.small { font-size: .8rem; }

/* cards ------------------------------------------------------------------ */
.card { background: #131c2e; border: 1px solid #1f2a40; border-radius: 14px; padding: 1rem 1.1rem; }
.card + .card { margin-top: .6rem; }
.kpi { background: #131c2e; border: 1px solid #1f2a40; border-radius: 14px; padding: .8rem 1rem; height: 100%; }
.kpi .label { color: #94a3b8; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.kpi .value { font-size: 1.45rem; font-weight: 800; margin-top: .15rem; }
.kpi .sub { color: #64748b; font-size: .78rem; }
.pos { color: #22c55e; } .neg { color: #ef4444; }

/* league group header in the match list ------------------------------------ */
.league-head { display: flex; align-items: center; gap: .55rem; margin: 1.1rem 0 .45rem 0;
  color: #cbd5e1; font-weight: 700; font-size: .9rem; }
.league-head .country { color: #64748b; font-weight: 500; }

/* match card ------------------------------------------------------------- */
a.match-card { display: grid; grid-template-columns: 64px minmax(180px, 1.5fr) minmax(220px, 1.6fr) 1fr 1fr;
  gap: 1rem; align-items: center; background: #131c2e; border: 1px solid #1f2a40; border-radius: 12px;
  padding: .7rem 1rem; margin-bottom: .45rem; color: #e2e8f0 !important; transition: all .12s ease; }
a.match-card:hover { border-color: #22c55e; background: #16213a; transform: translateY(-1px); }
.mc-time { text-align: center; font-weight: 700; font-size: .95rem; }
.mc-time .st { display: block; font-size: .68rem; font-weight: 600; color: #64748b; margin-top: 2px; }
.mc-time .st.ft { color: #22c55e; }
.mc-teams .team { display: flex; justify-content: space-between; font-weight: 600; font-size: .93rem; padding: 1px 0; }
.mc-teams .team b { font-weight: 800; min-width: 18px; text-align: right; }
.mc-teams .team.win { color: #fff; } .mc-teams .team.lose { color: #94a3b8; }
.mc-block .cap { color: #64748b; font-size: .68rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; margin-bottom: 4px; }
.mc-block .val { font-weight: 700; font-size: .95rem; }
.mc-block .odd { color: #94a3b8; font-size: .78rem; font-weight: 500; margin-left: .25rem; }
.mc-block .hit { color: #22c55e; } .mc-block .miss { color: #ef4444; }

/* probability bar --------------------------------------------------------- */
.pbar { display: flex; height: 8px; border-radius: 6px; overflow: hidden; background: #1f2a40; }
.pbar span { display: block; height: 100%; }
.plabels { display: flex; justify-content: space-between; font-size: .76rem; margin-top: 4px; color: #cbd5e1; }
.plabels b { font-weight: 700; }
.plabels .o { color: #64748b; font-size: .7rem; margin-left: 3px; }
.plabels .best { color: #fff; }

/* badges & chips ----------------------------------------------------------- */
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .68rem; font-weight: 700;
  letter-spacing: .03em; border: 1px solid currentColor; vertical-align: middle; }
.badge.market { color: #22c55e; } .badge.derived { color: #38bdf8; } .badge.simulated { color: #f59e0b; }
.badge.WON { color: #22c55e; } .badge.LOST { color: #ef4444; } .badge.UPCOMING { color: #94a3b8; } .badge.VOID { color: #64748b; }
.chip { display: inline-flex; width: 22px; height: 22px; border-radius: 6px; align-items: center; justify-content: center;
  font-size: .72rem; font-weight: 800; color: #0b1120; margin-right: 3px; }
.chip.W { background: #22c55e; } .chip.D { background: #94a3b8; } .chip.L { background: #ef4444; }

/* match header ------------------------------------------------------------ */
.match-header { background: linear-gradient(135deg, #131c2e 0%, #0f2a1f 100%); border: 1px solid #1f2a40;
  border-radius: 18px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.mh-meta { color: #94a3b8; font-size: .85rem; display: flex; gap: 1rem; flex-wrap: wrap; }
.mh-row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin-top: .8rem; }
.mh-team { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; }
.mh-team.away { text-align: right; }
.mh-team .sub { display: block; font-size: .8rem; font-weight: 500; color: #94a3b8; letter-spacing: 0; }
.mh-score { font-size: 2.3rem; font-weight: 800; padding: 0 1.5rem; text-align: center; }
.mh-score .vs { font-size: 1rem; color: #64748b; font-weight: 600; }

/* recommendation cards ------------------------------------------------------ */
.rec { border-radius: 16px; padding: 1.1rem 1.25rem; border: 1px solid #1f2a40; height: 100%; }
.rec.main { background: linear-gradient(135deg, rgba(34,197,94,.14), rgba(19,28,46,1) 65%); border-color: rgba(34,197,94,.45); }
.rec.risk { background: linear-gradient(135deg, rgba(245,158,11,.14), rgba(19,28,46,1) 65%); border-color: rgba(245,158,11,.45); }
.rec .tag { font-size: .7rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
.rec.main .tag { color: #22c55e; } .rec.risk .tag { color: #f59e0b; }
.rec .pick { font-size: 1.35rem; font-weight: 800; margin: .3rem 0 .1rem 0; }
.rec .mkt { color: #94a3b8; font-size: .82rem; }
.rec .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; margin-top: .8rem; }
.rec .grid .l { color: #64748b; font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.rec .grid .v { font-weight: 800; font-size: 1.05rem; }
.rec .note { color: #94a3b8; font-size: .75rem; margin-top: .6rem; }

/* market tables ----------------------------------------------------------- */
table.mkt { width: 100%; border-collapse: collapse; font-size: .84rem; }
table.mkt th { text-align: left; color: #64748b; font-weight: 600; font-size: .7rem; text-transform: uppercase;
  letter-spacing: .05em; padding: 4px 6px; border-bottom: 1px solid #1f2a40; }
table.mkt td { padding: 5px 6px; border-bottom: 1px solid #172036; }
table.mkt td.num, table.mkt th.num { white-space: nowrap; }
table.mkt tr:last-child td { border-bottom: none; }
table.mkt td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.mkt tr.top td { color: #fff; font-weight: 700; }
.minibar { height: 6px; border-radius: 4px; background: #1f2a40; overflow: hidden; min-width: 70px; }
.minibar span { display: block; height: 100%; background: #22c55e; }

/* team comparison --------------------------------------------------------- */
.cmp-row { display: grid; grid-template-columns: 70px 1fr 170px 1fr 70px; align-items: center; gap: .6rem; padding: 5px 0;
  border-bottom: 1px solid #172036; font-size: .86rem; }
.cmp-row:last-child { border-bottom: none; }
.cmp-row .lbl { text-align: center; color: #94a3b8; font-size: .78rem; }
.cmp-row .hv { text-align: right; font-weight: 700; } .cmp-row .av { font-weight: 700; }
.cmp-bar { height: 7px; background: #1f2a40; border-radius: 4px; overflow: hidden; display: flex; }
.cmp-bar.h { justify-content: flex-end; }
.cmp-bar span { display: block; height: 100%; }
.cmp-row .better { color: #fff; } .cmp-row .worse { color: #94a3b8; font-weight: 500; }

/* why this prediction ------------------------------------------------------ */
.why { background: #131c2e; border: 1px solid #1f2a40; border-left: 4px solid #22c55e; border-radius: 12px; padding: 1rem 1.2rem; }
.why p { margin: 0 0 .6rem 0; font-size: .95rem; line-height: 1.5; }
.factor { display: flex; gap: .6rem; align-items: baseline; padding: 3px 0; font-size: .88rem; }
.factor .arrow { font-weight: 800; width: 16px; }
.factor .arrow.up { color: #22c55e; } .factor .arrow.down { color: #ef4444; }
.factor .val { color: #64748b; font-size: .78rem; }

/* recent matches list --------------------------------------------------------- */
.rm { display: grid; grid-template-columns: 26px 70px 22px 1fr 54px; gap: .5rem; align-items: center; font-size: .84rem;
  padding: 4px 0; border-bottom: 1px solid #172036; }
.rm:last-child { border-bottom: none; }
.rm .d { color: #64748b; font-size: .76rem; } .rm .v { color: #64748b; font-size: .72rem; font-weight: 700; }
.rm .s { font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; }

/* sidebar-like league list ------------------------------------------------- */
.country-label { color: #64748b; font-size: .7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; margin: .7rem 0 .2rem .2rem; }
.st-key-league_nav { gap: .15rem; }
.st-key-league_nav button { justify-content: flex-start !important; min-height: 2rem; padding: .25rem .7rem; }
.st-key-league_nav button > div { justify-content: flex-start !important; width: 100%; }
.st-key-league_nav button p { font-size: .86rem; }
.stButton > button { border-radius: 10px; font-weight: 600; }
.empty { text-align: center; padding: 2.4rem 1rem; color: #64748b; border: 1px dashed #1f2a40; border-radius: 14px; }
.empty b { color: #cbd5e1; display: block; font-size: 1rem; margin-bottom: .3rem; }

/* metric explainer ------------------------------------------------------- */
.mx { display: grid; grid-template-columns: 34px 1fr; gap: .6rem; padding: .55rem 0; border-bottom: 1px solid #172036; }
.mx:last-child { border-bottom: none; }
.mx .ic { font-size: 1.3rem; }
.mx b { display: block; font-size: .9rem; } .mx span { color: #94a3b8; font-size: .84rem; }
"""


def inject_css() -> None:
    st.html(f"<style>{CSS}</style>")
