"""v6.2 browser checks: the logo navigates inside the app, every visible text reaches WCAG contrast in both
themes, chart labels are not cut at 1920×1080, 1366×768 and 125 %, the refund tooltip on History, the v6.2
colours and the note on explanations of predictions made before inputs were stored."""
import importlib.util

import pandas as pd
import pytest
from playwright.sync_api import expect

from src.ui.theme import PALETTES_V62
from tests.e2e.conftest import ROOT
from tests.e2e.harness import App

_spec = importlib.util.spec_from_file_location("ui_contrast_audit", ROOT / "scripts" / "ui_contrast_audit.py")
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)

PAGES = {"matches": "", "history": "history", "leagues": "leagues", "analytics": "analytics", "models": "models"}
SIZES = {"1920x1080": (1920, 1080, 1.0), "1366x768": (1366, 768, 1.0), "1920x1080@125": (1536, 864, 1.25)}

# every text drawn in a Plotly chart: inside the chart, and y labels not covered by values or other labels
CLIP_JS = r"""
() => {
  const out = [];
  for (const plot of document.querySelectorAll('.js-plotly-plot')) {
    const box = plot.getBoundingClientRect();
    if (box.width < 50 || box.height < 50) continue;
    const texts = Array.from(plot.querySelectorAll('svg text')).filter(t => {
      const r = t.getBoundingClientRect(); return t.textContent.trim() && r.width > 0 && r.height > 0; });
    const title = (plot.querySelector('.gtitle') || {}).textContent || '';
    for (const t of texts) {
      const r = t.getBoundingClientRect();
      if (r.left < box.left - 1 || r.right > box.right + 1 || r.top < box.top - 1 || r.bottom > box.bottom + 1)
        out.push({chart: title, text: t.textContent.trim(), problem: 'outside the chart'});
    }
    const ylabels = texts.filter(t => t.closest('.ytick'));
    const others = texts.filter(t => !t.closest('.ytick') && !t.closest('.xtick') && !t.closest('.gtitle'));
    for (const y of ylabels) {
      const a = y.getBoundingClientRect();
      for (const o of others) {
        const b = o.getBoundingClientRect();
        const w = Math.min(a.right, b.right) - Math.max(a.left, b.left), h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
        if (w > 2 && h > 2) out.push({chart: title, text: y.textContent.trim() + ' / ' + o.textContent.trim(), problem: 'overlap'});
      }
    }
  }
  return out;
}
"""


def _rgb(hex_colour: str) -> str:
    h = hex_colour.lstrip("#")
    return "rgb({}, {}, {})".format(*(int(h[i:i + 2], 16) for i in (0, 2, 4)))


# -- logo -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("start", ["history?theme=light", "leagues?theme=light", "?date=2026-11-08&league=E0&theme=light"])
def test_the_logo_opens_matches_inside_the_app(app, start):
    """A click anywhere on the logo opens Matches (today, all leagues) without reloading the page, so
    Streamlit's start-up screen does not flash; the light theme is kept."""
    app.open(start)
    app.page.evaluate("window.__fpNoReload = 1")
    box = app.page.locator(".fp-brand").bounding_box()
    for x in (box["x"] + 20, box["x"] + box["width"] - 25):               # the ball and the name
        app.page.mouse.click(x, box["y"] + box["height"] / 2)
        app.wait()
        expect(app.title).to_have_text("Matches")
        assert app.page.evaluate("window.__fpNoReload === 1"), "the page was reloaded"
        assert "date=" not in app.page.url and "league=" not in app.page.url
        assert "theme=light" in app.page.url
        assert app.background() == _rgb(PALETTES_V62["light"]["bg"])
        assert "not found" not in app.page.locator("body").inner_text().lower()


# -- contrast ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("name", list(PAGES))
def test_every_visible_text_is_readable(app, name, theme, predicted_day):
    """Normal text at least 4.5:1, large text 3:1 against what is really behind it (opacity included)."""
    day, _ = predicted_day
    path = PAGES[name] if name != "matches" else f"?date={day}"
    app.open(path + ("&" if "?" in path else "?") + f"theme={theme}")
    app.page.wait_for_timeout(1500)
    items = app.page.evaluate(_audit.AUDIT_JS)
    low = [i for i in items if i["ratio"] < i["need"] and not i["disabled"]]
    assert len(items) > 15
    assert not low, low[:8]


def test_captions_are_not_drawn_half_transparent(app):
    app.open("history")
    caption = app.page.locator('[data-testid="stCaptionContainer"]').first
    expect(caption).to_be_visible()
    assert float(caption.evaluate("e => getComputedStyle(e).opacity")) == 1.0


def test_v62_colours_are_used(app):
    app.open("")
    assert app.background() == _rgb(PALETTES_V62["dark"]["bg"])             # neutral, less graphite than v6.1
    app.open("?theme=light")
    assert app.background() == _rgb(PALETTES_V62["light"]["bg"])


def test_league_table_clubs_read_as_names(app):
    app.open("leagues")
    link = app.page.locator("table.mkt a").first
    expect(link).to_be_visible()
    assert link.evaluate("e => getComputedStyle(e).color") == _rgb(PALETTES_V62["dark"]["text-strong"])


# -- charts -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("name", ["analytics", "models"])
def test_chart_labels_are_not_cut(browser, base_url, name, size):
    """Long league and market names and the values next to bars stay inside each chart and do not overlap."""
    w, h, scale = SIZES[size]
    ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=scale)
    try:
        app = App(ctx.new_page(), base_url).open(PAGES[name])
        app.page.wait_for_selector(".js-plotly-plot", timeout=60_000)
        app.page.wait_for_timeout(2500)
        assert app.page.locator(".js-plotly-plot").count() >= 2
        problems = app.page.evaluate(CLIP_JS)
        assert not problems, problems[:8]
    finally:
        ctx.close()


def test_roi_by_league_shows_full_league_names(app):
    app.open("analytics")
    app.page.wait_for_selector(".js-plotly-plot .ytick text", timeout=60_000)
    app.page.wait_for_timeout(1500)
    labels = app.page.locator(".js-plotly-plot .ytick text").all_text_contents()
    long_names = [n for n in labels if len(n) >= 18]
    assert long_names, labels                                  # e.g. "Portugal Primeira Liga" is not shortened
    assert not any(n.endswith("…") or n.endswith("...") for n in labels)


# -- history ----------------------------------------------------------------------------------------------
def test_refunds_are_explained_next_to_predictions(app):
    app.open("history")
    kpi = app.kpi("Predictions")
    tip = kpi.locator(".tip")
    expect(tip).to_be_visible()
    assert "refunded prediction" in tip.get_attribute("title")
    assert "not counted in the hit rate or ROI" in tip.get_attribute("title")


# -- explanations -----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def old_finished_match():
    """A finished match predicted before the model inputs were stored (no row in inputs.parquet)."""
    picks = pd.read_parquet(ROOT / "local_data/predictions/picks.parquet", columns=["match_id", "role", "status",
                                                                                   "source", "date"])
    inputs = ROOT / "local_data/predictions/inputs.parquet"
    stored = set(pd.read_parquet(inputs, columns=["match_id"])["match_id"]) if inputs.exists() else set()
    done = picks.loc[(picks["role"] == "main") & (picks["source"] == "live") & picks["status"].isin(["WON", "LOST"])
                     & ~picks["match_id"].isin(stored)]
    return done.sort_values("date")["match_id"].iloc[-1]


def test_an_older_prediction_says_it_is_explained_with_current_data(app, old_finished_match):
    app.open(f"match?match_id={old_finished_match}")
    note = app.page.locator(".why .why-note").first
    expect(note).to_be_visible()
    expect(note).to_contain_text("were not stored")
