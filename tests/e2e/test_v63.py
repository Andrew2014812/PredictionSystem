"""v6.3 browser checks: the blue-graphite dark theme without title rules, flags next to league names, the Main
result plaque on finished matches (same history as Prediction History, never on unplayed matches), readable factor
values and no technical note on the "Why …?" card, the Teams header / legends / two-decimal hover values, the
League header and the new order of the Analytics charts."""
import pandas as pd
import pytest
from playwright.sync_api import expect

from src.analytics.performance import unified_history
from src.ui.theme import PALETTES_V63
from tests.e2e.conftest import ROOT

DAY = "2026-11-08"


def _rgb(hex_colour: str) -> str:
    h = hex_colour.lstrip("#")
    return "rgb({}, {}, {})".format(*(int(h[i:i + 2], 16) for i in (0, 2, 4)))


@pytest.fixture(scope="module")
def main_results():
    picks = pd.read_parquet(ROOT / "local_data/predictions/picks.parquet")
    main = unified_history(picks).loc[lambda d: d["role"] == "main"]
    return dict(zip(main["match_id"], main["status"]))


# -- theme ------------------------------------------------------------------------------------------------
def test_dark_theme_is_blue_graphite_and_titles_have_no_rule(app):
    app.open("history")
    assert app.background() == _rgb(PALETTES_V63["dark"]["bg"])
    title = app.page.locator(".page-title").first
    assert title.evaluate("e => getComputedStyle(e).borderBottomStyle") == "none"
    section = app.page.locator(".section-title").first
    assert section.evaluate("e => getComputedStyle(e).borderBottomStyle") == "none"


# -- flags ------------------------------------------------------------------------------------------------
def test_league_names_on_matches_have_their_own_flags(app):
    app.open(f"?date={DAY}")
    heads = app.page.locator(".league-head")
    expect(heads.first).to_be_visible()
    for i in range(heads.count()):
        flag = heads.nth(i).locator("img.flag")
        expect(flag).to_have_count(1)
        assert flag.evaluate("e => e.complete && e.naturalWidth > 0")                  # the SVG really draws
    # England and Scotland have different flags, not the same British symbol
    nav = app.page.locator(".country-label")
    england = nav.filter(has_text="England").locator("img.flag").get_attribute("src")
    scotland = nav.filter(has_text="Scotland").locator("img.flag").get_attribute("src")
    assert england and scotland and england != scotland


# -- Main result on Matches -------------------------------------------------------------------------------
def test_finished_matches_show_the_main_result_of_the_history(app, main_results):
    app.open(f"?date={DAY}")
    cards = app.page.locator("a.match-card")
    expect(cards.first).to_be_visible()
    checked = 0
    for i in range(cards.count()):
        card = cards.nth(i)
        match_id = card.get_attribute("href").split("match_id=")[1].split("&")[0]
        status = main_results.get(match_id)
        mark = card.locator(".mres")
        if status in ("WON", "LOST", "PUSH"):
            expect(mark).to_have_count(1)
            assert status in mark.get_attribute("class")
            text = {"WON": "MAIN WON", "LOST": "MAIN LOST", "PUSH": "MAIN REFUND"}[status]
            assert mark.inner_text().strip() == text                       # what the visitor sees
            checked += 1
        else:
            expect(mark).to_have_count(0)
    assert checked >= 5


def test_unplayed_matches_have_no_main_result(app, predicted_day):
    app.open(f"?date={predicted_day}")
    expect(app.page.locator("a.match-card").first).to_be_visible()
    unplayed = app.page.locator("a.match-card").filter(has_not=app.page.locator(".mc-time .st.ft"))
    assert unplayed.count() > 0
    expect(unplayed.locator(".mres")).to_have_count(0)


# -- Why this prediction ----------------------------------------------------------------------------------
def test_factor_values_say_who_is_ahead_and_in_what_unit(app):
    app.open("match?match_id=E0-20261108-sunderland-chelsea")
    evidence = app.page.locator(".why .fx-e")
    expect(evidence.first).to_be_visible()
    import re
    texts = [x.strip() for x in evidence.all_inner_texts() if x.strip()]        # the shown (Main) factors
    assert texts
    # every value names a team or the match level, never a bare number like "-108" or "0.35"
    assert not any(re.fullmatch(r"[+-]?\d+(\.\d+)?%?", x) for x in texts), texts
    assert any(u in " ".join(texts) for u in ("Elo points", "/game", "/match", "expected result", "places higher")), texts
    expect(app.page.locator(".why .why-note")).to_have_count(0)


# -- Teams ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("size", [(1920, 1080), (1366, 768)])
def test_team_trend_legends_are_inside_their_charts(browser, base_url, size):
    from tests.e2e.harness import App
    ctx = browser.new_context(viewport={"width": size[0], "height": size[1]})
    app = App(ctx.new_page(), base_url)
    app.open("teams?team=Wolves&country=England")
    app.page.wait_for_selector(".js-plotly-plot .legendtext", timeout=60_000)
    app.page.wait_for_timeout(800)
    problems = app.page.evaluate("""() => {
      const out = [];
      for (const plot of document.querySelectorAll('.js-plotly-plot')) {
        const box = plot.getBoundingClientRect();
        const bg = plot.querySelector('.legend rect.bg');
        const lb = bg ? bg.getBoundingClientRect() : box;
        for (const t of plot.querySelectorAll('.legendtext')) {
          const r = t.getBoundingClientRect();
          // inside the chart, and inside the legend box (Plotly clips what lies outside it)
          if (r.top < box.top - 1 || r.bottom > box.bottom + 1 || r.left < box.left - 1 || r.right > box.right + 1
              || r.right > lb.right + 1)
            out.push(t.textContent);
        }
      }
      return out; }""")
    legends = app.page.locator(".js-plotly-plot .legendtext").all_text_contents()
    assert "Conceded" in legends and "Corners against" in legends
    assert problems == [], problems
    ctx.close()


def test_team_chart_hover_values_have_two_decimals(app):
    app.open("teams?team=Wolves&country=England")
    app.page.wait_for_selector(".js-plotly-plot", timeout=60_000)
    templates = app.page.evaluate(
        "() => Array.from(document.querySelectorAll('.js-plotly-plot')).flatMap(p => p.data.map(d => d.hovertemplate))")
    assert templates and all(t and "%{y:.2f}" in t for t in templates), templates


def test_team_header_shows_badge_league_and_position(app):
    app.open("teams?team=Wolves&country=England")
    head = app.page.locator(".tm-head")
    expect(head).to_be_visible()
    expect(head.locator("img.flag")).to_have_count(1)
    expect(head.locator(".nm")).to_have_text("Wolves")
    expect(head.locator(".tpos .v")).to_be_visible()


# -- Leagues ----------------------------------------------------------------------------------------------
def test_league_page_has_a_header_with_its_flag(app):
    app.open("leagues?league=SC0")
    head = app.page.locator(".lg-head")
    expect(head).to_be_visible()
    expect(head.locator("img.flag")).to_have_attribute("alt", "Scotland")
    expect(head.locator(".nm")).to_have_text("Premiership")
    expect(app.page.locator("table.lt img.badge").first).to_be_visible()


# -- Analytics --------------------------------------------------------------------------------------------
def test_hit_rate_and_roi_share_the_top_row(app):
    app.open("analytics")
    app.page.wait_for_selector(".js-plotly-plot .gtitle", timeout=60_000)
    titles = app.page.evaluate("""() => Array.from(document.querySelectorAll('.js-plotly-plot')).map(p => {
        const r = p.getBoundingClientRect(); const t = p.querySelector('.gtitle');
        return {t: t ? t.textContent : '', x: Math.round(r.left), y: Math.round(r.top)}; })""")
    by = {x["t"].split(",")[0].split(" (")[0]: x for x in titles}
    hit, roi, cum, mk = by["Hit rate"], by["ROI"], by["Cumulative profit"], by["ROI by market"]
    assert hit["y"] == roi["y"] and hit["x"] < roi["x"]                 # top row: hit rate, then ROI
    assert cum["y"] == mk["y"] > hit["y"] and cum["x"] < mk["x"]        # next row: cumulative profit, ROI by market
