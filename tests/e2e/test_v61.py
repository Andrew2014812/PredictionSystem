"""v6.1 browser checks: no visible Streamlit controls, the new design, an English calendar in any browser
language, profit in units, Main / Risk metrics on one line, charts that keep a 100 % label, the confusion
matrix of the rule in use and no horizontal overflow on laptop screens."""
import re

import pandas as pd
import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import ROOT
from tests.e2e.harness import App

PAGES = ["", "history", "leagues", "teams", "analytics", "models"]
STREAMLIT_CONTROLS = ['[data-testid="stMainMenu"]', '[data-testid="stAppDeployButton"]',
                      '[data-testid="stToolbarActions"]', '[data-testid="stStatusWidget"]',
                      '[data-testid="stDecoration"]', "footer"]


@pytest.fixture(scope="module")
def refund_match():
    """A finished match whose Risk ended as a refund: the widest card (six metrics)."""
    picks = pd.read_parquet(ROOT / "local_data/predictions/picks.parquet")
    risk = picks.loc[(picks["role"] == "risk") & (picks["status"] == "PUSH") & picks["odds"].notna()]
    if risk.empty:
        risk = picks.loc[(picks["role"] == "risk") & picks["status"].isin(["WON", "LOST"]) & picks["odds"].notna()]
    return risk.sort_values("date")["match_id"].iloc[-1]


def test_a_visitor_sees_no_streamlit_controls(app):
    app.open("")
    for selector in STREAMLIT_CONTROLS:
        for el in app.page.locator(selector).all():
            assert not el.is_visible(), selector
    # the only things in the header are FootPredict's: brand, menu, search, theme switch
    expect(app.page.locator(".fp-brand")).to_be_visible()
    expect(app.page.get_by_placeholder("Search...")).to_be_visible()


def test_cloud_badges_are_hidden_and_the_tab_says_footpredict(page, base_url):
    """Streamlit Community Cloud wraps the app in a page of the same address with its own corner badges;
    a local stand-in of that page checks that the app hides them and fixes the tab title."""
    wrapper = f"{base_url}/__cloud_wrapper"
    page.route(wrapper, lambda route: route.fulfill(content_type="text/html", body=(
        "<html><head><title>FootPredict · Streamlit</title></head><body style='margin:0'>"
        f"<iframe src='{base_url}/' style='width:1400px;height:900px;border:0'></iframe>"
        "<a class='_container_gzau3_1 _viewerBadge_1j65n_23' style='position:fixed;right:10px;bottom:10px'>"
        "Hosted with Streamlit</a><div class='_profileContainer_gzau3_53' "
        "style='position:fixed;right:80px;bottom:10px'>avatar</div></body></html>")))
    page.goto(wrapper)
    page.frame_locator("iframe").locator(".fp-brand").wait_for(timeout=90_000)
    page.wait_for_function("document.title === 'FootPredict'", timeout=30_000)
    expect(page.locator("[class*=viewerBadge]")).to_be_hidden()
    expect(page.locator("[class*=_profileContainer]")).to_be_hidden()


def test_the_v61_design_uses_its_fonts_and_colours(app):
    app.open("history")
    title_font = app.page.evaluate("getComputedStyle(document.querySelector('.page-title')).fontFamily")
    body_font = app.page.evaluate("getComputedStyle(document.querySelector('.kpi .label')).fontFamily")
    weight = app.page.evaluate("getComputedStyle(document.querySelector('.page-title')).fontWeight")
    assert "IBM Plex Sans" in title_font and "Source Sans 3" in body_font
    assert int(weight) <= 700                                                 # lighter than v6's 800
    assert app.background() == "rgb(16, 24, 32)"                              # graphite #101820, not navy


def test_the_calendar_is_english_in_a_russian_browser(browser, base_url):
    ctx = browser.new_context(locale="ru-RU", timezone_id="Europe/Kyiv", viewport={"width": 1366, "height": 900})
    try:
        app = App(ctx.new_page(), base_url).open("?date=2026-11-01")
        try:                                                     # give the page's own script time to run
            app.page.wait_for_function("navigator.language === 'en-GB'", timeout=10_000)
        except Exception:                                        # noqa: BLE001 - the text check below decides
            pass
        field = app.page.locator('[data-testid="stDateInput"]').first
        expect(field).to_contain_text("01")
        (field.locator("button").first if field.locator("button").count() else field).click()
        grid = app.page.locator('[role="grid"]').first
        expect(grid).to_be_visible()
        try:                                   # the picker switches language right after it is drawn
            app.page.wait_for_function("!/[А-Яа-яЁё]/.test(document.body.innerText)", timeout=5_000)
        except Exception:                                        # noqa: BLE001 - the assertion below reports it
            pass
        text = app.page.locator("body").inner_text()                       # the month header is in a popover
        assert not re.search(r"[А-Яа-яЁё]", text), text[-200:]              # nothing Russian anywhere
        heads = [h.strip() for h in grid.locator('[role="columnheader"], th').all_inner_texts() if h.strip()]
        assert heads and all(re.fullmatch(r"[A-Z][a-z]?", h) for h in heads), heads   # M T W T F S S
        assert not re.search(r"[А-Яа-яЁё]", grid.inner_text()), grid.inner_text()[:80]
    finally:
        ctx.close()


def test_profit_is_shown_in_units_without_a_u(app):
    for path in ("history", "analytics"):
        app.open(path)
        card = app.page.locator(".kpi").filter(has_text=re.compile("profit · units", re.I)).first
        expect(card).to_be_visible()
        value = card.locator(".value").inner_text().strip()
        assert re.fullmatch(r"[+−-]?\d+\.\d{2}|—", value), value                 # "−81.12", no " u"
        tip_text = card.locator("[title]").first.get_attribute("title") or ""
        assert "fixed stake of 1 unit" in tip_text


@pytest.mark.parametrize("viewport,scale", [((1366, 768), 1), ((1093, 614), 1.25)])
def test_main_and_risk_metrics_stay_on_one_line(browser, base_url, refund_match, viewport, scale):
    ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=scale)
    try:
        app = App(ctx.new_page(), base_url).open(f"match?match_id={refund_match}")
        cards = app.page.locator(".recs .rec")
        expect(cards).to_have_count(2)
        heights = [c.bounding_box()["height"] for c in cards.all()]
        assert abs(heights[0] - heights[1]) <= 1, heights                       # equal height
        for card in cards.all():
            values = [v.bounding_box() for v in card.locator(".grid > div > .v").all()]
            assert len(values) >= 4
            assert max(v["y"] for v in values) - min(v["y"] for v in values) <= 3   # one line
            for left, right in zip(values, values[1:]):
                assert left["x"] + left["width"] <= right["x"] + 1, (left, right)  # no overlap
    finally:
        ctx.close()


def test_bar_labels_are_not_cut_at_the_end_of_the_axis(app):
    app.open("analytics")
    app.page.wait_for_selector(".js-plotly-plot", timeout=60_000)
    for plot in app.page.locator(".js-plotly-plot").all():
        box = plot.bounding_box()
        for label in plot.locator("g.textpoint text").all():
            b = label.bounding_box()
            if b and box:
                assert b["x"] + b["width"] <= box["x"] + box["width"] + 1, label.text_content()


def test_the_confusion_matrix_is_the_rule_in_use(app):
    app.open("models")
    expect(app.section("Confusion matrix")).to_be_visible()
    text = app.page.locator(".js-plotly-plot").first.text_content()
    from src.analytics.outcome_rule import rule_metrics
    from src.prediction import store
    feats = pd.read_parquet(ROOT / "local_data/processed/features.parquet",
                            columns=["match_id", "played", "home_goals", "away_goals"])
    m = rule_metrics(store.load_snapshots(), feats)
    named_draws = m["confusion_matrix"][1][1]
    assert str(named_draws) in text                                       # draws named by the rule, not ~1
    accuracy = app.page.locator(".kpi").filter(has_text="Accuracy").first.locator(".value").inner_text()
    assert accuracy.strip() == f"{m['accuracy'] * 100:.1f}%"


@pytest.mark.parametrize("viewport,scale", [((1366, 768), 1), ((1093, 614), 1.25)])
def test_no_horizontal_overflow_on_laptops(browser, base_url, viewport, scale):
    ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=scale)
    try:
        app = App(ctx.new_page(), base_url)
        for path in PAGES:
            app.open(path)
            over = app.page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            assert over <= 1, (path, over)
            nav = app.page.locator("header").get_by_role("link").all()
            search = app.page.locator(".st-key-fp_prefs").bounding_box()
            right_edge = max(n.bounding_box()["x"] + n.bounding_box()["width"] for n in nav if n.is_visible())
            assert right_edge <= search["x"], (path, right_edge, search)        # the menu never runs under it
    finally:
        ctx.close()
