"""Browser journeys over every screen of FootPredict."""
import re

import pytest
from playwright.sync_api import expect

from src.leagues import get_league

pytestmark = pytest.mark.e2e

SYNTHETIC = re.compile(r"\b(BOOK|DERIVED|SIM)\b|Fair odds|Estimated odds|Simulated odds|Not a value bet")


def _open_first_match(app, day):
    app.open(f"?date={day}")
    expect(app.match_cards().first).to_be_visible(timeout=60_000)
    app.match_cards().first.click()
    app.wait()
    expect(app.page.locator(".match-header")).to_be_visible()


def test_brand_and_navigation(app):
    app.open("")
    expect(app.page.locator(".fp-brand")).to_contain_text("FootPredict")
    labels = [x.strip() for x in app.nav_links().all_inner_texts()]
    for name in ("Matches", "Prediction History", "Leagues", "Teams", "Analytics", "Model Analysis"):
        assert any(name in label for label in labels), (name, labels)
    brand_right = app.page.locator(".fp-brand").bounding_box()
    first_link = app.nav_links().first.bounding_box()
    assert first_link["x"] > brand_right["x"] + 120                 # navigation starts after the brand


def test_matches_filter_by_league_and_open_details(app, predicted_day):
    day, league = predicted_day
    app.open(f"?date={day}")
    expect(app.title).to_have_text("Matches")
    expect(app.match_cards().first).to_be_visible(timeout=60_000)
    all_cards = app.match_cards().count()
    assert all_cards >= 5
    first = app.match_cards().first
    for label in ("Home", "Draw", "Away"):
        expect(first).to_contain_text(label)
    expect(app.button(re.compile(r"^All leagues$"))).to_be_visible()   # no confusing counter
    app.button(re.compile(rf"^{re.escape(get_league(league).name)}$")).first.click()
    app.wait()
    assert 0 < app.match_cards().count() <= all_cards
    app.match_cards().first.click()
    app.wait()
    expect(app.title).to_have_count(0)
    expect(app.page.locator(".rec.main, .outcome").first).to_be_visible()     # outcome card once finished
    expect(app.text("Match result").first).to_be_visible()
    app.assert_healthy()


def test_date_navigation_moves_one_day(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    app.button("›").click()
    app.wait()
    expect(app.page).to_have_url(re.compile(f"date={day.fromordinal(day.toordinal() + 1)}"))
    app.assert_healthy()


def test_team_search_from_matches_page(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    app.page.get_by_placeholder("Search team...").fill("Arsenal")
    app.page.keyboard.press("Enter")
    app.wait()
    link = app.page.locator(".search-hit a", has_text="Arsenal").first
    expect(link).to_be_visible()
    link.click()
    app.wait()
    expect(app.page.locator(".mh-team")).to_have_text("Arsenal")


def test_match_details_main_explanation_and_no_synthetic_odds(app, predicted_day):
    day, _ = predicted_day
    _open_first_match(app, day)
    expect(app.section("Why this prediction?")).to_be_visible()
    expect(app.page.locator(".why p").first).to_contain_text("FootPredict favours")
    expect(app.page.locator(".why .infl").first).to_be_visible()
    body = app.page.locator("[data-testid=stMain]").inner_text()
    assert not SYNTHETIC.search(body), SYNTHETIC.search(body).group()
    assert "prod_" not in body and "backtest_" not in body           # no technical ids
    app.tab("Head-to-head").click()
    expect(app.page.get_by_role("tabpanel").filter(has_text=re.compile("Meetings|have not met"))).to_be_visible()
    assert "Few meetings" not in app.page.locator("[data-testid=stMain]").inner_text()
    app.tab("Recent form").click()
    expect(app.page.locator(".rm").first).to_be_visible()
    app.assert_healthy()


def test_history_shows_completed_predictions_by_default(app):
    app.open("history")
    expect(app.title).to_have_text("Prediction History")
    app.select("Period", "All time")
    statuses = set(app.page.locator("table.mkt .status").all_inner_texts())
    assert statuses and statuses <= {"Won", "Lost"}, statuses
    expect(app.kpi("ROI")).to_be_visible()
    app.select("Prediction type", "Risk predictions")
    expect(app.page.locator("table.mkt").last).to_be_visible()
    app.assert_healthy()


def test_history_period_filter_changes_summary(app):
    app.open("history")
    app.select("Period", "All time")
    all_time = app.kpi("Predictions").first.locator(".value").inner_text()
    app.select("Period", "Last 7 days")
    week = app.kpi("Predictions").first.locator(".value").inner_text()
    assert int(week.replace(",", "")) < int(all_time.replace(",", ""))


def test_leagues_table_and_main_performance(app):
    app.open("leagues")
    expect(app.title).to_have_text("Leagues")
    expect(app.section("League table")).to_be_visible()
    expect(app.page.locator("table.mkt").first).to_contain_text("Pts")
    expect(app.kpi("Main ROI")).to_be_visible()
    app.assert_healthy()


def test_team_period_controls_the_match_list(app):
    app.open("teams")
    app.select("Search team", "Arsenal — Premier League (England)")
    expect(app.page.locator(".mh-team")).to_have_text("Arsenal")
    app.option("Last 5").click()
    app.wait()
    five = app.page.locator(".card .rm .chip.W, .card .rm .chip.D, .card .rm .chip.L").count()
    app.option("Last 15").click()
    app.wait()
    fifteen = app.page.locator(".card .rm .chip.W, .card .rm .chip.D, .card .rm .chip.L").count()
    assert five == 5 and fifteen == 15
    app.assert_healthy()


def test_analytics_defaults_to_main_predictions(app):
    app.open("analytics")
    expect(app.section("FootPredict performance")).to_be_visible()
    for label in ("Main predictions", "Main hit rate", "Main profit", "Main ROI"):
        expect(app.kpi(label)).to_be_visible()
    expect(app.text("Best market").first).to_be_visible()
    app.expander("Advanced analytics").click()
    app.wait()
    expect(app.charts().first).to_be_visible()
    assert app.charts().count() >= 4
    app.assert_healthy()


def test_model_analysis_draw_section(app):
    app.open("models")
    expect(app.title).to_have_text("Model Analysis")
    expect(app.text("Active production model")).to_be_visible()
    expect(app.kpi("Log loss ↓").first).to_be_visible()                 # says which way is better
    expect(app.section("Draw predictions")).to_be_visible()
    expect(app.page.locator("table.mkt").filter(has_text="draw-aware rule").first).to_be_visible()
    expect(app.section("Confusion matrix")).to_be_visible()
    app.tab("Corners model").click()
    expect(app.kpi("Poisson deviance").last).to_be_visible()
    app.assert_healthy()


def test_language_switch_translates_and_persists(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    app.pref("UA").click()
    app.wait()
    expect(app.title).to_have_text("Матчі")
    expect(app.nav_links().filter(has_text="Історія прогнозів")).to_have_count(1)
    expect(app.match_cards().first).to_contain_text("Нічия")
    app.match_cards().first.click()                                  # full page load keeps the language
    app.wait()
    expect(app.section("Чому такий прогноз?")).to_be_visible()
    app.pref("EN").click()
    app.wait()
    expect(app.section("Why this prediction?")).to_be_visible()


def test_theme_switch(app):
    app.open("")
    dark = app.background()
    app.pref("light_mode").click()
    app.wait()
    light = app.background()
    assert dark != light and light in ("rgb(245, 247, 251)",), light
    app.nav("Analytics")
    assert app.background() == light                                 # stays light across pages
    app.pref("dark_mode").click()
    app.wait()
    assert app.background() == dark


NAN = re.compile(r"\bnan\b|\+nan%|NaN")


def test_finished_match_shows_result_before_predictions(app, predicted_day):
    day, _ = predicted_day
    _open_first_match(app, day)
    main = app.page.locator("[data-testid=stMain]")
    outcome = app.page.locator(".outcome").first
    expect(outcome).to_be_visible()
    expect(outcome).to_contain_text(re.compile("Main prediction"))
    expect(outcome.locator(".status")).to_have_text(re.compile("^(Won|Lost)$"))
    order = main.inner_text().lower()                                   # labels may be upper-cased by CSS
    assert order.index("main prediction") < order.index("why this prediction?") < order.index("markets")
    expect(app.page.locator(".why .why-h").first).to_have_text("The numbers")   # facts before model factors
    expect(app.page.locator(".why .lvl").first).to_contain_text("%")
    assert not NAN.search(order), NAN.search(order).group()
    app.assert_healthy()


def test_history_separates_recorded_and_evaluation_predictions(app):
    app.open("history")
    body = app.page.locator("[data-testid=stMain]")
    expect(app.option("Recorded predictions")).to_be_visible()
    assert "Show upcoming" not in body.inner_text() and "Origin" not in body.inner_text()
    expect(app.section("Summary by period")).to_be_visible()                  # visible without a click
    app.select("Period", "All time")
    recorded = app.kpi("Predictions").first.locator(".value").inner_text()
    app.option("Historical evaluation").click()
    app.wait()
    evaluation = app.kpi("Predictions").first.locator(".value").inner_text()
    assert recorded != evaluation
    assert not NAN.search(body.inner_text())
    app.assert_healthy()


def test_analytics_highlights_follow_the_filters(app):
    app.open("analytics")
    expect(app.page.locator('[data-testid="stSelectbox"]').filter(has_text="Data")).to_be_visible()
    expect(app.section("Performance over time")).to_be_visible()
    app.select("Prediction set", "Total goals")
    expect(app.text("Best league").first).to_be_visible()                     # one market: leagues compared
    app.assert_healthy()


def test_team_statistics_are_grouped_and_all_matches_paginated(app):
    app.open("teams")
    app.select("Search team", "Arsenal — Premier League (England)")
    for group in ("Results", "Attack", "Defence", "Corners"):
        expect(app.page.locator(".st-group").filter(has_text=group).first).to_be_visible()
    app.option("All available").click()
    app.wait()
    expect(app.text(re.compile(r"Page 1 of \d+")).first).to_be_visible()
    app.assert_healthy()


def test_model_analysis_feature_dictionary_and_decision_rules(app):
    app.open("models")
    app.expander("Feature dictionary").first.click()
    app.page.get_by_label("Search features").first.fill("Elo")
    app.page.keyboard.press("Enter")
    app.wait()
    expect(app.text("Elo strength rating").first).to_be_visible()
    app.tab("Decision rules").click()
    expect(app.section("Main prediction rule")).to_be_visible()
    expect(app.section("Walk-forward check of the result model")).to_be_visible()
    app.assert_healthy()


def _contrast(rgb_a: str, rgb_b: str) -> float:
    def lum(rgb):
        vals = [int(x) / 255 for x in re.findall(r"\d+", rgb)[:3]]
        vals = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in vals]
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]
    hi, lo = sorted((lum(rgb_a), lum(rgb_b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_light_theme_text_is_readable(app):
    app.open("history")
    app.pref("light_mode").click()
    app.wait()
    bg = app.background()
    for selector in (".note", '[data-testid="stCaptionContainer"] p', '[data-testid="stWidgetLabel"] p', ".kpi .label"):
        colors = app.page.locator(selector).evaluate_all(
            "els => els.filter(e => e.getBoundingClientRect().width > 0).map(e => getComputedStyle(e).color)")
        assert colors, selector
        for color in set(colors):
            assert _contrast(color, bg) >= 4.5, (selector, color, bg)
    controls = app.page.locator('[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"]')
    for color in set(controls.evaluate_all("els => els.map(e => getComputedStyle(e).backgroundColor)")):
        assert _contrast(color, bg) < 1.3, ("dark select control in the light theme", color)
    app.pref("dark_mode").click()
    app.wait()
