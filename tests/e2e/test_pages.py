"""Browser journeys over every screen of the app."""
import re

import pytest
from playwright.sync_api import expect

from src.leagues import get_league

pytestmark = pytest.mark.e2e


def test_matches_filter_by_league_and_open_details(app, predicted_day):
    day, league = predicted_day
    app.open(f"?date={day}")
    expect(app.title).to_have_text("Matches")
    expect(app.match_cards().first).to_be_visible(timeout=60_000)
    all_cards = app.match_cards().count()
    assert all_cards >= 5
    expect(app.match_cards().first).to_contain_text("%")             # probabilities shown

    app.button(re.compile(rf"^{re.escape(get_league(league).name)}")).first.click()
    app.wait()
    league_cards = app.match_cards().count()
    assert 0 < league_cards <= all_cards

    app.match_cards().first.click()
    app.wait()
    expect(app.title).to_have_text("Match Details")
    expect(app.text("Main prediction").first).to_be_visible()
    expect(app.text("Match result (1X2)")).to_be_visible()
    expect(app.text("Exact score").first).to_be_visible()
    app.assert_healthy()


def test_date_navigation_moves_one_day(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    app.button("›").click()
    app.wait()
    assert f"date={day.fromordinal(day.toordinal() + 1)}" in app.page.url
    app.assert_healthy()


def test_match_details_explanation_and_team_tabs(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    expect(app.match_cards().first).to_be_visible(timeout=60_000)
    app.match_cards().first.click()
    app.wait()
    expect(app.section("Why this prediction?")).to_be_visible()
    expect(app.page.locator(".why .factor").first).to_be_visible()
    app.expander("Detailed SHAP explanation").click()
    app.wait()
    expect(app.charts().first).to_be_visible()
    app.tab("Head-to-head").click()
    expect(app.page.get_by_role("tabpanel").filter(has_text=re.compile("Meetings|have not met"))).to_be_visible()
    app.tab("Recent matches").click()
    expect(app.page.locator(".rm").first).to_be_visible()
    app.assert_healthy()


def test_history_filters_main_predictions(app):
    app.open("history")
    expect(app.title).to_have_text("Prediction History")
    app.select("Prediction type", "Main prediction")
    grid = app.page.locator('[data-testid="stDataFrame"]').last
    expect(grid).to_be_visible()
    expect(app.kpi("ROI")).to_be_visible()
    app.assert_healthy()


def test_leagues_table_and_fixtures(app):
    app.open("leagues")
    expect(app.title).to_have_text("Leagues")
    expect(app.section("League table")).to_be_visible()
    expect(app.page.locator("table.mkt").first).to_contain_text("Pts")
    expect(app.section("Recent results")).to_be_visible()
    app.assert_healthy()


def test_team_search_shows_form(app):
    app.open("teams")
    app.select("Search team", "Arsenal — Premier League (England)")
    expect(app.page.locator(".mh-team")).to_have_text("Arsenal")
    expect(app.text("Form").first).to_be_visible()
    app.option("Last 15").click()
    app.wait()
    expect(app.page.locator(".rm").first).to_be_visible()
    app.assert_healthy()


def test_analytics_advanced_charts(app):
    app.open("analytics")
    expect(app.section("Overall performance")).to_be_visible()
    expect(app.section("League performance")).to_be_visible()
    app.expander("Advanced Analytics").click()
    app.wait()
    expect(app.charts().first).to_be_visible()
    assert app.charts().count() >= 4
    app.assert_healthy()


def test_model_analysis_metrics_and_confusion_matrix(app):
    app.open("models")
    expect(app.title).to_have_text("Model Analysis")
    expect(app.kpi("Log loss").first).to_be_visible()
    expect(app.section("Confusion matrix")).to_be_visible()
    app.tab("Corners model").click()
    expect(app.kpi("Poisson deviance").last).to_be_visible()
    app.expander("What do these metrics mean?").click()
    expect(app.text("Balanced accuracy").first).to_be_visible()
    app.assert_healthy()
