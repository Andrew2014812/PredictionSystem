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


def test_brand_and_navigation(app, predicted_day):
    day, _ = predicted_day
    app.open("history")
    expect(app.page.locator(".fp-brand")).to_contain_text("FootPredict")
    expect(app.page.locator(".fp-brand img.mark")).to_be_visible()                  # the FP mark, not an emoji
    assert "⚽" not in app.page.locator(".fp-brand").inner_text()
    labels = [x.strip() for x in app.nav_links().all_inner_texts()]
    for name in ("Matches", "Prediction History", "Leagues", "Teams", "Analytics", "Model Analysis"):
        assert any(name in label for label in labels), (name, labels)
    brand_right = app.page.locator(".fp-brand").bounding_box()
    first_link = app.nav_links().first.bounding_box()
    assert first_link["x"] > brand_right["x"] + brand_right["width"] - 5          # no overlap with the menu
    for start in ("history", f"?date={day}"):                                      # from any page
        app.open(start)
        app.page.locator(".fp-brand a").click()                     # the whole logo leads to Matches
        app.wait()
        expect(app.title).to_have_text("Matches")
        assert "not found" not in app.page.locator("body").inner_text().lower()   # no "Page not found"
        assert "/matches" not in app.page.url


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


def test_global_team_search_from_the_header(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    field = app.page.locator(".st-key-global_search")
    box = field.bounding_box()
    assert 110 <= box["width"] <= 160, box                                     # compact field, not an icon
    expect(field.locator("input")).to_have_attribute("placeholder", "Search...")
    field.locator("input").click()
    app.page.keyboard.type("Arse")                                              # results while typing, no Enter
    first = app.page.get_by_role("option").first
    expect(first).to_contain_text("Arsenal")                                    # the club starting with it first
    app.page.keyboard.press("Enter")
    app.wait()
    expect(app.page.locator(".mh-team")).to_contain_text("Arsenal")
    assert "teams" in app.page.url


def test_match_details_main_explanation_and_no_synthetic_odds(app, predicted_day):
    day, _ = predicted_day
    _open_first_match(app, day)
    main_card, risk_card = app.page.locator(".rec.main"), app.page.locator(".rec.risk")
    expect(main_card).to_be_visible()
    expect(risk_card).to_be_visible()                                 # Risk exists for every predicted match
    m_box, r_box = main_card.bounding_box(), risk_card.bounding_box()
    assert abs(m_box["y"] - r_box["y"]) < 5 and r_box["x"] > m_box["x"]            # side by side
    assert m_box["width"] > r_box["width"] * 1.3                                   # Main 60 %, Risk 40 %
    assert abs(m_box["height"] - r_box["height"]) < 2                              # the same height
    expect(app.section("Why this prediction?")).to_be_visible()
    card = app.page.locator(".why").first
    expect(card.locator(".why-t")).to_have_text(re.compile(r"^Why .+\?$"))
    expect(card.locator(".why-p")).to_contain_text("model probability")
    factors = card.locator(".fx")
    assert 3 <= factors.count() <= 5
    shares = [int(x.rstrip("%")) for x in card.locator(".fx-s").all_inner_texts()]
    assert shares and all(0 < x <= 100 for x in shares) and sum(shares) <= 100      # real shares of the reasoning
    assert all(e.strip() for e in card.locator(".fx-e").all_inner_texts())         # each factor has its numbers
    body = app.page.locator("[data-testid=stMain]").inner_text()
    for template in ("Strong support", "Moderate support", "Small support", "Works against", "The numbers",
                     "What the model weighed"):
        assert template not in body, template
    order = body.lower()
    assert order.index("why this risk prediction?") < order.index("explain another market") < order.index("markets")
    assert not SYNTHETIC.search(body), SYNTHETIC.search(body).group()
    assert "prod_" not in body and "backtest_" not in body           # no technical ids
    app.tab("Head-to-head").click()
    expect(app.page.get_by_role("tabpanel").filter(has_text=re.compile("Meetings|have not met"))).to_be_visible()
    app.tab("Recent form").click()
    expect(app.page.locator(".rm").first).to_be_visible()
    app.assert_healthy()


INFLUENCE_TIP = ("Only the most influential factors are shown. Percentages represent their share of total model "
                 "influence.")


def test_why_card_shows_relative_model_influence_and_the_chain(app, predicted_day):
    """v6.1 card: factors of the market's own probability with their relative influence, the recent form
    for this market and a short note on the expected goals / distribution — no repeated factor names."""
    day, _ = predicted_day
    _open_first_match(app, day)
    card = app.page.locator(".why").first
    header = card.locator(".fx-h")
    expect(header).to_contain_text("Main factors")
    expect(header).to_contain_text("Relative model influence")
    expect(header.locator(".tip")).to_have_attribute("title", INFLUENCE_TIP)
    expect(card.locator(".why-r > span")).to_have_text("Recent form")
    rows = card.locator(".fx")
    assert 3 <= rows.count() <= 5
    shares = [int(x.rstrip("%")) for x in card.locator(".fx-s").all_inner_texts()]
    assert all(0 <= x <= 100 for x in shares) and shares == sorted(shares, reverse=True)   # by influence
    text = card.inner_text()
    for old in ("Share of the model’s reasoning", "SHARE OF THE MODEL", "Recent record", "RECENT RECORD",
                "biggest reasons", "percentage points", " pp"):
        assert old not in text, old
    summary = card.locator(".why-sum").inner_text()
    names = [n.split("\n")[0].strip() for n in card.locator(".fx-n").all_inner_texts()]
    assert not any(n.lower() in summary.lower() for n in names), (names, summary)   # the note does not repeat them
    # another market: the Main / Risk picks are not offered again; an exact score shows no shares
    app.expander("Explain another market").click()
    main_label = app.page.locator(".rec.main").first.inner_text()
    box = app.page.locator('[data-testid="stSelectbox"]').filter(has_text="Market")
    box.locator("input").click()
    options = [o.strip() for o in app.page.get_by_role("option").all_inner_texts()]
    assert options and not any(o.split(": ", 1)[-1] in main_label for o in options), (options, main_label)
    exact = app.page.get_by_role("option").filter(has_text="Exact score")
    if exact.count():
        exact.first.click()
        app.wait()
        other = app.page.locator(".why").last
        expect(other.locator(".why-t")).to_contain_text("Exact score")
        expect(other.locator(".fx-s")).to_have_count(0)
        expect(other.locator(".why-sum")).to_contain_text("expected goals")
    app.assert_healthy()


def test_handicap_separates_half_and_whole_asian_lines(app, predicted_day):
    day, _ = predicted_day
    _open_first_match(app, day)
    assert app.page.get_by_text("Show all lines").count() == 0                 # only priced lines, no toggle
    card = app.page.locator(".card").filter(has_text="Handicap").first
    tables = card.locator("table")
    if tables.count() == 0:
        expect(card).to_contain_text("No bookmaker prices")
        return
    for table in tables.all():
        head = table.locator("tr").first.inner_text()
        assert re.search(r"Selection\s*Win probability\s*Odds", head, re.I) or re.search(
            r"Selection\s*Win\s*Refund\s*Lose\s*Odds", head, re.I), head          # headers are upper-cased by CSS
        odds = table.locator("td:last-child").all_inner_texts()
        assert odds and all(re.fullmatch(r"\d+\.\d\d", o) for o in odds), odds   # every listed line is priced
        lines = [re.search(r"([+-]\d+(\.5)?)$", r) for r in table.locator("td:first-child").all_inner_texts()]
        assert all(lines)                                              # every row names the side and its line
    assert "Draw" not in card.inner_text()                            # a refund is never called a draw


def test_history_shows_completed_predictions_by_default(app):
    app.open("history")
    expect(app.title).to_have_text("Prediction History")
    app.select("Period", "All time")
    statuses = set(app.page.locator("table.mkt .status").all_inner_texts())
    assert statuses and statuses <= {"Won", "Lost", "Refund"}, statuses
    scores = [s for s in app.page.locator("table.mkt").last.locator("tr td:nth-child(9)").all_inner_texts() if s]
    assert scores and all(re.fullmatch(r"\d+:\d+", s) for s in scores), scores[:3]      # 2:1, not 2-1
    expect(app.kpi("ROI")).to_be_visible()
    app.select("Prediction type", "Risk predictions")
    expect(app.page.locator("table.mkt").last).to_be_visible()
    app.assert_healthy()


def test_history_is_one_history_with_simple_filters(app):
    app.open("history")
    app.select("Period", "All time")
    body = app.page.locator("[data-testid=stMain]").inner_text()
    for gone in ("Recorded", "Backtest", "Both"):                     # one history, no data switch
        assert app.option(gone).count() == 0, gone
    labels = [x.strip() for x in app.page.locator('[data-testid="stWidgetLabel"]').all_inner_texts()]
    for label in ("Period", "Prediction type", "League", "Team or match"):
        assert label in labels, labels
    assert "Market" not in labels and "Data" not in labels             # market filter only under "More filters"
    assert app.page.locator(".kpi").filter(has_text="ROI").count() == 1   # one set of figures
    assert "predictions with odds" not in body
    assert app.page.get_by_role("radio", name="Yearly").count() == 0
    weeks = app.page.locator("table.mkt").first.locator("tr td:first-child").all_inner_texts()
    assert weeks and re.fullmatch(r"\d\d\.\d\d – \d\d\.\d\d\.\d{4}", weeks[0]), weeks[:2]   # Mon–Sun interval
    app.page.get_by_label("Team or match").fill("Arsenal")
    app.page.keyboard.press("Enter")
    app.wait()
    teams = app.page.locator("table.mkt").last.locator("td:nth-child(3)").all_inner_texts()
    assert teams and all("Arsenal" in x for x in teams), teams[:3]


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
    assert app.option("All time").count() == 0                         # season statistics only
    body = app.page.locator("[data-testid=stMain]").inner_text()
    assert "Recorded before kick-off" not in body and "Backtest" not in body   # one set of figures
    app.assert_healthy()


def test_leagues_show_kick_off_times_like_matches(browser, base_url):
    from tests.e2e.harness import App
    ctx = browser.new_context(timezone_id="Europe/Kyiv", viewport={"width": 1500, "height": 1100})
    app = App(ctx.new_page(), base_url)
    app.open("leagues")
    row = app.page.locator("a.rm").filter(has=app.page.locator(".s", has_text=re.compile(r"^\d\d:\d\d$"))).first
    expect(row).to_be_visible()
    href, league_time = row.get_attribute("href"), row.locator(".s").inner_text()
    app.page.goto(f"{base_url}/match?{href.split('?', 1)[1]}")
    app.wait()
    assert league_time in app.page.locator(".match-header .mh-meta").inner_text()   # the same local time
    ctx.close()


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
    expect(app.text("Highest ROI market").first).to_be_visible()
    assert "Not enough data" not in app.page.locator("[data-testid=stMain]").inner_text()
    expect(app.charts().first).to_be_visible()
    assert app.charts().count() == 6                                     # exactly six charts
    expect(app.page.locator(".gtitle").filter(has_text="Hit rate by market")).to_have_count(1)
    expect(app.text("ROI (%), last 50 bets").first).to_be_visible()
    app.option("Last 20").click()                                        # rolling window 20 / 50 / 100 / All
    app.wait()
    expect(app.text("ROI (%), last 20 bets").first).to_be_visible()
    app.option("All").click()
    app.wait()
    expect(app.text("ROI (%), cumulative").first).to_be_visible()
    assert app.charts().count() == 6
    boxes = [c.bounding_box() for c in app.charts().all()]
    assert abs(boxes[0]["y"] - boxes[1]["y"]) < 5 and boxes[1]["x"] > boxes[0]["x"]      # two per row
    league = app.page.locator(".stPlotlyChart, [data-testid=stPlotlyChart]").filter(
        has=app.page.locator(".gtitle", has_text="ROI by league")).first
    assert league.bounding_box()["width"] > boxes[0]["width"] * 1.6              # ROI by league: full width
    assert "n=" not in app.page.locator(".gtitle").filter(has_text="Hit rate by market").first.text_content()
    assert app.page.get_by_text("Window", exact=True).locator("visible=true").count() == 0   # no label above the buttons
    assert "volume" not in app.page.locator("[data-testid=stMain]").inner_text().lower()
    app.assert_healthy()


def test_model_analysis_draw_section(app):
    app.open("models")
    expect(app.title).to_have_text("Model Analysis")
    body = app.page.locator("[data-testid=stMain]")
    expect(app.text("Active model: trained on").first).to_be_visible()
    for gone in ("Active production model", "Target", "Decision rules", "Where do the stored predictions come from?",
                 "↓", "↑"):
        assert gone not in body.inner_text(), gone
    expect(app.kpi("Log loss").first).to_be_visible()
    expect(app.section("Draw predictions")).to_be_visible()
    expect(app.section("Confusion matrix")).to_be_visible()
    app.tab("Corners model").click()
    expect(app.kpi("Poisson deviance").last).to_be_visible()
    app.assert_healthy()


def test_interface_is_english_only(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}&lang=uk")                                  # an old Ukrainian link
    expect(app.title).to_have_text("Matches")
    assert app.pref("UA").count() == 0 and app.pref("EN").count() == 0
    expect(app.match_cards().first).to_contain_text("Draw")


def test_kick_off_times_follow_the_browser_time_zone(browser, base_url, predicted_day):
    from tests.e2e.harness import App
    day, _ = predicted_day
    times = {}
    for zone in ("Europe/London", "Asia/Tokyo"):
        ctx = browser.new_context(timezone_id=zone, viewport={"width": 1500, "height": 1100})
        app = App(ctx.new_page(), base_url)
        app.open(f"?date={day}")
        first = app.match_cards().first.get_attribute("href")
        times[zone] = (first, app.match_cards().first.locator(".mc-time").inner_text().split()[0])
        app.page.goto(f"{base_url}/match?match_id={first.split('match_id=')[1].split('&')[0]}")
        app.wait()
        times[zone] += (app.page.locator(".match-header .mh-meta").inner_text(),)
        ctx.close()
    london, tokyo = times["Europe/London"], times["Asia/Tokyo"]
    assert london[1] != tokyo[1]                                       # e.g. 15:00 in London, 23:00 in Tokyo
    assert london[2] != tokyo[2]


def test_cloud_toolbar_buttons_do_not_cover_the_theme_switch(app):
    app.open("")
    # Streamlit Cloud puts a "Fork" button for visitors into the header toolbar; locally it is absent,
    # so the same element is added here the way Cloud renders it
    app.page.evaluate("""() => {
        const bar = document.querySelector('[data-testid="stToolbar"]');
        const actions = document.createElement('div');
        actions.setAttribute('data-testid', 'stToolbarActions');
        actions.innerHTML = '<div data-testid="stToolbarActionButton"><button>Fork</button></div>';
        actions.style.cssText = 'position:fixed; top:20px; right:60px; z-index:999999;';
        bar.appendChild(actions);
    }""")
    expect(app.page.locator('[data-testid="stToolbarActions"]')).to_be_hidden()
    app.pref("light_mode").click()                                   # the switch can still be clicked
    app.wait()
    assert app.background() == "rgb(233, 238, 244)"


def test_theme_switch(app):
    app.open("")
    dark = app.background()
    app.pref("light_mode").click()
    app.wait()
    light = app.background()
    assert dark != light and light in ("rgb(233, 238, 244)",), light          # soft grey-blue, not white
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
    card = app.page.locator(".rec.main").first
    expect(card).to_be_visible()
    expect(card.locator(".status")).to_have_text(re.compile("^(Won|Lost|Refund)$"))
    expect(app.page.locator(".rec.risk .status")).to_have_text(re.compile("^(Won|Lost|Refund)$"))
    order = main.inner_text().lower()                                   # labels may be upper-cased by CSS
    assert order.index("main prediction") < order.index("why this prediction?") < order.index("markets")
    expect(app.page.locator(".why .why-sum").first).to_be_visible()     # one closing line
    assert not NAN.search(order), NAN.search(order).group()
    app.assert_healthy()


def test_history_is_one_simple_record_with_month_selection(app):
    app.open("history")
    body = app.page.locator("[data-testid=stMain]")
    for gone in ("Recorded predictions", "Historical evaluation", "Show upcoming", "Origin"):
        assert gone not in body.inner_text(), gone
    expect(app.section("Summary by period")).to_be_visible()                  # visible without a click
    app.select("Period", "Specific month")
    expect(app.page.locator('[data-testid="stSelectbox"]').filter(has_text="Month")).to_be_visible()
    expect(app.page.locator('[data-testid="stSelectbox"]').filter(has_text="Year")).to_be_visible()
    assert not NAN.search(body.inner_text())
    app.assert_healthy()


def test_analytics_highlights_follow_the_filters(app):
    app.open("analytics")
    assert app.option("Recorded").count() == 0                              # the same one history
    expect(app.section("Performance over time")).to_be_visible()
    highlight = app.page.locator(".kpi").filter(has_text="Highest ROI market").first
    assert "predictions with odds" not in highlight.inner_text()            # sample size in the tooltip
    assert "bets with real odds" in (highlight.get_attribute("title") or "")
    app.select("Prediction set", "Total goals")
    expect(app.text("Highest ROI league").first).to_be_visible()             # one market: leagues compared
    app.assert_healthy()


def test_team_statistics_are_grouped_and_all_matches_paginated(app):
    app.open("teams")
    app.select("Search team", "Arsenal — Premier League (England)")
    expect(app.page.locator(".mh-team img.badge")).to_be_visible()             # club badge (or neutral shield)
    heights = {round(b["height"]) for b in (k.bounding_box() for k in app.page.locator(".kpi").all()[:4])}
    assert len(heights) == 1, heights                                         # equal KPI cards
    for group in ("Results", "Attack", "Defence", "Corners"):
        expect(app.page.locator(".st-group").filter(has_text=group).first).to_be_visible()
    app.option("All available").click()
    app.wait()
    expect(app.text(re.compile(r"Page 1 of \d+")).first).to_be_visible()
    app.assert_healthy()


def test_model_analysis_features_and_historical_evaluation(app):
    app.open("models")
    expect(app.section("Model Features").first).to_be_visible()
    app.page.get_by_label("Search features").first.fill("rest")                # result-model tab is open
    app.page.keyboard.press("Enter")
    app.wait()
    expect(app.page.locator("b", has_text="Rest days").locator("visible=true").first).to_be_visible()
    body = app.page.locator("[data-testid=stMain]").inner_text()
    assert "h_rest_days" not in body and "Technical column names" not in body     # no raw column names
    assert "Elo" not in body
    app.page.get_by_label("Search features").first.fill("")
    app.page.keyboard.press("Enter")
    app.wait()
    expect(app.page.get_by_text(re.compile(r"Show all features \(\d+ groups\)")).first).to_be_visible()
    app.tab("Historical Model Testing").click()
    expect(app.page.get_by_role("tabpanel").locator("table.mkt").first).to_contain_text("Test season")
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


def test_light_theme_date_field_is_readable(app, predicted_day):
    day, _ = predicted_day
    app.open(f"?date={day}")
    app.pref("light_mode").click()
    app.wait()
    field = app.page.locator('[data-testid="stDateInput"] input').first
    color, bg = field.evaluate("e => [getComputedStyle(e).color, getComputedStyle(e.closest('[data-baseweb=input]')"
                               " || e).backgroundColor]")
    assert _contrast(color, bg) >= 4.5, (color, bg)
    app.pref("dark_mode").click()
    app.wait()


def test_browser_tab_title_and_favicon(app):
    app.open("history")
    expect(app.page).to_have_title("FootPredict")                    # not the page name, not "Streamlit"
    icons = app.page.locator("link[rel~='icon']").evaluate_all("els => els.map(e => e.href)")
    assert icons and not any("favicon.png" == i.rsplit("/", 1)[-1] and "streamlit" in i for i in icons), icons
    href = icons[0]
    resp = app.page.request.get(href)
    assert resp.ok and resp.headers.get("content-type", "").startswith("image/"), href
    assert "FP" not in app.page.locator(".fp-brand").inner_text().replace("FootPredict", "")   # no FP repeat
