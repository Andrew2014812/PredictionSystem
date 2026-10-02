"""Page harness for the Streamlit app: navigation, waiting and lookups."""
from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

RUNNING = '[data-testid="stStatusWidget"]'


class App:
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url.rstrip("/")

    # -- navigation -----------------------------------------------------------
    def open(self, path: str = "") -> "App":
        self.page.goto(f"{self.base_url}/{path.lstrip('/')}")
        self.wait()
        return self

    def nav(self, title: str) -> "App":
        self.page.locator("header").get_by_role("link", name=title).first.click()
        self.wait()
        return self

    def wait(self) -> None:
        self.page.wait_for_selector(".page-title", timeout=90_000)
        self.page.wait_for_timeout(600)
        self.page.wait_for_selector(RUNNING, state="detached", timeout=120_000)
        self.page.wait_for_timeout(300)

    # -- lookups ------------------------------------------------------------------
    @property
    def title(self) -> Locator:
        return self.page.locator(".page-title")

    def text(self, text: str) -> Locator:
        return self.page.get_by_text(text, exact=False)

    def button(self, name: str) -> Locator:
        return self.page.get_by_role("button", name=name)

    def match_cards(self) -> Locator:
        return self.page.locator("a.match-card")

    def option(self, name: str) -> Locator:
        """An option of a segmented control / pills."""
        return self.page.get_by_role("radio", name=name, exact=True)

    def section(self, title: str) -> Locator:
        return self.page.locator(".section-title").filter(has_text=title)

    def kpi(self, label: str) -> Locator:
        return self.page.locator(".kpi").filter(has_text=label)

    def tab(self, name: str) -> Locator:
        return self.page.get_by_role("tab", name=name)

    def expander(self, name: str) -> Locator:
        return self.page.locator('[data-testid="stExpander"] summary').filter(has_text=name)

    def charts(self) -> Locator:
        return self.page.locator('[data-testid="stPlotlyChart"]')

    def select(self, label: str, option: str) -> None:
        box = self.page.locator('[data-testid="stSelectbox"]').filter(has_text=label)
        box.locator("input").click()
        box.locator("input").fill(option.split(" ")[0])
        self.page.get_by_role("option", name=option).first.click()
        self.wait()

    def assert_healthy(self) -> None:
        """No exception box, no error alert and no leaked Python object dump."""
        expect(self.page.locator('[data-testid="stException"]')).to_have_count(0)
        expect(self.page.locator('[data-testid="stAlertContentError"]')).to_have_count(0)
        expect(self.page.get_by_text("DeltaGenerator", exact=False)).to_have_count(0)
