"""Run against a deployed app to prove it serves exactly the committed data:

    E2E_BASE_URL=https://football-prediction-diploma.streamlit.app/~/+ pytest tests/e2e --e2e -k sync
"""
import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]


def test_app_shows_the_data_version_of_this_checkout(app):
    status_file = ROOT / "local_data" / "data_status.json"
    if not status_file.exists():
        pytest.skip("no data_status.json in this checkout")
    expected = json.loads(status_file.read_text())["updated_at"].replace("T", " ")[:16]
    app.open("")
    stamp = app.page.locator(".fp-stamp")
    expect(stamp).to_be_visible()
    where = "deployed app" if os.environ.get("E2E_BASE_URL") else "local app"
    assert stamp.inner_text() == expected, f"{where} serves data from {stamp.inner_text()}, checkout has {expected}"
