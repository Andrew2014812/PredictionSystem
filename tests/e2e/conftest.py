import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import requests

from tests.e2e.harness import App

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url():
    external = os.environ.get("E2E_BASE_URL")
    if external:                       # e.g. a deployed app
        yield external.rstrip("/")
        return
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.headless", "true"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if requests.get(f"{url}/_stcore/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("streamlit did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def app(page, base_url) -> App:
    page.set_viewport_size({"width": 1500, "height": 1100})
    return App(page, base_url)


@pytest.fixture(scope="session")
def predicted_day():
    """A date with settled, predicted matches (discovered, not hard-coded)."""
    snaps = pd.read_parquet(ROOT / "local_data/predictions/snapshots.parquet")
    played = snaps.loc[snaps["source"] != "live"]
    counts = played.groupby("date").size()
    day = counts.loc[counts >= 5].index.max()
    league = played.loc[played["date"] == day, "league"].value_counts().index[0]
    return pd.Timestamp(day).date(), league
