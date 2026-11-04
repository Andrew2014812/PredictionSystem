"""Screenshots and texts of the "Why this prediction?" card for a few real matches.

    .venv/bin/python experiments/v6_1/explain/screenshots.py before   # on the v6 card
    .venv/bin/python experiments/v6_1/explain/screenshots.py after    # on the v6.1 card

Starts its own headless Streamlit server on a free port (log: cache/streamlit_<tag>.log), opens
each match page, saves the Main card, the opened Risk card and their texts
(figures/card_<tag>_<match>.png, results/cards_<tag>.json), then stops the server by its PID.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MATCHES = ["I1-20261011-como-roma", "E1-20261011-southampton-portsmouth", "SP1-20260916-barcelona-santander",
           "G1-20261011-aris-volos-nfc"]


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(page) -> None:
    page.wait_for_selector(".fp-brand", timeout=90_000)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="attached", timeout=1500)
    except Exception:
        pass
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=120_000)
    page.wait_for_timeout(600)


def main(tag: str) -> None:
    (HERE / "cache").mkdir(exist_ok=True)
    log = open(HERE / "cache" / f"streamlit_{tag}.log", "w")
    port = _port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.headless", "true"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    texts = {}
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if requests.get(f"{url}/_stcore/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            raise RuntimeError("streamlit did not start")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1300})
            for mid in MATCHES:
                page.goto(f"{url}/match?match_id={mid}")
                _wait(page)
                page.locator('[data-testid="stExpander"] summary').filter(has_text="Why this risk prediction?").click()
                page.wait_for_timeout(800)
                cards = page.locator(".why")
                out = []
                for i in range(min(cards.count(), 2)):
                    card = cards.nth(i)
                    card.evaluate("e => e.scrollIntoView({block: 'center'})")
                    page.wait_for_timeout(300)
                    card.screenshot(path=str(HERE / "figures" / f"card_{tag}_{mid}_{'main' if i == 0 else 'risk'}.png"))
                    out.append(card.inner_text())
                texts[mid] = out
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    (HERE / "results" / f"cards_{tag}.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(texts, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "after")
