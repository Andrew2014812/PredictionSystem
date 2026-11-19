"""Screenshots of the Main-result indicator variants on Matches (v6.3 design choice), both themes, two sizes.

    python scripts/ui_main_mark_variants.py --out reports/v6_3/design/main_mark

The page is the real Matches page; a variant differs only in CSS, injected into the page before the screenshot
(the site itself has one indicator, no switch). Writes one PNG per variant x theme x size and a contact sheet.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_screenshots import free_port  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLAIN = ".mres{background:none!important;border-color:transparent!important;padding:0!important}"
VARIANTS = {
    "A_text": PLAIN + ".mres .ic,.mres .tx-short{display:none!important}.mres .tx-long{display:inline!important}",
    "B_icon_label": PLAIN + ".mres .tx-long{display:none!important}.mres .tx-short{display:inline!important}",
    "C_plaque": ".mres .ic,.mres .tx-short{display:none!important}.mres .tx-long{display:inline!important}",
    "D_plaque_icon_short": ".mres .tx-long{display:none!important}.mres .tx-short{display:inline!important}",
    "E_plaque_icon_text": ".mres .tx-short{display:none!important}.mres .tx-long{display:inline!important}",
}
SIZES = {"1920x1080": (1920, 1080), "1366x768": (1366, 768)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/v6_3/design/main_mark"))
    ap.add_argument("--day", default="2026-11-08")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log = open(out / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ})
    base = f"http://127.0.0.1:{port}"
    shots = []
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if requests.get(f"{base}/_stcore/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            sys.exit("server did not start")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for size, (w, h) in SIZES.items():
                page = browser.new_page(viewport={"width": w, "height": h})
                for theme in ("dark", "light"):
                    page.goto(f"{base}/?date={args.day}&theme={theme}", timeout=120_000)
                    page.wait_for_selector(".mres", timeout=120_000)
                    page.wait_for_timeout(1500)
                    for name, css in VARIANTS.items():
                        handle = page.add_style_tag(content=css)
                        page.wait_for_timeout(250)
                        box = page.locator(".league-head").first.bounding_box()
                        cards = page.locator("a.match-card")
                        last = cards.nth(min(cards.count() - 1, 5)).bounding_box()
                        clip = {"x": box["x"] - 8, "y": box["y"] - 8, "width": last["x"] + last["width"] - box["x"] + 16,
                                "height": last["y"] + last["height"] - box["y"] + 16}
                        path = out / f"{name}_{theme}_{size}.png"
                        page.screenshot(path=str(path), clip=clip)
                        shots.append(path)
                        handle.evaluate("el => el.remove()")
                page.close()
            browser.close()
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
        log.close()
    print("\n".join(str(s.relative_to(ROOT)) for s in shots))


if __name__ == "__main__":
    main()
