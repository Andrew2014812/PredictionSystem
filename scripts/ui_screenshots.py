"""Screenshots of every page, both themes and several window sizes (the before / after of a UI change).

Starts a Streamlit server for the given checkout (default: this repository) on a free port, captures
the pages and stops the server by its PID. Used for reports/v6_1/screenshots.

    python scripts/ui_screenshots.py --out reports/v6_1/screenshots/after
    python scripts/ui_screenshots.py --root /path/to/v6/checkout --out .../before --sizes 1366x768
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

MATCH_FINISHED = "E0-20261025-man-united-brighton"     # overridden by --match
PAGES = [("matches", "?date={day}"), ("match", "match?match_id={match}"), ("history", "history"),
         ("leagues", "leagues"), ("teams", "teams"), ("analytics", "analytics"), ("models", "models")]
SIZES = {"1366x768": (1366, 768, 1.0), "1920x1080": (1920, 1080, 1.0), "1366x768@125": (1093, 614, 1.25)}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--day", default="2026-11-01")
    ap.add_argument("--match", default=MATCH_FINISHED)
    ap.add_argument("--pages", default=",".join(p for p, _ in PAGES))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log = open(out / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=args.root, stdout=log, stderr=subprocess.STDOUT, env={**os.environ})
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if requests.get(f"{base}/_stcore/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            sys.exit("server did not start (see server.log)")
        wanted = set(args.pages.split(","))
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for size in args.sizes.split(","):
                w, h, scale = SIZES[size]
                ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=scale)
                page = ctx.new_page()
                for theme in args.themes.split(","):
                    for name, path in PAGES:
                        if name not in wanted:
                            continue
                        url = f"{base}/{path.format(day=args.day, match=args.match)}"
                        url += ("&" if "?" in url else "?") + f"theme={theme}"
                        page.goto(url, timeout=120_000)
                        page.wait_for_selector(".fp-brand", timeout=120_000)
                        page.wait_for_function("document.querySelector('[data-testid=stStatusWidget]') === null",
                                               timeout=120_000)
                        page.wait_for_timeout(2500)
                        page.screenshot(path=str(out / f"{name}_{theme}_{size}.png"), full_page=False)
                        print("saved", name, theme, size, flush=True)
                ctx.close()
            browser.close()
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
        log.close()


if __name__ == "__main__":
    main()
