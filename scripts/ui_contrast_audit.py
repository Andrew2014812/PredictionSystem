"""Measure the real contrast of every visible text on every page, both themes (the v6.2 contrast check).

The palette alone does not say how text looks: Streamlit widgets, opacity on a parent, semi-transparent
colours and Plotly labels change it. In a real browser this script takes each visible element that holds
text (HTML and SVG), composes its colour with every parent's opacity and the backgrounds behind it, and
computes the WCAG contrast ratio. Normal text needs 4.5:1, large text (≥ 24 px, or ≥ 18.7 px bold) 3:1;
disabled controls are listed apart.

    python scripts/ui_contrast_audit.py --out reports/v6_2/data/contrast_after.json
    python scripts/ui_contrast_audit.py --root /path/to/v6.1/checkout --out .../contrast_before.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_screenshots import PAGES, free_port  # noqa: E402

AUDIT_JS = r"""
() => {
  const parse = (c) => {
    const m = (c || '').match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number);
    return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
  };
  const over = (fg, bg) => [0, 1, 2].map(i => fg[i] * fg[3] + bg[i] * (1 - fg[3])).concat([1]);
  const lum = (c) => {
    const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const background = (el) => {
    const layers = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const s = getComputedStyle(n);
      const c = parse(s.backgroundColor);
      if (c && c[3] > 0) { layers.push(c); if (c[3] >= 1) break; }
    }
    let bg = [255, 255, 255, 1];
    const body = parse(getComputedStyle(document.body).backgroundColor);
    if (body && body[3] > 0) bg = over(body, bg);
    for (let i = layers.length - 1; i >= 0; i--) bg = over(layers[i], bg);
    return bg;
  };
  const opacity = (el) => {
    let o = 1;
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) o *= parseFloat(getComputedStyle(n).opacity || '1');
    return o;
  };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const s = getComputedStyle(n);
      if (s.display === 'none' || s.visibility === 'hidden') return false;
    }
    return true;
  };
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll('body *')) {
    if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME'].includes(el.tagName)) continue;
    // not judged here: the logo's transparent click target (its label is for screen readers) and a slider's
    // value, which sits ABOVE the thumb it belongs to (its DOM parent's colour is not behind it)
    if (el.closest('.st-key-fp_home, [data-testid="stSliderThumbValue"]')) continue;
    // a heatmap's cell is a raster image, not a parent: its counts are coloured per cell by readable_on()
    const plot = el.closest('.js-plotly-plot');
    if (plot && plot.querySelector('.heatmaplayer .hm') && el.closest('.annotation')) continue;
    const own = Array.from(el.childNodes).filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').trim();
    if (!own || !visible(el)) continue;
    const s = getComputedStyle(el);
    const svg = el instanceof SVGElement;
    const fgRaw = parse(svg ? (s.fill && s.fill !== 'none' ? s.fill : s.color) : s.color);
    if (!fgRaw) continue;
    const fillOpacity = svg ? parseFloat(s.fillOpacity || '1') : 1;
    const a = fgRaw[3] * opacity(el) * fillOpacity;
    if (a < 0.02) continue;                    // not drawn (e.g. a slider's end labels until hovered)
    const bg = background(svg ? el.ownerSVGElement || el : el);
    const fg = over([fgRaw[0], fgRaw[1], fgRaw[2], a], bg);
    const size = parseFloat(s.fontSize);
    const bold = parseInt(s.fontWeight || '400') >= 700;
    const large = size >= 24 || (bold && size >= 18.66);
    const need = large ? 3 : 4.5;
    const r = ratio(fg, bg);
    const disabled = !!el.closest('[disabled], [aria-disabled="true"]');
    const key = own.slice(0, 40) + '|' + (el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({text: own.slice(0, 60), tag: el.tagName.toLowerCase(), cls: String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className).slice(0, 60),
              size: Math.round(size * 10) / 10, ratio: Math.round(r * 100) / 100, need, disabled, svg,
              color: s.color, alpha: Math.round(a * 100) / 100});
  }
  return out;
}
"""


def audit_page(page) -> list[dict]:
    return page.evaluate(AUDIT_JS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", required=True)
    ap.add_argument("--day", default="2026-11-08")
    ap.add_argument("--match", default="E0-20261108-sunderland-chelsea")
    ap.add_argument("--size", default="1920x1080")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=args.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ})
    base = f"http://127.0.0.1:{port}"
    report = {"root": args.root, "size": args.size, "pages": {}}
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
            page = browser.new_page(viewport={"width": w, "height": h})
            for theme in ("dark", "light"):
                for name, path in PAGES:
                    url = f"{base}/{path.format(day=args.day, match=args.match)}"
                    url += ("&" if "?" in url else "?") + f"theme={theme}"
                    page.goto(url, timeout=120_000)
                    page.wait_for_selector(".fp-brand", timeout=120_000)
                    page.wait_for_function(
                        """() => { const n = document.querySelectorAll('[data-testid=stMain] *').length;
                                   const now = performance.now();
                                   if (window.__n !== n) { window.__n = n; window.__t = now; }
                                   return n > 40 && now - window.__t > 1500; }""", polling=200, timeout=180_000)
                    page.wait_for_timeout(1500)
                    items = audit_page(page)
                    low = [i for i in items if i["ratio"] < i["need"] and not i["disabled"]]
                    report["pages"][f"{name}_{theme}"] = {"texts": len(items), "below": low,
                                                         "min_ratio": min((i["ratio"] for i in items), default=None)}
                    print(f"{name:10s} {theme:5s} texts {len(items):4d}  below threshold {len(low):3d}", flush=True)
            browser.close()
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
