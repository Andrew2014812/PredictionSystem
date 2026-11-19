"""Small national flags next to league names, drawn as inline SVG.

Inline SVG looks the same on every system (emoji flags are missing on Windows, and the England and Scotland
flags are not emoji at all on most systems). England and Scotland get their own flags (St George's cross and
the Saltire), not the Union Flag. The flags are simplified to stay crisp at 18 × 12 px (no coats of arms).
"""
from __future__ import annotations

import html
from urllib.parse import quote

W, H = 30, 20


def _star(cx: float, cy: float, r: float, rot: float = -90.0) -> str:
    import math
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.382
        a = math.radians(rot + i * 36)
        pts.append(f"{cx + rr * math.cos(a):.2f},{cy + rr * math.sin(a):.2f}")
    return " ".join(pts)


def _h(*colours: str) -> str:
    """Horizontal stripes."""
    n = len(colours)
    return "".join(f'<rect y="{H * i / n:.3f}" width="{W}" height="{H / n + 0.05:.3f}" fill="{c}"/>'
                   for i, c in enumerate(colours))


def _v(*colours: str) -> str:
    """Vertical stripes."""
    n = len(colours)
    return "".join(f'<rect x="{W * i / n:.3f}" width="{W / n + 0.05:.3f}" height="{H}" fill="{c}"/>'
                   for i, c in enumerate(colours))


def _greece() -> str:
    blue, white = "#0d5eaf", "#ffffff"
    stripes = "".join(f'<rect y="{H * i / 9:.3f}" width="{W}" height="{H / 9 + 0.05:.3f}" '
                      f'fill="{blue if i % 2 == 0 else white}"/>' for i in range(9))
    s = H * 5 / 9
    cross = (f'<rect width="{s:.3f}" height="{s:.3f}" fill="{blue}"/>'
             f'<rect x="{s * 2 / 5:.3f}" width="{s / 5:.3f}" height="{s:.3f}" fill="{white}"/>'
             f'<rect y="{s * 2 / 5:.3f}" width="{s:.3f}" height="{s / 5:.3f}" fill="{white}"/>')
    return stripes + cross


FLAGS = {
    "England": (f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
                f'<rect x="{W / 2 - 2:.1f}" width="4" height="{H}" fill="#ce1124"/>'
                f'<rect y="{H / 2 - 2:.1f}" width="{W}" height="4" fill="#ce1124"/>'),
    "Scotland": (f'<rect width="{W}" height="{H}" fill="#005eb8"/>'
                 f'<path d="M0,0 L{W},{H} M{W},0 L0,{H}" stroke="#ffffff" stroke-width="3.6"/>'),
    "Spain": (f'<rect width="{W}" height="{H}" fill="#aa151b"/>'
              f'<rect y="{H / 4}" width="{W}" height="{H / 2}" fill="#f1bf00"/>'),
    "Italy": _v("#009246", "#ffffff", "#ce2b37"),
    "Germany": _h("#000000", "#dd0000", "#ffce00"),
    "France": _v("#002654", "#ffffff", "#ce1126"),
    "Netherlands": _h("#ae1c28", "#ffffff", "#21468b"),
    "Belgium": _v("#000000", "#fdda24", "#ef3340"),
    "Portugal": (f'<rect width="{W}" height="{H}" fill="#da291c"/><rect width="12" height="{H}" fill="#046a38"/>'
                 f'<circle cx="12" cy="{H / 2}" r="4" fill="#ffe900" stroke="#da291c" stroke-width=".6"/>'),
    "Turkey": (f'<rect width="{W}" height="{H}" fill="#e30a17"/>'
               f'<circle cx="11" cy="{H / 2}" r="5" fill="#ffffff"/><circle cx="12.3" cy="{H / 2}" r="4" fill="#e30a17"/>'
               f'<polygon points="{_star(17.2, H / 2, 2.4, -90 + 18)}" fill="#ffffff"/>'),
    "Greece": _greece(),
}


def flag_svg(country: str) -> str | None:
    body = FLAGS.get(country)
    if body is None:
        return None
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'preserveAspectRatio="none">{body}</svg>')


def flag(country: str, label: str | None = None, size: int = 18) -> str:
    """``<img>`` with the flag of ``country`` (an inline data URI), or '' for an unknown country."""
    svg = flag_svg(country)
    if svg is None:
        return ""
    alt = html.escape(label or country, quote=True)
    height = round(size * H / W)
    return (f'<img class="flag" src="data:image/svg+xml;charset=utf-8,{quote(svg)}" width="{size}" '
            f'height="{height}" alt="{alt}" title="{alt}">')
