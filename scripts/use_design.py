"""Switch the website's look between the FootPredict designs (no switch is shown on the site).

    python scripts/use_design.py v6      # back to the v6 look (navy, Inter, weight 800)
    python scripts/use_design.py v6.1    # the v6.1 look (graphite-green, IBM Plex Sans + Source Sans 3)

It sets ``UI_DESIGN`` in src/config.py and the widget colours of ``.streamlit/config.toml`` (Streamlit reads
them at start-up, so restart a local server; Streamlit Cloud redeploys after the push). Without editing
files, the environment variable / Streamlit secret ``FP_DESIGN=v6`` has the same effect on the page styles.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIDGET_COLOURS = {
    "v6": {"primaryColor": "#22c55e", "backgroundColor": "#0b1120", "secondaryBackgroundColor": "#131c2e",
           "textColor": "#e2e8f0", "borderColor": "#1f2a40"},
    "v6.1": {"primaryColor": "#269d7d", "backgroundColor": "#101820", "secondaryBackgroundColor": "#1b292d",
             "textColor": "#e1e7e5", "borderColor": "#2b3c40"},
}


def use(name: str) -> None:
    if name not in WIDGET_COLOURS:
        sys.exit(f"unknown design {name!r}; choose one of {', '.join(WIDGET_COLOURS)}")
    cfg = ROOT / "src" / "config.py"
    text = cfg.read_text(encoding="utf-8")
    text, n = re.subn(r'^UI_DESIGN = "[^"]*"', f'UI_DESIGN = "{name}"', text, flags=re.M)
    if n != 1:
        sys.exit("UI_DESIGN not found in src/config.py")
    cfg.write_text(text, encoding="utf-8")
    toml = ROOT / ".streamlit" / "config.toml"
    body = toml.read_text(encoding="utf-8")
    for key, value in WIDGET_COLOURS[name].items():
        body, n = re.subn(rf'^{key} = "[^"]*"', f'{key} = "{value}"', body, flags=re.M)
        if n != 1:
            sys.exit(f"{key} not found in .streamlit/config.toml")
    toml.write_text(body, encoding="utf-8")
    print(f"design {name}: src/config.py and .streamlit/config.toml updated")


if __name__ == "__main__":
    use(sys.argv[1] if len(sys.argv) > 1 else "v6.1")
