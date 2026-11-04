"""Downloading raw Football-Data CSV files to local storage."""
from __future__ import annotations

import logging

import requests

from .. import config
from ..leagues import league_codes
from ..storage import fixtures_storage, results_storage

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (football-prediction-system)"}


# Columns checked in every current-season file (they differ by league and season).
STAT_COLUMNS = {"goals": ("FTHG", "FTAG", "FTR"), "half_time": ("HTHG", "HTAG"), "shots": ("HS", "AS"),
                "shots_on_target": ("HST", "AST"), "corners": ("HC", "AC"), "fouls": ("HF", "AF"),
                "cards": ("HY", "AY", "HR", "AR")}
PRICE_COLUMNS = {"bet365_1x2": ("B365H",), "pinnacle_1x2": ("PSH",), "max_avg_1x2": ("MaxH", "AvgH"),
                 "bet365_total_2_5": ("B365>2.5",), "pinnacle_total_2_5": ("P>2.5",),
                 "asian_handicap": ("AHh", "B365AHH"), "closing_1x2": ("B365CH", "PSCH"),
                 "closing_total_2_5": ("B365C>2.5", "PC>2.5"), "closing_asian": ("AHCh", "B365CAHH")}


def _fetch_csv(url: str) -> bytes | None:
    """Return the CSV body, or None when the file is missing / not a CSV."""
    from ..providers.diagnostics import diag
    diag.add("football_data", "requests")
    try:
        response = requests.get(url, headers=_HEADERS, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException as exc:
        log.warning("download failed %s: %s", url, exc)
        diag.http("football_data", "network")
        return None
    diag.http("football_data", response.status_code)
    body = response.content
    if response.status_code != 200 or not body.lstrip(b"\xef\xbb\xbf").startswith(b"Div"):
        log.warning("no CSV at %s (HTTP %s)", url, response.status_code)
        diag.error("football_data", f"no CSV at {url.rsplit('/', 2)[-2:]} (HTTP {response.status_code})")
        return None
    return body


def describe_results_file(body: bytes, previous: bytes | None) -> dict:
    """Rows, played matches, new results against the previous copy, last match date and which statistics
    and price columns the file really has."""
    import io

    import pandas as pd
    def frame(raw):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
        except Exception:                                         # noqa: BLE001 - a broken file is reported
            return pd.DataFrame()
    new = frame(body)
    played = new.dropna(subset=[c for c in ("FTHG", "FTAG") if c in new]) if not new.empty else new
    old = frame(previous) if previous else pd.DataFrame()
    key = [c for c in ("Date", "HomeTeam", "AwayTeam") if c in new]
    old_played = old.dropna(subset=[c for c in ("FTHG", "FTAG") if c in old]) if not old.empty else old
    if key and not old_played.empty and set(key) <= set(old_played):
        before = set(map(tuple, old_played[key].astype(str).to_numpy()))
        new_results = sum(tuple(r) not in before for r in played[key].astype(str).to_numpy())
    else:
        new_results = len(played)
    def present(cols):
        return all(c in new and new[c].notna().any() for c in cols)
    return {"rows": int(len(new)), "played": int(len(played)), "new_results": int(new_results),
            "last_date": str(played["Date"].iloc[-1]) if len(played) and "Date" in played else None,
            "stats": {k: present(v) for k, v in STAT_COLUMNS.items()},
            "prices": {k: present(v) for k, v in PRICE_COLUMNS.items()},
            "xg": any(c.lower() in ("hxg", "axg", "xg") for c in new.columns)}


def download_results(seasons: list[int], leagues: list[str] | None = None) -> list[str]:
    """Download season result files; returns the saved relative paths."""
    saved = []
    for season in seasons:
        code = config.season_code(season)
        for div in leagues or league_codes():
            url = f"{config.FOOTBALL_DATA_URL}/{config.FOOTBALL_DATA_RESULTS_PATH}/{code}/{div}.csv"
            body = _fetch_csv(url)
            if body is None:
                continue
            target = results_storage.writable_path(f"{code}/{div}.csv")
            previous = target.read_bytes() if target.exists() else None
            _record_results_file(div, body, previous, season == config.current_season())
            target.write_bytes(body)
            saved.append(f"{code}/{div}.csv")
            log.info("saved %s", target)
    return saved


def _record_results_file(div: str, body: bytes, previous: bytes | None, current: bool) -> None:
    from ..providers.diagnostics import diag
    diag.add("football_data", "files_unchanged" if previous == body else "files")
    if not current:
        return
    info = describe_results_file(body, previous)
    diag.add("football_data", "season_played", info["played"])
    diag.add("football_data", "new_results", info["new_results"])
    leagues = diag._src("football_data")["info"].setdefault("leagues", {})
    leagues[div] = info


def download_fixtures() -> bool:
    """Replace the local fixtures file with the current Football-Data one."""
    url = f"{config.FOOTBALL_DATA_URL}/{config.FOOTBALL_DATA_FIXTURES_FILE}"
    body = _fetch_csv(url)
    if body is None:
        return False
    fixtures_storage.writable_path(config.FOOTBALL_DATA_FIXTURES_FILE).write_bytes(body)
    log.info("saved fixtures")
    return True
