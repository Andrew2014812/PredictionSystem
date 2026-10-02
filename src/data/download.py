"""Downloading raw Football-Data CSV files to local storage."""
from __future__ import annotations

import logging

import requests

from .. import config
from ..leagues import league_codes
from ..storage import fixtures_storage, results_storage

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (football-prediction-system)"}


def _fetch_csv(url: str) -> bytes | None:
    """Return the CSV body, or None when the file is missing / not a CSV."""
    try:
        response = requests.get(url, headers=_HEADERS, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException as exc:
        log.warning("download failed %s: %s", url, exc)
        return None
    body = response.content
    if response.status_code != 200 or not body.lstrip(b"\xef\xbb\xbf").startswith(b"Div"):
        log.warning("no CSV at %s (HTTP %s)", url, response.status_code)
        return None
    return body


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
            target.write_bytes(body)
            saved.append(f"{code}/{div}.csv")
            log.info("saved %s", target)
    return saved


def download_fixtures() -> bool:
    """Replace the local fixtures file with the current Football-Data one."""
    url = f"{config.FOOTBALL_DATA_URL}/{config.FOOTBALL_DATA_FIXTURES_FILE}"
    body = _fetch_csv(url)
    if body is None:
        return False
    fixtures_storage.writable_path(config.FOOTBALL_DATA_FIXTURES_FILE).write_bytes(body)
    log.info("saved fixtures")
    return True
