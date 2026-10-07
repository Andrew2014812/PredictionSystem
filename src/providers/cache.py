"""HTTP access to the API providers with an on-disk response cache.

Every response is stored under ``local_data/api/<provider>/`` together with
the time it was fetched, and every request is appended to a quota log.
The web interface never calls an API; it only reads files produced by the
pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from .. import config

log = logging.getLogger(__name__)


class ApiCache:
    def __init__(self, provider: str, root: Path | None = None):
        self.provider = provider
        self.root = (root or config.API_DIR) / provider
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str, params: dict) -> Path:
        key = json.dumps({"url": url, "params": {k: v for k, v in params.items()
                                                   if k.lower() not in ("apikey", "api_key")}},
                         sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.root / f"{digest}.json"

    def get(self, url: str, params: dict, max_age_seconds: float) -> dict | None:
        path = self._path(url, params)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - entry["fetched_at"] > max_age_seconds:
            return None
        return entry["body"]

    def put(self, url: str, params: dict, body) -> None:
        safe = {k: v for k, v in params.items() if k.lower() not in ("apikey", "api_key")}
        entry = {"fetched_at": time.time(), "url": url, "params": safe, "body": body}
        self._path(url, params).write_text(json.dumps(entry), encoding="utf-8")

    def log_request(self, url: str, cost: float | None, remaining) -> None:
        line = json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "url": url,
                           "cost": cost, "remaining": remaining})
        with open(self.root / "quota.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def requests_today(self) -> int:
        path = self.root / "quota.log"
        if not path.exists():
            return 0
        today = datetime.now().date().isoformat()
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                   if json.loads(line)["at"].startswith(today))


def fetch_json(cache: ApiCache, url: str, params: dict, headers: dict, max_age_seconds: float,
               cost_header: str | None = None, remaining_header: str | None = None):
    """Cached GET; returns the JSON body or None on any error."""
    cached = cache.get(url, params, max_age_seconds)
    if cached is not None:
        return cached
    try:
        response = requests.get(url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException as exc:
        log.warning("%s request failed: %s", cache.provider, exc)
        return None
    cost = response.headers.get(cost_header) if cost_header else 1
    cache.log_request(url, float(cost) if cost not in (None, "") else None,
                      response.headers.get(remaining_header) if remaining_header else None)
    if response.status_code != 200:
        log.warning("%s HTTP %s for %s", cache.provider, response.status_code, url)
        return None
    body = response.json()
    cache.put(url, params, body)
    return body
