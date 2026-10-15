"""Persistent request ledger for the credit-based The Odds API.

The raw-response cache (``local_data/api/<provider>/*.json``) is not committed,
so it does not survive between GitHub Actions runs. This ledger is a small
committed JSON file that does:

    local_data/api/the_odds_api/ledger.json
    {
      "remaining": 315, "used": 185, "checked_at": "...",      # from the API's own headers
      "month": "2026-10", "month_credits": 185,
      "requests": {"odds|soccer_epl": {"at": "...", "cost": 2, "items": 9}, ...}
    }

Every request that costs credits is recorded — also when it returned no
prices — so the same league / match is not asked again within its refresh
window, and every run respects a per-run budget and a monthly reserve.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from .. import config


class CreditLedger:
    def __init__(self, path: Path | None = None, monthly_reserve: int | None = None):
        self.path = path or (config.API_DIR / "the_odds_api" / "ledger.json")
        self.reserve = config.ODDS_API_MONTHLY_RESERVE if monthly_reserve is None else monthly_reserve
        self.data = {"requests": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.data.setdefault("requests", {})
        self.spent_this_run = 0

    # -- reading ----------------------------------------------------------------
    @property
    def remaining(self) -> int | None:
        value = self.data.get("remaining")
        return int(value) if value is not None else None

    def last_request(self, key: str) -> datetime | None:
        entry = self.data["requests"].get(key)
        return datetime.fromisoformat(entry["at"]) if entry else None

    def requested_within(self, key: str, hours: float, now: datetime) -> bool:
        """True if ``key`` was asked within ``hours`` and got an ANSWER (prices or "no market").
        A temporary failure (network, 5xx) may be retried at the next run."""
        entry = self.data["requests"].get(key)
        if not entry or str(entry.get("status", "200")) not in ("200", "not_found", "cached"):
            return False
        return now - datetime.fromisoformat(entry["at"]) < timedelta(hours=hours)

    def can_spend(self, cost: int, run_budget: int) -> bool:
        if self.spent_this_run + cost > run_budget:
            return False
        if self.remaining is not None and self.remaining - cost < self.reserve:
            return False
        return True

    # -- writing ----------------------------------------------------------------
    def record(self, key: str, cost: int, items: int, now: datetime, used=None, remaining=None,
               status=200) -> None:
        self.spent_this_run += cost
        self.data["requests"][key] = {"at": now.isoformat(timespec="seconds"), "cost": cost, "items": items,
                                      "status": str(status)}
        month = now.strftime("%Y-%m")
        if self.data.get("month") != month:
            self.data["month"], self.data["month_credits"] = month, 0
        self.data["month_credits"] = self.data.get("month_credits", 0) + cost
        if used is not None:
            self.data["used"] = int(float(used))
        if remaining is not None:
            self.data["remaining"] = int(float(remaining))
        self.data["checked_at"] = now.isoformat(timespec="seconds")

    def prune(self, now: datetime, days: int = 40) -> None:
        cut = now - timedelta(days=days)
        self.data["requests"] = {k: v for k, v in self.data["requests"].items()
                                 if datetime.fromisoformat(v["at"]) >= cut}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")
