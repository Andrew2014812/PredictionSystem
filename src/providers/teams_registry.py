"""One internal team identifier and the identifiers each provider uses for it.

Internal id: ``"<country>|<Football-Data name>"`` (e.g. ``"England|Man United"``), the name every
model, table and prediction already uses. Providers name and number teams their own way, so the
registry ``local_data/teams/provider_ids.json`` stores, per provider, ``provider team id ->
internal id``. A provider id is learned only from an UNAMBIGUOUS name match inside the right league
(aliases.resolve) and is never overwritten by a different team: a conflict is reported instead, so
odds of one match can never be attached to another.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import config
from . import aliases

log = logging.getLogger(__name__)
PATH = config.DATA_DIR / "teams" / "provider_ids.json"


def internal_id(country: str, team: str) -> str:
    return f"{country}|{team}"


class TeamRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or PATH
        self.data: dict[str, dict[str, str]] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.conflicts: list[tuple] = []
        self.unmatched: set[tuple] = set()

    def resolve(self, provider: str, league: str, country: str, name: str, provider_id=None,
                known: set[str] | None = None) -> str | None:
        """Football-Data name of a provider team, or None (never a guess)."""
        ids = self.data.setdefault(provider, {})
        by_name = aliases.resolve(name, known)
        if provider_id is not None and str(provider_id) in ids:
            iid = ids[str(provider_id)]
            team_country, team = iid.split("|", 1)
            if by_name is not None and internal_id(country, by_name) != iid:
                # the id says one club, the name another: attach nothing
                self.conflicts.append((provider, str(provider_id), iid, internal_id(country, by_name)))
                log.warning("team id conflict %s %s: %s vs %s", provider, provider_id, iid, by_name)
                return None
            if team_country == country and (known is None or team in known):
                return team                    # a learned id also covers spellings the aliases miss
        team = by_name
        if team is None:
            self.unmatched.add((provider, league, name))
            return None
        if provider_id is not None:
            iid, pid = internal_id(country, team), str(provider_id)
            if pid in ids and ids[pid] != iid:
                self.conflicts.append((provider, pid, ids[pid], iid))
                log.warning("team id conflict %s %s: %s vs %s", provider, pid, ids[pid], iid)
                return None
            if pid not in ids and iid in ids.values():
                # one club, one id per provider: a second id for the same club is a mis-read name
                other = next(k for k, v in ids.items() if v == iid)
                self.conflicts.append((provider, pid, iid, f"already id {other}"))
                log.warning("team id conflict %s %s: %s already has id %s", provider, pid, iid, other)
                return None
            ids[pid] = iid
        return team

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({p: dict(sorted(v.items())) for p, v in sorted(self.data.items())},
                                        indent=1, ensure_ascii=False), encoding="utf-8")
