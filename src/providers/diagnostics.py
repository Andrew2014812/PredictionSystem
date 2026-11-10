"""What each data source really delivered in one pipeline run.

A request count alone says little ("6 requests today"), so every source records, during the run:

* requests made and served from the cache, HTTP statuses, API errors (e.g. a subscription message);
* records returned, kept for our leagues, matched to our matches, and why the rest were skipped;
* new fixtures, new finished results, new and updated price quotes;
* quotas / credits left and the time the source took.

The run writes ``local_data/api/diagnostics/last_run.json`` (full detail), appends one line per source to
``diagnostics/history.csv`` and prints short lines such as
``API-Football: 4 requests · 126 fixtures returned · 92 in our leagues · 90 matched · 21 finished (6 new)``
for the GitHub Actions summary. Nothing here changes data.
"""
from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

SOURCES = ("football_data", "api_football", "oddspapi", "the_odds_api")
LABELS = {"football_data": "Football-Data", "api_football": "API-Football", "oddspapi": "OddsPapi",
          "the_odds_api": "The Odds API", "predictions": "Predictions"}
STAGE_LABELS = {"features": "features", "settlement": "settlement", "history_repair": "history repair",
                "predictions": "predictions", "shadow": "research shadows", "checks": "checks", "total": "total"}


class RunDiagnostics:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.started = datetime.now()
        self.data: dict[str, dict] = {}
        self.stages: dict[str, float] = {}        # pipeline steps that are not a data source
        self._open: dict[str, float] = {}

    # -- recording ---------------------------------------------------------------------------------------
    def _src(self, source: str) -> dict:
        return self.data.setdefault(source, {"counts": Counter(), "http": Counter(), "skipped": Counter(),
                                             "errors": [], "info": {}, "seconds": 0.0})

    def add(self, source: str, key: str, n: int = 1) -> None:
        self._src(source)["counts"][key] += int(n)

    def http(self, source: str, status) -> None:
        self._src(source)["http"][str(status)] += 1

    def skip(self, source: str, reason: str, n: int = 1) -> None:
        if n:
            self._src(source)["skipped"][reason] += int(n)

    def error(self, source: str, message: str) -> None:
        errors = self._src(source)["errors"]
        if message not in errors and len(errors) < 20:
            errors.append(message)

    def info(self, source: str, **values) -> None:
        self._src(source)["info"].update(values)

    def league(self, source: str, code: str) -> None:
        """A league whose data this source really changed in this run (new results, fixtures or prices)."""
        if code:
            self._src(source).setdefault("leagues_updated", set()).add(str(code))

    @contextmanager
    def timer(self, source: str):
        """Time spent in one source's own requests and parsing (nested sources are timed separately)."""
        self.begin(source)
        try:
            yield
        finally:
            self.end(source)

    def begin(self, source: str) -> None:
        self._open[source] = time.time()

    def end(self, source: str) -> None:
        start = self._open.pop(source, None)
        if start is not None:
            self._src(source)["seconds"] += time.time() - start

    @contextmanager
    def stage(self, name: str):
        """Time of a processing step of the pipeline (features, settlement, predictions...)."""
        start = time.time()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.time() - start

    # -- output --------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {"started_at": self.started.isoformat(timespec="seconds"),
               "finished_at": datetime.now().isoformat(timespec="seconds"), "sources": {}}
        for source, d in self.data.items():
            out["sources"][source] = {"counts": dict(d["counts"]), "http": dict(d["http"]),
                                      "skipped": dict(d["skipped"]), "errors": list(d["errors"]),
                                      "info": d["info"], "seconds": round(d["seconds"], 1),
                                      "leagues_updated": sorted(d.get("leagues_updated", ()))}
        out["timing_seconds"] = self.timing()
        return out

    def timing(self) -> dict:
        """Seconds per source and per processing step, and the whole run."""
        out = {s: round(self.data[s]["seconds"], 1) for s in SOURCES if s in self.data}
        out.update({k: round(v, 1) for k, v in self.stages.items()})
        out["total"] = round((datetime.now() - self.started).total_seconds(), 1)
        return out

    def timing_line(self) -> str:
        def fmt(sec: float) -> str:
            return f"{sec / 60:.1f} min" if sec >= 90 else f"{sec:.0f} s"
        parts = [f"{LABELS.get(k, STAGE_LABELS.get(k, k))} {fmt(v)}" for k, v in self.timing().items()]
        return "Timing: " + " · ".join(parts)

    def line(self, source: str) -> str:
        """One readable line: the source's requests, what came back and what was used."""
        d = self.data.get(source)
        label = LABELS.get(source, source)
        if not d:
            return f"{label}: not used"
        c, info = d["counts"], d["info"]
        parts = []
        if "status" in info:
            parts.append(str(info["status"]))
        if c.get("requests") or c.get("cached"):
            ok = d["http"].get("200", 0)
            parts.append(f"{c.get('requests', 0)} requests" + (f" ({ok} answered OK)" if d["http"] else "")
                         + (f" (+{c['cached']} cached)" if c.get("cached") else ""))
        if c.get("billable") is not None and source == "oddspapi":
            parts.append(f"{c.get('billable', 0)} billable")
        order = [("files", "files downloaded"), ("files_unchanged", "files unchanged"),
                 ("season_played", "played matches in the season files"), ("returned", "fixtures returned"), ("in_leagues", "in our leagues"), ("matched", "matched"),
                 ("new_fixtures", "new fixtures"), ("finished", "finished"), ("new_results", "new results"),
                 ("matches_asked", "matches asked"), ("matches_priced", "with real prices"),
                 ("quotes_new", "new quotes"), ("quotes_updated", "updated quotes"),
                 ("credits", "credits spent"), ("settled_now", "predictions settled"),
                 ("voided_now", "voided (no result after the waiting time)")]
        parts += [f"{c[k]:,} {text}" for k, text in order if c.get(k)]
        if c.get("settled_now"):
            parts.append(f"on {c.get('settled_matches', 0)} matches: Main {c.get('settled_main', 0)}, "
                         f"Risk {c.get('settled_risk', 0)}, market forecasts {c.get('settled_market', 0)}, "
                         f"refunds {c.get('settled_push', 0)}")
        if d.get("leagues_updated"):
            parts.append(f"{len(d['leagues_updated'])} leagues updated")
        if d["skipped"]:
            parts.append("skipped: " + ", ".join(f"{n} {r}" for r, n in d["skipped"].most_common(4)))
        if d["http"] and set(d["http"]) - {"200"}:
            parts.append("HTTP " + ", ".join(f"{s}×{n}" for s, n in sorted(d["http"].items())))
        if isinstance(info.get("history_checks"), dict):
            parts.append("history checks failed: " + ", ".join(f"{k} ({n})" for k, n in info["history_checks"].items()))
        elif info.get("history_checks"):
            parts.append("history checks: all passed")
        if info.get("waiting_for_a_result"):
            parts.append(f"{info['waiting_for_a_result']} past matches still waiting for a result")
        if info.get("waiting_reasons"):
            parts.append("waiting: " + ", ".join(f"{n} {r}" for r, n in info["waiting_reasons"].items()))
        if info.get("shadow_rows") or info.get("shadow_errors"):
            parts.append(f"{info.get('shadow_rows', 0)} research shadow rows"
                         + (f" ({info['shadow_errors']} shadow errors)" if info.get("shadow_errors") else ""))
        if "left" in info:
            parts.append(f"{info['left']} left")
        if d["errors"]:
            parts.append("errors: " + "; ".join(d["errors"][:2]))
        if d["seconds"] >= 1:
            parts.append(f"{d['seconds'] / 60:.1f} min" if d["seconds"] >= 90 else f"{d['seconds']:.0f} s")
        return f"{label}: " + " · ".join(parts or ["nothing"])

    def lines(self) -> list[str]:
        return [self.line(s) for s in (*SOURCES, "predictions") if s in self.data or s in SOURCES] \
            + [self.timing_line()]

    def save(self, storage) -> dict:
        """Write last_run.json and append the per-source lines to history.csv (storage = API storage)."""
        report = self.to_dict()
        report["lines"] = self.lines()
        storage.write_json("diagnostics/last_run.json", report)
        rows = [{"run_at": report["started_at"], "source": s, "line": self.line(s),
                 **{f"n_{k}": v for k, v in report["sources"].get(s, {}).get("counts", {}).items()}}
                for s in SOURCES if s in report["sources"]]
        if rows:
            path = storage.path("diagnostics/history.csv")
            new = pd.DataFrame(rows)
            old = pd.read_csv(path) if path.exists() else pd.DataFrame()
            merged = pd.concat([old, new], ignore_index=True).tail(4000)        # about three years of runs
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(path, index=False)
        return report


diag = RunDiagnostics()
