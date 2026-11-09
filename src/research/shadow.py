"""Shadow predictions: frozen research hypotheses recorded next to production, append-only.

One call per daily pipeline run, AFTER production stored its predictions::

    from src.research.shadow import record_shadow_predictions
    record_shadow_predictions(features)          # reads the production store, writes local_data/research/

It (1) settles shadow records of finished matches (appends their results), (2) records every registered
shadow (``registry.py``) for every match whose stored production prediction is a pre-match ("live") one and
whose kick-off is still in the future, and (3) refreshes the comparison report (``evaluate.py``).

Files (``local_data/research/``):

* ``shadow_predictions.parquet`` — append-only log, one row per (run, shadow, match, item); a row is added only
  when its content differs from the last row of the same (shadow, match, item), so a re-run on the same data
  adds nothing. Rows are never edited or deleted; a match that has kicked off never gets a new row. Each row
  carries ``seq``, ``row_hash`` and ``chain`` = sha256(previous chain + row_hash): editing any stored row breaks
  the chain (``verify_log``).
* ``shadow_results.parquet`` — append-only results of finished matches (goals, corners, Football-Data Bet365
  opening prices); a corrected result is a new row, the latest one counts.
* ``manifest.json`` — the registry entries as first registered (an entry that later changes is refused), one
  entry per run (rows appended, chain head, runtime) and the decisions of the forward tests once taken.
* ``shadow_report.json`` — the latest comparison of every shadow with production.

Shadows only READ the production tables; nothing under ``local_data/predictions`` is written, so the website
cannot change because of them. Any error inside is caught and returned in the summary, never raised into the
pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import config
from . import registry as R
from . import variants as V

log = logging.getLogger(__name__)

DEFAULT_DIR = config.DATA_DIR / "research"
PREDICTIONS = "shadow_predictions.parquet"
RESULTS = "shadow_results.parquet"
MANIFEST = "manifest.json"
REPORT = "shadow_report.json"
UK = ZoneInfo("Europe/London")
SCHEMA = 1

KEY = ["shadow_key", "match_id", "item"]
CONTENT = ["shadow_key", "match_id", "item", "market", "selection", "p", "base_market", "base_selection", "base_p",
           "odds", "bookmaker", "provider", "priced_at", "base_odds", "base_priced_at", "payload",
           "prod_generated_at", "prod_model_version", "kickoff"]
META_COLUMNS = ["league", "season", "date", "kickoff", "home_team", "away_team"]
RESULT_COLUMNS = ["match_id", "home_goals", "away_goals", "total_corners", "odds_home", "odds_draw", "odds_away",
                  "odds_over25", "odds_under25"]


# ------------------------------------------------------------------------------------------------ time / ids
def uk_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).astimezone(UK).replace(tzinfo=None))


def uk_to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize(UK, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC") \
        .tz_localize(None)


def utc_to_uk(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_convert(UK).tz_localize(None)


def code_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT_DIR, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return "unknown"


def _sha(text: str, algo=hashlib.sha1) -> str:
    return algo(text.encode()).hexdigest()


def _json(obj) -> str:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [clean(v) for v in list(o)]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 10)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, pd.Timestamp):
            return None if pd.isna(o) else str(o)
        if o is pd.NaT:
            return None
        return o
    return json.dumps(clean(obj), sort_keys=True, ensure_ascii=False)


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, pd.Timestamp):
        return "" if pd.isna(v) else v.isoformat()
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.9g}"
    if v is pd.NaT:
        return ""
    return str(v)


def content_hash(row: dict) -> str:
    return _sha("\x1f".join(_cell(row.get(c)) for c in CONTENT))


# ------------------------------------------------------------------------------------------------ files
def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _write_atomic(root: Path, name: str, df: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{name}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, root / name)


def load_manifest(root: Path) -> dict:
    path = root / MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": SCHEMA, "registry": {}, "runs": [], "decisions": {}}


def save_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, root / MANIFEST)


def load_predictions(root: Path | None = None) -> pd.DataFrame:
    return _read(Path(root or DEFAULT_DIR), PREDICTIONS)


def load_results(root: Path | None = None) -> pd.DataFrame:
    return _read(Path(root or DEFAULT_DIR), RESULTS)


def register(manifest: dict, specs: list[R.ShadowSpec], now: pd.Timestamp) -> dict[str, str]:
    """Add new registry entries; return {key: problem} for entries whose content changed since registration."""
    problems = {}
    for spec in R.SHADOWS:
        entry = spec.as_dict()
        old = manifest["registry"].get(spec.key)
        if old is None:
            manifest["registry"][spec.key] = {**entry, "first_registered_at": str(now)}
            continue
        stored = {k: v for k, v in old.items() if k != "first_registered_at"}
        if _json(stored) != _json(entry):
            problems[spec.key] = "registry entry changed after its first registration: register a new version"
    return problems


# ------------------------------------------------------------------------------------------------ production view
class ProductionView:
    """Read-only access to the production prediction tables (store module, a folder, a batch or an object with
    ``load_snapshots`` / ``load_markets`` / ``load_picks``)."""

    def __init__(self, source=None):
        self.batch = None
        if source is None:
            from ..prediction import store as source
        if hasattr(source, "snapshots") and hasattr(source, "markets") and hasattr(source, "picks"):
            self.batch = source                         # a PredictionBatch: its matches; history from the store
            from ..prediction import store as source
        if isinstance(source, (str, Path)):
            folder = Path(source)
            self._load = lambda name: (pd.read_parquet(folder / f"{name}.parquet")
                                       if (folder / f"{name}.parquet").exists() else pd.DataFrame())
        else:
            self._load = lambda name: getattr(source, f"load_{name}")()

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        snaps, markets, picks = self._load("snapshots"), self._load("markets"), self._load("picks")
        if self.batch is not None:
            ids = set(self.batch.snapshots["match_id"]) if not self.batch.snapshots.empty else set()
            snaps = pd.concat([snaps.loc[~snaps["match_id"].isin(ids)] if not snaps.empty else snaps,
                               self.batch.snapshots], ignore_index=True)
            markets = pd.concat([markets.loc[~markets["match_id"].isin(ids)] if not markets.empty else markets,
                                 self.batch.markets], ignore_index=True)
            picks = pd.concat([picks.loc[~picks["match_id"].isin(ids)] if not picks.empty else picks,
                               self.batch.picks], ignore_index=True)
        return snaps, markets, picks


def _model_meta() -> dict:
    try:
        from ..modelling.registry import active_version, version_metadata
        meta = version_metadata(active_version()) or {}
        return {"dc_rho": float(meta["tasks"]["home_goals"].get("extra", {}).get("dc_rho", 0.0))}
    except Exception:
        return {"dc_rho": 0.0}


def upcoming(snapshots: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Stored pre-match predictions whose match has not kicked off (the only ones a shadow may record)."""
    if snapshots.empty:
        return snapshots
    kickoff = pd.to_datetime(snapshots["kickoff"]).fillna(pd.to_datetime(snapshots["date"]))
    out = snapshots.loc[(snapshots["source"] == "live") & (kickoff > now)].copy()
    out["kickoff"] = kickoff[out.index]
    return out.drop_duplicates("match_id", keep="last").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ settlement
def settle_shadow(features: pd.DataFrame, root: Path | None = None, now: pd.Timestamp | None = None) -> dict:
    """Append the results of finished matches that have shadow records (a changed result is a new row)."""
    root = Path(root or DEFAULT_DIR)
    now = now if now is not None else uk_now()
    log_ = load_predictions(root)
    if log_.empty or features is None or features.empty:
        return {"results_appended": 0, "corrections": 0}
    ids = set(log_["match_id"])
    cols = [c for c in RESULT_COLUMNS if c in features.columns]
    f = features.loc[features["match_id"].isin(ids) & features["played"].astype(bool)
                     & features["home_goals"].notna(), cols + ["kickoff", "date"]].copy()
    kick = pd.to_datetime(f["kickoff"]).fillna(pd.to_datetime(f["date"]))
    f = f.loc[kick <= now].drop_duplicates("match_id", keep="last")
    for c in RESULT_COLUMNS:
        if c not in f:
            f[c] = np.nan
    f = f[RESULT_COLUMNS].copy()
    f["result_hash"] = [_sha("\x1f".join(_cell(v) for v in r)) for r in f[RESULT_COLUMNS].itertuples(index=False)]
    old = load_results(root)
    if not old.empty:
        last = old.drop_duplicates("match_id", keep="last").set_index("match_id")["result_hash"]
        changed = f["match_id"].map(last)
        corrections = int((changed.notna() & (changed != f["result_hash"])).sum())
        f = f.loc[changed.isna() | (changed != f["result_hash"])]
    else:
        corrections = 0
    if f.empty:
        return {"results_appended": 0, "corrections": 0}
    f["observed_at"] = pd.Series(pd.Timestamp(now), index=f.index).astype("datetime64[ns]")
    out = pd.concat([old, f], ignore_index=True) if not old.empty else f.reset_index(drop=True)
    _write_atomic(root, RESULTS, out)
    return {"results_appended": int(len(f)), "corrections": corrections}


# ------------------------------------------------------------------------------------------------ recording
def _decorate(rows: pd.DataFrame, spec: R.ShadowSpec, snaps: pd.DataFrame, now: pd.Timestamp, run_id: str,
              sha: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    meta = snaps.set_index("match_id")
    out = rows.copy()
    for c in META_COLUMNS:
        out[c] = out["match_id"].map(meta[c]) if c in meta else None
    out["prod_generated_at"] = pd.to_datetime(out["match_id"].map(meta["generated_at"]))
    out["prod_model_version"] = out["match_id"].map(meta["model_version"]).astype(str)
    out["prod_version"] = out["match_id"].map(meta["version"]) if "version" in meta else np.nan
    out["shadow_id"], out["shadow_version"], out["shadow_key"] = spec.id, spec.version, spec.key
    out["prereg_sha256"], out["freeze_sha"], out["code_sha"] = spec.prereg_sha256, spec.freeze_sha, sha
    out["recorded_at"] = now
    out["recorded_at_utc"] = uk_to_utc(now)
    out["run_id"] = run_id
    out["payload"] = [_json(p) for p in out["payload"]]
    out["kickoff"] = pd.to_datetime(out["kickoff"])
    out["date"] = pd.to_datetime(out["date"])
    for c in ("priced_at", "base_priced_at", "recorded_at", "recorded_at_utc", "prod_generated_at", "kickoff", "date"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ns]")
    records = out.to_dict("records")
    out["content_hash"] = [content_hash(r) for r in records]
    return out


def _append(root: Path, new: pd.DataFrame) -> tuple[int, str, int]:
    """Append rows whose content differs from the last stored row of the same key; returns (added, chain, total)."""
    old = load_predictions(root)
    if not old.empty:
        last = old.drop_duplicates(KEY, keep="last").set_index(KEY)["content_hash"]
        key = pd.MultiIndex.from_frame(new[KEY])
        prev = last.reindex(key).to_numpy()
        new = new.loc[prev != new["content_hash"].to_numpy()]
    new = new.drop_duplicates(KEY + ["content_hash"])
    head = str(old["chain"].iloc[-1]) if not old.empty else ""
    seq0 = int(old["seq"].iloc[-1]) + 1 if not old.empty else 0
    if new.empty:
        return 0, head, int(len(old))
    new = new.reset_index(drop=True)
    new["seq"] = np.arange(seq0, seq0 + len(new), dtype=np.int64)
    new["row_hash"] = [_sha(f"{c}|{r}|{s}") for c, r, s in zip(new["content_hash"], new["run_id"], new["seq"])]
    chain = []
    for h in new["row_hash"]:
        head = _sha(head + h, hashlib.sha256)
        chain.append(head)
    new["chain"] = chain
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new
    _write_atomic(root, PREDICTIONS, out)
    return int(len(new)), head, int(len(out))


def verify_log(root: Path | None = None) -> dict:
    """Recompute the hash chain of the prediction log and compare with the heads stored by every run."""
    root = Path(root or DEFAULT_DIR)
    df = load_predictions(root)
    manifest = load_manifest(root)
    problems = []
    head = ""
    heads_by_total = {}
    if not df.empty:
        if not (np.diff(df["seq"].to_numpy()) == 1).all():
            problems.append("sequence numbers are not consecutive")
        for i, (c, r, s, h, ch) in enumerate(zip(df["content_hash"], df["run_id"], df["seq"], df["row_hash"],
                                                 df["chain"])):
            if _sha(f"{c}|{r}|{s}") != h:
                problems.append(f"row {s}: row hash does not match its content hash")
                break
            head = _sha(head + h, hashlib.sha256)
            if head != ch:
                problems.append(f"row {s}: chain broken")
                break
            heads_by_total[i + 1] = head
        recomputed = [content_hash(r) for r in df.to_dict("records")]
        bad = int(np.sum(np.array(recomputed) != df["content_hash"].to_numpy()))
        if bad:
            problems.append(f"{bad} rows whose content no longer matches their hash")
    for run in manifest.get("runs", []):
        total, chead = run.get("rows_total"), run.get("chain_head")
        if total and chead and heads_by_total.get(total) != chead:
            problems.append(f"run {run.get('run_id')}: stored chain head not found in the log")
    return {"rows": int(len(df)), "ok": not problems, "problems": problems}


def record_shadow_predictions(features: pd.DataFrame | None, snapshots_batch_or_store=None,
                              now: pd.Timestamp | None = None, root: Path | str | None = None,
                              shadows: list[str] | None = None, evaluate: bool = True,
                              context_extra: dict | None = None) -> dict:
    """Settle, record and report every registered shadow for the upcoming matches of the latest production run.

    ``snapshots_batch_or_store``: None (the production store), a PredictionBatch, a folder with the production
    parquet files, or any object with load_snapshots / load_markets / load_picks. ``now`` is UK local time (the
    clock of kick-off times; default: the current time). Returns a summary; never raises."""
    t0 = time.time()
    root = Path(root or DEFAULT_DIR)
    now = pd.Timestamp(now) if now is not None else uk_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    summary: dict = {"run_id": run_id, "now_uk": str(now), "root": str(root), "recorded": {}, "errors": {}}
    try:
        manifest = load_manifest(root)
        problems = register(manifest, R.active(), now)
        sha = code_sha()
        summary["settlement"] = settle_shadow(features, root, now)
        snaps, markets, picks = ProductionView(snapshots_batch_or_store).tables()
        todo = upcoming(snaps, now)
        summary["upcoming_matches"] = int(len(todo))
        ids = set(todo["match_id"])
        hist_ids = set(picks.loc[picks["role"] == "main", "match_id"]) if not picks.empty else set()
        mk_all = markets
        markets = mk_all.loc[mk_all["match_id"].isin(ids)] if not mk_all.empty else mk_all
        history_push = (mk_all.loc[mk_all["match_id"].isin(hist_ids),
                                   ["match_id", "market", "selection", "push_probability"]]
                        .drop_duplicates(["match_id", "market", "selection"], keep="last")
                        if not mk_all.empty else pd.DataFrame(columns=["match_id", "market", "selection",
                                                                       "push_probability"]))
        meta = _model_meta()
        hist_snaps = snaps[["match_id", "xg_home", "xg_away"]] if not snaps.empty else pd.DataFrame()
        meta.update({"history_snapshots": hist_snaps})
        meta.update(context_extra or {})
        ctx = V.Context(now=now, snapshots=todo, markets=markets,
                        picks=picks.loc[picks["match_id"].isin(ids)] if not picks.empty else picks,
                        history=picks, history_push=history_push, features=features, research_dir=root,
                        model_meta=meta)
        frames = []
        for spec in R.active():
            if shadows is not None and spec.id not in shadows:
                continue
            problem = problems.get(spec.key) or spec.check()
            if problem:
                summary["errors"][spec.key] = problem
                continue
            t1 = time.time()
            try:
                rows = V.COMPUTE[spec.id](ctx, spec.document()) if not todo.empty else V._frame([])
                frames.append(_decorate(rows, spec, todo, now, run_id, sha))
                summary["recorded"][spec.key] = {"rows": int(len(rows)), "seconds": round(time.time() - t1, 2)}
            except Exception as exc:                                   # one shadow never stops the others
                log.exception("shadow %s failed", spec.key)
                summary["errors"][spec.key] = repr(exc)
        frames = [f for f in frames if not f.empty]
        added, head, total = _append(root, pd.concat(frames, ignore_index=True)) if frames else (
            0, (str(load_predictions(root)["chain"].iloc[-1]) if not load_predictions(root).empty else ""),
            int(len(load_predictions(root))))
        summary.update({"rows_appended": added, "rows_total": total, "chain_head": head})
        manifest["runs"].append({"run_id": run_id, "now_uk": str(now), "code_sha": sha,
                                 "upcoming_matches": summary["upcoming_matches"], "rows_appended": added,
                                 "rows_total": total, "chain_head": head,
                                 "results_appended": summary["settlement"]["results_appended"],
                                 "errors": summary["errors"], "seconds": round(time.time() - t0, 2)})
        save_manifest(root, manifest)
        if evaluate:
            from .evaluate import evaluate_shadow
            rep = evaluate_shadow(root, now=now, write=True)
            summary["report"] = {k: {"status": v.get("status"), "n": v.get("n")} for k, v in rep["shadows"].items()}
    except Exception as exc:
        log.exception("shadow recording failed")
        summary["errors"]["run"] = repr(exc)
    summary["seconds"] = round(time.time() - t0, 2)
    return summary
