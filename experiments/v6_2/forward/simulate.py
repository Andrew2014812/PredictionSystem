"""Replay of the forward-validation loop on past data, in a temporary folder (proof of mechanics, NOT evidence).

Pretend freeze: every registered shadow is treated as frozen on PRETEND_FREEZE (2026-10-01) instead of its real
freeze (2026-11-09). Production stored one final pre-kick-off ("live") prediction per match; the replay runs the
shadows right after the production run that made it (its ``generated_at``), with the state production had then:

* the matches of that run only (a match whose final version came from a later run is recorded at that run);
* their prices as recorded in markets.parquet at generation (prices added to picks later by the history repair
  are removed: no look-ahead);
* prediction history and results only of matches played before the day of the run (for PS-Platt monthly and
  for settlement); Football-Data xG on disk is used with the lag-0 rule of the xG study (earlier days only).

Every run is repeated (same day, same data -> must append nothing), and one extra call is made after the last
kick-off (must append nothing for started matches). At the end the hash chain is verified, a tampered copy must
fail the verification, the production files must be byte-identical to before, and the comparison report is
evaluated at weekly cut-offs to show the "no conclusion before N" rule and the frozen decision.

The hypotheses were selected after looking at 2026/27, so whatever the replay reports is in-sample with respect to
that selection: it demonstrates that the loop works, not that a shadow is better.

    python -m experiments.v6_2.forward.simulate
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

PRETEND_FREEZE_UK = pd.Timestamp("2026-10-01 00:00:00")
END = pd.Timestamp("2026-11-09 07:00:00")


def _hashes(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.parquet"))}


class ReplayView:
    """Production tables as they were right after one production run (read-only, in memory)."""

    def __init__(self, snaps, markets, picks):
        self.s, self.m, self.p = snaps, markets, picks

    def load_snapshots(self):
        return self.s

    def load_markets(self):
        return self.m

    def load_picks(self):
        return self.p


def pretend_registry():
    from src.research import registry as R
    real = R.SHADOWS
    fake = []
    for s in real:
        rule = ({"kind": "date_from", "date": str(PRETEND_FREEZE_UK.date())} if s.forward_rule["kind"] == "date_from"
                else {"kind": "kickoff_after_utc", "utc": "2026-10-01T00:00:00Z"})
        fake.append(replace(s, forward_rule=rule))
    R.SHADOWS = tuple(fake)
    return real


def main() -> dict:
    C.setup_logging("forward-sim")
    from src.features.store import load_features
    from src.research import evaluate as EV
    from src.research import registry as R
    from src.research import shadow as SH
    from src.research.xg_x5 import load_xg

    prod_dir = C.ROOT / "local_data" / "predictions"
    before = _hashes(prod_dir)
    root = C.CACHE / "replay"
    shutil.rmtree(root, ignore_errors=True)
    real_registry = pretend_registry()
    t_start = time.time()

    snaps = pd.read_parquet(prod_dir / "snapshots.parquet")
    markets = pd.read_parquet(prod_dir / "markets.parquet")
    picks = pd.read_parquet(prod_dir / "picks.parquet")
    feats = load_features()
    xg = load_xg()
    snaps["kickoff"] = pd.to_datetime(snaps["kickoff"]).fillna(pd.to_datetime(snaps["date"]))
    snaps["gen_uk"] = [SH.utc_to_uk(pd.Timestamp(g).tz_localize("UTC").isoformat()) for g in snaps["generated_at"]]
    live = snaps.loc[(snaps["source"] == "live") & (snaps["kickoff"] > PRETEND_FREEZE_UK)
                     & (snaps["gen_uk"] < snaps["kickoff"]) & (snaps["gen_uk"] < END)]
    runs = sorted(live["gen_uk"].dt.floor("min").unique())
    C.log.info("replay: %d live matches in %d production runs", len(live), len(runs))

    # prices exactly as recorded at generation (markets.parquet); picks of the run carry those prices
    mk_price = markets.drop_duplicates(["match_id", "market", "selection"], keep="last").set_index(
        ["match_id", "market", "selection"])[["odds", "priced_at", "bookmaker", "provider"]]

    run_rows, size_rows, evals = [], [], []
    cutoffs = list(pd.date_range("2026-10-08", END.normalize(), freq="7D")) + [END]
    next_cut = 0
    for t in runs:
        t_run = pd.Timestamp(t) + pd.Timedelta(minutes=1)
        day = t_run.normalize()
        while next_cut < len(cutoffs) and cutoffs[next_cut] <= t_run:
            rep = EV.evaluate_shadow(root, now=cutoffs[next_cut], write=True)
            evals.append({"cutoff": str(cutoffs[next_cut]), **{k: {"status": v.get("status"), "n": v.get("n"),
                                                                     "frozen": v.get("frozen", False)}
                                                                 for k, v in rep["shadows"].items()}})
            next_cut += 1
        this = live.loc[live["gen_uk"].dt.floor("min") == t]
        ids = set(this["match_id"])
        hist_s = snaps.loc[(snaps["kickoff"] <= t_run)]
        s_run = pd.concat([hist_s, this], ignore_index=True).drop(columns=["gen_uk"])
        m_run = markets.loc[markets["match_id"].isin(ids | set(hist_s["match_id"]))]
        p_hist = picks.loc[pd.to_datetime(picks["date"]) < day].copy()
        p_now = picks.loc[picks["match_id"].isin(ids)].copy()
        key = pd.MultiIndex.from_frame(p_now[["match_id", "market", "selection"]])
        rec = mk_price.reindex(key)
        p_now["odds"] = rec["odds"].to_numpy(float)
        p_now["odds_priced_at"] = pd.to_datetime(rec["priced_at"].to_numpy())
        p_now["bookmaker"], p_now["provider"] = rec["bookmaker"].to_numpy(), rec["provider"].to_numpy()
        p_now[["status", "profit", "home_goals", "away_goals", "total_corners"]] = ["UPCOMING", np.nan, np.nan,
                                                                                     np.nan, np.nan]
        p_run = pd.concat([p_hist.loc[~p_hist["match_id"].isin(ids)], p_now], ignore_index=True)
        f_run = feats.copy()
        future = pd.to_datetime(f_run["date"]) >= day
        f_run.loc[future, "played"] = False
        f_run.loc[future, ["home_goals", "away_goals", "total_corners"]] = np.nan
        view = ReplayView(s_run, m_run, p_run)
        x_run = xg.loc[pd.to_datetime(xg["date"]) < day]
        extra = {"xg_table": x_run}
        s1 = SH.record_shadow_predictions(f_run, view, now=t_run, root=root, evaluate=False, context_extra=extra)
        s2 = SH.record_shadow_predictions(f_run, view, now=t_run + pd.Timedelta(minutes=5), root=root,
                                          evaluate=False, context_extra=extra)
        run_rows.append({"run_uk": str(t_run), "matches": int(len(ids)), "upcoming_seen": s1.get("upcoming_matches"),
                         "rows_appended": s1.get("rows_appended"), "rerun_rows_appended": s2.get("rows_appended"),
                         "seconds": s1.get("seconds"), "rerun_seconds": s2.get("seconds"),
                         "results_appended": s1.get("settlement", {}).get("results_appended"),
                         "per_shadow_seconds": {k: v["seconds"] for k, v in s1.get("recorded", {}).items()},
                         "errors": {**s1.get("errors", {}), **s2.get("errors", {})}})
        size_rows.append({"run_uk": str(t_run), "bytes": int((root / SH.PREDICTIONS).stat().st_size)
                          if (root / SH.PREDICTIONS).exists() else 0,
                          "rows_total": s1.get("rows_total")})
        C.log.info("run %s: %d matches, +%d rows (re-run +%s) in %.1f s", t_run, len(ids), s1.get("rows_appended", 0),
                   s2.get("rows_appended"), s1.get("seconds", 0))

    # settle everything played before END, then a call after every kick-off: no new prediction rows
    f_end = feats.copy()
    late = pd.to_datetime(f_end["date"]) >= END.normalize()
    f_end.loc[late, "played"] = False
    after = SH.record_shadow_predictions(f_end, ReplayView(snaps.loc[snaps["kickoff"] <= END].drop(columns=["gen_uk"]),
                                                           markets, picks),
                                         now=END, root=root, evaluate=False)
    final = EV.evaluate_shadow(root, now=END, write=True)
    while next_cut < len(cutoffs):
        next_cut += 1
    evals.append({"cutoff": str(END), **{k: {"status": v.get("status"), "n": v.get("n"),
                                             "frozen": v.get("frozen", False)} for k, v in final["shadows"].items()}})
    # repeat the final evaluation one week later with the same data: the frozen decision must not change
    again = EV.evaluate_shadow(root, now=END + pd.Timedelta(days=7), write=False)
    frozen_same = all(json.dumps(final["shadows"][k].get("test"), sort_keys=True, default=str)
                      == json.dumps(again["shadows"][k].get("test"), sort_keys=True, default=str)
                      for k in final["shadows"] if final["shadows"][k].get("test"))

    verify = SH.verify_log(root)
    # tamper test on a copy
    tam = C.CACHE / "replay_tampered"
    shutil.rmtree(tam, ignore_errors=True)
    shutil.copytree(root, tam)
    df = pd.read_parquet(tam / SH.PREDICTIONS)
    i = int(len(df) // 2)
    df.loc[i, "p"] = (df.loc[i, "p"] if df.loc[i, "p"] == df.loc[i, "p"] else 0.5) + 0.01
    df.to_parquet(tam / SH.PREDICTIONS, index=False)
    tampered = SH.verify_log(tam)
    shutil.rmtree(tam, ignore_errors=True)

    after_hashes = _hashes(prod_dir)
    log_df = pd.read_parquet(root / SH.PREDICTIONS)
    res_df = pd.read_parquet(root / SH.RESULTS)
    late_rows = int((pd.to_datetime(log_df["recorded_at"]) >= pd.to_datetime(log_df["kickoff"])).sum())
    by_shadow = log_df.groupby("shadow_key").agg(rows=("seq", "size"), matches=("match_id", "nunique")).reset_index()
    sizes = pd.DataFrame(size_rows)
    days = (pd.Timestamp(runs[-1]) - pd.Timestamp(runs[0])).days + 1 if runs else 0
    out = {
        "pretend_freeze_uk": str(PRETEND_FREEZE_UK), "end_uk": str(END), "runs": run_rows,
        "n_runs": len(run_rows), "live_matches": int(len(live)),
        "idempotent_reruns": all(r["rerun_rows_appended"] == 0 for r in run_rows),
        "after_kickoff_call_rows_appended": after.get("rows_appended"),
        "rows_recorded_at_or_after_kickoff": late_rows,
        "verify": verify, "tamper_detected": not tampered["ok"], "tamper_problems": tampered["problems"][:3],
        "production_unchanged": before == after_hashes, "production_files": list(before),
        "frozen_decision_stable": frozen_same,
        "evaluations": evals, "final_report": final,
        "storage": {"rows_total": int(len(log_df)), "bytes_predictions": int((root / SH.PREDICTIONS).stat().st_size),
                    "bytes_results": int((root / SH.RESULTS).stat().st_size),
                    "bytes_manifest": int((root / SH.MANIFEST).stat().st_size),
                    "bytes_x5_cache": int((root / "cache" / "x5_expected_xg.parquet").stat().st_size)
                    if (root / "cache" / "x5_expected_xg.parquet").exists() else 0,
                    "days": days, "results_rows": int(len(res_df)),
                    "by_shadow": by_shadow.to_dict("records"), "size_by_run": sizes.to_dict("records")},
        "runtime": {"mean_s": float(np.mean([r["seconds"] for r in run_rows])),
                    "median_s": float(np.median([r["seconds"] for r in run_rows])),
                    "max_s": float(np.max([r["seconds"] for r in run_rows])),
                    "rerun_mean_s": float(np.mean([r["rerun_seconds"] for r in run_rows])),
                    "total_minutes": round((time.time() - t_start) / 60, 2)},
    }
    from src.research import registry as R2
    R2.SHADOWS = real_registry
    C.save_json("simulation.json", out)
    return out


if __name__ == "__main__":
    main()
