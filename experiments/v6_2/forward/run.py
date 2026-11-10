"""Forward-validation study: registry snapshot, a run on the current production state, costs and time-to-N.

    python -m experiments.v6_2.forward.simulate     # replay (results/simulation.json), run first
    python -m experiments.v6_2.forward.run          # this file (results/registry.json, current_run.json, eta.json)

The run on the current production state writes into a temporary folder under cache/, never into
local_data/research/: the first real forward records are made by the daily pipeline once it calls
record_shadow_predictions. No forward result exists yet: every registered hypothesis was frozen on 2026-11-09 and
counts only matches that kick off after that.
"""
from __future__ import annotations

import hashlib
import shutil
import time

import numpy as np
import pandas as pd

from . import common as C


def registry_snapshot() -> dict:
    from src.research import registry as R
    out = {"shadows": [], "preregistrations": {}}
    for s in R.SHADOWS:
        d = s.as_dict()
        d["prereg_check"] = s.check() or "ok"
        out["shadows"].append(d)
    for path in sorted((C.ROOT / "experiments" / "v6_2").glob("*/results/preregistration.json")):
        rel = path.relative_to(C.ROOT).as_posix()
        used = [s.key for s in R.SHADOWS if s.prereg == rel]
        out["preregistrations"][rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "shadows": used}
    return out


def current_run() -> dict:
    """Two calls on the current production state (first one builds the X5 cache, second is a same-day re-run)."""
    from src.features.store import load_features
    from src.research import shadow as SH
    prod = C.ROOT / "local_data" / "predictions"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in prod.glob("*.parquet")}
    root = C.CACHE / "current_run"
    shutil.rmtree(root, ignore_errors=True)
    t = time.time()
    feats = load_features()
    load_s = time.time() - t
    first = SH.record_shadow_predictions(feats, None, root=root)
    second = SH.record_shadow_predictions(feats, None, root=root)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in prod.glob("*.parquet")}
    size = {p.name: p.stat().st_size for p in root.rglob("*") if p.is_file()}
    return {"features_load_seconds": round(load_s, 2), "first": first, "second": second,
            "production_unchanged": before == after, "files": size, "verify": SH.verify_log(root)}


def eta(sim: dict) -> dict:
    """Weeks until each preregistered N, at the 2026/27 rate of played matches and the replay's qualifying share."""
    f = pd.read_parquet(C.ROOT / "local_data/processed/features.parquet", columns=["season", "date", "played",
                                                                                  "league"])
    f = f.loc[(f["season"] == 2026) & f["played"].astype(bool)]
    last = pd.to_datetime(f["date"]).max()
    recent = f.loc[pd.to_datetime(f["date"]) > last - pd.Timedelta(days=28)]
    per_week = len(recent) / 4
    xg_leagues = 18
    xg_share = float(recent["league"].isin(["B1", "D1", "D2", "E0", "E1", "E2", "E3", "F1", "F2", "G1", "I1", "I2",
                                            "N1", "P1", "SC0", "SP1", "SP2", "T1"]).mean())
    from dataclasses import replace
    from src.research import evaluate as EV
    from src.research import registry as R
    from src.research import shadow as SH
    from . import simulate as S
    real = S.pretend_registry()
    R.SHADOWS = tuple(replace(s, min_n=10 ** 9) for s in R.SHADOWS)
    try:
        rep = EV.evaluate_shadow(C.CACHE / "replay", now=S.END, write=False)["shadows"]
    finally:
        R.SHADOWS = real
    live = max(int(sim.get("live_matches", 0)), 1)
    _ = SH
    out = {"played_per_week": per_week, "window": [str((last - pd.Timedelta(days=28)).date()), str(last.date())],
           "xg_league_share": xg_share, "xg_leagues": xg_leagues, "shadows": {}}
    start = pd.Timestamp("2026-11-10")
    for key, r in rep.items():
        if "markets" in r:
            n = r["markets"]["ou25"]["n"]
        else:
            n = r.get("n", 0)
        share = n / live if live else np.nan
        weekly = per_week * share
        need = R.get(key.split("@")[0]).min_n
        weeks = need / weekly if weekly and need else None
        out["shadows"][key] = {"replay_cases": n, "replay_live_matches": live, "qualifying_share": share,
                               "cases_per_week": weekly, "min_n": need, "weeks_to_n": weeks,
                               "earliest_date": str((start + pd.Timedelta(weeks=weeks)).date()) if weeks else None}
    return out


def fidelity() -> dict:
    """Do the shadows compute what the studies froze? X5 weekly w and the PS-Platt fit against the preregistrations."""
    from src.research import registry as R
    from src.research import variants as V
    from src.research import xg_x5 as X
    out = {}
    doc = R.get("xg_x5").document()
    xg = X.load_xg()
    snaps = pd.read_parquet(C.ROOT / "local_data/predictions/snapshots.parquet",
                            columns=["match_id", "xg_home", "xg_away"])
    lam = snaps.rename(columns={"xg_home": "lam_h", "xg_away": "lam_a"})
    e_all = X.expected_xg(xg, xg[["match_id", "league", "date", "home_team", "away_team"]])
    weeks = []
    for i, w_pre in enumerate(doc["primary"]["w_history"]):
        ws = X.EVAL_START + pd.Timedelta(days=7 * i)
        r = X.fit_w(xg, lam, ws, cache=e_all)
        weeks.append({"week": str(ws.date()), "w_preregistered": w_pre, "w_shadow": r["w"], "n": r["n"],
                      "diff": (r["w"] - w_pre) if r["w"] == r["w"] else None})
    out["x5_weekly_w"] = weeks
    sv = pd.read_parquet(C.ROOT / "local_data/predictions/snapshots.parquet",
                         columns=["season", "source", "model_version", "date"])
    sv = sv.loc[sv["season"] == 2026]
    out["stored_model_versions_2026"] = [
        {"source": s, "model_version": v, "first": str(g["date"].min().date()), "last": str(g["date"].max().date()),
         "matches": int(len(g))} for (s, v), g in sv.groupby(["source", "model_version"])]
    mc = R.get("maincal_ps_platt_monthly").document()["hypotheses"]["H1_primary"]["current_fit_for_information"]
    picks = pd.read_parquet(C.ROOT / "local_data/predictions/picks.parquet")
    mk = pd.read_parquet(C.ROOT / "local_data/predictions/markets.parquet",
                         columns=["match_id", "market", "selection", "push_probability"]).drop_duplicates(
        ["match_id", "market", "selection"], keep="last")
    fit = V.monthly_platt_fit(picks, mk, pd.Timestamp(mc["last_match_date"]) + pd.Timedelta(days=1), 2026)
    out["ps_platt_current_fit"] = {"preregistered": mc, "shadow": fit,
                                   "note": "population = production unified_history; commit 466aab5 changed it "
                                           "after the study computed the preregistered fit"}
    return out


def volume(sim: dict) -> dict:
    """Expected daily volume in real operation: production refreshes every upcoming match each day, and a shadow
    row is added whenever its content changes. Upper bound = every refreshed (match, version) of the last 28 days
    gives a new row for every item; bytes per row from the replay."""
    v = pd.read_parquet(C.ROOT / "local_data/predictions/versions.parquet",
                        columns=["match_id", "version", "generated_at", "source", "role"])
    v = v.loc[(v["source"] == "live") & (v["role"] == "main")].drop_duplicates(["match_id", "version"])
    v["day"] = pd.to_datetime(v["generated_at"]).dt.normalize()
    last = v["day"].max()
    recent = v.loc[v["day"] > last - pd.Timedelta(days=28)]
    per_day = recent.groupby("day").size()
    st = sim.get("storage", {})
    rows_per_match = st.get("rows_total", 0) / max(sim.get("live_matches", 1), 1)
    bytes_per_row = st.get("bytes_predictions", 0) / max(st.get("rows_total", 1), 1)
    days_with_runs = int(per_day.size)
    mean_versions = float(per_day.sum() / 28)
    return {"window": [str((last - pd.Timedelta(days=28)).date()), str(last.date())],
            "versions_per_day_mean": mean_versions, "days_with_runs": days_with_runs,
            "versions_per_run_max": int(per_day.max()) if len(per_day) else 0,
            "rows_per_match_version": rows_per_match, "bytes_per_row": bytes_per_row,
            "rows_per_day_upper": mean_versions * rows_per_match,
            "kb_per_day_upper": mean_versions * rows_per_match * bytes_per_row / 1024,
            "mb_per_season_upper": mean_versions * rows_per_match * bytes_per_row * 300 / 1024 ** 2,
            "production_tables_kb": {p.name: round(p.stat().st_size / 1024) for p in
                                     sorted((C.ROOT / "local_data/predictions").glob("*.parquet"))}}


def main() -> None:
    C.setup_logging("forward-run")
    t0 = time.time()
    C.save_json("registry.json", registry_snapshot())
    C.save_json("fidelity.json", fidelity())
    cur = current_run()
    C.save_json("current_run.json", cur)
    sim = C.load_json("simulation.json", {})
    C.save_json("eta.json", eta(sim))
    C.save_json("volume.json", volume(sim))
    C.save_json("run_info.json", {"environment": C.environment(), "minutes": round((time.time() - t0) / 60, 2)})


if __name__ == "__main__":
    main()
