"""Risk strategy research of FootPredict v6.1 (research only - production Risk is not changed).

    python -m experiments.v6_1.risk.run --history        # walk-forward 2023/24, 2024/25 (+ 2025/26 info)
    python -m experiments.v6_1.risk.run --analysis       # why model EV does not turn into ROI (history)
    python -m experiments.v6_1.risk.run --preregister    # writes results/preregistration.json (commit it!)
    python -m experiments.v6_1.risk.run --independent    # 2026/27 production history (needs the committed
                                                         #   pre-registration)
    python -m experiments.v6_1.risk.run --figures
    python -m experiments.v6_1.risk.run --all            # everything in this order
    add --refresh to rebuild the cached model predictions and candidate tables

Protocol: every strategy is run on the same matches; a strategy is CHOSEN on 2023/24 + 2024/25 only;
2025/26 is reported for information; the stored 2026/27 production history (pre-kick-off
probabilities, recorded / archived real prices) is the independent check, evaluated only after the
choice has been written to results/preregistration.json and committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_research import common as C
from src import config

from . import analysis as A
from . import data as D
from . import strategies as S
from .calibration import Calibrator
from .evaluate import Season, metrics, pooled

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS, FIGURES, CACHE = HERE / "results", HERE / "figures", HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(exist_ok=True)
SELECT, INFO, INDEPENDENT = (2023, 2024), 2025, 2026
CURRENT = "Текущий Risk"
CONSTRAINTS = {"min_coverage_of_main": 0.995, "min_priced_share": 0.85, "min_avg_odds": 2.80}
PICK_COLS = ["match_id", "date", "league", "market", "selection", "group", "key", "p", "p_cal", "p_blend",
             "p_mkt", "odds", "ev", "ev_p_cal", "ev_p_blend", "status", "decided", "priced", "won", "profit", "tier"]
log = C.setup_logging("risk")


# ------------------------------------------------------------------------------------------------ helpers
def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def save(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(C._clean(obj), indent=1, ensure_ascii=False))
    log.info("wrote %s", path)
    return path


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_cache() -> None:
    from .base import block
    for s in D.HISTORY:
        block(s, refresh=True)
        D.history_season(s, refresh=True)
    D.independent_season(refresh=True)


def season_frame(season: int) -> tuple[pd.DataFrame, Calibrator]:
    """Prepared candidate table of a season and the calibrator fitted on the previous season."""
    df = S.prepare(D.independent_season()[0] if season == INDEPENDENT else D.history_season(season))
    cal = Calibrator().fit(S.prepare(D.history_season(season - 1)))
    return cal.apply(df), cal


def evaluate_season(season: int, df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    sea = Season(df, seed=season)
    res, bands, boots, picks = {}, {}, {}, {}
    base = None
    for name, (family, fn) in S.catalogue().items():
        pk = fn(df)
        m, boot = metrics(pk, sea, base)
        if name == CURRENT:
            base = boot
        m["family"] = family
        prof, stake = sea.vectors(pk)
        boots[(str(season), name)] = (boot, sea.w @ prof, sea.w @ stake)
        res[name] = m
        picks[name] = pk.reindex(columns=PICK_COLS)
    for (bl, fl), fn in S.band_catalogue().items():
        pk = fn(df)
        m, _ = metrics(pk, sea, base)
        m.pop("odds_bands", None)
        bands[f"{bl} | {fl}"] = {"band": bl, "floor": fl, **m}
    # odds-band breakdown of the bets of the current rule is inside res[CURRENT]["odds_bands"]
    return res, bands, boots, picks


def store_picks(season: int, picks: dict) -> None:
    frame = pd.concat([p.assign(strategy=n) for n, p in picks.items()], ignore_index=True)
    frame.to_parquet(CACHE / f"picks_{season}.parquet")


def summary_rows(results: dict) -> list[dict]:
    rows = []
    for season, per in results.items():
        for name, m in per.items():
            rows.append({"season": season, "strategy": name, "family": m.get("family"),
                         **{k: m.get(k) for k in ("matches", "picks", "coverage", "predictions", "bets",
                                                  "priced_share", "avg_odds", "hit_rate", "hit_rate_bets",
                                                  "roi", "profit", "avg_ev", "ev_minus_roi", "avg_p_bets",
                                                  "calibration_gap", "calibration_z", "max_drawdown",
                                                  "longest_losing_streak", "profit_sd")},
                         "roi_ci_low": (m.get("roi_ci95") or [None, None])[0],
                         "roi_ci_high": (m.get("roi_ci95") or [None, None])[1],
                         "diff_ci_low": (m.get("diff_vs_current_ci95") or [None, None])[0],
                         "diff_ci_high": (m.get("diff_vs_current_ci95") or [None, None])[1]})
    return rows


# ------------------------------------------------------------------------------------------------ stages
def stage_history() -> dict:
    t0 = time.time()
    out = {"seasons": {}, "bands": {}, "calibrators": {}, "checks": {}, "seconds": {}}
    all_boots = {}
    for s in (*SELECT, INFO):
        t = time.time()
        df, cal = season_frame(s)
        out["calibrators"][str(s)] = cal.to_dict()
        out["checks"][str(s)] = S.check_current(df)              # vectorised rule == production function
        res, bands, boots, picks = evaluate_season(s, df)
        out["seasons"][str(s)], out["bands"][str(s)] = res, bands
        all_boots.update(boots)
        store_picks(s, picks)
        out["seconds"][str(s)] = time.time() - t
        log.info("season %s: %d matches, current ROI %.4f (%.0fs)", s, res[CURRENT]["matches"],
                 res[CURRENT]["roi"], time.time() - t)
    names = list(S.catalogue())
    out["pooled_select"] = {n: pooled(out["seasons"], [str(s) for s in SELECT], n, all_boots) for n in names}
    out["pooled_all_history"] = {n: pooled(out["seasons"], [str(s) for s in (*SELECT, INFO)], n, all_boots)
                                 for n in names}
    out["pooled_bands_select"] = {}
    for key in out["bands"][str(SELECT[0])]:
        rows = [out["bands"][str(s)][key] for s in SELECT]
        bets = sum(r["bets"] for r in rows)
        out["pooled_bands_select"][key] = {"band": rows[0]["band"], "floor": rows[0]["floor"], "bets": bets,
                                           "roi": sum(r["profit"] for r in rows) / bets if bets else None,
                                           "coverage": float(np.mean([r["coverage"] for r in rows])),
                                           "avg_ev": float(np.nanmean([r["avg_ev"] or np.nan for r in rows])),
                                           "rois": [r["roi"] for r in rows]}
    out["selection"] = choose(out)
    out["minutes"] = (time.time() - t0) / 60
    save("history.json", out)
    pd.DataFrame(summary_rows(out["seasons"])).to_csv(RESULTS / "strategies_history.csv", index=False)
    pd.DataFrame([{"season": s, **v} for s, per in out["bands"].items() for v in per.values()]).drop(
        columns=["tier_shares", "group_shares", "roi_ci95", "diff_vs_current_ci95"], errors="ignore").to_csv(
        RESULTS / "odds_band_ev_floor.csv", index=False)
    return out


def choose(out: dict) -> dict:
    """Pre-registered choice on 2023/24 + 2024/25 only."""
    sel = [str(s) for s in SELECT]
    pooled_tab = out["pooled_select"]
    table = {}
    for n, p in pooled_tab.items():
        f = [out["seasons"][s][n] for s in sel]
        cur = [out["seasons"][s][CURRENT] for s in sel]
        ok = (p["min_coverage_of_main"] >= CONSTRAINTS["min_coverage_of_main"]
              and p["min_priced_share"] >= CONSTRAINTS["min_priced_share"]
              and p["min_avg_odds"] >= CONSTRAINTS["min_avg_odds"])
        table[n] = {"family": f[0]["family"], "pooled_roi": p["roi"], "worst_roi": p["worst_roi"],
                    "rois": [x["roi"] for x in f], "min_avg_odds": p["min_avg_odds"],
                    "min_priced_share": p["min_priced_share"], "min_coverage_of_main": p["min_coverage_of_main"],
                    "meets_constraints": bool(ok),
                    "better_than_current_both": bool(all(a["roi"] > b["roi"] for a, b in zip(f, cur))
                                                     and n != CURRENT),
                    "pooled_diff_ci95": p.get("diff_vs_current_ci95")}
    best_floor = {}
    for fam in ("floor", "calibrated", "calibrated_blend"):
        cands = [n for n, t in table.items() if t["family"] == fam and t["meets_constraints"]]
        if cands:
            best_floor[fam] = max(cands, key=lambda n: table[n]["pooled_roi"])
    eligible = [n for n, t in table.items() if t["meets_constraints"]]
    better = [n for n in eligible if table[n]["better_than_current_both"]]
    preferred = max(better, key=lambda n: (table[n]["pooled_roi"], table[n]["worst_roi"])) if better else CURRENT
    challengers = [n for n in eligible if n != CURRENT]
    challenger = max(challengers, key=lambda n: (table[n]["pooled_roi"], table[n]["worst_roi"]))
    return {"table": table, "constraints": CONSTRAINTS, "eligible": eligible, "better_than_current_both": better,
            "best_by_family": best_floor, "preferred": preferred, "challenger": challenger,
            "rule": "constraints on BOTH 2023/24 and 2024/25: a Risk for >= 99.5 % of the matches with a Main, "
                    ">= 85 % of the settled Risk picks priced, average odds of the bets >= 2.80; preferred = "
                    "the highest pooled 2023+2024 ROI among the rules whose ROI is higher than the current "
                    "rule's on BOTH seasons (ties: the higher worst-season ROI), otherwise the current rule; "
                    "challenger = the best eligible non-current rule by pooled ROI, tested on 2026/27 in any case"}


def stage_analysis() -> dict:
    t0 = time.time()
    out = {"seasons": {}}
    frames = []
    for s in (*SELECT, INFO):
        df, _ = season_frame(s)
        frames.append(df)
        out["seasons"][str(s)] = season_diagnostics(df)
    pooled_df = pd.concat(frames[:2], ignore_index=True)
    out["pooled_select"] = season_diagnostics(pooled_df, mc=False)
    out["variance"] = A.variance_table()
    out["minutes"] = (time.time() - t0) / 60
    save("analysis.json", out)
    return out


def season_diagnostics(df: pd.DataFrame, mc: bool = True) -> dict:
    d = {"calibration_rare": A.calibration_rare(df), "favourite_longshot": A.favourite_longshot(df),
         "ev_bins_model": A.ev_bins(df, "ev"), "ev_bins_platt": A.ev_bins(df, "ev_p_cal"),
         "ev_bins_blend": A.ev_bins(df, "ev_p_blend"), "who_is_right": A.who_is_right(df)}
    if mc:
        d["max_selection_mc"] = A.max_selection_mc(df)
    return d


def stage_preregister(force: bool = False) -> dict:
    path = RESULTS / "preregistration.json"
    if path.exists() and not force:
        log.info("pre-registration exists - not overwritten (use --force-preregister to replace it)")
        return json.loads(path.read_text())
    hist = load("history.json")
    if hist is None:
        raise SystemExit("run --history first")
    sel = hist["selection"]
    pred = Path(ROOT / "local_data" / "predictions")
    pre = {
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "code_commit": git("rev-parse", "HEAD"),
        "chosen_on": ["2023/24", "2024/25"], "information_only": ["2025/26"],
        "independent_check": "2026/27 production history: unified_history(store.load_picks()) season 2026 and "
                             "store.load_markets() (pre-kick-off probabilities, recorded / archived real prices); "
                             "Main of each match = the stored Main; calibrators fitted on 2025/26 walk-forward "
                             "predictions; matches settled in the data snapshot below",
        "data_snapshot": {"local_data_commit": git("log", "-1", "--format=%H", "--", "local_data/predictions"),
                          "sha256": {f.name: sha256(f) for f in sorted(pred.glob("*.parquet"))}},
        "selection_rule": sel["rule"], "constraints": sel["constraints"],
        "preferred_strategy": sel["preferred"], "challenger_strategy": sel["challenger"],
        "best_by_family": sel["best_by_family"],
        "frozen_parameters": {
            "risk_min_probability": S.RISK_MIN, "current_preferred_odds": list(S.PREFERRED),
            "softness": S.SOFTNESS, "probability_power": S.POWER, "ev_floors": S.FLOORS,
            "conservative_window": list(S.CONSERVATIVE), "high_odds_window": list(S.HIGH_ODDS),
            "odds_bands": {k: list(v) for k, v in S.BANDS.items()},
            "calibration": "Platt per selection type on the previous season, p in [0.05, 0.75], >= 300 rows; "
                           "blend: logistic on logit(p_model), logit(p_market); reliability test z >= -1.96, n >= 100"},
        "strategies_evaluated_on_2026": list(S.catalogue()) + ["odds band x EV floor grid (25 cells)"],
        "hypotheses": {
            "H1": "No Risk strategy has a positive ROI on 2026/27 (expected: all negative).",
            "H2": "The preferred / challenger strategy has a higher ROI than the current rule on 2026/27.",
            "H3": "In p 0.12-0.40 the model over-estimates rare outcomes (observed < mean p) on 2026/27.",
            "H4": "Among priced Risk candidates realised ROI rises with model EV with a slope well below 1.",
        },
        "decision_criteria_for_a_future_production_change": {
            "C1": "challenger (or preferred, if not the current rule) ROI on 2026/27 > current ROI on 2026/27",
            "C2": "paired match-bootstrap 95 % CI of the ROI difference on 2026/27 entirely above 0",
            "C3": "ROI difference > 0 in each of 2023/24, 2024/25, 2025/26 and 2026/27",
            "rule": "replace the current Risk only if C1, C2 and C3 all hold; otherwise keep it and re-test when "
                    "the 2026/27 history has at least twice as many settled priced Risk bets",
        },
        "pre_registration_values_on_history": {
            n: sel["table"][n] for n in {sel["preferred"], sel["challenger"], CURRENT}},
    }
    save("preregistration.json", pre)
    return pre


def prereg_commit() -> str:
    return git("log", "-1", "--format=%H", "--", "experiments/v6_1/risk/results/preregistration.json")


def stage_independent() -> dict:
    pre = load("preregistration.json")
    commit = prereg_commit()
    if pre is None or not commit:
        raise SystemExit("results/preregistration.json must exist and be committed before the 2026/27 check")
    if git("status", "--porcelain", "--", "experiments/v6_1/risk/results/preregistration.json"):
        raise SystemExit("results/preregistration.json differs from its committed version")
    t0 = time.time()
    df, cal = season_frame(INDEPENDENT)
    _, meta = D.independent_season()
    res, bands, boots, picks = evaluate_season(INDEPENDENT, df)
    store_picks(INDEPENDENT, picks)
    stored = stored_history()
    hist = load("history.json")
    out = {"preregistration_commit": commit, "preregistered": {k: pre[k] for k in
                                                                ("preferred_strategy", "challenger_strategy")},
           "data": meta, "check_current": S.check_current(df), "stored_production_risk": stored,
           "current_equals_stored_risk_share": float(picks[CURRENT].merge(
               df.loc[df["stored_risk"], ["match_id", "market", "selection"]], on=["match_id", "market", "selection"]
           ).shape[0] / max(len(picks[CURRENT]), 1)),
           "strategies": res, "bands": bands, "calibrator": cal.to_dict(),
           "diagnostics": season_diagnostics(df), "by_source": by_source(picks, df)}
    out["criteria"] = criteria(pre, hist, res)
    out["minutes"] = (time.time() - t0) / 60
    save("independent.json", out)
    pd.DataFrame(summary_rows({str(INDEPENDENT): res})).to_csv(RESULTS / "strategies_2026.csv", index=False)
    return out


def stored_history() -> dict:
    """The Risk history as the website shows it (unified_history), season 2026/27."""
    from src.analytics.performance import summarize, unified_history
    from src.prediction import store
    u = unified_history(store.load_picks())
    r = u.loc[(u["role"] == "risk") & (u["season"] == INDEPENDENT)]
    s = summarize(r)
    bets = r.loc[r["status"].isin(["WON", "LOST"]) & r["odds"].notna() & r["profit"].notna()]
    s.update({"priced_share": len(bets) / max(int(r["status"].isin(["WON", "LOST"]).sum()), 1),
              "avg_ev": float(bets["ev"].mean()), "first_date": str(r["date"].min().date()),
              "last_settled_date": str(r.loc[r["status"].isin(["WON", "LOST"]), "date"].max().date()),
              "by_source": {k: {"bets": int(len(g)), "roi": float(g["profit"].mean())}
                            for k, g in bets.groupby("source")},
              "by_provider": {k: {"bets": int(len(g)), "roi": float(g["profit"].mean())}
                              for k, g in bets.groupby("provider")}})
    return s


def by_source(picks: dict, df: pd.DataFrame) -> dict:
    src = df.drop_duplicates("match_id").set_index("match_id")["source"]
    out = {}
    for n, p in picks.items():
        b = p.loc[p["decided"] & p["priced"]].assign(source=lambda x: x["match_id"].map(src))
        out[n] = {k: {"bets": int(len(g)), "roi": float(g["profit"].mean())} for k, g in b.groupby("source")}
    return out


def criteria(pre: dict, hist: dict, res: dict) -> dict:
    out = {}
    for role in ("preferred_strategy", "challenger_strategy"):
        n = pre[role]
        if n == CURRENT:
            out[role] = {"strategy": n, "note": "the current rule itself"}
            continue
        diffs = [hist["seasons"][s][n]["roi"] - hist["seasons"][s][CURRENT]["roi"] for s in ("2023", "2024", "2025")]
        d26 = res[n]["roi"] - res[CURRENT]["roi"]
        ci = res[n].get("diff_vs_current_ci95")
        out[role] = {"strategy": n, "roi_2026": res[n]["roi"], "current_roi_2026": res[CURRENT]["roi"],
                     "diff_2026": d26, "diff_ci95_2026": ci, "diffs_history": diffs,
                     "C1": bool(d26 > 0), "C2": bool(ci is not None and ci[0] > 0),
                     "C3": bool(all(x > 0 for x in diffs) and d26 > 0)}
        out[role]["change_recommended"] = bool(out[role]["C1"] and out[role]["C2"] and out[role]["C3"])
    return out


def stage_info(timings: dict) -> None:
    import catboost, lightgbm, reportlab, scipy, sklearn, statsmodels, xgboost  # noqa: E401
    info = load("run_info.json") or {}
    info.update({"updated_at": datetime.now().isoformat(timespec="seconds"), "git_head": git("rev-parse", "HEAD"),
                 "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "python": platform.python_version(),
                 "platform": platform.platform(),
                 "packages": {m.__name__: m.__version__ for m in (np, pd, xgboost, sklearn, scipy, statsmodels,
                                                                  reportlab, lightgbm, catboost)},
                 "production_model_version": C.prod_meta().get("version", None) or
                 __import__("src.modelling.registry", fromlist=["active_version"]).active_version(),
                 "xgboost_threads": 3, "bootstrap_replicates": 2000,
                 "risk_config": {"RISK_MIN_PROBABILITY": config.RISK_MIN_PROBABILITY,
                                 "RISK_PREFERRED_ODDS": list(config.RISK_PREFERRED_ODDS),
                                 "RISK_ODDS_SOFTNESS": config.RISK_ODDS_SOFTNESS,
                                 "RISK_EV_WEIGHT": config.RISK_EV_WEIGHT,
                                 "RISK_UNPRICED_FACTOR": config.RISK_UNPRICED_FACTOR,
                                 "RISK_PROBABILITY_POWER": config.RISK_PROBABILITY_POWER}})
    info.setdefault("stage_minutes", {}).update(timings)
    blocks = {}
    for s in D.HISTORY:
        f = CACHE / f"block_{s}.json"
        if f.exists():
            j = json.loads(f.read_text())
            blocks[str(s)] = {"train_rows": j["train_rows"], "fit_seconds": j["seconds"],
                              "wall_seconds": j.get("wall_seconds")}
    info["model_blocks"] = blocks
    save("run_info.json", info)


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ("history", "analysis", "preregister", "independent", "figures", "all", "refresh",
                 "force_preregister"):
        ap.add_argument(f"--{flag.replace('_', '-')}", action="store_true")
    a = ap.parse_args()
    timings = {}

    def timed(name, fn, *args):
        t = time.time()
        fn(*args)
        timings[name] = (time.time() - t) / 60

    if a.refresh:
        timed("rebuild_cache", rebuild_cache)
    if a.all or a.history:
        timed("history", stage_history)
    if a.all or a.analysis:
        timed("analysis", stage_analysis)
    if a.all or a.preregister or a.force_preregister:
        timed("preregister", stage_preregister, a.force_preregister)
    if a.all or a.independent:
        timed("independent", stage_independent)
    if a.all or a.figures:
        from . import figures
        timed("figures", figures.build)
    stage_info(timings)


if __name__ == "__main__":
    main()
