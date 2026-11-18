"""Audit of FootPredict at the start of v6.3 (Prompt 9, §1): the facts every study starts from.

    python -m experiments.v6_3.audit.run      (after experiments.v6_3.replay.build)

Everything is measured from the data and code at the current commit, not copied from earlier reports.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from experiments.v6_3 import common as C
from src import config
from src.modelling.registry import active_version, version_metadata

DIRS = C.study_dirs("audit")


def model_metrics(m: pd.DataFrame, out: pd.DataFrame) -> dict:
    d = m.merge(out, on="match_id")
    d = d.loc[d["played"].astype(bool)]
    y = d["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    p = d[["p_home", "p_draw", "p_away"]].to_numpy()
    onehot = np.eye(3)[y]
    pred = p.argmax(1)
    res = {"n": int(len(d)), "accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
           "brier": float(((p - onehot) ** 2).sum(1).mean()),
           "mean_p": dict(zip("HDA", p.mean(0).round(4))), "freq": dict(zip("HDA", onehot.mean(0).round(4)))}
    mk = d[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    ok = np.isfinite(mk).all(1) & (mk > 1).all(1)
    q = 1 / mk[ok]
    q = q / q.sum(1, keepdims=True)
    res["bookmaker_log_loss_same_matches"] = float(log_loss(y[ok], q, labels=[0, 1, 2]))
    res["model_log_loss_same_matches"] = float(log_loss(y[ok], p[ok], labels=[0, 1, 2]))
    res["bookmaker_matches"] = int(ok.sum())
    goals = {}
    for side in ("home", "away"):
        e = d[f"lam_{side}"].to_numpy() - d[f"{side}_goals"].to_numpy()
        goals[side] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                       "mean_pred": float(d[f"lam_{side}"].mean()), "mean_true": float(d[f"{side}_goals"].mean())}
    c = d.loc[d["total_corners"].notna() & d["corner_mu"].notna()]
    e = c["corner_mu"] - c["total_corners"]
    var = c["total_corners"].var()
    corners = {"n": int(len(c)), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
               "r2": float(1 - (e ** 2).mean() / var), "mean_pred": float(c["corner_mu"].mean()),
               "mean_true": float(c["total_corners"].mean()),
               "mae_of_mean": float((c["total_corners"] - c["total_corners"].mean()).abs().mean())}
    return {"1x2": res, "goals": goals, "corners": corners}


def shadow_state() -> dict:
    try:
        from src.research.evaluate import evaluate_shadow
        rep = evaluate_shadow(write=False)
        return {k: {kk: v.get(kk) for kk in ("status", "n_settled", "n", "information_fraction") if kk in v}
                for k, v in rep.get("shadows", {}).items()} | {"_files": len(list((C.ROOT / "local_data/research/shadow_predictions").glob("*.parquet")))}
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def main() -> None:
    started = time.time()
    m = pd.read_parquet(C.SHARED_CACHE / "matches.parquet")
    out = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    base = C.read_json(C.SHARED_RESULTS / "baseline.json")
    meta = version_metadata(active_version())
    last_run = json.loads((C.ROOT / "local_data/api/diagnostics/last_run.json").read_text())
    waiting = pd.read_csv(C.ROOT / "local_data/api/diagnostics/waiting.csv")
    picks = pd.read_parquet(C.DATA_FILES["picks"], columns=["season", "role", "source", "status", "odds"])
    cur = picks.loc[picks["season"] == C.CURRENT]
    report = {
        "commit": C.git_sha(),
        "production_model": {"version": active_version(), "trained_to": meta.get("trained_to"),
                             "periods": meta.get("periods"),
                             "params": {k: v.get("params") for k, v in meta.get("tasks", {}).items()}},
        "season_2026_27": base["coverage"],
        "picks_by_source_role": cur.groupby(["source", "role"]).size().unstack(fill_value=0).to_dict(),
        "main_baseline_a": base["baseline_a"],
        "main_baseline_b": {u: base[u]["summary"] for u in ("t09", "close") if u in base},
        "model_metrics_2026_27": model_metrics(m, out),
        "main_rule": {k: getattr(config, k) for k in dir(config) if k.startswith(("MAIN_", "MARKET_RELIABILITY"))},
        "waiting_results": waiting.groupby("reason").size().to_dict(),
        "last_daily_run": {k: last_run.get(k) for k in list(last_run)[:12]},
        "shadows": shadow_state(),
    }
    C.write_json(DIRS["results"] / "audit.json", report)
    C.write_json(DIRS["results"] / "run_info.json", C.run_info("audit", started))
    print(json.dumps(C.to_jsonable({k: report[k] for k in ("commit", "season_2026_27", "model_metrics_2026_27")}),
                     indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
