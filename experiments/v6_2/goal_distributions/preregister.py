"""Writes results/preregistration.json for the B variants of the goal-distribution study (frozen variant and
forward criterion). Commit it before any match of the forward window is evaluated; production is not changed."""
from __future__ import annotations

import hashlib
import time

from src import config

from . import common as C
from . import maher


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    C.setup_logging()
    dec = C.load_json("decisions.json")
    fits = C.load_json("fits.json")
    comp = C.load_json("comparisons.json")["comparisons"]
    mr = C.load_json("main_risk.json")["variants"]
    bs = [v for v, d in dec["decisions"].items() if d["decision"] == "B"]
    if not bs:
        C.log.info("no B variant: nothing to preregister")
        return
    hyp = {}
    for i, v in enumerate(sorted(bs, key=lambda x: comp[x]["select"]["SCORE"]["diff"]), 1):
        rk = v if v.endswith("+R") else v + "+R"
        hyp[f"H{i}_{v}"] = {
            "variant": v, "name": fits["names"].get(v.replace("+R", ""), v),
            "frozen_parameters": fits["fits"][str(C.CURRENT)][v],
            "frozen_parameters_note": "fitted on 2025/26 (fold-2025 predictions) — applied unchanged to every forward "
                                      "match; production goal means / 1X2 probabilities of the active model.",
            "maher_spec": {"half_life_days": maher.HALF_LIFE, "window_days": maher.WINDOW, "alpha": maher.ALPHA,
                           "refit": "weekly per league, matches before the Monday of the match week"},
            "evidence_so_far": {"score_ll_diff_by_season": {s: comp[v][s]["SCORE"] for s in ("2023", "2024", "2025",
                                                                                              "2026")},
                                "score_ll_diff_select": comp[v]["select"]["SCORE"],
                                "main_roi_diff_select": mr.get(rk, {}).get("select", {}).get("main_vs_ref")},
            "forward_criterion": (
                "On matches played after the commit of this file (settled, all leagues of the system), the paired "
                "bootstrap 95 % CI (by match, 2000 resamples, seed 20261109) of mean log loss of the exact score "
                "(variant - production P+R) lies below 0, evaluated once at >= 3000 settled matches (or at the end of "
                "season 2026/27 if fewer: then 'inconclusive' unless the CI is below 0); AND, as a guard, the paired "
                "95 % CI of Main ROI difference at real pre-match prices does not lie entirely below 0."),
            "decision_now": "B. A only if the forward criterion holds; a better probability alone does not justify a "
                            "change that lowers Main ROI.",
        }
    doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_commit_at_writing": C.git_sha(),
           "study": "v6.2 Goal Distribution Models",
           "status_of_seasons": {"selection": ["2023/24", "2024/25"], "information": ["2025/26"],
                                 "current_already_looked_at": "2026/27 up to the data snapshot below (NOT a holdout)"},
           "data_snapshot": {"features.parquet": _sha(config.PROCESSED_DIR / "features.parquet"),
                             "data_version": C.data_version()},
           "decision_rule": dec["rule"], "hypotheses": hyp,
           "multiplicity": f"{len(hyp)} forward hypotheses; Holm correction over them."}
    C.save_json("preregistration.json", doc)


if __name__ == "__main__":
    main()
