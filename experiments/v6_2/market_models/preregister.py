"""Writes results/preregistration.json for the B variants of the market-models study (frozen variant and forward
criterion). Commit it before any match of the forward window is evaluated; production is not changed."""
from __future__ import annotations

import hashlib
import time

from src import config

from . import common as C
from . import direct as DR


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    C.setup_logging()
    dec = C.load_json("decisions.json")
    comp = C.load_json("comparisons.json")["comparisons"]
    mr = C.load_json("main_risk.json")
    st = C.load_json("stacking.json")
    bs = [k for k, d in dec["decisions"].items() if d["decision"] == "B"]
    if not bs:
        C.log.info("no B variant: nothing to preregister")
        return
    _, info = DR.fold(C.CURRENT)
    hyp = {}
    for i, key in enumerate(sorted(bs, key=lambda k: comp[k]["select"]["group"]["diff"]), 1):
        g, v = key.split("|")
        sub = next((k for k, s in mr["substitutions"].items() if s == {g: v}), None)
        hyp[f"H{i}_{g}_{v}"] = {
            "group": g, "variant": v,
            "frozen_models": {"fold": "2026/27 (fitted on 2021/22-2025/26)", "specs": DR.SPECS,
                              "trees": {k: x["trees"] for k, x in info["models"].items()}},
            "frozen_stacking_weights": st[str(C.CURRENT)] if v.startswith("S_") else None,
            "evidence_so_far": {"by_season": {s: comp[key][s]["group"] for s in ("2023", "2024", "2025", "2026")},
                                "select": comp[key]["select"]["group"],
                                "main_roi_select": mr["variants"].get(sub, {}).get("select", {}).get("main_vs_cur")},
            "forward_criterion": (
                "On matches played after the commit of this file, with the frozen fold-2026/27 models and weights, "
                "the paired bootstrap 95 % CI (by match, 2000 resamples, seed 20261109) of the group's mean log loss "
                "(variant - production) lies below 0, evaluated once at >= 3000 settled matches (or at the end of "
                "season 2026/27 if fewer: then 'inconclusive' unless the CI is below 0); AND the paired 95 % CI of "
                "Main ROI difference at real pre-match prices does not lie entirely below 0."),
            "decision_now": "B. A only if the forward criterion holds."}
    doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_commit_at_writing": C.git_sha(),
           "study": "v6.2 Market-Specific Models",
           "status_of_seasons": {"selection": ["2023/24", "2024/25"], "information": ["2025/26"],
                                 "current_already_looked_at": "2026/27 up to the data snapshot below (NOT a holdout)"},
           "data_snapshot": {"features.parquet": _sha(config.PROCESSED_DIR / "features.parquet"),
                             "data_version": C.data_version()},
           "decision_rule": dec["rule"], "hypotheses": hyp,
           "multiplicity": f"{len(hyp)} forward hypotheses; Holm correction over them."}
    C.save_json("preregistration.json", doc)


if __name__ == "__main__":
    main()
