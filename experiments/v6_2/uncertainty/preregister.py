"""Preregistration of the B decisions of the uncertainty study (results/preregistration.json).

Written only when results/decisions.json has a B; with no B there is nothing to freeze and no file is written.

    python -m experiments.v6_2.uncertainty.preregister
"""
from __future__ import annotations

import time

from . import common as C
from .run import MEASURES, WARMUP


def run() -> dict | None:
    dec = C.load_json("decisions.json")
    pred = C.load_json("predictive.json")
    filt = C.load_json("filters.json")
    hyps = {}
    if dec["flag_on_site"]["decision"] == "B":
        for c in dec["flag_on_site"]["significant_measures"]:
            r = pred["pooled_select"][c]
            hyps[f"flag_{c}"] = {
                "name": f"Метка «ненадёжный прогноз»: {MEASURES[c]}",
                "rule": f"a Main is flagged when {c} exceeds the 0.8 quantile of {c} over the Mains of the same season "
                        f"played on earlier days (no flag during the first {WARMUP} Mains); the pick is unchanged",
                "evidence_selection_seasons": {"beta_u": r["beta_u"], "beta_u_ci95": r["beta_u_ci95"],
                                               "p_holm": r["p_holm"]},
                "forward_criterion": "on Mains of matches played after this file was committed: gap(flagged) - "
                                     "gap(unflagged) > 0 with the 95 % bootstrap CI (by match, 2000, seed 20261109) "
                                     "above 0, evaluated once at >= 2000 settled Mains",
            }
    for c in (dec.get("current_signal") or {}).get("measures", []):
        r = pred["per_season"][str(C.CURRENT)][c]
        sel = pred["pooled_select"][c]
        scale = pred["frozen_scale"][str(C.CURRENT)][c]
        hyps[f"flag_{c}"] = {
            "name": f"Метка «ненадёжный прогноз» Main: {MEASURES[c]}",
            "prior": "low: found on 2026/27 (already looked at); NOT supported on the selection seasons",
            "measure": c, "definition": MEASURES[c],
            "ensemble": "u_1x2 = mean over H, D, A of the standard deviation over the 8 'boot' members of "
                        "experiments/v6_2/uncertainty/ensemble.py (production 1X2 model refitted on a bootstrap "
                        "resample of all matches of seasons < 2026, seeds as in that file)" if c == "u_1x2" else None,
            "frozen_threshold": scale["q80"], "frozen_mean": scale["mean"], "frozen_sd": scale["sd"],
            "rule": f"a Main is flagged when {c} > frozen_threshold (the 0.8 quantile of {c} over the settled Mains "
                    "of 2026/27 up to this file); the pick and its probability are unchanged",
            "evidence": {"current_2026_27": {"beta_u": r["beta_u"], "beta_u_ci95": r["beta_u_ci95"], "p": r["p"],
                                             "p_holm_season": r["p_holm_season"], "n": r["n"]},
                         "selection_seasons": {"beta_u": sel["beta_u"], "beta_u_ci95": sel["beta_u_ci95"],
                                               "p_holm": sel["p_holm"]}},
            "forward_criterion": "on the shown Mains of matches played after this file was committed (status WON / "
                                 "LOST): (1) logistic regression won ~ logit(pc) + (u - frozen_mean) / frozen_sd, "
                                 "95 % bootstrap CI of beta_u (by match, 2000, seed 20261109) entirely below 0; "
                                 "(2) gap(flagged) - gap(not flagged) > 0 with its 95 % CI above 0; evaluated once at "
                                 ">= 2000 settled Mains (or at the end of 2026/27: inconclusive unless both hold)",
            "decision": "A (display only: a 'less reliable' label) if (1) and (2) hold; otherwise C",
        }
    if dec["filter_main"]["decision"] == "B":
        for k, v in filt["variants"].items():
            if v["promising_roi"] or v["promising_gap"]:
                hyps[f"filter_{k}"] = {
                    "name": f"Фильтр Main: {MEASURES[v['measure']]}, убрать {int(v['drop'] * 100)} %",
                    "rule": f"drop the Main when {v['measure']} exceeds the {1 - v['drop']:.2f} quantile over earlier "
                            f"Mains of the season (warm-up {WARMUP})",
                    "evidence": {"roi": v["pooled_select_roi"], "abs_gap": v["pooled_select_abs_gap"]},
                    "forward_criterion": "paired 95 % CI of ROI(kept) - ROI(all) above 0 at >= 2000 settled priced Mains",
                }
    if not hyps:
        C.log.info("no B decision: nothing to preregister")
        return None
    out = {"study": "experiments/v6_2/uncertainty", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "git_sha_of_code": C.git_sha(), "hypotheses": hyps,
           "multiplicity": "Holm over the hypotheses of this file",
           "shadow": "registered as uncertainty_flag_u_1x2 in src/research/registry.py, not implemented yet: it "
                     "needs the 8 bootstrap 1X2 models stored once (fitted on seasons < 2026); the daily cost is then "
                     "8 predictions per match"}
    C.save_json("preregistration.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("unc-prereg")
    print(run())
