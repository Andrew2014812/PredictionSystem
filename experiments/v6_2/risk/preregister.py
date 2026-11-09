"""Writes results/preregistration.json of the Risk study (frozen rules and forward tests). Commit it before any
match of the forward window is played; production Risk is not changed."""
from __future__ import annotations

import hashlib
import time

from src import config

from . import common as C
from . import strategies as S


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    C.setup_logging()
    res = C.load_json("risk.json")
    sel = C.load_json("selection.json")
    st = res["strategies"]
    rule = "v6.1: стек с рынком, пол EV ≥ 0"
    d_each = {s: st[s][rule]["d_roi"] for s in ("2023", "2024", "2025", "2026")}
    post = {s: res["post"][s]["рынок (Шин→пропорц.)"]["gap"] for s in ("2023", "2024", "2025", "2026")}
    doc = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit_at_writing": C.git_sha(),
        "study": "v6.2 Risk Selection 2.0",
        "status_of_seasons": {"selection": ["2023/24", "2024/25"], "information": ["2025/26"],
                              "current_already_looked_at": "2026/27 up to the data snapshot below (NOT a holdout)"},
        "data_snapshot": {"markets.parquet": _sha(config.PREDICTIONS_DIR / "markets.parquet"),
                          "picks.parquet": _sha(config.PREDICTIONS_DIR / "picks.parquet"),
                          "data_version": C.data_version()},
        "selection_by_frozen_rule": {"rule": sel["rule"], "chosen": sel["chosen"], "promising": sel.get("promising")},
        "hypotheses": {
            "H1_rule": {
                "name": rule,
                "evidence_so_far": {"d_roi_vs_current_by_season": d_each,
                                    "pooled_2023_2024": res["pooled"]["rules"][rule]},
                "rule": "Production Risk pool and score (src/markets/recommendation.risk_prediction) unchanged, with one "
                        "extra first tier: among pool candidates with a real price whose EV at the stacked probability "
                        "p_s >= 0, the best production score computed with p_s; otherwise the production Risk. "
                        "p_s = sigmoid(b1_k logit(pc) + b2_k logit(qc) + b0_k) (1 - p_refund), pc = p / (1 - p_refund), "
                        "qc = proportional de-vigged price of a full book (Asian whole line: P(win | no refund)); "
                        "coefficients per selection key k (pooled key '*' below 300 rows) fitted by logistic regression "
                        "(C = 1e4) on the decided pool rows of 2022/23-2025/26 walk-forward predictions — the 2026/27 "
                        "fit of experiments/v6_2/risk (frozen numbers below).",
                "frozen_parameters": res["diagnostics"]["2026"]["fitted"]["stack"],
                "forward_criterion": "On Risk bets of matches played after the commit of this file, the paired bootstrap "
                                     "95 % CI (by match, 2000 resamples, seed 20261109) of ROI(H1) - ROI(production Risk) "
                                     "lies above 0, evaluated once at >= 2000 settled matches with both picks priced "
                                     "(or at the end of season 2026/27 if fewer: then the result is reported as "
                                     "inconclusive unless the CI is above 0).",
                "decision": "B now. A only if the forward criterion holds; the rule still loses money in every season "
                            "(it reduces the loss), so adoption would change WHICH Risk is shown, not make Risk a bet."},
            "H2_display": {
                "name": "Вероятность Risk = вероятность рынка без маржи",
                "evidence_so_far": {"gap_market_probability_by_season": post,
                                    "gap_model_probability_by_season": {s: res["post"][s]["сырая p"]["gap"]
                                                                         for s in ("2023", "2024", "2025", "2026")}},
                "rule": "For the shown Risk with a full real book, the stated probability is the proportional de-vigged "
                        "market probability of a win (Asian whole line: P(win | no refund) on the decided scale); the "
                        "pick is unchanged.",
                "forward_criterion": "On the shown Risk of matches played after the commit (decided, full book), the 95 % "
                                     "bootstrap CI of mean(stated) - hit rate contains 0 AND the paired CI of |gap(market)| - "
                                     "|gap(model)| lies below 0; evaluated at >= 1000 settled Risk picks.",
                "decision": "B now; A (display only) if the forward criterion holds."},
        },
        "multiplicity": "Two forward hypotheses; Holm correction over H1 and H2.",
        "risk_band_and_parameters": {"band": S.BAND, "z_lcb": S.Z_LCB},
    }
    C.save_json("preregistration.json", doc)


if __name__ == "__main__":
    main()
