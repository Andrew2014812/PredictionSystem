"""Stage 6 of v6.3: the list of combinations is fixed HERE, before any combination is evaluated.

    .venv/bin/python -m experiments.v6_3.summary.plan        # writes results/combinations_plan.json (once)

The file is written only if it does not exist yet (``--force`` overwrites it and records that it did): the plan
must be frozen before ``combine.py`` looks at a single combination result.

Building blocks (all of them are preregistered B candidates of the 13 studies that act on the Main choice):

* rule level (``experiments/v6_3/main_selection/rules.py``):
  REL   — MARKET_RELIABILITY['TOTAL'] 1.00 -> 0.95 (main_selection bt:reliability). It and the market-profitability
          candidate MARKET_RELIABILITY['HANDICAP'] 0.90 -> 1.035 move the Main in the same direction (from totals to
          Asian handicaps); only one may be combined. TOTAL 0.95 is taken because it is the smaller change of the
          production rule (it changes 17 % of the Main against 48 % for HANDICAP 1.035 — a property of the rule,
          not of its ROI) and it was chosen on 2025/26 by the study that owns the reliability grid.
  EVW   — MAIN_EV_WEIGHT 0.5 -> 1.0 (main_selection bt:ev_weight).
  FLOOR — real price >= 1.40 required, no eligible selection -> the statistical Main, shown, NOT a bet
          (main_selection floor_1.40).
* probability level (the rule then runs on the new probability: band, EV and score, as in the owning studies):
  CAL   — temperature calibration per market group, maps fitted on 2025/26 + 2026/27 months before the match
          (calibration ``temperature|group|pooled``, cache ``calibration/cache/calibrated_2026.parquet``).
  V6    — fixed 50/50 logit blend of the model with the single-book market probability (market_aware V6).
  C2    — walk-forward blend of the model with the multi-book consensus (consensus C2).
* model level (cached per-match outputs -> production engine ``engine.batch_from_outputs``):
  GOALS   — one team-goals XGBoost model (goals ``stacked``; λ home / away from ``goals/cache/stacked_2026``).
  CORNERS — Poisson GLM mean + negative binomial with r by league (corners ``best``; line probabilities from
            ``corners/results/corner_predictions_2026.csv``).

Not combined (stated in the plan): the alternative ML ensemble ens_xgb_best (its home-goals λ would replace the
same λ as GOALS; at most two model-level candidates are combined), squad PRE (covers 873 of 2 250 played matches
— 11 of 21 leagues), quality mkt_both / mkt_uniform (a 1X2 market blend that overlaps V6 / C2), recent weighting
(monthly refits; would need refitted outputs of every task), meta-selection meta_lr_ev (replaces the rule itself,
so it cannot be combined with rule parameters), timing (no B rule). Their combination is left to the forward check.
"""
from __future__ import annotations

import argparse

import pandas as pd

from experiments.v6_3 import common as C

DIRS = C.study_dirs("summary")
PLAN = DIRS["results"] / "combinations_plan.json"

BLOCKS = {
    "REL": {"level": "rule", "source": "main_selection bt:reliability (v63_main_selection:bt:reliability)",
            "change": "src/config.py MARKET_RELIABILITY['TOTAL'] 1.00 -> 0.95"},
    "EVW": {"level": "rule", "source": "main_selection bt:ev_weight (v63_main_selection:bt:ev_weight)",
            "change": "src/config.py MAIN_EV_WEIGHT 0.5 -> 1.0"},
    "FLOOR": {"level": "rule", "source": "main_selection floor_1.40 (v63_main_selection:floor_1.40)",
              "change": "real price >= 1.40 required; otherwise the statistical Main, not a bet"},
    "CAL": {"level": "probability", "source": "calibration temperature|group|pooled "
                                              "(v63_calibration_temperature_group_pooled@v1)",
            "change": "p' = sigma(logit(pc) / T_group) (1 - P(push)), monthly maps before the match"},
    "V6": {"level": "probability", "source": "market_aware V6 blend_5050 (v63-market-aware-V6)",
           "change": "p = softmax(0.5 log q_single_book + 0.5 log p_model) per market"},
    "C2": {"level": "probability", "source": "consensus C2 cons_blend_wf (v63-consensus-C2)",
           "change": "p = softmax(a + b_q log q_consensus + b_p log p_model), monthly walk-forward fit"},
    "GOALS": {"level": "model", "source": "goals stacked (preregistration 'Единая модель голов команды')",
              "change": "λ home / away of one team-goals model"},
    "CORNERS": {"level": "model", "source": "corners best (Poisson GLM + NB r by league)",
                "change": "corner line probabilities of the GLM mean with NB r by league"},
}

COMBINATIONS = [
    {"id": "K01", "parts": ["REL", "EVW"]},
    {"id": "K02", "parts": ["REL", "FLOOR"]},
    {"id": "K03", "parts": ["EVW", "FLOOR"]},
    {"id": "K04", "parts": ["CAL", "REL"]},
    {"id": "K05", "parts": ["CAL", "EVW"]},
    {"id": "K06", "parts": ["V6", "REL"]},
    {"id": "K07", "parts": ["V6", "EVW"]},
    {"id": "K08", "parts": ["C2", "REL"]},
    {"id": "K09", "parts": ["C2", "EVW"]},
    {"id": "K10", "parts": ["V6", "REL", "EVW"]},
    {"id": "K11", "parts": ["GOALS", "CORNERS"]},
    {"id": "K12", "parts": ["GOALS", "CORNERS", "REL", "EVW"]},
]

REFERENCES = [
    {"id": "S_" + b, "parts": [b], "role": "single part (reproduces the owning study; needed to judge whether a "
                                            "combination keeps the gains of its parts)"} for b in BLOCKS
] + [{"id": "R_combo_bt", "parts": ["REL", "EVW", "FLOOR"],
      "role": "replication of main_selection combo_bt (already reported by that study; not in the Holm family)"}]

PROTOCOL = {
    "universes": ["close (primary)", "t09 (robustness)"],
    "baseline": "Baseline B of the same universe (experiments/v6_3/replay/cache/baseline_b_*.parquet), paired "
                "bootstrap by match (common.paired_roi_diff, 2000 resamples, seed 20261118); Baseline A shown",
    "family": "Holm over the 12 combinations, separately in each universe (as the studies did); a joint Holm over "
              "all 24 combination x universe tests is reported as well",
    "keeps_gains": "a combination keeps the gains of its parts if, in BOTH universes, its ROI difference to "
                   "Baseline B is at least the largest difference of its single parts (point estimates); "
                   "additivity = at least the sum of the parts' differences; the paired difference combination - "
                   "best part (bootstrap) is reported",
    "probability_level_rule": "the rule runs on the new probability (band 0.40-0.85, EV and score), as in the "
                              "calibration / market_aware / consensus studies; rows without a new probability keep "
                              "the model's",
    "best_candidate_criterion": "for the table 'Current Production vs Best Validated Experimental Candidate' the "
                                "candidate is chosen among the preregistered B candidates of the studies and the 12 "
                                "combinations by EVIDENCE, not by ROI: (1) ROI difference to Baseline B > 0 in both "
                                "universes; (2) the smallest Holm-corrected p at close (Holm within the candidate's "
                                "own family); (3) ties -> the smallest max(raw p close, raw p t09). If no candidate "
                                "passes (1) the table says so and shows the candidate with the smallest Holm p.",
    "model_metrics": "1X2 log loss, goals Poisson deviance / exact-score log loss, corners MAE / deviance from the "
                     "owning studies' result files; production values from the same files",
}


def main(force: bool = False) -> None:
    if PLAN.exists() and not force:
        print(f"plan already frozen: {PLAN}")
        return
    plan = {"frozen_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "git_sha": C.git_sha(),
            "overwritten": bool(force and PLAN.exists()), "blocks": BLOCKS, "combinations": COMBINATIONS,
            "n_combinations": len(COMBINATIONS), "references": REFERENCES, "protocol": PROTOCOL,
            "not_combined": {
                "alt_ml ens_xgb_best / best_per_task": "home-goals λ overlaps GOALS; at most two model-level parts",
                "squad PRE": "covers 873 of 2 250 played matches (11 of 21 leagues); a combination would be "
                             "evaluated on a subset only",
                "quality mkt_both / mkt_uniform / mkt_season_0.75": "a 1X2 market blend overlapping V6 / C2",
                "recent_weighting wf:cur2 / wf:hl0": "monthly refits of every task; outputs of a joint scheme "
                                                     "are not cached",
                "meta_selection meta_lr_ev": "replaces the Main rule; rule parameters do not apply to it",
                "timing": "no B rule (CLV monitoring only)"}}
    C.write_json(PLAN, plan)
    print(f"written {PLAN}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
