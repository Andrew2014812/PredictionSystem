"""Pre-registration of the Main-strategy research (results/preregistration.json).

Stage "criteria" is written before any alternative strategy is evaluated; stage "strategy" is written after the
selection on the folds 2023/24 + 2024/25 and BEFORE any alternative strategy is evaluated on 2026/27. Both are
committed to git; the 2026/27 evaluation (independent.py) refuses to run without the "strategy" stage and records
the commit time of the file it used.

    python -m experiments.v6_1.main.preregister --criteria
    python -m experiments.v6_1.main.preregister --strategy       # reads results/selection.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from . import common as C

PATH = C.RESULTS / "preregistration.json"

CRITERIA = {
    "selection_folds": ["2023/24", "2024/25"],
    "information_only": "2025/26 (looked at in earlier development; never used to choose)",
    "independent_check": "2026/27 so far: production history store (unified_history + markets.parquet), real recorded "
                         "or archived pre-kick-off prices; evaluated once, after this file's 'strategy' stage is committed",
    "universe": "identical matches for every strategy: matches with real Bet365 opening prices for 1X2 and total 2.5 "
                "(Football-Data); ROI = mean profit per unit stake at the real price; refunds and unpriced picks are "
                "not bets",
    "variant_choice": "among variants of a family, the highest WORST-fold ROI over 2023/24 and 2024/25 (ties: pooled ROI); "
                      "a variant with fewer than 300 bets on either fold is not eligible",
    "adoption_rule": [
        "1. ROI higher than strategy A (current Main) on BOTH 2023/24 and 2024/25 on identical matches",
        "2. pooled 2023/24+2024/25 paired bootstrap (resampling matches, 2000 draws): 5th percentile of "
        "ROI(strategy) - ROI(A) > 0 (one-sided 95 %)",
        "3. calibration of the shown Main not worse: ECE of the Main picks not higher than A's by more than 0.01 and "
        "Brier not higher by more than 0.002 on both folds (a pure betting layer leaves the shown Main unchanged)",
        "4. 2026/27 confirmation within reason: ROI(strategy) - ROI(A) >= 0 on identical 2026/27 matches (point "
        "estimate; the 2026/27 sample is too small to require significance) and the strategy's 2026/27 ROI interval "
        "reported",
        "5. a positive ROI is claimed only if its 95 % bootstrap interval excludes 0; otherwise reported as "
        "'not distinguishable from 0'",
    ],
    "never": ["choose on 2025/26 or 2026/27", "use prices recorded after kick-off", "change stored predictions"],
}


def write(stage: str, payload: dict) -> dict:
    doc = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else {}
    if stage in doc:
        raise SystemExit(f"stage '{stage}' already pre-registered at {doc[stage]['timestamp_utc']}; not overwritten")
    doc[stage] = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "git_sha_before": C.git_sha(), **payload}
    PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", action="store_true")
    ap.add_argument("--strategy", action="store_true")
    a = ap.parse_args()
    if a.criteria:
        write("criteria", CRITERIA)
    if a.strategy:
        sel = C.load_json("selection.json")
        if not sel:
            raise SystemExit("results/selection.json missing: run the fold selection first")
        write("strategy", {"chosen": sel["preregistered"],
                           "betting_layer_note": "if the primary belongs to family G (the shown Main stays A), adoption "
                           "means an opt-in stake/skip decision next to the unchanged Main (disabled by default, the "
                           "interface is outside this study); the Main itself is not changed"})
    print(PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
