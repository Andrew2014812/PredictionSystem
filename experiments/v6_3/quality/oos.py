"""Out-of-sample production-model outputs of the selection seasons 2024/25 and 2025/26 (heavy job).

For season s ∈ {2024, 2025} the four production tasks are refitted (production features, parameters, tree counts,
recency weights, corner leagues — ``engine.fit_task``) on every match played before season s and predict season s.
These predictions are the "earlier seasons" on which the mitigations of this study choose their parameters
(never on 2026/27). Output: ``cache/oos_<s>.parquet`` (match_id + p_home, p_draw, p_away, lam_home, lam_away,
corner_mu) and the fitted bundles ``cache/oos_<s>_bundles.pkl`` (feature-group permutation of groups.py).

    experiments/v6_3/heavy.sh 3600 experiments/v6_3/quality/logs/oos.log \
        .venv/bin/python -m experiments.v6_3.quality.oos
"""
from __future__ import annotations

import time

import joblib

from experiments.v6_3 import common as C
from experiments.v6_3.replay import data as RD
from experiments.v6_3.replay import engine as EN

D = C.study_dirs("quality")
SEASONS = (2024, 2025)


def main() -> None:
    log = C.setup_logging("quality.oos")
    f = RD.load_features()
    for s in SEASONS:
        path = D["cache"] / f"oos_{s}.parquet"
        bpath = D["cache"] / f"oos_{s}_bundles.pkl"
        if path.exists() and bpath.exists():
            continue
        t0 = time.time()
        train = f.loc[(f["season"] < s) & f["played"].astype(bool)]
        target = f.loc[f["season"] == s]
        bundles = {k: EN.fit_task(k, train) for k in ("result", "home_goals", "away_goals", "corners")}
        out = EN.outputs_from_bundles(bundles, target)
        out.to_parquet(path)
        joblib.dump(bundles, bpath)
        log.info("season %s: trained on %d rows, predicted %d matches in %.0fs", s, len(train), len(target),
                 time.time() - t0)


if __name__ == "__main__":
    main()
