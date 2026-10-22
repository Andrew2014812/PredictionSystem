"""Compare Main-prediction rules on identical matches and identical real prices.

For each season s in 2023/24 and 2024/25 the four models are fitted on the seasons before s (tuned
parameters of the active model) and season s is predicted with its real Bet365 prices (Football-Data).
Rules are chosen on these two seasons together; 2025/26 (the v1–v4 test season) is printed for
information only and never used to choose.

    python scripts/main_rule_comparison.py            # -> local_data/models/main_rule_comparison.json
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.store import load_features  # noqa: E402
from src.markets.odds import PricedSelection  # noqa: E402
from src.markets.recommendation import main_params, main_prediction  # noqa: E402
from src.modelling.pipeline import _subset  # noqa: E402
from src.modelling.registry import active_version, version_metadata  # noqa: E402
from src.modelling.tasks import TASKS  # noqa: E402
from src.modelling.training import TaskTrainer, corner_leagues, negative_binomial_size  # noqa: E402
from src.prediction.engine import predict_matches  # noqa: E402
from src.providers import real_odds_lookup  # noqa: E402
from src.storage import models_storage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_study import legacy_main, settle, summarise  # noqa: E402

log = logging.getLogger("main_rules")
SELECT = (2023, 2024)
REPORT = 2025


def season_markets(features: pd.DataFrame, meta: dict, season: int):
    before = sorted(x for x in features["season"].unique() if x < season)
    tr, ev = _subset(features, before), _subset(features, [season])
    cl = corner_leagues(tr)
    bundles = {}
    for name, task in TASKS.items():
        rows = tr.loc[tr["league"].isin(cl)] if name == "corners" else tr
        info = meta["tasks"][name]
        b = TaskTrainer(task, features.columns).fit(rows, info["params"], info["n_estimators"])
        if name == "corners":
            fit = rows.loc[task.trainable(rows)]
            b.extra.update({"leagues": cl, "nb_size": negative_binomial_size(fit[task.target].to_numpy(float),
                                                                             b.predict(fit))})
        if name == "result":
            b.extra["draw_multiplier"] = 1.0
        bundles[name] = b
    ev = ev.loc[ev["home_goals"].notna()]
    batch = predict_matches(ev, bundles, real_odds_lookup(ev), "study", "study")
    results = ev.set_index("match_id")[["home_goals", "away_goals", "total_corners", "date"]].to_dict("index")
    by_match: dict[str, list] = {}
    push = batch.markets["push_probability"].fillna(0).to_numpy()
    for r, pp in zip(batch.markets.itertuples(index=False), push):
        by_match.setdefault(r.match_id, []).append(PricedSelection(
            r.market, r.selection, float(r.probability), None if pd.isna(r.odds) else float(r.odds),
            push_probability=float(pp)))
    return by_match, results


def evaluate(by_match, results, choose) -> dict:
    picks = []
    for mid, sels in by_match.items():
        p = choose(sels)
        picks.append((settle(p, results[mid]), p.odds if p is not None and p.has_odds else None,
                      p.odds if p is not None and p.has_odds else (1 / p.probability if p else None)))
    return summarise(picks)


def rules() -> dict:
    base = main_params()
    out = {"v4 (hard band 1.30–3.00, EV ≥ 3 %)": legacy_main}
    for name, params in {
        "v5 current": base,
        "v5 + value first (EV ≥ 0)": base.with_(value_first_min_ev=0.0),
        "v5 + value first (EV ≥ 3 %)": base.with_(value_first_min_ev=0.03),
        "v5 soft 0.30 + value first (EV ≥ 3 %)": base.with_(softness=0.30, value_first_min_ev=0.03),
        "v5 soft 0.30, EV weight 0 + value first (EV ≥ 3 %)": base.with_(softness=0.30, ev_weight=0.0,
                                                                          value_first_min_ev=0.03),
    }.items():
        out[name] = (lambda p: (lambda s: (lambda r: r.pick if r else None)(main_prediction(s, p))))(params)
    return out


def main() -> None:
    features = load_features()
    meta = version_metadata(active_version())
    table = {}
    for season in (*SELECT, REPORT):
        by_match, results = season_markets(features, meta, season)
        for name, choose in rules().items():
            table.setdefault(name, {})[str(season)] = evaluate(by_match, results, choose)
            log.info("%s %s %s", season, name, table[name][str(season)])
    for name, seasons in table.items():
        rois = [seasons[str(s)]["roi"] for s in SELECT]
        bets = [seasons[str(s)]["bets"] for s in SELECT]
        seasons["selection_roi"] = float(np.average(rois, weights=bets))
        seasons["selection_worst_roi"] = float(min(rois))
    best = max(table, key=lambda n: (table[n]["selection_worst_roi"], table[n]["selection_roi"]))
    out = {"seasons_for_choice": [str(s) for s in SELECT], "reported_only": str(REPORT), "rules": table,
           "chosen": best}
    models_storage.write_json("main_rule_comparison.json", out)
    for name, seasons in table.items():
        cells = " | ".join(f"{s}: ROI {seasons[str(s)]['roi'] * 100:+.1f}% hit {seasons[str(s)]['hit_rate'] * 100:.1f}% "
                           f"priced {seasons[str(s)]['priced_share'] * 100:.0f}% odds {seasons[str(s)]['avg_odds']:.2f}"
                           for s in (*SELECT, REPORT))
        print(f"{'*' if name == best else ' '} {name:<52} {cells}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    main()
