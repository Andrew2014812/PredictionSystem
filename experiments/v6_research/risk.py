"""Q5 — Risk rule with a small negative-EV floor, on real Bet365 prices (Football-Data B365*).

For every match of a fold the production chain is reproduced: the production models fitted on
the seasons before the fold (base predictions), score matrix reconciled with the classifier,
every goal market + corner lines (NB size from the inner season), real prices attached with
src.markets.odds.price from Football-Data rows only (1X2, total 2.5, the one Asian line per match),
Main = production main_prediction(). Risk variants:

    current            production risk_prediction() (no EV requirement)
    floor F, mode M    1. priced candidates with EV >= 0 -> best rule score (soft window 3.0–6.0 kept)
                       2. else priced candidates with F <= EV < 0 ->
                            M = "least_negative": highest EV (ties: rule score)
                            M = "best_scored":    best rule score
                       3. else the current rule (best-scored alternative)
    F in {0, -3 %, -5 %, -8 %}.

Candidates are always the production ones (no European handicap, never the Main selection, never a
selection with the same footprint as Main or a bet against Main, p >= RISK_MIN_PROBABILITY).
Settlement / summary use scripts/decision_study.py (settle, summarise).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.markets.distributions import corners_distribution, reconcile_with_1x2, score_matrix
from src.markets.markets import corner_markets, goal_markets
from src.markets.odds import price
from src.markets.recommendation import (EXCLUDED_GROUPS, _choose, effective_odds, main_prediction,
                                        opposite_outcome, risk_params, risk_prediction, rule_score,
                                        same_outcome)
from src.modelling.training import negative_binomial_size
from src.providers.football_data import odds_from_matches
from src.providers.schema import odds_lookup

from . import common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from decision_study import settle, summarise  # noqa: E402

log = C.setup_logging("risk")
FLOORS = (0.0, -0.03, -0.05, -0.08)


def risk_with_floor(sels, main, floor: float, mode: str):
    params = risk_params()
    eligible = [s for s in sels if s.group not in EXCLUDED_GROUPS
                and not (main is not None and (same_outcome(s, main) or opposite_outcome(s, main)))]
    cand = [s for s in eligible if s.probability >= params.min_probability]
    pos = [s for s in cand if s.has_odds and s.ev >= 0]
    if pos:
        return _choose(pos, params, "risk"), "positive_ev"
    within = [s for s in cand if s.has_odds and floor <= s.ev < 0]
    if within:
        if mode == "least_negative":
            best = max(within, key=lambda s: (s.ev, rule_score(s, params)))
            from src.markets.recommendation import Recommendation
            return Recommendation("risk", "priced", best, rule_score(best, params)), "within_floor"
        return _choose(within, params, "risk"), "within_floor"
    return _choose(cand or eligible, params, "risk"), "fallback_current"


def variants():
    out = {"current (no EV rule)": None}
    for f in FLOORS:
        for mode in ("least_negative", "best_scored"):
            if f == 0.0 and mode == "best_scored":
                continue
            label = "EV >= 0 first, else current" if f == 0.0 else f"floor {f * 100:+.0f} % ({mode})"
            out[label] = (f, mode)
    return out


def fold_selections(s: int):
    inner, outer, info = C.base_predictions(s)
    ok = inner["cmu"].notna() & inner["total_corners"].notna()
    r = negative_binomial_size(inner.loc[ok, "total_corners"].to_numpy(float), inner.loc[ok, "cmu"].to_numpy())
    df = C.features()
    rows = C.seasons(df, [s])
    lookup = odds_lookup(odds_from_matches(rows))
    by_match, results = {}, {}
    for row in outer.itertuples():
        m = reconcile_with_1x2(score_matrix(row.lh, row.la), np.array([row.pH, row.pD, row.pA]))
        sels = goal_markets(m)
        if row.cmu == row.cmu:
            sels += corner_markets(corners_distribution(row.cmu, r))
        by_match[row.match_id] = price(sels, lookup.get(row.match_id))
        results[row.match_id] = {"home_goals": row.home_goals, "away_goals": row.away_goals,
                                 "total_corners": row.total_corners}
    return by_match, results, r


def evaluate(by_match, results, rule, seed=11):
    picks, evs, steps, mains, risks_for_main, same_as_main = [], [], {}, 0, 0, 0
    profit_rows = []
    t0 = time.time()
    for mid, sels in by_match.items():
        main = main_prediction(sels)
        mp = main.pick if main else None
        if rule is None:
            rec, step = risk_prediction(sels, mp), "current"
        else:
            rec, step = risk_with_floor(sels, mp, *rule)
        steps[step] = steps.get(step, 0) + 1
        mains += main is not None
        risks_for_main += (main is not None) and (rec is not None)
        if rec is None:
            continue
        p = rec.pick
        same_as_main += mp is not None and (p.market, p.selection) == (mp.market, mp.selection)
        status = settle(p, results[mid])
        picks.append((status, p.odds if p.has_odds else None, effective_odds(p)))
        if p.has_odds:
            evs.append(p.ev)
            if status in ("WON", "LOST"):
                profit_rows.append(p.odds - 1 if status == "WON" else -1.0)
    summ = summarise(picks)
    prof = np.array(profit_rows)
    rng = np.random.default_rng(seed)
    boots = prof[rng.integers(0, len(prof), (1000, len(prof)))].mean(1) if len(prof) else np.array([np.nan])
    n = len(by_match)
    return {**summ, "matches": n, "coverage": (len([p for p in picks]) / n) if n else None,
            "matches_with_main": mains, "risk_for_every_main": bool(risks_for_main == mains),
            "matches_with_main_but_no_risk": int(mains - risks_for_main), "risk_equal_to_main": int(same_as_main),
            "avg_ev_priced": float(np.mean(evs)) if evs else None,
            "roi_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "step_shares": {k: v / n for k, v in steps.items()}, "seconds": time.time() - t0}


def run() -> dict:
    t0 = time.time()
    out = {"folds": {}, "variants": list(variants())}
    for s in C.FOLDS:
        by_match, results, r = fold_selections(s)
        fold = {"nb_size_inner": r}
        for name, rule in variants().items():
            fold[name] = evaluate(by_match, results, rule)
            log.info("fold %s %-40s %s", s, name, {k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in fold[name].items() if k in
                                                   ("coverage", "priced_share", "hit_rate", "avg_odds", "roi",
                                                    "avg_ev_priced")})
        out["folds"][str(s)] = fold
    # choice on 2023 + 2024: Risk must stay risky (avg effective odds >= 3) and mostly priced (>= 40 %),
    # as in scripts/decision_study.choose_risk_rule; then the highest worst-fold ROI, then pooled ROI
    table = {}
    for name in variants():
        f = [out["folds"][str(s)][name] for s in C.SELECT]
        bets = [x["bets"] for x in f]
        table[name] = {"worst_roi": min(x["roi"] for x in f),
                       "pooled_roi": float(np.average([x["roi"] for x in f], weights=bets)),
                       "min_avg_effective_odds": min(x["avg_effective_odds"] for x in f),
                       "min_priced_share": min(x["priced_share"] for x in f),
                       "risk_for_every_main_both": all(x["risk_for_every_main"] for x in f)}
    ok = [n for n, t in table.items() if t["min_avg_effective_odds"] >= 3.0 and t["min_priced_share"] >= 0.4
          and t["risk_for_every_main_both"]]
    worst_winner = max(ok or list(table), key=lambda n: (table[n]["worst_roi"], table[n]["pooled_roi"]))
    cur = "current (no EV rule)"
    for n in table:
        table[n]["roi_better_than_current_both"] = all(
            out["folds"][str(s)][n]["roi"] > out["folds"][str(s)][cur]["roi"] for s in C.SELECT)
    better = [n for n in ok if n != cur and table[n]["roi_better_than_current_both"]]
    chosen = max(better, key=lambda n: (table[n]["worst_roi"], table[n]["pooled_roi"])) if better else cur
    out["selection"] = {"table": table, "eligible": ok, "worst_fold_roi_winner": worst_winner,
                        "better_than_current_on_both": better, "chosen": chosen,
                        "rule": "avg effective odds >= 3.0 and priced share >= 40 % on both folds, a Risk for "
                                "every Main; a variant replaces the current rule only if its ROI is higher on BOTH "
                                "2023 and 2024; among those the highest worst-fold ROI (2023+2024 only)"}
    out["minutes"] = (time.time() - t0) / 60
    C.save_json("risk.json", out)
    figures(out)
    return out


def figures(out):
    plt = C.plot_style()
    names = out["variants"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))
    for ax, (key, title, scale) in zip(axes, (("roi", "ROI на реальных ценах Bet365, %", 100),
                                              ("hit_rate", "Доля выигрышей Risk, %", 100),
                                              ("avg_odds", "Средний коэффициент (с ценой)", 1))):
        for k, s in enumerate(C.FOLDS):
            f = out["folds"][str(s)]
            ax.bar(x + (k - 1) * 0.27, [f[n][key] * scale for n in names], 0.27, color=C.PALETTE[k],
                   label=config.season_label(s) + ("" if s in C.SELECT else " (только для информации)"))
            if key == "roi":
                for i, n in enumerate(names):
                    lo, hi = f[n]["roi_ci95"]
                    ax.plot([i + (k - 1) * 0.27] * 2, [lo * 100, hi * 100], color="black", lw=0.7)
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="grey", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Варианты правила Risk (отбор на 2023/24 + 2024/25; линии — 95 % бутстреп-интервал ROI)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "risk_variants.png"), plt.close(fig)


if __name__ == "__main__":
    run()
