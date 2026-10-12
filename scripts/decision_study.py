"""Validation study of the decision layer (no test-season data is used to choose).

Fits the four models on the TRAIN seasons with the tuned parameters of the
active model version, predicts the VALIDATION season and compares:

* the draw-aware 1X2 rule against plain arg-max (cross-fitted on two halves of
  the season, so the multiplier is never judged on the data it was chosen on);
* calibration (ECE) of 1X2 and of the derived markets, and temperature scaling;
* Dixon-Coles against independent Poisson (cross-fitted rho);
* Main-prediction ranking rules with real Football-Data (Bet365) prices;
* Risk-prediction EV thresholds;
* a walk-forward evaluation of the result model (train on all earlier seasons,
  evaluate the next one).

The result is written to ``local_data/models/decision_study.json`` and shown on
the Model Analysis page. Note: the 2025/26 test season has been looked at while
the system was developed, so it is no longer a strictly untouched test set —
the walk-forward table is the more honest view of out-of-sample quality.

    python scripts/decision_study.py
"""
import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from src import config  # noqa: E402
from src.features.store import load_features  # noqa: E402
from src.markets.distributions import (fit_dixon_coles_rho, reconcile_with_1x2,  # noqa: E402
                                       score_matrix)
from src.markets.markets import selection_won  # noqa: E402
from src.markets.odds import PricedSelection  # noqa: E402
from src.markets.recommendation import main_prediction, reliability, risk_prediction  # noqa: E402
from src.modelling.decision import choose_draw_multiplier, decide  # noqa: E402
from src.modelling.evaluation import classifier_metrics, multiclass_brier  # noqa: E402
from src.modelling.pipeline import _subset, season_split  # noqa: E402
from src.modelling.registry import active_version, version_metadata  # noqa: E402
from src.modelling.tasks import TASKS  # noqa: E402
from src.modelling.training import TaskTrainer, corner_leagues, negative_binomial_size  # noqa: E402
from src.prediction.engine import predict_matches  # noqa: E402
from src.providers import real_odds_lookup  # noqa: E402
from src.storage import models_storage  # noqa: E402

log = logging.getLogger("study")
OUT = "decision_study.json"


def ece(outcome: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: weighted mean |observed - predicted| over probability bins."""
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(outcome[m].mean() - proba[m].mean())
    return float(total)


def halves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    order = df["date"].rank(method="first").to_numpy()
    first = order <= len(df) / 2
    return first, ~first


def train_only_bundles(features, split, meta):
    tr = _subset(features, split["train"])
    va = _subset(features, split["validation"])
    cl = corner_leagues(pd.concat([tr, va]))
    bundles = {}
    for name, task in TASKS.items():
        t_tr, t_va = (tr[tr.league.isin(cl)], va[va.league.isin(cl)]) if name == "corners" else (tr, va)
        info = meta["tasks"][name]
        bundle = TaskTrainer(task, features.columns).fit(t_tr, info["params"], info["n_estimators"])
        if name == "corners":
            rows = t_va.loc[task.trainable(t_va)]
            bundle.extra.update({"leagues": cl, "nb_size": negative_binomial_size(
                rows[task.target].to_numpy(float), bundle.predict(rows))})
        if name == "result":
            bundle.extra["draw_multiplier"] = 1.0
        bundles[name] = bundle
    return bundles, tr, va


# -- 1. draw rule -------------------------------------------------------------
def draw_study(p, y, train_draw_share, va, odds):
    a, b = halves(va)
    named = {"argmax": np.empty(len(y), int), "draw_rule": np.empty(len(y), int)}
    chosen = []
    for fit, ev in ((a, b), (b, a)):
        m = choose_draw_multiplier(p[fit], y[fit], train_draw_share)["multiplier"]
        chosen.append(m)
        named["argmax"][ev] = decide(p[ev], 1.0)
        named["draw_rule"][ev] = decide(p[ev], m)
    out = {"multipliers_chosen_on_each_half": chosen}
    for key, pred in named.items():
        mets = classifier_metrics(y, p, pred)
        price = odds[np.arange(len(y)), pred]
        priced = np.isfinite(price) & (price > 1)
        profit = np.where(pred == y, price - 1, -1.0)[priced]
        out[key] = {"accuracy": mets["accuracy"], "macro_f1": mets["macro_f1"],
                    "draw_recall": mets["recall"]["D"], "draw_precision": mets["precision"]["D"],
                    "predicted_draw_share": mets["predicted_share"]["D"], "actual_draw_share": mets["actual_share"]["D"],
                    "roi_1x2_at_real_odds": float(profit.mean()) if priced.any() else None, "bets": int(priced.sum())}
    return out


# -- 2. calibration -----------------------------------------------------------
def temperature(p: np.ndarray, T: float) -> np.ndarray:
    q = np.power(np.clip(p, 1e-9, 1), 1 / T)
    return q / q.sum(axis=1, keepdims=True)


def calibration_study(p, y, va, matrices):
    a, b = halves(va)
    temp = {"raw": [], "scaled": [], "T": []}
    for fit, ev in ((a, b), (b, a)):
        T = minimize_scalar(lambda t: log_loss(y[fit], temperature(p[fit], t), labels=[0, 1, 2]),
                            bounds=(0.5, 2.0), method="bounded").x
        temp["T"].append(float(T))
        temp["raw"].append(log_loss(y[ev], p[ev], labels=[0, 1, 2]))
        temp["scaled"].append(log_loss(y[ev], temperature(p[ev], T), labels=[0, 1, 2]))
    hg, ag = va["home_goals"].to_numpy(int), va["away_goals"].to_numpy(int)
    ii, jj = np.indices(matrices[0].shape)
    over25 = np.array([m[ii + jj > 2.5].sum() for m in matrices])
    btts = np.array([m[(ii > 0) & (jj > 0)].sum() for m in matrices])
    dc_1x = p[:, 0] + p[:, 1]
    markets = {
        "1X2 home": ((y == 0), p[:, 0]), "1X2 draw": ((y == 1), p[:, 1]), "1X2 away": ((y == 2), p[:, 2]),
        "Over 2.5": ((hg + ag) > 2.5, over25), "BTTS yes": ((hg > 0) & (ag > 0), btts),
        "Double chance 1X": ((y != 2), dc_1x),
    }
    return {"ece": {k: ece(o, q) for k, (o, q) in markets.items()},
            "brier": {k: float(np.mean((q - o) ** 2)) for k, (o, q) in markets.items()},
            "temperature_scaling": {"T": temp["T"], "log_loss_raw": float(np.mean(temp["raw"])),
                                    "log_loss_scaled": float(np.mean(temp["scaled"]))}}


# -- 3. Dixon-Coles -----------------------------------------------------------
def dixon_coles_study(lh, la, p, va):
    hg, ag = va["home_goals"].to_numpy(int), va["away_goals"].to_numpy(int)
    a, b = halves(va)
    res = {"rho": [], "poisson": {"score_log_loss": [], "over25_brier": [], "btts_brier": []},
           "dixon_coles": {"score_log_loss": [], "over25_brier": [], "btts_brier": []}}
    size = config.MAX_GOALS + 1
    ii, jj = np.indices((size, size))
    for fit, ev in ((a, b), (b, a)):
        rho = fit_dixon_coles_rho(lh[fit], la[fit], hg[fit], ag[fit])
        res["rho"].append(rho)
        for key, r in (("poisson", 0.0), ("dixon_coles", rho)):
            ll, o25, bt = [], [], []
            for k in np.where(ev)[0]:
                m = reconcile_with_1x2(score_matrix(lh[k], la[k], rho=r), p[k])
                ll.append(-np.log(max(m[min(hg[k], size - 1), min(ag[k], size - 1)], 1e-12)))
                o25.append((m[ii + jj > 2.5].sum() - (hg[k] + ag[k] > 2.5)) ** 2)
                bt.append((m[(ii > 0) & (jj > 0)].sum() - (hg[k] > 0 and ag[k] > 0)) ** 2)
            res[key]["score_log_loss"].append(float(np.mean(ll)))
            res[key]["over25_brier"].append(float(np.mean(o25)))
            res[key]["btts_brier"].append(float(np.mean(bt)))
    for key in ("poisson", "dixon_coles"):
        res[key] = {k: float(np.mean(v)) for k, v in res[key].items()}
    res["rho_full_validation"] = fit_dixon_coles_rho(lh, la, hg, ag)
    return res


# -- 4. main / risk ranking -----------------------------------------------------
def legacy_main(sels):
    """The v3 rule (hard odds band 1.30–3.00, EV >= 3 %), kept for comparison."""
    cand = [s for s in sels if s.group != "EXACT_SCORE" and 0.45 <= s.probability <= 0.80]
    value = [s for s in cand if s.has_odds and 1.30 <= s.odds < 3.00 and s.ev >= 0.03]
    if value:
        return max(value, key=lambda s: (s.ev * reliability(s), s.probability))
    if cand:
        return max(cand, key=lambda s: (s.probability * reliability(s), s.has_odds))
    return None


def settle(pick, result):
    if pick is None:
        return None
    won = selection_won(pick.market, pick.selection, result["home_goals"], result["away_goals"],
                        result.get("total_corners"))
    return won


def summarise(picks) -> dict:
    """picks: list of (won, odds or None)."""
    picks = [p for p in picks if p[0] is not None]
    priced = [(w, o) for w, o in picks if o]
    profit = [o - 1 if w else -1.0 for w, o in priced]
    return {"predictions": len(picks), "hit_rate": float(np.mean([w for w, _ in picks])) if picks else None,
            "bets": len(priced), "priced_share": len(priced) / len(picks) if picks else None,
            "roi": float(np.mean(profit)) if profit else None,
            "avg_odds": float(np.mean([o for _, o in priced])) if priced else None}


def ranking_study(batch, va):
    results = va.set_index("match_id")[["home_goals", "away_goals", "total_corners", "date"]].to_dict("index")
    by_match: dict[str, list] = {}
    for r in batch.markets.itertuples(index=False):
        odds = None if pd.isna(r.odds) else float(r.odds)
        by_match.setdefault(r.match_id, []).append(PricedSelection(r.market, r.selection, float(r.probability), odds))
    ids = sorted(by_match, key=lambda m: results[m]["date"])
    half = {m: i < len(ids) / 2 for i, m in enumerate(ids)}

    def evaluate(choose):
        rows = {True: [], False: []}
        for m in ids:
            pick = choose(by_match[m])
            won = settle(pick, results[m])
            rows[half[m]].append((won, pick.odds if pick is not None and pick.has_odds else None))
        both = rows[True] + rows[False]
        return {**summarise(both), "first_half_roi": summarise(rows[True])["roi"],
                "second_half_roi": summarise(rows[False])["roi"]}

    grid = []
    for softness, ev_w, min_ev, min_p in itertools.product((0.15, 0.30, 0.60, 5.0), (0.0, 0.5, 1.0, 2.0),
                                                           (-1.0, 0.0, 0.03), (0.40, 0.45, 0.50)):
        r = evaluate(lambda s: (lambda rec: rec.pick if rec else None)(
            main_prediction(s, min_ev=min_ev, ev_weight=ev_w, softness=softness, min_probability=min_p)))
        grid.append({"softness": softness, "ev_weight": ev_w, "min_ev": min_ev, "min_probability": min_p, **r})
        log.info("main %s", grid[-1])
    legacy = evaluate(legacy_main)
    risk = []
    for min_ev in (0.05, 0.10, 0.15, 0.20):
        risk.append({"min_ev": min_ev, **evaluate(lambda s: (lambda rec: rec.pick if rec else None)(
            risk_prediction(s, min_ev=min_ev)))})
    return {"main_grid": grid, "main_legacy": legacy, "risk_grid": risk}


def choose_main(grid: list[dict]) -> dict:
    """Highest ROI among rules that price most main predictions and are profitable-or-best in
    BOTH halves of the validation season (stability), ties -> higher hit rate."""
    ok = [g for g in grid if g["roi"] is not None and (g["priced_share"] or 0) >= 0.6
          and g["first_half_roi"] is not None and g["second_half_roi"] is not None]
    return max(ok, key=lambda g: (min(g["first_half_roi"], g["second_half_roi"]), g["hit_rate"]))


# -- 5. walk-forward ------------------------------------------------------------
def walk_forward(features, meta):
    task, info = TASKS["result"], meta["tasks"]["result"]
    seasons = sorted(int(s) for s in features.loc[features["played"].astype(bool), "season"].unique())
    rows = []
    for s in seasons[2:]:
        if s >= config.current_season():
            continue
        tr = _subset(features, [x for x in seasons if x < s])
        ev = _subset(features, [s])
        ev = ev.loc[task.trainable(ev)]
        p = TaskTrainer(task, features.columns).fit(tr, info["params"], info["n_estimators"]).predict(ev)
        y = task.target_values(ev).to_numpy()
        prior = np.bincount(task.target_values(tr.loc[task.trainable(tr)]).to_numpy(), minlength=3) / len(task.target_values(tr.loc[task.trainable(tr)]))
        rows.append({"season": config.season_label(s), "matches": len(ev),
                     "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
                     "brier": multiclass_brier(y, p), "accuracy": float((p.argmax(1) == y).mean()),
                     "baseline_log_loss": float(log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2]))})
        log.info("walk-forward %s", rows[-1])
    return rows


def main() -> None:
    features = load_features()
    split = season_split()
    meta = version_metadata(active_version())
    bundles, tr, va = train_only_bundles(features, split, meta)
    va = va.loc[va["home_goals"].notna()].reset_index(drop=True)
    task = TASKS["result"]
    p = bundles["result"].predict(va)
    y = task.target_values(va).to_numpy()
    lh, la = bundles["home_goals"].predict(va), bundles["away_goals"].predict(va)
    train_draw = float((task.target_values(tr.loc[task.trainable(tr)]) == 1).mean())
    odds = va[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)

    study = {"validation_season": config.season_label(split["validation"][0]), "matches": len(va),
             "model_version": active_version()}
    study["draw_rule"] = draw_study(p, y, train_draw, va, odds)
    log.info("draw rule %s", study["draw_rule"])
    study["dixon_coles"] = dixon_coles_study(lh, la, p, va)
    log.info("dixon-coles %s", study["dixon_coles"])
    rho = study["dixon_coles"]["rho_full_validation"] if config.DIXON_COLES else 0.0
    matrices = [reconcile_with_1x2(score_matrix(lh[k], la[k], rho=rho), p[k]) for k in range(len(va))]
    study["calibration"] = calibration_study(p, y, va, matrices)
    log.info("calibration %s", study["calibration"])
    bundles["home_goals"].extra["dc_rho"] = rho
    batch = predict_matches(va, bundles, real_odds_lookup(va), "study", "study")
    study.update(ranking_study(batch, va))
    study["main_chosen"] = choose_main(study["main_grid"])
    study["walk_forward"] = walk_forward(features, meta)
    models_storage.write_json(OUT, study)
    print(json.dumps({k: study[k] for k in ("draw_rule", "dixon_coles", "calibration", "main_legacy",
                                             "main_chosen", "risk_grid", "walk_forward")}, indent=1, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
