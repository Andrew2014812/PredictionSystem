"""v5 study of the EXISTING XGBoost models (validation season only).

Every variant is fitted on the TRAIN seasons with the tuned parameters of the
active model and scored on the VALIDATION season; differences are given with a
paired bootstrap 95 % interval over matches, so a change is adopted only when
the interval is clearly on the good side. The test season is not used.

    1. Elo:        current features vs without Elo (and without Elo-derived features)
    2. Draws:      why arg-max never names a draw; draw-specific features; a separate
                   binary draw model blended into the 1X2 probabilities
    3. Features:   top-K selection by train-fold gain
    4. Corners:    one negative-binomial dispersion vs one per league
    5. Calibration of derived markets that v4 did not cover (corners lines, handicap)

    python experiments/xgb_v5/study.py            # writes experiments/xgb_v5/results.json
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from src.features.sets import GROUPS, MODEL_GROUPS  # noqa: E402
from src.features.store import load_features  # noqa: E402
from src.markets.distributions import corners_distribution, prob_over, reconcile_with_1x2, score_matrix  # noqa: E402
from src.markets.markets import goal_markets  # noqa: E402
from src.modelling.evaluation import classifier_metrics  # noqa: E402
from src.modelling.pipeline import _subset, season_split  # noqa: E402
from src.modelling.registry import active_version, version_metadata  # noqa: E402
from src.modelling.tasks import TASKS  # noqa: E402
from src.modelling.training import TaskTrainer, corner_leagues, negative_binomial_size  # noqa: E402

log = logging.getLogger("xgb_v5")
OUT = Path(__file__).with_name("results.json")
RNG = np.random.default_rng(7)
ELO_DERIVED = re.compile(r"(^[ha]_elo$|^elo_|_opp_elo$|_elo_surprise$)")


def columns_for(groups: list[str], cols: list[str]) -> list[str]:
    pats = [re.compile(p) for g in groups for p in GROUPS[g]]
    return [c for c in cols if any(p.search(c) for p in pats)]


def per_match_loss(task, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if task.kind == "classifier":
        return -np.log(np.clip(pred[np.arange(len(y)), y], 1e-12, 1))
    mu = np.clip(pred, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (term - (y - mu))                                    # Poisson deviance per match


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000) -> dict:
    """Mean of b - a (negative = b is better for a loss) with a paired 95 % interval."""
    d = b - a
    idx = RNG.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


class Fitter:
    def __init__(self, features: pd.DataFrame, meta: dict):
        split = season_split()
        self.cols = list(features.columns)
        self.tr, self.va = _subset(features, split["train"]), _subset(features, split["validation"])
        self.cl = corner_leagues(pd.concat([self.tr, self.va]))
        self.meta = meta

    def data(self, name):
        if name == "corners":
            return self.tr[self.tr.league.isin(self.cl)], self.va[self.va.league.isin(self.cl)]
        return self.tr, self.va

    def fit_predict(self, name: str, numeric: list[str], target=None):
        task = TASKS[name]
        tr, va = self.data(name)
        trainer = TaskTrainer(task, self.cols)
        trainer.numeric = numeric
        info = self.meta["tasks"][name]
        bundle = trainer.fit(tr, info["params"], info["n_estimators"])
        rows = va.loc[task.trainable(va)]
        return bundle, rows, bundle.predict(rows), task.target_values(rows).to_numpy()


def score(task, y, pred) -> dict:
    if task.kind == "classifier":
        m = classifier_metrics(y, pred)
        return {"log_loss": m["log_loss"], "brier": m["brier"], "accuracy": m["accuracy"],
                "draw_recall": m["recall"]["D"], "predicted_draw_share": m["predicted_share"]["D"]}
    loss = per_match_loss(task, y, pred)
    return {"poisson_deviance": float(loss.mean()), "mae": float(np.abs(y - pred).mean())}


# -- 1. Elo -----------------------------------------------------------------------
def elo_study(f: Fitter) -> list[dict]:
    rows = []
    for name in ("result", "home_goals", "away_goals"):
        task = TASKS[name]
        current = columns_for(MODEL_GROUPS[name], f.cols)
        without = [c for c in current if not ELO_DERIVED.search(c)]
        _, _, p_cur, y = f.fit_predict(name, current)
        _, _, p_no, _ = f.fit_predict(name, without)
        diff = bootstrap_diff(per_match_loss(task, y, p_cur), per_match_loss(task, y, p_no))
        rows.append({"model": name, "with_elo": score(task, y, p_cur), "without_elo": score(task, y, p_no),
                     "n_features": [len(current), len(without)], "loss_change_without_elo": diff})
        log.info("elo %s", rows[-1])
    return rows


# -- 2. Draws -----------------------------------------------------------------------
def draw_study(f: Fitter) -> dict:
    task = TASKS["result"]
    current = columns_for(MODEL_GROUPS["result"], f.cols)
    _, rows, p, y = f.fit_predict("result", current)
    pd_ = p[:, 1]
    out = {
        "actual_draw_share": float((y == 1).mean()),
        "mean_p_draw": float(pd_.mean()),
        "share_where_draw_is_most_likely": float((p.argmax(1) == 1).mean()),
        "p_draw_quantiles": {q: float(np.quantile(pd_, q)) for q in (0.1, 0.5, 0.9, 0.99)},
        "max_p_draw": float(pd_.max()),
        # the probability gap that a draw must close to become the arg-max
        "median_gap_to_top": float(np.median(p.max(1) - pd_)),
    }
    # draw rate by bins of predicted draw probability (calibration of P(D))
    bins = pd.cut(pd_, [0, .2, .24, .26, .28, .3, .32, 1])
    out["p_draw_calibration"] = (pd.DataFrame({"bin": bins.astype(str), "draw": (y == 1), "p": pd_})
                                 .groupby("bin", observed=True).agg(n=("draw", "size"), observed=("draw", "mean"),
                                                                    predicted=("p", "mean")).reset_index()
                                 .to_dict("records"))
    # draw rate by closeness of the teams (Elo expected score)
    if "elo_exp_home" in rows:
        closeness = (rows["elo_exp_home"] - 0.5).abs()
        q = pd.qcut(closeness, 5, duplicates="drop")
        out["draw_rate_by_closeness"] = (pd.DataFrame({"q": q.astype(str), "draw": (y == 1)})
                                         .groupby("q", observed=True)["draw"].agg(["size", "mean"])
                                         .reset_index().to_dict("records"))
    base = per_match_loss(task, y, p)

    # (a) draw-specific features: closeness and expected low scoring
    extra = {"abs_elo_diff": lambda d: d["elo_diff"].abs(),
             "abs_position_gap": lambda d: d["diff_position"].abs() if "diff_position" in d else np.nan,
             "goal_level": lambda d: d.filter(regex=r"^[ha]_season_gf$").sum(axis=1)}
    tr, va = f.data("result")
    tr, va = tr.copy(), va.copy()
    for k, fn in extra.items():
        tr[k], va[k] = fn(tr), fn(va)
    trainer = TaskTrainer(task, list(tr.columns))
    trainer.numeric = current + list(extra)
    info = f.meta["tasks"]["result"]
    b2 = trainer.fit(tr, info["params"], info["n_estimators"])
    p2 = b2.predict(va.loc[task.trainable(va)])
    out["draw_features"] = {**score(task, y, p2),
                            "loss_change": bootstrap_diff(base, per_match_loss(task, y, p2))}

    # (b) separate binary draw model blended into P(D) (weight chosen on the first half, judged on the second)
    import xgboost as xgb
    prep = b2.preprocessor
    tr_rows = tr.loc[task.trainable(tr)]
    y_tr_draw = (task.target_values(tr_rows).to_numpy() == 1).astype(int)
    params = {k: v for k, v in info["params"].items() if k != "half_life_years"}
    clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=info["n_estimators"], **params,
                            tree_method="hist", random_state=42, n_jobs=-1)
    clf.fit(prep.transform(tr_rows), y_tr_draw)
    pdraw = clf.predict_proba(prep.transform(va.loc[task.trainable(va)]))[:, 1]
    half = np.arange(len(y)) < len(y) / 2
    best_w, best_loss = 0.0, np.inf
    for w in np.linspace(0, 1, 11):
        pb = blend_draw(p, pdraw, w)
        loss = log_loss(y[half], pb[half], labels=[0, 1, 2])
        if loss < best_loss:
            best_w, best_loss = float(w), loss
    pb = blend_draw(p, pdraw, best_w)
    out["binary_draw_model"] = {"weight": best_w, **score(task, y[~half], pb[~half]),
                                "baseline_second_half": score(task, y[~half], p[~half]),
                                "loss_change": bootstrap_diff(base[~half], per_match_loss(task, y[~half], pb[~half]))}
    log.info("draws %s", {k: v for k, v in out.items() if k not in ("p_draw_calibration",)})
    return out


def blend_draw(p: np.ndarray, pdraw: np.ndarray, w: float) -> np.ndarray:
    """Replace P(D) by (1-w) P(D) + w P_bin(D) and rescale H / A to keep the sum 1."""
    new_d = (1 - w) * p[:, 1] + w * pdraw
    rest = (1 - new_d) / np.clip(p[:, 0] + p[:, 2], 1e-9, None)
    return np.column_stack([p[:, 0] * rest, new_d, p[:, 2] * rest])


# -- 3. Feature selection -----------------------------------------------------------
def selection_study(f: Fitter) -> list[dict]:
    out = []
    for name in ("result", "home_goals", "away_goals", "corners"):
        task = TASKS[name]
        current = columns_for(MODEL_GROUPS[name], f.cols)
        bundle, _, p_all, y = f.fit_predict(name, current)
        gain = pd.Series(bundle.model.get_booster().get_score(importance_type="gain"))
        names = bundle.preprocessor.feature_names
        gain.index = [names[int(i[1:])] if i.startswith("f") and i[1:].isdigit() else i for i in gain.index]
        ranked = [c for c in gain.sort_values(ascending=False).index if c in current]
        base = per_match_loss(task, y, p_all)
        for k in (30, 60, 100):
            if k >= len(current):
                continue
            _, _, pk, _ = f.fit_predict(name, ranked[:k])
            out.append({"model": name, "k": k, "of": len(current), **score(task, y, pk),
                        "loss_change": bootstrap_diff(base, per_match_loss(task, y, pk))})
            log.info("selection %s", out[-1])
    return out


# -- 4/5. Corners dispersion and calibration of corner / handicap lines -------------------------------
def ece(outcome, proba, bins=10) -> float:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(outcome[idx == b].mean() - proba[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def corners_and_handicap(f: Fitter) -> dict:
    out = {}
    bundle, rows, mu, y = f.fit_predict("corners", columns_for(MODEL_GROUPS["corners"], f.cols))
    tr, _ = f.data("corners")
    tr_rows = tr.loc[TASKS["corners"].trainable(tr)]
    mu_tr = bundle.predict(tr_rows)
    global_r = negative_binomial_size(tr_rows["total_corners"].to_numpy(float), mu_tr)
    per_league = {lg: negative_binomial_size(g["total_corners"].to_numpy(float), mu_tr[tr_rows["league"] == lg])
                  for lg, g in tr_rows.groupby("league")}
    lines = (8.5, 9.5, 10.5)
    res = {}
    for label, sizes in (("global", None), ("per_league", per_league)):
        ll, cal = [], {}
        for line in lines:
            p = np.array([prob_over(corners_distribution(m, sizes.get(lg, global_r) if sizes else global_r), line)
                          for m, lg in zip(mu, rows["league"])])
            o = (y > line).astype(float)
            ll.append(log_loss(o, np.clip(p, 1e-6, 1 - 1e-6)))
            cal[str(line)] = {"log_loss": ll[-1], "ece": ece(o, p), "brier": float(np.mean((p - o) ** 2))}
        res[label] = {"mean_line_log_loss": float(np.mean(ll)), "lines": cal}
    out["corners_dispersion"] = {"global_size": global_r, **res}

    # handicap calibration from the reconciled score matrix (Asian half and whole lines)

    _, rrows, p1x2, _ = f.fit_predict("result", columns_for(MODEL_GROUPS["result"], f.cols))
    _, _, lh, _ = f.fit_predict("home_goals", columns_for(MODEL_GROUPS["home_goals"], f.cols))
    _, _, la, _ = f.fit_predict("away_goals", columns_for(MODEL_GROUPS["away_goals"], f.cols))
    hg, ag = rrows["home_goals"].to_numpy(int), rrows["away_goals"].to_numpy(int)
    probs: dict[str, list] = {}
    for k in range(len(rrows)):
        m = reconcile_with_1x2(score_matrix(lh[k], la[k]), p1x2[k])
        for s in goal_markets(m):
            if s.market.startswith("HANDICAP_") and s.selection == "H":
                probs.setdefault(s.market, []).append((s.probability, s.push_probability, hg[k] - ag[k]))
    hcp = {}
    for market, vals in probs.items():
        line = float(market.split("_")[1])
        p = np.array([v[0] for v in vals])
        margin = np.array([v[2] for v in vals]) + line
        won, push = (margin > 0).astype(float), margin == 0
        keep = ~push                                           # refunds are not decided bets
        hcp[market] = {"ece_win_given_decided": ece(won[keep], (p / np.clip(1 - np.array([v[1] for v in vals]), 1e-9, None))[keep]),
                       "push_predicted": float(np.mean([v[1] for v in vals])), "push_actual": float(push.mean())}
    out["handicap_calibration"] = hcp
    log.info("corners/handicap %s", out["corners_dispersion"])
    return out


def main() -> None:
    features = load_features()
    meta = version_metadata(active_version())
    f = Fitter(features, meta)
    results = {"model_version": active_version(), "elo": elo_study(f), "draws": draw_study(f),
               "feature_selection": selection_study(f), "corners_handicap": corners_and_handicap(f)}
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps({k: v for k, v in results.items() if k != "draws"}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    main()
