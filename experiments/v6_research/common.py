"""Shared helpers of the v6 research (folds, production-faithful fits, metrics, IO, plots).

Walk-forward folds (season = start year):

    fold s:   OUTER  fit on seasons < s,      score season s
              INNER  fit on seasons < s - 1,  predict season s - 1  (inner validation:
                     calibration, blend weights, Dixon-Coles rho, NB size are fitted here)

Variants are CHOSEN on the folds 2023 and 2024 only; the fold 2025 (2025/26) has been
looked at during earlier development and is reported for information only.

Every XGBoost model uses the tuned parameters and tree counts of the active production
version (local_data/models/<active>/metadata.json); nothing is re-tuned here.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             r2_score)
from xgboost import XGBClassifier, XGBRegressor

from src import config
from src.features.sets import model_features
from src.features.store import load_features
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.registry import active_version, version_metadata
from src.modelling.tasks import TASKS
from src.modelling.training import corner_leagues, recency_weights

log = logging.getLogger("v6")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(exist_ok=True)

FOLDS = (2023, 2024, 2025)
SELECT = (2023, 2024)
REPORT = 2025
CLASSES = ("H", "D", "A")


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
_FEATURES: pd.DataFrame | None = None


def features() -> pd.DataFrame:
    global _FEATURES
    if _FEATURES is None:
        df = load_features()
        _FEATURES = df.loc[df["played"].astype(bool) & df["home_goals"].notna()
                           & (df["season"] >= config.FIRST_SEASON)].reset_index(drop=True)
    return _FEATURES


def seasons(df: pd.DataFrame, which) -> pd.DataFrame:
    return df.loc[df["season"].isin(list(which))]


def before(df: pd.DataFrame, season: int) -> pd.DataFrame:
    return df.loc[df["season"] < season]


def prod_meta() -> dict:
    return version_metadata(active_version())


def prod_params(task: str) -> tuple[dict, int]:
    info = prod_meta()["tasks"][task]
    return dict(info["params"]), int(info["n_estimators"])


def result_index(df: pd.DataFrame) -> np.ndarray:
    return TASKS["result"].target_values(df).to_numpy()


# ---------------------------------------------------------------------------------------
# production-faithful XGBoost
# ---------------------------------------------------------------------------------------
class XGBFit:
    """Same preprocessing and estimator settings as src/modelling/training.py."""

    def __init__(self, numeric: list[str], params: dict, n_estimators: int, objective: str,
                 eval_metric: str, classifier: bool):
        self.numeric, self.params, self.n = list(numeric), dict(params), int(n_estimators)
        self.objective, self.eval_metric, self.classifier = objective, eval_metric, classifier
        self.seconds = 0.0

    def _est(self, n, early=False):
        cls = XGBClassifier if self.classifier else XGBRegressor
        kw = dict(n_estimators=n, objective=self.objective, eval_metric=self.eval_metric,
                  tree_method="hist", random_state=config.RANDOM_STATE, n_jobs=-1,
                  **{k: v for k, v in self.params.items() if k != "half_life_years"})
        if early:
            kw["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS
        return cls(**kw)

    def _weights(self, rows, extra):
        w = recency_weights(rows["date"], self.params.get("half_life_years", 0))
        if extra is not None:
            w = np.asarray(extra, float) * (1.0 if w is None else w)
        return w

    def fit(self, rows: pd.DataFrame, y, sample_weight=None) -> "XGBFit":
        t0 = time.time()
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        self.model = self._est(self.n).fit(self.prep.transform(rows), np.asarray(y),
                                           sample_weight=self._weights(rows, sample_weight))
        self.seconds = time.time() - t0
        return self

    def early_stopped_trees(self, train, y_tr, valid, y_va, sample_weight=None) -> int:
        """Number of trees by early stopping on ``valid`` (used only for new model kinds)."""
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        m = self._est(config.MAX_ESTIMATORS, early=True)
        m.fit(prep.transform(train), np.asarray(y_tr), sample_weight=self._weights(train, sample_weight),
              eval_set=[(prep.transform(valid), np.asarray(y_va))], verbose=False)
        return int(m.best_iteration) + 1

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.prep.transform(rows)
        if self.classifier:
            p = self.model.predict_proba(X)
            return p[:, 1] if p.shape[1] == 2 else p
        return self.model.predict(X)


def task_numeric(task: str) -> list[str]:
    return model_features(task, list(features().columns))[0]


def production_model(task: str, numeric: list[str] | None = None) -> XGBFit:
    t = TASKS[task]
    params, n = prod_params(task)
    return XGBFit(numeric or task_numeric(task), params, n, t.objective, t.eval_metric,
                  t.kind == "classifier")


def target(task: str, rows: pd.DataFrame) -> np.ndarray:
    return TASKS[task].target_values(rows).to_numpy()


def fold_corner_leagues(train: pd.DataFrame) -> list[str]:
    return corner_leagues(train)


# ---------------------------------------------------------------------------------------
# base predictions of the production setup (cached; shared by all studies)
# ---------------------------------------------------------------------------------------
BASE_COLS = ["match_id", "league", "season", "date", "home_team", "away_team", "home_goals", "away_goals",
             "total_corners", "home_corners", "away_corners", "odds_home", "odds_draw", "odds_away",
             "odds_over25", "odds_under25", "odds_ah_line", "odds_ah_home", "odds_ah_away"]


def _predict_block(train: pd.DataFrame, score: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = score[BASE_COLS].copy()
    out["y"] = result_index(score)
    times = {}
    m = production_model("result").fit(train, target("result", train))
    p = m.predict(score)
    out["pH"], out["pD"], out["pA"] = p[:, 0], p[:, 1], p[:, 2]
    times["result"] = m.seconds
    for task, col in (("home_goals", "lh"), ("away_goals", "la")):
        m = production_model(task).fit(train, target(task, train))
        out[col] = m.predict(score)
        times[task] = m.seconds
    cl = fold_corner_leagues(train)
    ctr = train.loc[train["total_corners"].notna() & train["league"].isin(cl)]
    m = production_model("corners").fit(ctr, target("corners", ctr))
    out["cmu"] = np.where(score["league"].isin(cl), m.predict(score), np.nan)
    times["corners"] = m.seconds
    return out, {"seconds": times, "corner_leagues": cl, "train_rows": int(len(train))}


def base_predictions(season: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(inner, outer, info): production models fitted on < s-1 -> s-1 and on < s -> s."""
    f_in, f_out, f_info = (CACHE / f"base_{season}_{k}" for k in ("inner.parquet", "outer.parquet", "info.json"))
    if not refresh and f_in.exists() and f_out.exists() and f_info.exists():
        return pd.read_parquet(f_in), pd.read_parquet(f_out), json.loads(f_info.read_text())
    df = features()
    inner, info_in = _predict_block(before(df, season - 1), seasons(df, [season - 1]))
    outer, info_out = _predict_block(before(df, season), seasons(df, [season]))
    info = {"inner": info_in, "outer": info_out}
    inner.to_parquet(f_in), outer.to_parquet(f_out)
    f_info.write_text(json.dumps(info))
    log.info("base predictions %s: inner %d rows, outer %d rows", season, len(inner), len(outer))
    return inner, outer, info


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------
def ece(outcome, proba, bins: int = 10) -> float:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(outcome[idx == b].mean() - proba[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(outcome, proba, bins: int = 10, min_n: int = 20) -> list[dict]:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return [{"predicted": float(proba[idx == b].mean()), "observed": float(outcome[idx == b].mean()),
             "n": int((idx == b).sum())} for b in range(bins) if (idx == b).sum() >= min_n]


def multiclass_loss_per_match(y, p) -> np.ndarray:
    return -np.log(np.clip(np.asarray(p)[np.arange(len(y)), np.asarray(y)], 1e-15, 1))


def cls_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray | None = None) -> dict:
    """Probability metrics from ``p``; class metrics from ``pred`` (default arg-max)."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = p.argmax(1) if pred is None else np.asarray(pred, int)
    onehot = np.eye(3)[y]
    rec = {c: float(((pred == i) & (y == i)).sum() / max((y == i).sum(), 1)) for i, c in enumerate(CLASSES)}
    prec = {c: float(((pred == i) & (y == i)).sum() / max((pred == i).sum(), 1)) for i, c in enumerate(CLASSES)}
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "draw_precision": prec["D"], "draw_recall": rec["D"],
        "home_recall": rec["H"], "away_recall": rec["A"],
        "recall": rec, "precision": prec,
        "predicted_draw_share": float((pred == 1).mean()), "actual_draw_share": float((y == 1).mean()),
        "predicted_share": {c: float((pred == i).mean()) for i, c in enumerate(CLASSES)},
        "actual_share": {c: float((y == i).mean()) for i, c in enumerate(CLASSES)},
        "mean_probability": {c: float(p[:, i].mean()) for i, c in enumerate(CLASSES)},
        "ece": {c: ece(y == i, p[:, i]) for i, c in enumerate(CLASSES)},
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def poisson_dev(y, mu) -> np.ndarray:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))


def count_metrics(y, mu, buckets: int = 10) -> dict:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    q = pd.qcut(pd.Series(mu).rank(method="first"), buckets, labels=False)
    calib = [{"bucket": int(b), "predicted": float(mu[q == b].mean()), "observed": float(y[q == b].mean()),
              "n": int((q == b).sum())} for b in range(buckets)]
    res = y - mu
    return {"n": int(len(y)), "mae": float(np.abs(res).mean()), "rmse": float(np.sqrt(np.mean(res ** 2))),
            "r2": float(r2_score(y, mu)), "poisson_deviance": float(poisson_dev(y, mu).mean()),
            "mean_actual": float(y.mean()), "mean_predicted": float(mu.mean()),
            "bias": float(mu.mean() - y.mean()), "calibration_by_decile": calib,
            "residual_quantiles": {str(k): float(np.quantile(res, k)) for k in (0.05, 0.25, 0.5, 0.75, 0.95)}}


def binary_metrics(outcome, p, base_rate: float) -> dict:
    o = np.asarray(outcome, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    b = np.clip(base_rate, 1e-6, 1 - 1e-6)
    ll = float(log_loss(o, p, labels=[0, 1]))
    bll = float(log_loss(o, np.full(len(o), b), labels=[0, 1]))
    return {"n": int(len(o)), "log_loss": ll, "brier": float(np.mean((p - o) ** 2)),
            "base_rate_train": float(base_rate), "observed_rate": float(o.mean()), "mean_predicted": float(p.mean()),
            "baseline_log_loss": bll, "baseline_brier": float(np.mean((b - o) ** 2)),
            "log_loss_skill": float(1 - ll / bll), "ece": ece(o, p)}


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n: int = 1000, seed: int = 7) -> dict:
    """Mean of (a - b) per match and a 95 % bootstrap interval (negative = a better)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


# ---------------------------------------------------------------------------------------
# IO / plots
# ---------------------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False))
    log.info("wrote %s", path)
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
    return plt


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
           "#CCB974", "#64B5CD"]


def setup_logging(name: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)
