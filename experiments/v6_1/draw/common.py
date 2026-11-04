"""Shared pieces of the v6.1 draw research (folds, production-faithful fits, metrics, IO).

Blocks and folds (season = start year)

    block b      every production model (result, home goals, away goals, corners) fitted on the
                 played seasons < b with the tuned parameters of the active model, predicting season b.
    fold s       OUTER = block s, INNER = block s - 1 (inner validation: everything that is fitted —
                 calibration, blend weights, Dixon–Coles rho, inflation, stacking — is fitted here).

    selection folds  2023/24 and 2024/25   (every choice is made on these two only)
    information      2025/26               (looked at during v6 development; never used to choose)
    check            2026/27               (block 2026: models fitted on 2021–2025, i.e. production-like,
                                            plus the recorded production history of the website)

Every XGBoost fit is limited to ``N_JOBS`` threads (other studies run on the same machine).
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from xgboost import XGBClassifier

from src import config
from src.features.store import load_features
from src.modelling import training
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.registry import active_version, version_metadata

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES, CACHE = HERE / "results", HERE / "figures", HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

N_JOBS = 3
BLOCKS = (2022, 2023, 2024, 2025, 2026)
FOLDS = (2023, 2024, 2025, 2026)
SELECT = (2023, 2024)
INFO = 2025
CHECK = 2026
CLASSES = ("H", "D", "A")
ROLE = {2023: "отбор", 2024: "отбор", 2025: "информация", 2026: "проверка"}

log = logging.getLogger("draw61")
# XGBoost returns float32 probabilities whose rows sum to 1 only up to ~1e-7; sklearn warns about it on every call
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")


def setup_logging(name: str = "draw61") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def season_label(s) -> str:
    return config.season_label(int(s))


# ---------------------------------------------------------------------------------------------- threads
_ORIG_ESTIMATOR = training._estimator


def _limited_estimator(*args, **kwargs):
    est = _ORIG_ESTIMATOR(*args, **kwargs)
    est.set_params(n_jobs=N_JOBS)
    return est


training._estimator = _limited_estimator          # production TaskTrainer, but N_JOBS threads


# ---------------------------------------------------------------------------------------------- data
_FEATURES: pd.DataFrame | None = None


def features() -> pd.DataFrame:
    """Played matches with a result (all seasons from FIRST_SEASON)."""
    global _FEATURES
    if _FEATURES is None:
        df = load_features()
        _FEATURES = df.loc[df["played"].astype(bool) & df["home_goals"].notna() & df["result"].notna()
                           & (df["season"] >= config.FIRST_SEASON)].reset_index(drop=True)
    return _FEATURES


def prod_meta() -> dict:
    return version_metadata(active_version())


def y_index(result: pd.Series) -> np.ndarray:
    return result.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(int)


# parity / low-scoring features (absolute gaps and sums; the production model only has signed gaps)
def add_parity(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    new = {}
    for c in [c for c in df.columns if c.startswith("diff_")] + ["elo_diff"]:
        new[f"abs_{c}"] = df[c].abs()
    new["abs_elo_exp_gap"] = (df["elo_exp_home"] - 0.5).abs()
    new["abs_venue_form_gap"] = (df["h_venue_ppg"] - df["a_venue_ppg"]).abs()        # home form at home vs away form away
    new["abs_form5_gd_gap"] = (df["h_form5_gd"] - df["a_form5_gd"]).abs()
    new["sum_form10_draw_rate"] = df["h_form10_draw_rate"] + df["a_form10_draw_rate"]
    new["sum_form15_draw_rate"] = df["h_form15_draw_rate"] + df["a_form15_draw_rate"]
    new["sum_form10_goals"] = df["h_form10_gf"] + df["h_form10_ga"] + df["a_form10_gf"] + df["a_form10_ga"]
    new["sum_venue_attack"] = df["h_venue_gf"] + df["a_venue_gf"]
    new["sum_form10_sot"] = df["h_form10_sot_for"] + df["a_form10_sot_for"]
    new["low_score_index"] = df["league_avg_home_goals"] + df["league_avg_away_goals"]
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return out, list(new)


# ---------------------------------------------------------------------------------------------- XGBoost
class XGBFit:
    """XGBoost on the production preprocessing (one-hot league + numeric passthrough)."""

    def __init__(self, numeric, params: dict, n_estimators: int, objective: str, eval_metric: str):
        self.numeric, self.params, self.n = list(numeric), dict(params), int(n_estimators)
        self.objective, self.eval_metric = objective, eval_metric
        self.seconds = 0.0

    def _est(self, n, early=False):
        kw = dict(n_estimators=n, objective=self.objective, eval_metric=self.eval_metric, tree_method="hist",
                  random_state=config.RANDOM_STATE, n_jobs=N_JOBS,
                  **{k: v for k, v in self.params.items() if k != "half_life_years"})
        if early:
            kw["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS
        return XGBClassifier(**kw)

    def _w(self, rows):
        return training.recency_weights(rows["date"], self.params.get("half_life_years", 0))

    def fit(self, rows: pd.DataFrame, y) -> "XGBFit":
        t0 = time.time()
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        self.model = self._est(self.n).fit(self.prep.transform(rows), np.asarray(y), sample_weight=self._w(rows))
        self.seconds = time.time() - t0
        return self

    def early_stopped_trees(self, train, y_tr, valid, y_va) -> int:
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        m = self._est(config.MAX_ESTIMATORS, early=True)
        m.fit(prep.transform(train), np.asarray(y_tr), sample_weight=self._w(train),
              eval_set=[(prep.transform(valid), np.asarray(y_va))], verbose=False)
        return int(m.best_iteration) + 1

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(self.prep.transform(rows))
        return p[:, 1] if p.shape[1] == 2 else p


# ---------------------------------------------------------------------------------------------- metrics
def ece(outcome, proba, bins: int = 10) -> float:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(outcome[idx == b].mean() - proba[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(outcome, proba, bins: int = 20, min_n: int = 30) -> list[dict]:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return [{"predicted": float(proba[idx == b].mean()), "observed": float(outcome[idx == b].mean()),
             "n": int((idx == b).sum())} for b in range(bins) if (idx == b).sum() >= min_n]


def loss_per_match(y, p) -> np.ndarray:
    return -np.log(np.clip(np.asarray(p)[np.arange(len(y)), np.asarray(y)], 1e-15, 1))


def brier_per_match(y, p) -> np.ndarray:
    return np.sum((np.asarray(p) - np.eye(3)[np.asarray(y)]) ** 2, axis=1)


def prob_metrics(y, p) -> dict:
    """Probability quality (independent of the named outcome)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    d = (y == 1).astype(float)
    pd_ = np.clip(p[:, 1], 1e-12, 1 - 1e-12)
    return {"n": int(len(y)), "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "brier": float(brier_per_match(y, p).mean()),
            "draw_log_loss": float(-np.mean(d * np.log(pd_) + (1 - d) * np.log(1 - pd_))),
            "draw_brier": float(np.mean((pd_ - d) ** 2)), "draw_ece": ece(d, pd_),
            "mean_p_draw": float(p[:, 1].mean()), "actual_draw_share": float(d.mean()),
            "p_draw_sd": float(p[:, 1].std()), "p_draw_q90": float(np.quantile(p[:, 1], 0.9)),
            "p_draw_max": float(p[:, 1].max())}


def class_metrics(y, pred) -> dict:
    """Quality of the named outcome."""
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    rec = {c: float(((pred == i) & (y == i)).sum() / max((y == i).sum(), 1)) for i, c in enumerate(CLASSES)}
    prec = {c: float(((pred == i) & (y == i)).sum() / max((pred == i).sum(), 1)) for i, c in enumerate(CLASSES)}
    f1 = f1_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "draw_precision": prec["D"], "draw_recall": rec["D"], "draw_f1": float(f1[1]),
            "recall": rec, "precision": prec, "f1": {c: float(f1[i]) for i, c in enumerate(CLASSES)},
            "predicted_draw_share": float((pred == 1).mean()), "actual_draw_share": float((y == 1).mean()),
            "predicted_share": {c: float((pred == i).mean()) for i, c in enumerate(CLASSES)},
            "actual_share": {c: float((y == i).mean()) for i, c in enumerate(CLASSES)},
            "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 7) -> dict:
    """Mean of (a - b) per match and its 95 % bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def roi_ci(profit: np.ndarray, n: int = 2000, seed: int = 11) -> dict:
    """ROI (mean profit per unit stake) with a 95 % bootstrap interval over bets."""
    profit = np.asarray(profit, float)
    if len(profit) == 0:
        return {"bets": 0, "roi": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boots = profit[rng.integers(0, len(profit), size=(n, len(profit)))].mean(1)
    return {"bets": int(len(profit)), "roi": float(profit.mean()), "profit": float(profit.sum()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_roi_gt_0": float((boots > 0).mean())}


def demargin(odds: np.ndarray, method: str = "proportional") -> np.ndarray:
    """Bookmaker probabilities from 1X2 odds (rows with three prices > 1); NaN elsewhere."""
    odds = np.asarray(odds, float)
    out = np.full(odds.shape, np.nan)
    ok = np.all(np.isfinite(odds) & (odds > 1), 1)
    inv = 1 / odds[ok]
    if method == "proportional":
        out[ok] = inv / inv.sum(1, keepdims=True)
    elif method == "power":                       # p_i = inv_i^k with sum 1 (solved per row by bisection)
        lo, hi = np.full(len(inv), 0.5), np.full(len(inv), 3.0)
        for _ in range(60):
            k = (lo + hi) / 2
            s = (inv ** k[:, None]).sum(1)
            lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
        out[ok] = inv ** ((lo + hi) / 2)[:, None]
        out[ok] /= out[ok].sum(1, keepdims=True)
    return out


# ---------------------------------------------------------------------------------------------- IO
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))
