"""Metrics for every task and market (identical definitions for every candidate)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, r2_score

from src.markets.distributions import corners_distribution, prob_over, reconcile_with_1x2, score_matrix

CLASSES = ("H", "D", "A")


def ece(outcome, proba, bins: int = 10) -> float:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(outcome[idx == b].mean() - proba[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(outcome, proba, bins: int = 10) -> list[dict]:
    outcome, proba = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return [{"predicted": float(proba[idx == b].mean()), "observed": float(outcome[idx == b].mean()),
             "n": int((idx == b).sum())} for b in range(bins) if (idx == b).sum() >= 20]


def classification(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    onehot = np.eye(3)[y]
    recall = {c: float(((pred == i) & (y == i)).sum() / max((y == i).sum(), 1)) for i, c in enumerate(CLASSES)}
    precision = {c: float(((pred == i) & (y == i)).sum() / max((pred == i).sum(), 1)) for i, c in enumerate(CLASSES)}
    return {
        "n": int(len(y)), "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "ece": float(np.mean([ece(y == i, p[:, i]) for i in range(3)])),
        "recall": recall, "draw_precision": precision["D"],
        "predicted_share": {c: float((pred == i).mean()) for i, c in enumerate(CLASSES)},
        "actual_share": {c: float((y == i).mean()) for i, c in enumerate(CLASSES)},
        "mean_probability": {c: float(p[:, i].mean()) for i, c in enumerate(CLASSES)},
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "reliability": {c: reliability(y == i, p[:, i]) for i, c in enumerate(CLASSES)},
    }


def counts(y: np.ndarray, mu: np.ndarray) -> dict:
    mu = np.clip(mu, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
    # distribution calibration: predicted vs observed share of 0, 1, 2, 3+ (Poisson)
    from scipy.stats import poisson
    shares = {}
    for k in (0, 1, 2):
        shares[str(k)] = {"predicted": float(poisson.pmf(k, mu).mean()), "observed": float((y == k).mean())}
    shares["3+"] = {"predicted": float(1 - poisson.cdf(2, mu).mean()), "observed": float((y >= 3).mean())}
    return {"n": int(len(y)), "mae": float(np.abs(y - mu).mean()), "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
            "r2": float(r2_score(y, mu)), "poisson_deviance": float(dev.mean()),
            "mean_actual": float(y.mean()), "mean_predicted": float(mu.mean()), "count_distribution": shares}


def binary(outcome: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    o, p = np.asarray(outcome, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": int(len(o)), "accuracy": float(((p > 0.5) == o).mean()), "log_loss": float(log_loss(o, p, labels=[0, 1])),
            "brier": float(np.mean((p - o) ** 2)), "ece": ece(o, p),
            "naive_log_loss": float(log_loss(o, np.full(len(o), base_rate), labels=[0, 1])),
            "naive_brier": float(np.mean((base_rate - o) ** 2)), "reliability": reliability(o, p)}


# -- markets derived from a score matrix -------------------------------------------------------
HANDICAP_LINES = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)


def derived_goal_markets(p1x2: np.ndarray, lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> dict:
    """Probabilities of totals, BTTS, double chance, Asian handicap and exact score from the
    1X2-reconciled Poisson score matrix of each match, plus the per-match outcomes."""
    out = {k: [] for k in ("over15", "over25", "over35", "btts", "dc_1x", "dc_x2", "dc_12", "score_ll")}
    hcp = {line: [] for line in HANDICAP_LINES}
    for k in range(len(lh)):
        m = reconcile_with_1x2(score_matrix(lh[k], la[k]), p1x2[k])
        i, j = np.indices(m.shape)
        tot = i + j
        out["over15"].append(m[tot > 1.5].sum())
        out["over25"].append(m[tot > 2.5].sum())
        out["over35"].append(m[tot > 3.5].sum())
        out["btts"].append(m[(i > 0) & (j > 0)].sum())
        ph, pd_, pa = m[i > j].sum(), m[i == j].sum(), m[i < j].sum()
        out["dc_1x"].append(ph + pd_), out["dc_x2"].append(pd_ + pa), out["dc_12"].append(ph + pa)
        h, a = min(int(hg[k]), m.shape[0] - 1), min(int(ag[k]), m.shape[1] - 1)
        out["score_ll"].append(-np.log(max(m[h, a], 1e-12)))
        for line in HANDICAP_LINES:
            margin = i + line - j
            hcp[line].append((m[margin > 0].sum(), m[margin == 0].sum()))
    return {**{k: np.array(v) for k, v in out.items()}, "handicap": hcp}


def goal_market_metrics(d: dict, hg: np.ndarray, ag: np.ndarray, base: dict) -> dict:
    tot = hg + ag
    res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    outcomes = {"over15": tot > 1.5, "over25": tot > 2.5, "over35": tot > 3.5, "btts": (hg > 0) & (ag > 0),
                "dc_1x": res != 2, "dc_x2": res != 0, "dc_12": res != 1}
    out = {k: binary(outcomes[k], d[k], base[k]) for k in outcomes}
    out["exact_score"] = {"log_loss": float(np.mean(d["score_ll"])), "n": int(len(hg))}
    for line, vals in d["handicap"].items():
        win = np.array([v[0] for v in vals])
        push = np.array([v[1] for v in vals])
        margin = hg + line - ag
        decided = margin != 0
        p_cond = win / np.clip(1 - push, 1e-9, None)               # P(home covers | not refunded)
        out[f"asian_handicap_{line:+g}"] = {**binary((margin > 0)[decided], p_cond[decided],
                                                     base.get(f"ah{line:+g}", 0.5)),
                                           "push_predicted": float(push.mean()), "push_actual": float((~decided).mean())}
    return out


def corner_line_probs(mu: np.ndarray, size: float | None, line: float = 9.5) -> np.ndarray:
    return np.array([prob_over(corners_distribution(m, size), line) for m in mu])
