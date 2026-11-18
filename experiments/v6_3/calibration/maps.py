"""Calibration maps of the study (all on the no-refund scale pc, outcome y in {0, 1}).

Elementary maps
    raw          identity (the production probability)
    platt        y ~ sigmoid(a + b logit pc)                                  (Platt 1999)
    temperature  pc' = sigmoid(logit pc / T), one parameter                     (Guo et al. 2017, binary form)
    beta         y ~ sigmoid(c + a ln pc - b ln(1 - pc)), a, b >= 0             (Kull et al. 2017; implemented
                 here as a logistic regression on [ln pc, -ln(1 - pc)]; a negative coefficient drops its feature
                 and the regression is refitted, as in the original method)
    isotonic     monotone step function (PAV), clipped to [0.005, 0.995]
    histbin      20 equal-mass bins, Laplace-smoothed frequency per bin

Scopes
    global   one map for every row
    group    one map per market group (1X2, totals, BTTS, double chance, Asian handicaps, corners, ...)
    key      one map per group x selection (1X2|D, TOTAL|OVER, HANDICAP|A, ...)
A scope level with fewer than MIN_FIT settled rows falls back to the coarser level (key -> group -> global).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
MIN_FIT = 300
METHODS = ["raw", "platt", "temperature", "beta", "isotonic", "histbin"]
SCOPES = ["global", "group", "key"]
METHOD_RU = {"raw": "без калибровки", "platt": "Platt", "temperature": "температурная", "beta": "бета-калибровка",
             "isotonic": "изотоническая", "histbin": "гистограммная"}
SCOPE_RU = {"global": "общая", "group": "по группам рынков", "key": "по группе и исходу"}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


class Raw:
    def fit(self, pc, y):
        return self

    def predict(self, pc):
        return np.clip(np.asarray(pc, float), EPS, 1 - EPS)


class Platt:
    def fit(self, pc, y):
        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(logit(pc).reshape(-1, 1), y)
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return self

    def predict(self, pc):
        return sigmoid(self.a + self.b * logit(pc))


class Temperature:
    def fit(self, pc, y):
        z = logit(pc)
        y = np.asarray(y, float)

        def nll(lt):
            q = np.clip(sigmoid(z / np.exp(lt)), 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        self.T = float(np.exp(minimize_scalar(nll, bounds=(-2.5, 2.5), method="bounded").x))
        return self

    def predict(self, pc):
        return sigmoid(logit(pc) / self.T)


class Beta:
    @staticmethod
    def _x(pc):
        p = np.clip(np.asarray(pc, float), EPS, 1 - EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, pc, y):
        x = self._x(pc)
        keep = [0, 1]
        for _ in range(2):
            lr = LogisticRegression(C=1e4, max_iter=1000).fit(x[:, keep], y)
            coef = lr.coef_[0]
            if (coef >= 0).all() or len(keep) == 1:
                break
            keep = [keep[int(np.argmax(coef))]]
        self.w = np.zeros(2)
        self.w[keep] = np.maximum(lr.coef_[0], 0) if len(keep) == 1 else lr.coef_[0]
        self.c = float(lr.intercept_[0])
        return self

    def predict(self, pc):
        return sigmoid(self.c + self._x(pc) @ self.w)


class Isotonic:
    def fit(self, pc, y):
        self.m = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(np.asarray(pc, float), y)
        return self

    def predict(self, pc):
        return self.m.predict(np.asarray(pc, float))


class HistBin:
    def __init__(self, bins: int = 20):
        self.bins = bins

    def fit(self, pc, y):
        pc, y = np.asarray(pc, float), np.asarray(y, float)
        edges = np.unique(np.quantile(pc, np.linspace(0, 1, self.bins + 1)))
        self.edges = edges[1:-1]
        b = np.searchsorted(self.edges, pc, side="right")
        n = np.bincount(b, minlength=len(self.edges) + 1)
        k = np.bincount(b, weights=y, minlength=len(self.edges) + 1)
        prior = y.mean()
        self.rate = (k + 2 * prior) / (n + 2)
        return self

    def predict(self, pc):
        return np.clip(self.rate[np.searchsorted(self.edges, np.asarray(pc, float), side="right")], 0.005, 0.995)


ELEMENTARY = {"raw": Raw, "platt": Platt, "temperature": Temperature, "beta": Beta, "isotonic": Isotonic,
              "histbin": HistBin}


class ScopedMap:
    """A method fitted per scope level with fallback to coarser levels."""

    def __init__(self, method: str, scope: str):
        self.method, self.scope = method, scope

    def _levels(self):
        return {"global": ["global"], "group": ["group", "global"], "key": ["key", "group", "global"]}[self.scope]

    def fit(self, d: pd.DataFrame):
        d = d.loc[d["y"].notna()]
        self.maps = {}
        for level in self._levels():
            keys = pd.Series("all", index=d.index) if level == "global" else d[level]
            for k, idx in keys.groupby(keys).groups.items():
                if len(idx) >= MIN_FIT or level == "global":
                    g = d.loc[idx]
                    if g["y"].nunique() < 2:
                        continue
                    self.maps[(level, k)] = ELEMENTARY[self.method]().fit(g["pc"].to_numpy(), g["y"].to_numpy())
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        """Calibrated pc of every row (rows with no fitted level keep their pc)."""
        out = np.full(len(d), np.nan)
        pc = d["pc"].to_numpy(float)
        for level in self._levels():
            keys = np.full(len(d), "all", dtype=object) if level == "global" else d[level].to_numpy()
            todo = np.isnan(out)
            for k in pd.unique(keys[todo]):
                m = self.maps.get((level, k))
                if m is None:
                    continue
                sel = todo & (keys == k)
                out[sel] = m.predict(pc[sel])
        out = np.where(np.isnan(out), pc, out)
        return np.clip(out, EPS, 1 - EPS)


def to_p(d: pd.DataFrame, pc_cal: np.ndarray) -> np.ndarray:
    """Back to the production scale: p = pc (1 - P(push))."""
    return pc_cal * (1 - d["push_probability"].to_numpy(float))
