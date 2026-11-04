"""Probability transforms fitted on an INNER season and applied to the next season.

* calibration keys: group | selection kind (e.g. TOTAL|OVER, HANDICAP|A, 1X2|D);
* Asian whole lines are calibrated on the no-refund scale p / (1 - p_push) and converted back;
* isotonic and Platt (logistic on logit p) maps; a selection-conditional map fitted on the chosen Main picks;
* linear blend of the model with the de-margined bookmaker probability, weight per group by inner log loss;
* a logistic "disagreement" model y ~ logit q + (logit p - logit q);
* a logistic meta-model for the betting decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
MIN_FIT = 200


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def cal_key(df: pd.DataFrame) -> np.ndarray:
    g = df["group"].to_numpy().astype(object)
    s = df["selection"].to_numpy().astype(object)
    s = np.where(g == "DOUBLE_CHANCE", "ANY", s)
    return (g + "|" + s).astype(str)


def _cond(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    return p / np.clip(1 - df["p_push"].to_numpy(float), 1e-6, None)


def _decided(df: pd.DataFrame) -> np.ndarray:
    return np.isin(df["status"].to_numpy(), [0, 1])


class Calibrator:
    """Per-key map of the no-refund probability; keys with < MIN_FIT rows use the pooled group map."""

    def __init__(self, kind: str = "isotonic"):
        self.kind, self.maps = kind, {}

    def _fit_one(self, x, y):
        if self.kind == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
            m.fit(x, y)
            return m
        lr = LogisticRegression(C=1e4, max_iter=500)
        lr.fit(logit(x).reshape(-1, 1), y)
        return lr

    def _apply_one(self, m, x):
        if self.kind == "isotonic":
            return m.predict(x)
        return m.predict_proba(logit(x).reshape(-1, 1))[:, 1]

    def fit(self, df: pd.DataFrame, p_col: str = "p", mask: np.ndarray | None = None) -> "Calibrator":
        ok = _decided(df) if mask is None else (_decided(df) & mask)
        d = df.loc[ok]
        x = _cond(d, d[p_col].to_numpy(float))
        y = (d["status"].to_numpy() == 1).astype(float)
        keys, groups = cal_key(d), d["group"].to_numpy()
        for g in np.unique(groups):
            m = groups == g
            if m.sum() >= MIN_FIT:
                self.maps["G:" + g] = self._fit_one(x[m], y[m])
        for k in np.unique(keys):
            m = keys == k
            if m.sum() >= MIN_FIT:
                self.maps[k] = self._fit_one(x[m], y[m])
        if len(x) >= MIN_FIT:
            self.maps["ALL"] = self._fit_one(x, y)
        return self

    def transform(self, df: pd.DataFrame, p_col: str = "p") -> np.ndarray:
        p = df[p_col].to_numpy(float)
        x = _cond(df, p)
        out = x.copy()
        keys, groups = cal_key(df), df["group"].to_numpy()
        done = np.zeros(len(df), bool)
        for k in np.unique(keys):
            m = keys == k
            mp = self.maps.get(k) or self.maps.get("G:" + k.split("|")[0]) or self.maps.get("ALL")
            if mp is not None:
                out[m] = self._apply_one(mp, x[m])
                done[m] = True
        _ = groups
        return np.clip(out, 0.0, 1.0) * (1 - df["p_push"].to_numpy(float))


def market_q(df: pd.DataFrame, book: str = "prod", method: str = "shin") -> np.ndarray:
    return df[f"q_{book}_{method}"].to_numpy(float)


BLEND_GROUPS = {"1X2": "1X2", "DOUBLE_CHANCE": "1X2", "TOTAL": "TOTAL", "BTTS": "TOTAL", "CORNERS": "TOTAL",
                "HANDICAP": "HANDICAP"}


class Blend:
    """p' = w_g p + (1 - w_g) q on the no-refund scale, w_g on a 0.05 grid by inner log loss.

    Groups without fitted weight use the pooled weight (BLEND_GROUPS maps BTTS / corners to the totals weight,
    double chance to 1X2). Rows without a market probability keep the model probability."""

    def __init__(self, book="prod", method="shin", fixed: float | None = None):
        self.book, self.method, self.fixed, self.w = book, method, fixed, {}

    def fit(self, df: pd.DataFrame, p_col: str = "p") -> "Blend":
        if self.fixed is not None:
            return self
        q = market_q(df, self.book, self.method)
        ok = _decided(df) & np.isfinite(q)
        d = df.loc[ok]
        x, qq = _cond(d, d[p_col].to_numpy(float)), _cond(d, q[ok])
        y = (d["status"].to_numpy() == 1).astype(float)
        grid = np.round(np.arange(0, 1.0001, 0.05), 2)
        bg = d["group"].map(BLEND_GROUPS).to_numpy()
        for g in list(np.unique(bg)) + ["ALL"]:
            m = np.ones(len(d), bool) if g == "ALL" else bg == g
            if m.sum() < MIN_FIT:
                continue
            ll = []
            for w in grid:
                pb = np.clip(w * x[m] + (1 - w) * qq[m], EPS, 1 - EPS)
                ll.append(-np.mean(y[m] * np.log(pb) + (1 - y[m]) * np.log(1 - pb)))
            self.w[g] = float(grid[int(np.argmin(ll))])
        return self

    def weight(self, group: str) -> float:
        if self.fixed is not None:
            return self.fixed
        return self.w.get(BLEND_GROUPS.get(group, "ALL"), self.w.get("ALL", 1.0))

    def transform(self, df: pd.DataFrame, p_col: str = "p") -> np.ndarray:
        p = df[p_col].to_numpy(float)
        q = market_q(df, self.book, self.method)
        w = df["group"].map(self.weight).to_numpy(float)
        push = df["p_push"].to_numpy(float)
        x, qq = _cond(df, p), _cond(df, q)
        out = np.where(np.isfinite(qq), w * x + (1 - w) * qq, x)
        return out * (1 - push)


def disagreement_fit(df: pd.DataFrame, p_col="p", book="prod", method="shin") -> dict:
    """Logistic y ~ a + b logit(q) + c (logit p - logit q) per group, with standard errors (statsmodels)."""
    import statsmodels.api as sm
    q = market_q(df, book, method)
    ok = _decided(df) & np.isfinite(q)
    d = df.loc[ok]
    x, qq = _cond(d, d[p_col].to_numpy(float)), _cond(d, q[ok])
    y = (d["status"].to_numpy() == 1).astype(float)
    out = {}
    groups = d["group"].to_numpy()
    for g in list(np.unique(groups)) + ["ALL"]:
        m = np.ones(len(d), bool) if g == "ALL" else groups == g
        if m.sum() < MIN_FIT:
            continue
        X = sm.add_constant(np.column_stack([logit(qq[m]), logit(x[m]) - logit(qq[m])]))
        try:
            r = sm.Logit(y[m], X).fit(disp=0)
            out[g] = {"n": int(m.sum()), "const": float(r.params[0]), "b_market": float(r.params[1]),
                      "c_disagreement": float(r.params[2]), "se_market": float(r.bse[1]),
                      "se_disagreement": float(r.bse[2]),
                      "ci_disagreement": [float(r.params[2] - 1.96 * r.bse[2]), float(r.params[2] + 1.96 * r.bse[2])],
                      "mean_abs_disagreement": float(np.mean(np.abs(x[m] - qq[m])))}
        except Exception as exc:          # perfect separation etc.
            out[g] = {"n": int(m.sum()), "error": str(exc)}
    return out


META_FEATURES = ("lp", "lq", "dis", "lodds", "push", "ncand", "g_1X2", "g_TOTAL", "g_HANDICAP", "under", "line")


def meta_frame(df: pd.DataFrame, p_col: str, q: np.ndarray, ncand: np.ndarray) -> np.ndarray:
    p = df[p_col].to_numpy(float)
    x, qq = _cond(df, p), _cond(df, q)
    qq = np.where(np.isfinite(qq), qq, x)
    g = df["group"].to_numpy()
    cols = [logit(x), logit(qq), logit(x) - logit(qq), np.log(df["odds"].to_numpy(float)),
            df["p_push"].to_numpy(float), np.log1p(ncand), (g == "1X2").astype(float), (g == "TOTAL").astype(float),
            (g == "HANDICAP").astype(float), (df["selection"].to_numpy() == "UNDER").astype(float),
            np.nan_to_num(df["line"].to_numpy(float))]
    return np.column_stack(cols)


class Meta:
    """Logistic P(win | decided) of a chosen pick from model, market and price information (inner season)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Meta":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit((X - self.mu) / self.sd, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        bad = ~np.isfinite(X).all(1)
        p = self.lr.predict_proba((np.nan_to_num(X) - self.mu) / self.sd)[:, 1]
        return np.where(bad, np.nan, p)
