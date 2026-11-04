"""Calibration of selection probabilities, fitted on the PREVIOUS season only (walk-forward).

* ``platt``: per selection type (key, e.g. 1X2:D, TOTAL_3.5:OVER, HANDICAP:H) a logistic regression
  of the outcome (selection WON) on logit(p) of the rows with p in FIT_RANGE; fewer than MIN_N rows ->
  identity. Output column ``p_cal``.
* ``blend``: per key with a de-vigged market probability, a logistic regression of the outcome on
  [logit(p_model), logit(p_market)] (rows with a complete book). A priced selection of a key without
  its own blend (e.g. BTTS or corners in 2026/27, never priced in the Football-Data history) uses the
  blend pooled over all keys of the previous season; without a complete book ``p_cal``. Output ``p_blend``.
* ``reliability``: per key, the previous season's calibration in the Risk range p in RANGE:
  n, mean p, observed frequency and z = (observed - mean p) / sqrt(sum p(1 - p)) / n. A key is
  "reliable" (Market-specific Risk) if n >= REL_MIN_N and z >= -1.96, i.e. its probabilities were not
  significantly over-estimated. Output column ``reliable``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FIT_RANGE = (0.05, 0.75)
RANGE = (0.12, 0.40)
MIN_N = 300
REL_MIN_N = 100


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _settled(df):
    return df.loc[df["status"].notna()]


def _fit(X, y):
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(C=1e6, max_iter=1000).fit(X, y)


class Calibrator:
    def __init__(self, market_col: str = "p_mkt"):
        """``market_col``: "p_mkt" (as pre-registered: for a whole-line Asian handicap the de-vigged
        price is P(win | no refund)) or "p_mkt_u" (post-hoc correction: P(win) = P(win | no refund)
        x (1 - model refund probability))."""
        self.market_col = market_col
        self.platt, self.blend, self.rel = {}, {}, {}
        self.pooled_blend = None

    def fit(self, train: pd.DataFrame) -> "Calibrator":
        t = _settled(train)
        pb = t.loc[t["p"].between(*FIT_RANGE) & t[self.market_col].notna()]
        if len(pb) >= MIN_N:
            m = _fit(np.column_stack([logit(pb["p"]), logit(pb[self.market_col])]), pb["won"].to_numpy())
            if m is not None:
                self.pooled_blend = (float(m.coef_[0][0]), float(m.coef_[0][1]), float(m.intercept_[0]), int(len(pb)))
        for key, g in t.groupby("key"):
            r = g.loc[g["p"].between(*FIT_RANGE)]
            if len(r) >= MIN_N:
                m = _fit(logit(r["p"]).reshape(-1, 1), r["won"].to_numpy())
                if m is not None:
                    self.platt[key] = (float(m.coef_[0][0]), float(m.intercept_[0]), int(len(r)))
            b = r.loc[r[self.market_col].notna()]
            if len(b) >= MIN_N:
                m = _fit(np.column_stack([logit(b["p"]), logit(b[self.market_col])]), b["won"].to_numpy())
                if m is not None:
                    self.blend[key] = (float(m.coef_[0][0]), float(m.coef_[0][1]), float(m.intercept_[0]),
                                       int(len(b)))
            q = g.loc[g["p"].between(*RANGE)]
            n = len(q)
            if n:
                mp, obs = float(q["p"].mean()), float(q["won"].mean())
                se = float(np.sqrt((q["p"] * (1 - q["p"])).sum())) / n
                z = (obs - mp) / se if se > 0 else 0.0
                self.rel[key] = {"n": n, "mean_p": mp, "observed": obs, "z": z,
                                 "reliable": bool(n >= REL_MIN_N and z >= -1.96)}
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lp = logit(df["p"])
        a = df["key"].map(lambda k: self.platt.get(k, (1.0, 0.0, 0))[0]).to_numpy(float)
        c = df["key"].map(lambda k: self.platt.get(k, (1.0, 0.0, 0))[1]).to_numpy(float)
        df["p_cal"] = 1 / (1 + np.exp(-(a * lp + c)))
        has = df[self.market_col].notna().to_numpy() & (df["key"].isin(self.blend.keys()).to_numpy()
                                                   | (self.pooled_blend is not None))
        fallback = self.pooled_blend or (1.0, 0.0, 0.0, 0)
        bm = df["key"].map(lambda k: self.blend.get(k, fallback)).tolist()
        b1 = np.array([x[0] for x in bm]); b2 = np.array([x[1] for x in bm]); b0 = np.array([x[2] for x in bm])
        z = b1 * lp + b2 * logit(df[self.market_col].fillna(0.5)) + b0
        df["p_blend"] = np.where(has, 1 / (1 + np.exp(-z)), df["p_cal"])
        for col in ("p_cal", "p_blend"):
            df[f"ev_{col}"] = np.where(df["priced"], df[col] * df["odds"] - (1 - df["push_p"]), np.nan)
        df["reliable"] = df["key"].map(lambda k: self.rel.get(k, {}).get("reliable", False)).astype(bool)
        return df

    def to_dict(self) -> dict:
        return {"platt": {k: {"slope": v[0], "intercept": v[1], "n": v[2]} for k, v in self.platt.items()},
                "blend": {k: {"w_model": v[0], "w_market": v[1], "intercept": v[2], "n": v[3]}
                          for k, v in self.blend.items()},
                "pooled_blend": None if self.pooled_blend is None else dict(zip(
                    ("w_model", "w_market", "intercept", "n"), self.pooled_blend)),
                "reliability": self.rel}
