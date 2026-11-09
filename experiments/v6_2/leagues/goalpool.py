"""League-level corrections of the goal rates with partial pooling (Poisson–gamma empirical Bayes).

For season s the factors are estimated on the walk-forward predictions of 2022/23 ... s-1 (B0, seed 42), for the
home rate and the away rate separately:

    G1  global factor        F = Σ y / Σ λ                                                     (control)
    G2  per league, unpooled f_L = Σ_L y / Σ_L λ
    G3  per league, EB       y ~ Poisson(f_L λ),  f_L ~ Gamma(α, β) with mean F and variance τ²
                             (α = F²/τ², β = F/τ²),  f_L = (α + Σ_L y) / (β + Σ_L λ)
         τ² = max(0, var_L(f̂_L) − mean_L(f̂_L / Σ_L λ))   (method of moments)
The corrected rates are λ' = f λ; the 1X2 classifier is unchanged (Main reconciles the score matrix with it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VARIANTS = {"G1": "голы: глобальный множитель (контроль)", "G2": "голы: множитель лиги без сжатия",
            "G3": "голы: множитель лиги, EB (Пуассон–гамма)"}


def factors(y: np.ndarray, lam: np.ndarray, league: np.ndarray) -> dict:
    d = pd.DataFrame({"y": y, "l": lam, "g": league}).groupby("g")[["y", "l"]].sum()
    F = float(d["y"].sum() / d["l"].sum())
    raw = d["y"] / d["l"]
    tau2 = max(0.0, float(raw.var(ddof=1) - (raw / d["l"]).mean()))
    if tau2 > 0:
        a, b = F * F / tau2, F / tau2
        eb = (a + d["y"]) / (b + d["l"])
    else:
        eb = pd.Series(F, index=d.index)
    return {"F": F, "tau2": tau2, "raw": raw.to_dict(), "eb": eb.to_dict(), "sum_lambda": d["l"].to_dict()}


class GoalPool:
    def __init__(self, hist: pd.DataFrame):
        """hist: rows with league, home_goals, away_goals, lh, la of earlier seasons."""
        lg = hist["league"].to_numpy()
        self.h = factors(hist["home_goals"].to_numpy(float), hist["lh"].to_numpy(float), lg)
        self.a = factors(hist["away_goals"].to_numpy(float), hist["la"].to_numpy(float), lg)

    def transform(self, lh, la, league, variant: str):
        league = np.asarray(league)

        def fac(info):
            if variant == "G1":
                return np.full(len(league), info["F"])
            table = info["raw"] if variant == "G2" else info["eb"]
            return np.array([table.get(x, info["F"]) for x in league])
        return np.asarray(lh, float) * fac(self.h), np.asarray(la, float) * fac(self.a)

    def info(self) -> dict:
        return {"home": self.h, "away": self.a}
