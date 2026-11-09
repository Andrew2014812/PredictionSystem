"""Risk rules on a prepared candidate table (one chosen row per match), all on the same matches.

The production pool, the production score and the tie-breaking are the v6.1 vectorised copies
(experiments/v6_1/risk/strategies.py, checked against src/markets/recommendation.risk_prediction there and in
run.py here). Calibrators (``Cal``) work on the no-refund scale (model pc = p / (1 - p_push), market qc = de-vigged
P(win | no refund)) and are fitted on ALL history seasons before the evaluated season (expanding window), never on
the season itself; v6.1 fitted on the previous season only and used P(win) directly.

Alternative rankings choose among the priced pool candidates in the Risk odds band [2.50, 8.00] (so the pick stays
a Risk at odds of about 3+); a match without such a candidate keeps the production Risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from experiments.v6_1.risk import strategies as RS
from experiments.v6_2.main_calibration import methods as M

BAND = (2.50, 8.00)
Z_LCB = 1.0


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = RS.prepare(d)
    st = d["status"]
    d["y"] = np.where(d["decided"], (st == "WON").astype(float), np.nan)
    d["profit"] = np.where(d["priced"] & d["decided"], np.where(st == "WON", d["odds"] - 1, -1.0), np.nan)
    d["stake_profit"] = np.where(d["priced"] & (d["decided"] | (st == "PUSH")),
                                 np.where(st == "WON", d["odds"] - 1, np.where(st == "PUSH", 0.0, -1.0)), np.nan)
    push = d["push_p"].fillna(0.0).to_numpy(float)
    d["pc"] = d["p"].to_numpy(float) / np.clip(1 - push, 1e-6, None)
    d["qc"] = d["p_mkt"].to_numpy(float)                    # de-vigged P(win | no refund)
    d["ev_mkt"] = np.where(d["priced"], d["p_mkt_u"] * d["odds"] - (1 - push), np.nan)
    d["mi"] = pd.factorize(d["match_id"])[0]
    pool = d["cand"] & (d["p"] >= RS.RISK_MIN)
    d["ncand"] = pool.groupby(d["mi"]).transform("sum").to_numpy()
    d["ncand_priced"] = (pool & d["priced"]).groupby(d["mi"]).transform("sum").to_numpy()
    return d


class Cal:
    """Per-key Platt map of pc and a per-key logistic stack of (logit pc, logit qc), on the no-refund scale.

    Keys are the v6.1 selection keys (1X2:D, TOTAL_3.5:OVER, HANDICAP:H, ...); keys with fewer than MIN_N decided
    history rows use the pooled map. Only decided rows (won / lost) are fitted; a refund is neither."""

    MIN_N = 300

    def fit(self, rows: pd.DataFrame) -> "Cal":
        r = rows.loc[rows["decided"]]
        x, y = M.logit(r["pc"]), r["y"].to_numpy(float)
        qc = r["qc"].to_numpy(float)
        keys = r["key"].to_numpy()
        self.platt = {"*": LogisticRegression(C=1e4, max_iter=500).fit(x.reshape(-1, 1), y)}
        okq = np.isfinite(qc)
        self.stack = {"*": LogisticRegression(C=1e4, max_iter=500).fit(np.column_stack([x[okq], M.logit(qc[okq])]),
                                                                         y[okq])}
        for k in np.unique(keys):
            m = keys == k
            if m.sum() >= self.MIN_N and 0 < y[m].mean() < 1:
                self.platt[k] = LogisticRegression(C=1e4, max_iter=500).fit(x[m].reshape(-1, 1), y[m])
            mq = m & okq
            if mq.sum() >= self.MIN_N and 0 < y[mq].mean() < 1:
                self.stack[k] = LogisticRegression(C=1e4, max_iter=500).fit(
                    np.column_stack([x[mq], M.logit(qc[mq])]), y[mq])
        return self

    def apply(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = M.logit(d["pc"])
        qc = d["qc"].to_numpy(float)
        keys = d["key"].to_numpy()
        pcal = np.empty(len(d))
        pst = np.full(len(d), np.nan)
        for k in np.unique(keys):
            m = keys == k
            pm = self.platt.get(k, self.platt["*"])
            pcal[m] = pm.predict_proba(x[m].reshape(-1, 1))[:, 1]
            mq = m & np.isfinite(qc)
            if mq.any():
                sm = self.stack.get(k, self.stack["*"])
                pst[mq] = sm.predict_proba(np.column_stack([x[mq], M.logit(qc[mq])]))[:, 1]
        return pcal, np.where(np.isfinite(pst), pst, pcal)

    def params(self) -> dict:
        return {"platt": {k: [float(m.coef_[0, 0]), float(m.intercept_[0])] for k, m in self.platt.items()},
                "stack": {k: [float(m.coef_[0, 0]), float(m.coef_[0, 1]), float(m.intercept_[0])]
                          for k, m in self.stack.items()}}


class Fitted:
    """Calibrators fitted on the pool rows of the history seasons, and bin counts for the lower bound."""

    def __init__(self, hist: pd.DataFrame):
        pool = hist.loc[hist["cand"]]
        self.cal = Cal().fit(pool)
        pc_cal, pc_st = self.cal.apply(pool)
        dec = pool["decided"].to_numpy()
        b = np.floor(pc_st[dec] / 0.05).astype(int)
        self.counts = pd.Series(1, index=pd.MultiIndex.from_arrays([pool["key"].to_numpy()[dec], b])).groupby(
            level=[0, 1]).sum()

    def apply(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        push = d["push_p"].fillna(0.0).to_numpy(float)
        odds = d["odds"].to_numpy(float)
        pc_cal, pc_st = self.cal.apply(d)
        d["pc_cal"], d["pc_blend"] = pc_cal, pc_st
        d["p_cal"], d["p_blend"] = pc_cal * (1 - push), pc_st * (1 - push)
        for col in ("p_cal", "p_blend"):
            d[f"ev_{col}"] = np.where(d["priced"], d[col] * odds - (1 - push), np.nan)
        b = np.floor(pc_st / 0.05).astype(int)
        n = self.counts.reindex(pd.MultiIndex.from_arrays([d["key"].to_numpy(), b])).fillna(0).to_numpy(float)
        lb = np.clip(pc_st - Z_LCB * np.sqrt(pc_st * (1 - pc_st) / np.maximum(n, 1)), 0.001, 0.999) * (1 - push)
        d["p_lcb"] = lb
        d["ev_lcb"] = np.where(d["priced"], lb * odds - (1 - push), np.nan)
        pb = d["p_blend"].to_numpy(float)
        bb = odds - 1
        q = np.clip(1 - pb - push, 0, 1)
        with np.errstate(all="ignore"):
            d["kelly"] = np.where(d["priced"], (pb * bb - q) / (bb * np.clip(pb + q, 1e-9, None)), np.nan)
        d["edge"] = np.where(d["priced"] & d["p_mkt_u"].notna(), pb - d["p_mkt_u"], np.nan)
        return d


def _band(d):
    return d["cand"] & d["priced"] & (d["p"] >= RS.RISK_MIN) & (d["odds"] >= BAND[0]) & (d["odds"] <= BAND[1])


def ranked(d: pd.DataFrame, key_col: str, require=None) -> pd.DataFrame:
    """Best ``key_col`` among the priced pool candidates in the Risk band; else the production Risk."""
    first = _band(d) & d[key_col].notna()
    if require is not None:
        first &= require
    key = np.where(first, d[key_col].fillna(-9), RS.score(d))
    return RS.choose(d, [first, d["cand"]], key)


CATALOGUE = {
    "Текущий Risk (v6.1)": ("base", RS.current),
    "v6.1: стек с рынком, пол EV ≥ 0": ("v61", lambda d: RS.calibrated(d, 0.0, "p_blend")),
    "Окно 2.50–3.50": ("window", lambda d: RS.window(d, (2.50, 3.50))),
    "Окно 3.00–4.50": ("window", lambda d: RS.window(d, (3.00, 4.50))),
    "Макс. EV (сырая p)": ("rank", lambda d: ranked(d, "ev")),
    "Макс. EV (калибр. p)": ("rank", lambda d: ranked(d, "ev_p_cal")),
    "Макс. EV (стек с рынком)": ("rank", lambda d: ranked(d, "ev_p_blend")),
    "Макс. Келли (стек)": ("rank", lambda d: ranked(d, "kelly")),
    "Макс. нижняя граница EV": ("rank", lambda d: ranked(d, "ev_lcb")),
    "Макс. перевес над рынком": ("rank", lambda d: ranked(d, "edge")),
    "Макс. вероятность (стек)": ("rank", lambda d: ranked(d, "p_blend")),
    "Только рынок: макс. q": ("rank", lambda d: ranked(d, "p_mkt_u")),
}
# "fewer bets": bet only when the ranking key passes a fixed threshold (no look-ahead), or the top N of a day
THRESHOLDS = {"Макс. EV (стек с рынком)": ("ev_p_blend", (0.0, 0.05, 0.10)),
              "Макс. EV (сырая p)": ("ev", (0.0, 0.10, 0.20)),
              "Макс. перевес над рынком": ("edge", (0.0, 0.02, 0.04))}
TOP_PER_DAY = (1, 3, 5)


def picks_frame(d: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    cols = ["mi", "match_id", "date", "league", "market", "selection", "group", "key", "p", "pc", "push_p", "odds",
            "priced", "status", "y", "profit", "stake_profit", "ev", "p_mkt_u", "ev_mkt", "ncand", "ncand_priced"]
    extra = [c for c in ("p_cal", "p_blend", "ev_p_cal", "ev_p_blend", "edge", "kelly", "ev_lcb") if c in chosen]
    return chosen[cols + extra].sort_values("mi").reset_index(drop=True)
