"""Calibration maps, selection-bias corrections and ranking functions of the Main-calibration study.

Every probability is handled on the no-refund scale pc = p / (1 - p_push) (identical to p except for Asian
whole lines); a calibrated pc' is converted back with p' = pc' (1 - p_push). Outcome y = 1 won, 0 lost (refunds
are not decided and are left out of every fit and metric, as production leaves them out of the hit rate).

Post-selection maps (the Main pick does not change, only its stated probability):
    PS-Platt     y ~ a + b logit pc, fitted on the baseline Main picks of the history seasons
    PS-Iso       isotonic regression of y on pc (same picks)
    PS-Group     Platt per market group (pooled map for groups with < MIN_FIT picks)
    PS-Aware     logistic y ~ logit pc + log(1 + n_cand) + group + priced (selection-aware)
    PS-OS        order-statistic shrinkage: logit pc' = logit pc - c e(n_cand), e(n) = expected maximum of n
                 standard normals (Blom), one parameter c by maximum likelihood
    PS-Market    y ~ a + b1 logit pc + b2 logit qc on priced picks (qc = de-margined Bet365, Shin); PS-Platt
                 for picks without a full book
    PS-EB-League PS-Platt plus a league offset shrunk towards 0 by empirical Bayes (method of moments)
Pre-selection maps (every candidate is calibrated, then the production rule chooses on the new probabilities):
    KeyCal(kind, key) — Platt or isotonic per market key (``group`` = group|selection, ``market`` =
    market|selection, i.e. each line separately), pooled group / global maps below MIN_FIT rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
MIN_FIT = 300
GROUPS = ("1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "HANDICAP", "CORNERS")


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def blom(n) -> np.ndarray:
    """Expected maximum of n iid N(0, 1) (Blom approximation); 0 for n <= 1."""
    n = np.maximum(np.asarray(n, float), 1)
    return np.where(n <= 1, 0.0, norm.ppf((n - 0.375) / (n + 0.25)))


# ------------------------------------------------------------------------------------------------ elementary maps
class Platt:
    def fit(self, pc, y):
        lr = LogisticRegression(C=1e4, max_iter=500)
        lr.fit(logit(pc).reshape(-1, 1), np.asarray(y, float))
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return self

    def predict(self, pc):
        return sigmoid(self.a + self.b * logit(pc))

    def params(self):
        return {"a": self.a, "b": self.b}


class Iso:
    def fit(self, pc, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(pc, np.asarray(y, float))
        return self

    def predict(self, pc):
        return self.m.predict(np.asarray(pc, float))

    def params(self):
        return {}


# ------------------------------------------------------------------------------------------------ post-selection
class PSGroup:
    def fit(self, f: pd.DataFrame):
        self.pool = Platt().fit(f["pc"], f["y"])
        self.maps = {g: Platt().fit(x["pc"], x["y"]) for g, x in f.groupby("group") if len(x) >= MIN_FIT}
        return self

    def predict(self, f: pd.DataFrame):
        out = self.pool.predict(f["pc"])
        for g, m in self.maps.items():
            k = (f["group"] == g).to_numpy()
            out[k] = m.predict(f["pc"].to_numpy()[k])
        return out

    def params(self):
        return {"pool": self.pool.params(), **{g: m.params() for g, m in self.maps.items()}}


def _aware_x(f: pd.DataFrame) -> np.ndarray:
    cols = [logit(f["pc"]), np.log1p(f["ncand"].to_numpy(float)), f["priced"].to_numpy(float)]
    cols += [(f["group"] == g).to_numpy(float) for g in GROUPS[1:]]
    return np.column_stack(cols)


class PSAware:
    def fit(self, f: pd.DataFrame):
        X = _aware_x(f)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit((X - self.mu) / self.sd, f["y"].to_numpy(float))
        return self

    def predict(self, f: pd.DataFrame):
        return self.lr.predict_proba((_aware_x(f) - self.mu) / self.sd)[:, 1]

    def params(self):
        names = ["logit_p", "log1p_ncand", "priced"] + [f"g_{g}" for g in GROUPS[1:]]
        return {"intercept": float(self.lr.intercept_[0]),
                **{n: float(c / s) for n, c, s in zip(names, self.lr.coef_[0], self.sd)}}


class PSOrderStat:
    """logit pc' = logit pc - c e(n_cand)."""

    def fit(self, f: pd.DataFrame):
        x, e, y = logit(f["pc"]), blom(f["ncand"]), f["y"].to_numpy(float)

        def nll(c):
            q = np.clip(sigmoid(x - c * e), EPS, 1 - EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        self.c = float(minimize_scalar(nll, bounds=(-1.0, 2.0), method="bounded").x)
        return self

    def predict(self, f: pd.DataFrame):
        return sigmoid(logit(f["pc"]) - self.c * blom(f["ncand"]))

    def params(self):
        return {"c": self.c}


class PSMarket:
    def fit(self, f: pd.DataFrame):
        ok = np.isfinite(f["qc"].to_numpy(float))
        self.pool = Platt().fit(f["pc"], f["y"])
        g = f.loc[ok]
        self.lr = None
        if len(g) >= MIN_FIT:
            X = np.column_stack([logit(g["pc"]), logit(g["qc"])])
            self.lr = LogisticRegression(C=1e4, max_iter=500).fit(X, g["y"].to_numpy(float))
        return self

    def predict(self, f: pd.DataFrame):
        out = self.pool.predict(f["pc"])
        ok = np.isfinite(f["qc"].to_numpy(float))
        if self.lr is not None and ok.any():
            g = f.loc[ok]
            out[ok] = self.lr.predict_proba(np.column_stack([logit(g["pc"]), logit(g["qc"])]))[:, 1]
        return out

    def params(self):
        if self.lr is None:
            return {"pool": self.pool.params()}
        return {"intercept": float(self.lr.intercept_[0]), "b_model": float(self.lr.coef_[0, 0]),
                "b_market": float(self.lr.coef_[0, 1]), "pool": self.pool.params()}


class PSEBLeague:
    """PS-Platt + league offset u_L = B_L mean residual_L, B_L = tau^2 / (tau^2 + s^2 / n_L)."""

    def fit(self, f: pd.DataFrame):
        self.base = Platt().fit(f["pc"], f["y"])
        r = f["y"].to_numpy(float) - self.base.predict(f["pc"])
        s2 = float(np.mean(self.base.predict(f["pc"]) * (1 - self.base.predict(f["pc"]))))
        g = pd.DataFrame({"league": f["league"].to_numpy(), "r": r}).groupby("league")["r"].agg(["mean", "size"])
        tau2 = max(float(np.var(g["mean"]) - np.mean(s2 / g["size"])), 0.0)
        g["B"] = tau2 / (tau2 + s2 / g["size"]) if tau2 > 0 else 0.0
        self.u = (g["B"] * g["mean"]).to_dict()
        self.tau = float(np.sqrt(tau2))
        return self

    def predict(self, f: pd.DataFrame):
        u = f["league"].map(self.u).fillna(0.0).to_numpy(float)
        return np.clip(self.base.predict(f["pc"]) + u, 0.005, 0.995)

    def params(self):
        return {"tau": self.tau, "base": self.base.params(),
                "max_abs_offset": float(max([abs(v) for v in self.u.values()] or [0]))}


class Wrap:
    """A map that only needs pc (Platt / Iso) as a frame-level predictor."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, f):
        self.inner.fit(f["pc"].to_numpy(float), f["y"].to_numpy(float))
        return self

    def predict(self, f):
        return self.inner.predict(f["pc"].to_numpy(float))

    def params(self):
        return self.inner.params()


POST = {"PS-Platt": lambda: Wrap(Platt()), "PS-Iso": lambda: Wrap(Iso()), "PS-Group": PSGroup,
        "PS-Aware": PSAware, "PS-OS": PSOrderStat, "PS-Market": PSMarket, "PS-EB-League": PSEBLeague}


# ------------------------------------------------------------------------------------------------ pre-selection
def key_of(df: pd.DataFrame, key: str) -> np.ndarray:
    g = df["group"].to_numpy().astype(object)
    s = df["selection"].to_numpy().astype(object)
    if key == "group":
        s = np.where(g == "DOUBLE_CHANCE", "ANY", s)
        return (g + "|" + s).astype(str)
    return (df["market"].to_numpy().astype(object) + "|" + s).astype(str)


class KeyCal:
    """Per-key map of pc fitted on all decided candidates of the history seasons."""

    def __init__(self, kind: str = "platt", key: str = "market"):
        self.kind, self.key, self.maps = kind, key, {}

    def _new(self):
        return Platt() if self.kind == "platt" else Iso()

    def fit(self, rows: pd.DataFrame):
        """rows: decided candidates with columns group, market, selection, pc, y."""
        x, y = rows["pc"].to_numpy(float), rows["y"].to_numpy(float)
        keys, groups = key_of(rows, self.key), rows["group"].to_numpy()
        self.maps["ALL"] = self._new().fit(x, y)
        for g in np.unique(groups):
            m = groups == g
            if m.sum() >= MIN_FIT:
                self.maps["G:" + g] = self._new().fit(x[m], y[m])
        for k in np.unique(keys):
            m = keys == k
            if m.sum() >= MIN_FIT and 0 < y[m].mean() < 1:
                self.maps[k] = self._new().fit(x[m], y[m])
        self.n = {k: int((keys == k).sum()) for k in np.unique(keys)}
        return self

    def transform_c(self, df: pd.DataFrame, pc: np.ndarray) -> np.ndarray:
        out = pc.copy()
        keys = key_of(df, self.key)
        groups = df["group"].to_numpy()
        for k in np.unique(keys):
            m = keys == k
            g = groups[np.argmax(m)]
            mp = self.maps.get(k) or self.maps.get("G:" + g) or self.maps["ALL"]
            out[m] = mp.predict(pc[m])
        return np.clip(out, 0.001, 0.999)


def lcb_c(df: pd.DataFrame, pc_cal: np.ndarray, hist: pd.DataFrame, z: float = 1.0, width: float = 0.05) -> np.ndarray:
    """Lower bound pc - z sqrt(pc (1 - pc) / n_kb): n_kb = history candidates of the same market key in the same
    probability bin (calibration uncertainty; few observations -> larger penalty)."""
    bins = np.floor(pc_cal / width).astype(int)
    hk = key_of(hist, "market")
    hb = np.floor(hist["pc_cal"].to_numpy(float) / width).astype(int)
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([hk, hb])).groupby(level=[0, 1]).sum()
    idx = pd.MultiIndex.from_arrays([key_of(df, "market"), bins])
    n = counts.reindex(idx).fillna(0).to_numpy(float)
    return np.clip(pc_cal - z * np.sqrt(pc_cal * (1 - pc_cal) / np.maximum(n, 1.0)), 0.001, 0.999)


def kelly(p_win: np.ndarray, odds: np.ndarray, p_push: np.ndarray) -> np.ndarray:
    """Kelly fraction of a bet that wins (odds - 1), is refunded (0) or loses (-1)."""
    b = np.asarray(odds, float) - 1
    q = np.clip(1 - p_win - p_push, 0, 1)
    with np.errstate(all="ignore"):
        return (p_win * b - q) / (b * np.clip(p_win + q, 1e-9, None))
