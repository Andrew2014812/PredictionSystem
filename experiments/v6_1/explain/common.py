"""Shared code of the explainability research (FootPredict v6.1).

Everything here READS the stored data (features, stored predictions, production model files) and
writes only under experiments/v6_1/explain/. The production card code (src/explain/chain.py,
src/explain/card.py) is used as is, so the "new" method measured here is exactly what the site shows.

Methods compared on the probability of one market selection P = g(u(x)):

  old      v6 card: TreeSHAP of ONE source model's margin (src/explain/shap_explainer.explanation_plan)
  chain    path-integrated chain rule: c_j = Σ_k G_k φ_jk, G_k = ∫ ∂g/∂u_k along ū → u (v6.1 card)
  linear   local chain rule: c_j = Σ_k ∂g/∂u_k(u) φ_jk (derivative at the match only)
  mshap    exact Shapley values of the model OUTPUTS (5 players, baseline ū), spread over features in
           proportion to TreeSHAP (rescale rule)
  occl     occlusion: ΔP when one factor's features are set to their training means
  ref      reference: permutation Shapley values of the WHOLE composition g(u(x)) at factor level,
           interventional, with a background sample of real earlier matches
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import chain  # noqa: E402
from src.explain.card import _group_key, factor_table  # noqa: E402
from src.explain.shap_explainer import explain_selection, explanation_plan  # noqa: E402
from src.features.store import load_features  # noqa: E402
from src.markets.markets import market_group  # noqa: E402
from src.modelling.registry import load_bundles, version_metadata  # noqa: E402
from src.prediction.store import load_markets, load_snapshots  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for d in (RESULTS, FIGURES, CACHE):
    d.mkdir(exist_ok=True)

log = logging.getLogger("explain")
VERSIONS = {"backtest_2526": "backtest", "prod_20261012": "backfill"}      # versions with exact reproduction
PANEL = ["1X2", "DOUBLE_CHANCE", "TOTAL_1.5", "TOTAL_2.5", "TOTAL_3.5", "BTTS", "HANDICAP_-0.5", "HANDICAP_-1",
         "HANDICAP_-1.5", "HANDICAP_+1.5", "CORNERS_9.5", "EXACT_SCORE"]
N_JOBS = 3


def setup_logging(name: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(CACHE / f"{name}.log", "w")])


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        t = time.time()
        self.features = load_features()
        self.features["date"] = pd.to_datetime(self.features["date"])
        self.by_id = self.features.set_index("match_id", drop=False)
        self.snaps = load_snapshots()
        self.snaps["date"] = pd.to_datetime(self.snaps["date"])
        self.markets = load_markets()
        self.bundles = {v: load_bundles(v) for v in VERSIONS}
        for bundles in self.bundles.values():
            for b in bundles.values():
                b.model.set_params(n_jobs=N_JOBS)
        self.meta = {v: version_metadata(v) for v in VERSIONS}
        log.info("data loaded in %.1fs", time.time() - t)

    def row(self, match_id: str) -> pd.DataFrame:
        return self.by_id.loc[[match_id]].reset_index(drop=True)

    def match_markets(self, match_id: str) -> pd.DataFrame:
        return self.markets.loc[self.markets["match_id"] == match_id]

    def pool(self) -> pd.DataFrame:
        """Stored predictions whose model version and pre-match features reproduce them exactly."""
        s = self.snaps.loc[self.snaps["match_id"].isin(self.by_id.index)]
        return s.loc[[VERSIONS.get(v) == src for v, src in zip(s["model_version"], s["source"])]]


def panel(snap: pd.Series, mk: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(market, selection, role): the Main and Risk picks, then the likeliest selection of each panel market."""
    out = []
    for role in ("main", "risk"):
        m, s = snap.get(f"{role}_market"), snap.get(f"{role}_selection")
        if isinstance(m, str) and isinstance(s, str):
            out.append((m, s, role))
    seen = {(m, s) for m, s, _ in out}
    for market in PANEL:
        rows = mk.loc[mk["market"] == market]
        if rows.empty:
            continue
        best = rows.loc[rows["probability"].idxmax()]
        if (best["market"], best["selection"]) not in seen:
            out.append((best["market"], best["selection"], "panel"))
    return out


# ----------------------------------------------------------------------------- the composition g(u(x))
def raw_columns(bundles: dict) -> list[str]:
    cols = []
    for b in bundles.values():
        for c in b.preprocessor.categorical + b.preprocessor.numeric:
            if c not in cols:
                cols.append(c)
    return cols


def factor_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(_group_key(c)[0], []).append(c)
    return groups


def model_outputs(df: pd.DataFrame, bundles: dict) -> dict[str, np.ndarray]:
    """u for every row: 'goal' (n, 5) = (z_H, z_D, z_A, log λh, log λa) and 'corner' (n, 1)."""
    z = bundles["result"].model.predict(bundles["result"].matrix(df), output_margin=True)
    lh = bundles["home_goals"].model.predict(bundles["home_goals"].matrix(df), output_margin=True)
    la = bundles["away_goals"].model.predict(bundles["away_goals"].matrix(df), output_margin=True)
    out = {"goal": np.column_stack([np.asarray(z, float), lh, la])}
    if "corners" in bundles:
        out["corner"] = np.asarray(bundles["corners"].model.predict(bundles["corners"].matrix(df),
                                                                    output_margin=True), float)[:, None]
    return out


def g_of(market: str, selection: str, U: dict, bundles: dict) -> np.ndarray:
    rho = bundles["home_goals"].extra.get("dc_rho", 0.0) or 0.0
    nb = bundles["corners"].extra.get("nb_size") if "corners" in bundles else None
    key = "corner" if market_group(market) == "CORNERS" else "goal"
    return chain.market_probability(market, selection, U[key], rho, nb)


# ----------------------------------------------------------------------------- methods
def old_explain(row: pd.DataFrame, market: str, selection: str, bundles: dict) -> tuple[pd.Series, list[str]]:
    """v6: signed TreeSHAP of the source model(s) in margin units summed per factor, and the keys of
    the factors the v6 card showed (its exact rule: supporting factors by Σ|SHAP| share, at least 3,
    then those ≥ 3 %, plus the strongest factor against when ≥ 10 %)."""
    expl = explain_selection(row, market, selection, bundles)
    total = float(expl["shap"].abs().sum()) or 1.0
    groups: dict[str, list[float]] = {}
    for f, s in zip(expl["feature"].astype(str), expl["shap"]):
        g = groups.setdefault(_group_key(f)[0], [0.0, 0.0])
        g[0] += float(s)
        g[1] += abs(float(s))
    items = [(k, v[1] / total, v[0] > 0) for k, v in groups.items()]
    support = sorted((o for o in items if o[2]), key=lambda o: -o[1])
    against = sorted((o for o in items if not o[2]), key=lambda o: -o[1])
    shown = [o for i, o in enumerate(support[:5]) if i < 3 or o[1] >= 0.03]
    if against and against[0][1] >= 0.10:
        shown = (shown[:4] if len(shown) >= 5 else shown) + [against[0]]
    return pd.Series({k: v[0] for k, v in groups.items()}), [o[0] for o in shown]


def old_factors(row: pd.DataFrame, market: str, selection: str, bundles: dict) -> pd.Series:
    return old_explain(row, market, selection, bundles)[0]


def to_factors(contrib: pd.DataFrame, col: str = "contribution") -> pd.Series:
    ft = factor_table(contrib[["feature", col]].rename(columns={col: "contribution"}))
    return pd.Series(ft["contribution"].to_numpy(), index=ft["key"])


def linear_contributions(inputs: chain.ChainInputs, market: str, selection: str) -> tuple[pd.DataFrame, float]:
    grad = chain.gradient(market, selection, inputs.u[None, :], inputs.rho, inputs.nb_size)[0]
    c = inputs.phi.to_numpy() @ grad
    return pd.DataFrame({"feature": inputs.phi.index, "contribution": c}), float(c.sum())


def output_shapley(inputs: chain.ChainInputs, market: str, selection: str) -> np.ndarray:
    """Exact Shapley values of the model outputs (players) for g with baseline ū."""
    k = len(inputs.u)
    g = lambda U: chain.market_probability(market, selection, U, inputs.rho, inputs.nb_size)   # noqa: E731
    subsets = [s for r in range(k + 1) for s in combinations(range(k), r)]
    pts = np.array([[inputs.u[i] if i in s else inputs.base[i] for i in range(k)] for s in subsets])
    val = dict(zip(subsets, g(pts)))
    phi = np.zeros(k)
    for i in range(k):
        for s in subsets:
            if i in s:
                continue
            w = factorial(len(s)) * factorial(k - len(s) - 1) / factorial(k)
            phi[i] += w * (val[tuple(sorted(s + (i,)))] - val[s])
    return phi


def mshap_contributions(inputs: chain.ChainInputs, market: str, selection: str) -> pd.DataFrame:
    phi_out = output_shapley(inputs, market, selection)
    du = inputs.u - inputs.base
    grad = chain.gradient(market, selection, inputs.u[None, :], inputs.rho, inputs.nb_size)[0]
    slope = np.where(np.abs(du) > 1e-9, phi_out / np.where(np.abs(du) > 1e-9, du, 1.0), grad)
    return pd.DataFrame({"feature": inputs.phi.index, "contribution": inputs.phi.to_numpy() @ slope})


def occlusion(row: pd.DataFrame, market: str, selection: str, bundles: dict, groups: dict[str, list[str]],
              means: dict) -> pd.Series:
    """ΔP_g = P(x) − P(x with factor g at its training mean); the league factor is left out."""
    keys = [k for k in groups if k != "league"]
    frame = pd.concat([row] * (len(keys) + 1), ignore_index=True)
    for i, k in enumerate(keys, start=1):
        for c in groups[k]:
            if c in means:
                frame.loc[i, c] = means[c]
    p = g_of(market, selection, model_outputs(frame, bundles), bundles)
    return pd.Series(p[0] - p[1:], index=keys)


def permutation_reference(row: pd.DataFrame, background: pd.DataFrame, bundles: dict, groups: dict[str, list[str]],
                          selections: list[tuple[str, str]], n_perm: int, seed: int) -> dict:
    """Interventional permutation Shapley of g(u(x)) for several selections at once (antithetic pairs).

    Returns {"phi": (n_sel, G) mean, "se": (n_sel, G) standard error, "keys", "p_x", "p_bg"}."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    cols = [c for k in keys for c in groups[k]]
    G, B = len(keys), len(background)
    x = row[cols].iloc[0]
    bg = background[cols].reset_index(drop=True)
    col_group = np.array([keys.index(_group_key(c)[0]) for c in cols])
    draws = []
    for _ in range(n_perm // 2):
        perm = rng.permutation(G)
        for order in (perm, perm[::-1]):
            rank = np.empty(G, int)
            rank[order] = np.arange(G)
            col_rank = rank[col_group]                         # when each column switches to x
            steps = np.arange(G + 1)[:, None]                  # state s: groups with rank < s are from x
            take_x = col_rank[None, :] < steps                 # (G+1, ncols)
            frame = pd.DataFrame(np.repeat(bg.to_numpy(dtype=object)[None], G + 1, 0).reshape(-1, len(cols)),
                                 columns=cols)
            mask = np.repeat(take_x, B, axis=0)
            xv = np.tile(x.to_numpy(dtype=object), ((G + 1) * B, 1))
            frame = pd.DataFrame(np.where(mask, xv, frame.to_numpy()), columns=cols)
            for c in cols:
                if c != "league":
                    frame[c] = pd.to_numeric(frame[c], errors="coerce")
            U = model_outputs(frame, bundles)
            vals = np.stack([g_of(m, s, U, bundles).reshape(G + 1, B).mean(1) for m, s in selections])
            delta = np.diff(vals, axis=1)                      # (n_sel, G): contribution of the s-th group
            phi = np.zeros((len(selections), G))
            phi[:, order] = delta
            draws.append(phi)
    draws = np.stack(draws)
    return {"keys": keys, "phi": draws.mean(0), "se": draws.std(0, ddof=1) / np.sqrt(len(draws)),
            "draws": draws}


# ----------------------------------------------------------------------------- comparison metrics
def top(s: pd.Series, k: int) -> list[str]:
    return list(s.abs().sort_values(ascending=False, kind="stable").index[:k])


def sign_agreement(method: pd.Series, ref: pd.Series, k: int = 5) -> float:
    keys = [x for x in top(method, k) if x in ref.index]
    if not keys:
        return float("nan")
    return float(np.mean([np.sign(method[x]) == np.sign(ref[x]) for x in keys]))


def overlap(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    return len(set(top(a, k)) & set(top(b, k))) / k


def jaccard(a: list[str], b: list[str]) -> float:
    return len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)


def spearman(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        return float("nan")
    return float(pd.Series(a[idx]).rank().corr(pd.Series(b[idx]).rank()))


def plan_outputs(market: str, selection: str) -> set[str]:
    """Model outputs the v6 card explained for this market."""
    names = set()
    for task, cls, _ in explanation_plan(market, selection):
        if task == "result":
            names.add(["z_H", "z_D", "z_A"][cls])
        else:
            names.add({"home_goals": "log_lambda_home", "away_goals": "log_lambda_away", "corners": "log_mu"}[task])
    return names
