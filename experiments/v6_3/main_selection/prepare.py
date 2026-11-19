"""Enriched candidate tables (bt, close, t09) with every walk-forward column, cached in ``cache/``.

    python -m experiments.v6_3.main_selection.prepare        (light job: pandas only, ~1 min)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data as D
from . import features as F

UNIVERSES = ("bt", "close", "t09")
WC_SIGMA = 0.03                 # typical model - market disagreement used by the winner's-curse penalty
WC_GAMMAS = (0.5, 1.0, 2.0)


def winners_curse(c: pd.DataFrame) -> pd.DataFrame:
    """p − γ·σ·sqrt(2 ln n): expected maximum of n noisy estimates, n = Main-eligible selections of the match."""
    n = np.maximum(c["ncand"].to_numpy(float), 1.0)
    p = c["probability"].to_numpy(float)
    for g in WC_GAMMAS:
        c[f"p_wc{int(g * 10)}"] = np.clip(p - g * WC_SIGMA * np.sqrt(2 * np.log(n)), 1e-4, 1.0)
    return c


def build(refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    paths = {u: D.CACHE / f"enriched_{u}.parquet" for u in UNIVERSES}
    info_path = D.CACHE / "walk_forward_maps.json"
    if not refresh and all(p.exists() for p in paths.values()):
        return {u: pd.read_parquet(p) for u, p in paths.items()}
    info = {}
    bt = winners_curse(F.add_static(D.load("bt")))
    bt = F.walk_forward_columns(bt, None)
    info["bt"] = bt.attrs.pop("walk_forward")
    out["bt"] = bt
    for u in ("close", "t09"):
        c = winners_curse(F.add_static(D.load(u)))
        c = F.walk_forward_columns(c, bt)
        info[u] = c.attrs.pop("walk_forward")
        out[u] = c
    for u, c in out.items():
        c.attrs = {}
        c.to_parquet(paths[u])
    info_path.write_text(json.dumps(info, indent=1, default=str))
    return out


def load() -> dict[str, pd.DataFrame]:
    return build(refresh=False)


if __name__ == "__main__":
    d = build(refresh=True)
    for u, c in d.items():
        print(u, c.shape, c["match_id"].nunique())
