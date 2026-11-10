"""Refitted model variants: how much should the models specialise by league?

All variants use the production parameters and tree counts (no re-tuning), are fitted walk-forward on seasons < s
and are averaged over the three seeds (reference: the production setup averaged over the same seeds, "B0̄").

    M1   one global model WITHOUT the league one-hot (league context only through the league_* averages)
    M2   a separate model per league (its own matches only) — full specialisation, no pooling
    M3   a separate model per group of similar leagues (groups of fold s, profiles.py), league one-hot inside
    M4a  global model, then per league continued boosting on that league's matches (+20 % trees):
         fine-tuning with the global model as the prior
    M4b  the same with +50 % trees (weaker prior)
    M5   global model with the GROUP one-hot instead of the league one-hot (pooling inside groups)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import profiles as PR
from . import wf

VARIANTS = {"M1": "без one-hot лиги", "M2": "отдельная модель на лигу", "M3": "отдельная модель на группу лиг",
            "M4a": "дообучение на лиге (+20 % деревьев)", "M4b": "дообучение на лиге (+50 % деревьев)",
            "M5": "one-hot группы вместо лиги"}
TASKS = wf.GOAL_TASKS


def _write(out: pd.DataFrame, task: str, idx, p) -> None:
    if task == "result":
        out.loc[idx, ["pH", "pD", "pA"]] = p
    else:
        out.loc[idx, "lh" if task == "home_goals" else "la"] = p


def _predict(variant: str, s: int, seed: int) -> pd.DataFrame:
    df = wf.features()
    train, score = wf.before(df, s), wf.season(df, s).reset_index(drop=True)
    out = pd.DataFrame({"match_id": score["match_id"].to_numpy()})
    for c in ("pH", "pD", "pA", "lh", "la"):
        out[c] = np.nan
    if variant == "M1":
        return wf.predict_goal_tasks(train, score, seed, categorical=())
    if variant == "M5":
        groups = PR.groups_for(s)
        tr, sc = train.assign(lgroup=train["league"].map(groups)), score.assign(lgroup=score["league"].map(groups))
        return wf.predict_goal_tasks(tr, sc, seed, categorical=("lgroup",))
    if variant in ("M2", "M3"):
        key = (lambda d: d["league"]) if variant == "M2" else (lambda d: d["league"].map(PR.groups_for(s)))
        ktr, ksc = key(train).to_numpy(), key(score).to_numpy()
        for u in np.unique(ksc):
            tr = train.loc[ktr == u]
            idx = np.flatnonzero(ksc == u)
            for task in TASKS:
                m = wf.Fit(task, seed=seed).fit(tr)
                _write(out, task, idx, m.predict(score.iloc[idx]))
        return out
    if variant in ("M4a", "M4b"):
        frac = 0.2 if variant == "M4a" else 0.5
        for task in TASKS:
            g = wf.Fit(task, seed=seed).fit(train)
            extra = max(1, int(round(frac * g.n)))
            for lg in score["league"].unique():
                idx = np.flatnonzero(score["league"].to_numpy() == lg)
                ft = g.fine_tune(train.loc[train["league"] == lg], extra)
                _write(out, task, idx, ft.predict(score.iloc[idx]))
        return out
    raise ValueError(variant)


def predictions(variant: str, s: int, seed: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"var_{variant}_{s}_{seed}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    out = _predict(variant, s, seed)
    out.to_parquet(path)
    C.log.info("variant %s season %s seed %s done", variant, s, seed)
    return out


def averaged(variant: str, s: int) -> pd.DataFrame:
    return wf.seed_mean([predictions(variant, s, seed) for seed in C.SEEDS])


def base_averaged(s: int) -> pd.DataFrame:
    return wf.seed_mean([wf.base_predictions(s, seed) for seed in C.SEEDS])


def build_all() -> None:
    for s in C.EVAL:
        for v in VARIANTS:
            for seed in C.SEEDS:
                predictions(v, s, seed)


if __name__ == "__main__":
    C.setup_logging("models")
    build_all()
    print("MODELS_DONE")
