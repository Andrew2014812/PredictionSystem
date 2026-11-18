"""Reliability of the feature groups: permutation importance in two seasons, missingness, stability.

For every task and every feature group of its production feature set (``src/features/sets.MODEL_GROUPS``), the
columns of the group are permuted jointly across the matches of a season (the same permutation for all columns
of the group, so their mutual relations stay intact) and the model predicts again; importance = increase of the
loss (1X2 log loss / mean Poisson deviance), mean of 3 permutations. Seasons: 2025/26 with the out-of-sample
refit (trained ≤ 2024/25, oos.py) and 2026/27 with the production model (trained ≤ 2025/26). A group is
"reliable" when its importance is positive in both seasons and of the same order; a group whose importance
flips sign or collapses carries season-specific noise. Missing share of each group by league (2026/27) is
reported alongside. Prediction only, no fitting.
"""
from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN
from src.features.sets import GROUPS, MODEL_GROUPS
from src.modelling.tasks import TASKS

from . import data as Q

REPS = 3


def _loss(task_name, bundle, rows):
    from experiments.v6_3.alt_ml.models import score
    y = TASKS[task_name].target_values(rows).to_numpy()
    return score(task_name, y, bundle.predict(rows))


def group_columns(group: str, columns) -> list[str]:
    pats = [re.compile(p) for p in GROUPS[group]]
    return [c for c in columns if any(p.search(c) for p in pats)]


def importance(bundles: dict, season_rows: pd.DataFrame, label: str) -> list[dict]:
    rng = np.random.default_rng(C.SEED)
    out = []
    for task_name, groups in MODEL_GROUPS.items():
        b = bundles[task_name]
        rows = season_rows.loc[TASKS[task_name].trainable(season_rows)]
        if task_name == "corners":
            rows = rows.loc[rows["league"].isin(b.extra.get("leagues") or rows["league"].unique())]
        base = _loss(task_name, b, rows)
        feats = set(b.feature_names)
        for g in groups:
            cols = [c for c in group_columns(g, rows.columns) if c in feats]
            if not cols:
                continue
            deltas = []
            for _ in range(REPS):
                perm = rng.permutation(len(rows))
                x = rows.copy()
                x[cols] = rows[cols].to_numpy()[perm]
                deltas.append(_loss(task_name, b, x) - base)
            out.append({"season": label, "task": task_name, "group": g, "columns": len(cols), "base_loss": base,
                        "importance": float(np.mean(deltas)), "importance_sd": float(np.std(deltas)),
                        "missing_share": float(rows[cols].isna().mean().mean())})
    return out


def run() -> dict:
    f = Q.features()
    cur = f.loc[(f["season"] == C.CURRENT) & f["played"].astype(bool)]
    prev = f.loc[(f["season"] == 2025) & f["played"].astype(bool)]
    rows = importance(EN.production_bundles(), cur, "2026/27")
    oos = Q.D["cache"] / "oos_2025_bundles.pkl"
    if oos.exists():
        rows += importance(joblib.load(oos), prev, "2025/26")
    imp = pd.DataFrame(rows)
    # missing share by group and league (2026/27, result + corner groups)
    miss = []
    for g in GROUPS:
        cols = group_columns(g, cur.columns)
        if not cols or g == "market":
            continue
        for lg, part in cur.groupby("league"):
            miss.append({"group": g, "league": lg, "missing_share": float(part[cols].isna().mean().mean())})
    miss = pd.DataFrame(miss)
    wide = imp.pivot_table(index=["task", "group"], columns="season", values="importance").reset_index()
    stab = {}
    if {"2025/26", "2026/27"} <= set(wide.columns):
        for t, g in wide.groupby("task"):
            stab[t] = {"spearman": float(g["2025/26"].corr(g["2026/27"], method="spearman")),
                       "both_positive": int(((g["2025/26"] > 0) & (g["2026/27"] > 0)).sum()),
                       "sign_flip": int(((g["2025/26"] > 0) != (g["2026/27"] > 0)).sum()), "groups": int(len(g))}
    return {"importance": imp, "missing": miss, "stability": stab}
