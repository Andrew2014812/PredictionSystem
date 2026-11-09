"""Feature-set variants of the temporal study, fitted walk-forward with the production parameters.

Every variant changes only the numeric feature lists of the 1X2 classifier and of the two goal regressors
(corners unchanged); parameters, tree counts, preprocessing and recency weights are production's (nothing is
re-tuned, which favours the production feature set). Each variant is fitted on seasons < s, predicts season s
and is averaged over three seeds; the reference T0 is the production feature set averaged over the same seeds.

    T1..T8  production + one new family (builder.FAMILIES)
    T9      production + all eight families
    R0      "plain averages": production minus every dynamic feature (form windows, trends, venue form,
            frequencies, opponent-adjusted form, shot / corner shares, rest days) and minus Elo — season-to-date
            averages, table, head-to-head and league context only
    R1      R0 + Elo (Elo is itself an exponentially updated rating)
    R2      R1 + exponentially weighted means (EW instead of the production window means)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from experiments.v6_2.leagues import wf

from . import builder as B
from . import common as C

TASKS = wf.GOAL_TASKS
FAMILY_VARIANTS = {f"T{i + 1}": fam for i, fam in enumerate(B.FAMILIES)}
VARIANTS = {**{v: f"+ {B.FAMILY_RU[f]}" for v, f in FAMILY_VARIANTS.items()},
            "T9": "+ все восемь семейств",
            "R0": "простые сезонные средние (без оконной формы и Elo)",
            "R1": "R0 + Elo",
            "R2": "R1 + экспоненциальные средние"}
DYNAMIC = [r"form\d+_", r"_trend_", r"venue_form", r"_freq_", r"^diff_form", r"form_attack_vs", r"_sot_vs_",
           r"form_corners_sum", r"rest_days"]
ELO = [r"^[ha]_elo$", r"^elo_(diff|exp_home)$"]

_EXT: pd.DataFrame | None = None


def temporal_features(refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / "temporal_features.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    f = B.build(wf.features())
    f = f.drop(columns=[c for c in f.columns if "_rep_" in c or c.startswith("x_elo")])
    f.to_parquet(path)
    return f


def extended() -> pd.DataFrame:
    """The production feature table (played matches) with the candidate temporal columns joined by match_id."""
    global _EXT
    if _EXT is None:
        base = wf.features()
        t = temporal_features()
        _EXT = base.merge(t, on="match_id", how="left", validate="one_to_one")
        assert len(_EXT) == len(base)
    return _EXT


def new_columns() -> list[str]:
    return [c for c in temporal_features().columns if c != "match_id"]


def numeric(variant: str, task: str) -> list[str]:
    prod = wf.prod_numeric(task)
    new = new_columns()
    if variant == "T0":
        return prod
    if variant in FAMILY_VARIANTS:
        return prod + B.family_columns(new, [FAMILY_VARIANTS[variant]])
    if variant == "T9":
        return prod + B.family_columns(new, list(B.FAMILIES))
    pats = [re.compile(p) for p in DYNAMIC]
    plain = [c for c in prod if not any(p.search(c) for p in pats)]
    if variant == "R0":
        return [c for c in plain if not any(re.search(p, c) for p in ELO)]
    if variant == "R1":
        return plain
    if variant == "R2":
        return plain + B.family_columns(new, ["ewm"])
    raise ValueError(variant)


def predictions(variant: str, s: int, seed: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"tv_{variant}_{s}_{seed}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = extended()
    num = {t: numeric(variant, t) for t in TASKS}
    out = wf.predict_goal_tasks(wf.before(df, s), wf.season(df, s), seed, numeric=num)
    out.to_parquet(path)
    C.log.info("temporal variant %s season %s seed %s done", variant, s, seed)
    return out


def averaged(variant: str, s: int) -> pd.DataFrame:
    if variant == "T0":
        return wf.seed_mean([wf.base_predictions(s, seed) for seed in C.SEEDS])
    return wf.seed_mean([predictions(variant, s, seed) for seed in C.SEEDS])


def feature_counts() -> dict:
    return {v: {t: len(numeric(v, t)) for t in TASKS} for v in ["T0", *VARIANTS]}


def build_all() -> None:
    temporal_features()
    for s in C.EVAL:
        for v in VARIANTS:
            for seed in C.SEEDS:
                predictions(v, s, seed)


if __name__ == "__main__":
    C.setup_logging("variants")
    build_all()
    print("VARIANTS_DONE")
