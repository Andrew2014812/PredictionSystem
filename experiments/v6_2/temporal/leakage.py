"""Explicit leakage tests of the temporal features (and of the production feature builder).

    python -m experiments.v6_2.temporal.leakage          (writes results/leakage_test.json, exits 1 on failure)
    pytest experiments/v6_2/temporal/leakage.py           (the same checks as tests)

1. Reproduction. The study's builder recomputes production columns (Elo of both teams, last-5 goals and points,
   last-10 opponent-adjusted goals and Elo surprise, rest days, last-5 venue goals) on all 38 k matches: they must
   equal the production feature table exactly (same values, same missing pattern) — the study's timeline is the
   production timeline.
2. Future-blindness of the study's builder. On every English match (four divisions; history is kept per
   country, so the subset is closed) the results and statistics of every match ON OR AFTER a cut-off day D are
   replaced by random values and all features are rebuilt: every feature of every match played on or before D
   must be unchanged (day D included: results of the same day must not be used). Positive control: features
   after D must change (the test can detect a dependence).
3. Own result. Changing the score of one match must not change that match's features, and must change the
   features of the next match of its home team (positive control).
4. Future-blindness of the PRODUCTION builder (src/features/builder.py), same perturbation on the English raw
   results 2023/24–2025/26: every feature column of matches on or before D must be unchanged.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.v6_2.leagues import wf

from . import builder as B
from . import common as C

CUTOFFS = ("2023-01-02", "2024-08-20", "2025-03-15", "2026-09-20")
PROD_CUTOFFS = ("2023-10-01", "2024-08-20", "2025-03-15")      # inside the 2023/24-2025/26 raw subset
REPRO = [("x_elo_h", "h_elo"), ("x_elo_a", "a_elo"), ("h_rep_f5_gf", "h_form5_gf"), ("a_rep_f5_ppg", "a_form5_ppg"),
         ("h_rep_f10_adjgf", "h_form10_adj_gf"), ("a_rep_f10_surp", "a_form10_elo_surprise"),
         ("h_rep_rest", "h_rest_days"), ("a_rep_rest", "a_rest_days"), ("a_rep_vf5_gf", "a_venue_form5_gf")]
PERTURB = ["home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot"]


def _perturb(m: pd.DataFrame, mask: np.ndarray, seed: int, extra=()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = m.copy()
    n = int(mask.sum())
    for c in PERTURB + list(extra):
        if c in out:
            hi = 6 if "goals" in c else (25 if "shots" in c else (10 if "sot" in c or "corners" in c else 4))
            out.loc[mask, c] = rng.integers(0, hi, n).astype(float)
    if "result" in out:
        gd = out["home_goals"] - out["away_goals"]
        out["result"] = np.where(gd > 0, "H", np.where(gd < 0, "A", "D"))
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame, cols) -> tuple[int, float]:
    """(number of differing cells, max abs difference) between two aligned frames."""
    x, y = a[cols].to_numpy(float), b[cols].to_numpy(float)
    nan_diff = np.isnan(x) != np.isnan(y)
    both = ~np.isnan(x) & ~np.isnan(y)
    d = np.abs(np.where(both, x - y, 0.0))
    return int(nan_diff.sum() + (d > 1e-12).sum()), float(d.max()) if d.size else 0.0


def reproduction() -> dict:
    df = wf.features()
    f = B.build(df)
    m = df[["match_id"] + [b for _, b in REPRO]].merge(f[["match_id"] + [a for a, _ in REPRO]], on="match_id")
    out = {}
    for a, b in REPRO:
        x, y = m[a].to_numpy(float), m[b].to_numpy(float)
        both = ~np.isnan(x) & ~np.isnan(y)
        out[b] = {"rows": int(len(m)), "compared": int(both.sum()), "nan_pattern_mismatch": int((np.isnan(x) != np.isnan(y)).sum()),
                  "max_abs_diff": float(np.abs(x[both] - y[both]).max())}
    out["passed"] = all(v["nan_pattern_mismatch"] == 0 and v["max_abs_diff"] < 1e-9 for v in out.values())
    return out


def _england() -> pd.DataFrame:
    df = wf.features()
    return df.loc[df["country"] == "England", B.RAW].reset_index(drop=True)


def future_blind_study() -> dict:
    m = _england()
    base = B.build(m).set_index("match_id")
    cols = [c for c in base.columns]
    out = {"matches": int(len(m)), "features": len(cols), "cutoffs": {}}
    for i, d in enumerate(CUTOFFS):
        day = pd.Timestamp(d)
        mask = (pd.to_datetime(m["date"]) >= day).to_numpy()
        pert = B.build(_perturb(m, mask, 100 + i)).set_index("match_id").reindex(base.index)
        upto = pd.to_datetime(m.set_index("match_id").reindex(base.index)["date"]) <= day
        n_diff, mx = _same(base.loc[upto.to_numpy()], pert.loc[upto.to_numpy()], cols)
        after = ~upto.to_numpy()
        changed = (_rowdiff(base.loc[after], pert.loc[after], cols)).mean() if after.any() else np.nan
        out["cutoffs"][d] = {"rows_on_or_before": int(upto.sum()), "differing_cells_on_or_before": n_diff,
                             "max_abs_diff_on_or_before": mx, "rows_after": int(after.sum()),
                             "share_rows_changed_after": float(changed)}
    out["passed"] = all(v["differing_cells_on_or_before"] == 0 for v in out["cutoffs"].values()) and \
        all(v["share_rows_changed_after"] > 0.5 for v in out["cutoffs"].values() if v["rows_after"] > 100)
    return out


def _rowdiff(a, b, cols) -> np.ndarray:
    x, y = a[cols].to_numpy(float), b[cols].to_numpy(float)
    return ((np.isnan(x) != np.isnan(y)) | (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) > 1e-12)).any(1)


def own_result() -> dict:
    m = _england()
    m = m.loc[m["season"].isin([2024, 2025])].reset_index(drop=True)
    base = B.build(m).set_index("match_id")
    rng = np.random.default_rng(5)
    picks = rng.choice(len(m) - 200, 20, replace=False)
    own_changed, next_changed = 0, 0
    for i in picks:
        row = m.iloc[i]
        pert = m.copy()
        pert.loc[i, "home_goals"] = row["home_goals"] + 3
        pert.loc[i, "home_sot"] = (row["home_sot"] if row["home_sot"] == row["home_sot"] else 0) + 5
        f = B.build(pert).set_index("match_id")
        own_changed += int(_rowdiff(base.loc[[row["match_id"]]], f.loc[[row["match_id"]]], list(base.columns)).any())
        later = m.loc[(pd.to_datetime(m["date"]) > pd.Timestamp(row["date"]))
                      & ((m["home_team"] == row["home_team"]) | (m["away_team"] == row["home_team"]))]
        if len(later):
            nid = later.sort_values("date")["match_id"].iloc[0]
            next_changed += int(_rowdiff(base.loc[[nid]], f.loc[[nid]], list(base.columns)).any())
    return {"matches_tested": len(picks), "own_features_changed": own_changed, "next_match_features_changed": next_changed,
            "passed": own_changed == 0 and next_changed == len(picks)}


def future_blind_production() -> dict:
    from src.data.dataset import load_results
    from src.features.builder import ID_COLUMNS, MATCH_STAT_COLUMNS, ODDS_COLUMNS, TARGET_COLUMNS, build_features
    raw = load_results(seasons=[2023, 2024, 2025], leagues=["E0", "E1", "E2", "E3"]).reset_index(drop=True)
    base = build_features(raw).set_index("match_id")
    skip = set(ID_COLUMNS + ODDS_COLUMNS + MATCH_STAT_COLUMNS + TARGET_COLUMNS)
    cols = [c for c in base.columns if c not in skip and pd.api.types.is_numeric_dtype(base[c])]
    out = {"matches": int(len(raw)), "features": len(cols), "cutoffs": {}}
    for i, d in enumerate(PROD_CUTOFFS):
        day = pd.Timestamp(d)
        mask = (pd.to_datetime(raw["date"]) >= day).to_numpy()
        pert = build_features(_perturb(raw, mask, 200 + i, extra=("home_corners", "away_corners", "home_fouls",
                                                                    "away_fouls", "home_yellows", "away_yellows",
                                                                    "home_reds", "away_reds")))
        pert = pert.set_index("match_id").reindex(base.index)
        upto = (pd.to_datetime(base["date"]) <= day).to_numpy()
        n_diff, mx = _same(base.loc[upto], pert.loc[upto], cols)
        after = ~upto
        out["cutoffs"][d] = {"rows_on_or_before": int(upto.sum()), "differing_cells_on_or_before": n_diff,
                             "max_abs_diff_on_or_before": mx, "rows_after": int(after.sum()),
                             "share_rows_changed_after": float(_rowdiff(base.loc[after], pert.loc[after], cols).mean())}
    out["passed"] = all(v["differing_cells_on_or_before"] == 0 and v["share_rows_changed_after"] > 0.5
                        for v in out["cutoffs"].values())
    return out


# ------------------------------------------------------------------------------------------------ pytest entry points
def test_reproduction():
    assert reproduction()["passed"]


def test_future_blind_study():
    assert future_blind_study()["passed"]


def test_own_result():
    assert own_result()["passed"]


def test_future_blind_production():
    assert future_blind_production()["passed"]


def run() -> dict:
    res = {"reproduction": reproduction(), "future_blind_study": future_blind_study(), "own_result": own_result(),
           "future_blind_production": future_blind_production()}
    res["all_passed"] = all(v["passed"] for v in res.values())
    C.save_json("leakage_test.json", res)
    return res


if __name__ == "__main__":
    C.setup_logging("leakage")
    r = run()
    print("LEAKAGE_PASSED" if r["all_passed"] else "LEAKAGE_FAILED")
    sys.exit(0 if r["all_passed"] else 1)
