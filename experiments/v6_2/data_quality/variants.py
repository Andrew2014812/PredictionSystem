"""Walk-forward test of two ways to give clubs without history better features (research only).

The production feature builder is run on the same local files with one change each, the production models are
refitted (TaskTrainer, tuned parameters of the active version) on all seasons before the season predicted, and
the matches are predicted through the production engine with the real Bet365 prices (predictions.predict):

    base             the production builder unchanged (must reproduce local_data/processed/features.parquet)
    elo_new_-50      a club first seen after the first season of the data starts at base(tier) − 50 Elo
    elo_new_-100     the same with − 100
    ec_history       National League (EC, disabled for prediction) matches feed the state, so clubs promoted
                     to League Two arrive with form, Elo and head-to-head; EC rows are never trained on or predicted

Seasons: 2023/24 and 2024/25 select (pre-declared rule: a variant is promising only if it lowers the log loss of
the matches of clubs from outside on both, without raising the overall log loss); 2025/26 for information.
Every variant is evaluated on the same matches as ``base``.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from src.data.dataset import build_matches
from src.features import builder as B
from src.features.state import EloRatings
from src.leagues import league_codes
from src.modelling import training as T
from src.modelling.registry import load_bundles
from src.modelling.tasks import TASKS

from . import common as C
from . import predictions as PR          # noqa: F401  (limits the trainer to N_JOBS threads)

SEASONS = (2023, 2024, 2025)
VARIANTS = ("base", "elo_new_-50", "elo_new_-100", "ec_history")


class NewcomerElo(EloRatings):
    def __init__(self, penalty: float, first_season: int):
        super().__init__()
        self.penalty, self.first_season = penalty, first_season

    def get(self, team, season, tier):
        if team not in self.ratings and season > self.first_season:
            self.ratings[team], self.season[team] = self.base(tier) - self.penalty, season
        return super().get(team, season, tier)


def build(variant: str) -> pd.DataFrame:
    path = C.CACHE / f"features_{variant}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    leagues = league_codes() + (["EC"] if variant == "ec_history" else [])
    matches = build_matches(leagues)
    fb = B.FeatureBuilder()
    if variant.startswith("elo_new_"):
        fb.elo = NewcomerElo(float(variant.split("_")[-1].lstrip("-")), int(matches["season"].min()))
    f = fb.build(matches)
    f = f.loc[f["league"].isin(league_codes())].reset_index(drop=True)
    f.to_parquet(path)
    return f


def fold(f: pd.DataFrame, season: int) -> pd.DataFrame:
    prod = load_bundles(C.PROD_VERSION)
    train = f.loc[(f["season"] < season) & (f["season"] >= 2021) & f["played"].astype(bool)]
    cl = T.corner_leagues(train)
    bundles = {}
    for name, task in TASKS.items():
        rows = train.loc[train["league"].isin(cl)] if name == "corners" else train
        p = prod[name]
        b = T.TaskTrainer(task, f.columns).fit(rows, p.params, p.n_estimators)
        b.extra.update({k: v for k, v in p.extra.items() if k != "feature_means"})
        if name == "corners":
            b.extra["leagues"] = cl
        bundles[name] = b
    score = f.loc[(f["season"] == season) & f["played"].astype(bool) & f["home_goals"].notna()].reset_index(drop=True)
    return PR.predict(score, bundles)


def predictions(variant: str) -> pd.DataFrame:
    path = C.CACHE / f"variant_{variant}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    f = build(variant)
    out = pd.concat([fold(f, s).assign(variant=variant) for s in SEASONS], ignore_index=True)
    out.to_parquet(path)
    del f
    gc.collect()
    C.log.info("variant %s: %d predictions", variant, len(out))
    return out


def reproduces_production() -> dict:
    base = build("base")
    prod = C.features()
    cols = [c for c in prod.columns if c in base.columns and c not in ("match_id",)]
    a = prod.set_index("match_id").sort_index()
    b = base.set_index("match_id").sort_index()
    same_ids = a.index.equals(b.index)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(a[c]) and c in b]
    common = a.index.intersection(b.index)
    diff = 0
    for c in num:
        x, y = a.loc[common, c].astype(float), b.loc[common, c].astype(float)
        diff += int(((x - y).abs() > 1e-9).sum() + (x.isna() != y.isna()).sum())
    return {"rows_production": int(len(a)), "rows_rebuilt": int(len(b)), "same_match_ids": bool(same_ids),
            "numeric_columns": len(num), "cells_differing": diff}


def evaluate() -> dict:
    ms = pd.read_parquet(C.CACHE / "team_status.parquet")
    preds = {v: predictions(v) for v in VARIANTS}
    base = preds["base"].set_index("match_id")
    out = {"reproduction": reproduces_production(), "seasons": list(SEASONS), "variants": {}}
    st = ms.set_index("match_id")
    for v in VARIANTS[1:]:
        var = preds[v].set_index("match_id")
        ids = base.index.intersection(var.index)
        b, x = base.loc[ids], var.loc[ids]
        s = st.reindex(ids)
        outside = (s["h_status"] == "outside") | (s["a_status"] == "outside")
        early = outside & (((s["h_status"] == "outside") & (s["h_season_match_no"] <= 10))
                           | ((s["a_status"] == "outside") & (s["a_season_match_no"] <= 10)))
        e3_new = early & (b["league"] == "E3")
        res = {}
        for season in SEASONS + ("all",):
            sel = (b["season"] == season) if season != "all" else pd.Series(True, index=ids)
            part = {}
            for name, mask in (("all", sel), ("outside_first10", sel & early), ("outside_all", sel & outside),
                               ("e3_outside_first10", sel & e3_new)):
                m = mask.to_numpy()
                if m.sum() < 2:
                    continue
                part[name] = {"matches": int(m.sum()),
                              "log_loss_base": float(b.loc[m, "ll"].mean()), "log_loss_var": float(x.loc[m, "ll"].mean()),
                              "d_log_loss": C.paired_diff_ci(b.loc[m, "ll"], x.loc[m, "ll"]),
                              "main_base": C.roi_summary(b.loc[m, "main_profit"], b.loc[m, "main_won"]),
                              "main_var": C.roi_summary(x.loc[m, "main_profit"], x.loc[m, "main_won"]),
                              "d_main_roi": C.paired_nanmean_diff(b.loc[m, "main_profit"], x.loc[m, "main_profit"]),
                              "main_changed": float(((b.loc[m, "main_market"] != x.loc[m, "main_market"])
                                                     | (b.loc[m, "main_selection"] != x.loc[m, "main_selection"])).mean())}
            res[str(season)] = part
        out["variants"][v] = res
    rows = []
    for v, res in out["variants"].items():
        for season, part in res.items():
            for grp, r in part.items():
                rows.append({"variant": v, "season": season, "group": grp, "matches": r["matches"],
                             "log_loss_base": r["log_loss_base"], "d_log_loss": r["d_log_loss"]["mean"],
                             "lo": r["d_log_loss"]["ci95"][0], "hi": r["d_log_loss"]["ci95"][1],
                             "roi_base": r["main_base"]["roi"], "roi_var": r["main_var"]["roi"],
                             "bets_base": r["main_base"]["bets"], "bets_var": r["main_var"]["bets"],
                             "d_roi": r["d_main_roi"]["diff"], "d_roi_lo": r["d_main_roi"]["ci95"][0],
                             "d_roi_hi": r["d_main_roi"]["ci95"][1], "main_changed": r["main_changed"]})
    C.save_csv("variants_summary.csv", pd.DataFrame(rows))
    sel_rule = {}
    for v, res in out["variants"].items():
        ok = all(res[str(s)].get("outside_first10", {}).get("d_log_loss", {}).get("mean", 1) < 0
                 and res[str(s)]["all"]["d_log_loss"]["mean"] <= 0 for s in (2023, 2024))
        sel_rule[v] = bool(ok)
    out["meets_selection_rule"] = sel_rule
    # descriptive (decided after seeing the results, so it can only support a B): lower newcomer log loss in every
    # season and a pooled overall interval below 0
    consistent = {}
    for v, res in out["variants"].items():
        consistent[v] = bool(all(res[str(s)].get("outside_first10", {}).get("d_log_loss", {}).get("mean", 1) < 0
                                 for s in SEASONS) and (res["all"]["all"]["d_log_loss"]["ci95"][1] or 1) < 0)
    out["consistent_post_hoc"] = consistent
    C.save_json("variants.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("variants")
    evaluate()
