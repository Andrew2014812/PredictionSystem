"""Transfer of learned patterns between leagues (1X2 classifier, production parameters, seed 42).

For every league L and season s the classifier is fitted on seasons < s of
    LOLO-all     every OTHER league (L's one-hot is unknown -> all zeros: the league is new to the model)
    LOLO-group   only the other leagues of L's group (groups of fold s)
    LOLO-out     only the leagues OUTSIDE L's group
and predicts L in season s. Compared on the same matches with B0 (global model trained with L's own history)
and with M2 (L's own history only, seed 42): the value of a league's own data and of the other leagues' data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import profiles as PR
from . import wf

SEASONS = (2023, 2024, 2025)
KINDS = {"all": "все другие лиги", "group": "только своя группа", "out": "только другие группы"}


def predictions(kind: str, s: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"lolo_{kind}_{s}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = wf.features()
    train, score = wf.before(df, s), wf.season(df, s).reset_index(drop=True)
    groups = PR.groups_for(s)
    out = pd.DataFrame({"match_id": score["match_id"].to_numpy(), "pH": np.nan, "pD": np.nan, "pA": np.nan})
    gtr = train["league"].map(groups).to_numpy()
    for lg in sorted(score["league"].unique()):
        g = groups[lg]
        if kind == "all":
            m = train["league"].to_numpy() != lg
        elif kind == "group":
            m = (gtr == g) & (train["league"].to_numpy() != lg)
        else:
            m = gtr != g
        idx = np.flatnonzero(score["league"].to_numpy() == lg)
        if m.sum() < 500:
            continue
        fit = wf.Fit("result").fit(train.loc[m])
        out.loc[idx, ["pH", "pD", "pA"]] = fit.predict(score.iloc[idx])
    out.to_parquet(path)
    C.log.info("lolo %s %s done", kind, s)
    return out


def build_all() -> None:
    for s in SEASONS:
        for k in KINDS:
            predictions(k, s)


def run() -> dict:
    from . import models as MD
    res = {"kinds": KINDS, "seasons": {}, "pooled": {}}
    rows = []
    for s in SEASONS:
        fr = wf.frame(s)
        ev = fr.loc[fr["in_eval"]].reset_index(drop=True)
        y = ev["y"].to_numpy(int)

        def ll(pred):
            p = ev[["match_id"]].merge(pred, on="match_id", how="left")
            return C.logloss_rows(p[["pH", "pD", "pA"]].to_numpy(float), y)
        base = ll(wf.base_predictions(s, C.PROD_SEED))
        own = ll(MD.predictions("M2", s, C.PROD_SEED))
        blk = pd.DataFrame({"league": ev["league"], "season": s, "B0": base, "own": own})
        for k in KINDS:
            blk[k] = ll(predictions(k, s))
        rows.append(blk)
    allb = pd.concat(rows, ignore_index=True)
    allb.to_parquet(C.CACHE / "lolo_rows.parquet")
    for label, b in (("select", allb.loc[allb["season"].isin(C.SELECT)]), ("2025", allb.loc[allb["season"] == 2025])):
        out = {}
        for col in ("own", *KINDS):
            ok = b[col].notna()
            out[col] = {"pooled": C.paired_diff_ci(b.loc[ok, "B0"], b.loc[ok, col]),
                        "leagues": {lg: C.paired_diff_ci(g.loc[g[col].notna(), "B0"], g.loc[g[col].notna(), col])
                                    for lg, g in b.groupby("league")}}
        # same group vs other groups, head to head
        ok = b["group"].notna() & b["out"].notna()
        out["group_vs_out"] = C.paired_diff_ci(b.loc[ok, "out"], b.loc[ok, "group"])
        res["pooled" if label == "select" else "info_2025"] = out
    C.save_json("transfer.json", res)
    return res


if __name__ == "__main__":
    C.setup_logging("transfer")
    build_all()
    print("TRANSFER_DONE")
