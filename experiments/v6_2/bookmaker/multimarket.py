"""2026/27 (for information, not a holdout): the stored model probabilities of every market group vs the de-margined
OddsPapi CLOSING market (archive: the last quote before kick-off, one bookmaker per market — Pinnacle when it
priced the whole market, else Bet365).

Per (match, market) one binary event (OVER / YES / home covers / first double-chance selection) or the 1X2
three-way outcome; log loss of model vs market on the same events, cluster bootstrap by match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_2.odds_timing import clv as CLV
from experiments.v6_2.odds_timing import quotes as Q

from . import common as C

FIRST_SEL = {"TOTAL": "OVER", "CORNERS": "OVER", "BTTS": "YES", "HANDICAP": "H"}


def _group(m: str) -> str:
    return m if m in ("1X2", "BTTS", "DOUBLE_CHANCE") else m.split("_")[0]


def run() -> dict:
    from src.analytics.performance import unified_history
    from src.markets.markets import settle_selection
    hist = unified_history(pd.read_parquet(C.ROOT / "local_data/predictions/picks.parquet"))
    hist = hist.loc[hist["season"] == C.LIVE]
    src = hist.drop_duplicates("match_id").set_index("match_id")["source"]
    q = Q.load()
    fair = CLV._odp_fair(Q.closing(q))
    mk = pd.read_parquet(C.ROOT / "local_data/predictions/markets.parquet",
                         columns=["match_id", "market", "selection", "probability", "push_probability"])
    mk = mk.loc[mk["match_id"].isin(src.index)].drop_duplicates(["match_id", "market", "selection"], keep="last")
    mk["group"] = mk["market"].map(_group)
    mk = mk.loc[mk["group"].isin(["1X2", "TOTAL", "BTTS", "CORNERS", "HANDICAP"])]
    mk = mk.loc[~(mk["group"].eq("HANDICAP") & (mk["push_probability"].fillna(0) > 0))]
    m = mk.merge(fair, on=["match_id", "market", "selection"], how="inner")
    f = pd.read_parquet(C.ROOT / "local_data/processed/features.parquet",
                        columns=["match_id", "home_goals", "away_goals", "total_corners", "played"])
    f = f.loc[f["played"].astype(bool)].set_index("match_id")
    m = m.loc[m["match_id"].isin(f.index)].copy()
    r = f.reindex(m["match_id"])
    m["res"] = [settle_selection(a, b, h, g, c) for a, b, h, g, c in
                zip(m["market"], m["selection"], r["home_goals"], r["away_goals"], r["total_corners"])]
    m = m.loc[m["res"].isin(["WON", "LOST"])]
    m["e"] = (m["res"] == "WON").astype(float)
    m["source"] = m["match_id"].map(src)
    rows = []
    # 1X2: three-way
    x = m.loc[m["group"] == "1X2"]
    wp = x.pivot_table(index="match_id", columns="selection", values="probability")
    wq = x.pivot_table(index="match_id", columns="selection", values="q")
    we = x.pivot_table(index="match_id", columns="selection", values="e")
    ok = wp.index.intersection(wq.dropna().index).intersection(we.dropna().index)
    if len(ok):
        p = wp.loc[ok, ["H", "D", "A"]].to_numpy(float)
        p = p / p.sum(1, keepdims=True)
        qq = wq.loc[ok, ["H", "D", "A"]].to_numpy(float)
        y = we.loc[ok, ["H", "D", "A"]].to_numpy().argmax(1)
        rows.append(pd.DataFrame({"match_id": ok, "group": "1X2", "market": "1X2",
                                  "d": C.logloss_rows(p, y) - C.logloss_rows(qq, y)}))
    for grp, sel in FIRST_SEL.items():
        x = m.loc[(m["group"] == grp) & (m["selection"] == sel)]
        if x.empty:
            continue
        p = np.clip(x["probability"].to_numpy(float), 1e-6, 1 - 1e-6)
        qq = np.clip(x["q"].to_numpy(float), 1e-6, 1 - 1e-6)
        e = x["e"].to_numpy()
        d = -(e * np.log(p) + (1 - e) * np.log(1 - p)) + (e * np.log(qq) + (1 - e) * np.log(1 - qq))
        rows.append(pd.DataFrame({"match_id": x["match_id"].to_numpy(), "group": grp, "market": x["market"].to_numpy(),
                                  "d": d}))
    allr = pd.concat(rows, ignore_index=True)
    allr["source"] = allr["match_id"].map(src)
    out = {"matches": int(allr["match_id"].nunique()), "by_group": {}, "by_market": {}, "by_group_source": {}}
    for grp, g in allr.groupby("group"):
        out["by_group"][grp] = C.grouped_mean_ci(g["d"].to_numpy(), g["match_id"].to_numpy(), 1000)
        out["by_group"][grp]["share_model_better"] = float((g["d"] < 0).mean())
    for mkt, g in allr.groupby("market"):
        if len(g) >= 100:
            out["by_market"][mkt] = C.grouped_mean_ci(g["d"].to_numpy(), g["match_id"].to_numpy(), 1000)
    for (grp, s), g in allr.groupby(["group", "source"]):
        out["by_group_source"][f"{grp}|{s}"] = C.grouped_mean_ci(g["d"].to_numpy(), g["match_id"].to_numpy(), 1000)
    C.save_json("multimarket_2026.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("multi")
    import json
    print(json.dumps(C.clean(run()), indent=1)[:3000])
