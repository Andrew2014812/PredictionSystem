"""Betting check on real 2026/27 corner prices (the only season with corner prices).

Prices: OddsPapi archive in local_data/odds/odds.parquet (read only), bookmaker Bet365, markets
CORNERS_<line> OVER / UNDER. For every match, line and side the LAST price recorded before kick-off
is used. Football-Data files carry no corner prices, so 2021/22–2025/26 cannot be checked this way.

Strategies (flat stake 1, every model x distribution of fold 2026 — models fitted on 2021/22–2025/26):
    value_t      every side of the product lines 8.5 / 9.5 / 10.5 with EV = p * odds - 1 >= t
    best_t       at most one bet per match: the product-line side with the highest EV, if EV >= t
    ext_value_t  as value_t on the lines 7.5 .. 12.5
ROI confidence intervals: bootstrap over matches (2000 resamples). The ROI difference against the
reference uses the same resampled matches for both.
Market quality: Bet365 probabilities with the margin removed proportionally (q = (1/o) / sum 1/o).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import dist as Q
from .evaluate import EXT_LINES, REF

log = D.log
ODDS_FILE = Path(D.HERE).resolve().parents[2] / "local_data" / "odds" / "odds.parquet"
THRESHOLDS = (0.0, 0.03, 0.05, 0.10)
BOOT = 2000


def prices() -> pd.DataFrame:
    o = pd.read_parquet(ODDS_FILE)
    o = o.loc[o["market"].str.startswith("CORNERS_") & (o["bookmaker"] == "bet365") & o["odds"].notna()]
    o = o.loc[o["priced_at"] < o["kickoff"]].sort_values("priced_at")
    last = o.groupby(["match_id", "market", "selection"]).tail(1)
    w = last.pivot_table(index=["match_id", "market"], columns="selection", values="odds", aggfunc="last").reset_index()
    w["line"] = w["market"].str.replace("CORNERS_", "").astype(float)
    lead = last.assign(lead_h=(last["kickoff"] - last["priced_at"]).dt.total_seconds() / 3600)
    w.attrs["lead_hours"] = lead["lead_h"].describe().to_dict()
    return w.rename(columns={"OVER": "o_over", "UNDER": "o_under"})


def bets_frame(probs: pd.DataFrame, pr: pd.DataFrame, lines) -> pd.DataFrame:
    """Long table: one row per match x line x side with model probability, price and outcome."""
    rows = []
    for line in lines:
        p = probs[["match_id", "y", f"p{line}"]].rename(columns={f"p{line}": "p_over"})
        q = pr.loc[pr["line"] == line, ["match_id", "o_over", "o_under"]]
        m = p.merge(q, on="match_id")
        for side in ("over", "under"):
            prob = m["p_over"] if side == "over" else 1 - m["p_over"]
            odds = m[f"o_{side}"]
            win = (m["y"] > line) if side == "over" else (m["y"] < line)
            rows.append(pd.DataFrame({"match_id": m["match_id"], "line": line, "side": side, "p": prob, "odds": odds,
                                      "win": win.astype(float)}).dropna(subset=["odds"]))
    b = pd.concat(rows, ignore_index=True)
    b["ev"] = b["p"] * b["odds"] - 1
    b["profit"] = np.where(b["win"] > 0, b["odds"] - 1, -1.0)
    return b


def roi_boot(sel: pd.DataFrame, match_index: dict, draws: np.ndarray) -> dict:
    """ROI with a cluster bootstrap over matches; ``draws`` = (BOOT, n_matches) resampled match positions."""
    n = len(sel)
    if n == 0:
        return {"n": 0, "roi": None, "ci95": [None, None], "hit_rate": None, "mean_odds": None, "profit": 0.0}
    pos = sel["match_id"].map(match_index).to_numpy()
    nm = draws.shape[1]
    prof = np.bincount(pos, weights=sel["profit"].to_numpy(), minlength=nm)
    cnt = np.bincount(pos, minlength=nm).astype(float)
    bp, bc = prof[draws].sum(1), cnt[draws].sum(1)
    roi_b = bp / np.maximum(bc, 1)
    return {"n": int(n), "roi": float(sel["profit"].mean()), "ci95": [float(np.quantile(roi_b, 0.025)),
                                                                      float(np.quantile(roi_b, 0.975))],
            "hit_rate": float(sel["win"].mean()), "mean_odds": float(sel["odds"].mean()),
            "profit": float(sel["profit"].sum()), "_prof": prof, "_cnt": cnt}


def strategies(b: pd.DataFrame, ext: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for t in THRESHOLDS:
        out[f"value_{t:g}"] = b.loc[b["ev"] >= t]
        best = b.sort_values("ev", ascending=False).groupby("match_id").head(1)
        out[f"best_{t:g}"] = best.loc[best["ev"] >= t]
        out[f"ext_value_{t:g}"] = ext.loc[ext["ev"] >= t]
    return out


def market_quality(probs: pd.DataFrame, pr: pd.DataFrame) -> dict:
    out = {}
    for line in EXT_LINES:
        q = pr.loc[(pr["line"] == line) & pr["o_over"].notna() & pr["o_under"].notna()]
        m = probs[["match_id", "y", f"p{line}"]].merge(q, on="match_id")
        if m.empty:
            continue
        inv_o, inv_u = 1 / m["o_over"], 1 / m["o_under"]
        q_over = inv_o / (inv_o + inv_u)
        o = (m["y"] > line).to_numpy()
        out[str(line)] = {"n": int(len(m)), "margin": float((inv_o + inv_u - 1).mean()),
                          "market_log_loss": float(Q.binary_ll(o, q_over).mean()),
                          "model_log_loss": float(Q.binary_ll(o, m[f"p{line}"]).mean()),
                          "observed": float(o.mean()), "market_mean_p": float(q_over.mean()),
                          "model_mean_p": float(m[f"p{line}"].mean()),
                          "market_ece": Q.ece(o, q_over), "model_ece": Q.ece(o, m[f"p{line}"].to_numpy()),
                          "corr_model_market": float(np.corrcoef(q_over, m[f"p{line}"])[0, 1])}
    return out


def run() -> dict:
    lp = pd.read_parquet(D.CACHE / "line_probs.parquet")
    lp = lp.loc[lp["season"] == D.CHECK]
    pr = prices()
    matches = sorted(set(lp["match_id"]) & set(pr["match_id"]))
    idx = {m: i for i, m in enumerate(matches)}
    draws = np.random.default_rng(7).integers(0, len(matches), size=(BOOT, len(matches)))
    dates = D.frame().set_index("match_id")["date"]
    out = {"n_matches_priced": len(matches), "n_matches_2026": int(lp["match_id"].nunique()),
           "price_lead_hours": pr.attrs["lead_hours"],
           "lines_priced": pr.loc[pr["match_id"].isin(matches)].groupby("line").size().to_dict(),
           "combos": {}, "market": {}, "by_month": {}}
    ref_cache = {}
    combos = [(m, d) for (m, d), _ in lp.groupby(["model", "dist"])]
    combos.sort(key=lambda x: x != REF)            # reference first
    for model, dk in combos:
        probs = lp.loc[(lp["model"] == model) & (lp["dist"] == dk) & lp["match_id"].isin(matches)]
        b = bets_frame(probs, pr, D.LINES)
        ext = bets_frame(probs, pr, EXT_LINES)
        res = {}
        for name, sel in strategies(b, ext).items():
            r = roi_boot(sel, idx, draws)
            if (model, dk) == REF:
                ref_cache[name] = r
            elif r["n"] and ref_cache.get(name, {}).get("n"):
                rr = ref_cache[name]
                diff = (r["_prof"][draws].sum(1) / np.maximum(r["_cnt"][draws].sum(1), 1)
                        - rr["_prof"][draws].sum(1) / np.maximum(rr["_cnt"][draws].sum(1), 1))
                r["diff_vs_ref"] = {"mean": float(r["roi"] - rr["roi"]),
                                    "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]}
            res[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        # line log loss on the priced matches (product lines)
        res["line_log_loss_priced"] = float(np.mean([Q.binary_ll(probs["y"] > l, probs[f"p{l}"]).mean()
                                                     for l in D.LINES]))
        out["combos"][f"{model}|{dk}"] = res
        out["market"][f"{model}|{dk}"] = market_quality(probs, pr)
        if (model, dk) == REF or dk == "nb_mom":
            sel = b.loc[b["ev"] >= 0].copy()
            sel["month"] = sel["match_id"].map(dates).dt.strftime("%Y-%m")
            out["by_month"][f"{model}|{dk}"] = {m: {"n": int(len(g)), "roi": float(g["profit"].mean())}
                                                for m, g in sel.groupby("month")}
    D.save_result("betting.json", out)
    rows = [{"combo": k, **{f"{s}_{f}": v[s][f] for s in ("value_0", "value_0.05", "best_0", "ext_value_0")
                             for f in ("n", "roi")}, "line_ll_priced": v["line_log_loss_priced"]}
            for k, v in out["combos"].items()]
    pd.DataFrame(rows).to_csv(D.RESULTS / "betting_2026.csv", index=False, float_format="%.4f")
    return out
