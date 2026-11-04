"""The independent check on 2026/27 (run once, after the 'strategy' stage of preregistration.json is committed).

* A as stored: the Main predictions exactly as production recorded them (unified history), real prices
  recorded or restored from the archive — the official baseline (ROI, hit rate, calibration breakdowns);
* V1: the production candidate table (candidates.live) — every strategy, fitted on 2025/26 only;
* V2: the same matches restricted to the Football-Data markets at Football-Data Bet365 opening prices
  (the fold universe), plus closing-line value against the Bet365 closing price.

The pre-registered primary strategy is the confirmatory test; every other variant is shown for information.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from . import calibration as CAL
from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import strategies_eval as SE

PRE = C.RESULTS / "preregistration.json"


def _committed_prereg() -> dict:
    rel = PRE.relative_to(C.ROOT).as_posix()
    try:
        blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=C.ROOT, text=True)
        when = subprocess.check_output(["git", "log", "-1", "--format=%H %cI", "--", rel], cwd=C.ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        raise SystemExit("preregistration.json is not committed")
    doc = json.loads(blob)
    if "strategy" not in doc:
        raise SystemExit("the 'strategy' stage is not committed: pre-register before the 2026/27 check")
    doc["_commit"] = when
    return doc


def stored_baseline() -> dict:
    st = CD.stored_live_mains()
    st["decided"] = st["status"].isin(["WON", "LOST"])
    st["won"] = (st["status"] == "WON").astype(float)
    st["priced"] = st["odds"].notna() & st["decided"]
    st["p_push"] = 0.0
    st["line"] = st["market"].map(lambda m: float(m.split("_")[1]) if "_" in m and m.split("_")[0] in
                                  ("TOTAL", "HANDICAP", "CORNERS") else np.nan)
    st["p"] = st["probability"].astype(float)
    st["profit"] = np.where(st["priced"], st["profit"], np.nan)
    o = CAL.annotate(st)
    dec = o.loc[o["decided"]]
    bets = dec.loc[dec["priced"]]
    lo, hi = C.roi_ci(bets["profit"].to_numpy(float))
    out = {"matches": int(st["match_id"].nunique()), "settled": int(len(dec)), "bets": int(len(bets)),
           "priced_share": float(len(bets) / len(dec)), "hit_rate": float(dec["won"].mean()),
           "bet_hit_rate": float(bets["won"].mean()), "avg_odds": float(bets["odds"].mean()),
           "profit": float(bets["profit"].sum()), "roi": float(bets["profit"].mean()), "roi_ci95": [lo, hi],
           "mean_prob": float(dec["p"].mean()), "bet_mean_prob": float(bets["p"].mean()),
           "breakeven_hit": float(np.mean(1 / bets["odds"])),
           "stated_ev": float(np.mean(bets["p"] * bets["odds"] - 1)),
           "calibration": C.binary_scores(dec["won"], dec["p"]),
           "reliability": C.reliability_bins(dec["won"], dec["p"]),
           "by_group": CAL.breakdown(o, "group"), "by_league": CAL.breakdown(o, "league"),
           "by_odds_band": CAL.breakdown(o.loc[o["priced"]], "odds_band"), "by_prob_band": CAL.breakdown(o, "prob_band"),
           "by_line": CAL.breakdown(o, "line_key"), "by_side": CAL.breakdown(o, "side"),
           "by_source": CAL.breakdown(o, "source"),
           "by_odds_origin": CAL.breakdown(o.assign(oo=o["odds_origin"].fillna("recorded")), "oo"),
           "by_provider": CAL.breakdown(o.assign(pv=o["provider"].fillna("нет цены")), "pv")}
    return out, st


def v2_table(live: pd.DataFrame) -> pd.DataFrame:
    fdm = live["market"].isin(["1X2", "TOTAL_2.5"]) | live["o_b365"].notna()
    d = live.loc[fdm & (live["market"].isin(["1X2", "TOTAL_2.5"]) | (live["group"] == "HANDICAP"))].copy()
    d["odds"] = d["o_b365"]
    for m in ("prop", "shin"):
        d[f"q_prod_{m}"] = d[f"q_b365_{m}"]
    return d


def evaluate(V: pd.DataFrame, fit: SE.Fitted, stored: pd.DataFrame | None = None) -> tuple[dict, dict]:
    u = SE.universe(V)
    R = SE.run_all(u, fit, include_pinnacle=False)
    n = int(u["mi"].max() + 1)
    res = SE.evaluate(R, n)
    if stored is not None:                          # paired difference vs the stored production Main
        mi = pd.Series(np.arange(n), index=pd.factorize(u["match_id"])[1])
        s = stored.loc[stored["match_id"].isin(mi.index)]
        pm = np.full(n, np.nan)
        pm[mi[s["match_id"]].to_numpy()] = s["profit"].to_numpy(float)
        for name, r in R.items():
            o = r.picks.assign(profit=np.where(r.bet, r.picks["profit"], np.nan))
            res[name]["vs_A_stored"] = C.paired_roi_diff(pm, ST.per_match_profit(o, n))
        res["_A_stored_on_universe"] = {"bets": int(np.isfinite(pm).sum()), "roi": float(np.nanmean(pm)),
                                        "roi_ci95": list(C.roi_ci(pm[np.isfinite(pm)]))}
    res["_matches"] = n
    picks = {name: r.picks.assign(bet=r.bet) for name, r in R.items()}
    return res, picks


def clv_live(V2: pd.DataFrame, picks: pd.DataFrame) -> dict:
    u = SE.universe(V2)
    x = u.iloc[picks.loc[picks["bet"].to_numpy(bool), "row"].to_numpy()]      # only the staked picks
    o, oc = x["o_b365"].to_numpy(float), x["o_b365_c"].to_numpy(float)
    qc = x["q_b365_c_shin"].to_numpy(float)
    st = x["status"].to_numpy()
    m = np.isfinite(o) & np.isfinite(oc) & np.isfinite(qc) & np.isin(st, [0, 1])
    prof = np.where(st[m] == 1, o[m] - 1, -1.0)
    return {"n": int(m.sum()), "mean_clv_odds": float(np.mean(o[m] / oc[m] - 1)),
            "share_positive_clv_odds": float(np.mean(o[m] / oc[m] > 1)),
            "mean_clv_fair_b365close": float(np.mean(o[m] * qc[m] - 1)), "roi": float(prof.mean()),
            "roi_at_closing": float(np.mean(np.where(st[m] == 1, oc[m] - 1, -1.0)))}


def run() -> dict:
    pre = _committed_prereg()
    chosen = pre["strategy"]["chosen"]
    primary = chosen["primary"]["name"]
    secondary = [x["name"] for x in chosen["secondary"]]
    base, stored = stored_baseline()
    live = CD.live()
    inner = SE.universe(CD.historical(C.INFO))
    fit = SE.Fitted(inner, True)
    v1, picks1 = evaluate(live, fit, stored)
    v2, picks2 = evaluate(v2_table(live), fit)
    # rule 4 for the primary (and, for information, the secondary ones)
    rule4 = {n: {"V1_diff_vs_A": v1[n]["vs_A"]["diff"] if v1[n]["vs_A"] else None,
                 "V1_roi": v1[n]["roi"], "V1_roi_ci95": v1[n]["roi_ci95"], "V1_bets": v1[n]["bets"],
                 "V2_diff_vs_A": v2[n]["vs_A"]["diff"] if v2[n]["vs_A"] else None, "V2_roi": v2[n]["roi"],
                 "passes_rule4": bool(v1[n]["vs_A"] and v1[n]["vs_A"]["diff"] >= 0)}
             for n in [primary] + secondary if n in v1}
    fold_ok = chosen["primary"].get("passes_fold_rules_1_3", False)
    out = {"preregistration_commit": pre["_commit"], "preregistered_primary": primary, "secondary": secondary,
           "stored_A": base, "V1": v1, "V2": v2, "rule4": rule4, "fitted_on_2025": fit.summary(),
           "adopt": bool(fold_ok and rule4.get(primary, {}).get("passes_rule4", False)),
           "clv_V2": {n: clv_live(v2_table(live), picks2[n]) for n in ("A", primary) if n in picks2},
           "live_table": {"rows": int(len(live)), "matches": int(live["match_id"].nunique()),
                          "price_origin": live["odds_origin"].fillna("нет цены").value_counts().to_dict(),
                          "priced_share_by_group": live.assign(pr=live["odds"].notna()).groupby("group")["pr"].mean().to_dict(),
                          "same_main_as_stored": _same_main(live, stored)}}
    C.save_json("independent.json", out)
    pd.concat([p.assign(strategy=n, universe="V1") for n, p in picks1.items() if n in ("A", primary, *secondary)],
              ignore_index=True).to_parquet(C.CACHE / "independent_picks.parquet")
    figures(out, picks1, stored, primary)
    return out


def _same_main(live: pd.DataFrame, stored: pd.DataFrame) -> float:
    idx = ST.choose(live)
    o = ST.outcome(live, idx)
    m = o.merge(stored[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_st"))
    return float(((m["market"] == m["market_st"]) & (m["selection"] == m["selection_st"])).mean())


def figures(out: dict, picks: dict, stored: pd.DataFrame, primary: str) -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    fig, ax = plt.subplots(figsize=(10, 4.3))
    s = stored.loc[stored["priced"]].sort_values(["date", "match_id"])
    ax.plot(np.arange(len(s)), s["profit"].cumsum().to_numpy(), color="#333", lw=1.4, label="A (как записано)")
    names = ["A", primary] + [n for n in out["secondary"] if n in picks][:4]
    for k, n in enumerate(dict.fromkeys(names)):
        p = picks[n]
        b = p.loc[p["bet"] & p["priced"] & p["decided"]].sort_values(["date", "match_id"])
        ax.plot(np.arange(len(b)), b["profit"].cumsum().to_numpy(), color=PALETTE[k % len(PALETTE)], lw=1.1,
                label=n + (" (предрегистр.)" if n == primary else ""))
    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xlabel("Ставка № (по дате)"), ax.set_ylabel("Накопленная прибыль, ед.")
    ax.legend(fontsize=7)
    ax.set_title("2026/27 (независимая проверка, V1): накопленная прибыль, ставка 1 ед.", fontsize=10)
    save_chart(fig, C.FIGURES / "independent_cumulative.png")
    rel = out["stored_A"]["reliability"]
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    ax.errorbar([b["predicted"] for b in rel], [b["observed"] for b in rel], yerr=[1.96 * b["se"] for b in rel],
                marker="o", ms=4, capsize=2, color=PALETTE[0], label="Main как записано, 2026/27")
    ax.plot([0.35, 0.9], [0.35, 0.9], color="#888", ls="--", lw=0.8)
    ax.set_xlabel("Заявленная вероятность"), ax.set_ylabel("Доля выигрышей"), ax.legend(fontsize=7)
    ax.set_title("Надёжность записанных Main, 2026/27", fontsize=10)
    save_chart(fig, C.FIGURES / "independent_reliability.png")


if __name__ == "__main__":
    C.setup_logging("independent")
    run()
