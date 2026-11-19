"""Probability metrics and the Main evaluation shared by the calibration and meta-selection studies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

SMALL_BETS = 60          # breakdown cells with fewer bets are marked "small sample"


# ------------------------------------------------------------------------------------------------ probabilities
def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins: int = 10):
    """Expected calibration error: Σ_b n_b / n · |mean p_b − mean y_b| over 10 equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    sp = np.bincount(b, weights=p, minlength=bins)
    sy = np.bincount(b, weights=y, minlength=bins)
    ok = n > 0
    return float(np.sum(np.abs(sp[ok] - sy[ok])) / len(p))


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    d = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    return d.reset_index()


def prob_metrics(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"n": 0}
    return {"n": int(len(p)), "brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece(p, y),
            "mean_p": float(p.mean()), "hit": float(y.mean()), "gap": float(p.mean() - y.mean())}


def per_match_delta(d: pd.DataFrame, a: np.ndarray, b: np.ndarray, kind: str = "log_loss") -> dict:
    """Paired bootstrap (by match) of the mean per-row difference metric(a) − metric(b)."""
    y = d["y"].to_numpy(float)
    if kind == "log_loss":
        f = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))  # noqa: E731
    else:
        f = lambda p: (p - y) ** 2  # noqa: E731
    diff = pd.Series(f(np.asarray(a, float)) - f(np.asarray(b, float))).groupby(d["match_id"].to_numpy()).mean()
    return C.paired_mean_diff(diff.to_numpy(), np.zeros(len(diff)))


# ------------------------------------------------------------------------------------------------ Main
def gap_ci(main: pd.DataFrame, col: str = "stated_p") -> dict:
    """Stated probability − hit rate of the decided Main (WON / LOST), with a bootstrap interval by match."""
    d = main.loc[main["status"].isin(["WON", "LOST"])]
    # Baseline A (site history) has no P(push): treated as 0 (2 whole-line Asian handicaps of 2 389 rows)
    push = d["push_probability"].to_numpy(float) if "push_probability" in d else np.zeros(len(d))
    pc = d[col].to_numpy(float) / np.clip(1 - np.nan_to_num(push), 1e-6, None)
    y = (d["status"] == "WON").to_numpy(float)
    r = C.paired_mean_diff(pc, y)
    return {"gap": r["diff"], "gap_ci": r["ci95"], "n": r["n"], "mean_p": float(pc.mean()), "hit": float(y.mean())}


def main_eval(main: pd.DataFrame, base_b: pd.DataFrame, base_a: pd.DataFrame | None = None) -> dict:
    """Production figures, ROI CI, gap, negative-EV shares and paired comparisons of a one-Main-per-match table."""
    m = main.loc[main["status"].isin(["WON", "LOST", "PUSH"])]
    s = EV.summary(m)
    bets = m.loc[m["bet"] == 1]
    out = {"summary": s, "roi_ci": EV.roi_with_ci(m), "gap": gap_ci(m) if "stated_p" in m else None,
           "neg_ev_share_raw": float((pd.to_numeric(bets["ev"], errors="coerce") < 0).mean()) if len(bets) else None,
           "neg_ev_share_stated": float((bets["ev_stated"] < 0).mean()) if "ev_stated" in bets and len(bets) else None,
           "vs_b": EV.compare(m, base_b), "groups": m["group"].value_counts().to_dict(),
           "fallback": int((m.get("note", pd.Series("", index=m.index)).astype(str).str.startswith("fallback")).sum())}
    if base_a is not None:
        out["vs_a"] = EV.compare(m, base_a)
    return out


def robustness(main: pd.DataFrame, base_b: pd.DataFrame) -> dict:
    """Month / league breakdown (paired with B), dependence on the biggest wins, sensitivity to the price."""
    m = main.loc[main["status"].isin(["WON", "LOST", "PUSH"])].copy()
    b = base_b.loc[base_b["match_id"].isin(m["match_id"])]
    from experiments.v6_3.calibration.data import month_key
    m["month"], b = month_key(m["date"]), b.assign(month=month_key(b["date"]))
    rows = {}
    for by in ("month", "league"):
        x = EV.breakdown(m, by).set_index(by)
        y = EV.breakdown(b, by).set_index(by)
        t = x.join(y[["bets", "roi"]], rsuffix="_b")
        t["roi_diff"] = t["roi"] - t["roi_b"]
        t["small"] = t["bets"] < SMALL_BETS
        rows[by] = t.reset_index().to_dict("records")
    bets = m.loc[m["bet"] == 1].sort_values("profit", ascending=False)
    top = {}
    for k in (1, 5, 10, 20):
        rest = bets.iloc[k:]
        top[str(k)] = float(rest["profit"].sum() / len(rest)) if len(rest) else None
    sens = {}
    for delta in (0.0, 0.02, 0.05):
        o = bets["odds"].astype(float) * (1 - delta)
        prof = np.where(bets["status"] == "WON", o - 1, -1.0)
        sens[str(delta)] = float(prof.mean())
    pos_months = sum(1 for r in rows["month"] if r["roi_diff"] is not None and r["roi_diff"] > 0)
    pos_leagues = sum(1 for r in rows["league"] if r["roi_diff"] is not None and r["roi_diff"] > 0)
    return {"by_month": rows["month"], "by_league": rows["league"], "roi_without_top_wins": top,
            "roi_odds_haircut": sens, "months_better": pos_months, "months": len(rows["month"]),
            "leagues_better": pos_leagues, "leagues": len(rows["league"])}


def coverage(m: pd.DataFrame, mains: dict[str, pd.DataFrame]) -> list[dict]:
    """Season matches, played, processed, priced, few statistics, excluded from ROI with the reason."""
    played = set(m.loc[m["played"].astype(bool), "match_id"])
    few = set(m.loc[(m["h_season_games"] < 3) | (m["a_season_games"] < 3), "match_id"])
    rows = []
    for name, t in mains.items():
        t = t.loc[t["match_id"].isin(played)]
        rows.append({"method": name, "season_matches": int(len(m)), "played": len(played),
                     "processed": int(t["match_id"].nunique()), "real_price": int(t["odds"].notna().sum()),
                     "few_statistics": int(t["match_id"].isin(few).sum()), "bets": int(t["bet"].sum()),
                     "excluded_no_price": int((t["odds"].isna() & t["status"].isin(["WON", "LOST"])).sum()),
                     "excluded_push_void": int(t["status"].isin(["PUSH", "VOID"]).sum()),
                     "excluded_unknown": int((~t["status"].isin(["WON", "LOST", "PUSH", "VOID"])).sum()),
                     "not_played": int(len(m) - len(played))})
    return rows


def with_teams(t: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Fill home / away team names (the candidate tables do not carry them) for the per-match tables."""
    names = m.set_index("match_id")[["home_team", "away_team"]]
    t = t.copy()
    for col in ("home_team", "away_team"):
        known = t[col] if col in t else pd.Series(np.nan, index=t.index)
        t[col] = known.where(known.notna() & (known.astype(str) != ""), t["match_id"].map(names[col]))
    return t


def short(r: dict) -> dict:
    """Compact row for tables: ROI / hit / odds / bets / CI / paired diff / p."""
    s, v = r["summary"], r["vs_b"]
    return {"bets": s["bets"], "roi": s["roi"], "roi_ci": r["roi_ci"]["ci95"], "hit": s["hit_rate"],
            "avg_odds": s["avg_odds"], "neg_ev_raw": r["neg_ev_share_raw"], "neg_ev_stated": r["neg_ev_share_stated"],
            "gap": (r["gap"] or {}).get("gap"), "gap_ci": (r["gap"] or {}).get("gap_ci"),
            "d_roi": v["roi_diff"]["diff"], "d_roi_ci": v["roi_diff"]["ci95"], "p": v["roi_diff"]["p"],
            "same_main": v["same_main_share"]}
