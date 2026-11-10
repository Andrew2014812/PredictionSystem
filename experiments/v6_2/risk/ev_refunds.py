"""Expected value of bets with refunds: Asian whole lines (PUSH) and quarter lines (half win / half loss).

Unit stake, decimal odds o. Outcome probabilities come from the model score matrix.

    half line (±0.5, ±1.5, ...):   EV = p_w o - 1
    whole line (±1, ±2):           win p_w, refund p_r, loss p_l = 1 - p_w - p_r
                                   EV = p_w (o - 1) - p_l = p_w o - (1 - p_r)
    quarter line L (±0.25, ±0.75, ...) = half the stake on L - 1/4 and half on L + 1/4:
                                   EV = 1/2 EV(L - 1/4) + 1/2 EV(L + 1/4) at the same price o
                                      = p_W o + p_HW (o + 1)/2 - p_HL / 2 - p_L ... - 1 + (stake returned)
       with the four results full win W (both halves win), half win HW (one half wins, the other refunded),
       half loss HL (one half refunded, the other loses) and full loss L:
                                   EV = p_W (o - 1) + p_HW (o - 1)/2 - p_HL/2 - p_L

Production (src/markets/odds.py) computes EV = p_w o - (1 - p_r) — correct for half and whole lines; quarter lines
are never priced (all three providers drop them), so no quarter-line EV is computed. This module checks the stored
numbers and quantifies the inconsistencies around refunds that DO exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

from . import common as C


# ------------------------------------------------------------------------------------------------ formulas
def ev_half(p_w, o):
    return p_w * o - 1


def ev_whole(p_w, p_r, o):
    return p_w * o - (1 - p_r)


def ev_quarter(p_lo_w, p_lo_r, p_hi_w, p_hi_r, o):
    """Average of the two half-stakes (lines L - 1/4 and L + 1/4, one of them a half line with p_r = 0)."""
    return 0.5 * ev_whole(p_lo_w, p_lo_r, o) + 0.5 * ev_whole(p_hi_w, p_hi_r, o)


def worked_example() -> dict:
    """Formulas on a single illustrative score distribution (independent Poisson 1.6 : 1.1), home side."""
    from scipy.stats import poisson
    k = np.arange(11)
    m = np.outer(poisson.pmf(k, 1.6), poisson.pmf(k, 1.1))
    m /= m.sum()
    i, j = np.indices(m.shape)
    diff = i - j

    def side(line):            # home team with handicap ``line``: (P win, P refund)
        x = diff + line
        return float(m[x > 0].sum()), float(m[x == 0].sum())
    o = 1.90
    rows = []
    for line in (-0.5, -1.0, -0.75, -0.25):
        if line % 0.5 == 0:
            w, r = side(line)
            ev = ev_whole(w, r, o)
            naive = w * o - 1
            rows.append({"line": line, "kind": "whole" if r > 0 else "half", "p_win": w, "p_refund": r,
                         "ev": ev, "ev_without_refund": naive})
        else:
            lo, hi = line - 0.25, line + 0.25
            (w1, r1), (w2, r2) = side(lo), side(hi)
            ev = ev_quarter(w1, r1, w2, r2, o)
            rows.append({"line": line, "kind": "quarter", "halves": [lo, hi], "p_win_halves": [w1, w2],
                         "p_refund_halves": [r1, r2], "ev": ev,
                         "ev_if_treated_as_lower": ev_whole(w1, r1, o), "ev_if_treated_as_upper": ev_whole(w2, r2, o)})
    return {"odds": o, "lambda_home": 1.6, "lambda_away": 1.1, "rows": rows}


# ------------------------------------------------------------------------------------------------ stored data
def stored_checks() -> dict:
    from src.prediction import store
    mk = store.load_markets()
    ok = mk["odds"].notna() & mk["ev"].notna()
    push = mk["push_probability"].fillna(0.0)
    e = mk["probability"] * mk["odds"] - (1 - push)
    whole = ok & (push > 0)
    picks = store.load_picks()
    last = mk.drop_duplicates(["match_id", "market", "selection"], keep="last")
    x = picks.merge(last[["match_id", "market", "selection", "push_probability"]], how="left",
                    on=["match_id", "market", "selection"])
    okp = x["odds"].notna() & x["ev"].notna() & x["push_probability"].notna()
    ep = x["probability"] * x["odds"] - (1 - x["push_probability"])
    wp = okp & (x["push_probability"] > 0)
    odds = pd.read_parquet(config.ODDS_DIR / "odds.parquet", columns=["market", "line"])
    hl = odds.loc[odds["market"].str.startswith("HANDICAP"), "line"].astype(float)
    tl = odds.loc[odds["market"].str.startswith("TOTAL"), "line"].astype(float)
    return {"markets_rows_priced": int(ok.sum()), "markets_whole_line_priced": int(whole.sum()),
            "markets_max_abs_diff": float((e[ok] - mk.loc[ok, "ev"]).abs().max()),
            "picks_priced_with_push_known": int(okp.sum()), "picks_whole_line_priced": int(wp.sum()),
            "picks_max_abs_diff": float((ep[okp] - x.loc[okp, "ev"]).abs().max()),
            "picks_whole_line_matching_no_refund_formula": int(((x["probability"] * x["odds"] - 1 - x["ev"]).abs()[wp]
                                                                 < 1e-6).sum()),
            "archive_handicap_quotes": int(len(hl)),
            "archive_quarter_handicap_quotes": int(((hl * 4) % 2 == 1).sum()),
            "archive_quarter_total_quotes": int(((tl * 4) % 2 == 1).sum()),
            "archive_handicap_lines": sorted(float(v) for v in hl.unique())}


def scale_mismatch() -> dict:
    """Stated probability vs hit rate for whole-line picks: production shows P(win) but counts the hit rate over won
    and lost picks only (src/analytics/performance.py), so the comparable stated number is P(win | no refund)."""
    from src.analytics.performance import unified_history
    from src.prediction import store
    picks = unified_history(store.load_picks())
    mk = store.load_markets().drop_duplicates(["match_id", "market", "selection"], keep="last")
    x = picks.merge(mk[["match_id", "market", "selection", "push_probability"]], how="left",
                    on=["match_id", "market", "selection"])
    x["push_probability"] = x["push_probability"].fillna(0.0)
    out = {}
    for role in ("main", "risk"):
        r = x.loc[(x["role"] == role) & (x["season"] == 2026)]
        dec = r.loc[r["status"].isin(["WON", "LOST"])]
        whole = dec.loc[dec["push_probability"] > 0]
        pushes = r.loc[r["status"] == "PUSH"]
        pc = dec["probability"] / (1 - dec["push_probability"])
        out[role] = {"settled": int(len(dec)), "whole_line_settled": int(len(whole)), "push": int(len(pushes)),
                     "avg_probability_shown": float(dec["probability"].mean()),
                     "avg_probability_no_refund_scale": float(pc.mean()),
                     "hit_rate": float((dec["status"] == "WON").mean())}
        if len(whole):
            out[role]["whole"] = {"p_win": float(whole["probability"].mean()),
                                  "p_win_given_no_refund": float((whole["probability"] / (1 - whole["push_probability"])).mean()),
                                  "p_refund": float(whole["push_probability"].mean()),
                                  "hit_rate": float((whole["status"] == "WON").mean())}
            b = whole.loc[whole["odds"].notna()]
            if len(b):
                ev = b["probability"] * b["odds"] - (1 - b["push_probability"])
                out[role]["whole"].update({"bets": int(len(b)), "mean_ev_per_stake": float(ev.mean()),
                                           "mean_ev_per_decided_stake": float((ev / (1 - b["push_probability"])).mean()),
                                           "roi_production": float(b["profit"].mean())})
    return out


def fair_odds_effect(main_tables, risk_tables) -> dict:
    """Unpriced selections are ranked at 'effective odds' 1 / p_win (src/markets/recommendation.effective_odds).
    For an Asian whole line the break-even price is (1 - p_refund) / p_win, so the rule sees a longer price than the
    selection really has. Re-run the production Main and Risk rules with the corrected effective odds."""
    from experiments.v6_1.main import strategies as ST
    from experiments.v6_1.risk import strategies as RS
    out = {"main": {}, "risk": {}}
    for s, d in main_tables:
        P = ST.CURRENT
        idx0 = ST.choose(d, P)
        dd = d.copy()
        # rule_score uses 1 / p for unpriced rows: feed it p' with 1 / p' = (1 - push) / p only for the odds term
        p = dd["p"].to_numpy(float)
        push = dd["p_push"].to_numpy(float)
        odds = dd["odds"].to_numpy(float)
        priced = np.isfinite(odds) & (odds > 1)
        eff = np.where(priced, odds, (1 - push) / np.clip(p, 1e-9, None))
        rel = dd["group"].map(P.rel()).fillna(0.8).to_numpy(float)
        base = rel * np.clip(p, 0, None) ** P.power * ST.odds_preference(eff, P.preferred, P.softness)
        e = ST.ev(p, odds, push)
        bonus = 1 + P.ev_weight * np.clip(np.nan_to_num(e, nan=0.0), -P.ev_clip, P.ev_clip)
        score = np.where(priced, base * bonus, base * P.unpriced_factor)
        ok = (~dd["group"].isin(ST.EXCLUDED).to_numpy()) & (p >= P.min_p) & (p <= P.max_p)
        idx1 = ST.pick(dd, np.where(ok, score, -np.inf))
        out["main"][str(s)] = _compare(d, idx0, idx1, "p_push")
    for s, d in risk_tables:
        cur = RS.current(d)
        dd = d.copy()
        p = dd["p"].to_numpy(float)
        push = dd["push_p"].fillna(0.0).to_numpy(float)
        eff = np.where(dd["priced"], dd["odds"], (1 - push) / np.maximum(p, 1e-9))
        pref = RS.odds_preference(eff, *RS.PREFERRED, RS.SOFTNESS)
        base = dd["rel"].to_numpy() * np.maximum(p, 0.0) ** RS.POWER * pref
        ev = np.clip(np.nan_to_num(dd["ev"].to_numpy(float)), -RS.EV_CLIP, RS.EV_CLIP)
        sc = np.where(dd["priced"], base * (1 + RS.EV_WEIGHT * ev), base * RS.UNPRICED)
        new = RS.choose(dd, [dd["cand"]], sc)
        a = cur.set_index("match_id")
        b = new.set_index("match_id").reindex(a.index)
        changed = (a["market"] != b["market"]) | (a["selection"] != b["selection"])
        res = {"matches": int(len(a)), "changed": int(changed.sum()),
               "changed_whole_line_before": int(((a["push_p"] > 0) & changed).sum()),
               "changed_whole_line_after": int(((b["push_p"] > 0) & changed).sum())}
        for lab, x in (("before", a.loc[changed]), ("after", b.loc[changed])):
            dec = x.loc[x["status"].isin(["WON", "LOST"])]
            bets = dec.loc[dec["priced"]]
            res[lab] = {"hit": float((dec["status"] == "WON").mean()) if len(dec) else None,
                        "mean_p_no_refund": float((dec["p"] / (1 - dec["push_p"].fillna(0))).mean()) if len(dec) else None,
                        "eff_odds": float(np.where(x["priced"], x["odds"], (1 - x["push_p"].fillna(0)) / x["p"]).mean())
                        if len(x) else None,
                        "shown_eff_odds": float(np.where(x["priced"], x["odds"], 1 / x["p"]).mean()) if len(x) else None,
                        "bets": int(len(bets)),
                        "roi": float(np.where(bets["status"] == "WON", bets["odds"] - 1, -1).mean()) if len(bets) else None}
        out["risk"][str(s)] = res
    return out


def _compare(d, idx0, idx1, push_col):
    a = d.iloc[idx0].set_index("mi")
    b = d.iloc[idx1].set_index("mi").reindex(a.index)
    changed = (a["market"] != b["market"]) | (a["selection"] != b["selection"])
    res = {"matches": int(len(a)), "changed": int(changed.sum()),
           "changed_whole_line_before": int(((a[push_col] > 0) & changed).sum()),
           "changed_whole_line_after": int(((b[push_col] > 0) & changed).sum())}
    for lab, x in (("before", a.loc[changed]), ("after", b.loc[changed])):
        dec = x.loc[x["status"].isin([0, 1])]
        res[lab] = {"hit": float((dec["status"] == 1).mean()) if len(dec) else None,
                    "mean_p_no_refund": float((dec["p"] / (1 - dec[push_col])).mean()) if len(dec) else None}
    return res


def quarter_line_hypothetical() -> dict:
    """If quarter lines were added: the EV error of treating a quarter line L as its neighbouring half / whole line,
    on the production probabilities of 2026/27 (adjacent stored lines; price = the half-line price of the match)."""
    from src.prediction import store
    mk = store.load_markets()
    mk = mk.loc[mk["group"] == "HANDICAP"].drop_duplicates(["match_id", "market", "selection"], keep="last")
    mk["line"] = mk["market"].str.split("_").str[1].astype(float)
    w = mk.pivot_table(index=["match_id", "selection"], columns="line", values=["probability", "push_probability"],
                       aggfunc="first")
    rows = []
    for (lo, hi) in ((-2.5, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)):
        q = (lo + hi) / 2 if abs(hi - lo) == 0.5 else None
        if q is None:
            continue
        need = [("probability", lo), ("probability", hi), ("push_probability", lo), ("push_probability", hi)]
        if not all(c in w.columns for c in need):
            continue
        x = w[need].dropna()
        if x.empty:
            continue
        sel = x.index.get_level_values("selection")
        # selection H uses the home line; A of market L is the away side of the same market
        pw_lo, pw_hi = x[("probability", lo)].to_numpy(), x[("probability", hi)].to_numpy()
        pr_lo, pr_hi = x[("push_probability", lo)].to_numpy(), x[("push_probability", hi)].to_numpy()
        # for the home side a higher line is easier; for the away side the order is reversed but the average is
        # symmetric, so the quarter EV is the mean of the two halves in both cases
        o = 1.0 / np.clip(0.5 * (pw_lo + pw_hi) + 0.25 * (pr_lo + pr_hi), 1e-6, None) * 0.95   # a ~5 % margin price
        ev_q = ev_quarter(pw_lo, pr_lo, pw_hi, pr_hi, o)
        rows.append({"quarter_line": q, "n": int(len(x)), "home_rows": int((sel == "H").sum()),
                     "mean_ev_quarter": float(ev_q.mean()),
                     "mean_abs_err_as_lower": float(np.abs(ev_whole(pw_lo, pr_lo, o) - ev_q).mean()),
                     "mean_abs_err_as_upper": float(np.abs(ev_whole(pw_hi, pr_hi, o) - ev_q).mean()),
                     "mean_abs_err_ignoring_refund": float(np.abs(0.5 * (pw_lo + pw_hi) * o - 1 - ev_q).mean())})
    return {"note": "illustrative price o = 0.95 / (mean of the two halves' win-or-half probabilities)", "rows": rows}


def run(main_tables, risk_tables) -> dict:
    """``main_tables`` / ``risk_tables``: iterables of (season, table) (generators: one table in memory at a time)."""
    out = {"example": worked_example(), "stored": stored_checks(), "scale": scale_mismatch(),
           "fair_odds": fair_odds_effect(main_tables, risk_tables), "quarter": quarter_line_hypothetical()}
    C.save_json("ev_refunds.json", out)
    return out
