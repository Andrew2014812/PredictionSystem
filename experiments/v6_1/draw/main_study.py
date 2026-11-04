"""Step 4 (questions 2 and 4) — why a draw is never Main, and what a draw as Main would have done.

Sets: walk-forward blocks 2023–2026 (production engine, Main / Risk exactly as the website chooses them, real
pre-match Bet365 prices) and the 2026/27 production history (recorded Main picks and recorded pre-match draw
prices). Profit of a bet: win = odds − 1, loss = −1; refunds (PUSH) and selections without a real price are not
bets — the same definition the website uses.

Q2 — decomposition of the Main score of the draw with the production functions (src/markets/recommendation.py):
    gate      P(D) must be within [MAIN_MIN_PROBABILITY, MAIN_MAX_PROBABILITY] = [0.40, 0.85]
    odds      odds_preference(draw odds) with the preferred window 1.40–2.50 and softness 0.15
    score     reliability × p × odds_preference × (1 + 0.5 × clip(EV, ±0.3))
    counterfactuals: Main re-chosen with the gate lowered to 0.25, and also with the window widened to 1.40–3.60.

Q4 — "draw override" policies: Main := 1X2 draw when P(D) >= t, |P(H) − P(A)| <= g and the real draw price
gives EV = P(D) × odds − 1 >= e. Grid t ∈ {0.27, 0.28, 0.30, 0.32}, g ∈ {0.10, 0.20, 1.00}, e ∈ {0, 0.03, 0.05,
0.10, 0.15}. Pre-defined criterion for a production change: on the pooled selection folds the policy with the best
change of the overall Main ROI (at least 30 draw bets per selection fold); it is convincing only if the change is
positive on both selection folds, its pooled bootstrap 95 % CI is above zero, and it stays positive on the 2026/27
block and on the 2026/27 production history.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import config
from src.markets.markets import settle_selection
from src.markets.odds import PricedSelection
from src.markets.recommendation import main_params, main_prediction, odds_preference, rule_score

from . import blocks as B
from . import common as C
from . import history as Hm

log = C.setup_logging("draw61.main")
POLICY_T, POLICY_G, POLICY_E = (0.27, 0.28, 0.30, 0.32), (0.10, 0.20, 1.00), (0.0, 0.03, 0.05, 0.10, 0.15)
MIN_DRAW_BETS = 30
EV_BINS = [-1, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 5]
ODDS_BINS = [1, 2.8, 3.1, 3.4, 3.7, 4.0, 5.0, 50]


def _profit(status, odds):
    """Profit of a bet; NaN when it is not a bet (no real price, refund, unsettled)."""
    odds = np.asarray(odds, float)
    priced = np.isfinite(odds) & (odds > 1)
    return np.where(~priced, np.nan, np.where(status == "WON", odds - 1, np.where(status == "LOST", -1.0, np.nan)))


def block_frame(b: int) -> tuple[pd.DataFrame, dict]:
    """One row per match with Main (status / profit at the real price) and the draw price; plus selections."""
    pred, mk = B.block(b)
    fr = pred.copy()
    st = [settle_selection(m, s, h, a, c) if isinstance(m, str) else None
          for m, s, h, a, c in zip(fr["main_market"], fr["main_selection"], fr["home_goals"], fr["away_goals"],
                                   fr["total_corners"])]
    fr["main_status"] = st
    fr["main_profit"] = _profit(fr["main_status"].to_numpy(), fr["main_odds"].to_numpy(float))
    st = [settle_selection(m, s, h, a, c) if isinstance(m, str) else None
          for m, s, h, a, c in zip(fr["risk_market"], fr["risk_selection"], fr["home_goals"], fr["away_goals"],
                                   fr["total_corners"])]
    fr["risk_status"] = st
    fr["risk_profit"] = _profit(fr["risk_status"].to_numpy(), fr["risk_odds"].to_numpy(float))
    d = mk.loc[(mk["market"] == "1X2") & (mk["selection"] == "D")].set_index("match_id")
    fr["draw_odds"] = fr["match_id"].map(d["odds"]).astype(float)
    sels = {}
    for r in mk.itertuples(index=False):
        sels.setdefault(r.match_id, []).append(PricedSelection(
            r.market, r.selection, float(r.probability), None if pd.isna(r.odds) else float(r.odds),
            push_probability=float(0 if pd.isna(r.push_probability) else r.push_probability)))
    return fr, sels


def history_frame() -> pd.DataFrame:
    h = Hm.history().copy()
    h["draw_odds"] = h["odds_draw"]
    h["main_odds"] = h["main_odds"].astype(float)
    h["main_profit"] = _profit(h["main_status"].to_numpy(), h["main_odds"].to_numpy())
    h["risk_odds"] = h["risk_odds"].astype(float)
    h["risk_profit"] = _profit(h["risk_status"].to_numpy(), h["risk_odds"].to_numpy())
    return h


# ---------------------------------------------------------------------------------------------- Q2
def why_not_main(fr: pd.DataFrame, sels: dict | None) -> dict:
    p = fr[["pH", "pD", "pA"]].to_numpy()
    pdraw, odds = p[:, 1], fr["draw_odds"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1)
    pref = np.array([odds_preference(o) if ok else np.nan for o, ok in zip(odds, priced)])
    ev = pdraw * odds - 1
    params = main_params()
    score_d = np.array([rule_score(PricedSelection("1X2", "D", float(q), float(o) if ok else None), params)
                        for q, o, ok in zip(pdraw, odds, priced)])
    score_m = np.array([rule_score(PricedSelection(m, s, float(q), float(o) if o == o else None), params)
                        if isinstance(m, str) else np.nan
                        for m, s, q, o in zip(fr["main_market"], fr["main_selection"], fr["main_probability"].astype(float),
                                              fr["main_odds"].astype(float))])
    out = {
        "n": int(len(fr)), "main_is_draw": int(((fr["main_market"] == "1X2") & (fr["main_selection"] == "D")).sum()),
        "risk_is_draw": int(((fr["risk_market"] == "1X2") & (fr["risk_selection"] == "D")).sum()),
        "share_p_draw_ge_main_min": float((pdraw >= config.MAIN_MIN_PROBABILITY).mean()),
        "share_p_draw_ge_0.33": float((pdraw >= 0.33).mean()), "share_p_draw_ge_0.30": float((pdraw >= 0.30).mean()),
        "p_draw_quantiles": {str(q): float(np.quantile(pdraw, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)},
        "draw_priced_share": float(priced.mean()),
        "draw_odds_quantiles": {str(q): float(np.nanquantile(odds[priced], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "draw_odds_mean": float(odds[priced].mean()),
        "odds_preference_at_draw_odds_mean": float(np.nanmean(pref)),
        "odds_preference_at_draw_odds_quantiles": {str(q): float(np.nanquantile(pref, q)) for q in (0.05, 0.5, 0.95)},
        "draw_ev_mean": float(np.nanmean(ev[priced])), "draw_ev_ge_0_share": float((ev[priced] >= 0).mean()),
        "draw_ev_quantiles": {str(q): float(np.nanquantile(ev[priced], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "main_odds_mean": float(np.nanmean(fr["main_odds"].astype(float))),
        "main_probability_mean": float(fr["main_probability"].astype(float).mean()),
        "main_ev_mean": float(np.nanmean(fr["main_probability"].astype(float) * fr["main_odds"].astype(float) - 1)),
        "draw_score_ungated_mean": float(np.nanmean(score_d)), "main_score_mean": float(np.nanmean(score_m)),
        "main_to_draw_score_ratio_median": float(np.nanmedian(score_m / score_d)),
        "draw_score_beats_main_share_if_ungated": float(np.nanmean(score_d > score_m)),
    }
    # the best possible draw: highest P(D) at the highest EV clip
    out["max_draw_score_possible"] = float(np.nanmax(score_d))
    # upper bound: no gate, odds factor 1 and the full EV bonus (1 + 0.5 x 0.3) — could the draw then win?
    ceiling = config.MARKET_RELIABILITY["1X2"] * pdraw * (1 + params.ev_weight * 0.3)
    out["draw_score_ceiling_mean"] = float(ceiling.mean())
    out["share_main_score_below_draw_ceiling"] = float(np.nanmean(score_m < ceiling))
    out["main_score_quantiles"] = {str(q): float(np.nanquantile(score_m, q)) for q in (0.01, 0.05, 0.25, 0.5)}
    if sels is not None:
        cf = {}
        for tag, prm in (("gate_0.25", params.with_(min_probability=0.25)),
                         ("gate_0.25_window_3.60", params.with_(min_probability=0.25, preferred=(1.40, 3.60))),
                         ("gate_0.25_window_3.60_no_ev", params.with_(min_probability=0.25, preferred=(1.40, 3.60), ev_weight=0.0))):
            res_by = {"H": fr["home_goals"].to_numpy(), "A": fr["away_goals"].to_numpy()}
            picks = [main_prediction(sels[m], prm) for m in fr["match_id"]]
            is_draw = np.array([r is not None and r.pick.market == "1X2" and r.pick.selection == "D" for r in picks])
            prof = []
            for r, h, a, c in zip(picks, res_by["H"], res_by["A"], fr["total_corners"]):
                if r is None or not r.pick.has_odds:
                    prof.append(np.nan)
                    continue
                s = settle_selection(r.pick.market, r.pick.selection, h, a, c)
                prof.append(r.pick.odds - 1 if s == "WON" else (-1.0 if s == "LOST" else np.nan))
            prof = np.array(prof)
            cf[tag] = {"main_is_draw": int(is_draw.sum()), "main_roi": C.roi_ci(prof[np.isfinite(prof)]),
                       "draw_main_roi": C.roi_ci(prof[is_draw & np.isfinite(prof)])}
        out["counterfactual_main"] = cf
    return out


# ---------------------------------------------------------------------------------------------- Q4
def policy_mask(fr, t, g, e):
    p = fr[["pH", "pD", "pA"]].to_numpy()
    o = fr["draw_odds"].to_numpy(float)
    ok = np.isfinite(o) & (o > 1)
    ev = p[:, 1] * np.where(ok, o, np.nan) - 1
    return ok & (p[:, 1] >= t) & (np.abs(p[:, 0] - p[:, 2]) <= g) & (ev >= e)


def override(fr, mask, n_boot=2000, seed=5) -> dict:
    y = fr["y"].to_numpy()
    o = fr["draw_odds"].to_numpy(float)
    old = fr["main_profit"].to_numpy(float)                   # NaN = not a bet
    draw_prof = np.where(y == 1, o - 1, -1.0)
    new = np.where(mask, draw_prof, old)
    s_old, s_new = np.isfinite(old), np.isfinite(new)
    p_old, p_new = np.nan_to_num(old), np.nan_to_num(new)
    roi_old, roi_new = p_old.sum() / s_old.sum(), p_new.sum() / s_new.sum()
    rng = np.random.default_rng(seed)
    n = len(fr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        boots[i] = p_new[k].sum() / s_new[k].sum() - p_old[k].sum() / s_old[k].sum()
    replaced = mask & s_old
    ev_old = (fr["main_probability"].astype(float) * fr["main_odds"].astype(float) - 1).to_numpy()
    ev_new = np.where(mask, fr["pD"].to_numpy() * o - 1, ev_old)
    return {"main_mean_ev_before": float(np.nanmean(ev_old)), "main_mean_ev_after": float(np.nanmean(ev_new)),
            "main_avg_odds_before": float(np.nanmean(np.where(s_old, fr["main_odds"].astype(float), np.nan))),
            "main_avg_odds_after": float(np.nanmean(np.where(mask, o, np.where(s_old, fr["main_odds"].astype(float), np.nan)))),
            "main_hit_rate_before": float(np.mean(old[s_old] > 0)), "main_hit_rate_after": float(np.mean(new[s_new] > 0)),"draw_bets": int(mask.sum()), "draw_hit_rate": float((y[mask] == 1).mean()) if mask.any() else None,
            "draw_avg_odds": float(o[mask].mean()) if mask.any() else None,
            "draw_mean_ev": float((fr["pD"].to_numpy()[mask] * o[mask] - 1).mean()) if mask.any() else None,
            "draw_roi": C.roi_ci(draw_prof[mask]),
            "replaced_main_roi": C.roi_ci(old[replaced]),
            "replaced_main_hit_rate": float((old[replaced] > 0).mean()) if replaced.any() else None,
            "main_roi_before": float(roi_old), "main_roi_after": float(roi_new), "main_bets_before": int(s_old.sum()),
            "main_bets_after": int(s_new.sum()), "delta_main_roi": float(roi_new - roi_old),
            "delta_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def draw_bets_table(fr) -> dict:
    """ROI of a flat draw bet on every priced match, by EV bin and odds bin."""
    y, o, pdraw = fr["y"].to_numpy(), fr["draw_odds"].to_numpy(float), fr["pD"].to_numpy()
    ok = np.isfinite(o) & (o > 1)
    ev = pdraw * o - 1
    prof = np.where(y == 1, o - 1, -1.0)
    out = {"all": C.roi_ci(prof[ok])}
    out["by_ev"] = {}
    for lo, hi in zip(EV_BINS[:-1], EV_BINS[1:]):
        k = ok & (ev >= lo) & (ev < hi)
        out["by_ev"][f"[{lo:+.2f}, {hi:+.2f})"] = {**C.roi_ci(prof[k]), "hit_rate": float((y[k] == 1).mean()) if k.any() else None,
                                                    "mean_p_draw": float(pdraw[k].mean()) if k.any() else None,
                                                    "avg_odds": float(o[k].mean()) if k.any() else None}
    out["by_odds"] = {}
    for lo, hi in zip(ODDS_BINS[:-1], ODDS_BINS[1:]):
        k = ok & (o >= lo) & (o < hi)
        out["by_odds"][f"[{lo:.1f}, {hi:.1f})"] = {**C.roi_ci(prof[k]), "hit_rate": float((y[k] == 1).mean()) if k.any() else None,
                                                    "mean_p_draw": float(pdraw[k].mean()) if k.any() else None,
                                                    "implied": float((1 / o[k]).mean()) if k.any() else None}
    return out


def run() -> dict:
    sets = {}
    for b in C.FOLDS:
        sets[f"block_{b}"] = block_frame(b)
    sets["history_2026"] = (history_frame(), None)
    res = {"why_not_main": {}, "draw_bets": {}, "policies": {}, "risk_draws": {}}
    for name, (fr, sels) in sets.items():
        res["why_not_main"][name] = why_not_main(fr, sels)
        res["draw_bets"][name] = draw_bets_table(fr)
        rd = (fr["risk_market"] == "1X2") & (fr["risk_selection"] == "D")
        rp = fr["risk_profit"].to_numpy(float)
        res["risk_draws"][name] = {"share_of_matches": float(rd.mean()), "risk_draw_roi": C.roi_ci(rp[rd.to_numpy() & np.isfinite(rp)]),
                                   "risk_other_roi": C.roi_ci(rp[~rd.to_numpy() & np.isfinite(rp)])}
        log.info("Q2 %s: %s", name, {k: v for k, v in res["why_not_main"][name].items() if not isinstance(v, dict)})
    # policies
    grid = list(itertools.product(POLICY_T, POLICY_G, POLICY_E))
    table = {}
    for t, g, e in grid:
        key = f"t{t:.2f}_g{g:.2f}_e{e:.2f}"
        table[key] = {"params": {"t": t, "g": g, "e": e}}
        for name, (fr, _) in sets.items():
            table[key][name] = override(fr, policy_mask(fr, t, g, e), n_boot=500)
        sel = [sets[f"block_{s}"][0] for s in C.SELECT]
        pooled = pd.concat(sel, ignore_index=True)
        table[key]["pooled_selection"] = override(pooled, policy_mask(pooled, t, g, e), n_boot=500)
    ok = {k: v for k, v in table.items() if all(v[f"block_{s}"]["draw_bets"] >= MIN_DRAW_BETS for s in C.SELECT)}
    best = max(ok, key=lambda k: ok[k]["pooled_selection"]["delta_main_roi"]) if ok else None
    if best:                                                     # precise CIs for the chosen policy
        prm = table[best]["params"]
        for name, (fr, _) in sets.items():
            table[best][name] = override(fr, policy_mask(fr, prm["t"], prm["g"], prm["e"]), n_boot=4000)
        pooled = pd.concat([sets[f"block_{s}"][0] for s in C.SELECT], ignore_index=True)
        table[best]["pooled_selection"] = override(pooled, policy_mask(pooled, prm["t"], prm["g"], prm["e"]), n_boot=4000)
    b = table[best] if best else None
    convincing = bool(b and all(b[f"block_{s}"]["delta_main_roi"] > 0 for s in C.SELECT)
                      and b["pooled_selection"]["delta_ci95"][0] > 0
                      and b[f"block_{C.CHECK}"]["delta_main_roi"] > 0 and b["history_2026"]["delta_main_roi"] > 0)
    res["policies"] = {"grid": table, "best": best, "convincing": convincing, "min_draw_bets_per_selection_fold": MIN_DRAW_BETS}
    # draw bets by league (pooled walk-forward 2023–2025 and 2026 history), current-rule named draws
    by_league = {}
    for tag, frames in (("blocks_2023_2025", [sets[f"block_{s}"][0] for s in (2023, 2024, 2025)]),
                        ("history_2026", [sets["history_2026"][0]])):
        fr = pd.concat(frames, ignore_index=True)
        rows = {}
        for lg, g in fr.groupby("league"):
            m = policy_mask(g, config.DRAW_THRESHOLD, config.DRAW_MAX_GAP, -np.inf)
            mev = policy_mask(g, config.DRAW_THRESHOLD, config.DRAW_MAX_GAP, 0.0)
            o, y = g["draw_odds"].to_numpy(float), g["y"].to_numpy()
            prof = np.where(y == 1, o - 1, -1.0)
            mp = g["main_profit"].to_numpy(float)
            rows[lg] = {"n": int(len(g)), "named_draw_bets": C.roi_ci(prof[m]), "named_draw_ev_pos_bets": C.roi_ci(prof[mev]),
                        "main_roi": C.roi_ci(mp[np.isfinite(mp)])}
        by_league[tag] = rows
    res["by_league"] = by_league
    res["main_reconstruction_check_2025"] = B.check_against_backtest()
    C.save_json("draw_main.json", res)
    rows = []
    for key, v in table.items():
        for name in [*sets, "pooled_selection"]:
            r = v[name]
            rows.append({"policy": key, **v["params"], "set": name, "draw_bets": r["draw_bets"], "draw_hit_rate": r["draw_hit_rate"],
                         "draw_avg_odds": r["draw_avg_odds"], "draw_roi": r["draw_roi"]["roi"],
                         "draw_roi_lo": r["draw_roi"]["ci95"][0], "draw_roi_hi": r["draw_roi"]["ci95"][1],
                         "main_roi_before": r["main_roi_before"], "main_roi_after": r["main_roi_after"],
                         "delta_main_roi": r["delta_main_roi"], "delta_lo": r["delta_ci95"][0], "delta_hi": r["delta_ci95"][1]})
    pd.DataFrame(rows).to_csv(C.RESULTS / "draw_main_policies.csv", index=False)
    return res


if __name__ == "__main__":
    run()
