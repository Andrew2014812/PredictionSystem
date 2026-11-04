"""Strategies A–G on identical matches (walk-forward folds), plus the Main-vs-betting-decision split.

Universe ("bettable"): matches with a real Bet365 opening price for 1X2 and total 2.5; candidates are the
selections with a real price (1X2, total 2.5, the one Asian line) — every strategy picks one Main per match from
the same candidates and is settled at the same price. "production" mode keeps unpriced candidates (as the site
does; ROI only over picks that happen to have a price) and is reported as a secondary view.

Every fitted component for season s uses season s-1 predicted by the models trained on < s-1 (the inner season).

Families
    A  current Main (production parameters)
    B  calibrated Main: per-key isotonic (B1), Platt (B2), selection-conditional isotonic on A's inner picks (B3)
    C  value-aware Main: stronger EV weight, value first, maximum EV (raw and calibrated probabilities)
    D  market-specific rules: drop totals, only 1X2, drop groups whose inner-season Main picks were over-confident
    E  confidence-adjusted: lower confidence bound (E1), penalty for the number of candidates (E2)
    F  market consensus: model blended with the de-margined Bet365 probability (Shin / proportional, inner weight),
       fixed 50/50, market only; Pinnacle-based blend (folds only)
    G  selective betting on top of the shown Main of A: EV thresholds with raw / calibrated / blended / meta-model
       probabilities, agreement with the market, odds window
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import transforms as T

FAMILIES = {"A": "Текущий Main", "B": "Калиброванный Main", "C": "Main с учётом ценности (EV)",
            "D": "Правила по рынкам", "E": "Поправка на неопределённость", "F": "Консенсус с рынком",
            "G": "Выборочные ставки (Main не меняется)"}


@dataclass
class Result:
    name: str
    family: str
    picks: pd.DataFrame            # chosen rows with outcome (all matches of the universe)
    bet: np.ndarray                # bool per pick: stake (True) or skip
    p_display: np.ndarray          # the probability the strategy states for its pick
    note: str = ""


def universe(df: pd.DataFrame, price: str = "odds") -> pd.DataFrame:
    """Matches with a real price for 1X2 (all three) and total 2.5 (both); mi re-indexed 0..n-1."""
    has = df[price].notna()
    k = df.assign(_h=has).loc[df["market"].isin(["1X2", "TOTAL_2.5"])].groupby("match_id")["_h"].agg(["sum", "size"])
    ok = set(k.index[(k["sum"] == 5) & (k["size"] == 5)])
    out = df.loc[df["match_id"].isin(ok)].copy()
    out["mi"] = pd.factorize(out["match_id"])[0]
    return out.reset_index(drop=True)


def add_ncand(df: pd.DataFrame, priced_only: bool) -> pd.DataFrame:
    p = df["p"].to_numpy(float)
    ok = (~df["group"].isin(ST.EXCLUDED)).to_numpy() & (p >= 0.40) & (p <= 0.85)
    if priced_only:
        ok &= df["odds"].notna().to_numpy()
    df["ncand"] = pd.Series(ok.astype(int)).groupby(df["mi"].to_numpy()).transform("sum").to_numpy()
    return df


# ------------------------------------------------------------------------------------------------ fitting on inner
class Fitted:
    """Everything a fold needs, fitted on the inner season only."""

    def __init__(self, inner: pd.DataFrame, priced_only: bool, book: str = "prod"):
        t0 = time.time()
        self.priced_only = priced_only
        self.params = ST.CURRENT.with_(priced_only=priced_only)
        inner = add_ncand(inner, priced_only)
        self.iso = T.Calibrator("isotonic").fit(inner)
        self.platt = T.Calibrator("platt").fit(inner)
        a_idx = ST.choose(inner, self.params)
        mask = np.zeros(len(inner), bool)
        mask[a_idx] = True
        self.sel_iso = T.Calibrator("isotonic").fit(inner, mask=mask)
        self.blend_shin = T.Blend(book, "shin").fit(inner)
        self.blend_prop = T.Blend(book, "prop").fit(inner)
        self.blend_ps = T.Blend("ps", "shin").fit(inner) if "q_ps_shin" in inner and inner["q_ps_shin"].notna().mean() > 0.01 else None
        # selection gap of A's inner picks per group (D4) and the candidate-count penalty (E2)
        a = ST.outcome(inner, a_idx)
        dec = a.loc[a["decided"]]
        self.inner_gap = dec.groupby("group").apply(lambda g: g["p"].mean() - g["won"].mean(),
                                                    include_groups=False).to_dict()
        self.inner_n = dec.groupby("group").size().to_dict()
        self.drop_groups = sorted(g for g, v in self.inner_gap.items() if v > 0.02 and self.inner_n.get(g, 0) >= 100)
        self.lam = self._fit_lambda(inner)
        self.se_table = self._se_table(inner)
        # meta-model on A's inner picks with a real price
        a_rows = inner.iloc[a_idx]
        priced = a_rows["odds"].notna().to_numpy() & np.isin(a_rows["status"].to_numpy(), [0, 1])
        q = T.market_q(a_rows, book, "shin")
        X = T.meta_frame(a_rows, "p", q, a_rows["ncand"].to_numpy())
        self.meta_cols = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]          # without the candidate count (not transferable)
        self.meta = T.Meta().fit(X[priced][:, self.meta_cols], (a_rows["status"].to_numpy()[priced] == 1).astype(float))
        self.meta_groups = sorted(set(a_rows["group"].to_numpy()[priced]))
        self.book = book
        self.seconds = time.time() - t0

    def _fit_lambda(self, inner: pd.DataFrame) -> float:
        """lambda of p_adj = p - lambda * log(n_cand) that brings A-style picks' mean gap closest to 0."""
        best, best_gap = 0.0, None
        for lam in np.round(np.arange(0, 0.0801, 0.005), 3):
            d = inner.assign(p_adj=np.clip(inner["p"] - lam * np.log(np.maximum(inner["ncand"], 1)), 0.001, 0.999))
            idx = ST.choose(d, self.params, p_col="p_adj")
            o = ST.outcome(d, idx, p_col="p_adj")
            dec = o.loc[o["decided"]]
            gap = abs(dec["p"].mean() - dec["won"].mean())
            if best_gap is None or gap < best_gap:
                best, best_gap = float(lam), gap
        return best

    def _se_table(self, inner: pd.DataFrame) -> dict:
        """Standard error of the observed hit rate per calibration key and 0.05 probability bin (inner)."""
        dec = inner.loc[np.isin(inner["status"], [0, 1])]
        key = T.cal_key(dec)
        b = np.floor(dec["p"].to_numpy(float) / 0.05).astype(int)
        y = (dec["status"].to_numpy() == 1).astype(float)
        frame = pd.DataFrame({"k": key, "b": b, "y": y}).groupby(["k", "b"])["y"].agg(["mean", "size"])
        se = np.sqrt(np.clip(frame["mean"] * (1 - frame["mean"]), 0.01, None) / frame["size"])
        return {f"{k}|{bb}": float(v) for (k, bb), v in se.items()}

    def se(self, df: pd.DataFrame) -> np.ndarray:
        key = T.cal_key(df)
        b = np.floor(df["p"].to_numpy(float) / 0.05).astype(int)
        return np.array([self.se_table.get(f"{k}|{bb}", 0.05) for k, bb in zip(key, b)])

    def summary(self) -> dict:
        return {"blend_weight_shin": self.blend_shin.w, "blend_weight_prop": self.blend_prop.w,
                "blend_weight_pinnacle": self.blend_ps.w if self.blend_ps else None,
                "inner_gap_A": self.inner_gap, "inner_n_A": self.inner_n, "D4_dropped_groups": self.drop_groups,
                "E2_lambda": self.lam, "meta_coef": self.meta.lr.coef_.tolist(), "meta_groups": self.meta_groups,
                "fit_seconds": self.seconds}


# ------------------------------------------------------------------------------------------------ strategies
def _main(df, params, p_col="p", ev_values=None, name="", family="", note="", bet=None):
    idx = ST.choose(df, params, p_col=p_col, ev_values=ev_values)
    o = ST.outcome(df, idx, p_col=p_col)
    b = np.ones(len(o), bool) if bet is None else bet(o, idx)
    return Result(name, family, o, b, o["p"].to_numpy(float), note)


def run_all(outer: pd.DataFrame, fit: Fitted, include_pinnacle: bool = True) -> dict[str, Result]:
    P = fit.params
    df = add_ncand(outer.copy(), fit.priced_only)
    df["p_iso"] = np.clip(fit.iso.transform(df), 0.001, 0.999)
    df["p_platt"] = np.clip(fit.platt.transform(df), 0.001, 0.999)
    df["p_seliso"] = np.clip(fit.sel_iso.transform(df), 0.001, 0.999)
    df["p_blend"] = np.clip(fit.blend_shin.transform(df), 0.001, 0.999)
    df["p_blend_prop"] = np.clip(fit.blend_prop.transform(df), 0.001, 0.999)
    df["p_half"] = np.clip(T.Blend(fit.book, "shin", fixed=0.5).transform(df), 0.001, 0.999)
    df["p_mkt"] = np.clip(T.Blend(fit.book, "shin", fixed=0.0).transform(df), 0.001, 0.999)
    df["p_lcb"] = np.clip(df["p_iso"] - fit.se(df), 0.001, 0.999)
    df["p_ncand"] = np.clip(df["p"] - fit.lam * np.log(np.maximum(df["ncand"], 1)), 0.001, 0.999)
    if include_pinnacle and fit.blend_ps is not None and "q_ps_shin" in df:
        df["p_blend_ps"] = np.clip(fit.blend_ps.transform(df), 0.001, 0.999)
    R: dict[str, Result] = {}

    def add(r: Result):
        R[r.name] = r

    add(_main(df, P, name="A", family="A", note="production parameters"))
    add(_main(df, P, "p_iso", name="B1 isotonic", family="B"))
    add(_main(df, P, "p_platt", name="B2 Platt", family="B"))
    add(_main(df, P, "p_seliso", name="B3 selection-conditional", family="B"))
    for w in (1.0, 2.0, 4.0):
        add(_main(df, P.with_(ev_weight=w), name=f"C1 EV weight {w:g}", family="C"))
    for t in (0.0, 0.03):
        add(_main(df, P.with_(value_first_min_ev=t), name=f"C2 value first EV>={t:g}", family="C"))
    R["C3 max EV"] = _max_ev(df, "p", P, "C3 max EV")
    add(_main(df, P.with_(ev_weight=2.0), "p_iso", name="C4 calibrated + EV weight 2", family="C"))
    R["C5 calibrated max EV"] = _max_ev(df, "p_iso", P, "C5 calibrated max EV")
    add(_main(df, P.with_(reliability=_rel(P, TOTAL=0.0)), name="D1 no totals", family="D"))
    add(_main(df, P.with_(reliability=_rel(P, TOTAL=0.0, HANDICAP=0.0, DOUBLE_CHANCE=0.0, BTTS=0.0, CORNERS=0.0)),
              name="D2 only 1X2", family="D"))
    drop = {g: 0.0 for g in fit.drop_groups}
    add(_main(df, P.with_(reliability=_rel(P, **drop)) if drop else P, name="D3 drop over-confident groups",
              family="D", note=f"dropped on inner: {fit.drop_groups}"))
    add(_main(df, P.with_(reliability=_rel(P, TOTAL=0.9)), name="D4 totals reliability 0.9", family="D"))
    add(_main(df, P, "p_lcb", name="E1 lower confidence bound", family="E"))
    add(_main(df, P, "p_ncand", name="E2 candidate-count penalty", family="E", note=f"lambda={fit.lam}"))
    add(_main(df, P, "p_blend", name="F1 blend Shin (inner weight)", family="F"))
    add(_main(df, P, "p_blend_prop", name="F2 blend proportional (inner weight)", family="F"))
    add(_main(df, P, "p_half", name="F3 blend 50/50", family="F"))
    add(_main(df, P, "p_mkt", name="F4 market only", family="F"))
    if "p_blend_ps" in df:
        add(_main(df, P, "p_blend_ps", name="F5 blend Pinnacle", family="F"))
    # G: shown Main = A; stake only when the condition holds
    a = R["A"]
    rows = df.iloc[a.picks["row"].to_numpy()]
    odds = rows["odds"].to_numpy(float)
    push = rows["p_push"].to_numpy(float)

    def ev_of(col):
        return rows[col].to_numpy(float) * odds - (1 - push)

    q = T.market_q(rows, fit.book, "shin")
    X = T.meta_frame(rows, "p", q, rows["ncand"].to_numpy())[:, fit.meta_cols]
    p_meta = fit.meta.predict(X)
    known = rows["group"].isin(fit.meta_groups).to_numpy()
    conds = {
        "G1 EV>0 (raw)": ev_of("p") > 0,
        "G2 EV>5% (raw)": ev_of("p") > 0.05,
        "G3 EV>0 (calibrated)": ev_of("p_iso") > 0,
        "G4 EV>3% (calibrated)": ev_of("p_iso") > 0.03,
        "G5 EV>0 (blend)": ev_of("p_blend") > 0,
        "G6 EV>2% (blend)": ev_of("p_blend") > 0.02,
        "G7 EV>0 (meta-model)": known & (p_meta * odds - (1 - push) > 0),
        "G8 EV>3% (meta-model)": known & (p_meta * odds - (1 - push) > 0.03),
        "G9 agrees with market (|p-q|<3pp)": np.isfinite(q) & (np.abs(rows["p"].to_numpy(float) - q) < 0.03),
        "G10 odds 1.60–2.50": (odds >= 1.60) & (odds <= 2.50),
    }
    for name, m in conds.items():
        R[name] = Result(name, "G", a.picks, np.asarray(m, bool) & np.isfinite(odds), a.p_display)
    f1 = R["F1 blend Shin (inner weight)"]
    rows_f = df.iloc[f1.picks["row"].to_numpy()]
    ev_f = rows_f["p_blend"].to_numpy(float) * rows_f["odds"].to_numpy(float) - (1 - rows_f["p_push"].to_numpy(float))
    R["G11 F1 Main, EV>0 (blend)"] = Result("G11 F1 Main, EV>0 (blend)", "G", f1.picks, ev_f > 0, f1.p_display)
    return R


def _rel(P: ST.Params, **kw) -> tuple:
    r = P.rel()
    r.update(kw)
    return tuple(sorted(r.items()))


def _max_ev(df: pd.DataFrame, p_col: str, P: ST.Params, name: str) -> Result:
    """The priced candidate in the probability band with the highest EV (ties: higher probability)."""
    p = df[p_col].to_numpy(float)
    e = ST.ev(p, df["odds"], df["p_push"])
    ok = (~df["group"].isin(ST.EXCLUDED).to_numpy()) & (p >= P.min_p) & (p <= P.max_p) & np.isfinite(e)
    score = np.where(ok, e, -np.inf)
    idx = ST.pick(df, score, p_col, "odds", P.priced_only)
    o = ST.outcome(df, idx, p_col=p_col)
    return Result(name, "C", o, np.ones(len(o), bool), o["p"].to_numpy(float))


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(R: dict[str, Result], n_matches: int, price_cols: tuple = ()) -> dict:
    out = {}
    base = R["A"]
    base_pm = ST.per_match_profit(base.picks.assign(profit=np.where(base.bet, base.picks["profit"], np.nan)), n_matches)
    for name, r in R.items():
        o = r.picks.assign(profit=np.where(r.bet, r.picks["profit"], np.nan),
                           priced=r.picks["priced"] & r.bet)
        s = ST.summary(o, n_matches)
        dec = r.picks.loc[r.picks["decided"]]
        pd_ = (r.p_display / np.clip(1 - r.picks["p_push"].to_numpy(float), 1e-6, None))[r.picks["decided"].to_numpy()]
        s["calibration_main"] = C.binary_scores(dec["won"], pd_)
        s["share_groups"] = r.picks["group"].value_counts(normalize=True).round(4).to_dict()
        s["bets_share_of_matches"] = s["bets"] / n_matches
        pm = ST.per_match_profit(o, n_matches)
        s["vs_A"] = C.paired_roi_diff(base_pm, pm) if name != "A" else None
        s["family"], s["note"] = r.family, r.note
        out[name] = s
    return out


def fold_run(season: int, mode: str = "bettable") -> tuple[dict, dict, dict, dict]:
    priced_only = mode == "bettable"
    inner = CD.historical(season - 1)
    outer = CD.historical(season)
    if priced_only:
        inner, outer = universe(inner), universe(outer)
    fit = Fitted(inner, priced_only)
    R = run_all(outer, fit)
    n = int(outer["mi"].max() + 1)
    ev = evaluate(R, n)
    pm = {name: ST.per_match_profit(r.picks.assign(profit=np.where(r.bet, r.picks["profit"], np.nan)), n)
          for name, r in R.items()}
    picks = {name: r.picks.assign(bet=r.bet, p_display=r.p_display) for name, r in R.items()
             if name in ("A", "B1 isotonic", "F1 blend Shin (inner weight)", "C3 max EV", "E2 candidate-count penalty",
                         "G7 EV>0 (meta-model)", "B3 selection-conditional", "C2 value first EV>=0", "D2 only 1X2")}
    return ev, fit.summary(), picks, pm


def select(results: dict) -> dict:
    """Variant choice on 2023/24 + 2024/25 only: highest worst-fold ROI, >= 300 bets on each fold."""
    names = list(results[str(C.SELECT[0])].keys())
    table = {}
    for n in names:
        f = [results[str(s)][n] for s in C.SELECT]
        roi = [x["roi"] if x["roi"] is not None and x["roi"] == x["roi"] else -np.inf for x in f]
        bets = [x["bets"] for x in f]
        cal_ok = all(
            (x["calibration_main"]["ece"] <= results[str(s)]["A"]["calibration_main"]["ece"] + 0.01)
            and (x["calibration_main"]["brier"] <= results[str(s)]["A"]["calibration_main"]["brier"] + 0.002)
            for s, x in zip(C.SELECT, f))
        better_both = all(r > results[str(s)]["A"]["roi"] for s, r in zip(C.SELECT, roi)) if n != "A" else False
        table[n] = {"family": f[0]["family"], "roi": roi, "bets": bets, "worst_roi": min(roi),
                    "pooled_roi": float(np.average(roi, weights=bets)) if sum(bets) and min(roi) > -np.inf else -np.inf,
                    "eligible": min(bets) >= 300,
                    "better_than_A_both": better_both, "calibration_not_worse": cal_ok}
    fam_best = {}
    for fam in FAMILIES:
        cands = [n for n, t in table.items() if t["family"] == fam and t["eligible"]]
        if cands:
            fam_best[fam] = max(cands, key=lambda n: (table[n]["worst_roi"], table[n]["pooled_roi"]))
    return {"table": table, "family_best": fam_best}


def run() -> dict:
    t0 = time.time()
    out = {"modes": {}, "fitted": {}}
    picks_store, pms = {}, {}
    for mode in ("bettable", "production"):
        res, fits = {}, {}
        for s in C.FOLDS:
            ev, fit, picks, pm = fold_run(s, mode)
            res[str(s)], fits[str(s)] = ev, fit
            if mode == "bettable" and s in C.SELECT:
                for name, v in pm.items():
                    pms.setdefault(name, []).append(v)
            for name, p in picks.items():
                picks_store[(mode, s, name)] = p
            C.log.info("%s %s done (%.1f min)", mode, s, (time.time() - t0) / 60)
        out["modes"][mode] = {"folds": res, "selection": select(res)}
        out["fitted"][mode] = fits
    # pooled paired comparison (2023/24 + 2024/25) of every variant vs A, bettable universe
    out["pooled_vs_A"] = pooled_vs_a(pms)
    out["minutes"] = (time.time() - t0) / 60
    C.save_json("strategies.json", out)
    frames = []
    for (mode, s, name), p in picks_store.items():
        frames.append(p.assign(mode=mode, season=s, strategy=name))
    pd.concat(frames, ignore_index=True).to_parquet(C.CACHE / "strategy_picks.parquet")
    return out


def pooled_vs_a(pm: dict) -> dict:
    """Paired bootstrap of ROI(variant) - ROI(A) over the pooled 2023/24 + 2024/25 bettable matches;
    p05 = 5th percentile of the bootstrap difference (one-sided 95 % lower bound)."""
    base = np.concatenate(pm["A"])
    out = {}
    for name, parts in pm.items():
        if name == "A":
            continue
        b = np.concatenate(parts)
        d = C.paired_roi_diff(base, b)
        rng = np.random.default_rng(C.SEED)
        n, diffs = len(base), []
        for _ in range(C.N_BOOT // 250):
            idx = rng.integers(0, n, size=(250, n))
            with np.errstate(all="ignore"):
                diffs.append(np.nanmean(b[idx], 1) - np.nanmean(base[idx], 1))
        d["p05"] = float(np.nanquantile(np.concatenate(diffs), 0.05))
        out[name] = d
    return out


if __name__ == "__main__":
    C.setup_logging("strategies")
    r = run()
    for mode in r["modes"]:
        print(mode, r["modes"][mode]["selection"]["family_best"])
