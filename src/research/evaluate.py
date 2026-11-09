"""Paired comparison of every shadow with production on the same forward matches, with the preregistered
"no conclusion before N" rule.

For every (shadow, match, item) the record that counts is the LAST one stored before kick-off (records are
only made before kick-off; the filter is repeated here). A match counts as forward when it satisfies the
shadow's ``forward_rule`` (kick-off / date after the freeze of its preregistration).

Interim-safe reporting: while fewer than ``min_n`` forward cases are settled the report gives the count, the
information fraction and point estimates only — no interval, no p-value, no decision. At the first evaluation
with n >= min_n (or at the end of the preregistered window) the test is run once and its result is frozen in
the manifest; later reports repeat the frozen decision and never re-test on more data (no optional stopping).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..markets.markets import settle_selection
from . import registry as R
from .variants import pool_predict, power_demargin

SEED = 20261109
N_BOOT = 2000
SEASON_END = pd.Timestamp("2027-06-30")


# ------------------------------------------------------------------------------------------------ statistics
def _boot(n: int, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(250, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def _ci(stat_fn, n: int) -> np.ndarray:
    return np.concatenate([stat_fn(idx) for idx in _boot(n)])


def _p_two(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(pvalues: dict) -> dict:
    keys = sorted([k for k, v in pvalues.items() if v is not None and np.isfinite(v)], key=lambda k: pvalues[k])
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * pvalues[k]))
        out[k] = running
    return out


def gap_stats(p: np.ndarray, y: np.ndarray) -> dict:
    d = p - y
    b = _ci(lambda idx: d[idx].mean(1), len(d))
    return {"gap": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
            "mean_p": float(p.mean()), "hit_rate": float(y.mean())}


def abs_gap_diff(p_new: np.ndarray, p_old: np.ndarray, y: np.ndarray) -> dict:
    """|gap(p_new)| - |gap(p_old)|, paired bootstrap by match."""
    a, b = p_new - y, p_old - y
    d = _ci(lambda idx: np.abs(a[idx].mean(1)) - np.abs(b[idx].mean(1)), len(y))
    return {"diff": float(abs(a.mean()) - abs(b.mean())),
            "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))], "p": _p_two(d)}


def paired_mean_diff(x_new: np.ndarray, x_old: np.ndarray, one_sided_less: bool = False) -> dict:
    """mean(x_new - x_old) with its paired bootstrap interval (NaN-aware for ROI per match)."""
    a, b = np.asarray(x_new, float), np.asarray(x_old, float)
    d = _ci(lambda idx: np.nanmean(a[idx], 1) - np.nanmean(b[idx], 1), len(a))
    out = {"diff": float(np.nanmean(a) - np.nanmean(b)),
           "ci95": [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))], "p": _p_two(d)}
    if one_sided_less:
        out["p_one_sided"] = float(np.mean(d >= 0))
    return out


# ------------------------------------------------------------------------------------------------ data
def forward_mask(df: pd.DataFrame, spec: R.ShadowSpec) -> np.ndarray:
    rule = spec.forward_rule
    if rule["kind"] == "date_from":
        return (pd.to_datetime(df["date"]) >= pd.Timestamp(rule["date"])).to_numpy()
    if rule["kind"] == "kickoff_after_utc":
        from .shadow import utc_to_uk
        return (pd.to_datetime(df["kickoff"]) > utc_to_uk(rule["utc"])).to_numpy()
    raise ValueError(rule)


def final_records(log: pd.DataFrame, spec: R.ShadowSpec, item: str) -> pd.DataFrame:
    d = log.loc[(log["shadow_key"] == spec.key) & (log["item"] == item)]
    d = d.loc[pd.to_datetime(d["recorded_at"]) < pd.to_datetime(d["kickoff"])]
    d = d.sort_values("seq").drop_duplicates("match_id", keep="last")
    return d.loc[forward_mask(d, spec)].reset_index(drop=True)


def with_results(d: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if d.empty or results.empty:
        return d.iloc[:0].assign(home_goals=np.nan, away_goals=np.nan, total_corners=np.nan)
    r = results.drop_duplicates("match_id", keep="last").drop(columns=["observed_at", "result_hash"],
                                                              errors="ignore")
    return d.merge(r, on="match_id", how="inner")


def _status(df: pd.DataFrame, market_col="market", sel_col="selection") -> np.ndarray:
    return np.array([settle_selection(m, s, h, a, c) if isinstance(m, str) and m else None
                     for m, s, h, a, c in zip(df[market_col], df[sel_col], df["home_goals"], df["away_goals"],
                                              df["total_corners"])], dtype=object)


def _payload(df: pd.DataFrame) -> list[dict]:
    return [json.loads(p) if isinstance(p, str) and p else {} for p in df["payload"]]


def _profit(status: np.ndarray, odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, float)
    priced = np.isfinite(odds) & (odds > 1.0)
    out = np.where(status == "WON", odds - 1.0, np.where(status == "LOST", -1.0, np.nan))
    return np.where(priced, out, np.nan)


# ------------------------------------------------------------------------------------------------ per shadow
def _status_block(n: int, spec: R.ShadowSpec, now: pd.Timestamp, unit: str) -> dict:
    window_over = now is not None and now > SEASON_END and spec.family in ("main_calibration", "risk")
    ready = n >= spec.min_n or (window_over and n > 0)
    return {"n": int(n), "min_n": spec.min_n, "unit": unit, "information_fraction": round(n / spec.min_n, 4),
            "status": ("ready" if n >= spec.min_n else "end_of_window") if ready else
            ("no_forward_data" if n == 0 else "accumulating"), "ready": bool(ready)}


def eval_main_calibration(spec, log, results, now) -> dict:
    d = with_results(final_records(log, spec, "main"), results)
    if not d.empty:
        d = d.assign(status=_status(d)).loc[lambda x: x["status"].isin(["WON", "LOST"]) & x["p"].notna()]
        d = d.sort_values(["date", "kickoff", "match_id"]).head(spec.min_n)      # the window: first min_n
    out = _status_block(len(d), spec, now, "рассчитанных показанных Main")
    if d.empty:
        return out
    y = (d["status"] == "WON").to_numpy(float)
    p = d["p"].to_numpy(float)
    pc = np.array([pl.get("pc", np.nan) for pl in _payload(d)], float)
    out["point"] = {"gap_shadow": float((p - y).mean()), "gap_production": float((pc - y).mean())}
    if out["ready"]:
        g = gap_stats(p, y)
        c2 = abs_gap_diff(p, pc, y)
        out["test"] = {"gap_shadow": g, "gap_production": gap_stats(pc, y), "abs_gap_diff": c2,
                       "C1": bool(g["ci95"][0] <= 0 <= g["ci95"][1]), "C2": bool(c2["ci95"][1] < 0)}
        out["p_for_holm"] = c2["p"]
    return out


def eval_risk_stacked(spec, log, results, now) -> dict:
    d = with_results(final_records(log, spec, "risk"), results)
    if not d.empty:
        d = d.assign(st_new=_status(d), st_old=_status(d, "base_market", "base_selection"))
        d["pr_new"], d["pr_old"] = _profit(d["st_new"].to_numpy(), d["odds"]), _profit(d["st_old"].to_numpy(),
                                                                                         d["base_odds"])
        both = (np.isfinite(d["odds"]) & np.isfinite(d["base_odds"]) & d["st_new"].notna() & d["st_old"].notna())
        d = d.loc[both]
    out = _status_block(len(d), spec, now, "матчей с ценой у обоих Risk")
    if d.empty:
        return out
    out["point"] = {"roi_shadow": float(np.nanmean(d["pr_new"])), "roi_production": float(np.nanmean(d["pr_old"])),
                    "changed_picks": float(((d["market"] != d["base_market"]) |
                                            (d["selection"] != d["base_selection"])).mean())}
    if out["ready"]:
        t = paired_mean_diff(d["pr_new"].to_numpy(), d["pr_old"].to_numpy())
        out["test"] = {"roi_diff": t, "passed": bool(t["ci95"][0] > 0)}
        out["p_for_holm"] = t["p"]
    return out


def eval_risk_market(spec, log, results, now) -> dict:
    d = with_results(final_records(log, spec, "risk"), results)
    if not d.empty:
        d = d.assign(status=_status(d)).loc[lambda x: x["status"].isin(["WON", "LOST"]) & x["p"].notna()]
    out = _status_block(len(d), spec, now, "рассчитанных Risk с полной книгой")
    if d.empty:
        return out
    y = (d["status"] == "WON").to_numpy(float)
    q = d["p"].to_numpy(float)
    pc = np.array([pl.get("pc", np.nan) for pl in _payload(d)], float)
    out["point"] = {"gap_market": float((q - y).mean()), "gap_model": float((pc - y).mean())}
    if out["ready"]:
        g = gap_stats(q, y)
        c = abs_gap_diff(q, pc, y)
        out["test"] = {"gap_market": g, "gap_model": gap_stats(pc, y), "abs_gap_diff": c,
                       "passed": bool(g["ci95"][0] <= 0 <= g["ci95"][1] and c["ci95"][1] < 0)}
        out["p_for_holm"] = c["p"]
    return out


def _bookmaker_rows(spec, log, results, item: str, doc: dict) -> pd.DataFrame:
    d = with_results(final_records(log, spec, item), results)
    if d.empty:
        return d
    if item == "1x2":
        o = d[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
        y = np.where(d["home_goals"] > d["away_goals"], 0, np.where(d["home_goals"] == d["away_goals"], 1, 2))
        theta = doc["frozen_variant"]["theta_1x2"]
    else:
        o = d[["odds_over25", "odds_under25"]].to_numpy(float)
        y = np.where(d["home_goals"] + d["away_goals"] > 2.5, 0, 1)
        theta = doc["frozen_variant"]["theta_ou25"]
    ok = np.all(np.isfinite(o) & (o > 1.0), 1)
    d, o, y = d.loc[ok].reset_index(drop=True), o[ok], y[ok]
    if d.empty:
        return d
    p = np.array([pl["p_model"] for pl in _payload(d)], float)
    p = np.clip(p, 1e-6, 1)
    p = p / p.sum(1, keepdims=True)
    q = power_demargin(o)
    h2 = pool_predict(theta, q, p)
    idx = np.arange(len(d))
    d["ll_h2"] = -np.log(np.clip(h2[idx, y], 1e-15, 1))
    d["ll_h0"] = -np.log(np.clip(q[idx, y], 1e-15, 1))
    d["ll_model"] = -np.log(np.clip(p[idx, y], 1e-15, 1))
    return d


def eval_bookmaker(spec, log, results, now) -> dict:
    doc = spec.document()
    out = {"markets": {}}
    for item in ("ou25", "1x2"):
        d = _bookmaker_rows(spec, log, results, item, doc)
        m = _status_block(len(d), spec, now, "матчей с ценами Bet365 открытия")
        if not d.empty:
            m["point"] = {"ll_h2_minus_h0": float((d["ll_h2"] - d["ll_h0"]).mean()),
                          "ll_model_minus_h0": float((d["ll_model"] - d["ll_h0"]).mean())}
            if m["ready"]:
                t = paired_mean_diff(d["ll_h2"].to_numpy(), d["ll_h0"].to_numpy())
                m["test"] = {"ll_diff": t, "upper_below_0": bool(t["ci95"][1] < 0)}
        out["markets"][item] = m
    primary = out["markets"]["ou25"]
    out.update({k: primary[k] for k in ("n", "min_n", "unit", "information_fraction", "status", "ready")})
    if all(m.get("test") for m in out["markets"].values()):
        h = holm({k: m["test"]["ll_diff"]["p"] for k, m in out["markets"].items()})
        out["holm"] = h
        out["decision"] = ("B: предложить H2 как вход EV Main / Risk и отдельный форвард-тест правила ставок"
                           if out["markets"]["ou25"]["test"]["upper_below_0"] else "C: отказаться")
    return out


def eval_xg(spec, log, results, now) -> dict:
    g = with_results(final_records(log, spec, "goals"), results)
    out = _status_block(len(g), spec, now, "сыгранных матчей лиг с xG")
    if g.empty:
        return out
    pl = _payload(g)
    mu = np.array([x["mu"] for x in pl], float)
    lam = np.array([x["lam"] for x in pl], float)
    y = g[["home_goals", "away_goals"]].to_numpy(float)

    def dev(yy, m):
        m = np.clip(m, 1e-6, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2 * (np.where(yy > 0, yy * np.log(yy / m), 0.0) - (yy - m))
    dd = dev(y, mu).sum(1) - dev(y, lam).sum(1)
    res = np.where(y[:, 0] > y[:, 1], 0, np.where(y[:, 0] == y[:, 1], 1, 2))
    p5 = np.array([x["p_x5"] for x in pl], float)
    p0 = np.array([x["p_b0"] for x in pl], float)
    idx = np.arange(len(g))
    dll = -np.log(np.clip(p5[idx, res], 1e-15, 1)) + np.log(np.clip(p0[idx, res], 1e-15, 1))
    out["point"] = {"goals_dev_diff": float(dd.mean()), "logloss_1x2_diff": float(dll.mean()),
                    "fallback_share": float(np.mean([x.get("fallback", False) for x in pl]))}
    if out["ready"]:
        prim = paired_mean_diff(dd, np.zeros_like(dd), one_sided_less=True)
        ll = paired_mean_diff(dll, np.zeros_like(dll))
        m = with_results(final_records(log, spec, "main"), results)
        roi, gap = None, None
        if not m.empty:
            m = m.assign(st_new=_status(m), st_old=_status(m, "base_market", "base_selection"))
            pn, po = _profit(m["st_new"].to_numpy(), m["odds"]), _profit(m["st_old"].to_numpy(), m["base_odds"])
            roi = paired_mean_diff(pn, po)
            dec_n, dec_o = m["st_new"].isin(["WON", "LOST"]), m["st_old"].isin(["WON", "LOST"])
            gap = {"x5": float((m.loc[dec_n, "p"] - (m.loc[dec_n, "st_new"] == "WON")).mean()),
                   "b0": float((m.loc[dec_o, "base_p"] - (m.loc[dec_o, "st_old"] == "WON")).mean())}
        crit = {"primary": prim["p_one_sided"] < 0.05 and prim["diff"] < 0,
                "logloss": ll["ci95"][1] < 0.002,
                "roi": roi is not None and roi["ci95"][0] > -0.02,
                "overconfidence": gap is not None and gap["x5"] <= gap["b0"]}
        out["test"] = {"goals_dev_diff": prim, "logloss_1x2_diff": ll, "roi_main_diff": roi, "main_gap": gap,
                       "criteria": crit}
        out["p_for_holm"] = prim["p_one_sided"]
        out["decision"] = "A" if all(crit.values()) else ("B" if prim["diff"] < 0 else "C")
    return out


EVALUATORS = {"maincal_ps_platt_monthly": eval_main_calibration, "maincal_ps_group": eval_main_calibration,
              "risk_stacked": eval_risk_stacked, "risk_market_probability": eval_risk_market,
              "bookmaker_h2_pool": eval_bookmaker, "xg_x5": eval_xg}


def _family_decisions(shadows: dict) -> dict:
    out = {}
    mc = {k: v for k, v in shadows.items() if v.get("family") == "main_calibration" and v.get("test")}
    if len(mc) == 2:
        h = holm({k: v["p_for_holm"] for k, v in mc.items()})
        ok = {k: v["test"]["C1"] and v["test"]["C2"] and h.get(k, 1) < 0.05 for k, v in mc.items()}
        first = next((k for k in ("maincal_ps_platt_monthly@v1", "maincal_ps_group@v1") if ok.get(k)), None)
        out["main_calibration"] = {"holm": h, "passed": ok,
                                   "decision": f"A (только показ): {first}" if first else "C"}
    rk = {k: v for k, v in shadows.items() if v.get("family") == "risk" and v.get("test")}
    if len(rk) == 2:
        h = holm({k: v["p_for_holm"] for k, v in rk.items()})
        out["risk"] = {"holm": h, "passed": {k: v["test"]["passed"] and h.get(k, 1) < 0.05 for k, v in rk.items()}}
    return out


def evaluate_shadow(root: Path | str | None = None, now: pd.Timestamp | None = None, write: bool = True) -> dict:
    from .shadow import DEFAULT_DIR, REPORT, load_manifest, load_predictions, load_results, save_manifest, uk_now
    root = Path(root or DEFAULT_DIR)
    now = pd.Timestamp(now) if now is not None else uk_now()
    log = load_predictions(root)
    results = load_results(root)
    manifest = load_manifest(root)
    decisions = manifest.setdefault("decisions", {})
    report = {"evaluated_at_uk": str(now), "policy": "no conclusion before the preregistered N; one evaluation, "
                                                     "then frozen", "shadows": {}, "not_implemented": {}}
    for spec in R.SHADOWS:
        if not spec.implemented:
            report["not_implemented"][spec.key] = spec.not_implemented_reason
            continue
        if spec.key in decisions:                                     # frozen: never re-tested
            report["shadows"][spec.key] = {**decisions[spec.key], "frozen": True}
            continue
        if log.empty:
            res = _status_block(0, spec, now, "")
        else:
            res = EVALUATORS[spec.id](spec, log, results, now)
        res.update({"family": spec.family, "prereg": spec.prereg, "freeze_sha": spec.freeze_sha,
                    "criterion": spec.criterion})
        report["shadows"][spec.key] = res
    report["families"] = _family_decisions(report["shadows"])
    if write:
        for key, res in report["shadows"].items():
            if res.get("ready") and not res.get("frozen") and (res.get("test") or res.get("markets")):
                decisions[key] = {**res, "decided_at_uk": str(now)}
        if report["families"]:
            manifest.setdefault("family_decisions", {}).update(
                {k: {**v, "decided_at_uk": str(now)} for k, v in report["families"].items()
                 if k not in manifest.get("family_decisions", {})})
        save_manifest(root, manifest)
        root.mkdir(parents=True, exist_ok=True)
        (root / REPORT).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=_default),
                                   encoding="utf-8")
    return report


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
