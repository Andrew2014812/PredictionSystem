"""All Main variants on identical matches, season by season, with maps fitted on earlier seasons only.

Variant families (``FAMILY``):
    base   production v6.1 Main, raw model probability (the baseline)
    post   post-selection calibration (the pick is the baseline pick; only its stated probability changes)
    pre    pre-selection calibration of every candidate, then the production rule
    rank   other ranking functions on calibrated candidates (probability, EV, calibrated EV, Kelly, lower bound)
    few    fewer candidate markets (production rule, raw probability)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_1.main import strategies as ST

from . import common as C
from . import data as D
from . import methods as M

FAMILY_RU = {"base": "Базовый (продакшн v6.1)", "post": "Калибровка после выбора",
             "pre": "Калибровка всех кандидатов до выбора", "rank": "Другая функция ранжирования",
             "few": "Меньше рынков-кандидатов"}
FEW = {"F1 без форы и угловых": {"1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS"},
       "F2 1X2 + ДШ + тотал": {"1X2", "DOUBLE_CHANCE", "TOTAL"},
       "F3 только 1X2 и ДШ": {"1X2", "DOUBLE_CHANCE"}}
VARIANT_RU = {
    "A базовый": "production v6.1",
    "PS-Platt": "Платт по выбранным Main", "PS-Iso": "изотоническая по выбранным Main",
    "PS-Group": "Платт по выбранным, отдельно по группам рынков",
    "PS-Aware": "логистическая: p, число кандидатов, группа, цена",
    "PS-OS": "поправка порядковой статистики c·e(n)",
    "PS-Market": "Платт по выбранным + цена рынка (Шин)",
    "PS-EB-League": "Платт + смещение лиги (эмпирический Байес)",
    "Q1 изотоника по типам": "изотоническая по группа|исход до выбора",
    "Q2 Платт по рынкам": "Платт по рынок|исход (каждая линия) до выбора",
    "Q3 Q2 + Платт после выбора": "двухэтапная: Q2, затем Платт по выбранным",
    "Q4 Q2 + смесь с рынком": "Q2, затем смесь с ценой рынка (вес по истории)",
    "R1 макс. вероятность": "максимум калиброванной p в диапазоне",
    "R3 макс. EV": "максимум EV по сырой p",
    "R4 макс. калибр. EV": "максимум EV по калиброванной p",
    "R5 Келли": "максимум доли Келли (калибр.)",
    "R6 нижняя граница p": "максимум нижней границы калибр. p",
    "R7 нижняя граница EV": "максимум EV по нижней границе p",
}


def params(kind: str) -> ST.Params:
    return ST.CURRENT.with_(priced_only=kind == "bet")


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d["pc"] = D.p_cond(d)
    q = d["q_prod_shin"].to_numpy(float)
    d["qc"] = q / np.clip(1 - d["p_push"].to_numpy(float), 1e-6, None)
    st = d["status"].to_numpy()
    d["y"] = np.where(np.isin(st, [0, 1]), (st == 1).astype(float), np.nan)
    return d


def picks_frame(d: pd.DataFrame, idx: np.ndarray, pc_stated: np.ndarray | None = None) -> pd.DataFrame:
    r = d.iloc[idx]
    odds = r["odds"].to_numpy(float)
    push = r["p_push"].to_numpy(float)
    y = r["y"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1.0)
    pc = r["pc"].to_numpy(float) if pc_stated is None else np.asarray(pc_stated, float)
    profit = np.where(priced & np.isfinite(y), np.where(y == 1, odds - 1, -1.0), np.nan)
    st = r["status"].to_numpy()
    stake_profit = np.where(priced & np.isin(st, [0, 1, 2]), np.where(st == 1, odds - 1, np.where(st == 2, 0.0, -1.0)),
                            np.nan)
    return pd.DataFrame({"mi": r["mi"].to_numpy(), "date": r["date"].to_numpy(), "league": r["league"].to_numpy(),
                         "group": r["group"].to_numpy(), "market": r["market"].to_numpy(),
                         "selection": r["selection"].to_numpy(), "ncand": r["ncand"].to_numpy(),
                         "pc": pc, "pc_raw": r["pc"].to_numpy(float), "y": y, "odds": odds, "p_push": push,
                         "qc": r["qc"].to_numpy(float), "priced": priced, "profit": profit,
                         "stake_profit": stake_profit,
                         "ev": np.where(priced, pc * (1 - push) * odds - (1 - push), np.nan),
                         "ev_raw": np.where(priced, r["pc"].to_numpy(float) * (1 - push) * odds - (1 - push), np.nan)})


def decided_rows(d: pd.DataFrame) -> pd.DataFrame:
    m = np.isfinite(d["y"].to_numpy(float))
    return d.loc[m, ["group", "market", "selection", "pc", "y", "qc", "p_push"]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ history material
class History:
    """Universe tables of the history seasons (loaded lazily, one at a time) and their baseline picks."""

    def __init__(self, kind: str):
        self.kind = kind
        self._base: dict[int, pd.DataFrame] = {}
        self._rows: dict[int, pd.DataFrame] = {}

    def table(self, h: int) -> pd.DataFrame:
        return prepare(D.universe(D.load(h), self.kind))

    def material(self, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if h not in self._base:
            d = self.table(h)
            self._base[h] = picks_frame(d, ST.choose(d, params(self.kind))).assign(season=h)
            self._rows[h] = decided_rows(d)
        return self._base[h], self._rows[h]

    def base_picks(self, hs) -> pd.DataFrame:
        return pd.concat([self.material(h)[0] for h in hs], ignore_index=True)

    def rows(self, hs) -> pd.DataFrame:
        return pd.concat([self.material(h)[1] for h in hs], ignore_index=True)


# ------------------------------------------------------------------------------------------------ variants
def _decided(f):
    return f.loc[np.isfinite(f["y"].to_numpy(float))]


def blend_weights(rows: pd.DataFrame, pc_col: str = "pc_cal") -> dict:
    """Weight w per blend group of w pc + (1 - w) qc (0.05 grid, history log loss)."""
    groups = {"1X2": "1X2", "DOUBLE_CHANCE": "1X2", "TOTAL": "TOTAL", "BTTS": "TOTAL", "CORNERS": "TOTAL",
              "HANDICAP": "HANDICAP"}
    ok = np.isfinite(rows["qc"].to_numpy(float))
    r = rows.loc[ok]
    bg = r["group"].map(groups).to_numpy()
    out = {}
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    for g in np.unique(bg):
        m = bg == g
        if m.sum() < M.MIN_FIT:
            continue
        x, q, y = r[pc_col].to_numpy(float)[m], r["qc"].to_numpy(float)[m], r["y"].to_numpy(float)[m]
        ll = [np.mean(-(y * np.log(np.clip(w * x + (1 - w) * q, M.EPS, 1 - M.EPS))
                        + (1 - y) * np.log(np.clip(1 - w * x - (1 - w) * q, M.EPS, 1 - M.EPS)))) for w in grid]
        out[g] = float(grid[int(np.argmin(ll))])
    return {grp: out.get(bgp, 1.0) for grp, bgp in groups.items()}


MONTHLY = ("PS-Platt", "PS-Group", "PS-Market")


def _monthly(f: pd.DataFrame, name: str, history_picks: pd.DataFrame) -> np.ndarray:
    """Stated probability from a map fitted on the season's decided picks of earlier months only."""
    month = pd.to_datetime(f["date"]).dt.to_period("M").to_numpy()
    out = np.full(len(f), np.nan)
    dec = np.isfinite(f["y"].to_numpy(float))
    for mth in sorted(set(month)):
        cur = month == mth
        prev = (month < mth) & dec
        train = f.loc[prev] if prev.sum() >= M.MIN_FIT else history_picks
        out[cur] = M.POST[name]().fit(train).predict(f.loc[cur])
    return out


def season_variants(s: int, kind: str, hist: History) -> tuple[pd.DataFrame, dict[str, tuple[str, pd.DataFrame]], dict]:
    """All variants of one season / universe. Returns (table, {name: (family, picks)}, fitted parameters)."""
    t0 = time.time()
    P = params(kind)
    H = [h for h in C.HISTORY if h < s]
    d = prepare(D.universe(D.load(s), kind))
    out: dict[str, tuple[str, pd.DataFrame]] = {}
    info: dict = {"history": H}
    idx0 = ST.choose(d, P)
    base = picks_frame(d, idx0)
    out["A базовый"] = ("base", base)
    if s == C.CURRENT and kind == "site" and "stored_main" in d:
        sidx = np.flatnonzero(d["stored_main"].to_numpy())
        sidx = sidx[np.r_[True, d["mi"].to_numpy()[sidx][1:] != d["mi"].to_numpy()[sidx][:-1]]]
        info["stored_vs_recomputed_same"] = float(np.isin(sidx, idx0).mean())
        out["A как показано"] = ("base", picks_frame(d, sidx))

    # ---- post-selection maps on the baseline picks of the history seasons
    hb = _decided(hist.base_picks(H))
    info["history_base_picks"] = int(len(hb))
    for name, ctor in M.POST.items():
        m = ctor().fit(hb)
        f = base.copy()
        f["pc"] = m.predict(f)
        f["ev"] = np.where(f["priced"], f["pc"] * (1 - f["p_push"]) * f["odds"] - (1 - f["p_push"]), np.nan)
        out[name] = ("post", f)
        info[name] = m.params()
    if s == C.CURRENT:
        # within-season walk-forward: the map for a month is fitted on this season's picks of the EARLIER months
        # (the first month keeps the history map); applied to the recomputed and to the shown Main
        targets = [("", base)] + ([("показано + ", out["A как показано"][1])] if "A как показано" in out else [])
        for prefix, tgt in targets:
            for name in MONTHLY:
                f = tgt.copy()
                f["pc"] = _monthly(tgt, name, hb)
                f["ev"] = np.where(f["priced"], f["pc"] * (1 - f["p_push"]) * f["odds"] - (1 - f["p_push"]), np.nan)
                out[f"{prefix}{name} по месяцам"] = ("post_month", f)
            if prefix:
                for name in MONTHLY:
                    m = M.POST[name]().fit(hb)
                    f = tgt.copy()
                    f["pc"] = m.predict(f)
                    f["ev"] = np.where(f["priced"], f["pc"] * (1 - f["p_push"]) * f["odds"] - (1 - f["p_push"]), np.nan)
                    out[f"{prefix}{name}"] = ("post_shown", f)

    # ---- pre-selection calibration of every candidate
    rows = hist.rows(H)
    pc = d["pc"].to_numpy(float)
    push = d["p_push"].to_numpy(float)
    cals = {"Q1 изотоника по типам": M.KeyCal("isotonic", "group").fit(rows),
            "Q2 Платт по рынкам": M.KeyCal("platt", "market").fit(rows)}
    for name, cal in cals.items():
        pcc = cal.transform_c(d, pc)
        d["p_cal"] = pcc * (1 - push)
        idx = ST.choose(d, P, p_col="p_cal")
        out[name] = ("pre", picks_frame(d, idx, pcc[idx]))
    q2 = cals["Q2 Платт по рынкам"]
    d["pc_cal"] = q2.transform_c(d, pc)
    d["p_cal"] = d["pc_cal"] * (1 - push)

    # Q3: two-stage — post-selection Platt fitted on Q2-chosen history picks (Q2 map of each history season
    # fitted on the OTHER history seasons when there are several, else in-sample)
    hp = []
    for h in H:
        others = [x for x in H if x != h] or [h]
        cal_h = M.KeyCal("platt", "market").fit(hist.rows(others))
        dh = hist.table(h)
        dh["pc_cal"] = cal_h.transform_c(dh, dh["pc"].to_numpy(float))
        dh["p_cal"] = dh["pc_cal"] * (1 - dh["p_push"].to_numpy(float))
        ih = ST.choose(dh, P, p_col="p_cal")
        hp.append(_decided(picks_frame(dh, ih, dh["pc_cal"].to_numpy()[ih])))
        del dh
    hp = pd.concat(hp, ignore_index=True)
    two = M.Platt().fit(hp["pc"], hp["y"])
    info["Q3"] = two.params()
    idx_q2 = ST.choose(d, P, p_col="p_cal")
    f = picks_frame(d, idx_q2, two.predict(d["pc_cal"].to_numpy()[idx_q2]))
    out["Q3 Q2 + Платт после выбора"] = ("pre", f)

    # Q4: Q2 then blend with the de-margined price (weights from history, Q2 applied in-sample there)
    rows_cal = rows.copy()
    rows_cal["pc_cal"] = q2.transform_c(rows_cal, rows_cal["pc"].to_numpy(float))
    w = blend_weights(rows_cal)
    info["Q4_weights"] = w
    wv = d["group"].map(w).fillna(1.0).to_numpy(float)
    qc = d["qc"].to_numpy(float)
    pcb = np.where(np.isfinite(qc), wv * d["pc_cal"].to_numpy() + (1 - wv) * qc, d["pc_cal"].to_numpy())
    d["p_blend"] = pcb * (1 - push)
    idx = ST.choose(d, P, p_col="p_blend")
    out["Q4 Q2 + смесь с рынком"] = ("pre", picks_frame(d, idx, pcb[idx]))

    # ---- ranking functions (calibrated candidates, band on calibrated probability)
    odds = d["odds"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1.0)
    pcal = d["p_cal"].to_numpy(float)
    band = (~d["group"].isin(ST.EXCLUDED)).to_numpy() & (pcal >= P.min_p) & (pcal <= P.max_p)
    if kind == "bet":
        band &= priced
    lb = M.lcb_c(d, d["pc_cal"].to_numpy(float), rows_cal)
    ev_raw = d["p"].to_numpy(float) * odds - (1 - push)
    ev_cal = pcal * odds - (1 - push)
    ev_lb = lb * (1 - push) * odds - (1 - push)
    kel = M.kelly(pcal, odds, push)
    # probability keys on the production scale (probability of a WIN), as the production band and score use it
    keys = {"R1 макс. вероятность": pcal, "R6 нижняя граница p": lb * (1 - push)}
    if kind == "bet":
        keys.update({"R3 макс. EV": ev_raw, "R4 макс. калибр. EV": ev_cal, "R5 Келли": kel,
                     "R7 нижняя граница EV": ev_lb})
    for name, key in keys.items():
        sc = np.where(band & np.isfinite(key), key, -np.inf)
        idx = ST.pick(d, sc, p_col="p_cal", priced_only=kind == "bet")
        out[name] = ("rank", picks_frame(d, idx, d["pc_cal"].to_numpy()[idx]))

    # ---- fewer candidate markets (raw probability, production score)
    score = ST.rule_score(d, P)
    for name, groups in FEW.items():
        sc = np.where(d["group"].isin(groups).to_numpy(), score, -np.inf)
        idx = ST.pick(d, sc, priced_only=kind == "bet")
        out[name] = ("few", picks_frame(d, idx))
    info["n_matches"] = int(d["mi"].nunique())
    info["seconds"] = round(time.time() - t0, 1)
    return d[["mi", "match_id", "date", "league"]].drop_duplicates("mi").sort_values("mi").reset_index(drop=True), out, info
