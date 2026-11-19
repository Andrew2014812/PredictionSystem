"""Formatting helpers of the two v6.3 selection reports (numbers always come from results/)."""
from __future__ import annotations

import pandas as pd

from reportlab.lib.units import cm

from scripts import report_kit as K

ULAB = {"close": "close (до старта)", "t09": "T (утренний прогон)", "bt": "2025/26 бэктест"}
USHORT = {"close": "close", "t09": "T", "bt": "2025/26"}


def pp(v, d=1, signed=True):
    """fraction -> percentage points / percent with sign."""
    if v is None or v != v:
        return "—"
    return f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}"


def pc(v, d=1, signed=False):
    return "—" if v is None or v != v else (f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%")


def ci(lo_hi, d=1):
    if not lo_hi or lo_hi[0] is None:
        return "—"
    return f"[{pp(lo_hi[0], d)}; {pp(lo_hi[1], d)}]"


def pv(v):
    if v is None or v != v:
        return "—"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def num(v, d=2):
    return "—" if v is None or v != v else f"{v:.{d}f}"


def candidates_table(rows: list[dict], universe: str) -> list[list]:
    """Pre-chosen candidates of one universe: ≤ 11 columns."""
    out = [["Кандидат", "Ставок", "ROI", "Попад.", "Ср. коэф.", "ΔROI к B, п.п. [95 % ДИ]", "p", "p Холма",
            "Изменено", "Мес. лучше B", "Резерв (не ставка)"]]
    for r in rows:
        if r["universe"] != universe:
            continue
        out.append([r["candidate"], r["bets"], pc(r["roi"]), pc(r["hit_rate"]), num(r["avg_odds"]),
                    f"{pp(r['roi_diff'])} {ci(r['diff_ci'])}", pv(r["p"]), pv(r.get("p_holm")),
                    pc(r["changed_share"], 0), f"{r['months_better']}/{r['months_n']}", r["fallback_not_bet"]])
    return out


def grid_table(grid: pd.DataFrame, families: list[str] | None = None) -> list[list]:
    """Full grid, one row per rule: bt / close / t09 side by side (≤ 11 columns)."""
    out = [["Семейство", "Правило", "ROI bt", "ROI close", "ΔROI close", "p Холма close", "ROI T", "ΔROI T",
            "p Холма T", "Изменено close", "Ставок T"]]
    g = grid.loc[grid["rule"] != "production"]
    if families:
        g = g.loc[g["family"].isin(families)]
    for (fam, rule), x in g.groupby(["family", "rule"], sort=False):
        x = x.set_index("universe")
        get = lambda u, c: x.loc[u, c] if u in x.index else None  # noqa: E731
        out.append([fam, rule, pc(get("bt", "roi")), pc(get("close", "roi")), pp(get("close", "roi_diff")),
                    pv(get("close", "p_holm_grid")), pc(get("t09", "roi")), pp(get("t09", "roi_diff")),
                    pv(get("t09", "p_holm_grid")), pc(get("close", "changed_share"), 0),
                    int(get("t09", "bets")) if get("t09", "bets") is not None else "—"])
    return out


def months_table(details: dict, names: list[str], universe: str) -> list[list]:
    months = [m["month"] for m in details[names[0]][universe]["months"]]
    out = [["Кандидат"] + [f"{m}: ΔROI" for m in months]]
    for n in names:
        mm = {m["month"]: m for m in details[n][universe]["months"]}
        out.append([n] + [f"{pp(mm[m]['diff'])} ({mm[m]['bets']})" if m in mm else "—" for m in months])
    return out


def league_table(details: dict, base: dict, name: str, universe: str, top: int = 40) -> list[list]:
    lb = {r["league"]: r for r in base[universe]["leagues"]}
    out = [["Лига", "Ставок", "ROI кандидата", "ROI B", "Разница, п.п.", "Малая выборка"]]
    for r in sorted(details[name][universe]["leagues"], key=lambda r: r["league"])[:top]:
        b = lb.get(r["league"], {})
        diff = None if r["roi"] is None or b.get("roi") is None else r["roi"] - b["roi"]
        out.append([r["league"], r["bets"], pc(r["roi"]), pc(b.get("roi")), pp(diff), "да" if r["small"] else ""])
    return out


def sensitivity_table(details: dict, base: dict, names: list[str], universe: str) -> list[list]:
    out = [["Метод", "ROI", "коэф. −2 %", "коэф. −5 %", "без 5 лучших", "без 10 лучших", "Макс. выигрыш",
            "Доля EV<0", "Переоценка, п.п."]]
    rows = [("Baseline B", base[universe])] + [(n, details[n][universe]) for n in names]
    for n, d in rows:
        s, z = d["summary"], d["sensitivity"]
        gap = d.get("gap")
        if gap is None and s.get("hit_rate") is not None:
            gap = s["avg_probability"] - s["hit_rate"]
        out.append([n, pc(s["roi"]), pc(z.get("roi_odds_minus_2pct")), pc(z.get("roi_odds_minus_5pct")),
                    pc(z.get("roi_without_top5_wins")), pc(z.get("roi_without_top10_wins")),
                    num(z.get("largest_win")), pc(s["neg_ev_share"]), pp(gap)])
    return out


def coverage_rows(cov: pd.DataFrame) -> list[list]:
    out = [["Метод", "Вселенная", "Обработано", "С реальной ценой", "Ставок", "Без цены (не ставка)",
            "Резерв (не ставка)", "Возврат", "Неизв. итог"]]
    for _, r in cov.iterrows():
        out.append([r["method"], r["universe"], r["processed"], r["main_with_real_price"], r["bets"],
                    r["not_bet_no_price"], r["not_bet_fallback"], r["push"], r["unknown_result"]])
    return out


W = {"cand": [3.1, 1.25, 1.2, 1.2, 1.1, 3.0, 1.1, 1.1, 1.2, 1.25, 1.5],
     "grid": [2.5, 2.8, 1.25, 1.35, 1.25, 1.35, 1.25, 1.25, 1.35, 1.35, 1.25],
     "sens": [3.0, 1.5, 1.5, 1.5, 1.7, 1.7, 1.6, 1.5, 2.0],
     "cov": [3.2, 1.6, 1.8, 1.9, 1.4, 2.0, 2.0, 1.5, 1.6],
     "floor": [3.2, 1.6, 1.3, 1.3, 1.4, 1.2, 1.4, 1.4, 1.3, 1.2, 1.9],
     "base": [3.2, 1.5, 3.6, 1.8, 1.6, 1.8, 2.6]}


def widths(key: str) -> list[float]:
    return [w * cm for w in W[key]]


def mix_rows(groups: dict, label: str) -> list:
    return [label] + [pc(groups.get(g, 0.0)) for g in ("TOTAL", "HANDICAP", "1X2", "DOUBLE_CHANCE", "CORNERS", "BTTS")]


MIX_HEAD = ["Метод", "TOTAL", "HANDICAP", "1X2", "DC", "CORNERS", "BTTS"]


class V63Report(K.Report):
    pass


def ew_table(ew: dict, universe: str) -> list[list]:
    """Member chosen by the expanding window in every month (one row per family)."""
    fams = list(ew)
    months = [c["month"] for c in ew[fams[0]][universe]]
    out = [["Семейство"] + months]
    for f in fams:
        out.append([f] + [c["chosen"] for c in ew[f][universe]])
    return out
