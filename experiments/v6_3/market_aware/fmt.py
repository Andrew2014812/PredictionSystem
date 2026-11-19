"""Formatting helpers of the three market-study reports (numbers always come from results/)."""
from __future__ import annotations

from scripts.report_kit import P, table

U_RU = {"close": "close", "t09": "t09"}


def pp(v, d=1, signed=True):
    if v is None:
        return "—"
    return f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}"


def ci(c, d=1, mult=100):
    if not c or c[0] is None:
        return "—"
    return f"[{c[0] * mult:+.{d}f}; {c[1] * mult:+.{d}f}]"


def pval(v):
    if v is None:
        return "—"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def num(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def roi_cell(r):
    return f"{pp(r['roi'])}% {ci(r['ci95'])}" if r and r.get("roi") is not None else "—"


def variant_rows(variants: dict, names: dict, universe: str, holm: dict, keys=None, a: dict | None = None):
    """Rows: name, bets, hit, odds, ROI [CI], Δ vs B [CI], p, p Holm, same Main."""
    rows = [["Вариант", "Ставок", "Hit", "Ср. кэф", "ROI, % [95 % ДИ]", "Δ к B, п.п. [95 % ДИ]", "p", "p Холма",
             "Тот же Main"]]
    keys = keys or list(names)
    if a is not None:
        s = a["summary"]
        rows.append(["Baseline A (история сайта)", s["bets"], pp(s["hit_rate"], 1, False), num(s["avg_odds"], 2),
                     roi_cell(a["roi_ci"]), "—", "—", "—", "—"])
    for k in keys:
        if k not in variants:
            continue
        v = variants[k]
        s = v["summary"]
        vs = v.get("vs_b")
        rows.append([names.get(k, k), s["bets"], pp(s["hit_rate"], 1, False), num(s["avg_odds"], 2),
                     roi_cell(v["roi_ci"]),
                     f"{pp(vs['roi_diff']['diff'])} {ci(vs['roi_diff']['ci95'])}" if vs else "—",
                     pval(vs["roi_diff"]["p"]) if vs else "—",
                     pval(holm.get(f"{k}|{universe}")) if vs else "—",
                     pp(vs["same_main_share"], 0, False) + "%" if vs else "—"])
    return rows


def stability_rows(variants: dict, names: dict, keys):
    rows = [["Вариант", "Месяцев лучше B", "Лиг лучше B", "ROI без 5 лучших", "ROI без 10 лучших",
             "ROI, цены −3 %", "ROI, цены −5 %"]]
    for k in keys:
        if k not in variants:
            continue
        s = variants[k]["stability"]
        rows.append([names.get(k, k), f"{s['months_better']} / {s['months_total']}",
                     f"{s['leagues_better']} / {s['leagues_total']}", pp(s.get("roi_without_top5")) + "%",
                     pp(s.get("roi_without_top10")) + "%", pp(s.get("roi_odds_x0.97")) + "%",
                     pp(s.get("roi_odds_x0.95")) + "%"])
    return rows


def month_rows(variants: dict, key: str, name: str):
    rows = [["Месяц", f"Ставок ({name})", f"ROI ({name}), %", "Ставок B", "ROI B, %", "Hit", "Hit B"]]
    for r in variants[key]["stability"]["months"]:
        small = " (мало)" if r["bets"] < 100 else ""
        rows.append([r["month"] + small, r["bets"], pp(r["roi"]), r["bets_b"], pp(r["roi_b"]),
                     pp(r["hit_rate"], 1, False), pp(r["hit_rate_b"], 1, False)])
    return rows


def league_rows(variants: dict, key: str, name: str):
    rows = [["Лига", f"Ставок ({name})", f"ROI ({name}), %", "ROI B, %", "Δ, п.п."]]
    lg = sorted(variants[key]["stability"]["leagues"], key=lambda r: r["league"])
    for r in lg:
        d = (r["roi"] - r["roi_b"]) if r["roi"] is not None and r["roi_b"] is not None else None
        rows.append([r["league"] + (" *" if r["small"] else ""), r["bets"], pp(r["roi"]), pp(r["roi_b"]), pp(d)])
    return rows


def smart_widths(rows, total_cm: float = 17.0):
    """Column widths from the longest body cell and the longest header word (numbers never wrap)."""
    from reportlab.lib.units import cm
    n = max(len(r) for r in rows)
    need = []
    for i in range(n):
        body = max([len(str(r[i])) for r in rows[1:] if i < len(r)] or [1])
        head = max(len(w) for w in str(rows[0][i]).split()) if i < len(rows[0]) and str(rows[0][i]).split() else 1
        need.append(max(min(body, 14), head * 1.25, 5) + 3)
    tot = sum(need)
    return [total_cm * cm * x / tot for x in need]


def T(rows, total_cm: float = 17.0):
    return table(rows, widths=smart_widths(rows, total_cm))


def coverage_table(df):
    cols = ["universe", "method", "played", "processed", "bets_real_price", "no_price", "push",
            "without_statistics"]
    names = ["Мир", "Метод", "Матчей сезона", "Сыграно", "Обработано", "Ставок (реальная цена)", "Без цены",
             "PUSH", "Нет статистики", "Не сыграны"]
    cols = [c for c in cols if c in df.columns]
    rows = [[names[["universe", "method", "season_matches", "played", "processed", "bets_real_price", "no_price",
                    "push", "without_statistics", "not_played_excluded"].index(c)] for c in cols]]
    for _, r in df.iterrows():
        rows.append([r[c] for c in cols])
    return T(rows)


def para_list(items):
    return [P(i) for i in items]


def H1c(text):
    """A section heading that never stays alone at the bottom of a page."""
    from reportlab.lib.units import cm
    from reportlab.platypus import CondPageBreak

    from scripts.report_kit import H1
    return [CondPageBreak(5 * cm), H1(text)]
