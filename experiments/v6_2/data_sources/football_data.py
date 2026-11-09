"""Football-Data.co.uk: what the season files really contain, what becomes a feature and what xG would add.

Inputs: ``local_data/results/<season>/*.csv`` (committed Football-Data files), ``local_data/fixtures/fixtures.csv``,
the production model metadata (feature lists) and the cleaning rules of ``src/data/cleaning.py``.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import common as C

LEAGUES = ["E0", "E1", "E2", "E3", "SC0", "SC1", "SC2", "SC3", "SP1", "SP2", "I1", "I2", "D1", "D2", "F1", "F2",
           "N1", "B1", "P1", "T1", "G1"]

# column groups checked per file (share of played rows with a value)
GROUPS = {
    "goals FTHG/FTAG": ["FTHG", "FTAG"], "half-time HTHG/HTAG": ["HTHG", "HTAG"], "referee": ["Referee"],
    "xG HxG/AxG": ["HxG", "AxG"], "shots HS/AS": ["HS", "AS"], "on target HST/AST": ["HST", "AST"],
    "corners HC/AC": ["HC", "AC"], "fouls HF/AF": ["HF", "AF"], "yellow HY/AY": ["HY", "AY"],
    "red HR/AR": ["HR", "AR"],
    "Bet365 1X2 (B365H/D/A)": ["B365H", "B365D", "B365A"], "Pinnacle 1X2 (PSH/D/A)": ["PSH", "PSD", "PSA"],
    "Max/Avg 1X2": ["MaxH", "AvgH"], "Bet365 O/U 2.5": ["B365>2.5", "B365<2.5"],
    "Pinnacle O/U 2.5": ["P>2.5", "P<2.5"], "Asian line AHh + Bet365": ["AHh", "B365AHH", "B365AHA"],
    "Bet365 closing 1X2 (B365CH)": ["B365CH", "B365CD", "B365CA"], "Pinnacle closing 1X2 (PSCH)": ["PSCH"],
    "Bet365 closing O/U 2.5": ["B365C>2.5", "B365C<2.5"], "closing Asian (AHCh + B365CAHH)": ["AHCh", "B365CAHH"],
}

# how the system uses each raw column (src/data/cleaning.py, src/providers/football_data.py, src/features/*)
USE = [
    ("Div, Date, Time, HomeTeam, AwayTeam", "идентификатор матча, сезон, дата/время начала", "модель (ключ), интерфейс"),
    ("FTHG, FTAG (FTR пересчитывается из голов)", "голы → таблица, форма, Elo, H2H, частоты, цели моделей", "модель + расчёт ставок"),
    ("HS, AS", "удары → ~shots_* признаки", "модель"),
    ("HST, AST", "удары в створ → ~sot_* признаки", "модель"),
    ("HC, AC", "угловые → corners_* признаки, цель модели угловых, расчёт рынков угловых", "модель + расчёт ставок"),
    ("HF, AF", "фолы → season_fouls (группа discipline)", "модель (только модель исхода)"),
    ("HY, AY, HR, AR", "карточки → season_yellows / season_reds", "модель (только модель исхода)"),
    ("B365H/D/A, B365>2.5/<2.5, AHh + B365AHH/AHA", "реальные цены Bet365 (provider football_data, низший приоритет)",
     "цены/ROI; признаки рынка выключены (USE_ODDS_FEATURES=False)"),
    ("HTHG, HTAG, HTR, Referee", "не читаются", "—"),
    ("HxG, AxG (с 2026/27)", "не читаются продакшн-кодом (только исследование xG v6.2)", "—"),
    ("B365C*, PSCH, Max*, Avg*, BFE*, другие букмекеры", "не читаются продакшн-кодом (исследования v6.1/v6.2: закрытие, CLV)", "—"),
]

RAW_RULES = [  # feature name (side / window removed) -> raw Football-Data columns it is computed from
    (r"^league=", "Div (лига, категориальный)"),
    (r"corner", "HC, AC (угловые)"),
    (r"sot", "HST, AST (удары в створ)"),
    (r"shot", "HS, AS (удары)"),
    (r"fouls", "HF, AF (фолы)"),
    (r"yellows|reds", "HY, AY, HR, AR (карточки)"),
    (r"rest_days|^form\d+_n$|^season_games$|^venue_games$|^h2h_n$", "Date (календарь, число матчей)"),
    (r".", "FTHG, FTAG (голы и результат)"),
]


def raw_source(feature: str) -> str:
    core = re.sub(r"^[ha]_", "", feature)
    for rx, raw in RAW_RULES:
        if re.search(rx, core):
            return raw
    return "?"


def feature_map() -> dict:
    meta = json.loads((C.DATA / "models" / "metadata.json").read_text())
    active = meta["active"]
    m = json.loads((C.DATA / "models" / active / "metadata.json").read_text())
    rows = []
    for task, info in m["tasks"].items():
        for f in info["features"]:
            rows.append({"model": task, "feature": f, "raw": raw_source(f)})
    df = pd.DataFrame(rows)
    table = df.pivot_table(index="raw", columns="model", values="feature", aggfunc="count", fill_value=0)
    table["unique_features"] = df.drop_duplicates("feature").groupby("raw").size()
    table = table.reset_index().sort_values("unique_features", ascending=False)
    C.save_csv("fd_feature_map_detail.csv", df)
    return {"active_model": active, "use_odds_features": m.get("use_odds_features"),
            "n_features": {t: len(i["features"]) for t, i in m["tasks"].items()},
            "unique_features": int(df["feature"].nunique()), "by_raw": table.to_dict("records")}


def coverage(df: pd.DataFrame) -> dict:
    played = df.dropna(subset=[c for c in ("FTHG", "FTAG") if c in df])
    out = {"rows": int(len(df)), "played": int(len(played))}
    for name, cols in GROUPS.items():
        if not all(c in played for c in cols) or not len(played):
            out[name] = None if not all(c in played for c in cols) else 0.0
            continue
        out[name] = float(played[cols].notna().all(axis=1).mean())
    return out


def season_files() -> tuple[pd.DataFrame, dict]:
    rows = []
    columns = {}
    for season_dir in sorted((C.DATA / "results").iterdir()):
        for lg in LEAGUES:
            f = season_dir / f"{lg}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip")
            rows.append({"season": season_dir.name, "league": lg, "n_columns": df.shape[1], **coverage(df)})
            if season_dir.name == C.SEASON_CODE:
                columns[lg] = [c for c in df.columns if not c.startswith("Unnamed")]
    return pd.DataFrame(rows), columns


def xg_detail(cur: pd.DataFrame) -> dict:
    """xG of 2026/27: coverage per league and how close xG is to goals (a plausibility check)."""
    out = []
    for lg in LEAGUES:
        f = C.DATA / "results" / C.SEASON_CODE / f"{lg}.csv"
        df = pd.read_csv(f, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip")
        played = df.dropna(subset=["FTHG", "FTAG"])
        has = played[["HxG", "AxG"]].notna().all(axis=1) if {"HxG", "AxG"} <= set(played) else pd.Series(False, index=played.index)
        g = played.loc[has]
        out.append({"league": lg, "played": int(len(played)), "with_xg": int(has.sum()),
                    "share": float(has.mean()) if len(played) else None,
                    "mean_xg_total": float((g["HxG"] + g["AxG"]).mean()) if len(g) else None,
                    "mean_goals_total": float((g["FTHG"] + g["FTAG"]).mean()) if len(g) else None,
                    "corr_xg_goals": float((g["HxG"] + g["AxG"]).corr(g["FTHG"] + g["FTAG"])) if len(g) > 5 else None})
    return {"leagues": out, "total_played": sum(r["played"] for r in out),
            "total_with_xg": sum(r["with_xg"] for r in out)}


def fixtures_file() -> dict:
    f = C.DATA / "fixtures" / "fixtures.csv"
    df = pd.read_csv(f, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip")
    return {"rows": int(len(df)), "columns": int(df.shape[1]),
            "leagues": sorted(df["Div"].dropna().unique().tolist()),
            "with_b365_1x2": float(df[["B365H", "B365D", "B365A"]].notna().all(axis=1).mean()) if len(df) else None,
            "with_closing": float(df["B365CH"].notna().mean()) if "B365CH" in df and len(df) else None}


def run() -> dict:
    files, columns = season_files()
    C.save_csv("fd_season_coverage.csv", files)
    cur = files.loc[files["season"] == C.SEASON_CODE]
    xg = xg_detail(cur)
    out = {"season_coverage_current": cur.to_dict("records"),
           "xg_by_season": files.groupby("season")["xG HxG/AxG"].apply(lambda s: None if s.isna().all() else float(s.mean())).to_dict(),
           "columns_current": {lg: len(c) for lg, c in columns.items()},
           "column_names_E0": columns.get("E0", []),
           "xg": xg, "fixtures_file": fixtures_file(), "feature_map": feature_map(),
           "use": [{"columns": a, "becomes": b, "used_by": c} for a, b, c in USE]}
    C.save_json("football_data.json", out)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("xg_by_season", "fixtures_file")}, indent=1))
    print(pd.DataFrame(r["feature_map"]["by_raw"]).to_string())
