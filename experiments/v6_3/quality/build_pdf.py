"""Builds reports/v6_3/Quality_Coverage_Research.pdf and findings.md from results/ (no number typed by hand).

    .venv/bin/python -m experiments.v6_3.quality.build_pdf
"""
from __future__ import annotations

import re

import pandas as pd

from experiments.v6_3 import common as C
from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, figure, formula, note, table

from . import mitigate as MT

D = C.study_dirs("quality")
RES, FIG = D["results"], D["figures"]
OUT = C.ROOT / "reports" / "v6_3" / "Quality_Coverage_Research.pdf"


def j(name):
    return C.read_json(RES / name)


def pp(v, d=2, signed=True):
    """fraction -> percentage points / percent text."""
    if v is None or v != v:
        return "—"
    return f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}"


def W(*cm_):
    return [c * K.cm for c in cm_]


def ci_pp(m, key="diff", d=2):
    if not m or m.get(key) is None:
        return "—"
    lo, hi = m["ci95"]
    return f"{pp(m[key], d)} [{pp(lo, d)}; {pp(hi, d)}]"


def ci_k(m, key=None, d=1, scale=1000):
    """×10⁻³ with CI."""
    if m and key is None:
        key = "diff" if "diff" in m else "mean"
    if not m or m.get(key) is None:
        return "—"
    lo, hi = m["ci95"]
    return f"{m[key] * scale:+.{d}f} [{lo * scale:+.{d}f}; {hi * scale:+.{d}f}]"


def roi_txt(r):
    if not r or r.get("roi") is None:
        return "—"
    lo, hi = r["ci95"]
    return f"{r['roi'] * 100:+.1f} [{lo * 100:+.1f}; {hi * 100:+.1f}]"


def pval(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def small(r):
    return " *" if r and r.get("small") else ""


def strata_table(st: dict, dim: str, keys=None, current=True) -> list:
    d = st["current" if current else "history"][dim]
    keys = keys or sorted(d)
    col = "excess_close" if current else "excess_fd"
    if current:
        rows = [["Группа", "N", "LL", "Excess, ·10⁻³", "ROI close, %", "Ст.", "ROI t09, %", "Hit"]]
        for k in keys:
            if k not in d:
                continue
            g = d[k]
            rows.append([k, str(g["matches"]), K.num(g["log_loss"]["mean"], 4), ci_k(g[col]),
                         roi_txt(g["main_close"]) + small(g["main_close"]), str(g["main_close"]["bets"]),
                         roi_txt(g["main_t09"]) + small(g["main_t09"]), K.share(g["main_close"]["hit"])])
    else:
        rows = [["Группа", "N", "LL", "Excess, ·10⁻³", "ROI (Bet365), %", "Ст.", "Hit"]]
        for k in keys:
            if k not in d:
                continue
            g = d[k]
            rows.append([k, str(g["matches"]), K.num(g["log_loss"]["mean"], 4), ci_k(g[col]),
                         roi_txt(g["main_fd"]) + small(g["main_fd"]), str(g["main_fd"]["bets"]), K.share(g["main_fd"]["hit"])])
    return rows


W_CUR = [2.8, 0.9, 1.2, 3.4, 3.3, 0.9, 3.3, 1.2]
W_HIST = [2.8, 1.1, 1.3, 3.6, 3.6, 1.1, 1.3]


def build() -> None:
    st, sel, mit = j("strata.json"), j("selection.json"), j("mitigations_2026.json")
    idn, grp, info = j("identification.json"), j("feature_groups.json"), j("run_info.json")
    hist = pd.read_csv(RES / "selection_history.csv")
    cov = pd.read_csv(RES / "coverage.csv")
    imp = pd.read_csv(RES / "feature_groups_importance.csv")
    miss = pd.read_csv(RES / "feature_groups_missing.csv")
    base = C.read_json(C.SHARED_RESULTS / "baseline.json")
    chk = j("self_check.json") if (RES / "self_check.json").exists() else {}
    V, dec = mit["variants"], mit["decisions"]
    prim = V[mit["primary"]]["universes"]
    cnt = st["counts"]
    cur, hst = st["current"], st["history"]
    nb_c, nb_h = cur["n_bucket"], hst["n_bucket"]
    rep = K.Report(OUT, "Качество и полнота данных", "FootPredict v6.3 — исследование 6 (Quality vs Coverage)",
                   {"Сезон проверки": f"2026/27 (сыграно {cnt['current_matches']} матчей, уже изучен в v6.1/v6.2)",
                    "Сезоны выбора параметров": f"2024/25 + 2025/26 ({cnt['history_matches']} матчей, out-of-sample)",
                    "Модели": "prod_20261012 (XGBoost, 4 задачи) — Baseline B",
                    "Данные": f"features.parquet от {info['data']['features']['mtime'][:10]}"})

    # 1
    rep.add(H1("1. Название исследования"),
            P("**Влияние полноты и качества входных данных на прогнозы и Main FootPredict; смягчение разреженных данных "
              "без исключения матчей.** Исследование 6 технического задания v6.3 (Quality vs Coverage). Строится на "
              "v6.2 Data Quality (`experiments/v6_2/data_quality`): там проверены побитная воспроизводимость признаков, "
              "обнуление признаков одной команды и Elo новичка; здесь — полный пересчёт 2026/27 в двух моментах решения, "
              "страты по полноте данных и проверка смягчающих правил, сохраняющих один Main на матч."))
    # 2
    rep.add(H1("2. Постановка проблемы"),
            P("FootPredict выдаёт Main для каждого матча 21 лиги, включая начало сезона (у команд 0–2 сыгранных матча), "
              "клубы, поднявшиеся из лиг, которые система не загружает (нет истории), лиги с неполной статистикой и "
              "матчи без реальной цены на момент ежедневного запуска. В v6.2 найдены фантомные клубы из-за опечаток в "
              "названиях: такой клуб для модели — новичок без истории. Вопрос: насколько хуже прогнозы и Main на "
              "неполных данных и можно ли уменьшить потерю, **не исключая** неудобные матчи (одно Main на матч "
              "сохраняется всегда)."))
    # 3
    rep.add(H1("3. Гипотезы"),
            bullets(["H1. Чем меньше сыграно матчей сезона, тем больше проигрыш модели рынку (excess log loss).",
                     "H2. Матчи с клубами без истории (outside, returner) и фантомы (неверная идентификация) "
                     "прогнозируются хуже; Main в них хуже.",
                     "H3. Неполная статистика (пропуски признаков ударов / угловых) ухудшает прогноз.",
                     "H4. Лиги различаются по качеству прогноза и покрытию цен.",
                     "H5. Отсутствие реальной цены в момент решения (t09) меняет состав и доходность Main.",
                     "H6. Часть групп признаков ненадёжна: их важность не переносится между сезонами.",
                     "H7. Сжатие вероятностей к рынку / к приорам лиги в матчах с разреженными данными и ранжирование "
                     "без EV там снижают log loss и не снижают ROI Main."]))
    # 4
    c0 = cov.loc[(cov["method"] == "baseline_b") & (cov["universe"] == "close")].iloc[0]
    rep.add(H1("4. Исходные данные"),
            P(f"Сезон 2026/27: {base['coverage']['season_matches']} матчей, сыграно {base['coverage']['played']} "
              f"(по {base['coverage']['last_played']}). Матчи, цены и Baseline B — общий replay v6.3 "
              "(`experiments/v6_3/replay`). Сезоны 2024/25 и 2025/26 — прогнозы production-моделей, переобученных "
              "только на более ранних сезонах (`oos.py`), с ценами Bet365 из файлов Football-Data (1X2, тотал 2.5, "
              "одна азиатская линия)."),
            table([["Показатель", "Значение"],
                   ["Матчей сезона / сыграно", f"{int(c0['season_matches'])} / {int(c0['played'])}"],
                   ["Обработано (Main есть у каждого сыгранного матча)", str(int(c0["processed"]))],
                   ["Main с реальной ценой (close)", str(int(c0["with_real_price"]))],
                   ["Матчи с разреженными данными (max(s) ≥ 0.5)", str(int(c0["low_data_matches"]))],
                   ["Матчи с клубом без истории (form10_n < 10)", str(int(c0["newcomer_like_matches"]))],
                   ["Матчи с пропусками статистики (удары / угловые)", str(int(c0["matches_partial_stats"]))],
                   ["Исключено из ROI: без цены / PUSH,VOID / результат рынка неизвестен",
                    f"{int(c0['no_price_not_a_bet'])} / {int(c0['push_or_void'])} / {int(c0['unknown_result'])}"]]),
            P("Ни один сыгранный матч не исключён: каждая страта — разбиение всех матчей; матчи без цены считаются "
              "прогнозами, но не ставками (правило production).", "small"))
    # 5
    rep.add(H1("5. Методика"),
            P("**Страты** (известны до начала матча, из признаков системы):"),
            bullets(["n_{min} = min(h_season_games, a_season_games) — сколько матчей сезона сыграла «младшая» команда;",
                     "hist_{min} = min(h_form10_n, a_form10_n) — сколько матчей в окне формы (через сезоны); < 10 — "
                     "новичок без истории или фантом;",
                     "статус клуба (как в v6.2): established, moved_within (переход между нашими лигами), returner, "
                     "outside — худший из двух клубов, в разрезе n_{min} ≤ 5 / ≥ 6;",
                     "dq_missing — доля пустых признаков модели 1X2; stats — есть ли пропуски признаков ударов / угловых;",
                     "лига и уровень лиги; доступность цены на момент T (OddsPapi есть / только Football-Data; цена Main)."]),
            P("**Потери по матчу:** log loss 1X2 модели и рынка (цена без маржи, пропорционально), excess = модель − "
              "рынок (сложность матча не выдаётся за плохие данные), девиация Пуассона голов и угловых. Интервалы — "
              f"бутстреп по матчам ({C.N_BOOT} повторов); разница страт — независимый бутстреп двух групп."),
            P("**Смягчения** (один Main на матч сохраняется; все матчи остаются):"),
            formula("s_{season} = clip((10 − n_{min}) / 10, 0, 1),   s_{hist} = clip((10 − hist_{min}) / 10, 0, 1)"),
            formula("mkt: p′ = (1 − ω)·p + ω·q,   ω = w·s  (q — 1X2 без маржи в момент решения)"),
            formula("prior: λ′ = (1 − ω)·λ + ω·λ_{league},   ω = w·max(s_{season}, s_{hist})"),
            bullets([f"mkt_season(w), w ∈ {MT.GRID['mkt_season']}; mkt_hist(w), w ∈ {MT.GRID['mkt_hist']}; prior(w), "
                     f"w ∈ {MT.GRID['prior']};",
                     "mkt_both — ω = max(w_{1}·s_{season}, w_{2}·s_{hist}) с выбранными w_{1}, w_{2} (основной вариант, "
                     "объявлен до запуска);",
                     "mkt_uniform — тот же средний вес для всех матчей (контроль: важна ли адресность);",
                     f"lowdata_rank — правило Main без члена EV (EV weight = 0) в матчах с max(s) ≥ {MT.LOWDATA}."]),
            P("**Выбор параметров:** вес каждого семейства выбран на 2024/25 + 2025/26 (минимум log loss 1X2; для prior — "
              "минимум девиации голов), до просмотра 2026/27. На 2026/27 показаны все варианты; поправка Холма по всем "
              "вариантам в каждом универсуме. Main считается кодом production (`engine.batch_from_outputs`): изменённые "
              "вероятности проходят через ту же математику рынков и правило Main."),
            P("**Идентификация команд:** аудит названий 2026/27 и контролируемый эксперимент — в каждом матче один клуб "
              "(хозяева в чётных строках, гости в нечётных) превращается в «фантом» так, как построитель признаков видит "
              "незнакомый клуб (обнуление v6.2 newcomer); затем проверяется, сколько ущерба возвращают смягчения."),
            P("**Группы признаков:** перестановочная важность каждой группы (совместная перестановка столбцов группы) "
              "на 2025/26 (модель ≤ 2024/25) и 2026/27 (production) для всех четырёх задач; стабильность — ранговая "
              "корреляция важностей двух сезонов и смены знака."))
    # 6
    rep.add(H1("6. Эксперименты"),
            table([["Эксперимент", "Объём"],
                   ["Страты 2026/27 (production, оба универсума)", f"{cnt['current_matches']} матчей × 8 измерений"],
                   ["Страты 2024/25 + 2025/26 (out-of-sample)", f"{cnt['history_matches']} матчей × 8 измерений"],
                   ["Смягчения: выбор на 2024/25 + 2025/26", f"{mit['n_variants_tried']} вариантов"],
                   ["Смягчения: проверка на 2026/27", f"{mit['n_variants_on_2026']} вариантов × 2 универсума"],
                   ["Ошибочная идентификация (фантом)", f"{idn['matches']} матчей, 4 сценария × 2 универсума"],
                   ["Перестановочная важность групп", f"{len(imp)} пар (задача, группа, сезон)"]]))
    # 7
    rep.add(H1("7. Результаты"), H2("7.1 Число сыгранных матчей сезона (H1)"),
            table(strata_table(st, "n_bucket", ["0–2", "3–5", "6–9", "10+"]), widths=W(*W_CUR), font=7.0),
            P("2026/27. Строки — n_{min}; N — матчи, LL — log loss 1X2, Excess — log loss модели − рынка (closing), "
              "Ст. — ставки; * — меньше 100 ставок. ROI и hit — Main Baseline B (close).", "small"),
            table(strata_table(st, "n_bucket", ["0–2", "3–5", "6–9", "10+"], current=False), widths=W(*W_HIST), font=7.4),
            P("То же на сезонах выбора (out-of-sample, цены Bet365).", "small"))
    keys_club = ["established | 6+", "established | 0–5", "moved_within | 6+", "moved_within | 0–5", "returner | 6+",
                 "returner | 0–5", "outside | 6+", "outside | 0–5"]
    rep.add(H2("7.2 История клубов и новички (H2)"), table(strata_table(st, "club_early", keys_club), widths=W(*W_CUR), font=7.0),
            table(strata_table(st, "club_early", keys_club, current=False), widths=W(*W_HIST), font=7.2),
            P("Верх — 2026/27, низ — 2024/25 + 2025/26. Группа — худший статус из двух клубов | n_{min}.", "small"))
    rep.add(H2("7.3 Полнота статистики (H3)"), table(strata_table(st, "dq_bin", ["0 %", "0–5 %", "5–20 %", "> 20 %"]), widths=W(*W_CUR), font=7.0),
            table(strata_table(st, "stats_bin", ["full", "partial"]), widths=W(*W_CUR), font=7.0),
            P(f"Ранговая корреляция доли пропусков с excess log loss: 2026/27 {st['spearman']['dq_vs_excess_close']:+.3f}, "
              f"история {st['spearman']['dq_vs_excess_fd_hist']:+.3f}; n_{{min}} с excess: "
              f"{st['spearman']['nmin_vs_excess_close']:+.3f} и {st['spearman']['nmin_vs_excess_fd_hist']:+.3f}.", "small"))
    tier = cur["tier"]
    rows = [["Уровень", "Матчи", "Модель − рынок, ·10⁻³", "ROI close, %", "Ставки", "ROI t09, %"]]
    for k in sorted(tier, key=lambda s: int(s)):
        g = tier[k]
        rows.append([k, str(g["matches"]), ci_k(g["excess_close"]), roi_txt(g["main_close"]) + small(g["main_close"]),
                     str(g["main_close"]["bets"]), roi_txt(g["main_t09"]) + small(g["main_t09"])])
    lg = cur["league"]
    rows_l = [["Лига", "Матчи", "Модель − рынок", "ROI close, %", "Ставки", "OddsPapi"]]
    opc = idn["audit"]["oddspapi_closing_share_by_league"]
    for k in sorted(lg, key=lambda s: lg[s]["excess_close"]["mean"]):
        g = lg[k]
        rows_l.append([k, str(g["matches"]), ci_k(g["excess_close"]), roi_txt(g["main_close"]) + small(g["main_close"]),
                       str(g["main_close"]["bets"]), K.share(opc.get(k), 0)])
    rep.add(H2("7.4 Лиги и их покрытие (H4)"), table(rows, widths=W(1.8, 1.4, 4.4, 4.0, 1.6, 3.8), font=7.6),
            table(rows_l, widths=W(1.6, 1.4, 4.4, 4.2, 1.6, 2.0), font=7.2),
            P("Лиги упорядочены по excess log loss (2026/27, рынок closing); OddsPapi — доля сыгранных матчей с ценой "
              "OddsPapi на закрытии (идентификация матча в фиде цен). * — меньше 100 ставок.", "small"))
    pa = st["price_availability"]
    rows = [["Группа (t09)", "Матчи", "ROI close, %", "ROI t09, %", "Ставки t09", "Hit close"]]
    for k, lab in (("t09_main_priced", "Main с ценой в T"), ("t09_main_unpriced", "Main без цены в T"),
                   ("oddspapi_at_t09", "OddsPapi есть в T"), ("football_data_only_at_t09", "только Football-Data в T")):
        if k in pa:
            g = pa[k]
            rows.append([lab, str(g["matches"]), roi_txt(g["close"]), roi_txt(g["t09"]), str(g["t09"]["bets"]),
                         K.share(g["close"]["hit"])])
    for k, g in pa["by_priced_count"].items():
        rows.append([f"цен в T: {k}", str(g["matches"]), roi_txt(g["close"]), roi_txt(g["t09"]), str(g["t09"]["bets"]),
                     K.share(g["close"]["hit"])])
    rep.add(H2("7.5 Доступность реальных цен (H5)"), table(rows, widths=W(4.0, 1.4, 3.5, 3.5, 1.6, 1.8), font=7.6),
            P("Одни и те же матчи оценены в двух моментах решения: сравнение «close» внутри групп показывает, хуже ли "
              "сами матчи, у которых в T нет цены Main, или дело только в том, что они не становятся ставками.", "small"))
    a = idn["audit"]
    rows = [["Сценарий", "Univ.", "Log loss Δ, ·10⁻³", "ROI, %", "ΔROI к B, п.п.", "p", "Тот же Main"]]
    for name in ("misidentified", "misidentified + mkt_hist", "misidentified + mkt_both", "correct + mkt_both"):
        for u in ("close", "t09"):
            r = idn[name][u]
            rows.append([name, u, ci_k(r["d_log_loss"]), roi_txt(r["roi_ci"]), ci_pp(r["vs_b"]["roi_diff"], d=2),
                         pval(r["vs_b"]["roi_diff"]["p"]), K.share(r["vs_b"]["same_main_share"], 0)])
    rep.add(H2("7.6 Идентификация команд"),
            P(f"Аудит 2026/27: {a['clubs_2026']} клубов, из них {a['outside_clubs_2026']} впервые в данных; пар «новое имя "
              f"похоже на известное»: {len(a['similar_name_pairs'])}; клубов с числом матчей < половины медианы лиги: "
              f"{len(a['clubs_with_few_matches'])}; фантомные матчи v6.2 в данных: {a['v62_phantoms_present']}. Доля "
              f"сыгранных матчей с ценой OddsPapi на закрытии — {K.share(a['oddspapi_closing_share'])}."),
            table(rows, widths=W(3.9, 1.0, 2.9, 2.6, 3.7, 1.0, 1.5), font=7.0),
            P("misidentified — один клуб в каждом матче превращён в фантом (без истории); + mkt_* — то же со смягчением "
              "(веса выбраны на истории; у фантома s_hist = s_season = 1, поэтому оба варианта дают вес 1 и совпадают); "
              "correct + mkt_both — смягчение без ошибок идентификации.", "small"))
    if a["similar_name_pairs"]:
        rows = [["Лига", "Новое имя", "Известное имя", "Сходство", "Общие слова"]]
        for r in a["similar_name_pairs"][:12]:
            rows.append([r["league"], r["new_name"], r["known_name"], K.num(r["ratio"], 2), r["shared"]])
        reserve = all(re.search(r"\b(B|II|U23|U21)$", r["new_name"]) for r in a["similar_name_pairs"])
        rep.add(table(rows, widths=W(1.4, 4.0, 4.0, 2.0, 5.6), font=7.6),
                P("Кандидаты в алиасы по сходству названий. " + ("Все найденные пары — вторые (резервные) команды "
                  "клубов, это разные клубы, а не опечатки." if reserve else "Пары требуют ручной проверки по "
                  "составу лиги."), "small"))
    stab = grp["stability"]
    rows = [["Задача", "Групп", "Spearman 2025/26 ↔ 2026/27", "Обе > 0", "Смена знака"]]
    for t, v in stab.items():
        rows.append([t, str(v["groups"]), K.num(v["spearman"], 2), str(v["both_positive"]), str(v["sign_flip"])])
    w = imp.pivot_table(index=["task", "group"], columns="season", values="importance").reset_index()
    flips = w.loc[(w.get("2025/26", 0) > 0) != (w.get("2026/27", 0) > 0)] if "2025/26" in w else w.iloc[0:0]
    top = imp.loc[imp["season"] == "2026/27"].sort_values("importance", ascending=False).groupby("task").head(3)
    rows2 = [["Задача", "Группа", "Важн. 2026/27", "Важн. 2025/26", "Пропуски"]]
    for r in top.itertuples():
        prev = imp.loc[(imp["season"] == "2025/26") & (imp["task"] == r.task) & (imp["group"] == r.group), "importance"]
        rows2.append([r.task, r.group, f"{r.importance * 1000:.2f}", f"{prev.iloc[0] * 1000:.2f}" if len(prev) else "—",
                      K.share(r.missing_share)])
    mg = miss.groupby("group")["missing_share"].agg(["mean", "max"]).sort_values("max", ascending=False).head(6)
    rep.add(H2("7.7 Надёжность групп признаков (H6)"), table(rows, widths=W(3.0, 1.5, 5.0, 2.0, 2.5), font=8),
            table(rows2, widths=W(3.0, 4.0, 3.0, 3.0, 2.5), font=7.8),
            P("Важность — рост потерь (·10⁻³) при перестановке группы; три самые важные группы каждой задачи. "
              f"Групп со сменой знака важности между сезонами: {len(flips)} "
              f"({', '.join(sorted(set(flips['group'])))[:200] if len(flips) else 'нет'}). Наибольшие доли пропусков по "
              "лигам 2026/27: " + "; ".join(f"{g} {K.share(r['max'], 0)} (в среднем {K.share(r['mean'], 1)})"
                                           for g, r in mg.iterrows()) + ".", "small"))
    # mitigations
    hb = sel["history_baseline"]
    rows = [["Вариант", "Δ log loss, ·10⁻³", "Δ девиация голов", "ROI, %", "ΔROI, п.п.", "Тот же Main"]]
    for r in hist.itertuples():
        rows.append([r.variant, f"{r.d_log_loss * 1000:+.2f} [{r.d_ll_lo * 1000:+.2f}; {r.d_ll_hi * 1000:+.2f}]",
                     f"{r.d_goal_dev * 1000:+.2f}", f"{r.roi * 100:+.2f}", f"{r.d_roi * 100:+.2f} [{r.d_roi_lo * 100:+.2f}; {r.d_roi_hi * 100:+.2f}]",
                     K.share(r.same_main, 0)])
    rep.add(H2("7.8 Смягчения: выбор на 2024/25 + 2025/26"),
            P(f"База истории: log loss {hb['log_loss']:.4f}, ROI Main {pp(hb['roi'])} % на {hb['bets']} ставках. "
              f"Выбрано: mkt_season w = {sel['mkt_season']}, mkt_hist w = {sel['mkt_hist']}, prior w = {sel['prior']}; "
              f"средний вес mkt_both на истории {sel['uniform_w']:.3f} (вес контроля mkt_uniform)."),
            table(rows, widths=W(3.6, 3.6, 2.2, 1.4, 3.6, 1.6), font=7.2))
    for u, title in (("close", "close (основной)"), ("t09", "t09 (проверка устойчивости)")):
        rows = [["Вариант", "ROI, %", "Ст.", "Hit", "ΔROI к B, п.п.", "p Холма", "Δ log loss, ·10⁻³", "Разреж. ΔROI"]]
        for k, v in V.items():
            e = v["universes"][u]
            rows.append([k + (" ✓" if v.get("chosen") else ""), roi_txt(e["roi_ci"]), str(e["summary"]["bets"]),
                         K.share(e["summary"]["hit_rate"]), ci_pp(e["vs_b"]["roi_diff"]), pval(e.get("holm_p")),
                         ci_k(e["d_log_loss"], d=2), pp(e["lowdata"]["vs_b"]["roi_diff"]["diff"])])
        bb = base[u]["summary"]
        rows.append(["Baseline B", roi_txt(base[u]["roi_ci"]), str(bb["bets"]), K.share(bb["hit_rate"]), "0", "—", "0", "0"])
        rep.add(H2(f"7.{9 if u == 'close' else 10} Смягчения на 2026/27 — {title}"),
                table(rows, widths=W(2.9, 2.7, 0.9, 1.2, 3.5, 1.1, 3.4, 1.3), font=6.6))
    ba = base["baseline_a"]
    rep.add(P(f"✓ — вариант с весом, выбранным на истории (или без параметра). Baseline A (история сайта): ROI "
              f"{roi_txt(ba['roi_ci'])} %, {ba['summary']['bets']} ставок, hit {K.share(ba['summary']['hit_rate'])}. "
              "«Разреж.: ΔROI» — разница ROI с Baseline B на матчах с max(s) ≥ 0.5 (только там варианты меняют вход).", "small"))
    # 8 figures
    rep.add(H1("8. Графики"),
            figure(FIG / "season_progress.png", "Рис. 1. Слева: excess log loss (модель − рынок) по числу сыгранных "
                   "матчей сезона; справа: ROI Main Baseline B в 2026/27 (95 % ДИ)."),
            figure(FIG / "club_status.png", "Рис. 2. Excess log loss по статусу клубов и началу сезона: 2026/27 и сезоны выбора."),
            figure(FIG / "selection_history.png", "Рис. 3. Выбор веса на 2024/25 + 2025/26: изменение log loss 1X2.", width=12 * K.cm),
            figure(FIG / "mitigations_2026.png", "Рис. 4. Смягчения на 2026/27: ΔROI Main к Baseline B (оба универсума) и Δ log loss."),
            figure(FIG / "feature_groups.png", "Рис. 5. Перестановочная важность групп признаков в двух сезонах."))
    # 9 statistics
    pc = prim["close"]
    mon = pd.DataFrame(pc["monthly"])
    rows = [["Месяц", "Ставки", "ROI mkt_both, %", "ROI B, %", "Hit", "Hit B"]]
    for r in mon.itertuples():
        rows.append([r.month + (" *" if r.small else ""), str(r.bets), pp(r.roi), pp(r.roi_b), K.share(r.hit), K.share(r.hit_b)])
    lgd = pd.DataFrame(pc["league"]).sort_values("d_roi")
    rep.add(H1("9. Статистический анализ"),
            P(f"Основной вариант {mit['primary']} (объявлен заранее). close: ΔROI {ci_pp(pc['vs_b']['roi_diff'])} п.п., "
              f"p = {pval(pc['vs_b']['roi_diff']['p'])}, Холм {pval(pc.get('holm_p'))}; Main совпадает с Baseline B в "
              f"{K.share(pc['vs_b']['same_main_share'], 1)} матчей. t09: ΔROI {ci_pp(prim['t09']['vs_b']['roi_diff'])} п.п., "
              f"Холм {pval(prim['t09'].get('holm_p'))}. Log loss 1X2: {ci_k(pc['d_log_loss'], d=2)}·10⁻³ на всех матчах, "
              f"{ci_k(pc['d_log_loss_lowdata'], d=2)}·10⁻³ на разреженных."),
            table(rows, widths=W(3.0, 2.0, 3.2, 2.6, 2.0, 2.0), font=8),
            P(f"Лиги с наибольшим ухудшением / улучшением ΔROI (close): "
              + "; ".join(f"{r.league} {pp(r.d_roi)}" + (" *" if r.small else "") for r in lgd.head(3).itertuples()) + " / "
              + "; ".join(f"{r.league} {pp(r.d_roi)}" + (" *" if r.small else "") for r in lgd.tail(3).itertuples())
              + f". Без 5 крупнейших выигрышей ROI {pp(pc['top_wins'].get('roi_without_top5'))} % (с ними "
              f"{pp(pc['top_wins'].get('roi'))} %). При ценах ×0.98 / ×0.95: {pp(pc['odds_sensitivity'].get('0.98'))} / "
              f"{pp(pc['odds_sensitivity'].get('0.95'))} %. Всего вариантов: {mit['n_variants_tried']} на истории, "
              f"{mit['n_variants_on_2026']} на 2026/27; Холм — по {mit['n_variants_on_2026']} вариантам 2026/27 в каждом универсуме."),
            P("2026/27 не является независимой проверкой: сезон изучался в v6.1/v6.2. Веса выбраны на 2024/25 + 2025/26, "
              "поэтому результат 2026/27 — хронологическая проверка, а не доказательство.", "small"))
    # 10 conclusions
    def worse_than(a_, b_):
        """True when interval a lies entirely above interval b (a significantly larger)."""
        return a_ and b_ and a_["ci95"][0] is not None and b_["ci95"][1] is not None and a_["ci95"][0] > b_["ci95"][1]

    e_c, l_c = nb_c.get("0–2", {}).get("excess_close"), nb_c.get("10+", {}).get("excess_close")
    e_h, l_h = nb_h.get("0–2", {}).get("excess_fd"), nb_h.get("10+", {}).get("excess_fd")
    h1_hist = "подтверждается" if (e_h and l_h and e_h["mean"] > l_h["mean"] and
                                   (nb_h["0–2"].get("vs_reference") or {}).get("excess_ll_diff", {}).get("ci95", [None, None])[0] is not None
                                   and nb_h["0–2"]["vs_reference"]["excess_ll_diff"]["ci95"][0] > 0) else "не подтверждается"
    h1_cur = "подтверждается" if (e_c and l_c and e_c["mean"] > l_c["mean"] and
                                  nb_c["0–2"]["vs_reference"]["excess_ll_diff"]["ci95"][0] > 0) else "не подтверждается"
    out_e = cur["club_early"].get("outside | 0–5", {})
    out_h = hst["club_early"].get("outside | 0–5", {})
    sb_c, sb_h = cur["stats_bin"], hst["stats_bin"]
    d_stat_h = sb_h["partial"]["vs_reference"]["excess_ll_diff"]
    d_stat_c = sb_c["partial"]["vs_reference"]["excess_ll_diff"]
    pa = st["price_availability"]
    mis = idn["misidentified"]
    rec = idn["misidentified + mkt_both"]
    pv, pr = V[mit["primary"]]["universes"], mit["primary"]
    stab_txt = "; ".join(f"{t}: ρ = {v['spearman']:.2f}, смен знака {v['sign_flip']} из {v['groups']}"
                         for t, v in grp["stability"].items()) or "не рассчитана"
    concl = ([f"**H1 (начало сезона).** На истории {h1_hist}: excess при n_min 0–2 {ci_k(e_h)}·10⁻³ против {ci_k(l_h)} "
                     f"при n_min ≥ 10. В 2026/27 {h1_cur}: {ci_k(e_c)} против {ci_k(l_c)} — в текущем сезоне модель хуже "
                     "рынка скорее в поздних турах (модель обучена до 31.05.2026 и не переобучалась). ROI Main по стратам "
                     "различается в пределах интервалов.",
                     f"**H2 (новички и фантомы).** Клуб outside в первых матчах: на истории excess {ci_k(out_h.get('excess_fd'))}·10⁻³ "
                     f"и ROI {roi_txt(out_h.get('main_fd'))} % ({out_h.get('matches', 0)} матчей); в 2026/27 — "
                     f"{ci_k(out_e.get('excess_close'))}·10⁻³ ({out_e.get('matches', 0)} матчей, интервал широкий). "
                     f"Фантом (один клуб матча без истории) ухудшает log loss на {ci_k(mis['close']['d_log_loss'])}·10⁻³ и "
                     f"меняет Main в {K.share(1 - mis['close']['vs_b']['same_main_share'], 0)} матчей; ΔROI close "
                     f"{ci_pp(mis['close']['vs_b']['roi_diff'])}, t09 {ci_pp(mis['t09']['vs_b']['roi_diff'])} п.п. "
                     f"Смягчение mkt_both возвращает log loss к {ci_k(rec['close']['d_log_loss'])}·10⁻³ относительно "
                     "production без ошибки (рынок в таких матчах заменяет модель), но ROI фантомных матчей не "
                     f"восстанавливает (ΔROI close {ci_pp(rec['close']['vs_b']['roi_diff'])} п.п.).",
                     f"**H3 (полнота статистики).** На истории матчи с пропусками ударов / угловых хуже: разница excess "
                     f"{ci_k(d_stat_h)}·10⁻³; в 2026/27 {ci_k(d_stat_c)}·10⁻³. Пропуски почти совпадают с началом сезона и "
                     f"новичками; ранговая корреляция доли пропусков с excess "
                     f"{st['spearman']['dq_vs_excess_close']:+.3f} (2026/27) и {st['spearman']['dq_vs_excess_fd_hist']:+.3f} (история).",
                     f"**H4 (лиги).** Excess различается по лигам в широких интервалах; цены OddsPapi на закрытии есть у "
                     f"{K.share(idn['audit']['oddspapi_closing_share'])} сыгранных матчей — покрытие цен лиг практически полное.",
                     f"**H5 (цены в T).** {pa['t09_main_unpriced']['matches']} матчей, у Main которых в T нет цены (почти "
                     f"все — тоталы), при цене закрытия дают ROI {roi_txt(pa['t09_main_unpriced']['close'])} % против "
                     f"{roi_txt(pa['t09_main_priced']['close'])} % у остальных (интервалы перекрываются); прямое следствие "
                     "отсутствия цены — Main не становится ставкой в t09 (поэтому у Baseline B t09 меньше ставок).",
                     f"**H6 (группы признаков).** Стабильность важности между 2025/26 и 2026/27: {stab_txt}.",
                     f"**H7 (смягчения).** Основной вариант {pr}: log loss {ci_k(pv['close']['d_log_loss'], d=2)}·10⁻³ "
                     f"(на разреженных матчах {ci_k(pv['close']['d_log_loss_lowdata'], d=2)}), ΔROI close "
                     f"{ci_pp(pv['close']['vs_b']['roi_diff'])} п.п., t09 {ci_pp(pv['t09']['vs_b']['roi_diff'])} п.п. — "
                     f"решение {dec.get(pr)}. Контроль mkt_uniform: log loss {ci_k(V['mkt_uniform']['universes']['close']['d_log_loss'], d=2)}, "
                     f"ΔROI close {ci_pp(V['mkt_uniform']['universes']['close']['vs_b']['roi_diff'])} п.п. (p Холма "
                     f"{pval(V['mkt_uniform']['universes']['close'].get('holm_p'))}). Сжатие λ к приорам лиги ухудшает "
                     "девиацию голов; ранжирование без EV в разреженных матчах ROI не меняет."])
    rep.add(H1("10. Выводы"), bullets(concl))
    # 11 recommendations
    why_ru = {"not selected on 2024/25 + 2025/26": "не выбран на 2024/25 + 2025/26",
              "ROI better in both universes (Holm), metrics not worse": "ROI лучше в обоих универсумах (Холм)",
              "a model metric significantly worse": "метрика модели значимо хуже",
              "metric better / ROI not lower; not significant for ROI": "метрика лучше, ROI не ниже; ROI незначим",
              "no improvement": "улучшения нет"}
    why = {k: why_ru.get(v, v) for k, v in mit.get("decision_reasons", {}).items()}
    rows = [["Вариант", "Решение", "Обоснование"]]
    for k, d in dec.items():
        e = V[k]["universes"]
        rows.append([k, d, f"{why.get(k, '')}: ΔROI close {pp(e['close']['vs_b']['roi_diff']['diff'])} / t09 "
                           f"{pp(e['t09']['vs_b']['roi_diff']['diff'])} п.п.; Δ log loss {ci_k(e['close']['d_log_loss'], d=2)}; "
                           f"Δ дев. голов {ci_k(e['close']['d_goal_dev'], d=2)}"])
    rep.add(H1("11. Рекомендации (A / B / C)"), table(rows, widths=W(2.9, 1.7, 12.4), font=7.0),
            P("Правило (зафиксировано до просмотра 2026/27): A или B возможны только для вариантов, чей параметр выбран на "
              "2024/25 + 2025/26; A — ΔROI > 0 в обоих универсумах, p Холма < 0.05 (close), ни одна метрика модели не "
              "хуже; B — метрика модели значимо лучше без падения ROI в обоих универсумах (или ΔROI > 0 в обоих) и ни "
              "одна метрика не хуже; C — иначе. Кандидаты B записаны в `results/preregistration.json` для "
              "форвард-проверки. Production не меняется.", "small"),
            note("Предложение для production (только после форвард-проверки, не внедрено): в `src/prediction/engine.py` "
                 "перед построением рынков смешивать вероятности 1X2 модели с ценой без маржи, ω = max(w₁·s_season, "
                 f"w₂·s_hist), w₁ = {sel['mkt_season']}, w₂ = {sel['mkt_hist']}; параметры — в `src/config.py`. Один Main на "
                 "матч сохраняется, матчи не исключаются. Отдельно: ошибки идентификации дороги (см. 7.6) — проверка "
                 "новых названий клубов перед расчётом (v6.2 D1) остаётся обязательной."))
    # 12 limitations
    rep.add(H1("12. Ограничения"),
            bullets(["2026/27 — уже изученный сезон; выводы о ROI — хронологическая проверка, не доказательство.",
                     "История цен 2024/25–2025/26 — только Bet365 Football-Data (1X2, тотал 2.5, одна фора), поэтому ROI "
                     "истории не сравним напрямую с 2026/27, где цены есть почти у всех рынков.",
                     "Фантом смоделирован обнулением признаков (как v6.2), а не повторной сборкой признаков с опечаткой; "
                     "реальная частота ошибок идентификации в 2026/27 — несколько матчей (v6.2), поэтому эффект в отчёте — "
                     "оценка «на матч с ошибкой», а не вклад в ROI сезона.",
                     "Цены на момент T известны лишь частично (архив OddsPapi хранит в основном последнюю котировку).",
                     "Малые страты (outside в начале сезона, отдельные лиги) имеют широкие интервалы и помечены *.",
                     f"Лучший вес mkt_season на истории — край заранее объявленной сетки (w = {sel['mkt_season']}); больший "
                     "вес (полная замена модели рынком в первых турах) не проверялся.",
                     "Контроль mkt_uniform — это общее смешивание модели с рынком для всех матчей; эта тема — предмет "
                     "исследования 3 (Market-aware Main), здесь он нужен только чтобы отделить эффект адресности.",
                     f"Самопроверка: вероятности 1X2 2025/26 из oos.py и конфигурации xgb:prod исследования 8 (та же "
                     f"модель, то же обучение, независимый запуск) — {chk.get('matches', '—')} матчей, максимальное "
                     f"расхождение {chk.get('max_abs_diff_p_home', '—')}."]))
    rep.add(H1("Как воспроизвести"),
            K.reproduce(["experiments/v6_3/heavy.sh 3600 experiments/v6_3/quality/logs/oos.log \\",
                         "    .venv/bin/python -m experiments.v6_3.quality.oos      # переобучение 2024/25, 2025/26",
                         ".venv/bin/python -m experiments.v6_3.quality.run          # страты, смягчения, идентификация",
                         ".venv/bin/python -m experiments.v6_3.quality.build_pdf    # этот отчёт и findings.md"]),
            P("Файлы: `results/strata.json`, `selection_history.csv`, `mitigations_2026.json`, `identification.json`, "
              "`feature_groups_*.csv`, `per_match_*.csv` (поля §5), `coverage.csv`, `match_strata.csv`, "
              "`preregistration.json`, `run_info.json`; рисунки — `figures/`.", "small"))
    rep.build()
    print("pages", rep.check(scale=1.0))
    findings(st, sel, mit, idn, grp, hist, concl)


def findings(st, sel, mit, idn, grp, hist, concl) -> None:
    V, dec = mit["variants"], mit["decisions"]
    nb = st["current"]["n_bucket"]
    lines = ["# FootPredict v6.3 — исследование 6: качество и полнота данных — выводы", "",
             "Все числа — из `results/` (пишет `build_pdf.py`). PDF: `reports/v6_3/Quality_Coverage_Research.pdf`.", "",
             "## Дефекты", "",
             f"- Фантомные матчи v6.2 в данных 2026/27: {idn['audit']['v62_phantoms_present']}; подозрительных пар названий: "
             f"{len(idn['audit']['similar_name_pairs'])}; клубов с аномально малым числом матчей: "
             f"{len(idn['audit']['clubs_with_few_matches'])}.", "",
             "## Главное", ""] + [f"- {c}" for c in concl] + ["",
             "## Страты по числу сыгранных матчей (2026/27, Baseline B, рынок closing)", ""]
    for k in ("0–2", "3–5", "6–9", "10+"):
        g = nb.get(k)
        if g:
            lines.append(f"- n_min {k}: {g['matches']} матчей, модель − рынок {ci_k(g['excess_close'])}·10⁻³, ROI close "
                         f"{roi_txt(g['main_close'])} %, t09 {roi_txt(g['main_t09'])} %.")
    m = idn["misidentified"]["close"]
    lines += [f"- Фантом (один клуб матча без истории): Δ log loss {ci_k(m['d_log_loss'])}·10⁻³, ΔROI close "
              f"{ci_pp(m['vs_b']['roi_diff'])} п.п.; со смягчением mkt_both: Δ log loss "
              f"{ci_k(idn['misidentified + mkt_both']['close']['d_log_loss'])}·10⁻³.",
              f"- Выбор на 2024/25 + 2025/26: mkt_season w = {sel['mkt_season']}, mkt_hist w = {sel['mkt_hist']}, "
              f"prior w = {sel['prior']}.", "", "## Смягчения на 2026/27 и решения", "",
              "| Вариант | ΔROI close, п.п. | p Холма | ΔROI t09, п.п. | Δ log loss ·10⁻³ | Решение |", "|---|---|---|---|---|---|"]
    for k, v in V.items():
        c, t = v["universes"]["close"], v["universes"]["t09"]
        lines.append(f"| {k} | {ci_pp(c['vs_b']['roi_diff'])} | {pval(c.get('holm_p'))} | {ci_pp(t['vs_b']['roi_diff'])} | "
                     f"{ci_k(c['d_log_loss'], d=2)} | {dec[k]} |")
    lines += ["", "## Что было бы изменением production (не внесено)", "",
              "- Вариант B (если подтвердится форвард-проверкой): в `src/prediction/engine.py` перед построением рынков "
              "смешать p 1X2 с ценой без маржи с весом ω = max(w₁·s_season, w₂·s_hist) "
              f"(w₁ = {sel['mkt_season']}, w₂ = {sel['mkt_hist']}; s_season = clip((10 − min(h_season_games, a_season_games))/10, 0, 1), "
              "s_hist = clip((10 − min(h_form10_n, a_form10_n))/10, 0, 1)), параметры — в `src/config.py`.",
              "- Один Main на матч сохраняется; матчи не исключаются.", "",
              "Предрегистрация: `results/preregistration.json`.", "",
              "## Ограничения", "", "- 2026/27 — уже изученный сезон; история цен — только Bet365 Football-Data.",
              "- Фантом смоделирован обнулением признаков.", ""]
    (C.V63 / "quality" / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
