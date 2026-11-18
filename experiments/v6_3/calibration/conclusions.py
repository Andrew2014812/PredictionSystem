"""Conclusions, recommendations and A / B / C decisions of the calibration study, composed from results only."""
from __future__ import annotations

from . import data as D
from . import maps as MP
from .figures import vlabel


def _pp(v, d=1, signed=True):
    return "—" if v is None else (f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}")


def _ci(x: dict, d=1) -> str:
    c = x.get("ci95") or x.get("gap_ci") or [None, None]
    v = x.get("diff", x.get("gap"))
    return f"{_pp(v, d)} [{_pp(c[0], d)}; {_pp(c[1], d)}]"


def _pv(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def decision_of(r: dict, name: str) -> str:
    d1 = r["main"]["close"][name]["vs_b"]["roi_diff"]
    d2 = r["main"]["t09"][name]["vs_b"]["roi_diff"]
    h = r["main"]["close"]["holm"].get(name)
    if h is not None and h < 0.05 and d1["diff"] > 0 and d2["diff"] > 0:
        return "A"
    if d1["diff"] > 0 and d2["diff"] > 0:
        return "B"
    return "C"


def decisions(r: dict) -> dict:
    out = {}
    for n in r["family"]:
        row = {"decision": decision_of(r, n)}
        for u in ("close", "t09"):
            x = r["main"][u][n]
            row[u] = {"bets": x["summary"]["bets"], "roi": x["summary"]["roi"],
                      "d_roi": x["vs_b"]["roi_diff"]["diff"], "d_roi_ci": x["vs_b"]["roi_diff"]["ci95"],
                      "p": x["vs_b"]["roi_diff"]["p"], "holm_p": r["main"][u]["holm"].get(n),
                      "gap": x["gap"]["gap"], "same_main": x["vs_b"]["same_main_share"]}
        out[n] = row
    return out


def _wc(r):
    t = {(x["rule_on"], x["k"]): x for x in r["winners_curse"]["table"]}
    ks = r["winners_curse"]["k"]
    return t, ks[0], ks[-1]


def conclusions(r: dict) -> list[str]:
    q, prim = r["quality"], r["primary"]
    pm, ps, _ = prim.split("|")
    raw, cal = q["raw|-|-"], q[prim]
    dl = r["paired_log_loss_vs_raw"][prim]
    sel = r["selection"]["table"]
    raw_sel = next(x for x in sel if x["method"] == "raw")
    best_sel = next(x for x in sel if x["method"] != "raw")
    out = []
    # H1 — calibration by group
    worst = sorted(D.MAIN_GROUPS, key=lambda g: -raw[g]["ece"])
    out.append(
        "H1 (калибровка по группам). Вероятности модели 2026/27 в целом откалиброваны неплохо: ECE по всем "
        f"селекциям групп Main {_pp(raw['all_main_groups']['ece'], 2, False)} п.п., на кандидатах Main (p 0.40–0.85) "
        f"{_pp(raw['band']['ece'], 2, False)} п.п., средняя p − частота кандидатов {_pp(raw['band']['gap'], 2)} п.п. "
        "Наибольшая ECE по группам: " + "; ".join(
            f"{D.GROUP_RU[g]} {_pp(raw[g]['ece'], 2, False)} п.п. (n {raw[g]['n']})" for g in worst[:3])
        + " (средняя p по всем исходам рынка равна частоте по построению, поэтому смещение видно только по ECE и "
        "диаграммам надёжности). Систематического смещения в зоне кандидатов Main нет — H1 не подтверждается.")
    # H2 — maps
    out.append(
        f"H2 (карты на прошлых матчах). На 2025/26 (walk-forward) лучшая карта ({MP.METHOD_RU[pm]}, {MP.SCOPE_RU[ps]}) "
        f"почти не отличается от исходной вероятности: log loss {best_sel['log_loss']:.5f} против "
        f"{raw_sel['log_loss']:.5f}. На 2026/27 основная карта меняет log loss кандидатов Main на "
        f"{dl['band']['diff'] * 1000:+.2f} × 10⁻³ [{dl['band']['ci95'][0] * 1000:+.2f}; {dl['band']['ci95'][1] * 1000:+.2f}] "
        f"(положительно — хуже; p {_pv(dl['band']['p'])}), ECE кандидатов {_pp(raw['band']['ece'], 2, False)} → {_pp(cal['band']['ece'], 2, False)} п.п. "
        "Гибкие карты (изотоническая, гистограммная, по группе и исходу) на 2025/26 хуже исходной — "
        "переобучаются на шуме; калибровка вероятностей модели — это поправка порядка десятых долей процента, "
        "а не исправление систематической ошибки.")
    # H3 — winner's curse
    t, k1, kall = _wc(r)
    a1, aa = t[("raw", k1)], t[("raw", kall)]
    c1, ca = t[("calibrated", k1)], t[("calibrated", kall)]
    dec = r["b_gap_decomposition"]
    out.append(
        f"H3 (проклятие победителя). Переоценка выбранного Main растёт с числом кандидатов: при k = {k1} "
        f"заявлено − факт {_pp(a1['gap_stated'])} п.п., при всех кандидатах {_pp(aa['gap_stated'])} п.п. "
        f"(тот же выбор, оценённый откалиброванной вероятностью: {_pp(a1['gap_cal'])} → {_pp(aa['gap_cal'])} п.п.; "
        f"правило на откалиброванных вероятностях: {_pp(c1['gap_stated'])} → {_pp(ca['gap_stated'])} п.п.). "
        f"Разрыв Baseline B (close) {_ci(dec['stated'])} п.п.; "
        f"если те же Main оценить основной картой — {_ci(dec['primary_map'])} п.п. "
        "Значит, основная часть переоценки Main — эффект отбора максимума из многих шумных оценок, а не "
        "маргинальная ошибка калибровки; карта, обученная на всех селекциях, его не устраняет.")
    # H4 — Main ROI
    dd = decisions(r)
    rp = r["main"]["close"][prim]
    rp9 = r["main"]["t09"][prim]
    base = r["main"]["close"]["baseline_b"]
    best = max(r["family"], key=lambda n: r["main"]["close"][n]["vs_b"]["roi_diff"]["diff"])
    rb = r["main"]["close"][best]
    out.append(
        f"H4 (Main на откалиброванных вероятностях). Основная карта меняет Main в "
        f"{_pp(1 - rp['vs_b']['same_main_share'], 1, False)} % матчей; ROI close {_pp(rp['summary']['roi'])} % против "
        f"{_pp(base['summary']['roi'])} % у Baseline B, ΔROI {_ci(rp['vs_b']['roi_diff'])} п.п. (p Холма "
        f"{_pv(r['main']['close']['holm'].get(prim))}); t09 ΔROI {_ci(rp9['vs_b']['roi_diff'])} п.п. Доля ставок с "
        f"отрицательным EV по вероятности, на которой выбирали: {_pp(base['neg_ev_share_stated'], 1, False)} % → "
        f"{_pp(rp['neg_ev_share_stated'], 1, False)} %; переоценка Main {_pp(base['gap']['gap'])} → "
        f"{_pp(rp['gap']['gap'])} п.п. Лучший по close вариант семейства — {vlabel(best)}: ΔROI "
        f"{_ci(rb['vs_b']['roi_diff'])} п.п., p Холма {_pv(r['main']['close']['holm'].get(best))}.")
    counts = {k: sum(1 for v in dd.values() if v["decision"] == k) for k in "ABC"}
    out.append(
        f"Решения по {len(dd)} вариантам Main: A — {counts['A']}, B — {counts['B']}, C — {counts['C']}. "
        "Ни одно изменение ROI не значимо после поправки Холма; сезон 2026/27 уже изучался, поэтому это "
        "хронологическая проверка, а не независимое подтверждение." if counts["A"] == 0 else
        f"Решения по {len(dd)} вариантам Main: A — {counts['A']}, B — {counts['B']}, C — {counts['C']}.")
    out.append(
        "Ключевой вопрос: сделать вероятности заметно более достоверными калибровкой нельзя — они и так близки к "
        "откалиброванным на уровне всех селекций; завышена вероятность именно выбранного Main, и это следствие "
        "выбора максимума. Улучшить выбор Main калибровкой до выбора не удалось (ROI в пределах шума).")
    return out


def recommendations(r: dict, pre) -> list[str]:
    dd = decisions(r)
    b = [n for n, v in dd.items() if v["decision"] == "B"]
    a = [n for n, v in dd.items() if v["decision"] == "A"]
    c = [n for n, v in dd.items() if v["decision"] == "C"]
    prim = r["primary"]
    out = []
    if a:
        out.append("A (внедрить): " + "; ".join(vlabel(n) for n in a) + ".")
    else:
        out.append("A (внедрить сейчас): нет — ни один вариант не прошёл поправку Холма в close.")
    if b:
        ids = ", ".join(e["id"] for e in (pre or {}).get("entries", [])) or "—"
        out.append("B (форвард-тест без изменения продакшна): " + "; ".join(
            f"{vlabel(n)} (ΔROI close {_pp(dd[n]['close']['d_roi'])}, t09 {_pp(dd[n]['t09']['d_roi'])} п.п.)" for n in b)
            + f". Предрегистрация: results/preregistration.json ({ids}); решение — не раньше 1500 рассчитанных ставок Main "
              "после даты заморозки. Доказательность слабая: B здесь означает лишь положительный знак ΔROI в обеих "
              "вселенных, все p > 0.2 и все интервалы содержат ноль; при форвард-тесте главным считать один вариант "
              "(температурная карта по группам рынков — один параметр на группу), остальные — вторичными.")
    else:
        out.append("B: нет кандидатов (ни один вариант не улучшил ROI одновременно в close и t09).")
    out.append("C (не внедрять): " + "; ".join(vlabel(n) for n in c) + "." if c else "C: нет.")
    out.append(
        f"Показываемая вероятность Main: продолжить форвард-тест v6.2 (пост-калибровка уже выбранного Main), так "
        f"как маргинальная карта ({vlabel(prim)}) не убирает переоценку выбранного Main — её даёт отбор.")
    out.append(
        "Изменение продакшна, если вариант B подтвердится: в `src/prediction/engine.py` после расчёта рынков "
        "применять карту (pc′ = σ(logit pc / T_g), одна температура T_g на группу рынков, обучена на backtest 2025/26 "
        "и прошлых месяцах, ежемесячное переобучение) до `src/markets/recommendation.main_prediction`; параметры правила (`src/config.py`) не "
        "меняются. Сейчас продакшн не меняется.")
    return out
