"""Conclusions, recommendations, limitations and A / B / C decisions of the meta-selection study (from results)."""
from __future__ import annotations

import numpy as np

from .figures import NAME_RU


def _pp(v, d=1, signed=True):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}"


def _ci(x: dict, d=1) -> str:
    return f"{_pp(x['diff'], d)} [{_pp(x['ci95'][0], d)}; {_pp(x['ci95'][1], d)}]"


def _pv(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def decision_of(r: dict, n: str) -> str:
    a, b = r["close"][n]["vs_b"]["roi_diff"], r["t09"][n]["vs_b"]["roi_diff"]
    h = r["close"]["holm"].get(n)
    if h is not None and h < 0.05 and a["diff"] > 0 and b["diff"] > 0:
        return "A"
    if a["diff"] > 0 and b["diff"] > 0:
        return "B"
    return "C"


def decisions(r: dict) -> dict:
    out = {}
    for n in r["meta"] + r["simple"]:
        row = {"decision": decision_of(r, n)}
        for u in ("close", "t09"):
            x = r[u][n]
            row[u] = {"bets": x["summary"]["bets"], "roi": x["summary"]["roi"], "hit": x["summary"]["hit_rate"],
                      "avg_odds": x["summary"]["avg_odds"], "d_roi": x["vs_b"]["roi_diff"]["diff"],
                      "d_roi_ci": x["vs_b"]["roi_diff"]["ci95"], "p": x["vs_b"]["roi_diff"]["p"],
                      "holm_p": r[u]["holm"].get(n), "same_main": x["vs_b"]["same_main_share"]}
        out[n] = row
    return out


def conclusions(r: dict) -> list[str]:
    rc, r9, ch = r["close"], r["t09"], r["checks"]["close"]
    prim = r["primary"]
    dd = decisions(r)
    names = r["meta"] + r["simple"]
    out = []
    best = max(names, key=lambda n: rc[n]["vs_b"]["roi_diff"]["diff"])
    out.append(
        f"H1. Основная метамодель ({NAME_RU[prim]}): close ROI {_pp(rc[prim]['summary']['roi'])} % против "
        f"{_pp(rc['baseline_b']['summary']['roi'])} % у Baseline B, ΔROI {_ci(rc[prim]['vs_b']['roi_diff'])} п.п. "
        f"(p Холма {_pv(rc['holm'].get(prim))}); t09 ΔROI {_ci(r9[prim]['vs_b']['roi_diff'])} п.п. Лучший по close "
        f"вариант — {NAME_RU[best]}: ΔROI {_ci(rc[best]['vs_b']['roi_diff'])} п.п., t09 "
        f"{_ci(r9[best]['vs_b']['roi_diff'])} п.п., p Холма {_pv(rc['holm'].get(best))}.")
    if "rule_priced" in rc:
        out.append(
            f"Контроль «{NAME_RU['rule_priced']}» (то же правило продакшна, но только среди кандидатов с ценой): close "
            f"ΔROI {_ci(rc['rule_priced']['vs_b']['roi_diff'])} п.п., t09 {_ci(r9['rule_priced']['vs_b']['roi_diff'])} п.п. "
            f"(ставок {r9['rule_priced']['summary']['bets']} против {r9['baseline_b']['summary']['bets']}). В t09 метамодель "
            "сравнивается с Baseline B на другом наборе ставок: относительно этого контроля разница основной метамодели "
            f"{_pp(r9[prim]['summary']['roi'] - r9['rule_priced']['summary']['roi'])} п.п. ROI.")
    pq = rc["prob_quality"]
    out.append(
        f"H2. Log loss пула 2026/27: модель продакшна {pq['raw']['log_loss']:.4f}, откалиброванная "
        f"{pq['calibrated']['log_loss']:.4f}, рынок без маржи {pq['market_q']['log_loss']:.4f}, метамодель LR "
        f"{pq['p_lr']['log_loss']:.4f}, GBM {pq['p_gbm']['log_loss']:.4f}. "
        + ("Метамодель предсказывает исход лучше модели продакшна — в основном за счёт цены (наибольший коэффициент "
           "логрегрессии — у logit вероятности рынка)" if pq["p_lr"]["log_loss"] < pq["raw"]["log_loss"] else
           "Метамодель не предсказывает исход лучше модели продакшна")
        + (", но хуже самого рынка без маржи." if pq["p_lr"]["log_loss"] > pq["market_q"]["log_loss"] else "."))
    perm = ch["permutation"]
    rows = []
    for n in ("meta_lr_ev", "meta_gbm_ev", "meta_gbm_ret"):
        real = rc[n]["summary"]["roi"]
        share = float(np.mean(np.array(perm[n]["values"]) >= real))
        rows.append(f"{NAME_RU[n]}: walk-forward {_pp(real)} %, на перемешанных целях {_pp(perm[n]['mean'])} % "
                    f"(доля ≥ реального {share:.2f}), в выборке обучения {_pp(ch['in_sample'][n])} %")
    rnd = rc["random_pick"]
    out.append("H3 (переобучение). " + "; ".join(rows) + f". Случайный выбор из пула: {_pp(rnd['mean'])} % "
               f"(5–95 %: {_pp(rnd['q05'])}; {_pp(rnd['q95'])}). Разница между ROI в выборке обучения и walk-forward "
               "— мера оптимизма: именно столько «находит» модель, видящая исходы тех же матчей.")
    lc = {}
    for x in ch["learning_curve"]:
        lc.setdefault(x["frac"], []).append(x["meta_lr_ev_roi"])
    fr = sorted(lc)
    out.append(f"Кривая обучения LR: ROI при {fr[0] * 100:.0f} % обучающих матчей {_pp(np.mean(lc[fr[0]]))} %, при "
               f"100 % {_pp(np.mean(lc[fr[-1]]))} %; обучение только на 2025/26: LR {_pp(ch['history_only']['meta_lr_ev'])} %, "
               f"GBM {_pp(ch['history_only']['meta_gbm_ev'])} %.")
    simple = ", ".join(f"{NAME_RU[n]} {_pp(rc[n]['summary']['roi'])} % (t09 {_pp(r9[n]['summary']['roi'])} %)"
                       for n in r["simple"])
    out.append(f"H4. Простые правила на том же пуле (close): {simple}; Baseline B {_pp(rc['baseline_b']['summary']['roi'])} % "
               f"(t09 {_pp(r9['baseline_b']['summary']['roi'])} %).")
    counts = {k: sum(1 for v in dd.values() if v["decision"] == k) for k in "ABC"}
    out.append(f"Решения по {len(dd)} вариантам: A — {counts['A']}, B — {counts['B']}, C — {counts['C']}. "
               "Сезон 2026/27 уже изучался; результат — хронологическая проверка на 4 месяцах, а не независимое "
               "подтверждение.")
    return out


def recommendations(r: dict, pre) -> list[str]:
    dd = decisions(r)
    out = []
    a = [n for n, v in dd.items() if v["decision"] == "A"]
    b = [n for n, v in dd.items() if v["decision"] == "B"]
    c = [n for n, v in dd.items() if v["decision"] == "C"]
    out.append("A (внедрить сейчас): " + ("; ".join(NAME_RU[n] for n in a) if a else
                                         "нет — ни один вариант не прошёл поправку Холма в close") + ".")
    if b:
        ids = ", ".join(e["id"] for e in (pre or {}).get("entries", [])) or "—"
        out.append("B (форвард-тест, продакшн не меняется): " + "; ".join(
            f"{NAME_RU[n]} (ΔROI close {_pp(dd[n]['close']['d_roi'])}, t09 {_pp(dd[n]['t09']['d_roi'])} п.п.)"
            for n in b) + f". Предрегистрация: results/preregistration.json ({ids}).")
    else:
        out.append("B: нет (ни один вариант не улучшил ROI одновременно в close и t09).")
    out.append("C (не внедрять): " + ("; ".join(NAME_RU[n] for n in c) if c else "нет") + ".")
    out.append("Если вариант B подтвердится на форвард-выборке, изменение продакшна — отдельный шаг после "
               "`src/markets/recommendation.main_prediction`: метамодель ранжирует кандидатов с реальной ценой, "
               "переобучается раз в месяц на рассчитанных Main и кандидатах прошлых матчей; без цены — правило "
               "продакшна. Сейчас продакшн не меняется.")
    return out


def limitations(r: dict) -> list[str]:
    ii = r["close"]["info"]
    return [
        f"Пул 2025/26 ({ii['history_matches']} матчей) содержит цены только 1X2, тотала 2.5 и одной азиатской линии; "
        "двойной шанс, обе забьют и угловые метамодель видит только по месяцам 2026/27 (в августе — не видит).",
        "2026/27 — 4 месяца (август–ноябрь), около 2 250 сыгранных матчей; разница ROI в 1–2 п.п. не отличима "
        "от шума, месячные и лиговые оценки малы.",
        "Метамодель выбирает только среди кандидатов с реальной ценой; если у матча нет такой строки, Main "
        "берётся у Baseline B (fallback), поэтому число ставок может отличаться от Baseline B (особенно в t09, где "
        "у четверти Main продакшна нет цены).",
        "Признаки калиброванной вероятности для строк 2025/26 получены без своего месяца (leave-one-month-out) — "
        "это исторические данные, но распределение признака чуть отличается от 2026/27 (там карта обучена только "
        "на прошлом).",
        f"Основная метамодель часто выбирает целые азиатские линии: возвратов (PUSH) у её Main в close "
        f"{next(x['excluded_push_void'] for x in r['close']['coverage'] if x['method'] == r['primary'])}; по правилам "
        "продакшна возврат — не ставка, поэтому число ставок меньше, чем у Baseline B.",
        "Гиперпараметры заданы заранее и не подбирались; другая настройка могла бы дать другой результат, но "
        "подбор на 2026/27 сделал бы оценку смещённой.",
        "Сезон 2026/27 уже изучался в v6.1/v6.2 и других исследованиях v6.3; выводы — хронологическая проверка."]
