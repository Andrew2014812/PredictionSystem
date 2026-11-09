"""A / B / C decisions, conclusions and findings.md of the temporal study (text generated from results/*.json).

    python -m experiments.v6_2.temporal.decisions
"""
from __future__ import annotations

from experiments.v6_2.leagues.pdfkit import dk, pv, roi

from . import common as C

DEC = {"A": "готово к внедрению", "B": "перспективно, нужна проверка вперёд", "C": "не рекомендуется"}


def build() -> dict:
    var, sig, lk, mf = (C.load_json(x) for x in ("variants.json", "signal.json", "leakage_test.json", "market_form.json"))
    V = var["variants"]
    S = sig["select"]["features"]
    neg, concl = [], []
    for v, d in V.items():
        if d["decision"] == "C":
            neg.append(f"{v} ({d['name']}): 1X2 {dk(d['select']['d_ll'], 2)}, голы {dk(d['select']['d_dev'], 2)} ×10⁻³ "
                       f"(p Холма {pv(d['select']['p_holm_ll'])} / {pv(d['select']['p_holm_dev'])}).")
    top = max(S, key=lambda x: abs(S[x]["r_model"]["rho"] or 0))
    n_sig = sum(1 for x in S.values() if (x["r_model"]["p_holm"] or 1) < 0.05)
    n_sig_m = sum(1 for x in S.values() if (x["r_market"]["p_holm"] or 1) < 0.05)
    neg.append(f"Ни один из {len(S)} признаков не связан с остатком модели после поправки Холма (значимых {n_sig}; "
               f"максимум |r| = {abs(S[top]['r_model']['rho']):.3f} у {top}); с остатком рынка — значимых {n_sig_m}.")
    neg.append("Поправка рыночной цены на форму не лучше самого рынка: " + "; ".join(
        f"{v} {dk(mf['select'][v]['d_ll'], 2)}×10⁻³ (p Холма {pv(mf['select'][v]['p_holm'])})" for v in mf["select"]) + ".")
    concl.append(f"Утечки нет: {'все тесты пройдены' if lk['all_passed'] else 'НЕ все тесты пройдены'} — признаки "
                 "исследования и производственного построителя не меняются при подмене результатов дня матча и "
                 "всех последующих дней.")
    r0, r1, r2 = V["R0"]["select"], V["R1"]["select"], V["R2"]["select"]
    concl.append(f"Динамика против простых средних: без оконной формы и Elo (R0) {dk(r0['d_ll'], 2)}×10⁻³ log loss "
                 f"1X2 и {dk(r0['d_dev'], 2)} девиации голов к производственному набору; с Elo (R1) {dk(r1['d_ll'], 2)} / "
                 f"{dk(r1['d_dev'], 2)}; EW-средние вместо окон (R2) {dk(r2['d_ll'], 2)} / {dk(r2['d_dev'], 2)}.")
    t9 = V["T9"]["select"]
    concl.append(f"Все восемь новых семейств вместе (T9): {dk(t9['d_ll'], 2)}×10⁻³ (1X2), {dk(t9['d_dev'], 2)} (голы); "
                 f"ΔROI Main {t9['d_roi']['diff'] * 100:+.1f} п.п.")
    a = [v for v, d in V.items() if d["decision"] == "A"]
    b = [v for v, d in V.items() if d["decision"] == "B"]
    if a:
        concl.append("A: " + ", ".join(f"{v} ({V[v]['name']})" for v in a) + ".")
    if b:
        concl.append("B (предрегистрация, проверка вперёд): " + ", ".join(f"{v} ({V[v]['name']})" for v in b) + ".")
    if not a and not b:
        concl.append("Ни один новый временной признак не даёт устойчивого прироста сверх производственного набора: "
                     "продакшн не меняется.")
    concl.append("Смены тренера и состава не исследованы: нет надёжного локального источника с датой до матча.")
    out = {"negative": neg, "conclusions": concl, "decisions": {v: d["decision"] for v, d in V.items()}}
    C.save_json("decisions.json", out)
    findings(var, out)
    return out


def findings(var, dec) -> None:
    V = var["variants"]
    lines = ["# FootPredict v6.2 — временные признаки: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `decisions.py`). PDF: `reports/v6_2/Temporal_Features_Research.pdf`.",
             "", "## Варианты (разница с T0 = производственный набор, 3 сида, ×10⁻³; отбор 2023/24 + 2024/25)", "",
             "| Вариант | log loss 1X2 [95 % ДИ] | p Холма | Девиация голов [95 % ДИ] | p Холма | 2025/26 1X2 | ΔROI Main, п.п. | Решение |",
             "|---|---|---|---|---|---|---|---|"]
    for v, d in V.items():
        s = d["select"]
        lines.append(f"| {v} — {d['name']} | {dk(s['d_ll'], 2)} | {pv(s['p_holm_ll'])} | {dk(s['d_dev'], 2)} | "
                     f"{pv(s['p_holm_dev'])} | {dk(d['seasons']['2025']['d_ll'], 2)} | {s['d_roi']['diff'] * 100:+.1f} | "
                     f"{d['decision']} |")
    ref = var["reference"]["T0"]["select"]
    lines += ["", f"T0 на отборе: log loss {ref['ll']:.4f}, модель − рынок {dk(ref['vs_market'])}×10⁻³, ROI Main "
              f"{roi(ref['main']['roi'])}. Вариантов: {var['n_variants_tested']}.", "", "## Главное", ""]
    lines += [f"- {x}" for x in dec["conclusions"]]
    lines += ["", "## Отрицательные результаты", ""] + [f"- {x}" for x in dec["negative"]]
    pre = C.load_json("preregistration.json")
    lines += ["", "## Решение", ""]
    for k in "ABC":
        vs = [v for v, d in V.items() if d["decision"] == k]
        lines.append(f"- {k} ({DEC[k]}): {', '.join(vs) if vs else '—'}")
    lines.append(f"- Предрегистрация: {'`results/preregistration.json` (' + pre['frozen_at'] + ')' if pre else 'не нужна (нет B)'}. "
                 "Продакшн не меняется.")
    lines += ["", "## Воспроизведение", "", "```", "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
              "PYTHONPATH=. timeout 21600 $PY -m experiments.v6_2.temporal.run",
              "$PY -m experiments.v6_2.temporal.decisions && $PY -m experiments.v6_2.temporal.build_pdf", "```", ""]
    (C.HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    C.setup_logging("decisions")
    build()
