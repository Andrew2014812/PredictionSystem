"""A / B / C decisions, conclusions and findings.md of the league study (text generated from results/*.json).

    python -m experiments.v6_2.leagues.decisions
"""
from __future__ import annotations

from . import common as C
from .pdfkit import dk, pv, roi

DEC = {"A": "готово к внедрению", "B": "перспективно, нужна проверка вперёд", "C": "не рекомендуется"}


def build() -> dict:
    perf, var, tr, prof = (C.load_json(x) for x in ("perf.json", "variants.json", "transfer.json", "profiles.json"))
    V = var["variants"]
    sel = perf["select"]
    meta = sel["meta"]
    L = sel["leagues"]
    worse = sum(1 for r in L.values() if r["d"]["diff"] > 0)
    neg, concl = [], []
    for v, d in V.items():
        if d["decision"] == "C":
            m = "d_" + d["metric"]
            neg.append(f"{v} ({d['name']}): {dk(d['select'][m], 2)}×10⁻³ на объединённом отборе "
                       f"(2023/24 {dk(d['seasons']['2023'][m], 2)}, 2024/25 {dk(d['seasons']['2024'][m], 2)}), "
                       f"p Холма = {pv(d['select']['p_holm'])}.")
    ps = perf["persistence"]
    neg.append(f"Устойчивость «трудных» лиг не подтверждается: ранговая корреляция отставания между сезонами "
               f"{ps['2023-2024']['rho']:.2f} (2023/24↔2024/25), {ps['2024-2025']['rho']:.2f} (2024/25↔2025/26), "
               f"{ps['2023-2025']['rho']:.2f} (2023/24↔2025/26).")
    from scipy import stats
    n = {r["league"]: r["n"] for r in prof["folds"]["2024"]["profile"]}
    size = {}
    for v in ("C2", "C3", "M2", "M4a"):
        Lv = V[v]["select"]["leagues"]
        r = stats.spearmanr([n[x] for x in Lv], [Lv[x]["mean"] for x in Lv])
        size[v] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    neg.append("Связь вреда специализации с размером лиги (Спирмен ρ между числом матчей лиги в обучении и Δ лиги; "
               "< 0 — малым лигам хуже): " + ", ".join(f"{v} ρ = {x['rho']:.2f} (p = {pv(x['p'])})" for v, x in size.items()) + ".")
    ari = prof["ari"]
    neg.append(f"Группы лиг неустойчивы: ARI между соседними сезонами {min(ari.values()):.2f}–{max(ari.values()):.2f}, "
               f"силуэт ≤ {max(max(x['silhouette'].values()) for x in prof['folds'].values()):.2f}.")
    tier = perf["explain"]["tier"]
    concl.append(f"Модель уступает рынку в {worse} из {len(L)} лиг; среднее отставание "
                 f"{meta['mu'] * 1e3:+.1f}×10⁻³ log loss, межлиговое s.d. τ = {meta['tau'] * 1e3:.1f}×10⁻³ "
                 f"(I² = {meta['I2'] * 100:.0f} %). Есть лишь тенденция к большему отставанию в высших дивизионах, где рынок "
                 f"точнее (ρ с уровнем лиги {tier['rho']:.2f}, p = {pv(tier['p'])}, после Холма {pv(tier['p_holm'])} — "
                 "не значимо).")
    mr = sel["meta_main_roi"]
    concl.append(f"ROI и переоценка Main по лигам различаются не больше случайного (ROI: I² = {mr['I2'] * 100:.0f} %, "
                 f"p = {pv(mr['p_Q'])}); отбирать или исключать лиги по прошлому ROI нет оснований.")
    if tr:
        a, o = tr["pooled"]["all"]["pooled"], tr["pooled"]["own"]["pooled"]
        g = tr["pooled"]["group_vs_out"]
        concl.append(f"Перенос работает: модель, не видевшая лигу, хуже глобальной на {dk(a, 1)}×10⁻³, а модель только "
                     f"на своей лиге — на {dk(o, 1)}×10⁻³; своя группа против чужих групп {dk(g, 1)}×10⁻³. "
                     "Общая модель с данными всех лиг — лучший выбор для малых лиг.")
    def d_(v):
        return dk(V[v]["select"]["d_" + V[v]["metric"]], 2)
    sil = max(max(x["silhouette"].values()) for x in prof["folds"].values())
    hyp = [
        f"H1 подтверждается лишь частично: на объединённом отборе неоднородность есть (Q-тест p = {pv(meta['p_Q'])}, "
        f"I² = {meta['I2'] * 100:.0f} %), но по отдельным сезонам — нет, и порядок лиг не повторяется "
        f"(ρ 2024/25↔2025/26 = {ps['2024-2025']['rho']:.2f}).",
        f"H2 отвергается: калибровка по лиге без сжатия C2 {d_('C2')}, с EB-сжатием C3 {d_('C3')}, случайные эффекты "
        f"C3i {d_('C3i')}, иерархия C5 {d_('C5')} ×10⁻³ — все хуже исходных вероятностей; даже глобальная "
        f"перекалибровка C1 {d_('C1')}: модель уже откалибрована, и любая карта добавляет шум оценки.",
        f"H3 отвергается: кластерная структура слабая (силуэт ≤ {sil:.2f}, ARI {min(ari.values()):.2f}–"
        f"{max(ari.values()):.2f}); групповые варианты C4 {d_('C4')}, M3 {d_('M3')}, M5 {d_('M5')} ×10⁻³.",
        f"H4 отвергается: отдельная модель на лигу M2 {d_('M2')}, дообучение M4a {d_('M4a')}, M4b {d_('M4b')} ×10⁻³.",
    ]
    if tr:
        hyp.append(f"H5 подтверждается для переноса и не подтверждается для «похожести»: без собственной истории лиги "
                   f"потеря лишь {dk(tr['pooled']['all']['pooled'], 1)}×10⁻³, тогда как данные одной лиги дают "
                   f"{dk(tr['pooled']['own']['pooled'], 1)}; данные своей группы не лучше данных чужих групп "
                   f"({dk(tr['pooled']['group_vs_out'], 1)}).")
    concl = hyp + concl
    b = [v for v, d in V.items() if d["decision"] == "B"]
    a = [v for v, d in V.items() if d["decision"] == "A"]
    if a:
        concl.append("A: " + ", ".join(f"{v} ({V[v]['name']})" for v in a) + ".")
    if b:
        concl.append("B (нужна проверка вперёд, предрегистрация): " + ", ".join(f"{v} ({V[v]['name']})" for v in b) + ".")
    if not a and not b:
        concl.append("Ни одна специализация (калибровка по лигам/группам с EB-сжатием, групповые, отдельные, "
                     "дообученные модели, поправка голов) не улучшает глобальную модель устойчиво: продакшн "
                     "оставить как есть (одна глобальная модель с one-hot лиги).")
    out = {"negative": neg, "conclusions": concl, "decisions": {v: d["decision"] for v, d in V.items()}, "size_effect": size}
    C.save_json("decisions.json", out)
    findings(perf, var, tr, out)
    return out


def findings(perf, var, tr, dec) -> None:
    V = var["variants"]
    sel = perf["select"]
    lines = ["# FootPredict v6.2 — специализация по лигам: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `decisions.py`). PDF: `reports/v6_2/League_Specialization_Research.pdf`.",
             "", "## Модель против рынка по сезонам (log loss модель − Bet365 открытие, ×10⁻³)", "",
             "| Сезон | Матчей | Модель − рынок | τ | I² | p (Q) |", "|---|---|---|---|---|---|"]
    for s in C.EVAL:
        x = perf["seasons"][str(s)]
        lines.append(f"| {C.period_label(s)} | {x['n']} | {dk(x['d'])} | {x['meta']['tau'] * 1e3:.1f} | "
                     f"{x['meta']['I2'] * 100:.0f} % | {pv(x['meta']['p_Q'])} |")
    m = sel["meta"]
    lines.append(f"| 2023/24 + 2024/25 | {sel['n']} | {dk(sel['d'])} | {m['tau'] * 1e3:.1f} | {m['I2'] * 100:.0f} % | {pv(m['p_Q'])} |")
    lines += ["", "## Варианты (основная метрика, разница с эталоном ×10⁻³)", "",
              "| Вариант | Отбор [95 % ДИ] | p Холма | 2025/26 (инф.) | 2026/27 (тек.) | ΔROI Main, п.п. | Решение |",
              "|---|---|---|---|---|---|---|"]
    for v, d in V.items():
        k = "d_" + d["metric"]
        lines.append(f"| {v} — {d['name']} | {dk(d['select'][k], 2)} | {pv(d['select']['p_holm'])} | "
                     f"{dk(d['seasons']['2025'][k], 2)} | {dk(d['seasons']['2026'][k], 2)} | "
                     f"{d['select']['d_roi']['diff'] * 100:+.1f} | {d['decision']} |")
    lines += ["", f"Эталон: B0 (продакшн, сид 42) для C*/G*, B0̄ (3 сида) для M*. Вариантов: {var['n_variants_tested']}.",
              f"ROI Main B0 (отбор): {roi(var['reference']['B0']['select']['main']['roi'])}.", "", "## Главное", ""]
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
              "PYTHONPATH=. timeout 21600 $PY -m experiments.v6_2.leagues.run",
              "$PY -m experiments.v6_2.leagues.decisions && $PY -m experiments.v6_2.leagues.build_pdf", "```", ""]
    (C.HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    C.setup_logging("decisions")
    build()
