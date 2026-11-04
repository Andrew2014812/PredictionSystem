"""Writes experiments/v6_1/main/findings.md from the result files (every number is read, none typed in).

    python -m experiments.v6_1.main.findings_build
"""
from __future__ import annotations

from . import build_pdf as B
from . import common as C


def row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main() -> None:
    B.K.update(B.facts())
    K, ST, SEL, IND, PRE = B.K, B.ST, B.SEL, B.IND, B.PRE
    fb = ST["modes"]["bettable"]["folds"]
    L = ["# FootPredict v6.1 — исследование стратегии Main: выводы", "",
         "Все числа — из `results/*.json` (этот файл собирается `findings_build.py`). PDF: "
         "`reports/v6_1/Main_Strategy_Research.pdf` (`build_pdf.py`).", "",
         "## Периоды и протокол", "",
         "- Выбор вариантов: 2023/24 и 2024/25 (walk-forward, модели обучены на сезонах < s, подгоняемые части — на s−1).",
         "- 2025/26 — только для информации. 2026/27 — независимая проверка по хранилищу продакшна.",
         f"- Критерии внедрения предрегистрированы: {PRE['criteria']['timestamp_utc']} UTC; стратегия — "
         f"{PRE.get('strategy', {}).get('timestamp_utc', 'не записана')} UTC (коммит до проверки: "
         f"{(IND or {}).get('preregistration_commit', '—')}).", ""]
    if IND:
        b = IND["stored_A"]
        L += ["## Исходная проблема (2026/27, Main как записано)", "",
              row("Рассчитано", "Ставок", "Угадано (ставки)", "Заявлено (ставки)", "Ср. коэф.", "Безубыточно", "ROI [95 % ДИ]"),
              row(*["---"] * 7),
              row(b["settled"], b["bets"], B.share(b["bet_hit_rate"]), B.share(b["bet_mean_prob"]), B.num(b["avg_odds"], 3),
                  B.share(b["breakeven_hit"]), f"{B.f(b['roi'])} {B.ci(b['roi_ci95'])}"), ""]
    L += ["## Почему Main убыточен", "",
          f"- Все кандидаты модели откалиброваны (разница заявлено − факт {K['gap_all']}, ECE {K['ece_all']}), "
          f"выбранный Main переоценён на {K['gap_main_prod']} (только с ценой — {K['gap_main_bet']}): проклятие победителя.",
          f"- Отклонение модели от вероятности Bet365 без маржи не несёт сигнала: коэффициент c {K['c_range']}, "
          f"вес модели в смеси с рынком {K['w_blend']}.",
          f"- Симуляция «модель = рынок + шум» даёт переоценку {K['sim_gap']} и истинный EV {K['sim_ev']} — "
          f"как наблюдаемые ({K['gap_main_bet']}, ROI {K['obs_roi_A']}). Маржа на выбранных ценах {K['margin']}.",
          f"- Заявленный EV не связан с доходностью: ROI по бинам EV {K['ev_corridor']}.",
          f"- Тоталы — {K['share_tot_prod']} Main, но не причина; константы правила меняют ROI на {K['sens_spread']}",
          f"- Лучшая одновременная цена Bet365/Pinnacle: {K['best_gain']}; CLV к закрытию Pinnacle {K['clv_fair']}.",
          f"- Другие алгоритмы (CatBoost, LightGBM, GLM, ансамбли, Пуассон): разрыв log loss 1X2 с рынком {K['ll_gap']}, "
          f"ROI Main {K['mod_roi']}.", "",
          "## Стратегии A–G (только кандидаты с ценой, одинаковые матчи)", "",
          row("Стратегия", *[f"ROI {C.period_label(int(s))} [95 % ДИ]" for s in B.FOLDS], *[f"ставок {s}" for s in B.FOLDS]),
          row(*["---"] * 7)]
    for n in fb[B.FOLDS[0]]:
        L.append(row(B.ru(n), *[f"{B.f(fb[s][n]['roi'])} {B.ci(fb[s][n]['roi_ci95'])}" if fb[s][n]["roi"] is not None else "—"
                                for s in B.FOLDS], *[fb[s][n]["bets"] for s in B.FOLDS]))
    L += ["", f"Предрегистрирована: **{B.ru(SEL['primary'])}** (наибольший ROI худшего фолда); правила 1–3 на фолдах: "
          f"{'выполнены' if SEL['preregistered']['primary']['passes_fold_rules_1_3'] else 'не выполнены'}.", ""]
    if IND:
        prim = IND["preregistered_primary"]
        L += ["## Независимая проверка 2026/27", "",
              row("Стратегия", "V1 ставок", "V1 ROI [95 % ДИ]", "V1 разница с A", "V2 ставок", "V2 ROI"), row(*["---"] * 6)]
        for n in ["A", prim] + IND["secondary"]:
            x, y = IND["V1"].get(n), IND["V2"].get(n)
            if not x:
                continue
            L.append(row(B.ru(n), x["bets"], f"{B.f(x['roi'])} {B.ci(x['roi_ci95'])}",
                         B.pp(x["vs_A"]["diff"]) if x.get("vs_A") and x["vs_A"]["diff"] is not None else "—",
                         y["bets"] if y else "—", B.f(y["roi"]) if y and y["roi"] is not None else "—"))
        L += ["", f"Решение по предрегистрированным правилам: **{'внедрить' if IND['adopt'] else 'продакшн не менять'}**.", ""]
    L += ["## Воспроизведение", "", "```",
          "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
          "$PY -m experiments.v6_1.main.run --base --candidates",
          "$PY -m experiments.v6_1.main.run --strategies --calibration --bias --mix --betting --prices --models --figures --select",
          "$PY -m experiments.v6_1.main.preregister --strategy   # + git commit",
          "$PY -m experiments.v6_1.main.run --independent",
          "$PY -m experiments.v6_1.main.build_pdf && $PY -m experiments.v6_1.main.findings_build", "```", ""]
    (C.HERE / "findings.md").write_text("\n".join(L), encoding="utf-8")
    print("findings.md written")


if __name__ == "__main__":
    main()
