"""Builds reports/v6_3/Meta_Selection_Research.pdf from results/ (no number is typed by hand).

    python -m experiments.v6_3.meta_selection.build_pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from reportlab.lib.units import cm

from scripts import report_kit as K
from scripts.report_kit import H1, H2, H3, P, bullets, figure, formula, note, table

from experiments.v6_3 import common as C
from experiments.v6_3.calibration import data as D

from . import features as F
from .figures import NAME_RU

OUT = C.ROOT / "reports" / "v6_3" / "Meta_Selection_Research.pdf"
RES, FIG = F.DIRS["results"], F.DIRS["figures"]


def pp(v, d=1, signed=True):
    return "—" if v is None or (isinstance(v, float) and v != v) else (f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}")


def ci(v, c, d=1):
    return "—" if v is None else f"{pp(v, d)} [{pp(c[0], d)}; {pp(c[1], d)}]"


def pv(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def f3(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def decision(r: dict, n: str) -> str:
    a, b = r["close"][n]["vs_b"]["roi_diff"], r["t09"][n]["vs_b"]["roi_diff"]
    h = r["close"]["holm"].get(n)
    if h is not None and h < 0.05 and a["diff"] > 0 and b["diff"] > 0:
        return "A"
    if a["diff"] > 0 and b["diff"] > 0:
        return "B"
    return "C"


def build() -> None:
    r = C.read_json(RES / "meta_selection.json")
    info = C.read_json(RES / "run_info.json")
    pre = C.read_json(RES / "preregistration.json") if (RES / "preregistration.json").exists() else None
    ch = r["checks"]["close"]
    names = r["meta"] + r["simple"]
    prim = r["primary"]
    rc, r9 = r["close"], r["t09"]
    rep = K.Report(OUT, "Метамодель выбора Main", "FootPredict v6.3 — исследование 9",
                   {"Сезон проверки": "2026/27 (walk-forward по месяцам; сезон уже изучался)",
                    "Обучение": "пул backtest 2025/26 + месяцы 2026/27 до прогнозируемого",
                    "Данные": f"{rc['info']['season_matches']} матчей 2026/27 с кандидатами, "
                              f"{rc['info']['history_matches']} матчей 2025/26"})
    b = rc["baseline_b"]["summary"]
    m = rc[prim]
    rep.add(H1("1. Название и краткие итоги"),
            P("**Метамодель выбора наиболее перспективного Main среди рассчитанных рынков: сравнение с правилом "
              "продакшна и простыми правилами при строгой walk-forward проверке.**"),
            note(f"Итог. Основная метамодель (логистическая регрессия, Main = максимум EV по вероятности метамодели) "
                 f"на close: ROI {pp(m['summary']['roi'])} % против {pp(b['roi'])} % у Baseline B, ΔROI "
                 f"{ci(m['vs_b']['roi_diff']['diff'], m['vs_b']['roi_diff']['ci95'])} п.п. (p Холма "
                 f"{pv(rc['holm'].get(prim))}); t09: ΔROI {ci(r9[prim]['vs_b']['roi_diff']['diff'], r9[prim]['vs_b']['roi_diff']['ci95'])} п.п. "
                 f"Перемешивание целей: средний ROI {pp(ch['permutation'][prim]['mean'])} %; в выборке обучения "
                 f"ROI {pp(ch['in_sample'][prim])} %. Решение: {decision(r, prim)}."))
    rep.add(H1("2. Постановка проблемы"),
            P("Main выбирается фиксированным правилом: надёжность рынка × вероятность × предпочтение коэффициента × "
              "(1 + 0.5 · EV). Вес каждого множителя задан вручную на сезоне валидации. Метамодель может учесть то, "
              "что правило не видит: расхождение модели с рынком без маржи, историческую надёжность рынка, "
              "доступность статистики, характеристики матча. Риск — переобучиться на истории ставок: ROI зависит "
              "от нескольких сотен выигрышей с шумом, и сложная модель легко находит в нём случайные закономерности."))
    rep.add(H1("3. Гипотезы"), *bullets([
        "H1. Метамодель, обученная только на прошлом (walk-forward), выбирает Main с более высоким ROI, чем правило "
        "продакшна, в обеих вселенных (close и t09).",
        "H2. Вероятность метамодели калибрована лучше вероятности модели продакшна на кандидатах Main.",
        "H3. Преимущество (если есть) не объясняется переобучением: оно растёт с объёмом обучения, держится по "
        "месяцам и исчезает при перемешивании целей.",
        "H4. Простые правила (максимум вероятности, максимум EV, максимум вероятности рынка) не хуже метамодели."]))
    ii = rc["info"]
    rows = [["Набор", "Строк пула", "Матчей", "Группы рынков (строк)"],
            ["2025/26 backtest (только обучение)", ii["history_rows"], ii["history_matches"],
             ", ".join(f"{D.GROUP_RU[g]} {v}" for g, v in ii["history_groups"].items())],
            ["2026/27 close", ii["season_rows"], ii["season_matches"],
             ", ".join(f"{D.GROUP_RU[g]} {v}" for g, v in ii["season_groups"].items())],
            ["2026/27 t09", r9["info"]["season_rows"], r9["info"]["season_matches"],
             ", ".join(f"{D.GROUP_RU[g]} {v}" for g, v in r9["info"]["season_groups"].items())]]
    rep.add(H1("4. Исходные данные"),
            P("Пул матча — все селекции, которые правило продакшна может выбрать как Main (не европейская фора, не "
              "точный счёт, вероятность модели 0.40–0.85), **с реальной ценой** в момент решения. Обучение: "
              "out-of-sample прогнозы backtest 2025/26 (модель обучена до 2025-06-01; цены Bet365 из Football-Data "
              "только для 1X2, тотала 2.5 и одной азиатской линии) и кандидаты 2026/27 общего replay v6.3."),
            table(rows, widths=[4.2 * cm, 2 * cm, 1.8 * cm, 9 * cm]),
            P(f"Сыгранные матчи 2026/27 без единого кандидата в пуле (Main берётся у Baseline B, помечено "
              f"«fallback»): close {rc['info']['season_played_matches_without_pool']}, t09 "
              f"{r9['info']['season_played_matches_without_pool']}.", "small"))
    rep.add(H1("5. Методика"),
            H2("5.1. Признаки (все известны до матча)"),
            *bullets(["вероятность модели (logit pc), откалиброванная вероятность (основная карта исследования 2: для "
                      "2026/27 — обучена на 2025/26 и месяцах до матча, для строк 2025/26 — без своего месяца), EV;",
                      "цена: ln odds, вероятность рынка без маржи q = (1/o)/Σ(1/o) (полная книга рынка; иначе 1/o, "
                      "делённое на медианную маржу группы), признак полной книги, разность logit pc − logit q;",
                      "тип рынка (группа, сторона: хозяева / гости / ничья / тотал больше / меньше / обе забьют), "
                      "линия, P(push);",
                      "матч: вероятности 1X2, P(тотал > 2.5), P(угловые > 9.5), число кандидатов всего и с ценой, "
                      "оценка правила продакшна и признак «правило выбрало бы эту строку», месяц сезона;",
                      "доступность статистики: минимум сыгранных матчей сезона двух команд, число очных встреч, "
                      "минимум матчей в окне 10;",
                      "надёжность группы рынков до месяца матча: разрыв (средняя p − частота), ROI и объём строк этой "
                      "группы в пуле более ранних месяцев."]),
            H2("5.2. Модели"),
            *bullets(["**meta_lr_ev** (основная, зафиксирована заранее): стандартизация + логистическая регрессия "
                      "(C = 0.1), P(выигрыш | без возврата); Main = max EV_meta;",
                      "**meta_gbm_ev**: XGBoost, глубина 3, 300 деревьев, шаг 0.03, 2 потока, P(выигрыш); max EV_meta;",
                      "**meta_gbm_ret**: XGBoost-регрессия прибыли на единицу ставки; Main = max прогноза доходности;",
                      "**meta_lr_rule**: правило продакшна, где вероятность модели заменена на P метамодели."]),
            formula("EV_meta = P_meta · (1 − P(push)) · odds − (1 − P(push))"),
            P("Гиперпараметры заданы заранее и на 2026/27 не подбирались. Простые правила на том же пуле: max p "
              "модели, max EV, max вероятности рынка q, правило продакшна только среди кандидатов с ценой, правило продакшна на откалиброванной вероятности."),
            H2("5.3. Walk-forward"),
            P("Main матчей месяца M выбирает модель, обученная на пуле 2025/26 и пуле 2026/27 той же вселенной из "
              "матчей до 1-го числа M (август — только 2025/26). Метамодель никогда не видит исходы матчей, для "
              "которых выбирает. ROI — по правилам продакшна, парный бутстреп ΔROI к Baseline B того же момента "
              f"решения, поправка Холма по {len(names)} вариантам ({len(r['meta'])} метамодели + {len(r['simple'])} простых "
              "правил). Контроль «правило продакшна, только с ценой» отделяет эффект выбора только среди кандидатов с "
              "ценой (важен в t09, где у четверти Main продакшна нет цены) от эффекта самой метамодели."),
            H2("5.4. Проверки переобучения"),
            *bullets(["кривые обучения: случайные 5–100 % обучающих матчей (по 3 повтора), log loss и ROI;",
                      f"перестановочный тест: цели перемешаны внутри каждого обучающего набора "
                      f"({', '.join(str(v['n']) for v in ch['permutation'].values())} повторов для трёх моделей);",
                      f"нулевое распределение: случайный выбор из пула ({rc['random_pick']['n']} повторов);",
                      "оптимизм в выборке: модель на всём 2026/27 оценена на тех же матчах;",
                      "обучение только на 2025/26 (без месяцев 2026/27); стабильность по месяцам и лигам."]))
    rep.add(H1("6. Эксперименты"), *bullets([
        f"{len(r['meta'])} метамодели и {len(r['simple'])} простых правила × 2 вселенные (close, t09), "
        "4 месячных переобучения каждая.",
        f"Кривая обучения: {len(ch['learning_curve'])} прогонов walk-forward.",
        f"Перестановки: {sum(v['n'] for v in ch['permutation'].values())} прогонов walk-forward; случайный выбор: "
        f"{rc['random_pick']['n']} × 2.",
        f"Всего вариантов выбора Main, оценённых по ROI: {len(names)} (Холм по ним)."]))

    rep.add(H1("7. Результаты"))
    for u in ("close", "t09"):
        rr = r[u]
        rows = [["Вариант", "Ставки", "ROI, % [95 % ДИ]", "Попад., %", "Коэф.", "ΔROI к B, п.п. [95 % ДИ]",
                 "p Холма", "Тот же Main", "EV<0, %"]]
        for n in ["baseline_b"] + names:
            x = rr[n]
            s, d = x["summary"], x["vs_b"]["roi_diff"]
            rows.append([NAME_RU[n], s["bets"], ci(s["roi"], x["roi_ci"]["ci95"]), pp(s["hit_rate"], 1, False),
                         f"{s['avg_odds']:.2f}", "—" if n == "baseline_b" else ci(d["diff"], d["ci95"]),
                         "—" if n == "baseline_b" else pv(rr["holm"].get(n)),
                         pp(x["vs_b"]["same_main_share"], 0, False) + " %", pp(x["neg_ev_share_raw"], 1, False)])
        rep.add(H2(f"7.{1 if u == 'close' else 2}. Вселенная {u}"),
                table(rows, widths=[3.6 * cm, 1.4 * cm, 2.8 * cm, 1.6 * cm, 1.2 * cm, 2.8 * cm, 1.4 * cm, 1.2 * cm, 1.3 * cm]),
                P(f"Таблица {1 if u == 'close' else 2}. Main по каждому варианту, парно с Baseline B "
                  f"({rr['baseline_b']['vs_b']['matches']} сыгранных матчей). Случайный выбор из пула: средний ROI "
                  f"{pp(rr['random_pick']['mean'])} % (5–95 %: {pp(rr['random_pick']['q05'])}; "
                  f"{pp(rr['random_pick']['q95'])}). Baseline A (история сайта): {rr['baseline_a']['summary']['bets']} ставок, ROI {pp(rr['baseline_a']['summary']['roi'])} %.",
                  "small"))
    rows = [["Вариант", "Группы Main (близкий момент close)"]]
    for n in ["baseline_b"] + names:
        g = rc[n]["groups"]
        rows.append([NAME_RU[n], ", ".join(f"{D.GROUP_RU[k]} {v}" for k, v in sorted(g.items(), key=lambda kv: -kv[1]))])
    rep.add(table(rows, widths=[5 * cm, 12 * cm]), P("Таблица 3. Состав выбранных Main по группам рынков (close).", "small"))
    pq = rc["prob_quality"]
    rows = [["Вероятность", "n", "LogLoss", "Brier", "ECE, п.п.", "Разрыв, п.п."]]
    for k, lab in (("raw", "модель продакшна"), ("calibrated", "откалиброванная (исследование 2)"),
                   ("market_q", "рынок без маржи"), ("market_q_full_book", "рынок, полная книга"),
                   ("p_lr", "метамодель LR"), ("p_gbm", "метамодель GBM")):
        x = pq[k]
        rows.append([lab, x["n"], f3(x["log_loss"]), f3(x["brier"]), pp(x["ece"], 2, False), pp(x["gap"])])
    rep.add(H2("7.3. Качество вероятностей на пуле 2026/27"), table(rows),
            P("Таблица 4. Все строки пула сыгранных матчей 2026/27 (close), вероятности вне выборки обучения.",
              "small"))
    rep.page_break()
    rep.add(H1("8. Графики"),
            *figure(FIG / "roi_forest.png", "Рис. 1. ΔROI к Baseline B (95 % ДИ парного бутстрепа): метамодели "
                    "(зелёный) и простые правила (синий)."),
            *figure(FIG / "learning_curve.png", "Рис. 2. Кривые обучения (close): log loss пула месяца M и ROI "
                    "выбранного Main в зависимости от доли обучающих матчей."),
            *figure(FIG / "permutation.png", "Рис. 3. Перестановочный тест и случайный выбор: ROI метамодели против "
                    "моделей на перемешанных целях и случайного выбора из того же пула."),
            *figure(FIG / "monthly.png", "Рис. 4. ROI по месяцам."),
            *figure(FIG / "lr_coef.png", "Рис. 5. Наибольшие стандартизированные коэффициенты итоговой "
                    "логистической регрессии (обучена на всех данных — только для интерпретации). lq — logit вероятности рынка "
                    "без маржи, lp / lp_cal — logit вероятности модели / откалиброванной, line — линия рынка, edge — "
                    "lp − lq, g_* — группа рынка, s_* — сторона, ev_raw — EV модели, lodds — ln коэффициента, rel_* — "
                    "надёжность группы до месяца, ncand* — число кандидатов, is_rule_main — выбор правила продакшна.",
                    width=12 * cm))
    rep.add(H1("9. Статистический анализ"))
    lc = pd.DataFrame(ch["learning_curve"]).groupby("frac").mean(numeric_only=True).reset_index()
    rows = [["Доля обучения", "LogLoss LR", "ROI LR, %", "LogLoss GBM", "ROI GBM, %"]]
    for _, x in lc.iterrows():
        rows.append([f"{x['frac'] * 100:.0f} %", f3(x["meta_lr_ev_log_loss"]), pp(x["meta_lr_ev_roi"]),
                     f3(x["meta_gbm_ev_log_loss"]), pp(x["meta_gbm_ev_roi"])])
    rep.add(H3("Кривая обучения (close)"), table(rows, widths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]))
    rows = [["Модель", "ROI вне выборки, %", "ROI перемеш. (среднее / макс.), %", "Доля перемеш. ≥ реального",
             "ROI в выборке обучения, %", "Только 2025/26, %"]]
    for n in ("meta_lr_ev", "meta_gbm_ev", "meta_gbm_ret"):
        pr = ch["permutation"][n]
        real = rc[n]["summary"]["roi"]
        rows.append([NAME_RU[n], pp(real), f"{pp(pr['mean'])} / {pp(pr['max'])}",
                     f"{np.mean(np.array(pr['values']) >= real):.2f}", pp(ch["in_sample"][n]),
                     pp(ch["history_only"].get(n))])
    rep.add(H3("Перестановки, оптимизм в выборке, источник обучения (close)"),
            table(rows, widths=[4.0 * cm, 2.4 * cm, 3.2 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm]),
            P(f"LogLoss в выборке обучения: LR {f3(ch['in_sample_log_loss']['p_lr'])}, GBM "
              f"{f3(ch['in_sample_log_loss']['p_gbm'])}; только 2025/26: LR {f3(ch['history_only_log_loss']['p_lr'])}, "
              f"GBM {f3(ch['history_only_log_loss']['p_gbm'])}.", "small"))
    for u in ("close", "t09"):
        rb = r[u]["robustness"][prim]
        rbb = r[u]["robustness"]["baseline_b"]
        rows = [["Месяц", "Ставки", "ROI, %", "ROI B, %", "Δ, п.п.", "Малая выборка"]]
        for x in rb["by_month"]:
            rows.append([x["month"], x["bets"], pp(x["roi"]), pp(x["roi_b"]), pp(x["roi_diff"]), "да" if x["small"] else ""])
        rep.add(H3(f"Основная метамодель по месяцам — {u}"),
                table(rows, widths=[2.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2.5 * cm]),
                P(f"ROI основной метамодели и Baseline B по месяцам ({u}); ниже — зависимость от крупных выигрышей и "
                  "цены.", "small"))
        rows = [["Показатель", "Метамодель LR", "Baseline B"]]
        for k in ("1", "5", "10", "20"):
            rows.append([f"ROI без {k} крупнейших выигрышей, %", pp(rb["roi_without_top_wins"][k]),
                         pp(rbb["roi_without_top_wins"][k])])
        for k, lab in (("0.02", "2 %"), ("0.05", "5 %")):
            rows.append([f"ROI при коэффициентах ниже на {lab}, %", pp(rb["roi_odds_haircut"][k]),
                         pp(rbb["roi_odds_haircut"][k])])
        rows.append(["Лиг лучше B", f"{rb['leagues_better']} из {rb['leagues']}", ""])
        rep.add(table(rows, widths=[7 * cm, 3.5 * cm, 3.5 * cm]))
    rows = [["Цена", "Метод", "Матчей", "Сыгр.", "Обраб.", "С ценой", "Мало стат.",
             "Ставки", "Без цены", "PUSH/ VOID", "Не сыгр."]]
    for u in ("close", "t09"):
        for x in r[u]["coverage"]:
            rows.append([u, NAME_RU.get(x["method"], x["method"]), x["season_matches"], x["played"], x["processed"],
                         x["real_price"], x["few_statistics"], x["bets"], x["excluded_no_price"],
                         x["excluded_push_void"], x["not_played"]])
    rep.add(H3("Охват"), table(rows, widths=[1.2 * cm, 3.0 * cm, 1.5 * cm, 1.6 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm,
                                             1.5 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm]), P("«Мало статистики» — у одной из команд меньше 3 матчей сезона (матч не "
                                        "исключается); PUSH/VOID — возврат ставки (целые азиатские линии), в ROI ставкой не считается. Ничего не "
                                        "исключено молча.", "small"))
    from .conclusions import conclusions, limitations, recommendations
    rep.add(H1("10. Выводы"), *bullets(conclusions(r)))
    rep.add(H1("11. Рекомендации"), *bullets(recommendations(r, pre)))
    rep.add(H1("12. Ограничения"), *bullets(limitations(r)))
    rep.add(H1("Как воспроизвести"), *K.reproduce([
        "# сначала исследование 2 (калибровка): его кэш даёт откалиброванные вероятности",
        "experiments/v6_3/heavy.sh 3600 experiments/v6_3/calibration/logs/run.log \\",
        "    .venv/bin/python -m experiments.v6_3.calibration.run",
        "experiments/v6_3/heavy.sh 5400 experiments/v6_3/meta_selection/logs/run.log \\",
        "    .venv/bin/python -m experiments.v6_3.meta_selection.run",
        ".venv/bin/python -m experiments.v6_3.meta_selection.figures",
        ".venv/bin/python -m experiments.v6_3.meta_selection.preregister",
        ".venv/bin/python -m experiments.v6_3.meta_selection.build_pdf"]),
        P(f"Git: {info['git_sha']}; данные: picks {info['data']['picks']['mtime']}; время расчёта {info['seconds']} с.",
          "small"))
    rep.build()
    print("pages", rep.check())


if __name__ == "__main__":
    build()
