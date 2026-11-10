"""Builds reports/v6_2/Bookmaker_Benchmark_Research.pdf from results/*.json and figures/*.png (Russian).

    python -m experiments.v6_2.bookmaker.build_pdf

Every number is read from the result files of this folder; nothing is typed in by hand.
"""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from scripts.report_kit import H1, H2, P, Report, bullets, figure, formula, note, reproduce, table

from . import common as C

OUT = C.ROOT / "reports" / "v6_2" / "Bookmaker_Benchmark_Research.pdf"
J = C.load_json
DM, BM, ENC, HY, BET, MM, PRE, RUN = (J(x) for x in ("demargin.json", "benchmark.json", "encompassing.json",
                                                     "hybrids.json", "betting.json", "multimarket_2026.json",
                                                     "preregistration.json", "run_info.json"))
FIG = C.FIGURES
MK = {"1X2": "1X2", "OU25": "Тотал 2.5", "AH": "Аз. фора"}
REF = {"b365_open": "Bet365 откр.", "ps_open": "Pinnacle откр.", "avg_open": "среднее откр.", "bfe_open": "Betfair откр.",
       "b365_close": "Bet365 закр.", "ps_close": "Pinnacle закр.", "bfe_close": "Betfair закр."}
HYB = {"model": "модель", "H0": "H0 рынок", "H1": "H1 смесь", "H2": "H2 пул", "H3": "H3 коррекция рынка",
       "H4": "H4 мета-модель", "H5": "H5 остатки (XGB)"}
SEAS = (2023, 2024, 2025, 2026)


class Report62(Report):
    def build(self):
        from scripts.report_kit import _Doc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = _Doc(self.path, f"FootPredict — {self.title}", f"FootPredict v6.2 — {self.title}")
        doc.multiBuild(self._front() + self.story)
        return self.path


def T(rows, widths=None):
    return [table(rows, widths=[w * cm for w in widths] if widths else None), Spacer(1, 6)]


def f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def pc(v, d=1, signed=True):
    if v is None:
        return "—"
    return f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%"


def ci(c, mult=100, d=1):
    if not c or c[0] is None:
        return "—"
    return f"[{c[0] * mult:+.{d}f}; {c[1] * mult:+.{d}f}]"


def dll(r, d=1):
    """paired diff dict -> 'x.x [lo; hi]' in units of 10^-3."""
    if not r or r.get("diff") is None:
        return "—"
    return f"{r['diff'] * 1000:+.{d}f} {ci(r['ci95'], 1000, d)}"


def sl(s):
    return C.period_label(int(s))


def front(rep):
    sel = [BM["model_vs_market"][f"1X2|{s}|b365_open"] for s in C.SELECT]
    rep.add(H1("Краткое содержание"))
    rep.add(note(
        "Модель FootPredict не лучше вероятности букмекера ни на одном рынке и ни в одном сезоне: log loss 1X2 хуже, чем "
        f"у Bet365 на открытии, на {sel[0]['d_ll']['diff'] * 1000:+.1f} и {sel[1]['d_ll']['diff'] * 1000:+.1f}×10⁻³ "
        "(2023/24, 2024/25), тотала 2.5 — на несколько тысячных. Рынок «охватывает» модель: вес модели рядом с ценой "
        "незначим во всех тестах сезонов выбора. Ни один гибрид (смесь, пул, коррекция рынка, мета-модель, обучение на "
        "остатках к рынку) не улучшает рынок значимо после поправки Холма."))
    h2 = HY["results"]
    rep.add(bullets([
        "Лучший способ снятия маржи (по log loss 2023/24+2024/25): " + "; ".join(
            f"{k.replace('|', ' ')} — {v}" for k, v in BM["choice"].items() if k.endswith("b365_open")) + ".",
        f"Ставки по модели при EV > 0 по цене Bet365 открытия убыточны: 1X2 {pc(BET['1X2|2023|model|0.0']['roi'])} и "
        f"{pc(BET['1X2|2024|model|0.0']['roi'])}, тотал 2.5 {pc(BET['OU25|2023|model|0.0']['roi'])} и "
        f"{pc(BET['OU25|2024|model|0.0']['roi'])}; CLV к справедливому закрытию отрицателен.",
        f"2026/27 (только для информации): вес модели рядом с Bet365 открытия положителен (1X2 "
        f"{f(ENC['tests']['1X2|2026|b365_open']['b_model'], 2)}, тотал 2.5 {f(ENC['tests']['OU25|2026|b365_open']['b_model'], 2)}), "
        f"пул H2 по тоталу 2.5 лучше рынка на {dll(h2['OU25|2026']['H2']['d_ll_vs_H0'])}×10⁻³ — это зафиксировано как "
        "гипотеза для проверки вперёд (`results/preregistration.json`), продакшн не меняется.",
    ]))


def intro(rep):
    rep.add(H1("1. Цель исследования"))
    rep.add(P("Найти рынки, лиги, диапазоны коэффициентов и сезоны, где оценка вероятности FootPredict лучше оценки "
              "букмекера; правильно снять маржу; сравнить по log loss, Brier, калибровке, ROI и CLV; проверить, есть ли в "
              "модели информация, которой нет в цене, и можно ли построить гибрид модели и рынка без утечки."))
    rep.add(H1("2. Проблема"))
    rep.add(P("Main и Risk выбираются по вероятности модели и EV = p·o − 1. Если цена букмекера уже содержит всю "
              "информацию модели (и больше), то EV по модели систематически завышен для выборов, где модель расходится с "
              "рынком (проклятие победителя, см. v6.1), и ставки убыточны. Нужно понять, где модель знает больше рынка, "
              "если такое место вообще есть."))
    rep.add(H1("3. Гипотезы"))
    rep.add(bullets([
        "H1. На части рынков/лиг/диапазонов коэффициентов модель точнее вероятности букмекера без маржи.",
        "H2. Модель несёт информацию, которой нет в цене (тест охвата: вес модели > 0 рядом с ценой).",
        "H3. Гибрид модели и рынка, обученный только на прошлых сезонах, точнее рынка.",
        "H4. Ставки по такому гибриду имеют положительный ROI/CLV.",
    ]))
    rep.add(H1("4. Данные"))
    rows = [["Файл", "Строк", "Изменён"]]
    for k, v in (RUN or {}).get("data_version", {}).items():
        if v and k.endswith(".parquet"):
            rows.append([f"`{k}`", str(v.get("rows", "—")), v.get("mtime", "—")])
    rep.add(T(rows, [10, 2.5, 4.5]))
    n = {m: {s: HY["results"][f"{m}|{s}"]["n"] for s in SEAS} for m in ("1X2", "OU25", "AH")}
    rep.add(P("Вероятности модели: 2022/23–2025/26 — walk-forward кэш v6.1 (модели продакшна обучены только на сезонах "
              "до прогнозируемого); 2026/27 — сохранённые прогнозы продакшна (live, иначе backfill). Цены: Football-Data, "
              "снимок открытия (пятница/вторник, доступен до решения) и закрытия: Bet365 (книга продакшна), Pinnacle "
              "(до 2025/26), среднее рынка, Betfair Exchange (с 2024/25). Рынки: 1X2, тотал 2.5 и азиатская линия "
              "Football-Data (только половинные линии без возврата и неизменные к закрытию)."))
    rows = [["Рынок"] + [sl(s) for s in SEAS]] + [[MK[m]] + [str(n[m][s]) for s in SEAS] for m in n]
    rep.add(T(rows, [3, 3.5, 3.5, 3.5, 3.5]))


def methods(rep):
    rep.add(H1("5. Методы"))
    rep.add(P("Снятие маржи для полного рынка с ценами o_{i} (π_{i} = 1/o_{i}, Σπ = 1 + маржа):"))
    rep.add(bullets(["пропорциональный: q_{i} = π_{i}/Σπ;",
                     "Шин: q_{i} = (√(z² + 4(1−z)π_{i}²/Σπ) − z)/(2(1−z)), z — доля инсайдеров, Σq = 1;",
                     "степенной: q_{i} = π_{i}^{k}, k подобран так, что Σq = 1;",
                     "логарифмический (odds ratio): logit q_{i} = logit π_{i} − c, Σq = 1;",
                     "аддитивный: q_{i} = π_{i} − (Σπ − 1)/K."]))
    rep.add(P("Метрики: log loss −log q_{y}, Brier Σ(q_{k} − 1[y=k])², калибровка — наклон и сдвиг логистической регрессии "
              "исхода на logit прогноза, ECE (10 бинов). Разности модель − рынок — парный бутстреп по матчам (1000)."))
    rep.add(P("Тест охвата (encompassing) — объединённый мультиномиальный логит, оценка ММП:"))
    rep.add(formula("P(k) = softmax_{k}(a_{k} + b_{1}·log q_{k} + b_{2}·log p_{k}),  a_{1} = 0"))
    rep.add(P("b_{2} = 0 — рынок охватывает модель. Тест отношения правдоподобия (χ², 1 ст. св.), интервал b_{2} — бутстреп "
              "по матчам; поправка Холма по тестам сезонов выбора."))
    rep.add(P("Гибриды (все параметры для сезона s обучены на сезонах 2022/23 … s−1; рыночный вход — Bet365 открытия, "
              "доступный в момент решения): H0 — рынок; H1 — w·p + (1−w)·q; H2 — пул выше; H3 — коррекция рынка "
              "softmax(a + b₁ log q) без модели; H4 — мультиномиальная логистическая регрессия на log q, log p, марже и лиге; "
              "H5 — градиентный бустинг на признаках матча с логитом рынка как base margin (учит только то, чего нет в "
              "цене; деревья на сезонах 2021/22 … s−1, число деревьев — ранняя остановка на s−1)."))
    rep.add(P("Ставки: в каждом матче выбор с наибольшим EV = p·o_{Bet365, откр.} − 1, ставка при EV > τ (τ = 0; 3 %; 6 %). "
              "CLV_{спр} = o·q_{закр} − 1, q_{закр} — Betfair Exchange закрытия (Шин), иначе Pinnacle, иначе Bet365."))
    rep.add(H1("6. Параметры"))
    p = (RUN or {}).get("parameters", {})
    env = (RUN or {}).get("environment", {})
    rows = [["Параметр", "Значение"]] + [[k, str(v)] for k, v in p.items()]
    rows += [["seed", str(env.get("seed"))], ["n_jobs / потоки", "2"], ["git SHA прогона", str(env.get("git_sha", "—"))[:12]]]
    rep.add(T(rows, [4.5, 12.5]))


def demargin(rep):
    rep.add(H1("7. Снятие маржи"))
    rows = [["Рынок|сезон|снимок", "Матчей", "Маржа", "пропорц.", "Шин", "степен.", "логар.", "аддит."]]
    for k, v in DM["by_key"].items():
        m, s, ref = k.split("|")
        if s not in ("2023", "2024") or ref not in ("b365_open", "ps_open", "b365_close", "ps_close", "bfe_close"):
            continue
        rows.append([f"{MK[m]} {C.season_label(int(s))} {REF[ref]}", str(v["n"]), pc(v["margin"], 1, False)] +
                    [f(v[x], 4) for x in ("prop", "shin", "power", "log", "add")])
    rep.add(T(rows, [5.2, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8, 1.8]))
    rep.add(P("Выбор метода по сезонам выбора: " + "; ".join(f"{k.replace('|', ' ')} — {v}" for k, v in DM["choice"].items())
              + ". Различия между методами — десятитысячные доли log loss; для 1X2 все методы, учитывающие перекос "
                "фаворит–аутсайдер (Шин, степенной, аддитивный), лучше пропорционального."))
    rep.add(figure(FIG / "demargin.png", "Рис. 1. Log loss методов снятия маржи относительно пропорционального (×1000)."))


def vs_market(rep):
    rep.add(H1("8. Модель против рынка"))
    rows = [["Рынок", "Сезон", "Эталон", "Матчей", "LL модели", "LL рынка", "Δ LL ×10⁻³ [ДИ]", "Δ Brier ×10⁻³ [ДИ]",
             "Модель лучше, доля"]]
    for k, v in BM["model_vs_market"].items():
        m, s, ref = k.split("|")
        if ref not in ("b365_open", "ps_open", "b365_close", "bfe_close"):
            continue
        rows.append([MK[m], sl(s), REF[ref], str(v["n"]), f(v["ll_model"], 4), f(v["ll_market"], 4), dll(v["d_ll"]),
                     dll(v["d_brier"]), pc(v["share_model_better_ll"], 1, False)])
    rep.add(T(rows, [1.4, 2.2, 1.9, 1.2, 1.4, 1.4, 2.9, 2.9, 1.7]))
    rep.add(figure(FIG / "model_vs_market.png", "Рис. 2. Разность log loss модели и рынка (×1000, 95 % ДИ); выше нуля — рынок точнее."))
    rep.add(H2("8.1 По лигам"))
    for market in ("1X2", "OU25"):
        L = BM["by_league"][market]
        rows = [["Лига", "Матчей", "Δ LL 23/24+24/25 ×10⁻³ [ДИ]", "p (Холм)", "Δ 2025/26", "Δ 2026/27"]]
        for lg, v in sorted(L["leagues"].items(), key=lambda kv: kv[1]["select"]["mean"]):
            s = v["select"]
            rows.append([lg, str(s["n"]), f"{s['mean'] * 1000:+.1f} {ci(s['ci95'], 1000)}", f(v.get("p_holm"), 3),
                         f"{v['2025']['mean'] * 1000:+.1f}" if v["2025"].get("mean") is not None else "—",
                         f"{v['2026']['mean'] * 1000:+.1f}" if v["2026"].get("mean") is not None else "—"])
        rep.add(P(f"{MK[market]}: модель − {REF[L['reference']]} (метод {L['method']}), {L['n_leagues_tested']} лиг; "
                  "отрицательное значение — модель точнее."))
        rep.add(T(rows, [1.6, 1.6, 5.0, 2.0, 3.4, 3.4]))
    rep.add(figure(FIG / "leagues.png", "Рис. 3. 1X2 по лигам: модель − Bet365 открытия (×1000, 95 % ДИ)."))
    better = [lg for m in ("1X2", "OU25") for lg, v in BM["by_league"][m]["leagues"].items()
              if v["select"]["ci95"][1] is not None and v["select"]["ci95"][1] < 0]
    rep.add(P("Лиг, где модель значимо (без поправки) точнее рынка на сезонах выбора: " + (", ".join(better) if better else "нет")
              + "."))
    rep.add(H2("8.2 По диапазонам коэффициента"))
    rows = [["Рынок", "Период", "Коэф.", "Исходов", "Δ LL ×10⁻³ [ДИ]", "Ср. p модели", "Ср. q рынка", "Частота"]]
    for k, v in BM["by_odds_range"].items():
        m, per, b = k.split("|")
        rows.append([MK[m], {"select": "23/24+24/25", "2025": "2025/26 (инф.)", "2026": "2026/27 (инф.)"}[per], b, str(v["n"]),
                     f"{v['d_ll']['mean'] * 1000:+.1f} {ci(v['d_ll']['ci95'], 1000)}", f(v["mean_p"], 3), f(v["mean_q"], 3),
                     f(v["rate"], 3)])
    rep.add(T(rows, [1.6, 2.6, 1.6, 1.4, 3.6, 2.0, 2.0, 2.2]))
    rep.add(figure(FIG / "odds_range.png", "Рис. 4. Разность log loss исхода по диапазонам коэффициента Bet365 открытия."))
    rep.add(H2("8.3 Калибровка"))
    rows = [["Рынок", "Сезон", "Наклон модели", "Наклон рынка", "ECE модели", "ECE рынка"]]
    for k, v in BM["model_vs_market"].items():
        m, s, ref = k.split("|")
        if ref != "b365_open":
            continue
        rows.append([MK[m], sl(s), f(v["calib_model"]["slope"], 2), f(v["calib_market"]["slope"], 2),
                     f(v["calib_model"]["ece"], 3), f(v["calib_market"]["ece"], 3)])
    rep.add(T(rows, [2, 3.5, 2.9, 2.9, 2.85, 2.85]))
    rep.add(P("Наклон < 1 — прогноз слишком уверенный, > 1 — слишком осторожный (событие: победа хозяев / больше 2.5 / "
              "хозяева с форой). Для азиатской линии наклоны неустойчивы: вероятности там близки к 0.5 и выборка мала."))
    rep.add(figure(FIG / "reliability.png", "Рис. 5. Надёжность модели и Bet365 открытия (2023/24+2024/25, децили прогноза)."))


def encompassing(rep):
    rep.add(H1("9. Есть ли в модели информация сверх цены"))
    rows = [["Рынок", "Сезон", "Эталон", "Матчей", "b₁ рынок", "b₂ модель [ДИ]", "p (LR)", "p (Холм)"]]
    for k, v in ENC["tests"].items():
        m, s, ref = k.split("|")
        if ref not in ("b365_open", "ps_open", "b365_close", "bfe_close"):
            continue
        rows.append([MK[m], "23/24+24/25" if s == "select" else sl(s), REF[ref], str(v["n"]), f(v["b_market"], 2),
                     f"{v['b_model']:+.2f} [{v['b_model_ci95'][0]:+.2f}; {v['b_model_ci95'][1]:+.2f}]", f(v["p_lr"], 3),
                     f(v.get("p_holm"), 3)])
    rep.add(T(rows, [1.4, 2.6, 2.0, 1.3, 1.6, 3.6, 1.6, 1.6]))
    rep.add(P(f"Поправка Холма — по {ENC['n_tests_holm']} тестам отдельных сезонов выбора. На сезонах выбора вес модели "
              "незначим везде, а относительно закрытия Pinnacle/Betfair даже отрицателен: всё, что знает модель, уже есть в "
              "цене, а цена знает больше. Положительный вес в 2026/27 — только для информации (сезон использовался в v6.1)."))
    rep.add(figure(FIG / "encompassing.png", "Рис. 6. Вес модели b₂ рядом с ценой (95 % бутстреп-интервал)."))


def hybrids(rep):
    rep.add(H1("10. Гибриды модели и рынка"))
    rows = [["Рынок", "Сезон", "Вариант", "LL", "Δ LL к H0 ×10⁻³ [ДИ]", "p", "p (Холм)"]]
    for k, v in HY["results"].items():
        m, s = k.split("|")
        for nm in ("H0", "H1", "H2", "H3", "H4", "H5", "model"):
            if nm not in v:
                continue
            d = v[nm].get("d_ll_vs_H0")
            rows.append([MK[m], sl(s), HYB[nm], f(v[nm]["ll"], 4), dll(d) if d else "—",
                         f(d["p_two_sided"], 3) if d else "—", f(v[nm].get("p_holm"), 3)])
    rep.add(T(rows, [1.5, 3.0, 3.0, 1.6, 4.1, 1.8, 2.0]))
    w = {k: v["params"].get("H1_w") for k, v in HY["results"].items()}
    rep.add(P("Вес модели в линейной смеси H1, подобранный на прошлых сезонах: " + "; ".join(
        f"{k.replace('|', ' ')}: {x:.2f}" for k, x in w.items() if x is not None) + f". Тестов с поправкой Холма: "
        f"{HY['n_tests_holm']}. Деревьев H5 (ранняя остановка): " + "; ".join(
        f"{k}: {v['rounds']}" for k, v in HY["xgb_rounds"].items()) + "."))
    rep.add(figure(FIG / "hybrids.png", "Рис. 7. Гибриды против рынка H0 (×1000, 95 % ДИ); ниже нуля — лучше рынка."))


def betting(rep):
    rep.add(H1("11. Ставки и CLV"))
    rows = [["Рынок", "Сезон", "Вариант", "τ", "Ставок", "Ср. коэф.", "ROI [95 % ДИ]", "CLV спр. [ДИ]"]]
    for k, v in BET.items():
        m, s, nm, tau = k.split("|")
        if v.get("bets", 0) < 30 or nm not in ("model", "H2", "H3", "H4", "H5") or tau == "0.06":
            continue
        rows.append([MK[m], sl(s), HYB[nm], f"{float(tau):.2f}", str(v["bets"]), f(v["avg_odds"], 2),
                     f"{pc(v['roi'])} {ci(v['roi_ci95'])}", f"{pc(v['clv_fair']['mean'])} {ci(v['clv_fair']['ci95'])}"])
    rep.add(T(rows, [1.5, 2.9, 2.6, 1.0, 1.3, 1.4, 3.4, 2.9]))
    rep.add(P("Показаны варианты с ≥ 30 ставками. Модель «находит» EV в основном на длинных коэффициентах 1X2 (средний "
              f"коэффициент {f(BET['1X2|2023|model|0.0']['avg_odds'], 2)} и {f(BET['1X2|2024|model|0.0']['avg_odds'], 2)} "
              "в 2023/24 и 2024/25): это перекос модели к аутсайдерам относительно рынка, а не ценность. Гибриды, "
              "привязанные к рынку, почти не находят ставок с EV > 0, и их CLV тоже отрицателен."))
    rep.add(figure(FIG / "betting.png", "Рис. 8. ROI ставок при EV > 0 (≥ 30 ставок, 95 % ДИ)."))


def multimarket(rep):
    rep.add(H1("12. 2026/27: все группы рынков против закрытия OddsPapi (для информации)"))
    rows = [["Группа", "Событий", "Δ LL ×10⁻³ [ДИ]", "Модель лучше, доля"]]
    for k, v in MM["by_group"].items():
        rows.append([C.GROUP_RU.get(k, k), str(v["n"]), f"{v['mean'] * 1000:+.1f} {ci(v['ci95'], 1000)}",
                     pc(v["share_model_better"], 1, False)])
    rep.add(T(rows, [4, 2.5, 6, 4.5]))
    rows = [["Рынок", "Событий", "Δ LL ×10⁻³ [ДИ]"]]
    for k, v in MM["by_market"].items():
        rows.append([k, str(v["n"]), f"{v['mean'] * 1000:+.1f} {ci(v['ci95'], 1000)}"])
    rep.add(T(rows, [5, 3, 9]))
    rep.add(P(f"Матчей: {MM['matches']}. Угловые и «обе забьют» — на уровне закрытия (интервал включает 0), остальные "
              "группы — хуже закрытия. Закрытие — самая трудная планка (недоступна в момент решения)."))
    rep.add(figure(FIG / "multimarket.png", "Рис. 9. 2026/27: модель − закрытие OddsPapi (×1000, 95 % ДИ)."))


def prereg(rep):
    rep.add(H1("13. Предрегистрация"))
    if not PRE:
        rep.add(P("Файл предрегистрации не создан."))
        return
    v = PRE["frozen_variant"]
    rep.add(P(f"Статус: {PRE['status']}. Зафиксировано (UTC): {PRE['frozen_at_utc']}. Вариант: {v['name']}; "
              f"{v['formula']}; q — {v['q']}; p — {v['p']}. Параметры 1X2: {[round(x, 4) for x in v['theta_1x2']]}, "
              f"тотал 2.5: {[round(x, 4) for x in v['theta_ou25']]} ({v['theta_order']}; обучены на {v['fitted_on']})."))
    rep.add(P(f"Выборка вперёд: {PRE['forward_sample']['matches']}; минимум — {PRE['forward_sample']['minimum']}. "
              f"Критерий: {PRE['primary_criterion']['metric']}; успех — {PRE['primary_criterion']['pass']}. "
              f"Решение: {PRE['decision_rule']}."))


def tail(rep):
    rep.add(H1("14. Отрицательные результаты"))
    rep.add(bullets([
        "H1 отвергнута: ни на одном рынке и ни в одном сезоне модель не точнее Bet365 открытия; ни одна лига не значима после "
        "поправки Холма.",
        "H2 отвергнута на сезонах выбора: рынок охватывает модель.",
        "H3 отвергнута: гибриды не лучше рынка значимо; лучшие из них (H2, H3, H5) совпадают с рынком с точностью до "
        "десятых долей 10⁻³, H4 (мета-модель с лигами) хуже рынка — переобучение.",
        "H4 отвергнута: ставки по модели убыточны, CLV отрицателен; гибриды почти не дают ставок.",
    ]))
    rep.add(H1("15. Ограничения"))
    rep.add(bullets([
        "Для прошлых сезонов цены есть только для 1X2, тотала 2.5 и одной азиатской линии Football-Data; остальные рынки "
        "сравнены только в 2026/27 по закрытию OddsPapi.",
        "Снимок «открытия» Football-Data — пятница/вторник, а не момент генерации прогноза.",
        "2026/27 и 2025/26 использовались ранее; их результаты — для информации. Независима только проверка вперёд.",
        "Pinnacle в Football-Data есть до 2025/26 (частично), биржа Betfair — с 2024/25.",
    ]))
    rep.add(H1("16. Выводы"))
    rep.add(P("Цена букмекера (особенно без маржи методом, учитывающим перекос фаворит–аутсайдер) — лучший доступный "
              "прогноз. Модель FootPredict систематически уступает ей и не добавляет к ней значимой информации; поэтому EV, "
              "посчитанный по модели, отражает в основном ошибку модели. Исследование времени цены показывает, что модель "
              "слегка предсказывает движение линии (несколько процентов разрыва), но этого недостаточно, чтобы перекрыть маржу."))
    rep.add(H1("17. Рекомендации (A / B / C)"))
    rows = [["Решение", "Кат.", "Обоснование"],
            ["Показывать рядом с вероятностью модели вероятность рынка без маржи (степенной метод/Шин) и разницу", "A",
             "Разделы 7–8: рынок точнее; пользователь видит, где модель расходится с ценой"],
            ["Перевести EV Main/Risk на пул H2 модели и Bet365 открытия", "B",
             "Только после успешной проверки вперёд по предрегистрации (раздел 13)"],
            ["Заменить вероятность модели на гибрид H1/H3/H4/H5", "C", "Раздел 10: без значимого выигрыша"],
            ["Ставки по EV модели (value betting)", "C", "Раздел 11: ROI и CLV отрицательны"],
            ["Специализация по лигам/диапазонам коэффициентов", "C", "Разделы 8.1–8.2: значимых зон нет"]]
    rep.add(T(rows, [8.5, 1.2, 7.3]))
    rep.add(H1("18. Воспроизведение"))
    rep.add(reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                       "# walk-forward кэш v6.1: $PY -m experiments.v6_1.main.run --base --candidates",
                       "bash experiments/v6_2/bookmaker/launch.sh run      # фон, лог cache/run.log, маркер cache/run.done",
                       "$PY -m experiments.v6_2.bookmaker.preregister      # затем коммит results/preregistration.json",
                       "$PY -m experiments.v6_2.bookmaker.build_pdf"]))


def findings() -> str:
    h = HY["results"]
    lines = ["# FootPredict v6.2 — модель против букмекера: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: `reports/v6_2/Bookmaker_Benchmark_Research.pdf`.",
             "", "## Модель против рынка (log loss модель − рынок, ×10⁻³, парный бутстреп)", "",
             "| Рынок | Сезон | vs Bet365 откр. | vs Pinnacle откр. | vs Bet365 закр. | vs Betfair закр. |",
             "| --- | --- | --- | --- | --- | --- |"]
    for m in ("1X2", "OU25", "AH"):
        for s in SEAS:
            cells = [dll(BM["model_vs_market"].get(f"{m}|{s}|{r}", {}).get("d_ll")) for r in
                     ("b365_open", "ps_open", "b365_close", "bfe_close")]
            lines.append(f"| {MK[m]} | {sl(s)} | " + " | ".join(cells) + " |")
    lines += ["", "## Охват (вес модели b₂ рядом с Bet365 открытия)", ""]
    for m in ("1X2", "OU25", "AH"):
        for s in ("select", 2025, 2026):
            v = ENC["tests"].get(f"{m}|{s}|b365_open")
            if v:
                lines.append(f"- {MK[m]} {s}: b₂ = {v['b_model']:+.3f} [{v['b_model_ci95'][0]:+.3f}; "
                             f"{v['b_model_ci95'][1]:+.3f}], p = {v['p_lr']:.3f}")
    lines += ["", "## Лучший гибрид против рынка H0 (Δ log loss ×10⁻³)", ""]
    for k, v in h.items():
        best = min((nm for nm in ("H1", "H2", "H3", "H4", "H5") if nm in v), key=lambda nm: v[nm]["ll"])
        d = v[best]["d_ll_vs_H0"]
        lines.append(f"- {k}: {HYB[best]} {dll(d)}, p = {d['p_two_sided']:.3f}, p Холма = {f(v[best].get('p_holm'), 3)}")
    lines += ["", "## Решения", "", "- A: показывать вероятность рынка без маржи рядом с моделью.",
              "- B: EV Main/Risk по пулу H2 — только после проверки вперёд (`results/preregistration.json`).",
              "- C: гибриды H1/H3/H4/H5, value betting по модели, специализация по лигам/коэффициентам.", "",
              "## Воспроизведение", "", "```", "bash experiments/v6_2/bookmaker/launch.sh run",
              "$PY -m experiments.v6_2.bookmaker.preregister", "$PY -m experiments.v6_2.bookmaker.build_pdf", "```", ""]
    return "\n".join(lines)


def run():
    rep = Report62(OUT, "Модель против букмекера", "FootPredict v6.2 — сравнение с оценкой букмекера",
                   facts={"Сезоны выбора": "2023/24, 2024/25", "Для информации": "2025/26, 2026/27 (не holdout)",
                          "Код": "experiments/v6_2/bookmaker/"})
    front(rep)
    intro(rep)
    methods(rep)
    demargin(rep)
    vs_market(rep)
    encompassing(rep)
    hybrids(rep)
    betting(rep)
    multimarket(rep)
    prereg(rep)
    tail(rep)
    rep.build()
    pages = rep.check()
    (C.HERE / "findings.md").write_text(findings(), encoding="utf-8")
    print("pages", pages)


if __name__ == "__main__":
    run()
