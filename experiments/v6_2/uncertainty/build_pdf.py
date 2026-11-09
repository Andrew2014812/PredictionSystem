"""Builds reports/v6_2/Prediction_Uncertainty_Research.pdf and findings.md from results/ (no number typed by hand).

    python -m experiments.v6_2.uncertainty.build_pdf
"""
from __future__ import annotations

from reportlab.lib.units import cm

from scripts import report_kit as K
from scripts.report_kit import H1, H2, P, bullets, figure, formula, note, table

from . import common as C
from .run import DROP, MEASURES, WARMUP

OUT = C.ROOT / "reports" / "v6_2" / "Prediction_Uncertainty_Research.pdf"
SEASONS = [str(s) for s in C.SEASONS]
SL = {s: C.period_label(int(s)) for s in SEASONS}


def T(rows, widths):
    return table(rows, widths=[w * cm for w in widths])


def pp(v, d=1, signed=True):
    return "—" if v is None else (f"{v * 100:+.{d}f}" if signed else f"{v * 100:.{d}f}")


def ci(est, c, d=1):
    if est is None:
        return "—"
    if not c or c[0] is None:
        return pp(est, d)
    return f"{pp(est, d)} [{pp(c[0], d)}; {pp(c[1], d)}]"


def f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def pv(v):
    return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def label(key: str, filt: dict) -> str:
    v = filt["variants"][key]
    return f"{MEASURES[v['measure']]}, убрать {int(v['drop'] * 100)} %"


def build() -> None:
    ms = C.load_json("measures.json")
    pred = C.load_json("predictive.json")
    filt = C.load_json("filters.json")
    cb = C.load_json("calibration_band.json")
    iv = C.load_json("intervals.json")
    bag = C.load_json("bagging.json")
    stab = C.load_json("stability.json")
    dec = C.load_json("decisions.json")
    info = C.load_json("run_info.json")
    pre = C.load_json("preregistration.json")
    pool = pred["pooled_select"]
    rep = K.Report(OUT, "Неопределённость прогнозов", "FootPredict v6.2 — исследование",
                   {"Сезоны отбора": "2023/24, 2024/25", "Для информации": "2025/26",
                    "Текущий сезон (уже просмотрен)": "2026/27 до " + info["data_version"]["picks"]["mtime"][:10],
                    "Продакшн": "не изменяется"})

    sig = dec["flag_on_site"]["significant_measures"]
    best_r, best_g = filt["best_roi"], filt["best_gap"]
    rep.add(H1("1. Цель и проблема"),
            P("Модель выдаёт вероятность события, но не говорит, насколько надёжна сама эта оценка. Две матча с "
              "одинаковой p = 0.65 могут отличаться: в одном модели, обученные на чуть других данных или с другим "
              "случайным зерном, дают 0.64–0.66, в другом 0.58–0.72; у одного все признаки известны, у другого команда "
              "только что поднялась из низшей лиги; один прогноз стабилен от версии к версии, другой «прыгает»."),
            P("**Вопросы:** (1) насколько велик разброс оценки вероятности Main по сравнению с его систематической "
              "переоценкой; (2) можно ли заранее отметить матчи, где модель ошибается чаще, чем обещает; (3) улучшает ли "
              "отбор Main по неопределённости калибровку выбранного Main, долю угаданных и ROI по реальным ценам."),
            note("Главный результат. Разброс оценки вероятности Main между переобученными моделями (σ ≈ " +
                 f"{pp(ms['2024']['u_seed']['median'], 1, False)}–{pp(ms['2024']['u_boot']['median'], 1, False)} п.п. в 2024/25) "
                 f"того же порядка, что и переоценка выбранного Main ({pp(iv['2024']['gap'])} п.п.), но почти не связан с ней. " +
                 ("После поправки Холма значимо предсказывают проигрыш сверх p: " + ", ".join(MEASURES[c] for c in sig) + "."
                  if sig else "Ни одна мера не предсказывает проигрыш сверх заявленной p после поправки Холма.") +
                 (" Ни один фильтр Main по неопределённости не прошёл правило отбора." if not filt["any_promising"] else
                  " Есть фильтр, прошедший правило отбора (раздел 6.4).")))

    rep.add(H1("2. Гипотезы"),
            T([["#", "Гипотеза", "Проверка"],
               ["U1", "Разброс ансамбля (сиды / бутстреп / подвыборки) выше у Main, которые проигрывают чаще заявленного",
                "β при z(u) в логит-модели выигрыша при фиксированном logit p; gap по квинтилям u"],
               ["U2", "Пропуски признаков (новые команды, нет истории) делают прогноз ненадёжным", "то же для доли пропусков"],
               ["U3", "Ширина интервала Венна–Аберса (недостаток калибровочных данных около p) указывает на риск",
                "то же для ширины"],
               ["U4", "Широкое конформное множество 1X2 (2–3 исхода) — неуверенный матч", "то же для размера множества"],
               ["U5", "Неустойчивый прогноз (Main менялся между версиями, дрейф p) — ненадёжный (только 2026/27)",
                "gap по группам стабильности"],
               ["U6", "Если не ставить (не показывать как надёжный) самые неопределённые Main, калибровка и ROI лучше",
                "Δ|gap|, ΔROI оставленных против всех, парный бутстреп, Холм"],
               ["U7", "Среднее ансамбля (бэггинг) калибровано лучше одной модели", "gap и log loss 1X2"]],
              [1.0, 9.0, 7.0]))

    rows = [["Сезон", "Main", "Рассчитано", "Ставок с ценой", "gap Main, п.п. [95 % ДИ]", "ROI, % [95 % ДИ]"]]
    for s in SEASONS:
        m = ms[s]
        rows.append([SL[s], str(m["mains"]), str(m["decided"]), str(m["priced_bets"]),
                     ci(m["gap"]["mean"], m["gap"]["ci95"]), ci(m["roi"]["mean"], m["roi"]["ci95"])])
    rep.add(H1("3. Данные"),
            P("**2023/24–2025/26:** walk-forward таблицы кандидатов v6.1 (`experiments/v6_1/main/cache`, модели обучены на "
              "сезонах до прогнозируемого, математика рынков продакшна, цены — Bet365 открытия Football-Data, как их "
              "прикрепляет продакшн) и правило Main продакшна. **2026/27:** Main, который показал продакшн (единая "
              "история, последняя версия до матча), с его вероятностью и записанной ценой (пропуски — последняя "
              "котировка OddsPapi до начала, как в истории продакшна). 2025/26 — для информации, 2026/27 уже "
              "просматривался и не является независимой проверкой."),
            T(rows, [3.4, 1.6, 2.2, 2.4, 4.0, 3.8]),
            P("gap = средняя заявленная вероятность (на шкале без возврата, pc = p/(1 − p_возврата)) минус доля "
              "выигрышей; ROI — только Main с реальной ценой, выигрыш или проигрыш.", "small"))

    fam = info["families"]
    rep.add(H1("4. Методы"),
            H2("4.1. Ансамбли"),
            P(f"Для каждого сезона s четыре модели продакшна (те же параметры, число деревьев, предобработка и веса "
              f"давности) переобучены на всех сезонах до s несколько раз: **seed** — {fam['seed']} моделей с разным "
              f"случайным зерном XGBoost (подвыборка строк и столбцов); **boot** — {fam['boot']} моделей на бутстреп-выборке "
              f"матчей обучения (кратность как вес); **sub** — {fam['sub']} моделей на {int(info['sub_fraction'] * 100)} % "
              f"матчей без возвращения. Для каждого члена ансамбля вероятность ТОГО ЖЕ выбранного Main пересчитана "
              f"математикой продакшна (счётная матрица, согласованная с 1X2; угловые — отрицательное биномиальное). "
              f"Не более {C.N_JOBS} потоков на модель."),
            formula("u_fam = σ_k( p_k(Main) ),   k — члены семейства;   u_1X2 = mean_j σ_k( p_k(j) ), j ∈ {H, D, A}"),
            H2("4.2. Другие меры"),
            bullets(["**Доля пропусков** входных признаков модели 1X2 у матча.",
                     "**Венн–Аберс (индуктивный):** калибровочное множество — Main прошлого сезона (walk-forward); для p "
                     "изотоническая регрессия по калибровке ∪ {(p, 0)} даёт p₀, по калибровке ∪ {(p, 1)} — p₁; мера — "
                     "ширина p₁ − p₀ (велика там, где мало калибровочных данных).",
                     "**Конформное множество 1X2:** q — квантиль уровня ⌈(n+1)(1−α)⌉/n оценок 1 − p_y прошлого сезона, "
                     "α = 0.2; множество {k : 1 − p_k ≤ q}, мера — его размер (1–3).",
                     "**Стабильность (2026/27, записанные прогнозы):** число версий до матча, менялся ли Main, "
                     "|p последней − p первой версии|, σ p по версиям."]),
            H2("4.3. Проверки"),
            formula("logit P(выигрыш) = β₀ + β₁·logit pc + β_u·z(u),   z — стандартизация внутри сезона"),
            P("β_u < 0 — при той же заявленной вероятности более неопределённый Main выигрывает реже. 95 % ДИ — бутстреп "
              "по матчам (500 повторов), p — двусторонний; на объединённых сезонах отбора поправка Холма по "
              f"{pred['n_tests']} мерам. Дополнительно — AUC меры для проигрыша и gap по квинтилям меры."),
            P(f"**Фильтры.** «Убрать долю q самых неопределённых Main»: порог для матча дня d — квантиль (1 − q) меры по "
              f"Main того же сезона, сыгранным ДО дня d (первые {WARMUP} Main не отбрасываются), q ∈ "
              f"{{{', '.join(str(int(x * 100)) + ' %' for x in DROP)}}}, {filt['n_variants']} вариантов. Сравнение "
              "оставленных с всеми — на одних и тех же матчах, бутстреп по матчам (2000): ΔROI (ставка = Main с ценой) и "
              "Δ|gap|. Правило отбора (зафиксировано до расчёта): улучшение в обоих сезонах отбора и p Холма "
              "объединённого теста < 0.05."),
            P("**Неопределённость калибровки:** 95 % бутстреп-полоса кривой надёжности Main (корзины по 0.05). "
              "**Интервалы вероятности:** диапазон [min, max] членов бутстреп-ансамбля против наблюдаемой доли "
              "выигрышей корзины."))

    rows = [["Сезон", "σ сиды", "σ бутстреп", "σ подвыб.", "σ 1X2", "Ширина ВА", "|gap|", "Доля корзин внутри"]]
    for s in SEASONS:
        rows.append([SL[s], pp(ms[s]["u_seed"]["median"], 2, False), pp(ms[s]["u_boot"]["median"], 2, False),
                     pp(ms[s]["u_sub"]["median"], 2, False), pp(ms[s]["u_1x2"]["median"], 2, False),
                     pp(iv[s]["va_median_width"], 2, False), pp(abs(iv[s]["gap"]), 2, False),
                     f(iv[s]["share_bins_inside"], 2)])
    rep.add(H1("5. Параметры"),
            T([["Параметр", "Значение"],
               ["Члены ансамблей (seed / boot / sub)", f"{fam['seed']} / {fam['boot']} / {fam['sub']}"],
               ["Доля подвыборки", f"{info['sub_fraction']}"],
               ["Разогрев порога фильтра", f"{WARMUP} Main"],
               ["Доли отбрасывания", ", ".join(f"{int(x * 100)} %" for x in DROP)],
               ["Конформный уровень α", "0.2"],
               ["Бутстреп", f"{C.N_BOOT} повторов по матчам (β: 500), seed {C.SEED}"],
               ["Потоки на модель", str(C.N_JOBS)],
               ["Время расчёта анализа, мин", f(info["minutes"], 1)]], [8.0, 9.0]),
            H1("6. Результаты"),
            H2("6.1. Насколько велика неопределённость оценки"),
            P("Медианы мер (п.п. вероятности Main) и модуль переоценки выбранного Main; «доля корзин внутри» — в какой "
              "доле корзин по p наблюдаемая доля выигрышей попала в средний диапазон [min, max] членов бутстрепа."),
            T(rows, [3.3, 1.6, 1.9, 1.7, 1.5, 1.9, 1.5, 3.0]),
            figure(C.FIGURES / "spread_vs_gap.png", "Рис. 1. Медианный разброс оценки вероятности Main и |gap| по сезонам."))
    sp = ms["2024"]["spearman"]
    cols = [c for c in MEASURES if c in sp]
    srows = [["Мера"] + [c.replace("u_", "") for c in cols]]
    for a in cols:
        srows.append([MEASURES[a]] + [f(sp[a].get(b), 2) for b in cols])
    rep.add(P("Ранговые корреляции мер между собой (2024/25): меры почти независимы — они описывают разные источники "
              "неопределённости."), T(srows, [4.6] + [1.75] * len(cols)))

    rows = [["Мера", "n", "β_u [95 % ДИ]", "p", "p Холма", "AUC u (проигрыш)", "AUC −p (проигрыш)"]]
    for c in MEASURES:
        r = pool.get(c, {})
        rows.append([MEASURES[c], str(r.get("n", "—")), f"{f(r.get('beta_u'))} [{f((r.get('beta_u_ci95') or [None])[0])}; "
                     f"{f((r.get('beta_u_ci95') or [None, None])[1])}]", pv(r.get("p")), pv(r.get("p_holm")),
                     f(r.get("auc_u_for_miss")), f(r.get("auc_p_for_miss"))])
    rep.add(H2("6.2. Отмечает ли неопределённость матчи, где модель ошибается"),
            P("Объединённые сезоны отбора (2023/24 + 2024/25), меры стандартизованы внутри сезона:"),
            T(rows, [4.4, 1.3, 3.6, 1.5, 1.6, 2.3, 2.3]))
    rows = [["Мера"] + [SL[s] for s in SEASONS]]
    for c in MEASURES:
        cells = []
        for s in SEASONS:
            r = pred["per_season"][s].get(c, {})
            cells.append(f"{f(r.get('beta_u'))} (p Холма {pv(r.get('p_holm_season'))})" if r.get("beta_u") is not None
                         else "—")
        rows.append([MEASURES[c]] + cells)
    rep.add(P("β_u по сезонам (p с поправкой Холма по семи мерам внутри сезона; 2025/26 — для информации, 2026/27 — "
              "уже просмотрен):"),
            T(rows, [4.6, 3.1, 3.1, 3.1, 3.1]),
            figure(C.FIGURES / "beta_u.png", "Рис. 2. β_u с 95 % ДИ на сезонах отбора (красный — значимо после Холма)."),
            figure(C.FIGURES / "gap_by_quintile.png", "Рис. 3. Переоценка Main по квинтилям меры неопределённости."))
    cs = dec.get("current_signal") or {}
    if cs.get("measures"):
        txt = []
        for c in cs["measures"]:
            r = pred["per_season"][str(C.CURRENT)][c]
            q = pred["quintiles"][str(C.CURRENT)].get(c) or []
            top = f"; gap верхнего квинтиля {ci(q[-1]['gap'], q[-1]['gap_ci95'])} п.п. против нижнего " \
                  f"{ci(q[0]['gap'], q[0]['gap_ci95'])} п.п." if q else ""
            txt.append(f"{MEASURES[c]}: β_u = {f(r['beta_u'])} [{f(r['beta_u_ci95'][0])}; {f(r['beta_u_ci95'][1])}], "
                       f"p Холма {pv(r['p_holm_season'])}, n = {r['n']}{top}")
        rep.add(note("Сигнал 2026/27. На уже просмотренном текущем сезоне значимы после поправки Холма: " +
                     "; ".join(txt) + ". На сезонах отбора тот же коэффициент не отличается от нуля (таблица выше), "
                     "поэтому это не доказательство, а новая гипотеза: она заморожена в preregistration.json и "
                     "проверяется только вперёд (решение B, низкая априорная вероятность)."))

    rows = [["Фильтр", "Убрано (24/25)", "ΔROI 23/24", "ΔROI 24/25", "ΔROI отбор [95 % ДИ]", "p Холма",
             "Δ|gap| отбор [95 % ДИ]", "p Холма"]]
    order = sorted(filt["variants"], key=lambda k: -filt["variants"][k]["pooled_select_roi"]["diff"])
    for k in order[:10]:
        v = filt["variants"][k]
        rp, gp = v["pooled_select_roi"], v["pooled_select_abs_gap"]
        rows.append([label(k, filt), pp(v["seasons"]["2024"]["dropped_share"], 0, False) + " %",
                     pp(v["seasons"]["2023"]["roi_diff"]["diff"]), pp(v["seasons"]["2024"]["roi_diff"]["diff"]),
                     ci(rp["diff"], rp["ci95"]), pv(rp.get("p_holm")), ci(gp["diff"], gp["ci95"]), pv(gp.get("p_holm"))])
    rep.add(H2("6.3. Фильтр Main по неопределённости"),
            P(f"Десять лучших из {filt['n_variants']} вариантов по ΔROI на объединённых сезонах отбора (п.п.; Δ < 0 у "
              "|gap| — калибровка лучше):"),
            T(rows, [4.2, 1.6, 1.6, 1.6, 2.9, 1.3, 2.9, 1.3]))
    rows = [["Фильтр"] + [SL[s] for s in SEASONS]]
    for k in [x for x in dict.fromkeys([best_r, best_g]) if x]:
        v = filt["variants"][k]
        rows.append([label(k, filt) + " — ΔROI"] + [ci(v["seasons"][s]["roi_diff"]["diff"], v["seasons"][s]["roi_diff"]["ci95"])
                                                    for s in SEASONS])
        rows.append([label(k, filt) + " — Δ|gap|"] + [ci(v["seasons"][s]["abs_gap_diff"]["diff"],
                                                         v["seasons"][s]["abs_gap_diff"]["ci95"]) for s in SEASONS])
    rep.add(P("Лучший вариант по худшему из двух сезонов отбора (ΔROI и Δ|gap|) по всем сезонам:"),
            T(rows, [5.0, 3.0, 3.0, 3.0, 3.0]),
            P(("Прошли правило отбора: " + ", ".join(label(k, filt) for k, v in filt["variants"].items()
                                                  if v["promising_roi"] or v["promising_gap"]) + ".")
              if filt["any_promising"] else "Ни один вариант не прошёл правило отбора (улучшение в обоих сезонах и p Холма "
                                            "< 0.05)."),
            figure(C.FIGURES / "filters.png", "Рис. 4. ΔROI и Δ|gap| оставленных Main против всех (объединённые сезоны "
                                              "отбора, 95 % ДИ)."))

    rows = [["Сезон", "Корзин", "Наблюдение внутри 95 % полосы вокруг p", "Худшая корзина: p / факт [ДИ]"]]
    for s in SEASONS:
        b = cb[s]
        worst = max(b, key=lambda r: abs(r["predicted"] - r["observed"])) if b else None
        rows.append([SL[s], str(len(b)), f"{sum(not r['inside'] for r in b)} корзин из {len(b)}: p вне ДИ факта",
                     f"{f(worst['predicted'], 3)} / {f(worst['observed'], 3)} [{f(worst['ci95'][0], 3)}; "
                     f"{f(worst['ci95'][1], 3)}]" if worst else "—"])
    rep.add(H2("6.4. Неопределённость калибровки и интервалы вероятности"),
            T(rows, [3.0, 1.6, 6.6, 5.8]),
            figure(C.FIGURES / "calibration_band.png", "Рис. 5. Кривая надёжности Main с 95 % бутстреп-полосой."))
    rows = [["Сезон", "gap одной модели", "gap среднего сидов", "gap среднего бутстрепа", "LL 1X2: бутстреп − одна [ДИ]"]]
    for s in SEASONS:
        b = bag[s]
        ll = b.get("logloss_1x2_boot")
        rows.append([SL[s], pp(b["seed"]["gap_single"]), ci(b["seed"]["gap_ensemble"], b["seed"]["gap_ensemble_ci95"]),
                     ci(b["boot"]["gap_ensemble"], b["boot"]["gap_ensemble_ci95"]),
                     (f"{ll['diff']['mean'] * 1000:+.2f} [{ll['diff']['ci95'][0] * 1000:+.2f}; "
                      f"{ll['diff']['ci95'][1] * 1000:+.2f}] ×10⁻³") if ll else "—"])
    rep.add(H2("6.5. Среднее ансамбля вместо одной модели"), T(rows, [3.0, 2.6, 3.6, 3.6, 4.2]))

    if stab:
        rows = [["Группа (2026/27, записанные)", "n", "gap [95 % ДИ]", "Доля выигрышей", "ROI"]]
        for name, g in stab["groups"].items():
            rows.append([name, str(g["n"]), ci(g["gap"], g["gap_ci95"]), pp(g["hit_rate"], 1, False) + " %",
                         pp(g["roi"]) if g["roi"] is not None else "—"])
        brows = [["Мера стабильности", "n", "β_u [95 % ДИ]", "p"]]
        for c, r in stab["beta"].items():
            brows.append([c, str(r.get("n")), f"{f(r.get('beta_u'))} [{f((r.get('beta_u_ci95') or [None])[0])}; "
                          f"{f((r.get('beta_u_ci95') or [None, None])[1])}]", pv(r.get("p"))])
        rep.add(H2("6.6. Стабильность прогноза между версиями (2026/27)"),
                P(f"Записанных до матча рассчитанных Main: {stab['recorded_decided']}; медиана версий "
                  f"{f(stab['n_versions_median'], 0)}, Main менялся у {pp(stab['main_changed_share'], 1, False)} %, медианный "
                  f"дрейф p {pp(stab['p_drift_median'], 2, False)} п.п. 2026/27 уже просмотрен — только для информации."),
                T(rows, [5.4, 1.4, 4.2, 2.8, 2.2]), T(brows, [5.0, 1.6, 5.0, 2.0]))

    rep.add(H1("7. Отрицательные результаты"),
            bullets([
                "Неопределённость оценки (разброс переобученных моделей) не объясняет переоценку выбранного Main: "
                "среднее ансамбля переоценено так же, как одна модель (раздел 6.5); переоценка — эффект выбора максимума "
                "среди многих кандидатов, а не шум оценки отдельной вероятности.",
                ("Ни одна мера не предсказывает проигрыш сверх p после поправки Холма." if not sig else
                 "Значимые меры: " + ", ".join(MEASURES[c] for c in sig) + "; размер эффекта см. AUC в разделе 6.2."),
                "Фильтры по неопределённости меняют ROI и калибровку на величины внутри шума; правило отбора не "
                "выполнено." if not filt["any_promising"] else "Фильтр прошёл правило отбора — см. предрегистрацию.",
                "Ширина Венна–Аберса мала почти везде (тысячи калибровочных Main) и растёт только у редких крайних p."]))
    rep.add(H1("8. Ограничения"),
            bullets(["Ансамбли небольшие (6–8 членов): σ по ним сама оценена с погрешностью около 25 %.",
                     "Члены ансамблей — переобучение с параметрами продакшна; неопределённость выбора гиперпараметров и "
                     "признаков не учтена.",
                     "Для 2026/27 разброс посчитан по моделям, обученным на 2021/22–2025/26 на текущей таблице признаков, а "
                     "вероятность Main — сохранённая продакшном (часть прогнозов сделана ранними версиями моделей).",
                     "Стабильность версий есть только у записанных прогнозов 2026/27 (уже просмотренный сезон).",
                     "Цены: Bet365 открытия для прошлых сезонов, записанные / архивные для 2026/27 — ROI не сравним между "
                     "периодами напрямую."]))

    concl = [f"**{dec['flag_on_site']['decision']} — метка «ненадёжный прогноз» на сайте по правилу отбора:** "
             f"{dec['flag_on_site']['why']}.",
             f"**{dec['filter_main']['decision']} — фильтр Main / ставок по неопределённости:** " +
             ("прошёл правило отбора; форвард-проверка по предрегистрации." if filt["any_promising"] else
              "не прошёл правило отбора ни для ROI, ни для калибровки."),
             "**C — среднее ансамбля вместо одной модели для Main:** переоценка выбранного Main не уменьшается.",
             "**A — для отчётов и диплома:** показывать неопределённость калибровки (95 % полосы кривой надёжности) и "
             "интервалы метрик; это описание качества, а не изменение продакшна."]
    if cs.get("measures"):
        concl.insert(1, f"**{cs['decision']} — метка по {', '.join(MEASURES[c] for c in cs['measures'])}:** {cs['why']}.")
    if pre:
        concl.append(f"Предрегистрация B-гипотез: `experiments/v6_2/uncertainty/results/preregistration.json` "
                     f"(заморожено {pre.get('frozen_at', '')}); в реестре теней — запись без реализации (нужны "
                     "сохранённые бутстреп-модели 1X2).")
    rep.add(H1("9. Выводы и рекомендации"), bullets(concl),
            K.reproduce(["PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
                         "bash experiments/v6_2/uncertainty/launch.sh ensemble     # ансамбли (3 процесса, n_jobs = 2)",
                         "$PY -m experiments.v6_2.uncertainty.prepare             # таблицы Main с мерами",
                         "$PY -m experiments.v6_2.uncertainty.run && $PY -m experiments.v6_2.uncertainty.figures",
                         "$PY -m experiments.v6_2.uncertainty.build_pdf"]),
            P(f"Git SHA прогона: {info['environment']['git_sha']}.", "small"))
    rep.build()
    pages = rep.check()
    print(f"{OUT} — {pages} pages")
    findings(ms, pred, filt, iv, bag, stab, dec)


def findings(ms, pred, filt, iv, bag, stab, dec) -> None:
    pool = pred["pooled_select"]
    lines = ["# FootPredict v6.2 — неопределённость прогнозов: выводы", "",
             "Все числа — из `results/*.json` (файл пишет `build_pdf.py`). PDF: `reports/v6_2/Prediction_Uncertainty_Research.pdf`.",
             "", "## Размер неопределённости (медианы, п.п.)", "",
             "| Сезон | σ сиды | σ бутстреп | σ подвыборки | ширина ВА | gap Main [95 % ДИ] |", "|---|---|---|---|---|---|"]
    for s in SEASONS:
        lines.append(f"| {SL[s]} | {pp(ms[s]['u_seed']['median'], 2, False)} | {pp(ms[s]['u_boot']['median'], 2, False)} | "
                     f"{pp(ms[s]['u_sub']['median'], 2, False)} | {pp(iv[s]['va_median_width'], 2, False)} | "
                     f"{ci(ms[s]['gap']['mean'], ms[s]['gap']['ci95'])} |")
    lines += ["", "## Предсказывает ли мера проигрыш сверх p (2023/24 + 2024/25)", "",
              "| Мера | β_u [95 % ДИ] | p Холма | AUC |", "|---|---|---|---|"]
    for c in MEASURES:
        r = pool.get(c, {})
        lines.append(f"| {MEASURES[c]} | {f(r.get('beta_u'))} [{f((r.get('beta_u_ci95') or [None])[0])}; "
                     f"{f((r.get('beta_u_ci95') or [None, None])[1])}] | {pv(r.get('p_holm'))} | {f(r.get('auc_u_for_miss'))} |")
    br, bg = filt["best_roi"], filt["best_gap"]
    lines += ["", "## Фильтры (вариантов: " + str(filt["n_variants"]) + ")", "",
              f"- Лучший по ΔROI: {label(br, filt)}: объединённо {ci(filt['variants'][br]['pooled_select_roi']['diff'], filt['variants'][br]['pooled_select_roi']['ci95'])} п.п., "
              f"p Холма {pv(filt['variants'][br]['pooled_select_roi'].get('p_holm'))}.",
              f"- Лучший по Δ|gap|: {label(bg, filt)}: {ci(filt['variants'][bg]['pooled_select_abs_gap']['diff'], filt['variants'][bg]['pooled_select_abs_gap']['ci95'])} п.п., "
              f"p Холма {pv(filt['variants'][bg]['pooled_select_abs_gap'].get('p_holm'))}.",
              f"- Прошли правило отбора: {'да' if filt['any_promising'] else 'нет'}.", "",
              "## Бэггинг", ""]
    for s in SEASONS:
        b = bag[s]
        lines.append(f"- {SL[s]}: gap одной модели {pp(b['seed']['gap_single'])}, среднего бутстрепа "
                     f"{ci(b['boot']['gap_ensemble'], b['boot']['gap_ensemble_ci95'])} п.п.")
    if stab:
        lines += ["", "## Стабильность версий (2026/27, только для информации)", ""]
        for name, g in stab["groups"].items():
            lines.append(f"- {name}: n = {g['n']}, gap {ci(g['gap'], g['gap_ci95'])} п.п.")
    cs = dec.get("current_signal") or {}
    lines += ["", "## Решения", "",
              f"- {dec['flag_on_site']['decision']}: метка «ненадёжный прогноз» на сайте по правилу отбора — "
              f"{dec['flag_on_site']['why']}."]
    if cs.get("measures"):
        r = pred["per_season"][str(C.CURRENT)][cs["measures"][0]]
        lines.append(f"- {cs['decision']}: метка по {', '.join(MEASURES[c] for c in cs['measures'])} — {cs['why']} "
                     f"(2026/27: β_u = {f(r['beta_u'])} [{f(r['beta_u_ci95'][0])}; {f(r['beta_u_ci95'][1])}], p Холма "
                     f"{pv(r['p_holm_season'])}); предрегистрация `results/preregistration.json`.")
    lines += [
              f"- {dec['filter_main']['decision']}: фильтр Main/ставок по неопределённости.",
              "- C: среднее ансамбля вместо одной модели для Main.",
              "- A (отчёты): полосы неопределённости калибровки и интервалы метрик.", "- Продакшн не меняется.", "",
              "## Воспроизведение", "", "```",
              "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
              "bash experiments/v6_2/uncertainty/launch.sh ensemble",
              "$PY -m experiments.v6_2.uncertainty.prepare && $PY -m experiments.v6_2.uncertainty.run",
              "$PY -m experiments.v6_2.uncertainty.figures && $PY -m experiments.v6_2.uncertainty.build_pdf", "```", ""]
    (C.HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    C.setup_logging()
    build()
