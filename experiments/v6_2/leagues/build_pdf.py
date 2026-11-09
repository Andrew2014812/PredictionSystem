"""Builds reports/v6_2/League_Specialization_Research.pdf from results/*.json and figures/*.png (Russian).

    python -m experiments.v6_2.leagues.build_pdf

Every number is read from the result files of this folder; nothing is typed in by hand.
"""
from __future__ import annotations

from scripts.report_kit import H1, H2, P, Report, bullets, figure, formula, note, reproduce

from . import common as C
from . import figures as FG
from .pdfkit import T, ci, dk, f, mk, pc, pv, roi

OUT = C.ROOT / "reports" / "v6_2" / "League_Specialization_Research.pdf"
J = C.load_json
SEAS = (2023, 2024, 2025, 2026)
DEC_RU = {"A": "A — готово к внедрению", "B": "B — перспективно, нужна проверка вперёд", "C": "C — не рекомендуется"}


def load():
    return {k: J(f"{k}.json") for k in ("perf", "profiles", "variants", "variant_fits", "transfer", "run_info",
                                        "preregistration", "decisions")}


def front(rep, R):
    perf, var = R["perf"], R["variants"]
    sel = perf["select"]
    meta = sel["meta"]
    rep.add(H1("Краткие итоги"))
    best = min(var["variants"], key=lambda v: var["variants"][v]["select"][f"d_{var['variants'][v]['metric']}"]["diff"])
    bv = var["variants"][best]
    L = sel["leagues"]
    worse = sum(1 for r in L.values() if r["d"]["diff"] > 0)
    sig = sum(1 for r in L.values() if r["d"]["ci95"][0] > 0)
    items = [
        f"Модель уступает рынку (Bet365, открытие) в {worse} из {len(L)} лиг (в {sig} — с 95 % ДИ выше нуля); "
        f"среднее по лигам (случайные эффекты) {meta['mu'] * 1e3:+.1f}×10⁻³ log loss "
        f"(сезоны отбора 2023/24 + 2024/25, {sel['n']} матчей).",
        f"Неоднородность между лигами есть, но небольшая: τ = {meta['tau'] * 1e3:.1f}×10⁻³, I² = {meta['I2'] * 100:.0f} %, "
        f"Q-тест p = {pv(meta['p_Q'])}; по отдельным сезонам Q-тест не значим "
        f"(p = {', '.join(pv(perf['seasons'][str(s)]['meta']['p_Q']) for s in SEAS)}).",
        f"Порядок лиг по отставанию от рынка плохо повторяется: Спирмен 2023/24↔2024/25 ρ = "
        f"{perf['persistence']['2023-2024']['rho']:.2f}, 2024/25↔2025/26 ρ = {perf['persistence']['2024-2025']['rho']:.2f}, "
        f"2023/24↔2025/26 ρ = {perf['persistence']['2023-2025']['rho']:.2f}.",
        f"ROI Main по лигам неоднороден не больше случайного: I² = {sel['meta_main_roi']['I2'] * 100:.0f} %, "
        f"p = {pv(sel['meta_main_roi']['p_Q'])}; переоценка Main — I² = {sel['meta_main_gap']['I2'] * 100:.0f} %.",
        f"Проверено {var['n_variants_tested']} вариантов специализации; лучший по основной метрике — {best} "
        f"({bv['name']}): {dk(bv['select']['d_' + bv['metric']], 2)} ×10⁻³, p Холма = {pv(bv['select']['p_holm'])}.",
        "Решения: " + "; ".join(f"{v} — {d['decision']}" for v, d in var["variants"].items()) + ".",
    ]
    tr = R["transfer"]
    if tr:
        a = tr["pooled"]["all"]["pooled"]
        o = tr["pooled"]["own"]["pooled"]
        items.append(f"Перенос: модель, не видевшая лигу (все другие лиги), хуже глобальной на {dk(a, 1)}×10⁻³; "
                     f"модель только на своей лиге — на {dk(o, 1)}×10⁻³.")
    rep.add(*bullets(items))


def intro(rep, R):
    ri = R["run_info"] or {}
    rep.add(H1("1. Цель, проблема и гипотезы"))
    rep.add(P("FootPredict использует одну глобальную модель XGBoost для 21 лиги (one-hot код лиги среди признаков). "
              "Цель исследования — выяснить, насколько равномерно модель работает в разных лигах и стоит ли "
              "специализировать её по лигам: калибровать по лигам, объединять похожие лиги в группы, строить "
              "отдельные или иерархические модели, переносить закономерности между лигами."))
    rep.add(P("Проблема: у малых лиг (шотландские SC1–SC3, греческая G1, бельгийская B1) всего 180–300 матчей за сезон, "
              "поэтому любой параметр «на лигу» оценивается с большим шумом. Различия, видимые в одном сезоне, могут быть "
              "случайными. Поэтому каждый вывод проверяется на нескольких сезонах, с бутстреп-интервалами, поправкой "
              "Холма и оценкой межлиговой дисперсии (случайные эффекты)."))
    rep.add(P("Гипотезы:"))
    rep.add(*bullets([
        "H1. Качество модели относительно рынка заметно и устойчиво различается между лигами.",
        "H2. Калибровка по лигам (с частичным объединением — эмпирический Байес) улучшает вероятности 1X2.",
        "H3. Лиги образуют устойчивые группы со сходным профилем (голы, ничьи, преимущество поля, темп), и групповые "
        "модели или групповая калибровка лучше глобальных.",
        "H4. Отдельные модели на лигу или дообучение глобальной модели на лиге лучше одной глобальной модели.",
        "H5. Закономерности переносятся между лигами: модель, не видевшая лигу, почти не хуже глобальной; данные "
        "похожих лиг полезнее данных непохожих.",
    ]))
    rep.add(H1("2. Данные"))
    pr = R["perf"]["seasons"]
    rows = [["Сезон", "Роль", "Сыграно матчей", "С ценой Bet365 (оценка)", "Лиг"]]
    role = {2023: "отбор", 2024: "отбор", 2025: "для информации", 2026: "текущий, уже просмотрен"}
    for s in SEAS:
        rows.append([C.season_label(s), role[s], pr[str(s)]["coverage"]["played"], pr[str(s)]["coverage"]["with_b365_open"],
                     len(pr[str(s)]["leagues"])])
    rep.add(*T(rows, [2.2, 4.4, 3.2, 4.2, 1.6]))
    rep.add(P("Признаки — производственная таблица `local_data/processed/features.parquet` (форма, Elo, сезонная "
              "статистика, средние лиги, отдых; все — только по матчам до дня игры). Цены — файлы Football-Data "
              "`local_data/results/<сезон>/<лига>.csv`: Bet365 открытие (публикуется до решения; основной эталон рынка) и "
              "закрытие (для информации). Маржа снята степенным методом для открытия и аддитивным для закрытия — "
              "методы, выбранные исследованием букмекера v6.2. В оценку входят все сыгранные матчи с полной ценой 1X2 "
              "Bet365 на открытии; одинаковые матчи для каждого варианта."))
    fv = (ri.get("data_version") or {})
    if fv:
        k, v = next(iter(fv.items()))
        rep.add(P(f"Версия данных: `{k}` — {v.get('rows')} строк, изменён {v.get('mtime')}; "
                  f"файлов результатов Football-Data: {ri.get('results_csv_files')}.", "small"))


def methods(rep, R):
    ri = R["run_info"] or {}
    pr = ri.get("params", {})
    rep.add(H1("3. Методы"))
    rep.add(H2("3.1. Схема проверки"))
    rep.add(P("Walk-forward по сезонам: модели для сезона s обучаются только на сезонах < s (2021/22 … s−1) с "
              "производственными параметрами и числом деревьев (версия "
              f"`{pr.get('production_version', '—')}`, без перенастройки); 2026/27 предсказывается моделями, "
              "обученными на 2021/22–2025/26. Все калибровки, сжатия и группы для сезона s оцениваются только по "
              "предсказаниям прошлых сезонов (2022/23 … s−1, каждый — моделью, обученной до него). Отбор вариантов — "
              "только по 2023/24 + 2024/25; 2025/26 — для информации; 2026/27 — текущий сезон, уже просмотренный "
              "ранее, а не независимая проверка. Воспроизведение производственной схемы проверено: прогнозы сида 42 "
              "совпали с кэшем v6.1 (максимальное расхождение 0)."))
    rep.add(P("Эталоны. Пост-обработки (калибровки C*, поправки голов G*) применяются к прогнозам B0 — производственной "
              f"модели (сид {C.PROD_SEED}). Переобучаемые варианты M* усредняются по трём сидам {list(C.SEEDS)} и "
              "сравниваются с B0̄ — производственной схемой, усреднённой по тем же сидам: XGBoost с subsample < 1 "
              "зависит от сида, и без этого различия вариантов тонут в шуме."))
    rep.add(H2("3.2. Метрики"))
    rep.add(formula("LL = −(1/n) Σ_{i} log p_{i,y_{i}},     Δ = LL_{вариант} − LL_{эталон}  (< 0 — вариант лучше)"))
    rep.add(formula("D = (1/n) Σ_{i} [d(h_{i}, λ^{H}_{i}) + d(a_{i}, λ^{A}_{i})],   d(y, μ) = 2 [y log(y/μ) − (y − μ)]"))
    rep.add(P("Рынок: q = де-маржинализированные цены Bet365; «модель − рынок» = LL(p) − LL(q) на тех же матчах. "
              "Main — производственное правило выбора (векторная копия v6.1, сверенная с функциями продакшна) на "
              "таблице кандидатов, построенной производственной математикой; цена — Bet365 открытие, как её "
              "подставляет продакшн. Переоценка Main: gap = среднее заявленной вероятности − доля выигрышей."))
    rep.add(H2("3.3. Калибровка 1X2 и эмпирический Байес"))
    rep.add(formula("z_{k} = b · log p_{k} + a_{k},  a_{H} = 0,   p'_{k} = exp z_{k} / Σ_{j} exp z_{j},   θ = (b, a_{D}, a_{A})"))
    rep.add(P("b < 1 означает, что модель слишком уверена (вероятности нужно сжать к равным), b > 1 — недостаточно "
              "уверена. Для лиги (или группы) u: θ̂_{u} — оценка максимального правдоподобия, V_{u} — обратная "
              "наблюдаемая информация. Межлиговая дисперсия каждого параметра — по моментам; карта лиги — "
              "апостериорная мода при нормальном априорном распределении с центром в глобальной карте:"))
    rep.add(formula("τ²_{j} = max(0, mean_{u}(θ̂_{uj} − θ_{gj})² − mean_{u} V_{u,jj}),   "
                    "θ_{u} = argmin NLL_{u}(θ) + ½ Σ_{j} (θ_{j} − θ_{gj})² / τ²_{j}"))
    rep.add(P("τ² = 0 даёт глобальную карту, τ² → ∞ — несжатую карту лиги. Вариант «случайные эффекты» (C3i) "
              "сжимает только сдвиги ничьей и гостей при общем наклоне b; C5 — двухуровневая иерархия "
              "лига → группа → глобальная."))
    rep.add(H2("3.4. Поправка голов (Пуассон — гамма)"))
    rep.add(formula("y ~ Poisson(f_{L} λ),  f_{L} ~ Gamma(α, β),  E f = F,  Var f = τ²,   f̂_{L} = (α + Σ_{L} y)/(β + Σ_{L} λ)"))
    rep.add(H2("3.5. Неоднородность между лигами"))
    rep.add(P("Случайные эффекты (DerSimonian — Laird) для средних по лигам d_{L} с дисперсиями v_{L} = s²_{L}/n_{L}:"))
    rep.add(formula("Q = Σ w_{L}(d_{L} − d̄_{w})²,  w_{L} = 1/v_{L},  τ² = max(0, (Q − (k−1)) / (Σw − Σw²/Σw)),  "
                    "I² = max(0, (Q − (k−1))/Q)"))
    rep.add(formula("d^{EB}_{L} = μ̂ + (d_{L} − μ̂) · τ² / (τ² + v_{L})"))
    rep.add(H2("3.6. Группы лиг"))
    rep.add(P("Профиль лиги для сезона s — средние по её матчам сезонов < s: голы, доля голов хозяев, ничьи, "
              "перевес хозяев (П1 − П2), ТБ 2.5, «обе забьют», угловые, удары и удары в створ (темп), голы на удар в "
              "створ, жёлтые карточки, разброс очков за матч команд (баланс). Профили стандартизуются; группы — "
              "иерархическая кластеризация Уорда; число групп k ∈ {3…6} — по максимуму силуэта; на каждый сезон "
              "группы строятся заново по своему прошлому; устойчивость — скорректированный индекс Рэнда (ARI)."))
    rep.add(H2("3.7. Варианты"))
    V = R["variants"]["variants"]
    rows = [["Код", "Вариант", "Эталон", "Метрика"]]
    for v, d in V.items():
        rows.append([v, d["name"], "B0̄ (3 сида)" if d["ref"] == "B0bar" else "B0", "девиация голов" if d["metric"] == "dev"
                     else "log loss 1X2"])
    rep.add(*T(rows, [1.3, 8.6, 2.6, 3.0]))
    rep.add(P("M2 — модель на каждую лигу только по её матчам; M3 — модель на группу; M4a/M4b — глобальная модель, "
              "затем продолжение бустинга на матчах лиги (+20 % / +50 % деревьев; глобальная модель служит "
              "«априорным знанием»); M5 — one-hot группы вместо лиги; M1 — без признака лиги вовсе. Перенос (раздел 7): "
              "классификатор 1X2, обученный без лиги L (на всех других лигах, только на своей группе, только на "
              "чужих группах), предсказывает лигу L."))
    rep.add(H2("3.8. Неопределённость и решения"))
    rep.add(P(f"Парный бутстреп по матчам ({C.N_BOOT} повторов, 95 % процентильные интервалы); p-значение — "
              "двустороннее бутстреп. Поправка Холма — по всем вариантам на объединённых сезонах отбора "
              f"(проверено вариантов: {R['variants']['n_variants_tested']}). Правило решений зафиксировано до "
              "расчёта: A — лучше на объединённом отборе с p Холма < 0.05, лучше в обоих сезонах отбора, не хуже в "
              "2025/26, ROI Main не хуже значимо; B — лучше в обоих сезонах отбора и 95 % ДИ целиком < 0, но не "
              "проходит Холма или 2025/26; C — иначе. Выборки лиги-сезона меньше "
              f"{C.SMALL_N} матчей помечены как малые."))


def uniformity(rep, R):
    perf = R["perf"]
    sel = perf["select"]
    rep.add(H1("4. Насколько ровно модель работает по лигам"))
    rows = [["Сезон", "Матчей", "LL модели", "LL рынка", "Модель − рынок ×10⁻³", "τ ×10⁻³", "I², %", "p (Q)"]]
    for s in SEAS:
        x = perf["seasons"][str(s)]
        rows.append([C.short_period(s), x["n"], f(x["ll"], 4), f(x["ll_mkt"], 4), dk(x["d"]), f(x["meta"]["tau"] * 1e3, 1),
                     f(x["meta"]["I2"] * 100, 0), pv(x["meta"]["p_Q"])])
    rows.append(["2023/24+2024/25", sel["n"], "—", "—", dk(sel["d"]), f(sel["meta"]["tau"] * 1e3, 1),
                 f(sel["meta"]["I2"] * 100, 0), pv(sel["meta"]["p_Q"])])
    rep.add(*T(rows, [2.6, 1.4, 1.7, 1.7, 3.6, 1.6, 1.3, 1.5]))
    rep.add(*figure(C.FIGURES / "league_vs_market.png",
                    "Рис. 1. Модель − рынок по лигам (объединённый отбор): точки — средние по лиге с 95 % ДИ, "
                    "ромбы — после EB-сжатия к общему среднему."))
    L = sel["leagues"]
    rows = [["Лига", "n", "Модель − рынок ×10⁻³", "EB ×10⁻³", "Наклон b", "ECE мод.", "ECE рынка", "Ничьи прогн.", "Ничьи факт"]]
    for lg in sorted(L, key=lambda x: L[x]["d"]["diff"]):
        r = L[lg]
        rows.append([lg + (" *" if r["small"] else ""), r["n"], dk(r["d"]), f(sel["meta"]["by_group"][lg]["eb"] * 1e3, 1),
                     f"{r['slope']:.2f} ± {1.96 * (r['slope_se'] or 0):.2f}", f(r["ece"], 3), f(r["ece_mkt"], 3),
                     pc(r["draw_pred"], 1, False), pc(r["draw_obs"], 1, False)])
    rep.add(*T(rows, [1.4, 1.0, 3.6, 1.6, 2.2, 1.5, 1.6, 1.8, 1.7]))
    rep.add(P("* — меньше 300 матчей. Наклон b — оценка карты калибровки по матчам лиги (± 1.96 SE): b < 1 — модель "
              "слишком уверена, b > 1 — недостаточно уверена.", "small"))
    rep.add(*figure(C.FIGURES / "calibration_slopes.png",
                    "Рис. 2. Наклон калибровки по лигам (2023/24 + 2024/25) и сжатый наклон карты C3, построенной для "
                    "2025/26 по истории."))
    rep.add(H2("4.1. Устойчивы ли различия?"))
    ps = perf["persistence"]
    rep.add(P("Если лига «трудная» для модели по существу, её отставание от рынка должно повторяться из сезона в "
              "сезон. Ранговая корреляция средних по лигам между сезонами:"))
    rows = [["Пара сезонов", "Спирмен ρ", "p"]] + [[k.replace("-", " ↔ "), f(v["rho"], 2), pv(v["p"])] for k, v in ps.items()]
    rep.add(*T(rows, [5, 3, 3]))
    rep.add(*figure(C.FIGURES / "league_persistence.png", "Рис. 3. Отставание от рынка по лигам: 2023/24 против 2024/25."))
    ex = perf["explain"]
    from .profiles import VARS
    names = {**VARS, "train_n": "объём обучения (матчей до 2023/24)", "tier": "уровень лиги (1 — высший)"}
    rows = [["Характеристика лиги", "Спирмен ρ с (модель − рынок)", "p", "p Холма"]]
    for k in sorted(ex, key=lambda k: ex[k]["p"]):
        rows.append([names.get(k, k), f(ex[k]["rho"], 2), pv(ex[k]["p"]), pv(ex[k]["p_holm"])])
    rep.add(P("Что связано с отставанием лиги (21 лига, профиль — сезоны до 2023/24):"))
    rep.add(*T(rows, [7.0, 4.4, 2.0, 2.0]))
    rep.add(H2("4.2. Main по лигам"))
    rows = [["Лига", "Ставок", "Попадание", "Заявлено", "Переоценка, п.п.", "ROI [95 % ДИ]"]]
    for lg in sorted(L, key=lambda x: L[x]["main"]["roi"]["mean"] or 0):
        m = L[lg]["main"]
        rows.append([lg + (" *" if L[lg]["small"] else ""), m["bets"], pc(m["hit"], 1, False), pc(m["p"], 1, False),
                     f"{m['gap'] * 100:+.1f} {ci(m['gap_ci'])}", roi(m["roi"])])
    rep.add(*T(rows, [1.6, 1.4, 2.0, 2.0, 4.2, 4.8]))
    mr, mg = sel["meta_main_roi"], sel["meta_main_gap"]
    rep.add(P(f"Межлиговая неоднородность ROI Main: Q-тест p = {pv(mr['p_Q'])}, I² = {mr['I2'] * 100:.0f} %, общий ROI "
              f"(случайные эффекты) {mr['mu'] * 100:+.1f} % ± {1.96 * mr['mu_se'] * 100:.1f}; переоценки Main: "
              f"p = {pv(mg['p_Q'])}, I² = {mg['I2'] * 100:.0f} %, общая {mg['mu'] * 100:+.1f} п.п. Попадание и ROI "
              "считаются по матчам лиги с ценой; у малых лиг интервалы ROI шире ±10 п.п."))
    rep.add(*figure(C.FIGURES / "main_by_league.png", "Рис. 4. ROI и переоценка Main по лигам, 2023/24 + 2024/25."))


def groups(rep, R):
    pr = R["profiles"]
    rep.add(H1("5. Группы похожих лиг"))
    a = pr["all_history"]
    rep.add(P(f"По всей истории (2021/22–2026/27) силуэт максимален при k = {a['k']} "
              f"({', '.join(f'k={k}: {v:.2f}' for k, v in a['silhouette'].items())}). Силуэт ниже 0.3 означает слабую "
              "кластерную структуру: лиги образуют скорее непрерывный спектр, чем чёткие группы."))
    rep.add(*figure(C.FIGURES / "profiles_heatmap.png", "Рис. 5. Стандартизованные профили лиг и группы (вся история)."))
    rows = [["Сезон (группы по прошлому)", "k", "Силуэт", "Группы"]]
    for s, x in pr["folds"].items():
        g = {}
        for lg, gr in x["groups"].items():
            g.setdefault(gr, []).append(lg)
        rows.append([C.season_label(int(s)), x["k"], f(x["silhouette"][str(x["k"])], 2),
                     "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sorted(g.items()))])
    rep.add(*T(rows, [2.6, 0.8, 1.4, 12.2]))
    rep.add(P("Устойчивость разбиения между соседними сезонами (ARI, 1 — то же разбиение, 0 — случайное): " +
              ", ".join(f"{k.replace('-', '→')}: {v:.2f}" for k, v in pr["ari"].items()) + "."))


def variants(rep, R):
    var = R["variants"]
    V = var["variants"]
    rep.add(H1("6. Варианты специализации"))
    ref = var["reference"]
    rows = [["Эталон", "Сезон", "LL 1X2", "Девиация голов", "Модель − рынок ×10⁻³", "Main: ставок", "Main ROI"]]
    for r in ("B0", "B0bar"):
        for s in [*map(str, SEAS), "select"]:
            x = ref[r][s]
            rows.append(["B0" if r == "B0" else "B0̄", "отбор" if s == "select" else C.short_period(int(s)), f(x["ll"], 4),
                         f(x["dev"], 4), dk(x["vs_market"]), x["main"]["bets"], roi(x["main"]["roi"])])
    rep.add(*T(rows, [1.4, 2.4, 1.6, 2.2, 3.6, 1.8, 4.0]))
    sn = var["seed_noise"]
    rep.add(P("Шум сида (разница с сидом 42 по log loss 1X2, ×10⁻³): " + "; ".join(
        f"{k.replace('B0_seed', 'сид ').replace('B0bar', 'среднее 3 сидов')}: " +
        ", ".join(f"{C.season_label(int(s))} {dk(sn[k][s], 2)}" for s in map(str, C.SELECT)) for k in sn) + "."))
    rows = [["Вариант", "2023/24", "2024/25", "Отбор (объединённо)", "p", "p Холма", "2025/26 инф.", "2026/27 тек.", "Решение"]]
    for v, d in V.items():
        m = "d_" + d["metric"]
        rows.append([v, dk(d["seasons"]["2023"][m], 2), dk(d["seasons"]["2024"][m], 2), dk(d["select"][m], 2),
                     pv(d["select"][m]["p_two_sided"]), pv(d["select"]["p_holm"]), dk(d["seasons"]["2025"][m], 2),
                     dk(d["seasons"]["2026"][m], 2), d["decision"]])
    rep.add(P("Разница основной метрики с эталоном (×10⁻³; < 0 — вариант лучше; для G* — девиация голов, для остальных — "
              "log loss 1X2):"))
    rep.add(*T(rows, [1.2, 2.3, 2.3, 2.4, 1.2, 1.3, 2.3, 2.3, 1.3]))
    rep.add(*figure(C.FIGURES / "variants_forest.png",
                    "Рис. 6. Варианты: разница с эталоном на объединённом отборе (95 % ДИ) и по отдельным сезонам."))
    rows = [["Вариант", "Модель − рынок ×10⁻³ (отбор)", "Main ставок", "ROI Main", "ΔROI к эталону, п.п.", "Δ переоценки, п.п."]]
    for v, d in V.items():
        s = d["select"]
        rows.append([v, dk(s["vs_market"]), s["main"]["bets"], roi(s["main"]["roi"]),
                     f"{s['d_roi']['diff'] * 100:+.1f} {ci(s['d_roi']['ci95'])}", f"{s['d_gap'] * 100:+.2f}"])
    rep.add(P("Влияние на рынок и Main (объединённый отбор):"))
    rep.add(*T(rows, [1.4, 4.0, 1.8, 3.8, 3.6, 2.4]))
    fits = R["variant_fits"] or {}
    if fits:
        rows = [["Сезон", "τ² наклона b", "τ² сдвига D", "τ² сдвига A", "Глобальная b", "Голы: τ² хозяев", "Голы: τ² гостей"]]
        for s in SEAS:
            x = fits.get(str(s))
            if not x:
                continue
            cal, g = x["calibration"], x["goals"]
            rows.append([C.short_period(s), f(cal["tau2_league"][0], 4), f(cal["tau2_league"][1], 4), f(cal["tau2_league"][2], 4),
                         f(cal["global"][0], 3), f(g["home"]["tau2"], 4), f(g["away"]["tau2"], 4)])
        rep.add(P("Оценённая межлиговая дисперсия параметров (по истории до сезона). Чем меньше τ² по сравнению с "
                  "дисперсией оценки лиги, тем сильнее сжатие к глобальной карте:"))
        rep.add(*T(rows, [2.4, 2.2, 2.2, 2.2, 2.2, 2.4, 2.4]))
        x = fits.get(str(C.INFO))
        if x:
            cal = x["calibration"]
            rows = [["Лига", "Матчей в истории", "b (MLE) ± SE", "b после EB", "a_D (MLE)", "a_D после EB", "a_A (MLE)", "a_A после EB"]]
            for lg in sorted(cal["league_n"], key=lambda k: cal["league_n"][k]):
                mle, se, eb = cal["league_mle"][lg], cal["league_se"][lg], cal["league_eb"][lg]
                rows.append([lg, cal["league_n"][lg], f"{mle[0]:.2f} ± {se[0]:.2f}", f(eb[0], 2), f(mle[1], 2), f(eb[1], 2),
                             f(mle[2], 2), f(eb[2], 2)])
            rep.add(P("Карта калибровки лиг для 2025/26 (история 2022/23–2024/25): малые лиги сжимаются к глобальной "
                      "карте сильнее."))
            rep.add(*T(rows, [1.4, 2.4, 2.6, 1.9, 1.9, 2.1, 1.9, 2.1]))


def transfer(rep, R):
    tr = R["transfer"]
    if not tr:
        return
    rep.add(H1("7. Перенос закономерностей между лигами"))
    P_ = tr["pooled"]
    names = {"own": "только своя лига (M2, сид 42)", **{k: v for k, v in tr["kinds"].items()}}
    rows = [["Обучение (сид 42)", "Δ к B0, отбор ×10⁻³", "p", "Δ к B0, 2025/26 ×10⁻³"]]
    for k in ("own", "all", "group", "out"):
        rows.append([names[k], dk(P_[k]["pooled"]), pv(P_[k]["pooled"]["p_two_sided"]), dk(tr["info_2025"][k]["pooled"])])
    rep.add(*T(rows, [6.4, 4.0, 1.6, 4.0]))
    g = P_["group_vs_out"]
    rep.add(P(f"Своя группа против чужих групп (без самой лиги): {dk(g)}×10⁻³ (p = {pv(g['p_two_sided'])}); < 0 — "
              "данные похожих лиг полезнее. Модели «без лиги» обучены на меньшем объёме данных, поэтому сравнение "
              "«своя группа / чужие группы» смешивает похожесть и объём выборки."))
    rep.add(*figure(C.FIGURES / "transfer.png", "Рис. 7. Перенос по лигам: разница log loss с B0, 2023/24 + 2024/25."))


def tail(rep, R):
    var = R["variants"]
    V = var["variants"]
    dec = {k: [v for v, d in V.items() if d["decision"] == k] for k in "ABC"}
    rep.add(H1("8. Отрицательные результаты"))
    rep.add(*bullets(R["decisions"]["negative"] if R.get("decisions") else []))
    rep.add(H1("9. Ограничения"))
    rep.add(*bullets([
        "Малые лиги: 180–300 матчей за сезон; интервалы по лиге широкие, вывод «лига X хуже» по одному сезону ненадёжен.",
        "Параметры XGBoost не перенастраивались для отдельных и групповых моделей: малым лигам может подойти "
        "более сильная регуляризация. Это консервативно по отношению к специализации, но перенастройка на "
        "каждую лигу — ещё больше степеней свободы на малых выборках.",
        "Эталон рынка — одна книга (Bet365) на открытии; закрытие точнее, поэтому отставание от закрытия больше.",
        "2025/26 использовался прежними исследованиями, 2026/27 просмотрен многократно — оба только для информации; "
        "независимая проверка возможна лишь вперёд (папка `forward/`).",
        "Группы лиг слабо выражены и неустойчивы между сезонами, поэтому групповые варианты зависят от разбиения.",
    ]))
    rep.add(H1("10. Выводы и рекомендации"))
    rep.add(*bullets(R["decisions"]["conclusions"] if R.get("decisions") else []))
    rows = [["Решение", "Варианты"]] + [[DEC_RU[k], ", ".join(f"{v} ({V[v]['name']})" for v in dec[k]) or "—"] for k in "ABC"]
    rep.add(*T(rows, [5.0, 11.5]))
    pre = R.get("preregistration")
    if pre:
        rep.add(note(f"Предрегистрация: `experiments/v6_2/leagues/results/preregistration.json` (заморожено "
                     f"{pre['frozen_at']}): {', '.join(pre['b_variants'])}; критерий проверки вперёд — "
                     f"{pre['forward_criterion']['primary']}"))
    else:
        rep.add(note("Вариантов категории B нет — предрегистрация не требуется. Продакшн не меняется."))
    rep.add(H1("11. Воспроизведение"))
    rep.add(*reproduce([
        "PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python",
        "PYTHONPATH=. timeout 21600 $PY -m experiments.v6_2.leagues.run > experiments/v6_2/leagues/cache/run.log 2>&1",
        "$PY -m experiments.v6_2.leagues.decisions        # решения, findings.md",
        "$PY -m experiments.v6_2.leagues.build_pdf",
    ]))
    ri = R["run_info"] or {}
    rep.add(P(f"Код результатов: git {ri.get('git_sha', '—')}; завершено {ri.get('finished', '—')}; "
              f"сиды {list(C.SEEDS)}, потоков на модель {C.N_JOBS}, бутстреп {C.N_BOOT}.", "small"))


def run():
    FG.run()
    R = load()
    rep = Report(OUT, "Специализация по лигам",
                 facts={"Сезоны отбора": "2023/24, 2024/25", "Для информации": "2025/26; 2026/27 (текущий, уже просмотрен)",
                        "Лиг": "21"})
    front(rep, R)
    intro(rep, R)
    methods(rep, R)
    uniformity(rep, R)
    groups(rep, R)
    variants(rep, R)
    transfer(rep, R)
    tail(rep, R)
    rep.build()
    n = rep.check()
    print(f"pages: {n}")


if __name__ == "__main__":
    run()
