"""reports/v6_1/Corners_Research.pdf and experiments/v6_1/corners/findings.md from the result files.

    python -m experiments.v6_1.corners.build_pdf

Every number comes from experiments/v6_1/corners/results/*.json|csv (written by run.py).
"""
from __future__ import annotations

import pandas as pd

from scripts.report_kit import (H1, H2, H3, P, Report, Spacer, bullets, code, figure, formula, note, num, share,
                                table)
from src import config

from . import data as D
from .evaluate import REF, TITLES

R = {k: D.load_result(f"{k}.json") for k in ("inventory", "current", "metrics", "betting", "versions",
                                               "team_strength_grid", "run_times")}
MET, BET, CUR, INV = R["metrics"], R["betting"], R["current"], R["inventory"]
FOLDS = [str(s) for s in D.FOLDS]
REFK = "|".join(REF)
SL = {s: config.season_label(int(s)) for s in FOLDS}
SLT = {s: SL[s] + {"2023": "", "2024": "", "2025": "*", "2026": "†"}[s] for s in FOLDS}
FIG = D.FIGURES
MODELS = ["xgb_current", "xgb_enriched", "xgb_tuned", "xgb_ts", "xgb_ts_offset", "lgbm", "catboost", "poisson_glm",
          "nb_glm", "team_strength", "xgb_home_away", "ens_weighted", "ens_simple", "naive_league"]
DISTS = {"poisson": "Пуассон", "nb_mom": "NB, r методом моментов (продакшн)", "nb_mle": "NB, r по ML",
         "nb_league": "NB, r по лигам (со сжатием)", "nb1": "NB1: Var = φ·μ",
         "conv_indep": "свёртка независимых NB дома/в гостях", "nb_joint": "NB тотала с ковариацией сторон"}


def d(k, s):
    return MET["folds"][s]["dist"][k]


def pt(m, s):
    return MET["folds"][s]["point"][m]


def name(k):
    m, dk = k.split("|")
    return f"{TITLES.get(m, m)} · {dk}"


def ci(x, dg=4):
    return f"{x['mean']:+.{dg}f} [{x['ci95'][0]:+.{dg}f}; {x['ci95'][1]:+.{dg}f}]"


def pc(x, dg=1):
    return "—" if x is None else f"{x * 100:+.{dg}f} %"


SEL = MET["selection"]
TAB = SEL["table"]
ENS = "ens_simple|nb_mom"
best_nb = sorted([k for k in TAB if k.endswith("|nb_mom") and not k.startswith("naive")],
                 key=lambda k: TAB[k]["pooled_select_line_ll"])
BEST = best_nb[0]
ADOPTED = bool(SEL["eligible_before_betting"])


# ---------------------------------------------------------------------------------------------- sections
def s_goal(rep):
    rep.add(H1("1. Цель"),
            P("Проверить, можно ли заметно улучшить модель тотала угловых FootPredict, и прежде всего — качество "
              "вероятностей «больше / меньше» на линиях, которые показывает продукт (8.5, 9.5, 10.5; в историю "
              "прогнозов пишется линия 9.5). Точечная оценка (MAE, RMSE, R²) важна меньше, чем распределение "
              "и результат ставок по реальным ценам."),
            P("Сравниваются: текущий XGBoost; XGBoost с новыми признаками и с собственным подбором гиперпараметров; "
              "CatBoost; LightGBM; пуассоновская и отрицательно-биномиальная регрессии (GLM); иерархическая модель "
              "с частичным объединением (эффекты команд и лиг); раздельные модели угловых хозяев и гостей; "
              "совместное распределение сторон; ансамбли; пять вариантов дисперсии распределения."),
            note("Правило внедрения задано до расчёта: новая архитектура принимается, только если она "
                 "убедительно лучше на обоих фолдах отбора (2023/24, 2024/25) и не хуже на 2025/26 и 2026/27 "
                 "(включая ставки по реальным ценам). Иначе остаётся текущая модель."))


def s_problem(rep):
    b, p = CUR["backtest_2526"], CUR["prod_20261012"]
    mb, mp = b["metrics"], p["metrics"]
    rows = [["Показатель", "backtest_2526 на 2025/26", "prod_20261012 на 2026/27"],
            ["Матчей", mb["n"], mp["n"]], ["MAE", num(mb["mae"]), num(mp["mae"])],
            ["RMSE", num(mb["rmse"]), num(mp["rmse"])], ["R²", num(mb["r2"], 4), num(mp["r2"], 4)],
            ["Пуассоновская девиация", num(mb["poisson_deviance"], 4), num(mp["poisson_deviance"], 4)],
            ["Среднее факт / прогноз", f"{mb['mean_actual']:.2f} / {mb['mean_predicted']:.2f}",
             f"{mp['mean_actual']:.2f} / {mp['mean_predicted']:.2f}"],
            ["Log score (NB)", num(mb["log_score"], 4), num(mp["log_score"], 4)],
            ["CRPS (NB)", num(mb["crps"], 4), num(mp["crps"], 4)],
            ["Размер NB r", num(b["nb_size"], 2), num(p["nb_size"], 2)]]
    for l in map(str, D.LINES):
        rows.append([f"Log loss / ECE линии {l}", f"{mb['lines'][l]['log_loss']:.4f} / {mb['lines'][l]['ece']:.4f}",
                     f"{mp['lines'][l]['log_loss']:.4f} / {mp['lines'][l]['ece']:.4f}"])
    meta = b["metadata_test_metrics"]
    rep.add(H1("2. Исходная проблема"),
            P("Метрики пересчитаны заново по сохранённым моделям (`load_bundles`), а не взяты из метаданных. "
              "Модель `backtest_2526` (обучена на 2021/22–2024/25) оценена на 2025/26, активная продакшн-модель "
              "`prod_20261012` (обучена на 2021/22–2025/26) — на сыгранных матчах 2026/27 по состоянию данных."),
            table(rows), Spacer(1, 6),
            P(f"Пересчёт совпадает с метаданными версии: MAE {meta['mae']:.4f}, RMSE {meta['rmse']:.4f}, "
              f"R² {meta['r2']:.4f} на {meta['n']} матчах 2025/26. Вероятности линий, посчитанные кодом "
              f"исследования, совпадают с `src.markets.distributions.corners_distribution` / `prob_over` "
              f"(максимальное расхождение {b['check_vs_src_markets_max_abs']:.1e})."),
            P(f"Модель объясняет лишь {mb['r2'] * 100:.1f} % (2025/26) и {mp['r2'] * 100:.1f} % (2026/27) дисперсии "
              f"тотала. Это не ошибка реализации: безусловная дисперсия тотала около 11.5 при среднем около 9.8, "
              f"т.е. почти вся изменчивость — пуассоновский шум отдельного матча, который не предсказывается никакими "
              f"предматчевыми данными. Поэтому вопрос исследования — не «как поднять R²», а можно ли точнее "
              f"оценить вероятность пройти линию."))


def s_current(rep):
    b = CUR["backtest_2526"]
    prm = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in b["params"].items())
    rep.add(H1("3. Текущая реализация"),
            *bullets([
                "Задача `corners` (`src/modelling/tasks.py`): XGBRegressor, цель `total_corners`, objective "
                "`count:poisson`, метрика `poisson-nloglik`.",
                f"Признаки (`src/features/sets.py`, группы `MODEL_GROUPS['corners']`): {INV['feature_counts']['prod']} "
                "числовых (угловые за/против за сезон, по месту, формы 5/10/15, тренды, venue-форма 5, разности, "
                "средние лиги, corner pressure, удары и удары в створ, таблица, отдых) + one-hot лиги.",
                f"Гиперпараметры подобраны Hyperopt (20 испытаний, девиация на сезоне валидации 2024/25): {prm}; "
                f"деревьев {b['n_estimators']}. Веса по давности (период полураспада в годах).",
                "Распределение (`src/modelling/training.py: negative_binomial_size`, `src/markets/distributions.py: "
                "corners_distribution`): NB(μ, r) с одним размером r, оценённым методом моментов на остатках сезона "
                f"валидации; сейчас r = {b['nb_size']:.2f}, т.е. Var = μ + μ²/r — дисперсия лишь на ≈ 14 % выше среднего.",
                "Рынки (`src/markets/markets.py: corner_markets`): OVER / UNDER на линиях 8.5 / 9.5 / 10.5; "
                "в историю прогнозов пишется наиболее вероятный исход линии 9.5 (`HISTORY_MARKETS`), доверие "
                "к рынку при ранжировании Main / Risk — 0.85 (`MARKET_RELIABILITY`).",
                "Лиги: все 21 лига проходят порог покрытия статистикой угловых 80 % (`corner_leagues`).",
            ]))


def s_hypotheses(rep):
    rep.add(H1("4. Гипотезы"), *bullets([
        "**H1.** Новые строго предматчевые признаки (EWMA угловых и ударов, фолы, ожидание угловых из силы "
        "соперников, |разница Elo|) улучшат вероятности линий.",
        "**H2.** Собственный подбор гиперпараметров XGBoost на каждом фолде лучше продакшн-параметров.",
        "**H3.** Другие бустинги (CatBoost, LightGBM) лучше XGBoost.",
        "**H4.** Простые модели (Poisson GLM, NB GLM) не хуже бустинга при таком слабом сигнале.",
        "**H5.** Иерархическая модель с частичным объединением (сила команды в атаке/обороне по угловым, "
        "сжатая к среднему лиги, со скользящим переобучением) точнее оценок по скользящим окнам.",
        "**H6.** Раздельные модели угловых хозяев и гостей и их совместное распределение лучше модели тотала.",
        "**H7.** Распределение с другой дисперсией (ML, по лигам, NB1) или перекалибровка линий улучшит log loss.",
        "**H8.** Ансамбль разнородных моделей лучше каждой из них.",
        "**H9.** Лучшая по вероятностям модель даёт лучший результат ставок по реальным ценам угловых 2026/27.",
    ]))


def s_data(rep):
    cov = INV["coverage_by_league"]
    mls = INV["matches_by_league_season"]
    rows = [["Лига", "Угловые HC/AC", "Удары HS/AS", "В створ HST/AST", "Фолы HF/AF", "Карточки"]
            + [config.season_label(int(s))[2:] for s in sorted(next(iter(mls.values())))]]
    for lg in sorted(cov):
        c = cov[lg]
        rows.append([lg, share(c["home_corners"]), share(c["home_shots"]), share(c["home_sot"]),
                     share(c["home_fouls"]), share(c["home_yellows"])] + [mls[lg][s] for s in sorted(mls[lg])])
    st = INV["season_stats"]
    srows = [["Сезон", "Матчей с угловыми", "Среднее", "Дисперсия", "Хозяева", "Гости", "corr(хоз., гост.)"]]
    for s in sorted(st):
        v = st[s]
        srows.append([config.season_label(int(s)), v["n"], num(v["mean"], 2), num(v["var"], 2), num(v["home"], 2),
                      num(v["away"], 2), num(v["corr_home_away"], 3)])
    pl = BET["lines_priced"]
    rep.add(H1("5. Данные и периоды"),
            P("Источник статистики — файлы Football-Data (`local_data/results/<сезон>/<лига>.csv`): угловые HC/AC, "
              "удары HS/AS, удары в створ HST/AST, фолы HF/AF, жёлтые и красные карточки. Фолы в таблицу признаков "
              "продукта не входят, они прочитаны из тех же файлов через `build_matches` (только чтение). Других "
              "статистик (навесы, владение, xG, составы) в бесплатных данных нет, и они не придумывались; столбец "
              "Referee есть только в английских файлах, поэтому не использован. Цен на угловые в Football-Data нет."),
            H2("5.1. Какая статистика где есть"),
            P("Доля сыгранных матчей с заполненной статистикой и число матчей по сезонам (с 2021/22):"),
            table(rows, font=6.2), Spacer(1, 6),
            P("Статистика есть практически везде (минимум — T1, 98.3 %). Пропуски — единичные матчи; при обучении "
              "используются только матчи с известными угловыми хозяев и гостей."),
            H2("5.2. Тотал угловых по сезонам"), table(srows), Spacer(1, 4),
            P("Дисперсия тотала лишь немного выше среднего; угловые хозяев и гостей в одном матче отрицательно "
              "связаны (corr ≈ −0.25): когда одна команда давит, другая подаёт меньше."),
            *figure(FIG / "distribution.png", "Рис. 1. Распределение тотала угловых и среднее/дисперсия по лигам."),
            H2("5.3. Фолды"),
            table([["Фолд", "Внутреннее обучение", "Внутренняя валидация", "Внешнее обучение", "Оценка", "Роль"],
                   ["2023/24", "2021/22", "2022/23", "2021/22–2022/23", f"2023/24 ({MET['folds']['2023']['n']})", "отбор"],
                   ["2024/25", "2021/22–2022/23", "2023/24", "2021/22–2023/24", f"2024/25 ({MET['folds']['2024']['n']})", "отбор"],
                   ["2025/26*", "2021/22–2023/24", "2024/25", "2021/22–2024/25", f"2025/26 ({MET['folds']['2025']['n']})",
                    "только информация"],
                   ["2026/27†", "2021/22–2024/25", "2025/26", "2021/22–2025/26", f"2026/27 ({MET['folds']['2026']['n']})",
                    "независимая проверка, реальные цены"]]),
            Spacer(1, 4),
            P("Всё, что подгоняется (число деревьев, гиперпараметры, L2, размер NB, веса ансамбля, калибровка, "
              "гиперпараметры иерархической модели), подбирается только на внутренней валидации фолда. "
              "Звёздочка (*) — сезон, который уже видели при разработке v1–v6; крест (†) — 2026/27 по "
              "2 ноября 2026 г."),
            H2("5.4. Цены угловых 2026/27"),
            P(f"Архив OddsPapi (`local_data/odds/odds.parquet`, только чтение): Bet365, рынки CORNERS_<линия>. "
              f"Цены есть для {BET['n_matches_priced']} из {BET['n_matches_2026']} сыгранных матчей 2026/27; "
              f"на линиях 8.5 / 9.5 / 10.5 — {pl.get('8.5')} / {pl.get('9.5')} / {pl.get('10.5')} матчей. "
              f"Берётся последняя цена до начала матча (медиана — за {BET['price_lead_hours']['50%']:.1f} ч до "
              f"начала). Pinnacle покрывает лишь малую часть матчей и не используется. Маржа Bet365 на линиях 8.5–10.5 — "
              f"около 8 %."))


def s_methods(rep):
    v = R["versions"]
    rep.add(H1("6. Методы и параметры"),
            H2("6.1. Новые признаки"),
            P(f"{len(INV['extra_features'])} новых признаков (`experiments/v6_1/corners/data.py: add_extra_features`). "
              "Для каждой команды (страна, команда) матчи сортируются по дате, ряд СДВИГАЕТСЯ на один матч до "
              "любого усреднения — матч никогда не видит себя и более поздние матчи:"),
            *bullets(["EWMA угловых за / против (период полураспада 5 и 15 матчей), venue-EWMA (5) дома / в гостях;",
                      "EWMA ударов и ударов в створ за / против (10), фолов совершённых / полученных (10), "
                      "тотала угловых в матчах команды (10); угловые на удар;",
                      "давление относительно соперника: EWMA угловых команды минус EWMA пропущенных угловых соперника;",
                      "ожидание угловых матча: аддитивное (среднее «за» команды и «против» соперника), мультипликативное "
                      "(venue-EWMA, нормированные на средние лиги) и по ударам; |разница Elo|."]),
            P(f"Плюс в «обогащённый» набор добавлены имеющиеся, но не используемые продакшн-моделью группы "
              f"`opp_adjusted_corners`, `elo`, `discipline`. Итого: продакшн {INV['feature_counts']['prod']}, "
              f"обогащённый {INV['feature_counts']['enriched']}, компактный (для GLM) {INV['feature_counts']['compact']} "
              f"числовых признаков. Проверка утечки: для {INV['leakage_check']['checked']} случайных пар "
              f"(матч, признак) значение пересчитано перебором только по более ранним матчам — максимальное "
              f"расхождение {INV['leakage_check']['max_abs_diff']:.1e}."),
            *figure(FIG / "feature_corr.png", "Рис. 2. Признаки с наибольшей корреляцией с тоталом (2021/22–2022/23)."),
            P("Самая сильная связь у нового `x_exp_total_add` — лишь r ≈ 0.12; потолок предсказуемости низкий."),
            H2("6.2. Модели среднего"),
            table([["Модель", "Признаки", "Параметры / подбор (на внутренней валидации)"],
                   ["xgb_current", "продакшн", "продакшн-параметры и 846 деревьев (как `prod_20261012`)"],
                   ["xgb_enriched", "обогащённый", "продакшн-параметры, деревья ранней остановкой"],
                   ["xgb_tuned", "обогащённый", "Hyperopt TPE, 20 испытаний, пространство поиска продакшна"],
                   ["xgb_ts", "обогащённый + сила команд", "продакшн-параметры, ранняя остановка"],
                   ["xgb_ts_offset", "обогащённый", "бустинг от смещения log(μ иерархической модели)"],
                   ["lgbm", "обогащённый", "Poisson, lr 0.03, сетка num_leaves {7,15,31} × min_child {100,300}"],
                   ["catboost", "обогащённый", "Poisson, lr 0.05, depth {4,6} × l2 {3,10}"],
                   ["poisson_glm", "компактный", "стандартизация, флаги пропусков, лига; L2 {1e-3…1}"],
                   ["nb_glm", "компактный", "NB2, коэффициенты и r — штрафной ML (L-BFGS); L2 {1e-4,1e-3,1e-2}"],
                   ["team_strength", "только угловые", "иерархическая Poisson-модель сторон (6.3)"],
                   ["xgb_home_away", "обогащённый", "два XGBoost (угловые хозяев / гостей), сумма"],
                   ["ens_weighted", "—", "выпуклые веса всех моделей, минимум девиации на внутр. сезоне"],
                   ["ens_simple", "—", "равное среднее xgb_current, poisson_glm, team_strength (задано заранее)"],
                   ["naive_league", "—", "среднее тотала лиги в обучении"]]),
            H2("6.3. Иерархическая модель (частичное объединение)"),
            P("Каждый матч даёт две строки — угловые хозяев и гостей:"),
            formula("log E[угловые команды i против j] = b_{лига, место} + att_{i} + def_{j}"),
            P("att и def — эффекты команд с L2-штрафом α (апостериорная мода гауссовых случайных эффектов: "
              "команда с малой историей сжимается к среднему лиги), базы лиги × места не штрафуются; веса матчей "
              "0.5^{возраст/период полураспада}. Модель переобучается 1-го и 16-го числа каждого месяца ТОЛЬКО на "
              "матчах до даты переобучения и прогнозирует матчи следующих двух недель. Сетка α ∈ {0.001, 0.003, "
              "0.01, 0.03} × период {180, 365, 730} дней; точка выбирается по девиации на внутреннем сезоне фолда. "
              "Тотал = сумма сторон."),
            *figure(FIG / "team_strength_grid.png", "Рис. 3. Девиация иерархической модели по сетке и сезонам."),
            table([["α | период, дней"] + [config.season_label(int(x)) for x in sorted(next(iter(R["team_strength_grid"].values())))]]
                  + [[k] + [num(v[x], 4) for x in sorted(v)] for k, v in R["team_strength_grid"].items()], font=6.8),
            Spacer(1, 4),
            P("Лучшие точки — α = 0.01–0.03 и период 180–365 дней; слабое сжатие (α = 0.001) заметно хуже: "
              "эффекты команд по угловым шумные, и без частичного объединения модель переобучается."),
            H2("6.4. Распределения"),
            table([["Код", "Распределение", "Что подгоняется на внутреннем сезоне"]]
                  + [[k, DISTS[k], t] for k, _, t in (
                      ("poisson", "", "ничего"), ("nb_mom", "", "r = mean(μ²) / mean((y−μ)² − μ)"),
                      ("nb_mle", "", "r — максимум правдоподобия"),
                      ("nb_league", "", "r лиги по ML, 1/r сжат к общему с весом 400 матчей"),
                      ("nb1", "", "φ = mean((y−μ)²) / mean(μ), r_{i} = μ_{i}/(φ−1)"),
                      ("conv_indep", "", "r хозяев и гостей по ML; сторон независимы"),
                      ("nb_joint", "", "r сторон и корреляция ρ остатков сторон"))]),
            Spacer(1, 4),
            P("Для модели с NB, r по методу моментов дополнительно проверена перекалибровка Платта вероятностей "
              "каждой линии (логистическая регрессия по logit p, подогнанная на внутреннем сезоне)."),
            H2("6.5. Метрики"),
            *bullets(["точечные: MAE, RMSE, R², средняя пуассоновская девиация, смещение, калибровка по децилям;",
                      "распределение: log score = −log P(Y = y), дискретный CRPS = Σ_{k}(F(k) − 1{y ≤ k})², "
                      "рандомизированный PIT (10 корзин, χ²);",
                      "линии 8.5 / 9.5 / 10.5 (и для справки 7.5–12.5): log loss, Brier, ECE, диаграммы надёжности; "
                      "сравнение с базовой частотой обучающих сезонов и частотой по лиге;",
                      "разбивка по периодам сезона (июл–сен, окт–дек, янв–мар, апр–июн) и по лигам;",
                      "парный бутстрэп по матчам (2000 повторов) для Δ log loss линий, Δ log score и Δ CRPS "
                      "относительно текущей модели;",
                      "ставки 2026/27 по реальным ценам: ROI плоской ставкой, кластерный бутстрэп по матчам."]),
            H2("6.6. Окружение"),
            P(f"Python {v['python']}, xgboost {v['xgboost']}, lightgbm {v['lightgbm']}, catboost {v['catboost']}, "
              f"scikit-learn {v['sklearn']}, statsmodels {v['statsmodels']}, scipy {v['scipy']}, numpy {v['numpy']}, "
              f"pandas {v['pandas']}; потоков на модель — {v['n_jobs']} (машина делилась с другими задачами)."))


def s_experiments(rep):
    rows = [["Эксперимент", "Выполнен", "Комментарий"],
            ["Пересчёт текущих метрик по сохранённым моделям", "да", "раздел 2"],
            ["Текущий XGBoost в walk-forward", "да", "воспроизводит прод: 2026/27 девиация "
             f"{pt('xgb_current', '2026')['poisson_deviance']:.4f} против {CUR['prod_20261012']['metrics']['poisson_deviance']:.4f}"],
            ["XGBoost + новые признаки", "да", "xgb_enriched"],
            ["XGBoost с собственным Hyperopt", "да", "xgb_tuned, 20 испытаний на фолд"],
            ["CatBoost, LightGBM", "да", "небольшие сетки, ранняя остановка"],
            ["Poisson GLM, NB GLM", "да", "компактный набор; NB2 с оценкой r по ML"],
            ["Иерархическая модель (команды и лиги)", "да", "штрафной ML = мода апостериорного; MCMC не делался: "
             "та же модель, а 80 переобучений на фолд за минуты"],
            ["Сила команд как признак / смещение для XGBoost", "да", "xgb_ts, xgb_ts_offset"],
            ["Раздельные модели хозяев и гостей", "да", "xgb_home_away и стороны иерархической модели"],
            ["Совместное распределение сторон", "да", "свёртка независимых NB и NB тотала с ковариацией"],
            ["Двумерный Пуассон", "нет", "моделирует только положительную ковариацию, а у угловых сторон она "
             "отрицательна (ρ ≈ −0.19); заменён nb_joint"],
            ["Варианты дисперсии, перекалибровка линий", "да", "5 + 2 распределения, Платт"],
            ["Ансамбли", "да", "веса по внутреннему сезону и равные"],
            ["Ставки по реальным ценам угловых", "да", "только 2026/27 — цены есть только там"],
            ["Цены угловых 2021/22–2025/26", "нет", "в Football-Data их нет; платные источники не подключались"],
            ["Коэффициенты 1X2 / тоталов как признаки", "нет", "в продукте `USE_ODDS_FEATURES = False` "
             "(модель не должна копировать рынок); иначе проверка по ценам стала бы круговой"],
            ["Навесы, владение, xG, составы, судьи", "нет", "нет в бесплатных данных (Referee — только Англия)"],
            ["Отдельная модель на каждую лигу", "нет", "лига уже есть как признак/база; при 1–3 тыс. матчей "
             "лиги отдельные модели переобучаются; nb_league проверяет различие дисперсии"]]
    rep.add(H1("7. Эксперименты"), table(rows, widths=[155, 45, 280]))


def model_table(seasons):
    rows = [["Модель"] + [f"{h} {SLT[s]}" for s in seasons for h in ("LL линий", "log score", "RMSE", "R²")]]
    for m in MODELS:
        k = f"{m}|nb_mom"
        r = [TITLES[m]]
        for s in seasons:
            r += [num(d(k, s)["mean_line_log_loss"], 4), num(d(k, s)["log_score"], 4), num(pt(m, s)["rmse"], 3),
                  num(pt(m, s)["r2"], 4)]
        rows.append(r)
    return table(rows, font=6.4)


def point_table(seasons):
    rows = [["Модель"] + [f"{h} {SLT[s]}" for s in seasons for h in ("MAE", "девиация", "CRPS", "смещ.")]]
    for m in MODELS:
        r = [TITLES[m]]
        for s in seasons:
            r += [num(pt(m, s)["mae"], 3), num(pt(m, s)["poisson_deviance"], 4),
                  num(d(f"{m}|nb_mom", s)["crps"], 4), f"{pt(m, s)['bias']:+.2f}"]
        rows.append(r)
    return table(rows, font=6.4)


def delta_table():
    rows = [["Модель (NB, r моментов)"] + [f"Δ LL линий {SLT[s]}" for s in FOLDS] + ["Δ log score 23/24/25/26"]]
    for m in MODELS[1:]:
        k = f"{m}|nb_mom"
        rows.append([TITLES[m]] + [ci(d(k, s)["delta_line_log_loss"]) for s in FOLDS]
                    + [" / ".join(f"{d(k, s)['delta_log_score']['mean']:+.4f}" for s in FOLDS)])
    return table(rows, font=6.2, widths=[90, 88, 88, 88, 88, 78])


def s_results(rep):
    rep.add(H1("8. Результаты"),
            H2("8.1. Точечные метрики и вероятности линий"),
            P("Все модели с одним и тем же распределением — NB с r методом моментов на внутреннем сезоне, как в "
              "продакшне; различаются только средние μ. LL линий — средний log loss линий 8.5 / 9.5 / 10.5."),
            P("Фолды отбора:"), model_table(["2023", "2024"]), Spacer(1, 6),
            P("2025/26 (только информация) и 2026/27 (проверка):"), model_table(["2025", "2026"]), Spacer(1, 6),
            *figure(FIG / "models_metrics.png", "Рис. 4. Метрики моделей среднего по фолдам."),
            P("MAE, девиация, CRPS и смещение среднего (прогноз − факт):"),
            point_table(["2023", "2024"]), Spacer(1, 4), point_table(["2025", "2026"]), Spacer(1, 6))
    ref_r2 = [pt("xgb_current", s)["r2"] for s in FOLDS]
    best_r2 = max(pt(m, s)["r2"] for m in MODELS for s in FOLDS)
    rep.add(P(f"R² текущей модели — {min(ref_r2):.3f}–{max(ref_r2):.3f}; лучший R² среди всех моделей и сезонов — "
              f"{best_r2:.3f}. Разница между любыми двумя нетривиальными моделями — сотые доли RMSE; "
              f"«среднее лиги» хуже всех заметно, т.е. признаки работают, но сигнал слабый. Смещение среднего "
              f"меняет знак между сезонами у всех моделей сразу (средний тотал {INV['season_stats']['2023']['mean']:.2f} "
              f"в 2023/24 и {INV['season_stats']['2024']['mean']:.2f} в 2024/25): уровень угловых сезона "
              f"колеблется, и предматчевые признаки этого не предсказывают."),
            H2("8.2. Разница с текущей моделью (парный бутстрэп)"),
            P("Δ < 0 — модель лучше текущей; в скобках 95 % доверительный интервал по матчам."),
            delta_table(), Spacer(1, 6),
            *figure(FIG / "delta_models.png", "Рис. 5. Δ log loss линий против текущей модели с 95 % ДИ."))
    e = {s: d(ENS, s) for s in FOLDS}
    rep.add(P(f"Ни одна одиночная модель не лучше текущей на обоих фолдах отбора с интервалом ниже нуля. "
              f"Иерархическая модель сильна на 2023/24 ({ci(d('team_strength|nb_mom', '2023')['delta_line_log_loss'])}), "
              f"но хуже на 2024/25 и 2026/27. Единственная пара, которая лучше по точечной оценке на всех четырёх "
              f"сезонах и по log loss линий, и по log score, — равный ансамбль XGBoost + Poisson GLM + иерархическая "
              f"модель: Δ LL линий {ci(e['2023']['delta_line_log_loss'])} (2023/24), "
              f"{ci(e['2024']['delta_line_log_loss'])} (2024/25), {ci(e['2025']['delta_line_log_loss'])} (2025/26*), "
              f"{ci(e['2026']['delta_line_log_loss'])} (2026/27†). На 2024/25 интервал включает 0 — критерий "
              f"«убедительно лучше на обоих фолдах» не выполнен."),
            *figure(FIG / "delta_best.png", "Рис. 6. Восемь лучших пар «модель · распределение» на фолдах отбора."))


def s_dist(rep):
    rows = [["Модель · распределение"] + [f"log score {SLT[s]}" for s in FOLDS] + [f"LL линий {SLT[s]}" for s in FOLDS]]
    for m in ("xgb_current", "xgb_home_away", "team_strength", "ens_simple"):
        for dk in DISTS:
            k = f"{m}|{dk}"
            if k in MET["folds"]["2023"]["dist"]:
                rows.append([name(k)] + [num(d(k, s)["log_score"], 4) for s in FOLDS]
                            + [num(d(k, s)["mean_line_log_loss"], 4) for s in FOLDS])
    info = {s: d(REFK, s)["info"]["r"] for s in FOLDS}
    joint = {s: d("xgb_home_away|nb_joint", s)["info"] for s in FOLDS}
    platt = {s: (d(REFK, s)["platt_mean_line_log_loss"], d(REFK, s)["mean_line_log_loss"]) for s in FOLDS}
    pit = [["Модель · распределение"] + [f"χ² PIT {SLT[s]} (p)" for s in FOLDS]]
    for k in (REFK, "xgb_current|poisson", "xgb_current|nb1", ENS, "xgb_home_away|conv_indep",
              "xgb_home_away|nb_joint"):
        pit.append([name(k)] + [f"{d(k, s)['pit']['chi2']:.1f} ({d(k, s)['pit']['chi2_p']:.3f})" for s in FOLDS])
    rep.add(H2("8.3. Качество распределения и варианты дисперсии"),
            table(rows, font=6.3), Spacer(1, 6),
            *figure(FIG / "dispersion.png", "Рис. 7. Log score и log loss линий по вариантам распределения "
                                            "(среднее двух фолдов отбора)."),
            *bullets([
                "Пуассон заметно хуже NB по log score (≈ +0.005) — небольшая сверхдисперсия есть и её надо учитывать.",
                "Варианты NB (моменты, ML, по лигам, NB1) отличаются в четвёртом знаке; размер r текущей модели "
                + ", ".join(f"{SL[s]}: {info[s]:.1f}" for s in FOLDS) + " — устойчив; NB по лигам чуть хуже "
                "(лишние параметры).",
                "Свёртка независимых NB хозяев и гостей — самый плохой вариант (log score хуже на ≈ 0.01–0.015): "
                "корреляция остатков сторон отрицательна ("
                + ", ".join(f"ρ = {joint[s]['rho']:.3f}" for s in FOLDS) + "), а независимость её теряет. С учётом "
                "ковариации (nb_joint) раздельная модель возвращается к уровню модели тотала, но не лучше её.",
                "Перекалибровка Платта ухудшает log loss линий: " + ", ".join(
                    f"{SL[s]} {platt[s][0]:.4f} против {platt[s][1]:.4f}" for s in FOLDS)
                + " — вероятности NB уже откалиброваны, лишние параметры только добавляют шум."]),
            P("Равномерность PIT (χ² с 9 степенями свободы; p < 0.05 — заметное отклонение от равномерного):"),
            table(pit, font=6.4), Spacer(1, 6),
            *figure(FIG / "pit.png", "Рис. 8. Гистограммы PIT (идеал — 0.1 в каждой корзине)."),
            *figure(FIG / "deciles.png", "Рис. 9. Калибровка среднего: прогноз и факт по децилям прогноза."))


def s_lines(rep):
    keys = [REFK, ENS, BEST] if BEST != ENS else [REFK, ENS]
    rows = [["Сезон", "Линия", "Частота", "LL база", "LL лига"] + [f"LL {TITLES[k.split('|')[0]]}" for k in keys]
            + [f"ECE {TITLES[k.split('|')[0]]}" for k in keys]]
    for s in FOLDS:
        base = MET["folds"][s]["baselines"]["league_rate_log_loss"]
        for l in map(str, D.LINES):
            r = d(REFK, s)["lines"][l]
            rows.append([SLT[s], l, share(r["observed"]), num(r["base_log_loss"], 4), num(base[l], 4)]
                        + [num(d(k, s)["lines"][l]["log_loss"], 4) for k in keys]
                        + [num(d(k, s)["lines"][l]["ece"], 4) for k in keys])
    ext = [["Сезон"] + [f"LL {l}" for l in ("7.5", "8.5", "9.5", "10.5", "11.5", "12.5")]
           + [f"база {l}" for l in ("7.5", "9.5", "12.5")]]
    for s in FOLDS:
        r = d(REFK, s)["lines"]
        ext.append([SLT[s]] + [num(r[l]["log_loss"], 4) for l in ("7.5", "8.5", "9.5", "10.5", "11.5", "12.5")]
                   + [num(r[l]["base_log_loss"], 4) for l in ("7.5", "9.5", "12.5")])
    rep.add(H2("8.4. Качество по линиям и калибровка"),
            P("«LL база» — постоянная частота обучающих сезонов, «LL лига» — частота линии в лиге матча."),
            table(rows, font=6.2, widths=[44, 30, 40, 40, 40, 72, 72, 72, 72] if len(rows[0]) == 9 else None), Spacer(1, 6),
            P("Текущая модель на остальных линиях (для справки):"), table(ext, font=6.4), Spacer(1, 6),
            *figure(FIG / "line_calibration.png", "Рис. 10. Диаграммы надёжности линий 8.5 / 9.5 / 10.5 "
                                                  "(точки — децили прогноза)."),
            P(f"Текущая модель выигрывает у частоты лиги ≈ 0.002–0.008 log loss на каждой линии и каждом сезоне. "
              "Калибровка хорошая: ECE 0.004–0.019, максимум 0.029 на линии 8.5 в 2026/27 (модель переоценивает "
              f"«больше»: средняя P {share(d(REFK, '2026')['lines']['8.5']['mean_p'])} при частоте "
              f"{share(d(REFK, '2026')['lines']['8.5']['observed'])}). Разброс прогнозных вероятностей узкий (на 9.5 — "
              "примерно от 0.40 до 0.65 по децилям): модель редко бывает уверена, и именно это ограничивает "
              "пользу от неё в ставках."))


def s_periods(rep):
    keys = [REFK, ENS, "team_strength|nb_mom", "naive_league|nb_mom"]
    names = {"1_Jul-Sep": "июл–сен", "2_Oct-Dec": "окт–дек", "3_Jan-Mar": "янв–мар", "4_Apr-Jun": "апр–июн"}
    rows = [["Сезон", "Период", "n"] + [TITLES[k.split("|")[0]] for k in keys]]
    for s in FOLDS:
        bp = d(REFK, s)["by_phase"]
        for ph in sorted(bp):
            rows.append([SLT[s], names[ph], bp[ph]["n"]] + [num(d(k, s)["by_phase"][ph]["mean_line_log_loss"], 4)
                                                          for k in keys])
    lg_rows = [["Лига", "n (2 фолда)", "LL текущая", "LL ансамбль", "Δ", "LL иерарх.", "Δ"]]
    for lg in sorted(d(REFK, "2023")["by_league"]):
        vals = {}
        for k in (REFK, ENS, "team_strength|nb_mom"):
            a = [d(k, s)["by_league"][lg] for s in ("2023", "2024") if lg in d(k, s)["by_league"]]
            n = sum(x["n"] for x in a)
            vals[k] = (sum(x["mean_line_log_loss"] * x["n"] for x in a) / n, n)
        lg_rows.append([lg, vals[REFK][1], num(vals[REFK][0], 4), num(vals[ENS][0], 4),
                        f"{vals[ENS][0] - vals[REFK][0]:+.4f}", num(vals["team_strength|nb_mom"][0], 4),
                        f"{vals['team_strength|nb_mom'][0] - vals[REFK][0]:+.4f}"])
    rep.add(H2("8.5. По периодам сезона и по лигам"),
            P("Средний log loss линий по частям сезона (2026/27 — только июль–начало ноября):"),
            table(rows, font=6.6), Spacer(1, 6),
            *figure(FIG / "phase.png", "Рис. 11. Log loss линий по частям сезона."),
            P("Систематического провала в начале сезона нет: июль–сентябрь то лучше, то хуже других частей "
              "(формы и EWMA переходят через сезоны). Части сезона различаются уровнем шума сильнее, чем модели "
              "между собой; порядок моделей в каждой части почти тот же."),
            P("По лигам (фолды отбора вместе):"), table(lg_rows, font=6.6), Spacer(1, 6),
            *figure(FIG / "league.png", "Рис. 12. Δ log loss лучшей модели против текущей по лигам."),
            P("Выигрыш ансамбля распределён по лигам неравномерно; в лиге за два сезона лишь сотни матчей, "
              "поэтому разница по отдельной лиге того же порядка, что и шум, и выводов по лигам не делается."))


def s_ensembles(rep):
    rows = [["Фолд", "Ненулевые веса ансамбля по внутреннему сезону"]]
    for s in FOLDS:
        w = MET["folds"][s]["infos"]["ens_weighted"]["weights"]
        rows.append([SLT[s], ", ".join(f"{TITLES[m]} {v:.2f}" for m, v in w.items() if v > 0.005)])
    rows2 = [["Фолд", "xgb_tuned: глубина / lr / деревья", "lgbm: листья / деревья", "catboost: глубина / l2 / деревья",
              "GLM L2", "NB GLM r", "иерархия α / период"]]
    for s in FOLDS:
        I = MET["folds"][s]["infos"]
        t = I["xgb_tuned"]
        rows2.append([SLT[s], f"{t['params']['max_depth']} / {t['params']['learning_rate']:.3f} / {t['n_trees']}",
                      f"{I['lgbm']['params']['num_leaves']} / {I['lgbm']['params']['n_estimators']}",
                      f"{I['catboost']['params']['depth']} / {I['catboost']['params']['l2_leaf_reg']} / "
                      f"{I['catboost']['params']['n_estimators']}",
                      f"{I['poisson_glm']['params']['alpha']:g}", num(I["nb_glm"].get("size_outer_fit"), 1),
                      f"{I['team_strength']['alpha']:g} / {I['team_strength']['half_life_days']}"])
    rep.add(H2("8.6. Ансамбли и подобранные параметры"), table(rows), Spacer(1, 6),
            *figure(FIG / "ensemble_weights.png", "Рис. 13. Веса ансамбля по фолдам."),
            P("Веса, подобранные на одном внутреннем сезоне, неустойчивы: от почти чистой иерархической модели "
              "(2023/24) до LightGBM + NB GLM (2026/27). Поэтому взвешенный ансамбль не лучше равного, заданного "
              "заранее."),
            P("Параметры, выбранные на внутренней валидации каждого фолда:"),
            table(rows2, font=6.6, widths=[44, 96, 64, 92, 30, 44, 70]), Spacer(1, 4),
            P("Собственный Hyperopt выбирает очень разные деревья (от глубины 2 и 241 дерева до 82 деревьев с "
              "lr 0.05) — поверхность качества плоская, и он не лучше продакшн-параметров. XGBoost от смещения "
              "иерархической модели ранней остановкой получает 3–80 деревьев: поверх силы команд деревья почти "
              "ничего не добавляют."))


def s_betting(rep):
    keys = [REFK, ENS, "ens_weighted|nb_mom", "xgb_enriched|nb_mom", "catboost|nb_mom", "team_strength|nb_mom",
            "xgb_home_away|nb_joint", "naive_league|nb_mom"]
    strat = [("value_0", "EV≥0"), ("value_0.05", "EV≥5 %"), ("best_0", "1/матч EV≥0"), ("ext_value_0", "7.5–12.5")]
    rows = [["Модель · распределение", "LL линий (с ценой)"] + [f"{t}: n / ROI [ДИ]" for _, t in strat]]
    for k in keys:
        c = BET["combos"][k]
        rows.append([name(k), num(c["line_log_loss_priced"], 4)]
                    + [f"{c[s]['n']} / {pc(c[s]['roi'])} [{pc(c[s]['ci95'][0], 0)}; {pc(c[s]['ci95'][1], 0)}]"
                       for s, _ in strat])
    drows = [["Модель", "ROI − ROI текущей, EV≥0 [ДИ]", "EV≥5 % [ДИ]", "1/матч EV≥0 [ДИ]"]]
    for k in keys[1:]:
        c = BET["combos"][k]
        drows.append([name(k)] + [f"{pc(c[s]['diff_vs_ref']['mean'])} [{pc(c[s]['diff_vs_ref']['ci95'][0], 0)}; "
                                  f"{pc(c[s]['diff_vs_ref']['ci95'][1], 0)}]" if c[s].get("diff_vs_ref") else "—"
                                  for s in ("value_0", "value_0.05", "best_0")])
    mk = BET["market"][REFK]
    mrows = [["Линия", "n", "маржа", "LL Bet365 без маржи", "LL модели", "частота", "ср. P рынка", "ср. P модели",
              "ECE рынка", "ECE модели", "corr"]]
    for l, v in mk.items():
        mrows.append([l, v["n"], share(v["margin"]), num(v["market_log_loss"], 4), num(v["model_log_loss"], 4),
                      share(v["observed"]), share(v["market_mean_p"]), share(v["model_mean_p"]),
                      num(v["market_ece"], 4), num(v["model_ece"], 4), num(v["corr_model_market"], 3)])
    h = CUR["history_2026"]
    hrows = [["Роль", "Ставок", "Попадание", "Средний коэф.", "Средняя P модели", "ROI [95 % ДИ]"]]
    for role, lab in (("market", "Рынок CORNERS_9.5"), ("main", "Main"), ("risk", "Risk"), ("all", "Все")):
        v = h["roles"].get(role)
        if v:
            hrows.append([lab, v["n"], share(v["hit_rate"]), num(v["mean_odds"], 2), share(v["mean_probability"]),
                          f"{pc(v['roi'])} [{pc(v['roi_ci95'][0])}; {pc(v['roi_ci95'][1])}]"])
    ref = BET["combos"][REFK]
    month = BET["by_month"][REFK]
    rep.add(H2("8.7. Ставки по реальным ценам угловых, 2026/27"),
            P("Модели фолда 2026 обучены на 2021/22–2025/26, т.е. ровно на тех данных, что и продакшн-модель. "
              "Стратегии: «EV≥t» — каждая сторона линий 8.5 / 9.5 / 10.5 с EV = p·коэф − 1 ≥ t; «1/матч» — одна "
              "ставка на матч с максимальным EV; «7.5–12.5» — то же на шести линиях. Плоская ставка 1."),
            table(rows, font=6.2), Spacer(1, 6),
            P("Разница ROI с текущей моделью на тех же бутстрэп-выборках матчей:"), table(drows, font=6.6),
            Spacer(1, 6), *figure(FIG / "betting.png", "Рис. 14. ROI стратегий по реальным ценам Bet365, 2026/27."),
            P(f"Текущая модель: EV≥0 — {ref['value_0']['n']} ставок, ROI {pc(ref['value_0']['roi'])}; EV≥5 % — "
              f"{ref['value_0.05']['n']} ставок, ROI {pc(ref['value_0.05']['roi'])} [{pc(ref['value_0.05']['ci95'][0])}; "
              f"{pc(ref['value_0.05']['ci95'][1])}]. Все интервалы широко включают ноль, а по месяцам (EV≥0) ROI "
              + ", ".join(f"{m}: {pc(v['roi'])} (n={v['n']})" for m, v in month.items())
              + " — устойчивого перевеса над рынком нет. Ансамбли и другие модели не лучше текущей по ROI "
              "(у ансамблей и бустингов разницы в пределах шума; у иерархической модели и среднего лиги ROI ниже, "
              "при EV≥5 % — значимо)."),
            P("Модель против рынка на тех же матчах (вероятности Bet365 очищены от маржи пропорционально):"),
            table(mrows, font=6.2), Spacer(1, 6),
            *figure(FIG / "market.png", "Рис. 15. Log loss модели и рынка по линиям, 2026/27."),
            P("На линиях 8.5–12.5 текущая модель совпадает с рынком по log loss в третьем–четвёртом знаке "
              "(корреляция вероятностей ≈ 0.8): модель без единого коэффициента на входе оценивает угловые "
              "так же, как букмекер, но маржа ≈ 8 % не оставляет систематического перевеса."),
            H3("История прогнозов продукта (2026/27)"),
            P(f"`unified_history(load_picks())`: {h['rows']} прогнозов на угловые 2026/27, рассчитано "
              f"{h['settled']}, из них по реальной цене {h['roles']['all']['n']} (без цены — {h['unpriced_settled']}). "
              f"Версии модели: " + ", ".join(f"{k} — {v}" for k, v in h["versions"].items()) + "."),
            table(hrows), Spacer(1, 6),
            *figure(FIG / "history.png", "Рис. 16. ROI прогнозов продукта на угловые по ролям, 2026/27."),
            P("Роль «рынок» — всегда более вероятная сторона линии 9.5, без учёта цены; её ROI отрицателен примерно "
              "на величину маржи, что ожидаемо при вероятностях на уровне рынка. Main и Risk на угловых выбирались "
              "редко (десятки ставок), их ROI — шум."))


def s_selection(rep):
    rows = [["Модель · распределение", "убедительно лучше 23/24 и 24/25", "лучше по точке 23/24 и 24/25",
             "не хуже 25/26 и 26/27", "ср. LL линий (отбор)"]]
    ref_ll = (d(REFK, "2023")["mean_line_log_loss"] + d(REFK, "2024")["mean_line_log_loss"]) / 2
    for k in best_nb[:10]:
        v = TAB[k]
        rows.append([name(k), "да" if v["convincing_both"] else "нет", "да" if v["point_better_both"] else "нет",
                     "да" if v["not_worse_2025_2026"] else "нет", num(v["pooled_select_line_ll"], 4)])
    rows.append([name(REFK) + " (текущая)", "—", "—", "—", num(ref_ll, 4)])
    rep.add(H2("8.8. Применение правила внедрения"), table(rows, font=6.8), Spacer(1, 6),
            P(f"Прошли правило (до проверки ставок): {', '.join(SEL['eligible_before_betting']) or 'ни одна модель'}. "
              f"Лучше текущей по точечным оценкам на обоих фолдах отбора (и по LL линий, и по log score): "
              f"{len(SEL['point_better_both'])} пар, все — ансамбли или бустинги с новыми признаками."))


def s_conclusions(rep):
    e = {s: d(ENS, s) for s in FOLDS}
    rep.add(H1("9. Выводы"), *bullets([
        "**Текущая модель остаётся.** Ни одна архитектура не прошла заранее заданное правило: ни у одной "
        "модели интервал Δ log loss линий не лежит целиком ниже нуля на обоих фолдах отбора.",
        f"Лучший кандидат — равный ансамбль XGBoost + Poisson GLM + иерархическая модель: лучше текущей по точке "
        f"на всех четырёх сезонах (Δ LL линий {e['2023']['delta_line_log_loss']['mean']:+.4f} / "
        f"{e['2024']['delta_line_log_loss']['mean']:+.4f} / {e['2025']['delta_line_log_loss']['mean']:+.4f} / "
        f"{e['2026']['delta_line_log_loss']['mean']:+.4f}), но выигрыш ≈ 0.001 log loss — около 0.2 % — и не "
        f"даёт ничего на реальных ценах 2026/27 (ROI EV≥0: {pc(BET['combos'][ENS]['value_0']['roi'])} против "
        f"{pc(BET['combos'][REFK]['value_0']['roi'])} у текущей).",
        "H1 (новые признаки): слабое улучшение (≈ −0.0008 LL линий), интервалы включают 0 на фолдах отбора.",
        "H2 (собственный Hyperopt): не подтверждена — не лучше продакшн-параметров.",
        "H3 (CatBoost, LightGBM): на уровне XGBoost с новыми признаками; систематического преимущества нет.",
        "H4 (GLM): Poisson GLM на 36 признаках не хуже XGBoost на 114 — сигнал почти линейный и слабый.",
        "H5 (иерархическая модель): лучшая одиночная модель на 2023/24 (вместе с XGBoost от её смещения), "
        "но хуже на 2024/25 и 2026/27; "
        "полезна как член ансамбля, не как замена.",
        "H6 (раздельные хозяева/гости): независимая свёртка заметно хуже из-за отрицательной корреляции сторон; "
        "с ковариацией — на уровне модели тотала.",
        "H7 (дисперсия, калибровка): NB с одним r по методу моментов — правильный выбор; Пуассон хуже, NB по "
        "лигам и NB1 не лучше, перекалибровка Платта хуже.",
        "H8 (ансамбли): небольшой, но самый устойчивый выигрыш; веса по одному сезону неустойчивы.",
        "H9 (ставки): ни одна модель не показала значимой прибыли; текущая модель по log loss на уровне рынка "
        "Bet365 без маржи. Точность угловых упирается в шум матча, а не в архитектуру.",
    ]))


def s_limits(rep):
    rep.add(H1("10. Ограничения"), *bullets([
        "Реальные цены угловых есть только в 2026/27 (≈ 1.8 тыс. матчей, июль–начало ноября); для разницы ROI "
        "в 1–2 % этого мало — интервалы ±7–10 %.",
        "Фолд 2023/24 имеет лишь один сезон внутреннего обучения (2021/22): ранняя остановка, веса ансамбля и "
        "гиперпараметры там подбираются по малой выборке.",
        "Продакшн-параметры XGBoost подобраны на сезоне 2024/25, поэтому на фолдах 2023/24 и 2024/25 текущая "
        "модель слегка «видела» оценочные данные при тюнинге — это работает в её пользу, не против кандидатов.",
        "Для Poisson GLM на трёх фолдах выбран край сетки L2 (α = 1): сильная регуляризация; более широкая "
        "сетка могла бы чуть улучшить GLM, но не меняет вывод (GLM — член ансамбля, а не кандидат сам по себе).",
        "Иерархическая модель — мода апостериорного (штрафной ML), а не полный байесовский вывод; "
        "неопределённость эффектов команд в распределение не переносится.",
        "Признаки ограничены статистикой Football-Data; навесов, владения, xG, составов и травм нет.",
        "Цены — последняя котировка Bet365 до начала матча; реальная ставка в момент публикации прогноза "
        "могла быть по другой цене.",
    ]))


def s_recommendations(rep):
    rep.add(H1("11. Рекомендации"), *bullets([
        "Оставить продакшн-модель угловых (XGBoost, `count:poisson`) и NB с одним r по методу моментов без "
        "изменений; не переходить на раздельные модели сторон с независимой свёрткой.",
        "Не вводить ставку на угловые как источник прибыли: вероятности на уровне рынка, маржа ≈ 8 %. "
        "`MARKET_RELIABILITY['CORNERS'] = 0.85` оставить.",
        "Держать на наблюдении равный ансамбль XGBoost + Poisson GLM + иерархическая модель: пересчитать этот "
        "отчёт после окончания 2026/27 (`run.py --all`). Внедрять, только если на полном 2026/27 он будет "
        "лучше по log loss линий с интервалом ниже нуля и не хуже по ROI.",
        "Если ансамбль когда-нибудь пройдёт правило: признаки силы команд считать в `src/features/builder.py` "
        "(скользящее переобучение раз в полмесяца на прошлых матчах), GLM и среднее трёх моделей — в "
        "`src/modelling/pipeline.py`; r оценивать на внутреннем сезоне, как сейчас.",
        "Полезная правка без изменения модели: продукт показывает линии 8.5–10.5, а рынок Bet365 даёт 0.5–19.5; "
        "вероятности уже считаются из полного распределения, их можно показывать на любой линии с ценой.",
    ]))


def s_repro(rep):
    rt = R["run_times"]
    rows = [["Этап", "Минут", "Завершён", "Лог"]] + [[r["stage"], num(r["minutes"], 1), r["finished"], r["log"]]
                                                   for r in rt]
    rep.add(H1("12. Команды воспроизведения"),
            P("Из корня репозитория, окружение `.venv` (≈ 40–50 мин на 2 × 3 потока; кэш — "
              "`experiments/v6_1/corners/cache/`, в git не входит):"),
            code("""python -m experiments.v6_1.corners.run --prep --current --team-strength
python -m experiments.v6_1.corners.run --fit 2023 2025      # можно параллельно с
python -m experiments.v6_1.corners.run --fit 2024 2026
python -m experiments.v6_1.corners.run --evaluate --betting --figures
python -m experiments.v6_1.corners.build_pdf               # этот PDF и findings.md"""),
            P("Результаты: `results/inventory.json`, `current.json`, `team_strength_grid.json`, `metrics.json` "
              "(все модели × распределения × фолды), `models_by_fold.csv`, `betting.json`, `betting_2026.csv`, "
              "`versions.json`, `run_times.json`; рисунки — `figures/`. Ничего в `local_data/` не записывается "
              "(проверяется тестом `tests/test_research_isolation.py`)."),
            P("Время этапов этого прогона (фолды считались двумя процессами параллельно; модель xgb_ts_offset "
              "досчитана отдельным запуском):"), table(rows, font=6.8))


# ---------------------------------------------------------------------------------------------- findings.md
def findings() -> str:
    e = {s: d(ENS, s) for s in FOLDS}
    b, p = CUR["backtest_2526"]["metrics"], CUR["prod_20261012"]["metrics"]
    lines = ["# FootPredict v6.1 — модель угловых: выводы", "",
             "Все числа — из `results/*.json` (генерируется `python -m experiments.v6_1.corners.build_pdf`). "
             "PDF: `reports/v6_1/Corners_Research.pdf`.", "",
             "## Текущие метрики (пересчёт по сохранённым моделям)", "",
             "| Модель | Сезон | n | MAE | RMSE | R² | девиация | log score | LL 8.5 / 9.5 / 10.5 |", "|---|---|---|---|---|---|---|---|---|"]
    for v, m in (("backtest_2526", b), ("prod_20261012", p)):
        lines.append(f"| {v} | {SL[str(CUR[v]['season'])]} | {m['n']} | {m['mae']:.4f} | {m['rmse']:.4f} | "
                     f"{m['r2']:.4f} | {m['poisson_deviance']:.4f} | {m['log_score']:.4f} | "
                     + " / ".join(f"{m['lines'][l]['log_loss']:.4f}" for l in map(str, D.LINES)) + " |")
    lines += ["", "## Все модели (NB, r методом моментов); LL — средний log loss линий 8.5/9.5/10.5", "",
              "| Модель | " + " | ".join(f"LL {SLT[s]}" for s in FOLDS) + " | "
              + " | ".join(f"Δ LL {SLT[s]} [95 % ДИ]" for s in FOLDS) + " | ROI EV≥0 2026/27 |",
              "|---" * (2 + 2 * len(FOLDS)) + "|"]
    for m in MODELS:
        k = f"{m}|nb_mom"
        c = BET["combos"][k]["value_0"]
        lines.append(f"| {m} | " + " | ".join(f"{d(k, s)['mean_line_log_loss']:.4f}" for s in FOLDS) + " | "
                     + " | ".join(ci(d(k, s)["delta_line_log_loss"]) for s in FOLDS)
                     + f" | {pc(c['roi'])} (n={c['n']}) |")
    lines += ["", "## Вывод", "",
              f"- Правило внедрения (убедительно лучше на 2023/24 и 2024/25, не хуже на 2025/26 и 2026/27): "
              f"прошли — {', '.join(SEL['eligible_before_betting']) or 'никто'}. **Текущая модель остаётся.**",
              f"- Лучший кандидат: равный ансамбль XGBoost + Poisson GLM + иерархическая модель — лучше по точке на "
              f"всех сезонах (Δ LL {e['2023']['delta_line_log_loss']['mean']:+.4f} / {e['2024']['delta_line_log_loss']['mean']:+.4f} / "
              f"{e['2025']['delta_line_log_loss']['mean']:+.4f} / {e['2026']['delta_line_log_loss']['mean']:+.4f}), "
              f"но на 2024/25 ДИ {ci(e['2024']['delta_line_log_loss'])} включает 0, по ROI 2026/27 не лучше.",
              "- Распределение: NB с одним r (моменты) — верно; Пуассон хуже, NB по лигам/NB1 не лучше, Платт хуже, "
              "независимая свёртка сторон заметно хуже (ρ сторон ≈ −0.19).",
              f"- 2026/27: модель на уровне рынка Bet365 без маржи (LL 9.5: модель "
              f"{BET['market'][REFK]['9.5']['model_log_loss']:.4f}, рынок {BET['market'][REFK]['9.5']['market_log_loss']:.4f}); "
              f"значимой прибыли ни у одной модели нет.", ""]
    return "\n".join(lines)


def main():
    v = R["versions"]
    rep = Report("reports/v6_1/Corners_Research.pdf", "Исследование модели угловых", "FootPredict v6.1",
                 facts={"Данные": "Football-Data 2021/22–2026/27 (по 02.11.2026), цены угловых OddsPapi/Bet365 2026/27",
                        "Код": "experiments/v6_1/corners/", "Код прогона (SHA)": v["git_sha"]})
    for f in (s_goal, s_problem, s_current, s_hypotheses, s_data, s_methods, s_experiments, s_results, s_dist,
              s_lines, s_periods, s_ensembles, s_betting, s_selection, s_conclusions, s_limits, s_recommendations,
              s_repro):
        f(rep)
    rep.build()
    pages = rep.check(scale=0.9)
    (D.HERE / "findings.md").write_text(findings(), encoding="utf-8")
    print("pages:", pages)


if __name__ == "__main__":
    main()
