"""Builds experiments/v6_research/findings.md: Russian text below + every table generated from
results/*.json (results/summary_tables.md and the before/after table), so no table number is typed by hand.

    python -m experiments.v6_research.findings_build      # after `run --all`
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TEMPLATE = r"""
# FootPredict v6 — исследование моделей (ничьи, голы, угловые, производные рынки, Risk)

## Воспроизведение

Все числа ниже взяты из `experiments/v6_research/results/*.json` (таблицы сгенерированы из них скриптом `findings_build.py`). Валидация строго временная: фолд *s* — модель обучается на сезонах < *s* (с 2021/22), оценивается на сезоне *s*; все подгоняемые константы (температура, vector scaling, веса блендов, rho Диксона–Коулса, размер NB, L2 у GLM, число деревьев новых моделей) — на внутреннем сезоне *s*−1 моделью, обученной на < *s*−1. **Выбор — только по фолдам 2023/24 и 2024/25. Сезон 2025/26 (помечен «*» или «2025») — только для информации.** Параметры XGBoost везде — настроенные параметры активной продакшн-версии `prod_20261012` (без новых поисков); базовая модель фолда 2025 воспроизводит продакшн-метрику теста (log loss 1.0154).

- Окружение: `/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python`, запуск из корня репозитория.
- Всё целиком (≈ 35–45 мин на 14 ядрах): `python -m experiments.v6_research.run --all`, затем `python -m experiments.v6_research.findings_build`.
- По частям: `--base` (кэш прогнозов продакшн-моделей по фолдам), `--draw` (ничьи), `--goals` (голы), `--corners` (угловые), `--markets` (производные рынки), `--risk` (правило Risk), `--timing` (чистый замер времени обучения; запускать без параллельной нагрузки), `--summary` (сводные таблицы и рисунок). `--refresh` пересчитывает кэш.
- Результаты: `results/draw.json`, `goals.json`, `corners.json`, `markets.json`, `risk.json`, `timing.json`, `summary.json`, `summary_tables.md` (все таблицы полностью). Рисунки: `figures/*.png`, подписи и раздел каждого рисунка — `figures/captions.json`. Кэш прогнозов и логи (`cache/`, `logs/`) в git не входят.
- Код: `common.py` (фолды, обучение как в продакшне, метрики), `scores.py` (векторизованная матрица счетов; сверена с `src/markets/distributions.py`, расхождение 0.0), `draw.py`, `goals.py`, `corners.py`, `markets.py`, `risk.py`, `timing.py`, `summary.py`, `findings_build.py`, `run.py`. Продакшн-код, данные и модели не изменялись.

## Ничьи (Draw)

**Диагностика (почему argmax почти никогда не даёт ничью).**

- Кодировка верна: `RESULT_CLASSES = ["H", "D", "A"]` → 0/1/2, `XGBClassifier.classes_ = [0, 1, 2]`, столбцы `predict_proba` идут в том же порядке. Проверка по данным: в матчах 2023/24, выигранных гостями, средняя P(A) = 0.381 > P(H) = 0.364; в выигранных хозяевами P(H) = 0.497.
- Цель `multi:softprob`, метрика `mlogloss`, тюнинг по log loss. Постобработки нет: `DRAW_RULE = False`, `draw_multiplier = 1.0` в метаданных модели.
- Доля ничьих по сезонам 2021–2025: 26.6 / 25.6 / 26.2 / 26.0 / 26.6 %. Средняя P(ничья) модели: 0.252 / 0.258 / 0.261 (2023 / 2024 / 2025*) при факте 26.2 / 26.0 / 26.6 % — вероятность ничьей откалибрована (ECE по ничьей 0.013 / 0.006 / 0.006; у букмекера 0.009 / 0.004 / 0.010).
- Ничья — argmax только в 0.26 / 0.13 / 0.08 % матчей; у нормированных цен Bet365 — в 0.11 / 0.08 / 0.04 %. Максимум P(ничья) у модели 0.390 / 0.361 / 0.358, P(ничья) ≥ 0.30 лишь у 10.8 / 14.2 / 12.0 % матчей.
- Причина структурная: даже в самых равных матчах (|P(П1) − P(П2)| ≤ 0.05; 2024/25: 843 матча) фактическая доля ничьих 29.5 %, средняя P(ничья) 0.290, а большая из P(П1), P(П2) в среднем 0.367. Ничья почти никогда не является самым вероятным исходом — ни у модели, ни у рынка. Это свойство задачи, а не ошибка кодировки или калибровки.
- Постобработка множителем `argmax(P_H, m·P_D, P_A)`: m = 1.3 даёт 9.2 / 12.9 / 12.8 % ничьих, m = 1.5 — 26.3 / 32.7 / 33.7 % (`draw.json → diagnosis.folds.*.multiplier_effect`).

**Качество вероятностей по вариантам** (log loss и Brier — главные метрики; «лучше на обоих» = ниже log loss И Brier базовой модели на 2023 и 2024; в столбцах «2025*» — только для информации):

<<draw.probability>>

Те же строки, только матчи с ценами Bet365 (нормированные 1/кэф):

<<draw.bookmaker>>

Классовые метрики (argmax) ключевых вариантов (все варианты — в `results/summary_tables.md`):

<<draw.classes_filtered>>

Что показали варианты:

- (b) Веса ничьих «покупают» recall ценой вероятностей: w = 1.4 — draw recall 18.0 / 24.6 %, но log loss +0.0064 / +0.0097, ECE ничьей 0.063 (2024); balanced — +0.0158 / +0.0209. Это прямое смещение вероятностей, вредное для EV и рынков.
- (c) Температура и vector scaling, подогнанные на внутреннем сезоне, не помогают: модель уже откалибрована, параметры нестабильны между фолдами (T = 0.950 / 1.005 / 1.010; сдвиг логита ничьей в vector scaling −0.64 / −0.17 / −0.31).
- (d) Бинарная модель «ничья / не ничья» (+14 признаков |разностей|): вес бленда на внутреннем сезоне 1.0 / 0.25 / 0.55; на 2023 хуже (+0.0005), на 2024 −0.0001.
- (e, f) Бленд с 1X2 из матрицы счетов моделей голов — лучший по сумме 2023+2024: Диксон–Коулс-бленд (rho = −0.014 / −0.056 / −0.027, вес 0.6 / 0.4 / 0.5), объединённый log loss 1.00827 против 1.00863; на 2024 −0.0008 [95 % ДИ −0.0015; −0.0001], но на 2023 +0.00007 [−0.0010; +0.0012]. На обоих фолдах не лучше → по правилу отбора не принимается; выигрыш ≈ 0.04 % log loss.
- Наивные базовые линии (константа частот обучающих сезонов; частоты по лиге): log loss 1.0762 / 1.0734 / 1.0771 — модель лучше на ≈ 0.06; букмекер лучше модели на ≈ 0.012–0.014.

**Правило решения (меняет только названный исход; вероятности те же).** Сетка: argmax; множитель m ∈ 1.05…1.80; порог «ничья, если P(D) ≥ t и |P(H) − P(A)| ≤ g» (t ∈ 0.24…0.34, g ∈ 0.04…0.30 и без условия). Критерий: максимум macro F1 при доле предсказанных ничьих не выше доли ничьих обучающих сезонов (26.1 %). Выбрано на объединённых 2023+2024: **t = 0.28, g = 0.20** (в выборке отбора: macro F1 0.441, accuracy 47.4 %, доля ничьих 25.3 %).

<<draw.rules>>

Честная оценка (правило выбирается на внутреннем сезоне каждого фолда и применяется к следующему):

<<draw.rules_honest>>

Draw precision ≈ 28–32 % при базовой доле ничьих 26 % — названные ничьи лишь немного точнее случайных; accuracy падает на 1.3–3.3 п.п. ROI названного исхода по Bet365 приведён только для информации (выбор по нему не делался).

**Рекомендация.**

1. Модель вероятностей: оставить продакшн XGBoost без изменений (ни один вариант не лучше на обоих фолдах; веса ничьих ухудшают log loss и Brier). Диксон–Коулс-бленд — кандидат для повторной проверки на 2026/27, не для внедрения сейчас.
2. Правило решения для названного исхода 1X2 (история прогнозов, `role = "market"`): если цель — показывать ничьи, принять пороговое правило «ничья, если P(D) ≥ 0.28 и |P(H) − P(A)| ≤ 0.20, иначе argmax» (macro F1 0.369 → 0.445 и 0.367 → 0.435, draw recall 0.2 % → 25.1 % и 33.0 % на 2023 и 2024; 2025*: 0.362 → 0.436, 35.2 %) с явной ценой в accuracy. Где менять: `src/modelling/decision.py` (функция `decide`, режим «threshold») и её вызов в `src/prediction/engine.py::predict_matches`; параметры в `src/config.py` (например `DRAW_RULE = True`, `DRAW_THRESHOLD = 0.28`, `DRAW_MAX_GAP = 0.20`). Main/Risk, EV и все рынки используют вероятности и не затрагиваются. Если ключевая метрика — accuracy, оставить argmax.

## Модели голов

Сравнение: продакшн XGBoost (`count:poisson`), Poisson GLM (`PoissonRegressor` на том же наборе признаков: медианная импутация + индикаторы пропусков + стандартизация, L2 по внутреннему сезону; выбрано α = 1.0 во всех фолдах — край сетки 1e-3…1) и бленд `(1 − w)·XGB + w·GLM` (w на внутреннем сезоне: хозяева 0.8 / 0.4 / 0.5, гости 0.9 / 0.7 / 0.7). Правило замены: девиация ниже на обоих фолдах и 95 % бутстреп-ДИ разницы целиком ниже 0 на обоих.

<<goals>>

- Смещения почти нет: средний прогноз хозяев 1.520 / 1.497 / 1.488 при факте 1.533 / 1.484 / 1.486; гостей 1.223 / 1.216 / 1.208 при факте 1.251 / 1.202 / 1.212.
- Калибровка по децилям прогноза (рисунок `goals_calibration.png`): крайние децили XGBoost 2024/25 — хозяева прогноз 0.98 → факт 0.95 и 2.38 → 2.37; гости 0.77 → 0.77 и 1.98 → 1.96. Медиана остатка отрицательна (−0.19 … −0.23 у хозяев), распределение остатков скошено вправо, как и положено для пуассоновского счёта (`goals_residuals.png`).
- Хозяева: GLM и бленд смешанно (2023 хуже, 2024 лучше) → оставить.
- Гости: GLM и бленд лучше на обоих фолдах (бленд −0.0030 [−0.0065; +0.0005] и −0.0021 [−0.0043; +0.0001]; 2025*: −0.0011), но ДИ включает 0 — это не явное улучшение (≈ 0.2 % девиации).

**Рекомендация:** оставить продакшн XGBoost для обеих моделей. Бленд для голов гостей (w_GLM ≈ 0.7, `PoissonRegressor(alpha=1.0)` на признаках `away_goals`) — кандидат на v6.x при подтверждении на 2026/27; тогда менять `src/modelling/pipeline.py::train_all` (обучение GLM рядом с XGBoost) и `ModelBundle.predict` (смешивание ставок).

## Угловые

**Признаки.** В продакшн-модели угловых 114 числовых признаков (+21 one-hot лиги = 135): угловые за/против за сезон, по месту, окна 5/10/15, тренды, venue form 5, разности, сумма форм, «угловые против пропущенных соперника», средние лиги (общие, дома, в гостях), corner pressure (доля угловых и ударов), удары и удары в створ за форму. Не используется группа `opp_adjusted_corners` (`[ha]_form10_adj_cf/ca`). Покрытие статистикой угловых ≥ 98.3 % во всех 21 лиге — `corner_leagues()` включает все лиги. В эксперименте добавлен 31 строго сдвинутый признак по истории матчей (EWMA угловых за/против с полураспадом 5 и 15 матчей, venue-EWMA, EWMA ударов и ударов в створ, средний тотал угловых за 38 матчей, аддитивное и мультипликативное ожидание угловых). Проверка утечки: для 40 случайных матчей признак пересчитан перебором только по более ранним матчам — расхождение 0.0. Ни один признак не коррелирует с тоталом угловых сильнее |r| = 0.118 (сезоны 2021–2022; лучший — новый `x_exp_total_add`; лучшие продакшн — `league_avg_corners` 0.105, `form_corners_sum` 0.104): потолок предсказуемости низкий.

**Варианты** (размер NB r — по прогнозам внутреннего сезона, как в продакшне; линии — NB-распределение `corners_distribution`; у варианта «дома + в гостях» — свёртка двух NB):

<<corners.0>>

Вероятности линий против базовой частоты обучающих сезонов и частоты по лиге (продакшн-модель):

<<corners.1>>

- Все модели лишь немного лучше среднего лиги: RMSE 3.385 / 3.350 / 3.350 против 3.412 / 3.381 / 3.369, R² 0.026 / 0.027 / 0.019. Log loss линий лучше базовой частоты на ≈ 1.1–1.4 %.
- Дисперсия тотала близка к пуассоновской: r ≈ 60–71, т.е. дисперсия лишь на ≈ 15 % выше среднего. У раздельных моделей r ≈ 11–13 на сторону: угловые хозяев и гостей отрицательно связаны, и свёртка независимых NB это теряет (на 2024 девиация хуже на +0.0046).
- Новые признаки (`xgb_enriched`, `xgb_clean`) улучшают log loss линий на 0.0001–0.0009, девиацию — смешанно; GLM с новыми признаками и NB GLM хуже (NB GLM: девиация +0.0153 / +0.0063).
- Бленд XGB + Poisson GLM с новыми признаками (w_GLM 0.5 / 0.4 / 0.3) — единственный вариант лучше и по девиации, и по log loss линий на обоих фолдах, но ДИ девиации включает 0 (−0.0017 [−0.0043; +0.0008], −0.0007 [−0.0027; +0.0012]); на 2025* −0.0033 [−0.0044; −0.0020].

**Рекомендация:** оставить продакшн-модель угловых и NB с одним r. Бленд — кандидат при подтверждении на 2026/27 (тогда: расчёт 31 признака в `src/features/builder.py`, GLM в `src/modelling/pipeline.py`). R² выше ≈ 0.03 на этих данных не достиг ни один вариант.

## Производные рынки

Цепочка как в продакшне: `score_matrix` (независимый Пуассон, rho = 0) → `reconcile_with_1x2` → `goal_markets`; угловые — `corners_distribution(mu, r)` → `corner_markets` (r по внутреннему сезону: 60.3 / 63.9 / 71.1). Проверено для каждого матча трёх сезонов (7 247 / 7 129 / 7 094) — 23 проверки, допуск 1e-6:

<<markets.0>>

Нарушений нет; единственное заметное ненулевое расхождение (6.5e-8) — float32-выход классификатора. Целые линии азиатского гандикапа: выигрыш + возврат + проигрыш = 1, возврат одинаков для обеих сторон и равен «ничьей» европейского гандикапа; AH −0.5 (хозяева) = П1, AH +0.5 = 1X; тоталы и линии угловых монотонны.

**Расхождение 1X2 матрицы счетов и классификатора** (до согласования):

<<markets.2>>

Пуассоновская матрица систематически занижает ничью (в среднем на 0.011–0.016) и хуже по log loss, поэтому согласование с классификатором оправдано. Для тоталов и BTTS оно почти ничего не меняет (тотал 2.5 на 2024/25: 0.6828 против 0.6829 без согласования).

**Качество рынков** (log loss против базовой частоты обучающих сезонов; для целых линий AH/EH — 3 исхода выигрыш/возврат/проигрыш):

<<markets.1>>

- Все рынки лучше базовой частоты на обоих фолдах, но прирост очень разный: AH/EH и двойной шанс 1X/X2 — 5.7–10.8 %, 1X2 — 5.7–6.6 %, точный счёт — 3.1–3.7 %, тоталы 1.5–3.5 — 1.5–2.6 %, угловые — 1.1–1.4 %, двойной шанс 12 — 0.7–1.1 %, BTTS — 0.4–0.6 %. Accuracy для частых исходов (тотал больше 1.5, «12») не показательна и не использовалась.
- Bet365 (нормированные цены) лучше модели: тотал 2.5 — 0.6730 / 0.6774 / 0.6812 против 0.6792 / 0.6828 / 0.6863; 1X2 — 0.9916 / 0.9988 / 1.0040 против 1.0057 / 1.0117 / 1.0157.
- Возвраты на целых линиях: AH −1 прогноз 0.207 / 0.207 / 0.208 против факта 0.209 / 0.225 / 0.222 (на 2024/25 победа ровно в 1 мяч недооценена), AH +1 — 0.170 / 0.169 / 0.169 против 0.167 / 0.169 / 0.172.

**Рекомендация:** изменений не требуется; рынки согласованы и все лучше базовой частоты. BTTS и «12» почти не несут информации сверх частоты — их вес надёжности в `MARKET_RELIABILITY` повышать не стоит.

## Правило Risk

Для каждого матча воспроизведена продакшн-цепочка (модели фолда, все рынки, реальные цены Bet365 из Football-Data: 1X2, тотал 2.5 и одна линия AH на матч — столбцы `B365H/D/A`, `B365>2.5/<2.5`, `B365AHH/AHA`; это предматчевые цены, не closing `B365C*`), Main = `main_prediction()`. Варианты Risk: текущее правило; «EV ≥ 0 сначала»; пол EV −3 / −5 / −8 % с порядком «положительный EV (лучший по score) → в пределах пола (наименее отрицательный EV или лучший score) → текущее правило». Мягкое окно 3.0–6.0 и прочие ограничения (не Main, не тот же исход, не ставка против Main, p ≥ 0.12) сохранены. Расчёт и сводка — `settle` / `summarise` из `scripts/decision_study.py`.

<<risk>>

- Гарантия выполняется во всех вариантах и сезонах: Risk есть у каждого матча с Main (7 247 / 7 129 / 7 094 матчей, 0 без Risk), Risk ни разу не совпал с Main. Покрытие 100 %.
- В варианте «пол −5 %» ветка «положительный EV» срабатывает в 79–86 % матчей, «в пределах пола» — в 13–19 %, откат к текущему правилу — в 1–2 %. Средний модельный EV выбранных ставок с ценой растёт до +8.2…9.4 % на фолдах отбора (+6.3…7.1 % на 2025*; у текущего правила +2.4 / +1.6 / +0.0 %), но фактический ROI не растёт: модельный EV на этих коэффициентах завышен примерно на 15–16 п.п.
- EV-варианты делают Risk менее рискованным: средний коэффициент 3.5 → 3.0–3.3, hit rate 28 % → 32–35 %.
- Ни один вариант не лучше текущего по ROI на обоих фолдах (на 2023 все лучше, на 2024 все хуже). Формальный победитель по худшему фолду — пол −5 % с наименее отрицательным EV: худший ROI −7.0 % против −7.2 %, но объединённый ROI −6.69 % против −6.55 % у текущего; разница много меньше 95 % ДИ (≈ ±3.5–3.8 п.п.). На 2025* все EV-варианты хуже (−8.3…−9.1 % против −5.8 %).

**Рекомендация:** оставить текущее правило Risk (`RISK_EV_WEIGHT = 0.0`, без пола EV). Если по продуктовым причинам нужен «сначала положительный EV», наименее затратный вариант — пол −5 % с выбором наименее отрицательного EV (реализация: `src/markets/recommendation.py::risk_prediction`, новый параметр, например `RISK_EV_FLOOR = -0.05`, и трёхступенчатый выбор как в `experiments/v6_research/risk.py::risk_with_floor`), но улучшения ROI это не даёт.

## Сводная таблица Before/After

Чистый замер времени обучения (фолд 2024: обучение на сезонах < 2024; последовательно, без параллельной нагрузки; `results/timing.json`). Времена внутри JSON отдельных исследований замерены при параллельных запусках и для сравнения не используются.

<<timing>>

Все варианты по задачам (время = все нужные обучения: внешняя и внутренняя модель и т.п., по чистому замеру; «—» — отдельно не замерялось):

<<before_after>>

Итог (рисунок `summary_before_after.png`): ни один вариант моделей не прошёл правило замены; изменение рекомендуется только для правила решения 1X2 (порог t = 0.28, g = 0.20), которое не меняет вероятности. Продакшн-конфигурация моделей, рынков и правила Risk остаётся.

| Задача | Рекомендация | Ключевые цифры 2023 / 2024 / 2025* |
|---|---|---|
| 1X2 вероятности | оставить продакшн XGBoost | log loss <<v.ll>> |
| 1X2 правило решения | порог P(D) ≥ 0.28 и \|P(H) − P(A)\| ≤ 0.20 | macro F1 <<v.f1>>; draw recall <<v.dr>>; draw precision <<v.dp>>; доля ничьих <<v.ds>>; accuracy <<v.acc>> |
| Голы хозяев / гостей | оставить XGBoost | девиация <<v.gh>> и <<v.ga>> |
| Угловые | оставить XGBoost + NB | RMSE <<v.cr>>; R² <<v.c2>>; log loss линии 9.5: <<v.c95>> (база <<v.c95b>>) |
| Производные рынки | без изменений | 0 нарушений согласованности; все рынки лучше базовой частоты |
| Risk | оставить текущее правило | ROI <<v.roi>>; покрытие 100 %; с ценой <<v.pr>>; hit rate <<v.hit>>; средний коэффициент <<v.odds>> |

## Неудачные эксперименты и ограничения

- Хуже базовой модели на обоих фолдах (вероятности 1X2): веса ничьих 1.2 / 1.4 / balanced, vector scaling, сдвиг логита ничьей, только Пуассон. Температура, бинарная модель ничьей, бленды с Пуассоном и Диксоном–Коулсом — смешанно, в пределах шума.
- Ухудшились метрики: accuracy при пороговом правиле (−1.3…−3.3 п.п.), ECE ничьей при весах (до 0.065), ROI Risk на 2024 и 2025* у всех EV-вариантов, девиация раздельной модели угловых на 2024 (+0.0046), NB GLM угловых (девиация +0.0153 / +0.0063), GLM голов хозяев на 2023 (+0.0028).
- Порог правила решения подобран на тех же 2023+2024, на которых показаны его итоговые метрики; честная оценка — таблица с выбором на внутреннем сезоне (macro F1 0.443 / 0.432 / 0.434).
- Параметры продакшн-XGBoost настраивались с сезоном 2024/25 как валидационным — для фолда 2024 это одинаковый для всех вариантов оптимистичный сдвиг; новые гиперпараметры не искались (по условию). Внутренняя модель фолда 2023 обучена только на 2021/22.
- Число деревьев бинарной модели ничьей и новых моделей угловых определено ранней остановкой на том же внутреннем сезоне, на котором подбирается вес бленда (утечки в оцениваемый сезон нет, но вес может быть слегка оптимистичным).
- L2 у Poisson GLM выбран на краю сетки (α = 1.0) — более сильная регуляризация не проверялась; у NB GLM проверено одно значение L2 (statsmodels медленный).
- Реальные цены есть только для 1X2, тотала 2.5 и одной линии AH на матч (Football-Data, Bet365, предматчевые `B365*`, не closing); остальные рынки без цены, поэтому пол EV влияет только на эти три рынка. Синтетические цены не использовались.
- Различия ROI вариантов Risk на фолдах отбора (≤ 1.7 п.п.) меньше половины ширины 95 % бутстреп-ДИ (≈ ±3.5–3.8 п.п.).
- Все исследования задания выполнены; не запускались только расширенные сетки гиперпараметров (сознательно, из-за ограничения вычислений).
"""

def before_after(R: Path) -> str:
    _buf = []

    def print(*a):
        _buf.append(" ".join(str(x) for x in a))

    d, g, c, k, t = (json.loads((R / f).read_text()) for f in ("draw.json", "goals.json", "corners.json", "risk.json", "timing.json"))
    T = t["seconds"]
    F = ["2023", "2024", "2025"]
    SEL = ["2023", "2024"]


    def verdict(vals, base, lower=True):
        better = [(v < b) if lower else (v > b) for v, b in zip(vals[:2], base[:2])]
        if all(better):
            return "лучше на обоих фолдах отбора"
        if not any(better):
            return "хуже на обоих фолдах отбора"
        return "смешанно (лучше на одном фолде)"


    rows = []
    TIME = {
        "a_baseline": T["result: production XGBoost"],
        "b_w1.2": T["result: draw-weighted refit (b)"], "b_w1.4": T["result: draw-weighted refit (b)"],
        "b_balanced": T["result: draw-weighted refit (b)"],
        "c1_temperature": T["result: production XGBoost"] * 2 + T["calibration: temperature fit"],
        "c2_vector": T["result: production XGBoost"] * 2 + T["calibration: vector scaling fit"],
        "c3_draw_bias": T["result: production XGBoost"] * 2 + T["calibration: vector scaling fit"],
        "d_draw_model_blend": T["result: production XGBoost"] + 2 * T["result: binary draw model (d), one fit"],
        "e0_poisson_only": T["home_goals: production XGBoost"] + T["away_goals: production XGBoost"],
        "e_poisson_blend": 2 * (T["result: production XGBoost"] + T["home_goals: production XGBoost"] + T["away_goals: production XGBoost"]),
        "f0_dixon_coles_only": 2 * (T["home_goals: production XGBoost"] + T["away_goals: production XGBoost"]),
        "f_dixon_coles_blend": 2 * (T["result: production XGBoost"] + T["home_goals: production XGBoost"] + T["away_goals: production XGBoost"]),
        "ec2_poisson_blend_vector": 2 * (T["result: production XGBoost"] + T["home_goals: production XGBoost"] + T["away_goals: production XGBoost"]),
    }
    CPLX = {
        "a_baseline": "1 модель (продакшн)", "b_w1.2": "1 модель, вес ничьих", "b_w1.4": "1 модель, вес ничьих",
        "b_balanced": "1 модель, веса 1/частота", "c1_temperature": "+ внутренняя модель, 1 параметр",
        "c2_vector": "+ внутренняя модель, 5 параметров", "c3_draw_bias": "+ внутренняя модель, 1 параметр",
        "d_draw_model_blend": "+ бинарная XGBoost ничьей, 1 вес", "e0_poisson_only": "только модели голов",
        "e_poisson_blend": "+ внутренние модели 1X2 и голов, 1 вес", "f0_dixon_coles_only": "модели голов + rho",
        "f_dixon_coles_blend": "+ внутренние модели, rho + 1 вес", "ec2_poisson_blend_vector": "+ внутренние модели, 1 вес + 5 параметров",
    }
    print("| Задача | Вариант | Метрика | 2023/24 | 2024/25 | 2025/26* | Время обучения, с | Сложность | Итог на фолдах отбора |")
    print("|---|---|---|---|---|---|---|---|---|")
    base = [d["folds"][f]["variants"]["a_baseline"]["log_loss"] for f in F]
    for n in d["folds"]["2023"]["variants"]:
        if n.startswith("naive"):
            continue
        v = [d["folds"][f]["variants"][n]["log_loss"] for f in F]
        vd = "исходная (оставить)" if n == "a_baseline" else verdict(v, base)
        print(f"| 1X2 вероятности | {n} | log loss | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {TIME[n]:.1f} | {CPLX[n]} | {vd} |")
    rule = d["decision_rules"]["a_baseline"]
    for name, get in (("argmax (продакшн)", lambda f: d["folds"][f]["variants"]["a_baseline"]),
                      (f"{rule['final_rule']['rule']} (рекомендация; порог подобран на 2023+2024)", lambda f: rule["final_rule_metrics"][f])):
        v = [get(f) for f in F]
        print(f"| 1X2 правило решения | {name} | macro F1 / draw recall / accuracy | "
              + " | ".join(f"{x['macro_f1']:.3f} / {x['draw_recall'] * 100:.1f} % / {x['accuracy'] * 100:.1f} %" for x in v)
              + f" | 0 | без обучения | {'исходное' if 'argmax' in name else 'macro F1 и draw recall выше, accuracy ниже на 1.3–3.3 п.п.'} |")
    for task, title in (("home_goals", "Голы хозяев"), ("away_goals", "Голы гостей")):
        b = [g["folds"][f][task]["variants"]["xgb_production"]["poisson_deviance"] for f in F]
        for n, tk, cp in (("xgb_production", f"{task}: production XGBoost", "1 модель (продакшн)"),
                          ("poisson_glm", f"{task}: Poisson GLM", "линейная GLM, L2 по внутр. сезону"),
                          ("blend", None, "XGBoost + GLM, 1 вес")):
            v = [g["folds"][f][task]["variants"][n]["poisson_deviance"] for f in F]
            tm = T[tk] if tk else T[f"{task}: production XGBoost"] * 2 + T[f"{task}: Poisson GLM"] * 2
            vd = "исходная (оставить)" if n == "xgb_production" else verdict(v, b)
            if g["decision"][task]["alternatives"].get(n, {}).get("better_both") and not g["decision"][task]["alternatives"][n]["clearly_better_both"]:
                vd += ", но 95 % ДИ включает 0"
            print(f"| {title} | {n} | девиация Пуассона | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {tm:.1f} | {cp} | {vd} |")
    b = [c["folds"][f]["variants"]["xgb_production"]["mean_line_log_loss"] for f in F]
    CT = {"xgb_production": T["corners: production XGBoost"], "xgb_enriched": T["corners: XGBoost enriched"] + T["corners: extra feature computation (all seasons)"],
          "xgb_clean": None, "xgb_home_away": T["corners: home + away XGBoost (2 fits)"], "poisson_glm": T["corners: Poisson GLM"],
          "poisson_glm_enriched": None, "negbin_glm_clean": T["corners: negative-binomial GLM"],
          "blend": T["corners: production XGBoost"] + T["corners: Poisson GLM"]}
    CC = {"xgb_production": "1 модель (продакшн)", "xgb_enriched": "+ 31 скользящий признак", "xgb_clean": "84 угловых признака",
          "xgb_home_away": "2 модели + свёртка NB", "poisson_glm": "линейная GLM", "poisson_glm_enriched": "GLM + новые признаки",
          "negbin_glm_clean": "NB GLM (statsmodels)", "blend": "XGBoost + GLM(+новые), 1 вес"}
    for n in c["folds"]["2023"]["variants"]:
        if n == "naive_league_mean":
            continue
        v = [c["folds"][f]["variants"][n]["mean_line_log_loss"] for f in F]
        r = [c["folds"][f]["variants"][n]["rmse"] for f in F]
        vd = "исходная (оставить)" if n == "xgb_production" else "log loss линий: " + verdict(v, b)
        if n != "xgb_production":
            st = c["selection"]["table"][n]
            vd += "; девиация: " + ("лучше на обоих" if st["deviance_better_both"] else "не лучше на обоих")
            if st["deviance_better_both"] and not st["deviance_ci_below_zero_both"]:
                vd += " (95 % ДИ включает 0)"
        tm = "—" if CT[n] is None else f"{CT[n]:.1f}"
        print(f"| Угловые | {n} | ср. log loss линий (RMSE) | {v[0]:.4f} ({r[0]:.3f}) | {v[1]:.4f} ({r[1]:.3f}) | {v[2]:.4f} ({r[2]:.3f}) | {tm} | {CC[n]} | {vd} |")
    b = [k["folds"][f]["current (no EV rule)"]["roi"] for f in F]
    for n in k["variants"]:
        v = [k["folds"][f][n]["roi"] for f in F]
        o = [k["folds"][f][n]["avg_odds"] for f in F]
        vd = "исходное (оставить)" if n.startswith("current") else verdict(v, b, lower=False)
        print(f"| Risk | {n} | ROI (ср. кэф) | {v[0] * 100:+.1f} % ({o[0]:.2f}) | {v[1] * 100:+.1f} % ({o[1]:.2f}) | {v[2] * 100:+.1f} % ({o[2]:.2f}) | 0 | правило отбора | {vd} |")
    return "\n".join(_buf)


def tables() -> dict:
    """Parse results/summary_tables.md into {section: [tables]} and {section.sub: table}."""
    text = (RESULTS / "summary_tables.md").read_text(encoding="utf-8")
    out = {}
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        name, body = block.split("\n", 1)
        name = name.strip()
        subs = re.split(r"^### ", body, flags=re.M)
        if len(subs) > 1:
            for sub in subs[1:]:
                sname, sbody = sub.split("\n", 1)
                out[f"{name}.{sname.strip()}"] = sbody.strip()
        else:
            parts, cur = [], []
            for line in body.splitlines():
                if line.startswith("|"):
                    cur.append(line)
                elif cur:
                    parts.append("\n".join(cur))
                    cur = []
            if cur:
                parts.append("\n".join(cur))
            # keep bold captions (goals) together with their tables
            if name == "goals":
                out[name] = body.strip()
            else:
                for i, p in enumerate(parts):
                    out[f"{name}.{i}"] = p
                out[name] = "\n\n".join(parts)
    return out


def key_values() -> dict:
    d, g, c, k = (json.loads((RESULTS / f).read_text()) for f in ("draw.json", "goals.json", "corners.json", "risk.json"))
    F = ["2023", "2024", "2025"]
    b = [d["folds"][f]["variants"]["a_baseline"] for f in F]
    r = [d["decision_rules"]["a_baseline"]["final_rule_metrics"][f] for f in F]

    def arrow(key, pct=True, digits=1):
        fmt = (lambda x: f"{x * 100:.{digits}f} %") if pct else (lambda x: f"{x:.3f}")
        return " / ".join(f"{fmt(x[key])} → {fmt(y[key])}" for x, y in zip(b, r))

    def seq(vals, fmt):
        return " / ".join(fmt(v) for v in vals)

    cur = [k["folds"][f]["current (no EV rule)"] for f in F]
    cp = [c["folds"][f]["variants"]["xgb_production"] for f in F]
    return {
        "v.ll": seq([x["log_loss"] for x in b], lambda v: f"{v:.4f}"),
        "v.f1": arrow("macro_f1", pct=False), "v.dr": arrow("draw_recall"), "v.dp": arrow("draw_precision"),
        "v.ds": arrow("predicted_draw_share"), "v.acc": arrow("accuracy"),
        "v.gh": seq([g["folds"][f]["home_goals"]["variants"]["xgb_production"]["poisson_deviance"] for f in F], lambda v: f"{v:.4f}"),
        "v.ga": seq([g["folds"][f]["away_goals"]["variants"]["xgb_production"]["poisson_deviance"] for f in F], lambda v: f"{v:.4f}"),
        "v.cr": seq([x["rmse"] for x in cp], lambda v: f"{v:.3f}"),
        "v.c2": seq([x["r2"] for x in cp], lambda v: f"{v:.3f}"),
        "v.c95": seq([x["lines"]["9.5"]["log_loss"] for x in cp], lambda v: f"{v:.4f}"),
        "v.c95b": seq([x["lines"]["9.5"]["baseline_log_loss"] for x in cp], lambda v: f"{v:.4f}"),
        "v.roi": seq([x["roi"] for x in cur], lambda v: f"{v * 100:+.1f} %"),
        "v.pr": seq([x["priced_share"] for x in cur], lambda v: f"{v * 100:.1f} %"),
        "v.hit": seq([x["hit_rate"] for x in cur], lambda v: f"{v * 100:.1f} %"),
        "v.odds": seq([x["avg_odds"] for x in cur], lambda v: f"{v:.2f}"),
    }


def build() -> Path:
    t = tables()
    keep = ("a_baseline", "b_w1.4", "b_balanced", "c2_vector", "d_draw_model_blend", "f_dixon_coles_blend",
            "naive_constant")
    cls = t["draw.classes"].splitlines()
    t["draw.classes_filtered"] = "\n".join(cls[:2] + [l for l in cls[2:] if l.split("|")[1].strip() in keep])
    t["before_after"] = before_after(RESULTS)
    t |= key_values()
    text = TEMPLATE.strip() + "\n"
    for key in re.findall(r"<<([^>]+)>>", text):
        text = text.replace(f"<<{key}>>", t[key])
    path = HERE / "findings.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build())
