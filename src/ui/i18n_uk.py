"""Ukrainian translations of the interface (English text -> Ukrainian)."""
from __future__ import annotations

UK: dict[str, str] = {
    # navigation / pages
    "Matches": "Матчі", "Match Details": "Деталі матчу", "Prediction History": "Історія прогнозів",
    "Leagues": "Ліги", "Teams": "Команди", "Analytics": "Аналітика", "Model Analysis": "Аналіз моделей",
    "Dark / light theme": "Темна / світла тема", "Data updated": "Дані оновлено",
    # generic
    "Home": "Господарі", "Draw": "Нічия", "Away": "Гості", "Today": "Сьогодні", "Date": "Дата",
    "Previous day": "Попередній день", "Next day": "Наступний день", "All": "Усі", "League": "Ліга",
    "All leagues": "Усі ліги", "Market": "Ринок", "All markets": "Усі ринки", "Season": "Сезон",
    "Period": "Період", "Group by": "Групувати за", "Status": "Статус", "Score": "Рахунок", "Result": "Результат",
    "Type": "Тип", "Team": "Команда", "Match": "Матч", "Page": "Сторінка", "Total": "Разом", "Show": "Показати",
    "FT": "Завершено", "Full time": "Матч завершено", "Upcoming": "Попереду", "Won": "Виграш", "Lost": "Програш",
    "Void": "Повернення", "Won and lost": "Виграші та програші",
    "Mon": "Пн", "Tue": "Вт", "Wed": "Ср", "Thu": "Чт", "Fri": "Пт", "Sat": "Сб", "Sun": "Нд",
    "England": "Англія", "Scotland": "Шотландія", "Spain": "Іспанія", "Italy": "Італія", "Germany": "Німеччина",
    "France": "Франція", "Netherlands": "Нідерланди", "Belgium": "Бельгія", "Portugal": "Португалія",
    "Turkey": "Туреччина", "Greece": "Греція",
    "W": "В", "D": "Н", "L": "П", "H": "Д", "A": "Г", "P": "І", "GF": "ЗМ", "GA": "ПМ", "GD": "РМ", "Pts": "О",
    "W-D-L": "В-Н-П",
    # matches page
    "Search team": "Пошук команди", "Search team...": "Пошук команди...", "No team found.": "Команду не знайдено.",
    "No matches available for this date.": "На цю дату матчів немає.",
    "Pick another date or league.": "Оберіть іншу дату або лігу.",
    "{n} matches": "Матчів: {n}", "{a} finished · {b} upcoming": "{a} завершено · {b} попереду",
    "No prediction": "Немає прогнозу", "Over 2.5": "Більше 2.5", "Under 2.5": "Менше 2.5",
    # markets / selections
    "Match result": "Результат матчу", "Double chance": "Подвійний шанс", "Total goals": "Тотал голів",
    "Both teams to score": "Обидві заб'ють", "Exact score": "Точний рахунок", "Handicap": "Фора",
    "Total corners": "Тотал кутових", "{team} win": "Перемога {team}", "{team} or draw": "{team} або нічия",
    "{a} or {b}": "{a} або {b}", "Over {line} goals": "Більше {line} голів", "Under {line} goals": "Менше {line} голів",
    "Both teams to score: Yes": "Обидві заб'ють: так", "Both teams to score: No": "Обидві заб'ють: ні",
    "Exact score {score}": "Точний рахунок {score}", "Over {line} corners": "Більше {line} кутових",
    "Under {line} corners": "Менше {line} кутових", "BTTS": "Обидві заб'ють",
    "Selection": "Варіант", "Probability": "Ймовірність", "Odds": "Коефіцієнт", "Expected Value": "Очікувана цінність",
    "Expected Value estimates the model's theoretical advantage at the current odds: EV = model probability × odds − 1. "
    "It is not a guaranteed profit.": "Очікувана цінність (EV) оцінює теоретичну перевагу моделі за поточним "
    "коефіцієнтом: EV = ймовірність моделі × коефіцієнт − 1. Це не гарантований прибуток.",
    "Not available for this match.": "Для цього матчу недоступно.",
    # match details
    "Back to matches": "До матчів", "Match not found.": "Матч не знайдено.",
    "There is no prediction for this match.": "Для цього матчу немає прогнозу.",
    "No predictions yet.": "Прогнозів ще немає.", "Search a match…": "Знайдіть матч…",
    "Main prediction": "Основний прогноз", "Risk prediction": "Ризиковий прогноз", "Forecast": "Прогноз ринку",
    "Prediction results": "Результати прогнозів", "Prediction": "Прогноз", "Profit": "Прибуток",
    "Higher odds, higher risk: most such predictions lose, a few pay well.":
        "Вищий коефіцієнт — вищий ризик: більшість таких прогнозів програє, деякі добре окуповуються.",
    "Why this prediction?": "Чому такий прогноз?", "Explain risk prediction": "Пояснити ризиковий прогноз",
    "No explanation is available for this prediction.": "Для цього прогнозу немає пояснення.",
    "Advanced: explain another market · technical SHAP view": "Додатково: пояснення іншого ринку · технічний вигляд SHAP",
    "SHAP values are in the model's internal scale (log-odds or log of the expected count). Green pushes towards "
    "the selection, red against it.": "Значення SHAP подано у внутрішній шкалі моделі (логарифм шансів або логарифм "
    "очікуваної кількості). Зелене підсилює варіант, червоне — послаблює.",
    "Markets": "Ринки", "Expected goals": "Очікувані голи", "Corner line": "Лінія кутових",
    "Expected total corners: **{n}**": "Очікувана кількість кутових: **{n}**",
    "European handicap": "Європейська фора", "Handicap (half lines)": "Фора (половинні лінії)",
    "European handicap: the handicap is added to the home team's goals, then the result (home / draw / away) "
    "is decided.": "Європейська фора: фору додають до голів господарів, після чого визначають результат "
    "(перемога господарів / нічия / перемога гостей).",
    "Half-line handicap: the handicap is added to the home team's goals; a draw is impossible, so there are only "
    "two outcomes.": "Фора з половинною лінією: фору додають до голів господарів; нічия неможлива, тому є лише два "
    "результати.",
    "League position": "Місце в таблиці", "Points": "Очки", "Points / game": "Очки за матч",
    "Points / game (last 5)": "Очки за матч (останні 5)", "Goals / game": "Голи за матч",
    "Conceded / game": "Пропущено за матч", "Shots / game": "Удари за матч",
    "Shots on target / game": "Удари в площину за матч", "Corners / game": "Кутові за матч",
    "Yellow cards / game": "Жовті картки за матч",
    "No season statistics yet for these teams.": "Для цих команд ще немає статистики сезону.",
    "Last 5 matches": "Останні 5 матчів", "Before kick-off": "Перед матчем", "No matches.": "Матчів немає.",
    "Goals": "Голи", "Conceded": "Пропущено", "Shots": "Удари", "On target": "У площину", "Corners": "Кутові",
    "Last matches": "Останні матчі", "Last {n}": "Останні {n}", "Scored": "Забито", "at home": "удома",
    "away": "на виїзді", "this season": "цей сезон",
    "These teams have not met in the available data.": "У доступних даних ці команди не зустрічалися.",
    "Meetings": "Зустрічі", "{team} wins": "Перемоги {team}", "Draws": "Нічиї", "Goals / match": "Голи за матч",
    "{team} rest": "Відпочинок {team}", "{n} days": "{n} дн.", "Rest difference": "Різниця у відпочинку",
    "Positive = the home team had more rest.": "Додатне значення — господарі відпочивали довше.",
    "Teams before the match": "Команди перед матчем", "Team comparison": "Порівняння команд",
    "Recent form": "Поточна форма", "Home / Away": "Удома / на виїзді", "Head-to-head": "Особисті зустрічі",
    "Rest days": "Дні відпочинку",
    "This page could not be displayed right now. Please try again later.":
        "Не вдалося показати сторінку. Спробуйте пізніше.",
    # history
    "Prediction results and performance history.": "Результати прогнозів та історія ефективності.",
    "No completed predictions yet.": "Завершених прогнозів ще немає.",
    "Last 7 days": "Останні 7 днів", "Last 30 days": "Останні 30 днів", "This month": "Цей місяць",
    "This season": "Цей сезон", "This year": "Цей рік", "All time": "Увесь час", "Custom": "Власний період",
    "Date range": "Діапазон дат", "Prediction type": "Тип прогнозу", "Main predictions": "Основні прогнози",
    "Risk predictions": "Ризикові прогнози", "Market forecasts": "Прогнози ринків", "Show upcoming": "Показати майбутні",
    "Advanced filters": "Додаткові фільтри", "Prediction origin": "Походження прогнозу",
    "Live (before kick-off)": "Наживо (до початку матчу)", "Backfill (current season)": "Дозаповнення (поточний сезон)",
    "Backtest (test season)": "Бектест (тестовий сезон)", "Only predictions with odds": "Лише прогнози з коефіцієнтами",
    "Predictions": "Прогнози", "{a} won · {b} lost": "{a} виграно · {b} програно", "Hit rate": "Влучність",
    "{n} predictions with odds": "{n} прогнозів із коефіцієнтами", "ROI": "ROI",
    "profit / amount staked": "прибуток / сума ставок", "Average odds": "Середній коефіцієнт",
    "Profit and ROI are calculated with a fixed stake of 1 unit per prediction, only for predictions with real "
    "bookmaker odds.": "Прибуток і ROI розраховано з фіксованою ставкою 1 одиниця на прогноз і лише для прогнозів "
    "із реальними букмекерськими коефіцієнтами.",
    "No completed predictions for this period.": "За цей період немає завершених прогнозів.",
    "Daily": "Щодня", "Weekly": "Щотижня", "Monthly": "Щомісяця", "Yearly": "Щороку",
    "Summary by period": "Підсумки за періодами", "Main": "Основний", "Risk": "Ризиковий",
    "Page {p} of {n} · {k} predictions": "Сторінка {p} з {n} · прогнозів: {k}",
    # analytics
    "Prediction set": "Набір прогнозів", "All market forecasts": "Усі прогнози ринків",
    "Match result (1X2)": "Результат матчу (1X2)", "FootPredict performance": "Ефективність FootPredict",
    "Main hit rate": "Влучність основних", "Main profit": "Прибуток основних", "Main ROI": "ROI основних",
    "Fixed stake of 1 unit per prediction; profit and ROI use real bookmaker odds only. One main prediction per match.":
        "Фіксована ставка 1 одиниця на прогноз; прибуток і ROI — лише за реальними коефіцієнтами. Один основний "
        "прогноз на матч.",
    "Fixed stake of 1 unit per prediction; profit and ROI use real bookmaker odds only.":
        "Фіксована ставка 1 одиниця на прогноз; прибуток і ROI — лише за реальними коефіцієнтами.",
    "Best market": "Найкращий ринок", "Weakest market": "Найслабший ринок",
    "Not enough data (at least {n} predictions with odds).": "Недостатньо даних (потрібно щонайменше {n} прогнозів "
                                                              "із коефіцієнтами).",
    "Performance by league": "Ефективність за лігами", "Performance by market": "Ефективність за ринками",
    "By period": "За періодами", "Cumulative profit": "Накопичений прибуток", "Advanced analytics": "Розширена аналітика",
    "No predictions with odds in this selection.": "У цій вибірці немає прогнозів із коефіцієнтами.",
    "Cumulative profit (units)": "Накопичений прибуток (одиниці)",
    "Rolling ROI, last 200 predictions (%)": "Ковзний ROI, останні 200 прогнозів (%)",
    "Rolling hit rate, last 200 predictions": "Ковзна влучність, останні 200 прогнозів",
    "Main prediction volume over time": "Кількість основних прогнозів у часі",
    "Prediction volume over time": "Кількість прогнозів у часі", "ROI by league (%)": "ROI за лігами (%)",
    "ROI by market (%)": "ROI за ринками (%)",
    # leagues
    "No data available.": "Дані відсутні.", "Matches played": "Зіграно матчів", "Home · Draw · Away": "П1 · Н · П2",
    "Over 2.5 · BTTS": "Більше 2.5 · Обидві заб'ють", "Corners / match": "Кутові за матч",
    "League table": "Турнірна таблиця", "No matches played yet this season.": "У цьому сезоні ще не зіграно матчів.",
    "Upcoming matches": "Найближчі матчі", "No upcoming fixtures published yet.": "Розклад найближчих матчів ще не "
                                                                                   "опубліковано.",
    "Recent results": "Останні результати", "No results yet.": "Результатів ще немає.",
    "FootPredict in this league": "FootPredict у цій лізі",
    "No completed predictions for this league yet.": "Для цієї ліги ще немає завершених прогнозів.",
    "Matches predicted": "Спрогнозовано матчів", "Market performance": "Ефективність ринків",
    # teams
    "Last 5": "Останні 5", "Last 10": "Останні 10", "Last 15": "Останні 15", "Current season": "Поточний сезон",
    "All available": "Усі доступні", "Matches": "Матчі", "Corners against": "Кутові суперника",
    "Clean sheets": "Сухі матчі", "Not enough matches for trend charts.": "Замало матчів для графіків тенденцій.",
    "Goals per game": "Голи за матч", "Corners per game": "Кутові за матч", "Points per game": "Очки за матч",
    "{n}-match rolling averages.": "Ковзні середні за {n} матчі.",
    "Type a team name, e.g. Arsenal": "Введіть назву команди, напр. Arsenal", "Choose a team": "Оберіть команду",
    "Start typing a club name in the search box above.": "Почніть вводити назву клубу в полі пошуку вище.",
    "of {n}": "з {n}", "{n} played": "зіграно {n}", "Form": "Форма", "Trends": "Тенденції",
    "Showing the latest {a} of {b} matches.": "Показано останні {a} з {b} матчів.",
    # model analysis
    "Accuracy": "Точність", "Macro F1": "Macro F1", "Balanced accuracy": "Збалансована точність",
    "Log loss": "Log loss", "Brier score": "Оцінка Браєра", "RMSE": "RMSE", "MAE": "MAE", "R²": "R²",
    "Poisson deviance": "Пуассонівське відхилення",
    "Share of matches where the predicted outcome was the actual one. Higher is better.":
        "Частка матчів, у яких прогнозований результат збігся з фактичним. Більше — краще.",
    "Average quality over the three outcomes (home, draw, away); draws count as much as home wins. Higher is better.":
        "Середня якість за трьома результатами (господарі, нічия, гості); нічиї важать стільки ж, скільки перемоги "
        "господарів. Більше — краще.",
    "Average recall per outcome: how often each actual outcome was recognised. Higher is better.":
        "Середня повнота за результатами: як часто розпізнано кожен фактичний результат. Більше — краще.",
    "Measures the quality of predicted probabilities; punishes confident wrong forecasts. Lower is better.":
        "Вимірює якість прогнозованих ймовірностей; карає впевнені хибні прогнози. Менше — краще.",
    "Measures probability forecast error (mean squared error of the probabilities). Lower is better.":
        "Вимірює похибку ймовірнісного прогнозу (середня квадратична похибка ймовірностей). Менше — краще.",
    "Typical prediction error, with larger mistakes penalised more heavily. Lower is better.":
        "Типова похибка прогнозу, великі помилки штрафуються сильніше. Менше — краще.",
    "Average absolute prediction error (in goals or corners). Lower is better.":
        "Середня абсолютна похибка прогнозу (у голах або кутових). Менше — краще.",
    "Shows how much of the variation in the target is explained by the model. Higher is better.":
        "Показує, яку частку мінливості цільової змінної пояснює модель. Більше — краще.",
    "Proper error measure for count forecasts (goals, corners). Lower is better.":
        "Коректна міра похибки для прогнозів кількості (голи, кутові). Менше — краще.",
    "Season statistics": "Статистика сезону", "Corner statistics": "Статистика кутових",
    "Discipline (cards, fouls)": "Дисципліна (картки, фоли)", "Home / away": "Удома / на виїзді",
    "Recent form 5 / 10 / 15": "Поточна форма 5 / 10 / 15", "Attack / defence": "Атака / оборона",
    "Event frequencies": "Частоти подій", "Strength differences": "Різниця в силі",
    "League context": "Контекст ліги", "Attacking pressure": "Атакувальний тиск",
    "Bookmaker probabilities": "Ймовірності букмекерів", "Result model (1X2)": "Модель результату (1X2)",
    "Home goals model": "Модель голів господарів", "Away goals model": "Модель голів гостей",
    "Corners model": "Модель кутових", "Active production model": "Робоча модель",
    "Trained through": "Навчена до", "Used for the current season": "Використовується в поточному сезоні",
    "Evaluation model": "Оцінювальна модель", "Test season": "Тестовий сезон", "Trained on": "Навчена на",
    "All test-season metrics come from the evaluation model, which never saw the test season; the production model "
    "is then refitted on all completed seasons with the same settings.":
        "Усі метрики тестового сезону отримано оцінювальною моделлю, яка не бачила тестового сезону; робочу модель "
        "потім перенавчено на всіх завершених сезонах з тими самими налаштуваннями.",
    "Models and data periods": "Моделі та періоди даних", "Model": "Модель", "Target": "Цільова змінна",
    "Training": "Навчання", "Validation": "Валідація", "Test": "Тест", "Features": "Ознаки",
    "What do these metrics mean?": "Що означають ці метрики?", "Derived markets": "Похідні ринки",
    "Test-season metrics": "Метрики тестового сезону", "Draw predictions": "Прогнози нічиїх", "Outcome": "Результат",
    "Actual share": "Фактична частка", "Predicted share": "Прогнозована частка", "Recall": "Повнота",
    "Precision": "Точність класу", "Average probability": "Середня ймовірність",
    "A draw is rarely the single most likely outcome, so a plain arg-max almost never predicts one. FootPredict keeps "
    "the probabilities and names an outcome with argmax(P(H), m·P(D), P(A)); the multiplier m is chosen on the "
    "validation season only.": "Нічия рідко буває найімовірнішим результатом, тому звичайний arg-max майже ніколи "
    "її не обирає. FootPredict зберігає ймовірності, а результат називає за argmax(P(H), m·P(D), P(A)); множник m "
    "підібрано лише на валідаційному сезоні.",
    "Evaluation": "Оцінювання", "Validation · arg-max": "Валідація · arg-max",
    "Validation · draw-aware rule": "Валідація · правило з нічиїми", "Test · arg-max": "Тест · arg-max",
    "Test · draw-aware rule": "Тест · правило з нічиїми", "Draw recall": "Повнота нічиїх",
    "Draw precision": "Точність нічиїх", "Predicted draw share": "Прогнозована частка нічиїх",
    "Actual draw share": "Фактична частка нічиїх", "FootPredict result model": "Модель результату FootPredict",
    "Naive baseline (always the historical outcome frequencies)": "Наївна базова модель (завжди історичні частоти "
                                                                   "результатів)",
    "Bookmaker probabilities (reference)": "Ймовірності букмекерів (для порівняння)",
    "Confusion matrix": "Матриця помилок", "test season, {n} matches": "тестовий сезон, {n} матчів",
    "Actual": "Факт", "Predicted": "Прогноз", "Model vs naive baseline": "Модель проти наївної базової",
    "The naive baseline does not analyse the specific match. The bookmaker row is only a reference — odds are not "
    "model inputs.": "Наївна базова модель не аналізує конкретний матч. Рядок букмекерів наведено лише для "
    "порівняння — коефіцієнти не є входами моделі.",
    "Calibration": "Калібрування", "predicted probability vs observed frequency": "прогнозована ймовірність проти "
                                                                                    "спостережуваної частоти",
    "Perfect": "Ідеально", "Predicted probability": "Прогнозована ймовірність", "Observed frequency": "Спостережувана частота",
    "baseline {v}": "базова {v}", "Mean predicted / actual": "Середнє: прогноз / факт",
    "Baseline = always predicting the training average. Football is very random, so even good models explain only "
    "a small part of the variance (low R²); what matters is that the expected values and resulting probabilities "
    "are well calibrated.": "Базова модель завжди прогнозує середнє навчальних даних. Футбол дуже випадковий, тож "
    "навіть добрі моделі пояснюють лише малу частку мінливості (низький R²); важливо, щоб очікувані значення та "
    "отримані ймовірності були добре відкалібровані.",
    "{n} features": "ознак: {n}", "Show technical features": "Показати технічні ознаки", "Top {n}": "Топ {n}",
    "Global SHAP (mean |SHAP|, test season)": "Глобальний SHAP (середнє |SHAP|, тестовий сезон)",
    "Feature importance (XGBoost gain)": "Важливість ознак (XGBoost gain)",
    "Global importance shows what matters on average across many matches; the explanation on a match page "
    "describes one particular prediction.": "Глобальна важливість показує, що впливає в середньому на багато матчів; "
    "пояснення на сторінці матчу описує один конкретний прогноз.",
    "Advanced: hyperparameter search": "Додатково: пошук гіперпараметрів", "Trial": "Спроба",
    "Best so far": "Найкраще досі", "validation": "валідація", "Base rate": "Базова частота",
    "Totals, BTTS and exact scores come from one score matrix built from the two goal models (Poisson) and aligned "
    "with the 1X2 probabilities, so all markets agree with each other.": "Тотали, «обидві заб'ють» і точні рахунки "
    "отримано з однієї матриці рахунків, побудованої з двох моделей голів (Пуассон) і узгодженої з ймовірностями "
    "1X2, тому всі ринки узгоджені між собою.",
    "No trained models available.": "Навчених моделей немає.", "Features and importance": "Ознаки та важливість",
    "Markets derived from the goal and corner models": "Ринки, похідні від моделей голів і кутових",
    "test season": "тестовий сезон",
    "Model settings: {a} Hyperopt trials for the result model, {b} for each regression model.":
        "Налаштування: {a} спроб Hyperopt для моделі результату, {b} — для кожної регресійної моделі.",
}
