"""Stage A — the player-data sources that were checked on 2026-11-18 and what was found (observations of that day,
recorded here so the report states them; the downloaded robots.txt files are kept in ``cache/robots/``).

Rules followed: no login, no paid service, robots.txt and terms respected, no attempt to pass a bot challenge, at
least 3–4 s between requests, the API-Football key is not used (the working key lives only in GitHub Secrets; the
local key belongs to a suspended account) and no GitHub workflow is triggered.
"""
from __future__ import annotations

SOURCES = [
    {"source": "API-Football (подключён)", "access": "ключ (бесплатный план, 100 запросов/сутки)",
     "data": "составы (/fixtures/lineups), статистика игроков в матче с рейтингом (/fixtures/players), "
             "/players (сезонная статистика), /injuries, /sidelined; /fixtures?ids= (до 20 матчей) отдаёт "
             "события, составы и статистику игроков сразу",
     "history": "нет: бесплатный план отдаёт матчи только со вчера по завтра (проверено 04.11.2026, "
                "config.API_FOOTBALL_PLAN_DAYS_BACK); в local_data/api/api_football — только teams и fixtures "
                "(125 ответов), ни одного ответа с составами",
     "verdict": "только сбор вперёд (пилот ниже, нужно согласие владельца); история 2021–2026 недоступна"},
    {"source": "Understat", "access": "открытые страницы, robots.txt отсутствует",
     "data": "составы, минуты, позиции, xG/xA игроков для E0, SP1, I1, D1, F1 (+ РПЛ) с 2014/15",
     "history": "18.11.2026 сайт отвечает страницей обслуживания «We will come back soon» "
                "(и с этой машины, и из внешней сети)",
     "verdict": "недоступен в день исследования; первый кандидат для 2026/27 топ-5, когда вернётся"},
    {"source": "FBref", "access": "Cloudflare-проверка (403, «Just a moment…»)",
     "data": "составы и статистика игроков; с 2026 г. без расширенных данных Opta",
     "history": "автоматический доступ закрыт проверкой браузера",
     "verdict": "исключён: обход защиты недопустим"},
    {"source": "Открытый датасет Transfermarkt (dcaribou/transfermarkt-datasets)",
     "access": "публичный R2-бакет, без входа, лицензия датасета CC0",
     "data": "все матчи клубов (лига, кубки, еврокубки), стартовые составы и запасные с позициями, минуты, голы, "
             "передачи, карточки, датированная рыночная стоимость игроков, трансферы",
     "history": "11 из 21 лиги FootPredict (E0 SP1 I1 D1 F1 N1 P1 B1 T1 G1 SC0) с 2012/13; обновления "
                "остановлены: матчи до 06.07.2026, стоимость до 12.06.2026 — сезона 2026/27 нет",
     "verdict": "использован для исследования (история 2021/22–2025/26 и состав на 1 июля 2026); "
                "ограничение: сайт Transfermarkt запрещает машинное использование своего контента, включая обучение моделей (license.xml, "
                "RSL), сам сайт не опрашивался; для продакшна нужен юридически чистый поддерживаемый "
                "источник"},
    {"source": "ESPN (site.api.espn.com)", "access": "недокументированный API; robots.txt API — 403 Access Denied",
     "data": "составы и простая статистика игроков почти для всех лиг",
     "history": "условия Disney запрещают автоматический сбор; robots.txt www.espn.com закрывает boxscore",
     "verdict": "исключён"},
    {"source": "FotMob", "access": "robots.txt: Disallow /api/* для всех роботов, кроме поисковиков",
     "data": "составы, рейтинги игроков", "history": "—", "verdict": "исключён (robots.txt)"},
    {"source": "SofaScore / WhoScored / SoFIFA", "access": "403 / Cloudflare-проверка",
     "data": "рейтинги игроков, составы; SoFIFA — датированные рейтинги EA FC",
     "history": "автоматический доступ закрыт", "verdict": "исключены"},
    {"source": "worldfootball.net", "access": "robots.txt: Content-Signal запрещает обучение моделей",
     "data": "составы и замены многих лиг, включая низшие", "history": "—",
     "verdict": "исключён: сайт запрещает использование для обучения моделей"},
    {"source": "football-data.org", "access": "нужен ключ (регистрация); составы — платный пакет",
     "data": "—", "history": "—", "verdict": "исключён (вход и платный пакет)"},
]

API_FOOTBALL_PILOT = {
    "goal": "собирать составы и статистику игроков сыгранных матчей всех 21 лиги вперёд, чтобы через сезон "
            "получить динамические признаки состава для 2027/28",
    "endpoint": "GET /fixtures?ids=<до 20 id через дефис> (события, составы, статистика игроков с рейтингом в "
                "одном ответе); id матчей уже известны из ежедневного /fixtures?date=",
    "when": "ежедневный запуск конвейера, для вчерашних завершённых матчей (в пределах окна бесплатного плана)",
    "leagues": "все 21 лига FootPredict (идентификаторы API-Football уже есть в src/leagues.py)",
    "cost_note": "число запросов в сутки = ceil(матчей вчера / 20); расчёт по календарю 2025/26 — в results",
    "fallback_if_ids_has_no_players": "/fixtures/lineups и /fixtures/players по одному запросу на матч — "
                                      "только для 5–11 топ-лиг, иначе лимит 100/сутки превышается в пиковые дни",
    "first_step": "один пробный запрос /fixtures?ids= на 2–3 вчерашних матча через ключ из GitHub Secrets "
                  "(ручной запуск workflow) — только после явного согласия владельца",
    "approval": "требуется согласие владельца: запуск через CI-ключ и расход дневной квоты",
}
