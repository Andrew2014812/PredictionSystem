"""Registry of the experimental (shadow) models: which frozen hypothesis each one implements.

Every shadow names the preregistration file it implements, the SHA-256 of that file and the git commit that
froze it. Before a shadow records anything its preregistration is read and its hash checked: a changed file
means the hypothesis is no longer the frozen one, and the shadow refuses to record (``ShadowSpec.params``
raises). The parameters a shadow uses are taken from that file only.

A new hypothesis gets a new entry (or a new ``version`` of an existing id); an entry is never edited after
its first recorded prediction — ``shadow.py`` keeps the first registration of every (id, version) in the
manifest and refuses an entry whose content changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config

ROOT = config.ROOT_DIR


@dataclass(frozen=True)
class ShadowSpec:
    id: str
    version: int
    created: str                    # date the shadow entered the registry
    prereg: str                     # preregistration file (relative to the repository root)
    prereg_sha256: str              # hash of that file at the freeze
    freeze_sha: str                 # git commit that froze the preregistration
    frozen_at_utc: str              # time of that commit (UTC)
    hypothesis: str                 # which hypothesis of the file
    items: tuple                    # what is recorded per match: main | risk | 1x2 | ou25 | goals
    description: str
    baseline: str
    metric: str
    criterion: str
    min_n: int                      # no conclusion before this many settled forward cases
    forward_rule: dict              # which matches count as forward (kick-off / date after the freeze)
    family: str                     # Holm correction runs over the shadows of one family
    implemented: bool = True
    not_implemented_reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"

    def prereg_path(self, root: Path | None = None) -> Path:
        return Path(root or ROOT) / self.prereg

    def check(self, root: Path | None = None) -> str | None:
        """None when the preregistration file is unchanged, otherwise the reason it cannot be used."""
        path = self.prereg_path(root)
        if not path.exists():
            return f"preregistration {self.prereg} missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.prereg_sha256:
            return f"preregistration {self.prereg} changed (sha256 {digest[:12]} != {self.prereg_sha256[:12]})"
        return None

    def document(self, root: Path | None = None) -> dict:
        problem = self.check(root)
        if problem:
            raise ValueError(problem)
        return json.loads(self.prereg_path(root).read_text(encoding="utf-8"))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["items"] = list(self.items)
        d["key"] = self.key
        return d


MAINCAL = "experiments/v6_2/main_calibration/results/preregistration.json"
RISK = "experiments/v6_2/risk/results/preregistration.json"
BOOKMAKER = "experiments/v6_2/bookmaker/results/preregistration.json"
XG = "experiments/v6_2/xg/results/preregistration.json"
UNCERTAINTY = "experiments/v6_2/uncertainty/results/preregistration.json"

SHA = {MAINCAL: "d6969f7aa0801e32256cc863b9e3546c72ae84dd01b7b60e715ee3e73957cfe2",
       RISK: "db84ccc1ce00273d708ff0b56f5375a6181e3095b9587b67928601d9bdaed9c3",
       BOOKMAKER: "9f07bd3ecab8399d560efa6108eb5b170896bd92f068fa69efc4c8ecb30da865",
       XG: "948c5faf795abb6536cb4108aaad10017aa385dcb082e5158edf4b1cb57647e2",
       UNCERTAINTY: "049092c556ff2813e931ee49de201b72e2f77d077ce8f1412e50924a912483cd"}
COMMIT = {MAINCAL: ("24039db5ed13051dc8a9a7bfa0a04ac61667ed17", "2026-11-09T17:05:59Z"),
          RISK: ("8525b600367c109d2ffa7c19a2ac74fc41fac838", "2026-11-09T17:06:18Z"),
          BOOKMAKER: ("7f5d360a8fcd9f1d5546d13812640e60bb4a3485", "2026-11-09T17:08:44Z"),
          XG: ("7b2a73435bd0a21273a6b7aa6b6d4d59bd675cd8", "2026-11-09T17:03:59Z"),
          UNCERTAINTY: ("8b089170823efba124d35f11eb9af40d58c5613d", "2026-11-09T18:29:30Z")}


def _spec(prereg: str, **kw) -> ShadowSpec:
    sha, at = COMMIT[prereg]
    return ShadowSpec(prereg=prereg, prereg_sha256=SHA[prereg], freeze_sha=sha, frozen_at_utc=at,
                      created="2026-11-09", **kw)


SHADOWS: tuple[ShadowSpec, ...] = (
    _spec(MAINCAL, id="maincal_ps_platt_monthly", version=1, hypothesis="H1_primary", items=("main",),
          description="PS-Platt по месяцам: показанная вероятность Main = sigmoid(a_m + b_m logit pc), (a_m, b_m) — "
                      "логистическая регрессия на всех рассчитанных показанных Main 2026/27 до 1-го числа месяца матча; "
                      "выбор Main не меняется",
          baseline="вероятность показанного Main продакшна (pc)",
          metric="gap = mean(p') - доля выигрышей; парный |gap(p')| - |gap(pc)|",
          criterion="C1: 95 % ДИ gap(p') содержит 0; C2: парный ДИ |gap(p')| - |gap(pc)| < 0; C3: Холм по H1, H2",
          min_n=600, forward_rule={"kind": "date_from", "date": "2026-11-10"}, family="main_calibration"),
    _spec(MAINCAL, id="maincal_ps_group", version=1, hypothesis="H2_challenger", items=("main",),
          description="PS-Group: показанная вероятность Main = sigmoid(a_g + b_g logit pc) с замороженными картами групп "
                      "рынков (история 2022/23–2025/26); выбор Main не меняется",
          baseline="вероятность показанного Main продакшна (pc)",
          metric="gap = mean(p') - доля выигрышей; парный |gap(p')| - |gap(pc)|",
          criterion="C1: 95 % ДИ gap(p') содержит 0; C2: парный ДИ |gap(p')| - |gap(pc)| < 0; C3: Холм по H1, H2",
          min_n=600, forward_rule={"kind": "date_from", "date": "2026-11-10"}, family="main_calibration"),
    _spec(RISK, id="risk_stacked", version=1, hypothesis="H1_rule", items=("risk",),
          description="Risk со стеком модели и рынка: среди кандидатов пула с реальной ценой и полной книгой, у которых "
                      "EV по p_s ≥ 0, — лучший по формуле продакшна с p_s; иначе Risk продакшна",
          baseline="Risk продакшна",
          metric="ROI(H1) - ROI(Risk продакшна) на матчах, где у обоих выборов есть цена",
          criterion="парный 95 % ДИ разности ROI > 0 (бутстреп по матчам, 2000, seed 20261109)",
          min_n=2000, forward_rule={"kind": "kickoff_after_utc", "utc": "2026-11-09T17:06:18Z"}, family="risk"),
    _spec(RISK, id="risk_market_probability", version=1, hypothesis="H2_display", items=("risk",),
          description="Показанная вероятность Risk = вероятность рынка без маржи (пропорционально, полная книга); "
                      "выбор Risk не меняется",
          baseline="вероятность модели для показанного Risk (pc)",
          metric="gap(рынок) и парный |gap(рынок)| - |gap(модель)|",
          criterion="95 % ДИ gap(рынок) содержит 0 и парный ДИ |gap(рынок)| - |gap(модель)| < 0",
          min_n=1000, forward_rule={"kind": "kickoff_after_utc", "utc": "2026-11-09T17:06:18Z"}, family="risk"),
    _spec(BOOKMAKER, id="bookmaker_h2_pool", version=1, hypothesis="frozen_variant", items=("1x2", "ou25"),
          description="Лог-линейный пул модели и цены Bet365 открытия: P(k) = softmax(a_k + b1 log q_k + b2 log p_k) "
                      "с замороженными θ; p — вероятность модели, записанная до начала матча",
          baseline="H0: Bet365 открытия без маржи (степенной метод)",
          metric="log loss H2 - log loss H0 по рынку (1X2, тотал 2.5), парно по матчам",
          criterion="верхняя граница 95 % парного ДИ < 0 для тотала 2.5; 1X2 — вторичный, Холм по двум рынкам",
          min_n=1500, forward_rule={"kind": "kickoff_after_utc", "utc": "2026-11-09T17:08:44Z"}, family="bookmaker"),
    _spec(XG, id="xg_x5", version=1, hypothesis="primary", items=("goals", "main"),
          description="X5: log mu = (1 - w) log lambda_B0 + w log E[xG] (рейтинг xG лиги по матчам до дня матча, "
                      "w — еженедельно по Пуассону); 1X2 и рынки — из скорректированных mu",
          baseline="B0 = прогноз продакшна",
          metric="средняя девиация Пуассона голов (дом + гости), X5 - B0",
          criterion="разность < 0 при одностороннем бутстреп-p < 0.05; ΔLL 1X2 < +0.002; ΔROI Main > -0.02; "
                    "переоценка Main не больше, чем у B0",
          min_n=3000, forward_rule={"kind": "kickoff_after_utc", "utc": "2026-11-09T17:03:59Z"}, family="xg"),
    _spec(XG, id="xg_t1", version=1, hypothesis="secondary", items=("1x2",),
          description="T1: модели v6.1, переобученные с признаками прокси-xG по ударам",
          baseline="B0 = прогноз продакшна", metric="log loss 1X2, T1 - B0",
          criterion="для информации (Холм по двум форвард-тестам xG)", min_n=3000,
          forward_rule={"kind": "kickoff_after_utc", "utc": "2026-11-09T17:03:59Z"}, family="xg",
          implemented=False,
          not_implemented_reason="нужна отдельная версия моделей XGBoost (переобучение на 2021/22–2025/26 с 9 "
                                 "признаками прокси-xG) и её загрузка в ежедневном прогоне; это не «дешёвая» тень "
                                 "без переобучения. Заглушка: запись не ведётся."),
    _spec(UNCERTAINTY, id="uncertainty_flag_u_1x2", version=1, hypothesis="flag_u_1x2", items=("main",),
          description="Метка «ненадёжный прогноз» у Main, если разброс 1X2 по 8 бутстреп-моделям выше замороженного "
                      "порога (0.8-квантиль 2026/27); выбор и вероятность Main не меняются",
          baseline="Main без метки", metric="β_u в логит-модели выигрыша при logit pc; gap(с меткой) − gap(без)",
          criterion="95 % ДИ β_u < 0 и 95 % ДИ gap(с меткой) − gap(без) > 0 (бутстреп по матчам, 2000)",
          min_n=2000, forward_rule={"kind": "kickoff_after_utc", "utc": "2026-11-09T18:29:30Z"},
          family="uncertainty", implemented=False,
          not_implemented_reason="нужны 8 бутстреп-моделей 1X2, обученных один раз на сезонах до 2026/27 и "
                                 "сохранённых рядом с моделями продакшна; после этого ежедневная цена — 8 прогнозов "
                                 "на матч. Модели в этом исследовании не сохранялись, запись не ведётся."),
)


def get(shadow_id: str, version: int | None = None) -> ShadowSpec:
    found = [s for s in SHADOWS if s.id == shadow_id and (version is None or s.version == version)]
    if not found:
        raise KeyError(shadow_id)
    return max(found, key=lambda s: s.version)


def active() -> list[ShadowSpec]:
    return [s for s in SHADOWS if s.implemented]
