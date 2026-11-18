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


# v6.3 (experiments/v6_3/*/results/preregistration.json), frozen by the commits that added each study
V63 = {
    "main_selection": ("e62797ad1597163dfc396489c5437ae66c82749814316ad2223a990d5f1be8df",
                       "2bd1e2c77c1bff873ed5332be3c0f3c8aba994ac", "2026-11-18T13:15:58Z"),
    "market_profitability": ("f36b2817ee8aa5c81cb035c49823b4913974afe59282879a4850a0af8e847d51",
                             "0a872ace3c6d6426ef5d9f5e323796a7b5d067a6", "2026-11-18T13:15:58Z"),
    "market_aware": ("01fbd6be20319098c7905f4819a98bb20776a85912c8af46eb9f97fe6dbea87e",
                     "0485c7754be5494af10f8e665737e4749374871a", "2026-11-18T13:16:03Z"),
    "consensus": ("28f7dfd5789f8a9b31f7b352ede37cb9de7b41261cee4585016c330bcc51d9fd",
                  "d9352574221603bfb7c50528e539176775a63abe", "2026-11-18T13:16:03Z"),
    "calibration": ("28b6d2b410fc81e203b3ea52232775cefdb7babe8527b69fbfaa53f97ff78a77",
                    "5c610f43b5c2d532f95213028dece51bba93d6bc", "2026-11-18T13:47:20Z"),
    "meta_selection": ("71da781da47bb3b2207187e6baa49640666319371eac94d03edf36225c5c55b9",
                       "a09cd09f4c68177a016e1ea99749458baa28a4de", "2026-11-18T13:47:20Z"),
    "goals": ("89607fa4ec2b957205ac8807682cef0fb58734246b8855b77415eabebba759a8",
              "7e88d1844354520fbfd07c5af8d6a998bed8cda0", "2026-11-18T14:24:30Z"),
    "corners": ("70f9be5051c2ffcd57d0aad97c3f602a8c12b09a01f5091031f7890a20a90310",
                "e889203302eaf8430c679c1723c614d5ac4169a8", "2026-11-18T13:57:13Z"),
    "alt_ml": ("77c476d4512b049b0ec02c24e128d34f5ad16a10e9fcebb39c22cef21aa3c5ae",
               "ef056c3d8770bb47cd65368234bf5f7c5a019e08", "2026-11-18T13:59:57Z"),
    "quality": ("c49c3a359064636b706cab4a373e6c6b24bb55dd28a101d07de6bb6d1d1e1aca",
                "2112d5af812139f787808aae3fdf3fae7a882165", "2026-11-18T13:59:57Z"),
    "recent_weighting": ("fb8b526632118a91c37696043d39fb789c69d149cbd8d04880510869c569aa76",
                         "160fff2a5830fa2d6c691beacf191fee07ee3fb5", "2026-11-18T14:24:30Z"),
    "squad": ("decb920995dc9a2c509e8b91c2332ec0bb9c84594734b99d8d3047c95ee4f37b",
              "44e87eea5de1b208cfd0354d882b12600fbd81f3", "2026-11-18T13:27:48Z"),
}
for _study, (_sha256, _commit, _at) in V63.items():
    _path = f"experiments/v6_3/{_study}/results/preregistration.json"
    SHA[_path] = _sha256
    COMMIT[_path] = (_commit, _at)


def _v63(study: str, **kw) -> ShadowSpec:
    path = f"experiments/v6_3/{study}/results/preregistration.json"
    kw.setdefault("forward_rule", {"kind": "kickoff_after_utc", "utc": COMMIT[path][1]})
    return _spec(path, created="2026-11-18", **kw)


_V63_RULE = dict(items=("main",), baseline="Main продакшна (те же сохранённые вероятности и цены момента прогноза)",
                 metric="парная разность ROI теневого Main и Main продакшна (ставка 1, цена момента прогноза)",
                 criterion="95 % ДИ разности ROI > 0 (бутстреп по матчам, 2000, seed 20261109); Холм по семье "
                           "v63_main_rule", family="v63_main_rule")
_V63_LATER = "нужна ежемесячная переподгонка параметров (или отдельные модели) в ежедневном прогоне; в v6.3 " \
             "записываются только правила выбора Main без обучения. Гипотеза заморожена, проверка — повторным " \
             "воспроизведением исследования на матчах после заморозки."


def _spec(prereg: str, created: str = "2026-11-09", **kw) -> ShadowSpec:
    sha, at = COMMIT[prereg]
    return ShadowSpec(prereg=prereg, prereg_sha256=SHA[prereg], freeze_sha=sha, frozen_at_utc=at,
                      created=created, **kw)


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
    # ---- v6.3: Main-rule variants recorded daily (no fitting); the rest are frozen but not recorded
    _v63("main_selection", id="v63_main_reliability_total", version=1, hypothesis="v63_main_selection:bt:reliability",
         description="Правило Main продакшна с MARKET_RELIABILITY['TOTAL'] = 0.95 (вместо 1.00)", min_n=2596,
         **_V63_RULE),
    _v63("market_profitability", id="v63_main_reliability_handicap", version=1,
         hypothesis="v63_market_profitability:bt:boost",
         description="Правило Main продакшна с MARKET_RELIABILITY['HANDICAP'] = 1.035 (вместо 0.90); тот же эффект, "
                     "что у TOTAL 0.95, — решение по одной из двух", min_n=3993, **_V63_RULE),
    _v63("main_selection", id="v63_main_ev_weight", version=1, hypothesis="v63_main_selection:bt:ev_weight",
         description="Правило Main продакшна с MAIN_EV_WEIGHT = 1.0 (вместо 0.5)", min_n=8227, **_V63_RULE),
    _v63("main_selection", id="v63_main_floor_140", version=1, hypothesis="v63_main_selection:floor_1.40",
         description="Main только среди кандидатов с реальной ценой ≥ 1.40 (по оценке продакшна); если таких нет — "
                     "статистический Main продакшна, который не считается ставкой", min_n=2342, **_V63_RULE),
    _v63("market_aware", id="v63_market_blend_v6", version=1, hypothesis="v63-market-aware-V6", items=("main",),
         description="Правило Main на смеси 50/50 логарифмов вероятностей модели и рынка", baseline="Baseline B",
         metric="парная разность ROI", criterion="см. предрегистрацию", min_n=1000, family="v63_market",
         implemented=False, not_implemented_reason="нужна полная книга одного букмекера на момент прогноза для "
                                                   "каждого рынка; в сохранённых ценах селекции одного рынка могут "
                                                   "быть от разных букмекеров. " + _V63_LATER),
    _v63("consensus", id="v63_consensus_c2", version=1, hypothesis="v63-consensus-C2", items=("main",),
         description="Правило Main на смеси модели и консенсуса нескольких букмекеров", baseline="Baseline B",
         metric="парная разность ROI", criterion="см. предрегистрацию", min_n=1000, family="v63_market",
         implemented=False, not_implemented_reason="нужны цены нескольких букмекеров на момент прогноза и "
                                                   "ежемесячная подгонка весов. " + _V63_LATER),
    _v63("calibration", id="v63_calibration_temperature_group", version=1,
         hypothesis="v63_calibration_temperature_group_pooled@v1", items=("main",),
         description="Правило Main на вероятностях после температурной калибровки по группам рынков",
         baseline="Baseline B", metric="парная разность ROI", criterion="см. предрегистрацию", min_n=1500,
         family="v63_calibration", implemented=False, not_implemented_reason=_V63_LATER),
    _v63("meta_selection", id="v63_meta_lr_ev", version=1, hypothesis="v63_meta_selection_meta_lr_ev@v1",
         items=("main",), description="Метамодель (логистическая регрессия) выбирает Main с максимальным EV",
         baseline="Baseline B", metric="парная разность ROI", criterion="см. предрегистрацию", min_n=1500,
         family="v63_meta", implemented=False, not_implemented_reason=_V63_LATER),
)


def get(shadow_id: str, version: int | None = None) -> ShadowSpec:
    found = [s for s in SHADOWS if s.id == shadow_id and (version is None or s.version == version)]
    if not found:
        raise KeyError(shadow_id)
    return max(found, key=lambda s: s.version)


def active() -> list[ShadowSpec]:
    return [s for s in SHADOWS if s.implemented]
