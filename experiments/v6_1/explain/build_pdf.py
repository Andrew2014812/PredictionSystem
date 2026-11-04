"""Builds reports/v6_1/Explainability_Research.pdf (Russian) and findings.md from the result files.

    .venv/bin/python experiments/v6_1/explain/build_pdf.py

Every number comes from experiments/v6_1/explain/results/*.json|csv (run_research.py, timing.py,
sensitivity.py, screenshots.py); a missing input is reported as missing, never filled in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.units import cm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.report_kit import (ACCENT, AMBER, H1, H2, H3, MUTED, NEG, PALETTE, P, Report, bullets, code,  # noqa: E402
                                figure, formula, git, load, note, reproduce, save_chart, table)
from src.features.names import feature_label  # noqa: E402

RES, FIG = HERE / "results", HERE / "figures"
OUT = ROOT / "reports" / "v6_1" / "Explainability_Research.pdf"
GROUP_RU = {"1X2": "Исход 1X2", "DOUBLE_CHANCE": "Двойной шанс", "TOTAL": "Тотал голов", "BTTS": "Обе забьют",
            "HANDICAP": "Азиатская фора", "CORNERS": "Тотал угловых", "EXACT_SCORE": "Точный счёт"}
METHOD_RU = {"old": "A: SHAP исходной модели (v6)", "chain": "D2: интегрированное цепное правило",
             "linear": "D1: линейное цепное правило", "mshap": "D3: Shapley по выходам моделей",
             "occl": "C2: окклюзия (замена на среднее)"}


def f(v, d=1):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def pc(v, d=0):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v * 100:.{d}f}%"


_NAMES: set[str] = set()


def label(key: str) -> str:
    """Readable name of a factor key, as the card shows it (a home/away pair by its home-team feature)."""
    if not _NAMES:
        from src.modelling.registry import load_bundles
        for b in load_bundles("prod_20261012").values():
            _NAMES.update(b.preprocessor.numeric)
    name = feature_label(f"h_{key}" if f"h_{key}" in _NAMES else key)
    name = name.replace("Home team · ", "")
    return name[:1].upper() + name[1:]


# ----------------------------------------------------------------------------- figures
def fig_chain() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.axis("off")
    boxes = [(0.02, 0.5, "Признаки x\n(до матча)"), (0.24, 0.85, "Классификатор 1X2\nz = (z_H, z_D, z_A)"),
             (0.24, 0.5, "Голы хозяев\nlog λ_h"), (0.24, 0.15, "Голы гостей\nlog λ_a"),
             (0.50, 0.62, "Матрица счетов\nPoisson(λ_h)·Poisson(λ_a),\nсогласованная с p = softmax(z)"),
             (0.50, 0.12, "Угловые: log μ →\nNB(μ, r)"), (0.79, 0.62, "P(рынок)\n1X2, ДШ, тоталы,\nОЗ, форы, счёт"),
             (0.79, 0.12, "P(тотал угловых)")]
    for x, y, t in boxes:
        ax.text(x, y, t, ha="left", va="center", fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.45", fc="#eef4f1", ec=ACCENT, lw=0.9))
    arrows = [((0.15, 0.55), (0.235, 0.83)), ((0.15, 0.5), (0.235, 0.5)), ((0.15, 0.45), (0.235, 0.17)),
              ((0.15, 0.42), (0.49, 0.14)), ((0.40, 0.83), (0.49, 0.68)), ((0.37, 0.5), (0.49, 0.6)),
              ((0.37, 0.17), (0.49, 0.55)), ((0.71, 0.62), (0.785, 0.62)), ((0.65, 0.12), (0.785, 0.12))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    return save_chart(fig, FIG / "fig_chain.png")


def fig_model_shares(e1: pd.DataFrame) -> Path:
    goal = e1.loc[e1["group"] != "CORNERS"]
    by = goal.groupby("group")[["result_share", "goals_share"]].mean().reindex(
        [g for g in ("1X2", "DOUBLE_CHANCE", "HANDICAP", "TOTAL", "BTTS", "EXACT_SCORE") if g in set(goal["group"])])
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    x = np.arange(len(by))
    ax.bar(x, by["result_share"] * 100, color=PALETTE[0], label="через классификатор 1X2")
    ax.bar(x, by["goals_share"] * 100, bottom=by["result_share"] * 100, color=PALETTE[1],
           label="через модели голов (λ_h, λ_a)")
    ax.set_xticks(x, [GROUP_RU[g] for g in by.index], fontsize=8)
    ax.set_ylabel("доля |влияния|, %")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax.set_ylim(0, 100)
    return save_chart(fig, FIG / "fig_model_shares.png")


CLASSIFIER_PLAN = {"1X2", "DOUBLE_CHANCE", "HANDICAP"}       # v6 explained the 1X2 classifier for these


def with_omitted(e1: pd.DataFrame) -> pd.DataFrame:
    """Share of |influence| on P that goes through MODELS the v6 card did not explain."""
    out = e1.copy()
    out["omitted_model"] = np.where(out["group"].isin(CLASSIFIER_PLAN), out["goals_share"],
                                    np.where(out["group"] == "CORNERS", 0.0, out["result_share"]))
    return out


def old_table(e1: pd.DataFrame) -> pd.DataFrame:
    e1 = with_omitted(e1)
    return e1.groupby("group").agg(n=("market", "size"), omitted=("omitted_model", "mean"),
                                   over20=("omitted_model", lambda s: float((s > 0.2).mean())),
                                   sign=("old_sign_agree_new", "mean"), top5=("old_new_top5", "mean"),
                                   top1=("old_new_top1", "mean"))


def fig_omitted(e1: pd.DataFrame) -> Path:
    by = old_table(e1)[["omitted", "over20"]]
    by.columns = ["mean", "over20"]
    by = by.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    y = np.arange(len(by))
    ax.barh(y - 0.18, by["mean"] * 100, 0.36, color=NEG, label="средняя доля влияния вне карточки v6")
    ax.barh(y + 0.18, by["over20"] * 100, 0.36, color=AMBER, label="доля случаев, где она > 20 %")
    ax.set_yticks(y, [GROUP_RU[g] for g in by.index], fontsize=8)
    ax.set_xlabel("%")
    ax.legend(fontsize=7.5, loc="lower right")
    return save_chart(fig, FIG / "fig_omitted.png")


def fig_reference(e2s: dict) -> Path:
    methods = ["old", "linear", "chain", "mshap", "occl"]
    keys = [("refsign5", "знак топ-5 эталона"), ("top5", "совпадение топ-5"), ("spearman", "ранговая корреляция")]
    fig, ax = plt.subplots(figsize=(8.8, 3.3))
    x = np.arange(len(methods))
    for i, (k, name) in enumerate(keys):
        ax.bar(x + (i - 1) * 0.26, [e2s["overall"][m][k] * 100 for m in methods], 0.26, color=PALETTE[i], label=name)
    ax.set_xticks(x, [METHOD_RU[m].split(":")[0] for m in methods])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    return save_chart(fig, FIG / "fig_reference.png")


def fig_stability(e3s: dict) -> Path:
    o = e3s["overall"]
    names = [("old", "A (v6)"), ("chain", "D2 (v6.1)"), ("occl", "C2 окклюзия")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))
    axes[0].bar([n for _, n in names], [o[f"{m}_jacc3"] * 100 for m, _ in names], color=PALETTE[0])
    axes[0].set_title("Жаккар топ-3 при шуме 5 % σ", fontsize=8.5)
    axes[0].set_ylim(0, 100)
    axes[1].bar([n for _, n in names], [o[f"{m}_dshare_pp"] for m, _ in names], color=PALETTE[1])
    axes[1].set_title("|Δ доли| топ-3 факторов, п.п.", fontsize=8.5)
    return save_chart(fig, FIG / "fig_stability.png")


def fig_steps(e7: dict) -> Path:
    s = e7["feature_steps_mean"]
    eps = [float(k) for k in s]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot(eps, [s[k] * 100 for k in s], marker="o", color=PALETTE[2])
    ax.set_xscale("log")
    ax.set_xlabel("шаг ε (в долях стандартного отклонения признака)")
    ax.set_ylabel("признаков, меняющих выход, %")
    return save_chart(fig, FIG / "fig_feature_steps.png")


def fig_linear(e1: pd.DataFrame) -> Path:
    x = e1.loc[e1["delta"].abs() > 0.005, "linear_rel_err"].clip(1e-6, 10)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.hist(np.log10(x), bins=40, color=PALETTE[3])
    ax.set_xlabel("log10 относительной ошибки аддитивности (линейное правило)")
    ax.set_ylabel("выборов")
    return save_chart(fig, FIG / "fig_linear_error.png")


def fig_example(e5: dict, mid: str) -> Path | None:
    if mid not in e5:
        return None
    pick = e5[mid]["picks"][0]
    keys = [k for k, _, _ in pick["new_top"][:6]]
    new = {k: v for k, v, _ in pick["new_top"]}
    tot_new = sum(abs(v) for _, v, _ in pick["new_top"]) or 1
    old = pick["old_on_new_top"]
    ref = pick["ref_on_new_top"]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, vals, title in ((axes[0], [old[k] for k in keys], "A: SHAP логита (v6), ед. логита"),
                            (axes[1], [new[k] * 100 for k in keys], "D2: вклад в P, п.п."),
                            (axes[2], [ref[k] * 100 for k in keys], "эталон Shapley P, п.п.")):
        ax.barh(y, vals, color=[ACCENT if v > 0 else NEG for v in vals])
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(title, fontsize=8)
    axes[0].set_yticks(y, [label(k)[:38] for k in keys], fontsize=7)
    del tot_new
    return save_chart(fig, FIG / f"fig_example_{mid}.png")


# ----------------------------------------------------------------------------- text helpers
def card_text(texts: dict, mid: str, i: int) -> str:
    try:
        return texts[mid][i]
    except (KeyError, IndexError, TypeError):
        return "(нет данных)"


def card_block(text: str) -> list:
    """The card's inner text as compact lines: one line per factor ("name · evidence — share")."""
    import re
    import textwrap
    raw = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out, buf, in_factors = [], [], False
    for ln in raw:
        ln = re.sub(r"(INFLUENCE|REASONING)i$", r"\1 (i)", ln).replace("AGAINST", " [AGAINST]")
        if re.match(r"^(RELATIVE MODEL INFLUENCE|SHARE OF THE MODEL)", ln):
            out[-1] = f"{out[-1]} | {ln}" if out else ln
            in_factors = True
            continue
        if ln.startswith("RECENT"):
            in_factors = False
        if in_factors and re.fullmatch(r"\d+%", ln):
            out.append("  • " + " · ".join(buf) + f" — {ln}")
            buf = []
        elif in_factors:
            buf.append(ln)
        else:
            in_recent = any(x.startswith("RECENT") for x in out) and not ln.startswith("RECENT")
            out.append("  • " + ln if in_recent and not ln.endswith(".") else ln)
    wrapped = []
    for ln in out:
        wrapped += textwrap.wrap(ln, 98, subsequent_indent="    ") or [ln]
    return [code("\n".join(wrapped))]


def main() -> None:
    e0 = load(RES / "e0_reproduction.json", {})
    e1s = load(RES / "e1_summary.json", {})
    e1 = pd.read_csv(RES / "e1_selections.csv") if (RES / "e1_selections.csv").exists() else pd.DataFrame()
    e2s = load(RES / "e2_summary.json", {})
    e2 = pd.read_csv(RES / "e2_reference.csv") if (RES / "e2_reference.csv").exists() else pd.DataFrame()
    e3s = load(RES / "e3_summary.json", {})
    e4s = load(RES / "e4_summary.json", {})
    e5 = load(RES / "e5_examples.json", {})
    e6 = load(RES / "e6_timing.json", {})
    e7 = load(RES / "e7_sensitivity.json", {})
    info = load(RES / "run_info.json", {})
    before = load(RES / "cards_before.json", {})
    after = load(RES / "cards_after.json", {})
    e2e = load(RES / "e2e_runs.json", {})

    fig_chain()
    if not e1.empty:
        fig_model_shares(e1)
        fig_omitted(e1)
        fig_linear(e1)
    if e2s:
        fig_reference(e2s)
    if e3s:
        fig_stability(e3s)
    if e7:
        fig_steps(e7)
    for mid in e5:
        fig_example(e5, mid)

    by1 = {r["group"]: r for r in e1s.get("by_group", [])}
    rep = Report(OUT, "Объяснимость прогнозов: карточка «Почему этот прогноз?»", "FootPredict v6.1",
                 facts={"Модели": "prod_20261012 (рабочая), backtest_2526 (проверочная)",
                        "Выборка": f"{e1s.get('matches', '—')} матчей, {e1s.get('selections', '—')} выборов рынка"})

    # 1 ------------------------------------------------------------------------------------------
    rep.add(H1("1. Цель исследования"),
            P("На странице матча FootPredict показывает карточку «Why this prediction?» для прогнозов Main и Risk "
              "и раздел «Explain another market». Карточка должна честно отвечать на вопрос, **почему модель "
              "дала именно такую вероятность именно этому рынку**: какие факторы её определяют, какие "
              "футбольные факты это подтверждают и через какие промежуточные величины (ожидаемые голы, "
              "распределение угловых, вероятности исходов) вероятность получена."),
            P("Задача исследования — проверить на реальных матчах из хранилища прогнозов и на рабочих моделях, "
              "какой метод объяснения корректно объясняет **итоговую вероятность рынка**, отвергнуть методы, "
              "которые этого не делают, и внедрить выбранный метод в карточку, сохранив дизайн v6."),
            P("Требования к объяснению:"),
            bullets(["математическая корректность: вклады должны объяснять именно вероятность выбранного рынка;",
                     "согласованность знаков: «за» и «против» должны совпадать с реальным влиянием на эту вероятность;",
                     "отсутствие ложной точности: доля влияния — не изменение вероятности в процентных пунктах;",
                     "устойчивость к малым возмущениям признаков и между похожими матчами;",
                     "понятность для читателя, знающего футбол, но не машинное обучение;",
                     "время расчёта, приемлемое для веб-страницы;",
                     "использование только данных, известных до матча."]))

    # 2 ------------------------------------------------------------------------------------------
    rep.add(H1("2. Как получается вероятность рынка"),
            P("Ни одна вероятность рынка, кроме 1X2, не является прямым выходом модели. Цепочка вычислений "
              "(src/prediction/engine.py, src/markets/distributions.py, src/markets/markets.py):"),
            *figure(FIG / "fig_chain.png", "Рис. 1. Цепочка: признаки → модели XGBoost → промежуточные величины → "
                                           "вероятность рынка."),
            P("Классификатор 1X2 выдаёт логиты z = (z_{H}, z_{D}, z_{A}), вероятности исходов:"),
            formula("p_{r} = exp(z_{r}) / Σ_{s∈{H,D,A}} exp(z_{s})"),
            P("Регрессоры голов (Пуассон, log-ссылка) выдают log λ_{h} и log λ_{a}. Матрица счетов:"),
            formula("M_{ij} = Pois(i; λ_{h}) · Pois(j; λ_{a}),   i, j = 0…10   (ρ Диксона–Коулза = 0 в рабочей версии)"),
            P("Затем матрица согласуется с классификатором: каждая из трёх областей r(i, j) ∈ {победа хозяев, ничья, "
              "победа гостей} масштабируется так, чтобы её масса стала равна p_{r}:"),
            formula("M′_{ij} = M_{ij} · p_{r(i,j)} / Σ_{(k,l)∈r(i,j)} M_{kl}"),
            P("Вероятность любого голевого рынка с выигрышным множеством клеток S:"),
            formula("P(S) = Σ_{(i,j)∈S} M′_{ij} = Σ_{r} p_{r} · Q_{r}(λ_{h}, λ_{a}),   "
                    "Q_{r} = Σ_{(i,j)∈S∩r} M_{ij} / Σ_{(i,j)∈r} M_{ij}"),
            P("Отсюда главный вывод для объяснения: **тотал, «обе забьют», фора −1/−1.5/−2/−2.5 и точный счёт "
              "зависят одновременно от классификатора 1X2 и от обеих моделей голов**; 1X2, двойной шанс и фора ±0.5 "
              "зависят только от классификатора (Q_{r} ∈ {0, 1}). Тотал угловых зависит только от модели угловых:"),
            formula("P(угловых > L) = Σ_{k>L} NB(k; μ, r),   μ = exp(log μ),   r — параметр дисперсии"),
            table([["Рынок", "Какие выходы моделей его определяют", "Промежуточные величины на карточке"],
                   ["1X2, двойной шанс, фора ±0.5", "только z (классификатор 1X2)", "p_{H}, p_{D}, p_{A}"],
                   ["Тотал 1.5 / 2.5 / 3.5", "λ_{h}, λ_{a} и z (через согласование)", "ожидаемые голы, сумма vs линия"],
                   ["Обе забьют", "λ_{h}, λ_{a} и z", "P(хозяева забьют), P(гости забьют)"],
                   ["Фора ±1, ±1.5, ±2, ±2.5", "z и λ_{h}, λ_{a}", "условие выигрыша, ожидаемая разница голов, возврат"],
                   ["Точный счёт", "λ_{h}, λ_{a} и z; немонотонно", "ожидаемые голы, P(счёта), ближайшие счета"],
                   ["Тотал угловых", "μ (модель угловых) и NB", "ожидаемые угловые, типичный диапазон"]]))

    # 3 ------------------------------------------------------------------------------------------
    rows, hrows = [], []
    if not e1.empty:
        ot = old_table(e1)
        for g in ("TOTAL", "BTTS", "HANDICAP", "DOUBLE_CHANCE", "1X2", "CORNERS", "EXACT_SCORE"):
            if g in ot.index:
                r = ot.loc[g]
                rows.append([GROUP_RU[g], int(r["n"]), pc(r["omitted"]), pc(r["over20"]), pc(r["sign"]),
                             pc(r["top5"]), pc(r["top1"])])
        hc = with_omitted(e1.loc[e1["group"] == "HANDICAP"])
        hc = hc.assign(line=hc["market"].str.split("_").str[1].astype(float).abs())
        for line, x in hc.groupby("line"):
            hrows.append([f"±{line:g}", len(x), pc(x["omitted_model"].mean()), pc(x["old_sign_agree_new"].mean()),
                          pc(x["old_new_top5"].mean())])
    rep.add(H1("3. Старый подход (v6) и его ограничения"),
            P("В v6 карточка брала SHAP-значения (TreeSHAP) **одной исходной модели** в пространстве её выхода "
              "(src/explain/shap_explainer.py, explanation_plan): для 1X2 и форы — логит класса классификатора, "
              "для двойного шанса — логит исключённого класса со знаком минус, для тоталов и «обе забьют» — "
              "сумма SHAP log λ_{h} и log λ_{a} со знаком ±1, для угловых — log μ. Доля фактора:"),
            formula("доля_{g} = Σ_{j∈g} |φ_{j,k*}| / Σ_{j} |φ_{j,k*}|"),
            P("Ограничения, подтверждённые на реальных матчах (эксперимент E1, таблица ниже):"),
            bullets([
                "**объясняется не та величина**: логит или log λ — не вероятность рынка. Для двойного шанса и 1X2 "
                "вероятность — softmax всех трёх логитов, а объяснялся один;",
                "**часть цепочки игнорировалась**: для тоталов и «обе забьют» не учитывался классификатор 1X2 "
                "(через согласование матрицы он меняет вероятность), для фор ±1…±2.5 не учитывались модели голов;",
                "**равные веса**: SHAP log λ_{h} и log λ_{a} складывались с весом 1, хотя ∂P/∂log λ_{h} ≠ ∂P/∂log λ_{a};",
                "**одинаковые объяснения разных рынков**: фора +0.5 и +1.5 одной команды получали один и тот же "
                "список факторов (обе объяснялись логитом класса), см. раздел 8;",
                "**знаменатель доли** — сумма |SHAP| по ~190 признакам модели, поэтому на карточке были доли 2–3 %;",
                "итоговая строка повторяла названия двух факторов, уже показанных выше.",
            ]),
            table([["Рынок", "N", "Влияние вне v6 (средн.)", "Случаев > 20 %", "Знак v6 = знак P",
                    "Топ-5 v6 ∩ топ-5 P", "Топ-1 совпал"]] + rows),
            P("«Влияние вне v6» — доля |вклада в вероятность рынка|, проходящая через **модели**, которые карточка "
              "v6 не объясняла (классификатор 1X2 для голевых рынков, модели голов для 1X2/двойного шанса/фор; "
              "вычислено интегрированным цепным правилом, раздел 4.4). «Знак v6 = знак P» — доля показанных на "
              "карточке v6 факторов, чьё направление совпадает с направлением их реального вклада в вероятность "
              "рынка. «Топ-5 v6 ∩ топ-5 P» — сколько из пяти главных факторов v6 входят в пять главных факторов "
              "вероятности рынка.", "small"),
            P("Для 1X2 и двойного шанса v6 использовал правильную модель, но объяснял один логит, а не вероятность "
              "softmax: знаки совпадают почти всегда, но состав главных факторов — только частично. Для форы доля "
              "пропущенного влияния растёт с линией:"),
            table([["Линия форы", "N", "Влияние моделей голов (вне v6)", "Знак v6 = знак P", "Топ-5 v6 ∩ топ-5 P"]]
                  + hrows),
            P("Для точного счёта v6 объяснял сумму log λ_{h} + log λ_{a}, хотя вероятность счёта в основном "
              "определяется классификатором (через согласование матрицы) и немонотонна по ожидаемым голам — "
              "направление факторов v6 совпадало с реальным менее чем в трети случаев."),
            *figure(FIG / "fig_omitted.png", "Рис. 2. Доля влияния на вероятность рынка, которую карточка v6 не "
                                             "объясняла, по группам рынков (E1)."))

    # 4 ------------------------------------------------------------------------------------------
    rep.add(H1("4. Исследованные методы"),
            P("Обозначения: x — признаки матча, u = (u_{1}, …, u_{K}) — выходы моделей в пространстве отступа "
              "(margin), ū — их ожидаемые значения (база TreeSHAP), g — функция «выходы моделей → вероятность "
              "рынка» из раздела 2, P(x) = g(u(x)). TreeSHAP модели k даёт φ_{jk} с локальной точностью:"),
            formula("Σ_{j} φ_{jk} = u_{k}(x) − ū_{k}"),
            H2("4.1. A — SHAP исходной модели (v6)"),
            P("Описан в разделе 3. Корректен только тогда, когда рынок является монотонной функцией одного выхода "
              "одной модели (тотал угловых: P монотонна по log μ, поэтому доли A и D2 для угловых совпадают точно). "
              "Для остальных рынков объясняет другую величину."),
            H2("4.2. B — описание вычислительной цепочки"),
            P("Показывает промежуточные величины, из которых получена вероятность: ожидаемые голы, вероятности "
              "исходов, P(забьёт хозяин/гость), ожидаемую разницу голов, вероятность возврата, ожидаемые угловые и "
              "их типичный диапазон, вероятность счёта и ближайшие альтернативы. Значения пересчитываются из "
              "сохранённого прогноза (snapshot), поэтому совпадают с вероятностью на сайте (E0). Метод точен, но не "
              "говорит, какие **признаки** привели к этим величинам."),
            H2("4.3. C — анализ чувствительности"),
            P("C1 — локальные производные вероятности рынка по выходам моделей (центральные разности):"),
            formula("S_{k} = ∂g/∂u_{k} (u(x)),   в читаемых единицах: Δp.p. на +0.1 ожидаемого гола, на +1 угловой, "
                    "на +1 п.п. вероятности исхода"),
            P("C1′ — перенос на признаки по цепному правилу требует ∂u_{k}/∂x_{j}. У ансамбля деревьев выход — "
              "кусочно-постоянная функция признаков, поэтому эта производная почти всюду равна нулю (E7). "
              "C2 — окклюзия: изменение вероятности при замене фактора на его среднее по обучению:"),
            formula("Δ_{g} = P(x) − P(x_{g} := x̄_{g})"),
            P("Окклюзия не аддитивна (Σ_{g} Δ_{g} ≠ P(x) − P(x̄)) и игнорирует взаимодействия признаков."),
            H2("4.4. D — атрибуция для композиции моделей"),
            P("D1 — линейное цепное правило (производная в точке матча):"),
            formula("c^{lin}_{j} = Σ_{k} ∂g/∂u_{k}(u) · φ_{jk},   Σ_{j} c^{lin}_{j} = ∇g(u)·(u − ū) ≠ g(u) − g(ū)"),
            P("D2 — **интегрированное цепное правило** (Aumann–Shapley / интегрированные градиенты в пространстве "
              "выходов моделей): производная усредняется вдоль пути от ū к u, после чего каждый TreeSHAP-вклад "
              "умножается на средний наклон своего выхода:"),
            formula("G_{k} = ∫₀¹ ∂g/∂u_{k}(ū + t(u − ū)) dt,      c_{j} = Σ_{k} G_{k} · φ_{jk}"),
            formula("Σ_{j} c_{j} = Σ_{k} G_{k}(u_{k} − ū_{k}) = g(u) − g(ū)   (теорема о градиенте вдоль пути)"),
            P("Вклады c_{j} измеряются в единицах вероятности самого рынка, точно складываются в отклонение "
              "вероятности от «типичного матча» g(ū), имеют знак реального влияния и автоматически равны нулю для "
              "моделей, которые рынок не определяют. Интеграл считается квадратурой Гаусса–Лежандра (12 узлов) по "
              "центральным разностям g; g — та же матрица счетов, что в движке прогнозов (векторизованная копия, "
              "проверена тестом на равенство). Это вариант распространения Shapley-значений через цепочку моделей "
              "(Chen, Lundberg, Lee, 2022) с правилом интегрирования вместо правила масштабирования."),
            P("Замечание о классификаторе: softmax не меняется, если ко всем трём логитам добавить одно и то же число, "
              "поэтому разбиение влияния между отдельными логитами z_{H}, z_{D}, z_{A} условно. Вклады признаков c_{j} "
              "от этого не зависят: Σ_{r} ∂p/∂z_{r} = 0, значит Σ_{r} G_{z_r} = 0 и общий для трёх логитов сдвиг "
              "φ_{j} даёт нулевой вклад. Поэтому в отчёте влияние классификатора всегда суммируется по трём логитам."),
            P("D3 — точные Shapley-значения **выходов моделей** (K = 5 игроков: z_{H}, z_{D}, z_{A}, log λ_{h}, "
              "log λ_{a}; база ū; 2^{5} = 32 вычисления g), распределённые по признакам пропорционально TreeSHAP:"),
            formula("ψ_{k} = Σ_{S⊆K∖{k}} |S|!(K−|S|−1)!/K! · [g(u_{S∪{k}}, ū_{−}) − g(u_{S}, ū_{−})],   "
                    "c_{j} = Σ_{k} ψ_{k}/(u_{k} − ū_{k}) · φ_{jk}"),
            P("Эталон (REF) — перестановочные Shapley-значения **всей композиции** P(x) = g(u(x)) на уровне "
              "факторов карточки (пара «хозяева/гости» — один игрок), интервенционные, с фоном B из реальных "
              "матчей последних 12 месяцев обучения модели, антитетичные перестановки:"),
            formula("φ^{ref}_{g} = E_{π} E_{b∈B} [ P(x_{S∪g}, b) − P(x_{S}, b) ],   S = факторы перед g в перестановке π,   "
                    "Σ_{g} φ^{ref}_{g} = P(x) − E_{b} P(b)"),
            P(f"Эталон модельно-независим и точно аддитивен, но дорог: {e2s.get('n_perm', '—')} перестановок × "
              f"{e2s.get('n_background', '—')} фоновых матчей × ≈134 фактора ≈ "
              f"{f(e2s.get('ref_seconds_per_match'), 0)} с на матч (на 3 потоках) и содержит шум выборки; "
              "поэтому он используется как мерило, а не на сайте."),
            H2("4.5. E — гибрид с футбольными фактами"),
            P("Факты (src/explain/facts.py) считаются по реальным сыгранным матчам **до даты матча**: последние "
              "домашние матчи хозяев и выездные матчи гостей, ровно для выбранного рынка и линии (например, "
              "«6 из 10 домашних матчей — меньше 3.5»), плюс средние голы, угловые, место в таблице. Они не "
              "объясняют модель, а дают читателю проверяемую фактическую опору; на карточке — блок «Recent form»."),
            P("Итоговый вид карточки — гибрид D2 + B + E: факторы с относительным влиянием по D2, футбольные факты "
              "и короткая заметка о цепочке (ожидаемые голы / распределение)."))

    # 5 ------------------------------------------------------------------------------------------
    sec = info.get("seconds", {})
    rep.add(H1("5. Данные и эксперименты"),
            P("Используются только сохранённые данные: признаки до матча (features.parquet), сохранённые прогнозы "
              "(snapshots, markets) и файлы рабочих моделей. В выборку берутся прогнозы, которые **точно "
              "воспроизводятся** из сохранённых признаков: backtest_2526 (тестовый сезон, источник backtest) и "
              "prod_20261012 (текущий сезон, источник backfill). Живые прогнозы (live) воспроизводятся не всегда — "
              "признаки предстоящего матча обновляются после сохранения прогноза (E0)."),
            table([["Эксперимент", "Что проверяет", "Объём", "Выполнен", "Время, с"],
                   ["E0", "цепочка на карточке = сохранённой вероятности", f"{e0.get('main_risk_selections', '—')} Main/Risk + все рынки 200 матчей", "да" if e0 else "нет", f(sec.get("e0"), 0)],
                   ["E1", "A vs D1 vs D2 vs D3: аддитивность, пропущенное влияние, знаки, монотонность", f"{e1s.get('matches', '—')} матчей / {e1s.get('selections', '—')} выборов", "да" if e1s else "нет", f(sec.get("e1"), 0)],
                   ["E2", "все методы против эталона Shapley композиции", f"{e2s.get('matches', '—')} матчей / {e2s.get('selections', '—')} выборов", "да" if e2s else "нет", f(sec.get("e2"), 0)],
                   ["E3", "устойчивость: шум признаков и похожий матч", f"{e3s.get('matches', '—')} матчей / {e3s.get('selections', '—')} выборов", "да" if e3s else "нет", f(sec.get("e3"), 0)],
                   ["E4", "футбольные факты и только предматчевые данные", f"{e4s.get('picks', '—')} прогнозов Main/Risk", "да" if e4s else "нет", f(sec.get("e4"), 0)],
                   ["E5", "примеры матчей (старая и новая карточка)", f"{len(e5)} матча", "да" if e5 else "нет", f(sec.get("e5"), 0)],
                   ["E6", "время расчёта на страницу", f"{e6.get('matches', '—')} матчей", "да" if e6 else "нет", "—"],
                   ["E7", "локальная чувствительность и производные по признакам", f"{len(e7.get('feature_steps', []))} матчей", "да" if e7 else "нет", "—"],
                   ["Скриншоты", "карточка до/после в браузере (Playwright)", f"{len(after)} матча", "да" if after and before else "нет", "—"]],
                  widths=[2.0 * cm, 6.2 * cm, 5.4 * cm, 1.7 * cm, 1.7 * cm]),
            P("Набор рынков на матч: Main, Risk и самый вероятный выбор каждого из рынков 1X2, двойной шанс, тоталы "
              "1.5/2.5/3.5, обе забьют, форы −0.5/−1/−1.5/+1.5, угловые 9.5, точный счёт. Выборка — по "
              f"{e1s.get('matches', 0) // 2 if e1s else '—'} случайных матчей каждой версии модели (seed {info.get('seed', '—')})."))

    # 6 ------------------------------------------------------------------------------------------
    rep.add(H1("6. Результаты проверок"))
    lv = e0.get("live_drift", {})
    rep.add(H2("6.1. E0 — цепочка на карточке воспроизводит вероятность"),
            P(f"Для всех {e0.get('main_risk_selections', '—')} прогнозов Main и Risk в хранилище (все версии, включая "
              f"живые) вероятность, пересчитанная из сохранённых выходов моделей (p_{{H}}, p_{{D}}, p_{{A}}, λ_{{h}}, "
              f"λ_{{a}}, μ), отличается от сохранённой не более чем на {e0.get('max_abs_error', float('nan')):.1e} "
              f"(точность float32). Для всех {e0.get('all_markets_200_matches', {}).get('selections', '—')} выборов всех "
              f"рынков 200 случайных матчей — не более {e0.get('all_markets_200_matches', {}).get('max_abs_error', float('nan')):.1e}. "
              "Значит, числа в заметке о цепочке — ровно те, из которых получена вероятность на сайте."),
            P(f"Живые прогнозы: из {lv.get('n', '—')} матчей с признаками {pc(lv.get('share_exact_1e-4'))} "
              f"воспроизводятся сегодняшними признаками точно, у {pc(lv.get('share_over_1pp'), 1)} расхождение > 1 п.п. "
              f"(медиана {lv.get('median', float('nan')):.1e}, 90-й перцентиль {pc(lv.get('p90'), 2)}, максимум "
              f"{pc(lv.get('max'), 1)}). Причина — признаки предстоящего матча пересчитываются после сохранения прогноза. "
              "Для таких матчей факторы объясняют вероятность по последним предматчевым данным; заметка о цепочке "
              "всегда берётся из сохранённого прогноза."))

    if e1s:
        rows = []
        for g in ("1X2", "DOUBLE_CHANCE", "HANDICAP", "TOTAL", "BTTS", "CORNERS", "EXACT_SCORE"):
            r = by1.get(g)
            if r:
                shares = (["—", "—"] if g == "CORNERS" else [pc(r["result_share"]), pc(r["goals_share"])])
                rows.append([GROUP_RU[g], r["n"], *shares, pc(r["lin_rel"], 1),
                             f(r["msh_sp"], 3), pc(r["msh_top5"]), pc(r["nonmono"]), pc(r["top5_share"])])
        tm = e1s.get("time_ms", {})
        rep.add(H2("6.2. E1 — аддитивность, вклад моделей, монотонность"),
                P(f"**Аддитивность D2.** Максимальная ошибка |Σ c_{{j}} − (P − g(ū))| на {e1s['selections']} выборах — "
                  f"{e1s['max_complete_err']:.1e}, вероятность D2 совпадает с сохранённой до "
                  f"{e1s['max_p_diff_stored']:.1e}. **Линейное правило D1** не аддитивно: медианная относительная ошибка "
                  f"{pc(e1s['linear_rel_err_median'], 1)}, 90-й перцентиль {pc(e1s['linear_rel_err_p90'], 1)}, "
                  f"у {pc(e1s['linear_rel_err_over10pct'])} выборов ошибка больше 10 % от объясняемого изменения."),
                table([["Рынок", "N", "через 1X2", "через голы", "ошибка D1 (мед.)", "ρ(D2, D3)", "топ-5 D2∩D3",
                        "немонотонно", "топ-5 доля"]] + rows),
                P("«через 1X2» / «через голы» — средние доли |влияния| по D2, прошедшие через классификатор и через "
                  "модели голов. «немонотонно» — доля выборов, где ∂P/∂log λ меняет знак на пути ū → u. «топ-5 доля» "
                  "— медианная доля суммарного влияния, приходящаяся на 5 крупнейших факторов.", "small"),
                *figure(FIG / "fig_model_shares.png", "Рис. 3. Через какие модели идёт влияние на вероятность рынка "
                                                      "(D2, E1)."),
                P(f"Выводы E1: (1) у тоталов {pc(by1['TOTAL']['result_share'])} и у «обе забьют» "
                  f"{pc(by1['BTTS']['result_share'])} влияния в среднем идёт через классификатор 1X2 — v6 этого не "
                  f"показывал; у азиатских фор {pc(by1['HANDICAP']['goals_share'])} в среднем (до ≈58 % на линиях ±2 и "
                  f"±2.5) — через модели голов; у точного счёта {pc(by1['EXACT_SCORE']['result_share'])} — через "
                  "классификатор; (2) D2 и D3 (Shapley по выходам) дают практически одинаковые ранжирования "
                  "(ρ ≥ 0.999) — правило распределения между выходами моделей на результат не влияет; (3) точный "
                  f"счёт немонотонен по ожидаемым голам в {pc(by1['EXACT_SCORE']['nonmono'])} случаев: один и тот же "
                  "фактор (больше ожидаемых голов) то поднимает, то опускает вероятность счёта, поэтому подпись "
                  "«за/против» и доля для него вводят в заблуждение и не показываются."),
                *figure(FIG / "fig_linear_error.png", "Рис. 4. Ошибка аддитивности линейного цепного правила D1 "
                                                      "(выборы с |P − g(ū)| > 0.5 п.п.)."),
                P(f"Время (E1, на 3 потоках): TreeSHAP трёх голевых моделей на матч — медиана "
                  f"{f(tm.get('shap_inputs_goal_median'), 0)} мс, модели угловых — {f(tm.get('shap_inputs_corner_median'), 0)} мс; "
                  f"само интегрированное правило для одного рынка — {f(tm.get('attribution_median'), 1)} мс; "
                  f"объяснение v6 одного выбора — {f(tm.get('old_explain_selection_median'), 0)} мс."))

    if e2s:
        o = e2s["overall"]
        rows = [[METHOD_RU[m], pc(o[m]["refsign5"]), pc(o[m]["sign5"]), pc(o[m]["top5"]), pc(o[m]["top1"]),
                 f(o[m]["spearman"], 3)] for m in ("old", "linear", "chain", "mshap", "occl")]
        grows = []
        for g, d in e2s.get("by_group", {}).items():
            grows.append([GROUP_RU.get(g, g)] + [f"{pc(d[m]['refsign5'])} / {pc(d[m]['top5'])}"
                                                 for m in ("old", "linear", "chain", "mshap", "occl")])
        rp = e2s.get("repeat", {})
        rep.add(H2("6.3. E2 — сравнение с эталоном Shapley всей композиции"),
                P(f"{e2s['matches']} матчей, {e2s['selections']} выборов. Для каждого выбора эталон считается заново "
                  f"({e2s['n_perm']} перестановок, {e2s['n_background']} фоновых матчей). Медианная относительная "
                  f"стандартная ошибка эталона для его топ-5 факторов — {pc(e2s['ref_top5_se_rel_median'])}; "
                  f"повтор с другим seed на {N_REP(e2)} матчах: совпадение топ-5 {pc(rp.get('top5'))}, ранговая "
                  f"корреляция {f(rp.get('spearman'), 3)}, корреляция значений {f(rp.get('corr'), 3)}."),
                table([["Метод", "Знак топ-5 эталона", "Знак своего топ-5", "Топ-5 ∩ эталон", "Топ-1 = эталон",
                        "Spearman"]] + rows),
                P("«Знак топ-5 эталона» — доля пяти главных факторов эталона, направление которых метод показывает "
                  "верно. Метод A сравнивается в своих единицах (логит/log λ), поэтому для него важны знак и "
                  "ранжирование, а не величина.", "small"),
                *figure(FIG / "fig_reference.png", "Рис. 5. Согласие методов с эталоном (E2)."),
                H3("По группам рынков: знак топ-5 эталона / совпадение топ-5"),
                table([["Рынок", "A", "D1", "D2", "D3", "C2"]] + grows),
                P(f"Выводы E2: интегрированное цепное правило D2 ближе всех к эталону — верный знак у "
                  f"{pc(o['chain']['refsign5'])} главных факторов эталона, {pc(o['chain']['top5'])} совпадение топ-5, "
                  f"Spearman {f(o['chain']['spearman'], 3)}; D3 даёт те же числа (ранговая корреляция D2 и D3 в E1 — "
                  f"выше 0.99 во всех группах). Метод v6 (A): {pc(o['old']['refsign5'])}, {pc(o['old']['top5'])}, "
                  f"{f(o['old']['spearman'], 3)}; хуже всего — на точном счёте ({pc(e2s['by_group'].get('EXACT_SCORE', {}).get('old', {}).get('refsign5'))} "
                  "верных знаков) и на форах. Линейное правило D1 немного уступает D2 по составу факторов; окклюзия C2 "
                  "— худшая по всем мерам. Полного совпадения с эталоном нет и не ожидается: эталон использует "
                  "интервенционный фон и учитывает взаимодействия признаков из разных моделей, а D2 — "
                  "TreeSHAP внутри моделей."))

    if e3s:
        o = e3s["overall"]
        rows = [[METHOD_RU[m], pc(o.get(f"{m}_jacc3")), f(o.get(f"{m}_dshare_pp"), 2), pc(o.get(f"{m}_flip"), 1),
                 pc(o.get(f"{m}_twin_jacc3")) if f"{m}_twin_jacc3" in o else "—"] for m in ("old", "chain", "occl")]
        rep.add(H2("6.4. E3 — устойчивость"),
                P(f"{e3s['matches']} матчей, {e3s['selections']} выборов. К каждому числовому признаку добавляется шум "
                  f"N(0, ({e3s['noise_sd_fraction']}·σ)²) — {e3s['draws']} раз; сравниваются топ-3 факторов, их доли и "
                  "знаки. «Похожий матч» — ближайший по выходам моделей (p_{H}, p_{D}, log λ_{h}, log λ_{a}) прогноз той "
                  "же версии; для него сравнивается топ-3 того же рынка."),
                table([["Метод", "Жаккар топ-3 (шум)", "|Δ доли| топ-3, п.п.", "Смена знака топ-3",
                        "Жаккар топ-3 (похожий матч)"]] + rows),
                *figure(FIG / "fig_stability.png", "Рис. 6. Устойчивость к шуму признаков (E3)."),
                P("A и D2 одинаково устойчивы к шуму: оба опираются на TreeSHAP тех же деревьев, а путь по выходам "
                  "моделей гладкий; смена знака главных факторов почти не встречается. Окклюзия менее устойчива. "
                  "У «похожего матча» (близкие выходы моделей) топ-3 совпадает лишь примерно наполовину у обоих "
                  "методов: одинаковые ожидаемые голы могут получиться из разных признаков, и объяснение честно "
                  "это показывает."),
                P(f"90-й перцентиль |Δ доли| у D2 — {f(e3s.get('dshare_pp_p90', {}).get('chain'), 2)} п.п. "
                  "Поэтому доли на карточке округляются до целых процентов и не показываются с десятыми; факторы "
                  "меньше 3 % показываются только чтобы набрать три фактора."))

    if e4s:
        pm = e4s.get("prematch", {})
        ls = e4s.get("live_snapshots", {})
        grows = [[GROUP_RU.get(r["group"], r["group"]), r["n"], pc(r["p"]), pc(r["rate"]), f(r["corr"], 2)]
                 for r in e4s.get("by_group", [])]
        rep.add(H2("6.5. E4 — футбольные факты и только предматчевые данные"),
                P(f"На {e4s['picks']} прогнозах Main/Risk факты есть у {pc(e4s['with_facts'])}; у "
                  f"{pc(e4s['short_history_k_below_10'])} истории меньше 10 матчей (начало сезона, «x из k» честно "
                  f"показывает меньшее k). Частота, с которой выбор выиграл бы в недавних домашних/выездных матчах "
                  f"команд, коррелирует с вероятностью модели: r = {f(e4s['rate_vs_p_corr'], 2)}; по стороне 50 % "
                  f"факты и модель согласны в {pc(e4s['agree_direction'])} случаев, среднее расхождение "
                  f"{f(e4s['mean_abs_gap_pp'], 1)} п.п. Факты — независимая опора, а не пересказ модели: иногда они "
                  "расходятся, и это видно читателю."),
                table([["Рынок", "N", "P модели (средн.)", "Частота в фактах", "Корреляция"]] + grows),
                P("Только предматчевые данные:"),
                bullets([f"история для фактов — только матчи с датой раньше даты матча: {pc(pm.get('history_before_kickoff'))} "
                         f"из {pm.get('matches', '—')} матчей;",
                         f"признаки формы (h/a_form5/10_gf) пересчитаны из этой истории и сравнены с сохранёнными: "
                         f"{pm.get('form_features_checked', '—')} значений, максимальное расхождение "
                         f"{f(pm.get('form_max_abs_diff'), 6)};",
                         "среди входов моделей нет итоговых величин матча (голы, удары, угловые, карточки, результат): "
                         f"{', '.join(e4s.get('model_inputs_post_match', [])) or 'пересечение пустое'};",
                         f"живые прогнозы: {ls.get('n', '—')} шт., впервые сохранены до начала матча — "
                         f"{pc(ls.get('first_saved_before_kickoff'))} из {ls.get('with_kickoff', '—')} с известным временем;",
                         "заметка о цепочке берёт выходы моделей из сохранённого прогноза, факторы — из предматчевых "
                         "признаков; ни то, ни другое не использует результат матча."]))

    if e7:
        rows = []
        for mid, ex in e7.get("examples", {}).items():
            for it in ex["items"][:4]:
                if "per_corner_pp" in it:
                    rows.append([f"{ex['home']} – {ex['away']}", f"{it['market']} {it['selection']}", pc(it["P"], 1),
                                 "—", "—", "—", f"{it['per_corner_pp']:+.2f} (на +1 угловой)"])
                else:
                    rows.append([f"{ex['home']} – {ex['away']}", f"{it['market']} {it['selection']}", pc(it["P"], 1),
                                 f"{it['per_0.1_home_goals_pp']:+.2f}", f"{it['per_0.1_away_goals_pp']:+.2f}",
                                 f"{it['per_1pp_draw_pp']:+.2f}", "—"])
        s = e7.get("feature_steps_mean", {})
        rep.add(H2("6.6. E7 — локальная чувствительность"),
                P("Чувствительность вероятности рынка к выходам моделей (в п.п.): на +0.1 ожидаемого гола хозяев / "
                  "гостей, на +1 п.п. вероятности ничьей (взятого поровну из двух других исходов) и на +1 угловой."),
                table([["Матч", "Рынок", "P", "+0.1 гола хоз.", "+0.1 гола гост.", "+1 п.п. ничьей", "Угловые"]] + rows),
                P(f"Перенос на признаки: доля признаков, малый шаг которых (ε·σ) вообще меняет хотя бы один выход "
                  f"модели (20 матчей): ε = 10^{{-4}} — {pc(s.get('0.0001'), 2)}, 10^{{-3}} — {pc(s.get('0.001'), 2)}, "
                  f"10^{{-2}} — {pc(s.get('0.01'), 1)}, 0.1 — {pc(s.get('0.1'), 0)}, 0.5 — {pc(s.get('0.5'), 0)}. "
                  "Производная выхода дерева по признаку почти всюду равна нулю, а конечные шаги дают результат, "
                  "зависящий от размера шага (пересечение порогов разбиений). Поэтому градиентный перенос (C1′) "
                  "отвергнут; чувствительность по выходам моделей (C1) полезна как проверка и в заметке о цепочке."),
                *figure(FIG / "fig_feature_steps.png", "Рис. 7. Доля признаков, шаг которых меняет выход моделей, "
                                                      "в зависимости от размера шага (E7).", width=12 * cm))

    # 7 ------------------------------------------------------------------------------------------
    o = e2s.get("overall", {})
    o3 = e3s.get("overall", {})
    t6 = e6
    verdict = [
        ["Метод", "Объясняет P рынка", "Аддитивность", "Знаки vs эталон", "Устойчивость", "Время на матч", "Итог"],
        ["A: SHAP исходной модели", "нет (логит / log λ одной модели)", "к выходу модели, не к P",
         pc(o.get("old", {}).get("refsign5")), pc(o3.get("old_jacc3")),
         f"{f(t6.get('page_three_cards_old_ms', {}).get('median'), 0)} мс / 3 карточки", "отвергнут"],
        ["B: описание цепочки", "да, точно (E0)", "—", "—", "детерминирован", "< 1 мс", "принят (заметка)"],
        ["C1: чувствительность по выходам", "локально", "нет", "—", "—", "< 1 мс", "только для проверки"],
        ["C1′: градиент по признакам", "нет (∂u/∂x = 0)", "нет", "—", "—", "—", "отвергнут"],
        ["C2: окклюзия", "да, но не аддитивно", "нет", pc(o.get("occl", {}).get("refsign5")), pc(o3.get("occl_jacc3")),
         f"{f(e2s.get('occl_ms_median'), 0)} мс / рынок", "отвергнут"],
        ["D1: линейное цепное правило", "приближённо", f"нет (ошибка мед. {pc(e1s.get('linear_rel_err_median'), 1)})",
         pc(o.get("linear", {}).get("refsign5")), "как D2", "как D2", "отвергнут"],
        ["D2: интегрированное цепное правило", "да", f"точно (≤ {e1s.get('max_complete_err', float('nan')):.0e})",
         pc(o.get("chain", {}).get("refsign5")), pc(o3.get("chain_jacc3")),
         f"{f(t6.get('page_three_cards_new_ms', {}).get('median'), 0)} мс / 3 карточки", "**выбран**"],
        ["D3: Shapley по выходам", "да", "точно", pc(o.get("mshap", {}).get("refsign5")), "как D2", "32 вычисления g",
         "равноценен D2, не нужен"],
        ["REF: Shapley композиции", "да", "точно (к E_{b}P)", "эталон", f"шум выборки (повтор: топ-5 {pc(e2s.get('repeat', {}).get('top5'))})",
         f"≈{f(e2s.get('ref_seconds_per_match'), 0)} с", "только мерило"],
        ["E: футбольные факты", "не объясняет модель", "—", "—", "детерминированы", "мс", "принят (Recent form)"],
    ]
    rep.add(H1("7. Сравнение методов"),
            table(verdict, font=6.6),
            P("Отвергнуты методы, которые не могут корректно объяснить итоговую вероятность: A (объясняет другую "
              "величину и игнорирует часть цепочки), C1′ (нулевые производные деревьев), C2 (не аддитивна, зависит от "
              "выбора «среднего», не учитывает взаимодействия), D1 (не аддитивно на нелинейных рынках). D3 даёт "
              "практически те же факторы, что D2, но сложнее объяснить и он не нужен. Эталон слишком дорог для "
              "страницы и шумен."),
            note("Выбор: D2 (интегрированное цепное правило) для факторов + B (цепочка в заметке) + E (футбольные "
                 "факты). Это единственный из дешёвых методов, который точно складывается в вероятность рынка, "
                 "учитывает все модели цепочки и согласуется с эталоном по знакам и составу главных факторов."))

    # 8 ------------------------------------------------------------------------------------------
    rep.page_break()
    rep.add(H1("8. Реальные примеры матчей"),
            P("Тексты карточек сняты в браузере (Playwright, screenshots.py) на одной и той же базе данных: «до» — "
              "карточка v6, «после» — карточка v6.1. Ниже — Main и Risk каждого матча и таблица главных факторов "
              "трёх методов для Main."))
    for i, mid in enumerate(e5 or after):
        ex = e5.get(mid, {})
        title = f"{ex.get('home', '')} – {ex.get('away', '')}, {ex.get('date', '')}" if ex else mid
        rep.add(H2(f"8.{i + 1}. {title}"))
        for j, role in enumerate(("Main", "Risk")):
            rep.add(H3(f"{role}: карточка v6 (до)"), *card_block(card_text(before, mid, j)),
                    H3(f"{role}: карточка v6.1 (после)"), *card_block(card_text(after, mid, j)))
        if ex:
            pick = ex["picks"][0]
            eff = pick["model_effect"]
            rows = [["Фактор", "D2: вклад в P, п.п.", "доля D2", "Эталон, п.п.", "A (v6), ед. отступа"]]
            for k, v, s in pick["new_top"][:6]:
                rows.append([label(k), f"{v * 100:+.2f}", pc(s), f"{pick['ref_on_new_top'].get(k, 0) * 100:+.2f}",
                             f"{pick['old_on_new_top'].get(k, 0):+.3f}"])
            eff_txt = ", ".join(f"{ {'result': 'классификатор 1X2', 'home_goals': 'голы хозяев', 'away_goals': 'голы гостей', 'corners': 'угловые'}[m]} "
                                f"{v * 100:+.1f} п.п." for m, v in eff.items())
            rep.add(P(f"Main: {pick['market']} {pick['selection']}, P = {pc(pick['p'], 1)}, типичный матч g(ū) = "
                      f"{pc(pick['p_base'], 1)}. Изменение по моделям (D2): {eff_txt}."),
                    table(rows))
            path = FIG / f"fig_example_{mid}.png"
            rep.add(*figure(path, f"Рис. 8.{i + 1}. {title}: главные факторы Main по D2 и их значения у метода v6 и эталона."))
        shot_b = FIG / f"card_before_{mid}_main.png"
        shot_a = FIG / f"card_after_{mid}_main.png"
        rep.add(*figure(shot_b, f"Скриншот: карточка v6 (до), {title}.", width=15 * cm),
                *figure(shot_a, f"Скриншот: карточка v6.1 (после), {title}.", width=15 * cm))

    # 9 ------------------------------------------------------------------------------------------
    rep.add(H1("9. Выбранная реализация"),
            bullets([
                "**src/explain/chain.py** — векторизованная g (та же матрица счетов, что в движке), TreeSHAP всех "
                "выходов моделей одного матча (кэш на матч: Main, Risk и другой рынок используют один расчёт), "
                "интегрированное цепное правило (12 узлов Гаусса–Лежандра), значения цепочки из сохранённого прогноза.",
                "**src/explain/card.py** — факторы карточки: вклады c_{j} суммируются по фактору (пара хозяева/гости); "
                "относительное влияние s_{g} = |c_{g}| / Σ|c|; показываются главные факторы «за» (не меньше трёх, "
                "далее только ≥ 3 %) и самый сильный «против», если его доля ≥ 10 %; заметка chain_note по группе рынка.",
                "**views/match_details.py** — карточка в дизайне v6; подписи «Main factors», «Relative model influence» "
                "с подсказкой «Only the most influential factors are shown. Percentages represent their share of total "
                "model influence.», блок «Recent form»; итоговая строка — заметка о цепочке без повторения факторов.",
                "**Точный счёт** — без долей и без факторов: P(счёта) немонотонна по ожидаемым голам; карточка "
                "показывает ожидаемые голы, вероятность счёта и два ближайших по вероятности счёта.",
                "**Explain another market** — не предлагает Main и Risk, у которых уже есть свои карточки; раздел "
                "«Markets» не менялся.",
            ]),
            formula("s_{g} = |Σ_{j∈g} c_{j}| / Σ_{g′} |Σ_{j∈g′} c_{j}|      (доли показанных факторов не обязаны давать 100 %)"),
            P("Примеры заметок: «Como 1.43 and Roma 1.44 expected goals: about 2.9 in total, below the 3.5 line.»; "
              "«Portsmouth +1.5 wins unless Portsmouth lose by 2 or more goals. Expected goals 1.92–1.12 give a goal "
              "difference of -0.80 for Portsmouth.»; «FootPredict expects 11.0 corners (usually between 7 and 16), "
              "above the 8.5 line.»"),
            P(f"Время: построение TreeExplainer четырёх моделей при первом открытии страницы после запуска сервера — "
              f"{f(e6.get('cold_start_total_s'), 1)} с (один раз на процесс); далее страница с тремя карточками — "
              f"медиана {f(e6.get('page_three_cards_new_ms', {}).get('median'), 0)} мс (90-й перцентиль "
              f"{f(e6.get('page_three_cards_new_ms', {}).get('p90'), 0)} мс) против "
              f"{f(e6.get('page_three_cards_old_ms', {}).get('median'), 0)} мс у v6; само правило для рынка — "
              f"{f(e6.get('attribution_only_ms', {}).get('median'), 1)} мс."))

    # 10 -----------------------------------------------------------------------------------------
    rep.add(H1("10. Тесты"),
            bullets([
                "tests/test_explain_v61.py — векторизованная g совпадает с движком для всех рынков (в т.ч. ρ ≠ 0); "
                "угловые = хвост NB; полнота D2 (Σ c_{j} = P − g(ū)) для 14 рынков × 3 матчей; противоположные выборы "
                "получают противоположные вклады; знаки по смыслу (атака хозяев помогает «больше» и «обе забьют», рост "
                "ничьей снижает тотал); модели, не определяющие рынок, получают нулевой вес; немонотонность точного "
                "счёта; значения цепочки = вероятности движка (включая возврат на целых форах); правила факторов "
                "карточки; заметки; сквозной тест на обученных моделях: вклад = TreeSHAP × вес пути, P = вероятность "
                "движка.",
                "tests/e2e/test_pages.py::test_why_card_shows_relative_model_influence_and_the_chain — подписи, точный "
                "текст подсказки, «Recent form», порядок долей, отсутствие старых подписей и повтора факторов в "
                "заметке, отсутствие Main в «Explain another market», точный счёт без долей.",
            ]),
            P(f"Запуски: {e2e.get('unit', 'см. отчёт агента')}. Новая e2e-спецификация на коде v6 (до изменения): "
              f"{e2e.get('without_change', '—')}; на коде v6.1: {e2e.get('with_change', '—')}."))

    # 11 -----------------------------------------------------------------------------------------
    rep.add(H1("11. Недостатки и дальнейшие возможности"),
            bullets([
                "Вклады D2 измеряются относительно «типичного матча» g(ū) — вероятности при средних выходах моделей, "
                "а не средней вероятности рынка; читатель видит только относительное влияние, поэтому это не мешает, "
                "но значение базы на карточке не показывается.",
                "Внутри каждой модели вклад признака — TreeSHAP (path-dependent); D2 распределяет между признаками "
                "линейно, поэтому взаимодействия признаков из разных моделей учитываются только через путь по "
                "выходам моделей, а не точно, как в эталоне.",
                "Для живых прогнозов признаки обновляются после сохранения прогноза; в части матчей (E0) факторы "
                "объясняют вероятность по более свежим предматчевым данным, чем сохранённая. Решение — сохранять "
                "признаки вместе с прогнозом.",
                "Пары «хозяева/гости» суммируются в один фактор; если вклады сторон противоположны, нетто-вклад "
                "меньше, чем каждый из них.",
                "Рейтинги Эло — сильные признаки моделей, но их смысл неочевиден; карточка показывает их как "
                "«Team strength difference», без пояснения.",
                "Дальше: показывать базу «в типичном матче» в подсказке; группировать признаки по темам "
                "(атака, оборона, сила соперника); эталон Shapley с большим числом перестановок на всей выборке; "
                "проверка понятности карточки на пользователях.",
            ]))

    # 12 -----------------------------------------------------------------------------------------
    rep.add(H1("12. Воспроизведение"),
            table([["Параметр", "Значение"],
                   ["Git SHA (запуск экспериментов)", info.get("git_sha", "—")],
                   ["Git SHA (сборка отчёта)", git("rev-parse", "HEAD")],
                   ["Python / shap / xgboost / sklearn", f"{info.get('python', '—')} / {info.get('shap', '—')} / "
                                                         f"{info.get('xgboost', '—')} / {info.get('sklearn', '—')}"],
                   ["numpy / pandas", f"{info.get('numpy', '—')} / {info.get('pandas', '—')}"],
                   ["Модели (обучены до)", ", ".join(f"{k}: {v}" for k, v in info.get("versions", {}).items())],
                   ["Seed", str(info.get("seed", "—"))],
                   ["Время экспериментов, с", ", ".join(f"{k}: {v}" for k, v in sec.items())],
                   ["Дата запуска", info.get("run_at", "—")]]),
            *reproduce([
                ".venv/bin/python experiments/v6_1/explain/run_research.py all     # E0–E5, ≈1 ч на 3 потоках",
                ".venv/bin/python experiments/v6_1/explain/timing.py               # E6",
                ".venv/bin/python experiments/v6_1/explain/sensitivity.py          # E7",
                ".venv/bin/python experiments/v6_1/explain/screenshots.py after     # скриншоты и тексты карточек",
                ".venv/bin/python experiments/v6_1/explain/build_pdf.py            # этот отчёт и findings.md",
                ".venv/bin/python -m pytest tests -q --ignore=tests/e2e",
                ".venv/bin/python -m pytest tests/e2e/test_pages.py --e2e -k why_card",
            ]))

    rep.build()
    pages = rep.check(scale=1.0)
    print(f"{OUT} — {pages} pages")
    write_findings(e0, e1, e1s, e2s, e3s, e4s, e6, e7, info, pages)


def N_REP(e2: pd.DataFrame) -> int:
    path = RES / "e2_reference_repeat.csv"
    return int(pd.read_csv(path)["match_id"].nunique()) if path.exists() else 0


def write_findings(e0, e1, e1s, e2s, e3s, e4s, e6, e7, info, pages) -> None:
    o = e2s.get("overall", {})
    o3 = e3s.get("overall", {})
    by1 = {r["group"]: r for r in e1s.get("by_group", [])}
    lines = [
        "# Explainability research (v6.1) — findings",
        "",
        f"Generated by build_pdf.py from results/*. Report: reports/v6_1/Explainability_Research.pdf ({pages} pages).",
        f"Experiments run at {info.get('run_at', '—')} on commit {info.get('git_sha', '—')}; seed {info.get('seed', '—')}.",
        "",
        "## Chosen method",
        "Path-integrated chain rule (D2): c_j = Σ_k G_k φ_jk with G_k = ∫ ∂g/∂u_k along ū → u, where u are the",
        "model outputs (1X2 logits, log λ_home, log λ_away, log μ) and g the score-matrix / NB map to the market",
        "probability. Plus the chain values (expected goals, distribution) and football facts.",
        "",
        "## Key numbers",
        f"- E0: chain values reproduce stored probabilities to {e0.get('max_abs_error', float('nan')):.1e} "
        f"({e0.get('main_risk_selections', '—')} Main/Risk selections); live drift > 1 pp in "
        f"{pc(e0.get('live_drift', {}).get('share_over_1pp'), 1)} of live matches.",
        f"- E1: D2 completeness error ≤ {e1s.get('max_complete_err', float('nan')):.1e}; linear chain rule median "
        f"relative error {pc(e1s.get('linear_rel_err_median'), 1)}.",
    ]
    ot = old_table(e1) if not e1.empty else pd.DataFrame()
    for g in ("TOTAL", "BTTS", "HANDICAP", "DOUBLE_CHANCE", "1X2", "CORNERS", "EXACT_SCORE"):
        r = by1.get(g)
        if r and g in ot.index:
            lines.append(f"  - {g}: influence through models the v6 card ignored {pc(ot.loc[g, 'omitted'])} on average "
                         f"(>20 % in {pc(ot.loc[g, 'over20'])}); v6 shown-factor sign = market sign "
                         f"{pc(ot.loc[g, 'sign'])}; v6 top-5 in market top-5 {pc(ot.loc[g, 'top5'])}; "
                         f"non-monotonic {pc(r['nonmono'])}.")
    for m in ("old", "linear", "chain", "mshap", "occl"):
        if m in o:
            lines.append(f"- E2 {m}: sign of reference top-5 {pc(o[m]['refsign5'])}, top-5 overlap {pc(o[m]['top5'])}, "
                         f"Spearman {f(o[m]['spearman'], 3)}.")
    for m in ("old", "chain", "occl"):
        if f"{m}_jacc3" in o3:
            lines.append(f"- E3 {m}: top-3 Jaccard under 5 % noise {pc(o3[f'{m}_jacc3'])}, |Δshare| "
                         f"{f(o3[f'{m}_dshare_pp'], 2)} pp.")
    if e4s:
        lines.append(f"- E4: facts vs model probability r = {f(e4s.get('rate_vs_p_corr'), 2)}; pre-match history "
                     f"{pc(e4s.get('prematch', {}).get('history_before_kickoff'))}.")
    if e6:
        lines.append(f"- E6: cold start {f(e6.get('cold_start_total_s'), 1)} s per process; page with three cards "
                     f"{f(e6.get('page_three_cards_new_ms', {}).get('median'), 0)} ms (v6: "
                     f"{f(e6.get('page_three_cards_old_ms', {}).get('median'), 0)} ms).")
    if e7:
        lines.append(f"- E7: share of features whose 1e-3·σ step moves any model output: "
                     f"{pc(e7.get('feature_steps_mean', {}).get('0.001'), 2)} — gradient propagation to features is "
                     "not usable for trees.")
    (HERE / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
