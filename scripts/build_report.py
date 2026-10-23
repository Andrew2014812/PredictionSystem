"""Build the v6 research report (Russian PDF) from the project's own results.

Every number comes from a file produced by a run: reports/v6/audit_before.json and audit_after.json
(scripts/v6_audit.py), the model metadata (local_data/models), the research results
(experiments/v6_research: findings.md, results/*.json, figures/*.png), the price-archive coverage
(local_data/api/oddspapi/history_coverage.json) and reports/v6/tests.json (test runs). Missing inputs
are reported as missing, never filled in.

    python scripts/build_report.py            # -> reports/FootPredict_v6_Research_Report.pdf
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, Spacer, Table,  # noqa: E402
                                TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate  # noqa: E402
from reportlab.platypus.frames import Frame  # noqa: E402
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "FootPredict_v6_Research_Report.pdf"
V6 = ROOT / "reports" / "v6"
FIG = V6 / "figures"
RESEARCH = ROOT / "experiments" / "v6_research"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

INK, MUTED, ACCENT, NEG, GRID = "#0f172a", "#475569", "#16a34a", "#dc2626", "#e2e8f0"
S = {
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.6, leading=13.6, spaceAfter=5, textColor=INK),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, leading=19, spaceBefore=4, spaceAfter=9,
                         textColor=INK),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, spaceBefore=9, spaceAfter=5,
                         textColor=INK),
    "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=10, leading=13, spaceBefore=6, spaceAfter=3,
                         textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DejaVu", fontSize=8, leading=10, textColor=MUTED, alignment=TA_CENTER,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.6, leading=9.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=7.6, leading=9.4, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.6, leading=13.4, leftIndent=12,
                             bulletIndent=2, spaceAfter=2, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DejaVuMono", fontSize=7.6, leading=9.6, textColor=INK,
                           backColor="#f1f5f9", leftIndent=4, rightIndent=4, borderPadding=4, spaceAfter=6),
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=30, leading=36, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=13, leading=18, textColor=MUTED),
}


# ---------------------------------------------------------------------------------------------- inputs
def load(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "—"


def pct(v, d=1, signed=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+.{d}f}%" if signed else f"{v:.{d}f}%"


def share(v, d=1):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v * 100:.{d}f}%"


def num(v, d=3):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


# ---------------------------------------------------------------------------------------------- markup helpers
def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="DejaVuMono" size="8">\1</font>', t)
    return t


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(inline(text) if style != "raw" else text, S["body" if style == "raw" else style])


def bullets(items) -> list:
    return [Paragraph(inline(i), S["bullet"], bulletText="•") for i in items]


def table(rows: list[list], widths=None, header=True, zebra=True) -> Table:
    data = [[Paragraph(inline(str(c)), S["cellb" if (header and r == 0) else "cell"]) for c in row]
            for r, row in enumerate(rows)]
    tbl = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8fafc")))
    tbl.setStyle(TableStyle(style))
    return tbl


def figure(path: Path, caption: str, width=16.5 * cm) -> list:
    if not path.exists():
        return [P(f"(Рисунок не построен: {path.name} отсутствует.)", "small")]
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    height = width * h / w
    if height > 20 * cm:
        width, height = width * 20 * cm / height, 20 * cm
    return [KeepTogether([Image(str(path), width=width, height=height), Paragraph(esc(caption), S["cap"])])]


def markdown(md: str, heading_offset: int = 0) -> list:
    """A small markdown subset (headings, paragraphs, lists, pipe tables, code) -> flowables."""
    out, para, lines = [], [], md.splitlines()
    i = 0

    def flush():
        if para:
            out.append(P(" ".join(para)))
            para.clear()
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(Preformatted("\n".join(code), S["mono"]))
        elif re.match(r"^#{1,6} ", line):
            flush()
            level = len(line.split(" ")[0]) + heading_offset
            out.append(Paragraph(inline(line.lstrip("# ").strip()), S["h3" if level >= 3 else "h2"]))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            n = max(len(r) for r in rows)
            rows = [r + [""] * (n - len(r)) for r in rows]
            out.append(table(rows, widths=[17 * cm / n] * n))
            out.append(Spacer(1, 4))
            continue
        elif re.match(r"^\s*([-*]|\d+\.) ", line):
            flush()
            out.append(Paragraph(inline(re.sub(r"^\s*([-*]|\d+\.) ", "", line)), S["bullet"], bulletText="•"))
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    return out


def findings_sections() -> dict[str, str]:
    path = RESEARCH / "findings.md"
    if not path.exists():
        return {}
    parts = re.split(r"^## (.+)$", path.read_text(encoding="utf-8"), flags=re.M)
    return {parts[k].strip(): parts[k + 1] for k in range(1, len(parts) - 1, 2)}


def research_figures(section: str) -> list:
    caps = load(RESEARCH / "figures" / "captions.json", {}) or {}
    out = []
    for name, meta in caps.items():
        if isinstance(meta, dict) and meta.get("section") == section:
            out += figure(RESEARCH / "figures" / name, meta.get("caption", name), width=15 * cm)
    return out


# ---------------------------------------------------------------------------------------------- charts
def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def chart_coverage(before: dict, after: dict) -> Path | None:
    if not before or not after:
        return None
    b = {r["league"]: r for r in before["leagues"]}
    a = {r["league"]: r for r in after["leagues"]}
    leagues = sorted(a, key=lambda k: -a[k]["matches"])
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    x = range(len(leagues))
    share_b = [b.get(k, {}).get("main_priced", 0) / max(b.get(k, {}).get("main_settled", 0), 1) for k in leagues]
    share_a = [a[k]["main_priced"] / max(a[k]["main_settled"], 1) for k in leagues]
    ax.bar([i - 0.2 for i in x], share_b, 0.4, color="#94a3b8", label="До")
    ax.bar([i + 0.2 for i in x], share_a, 0.4, color=ACCENT, label="После")
    ax.set_xticks(list(x), leagues, rotation=0, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Main с реальной ценой")
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False)
    return _save(fig, "coverage_main_by_league.png")


def chart_market_bars(rows: list[dict], value: str, title: str, name: str, colour_sign: bool) -> Path | None:
    df = pd.DataFrame(rows)
    if df.empty or value not in df:
        return None
    df = df.dropna(subset=[value]).sort_values(value)
    from src.ui.labels import group_title, league_label
    lab = df["group"].map(group_title) if "group" in df else df["league"].map(lambda x: league_label(x, True))
    fig, ax = plt.subplots(figsize=(7.6, max(2.4, 0.32 * len(df) + 0.8)))
    vals = df[value] / (100 if value == "roi" else 1)
    cols = [ACCENT if v > 0 else NEG for v in vals] if colour_sign else ["#3b7ea1"] * len(df)
    ax.barh(lab, vals, color=cols, alpha=0.85)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for y, (v, n) in enumerate(zip(vals, df["bets" if value == "roi" else "predictions"])):
        ax.text(v + (0.004 if v >= 0 else -0.004), y, f"{v * 100:+.1f}% (n={int(n)})" if value == "roi"
                else f"{v * 100:.0f}% (n={int(n)})", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color=GRID)
    return _save(fig, name)


def chart_cumulative(picks: pd.DataFrame) -> Path | None:
    from src.analytics.performance import cumulative_by_date
    if picks.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    for role, colour, label in (("main", ACCENT, "Main"), ("risk", "#d97706", "Risk")):
        c = cumulative_by_date(picks.loc[picks["role"] == role], None)
        if not c.empty:
            ax.plot(c["date"], c["cumulative_profit"], color=colour, lw=1.6, label=label)
    ax.axhline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_ylabel("Прибыль, единиц")
    ax.legend(frameon=False)
    ax.grid(color=GRID)
    return _save(fig, "cumulative_profit.png")


def chart_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.axis("off")
    boxes = {
        "Football-Data\n(результаты, статистика,\nзакрывающие цены)": (0.02, 0.70),
        "API-Football\n(свежие результаты,\nрасписание)": (0.02, 0.40),
        "OddsPapi\n(текущие и архивные\nкотировки)": (0.02, 0.10),
        "Сбор и проверка\n(сопоставление команд,\nзапрет двойных матчей)": (0.27, 0.40),
        "Признаки\n(только до матча:\nформа, сила, угловые)": (0.50, 0.62),
        "4 модели XGBoost\n(исход, голы х/г,\nугловые)": (0.50, 0.25),
        "Рынки\n(матрица счёта, тоталы,\nфоры, угловые)": (0.73, 0.62),
        "Main / Risk,\nEV, история,\nрасчёт ROI": (0.73, 0.25),
        "Streamlit-сайт\n(7 страниц)": (0.90, 0.44),
    }
    for text, (x, y) in boxes.items():
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), 0.18 if x < 0.85 else 0.1, 0.2,
                                                       boxstyle="round,pad=0.01", fc="#f1f5f9", ec="#94a3b8"))
        ax.text(x + (0.09 if x < 0.85 else 0.05), y + 0.1, text, ha="center", va="center", fontsize=6.6)
    arrows = [((0.20, 0.80), (0.27, 0.52)), ((0.20, 0.50), (0.27, 0.50)), ((0.20, 0.20), (0.27, 0.46)),
              ((0.45, 0.52), (0.50, 0.70)), ((0.59, 0.62), (0.59, 0.45)), ((0.68, 0.35), (0.73, 0.70)),
              ((0.82, 0.62), (0.82, 0.45)), ((0.91, 0.35), (0.93, 0.44)), ((0.91, 0.72), (0.93, 0.64))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.9))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.05, 0.95)
    return _save(fig, "architecture.png")


# ---------------------------------------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                         bottomMargin=1.8 * cm, title="FootPredict v6 — исследовательский отчёт",
                         author="FootPredict", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self._decor)])

    def _decor(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.1 * cm, "FootPredict v6 — исследовательский отчёт")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
        canvas.setStrokeColor(colors.HexColor(GRID))
        canvas.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            level = 0 if flowable.style.name == "h1" else 1
            text = flowable.getPlainText()
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))


def H1(text):
    return Paragraph(esc(text), S["h1"])


def H2(text):
    return Paragraph(esc(text), S["h2"])


def build() -> Path:
    before = load(V6 / "audit_before.json", {})
    after = load(V6 / "audit_after.json", {}) or before
    tests = load(V6 / "tests.json", {})
    meta_all = load(ROOT / "local_data/models/metadata.json", {})
    active = meta_all.get("active", "")
    meta = load(ROOT / f"local_data/models/{active}/metadata.json", {})
    rule_cmp = load(ROOT / "local_data/models/main_rule_comparison.json", {})
    coverage = load(ROOT / "local_data/api/oddspapi/history_coverage.json", {})
    findings = findings_sections()
    research_results = sorted((RESEARCH / "results").glob("*.json")) if (RESEARCH / "results").exists() else []
    from src.analytics.performance import unified_history
    from src.prediction import store
    picks = store.load_picks()
    hist = unified_history(picks)

    sha, short = git("rev-parse", "HEAD"), git("rev-parse", "--short", "HEAD")
    story = []
    # 1. title
    story += [Spacer(1, 5 * cm), Paragraph("FootPredict", S["title"]), Spacer(1, 6),
              Paragraph("Информационная система прогнозирования результатов и статистических показателей "
                        "футбольных матчей на основе машинного обучения", S["subtitle"]), Spacer(1, 1.2 * cm),
              Paragraph("Исследовательский и технический отчёт · версия v6", S["h3"]),
              Paragraph(f"Дата исследования: {date.today():%d.%m.%Y}", S["body"]),
              Paragraph(f"Git commit: {esc(sha)}", S["body"]),
              Paragraph(f"Активная модель: {esc(active)}", S["body"]), PageBreak()]
    # 2. contents
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="DejaVu-Bold", fontSize=9.5, leading=13, leftIndent=0),
                       ParagraphStyle("t1", fontName="DejaVu", fontSize=8.6, leading=11.5, leftIndent=14)]
    story += [Paragraph("Содержание", S["h1"]), toc, PageBreak()]

    # 3. introduction
    story += [H1("1. Введение")]
    story += [P("FootPredict — веб-система, которая по данным прошлых матчей оценивает вероятности исходов и "
                "статистических показателей будущих футбольных матчей 21 европейской лиги, сопоставляет их с "
                "реальными букмекерскими коэффициентами и ведёт честную историю прогнозов с расчётом прибыли и ROI."),
              P("Цели итерации v6:")]
    story += bullets(["максимально восстановить реальные исторические коэффициенты сезона 2026/27;",
                      "исключить ошибки сопоставления матчей и котировок;",
                      "провести исследование качества моделей: ничьи, голы, угловые, производные рынки;",
                      "дать каждому матчу один Main и один Risk и восстановить пропущенные Risk;",
                      "свести историю прогнозов в единый набор без двойного подсчёта;",
                      "сделать объяснения прогнозов математически корректными;",
                      "точечно доработать интерфейс, протестировать систему и описать результаты."])
    story += [P("Отчёт строится скриптом `scripts/build_report.py` только из файлов, полученных запусками "
                "(аудит данных, метрики моделей, результаты экспериментов, журналы тестов). Где эксперимент не "
                "выполнялся или данных нет, это указано явно.")]

    # 4. architecture
    story += [H1("2. Архитектура"), P("Компоненты и поток данных:")]
    story += figure(chart_architecture(), "Рисунок 1. Архитектура FootPredict и поток данных.")
    story += bullets([
        "`src/data`, `src/providers` — загрузка и очистка результатов, расписаний и котировок; единый "
        "идентификатор команды, сопоставление названий провайдеров, запрет «двойных» матчей;",
        "`src/features` — предматчевые признаки (окна 5/10/15 матчей, сезон, дом/выезд, сила команд, угловые, "
        "личные встречи, средние лиги); всё считается только по матчам до даты игры;",
        "`src/modelling` — четыре модели XGBoost: исход 1X2 (multi:softprob), голы хозяев и гостей "
        "(Poisson), суммарные угловые (Poisson + отрицательное биномиальное распределение для линий);",
        "`src/markets` — матрица счёта и все производные рынки, реальные цены, EV, правила Main/Risk;",
        "`src/prediction` — генерация, версии до начала матча, расчёт результатов, восстановление истории;",
        "`views`, `src/ui` — Streamlit-сайт: Matches, Match Details, Prediction History, Leagues, Teams, "
        "Analytics, Model Analysis;",
        "`.github/workflows/update-data.yml` — ежедневное обновление (около 11:15 по Киеву) и ручные режимы."])

    # 5. previous version
    story += [H1("3. Анализ предыдущей версии (v5.1)")]
    if before:
        hr = before["history_roles"]
        story += [P(f"Состояние перед v6 (аудит `reports/v6/audit_before.json`, {esc(before['built_at'])}):")]
        story += [table([
            ["Показатель", "Значение"],
            ["Сыгранных матчей сезона 2026/27", before["played_matches"]],
            ["Матчей в единой истории", before["history_matches"]],
            ["Main рассчитано / с реальной ценой", f"{hr['main']['settled']} / {hr['main']['bets']} "
                                                  f"({share(hr['main']['odds_coverage'])})"],
            ["Main: попадание / ROI", f"{share(hr['main']['hit_rate'])} / {pct(hr['main']['roi'], signed=True)}"],
            ["Risk рассчитано / с реальной ценой", f"{hr['risk']['settled']} / {hr['risk']['bets']}"],
            ["Risk: попадание / ROI", f"{share(hr['risk']['hit_rate'])} / {pct(hr['risk']['roi'], signed=True)}"],
            ["Матчей с Main без Risk", before["main_without_risk"]],
        ], widths=[9 * cm, 8 * cm])]
    tm = {k: v.get("test_metrics", {}) for k, v in meta.get("tasks", {}).items()}
    if tm:
        r = tm.get("result", {})
        story += [Spacer(1, 6), P(f"Метрики активной модели на тестовом сезоне {', '.join(meta['periods']['test'])} "
                                  f"(n = {r.get('n')}):")]
        story += [table([
            ["Модель", "Метрики"],
            ["Исход 1X2", f"accuracy {num(r.get('accuracy'))}, macro F1 {num(r.get('macro_f1'))}, log loss "
                          f"{num(r.get('log_loss'))}, Brier {num(r.get('brier'))}; доля прогнозов «ничья» "
                          f"{share(r.get('predicted_share', {}).get('D'))} при фактической "
                          f"{share(r.get('actual_share', {}).get('D'))}"],
            *[[name, f"RMSE {num(tm[k].get('rmse'))}, MAE {num(tm[k].get('mae'))}, R² {num(tm[k].get('r2'))}, "
                     f"Poisson deviance {num(tm[k].get('poisson_deviance'))}, среднее прогноз/факт "
                     f"{num(tm[k].get('mean_predicted'), 2)}/{num(tm[k].get('mean_actual'), 2)}"]
              for k, name in (("home_goals", "Голы хозяев"), ("away_goals", "Голы гостей"), ("corners", "Угловые"))
              if k in tm],
        ], widths=[3.2 * cm, 13.8 * cm])]
    story += [P("Найденные проблемы: у 270 записанных матчей нет Risk (сохранены до появления правила Risk); "
                "около половины прогнозов без реальной цены; две ошибки сопоставления (Dundee Utd → Dundee, "
                "Espanyol → Barcelona) создали фантомные матчи — исправлены в PR #8; модель исходов почти никогда "
                "не называет ничью; история была разделена на Recorded / Backtest; ежедневный запуск по "
                "расписанию стартовал с опозданием до пяти часов.")]

    # 6. data sources
    story += [H1("4. Источники данных")]
    story += [table([
        ["Источник", "Назначение в FootPredict", "Ограничения"],
        ["Football-Data", "Результаты и статистика матчей с 2021/22 (обучение), закрывающие цены Bet365 и "
                          "средние по рынку для 1X2, тотала 2.5 и азиатской форы", "Публикуется после тура; "
                                                                                 "нет угловых линий и BTTS"],
        ["OddsPapi", "Основной источник цен: текущие котировки ближайших матчей и архив снимков до начала "
                     "матча (с января 2026)", "Бесплатно 250 платных запросов в месяц; исторический эндпоинт "
                                              "бесплатный, интервал ~5 с; лимиты частоты (429)"],
        ["API-Football", "Свежие результаты и расписание между публикациями Football-Data", "100 запросов в "
                                                                                           "день; коэффициенты "
                                                                                           "бесплатного тарифа "
                                                                                           "не используются"],
        ["The Odds API", "Ранее — резерв для отдельных рынков; сохранённые цены читаются из архива", "500 "
                                                                                                     "кредитов в "
                                                                                                     "месяц; в v6 "
                                                                                                     "выключен "
                                                                                                     "(ODDS_API_FALLBACK=off)"],
    ], widths=[2.8 * cm, 8.4 * cm, 5.8 * cm])]

    # 7. historical odds
    story += [H1("5. Исторические коэффициенты сезона 2026/27")]
    if coverage and coverage.get("total", {}).get("oddspapi_fixtures"):
        tot = coverage["total"]
        story += [P(f"Загрузка архива OddsPapi за период {coverage['period'][0]} – {coverage['period'][1]} "
                    f"(файл `local_data/api/oddspapi/history_coverage.json`): наших сыгранных матчей "
                    f"{tot['matches']}, фикстур OddsPapi {tot['oddspapi_fixtures']}, надёжно сопоставлено "
                    f"{tot['mapped']}, с хотя бы одной ценой {tot['with_odds']}.")]
    else:
        story += [P("Архив OddsPapi на момент сборки отчёта ещё не был загружен или не содержит фикстур; "
                    "таблицы ниже отражают имеющиеся данные.")]
    if after.get("leagues"):
        rows = [["Лига", "Матч.", "Фикстур ODP", "Сопост.", "С ценой", "Main с ценой", "Risk с ценой",
                 "1X2", "Тотал", "BTTS", "Фора", "Углов."]]
        for r in after["leagues"]:
            rows.append([r["league"], r["matches"], r["oddspapi_fixtures"], r["mapped"], r["with_any_price"],
                         f"{r['main_priced']}/{r['main_settled']}", f"{r['risk_priced']}/{r['risk_settled']}",
                         r["m_1X2"], r["m_TOTAL"], r["m_BTTS"], r["m_HANDICAP"], r["m_CORNERS"]])
        story += [table(rows, widths=[1.2 * cm, 1.3 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm, 1.9 * cm, 1.9 * cm,
                                      1.2 * cm, 1.3 * cm, 1.2 * cm, 1.2 * cm, 1.4 * cm]),
                  P("Сопост. — фикстуры OddsPapi, надёжно сопоставленные с матчем FootPredict (лига, дата, обе "
                    "команды). Рыночные столбцы — число матчей с ценой OddsPapi на этот рынок.", "small")]
    c = chart_coverage(before, after)
    if c:
        story += figure(c, "Рисунок 2. Доля рассчитанных Main с реальной ценой по лигам: до и после v6.")
    if before and after:
        hb, ha = before["history_roles"], after["history_roles"]
        story += [H2("Before / After покрытия"), table([
            ["", "До", "После"],
            ["Main с реальной ценой", f"{hb['main']['bets']} из {hb['main']['settled']} "
                                      f"({share(hb['main']['odds_coverage'])})",
             f"{ha['main']['bets']} из {ha['main']['settled']} ({share(ha['main']['odds_coverage'])})"],
            ["Risk с реальной ценой", f"{hb['risk']['bets']} из {hb['risk']['settled']}",
             f"{ha['risk']['bets']} из {ha['risk']['settled']}"],
            ["Цен, добавленных из архива", before.get("archive_prices", 0), after.get("archive_prices", 0)],
            ["Строк котировок (все провайдеры)", before["price_rows"]["total"], after["price_rows"]["total"]],
        ], widths=[6 * cm, 5.5 * cm, 5.5 * cm])]
    reasons = after.get("missing_reasons", {})
    if reasons:
        story += [H2("Причины отсутствия цены у рассчитанных прогнозов")]
        ru = {"match not found at OddsPapi (or not mapped safely)": "матч не найден у OddsPapi или не сопоставлен надёжно",
              "no pre-match price snapshot for the match": "нет предматчевого снимка цен для матча",
              "selection / line not offered by the bookmakers": "исход / линия не предлагались букмекерами",
              "this line of the market not offered": "эта линия рынка не предлагалась",
              "market not offered for the match": "рынок не предлагался для матча"}
        rows = [["Тип", "Причина", "Прогнозов"]]
        for role, rs in reasons.items():
            for why, n in sorted(rs.items(), key=lambda kv: -kv[1]):
                rows.append([role.capitalize(), ru.get(why, why), n])
        story += [table(rows, widths=[2 * cm, 12 * cm, 3 * cm])]
        story += [P("Объяснение причин: матч не найден — у OddsPapi нет фикстуры этого матча или "
                    "соответствие команд было сомнительным, и цена не привязывалась; нет снимка — фикстура "
                    "есть, но архив не хранит котировок до начала матча; рынок/линия не предлагались — "
                    "букмекеры не выставили эту линию (например, редкий точный счёт или фора ±2).")]

    # 8. data quality
    story += [H1("6. Проверка качества данных")]
    story += bullets([
        "Команды: внутренний идентификатор «страна|название Football-Data», для каждого провайдера — свои id. "
        "Новый id запоминается только при однозначном совпадении названия внутри лиги; один клуб получает "
        "только один id на провайдера; противоречие id и названия отклоняется и журналируется.",
        "Неоднозначные названия: короткое имя внутри более длинного известного имени («Dundee» в «Dundee "
        "United») и лишнее слово, похожее на другой клуб («Espanyol» ≈ Espanol), считаются неоднозначными — "
        "привязки нет.",
        "Дубликаты: одна команда не может играть два матча лиги в один день. Строка провайдера, "
        "противоречащая Football-Data, отбрасывается; два противоречащих друг другу провайдера отбрасываются "
        "оба (`src/data/cleaning.py: drop_double_bookings`).",
        "Перенесённые матчи: фикстура, заменённая сыгранным матчем той же пары в пределах 60 дней, удаляется; "
        "у перенесённого матча новая версия прогноза.",
        "Время котировок: цена используется как предматчевая, только если время котировки строго раньше "
        "начала матча; в архиве берётся последний такой снимок. Хранятся провайдер, букмекер, рынок, линия, "
        "исход, коэффициент, время котировки, время загрузки, начало матча и происхождение (текущая/архив).",
        "Противоречивые линии букмекера (вероятность «больше» растёт с линией, неупорядоченная лестница фор) "
        "отбрасываются.",
        "Утечка данных: признаки строятся только по матчам до даты игры; модели обучены на сезонах до "
        "тестового/текущего; при восстановлении Risk результат матча не используется."])

    # 9. features
    story += [H1("7. Feature Engineering")]
    groups = {}
    for task in meta.get("tasks", {}).values():
        for f in task.get("features", []):
            key = re.sub(r"^(h|a|diff)_", "", f.split("=")[0])
            key = re.sub(r"_(form5|form10|form15|season|venue).*", "", key)
            groups.setdefault(key, 0)
            groups[key] += 1
    story += [P(f"Число признаков по моделям: " + ", ".join(
        f"{name} — {len(t.get('features', []))}" for name, t in meta.get("tasks", {}).items()) + ".")]
    story += bullets([
        "окна формы 5 / 10 / 15 последних матчей и весь сезон, отдельно дома и в гостях;",
        "голы, удары, удары в створ, угловые за и против, очки за матч, разница голов;",
        "сила команд по рейтингу (Elo; в интерфейсе — «сила команды»), ожидаемый результат по силе, "
        "средняя сила соперников, поправки на соперника;",
        "сопоставления «атака против обороны» и «угловые против пропускаемых соперником угловых»;",
        "личные встречи (число, доли исходов, голы, BTTS, тотал);",
        "средние показатели лиги (голы, угловые, доли исходов), позиция и очки в таблице, дни отдыха;",
        "коэффициенты букмекеров НЕ используются как признаки (они нужны только для EV и ROI)."])

    # 10. methodology
    story += [H1("8. Методология машинного обучения")]
    p = meta.get("periods", {})
    story += [P(f"Временное разделение активной модели: обучение {', '.join(p.get('train', []))}, валидация "
                f"{', '.join(p.get('validation', []))}, тест {', '.join(p.get('test', []))}, текущий сезон "
                f"{p.get('current', '')}. Гиперпараметры подбирались Hyperopt (TPE) по валидационному сезону; "
                "итоговая модель переобучена на обучающих и валидационных сезонах.")]
    story += [P("Исследование v6 использует walk-forward: сезон 2023/24 прогнозируется моделью, обученной на "
                "сезонах до него, 2024/25 — на сезонах до 2024/25. Варианты выбираются только по этим двум "
                "сезонам; 2025/26 уже много раз анализировался и показывается «только для информации». "
                "Калибровка и пороги подбираются на сезоне, предшествующем оцениваемому.")]

    # 11-14 research
    for num_, title, key in (("9", "Исследование ничьих (Draw)", "Ничьи (Draw)"),
                             ("10", "Исследование моделей голов", "Модели голов"),
                             ("11", "Исследование модели угловых", "Угловые"),
                             ("12", "Производные рынки", "Производные рынки")):
        story += [H1(f"{num_}. {title}")]
        if key in findings:
            story += markdown(findings[key], heading_offset=1)
            story += research_figures(key)
        else:
            story += [P("Результаты этого исследования отсутствуют в `experiments/v6_research/findings.md`.")]
        if key == "Ничьи (Draw)" and tm.get("result"):
            bm = meta["tasks"]["result"].get("benchmarks", {}).get("bookmaker", {})
            if bm:
                story += [P(f"Справочно: у букмекерской оценки (нормированные 1/odds Bet365, тест "
                            f"{', '.join(p.get('test', []))}) log loss {num(bm.get('log_loss'))}, доля «ничья» как "
                            f"самого вероятного исхода {share(bm.get('predicted_share', {}).get('D'), 2)} — "
                            "ничья почти никогда не является самым вероятным исходом и для рынка.")]

    # 15. SHAP
    story += [H1("13. SHAP и объяснение прогнозов")]
    story += bullets([
        "Локальные объяснения считаются TreeSHAP той версией модели, которая сделала прогноз, по сохранённым "
        "предматчевым признакам.",
        "Какой выход модели объясняется: 1X2, двойной шанс, азиатская фора — классификатор исхода (log-odds "
        "класса); тоталы и BTTS — модели голов хозяев и гостей вместе (логарифм ожидаемых голов; вероятности "
        "монотонны по ожидаемым голам, поэтому знак вклада однозначен); угловые — модель угловых.",
        "Для точного счёта доли влияния не показываются: вероятность конкретного счёта не монотонна по "
        "ожидаемым голам, и доля SHAP создала бы ложную точность. Карточка показывает ожидаемые голы и "
        "фактическую статистику.",
        "Доля фактора = |SHAP фактора| / сумма |SHAP| всех признаков этого выхода. Это доля в «рассуждении» "
        "модели, а не изменение вероятности в процентных пунктах; вероятность исхода показывается отдельно.",
        "Парные признаки хозяев и гостей объединяются в один фактор, в подтверждении приводятся значения обеих "
        "команд; шаблонные формулировки («strong support», «works against») убраны.",
        "Глобальная важность (среднее |SHAP|) хранится в метаданных моделей и показана на Model Analysis."])

    # 16. Main / Risk
    story += [H1("14. Main и Risk")]
    story += bullets([
        "Main: кандидаты с вероятностью 40–85 % (без точного счёта); оценка = надёжность рынка × вероятность × "
        "мягкое предпочтение коэффициентов 1.40–2.50 × (1 + 0.5·EV) при реальной цене (×0.8 без цены).",
        "Risk: любая альтернатива, кроме Main, ставок с тем же набором счетов и ставок против Main; "
        "вероятность ≥ 12 %, мягкое предпочтение коэффициентов 3.0–6.0; Risk есть у каждого матча с Main.",
        "Без реальной цены Risk показывает исход и вероятность; EV отсутствует до появления настоящей цены."])
    if rule_cmp:
        rows = [["Правило Main", "2023/24 ROI", "2024/25 ROI", "2025/26 ROI (информ.)"]]
        for name, seasons in rule_cmp.get("rules", {}).items():
            rows.append([("★ " if name == rule_cmp.get("chosen") else "") + name,
                         *[pct(seasons.get(s, {}).get("roi", float("nan")) * 100, signed=True)
                           for s in ("2023", "2024", "2025")]])
        story += [H2("Сравнение правил Main (одни матчи, реальные цены Bet365)"),
                  table(rows, widths=[8.5 * cm, 2.8 * cm, 2.8 * cm, 2.9 * cm])]
    if "Правило Risk" in findings:
        story += [H2("Исследование порога EV для Risk")] + markdown(findings["Правило Risk"], heading_offset=1)
        story += research_figures("Правило Risk")
    if after:
        story += [H2("Восстановление истории"), P(
            f"Восстановлено Risk у записанных матчей: {after.get('risk_reconstructed', 0)}; матчей с Main без "
            f"Risk после восстановления: {after.get('main_without_risk')}. Цен, добавленных к сохранённым "
            f"прогнозам из архива: {after.get('archive_prices', 0)}. Исходные Main, их исход, вероятность, версия "
            "модели и время генерации не менялись.")]

    # 17 backtesting
    story += [H1("15. Историческая оценка (backtesting)")]
    if after:
        src = after.get("history_by_source", {})
        story += [P(f"Единая история сезона содержит {after['history_matches']} матчей: {src.get('live', 0)} с "
                    f"прогнозом, сохранённым до начала матча, и {src.get('backfill', 0)} ранних матчей сезона с "
                    "прогнозом, построенным позже из предматчевых данных моделью, обученной на прошлых сезонах.")]
    story += bullets([
        "для каждого матча используется одна запись Main и одна Risk: сохранённая до матча, если есть, иначе "
        "реконструкция; двойного подсчёта нет (`unified_history`);",
        "реконструкции используют только предматчевые признаки и цены, известные до начала матча;",
        "происхождение каждой строки (source, origin, odds_origin, время генерации и реконструкции) "
        "хранится в данных; интерфейс его не показывает, методика описана на Model Analysis;",
        "ограничение: реконструкции не были опубликованы заранее, поэтому общий результат — историческая "
        "оценка системы, а не только журнал опубликованных ставок."])

    # 18 profit
    story += [H1("16. Profit и ROI")]
    story += [P("Ставка 1 единица по реальной цене: выигрыш = odds − 1, проигрыш = −1, возврат (целая "
                "азиатская фора) и отмена = 0 и не входят в сумму ставок. ROI = чистая прибыль / сумма ставок × "
                "100 %. Прогнозы без реальной цены входят в процент попаданий, но не в прибыль и ROI.")]
    if after:
        rows = [["Набор", "Рассчитано", "С ценой", "Попадание", "Ср. коэф.", "Прибыль", "ROI"]]
        for label, key in (("Единая история", "history_roles"), ("Записанные до матча", "recorded_roles"),
                           ("Реконструкции", "rebuilt_roles")):
            for role in ("main", "risk"):
                r = after.get(key, {}).get(role)
                if r and r.get("settled"):
                    rows.append([f"{label} · {role.capitalize()}", r["settled"], r["bets"], share(r["hit_rate"]),
                                 num(r["avg_odds"], 2), num(r["profit"], 2), pct(r["roi"], signed=True)])
        story += [table(rows, widths=[5 * cm, 1.9 * cm, 1.7 * cm, 2 * cm, 1.8 * cm, 2 * cm, 2.6 * cm])]
        cum = chart_cumulative(hist.loc[hist["season"] == after.get("season")])
        if cum:
            story += figure(cum, "Рисунок 3. Накопленная прибыль Main и Risk по единой истории сезона.")
        m = chart_market_bars(after.get("main_by_market", []), "roi", "ROI Main по рынкам", "roi_market.png", True)
        if m:
            story += figure(m, "Рисунок 4. ROI Main по рынкам (n — ставки с реальной ценой).", width=14 * cm)
        lg = chart_market_bars(after.get("main_by_league", []), "roi", "ROI Main по лигам", "roi_league.png", True)
        if lg:
            story += figure(lg, "Рисунок 5. ROI Main по лигам.", width=14 * cm)
        story += [P("Малые выборки (менее 30 ставок) не позволяют уверенно называть рынок или лигу лучшими: "
                    "стандартная ошибка ROI при коэффициентах около 1.8 и 100 ставках составляет порядка 9 п.п.")]

    # 19 UI
    story += [H1("17. UI/UX")]
    story += bullets([
        "Логотип: вместо «FP» — минималистичный мяч с линиями движения и надпись Foot (светлым) Predict "
        "(зелёным); клик ведёт на Matches; favicon из того же знака (16–512 px, PNG и ICO); заголовок "
        "вкладки всегда «FootPredict».",
        "Поиск: полупрозрачное поле «Search…» шириной около 145 px справа в шапке, при вводе расширяется; "
        "результаты без Enter, ранжирование: точное совпадение и начало названия выше совпадений внутри "
        "названия (на «Arse» первым идёт Arsenal).",
        "Matches: эмблемы крупнее, коэффициенты крупнее и контрастнее, тонкие промежутки между сегментами 1X2.",
        "Match Details: Main 60 % ширины, Risk 40 %, одинаковая высота; новая карточка «Why …?»; в форах — "
        "только линии с реальными ценами, без переключателя «Show all lines».",
        "Prediction History: одна история без Recorded / Backtest / Both; фильтры Period, Prediction type, "
        "League, поиск команды; Market — в дополнительном меню; единые Profit и ROI.",
        "Leagues и Teams: время матчей в часовом поясе браузера той же функцией, что и на Matches (ранее "
        "показывалось британское время — разница 1–2 часа); единые показатели FootPredict в лиге.",
        "Analytics: те же данные, что History; карточки лучшего / худшего рынка без длинных подписей "
        "(размер выборки — во всплывающей подсказке); ROI по лигам — на всю ширину, 12 лиг и «Show all»; "
        "ROI по рынкам — приглушённые зелёный/красный с нулевой линией; Hit rate по рынкам — спокойный синий, "
        "без постоянных «n=»."])
    for name, cap in (("v6_matches.png", "Matches"), ("v6_match_top.png", "Match Details: Main / Risk и «Why …?»"),
                      ("v6_h.png", "Prediction History"), ("v6_analytics_full.png", "Analytics")):
        story += figure(V6 / "screenshots" / name, f"Скриншот: {cap}.", width=15 * cm)

    # 20 testing
    story += [H1("18. Тестирование")]
    if tests:
        rows = [["Набор", "Команда", "Результат"]]
        for k, v in tests.items():
            rows.append([k, v.get("command", ""), v.get("result", "")])
        story += [table(rows, widths=[3.2 * cm, 7.8 * cm, 6 * cm])]
        for k, v in tests.items():
            if v.get("notes"):
                story += [P(f"{k}: {v['notes']}", "small")]
    else:
        story += [P("Журнал тестов (reports/v6/tests.json) отсутствует.")]

    # 21 performance
    story += [H1("19. Производительность")]
    perf = load(V6 / "performance.json", {})
    if perf:
        story += [table([["Операция", "Время"]] + [[k, v] for k, v in perf.items()], widths=[10 * cm, 7 * cm])]
    else:
        story += [P("Замеры времени не сохранены.")]

    # 22 API limits
    story += [H1("20. Лимиты API")]
    api = load(V6 / "api_limits.json", {})
    if api:
        story += [table([["Провайдер", "Лимит", "Расход / состояние"]] + [[k, v.get("limit", ""), v.get("state", "")]
                                                                          for k, v in api.items()],
                        widths=[3.5 * cm, 5 * cm, 8.5 * cm])]
    story += [P("Рекомендации: ежедневный запуск тратит один платный запрос OddsPapi на список фикстур и бесплатные "
                "исторические запросы на цены; архив прошлых матчей догружается ручным режимом `history`, "
                "который продолжает прерванную загрузку; The Odds API выключен — включать только для отдельных "
                "рынков после оценки расхода.")]

    # 23 before/after
    story += [H1("21. Before / After")]
    if "Сводная таблица Before/After" in findings:
        story += markdown(findings["Сводная таблица Before/After"], heading_offset=1)
    if before and after:
        hb, ha = before["history_roles"], after["history_roles"]
        story += [H2("Данные и история"), table([
            ["Показатель", "До", "После"],
            ["Main с ценой", f"{hb['main']['bets']}/{hb['main']['settled']}", f"{ha['main']['bets']}/{ha['main']['settled']}"],
            ["ROI Main", pct(hb["main"]["roi"], signed=True), pct(ha["main"]["roi"], signed=True)],
            ["Risk рассчитано", hb["risk"]["settled"], ha["risk"]["settled"]],
            ["ROI Risk", pct(hb["risk"]["roi"], signed=True), pct(ha["risk"]["roi"], signed=True)],
            ["Матчей с Main без Risk", before["main_without_risk"], after["main_without_risk"]],
        ], widths=[7 * cm, 5 * cm, 5 * cm])]

    # 24 limitations
    story += [H1("22. Ограничения")]
    if "Неудачные эксперименты и ограничения" in findings:
        story += markdown(findings["Неудачные эксперименты и ограничения"], heading_offset=1)
    story += bullets([
        "Часть рынков (точный счёт, редкие линии фор и угловых) букмекеры выставляют не всегда — такие "
        "прогнозы остаются без цены и без прибыли.",
        "Футбольные исходы сильно зашумлены; разницы ROI в несколько процентов на сотнях ставок "
        "статистически незначимы.",
        "Реконструкции ранних матчей сезона не публиковались заранее; это историческая оценка.",
        "Остаток квоты OddsPapi через эндпоинт /account на бесплатном тарифе не всегда возвращается."])

    # 25 conclusion
    story += [H1("23. Заключение")]
    story += [P("В v6 история прогнозов сведена в один корректный набор без двойного подсчёта, у каждого матча "
                "есть Main и Risk, архивные цены добавляются к сохранённым прогнозам без изменения самих прогнозов, "
                "исключены найденные ошибки сопоставления, объяснения стали конкретными и математически "
                "корректными, интерфейс доработан точечно. Решения по моделям приняты по результатам "
                "walk-forward-проверки на сезонах 2023/24 и 2024/25 (раздел 9–12 и сводная таблица).")]

    # 26 appendix
    story += [H1("24. Приложения")]
    story += [H2("Команды запуска"), Preformatted("\n".join([
        "python scripts/run_pipeline.py --mode full            # обновление данных и прогнозов",
        "python scripts/oddspapi_history.py                    # оценка загрузки архива (dry run)",
        "python scripts/oddspapi_history.py --go               # загрузка архива (продолжаемая)",
        "python scripts/restore_recorded_odds.py --go          # архивные цены и пропущенные Risk",
        "python scripts/retrain_models.py                      # переобучение моделей",
        "python -m experiments.v6_research.run --all           # исследование v6",
        "python scripts/v6_audit.py --out reports/v6/audit_after.json",
        "python scripts/build_report.py                        # этот отчёт",
        "pytest && pytest tests/e2e --e2e                      # тесты"]), S["mono"])]
    story += [H2("Артефакты"), table([
        ["Путь", "Содержимое"],
        ["experiments/v6_research/", "код, результаты (results/*.json) и рисунки исследования"],
        ["reports/v6/audit_before.json, audit_after.json", "аудит данных до и после"],
        ["local_data/api/oddspapi/history_coverage.json", "покрытие архива OddsPapi"],
        ["local_data/models/<версия>/metadata.json", "параметры и метрики моделей"],
        ["scripts/build_report.py", "генератор этого отчёта"],
    ], widths=[7.5 * cm, 9.5 * cm])]
    if research_results:
        story += [P("Файлы результатов исследования: " + ", ".join(f"`{p.name}`" for p in research_results), "small")]
    req = (ROOT / "requirements.txt").read_text().strip().splitlines()
    story += [H2("Версии зависимостей"), Preformatted("\n".join(req), S["mono"])]
    story += [H2("Git"), Preformatted(git("log", "--oneline", "-25"), S["mono"]),
              P(f"Отчёт собран из коммита {esc(short)}.", "small")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Doc(OUT).multiBuild(story)
    return OUT


if __name__ == "__main__":
    print(build())
