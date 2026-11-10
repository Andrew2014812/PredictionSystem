"""Shared building blocks for the FootPredict research PDFs (Russian, A4, DejaVu fonts).

Every v6.1 and v6.2 report (reports/v6_1/*.pdf, reports/v6_2/*.pdf) is built with this kit so the six documents look like one
series: a title page with the date and Git SHA, a table of contents, numbered pages, readable tables,
figures with captions, formulas and a "how to reproduce" block.

    from scripts.report_kit import Report, P, H1, H2, H3, bullets, table, figure, formula, code
    rep = Report("reports/v6_1/Draw_Research.pdf", "Исследование ничьих", "FootPredict v6.1")
    rep.add(H1("1. Цель исследования"), P("…"))
    rep.build()              # writes the PDF (two passes for the contents)
    rep.check()              # renders every page to PNG next to the PDF and returns the page count

Numbers in a report must come from result files produced by a run; when an input is missing the
report says so instead of filling it in.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"      # DejaVu ships with matplotlib

pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-Bold")

INK, MUTED, ACCENT, NEG, AMBER, GRID = "#14202a", "#4b5a63", "#167c66", "#c0392b", "#c88a12", "#dfe4e2"
PALETTE = ["#167c66", "#c88a12", "#3d6fb6", "#8e5aa8", "#c0392b", "#5f7a85"]


def plot_style() -> None:
    """Matplotlib defaults used by every chart of the series (Cyrillic-safe font)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.prop_cycle": plt.cycler(color=PALETTE)})


plot_style()

S = {
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.6, leading=13.6, spaceAfter=5, textColor=INK),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8.6, leading=11.2, textColor=MUTED, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, leading=19, spaceBefore=4, spaceAfter=9,
                         textColor=INK),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, spaceBefore=9, spaceAfter=5,
                         textColor=INK),
    "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=10, leading=13, spaceBefore=6, spaceAfter=3,
                         textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DejaVu", fontSize=8.6, leading=11, textColor=MUTED, alignment=TA_CENTER,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.6, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=8.6, leading=10.6, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.6, leading=13.4, leftIndent=12,
                             bulletIndent=2, spaceAfter=2, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DejaVuMono", fontSize=8.2, leading=10.2, textColor=INK,
                           backColor="#f1f4f2", leftIndent=4, rightIndent=4, borderPadding=4, spaceAfter=6),
    "formula": ParagraphStyle("formula", fontName="DejaVu", fontSize=10, leading=15, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=7, textColor=INK),
    "note": ParagraphStyle("note", fontName="DejaVu", fontSize=9, leading=12.6, textColor=INK, backColor="#eef5f2",
                           borderColor="#b9d8cd", borderWidth=0.5, borderPadding=6, spaceBefore=4, spaceAfter=9),
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=28, leading=34, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=13, leading=18, textColor=MUTED),
}


# ---------------------------------------------------------------------------------------------- values
def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "—"


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def pct(v, d=1, signed=False):
    """A value already in percent (12.3 -> '12.3%')."""
    return "—" if _missing(v) else (f"{v:+.{d}f}%" if signed else f"{v:.{d}f}%")


def share(v, d=1, signed=False):
    """A fraction shown in percent (0.123 -> '12.3%')."""
    return "—" if _missing(v) else (f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%")


def num(v, d=3):
    return "—" if _missing(v) else f"{v:.{d}f}"


# ---------------------------------------------------------------------------------------------- markup
def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text) -> str:
    """**bold**, *italic*, `code`, x_{sub} and x^{sup} inside ordinary text."""
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="DejaVuMono" size="8">\1</font>', t)
    t = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", t)
    t = re.sub(r"\^\{([^}]+)\}", r"<super>\1</super>", t)
    return t


def P(text, style: str = "body") -> Paragraph:
    return Paragraph(inline(text), S[style])


def note(text) -> Paragraph:
    """A highlighted box for a conclusion or a warning."""
    return Paragraph(inline(text), S["note"])


def formula(text) -> Paragraph:
    """A centred formula; use x_{i}, x^{2}, Σ, ≥, √ and Greek letters directly."""
    return Paragraph(inline(text), S["formula"])


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), S["mono"])


def bullets(items) -> list:
    return [Paragraph(inline(i), S["bullet"], bulletText="•") for i in items]


def H1(text) -> Paragraph:
    return Paragraph(esc(text), S["h1"])


def H2(text) -> Paragraph:
    return Paragraph(esc(text), S["h2"])


def H3(text) -> Paragraph:
    return Paragraph(esc(text), S["h3"])


def auto_widths(rows: list[list], total: float = 17 * cm) -> list[float]:
    """Column widths proportional to the longest text of each column (clamped)."""
    n = max(len(r) for r in rows)
    longest = [max(min(len(str(r[i])) if i < len(r) else 0, 34) for r in rows) for i in range(n)]
    longest = [max(x, 6) for x in longest]
    return [total * x / sum(longest) for x in longest]


MIN_TABLE_FONT = 8.0
MAX_TABLE_COLUMNS = 11


def table(rows: list[list], widths=None, header=True, zebra=True, font=None) -> Table:
    """A wrapped, gridded table. Text is never smaller than MIN_TABLE_FONT (8 pt): a table too wide for
    that must be split into two tables (or transposed) by the caller — ``table`` sets wider tables at 7.4 pt and warns."""
    rows = [[("—" if _missing(c) else c) for c in r] for r in rows]
    ncols = max(len(r) for r in rows)
    if ncols > MAX_TABLE_COLUMNS:          # older reports: still built, but flagged for splitting
        print(f"report_kit: table with {ncols} columns is set at 7.4 pt; split it to keep 8 pt", file=sys.stderr)
        size = 7.4
    else:
        size = max(font or (8.0 if ncols >= 9 else 8.6), MIN_TABLE_FONT)
    body = ParagraphStyle("cw", parent=S["cell"], fontSize=size, leading=size * 1.24)
    head = ParagraphStyle("cwb", parent=S["cellb"], fontSize=size, leading=size * 1.24)
    data = [[Paragraph(inline(str(c)), head if (header and r == 0) else body) for c in row]
            for r, row in enumerate(rows)]
    tbl = Table(data, colWidths=widths or auto_widths(rows), repeatRows=1 if header else 0, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d1cd")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ebe7")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9f8")))
    tbl.setStyle(TableStyle(style))
    return tbl


def df_table(df, columns=None, fmt=None, widths=None, max_rows=None) -> Table:
    """A pandas DataFrame as a table; ``fmt`` maps a column to a formatting function."""
    fmt = fmt or {}
    cols = list(columns or df.columns)
    frame = df if max_rows is None else df.head(max_rows)
    rows = [cols] + [[fmt[c](r[c]) if c in fmt else r[c] for c in cols] for _, r in frame.iterrows()]
    return table(rows, widths=widths)


def figure(path, caption: str, width=16.5 * cm) -> list:
    path = Path(path)
    if not path.exists():
        return [P(f"(Рисунок не построен: {path.name} отсутствует.)", "small")]
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    height = width * h / w
    if height > 21 * cm:
        width, height = width * 21 * cm / height, 21 * cm
    return [KeepTogether([Image(str(path), width=width, height=height), Paragraph(esc(caption), S["cap"])])]


def save_chart(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def markdown(md: str) -> list:
    """A small markdown subset (### headings, paragraphs, lists, pipe tables, code) -> flowables."""
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
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append(code("\n".join(block)))
        elif re.match(r"^#{1,6} ", line):
            flush()
            out.append(H3(line.lstrip("# ").strip()))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            n = max(len(r) for r in rows)
            out += [table([r + [""] * (n - len(r)) for r in rows]), Spacer(1, 4)]
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


# ---------------------------------------------------------------------------------------------- document
class _Doc(BaseDocTemplate):
    def __init__(self, path, title: str, running: str, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                         bottomMargin=1.8 * cm, title=title, author="FootPredict", **kw)
        self.running = running
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self._decor)])

    def _decor(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.1 * cm, self.running)
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
        canvas.setStrokeColor(colors.HexColor(GRID))
        canvas.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0 if flowable.style.name == "h1" else 1, flowable.getPlainText(),
                                     self.page, key))


def series_name(path: Path) -> str:
    """"FootPredict v6.2" for a report under reports/v6_2/ (the folder names the iteration), v6.1 otherwise."""
    m = re.search(r"v(\d+)_(\d+)", str(path))
    return f"FootPredict v{m.group(1)}.{m.group(2)}" if m else "FootPredict v6.1"


class Report:
    """One research PDF: title page, contents, body; ``add`` flowables, then ``build`` and ``check``."""

    def __init__(self, path, title: str, subtitle: str | None = None,
                 facts: dict[str, str] | None = None):
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.series = series_name(self.path)
        self.title, self.subtitle = title, subtitle or self.series
        self.facts = facts or {}
        self.story: list = []

    def add(self, *flowables) -> "Report":
        for f in flowables:
            self.story.extend(f if isinstance(f, list) else [f])
        return self

    def page_break(self) -> "Report":
        self.story.append(PageBreak())
        return self

    def _front(self) -> list:
        facts = {"Дата": f"{date.today():%d.%m.%Y}", "Git commit": git("rev-parse", "HEAD"), **self.facts}
        front = [Spacer(1, 4.5 * cm), Paragraph("FootPredict", S["subtitle"]), Spacer(1, 6),
                 Paragraph(esc(self.title), S["title"]), Spacer(1, 8),
                 Paragraph("Информационная система прогнозирования результатов и статистических показателей "
                           "футбольных матчей на основе машинного обучения", S["subtitle"]),
                 Spacer(1, 1.2 * cm), Paragraph(esc(self.subtitle), S["h3"])]
        front += [Paragraph(f"{esc(k)}: {esc(v)}", S["body"]) for k, v in facts.items()]
        toc = TableOfContents()
        toc.levelStyles = [ParagraphStyle("t0", fontName="DejaVu-Bold", fontSize=9.5, leading=13, leftIndent=0),
                           ParagraphStyle("t1", fontName="DejaVu", fontSize=8.6, leading=11.5, leftIndent=14)]
        return front + [PageBreak(), Paragraph("Содержание", S["h3"]), toc, PageBreak()]

    def build(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = _Doc(self.path, f"FootPredict — {self.title}", f"{self.series} — {self.title}")
        doc.multiBuild(self._front() + self.story)
        return self.path

    def check(self, scale: float = 1.0) -> int:
        """Render every page to PNG (folder <name>_pages next to the PDF) for a visual check."""
        import pypdfium2 as pdfium
        out = self.path.parent / f"{self.path.stem}_pages"
        out.mkdir(exist_ok=True)
        pdf = pdfium.PdfDocument(str(self.path))
        for i in range(len(pdf)):
            pdf[i].render(scale=scale).to_pil().save(out / f"page_{i + 1:02d}.png")
        return len(pdf)


def reproduce(commands: list[str]) -> list:
    """The standard "how to reproduce" block."""
    return [P("Команды воспроизведения (из корня репозитория, окружение `.venv`):"), code("\n".join(commands))]
