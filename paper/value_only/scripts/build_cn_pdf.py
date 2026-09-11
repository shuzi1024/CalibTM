#!/usr/bin/env python3
"""Build and render the Chinese CalibTM research manuscript.

Document-only build: this script reads ``paper/value_only/main_cn.md`` and
does not import experiment code or open data, checkpoints, result caches, or
sealed-test artifacts.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper" / "value_only" / "main_cn.md"
OUTPUT = ROOT / "output" / "pdf" / "CalibTM_中文方法论文_v3.pdf"
RENDER_DIR = ROOT / "tmp" / "pdfs" / "calibtm_cn_v3"

INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5F6B76")
NAVY = colors.HexColor("#173B57")
BLUE = colors.HexColor("#2D6E93")
BLUE_LIGHT = colors.HexColor("#EAF3F8")
TEAL = colors.HexColor("#0A7A78")
TEAL_DARK = colors.HexColor("#075E5D")
TEAL_LIGHT = colors.HexColor("#E7F5F3")
ORANGE = colors.HexColor("#D66B2C")
ORANGE_LIGHT = colors.HexColor("#FFF1E8")
RED = colors.HexColor("#B0443C")
RED_LIGHT = colors.HexColor("#FCEDEC")
GREEN = colors.HexColor("#2F7D58")
GRID = colors.HexColor("#C9D3DB")
PAPER = colors.white


def register_fonts() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdfmetrics.registerFontFamily(
        "STSong-Light",
        normal="STSong-Light",
        bold="STSong-Light",
        italic="STSong-Light",
        boldItalic="STSong-Light",
    )


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=sample["BodyText"],
        fontName="STSong-Light",
        fontSize=9.25,
        leading=14.0,
        textColor=INK,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        spaceAfter=4.5,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "TitleCN",
            parent=body,
            fontSize=20.5,
            leading=28,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceBefore=8,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=body,
            fontSize=10.4,
            leading=16,
            textColor=TEAL_DARK,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=body,
            fontSize=14.2,
            leading=20,
            textColor=TEAL_DARK,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=body,
            fontSize=11.4,
            leading=17,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3CN",
            parent=body,
            fontSize=10.1,
            leading=15,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "abstract": ParagraphStyle(
            "AbstractCN",
            parent=body,
            fontSize=8.9,
            leading=14.1,
            leftIndent=6 * mm,
            rightIndent=6 * mm,
            spaceAfter=6,
        ),
        "quote": ParagraphStyle(
            "QuoteCN",
            parent=body,
            fontSize=9.1,
            leading=14.4,
            leftIndent=5 * mm,
            rightIndent=4 * mm,
            textColor=TEAL_DARK,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=body,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=1.5 * mm,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=body,
            fontSize=7.9,
            leading=11.7,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=body,
            fontSize=7.6,
            leading=11.1,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "reference": ParagraphStyle(
            "ReferenceCN",
            parent=body,
            fontSize=7.2,
            leading=9.5,
            spaceAfter=1.0,
        ),
        "code": ParagraphStyle(
            "CodeCN",
            parent=body,
            fontName="Courier",
            fontSize=7.5,
            leading=11,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            backColor=colors.HexColor("#F4F6F7"),
            borderColor=GRID,
            borderWidth=0.4,
            borderPadding=4,
            spaceBefore=3,
            spaceAfter=6,
        ),
    }


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#2D6E93">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font color="#075E5D">\1</font>', text)
    return text


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(inline(text), style)


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, color=TEAL) -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.5))
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 6
    for delta in (2.55, -2.55):
        d.add(
            Line(
                x2,
                y2,
                x2 + length * math.cos(angle + delta),
                y2 + length * math.sin(angle + delta),
                strokeColor=color,
                strokeWidth=1.5,
            )
        )


def box(d: Drawing, x: float, y: float, w: float, h: float, label: str, fill, stroke, size=8.2) -> None:
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1))
    lines = label.split("\n")
    start = y + h / 2 + (len(lines) - 1) * 5
    for i, line in enumerate(lines):
        d.add(
            String(
                x + w / 2,
                start - i * 11,
                line,
                fontName="STSong-Light",
                fontSize=size,
                fillColor=INK,
                textAnchor="middle",
            )
        )


def method_figure() -> tuple[Drawing, str]:
    d = Drawing(500, 238)
    d.add(String(8, 220, "CalibTM：由真实锚点定义 scaffold，只学习区域化有界校准", fontName="STSong-Light", fontSize=11, fillColor=NAVY))

    box(d, 14, 161, 88, 43, "K=3 真实锚点\nobservation only", TEAL_LIGHT, TEAL, 7.5)
    box(d, 126, 161, 102, 43, "canonical Linear\nscaffold  B_t", TEAL_LIGHT, TEAL, 7.5)
    box(d, 254, 157, 126, 51, "scaffold-conditioned\n共享校准器  g_theta", BLUE_LIGHT, BLUE, 7.5)
    arrow(d, 102, 182.5, 126, 182.5, TEAL)
    arrow(d, 228, 182.5, 254, 182.5, TEAL)

    box(d, 220, 96, 112, 39, "内部缺口\n尺度有界 offset", TEAL_LIGHT, TEAL, 7.4)
    box(d, 350, 96, 112, 39, "单侧边界段\n尺度有界 shift", TEAL_LIGHT, TEAL, 7.4)
    arrow(d, 303, 157, 276, 135, BLUE)
    arrow(d, 331, 157, 406, 135, BLUE)

    box(d, 270, 38, 135, 38, "hard-copy anchors\n+ nonnegative projection", ORANGE_LIGHT, ORANGE, 7.2)
    arrow(d, 276, 96, 315, 76, TEAL)
    arrow(d, 406, 96, 360, 76, TEAL)
    box(d, 426, 38, 61, 38, "完成 flow\n一次前向", TEAL_LIGHT, TEAL, 7.2)
    arrow(d, 405, 57, 426, 57, TEAL)

    d.add(Rect(13, 31, 225, 65, rx=5, ry=5, fillColor=colors.HexColor("#F5F7F8"), strokeColor=GRID, strokeWidth=0.8, strokeDashArray=[3, 2]))
    d.add(String(24, 80, "设计空间中的操作点", fontName="STSong-Light", fontSize=8.2, fillColor=NAVY))
    d.add(String(24, 62, "Flexible decoder：直接估计完整缺失轨迹", fontName="STSong-Light", fontSize=7.2, fillColor=MUTED))
    d.add(String(24, 46, "CalibTM：只搜索 B_t 周围的有界区域形变", fontName="STSong-Light", fontSize=7.2, fillColor=TEAL_DARK))
    d.add(String(13, 13, "该对比表达设计原则，不构成对所有自由重建方法的 matched 因果结论。", fontName="STSong-Light", fontSize=7.2, fillColor=MUTED))
    return d, "图 1  CalibTM 的主机制。真实观测先定义 canonical Linear；共享校准器根据 scaffold 与观测导出尺度，对内部位置和单侧边界段施加显式有界的修正，随后精确复制锚点并保证缺失预测非负。下方设计空间对比表示本文选择的 operating point，而非其优于所有自由重建方法的因果结论。"


def gains_figure() -> tuple[Drawing, str]:
    d = Drawing(500, 235)
    d.add(String(10, 217, "Structured NMAE 相对改善（%）", fontName="STSong-Light", fontSize=10.2, fillColor=NAVY))
    values = {
        "Abilene": (1.3413, 1.8900),
        "GEANT": (0.7437, 1.1474),
        "BRAIN": (1.1396, 3.2627),
    }
    left, bottom, top = 64, 42, 198
    d.add(Line(left, bottom, left, top, strokeColor=GRID, strokeWidth=1))
    d.add(Line(left, bottom, 484, bottom, strokeColor=GRID, strokeWidth=1))
    for tick in range(5):
        y = bottom + tick * (top - bottom) / 4
        d.add(Line(left - 3, y, 484, y, strokeColor=colors.HexColor("#E5EAEE"), strokeWidth=0.5))
        d.add(String(left - 9, y - 3, str(tick), fontName="STSong-Light", fontSize=7, fillColor=MUTED, textAnchor="end"))
    group_w = 118
    bar_w = 30
    for idx, (name, (vs_static, vs_linear)) in enumerate(values.items()):
        center = left + 76 + idx * group_w
        for offset, value, color, label in [(-18, vs_static, TEAL, "vs context-free"), (18, vs_linear, ORANGE, "vs Linear")]:
            height = value / 4 * (top - bottom)
            d.add(Rect(center + offset - bar_w / 2, bottom, bar_w, height, fillColor=color, strokeColor=None))
            d.add(String(center + offset, bottom + height + 5, f"{value:.2f}", fontName="STSong-Light", fontSize=7.2, fillColor=INK, textAnchor="middle"))
        d.add(String(center, 25, name, fontName="STSong-Light", fontSize=8, fillColor=INK, textAnchor="middle"))
    d.add(Rect(335, 210, 9, 7, fillColor=TEAL, strokeColor=None))
    d.add(String(349, 210, "vs Context-Free", fontName="STSong-Light", fontSize=7, fillColor=MUTED))
    d.add(Rect(423, 210, 9, 7, fillColor=ORANGE, strokeColor=None))
    d.add(String(437, 210, "vs Linear", fontName="STSong-Light", fontSize=7, fillColor=MUTED))
    return d, "图 2  Scaffold-conditioned calibration 相对 matched context-free control 与 canonical Linear 的 structured effect。A/G 为 historical evidence，BRAIN 为冻结后的第三 WAN confirmation；该图不表示 CalibTM 在三个 WAN 上均优于所有 learned methods。"


def _map_log(value: float, low: float, high: float, x0: float, x1: float) -> float:
    return x0 + (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) * (x1 - x0)


def pareto_figure() -> tuple[Drawing, str]:
    d = Drawing(500, 250)
    d.add(String(10, 232, "H800 batch=8：accuracy–latency（越左、越上越好）", fontName="STSong-Light", fontSize=10.2, fillColor=NAVY))
    data = {
        "Abilene": {
            "Linear": (0.340, 0.237557), "static": (2.867, 0.236011), "CalibTM": (3.171, 0.232858),
            "ARI": (37.372, 0.230258), "Impute": (5.836, 0.222715),
        },
        "GEANT": {
            "Linear": (0.351, 0.217642), "static": (2.820, 0.216573), "CalibTM": (2.884, 0.215160),
            "ARI": (73.213, 0.227781), "Impute": (14.611, 0.312429),
        },
    }
    colors_by = {"Linear": MUTED, "static": BLUE, "CalibTM": TEAL, "ARI": ORANGE, "Impute": RED}
    for panel, name in enumerate(("Abilene", "GEANT")):
        x0 = 52 + panel * 245
        x1 = x0 + 195
        y0, y1 = 44, 205
        vals = data[name]
        ys = [v[1] for v in vals.values()]
        lo, hi = min(ys) - 0.006, max(ys) + 0.006
        d.add(Line(x0, y0, x1, y0, strokeColor=GRID))
        d.add(Line(x0, y0, x0, y1, strokeColor=GRID))
        d.add(String((x0+x1)/2, 17, f"{name}  median latency (ms, log)", fontName="STSong-Light", fontSize=7.4, fillColor=MUTED, textAnchor="middle"))
        d.add(String(x0, 213, name, fontName="STSong-Light", fontSize=8.5, fillColor=NAVY))
        for latency_tick in (0.3, 1, 3, 10, 30, 100):
            if latency_tick < 0.3 or latency_tick > 100:
                continue
            x = _map_log(latency_tick, 0.3, 100, x0, x1)
            d.add(Line(x, y0, x, y0 - 3, strokeColor=GRID))
            d.add(String(x, y0 - 12, f"{latency_tick:g}", fontName="STSong-Light", fontSize=6.2, fillColor=MUTED, textAnchor="middle"))
        for method, (lat, nmae) in vals.items():
            x = _map_log(lat, 0.3, 100, x0, x1)
            y = y1 - (nmae - lo) / (hi - lo) * (y1 - y0)
            d.add(Circle(x, y, 4.5 if method == "CalibTM" else 3.8, fillColor=colors_by[method], strokeColor=PAPER, strokeWidth=0.7))
            dx = 6
            dy = -2
            if method in {"static", "CalibTM"}:
                dy = 8 if method == "CalibTM" else -12
            display = "Ctx-Free" if method == "static" else method
            d.add(String(x + dx, y + dy, display, fontName="STSong-Light", fontSize=7.2, fillColor=colors_by[method]))
        d.add(String(x0 - 8, y1 - 3, f"{lo:.3f}", fontName="STSong-Light", fontSize=6.2, fillColor=MUTED, textAnchor="end"))
        d.add(String(x0 - 8, y0 - 1, f"{hi:.3f}", fontName="STSong-Light", fontSize=6.2, fillColor=MUTED, textAnchor="end"))
    d.add(String(8, 120, "NMAE", fontName="STSong-Light", fontSize=7, fillColor=MUTED))
    return d, "图 3  A/G exact historical all-mask accuracy 与 seed-1 H800 batch=8 latency 的联合视图。两个轴来自不同但冻结的实验；ImputeFormer timing 含 adapter pipeline overhead，不能解释为架构内禀速度。"


FIGURES = {"METHOD": method_figure, "GAINS": gains_figure, "PARETO": pareto_figure}


def parse_table(lines: list[str], style_map: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    cols = max(len(row) for row in rows)
    rows = [row + [""] * (cols - len(row)) for row in rows]
    total_width = 174 * mm
    weights = []
    for col in range(cols):
        max_len = max(len(re.sub(r"[*`]", "", row[col])) for row in rows)
        weights.append(max(1.0, min(max_len, 26) ** 0.72))
    widths = [total_width * w / sum(weights) for w in weights]
    font_size = 6.15 if cols >= 6 else 6.55 if cols >= 5 else 7.0
    cell_style = ParagraphStyle(
        "TableCellCN",
        parent=style_map["body"],
        fontSize=font_size,
        leading=font_size + 3.0,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    data = [[Paragraph(inline(cell), cell_style) for cell in row] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), PAPER),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    for row_idx in range(1, len(data)):
        if row_idx % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#F5F8FA"))]))
    return table


def build_story(source: str, style_map: dict[str, ParagraphStyle]) -> list[object]:
    lines = source.splitlines()
    story: list[object] = []
    paragraph_buffer: list[str] = []
    first_title = True

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if paragraph_buffer:
            text = " ".join(x.strip() for x in paragraph_buffer).strip()
            if text:
                if re.match(r"^\[\d+\]\s", text):
                    chosen = style_map["reference"]
                elif story and any(
                    isinstance(item, Paragraph) and getattr(item.style, "name", "") == "Heading1CN" and "摘要" in item.getPlainText()
                    for item in story[-2:]
                ):
                    chosen = style_map["abstract"]
                else:
                    chosen = style_map["body"]
                story.append(paragraph(text, chosen))
            paragraph_buffer = []

    i = 0
    in_code = False
    code_lines: list[str] = []
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), style_map["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        fig_match = re.fullmatch(r"<!-- FIGURE:([A-Z]+) -->", stripped)
        if fig_match:
            flush_paragraph()
            drawing, caption = FIGURES[fig_match.group(1)]()
            story.append(KeepTogether([drawing, paragraph(caption, style_map["caption"])]))
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(Spacer(1, 3))
            story.append(parse_table(table_lines, style_map))
            story.append(Spacer(1, 7))
            continue
        if stripped == "---":
            flush_paragraph()
            story.append(PageBreak())
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(paragraph(stripped[4:], style_map["h2"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(paragraph(stripped[3:], style_map["h1"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            if first_title:
                story.append(Spacer(1, 10 * mm))
                story.append(paragraph(stripped[2:], style_map["title"]))
                first_title = False
            else:
                story.append(paragraph(stripped[2:], style_map["h1"]))
            i += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            content = paragraph(stripped[2:], style_map["quote"])
            qtable = Table([[content]], colWidths=[164 * mm])
            qtable.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), TEAL_LIGHT),
                        ("BOX", (0, 0), (-1, -1), 0.6, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(qtable)
            story.append(Spacer(1, 5))
            i += 1
            continue
        bullet_match = re.match(r"^[-*] (.+)$", stripped)
        number_match = re.match(r"^(\d+)\. (.+)$", stripped)
        if bullet_match or number_match:
            flush_paragraph()
            marker = "•" if bullet_match else number_match.group(1) + "."
            content = bullet_match.group(1) if bullet_match else number_match.group(2)
            story.append(Paragraph(inline(content), style_map["bullet"], bulletText=marker))
            i += 1
            continue
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) < 180:
            flush_paragraph()
            story.append(paragraph(stripped, style_map["subtitle"]))
            i += 1
            continue
        paragraph_buffer.append(stripped)
        i += 1
    flush_paragraph()
    return story


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont("STSong-Light", 7.1)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9.5 * mm, "CalibTM 中文方法论文研究稿 v3 · 2026-08-12")
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    style_map = styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="CalibTM：极端时间稀疏下的轻量锚点保持流量矩阵校准补全",
        author="ARI-LLM project",
        subject="CalibTM Chinese method-paper research draft",
    )
    story = build_story(source, style_map)
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)


def render_pdf() -> int:
    import fitz

    document = fitz.open(OUTPUT)
    for old in RENDER_DIR.glob("page-*.png"):
        old.unlink()
    matrix = fitz.Matrix(1.55, 1.55)
    for index, page in enumerate(document):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(RENDER_DIR / f"page-{index + 1:02d}.png")
    return len(document)


def main() -> None:
    build_pdf()
    pages = render_pdf()
    print(f"built={OUTPUT}")
    print(f"pages={pages}")
    print(f"rendered={RENDER_DIR}")


if __name__ == "__main__":
    main()
