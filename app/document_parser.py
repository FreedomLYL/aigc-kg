"""文档解析模块：从 PPT(X) / PDF / DOCX / XLSX / TXT / Markdown 提取纯文本。

未知二进制格式（图片/视频/zip 等）无法解析文本时抛 UnsupportedFormat，
由上层捕获后仍允许“上传存档”，但不参与自动构图。
"""
from pathlib import Path


class UnsupportedFormat(Exception):
    """声明确实无法提取文本的格式（非致命，仅存档）。"""


def extract_text_from_bytes(raw: bytes, filename: str) -> str:
    """根据文件后缀路由到对应解析器，返回清洗后的文本。"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(raw)
    if suffix == ".docx":
        return _extract_docx(raw)
    if suffix == ".pptx":
        return _extract_pptx(raw)
    if suffix == ".ppt":
        raise ValueError(".ppt 为旧版二进制格式，请先另存为 .pptx 后再上传")
    if suffix in (".xlsx", ".xls"):
        return _extract_xlsx(raw)
    if suffix in (".txt", ".md", ".markdown"):
        return _extract_txt(raw)
    raise UnsupportedFormat(f"({suffix or '未知'} 文档)未能解析出文本")


def _extract_pdf(raw: bytes) -> str:
    import pdfplumber
    import io
    text = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return _clean("\n".join(text))


def _extract_docx(raw: bytes) -> str:
    import io
    from docx import Document
    doc = Document(io.BytesIO(raw))
    return _clean("\n".join(p.text for p in doc.paragraphs))


def _extract_pptx(raw: bytes) -> str:
    """提取 PowerPoint 每页的文本框与表格文本。"""
    import io
    from pptx import Presentation
    prs = Presentation(io.BytesIO(raw))
    lines = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in para.runs).strip()
                    if t:
                        lines.append(t)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
    return _clean("\n".join(lines))


def _extract_xlsx(raw: bytes) -> str:
    """提取 Excel 各工作表单元格文本（每行用 | 连接）。"""
    import io
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
    out = []
    try:
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                vals = [str(v) for v in row if v is not None and str(v).strip()]
                if vals:
                    out.append(" | ".join(vals))
    finally:
        wb.close()
    return _clean("\n".join(out))


def _extract_txt(raw: bytes) -> str:
    # 优先按 utf-8，失败则回退 gbk
    try:
        return _clean(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return _clean(raw.decode("gbk", errors="ignore"))


def _clean(text: str) -> str:
    """简单清洗：去空行，规范空白符号。"""
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    return "\n".join(lines)