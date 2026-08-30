"""Format-specific text extraction.

Every parser returns a list of :class:`ParsedPage` so that page numbers and
sections survive all the way to the report — provenance is never lost.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import DocumentError, MissingDependencyError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """One page (PDF) or one logical block (text formats) of a document."""

    text: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class ParsedDocument:
    """A reference document after text extraction."""

    name: str
    source_path: str
    pages: List[ParsedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def parse_pdf(path: Path, *, backend: str = "auto") -> ParsedDocument:
    """Extract text from a PDF, one :class:`ParsedPage` per page."""
    path = Path(path)
    errors: List[str] = []

    if backend in {"auto", "pymupdf"}:
        try:
            return _parse_pdf_pymupdf(path)
        except MissingDependencyError as exc:
            errors.append(str(exc.message))
        except Exception as exc:  # pragma: no cover - depends on the file
            errors.append(f"pymupdf: {exc}")

    if backend in {"auto", "pypdf"}:
        try:
            return _parse_pdf_pypdf(path)
        except MissingDependencyError as exc:
            errors.append(str(exc.message))
        except Exception as exc:  # pragma: no cover
            errors.append(f"pypdf: {exc}")

    raise DocumentError(
        f"Could not extract text from PDF: {path.name}",
        module="documents.parser",
        stage="parse_pdf",
        cause="; ".join(errors) or "no PDF backend available",
        action="Install a PDF backend: pip install pymupdf  (or: pip install pypdf)",
    )


def _parse_pdf_pymupdf(path: Path) -> ParsedDocument:
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError(
                "pymupdf", module="documents.parser", stage="parse_pdf", extra="PDF text extraction"
            ) from exc

    document = pymupdf.open(str(path))
    try:
        pages: List[ParsedPage] = []
        outline = _pymupdf_sections(document)
        for index in range(document.page_count):
            text = document.load_page(index).get_text("text")
            text = _clean_text(text)
            if not text:
                continue
            pages.append(ParsedPage(text=text, page=index + 1, section=outline.get(index + 1)))
        metadata = dict(document.metadata or {})
    finally:
        document.close()
    return ParsedDocument(name=path.name, source_path=str(path), pages=pages, metadata=metadata)


def _pymupdf_sections(document) -> Dict[int, str]:
    """Map page number -> nearest table-of-contents title, when available."""
    sections: Dict[int, str] = {}
    try:
        toc = document.get_toc() or []
    except Exception:  # pragma: no cover
        return sections
    current: Optional[str] = None
    entries = sorted(((int(page), str(title).strip()) for _, title, page in toc if page), key=lambda x: x[0])
    cursor = 0
    for page_number in range(1, document.page_count + 1):
        while cursor < len(entries) and entries[cursor][0] <= page_number:
            current = entries[cursor][1]
            cursor += 1
        if current:
            sections[page_number] = current
    return sections


def _parse_pdf_pypdf(path: Path) -> ParsedDocument:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "pypdf", module="documents.parser", stage="parse_pdf", extra="PDF text extraction"
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append(ParsedPage(text=text, page=index + 1))
    metadata = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    return ParsedDocument(name=path.name, source_path=str(path), pages=pages, metadata=metadata)


# --------------------------------------------------------------------------- #
# Plain text / Markdown
# --------------------------------------------------------------------------- #
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_text(path: Path, *, markdown: bool = False) -> ParsedDocument:
    """Parse TXT/Markdown, using Markdown headings as section boundaries."""
    path = Path(path)
    raw = _read_text_file(path)
    if not markdown:
        return ParsedDocument(
            name=path.name, source_path=str(path), pages=[ParsedPage(text=_clean_text(raw))]
        )

    pages: List[ParsedPage] = []
    section: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        text = _clean_text("\n".join(buffer))
        if text:
            pages.append(ParsedPage(text=text, section=section))
        buffer.clear()

    for line in raw.splitlines():
        heading = _MD_HEADING.match(line.strip())
        if heading:
            flush()
            section = heading.group(2).strip()
            continue
        buffer.append(line)
    flush()

    if not pages:
        pages = [ParsedPage(text=_clean_text(raw))]
    return ParsedDocument(name=path.name, source_path=str(path), pages=pages)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def parse_docx(path: Path) -> ParsedDocument:
    """Extract paragraphs and tables from a .docx, split by heading styles."""
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "python-docx", module="documents.parser", stage="parse_docx", extra="DOCX support"
        ) from exc

    path = Path(path)
    document = docx.Document(str(path))
    pages: List[ParsedPage] = []
    section: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        text = _clean_text("\n".join(buffer))
        if text:
            pages.append(ParsedPage(text=text, section=section))
        buffer.clear()

    for paragraph in document.paragraphs:
        style = (paragraph.style.name or "") if paragraph.style else ""
        text = paragraph.text.strip()
        if not text:
            continue
        if style.lower().startswith("heading") or style.lower().startswith("título"):
            flush()
            section = text
            continue
        buffer.append(text)
    flush()

    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        text = _clean_text("\n".join(rows))
        if text:
            pages.append(ParsedPage(text=text, section=(section or "") + " (tabela)" if section else "tabela"))

    return ParsedDocument(name=path.name, source_path=str(path), pages=pages)


# --------------------------------------------------------------------------- #
# HTML / URL
# --------------------------------------------------------------------------- #
def parse_html(raw_html: str, *, name: str, source: str, strip_tags: Optional[List[str]] = None) -> ParsedDocument:
    """Convert HTML into sectioned text, preferring BeautifulSoup when present."""
    strip_tags = strip_tags or ["script", "style", "nav", "footer", "header"]
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag_name in strip_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else None

        pages: List[ParsedPage] = []
        section = title
        buffer: List[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td"]):
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if element.name in {"h1", "h2", "h3", "h4"}:
                block = _clean_text("\n".join(buffer))
                if block:
                    pages.append(ParsedPage(text=block, section=section))
                buffer = []
                section = text
                continue
            buffer.append(text)
        block = _clean_text("\n".join(buffer))
        if block:
            pages.append(ParsedPage(text=block, section=section))
        if not pages:
            pages = [ParsedPage(text=_clean_text(soup.get_text(" ", strip=True)), section=title)]
        return ParsedDocument(name=name, source_path=source, pages=pages, metadata={"title": title or ""})
    except ImportError:
        logger.warning("beautifulsoup4 is not installed; falling back to a simple HTML stripper.")
        return ParsedDocument(
            name=name, source_path=source, pages=[ParsedPage(text=_strip_html(raw_html, strip_tags))]
        )


def _strip_html(raw_html: str, strip_tags: List[str]) -> str:
    """Dependency-free HTML to text conversion."""
    text = raw_html
    for tag in strip_tags:
        text = re.sub(rf"<{tag}\b.*?</{tag}>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean_text(html.unescape(text))


# --------------------------------------------------------------------------- #
def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentError(
        f"Could not decode text file: {path.name}",
        module="documents.parser",
        stage="read",
        cause="The file is not valid UTF-8 nor Latin-1.",
        action="Convert the file to UTF-8 and try again.",
    )


def _clean_text(text: str) -> str:
    """Normalise whitespace, join hyphenated line breaks, drop control chars."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-zà-ÿ])", "", text)          # hyphenated word wrap
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()
