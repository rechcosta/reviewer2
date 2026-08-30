"""Document loading: local files, directories and URLs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

from ..config import DocumentsConfig
from ..errors import DocumentError
from ..logging_utils import get_logger
from .parser import ParsedDocument, ParsedPage, parse_docx, parse_html, parse_pdf, parse_text

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm", ".rst", ".csv"}


def is_url(reference: str) -> bool:
    """True when ``reference`` looks like an http(s) URL."""
    parsed = urlparse(str(reference))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_document(reference: str, config: DocumentsConfig | None = None) -> ParsedDocument:
    """Load a single reference (path or URL) into a :class:`ParsedDocument`."""
    config = config or DocumentsConfig()
    if is_url(reference):
        return _load_url(reference, config)

    path = Path(reference).expanduser()
    if not path.exists():
        raise DocumentError(
            f"Reference document not found: {reference}",
            module="documents.loader",
            stage="load",
            cause="The path given with --reference does not exist.",
            action="Check the path, or place the file under data/documents/.",
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        document = parse_pdf(path, backend=config.pdf_backend)
    elif suffix in {".md", ".markdown", ".rst"}:
        document = parse_text(path, markdown=True)
    elif suffix in {".txt", ".csv"}:
        document = parse_text(path, markdown=False)
    elif suffix == ".docx":
        document = parse_docx(path)
    elif suffix in {".html", ".htm"}:
        document = parse_html(
            path.read_text(encoding="utf-8", errors="replace"),
            name=path.name,
            source=str(path),
            strip_tags=config.html_strip_tags,
        )
    else:
        raise DocumentError(
            f"Unsupported document format: '{suffix or path.name}'",
            module="documents.loader",
            stage="load",
            cause=f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))} and http(s) URLs.",
            action="Convert the file to PDF, TXT, Markdown, DOCX or HTML.",
        )

    if not document.pages:
        raise DocumentError(
            f"No text could be extracted from {path.name}.",
            module="documents.loader",
            stage="load",
            cause="The document may be a scanned image without a text layer.",
            action="Run OCR on the document (e.g. ocrmypdf) and try again.",
        )
    logger.info("Loaded %s (%d page(s)/block(s))", document.name, len(document.pages))
    return document


def load_documents(references: Iterable[str], config: DocumentsConfig | None = None) -> List[ParsedDocument]:
    """Load several references; directories are expanded to supported files."""
    config = config or DocumentsConfig()
    documents: List[ParsedDocument] = []
    for reference in references:
        if not is_url(reference) and Path(reference).expanduser().is_dir():
            directory = Path(reference).expanduser()
            files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            if not files:
                raise DocumentError(
                    f"No supported document found in directory: {directory}",
                    module="documents.loader",
                    stage="load",
                    cause=f"Looked for: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
                    action="Point --reference to a file, or add documents to the directory.",
                )
            for file_path in files:
                documents.append(load_document(str(file_path), config))
            continue
        documents.append(load_document(str(reference), config))
    return documents


def _load_url(url: str, config: DocumentsConfig) -> ParsedDocument:
    """Fetch a URL and parse it as HTML or PDF.

    Network access happens only for references the user explicitly passed.
    """
    try:
        import requests
    except ImportError as exc:
        raise DocumentError(
            "The 'requests' package is required to load URLs.",
            module="documents.loader",
            stage="fetch_url",
            cause="requests is not installed.",
            action="Install it with: pip install requests — or download the page and pass the local file.",
        ) from exc

    logger.info("Fetching reference URL: %s", url)
    try:
        response = requests.get(
            url, timeout=config.request_timeout, headers={"User-Agent": config.user_agent}
        )
        response.raise_for_status()
    except Exception as exc:
        raise DocumentError(
            f"Could not fetch the reference URL: {url}",
            module="documents.loader",
            stage="fetch_url",
            cause=str(exc)[:300],
            action="Check the URL and your connection, or download the file and pass the local path.",
        ) from exc

    content_type = response.headers.get("Content-Type", "").lower()
    name = Path(urlparse(url).path).name or urlparse(url).netloc

    if "application/pdf" in content_type or name.lower().endswith(".pdf"):
        cache_dir = Path(config.download_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / (name if name.lower().endswith(".pdf") else f"{name or 'download'}.pdf")
        local.write_bytes(response.content)
        document = parse_pdf(local, backend=config.pdf_backend)
        document.name = name or local.name
        document.source_path = url
        return document

    if "text/plain" in content_type:
        return ParsedDocument(
            name=name or url, source_path=url, pages=[ParsedPage(text=response.text.strip())]
        )

    return parse_html(response.text, name=name or url, source=url, strip_tags=config.html_strip_tags)
