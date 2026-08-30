"""Document loading, parsing and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import DocumentsConfig
from app.documents import chunk_document, chunk_documents, load_document, load_documents
from app.documents.parser import ParsedDocument, ParsedPage, parse_html, parse_text
from app.errors import DocumentError


def test_parse_markdown_splits_by_heading(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Intro\n\nTexto um.\n\n## Métodos\n\nTexto dois.\n", encoding="utf-8")
    document = parse_text(path, markdown=True)
    assert [page.section for page in document.pages] == ["Intro", "Métodos"]
    assert document.pages[1].text == "Texto dois."


def test_parse_plain_text_is_single_block(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("linha um\nlinha dois\n", encoding="utf-8")
    document = parse_text(path)
    assert len(document.pages) == 1
    assert document.pages[0].section is None


def test_parse_text_joins_hyphenated_line_breaks(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("A memória vola-\ntil perde os dados.\n", encoding="utf-8")
    document = parse_text(path)
    assert "volatil perde" in document.pages[0].text


def test_parse_html_drops_scripts_and_keeps_sections() -> None:
    html = (
        "<html><head><title>T</title></head><body>"
        "<script>var x = 1;</script><h2>Seção</h2><p>Conteúdo relevante.</p>"
        "</body></html>"
    )
    document = parse_html(html, name="page.html", source="http://example.test")
    text = document.text
    assert "Conteúdo relevante." in text
    assert "var x" not in text


def test_chunker_preserves_provenance(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Methodology\n\n" + ("Frase de teste com conteúdo técnico. " * 60), encoding="utf-8")
    chunks = chunk_document(load_document(str(path)), DocumentsConfig(chunk_size=300, chunk_overlap=60))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document == "paper.md"
        assert chunk.section == "Methodology"
        assert chunk.chunk_id.startswith("paper_")
        assert chunk.text.strip()


def test_chunk_ids_are_unique_across_documents() -> None:
    documents = [
        ParsedDocument(name="a.md", source_path="a.md", pages=[ParsedPage(text="Um texto.", page=1)]),
        ParsedDocument(name="a.md", source_path="b/a.md", pages=[ParsedPage(text="Outro texto.", page=1)]),
    ]
    chunks = chunk_documents(documents)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunks_overlap(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    sentences = " ".join(f"Sentença número {i} sobre arquitetura de computadores." for i in range(40))
    path.write_text(sentences, encoding="utf-8")
    chunks = chunk_document(load_document(str(path)), DocumentsConfig(chunk_size=300, chunk_overlap=100))
    assert len(chunks) >= 2
    tail = set(chunks[0].text.split()[-6:])
    assert tail & set(chunks[1].text.split())


def test_small_document_is_one_chunk(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    path.write_text("Uma única frase curta.", encoding="utf-8")
    assert len(chunk_document(load_document(str(path)))) == 1


def test_locator_is_human_readable() -> None:
    chunks = chunk_documents(
        [ParsedDocument(name="p.pdf", source_path="p.pdf", pages=[ParsedPage(text="Texto.", page=12, section="Methods")])]
    )
    assert chunks[0].locator == "p.pdf, página 12, Methods"


def test_missing_document_reports_action(tmp_path: Path) -> None:
    with pytest.raises(DocumentError) as excinfo:
        load_document(str(tmp_path / "missing.pdf"))
    assert "recommended action" in excinfo.value.format()


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.xyz"
    path.write_text("conteúdo", encoding="utf-8")
    with pytest.raises(DocumentError) as excinfo:
        load_document(str(path))
    assert "Unsupported document format" in excinfo.value.message


def test_load_documents_expands_directories(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nTexto A.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Texto B.", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00")
    documents = load_documents([str(tmp_path)])
    assert sorted(document.name for document in documents) == ["a.md", "b.txt"]


def test_empty_directory_reports_action(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(DocumentError):
        load_documents([str(tmp_path / "empty")])
