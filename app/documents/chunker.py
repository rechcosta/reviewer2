"""Chunking of reference documents.

Chunks are sentence-aware and overlapping, and always carry the metadata
needed to cite them: document, page, section and a stable ``chunk_id``.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..config import DocumentsConfig
from ..logging_utils import get_logger
from ..models import DocumentChunk
from .parser import ParsedDocument, ParsedPage

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")


def split_sentences(text: str) -> List[str]:
    """Split a block of text into sentence-like units."""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part and part.strip()]
    return parts


def chunk_document(document: ParsedDocument, config: Optional[DocumentsConfig] = None) -> List[DocumentChunk]:
    """Split one parsed document into overlapping, citable chunks."""
    config = config or DocumentsConfig()
    chunks: List[DocumentChunk] = []
    stem = _slug(document.name)

    for page_index, page in enumerate(document.pages):
        page_chunks = _chunk_page(page, config)
        for order, (text, char_start, char_end) in enumerate(page_chunks):
            page_label = page.page if page.page is not None else page_index + 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{stem}_{page_label:04d}_{order:02d}",
                    document=document.name,
                    text=text,
                    page=page.page,
                    section=page.section,
                    char_start=char_start,
                    char_end=char_end,
                    source_path=document.source_path,
                )
            )
    logger.debug("Document %s produced %d chunks", document.name, len(chunks))
    return chunks


def chunk_documents(
    documents: List[ParsedDocument], config: Optional[DocumentsConfig] = None
) -> List[DocumentChunk]:
    """Chunk several documents, keeping chunk ids unique across the corpus."""
    config = config or DocumentsConfig()
    all_chunks: List[DocumentChunk] = []
    seen: dict[str, int] = {}
    for document in documents:
        for chunk in chunk_document(document, config):
            if chunk.chunk_id in seen:
                seen[chunk.chunk_id] += 1
                chunk.chunk_id = f"{chunk.chunk_id}_{seen[chunk.chunk_id]}"
            else:
                seen[chunk.chunk_id] = 0
            all_chunks.append(chunk)
    return all_chunks


def _chunk_page(page: ParsedPage, config: DocumentsConfig) -> List[tuple[str, int, int]]:
    """Greedy sentence packing with character overlap between chunks."""
    text = (page.text or "").strip()
    if not text:
        return []
    if len(text) <= config.chunk_size:
        return [(text, 0, len(text))]

    sentences = split_sentences(text)
    if not sentences:
        return [(text, 0, len(text))]

    chunks: List[tuple[str, int, int]] = []
    buffer: List[str] = []
    buffer_len = 0
    cursor = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if buffer and buffer_len + sentence_len > config.chunk_size:
            body = " ".join(buffer)
            start = text.find(buffer[0], max(0, cursor - len(body)))
            start = start if start >= 0 else cursor
            chunks.append((body, start, start + len(body)))
            cursor = start + len(body)
            buffer, buffer_len = _carry_overlap(buffer, config.chunk_overlap)
            buffer_len = sum(len(s) + 1 for s in buffer)
        buffer.append(sentence)
        buffer_len += sentence_len

    if buffer:
        body = " ".join(buffer)
        start = text.find(buffer[0], max(0, cursor - len(body)))
        start = start if start >= 0 else cursor
        # Avoid emitting a tiny trailing chunk: merge it into the previous one.
        if chunks and len(body) < config.min_chunk_size:
            previous, previous_start, _ = chunks[-1]
            merged = f"{previous} {body}".strip()
            chunks[-1] = (merged, previous_start, previous_start + len(merged))
        else:
            chunks.append((body, start, start + len(body)))
    return chunks


def _carry_overlap(buffer: List[str], overlap_chars: int) -> tuple[List[str], int]:
    """Keep the trailing sentences of a chunk as the head of the next one."""
    if overlap_chars <= 0:
        return [], 0
    carried: List[str] = []
    total = 0
    for sentence in reversed(buffer):
        if total + len(sentence) > overlap_chars and carried:
            break
        carried.insert(0, sentence)
        total += len(sentence) + 1
    return carried, total


def _slug(name: str) -> str:
    """Filesystem/id friendly version of a document name."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", str(name))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug or "doc"
