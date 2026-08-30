"""Small text utilities shared by several layers.

Kept dependency-free and in one place so that the analysis modules and the
offline heuristic provider cannot drift apart in how they tokenise text or
look for linguistic markers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence

STOPWORDS = frozenset(
    {
        "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "em",
        "no", "na", "nos", "nas", "por", "para", "com", "sem", "que", "e", "ou",
        "se", "ao", "aos", "the", "of", "in", "on", "to", "and", "or", "is",
        "are", "be", "as", "at", "by", "an", "it", "this", "that", "was", "were",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])")


def strip_accents(text: str) -> str:
    """Lowercase ``text`` and remove diacritics."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def tokenize(text: str) -> List[str]:
    """Content words of ``text``: lowercase, unaccented, stopwords removed."""
    return [
        token
        for token in _TOKEN_RE.findall(strip_accents(text))
        if token not in STOPWORDS and len(token) > 2
    ]


def split_sentences(text: str) -> List[str]:
    """Split PT/EN technical prose into sentence-like units."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(cleaned) if part.strip()]


def overlap(a: str, b: str) -> float:
    """Fraction of ``a``'s content words that also occur in ``b``."""
    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def find_markers(text: str, markers: Sequence[str]) -> List[str]:
    """Return which ``markers`` occur in ``text`` as whole words.

    ASCII markers are matched against the accent-stripped text (so "porem"
    also matches "porém"), while accented markers are matched against the
    accent-preserving lowercase text — otherwise Portuguese "só" would match
    English "so".
    """
    lowered = str(text or "").lower()
    stripped = strip_accents(text)
    found: List[str] = []
    for marker in markers:
        marker_is_ascii = marker.isascii()
        haystack = stripped if marker_is_ascii else lowered
        needle = strip_accents(marker) if marker_is_ascii else marker.lower()
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
            found.append(marker)
    return found


# --------------------------------------------------------------------------- #
# Linguistic markers (shared by the analysis stages and the offline provider)
# --------------------------------------------------------------------------- #
UNIVERSAL_MARKERS: Sequence[str] = (
    "sempre", "nunca", "todos", "todas", "todo", "toda", "nenhum", "nenhuma",
    "necessariamente", "exclusivamente", "garante", "garantem", "garantido",
    "garantida", "impossível", "impossivel", "qualquer", "invariavelmente",
    "obrigatoriamente",
    "always", "never", "all", "none", "every", "necessarily", "exclusively",
    "guarantees", "guaranteed", "impossible", "any", "invariably",
)

CAUSAL_MARKERS: Sequence[str] = (
    "porque", "portanto", "logo", "causa", "causam", "causada", "causado",
    "provoca", "resulta em", "por isso", "devido a", "faz com que", "leva a",
    "consequentemente", "gera",
    "because", "therefore", "thus", "hence", "causes", "leads to", "due to",
    "results in", "consequently",
)

CONDITION_MARKERS: Sequence[str] = (
    "somente", "apenas", "só", "exceto", "depende", "dependendo", "desde que",
    "porém", "porem", "entretanto", "contudo", "no entanto", "limitado",
    "limitada", "restrito", "salvo", "sob certas condições",
    "em determinadas condições", "na ausência de", "trade-off", "compromisso",
    "only", "except", "depends", "however", "unless", "limited",
    "provided that", "whereas", "tradeoff", "under certain conditions",
)

HEDGE_MARKERS: Sequence[str] = (
    "pode", "podem", "talvez", "possivelmente", "provavelmente", "geralmente",
    "tipicamente", "em geral", "costuma",
    "may", "might", "possibly", "probably", "generally", "typically", "usually",
)
