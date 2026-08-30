"""Linguistic markers used across the analysis stages.

These are *signals*, never verdicts: an absolute word is only a problem when
the evidence fails to support a universal statement (project spec, sections
16 and 17), and a causal connective is only a problem when the source does
not demonstrate causality (section 17).
"""

from __future__ import annotations

from typing import List

from ..text_utils import (
    CAUSAL_MARKERS,
    CONDITION_MARKERS,
    HEDGE_MARKERS,
    UNIVERSAL_MARKERS,
    find_markers,
    strip_accents,
)

__all__ = [
    "UNIVERSAL_MARKERS",
    "CAUSAL_MARKERS",
    "CONDITION_MARKERS",
    "HEDGE_MARKERS",
    "find_markers",
    "find_universal_markers",
    "find_causal_markers",
    "find_condition_markers",
    "find_hedge_markers",
    "strip_accents",
]


def find_universal_markers(text: str) -> List[str]:
    """Absolute quantifiers/modals present in ``text``."""
    return find_markers(text, UNIVERSAL_MARKERS)


def find_causal_markers(text: str) -> List[str]:
    """Causal connectives present in ``text``."""
    return find_markers(text, CAUSAL_MARKERS)


def find_condition_markers(text: str) -> List[str]:
    """Conditional/limiting expressions present in ``text``."""
    return find_markers(text, CONDITION_MARKERS)


def find_hedge_markers(text: str) -> List[str]:
    """Hedging expressions present in ``text``."""
    return find_markers(text, HEDGE_MARKERS)
