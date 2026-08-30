"""Analysis layer: segmentation, claims, critique, contradictions, omissions."""

from .claims import ClaimExtractor
from .confidence import apply_adjustment, calibrate
from .contradictions import ContradictionDetector
from .critique import CritiqueEngine, verify_quote
from .markers import (
    find_causal_markers,
    find_condition_markers,
    find_hedge_markers,
    find_universal_markers,
)
from .omissions import collect_omissions, count_by_impact
from .segmentation import build_windows

__all__ = [
    "ClaimExtractor",
    "CritiqueEngine",
    "ContradictionDetector",
    "build_windows",
    "calibrate",
    "apply_adjustment",
    "collect_omissions",
    "count_by_impact",
    "verify_quote",
    "find_universal_markers",
    "find_causal_markers",
    "find_condition_markers",
    "find_hedge_markers",
]
