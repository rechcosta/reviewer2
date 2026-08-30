"""Aggregation of omissions detected while critiquing claims.

The detection itself happens in :mod:`app.analysis.critique` (it reuses the
evidence already retrieved for the claim). This module collects, ranks and
summarises the results.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..models import Critique, ImpactLevel, Omission

_IMPACT_RANK: Dict[ImpactLevel, int] = {
    ImpactLevel.HIGH: 0,
    ImpactLevel.MEDIUM: 1,
    ImpactLevel.LOW: 2,
}


def collect_omissions(critiques: Sequence[Critique]) -> List[Omission]:
    """Return every omission of the active critiques, most severe first."""
    omissions: List[Omission] = []
    for critique in critiques:
        if critique.dropped:
            continue
        omissions.extend(critique.omissions)
    return sorted(omissions, key=lambda omission: _IMPACT_RANK.get(omission.impact, 1))


def count_by_impact(omissions: Sequence[Omission]) -> Dict[str, int]:
    """Histogram of omissions per impact level."""
    counts: Dict[str, int] = {}
    for omission in omissions:
        counts[omission.impact.value] = counts.get(omission.impact.value, 0) + 1
    return counts
