"""Internal contradiction detection.

Candidate pairs are found cheaply with embeddings (claims that talk about the
same thing), and only those pairs are submitted to the model, which must
first rule out differences of sense, context, time or conditions before
declaring a contradiction.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np

from ..config import AnalysisConfig
from ..embeddings import EmbeddingModel
from ..errors import LLMError
from ..llm import LLMProvider, as_dict
from ..logging_utils import get_logger
from ..models import Claim, ConfidenceLevel, InternalContradiction, Severity
from ..prompts import SYSTEM_PROMPT, internal_contradiction_prompt
from ..text_utils import tokenize

logger = get_logger(__name__)


class ContradictionDetector:
    """Finds claims of the same video that cannot both be true."""

    def __init__(
        self,
        llm: LLMProvider,
        embeddings: EmbeddingModel,
        config: Optional[AnalysisConfig] = None,
        *,
        concurrency: int = 1,
    ) -> None:
        self.concurrency = max(1, int(concurrency))
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or AnalysisConfig()
        # Lexical (hashing) embeddings produce systematically lower cosine
        # values than semantic ones, so the candidate threshold adapts.
        self.similarity_threshold = (
            self.config.contradiction_similarity_threshold_lexical
            if getattr(embeddings, "name", "") == "hashing"
            else self.config.contradiction_similarity_threshold
        )

    def detect(self, claims: Sequence[Claim]) -> List[InternalContradiction]:
        """Return the contradictions confirmed among ``claims``."""
        claims = list(claims)
        if len(claims) < 2:
            return []

        pairs = self.candidate_pairs(claims)
        logger.debug("Checking %d candidate contradiction pairs", len(pairs))

        def check(item) -> Optional[InternalContradiction]:
            index, (first, second, similarity) = item
            return self._check_pair(first, second, similarity, index + 1)

        if self.concurrency == 1 or len(pairs) <= 1:
            results = [check(item) for item in enumerate(pairs)]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="contradiction") as pool:
                results = list(pool.map(check, list(enumerate(pairs))))

        contradictions = [r for r in results if r is not None and r.is_contradiction]
        logger.info("Confirmed %d internal contradiction(s)", len(contradictions))
        return contradictions

    # ------------------------------------------------------------------ #
    def candidate_pairs(self, claims: Sequence[Claim]) -> List[tuple[Claim, Claim, float]]:
        """Pairs of claims worth checking for a contradiction.

        Two independent signals are combined, because either alone misses
        real cases: embedding similarity (the claims mean similar things)
        and a shared distinctive topic word (the claims are about the same
        object, even when phrased very differently — which is exactly the
        shape of "cache is non-volatile" vs. "cache data is lost on power
        off").
        """
        vectors = self.embeddings.encode([claim.text for claim in claims])
        distinctive = _distinctive_tokens(claims)
        pairs: List[tuple[Claim, Claim, float]] = []

        for i, j in itertools.combinations(range(len(claims)), 2):
            similarity = float(np.dot(vectors[i], vectors[j]))
            shares_topic = bool(distinctive[i] & distinctive[j])
            if similarity >= self.similarity_threshold or shares_topic:
                pairs.append((claims[i], claims[j], similarity))

        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs[: self.config.max_contradiction_pairs]

    def _check_pair(
        self, first: Claim, second: Claim, similarity: float, index: int
    ) -> Optional[InternalContradiction]:
        prompt = internal_contradiction_prompt(
            first.text,
            second.text,
            context_a=first.context,
            context_b=second.context,
            language=self.config.language,
        )
        try:
            raw = as_dict(self.llm.generate_json(prompt, system=SYSTEM_PROMPT, expect="object"))
        except LLMError as exc:
            logger.debug("Contradiction check failed for %s/%s: %s", first.claim_id, second.claim_id, exc.message)
            return None

        is_contradiction = bool(raw.get("is_contradiction", False))
        # A resolving condition means the inconsistency is only apparent.
        resolving = str(raw.get("resolving_condition", "")).strip()
        if resolving:
            is_contradiction = False
        if not bool(raw.get("same_sense_of_terms", True)):
            is_contradiction = False

        return InternalContradiction(
            contradiction_id=f"contradiction_{index:03d}",
            claim_a=first,
            claim_b=second,
            is_contradiction=is_contradiction,
            explanation=str(raw.get("explanation", "")).strip(),
            same_sense_of_terms=bool(raw.get("same_sense_of_terms", True)),
            context_difference=str(raw.get("context_difference", "")).strip(),
            temporal_difference=str(raw.get("temporal_difference", "")).strip(),
            resolving_condition=resolving,
            confidence=ConfidenceLevel.parse(raw.get("confidence"), ConfidenceLevel.LOW),
            severity=Severity.parse(raw.get("severity"), Severity.MEDIUM),
            similarity=round(similarity, 3),
        )


def _distinctive_tokens(claims: Sequence[Claim], *, max_document_frequency: float = 0.4) -> List[set]:
    """Content words of each claim that are not spread across the whole video.

    A word shared by most claims (the video's overall subject) carries no
    information about which pairs are worth comparing; a word occurring in a
    minority of claims identifies the specific object being discussed.
    """
    token_sets = [set(tokenize(claim.text)) for claim in claims]
    if not token_sets:
        return []
    frequency: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1

    total = len(token_sets)
    limit = max(1, int(max_document_frequency * total))
    return [
        {token for token in tokens if 2 <= frequency[token] <= limit}
        for tokens in token_sets
    ]
