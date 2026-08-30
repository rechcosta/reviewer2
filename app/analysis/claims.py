"""Claim extraction.

The transcript is turned into standalone, analysable assertions. Every claim
keeps its timestamp, the transcript confidence of the segments it came from,
and the linguistic markers that later stages care about.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ..config import AnalysisConfig
from ..errors import LLMError
from ..llm import LLMProvider, as_list
from ..logging_utils import get_logger
from ..models import Claim, Importance, Transcript, TranscriptWindow, format_timestamp
from ..prompts import SYSTEM_PROMPT, claim_extraction_prompt
from .markers import find_causal_markers, find_universal_markers, strip_accents
from .segmentation import build_windows

logger = get_logger(__name__)


class ClaimExtractor:
    """Extracts technical claims from a transcript, window by window."""

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[AnalysisConfig] = None,
        *,
        low_confidence_threshold: float = 0.6,
        concurrency: int = 1,
    ) -> None:
        self.llm = llm
        self.config = config or AnalysisConfig()
        self.low_confidence_threshold = low_confidence_threshold
        self.concurrency = concurrency

    def extract(self, transcript: Transcript) -> List[Claim]:
        """Return every claim found in ``transcript``.

        Windows are independent, so they are processed in parallel when
        ``llm.concurrency`` allows it. Claim ids are assigned after all
        windows finish, in transcript order.
        """
        windows = build_windows(transcript, self.config)
        total = len(windows)
        workers = max(1, int(self.concurrency))
        per_window: List[List[Claim]] = [[] for _ in range(total)]

        def work(position: int) -> None:
            window = windows[position]
            previous = windows[position - 1].text if position else ""
            try:
                per_window[position] = self._extract_window(window, transcript, previous)
            except LLMError as exc:
                # A failing window must not abort the whole review.
                logger.warning("Claim extraction failed for window %d: %s", window.window_id, exc.message)

        if workers == 1 or total <= 1:
            for position in range(total):
                work(position)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="claims") as pool:
                list(pool.map(work, range(total)))

        # Logged after the join, in transcript order: workers finish out of
        # order, and an out-of-order progress log reads like lost control.
        for position, window in enumerate(windows):
            logger.info(
                "Extracted %d claim(s) from window %d/%d (%s)",
                len(per_window[position]), position + 1, total, window.timestamp,
            )

        claims = [claim for window_claims in per_window for claim in window_claims]
        deduped = _deduplicate(claims)
        logger.info("Extracted %d claims", len(deduped))
        return deduped

    # ------------------------------------------------------------------ #
    def _extract_window(
        self, window: TranscriptWindow, transcript: Transcript, context_before: str
    ) -> List[Claim]:
        prompt = claim_extraction_prompt(
            window.text,
            context_before=context_before,
            max_claims=self.config.max_claims_per_window,
            min_chars=self.config.min_claim_chars,
            language=self.config.language,
        )
        payload = self.llm.generate_json(prompt, system=SYSTEM_PROMPT, expect="object")
        raw_claims = as_list(payload, key="claims")

        claims: List[Claim] = []
        for index, item in enumerate(raw_claims):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if len(text) < self.config.min_claim_chars:
                continue
            verbatim = str(item.get("verbatim", "")).strip()
            start, end = _locate(verbatim or text, window, transcript)
            claims.append(
                Claim(
                    claim_id=f"w{window.window_id:03d}_{len(claims) + 1:02d}",
                    text=text,
                    timestamp=format_timestamp(start),
                    start=start,
                    end=end,
                    topic=str(item.get("topic", "")).strip(),
                    context=str(item.get("context", "")).strip() or window.text[:400],
                    importance=Importance.parse(item.get("importance")),
                    segment_ids=list(window.segment_ids),
                    verbatim=verbatim if verbatim and verbatim in window.text else "",
                    transcript_confidence=round(window.mean_confidence, 3),
                    low_confidence_transcript=window.mean_confidence < self.low_confidence_threshold,
                    universal_markers=find_universal_markers(text),
                    causal_markers=find_causal_markers(text),
                )
            )
            if len(claims) >= self.config.max_claims_per_window:
                break
        return claims


def _locate(quote: str, window: TranscriptWindow, transcript: Transcript) -> tuple[float, float]:
    """Find the tightest timestamp range for a claim inside its window.

    Falls back to the window boundaries — a timestamp is never invented.
    """
    if not quote:
        return window.start, window.end
    needle = strip_accents(quote)[:60]
    segments = [transcript.segment_by_id(sid) for sid in window.segment_ids]
    segments = [segment for segment in segments if segment is not None]

    for segment in segments:
        if needle and needle[:30] in strip_accents(segment.text):
            return segment.start, segment.end

    # The claim may span consecutive segments: locate the first one whose text
    # shares the opening words of the quote.
    head = needle.split(" ")[:4]
    if head:
        prefix = " ".join(head)
        for index, segment in enumerate(segments):
            if prefix and prefix in strip_accents(segment.text):
                end = segments[min(index + 1, len(segments) - 1)].end
                return segment.start, max(end, segment.end)
    return window.start, window.end


def _deduplicate(claims: List[Claim]) -> List[Claim]:
    """Drop claims repeated verbatim, keeping the first occurrence."""
    seen: set[str] = set()
    unique: List[Claim] = []
    for claim in claims:
        key = " ".join(strip_accents(claim.text).split())
        if key in seen:
            continue
        seen.add(key)
        claim.claim_id = f"claim_{len(unique) + 1:03d}"
        unique.append(claim)
    return unique
