"""Pipeline orchestration.

    video → audio → ASR → transcript → windows → claims
                                                   ↓
    references → chunks → embeddings → vector store → retrieval
                                                   ↓
                                        critique → verification → report
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .analysis import ClaimExtractor, ContradictionDetector, CritiqueEngine
from .config import Config
from .embeddings import HashingEmbeddings, build_embedding_model
from .errors import Reviewer2Error
from .llm import HeuristicProvider, LLMProvider, build_llm
from .logging_utils import get_logger
from .models import Claim, Critique, ReviewReport, Transcript
from .reports import ReportGenerator, build_statistics
from .retrieval import EvidenceRetriever
from .transcription import load_transcript, transcribe_media
from .verification import DevilsAdvocate

logger = get_logger(__name__)

#: The pipeline stages, in order. Consumers (the web UI, progress bars) use
#: these names instead of parsing log lines.
STAGES: tuple[str, ...] = (
    "transcription",
    "indexing",
    "claims",
    "critique",
    "verification",
    "report",
)

#: ``callback(stage_index, stage_name)`` — called as each stage starts.
ProgressCallback = Callable[[int, str], None]


class ReviewPipeline:
    """Runs the complete review of one video against its reference material."""

    def __init__(
        self,
        config: Config,
        *,
        llm: Optional[LLMProvider] = None,
        retriever: Optional[EvidenceRetriever] = None,
    ) -> None:
        self.config = config
        self.llm = llm or build_llm(
            config.llm, cache_dir=Path(config.paths.cache_dir) / "llm"
        )
        self.retriever = retriever or EvidenceRetriever(config)
        self.limitations: List[str] = []
        self._on_stage: Optional[ProgressCallback] = None

    # ------------------------------------------------------------------ #
    def run(
        self,
        video: str,
        references: Sequence[str],
        *,
        transcript_path: Optional[Path] = None,
        force_transcription: bool = False,
        rebuild_index: bool = False,
        output: Optional[Path] = None,
        on_stage: Optional[ProgressCallback] = None,
    ) -> ReviewReport:
        """Execute every stage and write the final report.

        ``on_stage`` is invoked with ``(index, name)`` as each stage begins,
        so callers can report progress without parsing log output.
        """
        started = time.perf_counter()
        self.config.paths.ensure()
        self._on_stage = on_stage

        self._stage(0)
        logger.info("Loading video")
        transcript = self._transcribe(video, transcript_path, force_transcription)

        self._stage(1)
        logger.info("Indexing reference documents")
        chunk_count = self._index(references, rebuild_index)

        self._stage(2)
        logger.info("Extracting claims")
        claims = self._extract_claims(transcript)

        self._stage(3)
        logger.info("Reviewing claims")
        critiques = self._review(claims)

        self._stage(4)
        logger.info("Running verification")
        critiques = DevilsAdvocate(
            self.llm, self.config.verification, concurrency=self.config.llm.concurrency
        ).verify_all(critiques)

        contradictions = []
        if self.config.analysis.detect_internal_contradictions and len(claims) > 1:
            logger.info("Checking internal contradictions")
            detector = ContradictionDetector(
                self.llm, self.retriever.embeddings, self.config.analysis,
                concurrency=self.config.llm.concurrency,
            )
            contradictions = detector.detect(claims)

        self._stage(5)
        reused = getattr(self.llm, "summary", lambda: "")()
        if reused:
            logger.info("Cache: %s", reused)

        logger.info("Generating report")
        report = ReviewReport(
            video=str(video),
            references=[str(reference) for reference in references],
            transcript=transcript,
            claims=claims,
            critiques=critiques,
            internal_contradictions=contradictions,
            llm_model=f"{self.llm.name}/{self.llm.model}",
            asr_model=transcript.model,
            embedding_model=getattr(self.retriever.embeddings, "model_name", self.retriever.embeddings.name),
            vector_store=self.retriever.store.name,
            duration_seconds=round(time.perf_counter() - started, 2),
        )
        report.statistics.total_chunks = chunk_count
        report.statistics = build_statistics(report)
        report.limitations = self._limitations(report)

        output_path = Path(output) if output else self._default_output(video)
        generator = ReportGenerator(self.config.report)
        generator.write(report, output_path)
        logger.info("Report written to %s", output_path)

        if self.config.report.include_audit_json:
            trace_path = output_path.with_name(f"{output_path.stem}_retrieval.json")
            self.retriever.save_traces(trace_path)
            logger.info("Retrieval trace written to %s", trace_path)

        return report

    def _stage(self, index: int) -> None:
        """Announce the start of a pipeline stage to the progress callback."""
        if self._on_stage is not None:
            try:
                self._on_stage(index, STAGES[index])
            except Exception:  # a broken callback must not abort the review
                logger.debug("Progress callback raised", exc_info=True)

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def _transcribe(
        self, video: str, transcript_path: Optional[Path], force: bool
    ) -> Transcript:
        """Transcribe the media, or load a transcript supplied by the user."""
        if transcript_path is not None and Path(transcript_path).exists() and not force:
            logger.info("Using the transcript provided: %s", transcript_path)
            transcript = load_transcript(Path(transcript_path))
            if not transcript.source:
                transcript.source = str(video)
            return transcript

        logger.info("Extracting audio")
        return transcribe_media(
            Path(video), self.config, force=force, transcript_path=transcript_path
        )

    def _index(self, references: Sequence[str], rebuild: bool) -> int:
        """Load, chunk, embed and index every reference document."""
        if not references:
            raise Reviewer2Error(
                "No reference material was provided.",
                module="pipeline",
                stage="index",
                cause="Reviewer2 cannot audit an explanation without sources to check it against.",
                action="Pass at least one --reference file, directory or URL.",
            )
        return self.retriever.index_references(references, rebuild=rebuild)

    def _extract_claims(self, transcript: Transcript) -> List[Claim]:
        extractor = ClaimExtractor(
            self.llm,
            self.config.analysis,
            low_confidence_threshold=self.config.transcription.low_confidence_threshold,
            concurrency=self.config.llm.concurrency,
        )
        return extractor.extract(transcript)

    def _review(self, claims: Sequence[Claim]) -> List[Critique]:
        engine = CritiqueEngine(self.llm, self.retriever, self.config)
        logger.info("Running retrieval")
        critiques = engine.review_claims(claims)
        logger.info("Produced %d critique(s)", len(critiques))
        return critiques

    # ------------------------------------------------------------------ #
    def _default_output(self, video: str) -> Path:
        stem = Path(video).stem or "review"
        return Path(self.config.paths.reports_dir) / f"{stem}{self.config.report.filename_suffix}.md"

    def _limitations(self, report: ReviewReport) -> List[str]:
        """State plainly what could weaken this analysis."""
        limitations = list(self.limitations)
        limitations.append(
            "A análise considera apenas os materiais de referência fornecidos; "
            "afirmações fora do escopo dessas fontes não podem ser verificadas."
        )
        if isinstance(self.llm, HeuristicProvider):
            limitations.append(
                "Execução em modo heurístico offline (sem modelo de linguagem): as classificações "
                "seguem regras lexicais simples e devem ser tratadas como indicativas, não conclusivas."
            )
        if isinstance(self.retriever.embeddings, HashingEmbeddings):
            limitations.append(
                "Embeddings lexicais (hashing) foram usados no lugar de embeddings semânticos: "
                "a recuperação pode falhar quando vídeo e fontes usam vocabulário ou idiomas diferentes."
            )
        if report.transcript is not None:
            ratio = report.transcript.low_confidence_ratio
            if ratio > 0.1:
                limitations.append(
                    f"{ratio:.0%} dos segmentos da transcrição têm baixa confiança; "
                    "erros de transcrição podem ter sido interpretados como erros técnicos."
                )
        if report.statistics.evidence_coverage < 1.0:
            limitations.append(
                f"{1 - report.statistics.evidence_coverage:.0%} das críticas não puderam ser "
                "acompanhadas de citação verificável e foram rebaixadas."
            )
        not_supported = report.statistics.by_classification.get("NAO_SUSTENTADA", 0)
        if not_supported:
            limitations.append(
                f"{not_supported} afirmação(ões) não puderam ser determinadas com as evidências disponíveis."
            )
        if report.statistics.dropped_critiques:
            limitations.append(
                f"{report.statistics.dropped_critiques} crítica(s) foram descartadas na verificação "
                "por não se sustentarem — elas não aparecem no relatório."
            )
        return limitations


def build_pipeline(config: Config, *, offline: bool = False) -> ReviewPipeline:
    """Convenience factory used by the CLI and the tests."""
    if offline:
        llm: LLMProvider = HeuristicProvider()
        retriever = EvidenceRetriever(
            config, embedding_model=build_embedding_model(config.embeddings)
        )
        return ReviewPipeline(config, llm=llm, retriever=retriever)
    return ReviewPipeline(config)
