"""Web interface strings in Portuguese and English.

The interface follows ``report.language``, so a run configured to write an
English report also serves an English page.
"""

from __future__ import annotations

from typing import Dict

PT: Dict[str, str] = {
    "html_lang": "pt-BR",
    "title": "Reviewer2",
    "tagline": "Revisor técnico independente — cada crítica é ancorada em uma citação verificável das fontes.",
    "checking_backends": "verificando backends…",
    "video_label": "Vídeo ou áudio",
    "video_hint": "mp4, mkv, mov, webm, mp3, wav…",
    "choose_file": "Escolher arquivo",
    "choose_files": "Escolher arquivos",
    "no_file": "nenhum arquivo escolhido",
    "n_files": "{count} arquivos escolhidos",
    "references_label": "Materiais de referência",
    "references_hint": "PDF, TXT, Markdown, DOCX, HTML — pode selecionar vários",
    "urls_label": "URLs de referência (uma por linha)",
    "submit": "Revisar",
    "offline_label": "modo offline (sem LLM, execução seca)",
    "queued": "na fila…",
    "report_heading": "Relatório",
    "download": "baixar .md",
    "download_filename": "revisao.md",
    # Strings the page's own script writes.
    "llm_unavailable": "LLM indisponível",
    "no_optional_backends": "nenhum opcional",
    "server_silent": "servidor sem resposta",
    "submit_failed": "falha ao enviar",
    "done": "concluído",
    "stat_analysed": "afirmações analisadas",
    "stat_problems": "problemas",
    "stat_coverage": "cobertura de evidência",
    "stat_dropped": "críticas descartadas",
    # Pipeline stages, shown on the progress bar.
    "stage_transcription": "transcrição",
    "stage_indexing": "indexação das referências",
    "stage_claims": "extração de afirmações",
    "stage_critique": "recuperação e crítica",
    "stage_verification": "verificação",
    "stage_report": "relatório",
    # Messages the API returns to the page.
    "err_unnamed_file": "Arquivo de {label} sem nome.",
    "err_unsupported": "Formato não suportado para {label}: '{suffix}'. Aceitos: {accepted}",
    "err_invalid_url": "URL inválida: {url}",
    "err_no_reference": "É necessário pelo menos um material de referência (arquivo ou URL).",
    "err_report_not_ready": "O relatório ainda não está pronto.",
    "err_job_not_found": "Job não encontrado.",
    "err_unexpected": "Erro inesperado: {error}",
    "label_video": "vídeo",
    "label_reference": "referência",
}

EN: Dict[str, str] = {
    "html_lang": "en",
    "title": "Reviewer2",
    "tagline": "An independent technical reviewer — every critique is anchored to a verifiable quote from your sources.",
    "checking_backends": "checking backends…",
    "video_label": "Video or audio",
    "video_hint": "mp4, mkv, mov, webm, mp3, wav…",
    "choose_file": "Choose file",
    "choose_files": "Choose files",
    "no_file": "no file selected",
    "n_files": "{count} files selected",
    "references_label": "Reference material",
    "references_hint": "PDF, TXT, Markdown, DOCX, HTML — you can pick several",
    "urls_label": "Reference URLs (one per line)",
    "submit": "Review",
    "offline_label": "offline mode (no LLM, dry run)",
    "queued": "queued…",
    "report_heading": "Report",
    "download": "download .md",
    "download_filename": "review.md",
    "llm_unavailable": "LLM unavailable",
    "no_optional_backends": "no optional ones",
    "server_silent": "no response from the server",
    "submit_failed": "upload failed",
    "done": "done",
    "stat_analysed": "claims analysed",
    "stat_problems": "problems",
    "stat_coverage": "evidence coverage",
    "stat_dropped": "critiques dropped",
    "stage_transcription": "transcription",
    "stage_indexing": "indexing the references",
    "stage_claims": "claim extraction",
    "stage_critique": "retrieval and critique",
    "stage_verification": "verification",
    "stage_report": "report",
    "err_unnamed_file": "The {label} file has no name.",
    "err_unsupported": "Unsupported format for the {label}: '{suffix}'. Accepted: {accepted}",
    "err_invalid_url": "Invalid URL: {url}",
    "err_no_reference": "At least one reference material is required (a file or a URL).",
    "err_report_not_ready": "The report is not ready yet.",
    "err_job_not_found": "Job not found.",
    "err_unexpected": "Unexpected error: {error}",
    "label_video": "video",
    "label_reference": "reference",
}


def ui_strings(language: str) -> Dict[str, str]:
    """Return the interface string table for ``language`` (defaults to Portuguese)."""
    return EN if str(language).lower().startswith("en") else PT
