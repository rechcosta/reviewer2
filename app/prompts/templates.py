"""Prompt templates for every LLM task in the pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

TASK_MARKER = "REVIEWER2_TASK:"

SYSTEM_PROMPT = """You are Reviewer2, an independent technical reviewer.

Absolute rules:
1. EVIDENCE OVER CONFIDENCE. Judge only from the evidence given to you.
2. NEVER invent sources, pages, citations, timestamps, results or facts.
3. Quotes must be copied VERBATIM from the provided excerpts. If you cannot
   copy an exact quote, return an empty quote.
4. Absence of evidence is NOT falsehood. When the evidence is insufficient,
   say so explicitly and use INSUFFICIENT_EVIDENCE.
5. Prefer few correct judgements over many speculative ones.
6. Do not confuse what the source states (FATO) with what you conclude
   (INFERENCIA/INTERPRETACAO/OPINIAO).
7. Answer with a single valid JSON value. No markdown fences, no prose."""


def _block(task: str, instructions: str, schema: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """Assemble a task prompt from its parts."""
    return (
        f"{TASK_MARKER} {task}\n\n"
        f"{instructions.strip()}\n\n"
        f"OUTPUT — answer with exactly this JSON shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


# --------------------------------------------------------------------------- #
# 1. Claim extraction
# --------------------------------------------------------------------------- #
def claim_extraction_prompt(
    window_text: str,
    *,
    context_before: str = "",
    max_claims: int = 6,
    min_chars: int = 15,
    language: str = "pt",
) -> str:
    """Prompt that turns one transcript window into analysable claims."""
    instructions = f"""Extract the technical CLAIMS asserted by the speaker in the excerpt below.

A claim is a self-contained, verifiable technical assertion.

THE MOST IMPORTANT RULE: every claim must be a COMPLETE sentence that stands on its own,
with its own subject, understandable by someone who did not watch the video. Never emit a
fragment. "o desempenho sempre aumenta" is WRONG — it has no subject; write "Aumentar a
frequência do processador sempre aumenta o desempenho". "O pipeline garante aceleração de
cinco vezes" loses the reason — write the whole assertion the speaker made.

Other rules:
- Split a sentence into two claims only when it asserts two things that could be judged
  separately (e.g. a fact AND a causal conclusion drawn from it). Each half must still be
  a complete sentence, repeating the subject.
- Resolve pronouns from the context, WITHOUT adding information the speaker did not state.
- Keep the speaker's meaning exactly, including absolute words such as
  "sempre"/"always" or "nunca"/"never" — do not soften them.
- Ignore greetings, logistics, questions and purely narrative sentences. Do keep analogies
  and simplifications the speaker uses to explain something: they are claims too.
- If the excerpt contains no technical claim, return an empty list.
- At most {max_claims} claims. Ignore fragments shorter than {min_chars} characters.
- "verbatim" must be an exact substring of the excerpt.
- Write "topic" and "context" in {'Portuguese' if language == 'pt' else 'English'}."""
    schema = {
        "claims": [
            {
                "text": "standalone claim",
                "topic": "short topic",
                "context": "what the speaker was explaining",
                "importance": "high | medium | low",
                "verbatim": "exact substring of the excerpt",
            }
        ]
    }
    payload = {
        "context_before": context_before[-600:],
        "text": window_text,
        "max_claims": max_claims,
        "min_chars": min_chars,
    }
    return _block("claim_extraction", instructions, schema, payload)


# --------------------------------------------------------------------------- #
# 2. Critique of one claim against retrieved evidence
# --------------------------------------------------------------------------- #
def critique_prompt(
    claim: str,
    excerpts: List[Dict[str, Any]],
    *,
    claim_context: str = "",
    transcript_note: str = "",
    universal_markers: Optional[List[str]] = None,
    causal_markers: Optional[List[str]] = None,
    language: str = "pt",
) -> str:
    """Prompt that audits a single claim using only the retrieved excerpts.

    Evidence is cited by sentence id, never re-typed. That makes a fabricated
    quote structurally impossible — the model can only point at a sentence it
    was given — and removes the largest slice of generated tokens, which on
    CPU is the dominant cost of a review.
    """
    instructions = f"""Audit the CLAIM using ONLY the EXCERPTS provided.

Classification (choose exactly one):
- CORRETA: consistent with the evidence.
- PARCIALMENTE_CORRETA: the core is right but relevant conditions or limits are omitted.
- INCORRETA: contradicts reliable evidence.
- IMPRECISA: the idea may hold but the formulation is technically inadequate or ambiguous.
- CONTRADITORIA: conflicts with a source under the same conditions.
- NAO_SUSTENTADA: the provided evidence is not sufficient to judge it.
- SIMPLIFICACAO_PEDAGOGICA: a deliberate simplification, not necessarily an error.

Severity: CRITICO | ALTO | MEDIO | BAIXO | PEDAGOGICO.
Confidence: MUITO_ALTA | ALTA | MEDIA | BAIXA.
evidence_status: OK | UNKNOWN | CONFLICTING_EVIDENCE | INSUFFICIENT_EVIDENCE.
statement_type: FATO (stated by the source) | INFERENCIA | INTERPRETACAO | OPINIAO.
compatibility (claim vs. sources): SIM | NAO | PARCIAL | INDETERMINADO.

Method:
1. Check definitions, terminology, units, dimensional consistency, orders of magnitude,
   validity conditions and the physical interpretation.
2. Absolute words ("sempre", "nunca", "todos", "garante") are not errors by themselves —
   check whether the evidence really supports a universal statement.
3. Do not treat correlation, association or a hypothesis as demonstrated causality
   unless the evidence demonstrates it.
4. A textual difference between the claim and the source is NOT automatically a contradiction.
5. If the excerpts do not cover the claim, answer NAO_SUSTENTADA with
   INSUFFICIENT_EVIDENCE and say "Não foi possível determinar com as evidências disponíveis."
6. CITE BY ID. In "evidence", list the ids of the sentences that justify your verdict —
   ids taken from the excerpts below and nothing else. Never write the sentence out,
   never invent an id. Leave the list empty only for NAO_SUSTENTADA.
7. Also report relevant OMISSIONS: information a sentence states and the claim leaves out,
   when it changes how the claim should be understood (a condition, a limitation, a
   trade-off, a validity range). Cite them by id too. Empty list when there is none.
8. Be brief: at most two sentences per text field.
9. Write the text fields in {'Portuguese' if language == 'pt' else 'English'}. Leave
   "problem" and "suggested_correction" empty when the claim is correct."""
    schema = {
        "classification": "CORRETA",
        "severity": "BAIXO",
        "confidence": "MEDIA",
        "evidence_status": "OK",
        "statement_type": "FATO",
        "compatibility": "SIM",
        "evidence": [{"id": "e1", "relation": "SUPPORTS | CONTRADICTS | PARTIAL | NEUTRAL"}],
        "analysis": "objective technical analysis",
        "problem": "precise description of the error, or empty",
        "suggested_correction": "corrected wording, or empty",
        "conclusion": "one-sentence verdict",
        "omissions": [{"id": "e2", "impact": "ALTA | MEDIA | BAIXA", "explanation": "why it matters"}],
    }
    payload = {
        "claim": claim,
        "claim_context": claim_context,
        "transcription_note": transcript_note,
        "absolute_words_detected": universal_markers or [],
        "causal_words_detected": causal_markers or [],
        "excerpts": excerpts,
    }
    return _block("claim_critique", instructions, schema, payload)


# --------------------------------------------------------------------------- #
# 3. Omissions
# --------------------------------------------------------------------------- #
def omission_prompt(claim: str, evidence: List[Dict[str, Any]], *, language: str = "pt") -> str:
    """Prompt that finds relevant conditions present in sources but not in the claim."""
    instructions = f"""Identify relevant information that the EVIDENCE states and the CLAIM omits.

Report an omission only when ALL of the following hold:
- the missing information is explicitly present in an excerpt;
- it materially changes how the claim should be understood
  (a condition, a limitation, a trade-off, a validity range, a counterexample);
- the claim does not already mention it.

Impact: ALTA | MEDIA | BAIXA. Return an empty list when there is no relevant omission.
Do not report stylistic or purely didactic omissions.
"missing_information" and "explanation" must be written in
{'Portuguese' if language == 'pt' else 'English'}; "quote" must be verbatim from an excerpt."""
    schema = {
        "omissions": [
            {
                "missing_information": "what is missing",
                "impact": "ALTA | MEDIA | BAIXA",
                "chunk_id": "id from the list",
                "quote": "verbatim quote",
                "explanation": "why it matters",
            }
        ]
    }
    return _block("omission_detection", instructions, schema, {"claim": claim, "evidence": evidence})


# --------------------------------------------------------------------------- #
# 4. Internal contradictions
# --------------------------------------------------------------------------- #
def internal_contradiction_prompt(
    claim_a: str, claim_b: str, *, context_a: str = "", context_b: str = "", language: str = "pt"
) -> str:
    """Prompt that checks whether two claims from the same video conflict."""
    instructions = f"""Decide whether the two claims, both made in the SAME video, contradict each other.

Before answering "yes", check whether:
- the terms are used in the same sense in both claims;
- there is a difference of context;
- there is a temporal difference (e.g. before/after an event, powered on/off);
- some condition resolves the apparent inconsistency;
- one claim is a simplification of the other rather than its negation.

Answer is_contradiction = true only when the claims cannot both be true under the
same conditions. Confidence: MUITO_ALTA | ALTA | MEDIA | BAIXA.
Severity: CRITICO | ALTO | MEDIO | BAIXO | PEDAGOGICO.
Write the explanation in {'Portuguese' if language == 'pt' else 'English'}."""
    schema = {
        "is_contradiction": False,
        "explanation": "why they do or do not conflict",
        "same_sense_of_terms": True,
        "context_difference": "",
        "temporal_difference": "",
        "resolving_condition": "",
        "confidence": "BAIXA",
        "severity": "MEDIO",
    }
    payload = {"claim_a": claim_a, "context_a": context_a, "claim_b": claim_b, "context_b": context_b}
    return _block("internal_contradiction", instructions, schema, payload)


# --------------------------------------------------------------------------- #
# 5. Devil's advocate
# --------------------------------------------------------------------------- #
def devil_advocate_prompt(
    claim: str,
    classification: str,
    analysis: str,
    problem: str,
    evidence_quotes: List[str],
    source_texts: List[str],
    *,
    transcript_note: str = "",
    language: str = "pt",
) -> str:
    """Prompt that attacks a critique before it is allowed into the report."""
    instructions = f"""Act as the devil's advocate against the CRITIQUE below. Your job is to
attack it, not to defend it. Answer these questions honestly:

1. Is there a reasonable alternative interpretation of the claim?
2. Does the critique depend on an unstated assumption?
3. Was the context of the explanation taken into account?
4. Does the source really support the critique?
5. Could the claim be valid under some condition?
6. Could a transcription error have created the apparent problem?
7. Is there, among the excerpts, anything that contradicts the critique?

WHAT "sustained" MEANS — read carefully, this decides whether the critique is deleted:
- sustained = false ONLY when the critique is WRONG: the source does not say what the
  critique claims it says, the critique misreads the claim, or the claim is in fact
  correct. A false answer DELETES the critique from the report.
- sustained = true when the critique is right, even if imperfect. If the critique is
  right but overstated, or a nuance applies, keep sustained = true and express that
  through revised_classification, revised_severity or a negative confidence_adjustment.
- Noting that "the claim could be read more charitably" is NOT grounds for false. Almost
  any claim can be read charitably; that is what confidence is for.
- Set source_supports_critique = false only when the cited sentence genuinely fails to
  back the critique.

confidence_adjustment is a number between -0.4 and 0.1.
Fill revised_classification / revised_severity only when they must change;
otherwise use null. Write the notes in {'Portuguese' if language == 'pt' else 'English'}."""
    schema = {
        "sustained": True,
        "alternative_interpretation": "",
        "assumptions": [],
        "context_considered": True,
        "source_supports_critique": True,
        "conditions_that_would_validate_claim": "",
        "transcription_risk": "",
        "conflicting_sources": "",
        "notes": "",
        "confidence_adjustment": 0.0,
        "revised_classification": None,
        "revised_severity": None,
    }
    payload = {
        "claim": claim,
        "classification": classification,
        "analysis": analysis,
        "problem": problem,
        "evidence_quotes": evidence_quotes,
        "source_texts": source_texts,
        "transcription_note": transcript_note,
    }
    return _block("devil_advocate", instructions, schema, payload)
