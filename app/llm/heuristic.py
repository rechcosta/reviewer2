"""Offline heuristic provider.

This is **not** a language model. It is a small, deterministic rule engine
that answers the same JSON contracts as the real providers, so that:

* the test suite runs without downloading any model;
* the pipeline can be exercised end to end on a machine with no GPU and no
  Ollama installation.

Its judgements are intentionally conservative (it prefers
``NAO_SUSTENTADA`` over guessing) and every quote it emits is copied
verbatim from the evidence it was given, never invented. Reports produced
with this backend are labelled accordingly.

Its prose is always Portuguese (it has fixed strings, not a model), so an
English report produced with ``--offline`` keeps Portuguese analysis text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from ..text_utils import (
    CAUSAL_MARKERS,
    CONDITION_MARKERS,
    UNIVERSAL_MARKERS,
    find_markers,
    overlap,
    split_sentences,
    strip_accents,
    tokenize,
)
from .interface import LLMProvider, parse_json

TASK_MARKER = "REVIEWER2_TASK:"


class HeuristicProvider(LLMProvider):
    """Deterministic, model-free implementation of the LLM contracts."""

    name = "heuristic"

    def __init__(self, model: str = "heuristic-rules-v1") -> None:
        self.model = model

    def health_check(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        task, payload = _read_task(prompt)
        handler = {
            "claim_extraction": self._claims,
            "claim_critique": self._critique,
            "internal_contradiction": self._contradiction,
            "devil_advocate": self._devil_advocate,
        }.get(task)
        if handler is None:
            return json.dumps({"error": "unsupported_task", "task": task}, ensure_ascii=False)
        return json.dumps(handler(payload), ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Task implementations
    # ------------------------------------------------------------------ #
    def _claims(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Turn a transcript window into candidate claims."""
        text = str(payload.get("text", ""))
        max_claims = int(payload.get("max_claims", 6))
        min_chars = int(payload.get("min_chars", 15))
        claims: List[Dict[str, Any]] = []
        for sentence in split_sentences(text):
            if len(sentence) < min_chars or sentence.endswith("?"):
                continue
            if len(tokenize(sentence)) < 3:
                continue
            universal = find_markers(sentence, UNIVERSAL_MARKERS)
            causal = find_markers(sentence, CAUSAL_MARKERS)
            importance = "high" if (universal or causal) else "medium"
            claims.append(
                {
                    "text": sentence,
                    "topic": _topic_of(sentence),
                    "context": text[:280],
                    "importance": importance,
                    "verbatim": sentence,
                }
            )
            if len(claims) >= max_claims:
                break
        return {"claims": claims}

    def _critique(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compare one claim with the retrieved excerpts, citing sentences by id."""
        claim = str(payload.get("claim", ""))
        universal = find_markers(claim, UNIVERSAL_MARKERS)
        causal = find_markers(claim, CAUSAL_MARKERS)

        best_id, best_text, best_score = "", "", 0.0
        for excerpt in list(payload.get("excerpts") or []):
            for sentence in list(excerpt.get("sentences") or []):
                text = str(sentence.get("text", ""))
                score = overlap(claim, text)
                if score > best_score:
                    best_id, best_text, best_score = str(sentence.get("id", "")), text, score

        if not best_id or best_score < 0.2:
            return {
                "classification": "NAO_SUSTENTADA",
                "severity": "BAIXO",
                "confidence": "BAIXA",
                "evidence_status": "INSUFFICIENT_EVIDENCE",
                "statement_type": "INFERENCIA",
                "compatibility": "INDETERMINADO",
                "evidence": [],
                "analysis": "Não foi possível determinar com as evidências disponíveis: "
                "nenhum trecho recuperado trata diretamente desta afirmação.",
                "problem": "",
                "suggested_correction": "",
                "conclusion": "Sem evidência suficiente nas fontes fornecidas.",
                "omissions": [],
            }

        conditions = find_markers(best_text, CONDITION_MARKERS)
        omissions = (
            []
            if (find_markers(claim, CONDITION_MARKERS) or not conditions)
            else [
                {
                    "id": best_id,
                    "impact": "MEDIA",
                    "explanation": "A fonte condiciona a afirmação; o vídeo apresenta-a sem a condição.",
                }
            ]
        )

        if universal and conditions:
            return {
                "classification": "PARCIALMENTE_CORRETA",
                "severity": "MEDIO",
                "confidence": "MEDIA",
                "evidence_status": "OK",
                "statement_type": "FATO",
                "compatibility": "PARCIAL",
                "evidence": [{"id": best_id, "relation": "PARTIAL"}],
                "analysis": (
                    f"A afirmação usa uma formulação universal ({', '.join(universal)}), "
                    "enquanto a fonte apresenta condições ou limitações para o mesmo fenômeno."
                ),
                "problem": "Generalização indevida: a evidência não sustenta uma afirmação universal.",
                "suggested_correction": "Reformular a afirmação indicando as condições em que ela vale.",
                "conclusion": "Fundamento correto, formulação universal não sustentada.",
                "omissions": omissions,
            }

        if causal and conditions:
            return {
                "classification": "IMPRECISA",
                "severity": "MEDIO",
                "confidence": "MEDIA",
                "evidence_status": "OK",
                "statement_type": "INFERENCIA",
                "compatibility": "PARCIAL",
                "evidence": [{"id": best_id, "relation": "PARTIAL"}],
                "analysis": "A afirmação apresenta uma relação causal; a fonte descreve a relação "
                "sob condições específicas, sem demonstrar causalidade geral.",
                "problem": "Causalidade não demonstrada pela evidência disponível.",
                "suggested_correction": "Descrever a relação como associação observada sob determinadas condições.",
                "conclusion": "A direção da afirmação é plausível, mas a causalidade não está demonstrada.",
                "omissions": omissions,
            }

        return {
            "classification": "CORRETA",
            "severity": "BAIXO",
            "confidence": "MEDIA",
            "evidence_status": "OK",
            "statement_type": "FATO",
            "compatibility": "SIM",
            "evidence": [{"id": best_id, "relation": "SUPPORTS"}],
            "analysis": "O trecho recuperado é consistente com a afirmação.",
            "problem": "",
            "suggested_correction": "",
            "conclusion": "Afirmação consistente com a evidência recuperada.",
            "omissions": omissions,
        }

    def _contradiction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check whether two claims of the same video conflict."""
        a = str(payload.get("claim_a", ""))
        b = str(payload.get("claim_b", ""))
        negations = ("nao", "not", "never", "nunca", "sem ")
        a_neg = any(n in strip_accents(a) for n in negations)
        b_neg = any(n in strip_accents(b) for n in negations)
        shared = overlap(a, b)
        opposed = a_neg != b_neg and shared >= 0.35
        return {
            "is_contradiction": bool(opposed),
            "explanation": (
                "As afirmações compartilham o mesmo objeto e têm polaridade oposta."
                if opposed
                else "Não foi identificado conflito direto entre as afirmações."
            ),
            "same_sense_of_terms": True,
            "context_difference": "",
            "temporal_difference": "",
            "resolving_condition": "",
            "confidence": "BAIXA",
            "severity": "MEDIO",
        }

    def _devil_advocate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Keep a critique only when its quote really comes from the source."""
        quotes: Iterable[str] = payload.get("evidence_quotes") or []
        sources = " ".join(str(t) for t in (payload.get("source_texts") or []))
        classification = str(payload.get("classification", ""))
        quotes = [q for q in quotes if q]
        # Whitespace must be flattened on both sides: quotes come from
        # sentence splitting (single spaces) while the source text still
        # carries the line breaks of the original document.
        flat_sources = _flatten(sources)
        grounded = all(_flatten(q)[:80] in flat_sources for q in quotes) if quotes else False

        if classification.upper().startswith("NAO_SUSTENTADA"):
            return {
                "sustained": True,
                "notes": "Ausência de evidência declarada explicitamente; nada a rebater.",
                "confidence_adjustment": 0.0,
                "assumptions": [],
            }
        if not grounded:
            return {
                "sustained": False,
                "notes": "A citação apresentada não foi localizada nas fontes fornecidas.",
                "confidence_adjustment": -0.3,
                "assumptions": ["A crítica dependia de uma citação não verificável."],
            }
        return {
            "sustained": True,
            "alternative_interpretation": "A afirmação poderia ser válida em um contexto mais restrito.",
            "assumptions": ["A transcrição reflete corretamente a fala."],
            "context_considered": True,
            "source_supports_critique": True,
            "transcription_risk": "",
            "conflicting_sources": "",
            "notes": "Crítica mantida com evidência verificada.",
            "confidence_adjustment": 0.0,
        }


# --------------------------------------------------------------------------- #
def _flatten(text: str) -> str:
    """Accent-free, whitespace-normalised form used for quote comparison."""
    return " ".join(strip_accents(text).split())


def _read_task(prompt: str) -> tuple[str, Dict[str, Any]]:
    """Extract the task name and the JSON payload embedded in a prompt."""
    task = ""
    match = re.search(rf"{TASK_MARKER}\s*([a-z_]+)", prompt)
    if match:
        task = match.group(1)
    payload = parse_json(_last_json_block(prompt), expect="object") or {}
    if not isinstance(payload, dict):
        payload = {}
    return task, payload


def _last_json_block(prompt: str) -> str:
    """Return the last ``{...}`` block of the prompt (the input payload)."""
    start = prompt.rfind("{")
    if start < 0:
        return ""
    depth, in_string, escaped = 0, False, False
    # Scan backwards is fragile; instead find the matching block from the first
    # '{' that follows the INPUT marker, falling back to the whole prompt.
    marker = prompt.rfind("INPUT")
    begin = prompt.find("{", marker if marker >= 0 else 0)
    if begin < 0:
        return ""
    for index in range(begin, len(prompt)):
        char = prompt[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return prompt[begin : index + 1]
    return prompt[begin:]


def _topic_of(sentence: str) -> str:
    """Cheap topic guess: the two most significant words of the sentence."""
    tokens = tokenize(sentence)
    return " ".join(tokens[:2]) if tokens else "geral"
