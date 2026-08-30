"""LLM interface: JSON extraction, provider contracts, error handling."""

from __future__ import annotations

import pytest

from app.errors import ConfigurationError, LLMError
from app.config import LLMConfig
from app.llm import HeuristicProvider, LLMProvider, build_llm, parse_json
from app.llm.interface import as_dict, as_list
from app.prompts import claim_extraction_prompt, critique_prompt, devil_advocate_prompt


class _ScriptedProvider(LLMProvider):
    """Returns pre-recorded answers, one per call."""

    name = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.model = "scripted"
        self.calls = 0

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None, stop=None):
        self.calls += 1
        return self.answers.pop(0) if self.answers else "{}"


def test_parse_json_plain() -> None:
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_inside_code_fence() -> None:
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_parse_json_with_surrounding_prose() -> None:
    raw = 'Claro! Aqui está o resultado:\n{"classification": "CORRETA"}\nEspero ter ajudado.'
    assert parse_json(raw)["classification"] == "CORRETA"


def test_parse_json_tolerates_trailing_commas() -> None:
    assert parse_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_parse_json_handles_braces_inside_strings() -> None:
    assert parse_json('{"text": "uma chave } dentro"}')["text"] == "uma chave } dentro"


def test_parse_json_returns_none_for_garbage() -> None:
    assert parse_json("desculpe, não consigo responder") is None
    assert parse_json("") is None


def test_generate_json_repairs_once() -> None:
    provider = _ScriptedProvider(["not json at all", '{"ok": true}'])
    assert provider.generate_json("prompt")["ok"] is True
    assert provider.calls == 2


def test_generate_json_raises_after_retries() -> None:
    provider = _ScriptedProvider(["nope", "still nope"])
    with pytest.raises(LLMError) as excinfo:
        provider.generate_json("prompt", retries=1)
    assert excinfo.value.stage == "generate_json"


def test_as_dict_and_as_list() -> None:
    assert as_dict({"a": 1}) == {"a": 1}
    assert as_dict([{"a": 1}]) == {"a": 1}
    assert as_dict("x") == {}
    assert as_list({"claims": [1, 2]}, key="claims") == [1, 2]
    assert as_list([1, 2]) == [1, 2]
    assert as_list("x") == []


def test_heuristic_extracts_claims(llm: HeuristicProvider) -> None:
    prompt = claim_extraction_prompt(
        "A cache é uma memória volátil. Aumentar a frequência sempre aumenta o desempenho."
    )
    payload = llm.generate_json(prompt)
    texts = [claim["text"] for claim in payload["claims"]]
    assert len(texts) == 2
    assert any("sempre" in text for text in texts)
    assert all(claim["verbatim"] in prompt for claim in payload["claims"])


def _excerpt(*sentences: str) -> list:
    return [
        {
            "chunk_id": "c1",
            "document": "d.md",
            "sentences": [{"id": f"e{i}", "text": t} for i, t in enumerate(sentences, start=1)],
        }
    ]


def test_heuristic_flags_universal_claim_against_conditional_source(llm: HeuristicProvider) -> None:
    excerpts = _excerpt(
        "Aumentar a frequência do processador aumenta o desempenho apenas quando a carga é limitada por computação."
    )
    payload = llm.generate_json(
        critique_prompt("Aumentar a frequência do processador sempre aumenta o desempenho.", excerpts)
    )
    assert payload["classification"] == "PARCIALMENTE_CORRETA"
    assert payload["evidence"][0]["id"] == "e1"      # cita por id, nunca reescreve


def test_heuristic_declares_insufficient_evidence(llm: HeuristicProvider) -> None:
    payload = llm.generate_json(
        critique_prompt("O pipeline tem cinco estágios.", _excerpt("Receita de bolo de chocolate com cobertura."))
    )
    assert payload["classification"] == "NAO_SUSTENTADA"
    assert payload["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["evidence"] == []


def test_heuristic_devil_advocate_rejects_ungrounded_quote(llm: HeuristicProvider) -> None:
    payload = llm.generate_json(
        devil_advocate_prompt(
            "afirmação", "INCORRETA", "análise", "problema",
            evidence_quotes=["uma citação que não existe em lugar nenhum das fontes"],
            source_texts=["texto completamente diferente da citação apresentada"],
        )
    )
    assert payload["sustained"] is False


def test_heuristic_never_invents_citations(llm: HeuristicProvider) -> None:
    excerpts = _excerpt("A memória cache é volátil e perde os dados sem energia.")
    payload = llm.generate_json(critique_prompt("A memória cache é volátil.", excerpts))
    valid = {s["id"] for s in excerpts[0]["sentences"]}
    for item in payload["evidence"]:
        assert item["id"] in valid


def test_build_llm_heuristic_needs_no_server() -> None:
    provider = build_llm(LLMConfig(provider="heuristic"))
    assert isinstance(provider, HeuristicProvider)
    assert provider.health_check() is True


def test_build_llm_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError):
        build_llm(LLMConfig(provider="magic"), check=False)


def test_build_llm_falls_back_when_configured() -> None:
    config = LLMConfig(provider="ollama", base_url="http://127.0.0.1:1", fallback_to_heuristic=True)
    assert isinstance(build_llm(config), HeuristicProvider)


def test_build_llm_reports_actionable_error_when_unreachable() -> None:
    config = LLMConfig(provider="ollama", base_url="http://127.0.0.1:1", model="nope")
    with pytest.raises(ConfigurationError) as excinfo:
        build_llm(config)
    assert "ollama pull" in excinfo.value.format()


# --------------------------------------------------------------------------- #
# Truncated answers (the most common failure of small models)
# --------------------------------------------------------------------------- #
def test_truncated_object_is_repaired() -> None:
    """A critique cut off by the token limit keeps its verdict and evidence."""
    truncated = (
        '{"classification": "PARCIALMENTE_CORRETA", "severity": "MEDIO", '
        '"evidence": [{"chunk_id": "ref_0001_00", "quote": "trecho literal", "relation": "PARTIAL"}], '
        '"analysis": "a fonte condiciona o ganho de desempenho'
    )
    parsed = parse_json(truncated, expect="object")
    assert isinstance(parsed, dict)
    assert parsed["classification"] == "PARCIALMENTE_CORRETA"
    assert parsed["evidence"][0]["chunk_id"] == "ref_0001_00"
    assert parsed["analysis"].startswith("a fonte condiciona")


def test_truncated_object_never_degrades_into_its_inner_array() -> None:
    """Regression: the inner "evidence" array must not be read as the verdict.

    Before this guard, a truncated object silently parsed as its own evidence
    list, so every claim lost its classification and fell back to the default.
    """
    truncated = '{"classification": "INCORRETA", "evidence": [{"chunk_id": "c1", "quote": "x"}], "analysis": "y'
    parsed = parse_json(truncated, expect="object")
    assert isinstance(parsed, dict)
    assert "classification" in parsed
    assert "chunk_id" not in parsed


def test_truncated_dangling_key_is_dropped() -> None:
    truncated = '{"classification": "CORRETA", "severity": "BAIXO", "analysis":'
    parsed = parse_json(truncated, expect="object")
    assert parsed["classification"] == "CORRETA"
    assert parsed["severity"] == "BAIXO"


def test_truncated_nested_structures_are_closed() -> None:
    truncated = '{"claims": [{"text": "uma afirmação", "topic": "t"}, {"text": "outra afirmação"'
    parsed = parse_json(truncated, expect="object")
    assert len(parsed["claims"]) == 2
    assert parsed["claims"][1]["text"] == "outra afirmação"


def test_bare_top_level_array_is_still_accepted() -> None:
    """Some models answer with a bare list; that must keep working."""
    parsed = parse_json('[{"text": "afirmação"}]', expect="object")
    assert isinstance(parsed, list)
    assert parsed[0]["text"] == "afirmação"


def test_complete_object_is_not_touched_by_the_repair() -> None:
    parsed = parse_json('{"a": 1, "b": [1, 2]}', expect="object")
    assert parsed == {"a": 1, "b": [1, 2]}


def test_truncated_answer_flows_through_generate_json() -> None:
    provider = _ScriptedProvider(
        ['{"classification": "INCORRETA", "problem": "definição errada']
    )
    payload = provider.generate_json("prompt")
    assert payload["classification"] == "INCORRETA"
    assert provider.calls == 1  # repaired locally, no extra model round trip


# --------------------------------------------------------------------------- #
# Response cache — a re-run must not repeat tens of minutes of model calls
# --------------------------------------------------------------------------- #
def test_cache_serves_a_repeated_call(tmp_path) -> None:
    from app.llm import CachingProvider

    inner = _ScriptedProvider(['{"a": 1}', '{"a": 2}'])
    cached = CachingProvider(inner, tmp_path / "llm")

    first = cached.generate("mesmo prompt", system="sys")
    second = cached.generate("mesmo prompt", system="sys")

    assert first == second == '{"a": 1}'
    assert inner.calls == 1                     # a segunda veio do disco
    assert cached.hits == 1 and cached.misses == 1


def test_cache_survives_a_new_process(tmp_path) -> None:
    """An interrupted run resumes: the cache lives on disk, not in memory."""
    from app.llm import CachingProvider

    CachingProvider(_ScriptedProvider(['{"a": 1}']), tmp_path / "llm").generate("p", system="s")

    inner = _ScriptedProvider(['{"a": 99}'])
    revived = CachingProvider(inner, tmp_path / "llm")
    assert revived.generate("p", system="s") == '{"a": 1}'
    assert inner.calls == 0


def test_cache_key_covers_every_input_that_changes_the_answer(tmp_path) -> None:
    from app.llm import CachingProvider

    cached = CachingProvider(_ScriptedProvider([]), tmp_path / "llm")
    base = cached._key("prompt", "system", 0.1, 100)

    assert cached._key("outro prompt", "system", 0.1, 100) != base
    assert cached._key("prompt", "outro system", 0.1, 100) != base
    assert cached._key("prompt", "system", 0.9, 100) != base
    assert cached._key("prompt", "system", 0.1, 500) != base


def test_cache_key_changes_with_the_model(tmp_path) -> None:
    from app.llm import CachingProvider

    small = _ScriptedProvider([]); small.model = "qwen2.5:3b"
    large = _ScriptedProvider([]); large.model = "qwen2.5:7b"
    key_small = CachingProvider(small, tmp_path / "llm")._key("p", "s", 0.1, 100)
    key_large = CachingProvider(large, tmp_path / "llm")._key("p", "s", 0.1, 100)
    assert key_small != key_large


def test_cache_can_be_disabled(tmp_path) -> None:
    from app.llm import CachingProvider

    inner = _ScriptedProvider(['{"a": 1}', '{"a": 2}'])
    cached = CachingProvider(inner, tmp_path / "llm", enabled=False)
    assert cached.generate("p") != cached.generate("p")
    assert inner.calls == 2


def test_build_llm_wraps_with_the_cache(tmp_path) -> None:
    from app.llm import CachingProvider

    config = LLMConfig(provider="ollama", base_url="http://127.0.0.1:1", cache=True)
    provider = build_llm(config, check=False, cache_dir=tmp_path / "llm")
    assert isinstance(provider, CachingProvider)
    assert provider.name == "ollama"

    without = build_llm(LLMConfig(provider="ollama", cache=False), check=False, cache_dir=tmp_path)
    assert not isinstance(without, CachingProvider)
