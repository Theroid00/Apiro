"""Contracts for the bounded simplified reasoning engine."""

import json

from apiro.application.runtime import RuntimeResources
from apiro.axioms.models import ClinicalAxiom
from apiro.graph.contradiction import NLIResult
from apiro.reasoning.simple import SimpleReasoner


class _Extractor:
    def extract(self, _narrative, max_axioms=None):
        axioms = [
            ClinicalAxiom(
                id="ax_0", text="The patient has fever", domain="symptom",
                polarity="affirmed", value=None, unit=None, weight=0.8,
                raw_text="fever",
            ),
            ClinicalAxiom(
                id="ax_1", text="The patient has no cough", domain="symptom",
                polarity="negated", value=None, unit=None, weight=0.6,
                raw_text="no cough",
            ),
        ]
        return axioms[:max_axioms] if max_axioms else axioms


class _Embedder:
    def __init__(self):
        self.calls = []

    def query(self, query, n_results):
        self.calls.append((query, n_results))
        return [
            {"chunk_id": "paper_1", "text": "Evidence one", "distance": 0.2},
            {"chunk_id": "paper_far", "text": "Unrelated", "distance": 0.99},
        ]


class _LLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        return self.response

    generate = chat


class _Detector:
    def __init__(self):
        self.calls = []

    def check_deterministic(self, diagnosis, fact):
        self.calls.append((diagnosis, fact))
        if diagnosis == "Diagnosis B" and "fever" in fact:
            return NLIResult("contradiction", 0.9, False)
        return NLIResult("neutral", 0.5, False)


def _reasoner(response):
    embedder = _Embedder()
    llm = _LLM(response)
    detector = _Detector()
    reasoner = SimpleReasoner(
        embedder=embedder,
        llm_client=llm,
        axiom_extractor=_Extractor(),
        contradiction_detector=detector,
        n_diagnoses=3,
    )
    return reasoner, embedder, llm, detector


def test_simple_reasoner_has_one_retrieval_and_one_generation_call():
    response = json.dumps({"hypotheses": [{
        "diagnosis": "Diagnosis A",
        "confidence": 0.8,
        "supporting_fact_ids": ["ax_0", "invented"],
        "conflicting_fact_ids": [],
        "evidence_ids": ["E1", "invented_evidence"],
    }]})
    reasoner, embedder, llm, detector = _reasoner(response)
    events = []

    result = reasoner.run("Fever without cough", on_event=events.append)

    assert len(embedder.calls) == 1
    assert len(llm.prompts) == 1
    assert len(detector.calls) == 2
    assert result.mode == "simple"
    assert result.synthesis == ["Diagnosis A"]
    assert result.hypotheses[0].supporting_fact_ids == ("ax_0",)
    assert result.hypotheses[0].evidence_ids == ("paper_1",)
    assert [chunk.id for chunk in result.evidence] == ["paper_1"]
    assert result.total_nodes == 3
    assert result.total_edges == 1
    assert events[-1]["event"] == "traversal_complete"


def test_deterministic_conflict_penalty_changes_ranking():
    response = json.dumps({"hypotheses": [
        {
            "diagnosis": "Diagnosis B", "confidence": 0.9,
            "supporting_fact_ids": [], "conflicting_fact_ids": [],
            "evidence_ids": [],
        },
        {
            "diagnosis": "Diagnosis A", "confidence": 0.7,
            "supporting_fact_ids": ["ax_0"], "conflicting_fact_ids": [],
            "evidence_ids": ["paper_1"],
        },
    ]})
    reasoner, _embedder, _llm, _detector = _reasoner(response)

    result = reasoner.run("Fever without cough")

    assert result.synthesis == ["Diagnosis A", "Diagnosis B"]
    assert result.hypotheses[1].conflicting_fact_ids == ("ax_0",)
    assert result.contradiction_count == 1


def test_text_fallback_keeps_engine_usable_when_json_is_malformed():
    reasoner, _embedder, _llm, _detector = _reasoner(
        "1. Influenza\n2. Bacterial pneumonia"
    )

    result = reasoner.run("Fever without cough")

    assert result.synthesis == ["Influenza", "Bacterial pneumonia"]


def test_text_fallback_recovers_diagnoses_from_malformed_json():
    reasoner, _embedder, _llm, _detector = _reasoner(
        '{"hypotheses":[{"diagnosis":"Crohn\'s disease","confidence":0.8},'
        '{"diagnosis":"Ulcerative colitis","confidence":0.6}'
    )

    result = reasoner.run("Abdominal pain")

    assert result.synthesis == ["Crohn's disease", "Ulcerative colitis"]
    assert result.hypotheses[0].confidence == 0.8


def test_abstention_allows_an_explicit_empty_differential():
    reasoner, _embedder, _llm, _detector = _reasoner('{"hypotheses": []}')
    reasoner.allow_abstention = True

    result = reasoner.run("Insufficient information")

    assert result.synthesis == []
    assert result.total_nodes == 2


def test_weak_result_gets_exactly_one_corrective_pass():
    response = json.dumps({"hypotheses": [{
        "diagnosis": "Uncertain diagnosis",
        "confidence": 0.4,
        "supporting_fact_ids": [],
        "conflicting_fact_ids": [],
        "evidence_ids": [],
    }]})
    reasoner, embedder, llm, _detector = _reasoner(response)

    result = reasoner.run("Fever without cough")

    assert len(embedder.calls) == 2
    assert len(llm.prompts) == 2
    assert result.stop_reason == "corrective_complete"
    assert result.retrieval_count == 2
    assert result.reasoning_call_count == 2
    assert result.rounds == 2


def test_strong_result_stays_on_the_single_pass_fast_path():
    response = json.dumps({"hypotheses": [{
        "diagnosis": "Diagnosis A",
        "confidence": 0.85,
        "supporting_fact_ids": ["ax_0"],
        "conflicting_fact_ids": [],
        "evidence_ids": ["E1"],
    }]})
    reasoner, embedder, llm, _detector = _reasoner(response)

    result = reasoner.run("Fever without cough")

    assert len(embedder.calls) == 1
    assert len(llm.prompts) == 1
    assert result.stop_reason == "bounded_complete"
    assert result.retrieval_count == 1
    assert result.reasoning_call_count == 1


def test_runtime_exposes_canonical_service_with_explicit_mode():
    resources = RuntimeResources(
        embedder=_Embedder(),
        llm_client=_LLM(""),
        axiom_extractor=_Extractor(),
        doc_count=2,
        model="stub",
        ollama_url="http://invalid.test",
    )

    service = resources.create_service(default_mode="simple")

    assert service.default_mode == "simple"
