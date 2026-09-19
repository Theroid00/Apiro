"""Mechanism checks for the bounded evidence-audit investigator."""

import json

import pytest

from apiro.axioms.models import ClinicalAxiom
from apiro.graph.contradiction import NLIResult
from apiro.reasoning.investigator import InvestigatorReasoner


class _Extractor:
    def extract(self, _narrative, max_axioms=None):
        rows = [
            ClinicalAxiom(
                id="ax_0", text="The patient has pleuritic chest pain",
                domain="symptom", polarity="affirmed", value=None, unit=None,
                weight=0.9, raw_text="pleuritic chest pain",
            ),
            ClinicalAxiom(
                id="ax_1", text="The patient has no fever",
                domain="symptom", polarity="negated", value=None, unit=None,
                weight=0.7, raw_text="no fever",
            ),
        ]
        return rows[:max_axioms] if max_axioms else rows


class _Embedder:
    def __init__(self):
        self.calls = []

    def query(self, query, n_results):
        self.calls.append((query, n_results))
        index = len(self.calls)
        return [{
            "chunk_id": f"evidence_{index}",
            "text": f"Relevant evidence round {index}",
            "distance": 0.2,
        }]


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.structured_calls = 0
        self.schemas = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]

    def generate_json(self, prompt, schema=None):
        self.structured_calls += 1
        self.schemas.append(schema)
        return self.chat(prompt)


class _Detector:
    def check_deterministic(self, _diagnosis, _fact):
        return NLIResult("neutral", 0.5, False)


def _candidate(
    name,
    confidence,
    *,
    facts=("ax_0",),
    evidence="E1",
    quote="Relevant evidence round 1",
):
    return {
        "diagnosis": name,
        "confidence": confidence,
        "supporting_fact_ids": list(facts),
        "conflicting_fact_ids": [],
        "evidence_ids": [evidence] if evidence else [],
        "evidence_spans": (
            [{"evidence_id": evidence, "quote": quote}]
            if evidence and quote else []
        ),
    }


def _response(*rows, questions=()):
    return json.dumps({
        "hypotheses": list(rows),
        "missing_information": list(questions),
    })


def _reasoner(responses, **kwargs):
    embedder = _Embedder()
    llm = _LLM(responses)
    reasoner = InvestigatorReasoner(
        embedder=embedder,
        llm_client=llm,
        axiom_extractor=_Extractor(),
        contradiction_detector=_Detector(),
        **kwargs,
    )
    return reasoner, embedder, llm


def test_robust_candidate_passes_without_extra_call():
    reasoner, embedder, llm = _reasoner([_response(
        _candidate("Pulmonary embolism", 0.85, facts=("ax_0", "ax_1")),
        _candidate("Pneumonia", 0.30),
    )])

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.stop_reason == "evidence_audit_passed"
    assert result.synthesis[:2] == ["Pulmonary embolism", "Pneumonia"]
    assert result.evidence_audit["initial"]["needs_revision"] is False
    assert len(embedder.calls) == len(llm.prompts) == 1
    assert llm.structured_calls == 1
    assert llm.schemas[0]["type"] == "object"


def test_fragile_result_gets_exactly_one_counterfactual_revision():
    ambiguous = _response(
        _candidate("Pulmonary embolism", 0.70),
        _candidate("Pneumonia", 0.68),
    )
    reasoner, embedder, llm = _reasoner(
        [ambiguous, ambiguous, ambiguous], max_rounds=9,
        max_retrievals=9, max_model_calls=9,
    )
    events = []

    result = reasoner.run(
        "Pleuritic chest pain without fever", on_event=events.append
    )

    assert result.stop_reason == "counterfactual_revision_complete"
    assert result.rounds == result.retrieval_count == result.reasoning_call_count == 2
    assert len(embedder.calls) == len(llm.prompts) == 2
    assert [row["kind"] for row in result.action_history] == [
        "counterfactual_evidence_audit"
    ]
    assert events[-1]["event"] == "traversal_complete"


def test_model_budget_can_disable_revision():
    weak = _response(_candidate(
        "Pulmonary embolism", 0.4, evidence="", quote=""
    ))
    reasoner, embedder, llm = _reasoner(
        [weak], max_rounds=2, max_model_calls=1, max_retrievals=2
    )

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.stop_reason == "model_call_budget_exhausted"
    assert len(embedder.calls) == len(llm.prompts) == 1


def test_counterfactual_audit_identifies_rank_changing_fact():
    first = _response(
        _candidate("A", 0.75, facts=("ax_0",)),
        _candidate("B", 0.72, facts=("ax_1",)),
    )
    second = _response(
        _candidate("B", 0.80, facts=("ax_0", "ax_1"), evidence="E2",
                   quote="Relevant evidence round 2"),
        _candidate("A", 0.40),
    )
    reasoner, _embedder, llm = _reasoner([first, second])

    result = reasoner.run("Pleuritic chest pain without fever")

    initial = result.evidence_audit["initial"]
    assert initial["influential_fact_id"] == "ax_0"
    assert initial["rank_flip_without_fact"] is True
    assert "High-impact fact: ax_0" in llm.prompts[1]
    assert result.synthesis[0] == "B"


def test_only_exact_retrieved_spans_count_as_medical_evidence():
    fabricated = _response(_candidate(
        "Pulmonary embolism", 0.8, facts=("ax_0", "ax_1"),
        quote="A fabricated passage that was never retrieved",
    ))
    reasoner, _embedder, _llm = _reasoner(
        [fabricated], max_model_calls=1
    )

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.hypotheses[0].evidence_spans == ()
    assert "no_verified_evidence_span" in result.evidence_audit["initial"]["reasons"]


def test_text_fallback_is_counted_in_result_telemetry():
    reasoner, _embedder, _llm = _reasoner(
        ["1. Pulmonary embolism\n2. Pneumonia"], max_model_calls=1
    )

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.synthesis == ["Pulmonary embolism", "Pneumonia"]
    assert result.parse_fallback_count == 1


def test_missing_patient_fact_remains_an_unresolved_question():
    response = _response(
        _candidate("Pulmonary embolism", 0.85, facts=("ax_0", "ax_1")),
        _candidate("Pneumonia", 0.30),
        questions=("Is the D-dimer elevated?",),
    )
    reasoner, embedder, _llm = _reasoner([response])

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.unresolved_questions == ["Is the D-dimer elevated?"]
    assert len(embedder.calls) == 1
    assert "D-dimer" not in embedder.calls[0][0]


def test_uncertainty_comes_from_candidate_distribution():
    response = _response(
        _candidate("A", 0.70),
        _candidate("B", 0.69),
        _candidate("C", 0.68),
    )
    reasoner, _embedder, _llm = _reasoner([response], max_model_calls=1)

    audit = reasoner.run("Pleuritic chest pain without fever").evidence_audit["initial"]
    probabilities = [row["probability"] for row in audit["distribution"]]

    assert sum(probabilities) == pytest.approx(1.0, abs=2e-6)
    assert audit["normalized_entropy"] > 0.99
    assert probabilities[0] != pytest.approx(0.70)


def test_output_is_capped_without_truncating_candidate_audit():
    response = _response(
        _candidate("A", 0.9, facts=("ax_0", "ax_1")),
        _candidate("B", 0.6),
        _candidate("C", 0.3),
    )
    reasoner, _embedder, _llm = _reasoner([response], n_diagnoses=2)

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.synthesis == ["A", "B"]
    assert result.total_nodes == 4


def test_entropy_alone_does_not_trigger_a_second_pass():
    response = _response(
        _candidate("A", 0.70, facts=("ax_0", "ax_1")),
        _candidate("B", 0.65, facts=("ax_0", "ax_1")),
    )
    reasoner, embedder, llm = _reasoner([response])
    original_extract = reasoner.axiom_extractor.extract
    reasoner.axiom_extractor.extract = lambda *args, **kwargs: [
        ClinicalAxiom(
            item.id, item.text, item.domain, item.polarity, item.value, item.unit,
            0.01, raw_text=item.raw_text,
        )
        for item in original_extract(*args, **kwargs)
    ]

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.evidence_audit["initial"]["reasons"] == [
        "small_candidate_margin", "high_differential_entropy"
    ]
    assert result.evidence_audit["initial"]["needs_revision"] is False
    assert len(embedder.calls) == len(llm.prompts) == 1


def test_unverified_confidence_is_capped_and_history_is_preserved():
    weak = _response(_candidate("A", 0.95, facts=("ax_0",), evidence="", quote=""))
    reasoner, _embedder, _llm = _reasoner([weak], max_model_calls=1)

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.hypotheses[0].confidence == 0.69
    assert result.candidate_history["A"][0]["confidence"] == 0.95


def test_obvious_junk_and_duplicate_facts_are_removed():
    reasoner, _embedder, _llm = _reasoner([_response(_candidate("A", 0.5))])
    facts = [
        ClinicalAxiom("old_0", "pain", "symptom", "affirmed", None, None, 0.1, raw_text="pain"),
        ClinicalAxiom("old_1", "Chest pain", "symptom", "affirmed", None, None, 0.8, raw_text="chest pain"),
        ClinicalAxiom("old_2", "Chest pain", "symptom", "affirmed", None, None, 0.8, raw_text="chest pain"),
    ]

    cleaned = reasoner._clean_axioms(facts)

    assert [(item.id, item.text) for item in cleaned] == [("ax_0", "Chest pain")]
