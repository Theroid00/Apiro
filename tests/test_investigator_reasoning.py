"""Contracts for the complete bounded investigator engine."""

import json

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

    def chat(self, prompt):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]


class _Detector:
    def check_deterministic(self, _diagnosis, _fact):
        return NLIResult("neutral", 0.5, False)


class _AdjudicatingDetector(_Detector):
    def __init__(self):
        self.llm_calls = 0

    @staticmethod
    def should_check(_diagnosis, fact):
        return "no fever" in fact.lower()

    def check(self, _diagnosis, _fact):
        self.llm_calls += 1
        return NLIResult("contradiction", 0.95, True)

    def cache_info(self):
        return {"llm_calls": self.llm_calls}


def _response(*rows):
    return json.dumps({"hypotheses": list(rows)})


def _response_with_questions(rows, questions):
    return json.dumps({
        "hypotheses": list(rows),
        "missing_information": list(questions),
    })


def _candidate(name, confidence, *, evidence="E1"):
    return {
        "diagnosis": name,
        "confidence": confidence,
        "supporting_fact_ids": ["ax_0"],
        "conflicting_fact_ids": [],
        "evidence_ids": [evidence] if evidence else [],
    }


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


def test_investigator_stops_early_when_initial_result_is_adequate():
    reasoner, embedder, llm = _reasoner([
        _response(
            _candidate("Pulmonary embolism", 0.82),
            _candidate("Pneumonia", 0.45),
        )
    ])

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.mode == "investigator"
    assert result.stop_reason == "bounded_adequate"
    assert result.synthesis == ["Pulmonary embolism", "Pneumonia"]
    assert len(embedder.calls) == 2
    assert len(llm.prompts) == 1
    assert result.rounds == 1
    assert result.action_history == []


def test_investigator_uses_discriminating_actions_and_hard_round_limit():
    ambiguous = _response(
        _candidate("Pulmonary embolism", 0.70),
        _candidate("Pneumonia", 0.68),
    )
    reasoner, embedder, llm = _reasoner(
        [ambiguous, ambiguous, ambiguous], max_rounds=3
    )
    events = []

    result = reasoner.run(
        "Pleuritic chest pain without fever", on_event=events.append
    )

    assert result.stop_reason == "round_budget_exhausted"
    assert result.rounds == 3
    assert result.retrieval_count == 4
    assert result.reasoning_call_count == 3
    assert len(embedder.calls) == 4
    assert len(llm.prompts) == 3
    assert len(result.action_history) == 2
    assert all(
        row["kind"] == "discriminating_retrieval"
        for row in result.action_history
    )
    assert events[-1]["event"] == "traversal_complete"


def test_model_call_budget_can_stop_before_round_budget():
    weak = _response(_candidate("Pulmonary embolism", 0.4, evidence=""))
    reasoner, embedder, llm = _reasoner(
        [weak, weak], max_rounds=5, max_model_calls=2, max_retrievals=5
    )

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.stop_reason == "model_call_budget_exhausted"
    assert len(embedder.calls) == 3
    assert len(llm.prompts) == 2
    assert result.reasoning_call_count == 2


def test_investigator_returns_only_requested_number_of_diagnoses():
    reasoner, _embedder, _llm = _reasoner([
        _response(
            _candidate("A", 0.9),
            _candidate("B", 0.6),
            _candidate("C", 0.3),
        )
    ], n_diagnoses=2)

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.synthesis == ["A", "B"]
    assert result.total_nodes == 4


def test_candidate_branch_survives_one_omitted_round_for_backtracking():
    first = _response(
        _candidate("A", 0.70),
        _candidate("B", 0.68),
    )
    second = _response(_candidate("B", 0.45))
    reasoner, _embedder, _llm = _reasoner(
        [first, second], max_rounds=2
    )

    result = reasoner.run("Pleuritic chest pain without fever")

    assert "A" in result.synthesis
    assert result.candidate_history["A"][-1]["status"] == "held_for_backtracking"


def test_missing_information_is_exposed_as_an_auditable_question():
    response = _response_with_questions(
        [_candidate("Pulmonary embolism", 0.82), _candidate("Pneumonia", 0.4)],
        ["Is the D-dimer elevated?"],
    )
    reasoner, _embedder, _llm = _reasoner([response])

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.unresolved_questions == ["Is the D-dimer elevated?"]


def test_high_impact_model_adjudication_is_called_at_most_once():
    embedder = _Embedder()
    llm = _LLM([_response(_candidate("Fever syndrome", 0.8))])
    detector = _AdjudicatingDetector()
    reasoner = InvestigatorReasoner(
        embedder=embedder,
        llm_client=llm,
        axiom_extractor=_Extractor(),
        contradiction_detector=detector,
        max_rounds=1,
        max_adjudications=1,
    )

    result = reasoner.run("Pleuritic chest pain without fever")

    assert result.adjudication_count == 1
    assert detector.llm_calls == 1
    assert result.reasoning_call_count == 2
    assert result.hypotheses[0].conflicting_fact_ids == ("ax_1",)
