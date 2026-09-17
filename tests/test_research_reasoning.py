"""Tests for the research-derived complete-mode components."""

import json

from apiro.reasoning.action_policy import ActionValuePolicy
from apiro.reasoning.associative_memory import AssociativeMemory
from apiro.reasoning.concept_graph import MedicalConceptGraph
from apiro.reasoning.models import EvidenceChunk


def _chunk(chunk_id, text, distance):
    return EvidenceChunk(chunk_id, text, distance, {})


def test_concept_graph_round_trip_and_personalized_propagation(tmp_path):
    graph = MedicalConceptGraph()
    graph.ingest_records([
        {
            "chunk_id": "hpo_1",
            "title": "Pulmonary embolism phenotype",
            "condition_tags": "pulmonary embolism",
            "hpo_name": "pleuritic chest pain",
            "medical_domain": "pathophysiology",
        },
        {
            "chunk_id": "hpo_2",
            "condition_tags": "pneumonia",
            "hpo_name": "fever",
            "medical_domain": "pathophysiology",
        },
    ])
    path = tmp_path / "concept_graph.json"
    graph.save(path)

    restored = MedicalConceptGraph.load(path)
    related = dict(restored.related_concepts("pleuritic chest pain", limit=10))
    scores = restored.candidate_scores(
        "pleuritic chest pain", ["pulmonary embolism", "pneumonia"]
    )

    assert "pulmonary embolism" in related
    assert scores["pulmonary embolism"] > scores["pneumonia"]


def test_associative_memory_separates_near_duplicate_cases():
    batches = [
        [
            _chunk("a", "PE with pleuritic pain and tachycardia", 0.1),
            _chunk("b", "PE with pleuritic pain and tachycardia", 0.11),
        ],
        [_chunk("c", "Pneumonia with fever and consolidation", 0.2)],
    ]
    calls = []

    def retrieve(query, _limit):
        calls.append(query)
        return batches[len(calls) - 1]

    result = AssociativeMemory(retrieve).recall(
        "pleuritic pain", ["tachycardia"], max_results=2
    )

    assert result.retrieval_count == 2
    assert [chunk.id for chunk in result.evidence] == ["a", "c"]


def test_action_policy_loads_external_weights(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"weights": {
        "bias": 0.0,
        "contradiction": 4.0,
        "cost": -1.0,
    }}), encoding="utf-8")

    policy = ActionValuePolicy.load(path)

    assert policy.score({"contradiction": 1.0, "cost": 0.1}) > 0.9
    assert policy.score({"contradiction": 0.0, "cost": 1.0}) < 0.5


def test_action_policy_can_learn_from_observed_action_rewards():
    policy = ActionValuePolicy({
        "bias": 0.0,
        "contradiction": 0.0,
        "cost": 0.0,
    })
    examples = [
        ({"contradiction": 1.0, "cost": 0.0}, 1.0),
        ({"contradiction": 0.0, "cost": 1.0}, 0.0),
    ] * 20

    policy.fit(examples, epochs=100, learning_rate=0.2)

    assert policy.score(examples[0][0]) > policy.score(examples[1][0])
