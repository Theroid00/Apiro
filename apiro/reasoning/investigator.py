"""Bounded multi-round clinical investigation with explicit resource budgets."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from apiro.axioms.seeding import axioms_to_seed_nodes
from apiro.config import (
    ACTION_POLICY_PATH,
    CONCEPT_GRAPH_PATH,
    INVESTIGATOR_MAX_ADJUDICATIONS,
    INVESTIGATOR_MAX_CANDIDATES,
    INVESTIGATOR_MAX_FACTS,
    INVESTIGATOR_MAX_GRAPH_NODES,
    INVESTIGATOR_MAX_MODEL_CALLS,
    INVESTIGATOR_MAX_RETRIEVALS,
    INVESTIGATOR_MAX_ROUNDS,
)
from apiro.context import select_clinical_context
from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.node import Node

from .models import DiagnosticHypothesis, EvidenceChunk, InvestigationResult
from .action_policy import ActionValuePolicy
from .associative_memory import AssociativeMemory
from .concept_graph import MedicalConceptGraph
from .simple import SimpleReasoner


@dataclass(frozen=True)
class InvestigationAction:
    """One auditable action selected by the bounded controller."""

    kind: str
    target: str
    query: str
    value: float
    reason: str


@dataclass
class CandidateBranch:
    """Persistent candidate state supporting revision and backtracking."""

    diagnosis: str
    current: DiagnosticHypothesis
    best_score: float
    misses: int = 0
    history: list[dict] = field(default_factory=list)

    def observe(self, candidate: DiagnosticHypothesis, round_number: int) -> None:
        self.current = candidate
        self.best_score = max(self.best_score, candidate.score)
        self.misses = 0
        self.history.append({
            "round": round_number,
            "score": candidate.score,
            "confidence": candidate.confidence,
            "conflicts": list(candidate.conflicting_fact_ids),
            "status": "active",
        })

    def decay(self, round_number: int) -> None:
        self.misses += 1
        self.current = replace(
            self.current,
            confidence=round(self.current.confidence * 0.9, 6),
            score=round(self.current.score - 0.08, 6),
        )
        self.history.append({
            "round": round_number,
            "score": self.current.score,
            "confidence": self.current.confidence,
            "status": "held_for_backtracking" if self.misses == 1 else "inactive",
        })


@dataclass
class CaseState:
    """Compact working memory for one investigator-mode run."""

    narrative: str
    facts: list[Node]
    evidence: list[EvidenceChunk] = field(default_factory=list)
    candidates: list[DiagnosticHypothesis] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    action_history: list[InvestigationAction] = field(default_factory=list)
    branches: dict[str, CandidateBranch] = field(default_factory=dict)
    graph_context: list[str] = field(default_factory=list)
    rounds: int = 0
    retrieval_count: int = 0
    reasoning_call_count: int = 0
    adjudication_count: int = 0


_INITIAL_PREFIX = """Generate a diverse candidate set before ranking it.
Keep rare but plausible diagnoses when patient-specific findings support them.
Compare candidates against negated findings and retrieved evidence.
The root JSON object must also contain "missing_information", an array of at
most two concise patient facts that would best distinguish the candidates.

"""

_REVISION_PREFIX = """You are revising a bounded differential after one targeted investigation.
Use the new evidence to compare the candidates. You may add, remove, or reorder
diagnoses. Do not preserve the old leader merely for consistency.
The root JSON object must also contain "missing_information", an array of at
most two concise patient facts that would best distinguish the candidates.

Controller action: {kind}
Target: {target}
Reason: {reason}
Prior candidates:
{candidates}

"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class InvestigatorReasoner(SimpleReasoner):
    """Investigate competing diagnoses without unbounded graph expansion."""

    def __init__(
        self,
        *,
        embedder,
        llm_client,
        axiom_extractor,
        contradiction_detector,
        n_diagnoses: int = 3,
        allow_abstention: bool = False,
        max_facts: int = INVESTIGATOR_MAX_FACTS,
        max_candidates: int = INVESTIGATOR_MAX_CANDIDATES,
        max_rounds: int = INVESTIGATOR_MAX_ROUNDS,
        max_retrievals: int = INVESTIGATOR_MAX_RETRIEVALS,
        max_model_calls: int = INVESTIGATOR_MAX_MODEL_CALLS,
        max_graph_nodes: int = INVESTIGATOR_MAX_GRAPH_NODES,
        max_adjudications: int = INVESTIGATOR_MAX_ADJUDICATIONS,
        concept_graph: MedicalConceptGraph | None = None,
        concept_graph_path: Path = CONCEPT_GRAPH_PATH,
        action_policy: ActionValuePolicy | None = None,
        action_policy_path: Path = ACTION_POLICY_PATH,
    ):
        self.output_diagnoses = max(1, int(n_diagnoses))
        self.max_candidates = max(self.output_diagnoses, int(max_candidates))
        self.max_rounds = max(1, int(max_rounds))
        self.max_retrievals = max(1, int(max_retrievals))
        self.max_model_calls = max(1, int(max_model_calls))
        self.max_graph_nodes = max(self.max_candidates + 1, int(max_graph_nodes))
        self.max_adjudications = max(0, int(max_adjudications))
        self.concept_graph = concept_graph or self._load_concept_graph(
            concept_graph_path
        )
        self.action_policy = action_policy or ActionValuePolicy.load(
            action_policy_path
        )
        super().__init__(
            embedder=embedder,
            llm_client=llm_client,
            axiom_extractor=axiom_extractor,
            contradiction_detector=contradiction_detector,
            n_diagnoses=self.max_candidates,
            max_facts=min(int(max_facts), self.max_graph_nodes - self.max_candidates),
            allow_abstention=allow_abstention,
            corrective_pass=False,
        )

    def run(
        self,
        narrative: str,
        *,
        on_event: Callable[[dict], None] | None = None,
    ) -> InvestigationResult:
        started = time.monotonic()
        timings: dict[str, float] = {}

        stage = time.monotonic()
        axioms = self._extract_axioms(narrative)
        facts = axioms_to_seed_nodes(axioms)
        if not facts:
            facts = [Node(
                id="ax_fallback_0",
                claim=(
                    "The patient presents with the following clinical picture: "
                    f"{narrative.strip()[:400]}"
                ),
                entropy_score=0.01,
                domain="pathophysiology",
                depth=0,
                metadata={"axiom_weight": 1.0, "polarity": "affirmed", "fallback": True},
            )]
        graph = BeliefGraph(max_depth=1, max_nodes=self.max_graph_nodes)
        for fact in facts:
            graph.add_node(fact)
            self._emit(on_event, {
                "event": "seed_added",
                "node_id": fact.id,
                "claim": fact.claim,
                "domain": fact.domain,
                "entropy": fact.entropy_score,
                "depth": 0,
            })
        timings["extraction"] = time.monotonic() - stage

        selected = select_clinical_context(
            narrative, max_characters=self.max_context_characters
        )
        state = CaseState(narrative=selected.text, facts=facts)

        stage = time.monotonic()
        memory = AssociativeMemory(
            lambda query, limit: self._retrieve(
                query, facts, n_results=limit
            ),
            self.concept_graph,
        )
        recalled = memory.recall(
            selected.text,
            [fact.claim for fact in facts],
            max_results=self.rag_top_k,
            max_calls=min(2, self.max_retrievals),
        )
        state.evidence = recalled.evidence
        state.retrieval_count = recalled.retrieval_count
        state.graph_context = list(recalled.graph_context)
        self._ingest_runtime_evidence(state.evidence)
        timings["initial_retrieval"] = time.monotonic() - stage
        self._emit(on_event, {
            "event": "retrieval_complete",
            "round": 1,
            "evidence_count": len(state.evidence),
        })

        stage = time.monotonic()
        raw_output = self._generate_json(
            _INITIAL_PREFIX + self._build_prompt(
                selected.text,
                facts,
                state.evidence,
                include_missing_information=True,
            )
        )
        state.reasoning_call_count = 1
        state.rounds = 1
        proposed = self._constrain_and_rank(
            self._parse_hypotheses(raw_output, facts, state.evidence), facts
        )
        self._update_candidates(state, proposed)
        self._add_missing_questions(state, raw_output)
        self._adjudicate_high_impact_conflict(state)
        timings["initial_reasoning"] = time.monotonic() - stage

        stop_reason = "bounded_adequate"
        while not self._is_adequate(state):
            budget_reason = self._budget_stop_reason(state)
            if budget_reason is not None:
                stop_reason = budget_reason
                break

            action = self._choose_action(state)
            state.action_history.append(action)
            self._emit(on_event, {
                "event": "investigation_action",
                "round": state.rounds + 1,
                "kind": action.kind,
                "target": action.target,
                "value": action.value,
                "reason": action.reason,
            })

            graph_terms = [
                label for label, _score in self.concept_graph.related_concepts(
                    action.query, limit=5
                )
            ]
            expanded_query = action.query
            if graph_terms:
                expanded_query += ". Graph-linked concepts: " + ", ".join(graph_terms)
            extra = self._retrieve(
                expanded_query, facts, n_results=min(4, self.rag_top_k)
            )
            state.retrieval_count += 1
            state.evidence = self._merge_evidence(state.evidence, extra)
            self._ingest_runtime_evidence(extra)

            revision_prompt = _REVISION_PREFIX.format(
                kind=action.kind,
                target=action.target,
                reason=action.reason,
                candidates="\n".join(
                    f"- {item.diagnosis}: confidence={item.confidence:.2f}, "
                    f"score={item.score:.2f}"
                    for item in state.candidates
                ) or "No usable candidates.",
            ) + self._build_prompt(
                selected.text,
                facts,
                state.evidence,
                include_missing_information=True,
            )
            raw_output = self._generate_json(revision_prompt)
            state.reasoning_call_count += 1
            state.rounds += 1
            proposed = self._constrain_and_rank(
                self._parse_hypotheses(raw_output, facts, state.evidence), facts
            )
            self._update_candidates(state, proposed)
            self._add_missing_questions(state, raw_output)
            self._adjudicate_high_impact_conflict(state)
            self._emit(on_event, {
                "event": "investigation_round_complete",
                "round": state.rounds,
                "candidate_count": len(state.candidates),
                "evidence_count": len(state.evidence),
            })
        else:
            stop_reason = "bounded_adequate"

        if not state.candidates and self.allow_abstention:
            stop_reason = "bounded_abstained"

        final_candidates = state.candidates[: self.output_diagnoses]
        stage = time.monotonic()
        self._add_hypotheses(graph, final_candidates, facts, on_event)
        timings["graph_and_validation"] = time.monotonic() - stage
        duration = time.monotonic() - started
        timings["total"] = duration

        result = InvestigationResult(
            mode="investigator",
            graph=graph,
            stop_reason=stop_reason,
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            rabbit_hole_count=0,
            contradiction_count=sum(
                len(item.conflicting_fact_ids) for item in final_candidates
            ),
            duration_seconds=duration,
            synthesis=[item.diagnosis for item in final_candidates],
            stage_timings={key: round(value, 6) for key, value in timings.items()},
            hypotheses=final_candidates,
            evidence=state.evidence,
            axioms=axioms,
            raw_output=raw_output,
            retrieval_count=state.retrieval_count,
            reasoning_call_count=state.reasoning_call_count,
            rounds=state.rounds,
            action_history=[action.__dict__.copy() for action in state.action_history],
            unresolved_questions=list(state.unresolved_questions),
            candidate_history={
                branch.diagnosis: list(branch.history)
                for branch in state.branches.values()
            },
            graph_context=list(state.graph_context),
            adjudication_count=state.adjudication_count,
        )
        self._emit(on_event, {
            "event": "traversal_complete",
            "mode": "investigator",
            "stop_reason": result.stop_reason,
            "total_nodes": result.total_nodes,
            "total_edges": result.total_edges,
            "rabbit_hole_count": 0,
            "contradiction_count": result.contradiction_count,
            "synthesis": result.synthesis,
            "rounds": result.rounds,
            "retrieval_count": result.retrieval_count,
            "reasoning_call_count": result.reasoning_call_count,
            "adjudication_count": result.adjudication_count,
            "duration_seconds": round(duration, 4),
        })
        return result

    @staticmethod
    def _load_concept_graph(path: Path) -> MedicalConceptGraph:
        try:
            return MedicalConceptGraph.load(path) if Path(path).exists() else MedicalConceptGraph()
        except (OSError, ValueError, json.JSONDecodeError):
            return MedicalConceptGraph()

    def _ingest_runtime_evidence(self, evidence: list[EvidenceChunk]) -> None:
        records = []
        for chunk in evidence:
            record = dict(chunk.metadata)
            record.update({"chunk_id": chunk.id, "text": chunk.text})
            records.append(record)
        self.concept_graph.ingest_records(records)

    def _update_candidates(
        self, state: CaseState, proposed: list[DiagnosticHypothesis]
    ) -> None:
        context = state.narrative + " " + " ".join(
            fact.claim for fact in state.facts
        )
        graph_scores = self.concept_graph.candidate_scores(
            context, [item.diagnosis for item in proposed]
        )
        reranked = [
            replace(
                item,
                score=round(
                    item.score + 0.12 * graph_scores.get(item.diagnosis, 0.0),
                    6,
                ),
            )
            for item in proposed
        ]
        observed: set[str] = set()
        for candidate in reranked:
            key = candidate.diagnosis.casefold()
            observed.add(key)
            branch = state.branches.get(key)
            if branch is None:
                branch = CandidateBranch(
                    diagnosis=candidate.diagnosis,
                    current=candidate,
                    best_score=candidate.score,
                )
                state.branches[key] = branch
            branch.observe(candidate, state.rounds)

        for key, branch in state.branches.items():
            if key not in observed:
                branch.decay(state.rounds)

        # Keep one omitted branch alive for a round. This makes backtracking
        # explicit without allowing stale candidates to grow without bound.
        active = [
            branch.current
            for branch in state.branches.values()
            if branch.misses <= 1
        ]
        state.candidates = sorted(
            active,
            key=lambda item: (item.score, item.confidence),
            reverse=True,
        )[: self.max_candidates]

    @staticmethod
    def _add_missing_questions(state: CaseState, raw_output: str) -> None:
        match = _JSON_OBJECT.search(raw_output or "")
        if not match:
            return
        try:
            payload = json.loads(match.group(0))
        except (TypeError, ValueError):
            return
        rows = payload.get("missing_information", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return
        for row in rows[:2]:
            question = str(row).strip()
            if question and question not in state.unresolved_questions:
                state.unresolved_questions.append(question[:240])

    def _adjudicate_high_impact_conflict(self, state: CaseState) -> None:
        if (
            state.adjudication_count >= self.max_adjudications
            or state.reasoning_call_count >= self.max_model_calls
            or not state.candidates
            or not hasattr(self.contradiction_detector, "check")
        ):
            return
        top = state.candidates[0]
        should_check = getattr(self.contradiction_detector, "should_check", None)
        for fact in state.facts:
            polarity = str(fact.metadata.get("polarity", "")).lower()
            if polarity != "negated":
                continue
            if callable(should_check) and not should_check(top.diagnosis, fact.claim):
                continue
            before = self._detector_llm_calls()
            result = self.contradiction_detector.check(top.diagnosis, fact.claim)
            after = self._detector_llm_calls()
            state.adjudication_count += 1
            state.reasoning_call_count += max(0, after - before)
            if result.label == "contradiction" and fact.id not in top.conflicting_fact_ids:
                revised = replace(
                    top,
                    score=round(top.score - 0.35, 6),
                    conflicting_fact_ids=top.conflicting_fact_ids + (fact.id,),
                )
                branch = state.branches[top.diagnosis.casefold()]
                branch.observe(revised, state.rounds)
                state.candidates[0] = revised
                state.candidates.sort(
                    key=lambda item: (item.score, item.confidence), reverse=True
                )
            return

    def _detector_llm_calls(self) -> int:
        info = getattr(self.contradiction_detector, "cache_info", None)
        if not callable(info):
            return 0
        return int(info().get("llm_calls", 0))

    def _is_adequate(self, state: CaseState) -> bool:
        if not state.candidates:
            return self.allow_abstention
        top = state.candidates[0]
        if top.conflicting_fact_ids or top.confidence < 0.65:
            return False
        if not top.supporting_fact_ids or not top.evidence_ids:
            return False
        if len(state.candidates) == 1:
            return True
        return top.score - state.candidates[1].score >= 0.12

    def _budget_stop_reason(self, state: CaseState) -> str | None:
        if state.rounds >= self.max_rounds:
            return "round_budget_exhausted"
        if state.retrieval_count >= self.max_retrievals:
            return "retrieval_budget_exhausted"
        if state.reasoning_call_count >= self.max_model_calls:
            return "model_call_budget_exhausted"
        return None

    def _choose_action(self, state: CaseState) -> InvestigationAction:
        findings = "; ".join(fact.claim for fact in state.facts[:10])
        if not state.candidates:
            value = self.action_policy.score({
                "uncertainty": 1.0,
                "small_margin": 1.0,
                "evidence_gap": 1.0,
                "novelty": 1.0,
                "cost": 0.5,
            })
            return InvestigationAction(
                kind="candidate_recovery",
                target="differential",
                query=f"Differential diagnosis for these findings: {findings}",
                value=value,
                reason="The previous round produced no usable candidate diagnosis.",
            )

        top = state.candidates[0]
        margin = (
            top.score - state.candidates[1].score
            if len(state.candidates) > 1 else 1.0
        )
        uncertainty = (
            -(top.confidence * math.log2(top.confidence)
              + (1.0 - top.confidence) * math.log2(1.0 - top.confidence))
            if 0.0 < top.confidence < 1.0 else 0.0
        )
        common = {
            "uncertainty": uncertainty,
            "small_margin": 1.0 - min(1.0, max(0.0, margin) / 0.3),
            "evidence_gap": float(not top.evidence_ids),
            "contradiction": float(bool(top.conflicting_fact_ids)),
        }
        actions: list[InvestigationAction] = []
        if top.conflicting_fact_ids:
            conflicts = ", ".join(top.conflicting_fact_ids)
            actions.append(InvestigationAction(
                kind="contradiction_investigation",
                target=top.diagnosis,
                query=(
                    f"Evidence for and against {top.diagnosis}; resolve findings "
                    f"{conflicts}. Patient findings: {findings}"
                ),
                value=self.action_policy.score(common | {"cost": 0.35}),
                reason=f"The leading candidate conflicts with facts {conflicts}.",
            ))

        if len(state.candidates) > 1:
            second = state.candidates[1]
            actions.append(InvestigationAction(
                    kind="discriminating_retrieval",
                    target=f"{top.diagnosis} vs {second.diagnosis}",
                    query=(
                        f"Findings that distinguish {top.diagnosis} from "
                        f"{second.diagnosis} in a patient with: {findings}"
                    ),
                    value=self.action_policy.score(
                        common | {"novelty": 0.6, "cost": 0.4}
                    ),
                    reason="The two leading candidates remain too close to separate.",
            ))

        if not top.evidence_ids:
            actions.append(InvestigationAction(
                kind="candidate_evidence",
                target=top.diagnosis,
                query=f"Clinical evidence for {top.diagnosis}: {findings}",
                value=self.action_policy.score(common | {"cost": 0.3}),
                reason="The leading candidate has no cited corpus evidence.",
            ))

        if state.unresolved_questions:
            question = state.unresolved_questions[-1]
            actions.append(InvestigationAction(
                kind="missing_fact_retrieval",
                target=question,
                query=f"Clinical significance of {question}. Patient findings: {findings}",
                value=self.action_policy.score(
                    common | {"novelty": 0.8, "cost": 0.45}
                ),
                reason="The model identified a missing discriminating fact.",
            ))

        actions.append(InvestigationAction(
            kind="similar_case_retrieval",
            target=top.diagnosis,
            query=(
                f"Similar clinical cases and rare alternatives to {top.diagnosis}: "
                f"{findings}"
            ),
            value=self.action_policy.score(
                common | {"novelty": 0.7, "cost": 0.6}
            ),
            reason="Confidence or patient-specific support remains inadequate.",
        ))
        return max(actions, key=lambda action: action.value)


__all__ = [
    "CandidateBranch",
    "CaseState",
    "InvestigationAction",
    "InvestigatorReasoner",
]
