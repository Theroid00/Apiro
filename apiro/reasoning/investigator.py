"""Bounded multi-round clinical investigation with explicit resource budgets."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from apiro.axioms.seeding import axioms_to_seed_nodes
from apiro.config import (
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
class CaseState:
    """Compact working memory for one investigator-mode run."""

    narrative: str
    facts: list[Node]
    evidence: list[EvidenceChunk] = field(default_factory=list)
    candidates: list[DiagnosticHypothesis] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    action_history: list[InvestigationAction] = field(default_factory=list)
    rounds: int = 0
    retrieval_count: int = 0
    reasoning_call_count: int = 0


_INITIAL_PREFIX = """Generate a diverse candidate set before ranking it.
Keep rare but plausible diagnoses when patient-specific findings support them.
Compare candidates against negated findings and retrieved evidence.

"""

_REVISION_PREFIX = """You are revising a bounded differential after one targeted investigation.
Use the new evidence to compare the candidates. You may add, remove, or reorder
diagnoses. Do not preserve the old leader merely for consistency.

Controller action: {kind}
Target: {target}
Reason: {reason}
Prior candidates:
{candidates}

"""


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
    ):
        self.output_diagnoses = max(1, int(n_diagnoses))
        self.max_candidates = max(self.output_diagnoses, int(max_candidates))
        self.max_rounds = max(1, int(max_rounds))
        self.max_retrievals = max(1, int(max_retrievals))
        self.max_model_calls = max(1, int(max_model_calls))
        self.max_graph_nodes = max(self.max_candidates + 1, int(max_graph_nodes))
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
        state.evidence = self._retrieve(selected.text, facts)
        state.retrieval_count = 1
        timings["initial_retrieval"] = time.monotonic() - stage
        self._emit(on_event, {
            "event": "retrieval_complete",
            "round": 1,
            "evidence_count": len(state.evidence),
        })

        stage = time.monotonic()
        raw_output = self.llm_client.chat(
            _INITIAL_PREFIX + self._build_prompt(selected.text, facts, state.evidence)
        )
        state.reasoning_call_count = 1
        state.rounds = 1
        state.candidates = self._constrain_and_rank(
            self._parse_hypotheses(raw_output, facts, state.evidence), facts
        )
        timings["initial_reasoning"] = time.monotonic() - stage

        stop_reason = "bounded_adequate"
        while not self._is_adequate(state):
            budget_reason = self._budget_stop_reason(state)
            if budget_reason is not None:
                stop_reason = budget_reason
                break

            action = self._choose_action(state)
            state.action_history.append(action)
            state.unresolved_questions.append(action.reason)
            self._emit(on_event, {
                "event": "investigation_action",
                "round": state.rounds + 1,
                "kind": action.kind,
                "target": action.target,
                "value": action.value,
                "reason": action.reason,
            })

            extra = self._retrieve(
                action.query, facts, n_results=min(4, self.rag_top_k)
            )
            state.retrieval_count += 1
            state.evidence = self._merge_evidence(state.evidence, extra)

            revision_prompt = _REVISION_PREFIX.format(
                kind=action.kind,
                target=action.target,
                reason=action.reason,
                candidates="\n".join(
                    f"- {item.diagnosis}: confidence={item.confidence:.2f}, "
                    f"score={item.score:.2f}"
                    for item in state.candidates
                ) or "No usable candidates.",
            ) + self._build_prompt(selected.text, facts, state.evidence)
            raw_output = self.llm_client.chat(revision_prompt)
            state.reasoning_call_count += 1
            state.rounds += 1
            state.candidates = self._constrain_and_rank(
                self._parse_hypotheses(raw_output, facts, state.evidence), facts
            )
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
            "duration_seconds": round(duration, 4),
        })
        return result

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
            return InvestigationAction(
                kind="candidate_recovery",
                target="differential",
                query=f"Differential diagnosis for these findings: {findings}",
                value=1.0,
                reason="The previous round produced no usable candidate diagnosis.",
            )

        top = state.candidates[0]
        if top.conflicting_fact_ids:
            conflicts = ", ".join(top.conflicting_fact_ids)
            return InvestigationAction(
                kind="contradiction_investigation",
                target=top.diagnosis,
                query=(
                    f"Evidence for and against {top.diagnosis}; resolve findings "
                    f"{conflicts}. Patient findings: {findings}"
                ),
                value=0.95,
                reason=f"The leading candidate conflicts with facts {conflicts}.",
            )

        if len(state.candidates) > 1:
            second = state.candidates[1]
            margin = top.score - second.score
            if margin < 0.12:
                return InvestigationAction(
                    kind="discriminating_retrieval",
                    target=f"{top.diagnosis} vs {second.diagnosis}",
                    query=(
                        f"Findings that distinguish {top.diagnosis} from "
                        f"{second.diagnosis} in a patient with: {findings}"
                    ),
                    value=round(0.9 - max(0.0, margin), 3),
                    reason="The two leading candidates remain too close to separate.",
                )

        if not top.evidence_ids:
            return InvestigationAction(
                kind="candidate_evidence",
                target=top.diagnosis,
                query=f"Clinical evidence for {top.diagnosis}: {findings}",
                value=0.8,
                reason="The leading candidate has no cited corpus evidence.",
            )

        return InvestigationAction(
            kind="similar_case_retrieval",
            target=top.diagnosis,
            query=(
                f"Similar clinical cases and rare alternatives to {top.diagnosis}: "
                f"{findings}"
            ),
            value=0.7,
            reason="Confidence or patient-specific support remains inadequate.",
        )


__all__ = ["CaseState", "InvestigationAction", "InvestigatorReasoner"]
