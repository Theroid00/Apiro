"""Bounded evidence audit for distractor-resistant differential diagnosis."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, replace
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
    """The single contrastive investigation allowed by the controller."""

    kind: str
    target: str
    query: str
    value: float
    reason: str


@dataclass
class CaseState:
    """Small auditable state for one case."""

    narrative: str
    facts: list[Node]
    evidence: list[EvidenceChunk] = field(default_factory=list)
    candidates: list[DiagnosticHypothesis] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    action_history: list[InvestigationAction] = field(default_factory=list)
    candidate_history: dict[str, list[dict]] = field(default_factory=dict)
    rounds: int = 0
    retrieval_count: int = 0
    reasoning_call_count: int = 0


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_INITIAL_PREFIX = """You are an evidence-auditing medical investigator.
The raw narrative may contain irrelevant or misleading details. Patient facts
and general medical knowledge are separate ledgers: only IDs in the patient
fact ledger describe this patient, while retrieved passages describe medicine
in general. An uncited detail must not affect the ranking.

Generate competing diagnoses, then test each against affirmed and negated
patient facts. For every cited evidence ID, include a short exact quote in
"evidence_spans" using this shape:
[{"evidence_id":"E1","quote":"exact words from the passage"}]
The root object must also include "missing_information", containing at most two
patient-specific questions. Never assume or retrieve an answer to those
questions.

"""

_REVISION_PREFIX = """Perform one counterfactual evidence audit of the prior differential.
The raw narrative may contain plausible distractors. Temporarily treat the
identified high-impact fact as non-discriminating and compare the two leading
diagnoses using the complete patient fact ledger and the newly retrieved
medical knowledge. Restore the prior leader only when independent patient fact
IDs support it. General medical knowledge cannot establish a missing patient
fact. Return the same strict JSON schema, including exact evidence_spans and at
most two missing_information questions.

High-impact fact: {fact}
Audit reason: {reason}
Prior candidate board:
{candidates}

"""


class InvestigatorReasoner(SimpleReasoner):
    """Rank a differential, audit its evidence use, and revise at most once."""

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
        self.max_rounds = min(2, max(1, int(max_rounds)))
        self.max_retrievals = min(2, max(1, int(max_retrievals)))
        self.max_model_calls = min(2, max(1, int(max_model_calls)))
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
        self.parse_fallback_count = 0
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
        timings["fact_ledger"] = time.monotonic() - stage

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
        raw_output = self._generate_json(
            _INITIAL_PREFIX + self._build_prompt(
                selected.text,
                facts,
                state.evidence,
                include_missing_information=True,
                include_evidence_spans=True,
            )
        )
        state.reasoning_call_count = 1
        state.rounds = 1
        state.candidates = self._rank_candidate_board(
            self._parse_audited_hypotheses(raw_output, facts, state.evidence),
            facts,
            state.evidence,
        )
        self._record_candidates(state)
        self._add_missing_questions(state, raw_output)
        initial_audit = self._audit(state.candidates, facts, state.evidence)
        timings["initial_reasoning_and_audit"] = time.monotonic() - stage

        stop_reason = "evidence_audit_passed"
        if initial_audit["needs_revision"]:
            budget_reason = self._revision_budget_reason(state)
            if budget_reason:
                stop_reason = budget_reason
            else:
                action = self._contrastive_action(state, initial_audit)
                state.action_history.append(action)
                self._emit(on_event, {
                    "event": "investigation_action",
                    "round": 2,
                    "kind": action.kind,
                    "target": action.target,
                    "value": action.value,
                    "reason": action.reason,
                })
                stage = time.monotonic()
                extra = self._retrieve(
                    action.query, facts, n_results=min(4, self.rag_top_k)
                )
                state.retrieval_count += 1
                state.evidence = self._merge_evidence(state.evidence, extra)
                raw_output = self._generate_json(
                    _REVISION_PREFIX.format(
                        fact=initial_audit.get("influential_fact_id") or "none identified",
                        reason="; ".join(initial_audit["reasons"]),
                        candidates=self._format_candidate_board(state.candidates),
                    )
                    + self._build_prompt(
                        selected.text,
                        facts,
                        state.evidence,
                        include_missing_information=True,
                        include_evidence_spans=True,
                    )
                )
                state.reasoning_call_count += 1
                state.rounds += 1
                state.candidates = self._rank_candidate_board(
                    self._parse_audited_hypotheses(
                        raw_output, facts, state.evidence
                    ),
                    facts,
                    state.evidence,
                )
                self._record_candidates(state)
                self._add_missing_questions(state, raw_output)
                timings["contrastive_revision"] = time.monotonic() - stage
                stop_reason = "counterfactual_revision_complete"

        final_audit = self._audit(state.candidates, facts, state.evidence)
        if not state.candidates and self.allow_abstention:
            stop_reason = "bounded_abstained"

        final_candidates = state.candidates[: self.output_diagnoses]
        stage = time.monotonic()
        self._add_hypotheses(graph, final_candidates, facts, on_event)
        timings["provenance_graph"] = time.monotonic() - stage
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
            candidate_history=state.candidate_history,
            evidence_audit={"initial": initial_audit, "final": final_audit},
            parse_fallback_count=self.parse_fallback_count,
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
            "parse_fallback_count": result.parse_fallback_count,
            "duration_seconds": round(duration, 4),
        })
        return result

    def _parse_audited_hypotheses(
        self, raw: str, facts: list[Node], evidence: list[EvidenceChunk]
    ) -> list[dict]:
        rows = self._parse_hypotheses(raw, facts, evidence)
        by_diagnosis: dict[str, list[tuple[str, str]]] = {}
        match = _JSON_OBJECT.search(raw or "")
        try:
            payload = json.loads(match.group(0)) if match else {}
        except (TypeError, ValueError):
            payload = {}
        aliases = {f"E{i}": chunk.id for i, chunk in enumerate(evidence, 1)}
        chunks = {chunk.id: chunk for chunk in evidence}
        allowed_ids = {
            row["diagnosis"].casefold(): set(row["evidence_ids"])
            for row in rows
        }
        for row in payload.get("hypotheses", []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            diagnosis_key = str(row.get("diagnosis") or "").strip().casefold()
            verified = []
            for span in row.get("evidence_spans", []):
                if not isinstance(span, dict):
                    continue
                evidence_id = aliases.get(
                    str(span.get("evidence_id") or ""),
                    str(span.get("evidence_id") or ""),
                )
                quote = str(span.get("quote") or "").strip()
                chunk = chunks.get(evidence_id)
                verified_span = (evidence_id, quote[:240])
                if (
                    chunk
                    and evidence_id in allowed_ids.get(diagnosis_key, set())
                    and len(quote) >= 8
                    and quote.casefold() in chunk.text.casefold()
                    and verified_span not in verified
                ):
                    verified.append(verified_span)
            by_diagnosis[diagnosis_key] = verified
        for row in rows:
            row["evidence_spans"] = by_diagnosis.get(
                row["diagnosis"].casefold(), []
            )
        return rows

    def _rank_candidate_board(
        self,
        rows: list[dict],
        facts: list[Node],
        evidence: list[EvidenceChunk],
    ) -> list[DiagnosticHypothesis]:
        fact_ids = {fact.id for fact in facts}
        ranked = []
        for row in rows:
            conflicts = list(row["conflicting_fact_ids"])
            for fact in facts:
                result = self.contradiction_detector.check_deterministic(
                    row["diagnosis"], fact.claim
                )
                if result.label == "contradiction" and fact.id not in conflicts:
                    conflicts.append(fact.id)
            supports = tuple(
                fact_id for fact_id in row["supporting_fact_ids"]
                if fact_id in fact_ids and fact_id not in conflicts
            )
            candidate = DiagnosticHypothesis(
                diagnosis=row["diagnosis"],
                confidence=row["confidence"],
                score=0.0,
                supporting_fact_ids=supports,
                conflicting_fact_ids=tuple(conflicts),
                evidence_ids=tuple(row["evidence_ids"]),
                evidence_spans=tuple(row.get("evidence_spans", [])),
            )
            ranked.append(replace(
                candidate,
                score=self._candidate_score(candidate, facts, evidence),
            ))
        return sorted(
            ranked, key=lambda item: (item.score, item.confidence), reverse=True
        )[: self.max_candidates]

    @staticmethod
    def _candidate_score(
        candidate: DiagnosticHypothesis,
        facts: list[Node],
        evidence: list[EvidenceChunk],
    ) -> float:
        facts_by_id = {fact.id: fact for fact in facts}
        evidence_by_id = {chunk.id: chunk for chunk in evidence}

        def weight(fact_id: str) -> float:
            fact = facts_by_id.get(fact_id)
            return max(0.0, float((fact.metadata if fact else {}).get("axiom_weight", 0.0)))

        support = 1.0 - math.exp(-sum(weight(i) for i in candidate.supporting_fact_ids))
        conflict = 1.0 - math.exp(-sum(weight(i) for i in candidate.conflicting_fact_ids))
        verified_ids = {evidence_id for evidence_id, _quote in candidate.evidence_spans}
        relevance = [
            1.0 - min(1.0, max(0.0, evidence_by_id[evidence_id].distance or 0.0))
            for evidence_id in verified_ids if evidence_id in evidence_by_id
        ]
        knowledge = sum(relevance) / len(relevance) if relevance else 0.0
        return round(
            0.35 * candidate.confidence
            + 0.50 * support
            + 0.15 * knowledge
            - 0.70 * conflict,
            6,
        )

    def _audit(
        self,
        candidates: list[DiagnosticHypothesis],
        facts: list[Node],
        evidence: list[EvidenceChunk],
    ) -> dict:
        if not candidates:
            return {
                "needs_revision": not self.allow_abstention,
                "reasons": ["no_usable_candidate"],
                "normalized_entropy": 1.0,
                "margin": 0.0,
                "distribution": [],
                "influential_fact_id": None,
                "rank_flip_without_fact": False,
            }

        peak = max(item.score for item in candidates)
        weights = [math.exp((item.score - peak) / 0.2) for item in candidates]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        entropy = (
            -sum(p * math.log2(p) for p in probabilities if p > 0.0)
            / math.log2(len(probabilities))
            if len(probabilities) > 1 else 0.0
        )
        top = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top.score - runner_up
        influential_fact_id = None
        largest_drop = 0.0
        score_without_fact = top.score
        for fact_id in top.supporting_fact_ids:
            reduced = replace(
                top,
                supporting_fact_ids=tuple(
                    item for item in top.supporting_fact_ids if item != fact_id
                ),
            )
            reduced_score = self._candidate_score(reduced, facts, evidence)
            drop = top.score - reduced_score
            if drop > largest_drop:
                largest_drop = drop
                influential_fact_id = fact_id
                score_without_fact = reduced_score

        rank_flip = bool(
            influential_fact_id and len(candidates) > 1
            and score_without_fact < candidates[1].score
        )
        reasons = []
        if top.conflicting_fact_ids:
            reasons.append("leading_candidate_conflicts_with_patient_fact")
        if not top.supporting_fact_ids:
            reasons.append("no_cited_patient_support")
        elif len(top.supporting_fact_ids) == 1:
            reasons.append("single_fact_dependence")
        if not top.evidence_spans:
            reasons.append("no_verified_evidence_span")
        if margin < 0.12:
            reasons.append("small_candidate_margin")
        if entropy > 0.82:
            reasons.append("high_differential_entropy")
        if rank_flip:
            reasons.append("counterfactual_rank_flip")
        return {
            "needs_revision": bool(reasons),
            "reasons": reasons,
            "normalized_entropy": round(entropy, 6),
            "margin": round(margin, 6),
            "distribution": [
                {"diagnosis": item.diagnosis, "probability": round(probability, 6)}
                for item, probability in zip(candidates, probabilities)
            ],
            "influential_fact_id": influential_fact_id,
            "rank_flip_without_fact": rank_flip,
            "score_without_influential_fact": round(score_without_fact, 6),
            "verified_evidence_spans": len(top.evidence_spans),
        }

    def _contrastive_action(
        self, state: CaseState, audit: dict
    ) -> InvestigationAction:
        names = [item.diagnosis for item in state.candidates[:2]]
        target = " vs ".join(names) or "differential recovery"
        findings = "; ".join(fact.claim for fact in state.facts[:10])
        query = (
            f"Medical evidence distinguishing {target}. Identify which findings "
            f"are specific, contradictory, or commonly misleading: {findings}"
        )
        return InvestigationAction(
            kind="counterfactual_evidence_audit",
            target=target,
            query=query,
            value=float(audit["normalized_entropy"]),
            reason="; ".join(audit["reasons"]),
        )

    def _revision_budget_reason(self, state: CaseState) -> str | None:
        if state.rounds >= self.max_rounds:
            return "round_budget_exhausted"
        if state.retrieval_count >= self.max_retrievals:
            return "retrieval_budget_exhausted"
        if state.reasoning_call_count >= self.max_model_calls:
            return "model_call_budget_exhausted"
        return None

    @staticmethod
    def _add_missing_questions(state: CaseState, raw_output: str) -> None:
        match = _JSON_OBJECT.search(raw_output or "")
        try:
            payload = json.loads(match.group(0)) if match else {}
        except (TypeError, ValueError):
            return
        rows = payload.get("missing_information", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return
        for row in rows[:2]:
            question = str(row).strip()
            if question and question not in state.unresolved_questions:
                state.unresolved_questions.append(question[:240])

    @staticmethod
    def _record_candidates(state: CaseState) -> None:
        for candidate in state.candidates:
            state.candidate_history.setdefault(candidate.diagnosis, []).append({
                "round": state.rounds,
                "score": candidate.score,
                "confidence": candidate.confidence,
                "supporting_fact_ids": list(candidate.supporting_fact_ids),
                "conflicting_fact_ids": list(candidate.conflicting_fact_ids),
                "verified_evidence_spans": len(candidate.evidence_spans),
            })

    @staticmethod
    def _format_candidate_board(candidates: list[DiagnosticHypothesis]) -> str:
        return "\n".join(
            f"- {item.diagnosis}: score={item.score:.3f}; "
            f"patient_support={list(item.supporting_fact_ids)}; "
            f"conflicts={list(item.conflicting_fact_ids)}; "
            f"verified_spans={len(item.evidence_spans)}"
            for item in candidates
        ) or "No usable candidates."


__all__ = ["CaseState", "InvestigationAction", "InvestigatorReasoner"]
