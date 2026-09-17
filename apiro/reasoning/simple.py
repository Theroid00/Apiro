"""A bounded clinical reasoning path with one retrieval and one model call."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable

from apiro.axioms.seeding import axioms_to_seed_nodes
from apiro.config import (
    RAG_MAX_DISTANCE,
    SIMPLE_MAX_CONTEXT_CHARS,
    SIMPLE_MAX_FACTS,
    SIMPLE_RAG_TOP_K,
    SIMPLE_CORRECTIVE_PASS,
)
from apiro.context import select_clinical_context
from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.edge import Edge
from apiro.graph.node import Node
from apiro.parsing import parse_differential

from .models import DiagnosticHypothesis, EvidenceChunk, InvestigationResult

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT = """You are ranking differential diagnoses for one patient.
Use the clinical presentation, deterministic facts, and retrieved evidence below.
Return strict JSON only, with this shape:
{{"hypotheses":[{{"diagnosis":"name","confidence":0.0,"supporting_fact_ids":["ax_0"],"conflicting_fact_ids":[],"evidence_ids":["E1"]}}]{extra_schema}}}

Rules:
- Return at most {n_diagnoses} distinct diagnoses, most likely first.
- Confidence is the probability that the diagnosis is the best explanation for this patient.
- Cite only IDs shown below. Do not invent IDs.
- Put facts that argue against a diagnosis in conflicting_fact_ids.
- Prefer a concise specific diagnosis name; do not include explanations.
- {abstention_rule}

Clinical presentation:
{narrative}

Deterministic facts:
{facts}

Retrieved evidence:
{evidence}
"""

_REVISION_PREFIX = """The first differential did not pass deterministic quality checks.
Revise it once using the additional targeted evidence. Do not preserve a prior
diagnosis merely for consistency. Return the same strict JSON schema.

Previous ranked diagnoses:
{previous}

"""


class SimpleReasoner:
    """Generate, constrain, and rank a differential with a fixed call budget."""

    def __init__(
        self,
        *,
        embedder,
        llm_client,
        axiom_extractor,
        contradiction_detector,
        n_diagnoses: int = 3,
        max_facts: int = SIMPLE_MAX_FACTS,
        rag_top_k: int = SIMPLE_RAG_TOP_K,
        max_context_characters: int = SIMPLE_MAX_CONTEXT_CHARS,
        allow_abstention: bool = False,
        corrective_pass: bool = SIMPLE_CORRECTIVE_PASS,
    ):
        self.embedder = embedder
        self.llm_client = llm_client
        self.axiom_extractor = axiom_extractor
        self.contradiction_detector = contradiction_detector
        self.n_diagnoses = max(1, int(n_diagnoses))
        self.max_facts = max(1, int(max_facts))
        self.rag_top_k = max(1, int(rag_top_k))
        self.max_context_characters = max(1000, int(max_context_characters))
        self.allow_abstention = allow_abstention
        self.corrective_pass = bool(corrective_pass)

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
        seeds = axioms_to_seed_nodes(axioms)
        if not seeds:
            seeds = [Node(
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
        graph = BeliefGraph(max_depth=1, max_nodes=len(seeds) + self.n_diagnoses)
        for seed in seeds:
            graph.add_node(seed)
            self._emit(on_event, {
                "event": "seed_added",
                "node_id": seed.id,
                "claim": seed.claim,
                "domain": seed.domain,
                "entropy": seed.entropy_score,
                "depth": 0,
            })
        timings["extraction"] = time.monotonic() - stage

        stage = time.monotonic()
        selected = select_clinical_context(
            narrative, max_characters=self.max_context_characters
        )
        evidence = self._retrieve(selected.text, seeds)
        timings["retrieval"] = time.monotonic() - stage
        self._emit(on_event, {
            "event": "retrieval_complete",
            "evidence_count": len(evidence),
        })

        stage = time.monotonic()
        prompt = self._build_prompt(selected.text, seeds, evidence)
        raw_output = self.llm_client.chat(prompt)
        parsed = self._parse_hypotheses(raw_output, seeds, evidence)
        hypotheses = self._constrain_and_rank(parsed, seeds)
        reasoning_calls = 1
        retrieval_count = 1
        corrected = False
        if self.corrective_pass and self._needs_correction(hypotheses, evidence):
            query = self._correction_query(selected.text, seeds, hypotheses)
            extra_evidence = self._retrieve(
                query, seeds, n_results=min(4, self.rag_top_k)
            )
            retrieval_count += 1
            evidence = self._merge_evidence(evidence, extra_evidence)
            revision_prompt = _REVISION_PREFIX.format(
                previous="\n".join(
                    f"- {item.diagnosis} ({item.confidence:.2f})"
                    for item in hypotheses
                ) or "No usable diagnosis was produced."
            ) + self._build_prompt(selected.text, seeds, evidence)
            raw_output = self.llm_client.chat(revision_prompt)
            reasoning_calls += 1
            parsed = self._parse_hypotheses(raw_output, seeds, evidence)
            hypotheses = self._constrain_and_rank(parsed, seeds)
            corrected = True
            self._emit(on_event, {
                "event": "corrective_pass_complete",
                "evidence_count": len(evidence),
            })
        timings["generation"] = time.monotonic() - stage

        stage = time.monotonic()
        self._add_hypotheses(graph, hypotheses, seeds, on_event)
        timings["constraints_and_ranking"] = time.monotonic() - stage

        duration = time.monotonic() - started
        timings["total"] = duration
        result = InvestigationResult(
            mode="simple",
            graph=graph,
            stop_reason=("corrective_complete" if corrected else "bounded_complete"),
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            rabbit_hole_count=0,
            contradiction_count=sum(len(h.conflicting_fact_ids) for h in hypotheses),
            duration_seconds=duration,
            synthesis=[h.diagnosis for h in hypotheses],
            stage_timings={key: round(value, 6) for key, value in timings.items()},
            hypotheses=hypotheses,
            evidence=evidence,
            axioms=axioms,
            raw_output=raw_output,
            retrieval_count=retrieval_count,
            reasoning_call_count=reasoning_calls,
            rounds=2 if corrected else 1,
        )
        self._emit(on_event, {
            "event": "traversal_complete",
            "mode": "simple",
            "stop_reason": result.stop_reason,
            "total_nodes": result.total_nodes,
            "total_edges": result.total_edges,
            "rabbit_hole_count": result.rabbit_hole_count,
            "contradiction_count": result.contradiction_count,
            "synthesis": result.synthesis,
            "duration_seconds": round(duration, 4),
        })
        return result

    def _extract_axioms(self, narrative: str) -> list:
        try:
            axioms = self.axiom_extractor.extract(
                narrative, max_axioms=self.max_facts
            )
        except TypeError:
            # Lightweight test doubles and older external extractors may not
            # yet accept max_axioms.
            axioms = self.axiom_extractor.extract(narrative)
        return list(axioms or [])[: self.max_facts]

    def _retrieve(
        self, narrative: str, seeds: list[Node], *, n_results: int | None = None
    ) -> list[EvidenceChunk]:
        facts = " ".join(seed.claim for seed in seeds)
        query = f"{narrative.strip()}\nKey findings: {facts}"[: self.max_context_characters]
        raw_chunks = self.embedder.query(
            query, n_results=n_results or self.rag_top_k
        )
        evidence: list[EvidenceChunk] = []
        for index, chunk in enumerate(raw_chunks or [], 1):
            distance = chunk.get("distance")
            if (
                RAG_MAX_DISTANCE is not None
                and distance is not None
                and distance > RAG_MAX_DISTANCE
            ):
                continue
            chunk_id = str(chunk.get("chunk_id") or f"retrieved_{index}")
            metadata = {
                key: value
                for key, value in chunk.items()
                if key not in {"text", "chunk_id", "distance"}
            }
            evidence.append(EvidenceChunk(
                id=chunk_id,
                text=str(chunk.get("text") or ""),
                distance=float(distance) if distance is not None else None,
                metadata=metadata,
            ))
        return evidence

    def _needs_correction(
        self,
        hypotheses: list[DiagnosticHypothesis],
        evidence: list[EvidenceChunk],
    ) -> bool:
        if not hypotheses:
            return not self.allow_abstention
        top = hypotheses[0]
        if not evidence or top.conflicting_fact_ids:
            return True
        if not top.supporting_fact_ids and not top.evidence_ids:
            return True
        if top.confidence < 0.55:
            return True
        return (
            len(hypotheses) > 1
            and top.score - hypotheses[1].score < 0.08
        )

    @staticmethod
    def _correction_query(
        narrative: str,
        seeds: list[Node],
        hypotheses: list[DiagnosticHypothesis],
    ) -> str:
        candidates = ", ".join(item.diagnosis for item in hypotheses[:3])
        findings = "; ".join(seed.claim for seed in seeds[:8])
        return (
            f"Evidence distinguishing {candidates or 'the leading differential'} "
            f"for this patient. Key findings: {findings}. Case: {narrative}"
        )

    @staticmethod
    def _merge_evidence(
        initial: list[EvidenceChunk], extra: list[EvidenceChunk]
    ) -> list[EvidenceChunk]:
        merged = list(initial)
        seen = {chunk.id for chunk in initial}
        for chunk in extra:
            if chunk.id not in seen:
                merged.append(chunk)
                seen.add(chunk.id)
        return merged

    def _build_prompt(
        self,
        narrative: str,
        seeds: list[Node],
        evidence: list[EvidenceChunk],
        *,
        include_missing_information: bool = False,
    ) -> str:
        facts_text = "\n".join(f"{seed.id}: {seed.claim}" for seed in seeds)
        evidence_text = "\n\n".join(
            f"E{index} [{chunk.id}]: {chunk.text}"
            for index, chunk in enumerate(evidence, 1)
        ) or "No sufficiently relevant corpus evidence was retrieved."
        return _PROMPT.format(
            n_diagnoses=self.n_diagnoses,
            extra_schema=(
                ',"missing_information":["one concise discriminating question"]'
                if include_missing_information else ""
            ),
            abstention_rule=(
                'If evidence is insufficient, return {"hypotheses":[]}.'
                if self.allow_abstention
                else "Return at least one diagnosis."
            ),
            narrative=narrative,
            facts=facts_text,
            evidence=evidence_text,
        )

    def _parse_hypotheses(
        self, raw: str, seeds: list[Node], evidence: list[EvidenceChunk]
    ) -> list[dict]:
        payload = None
        match = _JSON_OBJECT.search(raw or "")
        if match:
            try:
                payload = json.loads(match.group(0))
            except (TypeError, ValueError):
                logger.warning("Simple reasoner received malformed JSON; using text fallback")

        rows = payload.get("hypotheses", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []

        valid_fact_ids = {seed.id for seed in seeds}
        evidence_aliases = {
            f"E{index}": chunk.id for index, chunk in enumerate(evidence, 1)
        }
        valid_evidence_ids = {chunk.id for chunk in evidence}
        parsed: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            diagnosis = str(row.get("diagnosis") or "").strip()
            key = diagnosis.casefold()
            if not diagnosis or key in seen:
                continue
            seen.add(key)
            try:
                confidence = float(row.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            evidence_ids = []
            for item in self._as_string_list(row.get("evidence_ids")):
                mapped = evidence_aliases.get(item, item)
                if mapped in valid_evidence_ids and mapped not in evidence_ids:
                    evidence_ids.append(mapped)
            parsed.append({
                "diagnosis": diagnosis[:200],
                "confidence": min(1.0, max(0.0, confidence)),
                "supporting_fact_ids": self._valid_ids(
                    row.get("supporting_fact_ids"), valid_fact_ids
                ),
                "conflicting_fact_ids": self._valid_ids(
                    row.get("conflicting_fact_ids"), valid_fact_ids
                ),
                "evidence_ids": evidence_ids,
            })
            if len(parsed) >= self.n_diagnoses:
                break

        if parsed or (self.allow_abstention and isinstance(payload, dict)):
            return parsed

        fallback = parse_differential(raw or "", limit=self.n_diagnoses)
        return [
            {
                "diagnosis": diagnosis,
                "confidence": max(0.35, 0.6 - index * 0.1),
                "supporting_fact_ids": [],
                "conflicting_fact_ids": [],
                "evidence_ids": [],
            }
            for index, diagnosis in enumerate(fallback)
        ]

    @staticmethod
    def _as_string_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, (str, int))]

    @classmethod
    def _valid_ids(cls, value, allowed: set[str]) -> list[str]:
        output: list[str] = []
        for item in cls._as_string_list(value):
            if item in allowed and item not in output:
                output.append(item)
        return output

    def _constrain_and_rank(
        self, rows: list[dict], seeds: list[Node]
    ) -> list[DiagnosticHypothesis]:
        seed_by_id = {seed.id: seed for seed in seeds}
        ranked: list[DiagnosticHypothesis] = []
        for row in rows:
            conflicts = list(row["conflicting_fact_ids"])
            for seed in seeds:
                result = self.contradiction_detector.check_deterministic(
                    row["diagnosis"], seed.claim
                )
                if result.label == "contradiction" and seed.id not in conflicts:
                    conflicts.append(seed.id)

            supports = [
                fact_id for fact_id in row["supporting_fact_ids"]
                if fact_id in seed_by_id and fact_id not in conflicts
            ]
            score = (
                row["confidence"]
                + 0.04 * len(supports)
                + 0.02 * len(row["evidence_ids"])
                - 0.35 * len(conflicts)
            )
            ranked.append(DiagnosticHypothesis(
                diagnosis=row["diagnosis"],
                confidence=row["confidence"],
                score=round(score, 6),
                supporting_fact_ids=tuple(supports),
                conflicting_fact_ids=tuple(conflicts),
                evidence_ids=tuple(row["evidence_ids"]),
            ))
        return sorted(ranked, key=lambda item: (item.score, item.confidence), reverse=True)

    def _add_hypotheses(
        self,
        graph: BeliefGraph,
        hypotheses: list[DiagnosticHypothesis],
        seeds: list[Node],
        on_event,
    ) -> None:
        fallback_parent = seeds[0].id
        for index, hypothesis in enumerate(hypotheses, 1):
            parent_id = (
                hypothesis.supporting_fact_ids[0]
                if hypothesis.supporting_fact_ids
                else fallback_parent
            )
            node = Node(
                id=f"dx_{index}",
                claim=hypothesis.diagnosis,
                entropy_score=self._binary_entropy(hypothesis.confidence),
                domain="diagnosis",
                depth=1,
                parent_id=parent_id,
                sources=list(hypothesis.evidence_ids),
                metadata={
                    "confidence": hypothesis.confidence,
                    "rank_score": hypothesis.score,
                    "supporting_fact_ids": list(hypothesis.supporting_fact_ids),
                    "conflicting_fact_ids": list(hypothesis.conflicting_fact_ids),
                    "evidence_ids": list(hypothesis.evidence_ids),
                },
            )
            graph.add_node(node)
            for fact_id in hypothesis.supporting_fact_ids:
                graph.add_edge(Edge(
                    parent_id=fact_id,
                    child_id=node.id,
                    relation="supports",
                    confidence=hypothesis.confidence,
                ))
            if not hypothesis.supporting_fact_ids:
                graph.add_edge(Edge(
                    parent_id=fallback_parent,
                    child_id=node.id,
                    relation="expands",
                    confidence=hypothesis.confidence,
                ))
            for fact_id in hypothesis.conflicting_fact_ids:
                graph.add_edge(Edge(
                    parent_id=fact_id,
                    child_id=node.id,
                    relation="contradicts",
                    contradiction_flag=True,
                    confidence=0.9,
                ))
            self._emit(on_event, {
                "event": "node_expanded",
                "node_id": node.id,
                "claim": node.claim,
                "domain": node.domain,
                "entropy": node.entropy_score,
                "depth": node.depth,
                "parent_id": parent_id,
                "node": {
                    "id": node.id,
                    "claim": node.claim,
                    "domain": node.domain,
                    "entropy": node.entropy_score,
                    "depth": node.depth,
                    "confidence": hypothesis.confidence,
                },
            })

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        p = min(1.0, max(0.0, probability))
        if p in {0.0, 1.0}:
            return 0.0
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    @staticmethod
    def _emit(callback, event: dict) -> None:
        if callback is not None:
            callback(event)
