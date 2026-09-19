"""Small, serializable contracts for the simplified investigation path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EvidenceChunk:
    id: str
    text: str
    distance: Optional[float]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticHypothesis:
    diagnosis: str
    confidence: float
    score: float
    supporting_fact_ids: tuple[str, ...] = ()
    conflicting_fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_spans: tuple[tuple[str, str], ...] = ()


@dataclass
class InvestigationResult:
    """One result shape shared by the bounded reasoning engines."""

    mode: str
    graph: object
    stop_reason: str
    total_nodes: int
    total_edges: int
    rabbit_hole_count: int
    contradiction_count: int
    duration_seconds: float
    synthesis: list[str]
    saturation_status: Optional[object] = None
    stage_timings: Optional[dict[str, float]] = None
    hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)
    evidence: list[EvidenceChunk] = field(default_factory=list)
    axioms: list[object] = field(default_factory=list)
    raw_output: str = ""
    model_telemetry: Optional[dict] = None
    retrieval_count: int = 0
    reasoning_call_count: int = 0
    rounds: int = 0
    action_history: list[dict] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    candidate_history: dict[str, list[dict]] = field(default_factory=dict)
    graph_context: list[str] = field(default_factory=list)
    adjudication_count: int = 0
    evidence_audit: dict = field(default_factory=dict)
    parse_fallback_count: int = 0
