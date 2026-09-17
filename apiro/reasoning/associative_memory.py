"""Bounded associative retrieval with pattern completion and separation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .concept_graph import MedicalConceptGraph
from .models import EvidenceChunk


_WORDS = re.compile(r"[a-z0-9][a-z0-9+\-']{1,}", re.IGNORECASE)


def _terms(text: str) -> frozenset[str]:
    return frozenset(word.lower() for word in _WORDS.findall(text))


def _similarity(left: str, right: str) -> float:
    a, b = _terms(left), _terms(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class MemoryResult:
    evidence: list[EvidenceChunk]
    retrieval_count: int
    graph_context: tuple[str, ...]


class AssociativeMemory:
    """Retrieve general evidence and separated similar-case memories."""

    def __init__(
        self,
        retrieve: Callable[[str, int], list[EvidenceChunk]],
        concept_graph: MedicalConceptGraph | None = None,
        *,
        diversity_weight: float = 0.35,
    ):
        self.retrieve = retrieve
        self.concept_graph = concept_graph or MedicalConceptGraph()
        self.diversity_weight = min(1.0, max(0.0, float(diversity_weight)))

    def recall(
        self,
        narrative: str,
        findings: list[str],
        *,
        max_results: int = 8,
        max_calls: int = 2,
    ) -> MemoryResult:
        graph_context = tuple(
            label for label, _score in self.concept_graph.related_concepts(
                narrative + " " + " ".join(findings), limit=6
            )
        )
        queries = [
            narrative + "\nKey findings: " + "; ".join(findings),
            (
                "Similar clinical cases with final diagnoses and distinguishing "
                "findings: " + "; ".join(findings)
            ),
        ]
        if graph_context:
            queries[1] += ". Associated concepts: " + ", ".join(graph_context)

        candidates: list[EvidenceChunk] = []
        calls = 0
        for query in queries[: max(1, int(max_calls))]:
            candidates.extend(self.retrieve(query, max_results))
            calls += 1
        return MemoryResult(
            evidence=self.select_diverse(candidates, limit=max_results),
            retrieval_count=calls,
            graph_context=graph_context,
        )

    def select_diverse(
        self, chunks: list[EvidenceChunk], *, limit: int
    ) -> list[EvidenceChunk]:
        """MMR-like selection keeps near-duplicate cases from collapsing memory."""
        unique: list[EvidenceChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.id not in seen and chunk.text.strip():
                unique.append(chunk)
                seen.add(chunk.id)
        selected: list[EvidenceChunk] = []
        remaining = list(unique)
        while remaining and len(selected) < max(0, int(limit)):
            def utility(chunk: EvidenceChunk) -> tuple[float, float]:
                relevance = 1.0 - min(1.0, max(0.0, chunk.distance or 0.0))
                redundancy = max(
                    (_similarity(chunk.text, prior.text) for prior in selected),
                    default=0.0,
                )
                return (
                    relevance - self.diversity_weight * redundancy,
                    relevance,
                )

            best = max(remaining, key=utility)
            selected.append(best)
            remaining.remove(best)
        return selected


__all__ = ["AssociativeMemory", "MemoryResult"]
