"""Sparse typed medical concept graph with bounded personalized propagation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKENS = re.compile(r"[a-z0-9][a-z0-9+\-']{1,}", re.IGNORECASE)
_CONCEPT_FIELDS = (
    "condition_tags",
    "disease_name",
    "diagnosis",
    "condition",
    "hpo_name",
    "gene_symbol",
    "medical_domain",
)


def _normalise(value: str) -> str:
    return " ".join(_TOKENS.findall(str(value).lower()))


def _concept_values(record: dict) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for field in _CONCEPT_FIELDS:
        raw = record.get(field)
        if raw is None:
            continue
        values = raw if isinstance(raw, list) else str(raw).split(",")
        for value in values:
            label = str(value).strip()
            if label:
                output.append((field, label))
    return output


@dataclass(frozen=True)
class ConceptNode:
    id: str
    label: str
    kind: str


class MedicalConceptGraph:
    """Small serializable graph used for associative retrieval and reranking."""

    SCHEMA_VERSION = 1

    def __init__(self):
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: dict[str, dict[str, float]] = defaultdict(dict)
        self.edge_types: dict[tuple[str, str], str] = {}

    @staticmethod
    def concept_id(kind: str, label: str) -> str:
        return f"{kind}:{_normalise(label)}"

    def add_node(self, node_id: str, label: str, kind: str) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = ConceptNode(node_id, str(label), str(kind))

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        relation: str,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        if source not in self.nodes or target not in self.nodes or source == target:
            return
        bounded = min(1.0, max(0.0, float(weight)))
        self.edges[source][target] = max(self.edges[source].get(target, 0.0), bounded)
        self.edge_types[(source, target)] = relation
        if bidirectional:
            self.edges[target][source] = max(self.edges[target].get(source, 0.0), bounded)
            self.edge_types[(target, source)] = relation

    def ingest_records(self, records: Iterable[dict]) -> None:
        """Add document-to-concept and concept co-occurrence relationships."""
        for index, record in enumerate(records):
            document_id = str(
                record.get("chunk_id") or record.get("id") or f"record_{index}"
            )
            doc_node = f"document:{document_id}"
            title = str(record.get("title") or document_id)
            self.add_node(doc_node, title, "document")
            concept_ids: list[str] = []
            for field, label in _concept_values(record):
                kind = {
                    "condition_tags": "disease",
                    "disease_name": "disease",
                    "diagnosis": "disease",
                    "condition": "disease",
                    "hpo_name": "phenotype",
                    "gene_symbol": "gene",
                    "medical_domain": "domain",
                }[field]
                node_id = self.concept_id(kind, label)
                self.add_node(node_id, label, kind)
                self.add_edge(
                    node_id, doc_node, relation="supported_by", weight=0.9
                )
                concept_ids.append(node_id)
            for offset, source in enumerate(concept_ids):
                for target in concept_ids[offset + 1:]:
                    self.add_edge(
                        source, target, relation="co_occurs", weight=0.65
                    )

    def matching_nodes(self, text: str) -> list[str]:
        normalised = _normalise(text)
        tokens = set(normalised.split())
        matches: list[tuple[int, str]] = []
        for node_id, node in self.nodes.items():
            if node.kind == "document":
                continue
            label = _normalise(node.label)
            label_tokens = set(label.split())
            overlap = len(tokens & label_tokens)
            if label and (label in normalised or overlap >= max(1, len(label_tokens) // 2)):
                matches.append((overlap, node_id))
        return [node_id for _score, node_id in sorted(matches, reverse=True)]

    def personalized_pagerank(
        self,
        seed_nodes: Iterable[str],
        *,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[str, float]:
        seeds = [node_id for node_id in dict.fromkeys(seed_nodes) if node_id in self.nodes]
        if not seeds:
            return {}
        active = set(seeds)
        frontier = set(seeds)
        for _ in range(3):
            frontier = {
                target
                for source in frontier
                for target in self.edges.get(source, {})
                if target not in active
            }
            active.update(frontier)
            if not frontier:
                break
        teleport = {node_id: 1.0 / len(seeds) for node_id in seeds}
        ranks = {node_id: teleport.get(node_id, 0.0) for node_id in active}
        for _ in range(max(1, iterations)):
            updated = {
                node_id: (1.0 - damping) * teleport.get(node_id, 0.0)
                for node_id in active
            }
            for source in active:
                neighbours = {
                    target: weight
                    for target, weight in self.edges.get(source, {}).items()
                    if target in active
                }
                total = sum(neighbours.values())
                if total <= 0.0:
                    continue
                for target, weight in neighbours.items():
                    updated[target] += damping * ranks.get(source, 0.0) * weight / total
            ranks = updated
        return ranks

    def related_concepts(self, text: str, *, limit: int = 8) -> list[tuple[str, float]]:
        seeds = self.matching_nodes(text)
        ranks = self.personalized_pagerank(seeds)
        rows = [
            (node.label, score)
            for node_id, score in ranks.items()
            if node_id not in seeds
            for node in [self.nodes[node_id]]
            if node.kind in {"disease", "phenotype", "gene"}
        ]
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[: max(0, int(limit))]

    def candidate_scores(
        self, context: str, candidates: Iterable[str]
    ) -> dict[str, float]:
        ranks = self.personalized_pagerank(self.matching_nodes(context))
        output: dict[str, float] = {}
        for candidate in candidates:
            matches = self.matching_nodes(candidate)
            output[candidate] = max((ranks.get(node_id, 0.0) for node_id in matches), default=0.0)
        peak = max(output.values(), default=0.0)
        return {
            candidate: score / peak if peak > 0.0 else 0.0
            for candidate, score in output.items()
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "nodes": [node.__dict__ for node in self.nodes.values()],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "relation": self.edge_types.get((source, target), "related_to"),
                }
                for source, targets in self.edges.items()
                for target, weight in targets.items()
                if source < target
            ],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict) -> "MedicalConceptGraph":
        graph = cls()
        for row in payload.get("nodes", []):
            graph.add_node(str(row["id"]), str(row["label"]), str(row["kind"]))
        for row in payload.get("edges", []):
            graph.add_edge(
                str(row["source"]),
                str(row["target"]),
                relation=str(row.get("relation", "related_to")),
                weight=float(row.get("weight", 1.0)),
            )
        return graph

    @classmethod
    def load(cls, path: Path) -> "MedicalConceptGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported concept graph schema version")
        return cls.from_dict(payload)


__all__ = ["ConceptNode", "MedicalConceptGraph"]
