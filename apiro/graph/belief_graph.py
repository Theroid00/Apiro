"""Small provenance graph shared by the bounded reasoning engines."""

from __future__ import annotations

import json
from pathlib import Path

from apiro.graph.edge import Edge
from apiro.graph.node import Node


class BeliefGraph:
    """Store facts, hypotheses, and their evidence relationships.

    Reasoning happens in the bounded reasoners. This class only preserves the
    result for inspection, streaming, and optional JSON export.
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node once."""
        self.nodes.setdefault(node.id, node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge after validating both endpoints."""
        if edge.parent_id not in self.nodes:
            raise ValueError(f"Parent node {edge.parent_id!r} not in graph.")
        if edge.child_id not in self.nodes:
            raise ValueError(f"Child node {edge.child_id!r} not in graph.")
        self.edges.append(edge)

    def export_json(self, path: Path | None = None) -> dict:
        """Return the graph as JSON-compatible data and optionally write it."""
        data = {
            "nodes": [
                {
                    "id": node.id,
                    "claim": node.claim,
                    "domain": node.domain,
                    "entropy_score": node.entropy_score,
                    # Retained as stable UI fields; provenance nodes do not
                    # have traversal state.
                    "resolved": False,
                    "is_rabbit_hole": False,
                    "contradiction_penalty": node.contradiction_penalty,
                    "depth": node.depth,
                    "parent_id": node.parent_id,
                    "sources": node.sources,
                    "metadata": node.metadata,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "parent_id": edge.parent_id,
                    "child_id": edge.child_id,
                    "relation": edge.relation,
                    "contradiction_flag": edge.contradiction_flag,
                    "confidence": edge.confidence,
                }
                for edge in self.edges
            ],
            "stats": {
                "n_nodes": len(self.nodes),
                "n_edges": len(self.edges),
            },
        }
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def __repr__(self) -> str:
        return f"BeliefGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
