"""graph/node.py — provenance node for the bounded Apiro engines."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    """
    A single fact, hypothesis, or claim in the investigation provenance graph.

    Attributes:
        id:              Unique identifier (e.g. "node_0001").
        claim:           The clinical statement or hypothesis this node represents.
        domain:          Clinical domain label assigned to the node.
        entropy_score:   Uncertainty value displayed with the provenance.
        depth:           Provenance depth from the seed node (0 = seed).
        parent_id:       ID of the node that generated this one (None for seeds).
        sources:         List of PubMed IDs supporting this node's claim.
        metadata:        Arbitrary key-value store for extra data.
    """
    id:            str
    claim:         str
    domain:        str
    entropy_score: float
    contradiction_penalty: float = 0.0
    depth:         int           = 0
    parent_id:     str | None    = None
    sources:       list[str]     = field(default_factory=list)
    metadata:      dict          = field(default_factory=dict)

    def __post_init__(self):
        if self.entropy_score < 0:
            raise ValueError(f"entropy_score must be >= 0, got {self.entropy_score}")

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id!r}, "
            f"H={self.entropy_score:.3f}, "
            f"depth={self.depth}, "
            f"domain={self.domain!r}, "
            f"claim={self.claim[:60]!r})"
        )
