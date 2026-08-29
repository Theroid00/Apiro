"""graph/node.py — Node dataclass for the Apiro belief graph."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    """
    A single hypothesis, finding, or claim within the belief graph.

    Attributes:
        id:              Unique identifier (e.g. "node_0001").
        claim:           The clinical statement or hypothesis this node represents.
        domain:          One of the 7 medical domains from config.DOMAINS.
        entropy_score:   Shannon entropy of first token at time of node creation.
                         Higher = model is more uncertain about this claim.
        resolved:        True once this node has been expanded (children generated).
        is_rabbit_hole:  True if RabbitHoleDetector flagged this node.
        depth:           Depth from the seed node (0 = seed).
        parent_id:       ID of the node that generated this one (None for seeds).
        sources:         List of PubMed IDs supporting this node's claim.
        metadata:        Arbitrary key-value store for extra data.
    """
    id:            str
    claim:         str
    domain:        str
    entropy_score: float
    resolved:      bool          = False
    is_rabbit_hole: bool         = False
    contradiction_penalty: float = 0.0
    depth:         int           = 0
    parent_id:     str | None    = None
    sources:       list[str]     = field(default_factory=list)
    metadata:      dict          = field(default_factory=dict)

    def __post_init__(self):
        if self.entropy_score < 0:
            raise ValueError(f"entropy_score must be >= 0, got {self.entropy_score}")

    @property
    def entropy(self) -> float:
        """Alias for entropy_score — used by traversal/expander code."""
        return self.entropy_score

    def __repr__(self) -> str:
        status = "✓" if self.resolved else ("🐇" if self.is_rabbit_hole else "○")
        return (
            f"Node({status} id={self.id!r}, "
            f"H={self.entropy_score:.3f}, "
            f"depth={self.depth}, "
            f"domain={self.domain!r}, "
            f"claim={self.claim[:60]!r})"
        )
