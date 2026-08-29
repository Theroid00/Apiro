"""graph/edge.py — Edge dataclass for the Apiro belief graph."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Edge:
    """
    A directed relationship between two nodes in the belief graph.

    Attributes:
        parent_id:          Source node ID.
        child_id:           Target node ID.
        relation:           Semantic relationship type (default: "expands").
        contradiction_flag: True if ContradictionDetector flagged this pair.
        confidence:         Score from the contradiction detector (0–1).
    """
    parent_id:          str
    child_id:           str
    relation:           str   = "expands"   # supports | contradicts | refines | expands
    contradiction_flag: bool  = False
    confidence:         float = 1.0

    VALID_RELATIONS = frozenset({"supports", "contradicts", "refines", "expands"})

    def __post_init__(self):
        if self.relation not in self.VALID_RELATIONS:
            raise ValueError(
                f"Invalid relation {self.relation!r}. "
                f"Must be one of {sorted(self.VALID_RELATIONS)}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def __repr__(self) -> str:
        flag = " ⚡CONTRADICTION" if self.contradiction_flag else ""
        return (
            f"Edge({self.parent_id} --[{self.relation}]--> "
            f"{self.child_id}, conf={self.confidence:.2f}{flag})"
        )
