"""
graph/rabbit_hole.py — RabbitHoleDetector
==========================================
Detects when traversal has gone into a rabbit hole:
the entropy curve reverses (starts rising again) after an initial decline,
at depth >= min_depth.

Spec (TC-2.4): fires ONLY after 3+ consecutive decreases followed by
2+ consecutive increases. A single-step entropy blip must NOT fire.

When flagged, the traversal loop skips this node and picks frontier[1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.node import Node
from apiro.config import RABBIT_HOLE_MIN_DEPTH, RABBIT_HOLE_REVERSAL_WINDOW


@dataclass
class RabbitHoleEvent:
    """Logged whenever a rabbit hole node is flagged."""
    node_id:   str
    node_claim: str
    depth:     int
    trend:     float   # entropy trend at time of detection


class RabbitHoleDetector:
    """
    Fires when:
      1. current_node.depth >= min_depth (not a root-level node)
      2. The entropy trend over the last `reversal_window` expanded nodes
         has turned POSITIVE (entropy rising after initial decline).
    """

    def __init__(
        self,
        min_depth: int = RABBIT_HOLE_MIN_DEPTH,
        reversal_window: int = RABBIT_HOLE_REVERSAL_WINDOW,
    ):
        self.min_depth       = min_depth
        self.reversal_window = reversal_window
        self.events: list[RabbitHoleEvent] = []


    def ancestor_entropies(self, graph: BeliefGraph, node: Node) -> list[float]:
        """
        Entropy values along this node's own lineage, root first.

        A rabbit hole is a property of a *reasoning path*, not of the engine's
        global mood, so the reversal must be measured on the chain of claims
        that actually produced this node.
        """
        chain: list[float] = []
        seen: set[str] = set()
        cur: Node | None = node
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            h = cur.entropy_score if cur.entropy_score is not None else 0.0
            chain.append(h)
            parent_id = getattr(cur, "parent_id", None)
            cur = graph.nodes.get(parent_id) if parent_id else None
        chain.reverse()
        return chain

    @staticmethod
    def _reversal(series: list[float]) -> bool:
        """
        True when the series is rising overall AND still rising at the end.

        Condition 2 blocks completed blips:
          [0.5, 0.52, 0.4] → slope negative OR last pair falls → no fire.
          [0.6, 0.5, 0.8]  → slope positive AND last pair rises → FIRES.
        """
        if len(series) < 2:
            return False
        x = np.arange(len(series), dtype=float)
        slope = float(np.polyfit(x, series, 1)[0])
        if slope <= 0.0:
            return False
        return series[-1] > series[-2]

    def check(self, graph: BeliefGraph, current_node: Node) -> bool:
        """
        Return True if this node is a rabbit hole candidate.
        Does NOT mutate the node — call flag_rabbit_hole() to do that.

        Detection is path-local: the entropy curve along this node's ancestor
        chain must have reversed (declining, then rising again).

        This replaces a graph-global check that read the last N expansions
        anywhere in the graph. Because the entropy-first frontier hands this
        detector the highest-entropy node at depth >= 1, and a rising global
        trend is exactly what a fresh batch of open questions looks like, the
        global version systematically flagged the single most informative node
        on the frontier and permanently removed it from both the traversal and
        the final synthesis — regardless of whether that node's own reasoning
        path had gone anywhere bad.

        Falls back to the global window when the node has no lineage in the
        graph (detached nodes, as constructed in the unit tests).
        """
        if current_node.depth < self.min_depth:
            return False

        chain = self.ancestor_entropies(graph, current_node)
        if len(chain) >= 3:
            return self._reversal(chain[-self.reversal_window:])

        # No usable lineage — fall back to the global expansion window.
        recent = graph.get_recent_entropies(self.reversal_window)
        return self._reversal(recent)

    def flag_rabbit_hole(self, node: Node, graph: BeliefGraph) -> None:
        """
        Mark the node as a rabbit hole, log the event, and record metadata.
        The traversal loop then skips to frontier[1].
        """
        node.is_rabbit_hole = True
        trend = graph.get_entropy_trend(self.reversal_window)
        node.metadata["rabbit_hole_trend"] = trend
        node.metadata["rabbit_hole_depth"]  = node.depth

        self.events.append(RabbitHoleEvent(
            node_id=node.id,
            node_claim=node.claim,
            depth=node.depth,
            trend=trend,
        ))

    def get_status(self, graph: BeliefGraph, current_node: Node) -> dict:
        """Diagnostic dict for inspection/logging."""
        trend = graph.get_entropy_trend(self.reversal_window)
        return {
            "is_rabbit_hole":  self.check(graph, current_node),
            "node_depth":      current_node.depth,
            "min_depth":       self.min_depth,
            "entropy_trend":   round(trend, 5),
            "reversal_window": self.reversal_window,
            "total_flagged":   len(self.events),
        }
