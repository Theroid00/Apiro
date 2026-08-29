"""
tests/test_graph.py
====================
Unit tests for the Phase 1 graph layer:
    - Node dataclass
    - Edge dataclass
    - BeliefGraph (add, query, frontier, export/import)
    - SaturationDetector
    - RabbitHoleDetector

Run with:
    python -m pytest tests/test_graph.py -v
"""

import math
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
from apiro.graph.node import Node
from apiro.graph.edge import Edge
from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.saturation import SaturationDetector
from apiro.graph.rabbit_hole import RabbitHoleDetector


# ============================================================
# Helpers
# ============================================================

def make_node(id: str, entropy: float, domain: str = "pathophysiology", depth: int = 0) -> Node:
    return Node(id=id, claim=f"Claim for {id}", domain=domain, entropy_score=entropy, depth=depth)


def make_edge(parent: str, child: str, relation: str = "supports") -> Edge:
    return Edge(parent_id=parent, child_id=child, relation=relation)


def graph_with_nodes(*entropy_values) -> BeliefGraph:
    """Create a linear chain graph with given entropy values (for saturation/rabbit-hole tests)."""
    g = BeliefGraph()
    prev_id = None
    for i, h in enumerate(entropy_values):
        nid  = f"node_{i:03d}"
        node = make_node(nid, h, depth=i)
        g.add_node(node)
        if prev_id:
            g.add_edge(make_edge(prev_id, nid))
        prev_id = nid
    return g


# ============================================================
# Node tests
# ============================================================

class TestNode:
    def test_valid_creation(self):
        n = make_node("n1", 1.5)
        assert n.id == "n1"
        assert n.entropy_score == 1.5
        assert n.resolved is False
        assert n.is_rabbit_hole is False
        assert n.depth == 0
        assert n.sources == []

    def test_negative_entropy_raises(self):
        with pytest.raises(ValueError, match="entropy_score must be"):
            Node(id="bad", claim="x", domain="lab", entropy_score=-0.1)

    def test_zero_entropy_allowed(self):
        n = Node(id="zero", claim="certain", domain="lab", entropy_score=0.0)
        assert n.entropy_score == 0.0

    def test_repr_contains_id(self):
        n = make_node("abc", 2.0)
        assert "abc" in repr(n)
        assert "2.000" in repr(n)

    def test_resolved_repr(self):
        n = make_node("r1", 0.5)
        n.resolved = True
        assert "✓" in repr(n)

    def test_rabbit_hole_repr(self):
        n = make_node("rh1", 0.5)
        n.is_rabbit_hole = True
        assert "🐇" in repr(n)


# ============================================================
# Edge tests
# ============================================================

class TestEdge:
    def test_valid_relations(self):
        for rel in ("supports", "contradicts", "refines", "expands"):
            e = Edge(parent_id="p", child_id="c", relation=rel)
            assert e.relation == rel

    def test_invalid_relation_raises(self):
        with pytest.raises(ValueError, match="Invalid relation"):
            Edge(parent_id="p", child_id="c", relation="opposes")

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence must be"):
            Edge(parent_id="p", child_id="c", relation="supports", confidence=1.5)

    def test_contradiction_flag_default_false(self):
        e = make_edge("p", "c")
        assert e.contradiction_flag is False

    def test_repr_contradiction(self):
        e = Edge(parent_id="p", child_id="c", relation="contradicts", contradiction_flag=True)
        assert "CONTRADICTION" in repr(e)


# ============================================================
# BeliefGraph tests
# ============================================================

class TestBeliefGraph:
    def test_empty_graph(self):
        g = BeliefGraph()
        assert g.n_nodes == 0
        assert g.n_edges == 0
        assert g.get_frontier() == []
        assert g.mean_entropy is None

    def test_add_node(self):
        g = BeliefGraph()
        g.add_node(make_node("n1", 1.0))
        assert g.n_nodes == 1
        assert "n1" in g.nodes

    def test_add_duplicate_node_ignored(self):
        g = BeliefGraph()
        g.add_node(make_node("n1", 1.0))
        g.add_node(make_node("n1", 2.0))   # should be ignored
        assert g.n_nodes == 1
        assert g.nodes["n1"].entropy_score == 1.0   # original preserved

    def test_add_edge(self):
        g = BeliefGraph()
        g.add_node(make_node("p", 1.0))
        g.add_node(make_node("c", 0.5))
        g.add_edge(make_edge("p", "c"))
        assert g.n_edges == 1

    def test_add_edge_missing_parent_raises(self):
        g = BeliefGraph()
        g.add_node(make_node("c", 0.5))
        with pytest.raises(ValueError, match="Parent node"):
            g.add_edge(make_edge("missing", "c"))

    def test_add_edge_missing_child_raises(self):
        g = BeliefGraph()
        g.add_node(make_node("p", 1.0))
        with pytest.raises(ValueError, match="Child node"):
            g.add_edge(make_edge("p", "missing"))

    def test_frontier_sorted_entropy_desc(self):
        g = BeliefGraph()
        for nid, h in [("a", 0.3), ("b", 1.8), ("c", 0.9)]:
            g.add_node(make_node(nid, h))
        frontier = g.get_frontier()
        entropies = [n.entropy_score for n in frontier]
        assert entropies == sorted(entropies, reverse=True)
        assert frontier[0].id == "b"

    def test_frontier_excludes_resolved(self):
        g = BeliefGraph()
        g.add_node(make_node("a", 2.0))
        g.add_node(make_node("b", 1.0))
        g.mark_resolved("a")
        frontier_ids = {n.id for n in g.get_frontier()}
        assert "a" not in frontier_ids
        assert "b" in frontier_ids

    def test_frontier_excludes_rabbit_holes(self):
        g = BeliefGraph()
        g.add_node(make_node("a", 2.0))
        g.add_node(make_node("b", 1.0))
        g.nodes["a"].is_rabbit_hole = True
        frontier_ids = {n.id for n in g.get_frontier()}
        assert "a" not in frontier_ids

    def test_mark_resolved_logs_expansion(self):
        g = BeliefGraph()
        g.add_node(make_node("n1", 1.5))
        g.mark_resolved("n1")
        assert g.nodes["n1"].resolved is True
        assert len(g._expansion_log) == 1
        assert g._expansion_log[0]["entropy"] == 1.5

    def test_mark_resolved_missing_raises(self):
        g = BeliefGraph()
        with pytest.raises(KeyError):
            g.mark_resolved("ghost")

    def test_mean_entropy(self):
        g = BeliefGraph()
        g.add_node(make_node("a", 1.0))
        g.add_node(make_node("b", 3.0))
        assert math.isclose(g.mean_entropy, 2.0)

    def test_children_of(self):
        g = BeliefGraph()
        g.add_node(make_node("p", 1.0))
        g.add_node(make_node("c1", 0.5))
        g.add_node(make_node("c2", 0.4))
        g.add_edge(make_edge("p", "c1"))
        g.add_edge(make_edge("p", "c2"))
        children = {n.id for n in g.children_of("p")}
        assert children == {"c1", "c2"}

    def test_parents_of(self):
        g = BeliefGraph()
        g.add_node(make_node("p", 1.0))
        g.add_node(make_node("c", 0.5))
        g.add_edge(make_edge("p", "c"))
        parents = {n.id for n in g.parents_of("c")}
        assert parents == {"p"}

    def test_contradiction_edges(self):
        g = BeliefGraph()
        g.add_node(make_node("a", 1.0))
        g.add_node(make_node("b", 0.5))
        e = Edge(parent_id="a", child_id="b", relation="contradicts", contradiction_flag=True)
        g.add_edge(e)
        assert len(g.get_contradiction_edges()) == 1

    def test_export_import_json(self):
        g = BeliefGraph()
        g.add_node(make_node("n1", 1.5, depth=0))
        g.add_node(make_node("n2", 0.8, depth=1))
        g.add_edge(make_edge("n1", "n2"))
        g.mark_resolved("n1")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        g.export_json(path)
        assert path.exists()

        g2 = BeliefGraph.from_json(path)
        assert g2.n_nodes == 2
        assert g2.n_edges == 1
        assert g2.nodes["n1"].resolved is True
        assert len(g2._expansion_log) == 1
        path.unlink()

    def test_entropy_trend_positive(self):
        # Entropy rising → positive trend
        g = graph_with_nodes(0.5, 0.7, 1.0, 1.5, 2.0)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)
        trend = g.get_entropy_trend(window=5)
        assert trend > 0

    def test_entropy_trend_negative(self):
        # Entropy falling → negative trend
        g = graph_with_nodes(2.0, 1.5, 1.0, 0.7, 0.4)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)
        trend = g.get_entropy_trend(window=5)
        assert trend < 0

    def test_entropy_trend_too_few_expansions(self):
        g = BeliefGraph()
        g.add_node(make_node("only", 1.0))
        g.mark_resolved("only")
        assert g.get_entropy_trend() == 0.0

    def test_budget_exceeded_error(self):
        from apiro.graph.belief_graph import BudgetExceededError
        g = BeliefGraph(max_nodes=10)
        for i in range(10):
            g.add_node(make_node(f"n{i}", 0.5))
        with pytest.raises(BudgetExceededError):
            g.add_node(make_node("n10", 0.5))

    def test_depth_enforcement(self):
        g = BeliefGraph(max_depth=6)
        node_depth_6 = make_node("d6", 0.5, depth=6)
        node_depth_7 = make_node("d7", 0.5, depth=7)
        g.add_node(node_depth_6)
        g.add_node(node_depth_7)
        assert "d6" in g.nodes
        assert "d7" not in g.nodes


# ============================================================
# SaturationDetector tests
# ============================================================

class TestSaturationDetector:
    def _saturated_graph(self) -> BeliefGraph:
        """Graph with 5 low, stable, declining entropies."""
        g = graph_with_nodes(0.22, 0.20, 0.18, 0.17, 0.15)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)
        return g

    def _high_entropy_graph(self) -> BeliefGraph:
        g = graph_with_nodes(1.5, 1.8, 2.0, 2.2, 2.5)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)
        return g

    def test_saturated_when_all_conditions_met(self):
        g   = self._saturated_graph()
        det = SaturationDetector(theta=0.25, window=5, max_variance=0.04)
        assert det.is_saturated(g) is True

    def test_not_saturated_high_entropy(self):
        g   = self._high_entropy_graph()
        det = SaturationDetector(theta=0.25, window=5, max_variance=0.04)
        assert det.is_saturated(g) is False

    def test_not_saturated_too_few_expansions(self):
        g = BeliefGraph()
        g.add_node(make_node("x", 0.1))
        g.mark_resolved("x")
        det = SaturationDetector(theta=0.25, window=5)
        assert det.is_saturated(g) is False

    def test_get_status_keys(self):
        g      = self._saturated_graph()
        det    = SaturationDetector(theta=0.25)
        status = det.get_status(g)
        for key in ("saturated", "avg_entropy", "variance", "trend", "conditions"):
            assert key in status

    def test_rising_entropy_not_saturated(self):
        # Even with low average, rising trend should block saturation
        g = graph_with_nodes(0.10, 0.13, 0.17, 0.21, 0.24)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)
        det = SaturationDetector(theta=0.25, window=5)
        status = det.get_status(g)
        # trend is positive, so should not saturate
        assert status["conditions"]["non_rising"] is False
        assert det.is_saturated(g) is False

    def test_for_domain_genetics(self):
        det = SaturationDetector.for_domain("genetics")
        # Calibrated empirically for llama3.1:8b — ClinVar conflicting-classification
        # claims plateau at 0.66-0.69 nats, so theta=0.70 fires saturation correctly
        # while still requiring meaningful convergence below the initial entropy of 0.75.
        assert det.theta == 0.70

    def test_for_domain_unknown_uses_default(self):
        from apiro.config import DEFAULT_THETA
        det = SaturationDetector.for_domain("unknown_domain")
        assert det.theta == DEFAULT_THETA


# ============================================================
# RabbitHoleDetector tests
# ============================================================

class TestRabbitHoleDetector:
    def test_fires_on_rising_entropy_deep_node(self):
        # Rising entropy + depth >= min_depth → rabbit hole
        g = graph_with_nodes(2.0, 1.5, 1.0, 1.5, 2.0)   # V-shape, rising at end
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)

        deep_node = make_node("deep", 2.0, depth=4)
        det = RabbitHoleDetector(min_depth=3, reversal_window=4)
        assert det.check(g, deep_node) is True

    def test_does_not_fire_shallow_node(self):
        g = graph_with_nodes(2.0, 1.5, 1.0, 1.5, 2.0)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)

        shallow_node = make_node("shallow", 2.0, depth=1)
        det = RabbitHoleDetector(min_depth=3, reversal_window=4)
        assert det.check(g, shallow_node) is False

    def test_does_not_fire_declining_entropy(self):
        g = graph_with_nodes(2.5, 2.0, 1.5, 1.0, 0.5)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)

        deep_node = make_node("deep", 0.5, depth=4)
        det = RabbitHoleDetector(min_depth=3, reversal_window=4)
        assert det.check(g, deep_node) is False

    def test_flag_mutates_node(self):
        g = BeliefGraph()
        g.add_node(make_node("a", 1.0))
        g.add_node(make_node("b", 0.5))
        g.add_node(make_node("c", 1.2))
        for nid in ["a", "b", "c"]:
            g.mark_resolved(nid)

        node = make_node("flagged", 2.0, depth=5)
        det  = RabbitHoleDetector()
        det.flag_rabbit_hole(node, g)
        assert node.is_rabbit_hole is True
        assert "rabbit_hole_trend" in node.metadata

    def test_get_status_structure(self):
        g = graph_with_nodes(1.0, 1.5, 2.0)
        for nid in list(g.nodes.keys()):
            g.mark_resolved(nid)
        node = make_node("test", 2.0, depth=5)
        det  = RabbitHoleDetector()
        status = det.get_status(g, node)
        assert "is_rabbit_hole" in status
        assert "entropy_trend" in status
        assert "node_depth" in status
