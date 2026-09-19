"""Tests for the bounded provenance graph data model."""

import pytest

from apiro.graph.belief_graph import BeliefGraph, BudgetExceededError
from apiro.graph.edge import Edge
from apiro.graph.node import Node


def node(identifier="n", depth=0):
    return Node(
        id=identifier, claim=f"Claim for {identifier}",
        domain="pathophysiology", entropy_score=0.5, depth=depth,
    )


def test_node_and_edge_validate_values():
    assert node().id == "n"
    assert Edge(parent_id="a", child_id="b", relation="supports").relation == "supports"
    with pytest.raises(ValueError):
        Node(id="bad", claim="x", domain="lab", entropy_score=-0.1)
    with pytest.raises(ValueError):
        Edge(parent_id="a", child_id="b", relation="invalid")


def test_graph_adds_nodes_edges_and_queries_relationships():
    graph = BeliefGraph(max_depth=1, max_nodes=3)
    graph.add_node(node("fact"))
    graph.add_node(node("diagnosis", depth=1))
    graph.add_edge(Edge(parent_id="fact", child_id="diagnosis", relation="supports"))

    assert graph.get_frontier()
    assert graph.children_of("fact")[0].id == "diagnosis"
    assert graph.parents_of("diagnosis")[0].id == "fact"


def test_graph_enforces_depth_and_node_budgets():
    graph = BeliefGraph(max_depth=1, max_nodes=2)
    graph.add_node(node("root"))
    graph.add_node(node("too_deep", depth=2))
    assert "too_deep" not in graph.nodes
    graph.add_node(node("valid", depth=1))
    with pytest.raises(BudgetExceededError):
        graph.add_node(node("overflow"))
