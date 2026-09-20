"""Tests for the bounded provenance graph data model."""

import pytest

from apiro.graph.belief_graph import BeliefGraph
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


def test_graph_stores_nodes_edges_and_exports_json():
    graph = BeliefGraph()
    graph.add_node(node("fact"))
    graph.add_node(node("diagnosis", depth=1))
    graph.add_edge(Edge(parent_id="fact", child_id="diagnosis", relation="supports"))

    exported = graph.export_json()
    assert [item["id"] for item in exported["nodes"]] == ["fact", "diagnosis"]
    assert exported["edges"][0]["relation"] == "supports"
    assert exported["stats"] == {"n_nodes": 2, "n_edges": 1}


def test_graph_validates_edge_endpoints():
    graph = BeliefGraph()
    graph.add_node(node("fact"))
    with pytest.raises(ValueError):
        graph.add_edge(Edge(parent_id="missing", child_id="fact", relation="supports"))
