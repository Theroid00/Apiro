"""
apiro.graph — belief graph data model and orchestration components.

Data model:
  BeliefGraph, Node, Edge

Orchestration:
  ContradictionDetector (import directly: from apiro.graph.contradiction import ContradictionDetector)

  Stubs for the bounded engines live with their tests.
"""

from apiro.graph.node import Node
from apiro.graph.edge import Edge
from apiro.graph.belief_graph import BeliefGraph

# NOTE: ContradictionDetector is NOT imported here because it pulls in
# torch + transformers at import time (heavy dependencies, ~330MB model download).
# Import it directly when needed:
#   from apiro.graph.contradiction import ContradictionDetector

__all__ = [
    # Data model
    "Node",
    "Edge",
    "BeliefGraph",
]
