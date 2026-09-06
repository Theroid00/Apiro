"""
graph/belief_graph.py — BeliefGraph
=====================================
NetworkX-backed directed graph of clinical hypotheses.
The frontier (unresolved nodes sorted by entropy descending) is the
core data structure that drives the entropy-first traversal loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from apiro.graph.node import Node
from apiro.graph.edge import Edge
from apiro.config import GRAPH_MAX_DEPTH, GRAPH_MAX_NODES, RELEVANCE_FLOOR


#: Process-wide sentence embedder, loaded on first use by
#: BeliefGraph._get_embedder() and shared by every graph instance.
_SHARED_EMBEDDER = None


class BudgetExceededError(Exception):
    """Exception raised when the node budget is exceeded."""
    pass


class BeliefGraph:
    """
    Directed acyclic graph of clinical Nodes connected by typed Edges.

    The key invariant: `get_frontier()` always returns unresolved nodes
    sorted by `entropy_score` descending — the traversal loop always
    picks `frontier[0]` (highest uncertainty) to expand next.
    """

    def __init__(self, max_depth: int = GRAPH_MAX_DEPTH, max_nodes: int = GRAPH_MAX_NODES):
        self._graph: nx.DiGraph   = nx.DiGraph()
        self.nodes:  dict[str, Node] = {}   # id → Node
        self.edges:  list[Edge]      = []
        self._expansion_log: list[dict] = []  # ordered history of expanded nodes
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        # Semantic matching caches
        self._embedder = None
        self._embeddings: dict[str, np.ndarray] = {}  # node_id -> embedding
        # Case anchor: embedding of this patient's presentation, used to keep
        # the entropy-first frontier chasing uncertainty that is relevant to
        # THIS patient rather than uncertainty in the abstract.
        self._case_embedding: Optional[np.ndarray] = None
        self._relevance: dict[str, float] = {}       # node_id -> [0, 1]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node. Silently ignores duplicate IDs. Enforces budget and depth limits."""
        if node.id in self.nodes:
            return
        if len(self.nodes) >= self.max_nodes:
            raise BudgetExceededError(f"Attempted to exceed node budget of {self.max_nodes}")
        if node.depth > self.max_depth:
            # depth enforcement: node at depth 7 rejected when max_depth=6
            return
        self.nodes[node.id] = node
        self._graph.add_node(node.id, entropy=node.entropy_score, domain=node.domain)

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge. Both nodes must already exist."""
        if edge.parent_id not in self.nodes:
            raise ValueError(f"Parent node {edge.parent_id!r} not in graph.")
        if edge.child_id not in self.nodes:
            raise ValueError(f"Child node {edge.child_id!r} not in graph.")
        self.edges.append(edge)
        self._graph.add_edge(
            edge.parent_id, edge.child_id,
            relation=edge.relation,
            contradiction_flag=edge.contradiction_flag,
            confidence=edge.confidence,
        )

    def mark_resolved(self, node_id: str) -> None:
        """Mark a node as expanded. Records it in the expansion log."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id!r} not found.")
        node = self.nodes[node_id]
        node.resolved = True
        self._expansion_log.append({
            "node_id":      node_id,
            "entropy":      node.entropy_score,
            "domain":       node.domain,
            "depth":        node.depth,
            "is_rabbit_hole": node.is_rabbit_hole,
        })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def set_case_anchor(self, text: str) -> None:
        """
        Register the patient's presentation as the relevance anchor.

        Depth-0 findings use differential breadth. Generated hypotheses use
        binary entropy derived from verbalized patient-specific confidence.
        Either signal alone can favor a vague but irrelevant claim, so the
        anchor keeps traversal priority tied to the full presentation.

        With an anchor set, exploration priority becomes entropy scaled by how
        close the claim sits to the case.
        Without one the frontier falls back to raw entropy, so this is opt-in
        and never changes behaviour for callers that do not use it.
        """
        if not text or not text.strip():
            return
        try:
            embedder = self._get_embedder()
            self._case_embedding = embedder.encode(text.strip()[:4000], normalize_embeddings=True)
            self._relevance.clear()
        except Exception:
            # Embedding is best-effort: a missing model must not break traversal.
            self._case_embedding = None

    def relevance_of(self, node: Node) -> Optional[float]:
        """Cosine similarity of a node's claim to the case anchor, in [0, 1]."""
        if self._case_embedding is None:
            return None
        if node.id in self._relevance:
            return self._relevance[node.id]
        try:
            emb = self._embeddings.get(node.id)
            if emb is None:
                emb = self._get_embedder().encode(node.claim, normalize_embeddings=True)
                self._embeddings[node.id] = emb
            score = max(0.0, min(1.0, float(np.dot(self._case_embedding, emb))))
        except Exception:
            return None
        self._relevance[node.id] = score
        return score

    def get_frontier(self, depth_aware: bool = False) -> list[Node]:
        """
        Return all unresolved, non-rabbit-hole nodes sorted by score.

        Args:
            depth_aware: If False (default), sorts strictly by entropy_score descending
                         — the standard baseline contract expected by unit tests.
                         If True, uses depth-aware scoring for the entropy-first traversal:
                           - Depth 0 (seeds): score = 2.0 - entropy, guaranteeing all seeds
                             are expanded before any derived child node (starvation-proof).
                             Ties (all axioms share a fixed entropy) break on the
                             axiom's diagnostic weight, so the sharpest anchor
                             expands first.
                           - Depth >= 1 (derived): score = entropy, scaled by relevance to
                             the case anchor when one has been set (see set_case_anchor).
        """
        candidates = [n for n in self.nodes.values() if not n.resolved and not n.is_rabbit_hole]

        if depth_aware:
            def score(n: Node) -> float:
                h = n.entropy_score if n.entropy_score is not None else 0.5
                if n.depth == 0:
                    # Depth 0: always prioritize seed nodes to prevent starvation
                    base_score = 2.0 - h + 0.1 * float(n.metadata.get("axiom_weight", 0.0))
                else:
                    base_score = h
                    rel = self.relevance_of(n)
                    if rel is not None:
                        base_score *= (RELEVANCE_FLOOR + (1.0 - RELEVANCE_FLOOR) * rel)
                return base_score - getattr(n, "contradiction_penalty", 0.0)
        else:
            def score(n: Node) -> float:
                h = n.entropy_score if n.entropy_score is not None else 0.0
                return h - getattr(n, "contradiction_penalty", 0.0)

        return sorted(candidates, key=score, reverse=True)

    def get_entropy_trend(self, window: int = 5, min_depth: int | None = None) -> float:
        """
        Linear trend coefficient of entropy over the last `window` expansions.
        Positive = entropy rising (rabbit hole risk).
        Negative = entropy declining (converging toward saturation).
        Returns 0.0 if fewer than 2 expansions have occurred.

        Args:
            min_depth: when given, only expansions of nodes at this depth or
                       deeper are considered. Use min_depth=1 to measure the
                       *exploration* trend and ignore deterministic depth-0
                       seed anchors (whose entropy is a fixed constant and
                       therefore carries no convergence signal).
        """
        entropies = self.get_recent_entropies(window, min_depth=min_depth)
        if len(entropies) < 2:
            return 0.0
        x = np.arange(len(entropies), dtype=float)
        slope = float(np.polyfit(x, entropies, 1)[0])
        return round(slope, 9)

    def get_recent_entropies(self, window: int = 5, min_depth: int | None = None) -> list[float]:
        """
        Return entropy values from the last `window` expanded nodes.

        Args:
            min_depth: when given, expansions of nodes shallower than this are
                       filtered out *before* the window is applied.
        """
        log = self._expansion_log
        if min_depth is not None:
            log = [e for e in log if (e.get("depth") or 0) >= min_depth]
        return [e["entropy"] for e in log[-window:]]

    def count_expansions(self, min_depth: int | None = None) -> int:
        """Number of nodes expanded so far, optionally filtered by minimum depth."""
        if min_depth is None:
            return len(self._expansion_log)
        return sum(1 for e in self._expansion_log if (e.get("depth") or 0) >= min_depth)

    def get_contradiction_edges(self) -> list[Edge]:
        """Return all edges that were flagged as contradictions."""
        return [e for e in self.edges if e.contradiction_flag]

    def children_of(self, node_id: str) -> list[Node]:
        """Return direct children of a node."""
        return [self.nodes[c] for c in self._graph.successors(node_id) if c in self.nodes]

    def parents_of(self, node_id: str) -> list[Node]:
        """Return direct parents of a node."""
        return [self.nodes[p] for p in self._graph.predecessors(node_id) if p in self.nodes]

    def node_count(self) -> int:
        """Return total number of nodes (alias for n_nodes, used by traversal)."""
        return len(self.nodes)

    def _get_embedder(self):
        """Lazily load the process-wide sentence embedder.

        One model instance is shared by every BeliefGraph: loading
        all-mpnet-base-v2 costs seconds and hundreds of megabytes, and a
        benchmark constructs one graph per case.

        (This previously declared `global _SHARED_EMBEDDER`, never assigned to
        that name, and read and wrote the module dict by hand instead — the
        `global` statement did nothing and the name existed only after the
        first call.)
        """
        global _SHARED_EMBEDDER
        if self._embedder is None:
            if _SHARED_EMBEDDER is None:
                from sentence_transformers import SentenceTransformer
                from apiro.config import EMBED_MODEL
                _SHARED_EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
            self._embedder = _SHARED_EMBEDDER
        return self._embedder

    def ancestors_of(self, node_id: str) -> set[str]:
        """All transitive parents of a node, following Node.parent_id links."""
        chain: set[str] = set()
        cur = self.nodes.get(node_id)
        while cur is not None:
            parent_id = getattr(cur, "parent_id", None)
            if not parent_id or parent_id in chain:
                break
            chain.add(parent_id)
            cur = self.nodes.get(parent_id)
        return chain

    def find_semantic_match(
        self,
        claim: str,
        threshold: float = 0.92,
        exclude_ids: set[str] | None = None,
    ) -> Optional[Node]:
        """
        Find an existing node in the graph with a semantically equivalent claim.
        Returns the Node if one exists above the similarity threshold, else None.

        Args:
            exclude_ids: node IDs that must not be considered a match. Callers
                pass the expanding node and its ancestors: merging a child into
                its own parent creates a self-loop and silently discards the
                expansion, and merging a generated hypothesis into a depth-0
                axiom destroys the hypothesis outright.
        """
        if not self.nodes:
            return None

        # Embedding is best-effort (same policy as set_case_anchor() and
        # relevance_of()): semantic merging is a quality optimization, not a
        # hard requirement, so an unavailable/failing embedder must not take
        # down the traversal that's calling this on every generated hypothesis.
        try:
            embedder = self._get_embedder()
            new_emb = embedder.encode(claim, normalize_embeddings=True)

            # Ensure all existing nodes are embedded
            unembedded = [n for n in self.nodes.values() if n.id not in self._embeddings]
            if unembedded:
                texts = [n.claim for n in unembedded]
                embs = embedder.encode(texts, normalize_embeddings=True)
                for n, emb in zip(unembedded, embs):
                    self._embeddings[n.id] = emb
        except Exception:
            return None

        excluded = exclude_ids or set()

        # Find highest cosine similarity
        best_match = None
        best_score = -1.0

        for n_id, emb in self._embeddings.items():
            if n_id in excluded or n_id not in self.nodes:
                continue
            score = np.dot(new_emb, emb)
            if score > best_score:
                best_score = score
                best_match = self.nodes[n_id]

        if best_match and best_score >= threshold:
            return best_match

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_resolved(self) -> int:
        return sum(1 for n in self.nodes.values() if n.resolved)

    @property
    def n_rabbit_holes(self) -> int:
        return sum(1 for n in self.nodes.values() if n.is_rabbit_hole)

    @property
    def mean_entropy(self) -> Optional[float]:
        entropies = [n.entropy_score for n in self.nodes.values()]
        return float(np.mean(entropies)) if entropies else None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.DiGraph:
        """Return the underlying NetworkX DiGraph (read-only view)."""
        return self._graph

    def export_json(self, path: Path | None = None) -> dict:
        """
        Serialise the graph to a dict (always returned) and optionally write
        it to a JSON file when `path` is given.

        Format:
          {
            "nodes": [{id, claim, domain, entropy_score, resolved, depth, ...}],
            "edges": [{parent_id, child_id, relation, contradiction_flag, confidence}],
            "expansion_log": [...],
            "stats": {n_nodes, n_edges, n_resolved, n_rabbit_holes, mean_entropy}
          }
        """
        data = {
            "nodes": [
                {
                    "id":            n.id,
                    "claim":         n.claim,
                    "domain":        n.domain,
                    "entropy_score": n.entropy_score,
                    "resolved":      n.resolved,
                    "is_rabbit_hole": n.is_rabbit_hole,
                    "contradiction_penalty": getattr(n, "contradiction_penalty", 0.0),
                    "depth":         n.depth,
                    "parent_id":     n.parent_id,
                    "sources":       n.sources,
                    "metadata":      n.metadata,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "parent_id":         e.parent_id,
                    "child_id":          e.child_id,
                    "relation":          e.relation,
                    "contradiction_flag": e.contradiction_flag,
                    "confidence":        e.confidence,
                }
                for e in self.edges
            ],
            "expansion_log": self._expansion_log,
            "stats": {
                "n_nodes":        self.n_nodes,
                "n_edges":        self.n_edges,
                "n_resolved":     self.n_resolved,
                "n_rabbit_holes": self.n_rabbit_holes,
                "mean_entropy":   self.mean_entropy,
            },
        }
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
        return data

    @classmethod
    def from_json(cls, path: Path) -> "BeliefGraph":
        """Load a previously exported graph back into memory."""
        from apiro.graph.node import Node
        from apiro.graph.edge import Edge
        with open(path) as f:
            data = json.load(f)
        g = cls()
        for n in data["nodes"]:
            g.add_node(Node(**n))
        for e in data["edges"]:
            g.add_edge(Edge(**e))
        g._expansion_log = data.get("expansion_log", [])
        return g

    def __repr__(self) -> str:
        return (
            f"BeliefGraph(nodes={self.n_nodes}, edges={self.n_edges}, "
            f"resolved={self.n_resolved}, frontier={len(self.get_frontier())}, "
            f"rabbit_holes={self.n_rabbit_holes})"
        )
