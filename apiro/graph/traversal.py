"""
graph/traversal.py
------------------
Contains two traversal strategies:

1. ApiroTraversal (CLASSIC) — Phase 2 generative expansion:
   Entropy-first BFS that blindly expands frontier nodes via LLM.
   Kept for backwards compatibility and comparison.

2. HypothesisTestingTraversal (NEW) — hypothesis-testing inference:
   Phase 1: Extract structured PatientContext.
   Phase 2: Generate N candidate diagnoses via HypothesisOracle (1 LLM call).
   Phase 3: Score each hypothesis deterministically via EvidenceMatcher (0 LLM calls).
   Phase 4: Apply demographic constraints via BayesianScorer (0 LLM calls).
   Phase 5 (optional): Targeted graph enrichment for top-3 hypotheses only.

Select via the `--mode classic|hypothesis` flag in run.py.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from apiro.config import (
    CONTRADICTION_THRESHOLD_EF,
    CONTRADICTION_PENALTY,
    SATURATION_MIN_EXPLORATION,
    MAX_EXPLORATION_EXPANSIONS,
)
from apiro.graph.belief_graph import BudgetExceededError
from apiro.graph.critic import CriticEngine

logger = logging.getLogger(__name__)



@dataclass
class TraversalResult:
    """
    Summary returned by ApiroTraversal.run().

    Contains the final graph state and run statistics for export and testing.
    """
    graph:               object
    stop_reason:         str      # "saturation" | "max_depth" | "no_frontier"
    total_nodes:         int
    total_edges:         int
    rabbit_hole_count:   int
    contradiction_count: int
    duration_seconds:    float
    synthesis:           list[str]        # Final differential diagnosis
    saturation_status:   Optional[object] = None  # SaturationStatus if stop was saturation


@dataclass
class HypothesisTestingResult:
    """
    Summary returned by HypothesisTestingTraversal.run().

    Carries the ranked hypotheses, evidence scores, and timing stats.
    """
    synthesis:           list[str]              # Top 3 diagnoses
    ranked_hypotheses:   list                   # list[RankedHypothesis]
    duration_seconds:    float
    patient_context:     object                 # PatientContext
    stop_reason:         str = "scored"
    # Classic graph stats (populated if graph enrichment ran)
    total_nodes:         int = 0
    total_edges:         int = 0
    rabbit_hole_count:   int = 0
    contradiction_count: int = 0
    graph:               Optional[object] = None


class ApiroTraversal:
    """
    Orchestrates the entropy-first belief graph traversal.

    Args:
        expander:    NodeExpander instance
        saturation:  SaturationDetector instance
        rabbit_hole: RabbitHoleDetector instance
        contradiction: ContradictionDetector instance
        log_dir:     Directory to write traversal_log_{case_name}.jsonl
    """

    def __init__(
        self,
        expander,
        saturation,
        rabbit_hole,
        contradiction,
        log_dir: str = "data",
    ):
        self.expander    = expander
        self.saturation  = saturation
        self.rabbit_hole = rabbit_hole
        self.contradiction = contradiction
        self.log_dir     = log_dir
        self._traversal_log: list[dict] = []
        self.critic = CriticEngine(llm_client=expander.llm_client)

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log(self, event: dict) -> None:
        """Append a structured event to the in-memory traversal log."""
        self._traversal_log.append(event)
        if hasattr(self, "_on_event") and self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as e:
                logger.warning(f"Failed to trigger on_event callback: {e}")

    def _write_log(self, case_name: str) -> str:
        """Write the traversal log to a JSONL file and return the path."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"traversal_log_{case_name}.jsonl")
        with open(log_path, "w") as f:
            for event in self._traversal_log:
                f.write(json.dumps(event) + "\n")
        logger.info(f"[Traversal] Log written to {log_path}")
        return log_path

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        seed_nodes: list,
        graph,
        max_depth: int = 8,
        case_name: str = "run",
        vignette: str = None,
        on_event = None,
    ) -> TraversalResult:
        self._on_event = on_event
        """
        Run the entropy-first traversal loop.

        Args:
            seed_nodes: list of Node objects to start from
            graph:      empty BeliefGraph to populate
            max_depth:  hard stop — prevents infinite loops
            case_name:  used for log filename

        Returns:
            TraversalResult with the populated graph and run statistics
        """
        start_time = time.time()
        self._traversal_log = []

        # ── Anchor the frontier to this patient ───────────────────────────────
        # Entropy alone measures how many diagnoses a claim is compatible with,
        # which is a property of the sentence and not of the patient. Anchoring
        # makes the exploration frontier chase uncertainty about THIS case.
        anchor_text = vignette or "\n".join(s.claim for s in seed_nodes)
        if hasattr(graph, "set_case_anchor"):
            graph.set_case_anchor(anchor_text)

        # ── Seed the graph ────────────────────────────────────────────────────
        for seed in seed_nodes:
            graph.add_node(seed)
            self._log({
                "event":   "seed_added",
                "node_id": seed.id,
                "claim":   seed.claim,
                "entropy": seed.entropy_score,
                "domain":  seed.domain,
            })

        logger.info(f"[Traversal] Starting with {len(seed_nodes)} seed nodes")

        stop_reason = "no_frontier"
        iteration   = 0

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            iteration += 1

            # Number of *exploration* expansions so far (depth >= 1). Seed
            # expansions are excluded: they are deterministic anchors, not
            # evidence that the engine has learned anything.
            n_explored = graph.count_expansions(min_depth=1)

            # ── Stop condition -1: Exploration budget ─────────────────────────
            # Bounds wall-clock per case. Each exploration expansion costs one
            # generation call plus N_CHILD_HYPOTHESES entropy calls.
            if n_explored >= MAX_EXPLORATION_EXPANSIONS:
                stop_reason = "exploration_budget"
                logger.info(
                    f"[Traversal] Exploration budget reached "
                    f"({n_explored}/{MAX_EXPLORATION_EXPANSIONS} expansions)."
                )
                self._log({
                    "event":       "exploration_budget_reached",
                    "iteration":   iteration,
                    "expansions":  n_explored,
                })
                break

            # ── Stop condition 0: Global Critic Halting ───────────────────────
            if (
                n_explored >= SATURATION_MIN_EXPLORATION
                and iteration >= 10
                and iteration % 5 == 0
                and self.critic.evaluate_halting(graph, vignette=vignette)
            ):
                stop_reason = "critic_halt"
                logger.info(f"[Traversal] Global Critic approved halting at iteration {iteration}.")
                self._log({
                    "event":       "critic_halt_fired",
                    "iteration":   iteration,
                })
                break

            # ── Stop condition 1: Saturation ─────────────────────────────────
            # Warm-up guard: saturation is only meaningful once the engine has
            # actually explored. Without this, a run seeded with N >= window
            # deterministic axioms (all at entropy 0.01) "saturates" on its own
            # seed nodes and halts before generating a single hypothesis.
            if n_explored >= SATURATION_MIN_EXPLORATION and self.saturation.is_saturated(graph):
                status = self.saturation.get_status(graph)
                stop_reason = "saturation"
                logger.info(
                    f"[Traversal] SATURATED at iteration {iteration}. "
                    f"avg_entropy={status['avg_entropy']:.3f}, "
                    f"variance={status['variance']:.4f}, "
                    f"trend={status['trend']:.4f}"
                )
                self._log({
                    "event":       "saturation_fired",
                    "iteration":   iteration,
                    "avg_entropy": status["avg_entropy"],
                    "variance":    status["variance"],
                    "trend":       status["trend"],
                })
                break

            # ── Get frontier with depth-aware EF scoring ──────────────────────
            # depth_aware=True: depth-0 nodes sorted by certainty (1-H),
            # depth>=1 nodes sorted by uncertainty (H) to chase differentials.
            frontier = graph.get_frontier(depth_aware=True)

            if not frontier:
                stop_reason = "no_frontier"
                logger.info(f"[Traversal] Frontier empty at iteration {iteration}.")
                break

            # ── Stop condition 2: Max depth ───────────────────────────────────
            # Only nodes that are still allowed to expand are eligible. The old
            # code halted the entire run whenever the *top-scoring* node was at
            # max depth, discarding an otherwise healthy frontier of shallower
            # nodes — a single deep node could end the traversal outright.
            expandable = [n for n in frontier if n.depth < max_depth]
            if not expandable:
                stop_reason = "max_depth"
                logger.info(
                    f"[Traversal] Max depth {max_depth} reached — "
                    f"{len(frontier)} frontier node(s) all at or beyond the limit."
                )
                break

            # ── Pick best node ────────────────────────────────────────────────
            node = expandable[0]

            # ── Stop condition 3: Rabbit hole check ──────────────────────────
            # If the top node is a rabbit hole, flag it and restart the loop.
            # The next call to get_frontier() will naturally exclude it, so
            # frontier[1] becomes the new top and passes all safety checks.
            if self.rabbit_hole.check(graph, node):
                self.rabbit_hole.flag_rabbit_hole(node, graph)
                self._log({
                    "event":   "rabbit_hole_flagged",
                    "node_id": node.id,
                    "claim":   node.claim,
                    "depth":   node.depth,
                    "entropy": node.entropy_score,
                })
                logger.warning(f"[Traversal] Rabbit hole: '{node.claim[:60]}' — skipping.")
                continue  # restart loop; get_frontier will exclude flagged node

            # ── Expand the node ───────────────────────────────────────────────
            logger.info(
                f"[Traversal] iter={iteration} expanding: '{node.claim[:60]}' "
                f"(depth={node.depth}, entropy={node.entropy_score:.3f})"
            )

            self._log({
                "event":     "expanding",
                "iteration": iteration,
                "node_id":   node.id,
                "claim":     node.claim,
                "entropy":   node.entropy_score,
                "depth":     node.depth,
            })

            try:
                new_nodes = self.expander.expand(node, graph, vignette=vignette)
            except BudgetExceededError as e:
                logger.warning(f"[Traversal] Budget exceeded: {e}. Stopping traversal gracefully.")
                stop_reason = "budget_exceeded"
                break
            graph.mark_resolved(node.id)

            # ── Contradiction check: new nodes vs ALL existing nodes ───────────
            # Collect all pairs that pass the domain gate first, then run a
            # single batched NLI forward pass instead of one GPU call per pair.
            existing_nodes = list(graph.nodes.values())

            batch_pairs:   list[tuple[str, str]]      = []
            batch_meta:    list[tuple[object, object]] = []  # (new_node, existing)

            for new_node in new_nodes:
                self._log({
                    "event":     "node_expanded",
                    "node_id":   new_node.id,
                    "claim":     new_node.claim,
                    "entropy":   new_node.entropy_score,
                    "domain":    new_node.domain,
                    "depth":     new_node.depth,
                    "parent_id": new_node.parent_id,
                })
                for existing in existing_nodes:
                    if existing.id == new_node.id:
                        continue
                    if not self.contradiction.should_check(new_node.claim, existing.claim):
                        logger.debug(
                            f"[Traversal] Contradiction gate: skipping cross-abstraction "
                            f"pair '{new_node.claim[:30]}' vs '{existing.claim[:30]}'"
                        )
                        continue

                    batch_pairs.append((new_node.claim, existing.claim))
                    batch_meta.append((new_node, existing))

            # Run the batched NLI cross-encoder check
            if batch_pairs:
                results = self.contradiction.check_batch(batch_pairs)
                for (new_node, existing), result in zip(batch_meta, results):
                    # `>=`, not `>`: the deterministic fast filter emits a fixed
                    # confidence, and a strict comparison against an identical
                    # threshold silently discarded every keyword contradiction.
                    if result.label == "contradiction" and result.score >= CONTRADICTION_THRESHOLD_EF:
                        # Find the edge and flag it
                        for edge in graph.edges:
                            if edge.parent_id == new_node.parent_id and edge.child_id == new_node.id:
                                edge.contradiction_flag = True

                        self._log({
                            "event":              "contradiction_flagged",
                            "node_a":             new_node.claim[:80],
                            "node_b":             existing.claim[:80],
                            "score":              round(result.score, 3),
                            "negation_detected":  result.negation_detected,
                        })
                        logger.info(
                            f"[Traversal] Contradiction: '{new_node.claim[:40]}' "
                            f"vs '{existing.claim[:40]}' (score={result.score:.3f})"
                        )

                        # ── Contradiction-informed soft-pruning ───────────────
                        # Seed nodes (depth 0) are ground truth and must never be penalized.
                        # If a hypothesis contradicts ground truth, the hypothesis is penalized.
                        if new_node.depth == 0 and existing.depth == 0:
                            # Both are ground truth, do not penalize either
                            continue
                        elif new_node.depth == 0:
                            weaker = existing
                        elif existing.depth == 0:
                            weaker = new_node
                        else:
                            # Penalize the deeper (more speculative/derived) node
                            if new_node.depth != existing.depth:
                                weaker = new_node if new_node.depth > existing.depth else existing
                            else:
                                # Depths are equal: penalize the more uncertain (higher entropy) node
                                new_h      = new_node.entropy_score  or 0.0
                                existing_h = existing.entropy_score  or 0.0
                                weaker = new_node if new_h >= existing_h else existing

                        weaker.contradiction_penalty = CONTRADICTION_PENALTY
                        logger.info(
                            f"[Traversal] Soft-pruned weaker contradicting node: "
                            f"'{weaker.claim[:50]}' (entropy={weaker.entropy_score:.3f}, penalty={CONTRADICTION_PENALTY})"
                        )


        # ── Wrap up ───────────────────────────────────────────────────────────
        duration = round(time.time() - start_time, 2)

        # ── Synthesize differential ───────────────────────────────────────────
        # The vignette is passed through: the bare-LLM baseline reads the whole
        # case, so the synthesizer must too — otherwise Apiro argues its final
        # answer from graph fragments alone.
        try:
            synthesis = self.expander.synthesize_differential(graph, vignette=vignette)
        except TypeError:
            # Back-compat with expanders whose signature predates `vignette`.
            synthesis = self.expander.synthesize_differential(graph)

        self._log({
            "event":            "traversal_complete",
            "stop_reason":      stop_reason,
            "total_nodes":      graph.node_count(),
            "total_edges":      len(graph.edges),
            "synthesis":        synthesis,
            "duration_seconds": duration,
        })

        self._write_log(case_name)

        sat_status = self.saturation.get_status(graph) if stop_reason == "saturation" else None

        result = TraversalResult(
            graph=graph,
            stop_reason=stop_reason,
            total_nodes=graph.node_count(),
            total_edges=len(graph.edges),
            rabbit_hole_count=len(self.rabbit_hole.events),
            contradiction_count=len(graph.get_contradiction_edges()),
            duration_seconds=duration,
            synthesis=synthesis,
            saturation_status=sat_status,
        )

        logger.info(
            f"[Traversal] Done. stop={stop_reason}, nodes={result.total_nodes}, "
            f"edges={result.total_edges}, rabbit_holes={result.rabbit_hole_count}, "
            f"contradictions={result.contradiction_count}, time={duration}s"
        )

        return result
