"""
tests/test_traversal_regressions.py
-----------------------------------
Regression tests for the accuracy bugs fixed on feature/apiro-accuracy-fixes.

Every test here pins behaviour that was previously broken in a way that
silently crippled the engine (traversal halting early, patient facts missing
from synthesis, guardrails that could never fire). All tests run on stubs — no
Ollama, ChromaDB, or model download required.
"""

import pytest

from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.node import Node
from apiro.graph.saturation import SaturationDetector


def _seeded_graph(n_seeds: int = 8, entropy: float = 0.01) -> BeliefGraph:
    """Graph whose depth-0 axiom seeds have all been expanded."""
    g = BeliefGraph()
    for i in range(n_seeds):
        g.add_node(Node(id=f"s{i}", claim=f"seed {i}", domain="lab",
                        entropy_score=entropy, depth=0))
        g.mark_resolved(f"s{i}")
    return g


# ── Premature saturation on deterministic seed nodes ──────────────────────────

class TestSeedSaturation:
    def test_seeds_alone_do_not_saturate_in_exploration_mode(self):
        """
        Deterministic axiom seeds are injected at a fixed entropy (~0.01).
        Counting them makes the window read mean=0.01/var=0/trend=0, which
        fired saturation before a single hypothesis was ever generated.
        """
        g = _seeded_graph()
        det = SaturationDetector(theta=0.55, window=5, exploration_only=True)
        assert det.is_saturated(g) is False

    def test_legacy_mode_still_saturates_on_flat_window(self):
        """The original (depth-blind) contract is preserved when opted out."""
        g = _seeded_graph()
        det = SaturationDetector(theta=0.55, window=5, exploration_only=False)
        assert det.is_saturated(g) is True

    def test_saturates_once_exploration_actually_converges(self):
        g = _seeded_graph()
        for i, h in enumerate([0.30, 0.28, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21]):
            g.add_node(Node(id=f"h{i}", claim=f"hyp {i}", domain="lab",
                            entropy_score=h, depth=1))
            g.mark_resolved(f"h{i}")
        det = SaturationDetector(theta=0.55, window=5, exploration_only=True,
                                 min_expansions=8)
        assert det.is_saturated(g) is True

    def test_warmup_floor_blocks_early_saturation(self):
        """Even a flat exploration window cannot fire before the warm-up floor."""
        g = _seeded_graph()
        for i, h in enumerate([0.24, 0.24, 0.24, 0.24, 0.24]):
            g.add_node(Node(id=f"h{i}", claim=f"hyp {i}", domain="lab",
                            entropy_score=h, depth=1))
            g.mark_resolved(f"h{i}")
        det = SaturationDetector(theta=0.55, window=5, exploration_only=True,
                                 min_expansions=8)
        assert det.is_saturated(g) is False


# ── Depth-filtered expansion statistics ───────────────────────────────────────

class TestExpansionFilters:
    def test_count_expansions_filters_by_depth(self):
        g = _seeded_graph(n_seeds=3)
        g.add_node(Node(id="h0", claim="hyp", domain="lab",
                        entropy_score=0.5, depth=1))
        g.mark_resolved("h0")
        assert g.count_expansions() == 4
        assert g.count_expansions(min_depth=1) == 1

    def test_recent_entropies_filters_by_depth(self):
        g = _seeded_graph(n_seeds=3)
        g.add_node(Node(id="h0", claim="hyp", domain="lab",
                        entropy_score=0.5, depth=1))
        g.mark_resolved("h0")
        assert g.get_recent_entropies(5) == [0.01, 0.01, 0.01, 0.5]
        assert g.get_recent_entropies(5, min_depth=1) == [0.5]

    def test_trend_ignores_seed_plateau(self):
        g = _seeded_graph(n_seeds=4)
        for i, h in enumerate([0.6, 0.5, 0.4]):
            g.add_node(Node(id=f"h{i}", claim=f"hyp {i}", domain="lab",
                            entropy_score=h, depth=1))
            g.mark_resolved(f"h{i}")
        # Depth-blind view is dominated by the seed step-up; the exploration
        # view correctly reports a declining trend.
        assert g.get_entropy_trend(5, min_depth=1) < 0


# ── Final synthesis must see the patient, not just the graph ──────────────────

class _RecordingLLM:
    """Captures the prompt it is handed and returns a fixed differential."""

    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Acute pancreatitis\nCholedocholithiasis\nPeptic ulcer perforation"

    def generate(self, prompt: str) -> str:
        return self.chat(prompt)


def _graph_for_synthesis() -> BeliefGraph:
    g = BeliefGraph()
    g.add_node(Node(id="ax_0", claim="The patient has a lab result showing Lipase 1200 U/L.",
                    domain="lab", entropy_score=0.01, depth=0))
    g.add_node(Node(id="ax_1", claim="The patient denies the clinical finding of fever.",
                    domain="symptom", entropy_score=0.01, depth=0))
    # 20 vague, high-breadth exploration claims + 1 highly specific one.
    for i in range(20):
        g.add_node(Node(id=f"vague{i}", claim=f"Abdominal pain may reflect many causes {i}.",
                        domain="pathophysiology", entropy_score=0.693, depth=1,
                        parent_id="ax_0"))
    g.add_node(Node(id="specific", claim="Lipase above three times normal is diagnostic of pancreatitis.",
                    domain="lab", entropy_score=0.10, depth=2, parent_id="ax_0"))
    return g


def _expander(llm):
    from apiro.graph.expander import NodeExpander, StubEntropyEngine, StubChromaClient
    return NodeExpander(entropy_engine=StubEntropyEngine(),
                        chroma_client=StubChromaClient(), llm_client=llm)


class TestSynthesisContext:
    def test_vignette_reaches_the_synthesizer(self):
        llm = _RecordingLLM()
        _expander(llm).synthesize_differential(
            _graph_for_synthesis(), vignette="52M epigastric pain radiating to the back.")
        assert "52M epigastric pain radiating to the back." in llm.prompts[0]

    def test_confirmed_anchors_are_never_truncated_away(self):
        """
        Anchors carry entropy ~0.01. Under the old entropy-descending top_k cut
        they were dropped in favour of 15 vague claims, so the final prompt
        contained none of the patient's actual findings.
        """
        llm = _RecordingLLM()
        _expander(llm).synthesize_differential(_graph_for_synthesis(), top_k=5)
        prompt = llm.prompts[0]
        assert "Lipase 1200 U/L" in prompt

    def test_specific_claims_outrank_vague_ones(self):
        llm = _RecordingLLM()
        _expander(llm).synthesize_differential(_graph_for_synthesis(), top_k=3)
        assert "diagnostic of pancreatitis" in llm.prompts[0]

    def test_negated_findings_are_listed_as_ruled_out(self):
        llm = _RecordingLLM()
        _expander(llm).synthesize_differential(_graph_for_synthesis())
        assert "RULED-OUT" in llm.prompts[0]
        assert "denies the clinical finding of fever" in llm.prompts[0]

    def test_empty_graph_returns_empty_differential(self):
        llm = _RecordingLLM()
        assert _expander(llm).synthesize_differential(BeliefGraph()) == []
        assert llm.prompts == []


# ── Parser must not fabricate placeholder hypotheses ──────────────────────────

class TestHypothesisParsing:
    def test_no_placeholder_padding_by_default(self):
        llm = _RecordingLLM()
        parsed = _expander(llm)._parse_hypotheses("Only one real hypothesis here.")
        assert parsed == ["Only one real hypothesis here."]
        assert not any(p.startswith("[Expansion failed") for p in parsed)

    def test_numbering_and_bullets_stripped(self):
        llm = _RecordingLLM()
        parsed = _expander(llm)._parse_hypotheses("1. Alpha claim\n- Beta claim\n2) Gamma claim")
        assert parsed == ["Alpha claim", "Beta claim", "Gamma claim"]

    def test_preamble_dropped(self):
        llm = _RecordingLLM()
        parsed = _expander(llm)._parse_hypotheses("Hypotheses:\nAlpha claim\nBeta claim")
        assert parsed == ["Alpha claim", "Beta claim"]


# ── Deterministic contradiction guardrail ─────────────────────────────────────

class TestContradictionGuardrail:
    def _detector(self):
        from apiro.graph.contradiction import ContradictionDetector
        return ContradictionDetector()

    def test_fast_filter_score_clears_the_pruning_threshold(self):
        """
        The fast filter used to emit exactly CONTRADICTION_THRESHOLD_EF and
        every caller tested `score > threshold`, so deterministic keyword
        contradictions could never prune anything.
        """
        from apiro.config import CONTRADICTION_THRESHOLD_EF
        from apiro.graph.contradiction import FAST_FILTER_CONTRADICTION_SCORE
        assert FAST_FILTER_CONTRADICTION_SCORE > CONTRADICTION_THRESHOLD_EF

    def test_hyperkalemia_vs_hypokalemia_is_a_contradiction(self):
        r = self._detector()._fast_filter(
            "The patient has hyperkalemia with potassium 6.1.",
            "Hypokalemia is causing the patient's potassium wasting.",
            negation_detected=False,
        )
        assert r is not None and r.label == "contradiction"

    def test_substring_false_positive_does_not_fire(self):
        """'low' must not match inside 'blood flow'; 'high' not inside 'highly'."""
        r = self._detector()._fast_filter(
            "Coronary blood flow is reduced during the ischemic cascade.",
            "Elevated coronary perfusion is highly protective in ischemic cascade.",
            negation_detected=False,
        )
        assert r is None or r.label != "contradiction"

    def test_left_does_not_match_inside_cleft(self):
        det = self._detector()
        assert det._contains_term("cleft palate repair", "left") is False
        assert det._contains_term("left-sided weakness", "left") is True

    def test_check_batch_preserves_order_and_length(self):
        det = self._detector()
        pairs = [
            ("The patient has hyperkalemia with potassium 6.1.",
             "Hypokalemia explains the potassium level."),
            ("Ultrasound shows gallstones.", "The femur radiograph is unremarkable."),
        ]
        results = det.check_batch(pairs)
        assert len(results) == len(pairs)
        assert results[0].label == "contradiction"
        assert results[1].label == "neutral"


# ── Rabbit-hole detection is a property of a path, not of the whole graph ─────

class TestRabbitHoleLocality:
    def _detector(self):
        from apiro.graph.rabbit_hole import RabbitHoleDetector
        return RabbitHoleDetector(min_depth=3, reversal_window=4)

    def _chain(self, entropies: list[float]) -> tuple[BeliefGraph, Node]:
        g = BeliefGraph()
        parent = None
        node = None
        for i, h in enumerate(entropies):
            node = Node(id=f"n{i}", claim=f"claim {i}", domain="lab",
                        entropy_score=h, depth=i, parent_id=parent)
            g.add_node(node)
            parent = node.id
        return g, node

    def test_fires_on_a_genuinely_reversing_path(self):
        g, leaf = self._chain([0.65, 0.45, 0.30, 0.45, 0.60])
        assert self._detector().check(g, leaf) is True

    def test_healthy_converging_path_survives_a_noisy_graph(self):
        """
        A node whose own lineage is converging must not be flagged just because
        unrelated branches pushed the global entropy trend upward — the old
        global check killed exactly the highest-entropy frontier node this way.
        """
        g, leaf = self._chain([0.69, 0.55, 0.40, 0.25, 0.15])
        for i, h in enumerate([0.2, 0.4, 0.6, 0.69]):
            g.add_node(Node(id=f"other{i}", claim=f"other {i}", domain="lab",
                            entropy_score=h, depth=1, parent_id="n0"))
            g.mark_resolved(f"other{i}")
        assert self._detector().check(g, leaf) is False

    def test_shallow_nodes_are_never_flagged(self):
        g, leaf = self._chain([0.30, 0.45, 0.60])
        assert self._detector().check(g, leaf) is False

    def test_detached_node_falls_back_to_global_window(self):
        g = BeliefGraph()
        for i, h in enumerate([2.0, 1.5, 1.0, 1.5, 2.0]):
            g.add_node(Node(id=f"g{i}", claim=f"c{i}", domain="lab", entropy_score=h))
            g.mark_resolved(f"g{i}")
        detached = Node(id="deep", claim="deep", domain="lab", entropy_score=2.0, depth=4)
        assert self._detector().check(g, detached) is True
