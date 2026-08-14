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
