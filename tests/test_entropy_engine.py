"""
tests/test_entropy_engine.py
============================
Unit tests for EntropyEngine — the "differential breadth" signal.

These tests run WITHOUT Ollama (network calls are mocked). Prior to the
rewrite documented in apiro/entropy/engine.py's module docstring, this
signal asked the LLM a yes/no verification question and measured
first-token entropy. That path was replaced with differential_breadth_entropy():
the LLM is asked to count how many distinct diagnoses could plausibly
explain a finding, and the count is mapped through a fixed table
(_COUNT_TO_ENTROPY) to an uncertainty score. temperature_corrected_entropy()
and epistemic_certainty_entropy() are both still-supported entry points that
delegate to it — this file tests the actual delegation and the count→entropy
mapping, not the old logprob-based signal these tests previously covered.
"""

from unittest.mock import patch
import pytest

from apiro.entropy.engine import EntropyEngine, _COUNT_TO_ENTROPY, _DEFAULT_HIGH


def make_engine() -> EntropyEngine:
    """Return an EntropyEngine with no real Ollama connection needed."""
    return EntropyEngine(model="llama3.1:8b", ollama_url="http://localhost:11434")


# ── _build_verification_prompt / _extract_claim_from_prompt ───────────────────
# Kept for interface compatibility with old callers (e.g. NodeExpander._batch_entropy
# calls _build_verification_prompt() then temperature_corrected_entropy()), even
# though the new signal doesn't use a yes/no verification framing internally —
# it re-extracts the claim and counts diagnoses instead.

class TestVerificationPromptRoundTrip:
    def test_build_then_extract_recovers_the_claim(self):
        claim = "Troponin elevation indicates myocardial injury."
        prompt = EntropyEngine._build_verification_prompt(claim)
        assert EntropyEngine._extract_claim_from_prompt(prompt) == claim

    def test_build_prompt_contains_claim_and_yes_no_instruction(self):
        claim = "Aspirin is contraindicated in haemorrhagic stroke."
        prompt = EntropyEngine._build_verification_prompt(claim)
        assert claim in prompt
        assert "Yes or No" in prompt

    def test_context_chunks_are_accepted_but_do_not_affect_the_prompt(self):
        """
        context_chunks is accepted for interface compatibility with the old
        evidence-grounded verification prompt, but the current implementation
        does not use it — the differential-breadth signal doesn't need RAG
        context, it just counts plausible diagnoses for the claim itself.
        """
        claim = "Elevated troponin is seen in STEMI."
        without = EntropyEngine._build_verification_prompt(claim)
        with_chunks = EntropyEngine._build_verification_prompt(
            claim, context_chunks=["Troponin is a cardiac biomarker."]
        )
        assert without == with_chunks

    def test_extract_falls_back_to_whole_prompt_without_marker(self):
        """No 'Clinical claim:' marker → the whole string is treated as the claim."""
        assert EntropyEngine._extract_claim_from_prompt("just a bare claim") == "just a bare claim"


# ── differential_breadth_entropy — the live signal ─────────────────────────────

class TestDifferentialBreadthEntropy:
    @pytest.mark.parametrize("count,expected", sorted(_COUNT_TO_ENTROPY.items()))
    def test_count_maps_to_the_documented_entropy_table(self, count, expected):
        engine = make_engine()
        with patch.object(engine, "_query_differential_count", return_value=count):
            assert engine.differential_breadth_entropy("some claim") == expected

    def test_count_at_or_above_six_uses_default_high(self):
        engine = make_engine()
        with patch.object(engine, "_query_differential_count", return_value=6):
            assert engine.differential_breadth_entropy("vague claim") == _DEFAULT_HIGH

    def test_none_count_uses_default_high(self):
        """Total query failure (Ollama down, unparseable after retries) → max uncertainty."""
        engine = make_engine()
        with patch.object(engine, "_query_differential_count", return_value=None):
            assert engine.differential_breadth_entropy("some claim") == _DEFAULT_HIGH

    def test_empty_claim_uses_default_high_without_querying(self):
        engine = make_engine()
        with patch.object(engine, "_query_differential_count") as mock_query:
            assert engine.differential_breadth_entropy("") == _DEFAULT_HIGH
            mock_query.assert_not_called()

    def test_placeholder_claim_uses_default_high_without_querying(self):
        """Claims starting with '[' are synthetic placeholders (e.g. failed expansions)."""
        engine = make_engine()
        with patch.object(engine, "_query_differential_count") as mock_query:
            assert engine.differential_breadth_entropy("[Expansion failed]") == _DEFAULT_HIGH
            mock_query.assert_not_called()

    def test_result_is_cached_by_normalized_claim(self):
        engine = make_engine()
        with patch.object(engine, "_query_differential_count", return_value=2) as mock_query:
            first = engine.differential_breadth_entropy("Chest pain")
            second = engine.differential_breadth_entropy("  chest pain  ")  # same claim, different casing/whitespace
        assert first == second == _COUNT_TO_ENTROPY[2]
        mock_query.assert_called_once()


# ── Legacy entry points delegate correctly ─────────────────────────────────────

class TestLegacyEntryPointsDelegate:
    """
    temperature_corrected_entropy() and epistemic_certainty_entropy() are kept
    so existing callers (NodeExpander._batch_entropy, older test/eval scripts)
    don't need to change. Both must route to differential_breadth_entropy()
    with the correctly-extracted claim.
    """

    def test_temperature_corrected_entropy_extracts_claim_from_prompt(self):
        engine = make_engine()
        claim = "Beta-blockers reduce post-MI mortality."
        prompt = EntropyEngine._build_verification_prompt(claim)

        with patch.object(engine, "differential_breadth_entropy", return_value=0.42) as mock_dbe:
            result = engine.temperature_corrected_entropy(prompt)

        assert result == 0.42
        mock_dbe.assert_called_once_with(claim)

    def test_epistemic_certainty_entropy_ignores_context_chunks_and_delegates(self):
        engine = make_engine()
        claim = "Aquaporin-4 antibody positivity supports neuromyelitis optica."

        with patch.object(engine, "differential_breadth_entropy", return_value=0.10) as mock_dbe:
            result = engine.epistemic_certainty_entropy(claim, context_chunks=["some evidence"])

        assert result == 0.10
        mock_dbe.assert_called_once_with(claim)

    def test_first_token_entropy_delegates_to_temperature_corrected_entropy(self):
        engine = make_engine()
        with patch.object(engine, "temperature_corrected_entropy", return_value=0.55) as mock_tce:
            result = engine.first_token_entropy("some prompt")
        assert result == 0.55
        mock_tce.assert_called_once_with("some prompt")


# ── _parse_count ────────────────────────────────────────────────────────────────

class TestParseCount:
    def test_parses_a_bare_integer(self):
        assert EntropyEngine._parse_count("3") == 3

    def test_parses_integer_with_surrounding_text(self):
        assert EntropyEngine._parse_count("The count is 4 diagnoses.") == 4

    def test_caps_at_six(self):
        """Anything >= 6 maps to the same _DEFAULT_HIGH bucket, so parsing caps there."""
        assert EntropyEngine._parse_count("12") == 6

    def test_unparseable_text_returns_none(self):
        assert EntropyEngine._parse_count("I'm not sure, it depends.") is None


# ── Semantic rationale for the entropy bounds ──────────────────────────────────

class TestEntropyBoundsRationale:
    """
    Documents why the score range and saturation thresholds relate the way
    they do — not a behaviour test of engine internals.
    """

    def test_default_high_equals_max_binary_entropy(self):
        """
        _DEFAULT_HIGH (used for 'many/unknown diagnoses') is set to ln(2),
        the same maximum-uncertainty value a true binary Shannon entropy
        calculation would produce at a 50/50 split — so it composes
        correctly with SaturationDetector's theta thresholds, which were
        calibrated against that scale.
        """
        import math
        assert _DEFAULT_HIGH == pytest.approx(math.log(2), abs=1e-3)

    def test_saturation_theta_is_below_default_high(self):
        from apiro.config import DEFAULT_THETA
        assert DEFAULT_THETA < _DEFAULT_HIGH, (
            "Saturation theta must be well below the max-uncertainty score so "
            "the engine stops only when genuinely converging, not at the "
            "'many plausible diagnoses' ceiling."
        )
