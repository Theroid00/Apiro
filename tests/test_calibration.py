"""
tests/test_calibration.py
=========================
Unit tests for apiro/eval/calibration.py — Pillar 3 (ECE, Brier,
risk-coverage / AURC, selective abstention).

This module drives the headline safety numbers in the README and had no test
coverage at all. It also shipped two top-level definitions of
``_aurc_from_ranking``: the first was truncated mid-body (it computed a sort
order and then fell off the end, returning ``None``) and was silently shadowed
by a complete second definition further down. The dead stub is gone; the
hand-computed AURC cases below are the guard against anything like it
returning.

No Ollama, no ChromaDB, no model download.
"""

import math

import numpy as np
import pytest

from apiro.eval.calibration import (
    CalibratedDecision,
    _aurc_from_ranking,
    _optimal_aurc,
    compute_ece,
    compute_risk_coverage,
)


# --------------------------------------------------------------------------- #
# ECE / MCE / Brier
# --------------------------------------------------------------------------- #
class TestComputeECE:
    def test_perfect_confidence_on_correct_predictions_is_perfectly_calibrated(self):
        result = compute_ece([1.0] * 8, [True] * 8)
        assert result["ece"] == pytest.approx(0.0)
        assert result["mce"] == pytest.approx(0.0)
        assert result["brier_score"] == pytest.approx(0.0)

    def test_zero_confidence_on_wrong_predictions_is_perfectly_calibrated(self):
        result = compute_ece([0.0] * 8, [False] * 8)
        assert result["ece"] == pytest.approx(0.0)
        assert result["brier_score"] == pytest.approx(0.0)

    def test_maximally_overconfident_is_maximally_miscalibrated(self):
        # Certain, and wrong every time: the worst achievable calibration.
        result = compute_ece([1.0] * 8, [False] * 8)
        assert result["ece"] == pytest.approx(1.0)
        assert result["mce"] == pytest.approx(1.0)
        assert result["brier_score"] == pytest.approx(1.0)

    def test_half_confidence_half_correct_is_calibrated_but_unsharp(self):
        # One bin: mean confidence 0.5, mean accuracy 0.5 -> gap 0 -> ECE 0.
        # Brier still penalises the lack of sharpness: mean((0.5 - y)^2) = 0.25.
        result = compute_ece([0.5] * 4, [True, True, False, False])
        assert result["ece"] == pytest.approx(0.0)
        assert result["brier_score"] == pytest.approx(0.25)

    def test_mce_is_at_least_ece(self):
        conf = [0.1, 0.3, 0.55, 0.75, 0.95, 0.2, 0.85, 0.45]
        corr = [False, True, False, True, True, False, False, True]
        result = compute_ece(conf, corr)
        assert result["mce"] >= result["ece"] - 1e-12

    def test_bin_counts_sum_to_sample_count(self):
        conf = [0.05, 0.15, 0.5, 0.95, 1.0, 0.0]
        corr = [False, True, True, True, False, False]
        result = compute_ece(conf, corr)
        assert sum(result["bin_counts"]) == result["n_samples"] == 6

    def test_zero_confidence_lands_in_the_first_bin(self):
        # np.digitize with right=True would otherwise push an exact 0.0 out of
        # range; the clamp in _assign_bins keeps it in bin 0.
        result = compute_ece([0.0, 0.0], [False, False])
        assert result["bin_counts"][0] == 2

    def test_quantile_strategy_is_accepted(self):
        conf = [0.1, 0.2, 0.8, 0.9]
        corr = [False, False, True, True]
        result = compute_ece(conf, corr, n_bins=2, strategy="quantile")
        assert result["strategy"] == "quantile"
        assert sum(result["bin_counts"]) == 4

    def test_empty_input_returns_zeros_not_nan(self):
        result = compute_ece([], [])
        assert result["n_samples"] == 0
        assert result["ece"] == 0.0
        assert result["brier_score"] == 0.0

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            compute_ece([0.5, 0.5], [True])

    def test_out_of_range_confidence_is_rejected(self):
        with pytest.raises(ValueError):
            compute_ece([1.5], [True])
        with pytest.raises(ValueError):
            compute_ece([-0.1], [True])

    def test_non_finite_confidence_is_rejected(self):
        with pytest.raises(ValueError):
            compute_ece([float("nan")], [True])

    def test_invalid_bin_count_is_rejected(self):
        with pytest.raises(ValueError):
            compute_ece([0.5], [True], n_bins=0)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError):
            compute_ece([0.5], [True], strategy="logarithmic")

    def test_reproduces_the_published_apiro_underconfidence(self):
        # data/calibration_eval_results.json reports mean_confidence 0.228
        # against forced_accuracy 0.68 for the Apiro arm: severe *under*
        # confidence, and an ECE of 0.45. Confidence well below accuracy must
        # produce a large ECE, not a small one.
        result = compute_ece([0.23] * 25, [True] * 17 + [False] * 8)
        assert result["ece"] == pytest.approx(0.45, abs=0.02)


# --------------------------------------------------------------------------- #
# Risk-coverage / AURC
# --------------------------------------------------------------------------- #
class TestComputeRiskCoverage:
    #: Confidence ranks the two correct answers above the two wrong ones.
    PERFECT_CONF = [0.9, 0.8, 0.2, 0.1]
    PERFECT_CORR = [True, True, False, False]

    def test_perfect_ranking_matches_the_oracle(self):
        # Accepting in descending-confidence order gives errors [0,0,1,1],
        # coverage [.25,.5,.75,1], selective risk [0,0,1/3,1/2].
        # Trapezoidal area over coverage:
        #   .25*(0+0)/2 + .25*(0+0)/2 + .25*(0+1/3)/2 + .25*(1/3+1/2)/2
        #   = 0 + 0 + 0.0416667 + 0.1041667 = 0.1458333
        result = compute_risk_coverage(self.PERFECT_CONF, self.PERFECT_CORR)
        assert result["aurc"] == pytest.approx(0.1458333, abs=1e-6)
        assert result["optimal_aurc"] == pytest.approx(0.1458333, abs=1e-6)
        assert result["excess_risk"] == pytest.approx(0.0, abs=1e-9)

    def test_inverted_ranking_is_far_worse_than_the_oracle(self):
        # Confidence ranks both wrong answers first: errors [1,1,0,0],
        # selective risk [1, 1, 2/3, 1/2].
        #   .25*(0+1)/2 + .25*(1+1)/2 + .25*(1+2/3)/2 + .25*(2/3+1/2)/2
        #   = 0.125 + 0.25 + 0.2083333 + 0.1458333 = 0.7291667
        result = compute_risk_coverage([0.1, 0.2, 0.8, 0.9], self.PERFECT_CORR)
        assert result["aurc"] == pytest.approx(0.7291667, abs=1e-6)
        assert result["excess_risk"] == pytest.approx(0.5833333, abs=1e-6)

    def test_aurc_is_a_real_number_not_none(self):
        # Direct regression guard on the removed truncated `_aurc_from_ranking`
        # stub, which returned None. If it ever comes back and wins the name,
        # `float(aurc)` in compute_risk_coverage raises and this fails loudly.
        result = compute_risk_coverage(self.PERFECT_CONF, self.PERFECT_CORR)
        assert isinstance(result["aurc"], float)
        assert math.isfinite(result["aurc"])

    def test_all_correct_carries_no_risk(self):
        result = compute_risk_coverage([0.9, 0.5, 0.1], [True, True, True])
        assert result["aurc"] == pytest.approx(0.0)
        assert result["base_error_rate"] == pytest.approx(0.0)
        assert result["excess_risk"] == pytest.approx(0.0)

    def test_base_error_rate_is_the_full_coverage_error(self):
        result = compute_risk_coverage(self.PERFECT_CONF, self.PERFECT_CORR)
        assert result["base_error_rate"] == pytest.approx(0.5)

    def test_threshold_zero_gives_full_coverage(self):
        result = compute_risk_coverage(self.PERFECT_CONF, self.PERFECT_CORR)
        idx = result["thresholds"].index(0.0)
        assert result["coverages"][idx] == pytest.approx(1.0)
        assert result["selective_risks"][idx] == pytest.approx(0.5)

    def test_highest_threshold_keeps_only_the_most_confident_case(self):
        result = compute_risk_coverage(self.PERFECT_CONF, self.PERFECT_CORR)
        assert result["coverages"][-1] == pytest.approx(0.25)
        assert result["selective_risks"][-1] == pytest.approx(0.0)

    def test_thresholds_are_reported_in_ascending_order(self):
        result = compute_risk_coverage(self.PERFECT_CONF, self.PERFECT_CORR)
        assert result["thresholds"] == sorted(result["thresholds"])

    def test_coverage_is_monotonically_non_increasing_in_threshold(self):
        conf = [0.1, 0.35, 0.5, 0.65, 0.8, 0.95]
        corr = [False, False, True, True, True, True]
        result = compute_risk_coverage(conf, corr)
        coverages = result["coverages"]
        assert all(a >= b - 1e-12 for a, b in zip(coverages, coverages[1:]))

    def test_explicit_thresholds_are_honoured(self):
        result = compute_risk_coverage(
            self.PERFECT_CONF, self.PERFECT_CORR, thresholds=[0.0, 0.5, 1.0]
        )
        assert result["thresholds"] == [0.0, 0.5, 1.0]
        assert result["coverages"] == pytest.approx([1.0, 0.5, 0.0])

    def test_zero_coverage_reports_zero_risk_not_nan(self):
        result = compute_risk_coverage(
            self.PERFECT_CONF, self.PERFECT_CORR, thresholds=[1.0]
        )
        assert result["coverages"][0] == 0.0
        assert result["selective_risks"][0] == 0.0

    def test_excess_risk_is_never_negative(self):
        result = compute_risk_coverage([0.5, 0.5, 0.5, 0.5], [True, False, True, False])
        assert result["excess_risk"] >= 0.0

    def test_empty_input_returns_zeros(self):
        result = compute_risk_coverage([], [])
        assert result["n_samples"] == 0
        assert result["aurc"] == 0.0

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            compute_risk_coverage([0.5, 0.5], [True])

    def test_reproduces_the_published_apiro_rag_aurc_gap(self):
        # data/calibration_eval_results.json: Apiro AURC 0.119 vs RAG 0.536.
        # A confidence signal that ranks correct-before-incorrect must produce a
        # markedly lower AURC than one that does not, at equal accuracy.
        corr = [True] * 10 + [False] * 15
        good = [0.9] * 10 + [0.1] * 15          # ranks correct first
        bad = [0.1] * 10 + [0.9] * 15           # ranks correct last
        assert (compute_risk_coverage(good, corr)["aurc"]
                < compute_risk_coverage(bad, corr)["aurc"])


class TestAurcHelpers:
    def test_optimal_aurc_matches_a_perfectly_ranked_run(self):
        errors = np.array([0.0, 0.0, 1.0, 1.0])
        conf = np.array([0.9, 0.8, 0.2, 0.1])
        assert _aurc_from_ranking(conf, errors) == pytest.approx(_optimal_aurc(errors))

    def test_ascending_flag_reverses_the_acceptance_order(self):
        errors = np.array([0.0, 0.0, 1.0, 1.0])
        conf = np.array([0.9, 0.8, 0.2, 0.1])
        descending = _aurc_from_ranking(conf, errors, ascending_by_risk=False)
        ascending = _aurc_from_ranking(conf, errors, ascending_by_risk=True)
        assert ascending > descending

    def test_empty_arrays_are_zero(self):
        empty = np.array([])
        assert _aurc_from_ranking(empty, empty) == 0.0
        assert _optimal_aurc(empty) == 0.0


# --------------------------------------------------------------------------- #
# Selective abstention
# --------------------------------------------------------------------------- #
class TestCalibratedDecision:
    DIFFERENTIAL = [{"label": "Pheochromocytoma", "score": 0.8},
                    {"label": "Panic disorder", "score": 0.1}]

    def test_answers_above_threshold(self):
        decision = CalibratedDecision.decide(self.DIFFERENTIAL, confidence=0.9, threshold=0.65)
        assert decision.abstained is False
        assert decision.abstention_reason is None
        assert decision.top == self.DIFFERENTIAL[0]

    def test_threshold_is_inclusive(self):
        # tau = 0.65 is the README's operating point; a case sitting exactly on
        # it must be answered, not abstained.
        decision = CalibratedDecision.decide(self.DIFFERENTIAL, confidence=0.65, threshold=0.65)
        assert decision.abstained is False

    def test_abstains_below_threshold(self):
        decision = CalibratedDecision.decide(self.DIFFERENTIAL, confidence=0.4, threshold=0.65)
        assert decision.abstained is True
        assert "confidence" in decision.abstention_reason
        assert decision.top is None

    def test_unresolved_conflicts_force_abstention_even_when_confident(self):
        decision = CalibratedDecision.decide(
            self.DIFFERENTIAL, confidence=0.99, threshold=0.65,
            unresolved_conflicts=["anchor vs anchor: potassium"],
        )
        assert decision.abstained is True
        assert "unresolved conflict" in decision.abstention_reason

    def test_both_reasons_are_reported_together(self):
        decision = CalibratedDecision.decide(
            self.DIFFERENTIAL, confidence=0.1, threshold=0.65,
            unresolved_conflicts=["a", "b"],
        )
        assert "confidence" in decision.abstention_reason
        assert "2 unresolved conflict" in decision.abstention_reason

    def test_empty_differential_has_no_top(self):
        decision = CalibratedDecision.decide([], confidence=0.9, threshold=0.65)
        assert decision.abstained is False
        assert decision.top is None

    def test_out_of_range_confidence_is_rejected(self):
        with pytest.raises(ValueError):
            CalibratedDecision(confidence=1.5)
        with pytest.raises(ValueError):
            CalibratedDecision(confidence=-0.1)

    def test_out_of_range_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            CalibratedDecision.decide(self.DIFFERENTIAL, confidence=0.5, threshold=1.5)

    def test_abstention_without_a_reason_is_rejected(self):
        # An abstention the caller cannot explain is not an abstention a
        # clinician can act on.
        with pytest.raises(ValueError):
            CalibratedDecision(confidence=0.5, abstained=True)

    def test_reason_without_an_abstention_is_rejected(self):
        with pytest.raises(ValueError):
            CalibratedDecision(confidence=0.9, abstained=False, abstention_reason="why")

    def test_to_dict_is_json_safe(self):
        import json

        decision = CalibratedDecision.decide(self.DIFFERENTIAL, confidence=0.4, threshold=0.65)
        payload = decision.to_dict()
        json.dumps(payload)                     # must not raise
        assert payload["abstained"] is True
        assert set(payload) == {
            "differential", "confidence", "abstained",
            "abstention_reason", "unresolved_conflicts",
        }

    def test_differential_is_copied_not_aliased(self):
        source = [{"label": "X"}]
        decision = CalibratedDecision.decide(source, confidence=0.9, threshold=0.5)
        source.append({"label": "Y"})
        assert len(decision.differential) == 1
