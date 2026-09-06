"""
tests/test_eval_metrics.py
==========================
Unit tests for apiro/eval/metrics.py — the rank-aware differential-diagnosis
metrics and the paired significance machinery.

Pure arithmetic on stub inputs: no Ollama, no ChromaDB, no model download.
Every expected value below is hand-computed in the docstring or comment next
to the assertion, so a failure tells you which formula moved.
"""

import math

import pytest

from apiro.eval.metrics import (
    ArmScores,
    _normal_quantile,
    abstention_metrics,
    bias_trap_rate,
    compare_bias_traps,
    bootstrap_proportion_ci,
    compare_arms,
    compare_robustness,
    distractor_robustness,
    distractor_selection_rate,
    first_hit_rank,
    mcnemar_exact,
    mean_reciprocal_rank,
    paired_bootstrap_delta_ci,
    reciprocal_rank,
    score_arm,
    signal_health,
    top_k_accuracy,
    wilson_interval,
)


def exact_matcher(prediction: str, truth: str) -> bool:
    """Case-insensitive exact match — keeps the tests about the metrics."""
    return prediction.strip().lower() == truth.strip().lower()


# --------------------------------------------------------------------------- #
# Rank-aware hit metrics
# --------------------------------------------------------------------------- #
class TestFirstHitRank:
    def test_returns_one_based_rank_of_first_match(self):
        preds = ["Pneumonia", "Pulmonary embolism", "Asthma"]
        assert first_hit_rank(preds, "pulmonary embolism", exact_matcher) == 2

    def test_returns_none_when_nothing_matches(self):
        preds = ["Pneumonia", "Asthma"]
        assert first_hit_rank(preds, "sarcoidosis", exact_matcher) is None

    def test_first_match_wins_when_truth_appears_twice(self):
        preds = ["Sepsis", "Sepsis"]
        assert first_hit_rank(preds, "sepsis", exact_matcher) == 1

    def test_blank_entries_do_not_consume_a_rank_slot(self):
        # A blank line in the model's output must not push the real answer down
        # a rank; it is not a prediction.
        preds = ["", "   ", "Sepsis"]
        assert first_hit_rank(preds, "sepsis", exact_matcher) == 1

    def test_none_entries_are_tolerated(self):
        assert first_hit_rank([None, "Sepsis"], "sepsis", exact_matcher) == 1

    def test_empty_prediction_list(self):
        assert first_hit_rank([], "sepsis", exact_matcher) is None


class TestTopKAccuracy:
    RANKS = [1, 3, None, 2]      # 4 cases: hit@1, hit@3, miss, hit@2

    def test_top1(self):
        assert top_k_accuracy(self.RANKS, 1) == pytest.approx(0.25)

    def test_top3(self):
        assert top_k_accuracy(self.RANKS, 3) == pytest.approx(0.75)

    def test_top5_does_not_exceed_available_hits(self):
        assert top_k_accuracy(self.RANKS, 5) == pytest.approx(0.75)

    def test_empty_case_set_is_zero_not_an_error(self):
        assert top_k_accuracy([], 3) == 0.0

    def test_k_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            top_k_accuracy(self.RANKS, 0)


class TestReciprocalRank:
    def test_rank_one_is_one(self):
        assert reciprocal_rank(1) == pytest.approx(1.0)

    def test_rank_four_is_a_quarter(self):
        assert reciprocal_rank(4) == pytest.approx(0.25)

    def test_miss_contributes_zero(self):
        assert reciprocal_rank(None) == 0.0

    def test_mrr_over_mixed_ranks(self):
        # (1/1 + 1/3 + 0 + 1/2) / 4 = 1.8333.. / 4
        assert mean_reciprocal_rank([1, 3, None, 2]) == pytest.approx(1.8333333 / 4)

    def test_mrr_separates_what_top3_accuracy_conflates(self):
        # Both arms are 100% correct at top-3; only MRR sees that one of them
        # ranked every answer first. This is the signal the old pass/fail
        # harness discarded.
        assert top_k_accuracy([1, 1, 1], 3) == top_k_accuracy([3, 3, 3], 3)
        assert mean_reciprocal_rank([1, 1, 1]) > mean_reciprocal_rank([3, 3, 3])


class TestDistractorSelectionRate:
    def test_flags_a_distractor_in_the_top_prediction(self):
        assert distractor_selection_rate(
            ["Crohn disease", "Colon cancer"], ["Crohn disease"], exact_matcher
        ) is True

    def test_distractor_below_the_cutoff_is_not_counted(self):
        assert distractor_selection_rate(
            ["Colon cancer", "Crohn disease"], ["Crohn disease"], exact_matcher
        ) is False

    def test_wider_cutoff_sees_the_distractor(self):
        assert distractor_selection_rate(
            ["Colon cancer", "Crohn disease"], ["Crohn disease"],
            exact_matcher, top_n=2,
        ) is True

    def test_a_correct_answer_is_never_a_trap_capture(self):
        # CUPCase ships distractors that overlap the ground truth — e.g. truth
        # "Myasthenia Gravis (MG) and ..." with "Myasthenia Gravis" among the
        # distractors. Without the ground_truth guard the metric penalises an
        # arm for being right, which is how "Apiro selects 4x fewer
        # distractors" got reported when Apiro was simply wrong more often.
        assert distractor_selection_rate(
            ["Myasthenia gravis"], ["Myasthenia gravis", "Lambert-Eaton"],
            exact_matcher, ground_truth="Myasthenia gravis",
        ) is False

    def test_a_genuine_capture_is_still_detected(self):
        assert distractor_selection_rate(
            ["Lambert-Eaton"], ["Lambert-Eaton"],
            exact_matcher, ground_truth="Myasthenia gravis",
        ) is True

    def test_a_case_whose_distractors_all_match_the_truth_is_excluded(self):
        # Nothing on such a case can distinguish a trap capture from a correct
        # answer, so it must not contribute to the denominator either.
        assert distractor_selection_rate(
            ["Anything at all"], ["Myasthenia gravis"],
            exact_matcher, ground_truth="Myasthenia gravis",
        ) is None

    def test_none_when_case_ships_no_distractors(self):
        # None, not False: a case with no distractors cannot contribute to the
        # rate, and counting it as a non-selection would dilute the denominator.
        assert distractor_selection_rate(["Colon cancer"], [], exact_matcher) is None
        assert distractor_selection_rate(["Colon cancer"], ["", "  "], exact_matcher) is None


# --------------------------------------------------------------------------- #
# Intervals on a single proportion
# --------------------------------------------------------------------------- #
class TestWilsonInterval:
    def test_matches_hand_computed_value_for_the_published_niah_result(self):
        # 17/25 is the published Apiro C-NIAH figure. Wilson 95%:
        #   z = 1.959964, denom = 1 + z^2/25 = 1.153661
        #   centre = (0.68 + z^2/50) / denom = 0.656033
        #   half   = (z/denom) * sqrt(0.68*0.32/25 + z^2/2500) = 0.171927
        low, high = wilson_interval(17, 25)
        assert low == pytest.approx(0.4841, abs=1e-3)
        assert high == pytest.approx(0.8280, abs=1e-3)

    def test_interval_is_wide_enough_to_cover_the_rag_baseline(self):
        # The point of reporting it: 17/25 (68%) and 10/25 (40%) have
        # overlapping marginal intervals, so the headline delta needs the
        # paired test below, not a comparison of these two intervals.
        apiro_low, _ = wilson_interval(17, 25)
        _, rag_high = wilson_interval(10, 25)
        assert apiro_low < rag_high

    def test_zero_successes_does_not_collapse_to_a_point(self):
        # The Wald interval gives [0, 0] here, which would claim certainty from
        # 10 observations. Wilson does not.
        low, high = wilson_interval(0, 10)
        assert low == 0.0
        assert 0.2 < high < 0.35

    def test_all_successes_does_not_collapse_to_a_point(self):
        low, high = wilson_interval(10, 10)
        assert high == 1.0
        assert 0.65 < low < 0.80

    def test_no_data_means_no_constraint(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            wilson_interval(11, 10)
        with pytest.raises(ValueError):
            wilson_interval(-1, 10)

    def test_rejects_out_of_range_confidence(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 10, confidence=1.0)


class TestNormalQuantile:
    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.975, 1.959964),
            (0.95, 1.644854),
            (0.5, 0.0),
            (0.025, -1.959964),
        ],
    )
    def test_known_quantiles(self, p, expected):
        assert _normal_quantile(p) == pytest.approx(expected, abs=1e-5)

    def test_tail_branches_are_symmetric(self):
        assert _normal_quantile(0.001) == pytest.approx(-_normal_quantile(0.999), abs=1e-6)

    def test_rejects_closed_interval_endpoints(self):
        with pytest.raises(ValueError):
            _normal_quantile(0.0)
        with pytest.raises(ValueError):
            _normal_quantile(1.0)


class TestBootstrapProportionCI:
    def test_brackets_the_observed_proportion(self):
        outcomes = [True] * 17 + [False] * 8
        low, high = bootstrap_proportion_ci(outcomes)
        assert low < 0.68 < high

    def test_agrees_with_wilson_to_within_a_few_points(self):
        outcomes = [True] * 17 + [False] * 8
        b_low, b_high = bootstrap_proportion_ci(outcomes)
        w_low, w_high = wilson_interval(17, 25)
        assert abs(b_low - w_low) < 0.08
        assert abs(b_high - w_high) < 0.08

    def test_is_reproducible_under_a_fixed_seed(self):
        outcomes = [True, False, True, True, False]
        assert bootstrap_proportion_ci(outcomes, seed=7) == bootstrap_proportion_ci(outcomes, seed=7)

    def test_degenerate_input_has_zero_width(self):
        assert bootstrap_proportion_ci([True] * 10) == (1.0, 1.0)

    def test_empty_means_no_constraint(self):
        assert bootstrap_proportion_ci([]) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Paired comparison
# --------------------------------------------------------------------------- #
class TestMcNemarExact:
    def test_counts_only_discordant_pairs(self):
        a = [True, True, False, False, True]
        b = [True, False, True, False, False]
        result = mcnemar_exact(a, b)
        assert result["a_only"] == 2      # cases 2 and 5
        assert result["b_only"] == 1      # case 3
        assert result["n_discordant"] == 3
        assert result["n_cases"] == 5

    def test_no_disagreement_gives_no_evidence(self):
        result = mcnemar_exact([True, False], [True, False])
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0
        assert result["significant_at_05"] is False

    def test_five_zero_split_is_not_significant(self):
        # 2 * (1/2^5) = 0.0625 — the classic result that five straight wins are
        # not enough at alpha = 0.05.
        result = mcnemar_exact([True] * 5 + [True] * 3, [False] * 5 + [True] * 3)
        assert result["n_discordant"] == 5
        assert result["p_value"] == pytest.approx(0.0625)
        assert result["significant_at_05"] is False

    def test_six_zero_split_is_significant(self):
        # 2 * (1/2^6) = 0.03125
        result = mcnemar_exact([True] * 6, [False] * 6)
        assert result["p_value"] == pytest.approx(0.03125)
        assert result["significant_at_05"] is True

    def test_even_split_is_maximally_inconclusive(self):
        result = mcnemar_exact([True, False], [False, True])
        assert result["p_value"] == 1.0

    def test_p_value_never_exceeds_one(self):
        for n_a, n_b in [(1, 1), (2, 2), (3, 4), (5, 5)]:
            a = [True] * n_a + [False] * n_b
            b = [False] * n_a + [True] * n_b
            assert 0.0 <= mcnemar_exact(a, b)["p_value"] <= 1.0

    def test_direction_is_reported(self):
        forward = mcnemar_exact([True] * 6, [False] * 6)
        reverse = mcnemar_exact([False] * 6, [True] * 6)
        assert forward["a_only"] == reverse["b_only"] == 6
        assert forward["p_value"] == pytest.approx(reverse["p_value"])

    def test_mismatched_lengths_are_rejected(self):
        # Two arms scored on different case sets is not a paired design; a
        # silent zip() would have truncated and reported a test that means
        # nothing.
        with pytest.raises(ValueError):
            mcnemar_exact([True, False], [True])


class TestPairedBootstrapDeltaCI:
    def test_delta_is_the_plain_accuracy_difference(self):
        a = [True] * 17 + [False] * 8      # 0.68
        b = [True] * 10 + [False] * 15     # 0.40
        result = paired_bootstrap_delta_ci(a, b)
        assert result["delta"] == pytest.approx(0.28)
        assert result["n_cases"] == 25

    def test_interval_brackets_the_point_estimate(self):
        a = [True] * 17 + [False] * 8
        b = [True] * 10 + [False] * 15
        result = paired_bootstrap_delta_ci(a, b)
        assert result["ci_low"] <= result["delta"] <= result["ci_high"]

    def test_identical_arms_give_a_zero_width_interval(self):
        outcomes = [True, False, True, True]
        result = paired_bootstrap_delta_ci(outcomes, outcomes)
        assert result["delta"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0
        assert result["excludes_zero"] is False

    def test_a_clean_sweep_excludes_zero(self):
        result = paired_bootstrap_delta_ci([True] * 20, [False] * 20)
        assert result["delta"] == pytest.approx(1.0)
        assert result["excludes_zero"] is True

    def test_is_reproducible_under_a_fixed_seed(self):
        a = [True, False, True, True, False]
        b = [False, False, True, False, True]
        assert (paired_bootstrap_delta_ci(a, b, seed=11)
                == paired_bootstrap_delta_ci(a, b, seed=11))

    def test_empty_input(self):
        result = paired_bootstrap_delta_ci([], [])
        assert result["n_cases"] == 0
        assert result["delta"] == 0.0

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap_delta_ci([True, False], [True])


# --------------------------------------------------------------------------- #
# Signal health
# --------------------------------------------------------------------------- #
class TestSignalHealth:
    def test_a_constant_signal_is_degenerate(self):
        h = signal_health([0.1] * 100)
        assert h["degenerate"] is True
        assert h["n_distinct"] == 1
        assert h["modal_share"] == pytest.approx(1.0)
        assert h["normalized_entropy"] == pytest.approx(0.0)

    def test_a_spread_signal_is_not(self):
        h = signal_health([i / 100 for i in range(100)])
        assert h["degenerate"] is False
        assert h["normalized_entropy"] == pytest.approx(1.0, abs=1e-9)

    def test_reproduces_the_measured_collapse(self):
        # The real shape from the 2026-08-30 run: 64% at one value.
        values = [0.10] * 643 + [0.65] * 178 + [0.693] * 118 + [0.25] * 32 + [0.05] * 29
        h = signal_health(values)
        assert h["modal_value"] == pytest.approx(0.10)
        assert h["modal_share"] == pytest.approx(0.643, abs=0.01)
        assert h["degenerate"] is True

    def test_threshold_is_configurable(self):
        values = [0.1] * 45 + [0.2] * 55
        assert signal_health(values, top_share_warn=0.50)["degenerate"] is True
        assert signal_health(values, top_share_warn=0.60)["degenerate"] is False

    def test_empty_and_none_are_safe(self):
        assert signal_health([])["n"] == 0
        assert signal_health([None, None])["n"] == 0
        assert signal_health([0.1, None, 0.2])["n"] == 2

    def test_distribution_is_json_safe(self):
        import json
        json.dumps(signal_health([0.1, 0.2, 0.2]))


# --------------------------------------------------------------------------- #
# Counterfactual traps (MedEinst) and abstention (MedAbstain)
# --------------------------------------------------------------------------- #
class TestBiasTrapRate:
    def test_a_prior_driven_model_is_trapped_every_time(self):
        # Right on every control, wrong on every trap: the signature of
        # answering from statistical priors rather than from the evidence.
        t = bias_trap_rate([True] * 10, [False] * 10)
        assert t["trap_rate"] == pytest.approx(1.0)
        assert t["consistent"] == 0
        assert t["n_trapped"] == 10

    def test_an_evidence_driven_model_escapes_every_trap(self):
        t = bias_trap_rate([True] * 10, [True] * 10)
        assert t["trap_rate"] == pytest.approx(0.0)
        assert t["consistent"] == 10

    def test_denominator_is_controls_the_arm_got_right(self):
        # A pair the arm failed on the control says nothing about whether
        # flipping the evidence would have flipped its answer.
        t = bias_trap_rate([True, True, False, False], [True, False, False, False])
        assert t["n_control_correct"] == 2
        assert t["trap_rate"] == pytest.approx(0.5)

    def test_no_control_solved_is_undefined_not_a_crash(self):
        t = bias_trap_rate([False, False], [False, False])
        assert t["trap_rate"] is None
        assert t["trap_rate_ci"] == (0.0, 1.0)

    def test_trap_cannot_be_passed_by_ignoring_the_note(self):
        # The property that makes this design stronger than a fixed-answer
        # distractor: an arm that always says the prior diagnosis scores
        # perfectly on controls and zero on traps.
        always_prior_control = [True] * 8      # prior == truth on controls
        always_prior_trap = [False] * 8        # prior != truth on traps
        assert bias_trap_rate(always_prior_control, always_prior_trap)["trap_rate"] == 1.0

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            bias_trap_rate([True, False], [True])


class TestCompareBiasTraps:
    def test_detects_a_large_gap(self):
        c = compare_bias_traps([True] * 12, [True] * 12, [True] * 12, [False] * 12)
        assert c["n_comparable"] == 12
        assert c["a_escaped"] == 12 and c["b_escaped"] == 0
        assert c["mcnemar"]["significant_at_05"] is True

    def test_restricted_to_shared_correct_controls(self):
        c = compare_bias_traps([True, True, False], [True, True, False],
                               [True, False, False], [True, False, False])
        assert c["n_comparable"] == 1

    def test_no_shared_control_reports_nothing_comparable(self):
        c = compare_bias_traps([True, False], [True, False],
                               [False, True], [False, True])
        assert c["n_comparable"] == 0

    def test_arms_on_different_pair_counts_are_rejected(self):
        with pytest.raises(ValueError):
            compare_bias_traps([True], [True], [True, False], [True, False])


class TestAbstentionMetrics:
    SHOULD = [True] * 4 + [False] * 6          # 4 unanswerable, 6 answerable

    def test_a_model_that_never_declines_fabricates_on_all_of_them(self):
        a = abstention_metrics([False] * 10, self.SHOULD)
        assert a["fabrication_rate"] == pytest.approx(1.0)
        assert a["over_abstention_rate"] == pytest.approx(0.0)

    def test_a_perfect_abstainer(self):
        a = abstention_metrics([True] * 4 + [False] * 6, self.SHOULD)
        assert a["fabrication_rate"] == pytest.approx(0.0)
        assert a["abstention_recall"] == pytest.approx(1.0)
        assert a["over_abstention_rate"] == pytest.approx(0.0)

    def test_declining_everything_is_safe_but_useless(self):
        # Zero fabrication and zero coverage. Reported separately on purpose:
        # collapsing them into one accuracy number would hide which is which.
        a = abstention_metrics([True] * 10, self.SHOULD)
        assert a["fabrication_rate"] == pytest.approx(0.0)
        assert a["over_abstention_rate"] == pytest.approx(1.0)
        assert a["coverage"] == pytest.approx(0.0)

    def test_selective_accuracy_is_over_answered_answerable_cases(self):
        a = abstention_metrics(
            [True] * 4 + [False] * 6, self.SHOULD,
            correct_when_answered=[False] * 4 + [True] * 4 + [False] * 2,
        )
        assert a["n_answered_answerable"] == 6
        assert a["selective_accuracy"] == pytest.approx(4 / 6)

    def test_correctness_length_is_validated(self):
        with pytest.raises(ValueError):
            abstention_metrics([True], [True], correct_when_answered=[True, False])


# --------------------------------------------------------------------------- #
# Matched-pair distractor resilience
# --------------------------------------------------------------------------- #
class TestDistractorRobustness:
    def test_a_perfectly_robust_arm_loses_nothing(self):
        clean = [True] * 8 + [False] * 2
        r = distractor_robustness(clean, clean)
        assert r["broken"] == 0
        assert r["degradation"] == pytest.approx(0.0)
        assert r["retention"] == pytest.approx(1.0)

    def test_a_fragile_arm_is_separated_despite_equal_clean_accuracy(self):
        # The whole point of the paired design: both arms solve 8/10 clean, so
        # aggregate accuracy on the clean condition cannot tell them apart.
        clean = [True] * 8 + [False] * 2
        fragile = [True] * 3 + [False] * 7
        r = distractor_robustness(clean, fragile)
        assert r["clean_accuracy"] == pytest.approx(0.8)
        assert r["broken"] == 5
        # 8 solvable, 5 broken -> 3 survived.
        assert r["retention"] == pytest.approx(3 / 8)
        assert r["degradation"] == pytest.approx(0.5)

    def test_retention_denominator_is_the_solvable_set(self):
        # Cases the arm never solved clean cannot be "broken" by a distractor
        # and must not dilute the rate.
        r = distractor_robustness([True, True, False, False], [True, False, False, False])
        assert r["n_solvable"] == 2
        assert r["retention"] == pytest.approx(0.5)

    def test_rescued_cases_are_counted_separately(self):
        # Right on adversarial, wrong on clean: noise, not resilience.
        r = distractor_robustness([False, True], [True, True])
        assert r["rescued"] == 1
        assert r["broken"] == 0

    def test_degradation_ci_brackets_the_estimate(self):
        clean = [True] * 8 + [False] * 2
        adv = [True] * 3 + [False] * 7
        r = distractor_robustness(clean, adv)
        assert r["degradation_ci"][0] <= r["degradation"] <= r["degradation_ci"][1]

    def test_empty_input(self):
        r = distractor_robustness([], [])
        assert r["n_pairs"] == 0
        assert r["retention"] is None

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            distractor_robustness([True, False], [True])


class TestCompareRobustness:
    def test_restricted_to_pairs_both_arms_solved_clean(self):
        # Arm A solves 3, arm B solves 2; only 2 are comparable. Including the
        # third would confound resilience with baseline capability.
        a_clean = [True, True, True, False]
        b_clean = [True, True, False, False]
        result = compare_robustness(a_clean, a_clean, b_clean, b_clean)
        assert result["n_comparable"] == 2

    def test_detects_a_robustness_gap(self):
        a_clean = [True] * 10
        a_adv = [True] * 10                 # survives everything
        b_clean = [True] * 10
        b_adv = [False] * 10                # survives nothing
        result = compare_robustness(a_clean, a_adv, b_clean, b_adv)
        assert result["n_comparable"] == 10
        assert result["a_survived"] == 10
        assert result["b_survived"] == 0
        assert result["mcnemar"]["significant_at_05"] is True

    def test_identical_arms_are_not_separated(self):
        clean = [True] * 6
        adv = [True, True, False, True, False, True]
        result = compare_robustness(clean, adv, clean, adv)
        assert result["mcnemar"]["p_value"] == 1.0

    def test_no_shared_solvable_pair_reports_nothing_comparable(self):
        result = compare_robustness([True, False], [True, False],
                                    [False, True], [False, True])
        assert result["n_comparable"] == 0

    def test_arms_scored_on_different_pair_counts_are_rejected(self):
        with pytest.raises(ValueError):
            compare_robustness([True], [True], [True, False], [True, False])


# --------------------------------------------------------------------------- #
# Arm aggregation
# --------------------------------------------------------------------------- #
class TestScoreArm:
    PREDICTIONS = [
        ["Pulmonary embolism", "Pneumonia", "Asthma"],   # hit @1
        ["Pneumonia", "Sepsis", "Meningitis"],           # hit @3 (truth=meningitis)
        ["Asthma", "Anxiety", "GERD"],                   # miss
    ]
    TRUTHS = ["Pulmonary embolism", "Meningitis", "Aortic dissection"]
    DISTRACTORS = [["Pneumonia"], ["Migraine"], ["Asthma"]]

    def _scores(self):
        return score_arm(
            "apiro", self.PREDICTIONS, self.TRUTHS, exact_matcher,
            distractors_per_case=self.DISTRACTORS,
        )

    def test_ranks_and_top_k(self):
        s = self._scores()
        assert s.ranks == [1, 3, None]
        assert s.top_k[1] == pytest.approx(1 / 3)
        assert s.top_k[3] == pytest.approx(2 / 3)

    def test_mrr(self):
        # (1 + 1/3 + 0) / 3
        assert self._scores().mrr == pytest.approx((1 + 1 / 3) / 3)

    def test_distractor_rate_counts_only_top1_selections(self):
        # Case 3 leads with "Asthma", which is its curated distractor.
        # Cases 1 and 2 lead with something else.
        s = self._scores()
        assert s.n_distractor_cases == 3
        assert s.distractor_rate == pytest.approx(1 / 3)

    def test_confidence_intervals_are_attached(self):
        s = self._scores()
        assert s.top1_ci[0] <= s.top_k[1] <= s.top1_ci[1]
        assert s.top3_ci[0] <= s.top_k[3] <= s.top3_ci[1]

    def test_outcomes_at_feeds_the_paired_tests(self):
        assert self._scores().outcomes_at(1) == [True, False, False]
        assert self._scores().outcomes_at(3) == [True, True, False]

    def test_distractor_rate_is_none_without_distractors(self):
        s = score_arm("apiro", self.PREDICTIONS, self.TRUTHS, exact_matcher)
        assert s.distractor_rate is None
        assert s.n_distractor_cases == 0

    def test_to_dict_is_json_safe(self):
        import json

        payload = self._scores().to_dict()
        json.dumps(payload)                      # must not raise
        assert set(payload["top_k"]) == {"1", "3", "5"}

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            score_arm("apiro", self.PREDICTIONS, self.TRUTHS[:2], exact_matcher)
        with pytest.raises(ValueError):
            score_arm("apiro", self.PREDICTIONS, self.TRUTHS, exact_matcher,
                      distractors_per_case=[[]])


class TestCompareArms:
    def _arms(self):
        return {
            "apiro": ArmScores(arm="apiro", n_cases=4, ranks=[1, 2, 3, None]),
            "bare_llm": ArmScores(arm="bare_llm", n_cases=4, ranks=[None, None, 3, None]),
        }

    def test_compares_every_arm_against_the_reference(self):
        result = compare_arms(self._arms(), reference="bare_llm", k=3)
        assert set(result) == {"apiro"}
        assert result["apiro"]["vs"] == "bare_llm"
        assert result["apiro"]["k"] == 3

    def test_delta_reflects_the_cutoff(self):
        # At k=3 Apiro is 3/4 and the baseline 1/4.
        result = compare_arms(self._arms(), reference="bare_llm", k=3)
        assert result["apiro"]["delta_ci"]["delta"] == pytest.approx(0.5)
        assert result["apiro"]["mcnemar"]["a_only"] == 2
        assert result["apiro"]["mcnemar"]["b_only"] == 0

    def test_unknown_reference_is_rejected(self):
        with pytest.raises(KeyError):
            compare_arms(self._arms(), reference="rag")


# --------------------------------------------------------------------------- #
# Cross-cutting sanity
# --------------------------------------------------------------------------- #
def test_all_reported_quantities_are_finite():
    """No metric may emit NaN or inf — a results JSON must stay serialisable."""
    a = [True, False, True, True, False]
    b = [False, True, True, False, False]

    values = [
        top_k_accuracy([1, None, 2], 3),
        mean_reciprocal_rank([1, None, 2]),
        *wilson_interval(3, 5),
        *bootstrap_proportion_ci(a),
        paired_bootstrap_delta_ci(a, b)["delta"],
        paired_bootstrap_delta_ci(a, b)["ci_low"],
        paired_bootstrap_delta_ci(a, b)["ci_high"],
        mcnemar_exact(a, b)["p_value"],
    ]
    assert all(math.isfinite(v) for v in values)
