"""
apiro/eval/metrics.py — differential-diagnosis metrics and paired significance
==============================================================================

The evaluation harnesses in ``scripts/`` used to report a single number per
arm: the fraction of cases where the ground-truth diagnosis appeared anywhere
in the model's top-3 output. Two things were missing from every published
figure:

1. **Rank information.** "In the top 3" scores a first-place answer and a
   third-place answer identically. For a differential-diagnosis engine, where
   the differential is *ranked by design*, that discards the signal the system
   is built to produce. Top-1 accuracy and Mean Reciprocal Rank recover it.

2. **Uncertainty.** The README reports "68.0% vs 40.0%, +28% lift" on N = 25.
   The 95% Wilson interval for 17/25 is roughly [48%, 83%], and the two arms
   are scored *on the same cases*, so the comparison is paired and an unpaired
   eyeball of the two intervals is the wrong test. This module provides the
   right one — an exact McNemar test on the discordant pairs — plus a paired
   bootstrap CI on the delta itself.

Everything here is pure numpy/stdlib (no scipy/sklearn), deterministic under a
fixed seed, and free of any Apiro-specific imports, so it can be unit-tested
without Ollama, ChromaDB, or a model download.

Matching is delegated: callers pass a ``matcher(prediction, truth) -> bool``,
normally a partial of ``apiro.eval.evaluator._check_synthesis_hit``, so the
clinical concept-normalization cascade stays in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "clean_predictions",
    "first_hit_rank",
    "top_k_accuracy",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "distractor_selection_rate",
    "wilson_interval",
    "bootstrap_proportion_ci",
    "paired_bootstrap_delta_ci",
    "mcnemar_exact",
    "distractor_robustness",
    "compare_robustness",
    "bias_trap_rate",
    "compare_bias_traps",
    "abstention_metrics",
    "ArmScores",
    "score_arm",
    "compare_arms",
]

#: A predicate deciding whether one predicted diagnosis matches one ground
#: truth. Signature: ``matcher(prediction, ground_truth) -> bool``.
Matcher = Callable[[str, str], bool]

#: Ranks at which top-k accuracy is reported by default. A clinical
#: differential is conventionally read at 1, 3 and 5.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

#: Default number of bootstrap resamples. 10,000 gives a percentile interval
#: stable to well under a percentage point at the sample sizes used here.
DEFAULT_N_BOOTSTRAP = 10_000

#: Default seed, so a reported interval is reproducible from the results file.
DEFAULT_SEED = 20260829


# --------------------------------------------------------------------------- #
# Rank-aware hit metrics
# --------------------------------------------------------------------------- #
def clean_predictions(predictions: Sequence[str]) -> list[str]:
    """Drop empty/whitespace entries from a differential, preserving order.

    Blank lines are formatting artifacts, not candidate diagnoses, so they must
    not occupy a rank: an arm whose output happened to contain a stray newline
    would otherwise have its real answer pushed down and its MRR reduced for a
    reason that has nothing to do with its reasoning.
    """
    return [
        str(p).strip()
        for p in predictions
        if p is not None and str(p).strip()
    ]


def first_hit_rank(
    predictions: Sequence[str],
    ground_truth: str,
    matcher: Matcher,
) -> Optional[int]:
    """Return the 1-based rank of the first prediction matching ``ground_truth``.

    Blank entries are removed before ranking (see :func:`clean_predictions`).

    Args:
        predictions: The model's ranked differential, most likely first.
        ground_truth: The reference diagnosis.
        matcher: Predicate deciding clinical equivalence.

    Returns:
        The 1-based rank of the first match, or ``None`` if nothing matched.
    """
    for rank, prediction in enumerate(clean_predictions(predictions), start=1):
        if matcher(prediction, ground_truth):
            return rank
    return None


def top_k_accuracy(
    ranks: Sequence[Optional[int]],
    k: int,
) -> float:
    """Fraction of cases whose first matching prediction sits at rank <= k.

    Args:
        ranks: Per-case output of :func:`first_hit_rank` (``None`` = miss).
        k: The cut-off rank.

    Returns:
        Accuracy in ``[0, 1]``; ``0.0`` for an empty case list.

    Raises:
        ValueError: If ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def reciprocal_rank(rank: Optional[int]) -> float:
    """``1 / rank`` for a hit, ``0.0`` for a miss."""
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / float(rank)


def mean_reciprocal_rank(ranks: Sequence[Optional[int]]) -> float:
    """Mean of :func:`reciprocal_rank` over all cases (misses contribute 0)."""
    if not ranks:
        return 0.0
    return float(np.mean([reciprocal_rank(r) for r in ranks]))


def distractor_selection_rate(
    predictions: Sequence[str],
    distractors: Sequence[str],
    matcher: Matcher,
    top_n: int = 1,
    ground_truth: Optional[str] = None,
) -> Optional[bool]:
    """Did the model's top-``top_n`` output name one of the case's distractors?

    This is the metric that speaks directly to Apiro's central claim. Accuracy
    says whether the right answer was found; this says whether the *designed
    wrong answer* was chosen instead. A benchmark whose cases ship their own
    curated distractors (CUPCase does) can measure distractor rejection
    directly rather than inferring it from an accuracy gap.

    Args:
        predictions: The model's ranked differential.
        distractors: Curated plausible-but-wrong diagnoses for this case.
        matcher: Predicate deciding clinical equivalence.
        top_n: How many of the model's leading predictions to inspect.
        ground_truth: The correct answer. **Pass this whenever it is known.**
            A prediction that matches both a distractor and the truth is a
            correct answer, not a trap capture, and counting it as one
            penalises an arm for being right.

    Returns:
        ``True``/``False``, or ``None`` when the case cannot contribute — no
        distractors, or every distractor is itself a match for the ground
        truth, which makes "picked a distractor" undefined for that case.

    Note:
        Curated distractor lists are not always disjoint from the answer. In
        CUPCase, for example, the ground truth "Intestinal obstruction due to
        Ascaris lumbricoides" ships distractors "Unspecified intestinal
        obstruction" and "Other complete intestinal obstruction" — ICD-level
        near-misses rather than clinically wrong alternatives. On such a case
        this returns ``None``: naming the distractor there is a *granularity*
        error, not the failure this metric exists to detect, and scoring it as
        one would misreport what the number means.
    """
    usable = clean_predictions(distractors)
    if not usable:
        return None

    if ground_truth is not None and str(ground_truth).strip():
        # Drop distractors that are themselves matches for the truth; if that
        # empties the list, the case cannot discriminate and is excluded.
        usable = [d for d in usable if not matcher(d, ground_truth)]
        if not usable:
            return None

    for prediction in clean_predictions(predictions)[:top_n]:
        if ground_truth is not None and matcher(prediction, ground_truth):
            # Correct answers are never trap captures, whatever else they
            # happen to resemble.
            continue
        if any(matcher(prediction, d) for d in usable):
            return True
    return False


# --------------------------------------------------------------------------- #
# Uncertainty on a single proportion
# --------------------------------------------------------------------------- #
def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) approximation at the sample sizes this
    project evaluates at: Wald produces intervals that leave ``[0, 1]`` and
    collapses to zero width at 0% or 100%, both of which occur in the current
    per-family breakdowns (negation_trap is 0/2, long-context is 5/5).

    Args:
        successes: Number of correct cases.
        total: Number of cases evaluated.
        confidence: Two-sided confidence level.

    Returns:
        ``(low, high)``, clamped to ``[0, 1]``. Returns ``(0.0, 1.0)`` when
        ``total == 0`` — no data means no constraint, not a point estimate.

    Raises:
        ValueError: If the inputs are not a valid success/trial pair or the
            confidence level is outside ``(0, 1)``.
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(
            f"Invalid counts: successes={successes}, total={total}."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if total == 0:
        return (0.0, 1.0)

    z = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    n = float(total)
    p = successes / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_proportion_ci(
    outcomes: Sequence[bool],
    confidence: float = 0.95,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a boolean outcome vector.

    Provided alongside :func:`wilson_interval` so a report can show that the
    closed-form and resampled intervals agree; a large disagreement is a
    signal that the case set is too small to summarize with either.

    Args:
        outcomes: Per-case correctness flags.
        confidence: Two-sided confidence level.
        n_bootstrap: Number of resamples.
        seed: RNG seed, so the interval is reproducible.

    Returns:
        ``(low, high)``. ``(0.0, 1.0)`` for an empty input.
    """
    if not outcomes:
        return (0.0, 1.0)
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    values = np.asarray(outcomes, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(low), float(high))


# --------------------------------------------------------------------------- #
# Paired comparison between two arms
# --------------------------------------------------------------------------- #
def _validate_paired(a: Sequence[bool], b: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Coerce two per-case outcome vectors, requiring equal length."""
    arr_a = np.asarray(a, dtype=bool).ravel()
    arr_b = np.asarray(b, dtype=bool).ravel()
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"Paired comparison needs equal-length outcome vectors, got "
            f"{arr_a.shape[0]} and {arr_b.shape[0]}. Both arms must be scored "
            f"on the same cases, in the same order."
        )
    return arr_a, arr_b


def paired_bootstrap_delta_ci(
    a_outcomes: Sequence[bool],
    b_outcomes: Sequence[bool],
    confidence: float = 0.95,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap CI for ``accuracy(a) - accuracy(b)``, resampling *cases*.

    Resampling case indices (rather than each arm independently) preserves the
    pairing: both arms are re-scored on the same bootstrap sample, exactly as
    they were on the real one. Independent resampling would inflate the
    variance of the difference and widen the interval for no reason.

    Args:
        a_outcomes: Per-case correctness for the arm under test.
        b_outcomes: Per-case correctness for the comparison arm.
        confidence: Two-sided confidence level.
        n_bootstrap: Number of resamples.
        seed: RNG seed.

    Returns:
        ``{"delta", "ci_low", "ci_high", "n_cases", "excludes_zero"}``.
        ``excludes_zero`` is a convenience flag, not a p-value — use
        :func:`mcnemar_exact` for the test.
    """
    arr_a, arr_b = _validate_paired(a_outcomes, b_outcomes)
    n = arr_a.size
    if n == 0:
        return {
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_cases": 0,
            "excludes_zero": False,
        }

    delta = float(arr_a.mean() - arr_b.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    deltas = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "delta": delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_cases": int(n),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def mcnemar_exact(
    a_outcomes: Sequence[bool],
    b_outcomes: Sequence[bool],
) -> dict:
    """Exact (binomial) McNemar test on two paired correctness vectors.

    McNemar conditions on the *discordant* cases — those one arm got right and
    the other got wrong — and asks whether the split between the two kinds of
    disagreement is consistent with a fair coin. Cases both arms agree on carry
    no information about which is better and are correctly ignored.

    The exact binomial form is used rather than the chi-squared approximation
    because the discordant count here is small (on the published N = 25 NIAH
    run, Apiro and the bare LLM disagree on 11 cases), which is well inside the
    range where the asymptotic version is unreliable.

    Args:
        a_outcomes: Per-case correctness for the arm under test.
        b_outcomes: Per-case correctness for the comparison arm.

    Returns:
        ``{"n_cases", "a_only", "b_only", "n_discordant", "p_value",
        "significant_at_05"}`` where ``a_only`` counts cases only ``a`` got
        right. ``p_value`` is two-sided; it is ``1.0`` when there are no
        discordant pairs (no evidence either way).
    """
    arr_a, arr_b = _validate_paired(a_outcomes, b_outcomes)

    a_only = int(np.count_nonzero(arr_a & ~arr_b))
    b_only = int(np.count_nonzero(~arr_a & arr_b))
    n_discordant = a_only + b_only

    if n_discordant == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial: sum the probability of every split at
        # least as extreme as the observed one, under p = 0.5.
        k = min(a_only, b_only)
        tail = sum(
            math.comb(n_discordant, i) for i in range(0, k + 1)
        ) / (2.0 ** n_discordant)
        p_value = min(1.0, 2.0 * tail)

    return {
        "n_cases": int(arr_a.size),
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_discordant,
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
    }


# --------------------------------------------------------------------------- #
# Matched-pair distractor resilience — the on-thesis endpoint
# --------------------------------------------------------------------------- #
def distractor_robustness(
    clean_outcomes: Sequence[bool],
    adversarial_outcomes: Sequence[bool],
    confidence: float = 0.95,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict:
    """How much does one arm degrade when a distractor is added?

    Inputs are per-*pair*, aligned: ``clean_outcomes[i]`` and
    ``adversarial_outcomes[i]`` are the same arm on the same case with and
    without the adversarial sentence.

    This is the endpoint the architecture is actually about. Aggregate accuracy
    compares arms across different cases and therefore carries all the variance
    of case difficulty, which is large and unrelated to distractor resilience.
    Here each pair is its own control, so that variance cancels.

    ``retention`` — P(correct on adversarial | correct on clean) — is the
    headline number: of the cases an arm could solve, what fraction survived
    the distractor? A system whose claim is "rejects distractors instead of
    rationalizing them" should have a retention near 1.0 even if its raw
    accuracy is unremarkable.

    Returns:
        A dict with ``n_pairs``, ``clean_accuracy``, ``adversarial_accuracy``,
        ``degradation`` (clean − adversarial, with a paired bootstrap CI),
        ``broken`` (solved clean, lost adversarial), ``rescued`` (the reverse,
        i.e. noise), ``retention`` and ``n_solvable`` (the retention
        denominator).
    """
    clean, adversarial = _validate_paired(clean_outcomes, adversarial_outcomes)
    n = clean.size
    if n == 0:
        return {
            "n_pairs": 0, "clean_accuracy": 0.0, "adversarial_accuracy": 0.0,
            "degradation": 0.0, "degradation_ci": (0.0, 0.0),
            "broken": 0, "rescued": 0, "retention": None, "n_solvable": 0,
        }

    broken = int(np.count_nonzero(clean & ~adversarial))
    rescued = int(np.count_nonzero(~clean & adversarial))
    n_solvable = int(np.count_nonzero(clean))

    delta = paired_bootstrap_delta_ci(
        clean_outcomes, adversarial_outcomes,
        confidence=confidence, n_bootstrap=n_bootstrap, seed=seed,
    )

    return {
        "n_pairs": n,
        "clean_accuracy": float(clean.mean()),
        "adversarial_accuracy": float(adversarial.mean()),
        "degradation": delta["delta"],
        "degradation_ci": (delta["ci_low"], delta["ci_high"]),
        "broken": broken,
        "rescued": rescued,
        "n_solvable": n_solvable,
        "retention": (n_solvable - broken) / n_solvable if n_solvable else None,
    }


def compare_robustness(
    arm_a_clean: Sequence[bool],
    arm_a_adversarial: Sequence[bool],
    arm_b_clean: Sequence[bool],
    arm_b_adversarial: Sequence[bool],
) -> dict:
    """Which of two arms better survives the distractor, on shared ground?

    Restricted to pairs **both** arms solved in the clean condition. A pair
    neither arm could solve says nothing about distractor resilience, and a
    pair only one arm solved confounds resilience with baseline capability —
    including either would answer a different question.

    On that subset, McNemar asks: among the cases where the two arms disagree
    about surviving the distractor, is the split better than a coin?

    Returns:
        ``{"n_comparable", "a_survived", "b_survived", "mcnemar"}``, or
        ``n_comparable = 0`` when the arms share no solvable pair.
    """
    a_clean, a_adv = _validate_paired(arm_a_clean, arm_a_adversarial)
    b_clean, b_adv = _validate_paired(arm_b_clean, arm_b_adversarial)
    if a_clean.shape != b_clean.shape:
        raise ValueError(
            f"Both arms must be scored on the same pairs, got {a_clean.size} "
            f"and {b_clean.size}."
        )

    comparable = a_clean & b_clean
    n_comparable = int(np.count_nonzero(comparable))
    if n_comparable == 0:
        return {
            "n_comparable": 0, "a_survived": 0, "b_survived": 0,
            "mcnemar": mcnemar_exact([], []),
        }

    a_survived = a_adv[comparable]
    b_survived = b_adv[comparable]
    return {
        "n_comparable": n_comparable,
        "a_survived": int(np.count_nonzero(a_survived)),
        "b_survived": int(np.count_nonzero(b_survived)),
        "mcnemar": mcnemar_exact(a_survived.tolist(), b_survived.tolist()),
    }


# --------------------------------------------------------------------------- #
# Counterfactual traps and abstention — the two research-grounded endpoints
# --------------------------------------------------------------------------- #
def bias_trap_rate(
    control_outcomes: Sequence[bool],
    trap_outcomes: Sequence[bool],
    confidence: float = 0.95,
) -> dict:
    """P(wrong on the trap | right on the control).

    The metric from MedEinst (arXiv:2601.06636). Inputs are per-pair and
    aligned: index *i* is the same arm on the control and trap halves of one
    counterfactual pair, which share a presenting syndrome but whose buried
    discriminative evidence implies different diagnoses.

    Conditioning on the control being right is what makes this sharp. It
    isolates arms that *could* do the case from arms that could not, and asks
    only whether flipping the evidence flipped the answer. A model reasoning
    from statistical priors gets the control right for the wrong reason and
    then fails the trap: its rate approaches 1.0. A model reading the evidence
    holds near 0.0.

    Unlike a distractor that leaves the answer unchanged, a trap cannot be
    passed by ignoring the note — the prior-driven answer is wrong by
    construction.

    Returns:
        ``n_pairs``, ``n_control_correct`` (the denominator),
        ``trap_rate`` (None when the arm solved no control),
        ``trap_rate_ci`` (Wilson), ``n_trapped``, and ``consistent`` — pairs
        the arm got right in both directions, the only fully correct outcome.
    """
    control, trap = _validate_paired(control_outcomes, trap_outcomes)
    n = control.size
    n_control_correct = int(np.count_nonzero(control))
    n_trapped = int(np.count_nonzero(control & ~trap))
    consistent = int(np.count_nonzero(control & trap))

    return {
        "n_pairs": int(n),
        "n_control_correct": n_control_correct,
        "n_trapped": n_trapped,
        "consistent": consistent,
        "trap_rate": (n_trapped / n_control_correct) if n_control_correct else None,
        "trap_rate_ci": (
            wilson_interval(n_trapped, n_control_correct, confidence)
            if n_control_correct else (0.0, 1.0)
        ),
        "control_accuracy": float(control.mean()) if n else 0.0,
        "trap_accuracy": float(trap.mean()) if n else 0.0,
    }


def compare_bias_traps(
    arm_a_control: Sequence[bool],
    arm_a_trap: Sequence[bool],
    arm_b_control: Sequence[bool],
    arm_b_trap: Sequence[bool],
) -> dict:
    """Head-to-head trap survival, on pairs both arms got right on the control.

    Restricting to shared controls removes baseline capability from the
    comparison: a pair only one arm could solve says nothing about which is
    less prior-driven. On that subset, McNemar tests whether one arm escapes
    the trap more often than the other.
    """
    a_ctl, a_trap = _validate_paired(arm_a_control, arm_a_trap)
    b_ctl, b_trap = _validate_paired(arm_b_control, arm_b_trap)
    if a_ctl.shape != b_ctl.shape:
        raise ValueError(
            f"Both arms must be scored on the same pairs, got {a_ctl.size} and {b_ctl.size}."
        )

    comparable = a_ctl & b_ctl
    n_comparable = int(np.count_nonzero(comparable))
    if n_comparable == 0:
        return {
            "n_comparable": 0, "a_escaped": 0, "b_escaped": 0,
            "mcnemar": mcnemar_exact([], []),
        }
    a_escaped = a_trap[comparable]
    b_escaped = b_trap[comparable]
    return {
        "n_comparable": n_comparable,
        "a_escaped": int(np.count_nonzero(a_escaped)),
        "b_escaped": int(np.count_nonzero(b_escaped)),
        "mcnemar": mcnemar_exact(a_escaped.tolist(), b_escaped.tolist()),
    }


def abstention_metrics(
    abstained: Sequence[bool],
    should_abstain: Sequence[bool],
    correct_when_answered: Optional[Sequence[bool]] = None,
) -> dict:
    """Scoring for cases whose evidence was removed (MedAbstain, 2601.12471).

    On an unanswerable case the only correct behaviour is to decline; naming a
    diagnosis anyway is a confident fabrication, which is the failure mode this
    architecture exists to prevent. On an answerable case, declining is a
    missed opportunity rather than a safety failure — so the two directions are
    reported separately rather than collapsed into one accuracy number.

    Args:
        abstained: Did the arm decline, per case?
        should_abstain: Was the case unanswerable, per case?
        correct_when_answered: Optional per-case correctness, used to compute
            selective accuracy over the answered subset.

    Returns:
        ``fabrication_rate`` — answered a question with no answer, the number
        that matters; ``abstention_recall`` — declined when it should have;
        ``over_abstention_rate`` — declined when it need not have; plus
        ``coverage`` and ``selective_accuracy`` when correctness is supplied.
    """
    abst, should = _validate_paired(abstained, should_abstain)
    n = abst.size

    n_unanswerable = int(np.count_nonzero(should))
    n_answerable = int(n - n_unanswerable)
    fabricated = int(np.count_nonzero(should & ~abst))
    correctly_declined = int(np.count_nonzero(should & abst))
    over_abstained = int(np.count_nonzero(~should & abst))

    out = {
        "n_cases": int(n),
        "n_unanswerable": n_unanswerable,
        "n_answerable": n_answerable,
        "fabricated": fabricated,
        "fabrication_rate": (fabricated / n_unanswerable) if n_unanswerable else None,
        "fabrication_rate_ci": (
            wilson_interval(fabricated, n_unanswerable) if n_unanswerable else (0.0, 1.0)
        ),
        "abstention_recall": (correctly_declined / n_unanswerable) if n_unanswerable else None,
        "over_abstained": over_abstained,
        "over_abstention_rate": (over_abstained / n_answerable) if n_answerable else None,
        "coverage": float(np.count_nonzero(~abst) / n) if n else 0.0,
    }

    if correct_when_answered is not None:
        correct = np.asarray(correct_when_answered, dtype=bool).ravel()
        if correct.shape != abst.shape:
            raise ValueError(
                f"correct_when_answered has {correct.size} entries for {n} cases."
            )
        answered_answerable = (~abst) & (~should)
        n_aa = int(np.count_nonzero(answered_answerable))
        out["selective_accuracy"] = (
            float(np.count_nonzero(correct & answered_answerable) / n_aa) if n_aa else None
        )
        out["n_answered_answerable"] = n_aa

    return out


# --------------------------------------------------------------------------- #
# Arm-level aggregation
# --------------------------------------------------------------------------- #
@dataclass
class ArmScores:
    """All rank-aware metrics for one evaluated arm.

    Attributes:
        arm: Arm name (e.g. ``"apiro"``).
        n_cases: Number of cases scored.
        ranks: Per-case first-hit rank (``None`` = miss), in case order.
        top_k: ``{k: accuracy}`` for each reported cut-off.
        mrr: Mean Reciprocal Rank.
        top1_ci: Wilson 95% interval on top-1 accuracy.
        top3_ci: Wilson 95% interval on top-3 accuracy.
        distractor_rate: Fraction of distractor-bearing cases whose top
            prediction was a curated distractor, or ``None`` when the
            benchmark ships none.
        n_distractor_cases: How many cases contributed to ``distractor_rate``.
    """

    arm: str
    n_cases: int
    ranks: list[Optional[int]] = field(default_factory=list)
    top_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    top1_ci: tuple[float, float] = (0.0, 1.0)
    top3_ci: tuple[float, float] = (0.0, 1.0)
    distractor_rate: Optional[float] = None
    n_distractor_cases: int = 0

    def outcomes_at(self, k: int) -> list[bool]:
        """Per-case correctness at cut-off ``k`` — the input to paired tests."""
        return [r is not None and r <= k for r in self.ranks]

    def to_dict(self) -> dict:
        """JSON-serializable form (``top_k`` keys become strings)."""
        return {
            "arm": self.arm,
            "n_cases": self.n_cases,
            "ranks": list(self.ranks),
            "top_k": {str(k): v for k, v in sorted(self.top_k.items())},
            "mrr": self.mrr,
            "top1_ci": list(self.top1_ci),
            "top3_ci": list(self.top3_ci),
            "distractor_rate": self.distractor_rate,
            "n_distractor_cases": self.n_distractor_cases,
        }


def score_arm(
    arm: str,
    predictions_per_case: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    matcher: Matcher,
    distractors_per_case: Optional[Sequence[Sequence[str]]] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> ArmScores:
    """Score one arm across a whole case set.

    Args:
        arm: Arm name.
        predictions_per_case: Ranked differential per case, most likely first.
        ground_truths: Reference diagnosis per case.
        matcher: Predicate deciding clinical equivalence.
        distractors_per_case: Optional curated wrong answers per case; when
            supplied, the distractor-selection rate is reported.
        k_values: Cut-offs for top-k accuracy.

    Returns:
        An :class:`ArmScores`.

    Raises:
        ValueError: If the per-case sequences have mismatched lengths.
    """
    if len(predictions_per_case) != len(ground_truths):
        raise ValueError(
            f"Arm '{arm}': {len(predictions_per_case)} prediction lists for "
            f"{len(ground_truths)} ground truths."
        )
    if distractors_per_case is not None and len(distractors_per_case) != len(ground_truths):
        raise ValueError(
            f"Arm '{arm}': {len(distractors_per_case)} distractor lists for "
            f"{len(ground_truths)} ground truths."
        )

    ranks = [
        first_hit_rank(preds, truth, matcher)
        for preds, truth in zip(predictions_per_case, ground_truths)
    ]

    distractor_rate: Optional[float] = None
    n_distractor_cases = 0
    if distractors_per_case is not None:
        flags = [
            distractor_selection_rate(preds, dists, matcher, ground_truth=truth)
            for preds, dists, truth in zip(
                predictions_per_case, distractors_per_case, ground_truths
            )
        ]
        scored = [f for f in flags if f is not None]
        n_distractor_cases = len(scored)
        if scored:
            distractor_rate = sum(1 for f in scored if f) / len(scored)

    n = len(ranks)
    top_k = {k: top_k_accuracy(ranks, k) for k in k_values}
    top1_hits = sum(1 for r in ranks if r is not None and r <= 1)
    top3_hits = sum(1 for r in ranks if r is not None and r <= 3)

    return ArmScores(
        arm=arm,
        n_cases=n,
        ranks=ranks,
        top_k=top_k,
        mrr=mean_reciprocal_rank(ranks),
        top1_ci=wilson_interval(top1_hits, n),
        top3_ci=wilson_interval(top3_hits, n),
        distractor_rate=distractor_rate,
        n_distractor_cases=n_distractor_cases,
    )


def compare_arms(
    arms: dict[str, ArmScores],
    reference: str,
    k: int = 3,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Paired comparison of every arm against ``reference`` at cut-off ``k``.

    Args:
        arms: ``{arm_name: ArmScores}``, all scored on the same cases in the
            same order.
        reference: Name of the baseline arm to compare against.
        k: Cut-off at which correctness is defined for the comparison.
        seed: RNG seed for the paired bootstrap.

    Returns:
        ``{arm_name: {"delta_ci": ..., "mcnemar": ...}}`` for every arm other
        than ``reference``.

    Raises:
        KeyError: If ``reference`` is not present in ``arms``.
    """
    if reference not in arms:
        raise KeyError(
            f"Reference arm '{reference}' not among scored arms "
            f"{sorted(arms)}."
        )

    baseline = arms[reference].outcomes_at(k)
    out: dict[str, dict] = {}
    for name, scores in arms.items():
        if name == reference:
            continue
        candidate = scores.outcomes_at(k)
        out[name] = {
            "vs": reference,
            "k": k,
            "delta_ci": paired_bootstrap_delta_ci(candidate, baseline, seed=seed),
            "mcnemar": mcnemar_exact(candidate, baseline),
        }
    return out


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #
def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF.

    Implemented locally to keep this module free of a scipy dependency (the
    project deliberately has none — see requirements.txt). Uses the
    Beasley-Springer-Moro / Acklam rational approximation, accurate to about
    1.15e-9 in absolute value over the open unit interval, which is several
    orders of magnitude tighter than anything a 25-case interval needs.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}.")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
