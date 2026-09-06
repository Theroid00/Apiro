"""
apiro/eval/calibration.py

Pillar 3: Safety, Calibration & Selective Abstention.

This module provides calibration metrics (ECE, MCE, Brier score), reliability
diagram data, selective prediction / risk-coverage analysis (coverage, selective
risk, AURC, excess risk), and a calibrated decision wrapper used to represent
a model decision that may abstain when confidence is insufficient.

Design goals:
    * Zero external dependencies beyond the Python standard library and numpy.
    * Numerically robust and defensive against degenerate inputs.
    * Pure functions where possible; small, testable, composable pieces.

References:
    - Guo et al., "On Calibration of Modern Neural Networks" (ECE/MCE).
    - Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks"
      (risk-coverage, AURC).
    - Brier, "Verification of Forecasts Expressed in Terms of Probability".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "compute_ece",
    "compute_risk_coverage",
    "CalibratedDecision",
]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _validate_pairs(
    confidences: list[float],
    correctness: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce (confidence, correctness) inputs into arrays.

    Args:
        confidences: Sequence of predicted confidences (each in ``[0, 1]``).
        correctness: Sequence of booleans indicating whether each prediction
            was correct.

    Returns:
        A tuple ``(conf, corr)`` of ``float64`` numpy arrays of equal length.
        ``corr`` holds values in ``{0.0, 1.0}``.

    Raises:
        ValueError: If lengths mismatch or confidences fall outside ``[0, 1]``.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correctness, dtype=np.float64).ravel()

    if conf.shape[0] != corr.shape[0]:
        raise ValueError(
            f"Length mismatch: confidences ({conf.shape[0]}) != "
            f"correctness ({corr.shape[0]})."
        )

    if conf.size and (np.any(conf < 0.0) or np.any(conf > 1.0)):
        raise ValueError("All confidences must lie within the range [0, 1].")

    if np.any(~np.isfinite(conf)):
        raise ValueError("Confidences must be finite (no NaN or inf).")

    # Normalize correctness to a strict {0.0, 1.0} indicator.
    corr = (corr != 0.0).astype(np.float64)

    return conf, corr


def _make_bin_edges(
    conf: np.ndarray,
    n_bins: int,
    strategy: str,
) -> np.ndarray:
    """Construct monotonically increasing bin edges.

    Args:
        conf: Array of confidence values.
        n_bins: Number of bins (must be >= 1).
        strategy: Either ``"uniform"`` (equal-width bins on ``[0, 1]``) or
            ``"quantile"`` (equal-frequency bins based on the empirical
            distribution of ``conf``).

    Returns:
        An array of ``n_bins + 1`` strictly non-decreasing bin edges. The first
        edge is ``0.0`` and the last edge is ``1.0``.

    Raises:
        ValueError: If ``n_bins < 1`` or ``strategy`` is unknown.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)

    if strategy == "quantile":
        if conf.size == 0:
            return np.linspace(0.0, 1.0, n_bins + 1)
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(conf, quantiles)
        # Anchor the outer edges to the full probability range so that all
        # values are captured regardless of the observed distribution.
        edges[0] = 0.0
        edges[-1] = 1.0
        # Deduplicate collapsed edges (can happen with heavy point masses)
        # while preserving monotonicity.
        edges = np.maximum.accumulate(edges)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
        return edges

    raise ValueError(
        f"Unknown binning strategy '{strategy}'. "
        "Expected 'uniform' or 'quantile'."
    )


def _assign_bins(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each confidence to a bin index in ``[0, len(edges) - 2]``.

    Values are assigned to the bin ``(edges[i], edges[i+1]]`` with the leftmost
    bin also including its lower edge (so ``0.0`` is captured).

    Args:
        conf: Array of confidence values.
        edges: Monotonic bin edges of length ``n_bins + 1``.

    Returns:
        Integer array of bin indices, one per confidence value.
    """
    n_bins = edges.shape[0] - 1
    if conf.size == 0:
        return np.empty(0, dtype=np.int64)

    # np.digitize with right=True gives bins in [1, n_bins] for interior/upper
    # values; subtract 1 for zero-based indexing, then clamp to valid range so
    # that the extreme lower edge (0.0) lands in bin 0.
    idx = np.digitize(conf, edges[1:-1], right=True)
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


# --------------------------------------------------------------------------- #
# 1. Expected Calibration Error (ECE) & related metrics
# --------------------------------------------------------------------------- #
def compute_ece(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Compute calibration metrics and reliability-diagram bin data.

    Computes:
        * **ECE** (Expected Calibration Error): the weighted average absolute
          gap between per-bin confidence and per-bin accuracy.
        * **MCE** (Maximum Calibration Error): the maximum such gap across bins.
        * **Brier score**: ``mean((confidence - correct)^2)``.

    Also returns per-bin arrays suitable for plotting a reliability diagram.

    Args:
        confidences: Predicted confidences, each in ``[0, 1]``.
        correctness: Booleans indicating whether each prediction was correct.
        n_bins: Number of bins used to aggregate confidences. Default ``10``.
        strategy: Binning strategy, ``"uniform"`` or ``"quantile"``.
            Default ``"uniform"``.

    Returns:
        A dictionary with keys:
            * ``"ece"`` (float): Expected Calibration Error.
            * ``"mce"`` (float): Maximum Calibration Error.
            * ``"brier_score"`` (float): Mean squared calibration error.
            * ``"n_samples"`` (int): Number of samples used.
            * ``"n_bins"`` (int): Number of realized bins.
            * ``"strategy"`` (str): Binning strategy used.
            * ``"bin_edges"`` (list[float]): Bin edges (length ``n_bins + 1``).
            * ``"bin_confs"`` (list[float]): Mean confidence per bin.
            * ``"bin_accs"`` (list[float]): Mean accuracy per bin.
            * ``"bin_counts"`` (list[int]): Sample count per bin.

    Raises:
        ValueError: On input length mismatch, out-of-range confidences,
            or invalid ``n_bins`` / ``strategy``.

    Notes:
        Empty bins contribute zero to ECE/MCE and are reported with
        ``bin_conf = bin_acc = 0.0`` and ``bin_count = 0``. If no samples are
        provided, all metrics are ``0.0`` (or NaN-safe defaults).
    """
    conf, corr = _validate_pairs(confidences, correctness)
    edges = _make_bin_edges(conf, n_bins, strategy)
    realized_bins = edges.shape[0] - 1

    bin_confs = np.zeros(realized_bins, dtype=np.float64)
    bin_accs = np.zeros(realized_bins, dtype=np.float64)
    bin_counts = np.zeros(realized_bins, dtype=np.int64)

    n_samples = conf.shape[0]

    if n_samples == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "n_samples": 0,
            "n_bins": realized_bins,
            "strategy": strategy,
            "bin_edges": edges.tolist(),
            "bin_confs": bin_confs.tolist(),
            "bin_accs": bin_accs.tolist(),
            "bin_counts": bin_counts.tolist(),
        }

    assignments = _assign_bins(conf, edges)

    ece = 0.0
    mce = 0.0
    for b in range(realized_bins):
        mask = assignments == b
        count = int(np.count_nonzero(mask))
        bin_counts[b] = count
        if count == 0:
            continue
        mean_conf = float(np.mean(conf[mask]))
        mean_acc = float(np.mean(corr[mask]))
        bin_confs[b] = mean_conf
        bin_accs[b] = mean_acc

        gap = abs(mean_conf - mean_acc)
        ece += (count / n_samples) * gap
        mce = max(mce, gap)

    brier_score = float(np.mean((conf - corr) ** 2))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "n_samples": int(n_samples),
        "n_bins": int(realized_bins),
        "strategy": strategy,
        "bin_edges": edges.tolist(),
        "bin_confs": bin_confs.tolist(),
        "bin_accs": bin_accs.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


# --------------------------------------------------------------------------- #
# 2. Selective prediction & risk-coverage analysis
# --------------------------------------------------------------------------- #
def compute_risk_coverage(
    confidences: list[float],
    correctness: list[bool],
    thresholds: Optional[list[float]] = None,
) -> dict:
    """Compute the risk-coverage curve, AURC, and excess risk.

    For each confidence threshold ``t``, a prediction is *accepted* (not
    abstained) iff its confidence is ``>= t``. On the accepted subset:

        * **coverage(t)** = (# accepted) / (# total)
        * **selective_risk(t)** = (# errors among accepted) / (# accepted)

    Summary statistics:
        * **AURC** (Area Under the Risk-Coverage curve): computed by
          trapezoidal integration of selective risk with respect to coverage,
          normalized by the covered range. Lower is better.
        * **Excess risk**: AURC minus the AURC of a hypothetical *optimal*
          ranker that always accepts correct predictions before incorrect
          ones. Lower is better; ``0.0`` indicates optimal ranking.

    Args:
        confidences: Predicted confidences, each in ``[0, 1]``.
        correctness: Booleans indicating whether each prediction was correct.
        thresholds: Optional explicit thresholds to evaluate. If ``None``,
            thresholds are derived from the unique observed confidences
            (plus ``0.0``), yielding an exact curve.

    Returns:
        A dictionary with keys:
            * ``"thresholds"`` (list[float]): Thresholds evaluated (ascending).
            * ``"coverages"`` (list[float]): Coverage at each threshold.
            * ``"selective_risks"`` (list[float]): Selective risk at each
              threshold (``0.0`` where coverage is zero).
            * ``"aurc"`` (float): Area under the risk-coverage curve.
            * ``"optimal_aurc"`` (float): AURC of the oracle ranker.
            * ``"excess_risk"`` (float): ``aurc - optimal_aurc`` (clamped >= 0).
            * ``"base_error_rate"`` (float): Error rate at full coverage.
            * ``"n_samples"`` (int): Number of samples used.

    Raises:
        ValueError: On input length mismatch or out-of-range confidences.
    """
    conf, corr = _validate_pairs(confidences, correctness)
    n_samples = conf.shape[0]

    if n_samples == 0:
        return {
            "thresholds": [],
            "coverages": [],
            "selective_risks": [],
            "aurc": 0.0,
            "optimal_aurc": 0.0,
            "excess_risk": 0.0,
            "base_error_rate": 0.0,
            "n_samples": 0,
        }

    errors = 1.0 - corr  # 1.0 where incorrect, 0.0 where correct
    base_error_rate = float(np.mean(errors))

    # Build the set of thresholds to evaluate.
    if thresholds is None:
        thr = np.unique(conf)
        # Include 0.0 so that full coverage (accept everything) is represented.
        thr = np.unique(np.concatenate(([0.0], thr)))
    else:
        thr = np.unique(np.asarray(thresholds, dtype=np.float64).ravel())
        if thr.size == 0:
            thr = np.array([0.0])

    coverages = np.empty(thr.shape[0], dtype=np.float64)
    selective_risks = np.empty(thr.shape[0], dtype=np.float64)

    for i, t in enumerate(thr):
        accepted = conf >= t
        n_accepted = int(np.count_nonzero(accepted))
        coverages[i] = n_accepted / n_samples
        if n_accepted == 0:
            selective_risks[i] = 0.0
        else:
            selective_risks[i] = float(np.sum(errors[accepted]) / n_accepted)

    # --- AURC via the ideal (confidence-sorted) selective curve. ------------
    # Sort by descending confidence: accepting the k highest-confidence items.
    aurc = _aurc_from_ranking(conf, errors, ascending_by_risk=False)

    # --- Optimal (oracle) AURC: rank all correct before all incorrect. ------
    optimal_aurc = _optimal_aurc(errors)

    excess_risk = max(0.0, aurc - optimal_aurc)

    # Present the curve in ascending-threshold order for readability.
    order = np.argsort(thr)
    thr_sorted = thr[order]
    cov_sorted = coverages[order]
    risk_sorted = selective_risks[order]

    return {
        "thresholds": thr_sorted.tolist(),
        "coverages": cov_sorted.tolist(),
        "selective_risks": risk_sorted.tolist(),
        "aurc": float(aurc),
        "optimal_aurc": float(optimal_aurc),
        "excess_risk": float(excess_risk),
        "base_error_rate": base_error_rate,
        "n_samples": int(n_samples),
    }


def _integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if fn is not None:
        return float(fn(y, x))
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1])))


def _aurc_from_ranking(
    confidences: np.ndarray,
    errors: np.ndarray,
    ascending_by_risk: bool = False,
) -> float:
    n = confidences.shape[0]
    if n == 0:
        return 0.0

    if ascending_by_risk:
        order = np.argsort(confidences)
    else:
        order = np.argsort(-confidences)

    sorted_confidences = confidences[order]
    sorted_errors = errors[order]

    # A threshold cannot distinguish examples with identical confidence.
    # Emit one point only after each complete tie block, making the metric
    # invariant to input order within that block.
    block_ends = np.flatnonzero(
        np.r_[sorted_confidences[1:] != sorted_confidences[:-1], True]
    )
    cum_errors = np.cumsum(sorted_errors)[block_ends]
    k = (block_ends + 1).astype(np.float64)

    coverage = k / n
    selective_risk = cum_errors / k

    x = np.concatenate(([0.0], coverage))
    y = np.concatenate(([0.0], selective_risk))
    return _integrate_trapz(y, x)


def _optimal_aurc(errors: np.ndarray) -> float:
    n = errors.shape[0]
    if n == 0:
        return 0.0

    sorted_errors = np.sort(errors)
    cum_errors = np.cumsum(sorted_errors)
    k = np.arange(1, n + 1, dtype=np.float64)

    coverage = k / n
    selective_risk = cum_errors / k

    x = np.concatenate(([0.0], coverage))
    y = np.concatenate(([0.0], selective_risk))
    return _integrate_trapz(y, x)


# --------------------------------------------------------------------------- #
# 3. Calibrated decision wrapper
# --------------------------------------------------------------------------- #
@dataclass
class CalibratedDecision:
    """A calibrated model decision that may abstain under low confidence.

    This wrapper standardizes how Apiro represents a decision output alongside
    its calibrated confidence and any selective-abstention metadata. It is the
    canonical return type for decision-making components that participate in
    Pillar 3 safety guarantees.

    Attributes:
        differential: Ranked candidate hypotheses/actions. Each entry is a
            free-form ``dict`` (e.g. ``{"label": ..., "score": ...}``).
        confidence: Calibrated confidence in the top decision, in ``[0, 1]``.
        abstained: Whether the system declined to commit to a decision.
        abstention_reason: Human-readable reason for abstention, or ``None``
            when a decision was made.
        unresolved_conflicts: Identifiers/descriptions of conflicts that could
            not be resolved and may warrant human review.
    """

    differential: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    abstention_reason: Optional[str] = None
    unresolved_conflicts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and normalize field values.

        Raises:
            ValueError: If ``confidence`` is outside ``[0, 1]``, or if an
                abstention is asserted without a reason, or a non-abstention
                carries an abstention reason.
        """
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}."
            )
        self.confidence = float(self.confidence)

        if self.abstained and not self.abstention_reason:
            raise ValueError(
                "abstention_reason must be provided when abstained is True."
            )
        if not self.abstained and self.abstention_reason is not None:
            raise ValueError(
                "abstention_reason must be None when abstained is False."
            )

    @classmethod
    def decide(
        cls,
        differential: list[dict],
        confidence: float,
        threshold: float,
        unresolved_conflicts: Optional[list[str]] = None,
        low_confidence_reason: str = "confidence below abstention threshold",
    ) -> "CalibratedDecision":
        """Construct a decision, abstaining when confidence is insufficient.

        The system abstains if the calibrated ``confidence`` is strictly below
        ``threshold`` or if there are unresolved conflicts.

        Args:
            differential: Ranked candidate hypotheses/actions.
            confidence: Calibrated confidence in ``[0, 1]``.
            threshold: Minimum confidence required to commit to a decision.
            unresolved_conflicts: Conflicts that could not be resolved.
            low_confidence_reason: Reason text used when abstaining due to low
                confidence.

        Returns:
            A ``CalibratedDecision`` with ``abstained`` set appropriately.

        Raises:
            ValueError: If ``confidence`` or ``threshold`` is outside ``[0, 1]``.
        """
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold!r}."
            )

        conflicts = list(unresolved_conflicts or [])
        confidence = float(confidence)

        abstain = confidence < float(threshold) or bool(conflicts)
        reason: Optional[str] = None
        if abstain:
            if conflicts and confidence < float(threshold):
                reason = (
                    f"{low_confidence_reason}; "
                    f"{len(conflicts)} unresolved conflict(s)"
                )
            elif conflicts:
                reason = f"{len(conflicts)} unresolved conflict(s)"
            else:
                reason = low_confidence_reason

        return cls(
            differential=list(differential),
            confidence=confidence,
            abstained=abstain,
            abstention_reason=reason,
            unresolved_conflicts=conflicts,
        )

    @property
    def top(self) -> Optional[dict]:
        """Return the highest-ranked differential entry, if any.

        Returns:
            The first entry of ``differential`` when a decision was made and
            the differential is non-empty; otherwise ``None``.
        """
        if self.abstained or not self.differential:
            return None
        return self.differential[0]

    def to_dict(self) -> dict:
        """Serialize the decision to a plain dictionary.

        Returns:
            A JSON-serializable dictionary representation of the decision.
        """
        return {
            "differential": list(self.differential),
            "confidence": self.confidence,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "unresolved_conflicts": list(self.unresolved_conflicts),
        }
