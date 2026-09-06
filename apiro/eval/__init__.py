"""
apiro.eval — evaluation, scoring and calibration.

Three layers, each usable on its own:

  evaluator   — is a predicted diagnosis clinically equivalent to the ground
                truth? A cascade of exact match, curated concept
                normalization, an optional LLM judge, and an optional
                embedding fallback.
  metrics     — rank-aware differential-diagnosis metrics (top-k, MRR,
                distractor-selection rate) and the paired significance
                machinery (Wilson / bootstrap intervals, exact McNemar) that
                accuracy deltas on small case sets require.
  calibration — Expected Calibration Error, Brier score, risk-coverage /
                AURC, and the selective-abstention decision wrapper.

Nothing here imports Ollama, ChromaDB, torch or transformers, so the whole
package is importable and testable offline.
"""

from apiro.eval.calibration import (
    CalibratedDecision,
    compute_ece,
    compute_risk_coverage,
)
from apiro.eval.metrics import (
    ArmScores,
    compare_arms,
    distractor_selection_rate,
    first_hit_rank,
    mcnemar_exact,
    mean_reciprocal_rank,
    paired_bootstrap_delta_ci,
    score_arm,
    top_k_accuracy,
    wilson_interval,
)

__all__ = [
    # calibration
    "CalibratedDecision",
    "compute_ece",
    "compute_risk_coverage",
    # metrics
    "ArmScores",
    "compare_arms",
    "distractor_selection_rate",
    "first_hit_rank",
    "mcnemar_exact",
    "mean_reciprocal_rank",
    "paired_bootstrap_delta_ci",
    "score_arm",
    "top_k_accuracy",
    "wilson_interval",
]
