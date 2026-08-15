#!/usr/bin/env python3
"""
scripts/run_safety_calibration_eval.py

Safety / calibration evaluation harness for Apiro NIAH results.

Loads data/niah_eval_results.json, derives a per-case confidence for each arm,
and computes calibration + selective-prediction metrics via
apiro.eval.calibration.

IMPORTANT ASSUMPTIONS (see functions below, marked `# >>> ASSUMPTION`):
  * The mapping from Apiro traversal signals -> confidence is a heuristic and
    should be replaced with the project's real calibration model.
  * RAG / Bare-LLM expose no native confidence in the given schema. Deriving a
    "proxy" from the correctness label would leak the evaluation label and make
    ECE/AURC meaningless. This script therefore requires an explicit,
    label-free confidence source or a heuristic based only on the model OUTPUT.
    A leakage guard aborts if the confidence is suspiciously perfectly aligned
    with correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The interfaces you specified. These are imported, NOT reimplemented, because
# all reported numbers depend on their exact internals.
from apiro.eval.calibration import (  # noqa: E402
    compute_ece,
    compute_risk_coverage,
)

DEFAULT_INPUT = Path("data/niah_eval_results.json")
DEFAULT_OUTPUT = Path("data/calibration_eval_results.json")

ARMS = ("apiro", "rag", "bare_llm")
SELECTIVE_THRESHOLDS = [0.50, 0.65, 0.80]
REPORT_TAU = 0.65

# If confidence is >= this fraction "perfectly ordered" w.r.t. correctness,
# we assume the confidence was derived from the label and abort. Tune per arm.
LEAKAGE_ABORT_AUROC = 0.999


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input results not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cases = payload.get("case_results")
    if not isinstance(cases, list) or not cases:
        raise ValueError(
            f"{path} has no non-empty 'case_results' list; got "
            f"{type(cases).__name__}"
        )
    return cases


def _as_bool(value: Any) -> bool | None:
    """Extract a success flag; None if the arm produced nothing usable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and isinstance(value.get("success"), bool):
        return value["success"]
    return None


# --------------------------------------------------------------------------- #
# Confidence derivation
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def apiro_confidence(case: dict[str, Any]) -> float:
    """
    Derive an Apiro confidence in [0, 1] from traversal signals + top-candidate
    margin.

    # >>> ASSUMPTION (REPLACE WITH REAL MODEL) <<<
    The exact fields in `apiro.output` items were not specified, so the
    "top candidate margin" is looked up defensively under several likely keys
    ('margin', 'score_margin', 'delta'). If none is present, margin defaults to
    0.0 (i.e. no margin signal). The functional form below is a documented
    heuristic, not a fitted calibrator:

        base    = sigmoid( a - b * contradictions )        # more conflicts -> less confident
        margin  = clip01(top_margin)                       # candidate separation
        stop    = penalty for non-clean stop reasons       # budget/timeout -> less confident
        conf    = clip01( 0.5*base + 0.4*margin + stop )

    Coefficients (a, b, weights, penalties) are placeholders.
    """
    traversal = case.get("traversal") or {}
    contradictions = traversal.get("contradictions")
    contradictions = float(contradictions) if isinstance(contradictions, (int, float)) else 0.0
    stop_reason = str(traversal.get("stop_reason") or "").lower()

    # Top-candidate margin — defensive extraction from the (unspecified) output.
    top_margin = 0.0
    out = case.get("apiro", {}).get("output")
    if isinstance(out, list) and out and isinstance(out[0], dict):
        for key in ("margin", "score_margin", "delta", "confidence_margin"):
            if isinstance(out[0].get(key), (int, float)):
                top_margin = float(out[0][key])
                break

    # >>> ASSUMPTION: coefficients below are placeholders. <<<
    A, B = 2.0, 0.05  # contradictions ~55 -> sigmoid(2 - 2.75) ~ 0.32
    base = _sigmoid(A - B * contradictions)

    margin = _clip01(top_margin)

    clean_stops = {"answer_found", "converged", "resolved", "single_candidate"}
    if stop_reason in clean_stops:
        stop_bonus = 0.10
    elif stop_reason in {"exploration_budget", "timeout", "max_depth", "budget"}:
        stop_bonus = -0.10
    else:
        stop_bonus = 0.0

    conf = 0.5 * base + 0.4 * margin + stop_bonus + 0.1
    return _clip01(conf)


def output_length_proxy_confidence(output: Any) -> float:
    """
    A LABEL-FREE, deliberately weak confidence proxy for arms that expose only
    text output (RAG / Bare LLM). It depends ONLY on the produced output, never
    on correctness, so it does not leak the evaluation label.

    # >>> ASSUMPTION (REPLACE WITH REAL SIGNAL) <<<
    Uses hedging-language and non-empty-answer heuristics on the output string.
    This is a placeholder for a real token-logprob / verbalized-confidence
    signal, which the given schema does not contain.
    """
    if not isinstance(output, str) or not output.strip():
        return 0.05  # empty / no answer -> very low confidence

    text = output.lower()
    hedges = (
        "i don't know", "i do not know", "cannot", "can't determine",
        "not sure", "unclear", "no information", "unable to", "insufficient",
        "not mentioned", "not found",
    )
    if any(h in text for h in hedges):
        return 0.20

    # Length as a crude assertiveness proxy, squashed into a mild range.
    n_tokens = len(text.split())
    return _clip01(0.45 + 0.35 * _sigmoid((n_tokens - 20) / 20.0))


def rag_confidence(case: dict[str, Any]) -> float:
    return output_length_proxy_confidence(case.get("rag", {}).get("output"))


def bare_llm_confidence(case: dict[str, Any]) -> float:
    return output_length_proxy_confidence(case.get("bare_llm", {}).get("output"))


CONFIDENCE_FNS: dict[str, Callable[[dict[str, Any]], float]] = {
    "apiro": apiro_confidence,
    "rag": rag_confidence,
    "bare_llm": bare_llm_confidence,
}


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def _auroc(confidences: list[float], correctness: list[bool]) -> float | None:
    """Rank-based AUROC of confidence vs correctness (leakage sanity check)."""
    pos = [c for c, y in zip(confidences, correctness) if y]
    neg = [c for c, y in zip(confidences, correctness) if not y]
    if not pos or not neg:
        return None
    # Rank-sum (Mann–Whitney) formulation with tie handling.
    paired = sorted(
        [(c, 1) for c in pos] + [(c, 0) for c in neg], key=lambda t: t[0]
    )
    ranks: list[float] = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lbl) in zip(ranks, paired) if lbl == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def assert_no_label_leakage(arm: str, confidences: list[float],
                            correctness: list[bool]) -> float | None:
    auroc = _auroc(confidences, correctness)
    if auroc is not None and auroc >= LEAKAGE_ABORT_AUROC:
        raise RuntimeError(
            f"[LEAKAGE GUARD] Arm '{arm}' confidence perfectly separates "
            f"correct from incorrect (AUROC={auroc:.4f} >= "
            f"{LEAKAGE_ABORT_AUROC}). This strongly suggests the confidence "
            f"was derived from the correctness label, which invalidates ECE/"
            f"AURC. Provide a label-free confidence source."
        )
    return auroc


# --------------------------------------------------------------------------- #
# Selective metrics at fixed thresholds
# --------------------------------------------------------------------------- #
def selective_metrics_at(confidences: list[float], correctness: list[bool],
                         tau: float) -> dict[str, float]:
    n = len(confidences)
    kept = [(c, y) for c, y in zip(confidences, correctness) if c >= tau]
    n_kept = len(kept)
    abstention_rate = 1.0 - (n_kept / n) if n else 0.0
    coverage = n_kept / n if n else 0.0
    if n_kept:
        selective_acc = sum(1 for _, y in kept if y) / n_kept
    else:
        selective_acc = float("nan")
    return {
        "tau": tau,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_accuracy": selective_acc,
        "n_kept": n_kept,
        "n_total": n,
    }


# --------------------------------------------------------------------------- #
# Per-arm evaluation
# --------------------------------------------------------------------------- #
def evaluate_arm(arm: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    conf_fn = CONFIDENCE_FNS[arm]
    confidences: list[float] = []
    correctness: list[bool] = []
    skipped = 0

    for case in cases:
        success = _as_bool(case.get(arm))
        if success is None:
            skipped += 1
            continue
        correctness.append(bool(success))
        confidences.append(_clip01(float(conf_fn(case))))

    if not confidences:
        raise ValueError(f"Arm '{arm}' produced no usable cases.")

    auroc = assert_no_label_leakage(arm, confidences, correctness)

    ece_res = compute_ece(confidences, correctness, n_bins=10, strategy="uniform")
    rc_res = compute_risk_coverage(confidences, correctness, thresholds=None)

    forced_accuracy = sum(1 for y in correctness if y) / len(correctness)
    mean_conf = statistics.fmean(confidences)

    selective = {
        f"{tau:.2f}": selective_metrics_at(confidences, correctness, tau)
        for tau in SELECTIVE_THRESHOLDS
    }

    return {
        "arm": arm,
        "n_samples": len(correctness),
        "n_skipped": skipped,
        "forced_accuracy": forced_accuracy,
        "mean_confidence": mean_conf,
        "confidence_auroc": auroc,
        "ece": ece_res.get("ece"),
        "mce": ece_res.get("mce"),
        "brier_score": ece_res.get("brier_score"),
        "aurc": rc_res.get("aurc"),
        "optimal_aurc": rc_res.get("optimal_aurc"),
        "excess_risk": rc_res.get("excess_risk"),
        "selective": selective,
        "raw_ece": ece_res,
        "raw_risk_coverage": rc_res,
    }


# --------------------------------------------------------------------------- #
# ASCII reporting
# --------------------------------------------------------------------------- #
def _fmt(value: Any, fmt: str = "{:.4f}") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def render_table(results: dict[str, dict[str, Any]], tau: float) -> str:
    tau_key = f"{tau:.2f}"
    arm_labels = {"apiro": "Apiro", "rag": "RAG", "bare_llm": "Bare LLM"}
    ordered = [a for a in ARMS if a in results]

    rows: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("Forced Accuracy", lambda r: _fmt(r["forced_accuracy"])),
        ("ECE", lambda r: _fmt(r["ece"])),
        ("Brier Score", lambda r: _fmt(r["brier_score"])),
        ("AURC", lambda r: _fmt(r["aurc"])),
        (f"Selective Acc (tau={tau:.2f})",
         lambda r: _fmt(r["selective"][tau_key]["selective_accuracy"])),
        (f"Abstention Rate (tau={tau:.2f})",
         lambda r: _fmt(r["selective"][tau_key]["abstention_rate"])),
    ]

    metric_w = max(len(name) for name, _ in rows)
    metric_w = max(metric_w, len("Metric"))
    col_w = max(10, *(len(arm_labels[a]) for a in ordered))

    def sep(left: str, mid: str, right: str, fill: str = "─") -> str:
        parts = [fill * (metric_w + 2)] + [fill * (col_w + 2) for _ in ordered]
        return left + mid.join(parts) + right

    def row(cells: list[str]) -> str:
        head = f" {cells[0]:<{metric_w}} "
        body = "│".join(f" {c:>{col_w}} " for c in cells[1:])
        return "│" + head + "│" + body + "│"

    lines: list[str] = []
    title = "  SAFETY / CALIBRATION EVALUATION — ARM COMPARISON  "
    lines.append(sep("┌", "┬", "┐"))
    lines.append(row(["Metric"] + [arm_labels[a] for a in ordered]))
    lines.append(sep("├", "┼", "┤"))
    for name, getter in rows:
        lines.append(row([name] + [getter(results[a]) for a in ordered]))
    lines.append(sep("└", "┴", "┘"))

    header = "\n" + title.center(len(lines[0])) + "\n"
    return header + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run safety / calibration evaluation over NIAH results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Input results JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output results JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--tau", type=float, default=REPORT_TAU,
                        help=f"Reporting threshold (default: {REPORT_TAU})")
    args = parser.parse_args(argv)

    try:
        cases = load_results(args.input)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR loading input: {exc}", file=sys.stderr)
        return 2

    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        try:
            results[arm] = evaluate_arm(arm, cases)
        except RuntimeError as exc:  # leakage guard
            print(f"ABORTING for arm '{arm}': {exc}", file=sys.stderr)
            return 3
        except ValueError as exc:
            print(f"WARNING: skipping arm '{arm}': {exc}", file=sys.stderr)

    if not results:
        print("ERROR: no arm produced usable results.", file=sys.stderr)
        return 4

    print(render_table(results, args.tau))

    payload = {
        "meta": {
            "input_file": str(args.input),
            "n_cases": len(cases),
            "arms": list(results.keys()),
            "selective_thresholds": SELECTIVE_THRESHOLDS,
            "report_tau": args.tau,
            "ece_config": {"n_bins": 10, "strategy": "uniform"},
            "notes": (
                "Apiro confidence and RAG/Bare-LLM proxy confidence are "
                "heuristic placeholders (see script). Replace with the real "
                "calibration model / native confidence signal before publishing."
            ),
        },
        "arms": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"Saved complete results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
