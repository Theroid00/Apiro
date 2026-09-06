"""Paired robustness metrics used by MedEinst and MedDistractQA."""

from __future__ import annotations

from collections import defaultdict

from apiro.eval.metrics import distractor_robustness, first_hit_rank, wilson_interval


ARMS = ("apiro", "rag", "bare_llm")


def score_medeinst(records: list[dict], matcher, arms=ARMS) -> dict:
    """Compute MedEinst's rank-1 control-diagnosis retention endpoint.

    MedEinst defines an Einstellung trap as ``f(control) == control_truth``
    and ``f(trap) == control_truth``. Since ``f`` is one diagnosis, the
    primary endpoint must use the first prediction. Top-3 measures are kept as
    secondary Apiro differential-quality diagnostics.
    """
    grouped = defaultdict(dict)
    for record in records:
        grouped[str(record["case_id"])][record["case_type"]] = record
    pairs = [(v["control"], v["trap"]) for v in grouped.values() if {"control", "trap"} <= v.keys()]

    output = {}
    for arm in arms:
        control_correct, trap_correct, retained = [], [], []
        control_top3, trap_top3 = [], []
        rank_changes = []
        for control, trap in pairs:
            control_preds = control["predictions"][arm]
            trap_preds = trap["predictions"][arm]
            ctl_rank = first_hit_rank(control_preds, control["ground_truth"], matcher)
            trap_rank = first_hit_rank(trap_preds, trap["ground_truth"], matcher)
            retained_rank = first_hit_rank(trap_preds, control["ground_truth"], matcher)
            ctl_top1 = bool(control_preds) and matcher(
                control_preds[0], control["ground_truth"]
            )
            trap_top1 = bool(trap_preds) and matcher(
                trap_preds[0], trap["ground_truth"]
            )
            retained_top1 = bool(trap_preds) and matcher(
                trap_preds[0], control["ground_truth"]
            )
            control_correct.append(ctl_top1)
            trap_correct.append(trap_top1)
            retained.append(ctl_top1 and retained_top1)
            control_top3.append(ctl_rank is not None)
            trap_top3.append(trap_rank is not None)
            rank_changes.append({
                "case_id": control["case_id"],
                "control_truth_rank_control": ctl_rank,
                "control_truth_rank_trap": retained_rank,
                "trap_truth_rank_trap": trap_rank,
            })
        eligible = sum(control_correct)
        trapped = sum(retained)
        both = sum(c and t for c, t in zip(control_correct, trap_correct))
        trap_errors = sum(c and not t for c, t in zip(control_correct, trap_correct))
        both_top3 = sum(c and t for c, t in zip(control_top3, trap_top3))
        output[arm] = {
            "n_pairs": len(pairs),
            "n_control_correct": eligible,
            "n_bias_traps": trapped,
            "bias_trap_rate": trapped / eligible if eligible else None,
            "bias_trap_rate_ci": wilson_interval(trapped, eligible) if eligible else (0.0, 1.0),
            "control_accuracy": sum(control_correct) / len(pairs) if pairs else 0.0,
            "trap_accuracy": sum(trap_correct) / len(pairs) if pairs else 0.0,
            "pair_resilience": both / len(pairs) if pairs else 0.0,
            "conditional_trap_error_rate": trap_errors / eligible if eligible else None,
            "top3_control_accuracy": sum(control_top3) / len(pairs) if pairs else 0.0,
            "top3_trap_accuracy": sum(trap_top3) / len(pairs) if pairs else 0.0,
            "top3_pair_resilience": both_top3 / len(pairs) if pairs else 0.0,
            "rank_transitions": rank_changes,
        }
    return output


def score_meddistract(records: list[dict], matcher, arms=ARMS) -> dict:
    """Score clean/distracted accuracy retention and prediction changes."""
    grouped = defaultdict(dict)
    for record in records:
        grouped[str(record["case_id"])][record["condition"]] = record
    pairs = [(v["clean"], v["distracted"]) for v in grouped.values() if {"clean", "distracted"} <= v.keys()]
    output = {}
    for arm in arms:
        clean, distracted, flips = [], [], 0
        for base, noisy in pairs:
            base_preds = base["predictions"][arm]
            noisy_preds = noisy["predictions"][arm]
            clean.append(first_hit_rank(base_preds, base["ground_truth"], matcher) is not None)
            distracted.append(first_hit_rank(noisy_preds, noisy["ground_truth"], matcher) is not None)
            base_top = base_preds[0] if base_preds else ""
            noisy_top = noisy_preds[0] if noisy_preds else ""
            flips += int(bool(base_top or noisy_top) and not matcher(base_top, noisy_top))
        metrics = distractor_robustness(clean, distracted)
        metrics["top1_flip_rate"] = flips / len(pairs) if pairs else 0.0
        output[arm] = metrics
    return output
