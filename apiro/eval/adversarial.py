"""Paired robustness metrics used by MedEinst and MedDistractQA."""

from __future__ import annotations

from collections import defaultdict

from apiro.eval.metrics import distractor_robustness, first_hit_rank, wilson_interval


ARMS = ("apiro", "rag", "bare_llm")


def score_medeinst(records: list[dict], matcher, arms=ARMS) -> dict:
    """Compute MedEinst's exact control-diagnosis retention trap endpoint."""
    grouped = defaultdict(dict)
    for record in records:
        grouped[str(record["case_id"])][record["case_type"]] = record
    pairs = [(v["control"], v["trap"]) for v in grouped.values() if {"control", "trap"} <= v.keys()]

    output = {}
    for arm in arms:
        control_correct, trap_correct, retained = [], [], []
        rank_changes = []
        for control, trap in pairs:
            control_preds = control["predictions"][arm]
            trap_preds = trap["predictions"][arm]
            ctl_rank = first_hit_rank(control_preds, control["ground_truth"], matcher)
            trap_rank = first_hit_rank(trap_preds, trap["ground_truth"], matcher)
            retained_rank = first_hit_rank(trap_preds, control["ground_truth"], matcher)
            control_correct.append(ctl_rank is not None)
            trap_correct.append(trap_rank is not None)
            retained.append(ctl_rank is not None and retained_rank is not None and trap_rank is None)
            rank_changes.append({
                "case_id": control["case_id"],
                "control_truth_rank_control": ctl_rank,
                "control_truth_rank_trap": retained_rank,
                "trap_truth_rank_trap": trap_rank,
            })
        eligible = sum(control_correct)
        trapped = sum(retained)
        both = sum(c and t for c, t in zip(control_correct, trap_correct))
        output[arm] = {
            "n_pairs": len(pairs),
            "n_control_correct": eligible,
            "n_bias_traps": trapped,
            "bias_trap_rate": trapped / eligible if eligible else None,
            "bias_trap_rate_ci": wilson_interval(trapped, eligible) if eligible else (0.0, 1.0),
            "control_accuracy": sum(control_correct) / len(pairs) if pairs else 0.0,
            "trap_accuracy": sum(trap_correct) / len(pairs) if pairs else 0.0,
            "pair_resilience": both / len(pairs) if pairs else 0.0,
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
