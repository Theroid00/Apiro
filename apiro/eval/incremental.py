"""Longitudinal diagnosis sessions and MINT-style behavioral metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from apiro.eval.metrics import first_hit_rank
from apiro.parsing import detect_abstention


@dataclass
class IncrementalDiagnosisSession:
    """Accumulate evidence shards while retaining every diagnostic revision."""

    case_id: str
    resources: object
    max_depth: int = 6
    log_dir: object = None
    evidence: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def add_turn(self, shard: str, *, category: str = "unknown", is_lure: bool = False) -> dict:
        if not shard or not shard.strip():
            raise ValueError("evidence shard must not be empty")
        self.evidence.append(shard.strip())
        turn = len(self.evidence)
        from apiro.eval.live import evaluate_narrative_case
        result = evaluate_narrative_case(
            case_name=f"{self.case_id}_turn_{turn:02d}",
            narrative="\n".join(self.evidence), resources=self.resources,
            max_depth=self.max_depth, log_dir=self.log_dir,
            allow_abstention=True,
        )
        record = {
            "turn": turn, "category": category, "is_lure": is_lure,
            "evidence_shard": shard, **result,
            "abstained": {
                arm: detect_abstention("\n".join(predictions)) or not predictions
                for arm, predictions in result["predictions"].items()
            },
        }
        self.history.append(record)
        return record


def score_mint(cases: list[dict], matcher, arms=("apiro", "rag", "bare_llm")) -> dict:
    """Score commitment timing, revisions, final accuracy, and lure failures."""
    output = {}
    for arm in arms:
        first_commits, final_correct = [], []
        wrong_to_right = right_to_wrong = lure_flips = lure_opportunities = 0
        for case in cases:
            truth = case["ground_truth"]
            history = case["turns"]
            correctness = []
            committed = []
            for turn in history:
                predictions = turn["predictions"][arm]
                abstained = turn.get("abstained", {}).get(arm, not predictions)
                committed.append(not abstained)
                correctness.append(first_hit_rank(predictions, truth, matcher) is not None)
            first = next((i + 1 for i, value in enumerate(committed) if value), None)
            first_commits.append(first)
            final_correct.append(correctness[-1] if correctness else False)
            for before, after in zip(correctness, correctness[1:]):
                wrong_to_right += int(not before and after)
                right_to_wrong += int(before and not after)
            for index, turn in enumerate(history):
                if turn.get("is_lure") and index > 0:
                    lure_opportunities += 1
                    lure_flips += int(correctness[index - 1] and not correctness[index])
        committed_values = [value for value in first_commits if value is not None]
        output[arm] = {
            "n_cases": len(cases),
            "final_accuracy": sum(final_correct) / len(cases) if cases else 0.0,
            "coverage": len(committed_values) / len(cases) if cases else 0.0,
            "mean_first_commit_turn": sum(committed_values) / len(committed_values) if committed_values else None,
            "early_commitment_rate": sum(v <= 2 for v in committed_values) / len(committed_values) if committed_values else None,
            "wrong_to_correct_revisions": wrong_to_right,
            "correct_to_wrong_revisions": right_to_wrong,
            "lure_failure_rate": lure_flips / lure_opportunities if lure_opportunities else None,
            "n_lure_opportunities": lure_opportunities,
        }
    return output
