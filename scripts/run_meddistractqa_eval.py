#!/usr/bin/env python3
"""Run diagnosis-only clean/distracted pairs from MedDistractQA."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apiro.eval.adversarial import score_meddistract
from apiro.eval.harness import build_real_components, make_matcher
from apiro.eval.live import evaluate_narrative_case
from apiro.eval.manifest import build_manifest, create_run_directory

DATASET = "KrithikV/MedDistractQA"
REVISION = "main"


def clean_question(question: str, distraction: str) -> str:
    cleaned = question.replace(distraction, "").replace("  ", " ").strip()
    return cleaned


def diagnosis_pairs(rows: list[dict], n: int, seed: int) -> list[dict]:
    eligible = [r for r in rows if r.get("medical_competency") == "Patient Care: Diagnosis"]
    random.Random(seed).shuffle(eligible)
    pairs = []
    for index, row in enumerate(eligible[:min(n, len(eligible))]):
        choices = row.get("question_choices") or {}
        answer = str(row.get("correct_answer", ""))
        if answer not in choices:
            raise ValueError(f"MedDistractQA row has invalid answer key {answer!r}")
        distraction = row.get("distracting_sentence", "")
        case_id = str(row.get("id", index))
        common = {"case_id": case_id, "ground_truth": choices[answer], "choices": choices,
                  "distracting_sentence": distraction}
        pairs.extend([
            {**common, "condition": "clean", "narrative": clean_question(row["question"], distraction)},
            {**common, "condition": "distracted", "narrative": row["question"]},
        ])
    return pairs


def load_rows(dataset_json: Path | None = None) -> list[dict]:
    if dataset_json:
        data = json.loads(dataset_json.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data["rows"]
    from datasets import load_dataset
    return list(load_dataset(DATASET, split="train", revision=REVISION))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data" / "runs")
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be positive")
    selected = diagnosis_pairs(load_rows(args.dataset_json), args.n, args.seed)
    if args.describe_only:
        print(f"MedDistractQA: {len(selected)//2} diagnosis-only clean/distracted pairs")
        return 0

    components = build_real_components()
    manifest = build_manifest(
        benchmark="meddistractqa", dataset=DATASET, revision=REVISION,
        case_ids=[f"{r['case_id']}:{r['condition']}" for r in selected],
        config={"seed": args.seed, "n_pairs": len(selected)//2, "max_depth": args.max_depth,
                "subset": "Patient Care: Diagnosis", "n_diagnoses": 3,
                "model": components.resources.model},
    )
    run_dir = create_run_directory(manifest, args.runs_dir)
    records = []
    for row in selected:
        evaluated = evaluate_narrative_case(
            case_name=f"{row['case_id']}_{row['condition']}", narrative=row["narrative"],
            resources=components.resources, max_depth=args.max_depth, log_dir=run_dir / "logs",
        )
        records.append({k: row[k] for k in ("case_id", "condition", "ground_truth", "choices", "distracting_sentence")} | evaluated)
    payload = {"manifest": manifest, "scores": score_meddistract(records, make_matcher()), "case_results": records}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved MedDistractQA run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
