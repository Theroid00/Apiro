#!/usr/bin/env python3
"""Run Apiro, RAG and bare-LLM arms on paired MedEinst cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apiro.eval.adversarial import score_medeinst
from apiro.eval.harness import build_real_components, make_matcher
from apiro.eval.live import evaluate_narrative_case
from apiro.eval.manifest import build_manifest, create_run_directory

DATASET = "zhui711/MedEinst"
REVISION = "354f4b5"


def load_rows(split: str, dataset_json: Path | None = None) -> list[dict]:
    if dataset_json:
        with dataset_json.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    from datasets import load_dataset
    return list(load_dataset(DATASET, split=split, revision=REVISION))


def select_pairs(rows: list[dict], n_pairs: int, seed: int) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = {}
    required = {"case_id", "case_type", "narrative", "ground_truth"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"MedEinst row is missing {sorted(missing)}")
        grouped.setdefault(str(row["case_id"]), {})[row["case_type"]] = row
    ids = sorted(case_id for case_id, pair in grouped.items() if {"control", "trap"} <= pair.keys())
    if not ids:
        raise ValueError("MedEinst contains no complete control/trap pairs")
    random.Random(seed).shuffle(ids)
    chosen = ids[:min(n_pairs, len(ids))]
    return [grouped[case_id][kind] for case_id in chosen for kind in ("control", "trap")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=60)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data" / "runs")
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args(argv)
    if args.n_pairs < 1:
        parser.error("--n-pairs must be positive")

    selected = select_pairs(load_rows(args.split, args.dataset_json), args.n_pairs, args.seed)
    if args.describe_only:
        print(f"MedEinst: {len(selected) // 2} complete pairs ({len(selected)} rows)")
        return 0

    components = build_real_components()
    manifest = build_manifest(
        benchmark="medeinst", dataset=DATASET, revision=REVISION,
        case_ids=[f"{r['case_id']}:{r['case_type']}" for r in selected],
        config={"split": args.split, "seed": args.seed, "n_pairs": len(selected)//2,
                "max_depth": args.max_depth, "n_diagnoses": 3,
                "model": components.resources.model},
    )
    run_dir = create_run_directory(manifest, args.runs_dir)
    records = []
    for row in selected:
        evaluated = evaluate_narrative_case(
            case_name=f"{row['case_id']}_{row['case_type']}",
            narrative=row["narrative"], resources=components.resources,
            max_depth=args.max_depth, log_dir=run_dir / "logs",
        )
        records.append({
            "case_id": str(row["case_id"]), "case_type": row["case_type"],
            "ground_truth": row["ground_truth"], **evaluated,
        })
    matcher = make_matcher()
    payload = {"manifest": manifest, "scores": score_medeinst(records, matcher), "case_results": records}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved MedEinst run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
