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


def print_summary(scores: dict, n_pairs: int, source: Path | None = None) -> None:
    """Print primary rank-1 MedEinst metrics and secondary top-3 accuracy."""
    width = 122
    print("\n" + "=" * width)
    print(f"       MEDEINST PAIRED ADVERSARIAL BENCHMARK (N = {n_pairs} pairs)")
    if source is not None:
        print(f"       Rescored from: {source}")
    print("=" * width)
    print(
        f"{'Arm':<14} {'Control@1':>10} {'Trap@1':>9} {'BTR':>9} "
        f"{'Trapped':>9} {'BTR 95% CI':>15} {'Pair@1':>9} "
        f"{'Control@3':>11} {'Trap@3':>9} {'Pair@3':>9}"
    )
    print("-" * width)
    for arm, metrics in scores.items():
        btr = metrics["bias_trap_rate"]
        btr_text = "N/A" if btr is None else f"{btr * 100:.1f}%"
        lo, hi = metrics["bias_trap_rate_ci"]
        ci_text = f"[{lo * 100:.1f}, {hi * 100:.1f}]"
        trapped_text = f"{metrics['n_bias_traps']}/{metrics['n_control_correct']}"
        print(
            f"{arm:<14} {metrics['control_accuracy'] * 100:>9.1f}%"
            f" {metrics['trap_accuracy'] * 100:>8.1f}% {btr_text:>9}"
            f" {trapped_text:>9} {ci_text:>15}"
            f" {metrics['pair_resilience'] * 100:>8.1f}%"
            f" {metrics['top3_control_accuracy'] * 100:>10.1f}%"
            f" {metrics['top3_trap_accuracy'] * 100:>8.1f}%"
            f" {metrics['top3_pair_resilience'] * 100:>8.1f}%"
        )
    print("=" * width + "\n")


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
    parser.add_argument(
        "--rescore-results",
        type=Path,
        help="Recompute metrics from a saved results.json without model calls.",
    )
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args(argv)
    if args.n_pairs < 1:
        parser.error("--n-pairs must be positive")

    if args.rescore_results is not None:
        with args.rescore_results.open(encoding="utf-8") as handle:
            prior = json.load(handle)
        records = prior.get("case_results")
        if not isinstance(records, list):
            parser.error("--rescore-results must contain a case_results list")
        scores = score_medeinst(records, make_matcher())
        n_pairs = len({str(row["case_id"]) for row in records})
        print_summary(scores, n_pairs, source=args.rescore_results)
        return 0

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
    scores = score_medeinst(records, matcher)
    payload = {"manifest": manifest, "scores": scores, "case_results": records}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved MedEinst run to {run_dir}")
    print_summary(scores, len(selected) // 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
