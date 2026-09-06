#!/usr/bin/env python3
"""Run MINT-style incremental evidence cases from a local JSON file.

The paper does not currently link a public dataset repository, so this runner
uses an explicit local interchange schema: a JSON list of objects containing
``case_id``, ``ground_truth`` and ``turns``. Each turn contains ``text``, an
optional clinical ``category``, and optional ``is_lure`` boolean.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apiro.eval.harness import build_real_components, make_matcher
from apiro.eval.incremental import IncrementalDiagnosisSession, score_mint
from apiro.eval.manifest import build_manifest, create_run_directory


def load_cases(path: Path, n: int, seed: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data if isinstance(data, list) else data.get("cases", [])
    for case in cases:
        if not {"case_id", "ground_truth", "turns"} <= case.keys() or not case["turns"]:
            raise ValueError("each MINT case needs case_id, ground_truth, and non-empty turns")
    random.Random(seed).shuffle(cases)
    return cases[:min(n, len(cases))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data" / "runs")
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args(argv)
    cases = load_cases(args.dataset_json, args.n, args.seed)
    if args.describe_only:
        print(f"MINT: {len(cases)} cases, {sum(len(c['turns']) for c in cases)} evidence turns")
        return 0

    components = build_real_components()
    manifest = build_manifest(
        benchmark="mint", dataset=str(args.dataset_json), revision="local",
        case_ids=[str(c["case_id"]) for c in cases],
        config={"seed": args.seed, "n": len(cases), "max_depth": args.max_depth,
                "model": components.resources.model, "policy": "evaluate-every-turn"},
    )
    run_dir = create_run_directory(manifest, args.runs_dir)
    results = []
    for case in cases:
        session = IncrementalDiagnosisSession(
            str(case["case_id"]), components.resources, args.max_depth, run_dir / "logs"
        )
        for turn in case["turns"]:
            session.add_turn(turn["text"], category=turn.get("category", "unknown"),
                             is_lure=bool(turn.get("is_lure")))
        results.append({"case_id": str(case["case_id"]), "ground_truth": case["ground_truth"],
                        "turns": session.history})
    payload = {"manifest": manifest, "scores": score_mint(results, make_matcher()), "case_results": results}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved MINT run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
