#!/usr/bin/env python3
"""Fit investigator action-value weights from JSONL feedback records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apiro.config import ACTION_POLICY_PATH
from apiro.reasoning.action_policy import ActionValuePolicy


def load_examples(path: Path) -> list[tuple[dict[str, float], float]]:
    examples = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        features = row.get("features")
        if not isinstance(features, dict) or "reward" not in row:
            raise ValueError(
                f"{path}:{line_number} requires features and reward"
            )
        examples.append((features, float(row["reward"])))
    return examples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--output", type=Path, default=ACTION_POLICY_PATH)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args(argv)

    policy = ActionValuePolicy()
    examples = load_examples(args.feedback)
    policy.fit(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    policy.save(args.output)
    print(f"Saved policy trained on {len(examples)} actions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
