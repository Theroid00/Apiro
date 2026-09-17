#!/usr/bin/env python3
"""Build Apiro's offline sparse medical concept graph from corpus records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apiro.config import CONCEPT_GRAPH_PATH
from apiro.reasoning.concept_graph import MedicalConceptGraph


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("records", "chunks", "rows"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"{path} does not contain a record list")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=CONCEPT_GRAPH_PATH)
    args = parser.parse_args(argv)

    graph = MedicalConceptGraph()
    count = 0
    for path in args.inputs:
        records = load_records(path)
        graph.ingest_records(records)
        count += len(records)
    graph.save(args.output)
    print(
        f"Saved {len(graph.nodes)} nodes from {count} records to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
