#!/usr/bin/env python3
"""
scripts/investigate.py
======================
CLI runner for Apiro. The default simplified engine extracts bounded facts,
retrieves evidence once, and makes one structured reasoning call. The legacy
entropy-first traversal remains selectable for comparison.

Usage:
  python scripts/investigate.py --findings "49yo female, dyspnea, history of breast cancer"
  python scripts/investigate.py -f "..." --output data/graph_run.json
"""
import argparse
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("investigate")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def build_components():
    from apiro.application.runtime import RuntimeSetupError, build_runtime_resources
    try:
        resources = build_runtime_resources()
    except RuntimeSetupError as exc:
        print(f"\n[-] {exc}")
        sys.exit(1)
    return resources.create_service(), resources.doc_count

def print_report(result, elapsed: float) -> None:
    print("\n+" + "-" * 58 + "+")
    print("|" + "    APIRO DIFFERENTIAL DIAGNOSIS REPORT".center(58) + "|")
    print("+" + "-" * 58 + "+")

    print(f"\n  Time taken:             {elapsed:.1f} seconds")
    print(f"  Reasoning mode:         {getattr(result, 'mode', 'legacy')}")
    if getattr(result, "rounds", 0):
        print(f"  Investigation rounds:   {result.rounds}")
        print(f"  Retrievals:             {result.retrieval_count}")
        print(f"  Reasoning calls:         {result.reasoning_call_count}")
    if result.stop_reason:
        print(f"  Stop reason:            {result.stop_reason}")

    # NOTE: `result` is a TraversalResult. It has never carried
    # `patient_context` or `ranked_hypotheses` — those belong to the
    # hypothesis-testing engine that was purged in commit 8a001c7 — so this
    # report raised AttributeError at the end of every CLI run.
    print("\n  [ GRAPH ]")
    print(f"  nodes={result.total_nodes}  edges={result.total_edges}  "
          f"rabbit_holes={result.rabbit_hole_count}  "
          f"contradictions={result.contradiction_count}")

    print("\n  [ TOP DIFFERENTIAL DIAGNOSES ]")
    if not result.synthesis:
        print("  No viable hypotheses generated.")
    else:
        for i, dx in enumerate(result.synthesis, 1):
            print(f"  {i}. {dx}")

    graph = getattr(result, "graph", None)
    if graph is not None:
        anchors = [n for n in graph.nodes.values() if n.depth == 0]
        if anchors:
            print("\n  [ DETERMINISTIC ANCHORS ]")
            for n in anchors[:10]:
                print(f"  - {n.claim[:70]}")

    print("\n+" + "-" * 58 + "+\n")

def main():
    parser = argparse.ArgumentParser(
        description="Apiro AI Detective — free-text clinical findings → differential diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--findings", "-f",
        type=str,
        default=None,
        help="Free-text clinical findings. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=5,
        help="Max traversal depth (legacy mode only).",
    )
    parser.add_argument(
        "--mode", choices=("simple", "investigator", "legacy"), default=None,
        help="Reasoning engine. Defaults to APIRO_REASONING_MODE or simple.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write the belief graph as JSON.",
    )
    args = parser.parse_args()

    if args.findings:
        raw_findings = args.findings
    else:
        print("\n" + "=" * 60)
        print("    APIRO -- AI DIAGNOSTIC DETECTIVE")
        print("=" * 60)
        print("  Enter clinical findings (symptoms, labs, vitals, history).")
        print("  Press Enter twice when done.\n")
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        raw_findings = "\n".join(lines)

    if not raw_findings.strip():
        print("[-] No findings provided. Exiting.")
        sys.exit(1)

    print("\n[*] Initialising Apiro components...")
    service, doc_count = build_components()
    print(f"[+] Components ready. Corpus: {doc_count:,} documents.\n")

    print(f"\n[*] Apiro is investigating...")
    t0 = time.time()

    result = service.investigate(
        raw_findings,
        mode=args.mode,
        max_depth=args.max_depth,
        case_name="investigate",
    )
    elapsed = time.time() - t0

    print_report(result, elapsed)

    # --output was documented but never implemented; BeliefGraph.export_json
    # has always been able to serve it.
    if args.output:
        result.graph.export_json(Path(args.output))
        print(f"[+] Belief graph written to {args.output}\n")

if __name__ == "__main__":
    main()
