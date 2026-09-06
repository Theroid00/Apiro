#!/usr/bin/env python3
"""
scripts/investigate.py
======================
CLI runner for Apiro. Extracts deterministic clinical axioms from free-text
findings, seeds the belief graph with them, and runs the entropy-first
ApiroTraversal to a ranked differential.

(The "Hypothesis-Testing Engine" this docstring used to name was purged in
commit 8a001c7; ApiroTraversal is the only strategy in the codebase.)

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
    import requests
    from apiro.corpus.embedder import Embedder
    from apiro.entropy.engine import EntropyEngine
    from apiro.axioms.extractor import AxiomExtractor
    from apiro.graph.expander import NodeExpander
    from apiro.graph.saturation import SaturationDetector
    from apiro.graph.rabbit_hole import RabbitHoleDetector
    from apiro.graph.contradiction import ContradictionDetector
    from apiro.graph.traversal import ApiroTraversal
    from apiro.config import (
        OLLAMA_BASE_URL,
        PRIMARY_MODEL,
        SATURATION_EXPLORATION_ONLY,
    )

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"\n❌  Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
        print("    Start it with:  ollama serve")
        sys.exit(1)

    embedder = Embedder()
    doc_count = embedder.count
    if doc_count == 0:
        print("\n❌  ChromaDB corpus is empty.")
        print("    Build it with:  python -m apiro.corpus.build_corpus")
        sys.exit(1)

    class _ChromaAdapter:
        def __init__(self, emb: Embedder):
            self._emb = emb
        def query(self, collection_name: str = "", query_texts: list = None, n_results: int = 6, where: dict | None = None) -> dict:
            query_texts = query_texts or []
            text = query_texts[0] if query_texts else ""
            results = self._emb.query(text, n_results=n_results, where=where)
            # Distances are passed through so the expander can discard chunks
            # that are merely the nearest neighbours rather than real evidence.
            return {
                "documents": [[r["text"] for r in results]],
                "distances": [[r.get("distance") for r in results]],
            }

    chroma_adapter = _ChromaAdapter(embedder)
    entropy_engine = EntropyEngine(model=PRIMARY_MODEL, ollama_url=OLLAMA_BASE_URL)
    axiom_extractor = AxiomExtractor()
    contradiction = ContradictionDetector()

    from apiro.llm_client import OllamaLLMClient
    llm_client = OllamaLLMClient(OLLAMA_BASE_URL, PRIMARY_MODEL)

    expander = NodeExpander(
        entropy_engine=entropy_engine,
        chroma_client=chroma_adapter,
        llm_client=llm_client,
        contradiction_detector=contradiction,
    )
    # exploration_only: deterministic depth-0 axiom seeds carry a fixed entropy
    # and must not be allowed to trigger saturation (see config comments).
    saturation = SaturationDetector(exploration_only=SATURATION_EXPLORATION_ONLY)
    rabbit_hole = RabbitHoleDetector()

    traversal = ApiroTraversal(
        expander=expander,
        saturation=saturation,
        rabbit_hole=rabbit_hole,
        contradiction=contradiction,
    )

    return (traversal, axiom_extractor), doc_count

def print_report(result, elapsed: float) -> None:
    print("\n+" + "-" * 58 + "+")
    print("|" + "    APIRO DIFFERENTIAL DIAGNOSIS REPORT".center(58) + "|")
    print("+" + "-" * 58 + "+")

    print(f"\n  Time taken:             {elapsed:.1f} seconds")
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
        help="Max traversal depth.",
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
    (traversal, axiom_extractor), doc_count = build_components()
    print(f"[+] Components ready. Corpus: {doc_count:,} documents.\n")

    print(f"\n[*] Apiro is investigating...")
    t0 = time.time()

    from apiro.graph.belief_graph import BeliefGraph
    graph = BeliefGraph()

    # Extract deterministic axioms and seed the graph
    print("[*] Extracting deterministic clinical axioms...")
    from apiro.axioms.seeding import build_seeds
    seeds, axioms, enriched_vignette = build_seeds(raw_findings, axiom_extractor)
    print(f"[+] Extracted {len(axioms)} axioms and anchored {len(seeds)} of them to the graph.\n")

    result = traversal.run(
        seed_nodes=seeds,
        graph=graph,
        max_depth=args.max_depth,
        case_name="investigate",
        vignette=enriched_vignette,
    )
    elapsed = time.time() - t0

    print_report(result, elapsed)

    # --output was documented but never implemented; BeliefGraph.export_json
    # has always been able to serve it.
    if args.output:
        graph.export_json(Path(args.output))
        print(f"[+] Belief graph written to {args.output}\n")

if __name__ == "__main__":
    main()
