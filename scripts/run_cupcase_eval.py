#!/usr/bin/env python3
"""
scripts/run_cupcase_eval.py
===========================
CUPCase distractor-resilience benchmark.

WHY THIS BENCHMARK
------------------
Apiro's central claim is *distractor rejection*: it should decline a
plausible-but-wrong diagnosis that a bare LLM or a standard RAG pipeline
accepts. The two benchmarks already in the repo measure that only indirectly:

  * C-NIAH (``run_niah_eval.py``) is synthetic and self-authored. Its
    distractors, its needles, and the stub responses that recover them all come
    out of ``build_niah_cases.py``. It is a good instrumented probe of the
    mechanism and a weak claim about the world.
  * The PMC set (``run_pmc_eval.py``) is real but N = 10, and its
    ``target_diagnosis`` fields were written by an unconstrained LLM — four of
    the ten are multi-paragraph prose rather than a diagnosis label.

CUPCase (``ofir408/CupCase``, 3,562 real clinical cases) is external, public,
and — the reason it fits here specifically — **ships three curated distractors
per case**. That turns distractor rejection from something inferred out of an
accuracy gap into something measured directly: how often does each arm name a
distractor as its leading diagnosis?

The adapter this harness drives (``apiro/corpus/clinical_case_adapter.py``)
was already in the repository, complete and unreferenced by anything.

WHAT IT REPORTS
---------------
  * Top-1 / top-3 / top-5 accuracy and Mean Reciprocal Rank, so a first-place
    answer is not scored identically to a third-place one.
  * Distractor-selection rate — the direct measure of the claim.
  * Wilson 95% intervals per arm, plus a paired bootstrap CI on each delta and
    an exact McNemar test against the bare-LLM baseline. At the sample sizes
    this project runs at, an accuracy delta without a paired test is not
    evidence.

Requires a live stack: Ollama serving ``config.PRIMARY_MODEL`` and a populated
ChromaDB corpus. The first invocation also downloads the CUPCase split via
HuggingFace ``datasets`` (cached thereafter).

Usage:
    python scripts/run_cupcase_eval.py --n 50
    python scripts/run_cupcase_eval.py --n 50 --out data/cupcase_eval_results.json
    python scripts/run_cupcase_eval.py --n 20 --describe-only   # no model calls
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cupcase_eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apiro.eval.harness import build_real_components, make_matcher  # noqa: E402
from apiro.eval.metrics import compare_arms, score_arm  # noqa: E402
from apiro.graph.belief_graph import BeliefGraph  # noqa: E402

ARMS = ("apiro", "rag", "bare_llm")
ARM_LABELS = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}

#: How many diagnoses each arm is asked for. Fixed across arms — asking one arm
#: for more candidates than another would inflate its top-k for free.
N_DIFFERENTIAL = 5

BARE_PROMPT = (
    "Based on the following clinical presentation, list your top {n} differential "
    "diagnoses, most likely first.\n"
    "Output ONLY the {n} diagnosis names, one per line, no numbering, no "
    "explanation.\n\n"
    "{vignette}"
)

RAG_PROMPT = (
    "Based on the following clinical presentation and the retrieved medical "
    "context, list your top {n} differential diagnoses, most likely first.\n"
    "Output ONLY the {n} diagnosis names, one per line, no numbering, no "
    "explanation.\n\n"
    "Clinical presentation:\n{vignette}\n\nRetrieved context:\n{context}"
)


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def _parse_differential(raw: str, limit: int = N_DIFFERENTIAL) -> list[str]:
    """Parse an LLM's newline-separated differential into a ranked list.

    Strips list numbering and bullets, drops preamble lines, and preserves
    order — order *is* the ranking every rank-aware metric reads.
    """
    import re

    out: list[str] = []
    for line in (raw or "").splitlines():
        clean = re.sub(r"^\s*\d+\s*[.)]\s*|^\s*[-*•]\s*", "", line.strip()).strip()
        if not clean or clean.startswith("```"):
            continue
        # Drop header lines like "Differential:" / "Top 3 diagnoses:".
        if re.match(r"^(top\s*\d*|differential|diagnos[ei]s|answer|output)\s*:?\s*$",
                    clean, re.IGNORECASE):
            continue
        out.append(clean)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #
def _evaluate_case(case: dict, components, max_depth: int) -> dict:
    """Run all three arms on one CUPCase case and return its raw outputs."""
    embedder = components.embedder
    llm_client = components.llm_client
    traversal = components.traversal

    case_id = case["case_id"]
    ground_truth = case["ground_truth"]
    distractors = [d for d in case.get("distractors", []) if d and str(d).strip()]

    # The adapter turns the narrative into PatientFindings; the narrative
    # itself is what the two baselines read, so all three arms see the same
    # source text.
    vignette = "\n".join(
        f.get("value", "") for f in case.get("findings", []) if f.get("value")
    ).strip() or case.get("description", "")

    logger.info(f"\nCase {case_id} — truth={ground_truth!r}, {len(distractors)} distractor(s)")

    # ---- Arm 1: bare LLM zero-shot --------------------------------------- #
    bare_raw = llm_client.generate(
        BARE_PROMPT.format(n=N_DIFFERENTIAL, vignette=vignette)
    )
    bare_preds = _parse_differential(bare_raw)
    logger.info(f"  bare_llm : {bare_preds}")

    # ---- Arm 2: standard RAG --------------------------------------------- #
    rag_chunks = embedder.query(vignette, n_results=6)
    rag_context = "\n\n".join(chunk["text"] for chunk in rag_chunks)
    rag_raw = llm_client.generate(
        RAG_PROMPT.format(n=N_DIFFERENTIAL, vignette=vignette, context=rag_context)
    )
    rag_preds = _parse_differential(rag_raw)
    logger.info(f"  rag      : {rag_preds}")

    # ---- Arm 3: Apiro belief-graph traversal ----------------------------- #
    from apiro.axioms.seeding import build_seeds

    graph = BeliefGraph()
    seeds, axioms, enriched = build_seeds(vignette, components.axiom_extractor)
    result = traversal.run(
        seed_nodes=seeds,
        graph=graph,
        max_depth=max_depth,
        case_name=f"cupcase_{case_id}",
        vignette=enriched,
    )
    apiro_preds = list(result.synthesis or [])
    logger.info(f"  apiro    : {apiro_preds}  (stop={result.stop_reason})")

    return {
        "case_id": case_id,
        "source": case.get("source", "cupcase"),
        "specialty": case.get("specialty", "unknown"),
        "ground_truth": ground_truth,
        "distractors": distractors,
        "n_axioms": len(axioms),
        "predictions": {
            "bare_llm": bare_preds,
            "rag": rag_preds,
            "apiro": apiro_preds,
        },
        "raw_output": {
            "bare_llm": bare_raw,
            "rag": rag_raw,
            "apiro": apiro_preds,
        },
        "traversal": {
            "stop_reason": result.stop_reason,
            "total_nodes": result.total_nodes,
            "total_edges": result.total_edges,
            "seed_nodes": sum(1 for n in graph.nodes.values() if n.depth == 0),
            "explored_nodes": graph.count_expansions(min_depth=1),
            "max_depth_reached": max((n.depth for n in graph.nodes.values()), default=0),
            "rabbit_holes": result.rabbit_hole_count,
            "contradictions": result.contradiction_count,
            "duration_seconds": result.duration_seconds,
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _print_report(scores: dict, comparisons: dict, n_cases: int) -> None:
    width = 78
    print("\n" + "=" * width)
    print("CUPCASE DISTRACTOR-RESILIENCE BENCHMARK".center(width))
    print(f"N = {n_cases} cases".center(width))
    print("=" * width)

    print(f"\n{'Arm':<16}{'Top-1':>10}{'Top-3':>10}{'Top-5':>10}{'MRR':>10}"
          f"{'Top-3 95% CI':>22}")
    print("-" * width)
    for arm in ARMS:
        if arm not in scores:
            continue
        s = scores[arm]
        low, high = s.top3_ci
        print(
            f"{ARM_LABELS[arm]:<16}"
            f"{_fmt_pct(s.top_k.get(1, 0.0)):>10}"
            f"{_fmt_pct(s.top_k.get(3, 0.0)):>10}"
            f"{_fmt_pct(s.top_k.get(5, 0.0)):>10}"
            f"{s.mrr:>10.3f}"
            f"{f'[{low * 100:.1f}%, {high * 100:.1f}%]':>22}"
        )
    print("=" * width)

    # ---- The claim this benchmark exists to test ------------------------- #
    print("\n" + "=" * width)
    print("  DISTRACTOR SELECTION — top-1 prediction is a curated wrong answer")
    print("  (lower is better; this is the mechanism Apiro claims to have)")
    print("=" * width)
    for arm in ARMS:
        if arm not in scores:
            continue
        s = scores[arm]
        if s.distractor_rate is None:
            print(f"  {ARM_LABELS[arm]:<16} n/a (no distractors in this case set)")
        else:
            print(f"  {ARM_LABELS[arm]:<16}{_fmt_pct(s.distractor_rate):>10}"
                  f"   (over {s.n_distractor_cases} case(s))")
    print("=" * width)

    # ---- Paired significance --------------------------------------------- #
    print("\n" + "=" * width)
    print("  PAIRED COMPARISON vs Bare LLM (top-3 correctness)")
    print("=" * width)
    for arm, comp in comparisons.items():
        delta = comp["delta_ci"]
        mcn = comp["mcnemar"]
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(
            f"  {ARM_LABELS.get(arm, arm):<16}"
            f"delta={delta['delta'] * 100:+6.1f}pp  "
            f"95% CI [{delta['ci_low'] * 100:+.1f}, {delta['ci_high'] * 100:+.1f}]pp"
        )
        print(
            f"  {'':<16}McNemar: {mcn['a_only']} won / {mcn['b_only']} lost "
            f"of {mcn['n_discordant']} discordant, p={mcn['p_value']:.4f} "
            f"({verdict} at alpha=0.05)"
        )
    print("=" * width + "\n")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_evaluation(
    n: int,
    max_depth: int,
    out_path: str | None,
    seed: int,
    describe_only: bool,
) -> None:
    from apiro.corpus.clinical_case_adapter import ClinicalCaseAdapter

    adapter = ClinicalCaseAdapter()
    logger.info(f"Loading {n} CUPCase case(s)...")
    raw_cases = adapter.load_cupcase(n=n, seed=seed)
    # entropy_engine=None: seeds get the heuristic entropy from config rather
    # than one Ollama round-trip per seed node, which would dominate runtime
    # without changing the frontier ordering.
    cases = adapter.build_cases(raw_cases, entropy_engine=None)
    logger.info(f"Built {len(cases)} evaluation case(s).")

    if describe_only:
        # Dataset sanity check with no model calls — useful for confirming the
        # adapter, the distractor fields and the ground-truth labels look sane
        # before committing to a long benchmark run.
        n_with_distractors = sum(
            1 for c in cases if any(d and str(d).strip() for d in c.get("distractors", []))
        )
        print("\n" + "=" * 78)
        print("CUPCASE DATASET PREVIEW (no model calls made)".center(78))
        print("=" * 78)
        print(f"  cases loaded            : {len(cases)}")
        print(f"  cases with distractors  : {n_with_distractors}")
        print(f"  mean findings per case  : "
              f"{sum(len(c.get('findings', [])) for c in cases) / max(1, len(cases)):.1f}")
        print("-" * 78)
        for c in cases[:5]:
            print(f"  {c['case_id']}: truth={c['ground_truth'][:48]!r}")
            for d in c.get("distractors", [])[:3]:
                if d and str(d).strip():
                    print(f"      distractor: {str(d)[:60]!r}")
        print("=" * 78 + "\n")
        return

    components = build_real_components()
    logger.info(f"Components ready. Corpus: {components.doc_count:,} documents.")

    results = []
    for i, case in enumerate(cases, start=1):
        logger.info(f"\n[{i}/{len(cases)}] ---------------------------------")
        try:
            results.append(_evaluate_case(case, components, max_depth))
        except Exception as exc:  # noqa: BLE001
            # One bad case must not discard the whole run, but it must be
            # visible: a silently dropped case changes every denominator.
            logger.error(f"Case {case['case_id']} failed: {exc}", exc_info=True)

    if not results:
        logger.error("No case completed successfully — nothing to report.")
        sys.exit(1)
    if len(results) != len(cases):
        logger.warning(
            f"{len(cases) - len(results)} case(s) failed and are excluded. "
            f"All reported rates are over the {len(results)} that completed."
        )

    matcher = make_matcher(embedder=components.embedder, llm_client=components.llm_client)
    ground_truths = [r["ground_truth"] for r in results]
    distractors = [r["distractors"] for r in results]

    scores = {
        arm: score_arm(
            arm,
            [r["predictions"][arm] for r in results],
            ground_truths,
            matcher,
            distractors_per_case=distractors,
        )
        for arm in ARMS
    }
    comparisons = compare_arms(scores, reference="bare_llm", k=3)

    _print_report(scores, comparisons, len(results))

    if out_path:
        out_file = Path(out_path)
        if not out_file.is_absolute():
            out_file = PROJECT_ROOT / out_file
        out_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "cupcase",
            "dataset": "ofir408/CupCase",
            "n_requested": n,
            "n_completed": len(results),
            "sample_seed": seed,
            "n_differential": N_DIFFERENTIAL,
            "corpus_documents": components.doc_count,
            "scores": {arm: s.to_dict() for arm, s in scores.items()},
            "comparisons": comparisons,
            "case_results": results,
        }
        with open(out_file, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info(f"Saved results to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUPCase distractor-resilience benchmark (Apiro vs RAG vs bare LLM)."
    )
    parser.add_argument("--n", type=int, default=50,
                        help="Number of CUPCase cases to evaluate (default: 50).")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max traversal depth for the Apiro arm (default: 6).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed for case selection (default: 42).")
    parser.add_argument("--out", type=str, default="data/cupcase_eval_results.json",
                        help="Where to write the detailed results JSON.")
    parser.add_argument("--describe-only", action="store_true",
                        help="Load and summarise the case set without calling any model.")
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be a positive integer.")

    run_evaluation(
        n=args.n,
        max_depth=args.max_depth,
        out_path=args.out,
        seed=args.seed,
        describe_only=args.describe_only,
    )


if __name__ == "__main__":
    main()
