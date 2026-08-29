#!/usr/bin/env python3
"""
scripts/run_ddxplus_eval.py
===========================
DDXPlus differential-diagnosis benchmark.

WHY THIS BENCHMARK
------------------
Every other case set in this repository is either self-authored (C-NIAH) or
small (PMC, N = 10). DDXPlus (arXiv:2205.09148, CC-BY) is external, large, and
— uniquely here — ships a ground-truth **ranked differential** rather than a
single label. That is what makes top-k and MRR mean something: the reference
is an ordering, so a system that ranks the right answer second is measurably
better than one that ranks it fifth, and both are measurably worse than one
that ranks it first.

It is also the substrate MedEinst (arXiv:2601.06636) built its counterfactual
traps from, so results here are directly comparable in kind to that literature.

WHAT IT REPORTS
---------------
  * Top-1 / top-3 / top-5 accuracy against the ground-truth pathology, and MRR.
  * **Differential overlap** — how much of the reference differential each arm
    recovers, and whether it recovers it in the right order. Accuracy against a
    single label throws that away.
  * Wilson intervals per arm, a paired bootstrap CI on each delta, and an exact
    McNemar test against the bare-LLM baseline.

Requires: Ollama serving config.PRIMARY_MODEL, a populated ChromaDB corpus,
and `python scripts/fetch_datasets.py --only ddxplus`.

Usage:
    python scripts/run_ddxplus_eval.py --n 60
    python scripts/run_ddxplus_eval.py --n 20 --describe-only   # no model calls
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
logger = logging.getLogger("ddxplus_eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apiro.corpus.ddxplus_adapter import DDXPlusAdapter  # noqa: E402
from apiro.eval.harness import build_real_components, make_matcher  # noqa: E402
from apiro.eval.metrics import compare_arms, score_arm  # noqa: E402
from apiro.graph.belief_graph import BeliefGraph  # noqa: E402
from apiro.parsing import ABSTENTION_SENTINEL, detect_abstention, parse_differential  # noqa: E402

ARMS = ("apiro", "rag", "bare_llm")
ARM_LABELS = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}

#: Same candidate budget for every arm — see apiro/parsing.py for what happened
#: the last time the arms were allowed different ones.
N_DIFFERENTIAL = 5

BARE_PROMPT = (
    "Read the clinical note and give your top {n} differential diagnoses, most "
    "likely first.\n"
    "Output ONLY the {n} diagnosis names, one per line, no numbering, no "
    "explanation.\n"
    "If the note does not support any diagnosis, reply with exactly "
    "'{abstain}'.\n\n{vignette}"
)

RAG_PROMPT = (
    "Read the clinical note and the retrieved medical context, then give your "
    "top {n} differential diagnoses, most likely first.\n"
    "Output ONLY the {n} diagnosis names, one per line, no numbering, no "
    "explanation.\n"
    "If the note does not support any diagnosis, reply with exactly "
    "'{abstain}'.\n\nClinical note:\n{vignette}\n\nRetrieved context:\n{context}"
)


# --------------------------------------------------------------------------- #
# Differential overlap — what a ranked reference makes possible
# --------------------------------------------------------------------------- #
def differential_overlap(predicted, reference, matcher, k: int = 5) -> dict:
    """How much of the reference differential did this arm recover?

    DDXPlus gives a ranked list of plausible diagnoses, not one answer. An arm
    that names three of the reference's five is doing something different from
    one that names the top answer and then two unrelated diseases, and top-1
    accuracy scores them identically.

    Returns:
        ``recall_at_k`` — fraction of the reference's top-k the arm named;
        ``precision_at_k`` — fraction of the arm's k that appear in the
        reference at all; ``top1_agreement`` — whether both rank the same
        diagnosis first.
    """
    ref_k = [r for r in reference[:k] if r]
    pred_k = [p for p in predicted[:k] if p]
    if not ref_k:
        return {"recall_at_k": None, "precision_at_k": None, "top1_agreement": None}

    recovered = sum(1 for r in ref_k if any(matcher(p, r) for p in pred_k))
    relevant = sum(1 for p in pred_k if any(matcher(p, r) for r in reference))
    return {
        "recall_at_k": recovered / len(ref_k),
        "precision_at_k": (relevant / len(pred_k)) if pred_k else 0.0,
        "top1_agreement": bool(pred_k and matcher(pred_k[0], ref_k[0])),
    }


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #
def _evaluate_case(case, components, max_depth: int) -> dict:
    embedder = components.embedder
    llm_client = components.llm_client
    traversal = components.traversal
    vignette = case.vignette

    logger.info(f"\n{case.case_id} — truth={case.ground_truth!r}, "
                f"reference differential of {len(case.differential)}")

    bare_raw = llm_client.generate(BARE_PROMPT.format(
        n=N_DIFFERENTIAL, abstain=ABSTENTION_SENTINEL, vignette=vignette))
    bare_preds = parse_differential(bare_raw, limit=N_DIFFERENTIAL)
    logger.info(f"  bare_llm : {bare_preds}")

    rag_chunks = embedder.query(vignette, n_results=6)
    rag_raw = llm_client.generate(RAG_PROMPT.format(
        n=N_DIFFERENTIAL, abstain=ABSTENTION_SENTINEL, vignette=vignette,
        context="\n\n".join(c["text"] for c in rag_chunks)))
    rag_preds = parse_differential(rag_raw, limit=N_DIFFERENTIAL)
    logger.info(f"  rag      : {rag_preds}")

    from apiro.axioms.seeding import build_seeds

    graph = BeliefGraph()
    seeds, axioms, enriched = build_seeds(vignette, components.axiom_extractor)
    result = traversal.run(
        seed_nodes=seeds, graph=graph, max_depth=max_depth,
        case_name=case.case_id, vignette=enriched,
    )
    apiro_preds = list(result.synthesis or [])
    logger.info(f"  apiro    : {apiro_preds}  (stop={result.stop_reason})")

    return {
        "case_id": case.case_id,
        "ground_truth": case.ground_truth,
        "reference_differential": case.differential,
        "n_axioms": len(axioms),
        "n_evidences": case.n_evidences,
        "predictions": {"bare_llm": bare_preds, "rag": rag_preds, "apiro": apiro_preds},
        "abstained": {
            "bare_llm": detect_abstention(bare_raw),
            "rag": detect_abstention(rag_raw),
            "apiro": detect_abstention("\n".join(apiro_preds)),
        },
        "n_candidates": {
            "bare_llm": len(bare_preds), "rag": len(rag_preds), "apiro": len(apiro_preds),
        },
        "traversal": {
            "stop_reason": result.stop_reason,
            "total_nodes": result.total_nodes,
            "explored_nodes": graph.count_expansions(min_depth=1),
            "contradictions": result.contradiction_count,
            "duration_seconds": result.duration_seconds,
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _report(scores, comparisons, overlaps, n_cases) -> None:
    w = 78
    print("\n" + "=" * w)
    print("DDXPLUS DIFFERENTIAL-DIAGNOSIS BENCHMARK".center(w))
    print(f"N = {n_cases} cases  (external, CC-BY, ranked reference differential)".center(w))
    print("=" * w)

    print(f"\n{'Arm':<16}{'Top-1':>9}{'Top-3':>9}{'Top-5':>9}{'MRR':>9}{'Top-3 95% CI':>22}")
    print("-" * w)
    for arm in ARMS:
        s = scores[arm]
        lo, hi = s.top3_ci
        print(f"{ARM_LABELS[arm]:<16}"
              f"{s.top_k.get(1, 0.0) * 100:>8.1f}%{s.top_k.get(3, 0.0) * 100:>8.1f}%"
              f"{s.top_k.get(5, 0.0) * 100:>8.1f}%{s.mrr:>9.3f}"
              f"{f'[{lo * 100:.1f}%, {hi * 100:.1f}%]':>22}")
    print("=" * w)

    print("\n" + "=" * w)
    print("  DIFFERENTIAL OVERLAP — against the reference ranked differential")
    print("  (single-label accuracy above cannot see this)")
    print("=" * w)
    print(f"{'Arm':<16}{'recall@5':>12}{'precision@5':>14}{'top-1 agreement':>18}")
    print("-" * w)
    for arm in ARMS:
        o = overlaps[arm]
        print(f"{ARM_LABELS[arm]:<16}{o['recall_at_k'] * 100:>11.1f}%"
              f"{o['precision_at_k'] * 100:>13.1f}%{o['top1_agreement'] * 100:>17.1f}%")
    print("=" * w)

    print("\n" + "=" * w)
    print("  PAIRED COMPARISON vs Bare LLM (top-3 correctness)")
    print("=" * w)
    for arm, comp in comparisons.items():
        d, m = comp["delta_ci"], comp["mcnemar"]
        verdict = "significant" if m["significant_at_05"] else "NOT significant"
        print(f"  {ARM_LABELS.get(arm, arm):<16}delta={d['delta'] * 100:+6.1f}pp  "
              f"95% CI [{d['ci_low'] * 100:+.1f}, {d['ci_high'] * 100:+.1f}]pp")
        print(f"  {'':<16}McNemar: {m['a_only']} won / {m['b_only']} lost of "
              f"{m['n_discordant']} discordant, p={m['p_value']:.4f} ({verdict})")
    print("=" * w + "\n")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_evaluation(n, split, seed, max_depth, out_path, describe_only) -> None:
    adapter = DDXPlusAdapter()
    cases = adapter.load_cases(n=n, split=split, seed=seed)
    logger.info(f"Loaded {len(cases)} DDXPlus cases from split={split!r}.")

    if describe_only:
        print("\n" + "=" * 78)
        print("DDXPLUS PREVIEW (no model calls made)".center(78))
        print("=" * 78)
        print(f"  cases                       : {len(cases)}")
        print(f"  mean resolved evidences     : "
              f"{sum(c.n_evidences for c in cases) / max(1, len(cases)):.1f}")
        print(f"  mean reference differential : "
              f"{sum(len(c.differential) for c in cases) / max(1, len(cases)):.1f}")
        print(f"  distinct ground truths      : {len({c.ground_truth for c in cases})}")
        print("-" * 78)
        ex = cases[0]
        print(f"  example {ex.case_id}")
        print(f"    truth       : {ex.ground_truth!r}")
        print(f"    differential: {ex.differential[:5]}")
        print("    vignette:")
        for line in ex.vignette.splitlines()[:8]:
            print(f"      {line}")
        print("=" * 78 + "\n")
        return

    components = build_real_components()
    components.traversal.expander.n_diagnoses = N_DIFFERENTIAL
    logger.info(f"Components ready. Corpus: {components.doc_count:,} documents.")

    results = []
    for i, case in enumerate(cases, 1):
        logger.info(f"\n[{i}/{len(cases)}] -----------------------------")
        try:
            results.append(_evaluate_case(case, components, max_depth))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{case.case_id} failed: {exc}", exc_info=True)

    if not results:
        logger.error("No case completed — nothing to report.")
        sys.exit(1)
    if len(results) != len(cases):
        logger.warning(f"{len(cases) - len(results)} case(s) failed and are excluded; "
                       f"all rates are over the {len(results)} that completed.")

    matcher = make_matcher(embedder=components.embedder, llm_client=components.llm_client)
    truths = [r["ground_truth"] for r in results]

    scores = {
        arm: score_arm(arm, [r["predictions"][arm] for r in results], truths, matcher)
        for arm in ARMS
    }
    comparisons = compare_arms(scores, reference="bare_llm", k=3)

    overlaps = {}
    for arm in ARMS:
        per_case = [
            differential_overlap(r["predictions"][arm], r["reference_differential"], matcher)
            for r in results
        ]
        usable = [o for o in per_case if o["recall_at_k"] is not None]
        overlaps[arm] = {
            "recall_at_k": sum(o["recall_at_k"] for o in usable) / len(usable) if usable else 0.0,
            "precision_at_k": sum(o["precision_at_k"] for o in usable) / len(usable) if usable else 0.0,
            "top1_agreement": sum(1 for o in usable if o["top1_agreement"]) / len(usable) if usable else 0.0,
            "n_scored": len(usable),
        }

    _report(scores, comparisons, overlaps, len(results))

    if out_path:
        out = Path(out_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({
                "benchmark": "ddxplus",
                "dataset": "aai530-group6/ddxplus",
                "split": split, "sample_seed": seed,
                "n_requested": n, "n_completed": len(results),
                "n_differential": N_DIFFERENTIAL,
                "scores": {a: s.to_dict() for a, s in scores.items()},
                "differential_overlap": overlaps,
                "comparisons": comparisons,
                "case_results": results,
            }, fh, indent=2)
        logger.info(f"Saved results to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DDXPlus differential-diagnosis benchmark.")
    parser.add_argument("--n", type=int, default=60, help="Cases to evaluate (default: 60).")
    parser.add_argument("--split", default="test", choices=("train", "validate", "test"))
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--out", default="data/ddxplus_eval_results.json")
    parser.add_argument("--describe-only", action="store_true",
                        help="Load and summarise the cases without calling any model.")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be a positive integer.")
    run_evaluation(args.n, args.split, args.seed, args.max_depth,
                   args.out, args.describe_only)


if __name__ == "__main__":
    main()
