"""Shared live three-arm evaluation for narrative diagnosis cases."""

from __future__ import annotations

import hashlib
import time

from apiro.graph.belief_graph import BeliefGraph
from apiro.parsing import parse_differential


BARE_PROMPT = """Read the clinical presentation and list the top {n} differential diagnoses, most likely first.
Output only diagnosis names, one per line, with no numbering or explanation.

{narrative}"""

RAG_PROMPT = """Read the clinical presentation and retrieved medical context and list the top {n} differential diagnoses, most likely first.
Output only diagnosis names, one per line, with no numbering or explanation.

Clinical presentation:
{narrative}

Retrieved context:
{context}"""


def evaluate_narrative_case(
    *, case_name: str, narrative: str, resources, n_diagnoses: int = 3,
    max_depth: int = 6, log_dir=None,
) -> dict:
    """Evaluate bare, RAG and isolated Apiro arms with equal answer budgets."""
    started = time.time()
    bare_raw = resources.llm_client.generate(
        BARE_PROMPT.format(n=n_diagnoses, narrative=narrative)
    )
    bare = parse_differential(bare_raw, limit=n_diagnoses)

    rag_chunks = resources.embedder.query(narrative, n_results=6)
    rag_raw = resources.llm_client.generate(RAG_PROMPT.format(
        n=n_diagnoses, narrative=narrative,
        context="\n\n".join(chunk["text"] for chunk in rag_chunks),
    ))
    rag = parse_differential(rag_raw, limit=n_diagnoses)

    from apiro.axioms.seeding import build_seeds

    traversal = resources.create_traversal(
        n_diagnoses=n_diagnoses, log_dir=log_dir
    )
    graph = BeliefGraph()
    seeds, axioms, enriched = build_seeds(narrative, resources.axiom_extractor)
    result = traversal.run(
        seed_nodes=seeds, graph=graph, max_depth=max_depth,
        case_name=case_name, vignette=enriched,
    )
    apiro = list(result.synthesis or [])
    hypotheses = [
        {
            "claim": node.claim,
            "depth": node.depth,
            "entropy": node.entropy_score,
            "confidence": (node.metadata or {}).get("confidence"),
            "contradiction_penalty": node.contradiction_penalty,
        }
        for node in graph.nodes.values() if node.depth >= 1
    ]
    return {
        "predictions": {"apiro": apiro, "rag": rag, "bare_llm": bare},
        "raw_output": {"apiro": apiro, "rag": rag_raw, "bare_llm": bare_raw},
        "n_candidates": {"apiro": len(apiro), "rag": len(rag), "bare_llm": len(bare)},
        "input": {
            "sha256": hashlib.sha256(narrative.encode("utf-8")).hexdigest(),
            "characters": len(narrative),
            "approx_tokens": round(len(narrative.split()) / 0.75),
        },
        "traversal": {
            "stop_reason": result.stop_reason,
            "total_nodes": result.total_nodes,
            "contradictions": result.contradiction_count,
            "stage_timings": result.stage_timings,
            "hypotheses": hypotheses,
        },
        "wall_seconds_all_arms": round(time.time() - started, 4),
        "n_axioms": len(axioms),
    }
