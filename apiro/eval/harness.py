"""
apiro/eval/harness.py — shared evaluation wiring
=================================================

Every benchmark script needs the same live stack: an Embedder over the
persistent ChromaDB corpus, an adapter presenting it in the shape
NodeExpander expects, an EntropyEngine, a ContradictionDetector, and an
ApiroTraversal tying them together.

That wiring was copy-pasted into ``scripts/run_pmc_eval.py`` and
``scripts/run_niah_eval.py`` — including a private ``_ChromaAdapter`` class
defined twice, verbatim. The copies had already begun to drift (one passed a
120 s LLM timeout, the other the 90 s default), which is exactly the failure
mode ``apiro/llm_client.py`` was extracted to stop: two benchmarks reporting
comparable numbers from stacks that were not actually identical.

This module is the single implementation. Heavy imports (chromadb,
sentence-transformers, torch) stay inside the functions so importing
``apiro.eval`` remains cheap and offline-safe.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ChromaQueryAdapter",
    "RealComponents",
    "build_real_components",
    "make_matcher",
]


class ChromaQueryAdapter:
    """Presents an :class:`~apiro.corpus.embedder.Embedder` as a chroma client.

    ``NodeExpander._retrieve_context`` expects the raw ChromaDB response shape
    (``{"documents": [[...]], "distances": [[...]]}``), while ``Embedder.query``
    returns a flat list of dicts. This translates between them.

    Distances are passed through deliberately: the expander drops chunks beyond
    ``config.RAG_MAX_DISTANCE`` so a rare-disease query does not get six
    confidently-formatted nearest neighbours injected under "use ONLY what is
    stated here".
    """

    def __init__(self, embedder):
        self._emb = embedder

    def query(
        self,
        collection_name: str = "",
        query_texts: Optional[list] = None,
        n_results: int = 6,
        where: Optional[dict] = None,
    ) -> dict:
        query_texts = query_texts or []
        text = query_texts[0] if query_texts else ""
        results = self._emb.query(text, n_results=n_results, where=where)
        return {
            "documents": [[r["text"] for r in results]],
            "distances": [[r.get("distance") for r in results]],
        }


@dataclass
class RealComponents:
    """The live evaluation stack.

    Attributes:
        embedder:       Embedder over the persistent ChromaDB corpus.
        llm_client:     Shared OllamaLLMClient (``.chat`` / ``.generate``).
        traversal:      Configured ApiroTraversal.
        contradiction:  The detector the traversal uses (exposed so a harness
                        can report ``cache_info()`` after a run).
        axiom_extractor: Deterministic axiom extractor for seeding.
        doc_count:      Documents in the corpus collection.
    """

    embedder: object
    llm_client: object
    traversal: object
    contradiction: object
    axiom_extractor: object
    doc_count: int


def build_real_components(
    llm_timeout: int = 120,
    require_corpus: bool = True,
) -> RealComponents:
    """Build the live Ollama + ChromaDB evaluation stack.

    Args:
        llm_timeout: Per-request timeout for the generation client, in seconds.
            Long haystack prompts on a local 8B model routinely exceed the
            client default, so benchmarks pass a larger value.
        require_corpus: Exit with a message when the ChromaDB collection is
            empty. A benchmark that silently runs the "RAG" arm against zero
            documents reports a baseline that never existed.

    Returns:
        A :class:`RealComponents`.

    Raises:
        SystemExit: If Ollama is unreachable, or the corpus is empty and
            ``require_corpus`` is set. These are setup errors, not conditions a
            benchmark should try to recover from — proceeding would produce a
            results file that looks valid and is not.
    """
    import requests

    from apiro.config import (
        OLLAMA_BASE_URL,
        PRIMARY_MODEL,
        SATURATION_EXPLORATION_ONLY,
    )
    from apiro.axioms.extractor import AxiomExtractor
    from apiro.corpus.embedder import Embedder
    from apiro.entropy.engine import EntropyEngine
    from apiro.graph.contradiction import ContradictionDetector
    from apiro.graph.expander import NodeExpander
    from apiro.graph.rabbit_hole import RabbitHoleDetector
    from apiro.graph.saturation import SaturationDetector
    from apiro.graph.traversal import ApiroTraversal
    from apiro.llm_client import OllamaLLMClient

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal setup
        logger.error(
            f"[Ollama] Could not reach Ollama at {OLLAMA_BASE_URL}: {exc}\n"
            f"         Start it with: ollama serve"
        )
        sys.exit(1)

    embedder = Embedder()
    doc_count = embedder.count
    if doc_count == 0 and require_corpus:
        logger.error(
            "[Corpus] ChromaDB collection is empty.\n"
            "         Build it with: python -m apiro.corpus.build_corpus --sources medrag"
        )
        sys.exit(1)

    llm_client = OllamaLLMClient(OLLAMA_BASE_URL, PRIMARY_MODEL, timeout=llm_timeout)
    contradiction = ContradictionDetector()

    expander = NodeExpander(
        entropy_engine=EntropyEngine(model=PRIMARY_MODEL, ollama_url=OLLAMA_BASE_URL),
        chroma_client=ChromaQueryAdapter(embedder),
        llm_client=llm_client,
        contradiction_detector=contradiction,
    )

    traversal = ApiroTraversal(
        expander=expander,
        # exploration_only: depth-0 axiom seeds carry a fixed entropy and must
        # never be counted as evidence that the engine has converged.
        saturation=SaturationDetector(exploration_only=SATURATION_EXPLORATION_ONLY),
        rabbit_hole=RabbitHoleDetector(),
        contradiction=contradiction,
    )

    return RealComponents(
        embedder=embedder,
        llm_client=llm_client,
        traversal=traversal,
        contradiction=contradiction,
        axiom_extractor=AxiomExtractor(),
        doc_count=doc_count,
    )


def make_matcher(embedder=None, llm_client=None):
    """Build the ``matcher(prediction, truth) -> bool`` used by apiro.eval.metrics.

    Wraps the single-candidate case of
    :func:`apiro.eval.evaluator._check_synthesis_hit` so the rank-aware metrics
    grade with exactly the same concept-normalization cascade the pass/fail
    harnesses use. Two scores computed by different matchers are not
    comparable, so there is deliberately only one way to build this.

    Args:
        embedder: Optional encoder enabling the embedding-similarity fallback.
        llm_client: Optional client enabling the LLM-as-a-judge fallback.

    Returns:
        A predicate suitable for ``score_arm(..., matcher=...)``.
    """
    from apiro.eval.evaluator import _check_synthesis_hit

    def matcher(prediction: str, ground_truth: str) -> bool:
        hit, _ = _check_synthesis_hit(
            [prediction],
            ground_truth,
            embedder=embedder,
            llm_client=llm_client,
        )
        return bool(hit)

    return matcher
