"""Shared live benchmark wiring for the active bounded engines."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["RealComponents", "build_real_components", "make_matcher"]


@dataclass
class RealComponents:
    """Shared runtime resources used by live benchmark arms."""

    embedder: object
    llm_client: object
    axiom_extractor: object
    doc_count: int
    resources: object

    def create_service(self, **kwargs):
        return self.resources.create_service(**kwargs)


def build_real_components(
    llm_timeout: int = 120,
    require_corpus: bool = True,
) -> RealComponents:
    """Build the Ollama, ChromaDB, and extraction resources for a run."""
    from apiro.application.runtime import RuntimeSetupError, build_runtime_resources

    try:
        resources = build_runtime_resources(
            llm_timeout=llm_timeout, require_corpus=require_corpus
        )
    except RuntimeSetupError as exc:
        logger.error(str(exc))
        sys.exit(1)

    return RealComponents(
        embedder=resources.embedder,
        llm_client=resources.llm_client,
        axiom_extractor=resources.axiom_extractor,
        doc_count=resources.doc_count,
        resources=resources,
    )


def make_matcher(embedder=None, llm_client=None):
    """Build the one matcher shared by all active benchmark runners."""
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
