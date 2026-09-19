"""Construct shared runtime resources and isolated per-investigation state."""

from __future__ import annotations

from dataclasses import dataclass


class RuntimeSetupError(RuntimeError):
    """Raised when the local model or corpus is unavailable."""


@dataclass
class RuntimeResources:
    """Expensive resources safe to share across independent investigations."""

    embedder: object
    llm_client: object
    axiom_extractor: object
    doc_count: int
    model: str
    ollama_url: str
    model_scheduler: object | None = None

    def create_service(self, *, default_mode: str | None = None):
        """Return the canonical stateless investigation orchestrator."""
        from apiro.application.service import InvestigationService
        from apiro.config import REASONING_MODE

        return InvestigationService(
            self, default_mode=default_mode or REASONING_MODE
        )


def build_runtime_resources(
    *, llm_timeout: int = 120, require_corpus: bool = True
) -> RuntimeResources:
    """Build shared model, vector-store and extraction resources."""
    import requests

    from apiro.axioms.extractor import AxiomExtractor
    from apiro.application.model_scheduler import ModelCallScheduler
    from apiro.config import (
        MAX_MODEL_CONCURRENCY,
        MODEL_JSON_NUM_PREDICT,
        MODEL_SEED,
        MODEL_TEMPERATURE,
        OLLAMA_BASE_URL,
        PRIMARY_MODEL,
    )
    from apiro.corpus.embedder import Embedder
    from apiro.llm_client import OllamaLLMClient

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeSetupError(
            f"Ollama is not reachable at {OLLAMA_BASE_URL}: {exc}. "
            "Start it with: ollama serve"
        ) from exc

    embedder = Embedder()
    doc_count = embedder.count
    if require_corpus and doc_count == 0:
        raise RuntimeSetupError(
            "ChromaDB corpus is empty; build it with "
            "python -m apiro.corpus.build_corpus --sources medrag"
        )
    scheduler = ModelCallScheduler(MAX_MODEL_CONCURRENCY)
    return RuntimeResources(
        embedder=embedder,
        llm_client=OllamaLLMClient(
            OLLAMA_BASE_URL,
            PRIMARY_MODEL,
            temperature=MODEL_TEMPERATURE,
            seed=MODEL_SEED,
            json_num_predict=MODEL_JSON_NUM_PREDICT,
            timeout=llm_timeout,
            scheduler=scheduler,
        ),
        axiom_extractor=AxiomExtractor(),
        doc_count=doc_count,
        model=PRIMARY_MODEL,
        ollama_url=OLLAMA_BASE_URL,
        model_scheduler=scheduler,
    )
