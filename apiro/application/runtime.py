"""Construct shared runtime resources and isolated per-investigation state."""

from __future__ import annotations

from dataclasses import dataclass


class RuntimeSetupError(RuntimeError):
    """Raised when the local model or corpus is unavailable."""


class ChromaQueryAdapter:
    """Translate :class:`Embedder` results into the shape NodeExpander uses."""

    def __init__(self, embedder):
        self._embedder = embedder

    def query(self, collection_name="", query_texts=None, n_results=6, where=None):
        query_texts = query_texts or []
        query = query_texts[0] if query_texts else ""
        results = self._embedder.query(query, n_results=n_results, where=where)
        return {
            "documents": [[item["text"] for item in results]],
            "distances": [[item.get("distance") for item in results]],
        }


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

    def create_traversal(
        self,
        *,
        n_diagnoses: int | None = None,
        allow_abstention: bool = False,
        log_dir=None,
    ):
        """Return a traversal whose callbacks, logs and detectors are run-local."""
        from apiro.config import N_DIFFERENTIAL, SATURATION_EXPLORATION_ONLY
        from apiro.entropy.engine import EntropyEngine
        from apiro.graph.contradiction import ContradictionDetector
        from apiro.graph.expander import NodeExpander
        from apiro.graph.rabbit_hole import RabbitHoleDetector
        from apiro.graph.saturation import SaturationDetector
        from apiro.graph.traversal import ApiroTraversal

        contradiction = ContradictionDetector(
            model=self.model,
            ollama_url=self.ollama_url,
            scheduler=self.model_scheduler,
        )
        expander = NodeExpander(
            entropy_engine=EntropyEngine(
                model=self.model,
                ollama_url=self.ollama_url,
                scheduler=self.model_scheduler,
            ),
            chroma_client=ChromaQueryAdapter(self.embedder),
            llm_client=self.llm_client,
            contradiction_detector=contradiction,
            n_diagnoses=n_diagnoses or N_DIFFERENTIAL,
            allow_abstention=allow_abstention,
        )
        traversal = ApiroTraversal(
            expander=expander,
            saturation=SaturationDetector(
                exploration_only=SATURATION_EXPLORATION_ONLY
            ),
            rabbit_hole=RabbitHoleDetector(),
            contradiction=contradiction,
            log_dir=log_dir,
        )
        traversal.run_id = None
        return traversal


def build_runtime_resources(
    *, llm_timeout: int = 120, require_corpus: bool = True
) -> RuntimeResources:
    """Build shared model, vector-store and extraction resources."""
    import requests

    from apiro.axioms.extractor import AxiomExtractor
    from apiro.application.model_scheduler import ModelCallScheduler
    from apiro.config import MAX_MODEL_CONCURRENCY, OLLAMA_BASE_URL, PRIMARY_MODEL
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
            timeout=llm_timeout,
            scheduler=scheduler,
        ),
        axiom_extractor=AxiomExtractor(),
        doc_count=doc_count,
        model=PRIMARY_MODEL,
        ollama_url=OLLAMA_BASE_URL,
        model_scheduler=scheduler,
    )
