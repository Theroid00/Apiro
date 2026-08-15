"""
corpus/embedder.py
===================
Embeds chunks using all-mpnet-base-v2 and inserts them into ChromaDB.

Each chunk becomes one ChromaDB document:
  - document:  chunk["text"]
  - embedding: 768-dim float vector from all-mpnet-base-v2
  - metadata:  all other chunk fields (source_db, medical_domain, pmid, etc.)
  - id:        chunk["chunk_id"]

Usage:
    from corpus.embedder import Embedder
    embedder = Embedder()
    embedder.ingest(chunks)
    results = embedder.query("chest pain diagnosis", n_results=6)
"""

import logging
from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from apiro.config import EMBED_MODEL, EMBED_DIM, CHROMA_DIR, CHROMA_COLLECTION, RAG_TOP_K

logger = logging.getLogger(__name__)

# Fields that ChromaDB metadata does NOT accept as non-string types
_ALLOWED_META_TYPES = (str, int, float, bool)


def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str/int/float/bool as metadata values.
    Coerce or drop anything else.
    """
    clean = {}
    for k, v in meta.items():
        if isinstance(v, _ALLOWED_META_TYPES):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = ", ".join(str(x) for x in v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


class Embedder:
    """
    Manages embedding and ChromaDB ingestion for the Apiro corpus.

    Args:
        model_name:        Sentence transformer model to use.
        chroma_path:       Directory for persistent ChromaDB storage.
        collection_name:   ChromaDB collection name.
        batch_size:        Chunks per embedding/upsert batch.
        device:            PyTorch device for the embedding model.
                           Defaults to 'cpu' — this is intentional.
                           GPU (CUDA) crashes mid-run leave ChromaDB in an
                           inconsistent state with no recovery path.
                           CPU embedding is stable for any corpus size and
                           fast enough for our batch sizes (~256 chunks/3s).
                           Pass device='cuda' explicitly if you need GPU speed
                           and accept the crash risk.
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        chroma_path: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.model_name      = model_name
        self.collection_name = collection_name
        self.batch_size      = batch_size
        self.device          = device

        logger.info(f"Loading embedding model: {model_name} on device='{device}'")
        self._model = SentenceTransformer(model_name, device=device)

        logger.info(f"Connecting to ChromaDB at {chroma_path}")
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for medical text
        )
        existing = self._collection.count()
        logger.info(
            f"Collection '{collection_name}' has "
            f"{existing:,} existing documents. "
            f"Upsert is idempotent — safe to re-run."
        )

    def encode(self, texts, **kwargs):
        """Forward encoding calls to the underlying SentenceTransformer model."""
        return self._model.encode(texts, **kwargs)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, chunks: Sequence[dict], show_progress: bool = True) -> int:
        """
        Embed and upsert chunks into ChromaDB.

        Each batch is committed independently, so a mid-run failure only
        loses the current batch. Re-running is safe: upsert is idempotent
        and duplicate chunk_ids are silently skipped by ChromaDB.

        Args:
            chunks:        List of chunk dicts from Chunker.chunk_records().
            show_progress: Log progress every batch.

        Returns:
            Number of chunks successfully inserted/updated this run.
        """
        if not chunks:
            logger.warning("ingest() called with empty chunk list.")
            return 0

        total    = len(chunks)
        inserted = 0
        skipped  = 0

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]

            texts     = [c["text"] for c in batch]
            ids       = [c["chunk_id"] for c in batch]
            metadatas = [
                _sanitize_metadata({k: v for k, v in c.items() if k not in ("text", "chunk_id")})
                for c in batch
            ]

            try:
                # Embed on the configured device (CPU by default)
                embeddings = self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=min(64, len(texts)),
                    show_progress_bar=False,
                ).tolist()

                # Upsert (idempotent — safe to re-run; same chunk_id = update)
                self._collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                inserted += len(batch)

            except Exception as e:
                skipped += len(batch)
                logger.error(
                    f"  Batch {i//self.batch_size + 1} failed "
                    f"(chunks {i}–{i+len(batch)-1}): {e}. "
                    f"Skipping and continuing."
                )
                continue

            if show_progress and (inserted % (self.batch_size * 4) == 0 or inserted == total):
                logger.info(f"  Embedded {inserted}/{total} chunks...")

        if skipped:
            logger.warning(f"  {skipped} chunks skipped due to errors.")
        logger.info(f"Ingestion complete. {inserted} chunks embedded into '{self.collection_name}'.")
        return inserted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = RAG_TOP_K,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-n most semantically similar chunks.

        Args:
            query_text: Free-text query.
            n_results:  Number of results to return.
            where:      Optional ChromaDB metadata filter (e.g. {"source_db": "pubmed"}).

        Returns:
            List of dicts with keys: text, chunk_id, distance, and all metadata fields.
        """
        query_embedding = self._model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        kwargs = dict(
            query_embeddings=query_embedding,
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        if self._collection.count() == 0:
            logger.warning("ChromaDB collection is empty. Run corpus/build_corpus.py first.")
            return []

        results = self._collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            entry = {"text": doc, "distance": dist}
            entry.update(meta)
            output.append(entry)

        return output

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count()

    def stats(self) -> dict:
        """Return a summary dict of the collection."""
        return {
            "collection":  self.collection_name,
            "n_documents": self.count,
            "model":       self.model_name,
            "embed_dim":   EMBED_DIM,
            "chroma_path": str(CHROMA_DIR),
        }
