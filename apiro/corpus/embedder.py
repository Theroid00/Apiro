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

Caching
-------
This module maintains two in-memory caches to avoid redundant work during
BFS-style traversal where the same or repeated claims are encoded/queried
many times:

  * self._encode_cache : maps (text, kwargs_signature) -> np.ndarray
        Pure function of (model, text, encode kwargs). Safe to keep across
        ingests because it does not depend on collection contents.

  * self._query_cache  : maps (query_text, n_results, where, collection_count)
                         -> list[dict]
        Depends on collection contents, so it is CLEARED on every ingest()
        that mutates the collection. The collection count is also folded into
        the key as a defensive second layer against stale reads.

Both caches evict oldest entries (FIFO / insertion-order) once they exceed
_CACHE_MAX_ENTRIES.
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from apiro.config import EMBED_MODEL, EMBED_DIM, CHROMA_DIR, CHROMA_COLLECTION, RAG_TOP_K

logger = logging.getLogger(__name__)

# Fields that ChromaDB metadata does NOT accept as non-string types
_ALLOWED_META_TYPES = (str, int, float, bool)

# Maximum number of entries kept in each in-memory cache before eviction.
_CACHE_MAX_ENTRIES = 4096


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


def _stable_kwargs_key(kwargs: dict) -> str:
    """
    Build a deterministic, hashable signature for encode kwargs so that calls
    with different options (e.g. normalize_embeddings) do not collide in the
    cache. Falls back to repr() for any non-JSON-serializable value.
    """
    try:
        return json.dumps(kwargs, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(sorted(kwargs.items(), key=lambda kv: kv[0]))


def _stable_where_key(where: dict | None) -> str:
    """Deterministic signature for a ChromaDB `where` filter (or None)."""
    if where is None:
        return "null"
    try:
        return json.dumps(where, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(where)


def _cache_put(cache: "OrderedDict", key, value) -> None:
    """
    Insert into an insertion-ordered cache and evict the oldest entries once
    the cache grows beyond _CACHE_MAX_ENTRIES. Re-inserting an existing key
    moves it to the most-recently-used position.
    """
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > _CACHE_MAX_ENTRIES:
        # popitem(last=False) removes the oldest inserted entry (FIFO eviction).
        cache.popitem(last=False)


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

        # In-memory caches. OrderedDict gives us cheap FIFO/LRU eviction.
        self._encode_cache: "OrderedDict" = OrderedDict()
        self._query_cache: "OrderedDict" = OrderedDict()

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

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear both the encode and query caches."""
        self._encode_cache.clear()
        self._query_cache.clear()

    def _invalidate_query_cache(self) -> None:
        """
        Drop all cached query results. Called whenever the collection changes,
        because query results depend on collection contents. The encode cache
        is intentionally left intact (it does not depend on the collection).
        """
        if self._query_cache:
            logger.debug(
                f"Invalidating query cache ({len(self._query_cache)} entries) "
                f"after collection mutation."
            )
        self._query_cache.clear()

    def _encode_single_cached(self, text: str, **kwargs) -> np.ndarray:
        """
        Encode a single text string with caching. Returns a 1-D np.ndarray.
        Returns a COPY of the cached vector so callers cannot mutate the
        cached value in place.
        """
        key = (text, _stable_kwargs_key(kwargs))
        cached = self._encode_cache.get(key)
        if cached is not None:
            self._encode_cache.move_to_end(key)
            return cached.copy()

        vec = self._model.encode([text], **kwargs)
        vec = np.asarray(vec)[0]
        _cache_put(self._encode_cache, key, vec)
        return vec.copy()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts, **kwargs):
        """
        Forward encoding calls to the underlying SentenceTransformer model.

        Uses an in-memory cache keyed on (text, kwargs) so that repeated or
        near-duplicate claims during BFS traversal do not re-run the
        SentenceTransformer.

        Return type matches the underlying model:
          - A single string input returns a 1-D np.ndarray.
          - A sequence of strings returns a 2-D np.ndarray (one row per input).
        Cached vectors are returned as copies to prevent aliasing bugs.
        """
        # Single-string input: return a 1-D array, matching SentenceTransformer.
        if isinstance(texts, str):
            return self._encode_single_cached(texts, **kwargs)

        # Sequence input: encode each element through the single-item cache,
        # then stack into a 2-D array (matching SentenceTransformer's output).
        materialized = list(texts)
        if not materialized:
            # Preserve ndarray return type for empty input.
            return np.empty((0,))

        vectors = [self._encode_single_cached(t, **kwargs) for t in materialized]
        return np.vstack(vectors)

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
                # Embed on the configured device (CPU by default).
                # Ingestion uses the model directly (not the cache): these are
                # fresh corpus texts, so caching them wastes memory.
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

        # The collection contents changed: any cached query results are now
        # potentially stale, so drop them. Encode cache stays warm.
        if inserted:
            self._invalidate_query_cache()

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

        Results are cached in memory keyed on
        (query_text, n_results, where, collection_count). The cache is cleared
        on every ingest() that mutates the collection; the collection count is
        also folded into the key as a defensive guard against stale reads.

        Args:
            query_text: Free-text query.
            n_results:  Number of results to return.
            where:      Optional ChromaDB metadata filter (e.g. {"source_db": "pubmed"}).

        Returns:
            List of dicts with keys: text, chunk_id, distance, and all metadata fields.
        """
        collection_count = self._collection.count()

        if collection_count == 0:
            logger.warning("ChromaDB collection is empty. Run corpus/build_corpus.py first.")
            return []

        cache_key = (
            query_text,
            n_results,
            _stable_where_key(where),
            collection_count,
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            # Return a shallow-copied list of shallow-copied dicts so callers
            # cannot mutate the cached result in place.
            return [dict(entry) for entry in cached]

        # Reuse the encode cache for the query text (BFS often re-queries the
        # same claim text). encode() returns a 1-D ndarray for a str input.
        query_embedding = self.encode(
            query_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).reshape(1, -1).tolist()

        kwargs = dict(
            query_embeddings=query_embedding,
            n_results=min(n_results, collection_count),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

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

        # Cache a defensive copy so a later mutation of `output` by a caller
        # does not corrupt the cache.
        _cache_put(self._query_cache, cache_key, [dict(entry) for entry in output])

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