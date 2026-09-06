"""
tests/test_corpus.py
=====================
Unit tests for the Phase 1 corpus pipeline:
    - Chunker (chunk_text, chunk_record, chunk_records)
    - Embedder (ingest, query, stats) — uses an in-memory ChromaDB client

Run with:
    python -m pytest tests/test_corpus.py -v

Note: These tests do NOT make real network calls to PubMed/OMIM/ClinVar/DrugBank.
Scraper integration tests are run manually with real credentials.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from apiro.corpus.chunker import Chunker, _approx_tokens, _make_chunk_id


@pytest.fixture(autouse=True)
def _stub_sentence_transformers(monkeypatch):
    """
    apiro/corpus/embedder.py does `from sentence_transformers import
    SentenceTransformer` at module level, which would otherwise pull in
    torch/transformers just to import the module under test. Stub the
    module via monkeypatch (not a bare `sys.modules[...] = ...`
    assignment) so pytest reverts it after each test — a bare module-level
    assignment here previously leaked into every other test file that
    imports later in the same pytest session (alphabetically, most of
    them), silently turning unrelated sentence-transformers-dependent
    code paths in other test files into MagicMock-driven no-ops instead
    of the "unavailable, skip" behaviour they were designed to fall back to.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", MagicMock())


# ============================================================
# Chunker tests
# ============================================================

class TestChunker:
    SHORT_TEXT  = "The patient has chest pain."
    MEDIUM_TEXT = (
        "The patient presents with acute chest pain radiating to the left arm. "
        "Physical examination reveals diaphoresis and tachycardia. "
        "Troponin levels are elevated at 3.2 ng/mL. "
        "An ECG shows ST-segment elevation in leads II, III, and aVF. "
        "The patient is taken for emergency coronary angiography."
    )
    LONG_TEXT = " ".join([
        f"Sentence number {i} describes a clinical finding relevant to the diagnosis."
        for i in range(200)
    ])

    def test_short_text_single_chunk(self):
        chunker = Chunker(chunk_size=300, overlap=50)
        chunks  = chunker.chunk_text(self.SHORT_TEXT)
        assert len(chunks) == 1
        assert chunks[0] == self.SHORT_TEXT

    def test_medium_text_single_chunk(self):
        chunker = Chunker(chunk_size=300, overlap=50)
        chunks  = chunker.chunk_text(self.MEDIUM_TEXT)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

    def test_long_text_multiple_chunks(self):
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks  = chunker.chunk_text(self.LONG_TEXT)
        assert len(chunks) > 1

    def test_empty_text_returns_empty(self):
        chunker = Chunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_chunk_record_attaches_metadata(self):
        chunker = Chunker(chunk_size=300)
        record  = {
            "pmid":           "12345",
            "title":          "Test study",
            "abstract":       self.MEDIUM_TEXT,
            "text":           self.MEDIUM_TEXT,
            "source_db":      "pubmed",
            "medical_domain": "pathophysiology",
        }
        chunks = chunker.chunk_record(record, source_id_key="pmid")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "chunk_id"       in chunk
            assert "text"           in chunk
            assert "source_db"      in chunk
            assert "medical_domain" in chunk
            assert "pmid"           in chunk
            assert chunk["source_db"] == "pubmed"
            # Original text key replaced by chunk text
            assert chunk["text"] != record["abstract"] or len(chunks) == 1

    def test_chunk_record_empty_text_returns_empty(self):
        chunker = Chunker()
        record  = {"text": "", "source_db": "pubmed", "medical_domain": "lab"}
        assert chunker.chunk_record(record) == []

    def test_chunk_records_deduplication(self):
        chunker = Chunker(chunk_size=300)
        record  = {
            "pmid": "99999", "text": self.SHORT_TEXT,
            "source_db": "pubmed", "medical_domain": "lab",
        }
        # Passing the same record twice should deduplicate by chunk_id
        chunks = chunker.chunk_records([record, record])
        assert len(chunks) == 1

    def test_chunk_records_multiple_sources(self):
        chunker = Chunker(chunk_size=300)
        records = [
            {"pmid": "1", "text": self.MEDIUM_TEXT, "source_db": "pubmed", "medical_domain": "lab"},
            {"pmid": "2", "text": self.MEDIUM_TEXT, "source_db": "pubmed", "medical_domain": "lab"},
        ]
        chunks = chunker.chunk_records(records)
        # Should have at least 2 distinct chunks (one per record minimum)
        assert len(chunks) >= 2
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids should be unique"

    def test_chunk_index_present(self):
        chunker = Chunker(chunk_size=50, overlap=10)
        record  = {"pmid": "X", "text": self.LONG_TEXT, "source_db": "pubmed", "medical_domain": "lab"}
        chunks  = chunker.chunk_record(record)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert chunk["n_chunks"] == len(chunks)

    def test_approx_tokens_nonempty(self):
        tokens = _approx_tokens("hello world foo bar")
        assert tokens > 0

    def test_make_chunk_id_deterministic(self):
        cid1 = _make_chunk_id("pmid_123", 0)
        cid2 = _make_chunk_id("pmid_123", 0)
        cid3 = _make_chunk_id("pmid_123", 1)
        assert cid1 == cid2        # deterministic
        assert cid1 != cid3        # different index → different ID
        assert len(cid1) == 12     # 12-char hex


# ============================================================
# Embedder tests (mocked ChromaDB + SentenceTransformer)
# ============================================================

class TestEmbedder:
    """
    These tests mock ChromaDB and SentenceTransformer to avoid:
      - Network downloads
      - Disk I/O
      - GPU/CPU inference overhead

    They test the Embedder's logic: batching, metadata sanitization,
    query dispatch, and stats.
    """

    def _make_embedder(self):
        """Return an Embedder with mocked internals."""
        with (
            patch("apiro.corpus.embedder.SentenceTransformer") as mock_st,
            patch("apiro.corpus.embedder.chromadb.PersistentClient") as mock_chroma,
        ):
            # Set up mock encoder
            import numpy as np
            mock_model_instance = MagicMock()
            mock_model_instance.encode.return_value = np.zeros((1, 768))
            mock_st.return_value = mock_model_instance

            # Set up mock collection
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            from collections import OrderedDict
            from apiro.corpus.embedder import Embedder
            embedder = Embedder.__new__(Embedder)
            embedder.model_name      = "all-mpnet-base-v2"
            embedder.collection_name = "test_collection"
            embedder.batch_size      = 256
            embedder._model          = mock_model_instance
            embedder._collection     = mock_collection
            embedder._client         = mock_client
            # Mirrors __init__'s in-memory caches — a bare __new__() skips
            # __init__ entirely, so these must be set by hand or any method
            # that touches the cache (ingest, query) raises AttributeError.
            embedder._encode_cache   = OrderedDict()
            embedder._query_cache    = OrderedDict()
            return embedder

    def _sample_chunks(self, n: int = 3) -> list[dict]:
        return [
            {
                "chunk_id":      f"chunk_{i:04d}",
                "text":          f"Sample medical text chunk number {i}.",
                "source_db":     "pubmed",
                "medical_domain": "pathophysiology",
                "pmid":          str(1000 + i),
                "chunk_index":   i,
                "n_chunks":      n,
            }
            for i in range(n)
        ]

    def test_ingest_calls_upsert(self):
        import numpy as np
        embedder   = self._make_embedder()
        chunks     = self._sample_chunks(3)
        embedder._model.encode.return_value = np.zeros((3, 768))
        embedder._collection.count.return_value = 0

        result = embedder.ingest(chunks, show_progress=False)

        assert result == 3
        embedder._collection.upsert.assert_called_once()
        call_kwargs = embedder._collection.upsert.call_args[1]
        assert len(call_kwargs["ids"]) == 3

    def test_ingest_empty_chunks(self):
        embedder = self._make_embedder()
        result   = embedder.ingest([], show_progress=False)
        assert result == 0
        embedder._collection.upsert.assert_not_called()

    def test_query_returns_list(self):
        import numpy as np
        embedder = self._make_embedder()
        embedder._model.encode.return_value = np.zeros((1, 768))
        embedder._collection.count.return_value = 5
        embedder._collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [
                [{"source_db": "pubmed", "medical_domain": "lab"}] * 2
            ],
            "distances": [[0.1, 0.2]],
        }

        results = embedder.query("chest pain", n_results=2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["text"] == "doc1"
        assert results[0]["distance"] == 0.1
        assert results[0]["source_db"] == "pubmed"

    def test_query_empty_collection_returns_empty(self):
        embedder = self._make_embedder()
        embedder._collection.count.return_value = 0
        results = embedder.query("anything")
        assert results == []

    def test_stats_dict_keys(self):
        embedder = self._make_embedder()
        embedder._collection.count.return_value = 42
        stats = embedder.stats()
        assert "collection"  in stats
        assert "n_documents" in stats
        assert "model"       in stats
        assert stats["n_documents"] == 42

    def test_metadata_sanitization(self):
        """Lists and None values in metadata should be coerced to strings."""
        from apiro.corpus.embedder import _sanitize_metadata
        raw = {
            "pmid":    "12345",
            "authors": ["Smith J", "Doe J"],   # list → should become string
            "count":   10,
            "flag":    True,
            "nothing": None,
        }
        clean = _sanitize_metadata(raw)
        assert isinstance(clean["authors"], str)
        assert "Smith J" in clean["authors"]
        assert clean["count"] == 10
        assert clean["flag"] is True
        assert clean["nothing"] == ""
