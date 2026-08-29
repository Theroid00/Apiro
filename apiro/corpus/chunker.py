"""
corpus/chunker.py
==================
Splits raw record texts into overlapping chunks suitable for embedding.

Strategy:
  1. Sentence-tokenize using NLTK (sentence-boundary aware).
  2. Greedily accumulate sentences until chunk reaches ~CHUNK_SIZE_TOKENS.
  3. Slide forward by (CHUNK_SIZE - CHUNK_OVERLAP) tokens for the next chunk.
  4. Each chunk carries the full metadata from its source record.

Token counting is approximated as: n_tokens ≈ n_words * 1.3
(Good enough for chunking; not used for model inference.)

Usage:
    from corpus.chunker import Chunker
    chunker = Chunker()
    chunks = chunker.chunk_records(records)

Each chunk dict:
    {
        "chunk_id":      str,    # "{source_id}_{chunk_index}"
        "text":          str,    # the chunk text
        "source_db":     str,
        "medical_domain": str,
        # ...all other metadata from the source record...
    }
"""

import hashlib
import logging
import ssl
from typing import Sequence

import nltk
import certifi

from apiro.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

logger = logging.getLogger(__name__)

def _ensure_nltk_tokenizer() -> str:
    """
    Ensure an NLTK sentence tokenizer is available.
    Tries punkt_tab first, then punkt. Falls back to ssl-patched download
    on macOS where the default SSL context lacks CA certs.
    Returns the name of the available tokenizer ('punkt_tab' or 'punkt').
    """
    for tok in ("punkt_tab", "punkt"):
        resource = f"tokenizers/{tok}" if tok == "punkt" else f"tokenizers/{tok}"
        try:
            nltk.data.find(resource)
            return tok
        except LookupError:
            pass

    # Not found — try downloading with certifi SSL context (fixes macOS)
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = lambda: ssl_ctx
    try:
        for tok in ("punkt_tab", "punkt"):
            try:
                nltk.download(tok, quiet=True)
                nltk.data.find(f"tokenizers/{tok}")
                logger.info(f"Downloaded NLTK {tok} tokenizer.")
                return tok
            except Exception:
                continue
    finally:
        ssl._create_default_https_context = _orig

    logger.warning(
        "Could not download NLTK tokenizer. "
        "Falling back to naive sentence splitting on '. '"
    )
    return "naive"


_TOKENIZER_NAME: str = _ensure_nltk_tokenizer()


def _approx_tokens(text: str) -> int:
    """Approximate token count from word count (1 word ≈ 1.3 tokens)."""
    return int(len(text.split()) * 1.3)


def _make_chunk_id(source_id: str, index: int) -> str:
    """Deterministic chunk ID based on source ID and position."""
    raw = f"{source_id}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class Chunker:
    """
    Sentence-aware sliding-window chunker.

    Args:
        chunk_size:    Target token count per chunk.
        overlap:       Token overlap between adjacent chunks.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS,
    ):
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.step       = chunk_size - overlap   # tokens to advance per chunk

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks of approximately chunk_size tokens.

        Returns:
            List of chunk strings. Single-sentence texts return a single chunk.
        """
        if not text or not text.strip():
            return []

        if _TOKENIZER_NAME == "naive":
            # Simple fallback: split on ". " boundaries. Only append "." to
            # pieces that don't already end in sentence-ending punctuation —
            # the last piece from a str.split(". ") keeps the original
            # text's own trailing punctuation (or none, mid-sentence), so
            # unconditionally appending "." here used to double it up
            # ("...pain." -> "...pain..") whenever the input already ended
            # with '.', '!', or '?'.
            sentences = [
                s if s[-1] in ".!?" else s + "."
                for s in (piece.strip() for piece in text.split(". "))
                if s
            ]
        else:
            sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens: int = 0

        for sentence in sentences:
            s_tokens = _approx_tokens(sentence)

            # If adding this sentence would exceed chunk_size AND we have content, flush
            if current_tokens + s_tokens > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))

                # Roll back to create overlap: drop sentences from the front
                # until we're within the overlap budget
                while current_sentences and current_tokens > self.overlap:
                    removed = current_sentences.pop(0)
                    current_tokens -= _approx_tokens(removed)

            current_sentences.append(sentence)
            current_tokens += s_tokens

        # Flush any remaining sentences
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks if chunks else [text]

    def chunk_record(self, record: dict, source_id_key: str = "pmid") -> list[dict]:
        """
        Chunk a single record's text field and attach all metadata to each chunk.

        Args:
            record:        Source record dict (must contain "text" key).
            source_id_key: Key to use as the source identifier in chunk_id.

        Returns:
            List of chunk dicts.
        """
        text = record.get("text", "").strip()
        if not text:
            return []

        # Derive a stable source_id from the record
        source_id = str(
            record.get(source_id_key)
            or record.get("pmid")
            or record.get("drugbank_id")
            or record.get("mim_number")
            or record.get("variant_id")
            or "unknown"
        )

        raw_chunks = self.chunk_text(text)
        result = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk = {k: v for k, v in record.items() if k != "text"}
            chunk["chunk_id"]   = _make_chunk_id(source_id, i)
            chunk["text"]       = chunk_text
            chunk["chunk_index"] = i
            chunk["n_chunks"]   = len(raw_chunks)
            result.append(chunk)

        return result

    def chunk_records(self, records: Sequence[dict]) -> list[dict]:
        """
        Chunk all records and return the flat list of chunk dicts.

        Args:
            records: Iterable of source record dicts.

        Returns:
            Flat list of chunk dicts, deduplicated by chunk_id.
        """
        seen: set[str] = set()
        all_chunks: list[dict] = []

        for record in records:
            chunks = self.chunk_record(record)
            for chunk in chunks:
                cid = chunk["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    all_chunks.append(chunk)

        logger.info(
            f"Chunker: {len(records)} records → {len(all_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )
        return all_chunks
