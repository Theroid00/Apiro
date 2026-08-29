"""
corpus/scrapers/textbooks_scraper.py
=======================================
Scrapes the MedRAG/textbooks dataset from HuggingFace.

This dataset contains pre-chunked excerpts from curated medical textbooks:
  - Harrison's Principles of Internal Medicine
  - Oxford Handbook of Clinical Medicine
  - Davidson's Principles and Practice of Medicine
  - StatPearls (continuously updated NCBI clinical summaries)

These are much more clinically useful than raw PubMed abstracts because:
  1. They are organized by disease/condition (not lab methodology)
  2. They use modern diagnostic criteria (post-2010)
  3. They cover differential diagnosis explicitly
  4. They are the exact type of knowledge a clinician uses in reasoning

Dataset: https://huggingface.co/datasets/MedRAG/textbooks
Fields per record:
    id       — unique chunk ID
    title    — textbook name + chapter
    content  — text chunk (~200 tokens)
    contents — title + content concatenated

Usage:
    from apiro.corpus.scrapers.textbooks_scraper import TextbooksScraper
    scraper = TextbooksScraper(max_records=50_000)
    records = scraper.fetch()
"""

import logging

from apiro.corpus.scrapers.medrag_scraper import (
    _guess_domain,
    _extract_condition_tags,
)

logger = logging.getLogger(__name__)


class TextbooksScraper:
    """
    Streams MedRAG/textbooks from HuggingFace.

    Args:
        max_records:  Maximum records to return (0 = all available).
        split:        Dataset split. MedRAG/textbooks has only 'train'.
    """

    DATASET_NAME = "MedRAG/textbooks"

    def __init__(self, max_records: int = 50_000, split: str = "train"):
        self.max_records = max_records
        self.split       = split

    def fetch(self) -> list[dict]:
        """
        Stream MedRAG/textbooks and return chunk dicts for Embedder.ingest().

        Returns:
            List of pre-chunked record dicts compatible with the Apiro corpus schema.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace `datasets` library not installed. "
                "Run: pip install datasets"
            )

        logger.info(
            f"Streaming {self.DATASET_NAME} from HuggingFace "
            f"(max_records={self.max_records or 'unlimited'})..."
        )

        dataset = load_dataset(
            self.DATASET_NAME,
            split=self.split,
            streaming=True,
            trust_remote_code=False,
        )

        records: list[dict] = []
        skipped_empty = 0

        for i, example in enumerate(dataset):
            text  = example.get("contents") or example.get("content", "")
            title = example.get("title", "")
            uid   = str(example.get("id", f"textbook_{i:08d}"))

            if not text.strip():
                skipped_empty += 1
                continue

            records.append({
                "chunk_id":       f"tb_{uid}",
                "pmid":           "",         # no PMID for textbooks
                "title":          title,
                "text":           text,
                "source_db":      "medrag_textbooks",
                "medical_domain": _guess_domain(text),
                "condition_tags": _extract_condition_tags(text),
                "evidence_level": 3,  # textbook = authoritative, evidence level 3
                "chunk_index":    0,
                "n_chunks":       1,
            })

            if len(records) % 5_000 == 0 and len(records) > 0:
                logger.info(f"  Collected {len(records):,} textbook records...")

            if self.max_records and len(records) >= self.max_records:
                break

        logger.info(
            f"Textbooks fetch complete. "
            f"{len(records):,} records collected "
            f"({skipped_empty} empty skipped)."
        )
        return records
