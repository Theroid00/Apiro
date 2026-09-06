"""
corpus/build_corpus.py
=======================
Orchestrator CLI for the Apiro corpus pipeline.

Sources:
    medrag    — HuggingFace MedRAG/pubmed (pre-chunked, no credentials)
    hpo       — Human Phenotype Ontology (free download, no key)
    clinvar   — ClinVar FTP variant_summary.txt.gz (pathogenic variants)
    openfda   — OpenFDA + ChEMBL drug mechanism/indication records (no key)

Pipeline per source:
    1. Scrape → raw record dicts
    2. Chunk  → overlapping text chunks with metadata
               (MedRAG records are pre-chunked and skip this step)
    3. Embed  → ChromaDB persistent collection

Usage:
    python corpus/build_corpus.py --sources medrag
    python corpus/build_corpus.py --sources medrag hpo clinvar openfda
    python corpus/build_corpus.py --sources medrag --max-records 100000

Environment variables (none required for default sources):
    NCBI_EMAIL   — not required (ClinVar uses FTP, not Entrez)
"""

import argparse
import json
import logging
import time

from apiro.config import DATA_DIR
from apiro.corpus.chunker import Chunker
from apiro.corpus.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_corpus")

VALID_SOURCES = ["medrag", "textbooks", "hpo", "clinvar", "openfda"]

# Sources whose records are already chunked and skip corpus/chunker.py
PRE_CHUNKED_SOURCES = {"medrag", "textbooks"}


def scrape(source: str, max_records: int) -> list[dict]:
    """Dispatch to the correct scraper and return raw records."""

    if source == "medrag":
        from apiro.corpus.scrapers.medrag_scraper import MedRAGScraper
        return MedRAGScraper(max_records=max_records).fetch()

    elif source == "textbooks":
        from apiro.corpus.scrapers.textbooks_scraper import TextbooksScraper
        return TextbooksScraper(max_records=max_records).fetch()

    elif source == "hpo":
        from apiro.corpus.scrapers.hpo_scraper import HPOScraper
        return HPOScraper(max_records=max_records).fetch()

    elif source == "clinvar":
        from apiro.corpus.scrapers.clinvar_scraper import ClinVarScraper
        return ClinVarScraper().fetch(max_records=max_records)

    elif source == "openfda":
        from apiro.corpus.scrapers.openfda_chembl_scraper import OpenFDAChEMBLScraper
        return OpenFDAChEMBLScraper(
            max_openfda=min(max_records, 500),
            max_chembl=min(max_records, 1000),
        ).fetch()

    else:
        raise ValueError(f"Unknown source: {source!r}. Must be one of {VALID_SOURCES}")


def build(sources: list[str], max_records: int, clear: bool = False) -> dict:
    """
    Run the full pipeline: [optional clear] → scrape → chunk (if needed) → embed → stats.

    Args:
        sources:     List of source names to ingest.
        max_records: Max records per source.
        clear:       If True, delete the existing ChromaDB collection before
                     building. Use this when switching corpus composition to
                     avoid mixing old and new documents.

    Returns:
        corpus_stats dict (also written to data/corpus_stats.json).
    """
    chunker  = Chunker()
    embedder = Embedder()

    if clear:
        logger.warning(
            "--clear specified: deleting existing ChromaDB collection before rebuild. "
            "This cannot be undone."
        )
        from apiro.config import CHROMA_DIR, CHROMA_COLLECTION
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(CHROMA_COLLECTION)
            logger.info(f"  Deleted collection '{CHROMA_COLLECTION}'.")
        except Exception as e:
            logger.warning(f"  Could not delete collection (may not exist): {e}")
        # Re-create Embedder so it creates a fresh collection
        embedder = Embedder()

    stats: dict = {
        "sources":        {},
        "total_records":  0,
        "total_chunks":   0,
        "total_embedded": 0,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for source in sources:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Source: {source.upper()}")
        logger.info(f"{'='*55}")

        t0 = time.time()
        try:
            records = scrape(source, max_records)
        except Exception as e:
            logger.error(f"Scraping {source} failed: {e}")
            stats["sources"][source] = {"error": str(e)}
            continue

        n_records = len(records)
        logger.info(f"  Scraped {n_records:,} records from {source}")

        # MedRAG records are pre-chunked — use them directly
        if source in PRE_CHUNKED_SOURCES:
            chunks   = records
            n_chunks = len(chunks)
            logger.info(f"  Pre-chunked: {n_chunks:,} chunks (no chunker step)")
        else:
            chunks   = chunker.chunk_records(records)
            n_chunks = len(chunks)
            logger.info(f"  Chunked into {n_chunks:,} chunks")

        n_embedded = embedder.ingest(chunks)
        elapsed    = round(time.time() - t0, 1)

        stats["sources"][source] = {
            "n_records":   n_records,
            "n_chunks":    n_chunks,
            "n_embedded":  n_embedded,
            "pre_chunked": source in PRE_CHUNKED_SOURCES,
            "elapsed_s":   elapsed,
        }
        stats["total_records"]  += n_records
        stats["total_chunks"]   += n_chunks
        stats["total_embedded"] += n_embedded

        logger.info(f"  Done in {elapsed}s")

    stats["collection_stats"] = embedder.stats()

    stats_path = DATA_DIR / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nCorpus stats → {stats_path}")
    logger.info(
        f"Total: {stats['total_records']:,} records, "
        f"{stats['total_chunks']:,} chunks, "
        f"{stats['total_embedded']:,} embedded."
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build the Apiro biomedical corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
  medrag   — HuggingFace MedRAG/pubmed (pre-chunked, no credentials needed)
  hpo      — Human Phenotype Ontology (free download, no key)
  clinvar  — ClinVar FTP pathogenic variants (~150 MB download, cached)
  openfda  — OpenFDA drug labels + ChEMBL mechanisms (no key)

Examples:
  python corpus/build_corpus.py --sources medrag
  python corpus/build_corpus.py --sources medrag hpo clinvar openfda
  python corpus/build_corpus.py --sources medrag --max-records 50000
        """,
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=False,
        help=(
            "Delete the existing ChromaDB collection before building. "
            "Use when switching corpus composition (e.g. replacing 1970s papers "
            "with modern textbooks). Cannot be undone."
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=VALID_SOURCES,
        default=["textbooks"],
        help=f"Which sources to scrape (default: textbooks). Options: {VALID_SOURCES}",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=100_000,
        help="Max records per source (default: 100000).",
    )
    args = parser.parse_args()

    logger.info(f"Building corpus from: {args.sources}")
    logger.info(f"Max records per source: {args.max_records:,}")
    if args.clear:
        logger.warning("--clear flag set: existing corpus will be deleted first.")

    stats = build(sources=args.sources, max_records=args.max_records, clear=args.clear)

    print("\n✅ Corpus build complete.")
    print(f"   Total records : {stats['total_records']:,}")
    print(f"   Total chunks  : {stats['total_chunks']:,}")
    print(f"   Total embedded: {stats['total_embedded']:,}")
    print(f"   ChromaDB docs : {stats['collection_stats']['n_documents']:,}")


if __name__ == "__main__":
    main()
