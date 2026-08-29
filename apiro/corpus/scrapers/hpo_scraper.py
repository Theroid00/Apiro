"""
corpus/scrapers/hpo_scraper.py
================================
Fetches phenotype-gene-disease associations from the Human Phenotype
Ontology (HPO). No API key required.

Downloads two files from the HPO GitHub releases:
  1. genes_to_phenotype.txt  — gene → HPO term mappings
  2. phenotype.hpoa          — disease → HPO term mappings (with DiseaseName)

Both files are cached to data/corpus/ after first download.

Usage:
    from corpus.scrapers.hpo_scraper import HPOScraper
    scraper = HPOScraper()
    records = scraper.fetch()

Each record dict:
    {
        "hpo_id":        str,    # e.g. "HP:0001250"
        "hpo_name":      str,    # e.g. "Seizure"
        "gene_symbol":   str,    # e.g. "SCN1A"
        "disease_name":  str,    # e.g. "Dravet syndrome"
        "database_id":   str,    # e.g. "OMIM:607208"
        "text":          str,    # combined text for chunking
        "source_db":     "hpo",
        "medical_domain": "genetics",
    }
"""

import gzip
import logging
from pathlib import Path

import requests

from apiro.config import CORPUS_DIR

logger = logging.getLogger(__name__)

# HPO official persistent URLs (via GitHub releases)
_HPO_BASE = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download"
GENES_TO_PHENOTYPE_URL = f"{_HPO_BASE}/genes_to_phenotype.txt"
PHENOTYPE_HPOA_URL     = f"{_HPO_BASE}/phenotype.hpoa"

_CACHE_DIR = CORPUS_DIR / "hpo_cache"


def _download(url: str, cache_path: Path, force: bool = False) -> str:
    """
    Download a file if not cached. Returns file contents as a string.
    """
    if cache_path.exists() and not force:
        logger.info(f"  Using cached file: {cache_path.name}")
        return cache_path.read_text(encoding="utf-8", errors="replace")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    # Handle gzip transparently
    content = resp.content
    if url.endswith(".gz") or resp.headers.get("Content-Type", "").startswith("application/gzip"):
        content = gzip.decompress(content)

    text = content.decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    logger.info(f"  Saved {cache_path.name} ({len(text):,} chars)")
    return text


class HPOScraper:
    """
    Builds phenotype-gene-disease records from HPO annotation files.

    Args:
        force_download: Re-download even if cache files exist.
        max_records:    Maximum records to return (0 = unlimited).
    """

    def __init__(self, force_download: bool = False, max_records: int = 0):
        self.force_download = force_download
        self.max_records    = max_records

    def _parse_genes_to_phenotype(self, text: str) -> dict[str, str]:
        """
        Parse genes_to_phenotype.txt.
        Returns a dict: hpo_id → hpo_name (for cross-referencing).
        Also returns a list of (gene_symbol, hpo_id, hpo_name, disease_id) tuples.

        File columns (tab-separated, first line is header):
          ncbi_gene_id | gene_symbol | hpo_id | hpo_name | frequency | disease_id
        """
        hpo_id_to_name: dict[str, str] = {}
        rows: list[dict] = []

        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # Skip the header line
            if line.startswith("ncbi_gene_id") or line.startswith("entrez_id"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            gene_symbol = parts[1].strip()
            hpo_id      = parts[2].strip()
            hpo_name    = parts[3].strip()
            disease_id  = parts[5].strip() if len(parts) > 5 else ""

            hpo_id_to_name[hpo_id] = hpo_name
            rows.append({
                "gene_symbol": gene_symbol,
                "hpo_id":      hpo_id,
                "hpo_name":    hpo_name,
                "disease_id":  disease_id,
            })

        logger.info(f"  genes_to_phenotype: {len(rows)} gene-phenotype rows parsed")
        return rows, hpo_id_to_name

    def _parse_phenotype_hpoa(self, text: str) -> list[dict]:
        """
        Parse phenotype.hpoa for disease-phenotype associations.

        File columns (tab-separated, lines starting with '#' are comments):
          DatabaseID | DiseaseName | Qualifier | HPO_ID | DB_Reference |
          Evidence | Onset | Frequency | Sex | Modifier | Aspect | BiocurationBy

        Only keeps lines where Aspect == 'P' (phenotypic abnormality)
        and Qualifier is NOT 'NOT'.
        """
        rows: list[dict] = []

        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # Skip header
            if line.startswith("DatabaseID"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue

            database_id  = parts[0].strip()
            disease_name = parts[1].strip()
            qualifier    = parts[2].strip()
            hpo_id       = parts[3].strip()
            aspect       = parts[10].strip() if len(parts) > 10 else ""

            # Skip negated and non-phenotype rows
            if qualifier == "NOT" or aspect != "P":
                continue

            rows.append({
                "database_id":  database_id,
                "disease_name": disease_name,
                "hpo_id":       hpo_id,
            })

        logger.info(f"  phenotype.hpoa: {len(rows)} disease-phenotype rows parsed")
        return rows

    def fetch(self) -> list[dict]:
        """
        Download, parse, and join HPO annotation files.
        Produces one record per unique (gene, hpo_term, disease) triple.

        Returns:
            List of record dicts ready for Chunker.chunk_records().
        """
        # Download both files
        g2p_text = _download(
            GENES_TO_PHENOTYPE_URL,
            _CACHE_DIR / "genes_to_phenotype.txt",
            self.force_download,
        )
        hpoa_text = _download(
            PHENOTYPE_HPOA_URL,
            _CACHE_DIR / "phenotype.hpoa",
            self.force_download,
        )

        gene_rows, hpo_id_to_name = self._parse_genes_to_phenotype(g2p_text)
        disease_rows              = self._parse_phenotype_hpoa(hpoa_text)

        # Build disease lookup: hpo_id → list of disease_names
        hpo_to_diseases: dict[str, list[str]] = {}
        for row in disease_rows:
            hpo_to_diseases.setdefault(row["hpo_id"], []).append(row["disease_name"])

        # Build one record per gene-phenotype row, enriched with disease context
        records: list[dict] = []
        seen: set[str] = set()

        for row in gene_rows:
            hpo_id      = row["hpo_id"]
            hpo_name    = row["hpo_name"]
            gene_symbol = row["gene_symbol"]
            disease_id  = row["disease_id"]

            # Get disease names for this HPO term
            diseases = hpo_to_diseases.get(hpo_id, [])
            disease_name = diseases[0] if diseases else disease_id

            # Deduplicate by (gene, hpo_id)
            key = f"{gene_symbol}|{hpo_id}"
            if key in seen:
                continue
            seen.add(key)

            # Build human-readable text for RAG
            disease_str = "; ".join(diseases[:3]) if diseases else disease_id
            text = (
                f"Gene {gene_symbol} is associated with the phenotype '{hpo_name}' "
                f"(HPO: {hpo_id}). "
                f"Related disease(s): {disease_str}."
            )

            records.append({
                "hpo_id":        hpo_id,
                "hpo_name":      hpo_name,
                "gene_symbol":   gene_symbol,
                "disease_name":  disease_name,
                "database_id":   disease_id,
                "text":          text,
                "source_db":     "hpo",
                "medical_domain": "genetics",
            })

            if self.max_records and len(records) >= self.max_records:
                break

        logger.info(f"HPO scrape complete. {len(records)} unique gene-phenotype records.")
        return records
