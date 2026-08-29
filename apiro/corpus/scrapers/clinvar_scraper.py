"""
corpus/scrapers/clinvar_scraper.py
====================================
Downloads and parses ClinVar's variant_summary.txt.gz from the NCBI FTP.
No Entrez credentials, no API rate limits. Single download, local parse.

FTP URL:
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

File: tab-delimited TSV with header row. Relevant columns:
    #AlleleID           → variant_id
    GeneSymbol          → gene
    ClinicalSignificance → significance
    PhenotypeList       → pipe-delimited conditions
    Type                → variant type (SNV, deletion, etc.)
    Assembly            → genome assembly (filter to GRCh38)
    ReviewStatus        → evidence strength

Filters applied:
    - Assembly == "GRCh38" (or "GRCh37" as fallback)
    - ClinicalSignificance contains "Pathogenic" or "Likely pathogenic"
    - GeneSymbol is not empty

Cache: Saves gz file to data/corpus/clinvar_cache/ after first download.

Usage:
    from corpus.scrapers.clinvar_scraper import ClinVarScraper
    scraper = ClinVarScraper()
    records = scraper.fetch(max_records=10_000)

Each record dict:
    {
        "variant_id":    str,
        "gene":          str,
        "condition":     str,
        "significance":  str,
        "variant_type":  str,
        "text":          str,
        "source_db":     "clinvar",
        "medical_domain": "genetics",
    }
"""

import gzip
import logging
from pathlib import Path
from typing import Iterator

import requests

from apiro.config import CORPUS_DIR

logger = logging.getLogger(__name__)

_CLINVAR_FTP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
_CACHE_DIR  = CORPUS_DIR / "clinvar_cache"
_CACHE_FILE = _CACHE_DIR / "variant_summary.txt.gz"

_PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic"}
_TARGET_ASSEMBLIES = {"GRCh38", "GRCh37"}   # prefer GRCh38 but accept GRCh37 as fallback


def _download_gz(url: str, dest: Path) -> None:
    """Stream download a .gz file to disk with progress logging."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading ClinVar variant_summary.txt.gz (~150 MB)...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total     = int(resp.headers.get("Content-Length", 0))
    received  = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            received += len(chunk)
            if total:
                pct = 100 * received / total
                logger.info(f"  {pct:.0f}%  ({received / 1e6:.1f} MB)")
    logger.info(f"Download complete: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _is_pathogenic(significance: str) -> bool:
    """Return True if clinical significance contains any pathogenic label."""
    sig_lower = significance.lower()
    return any(term in sig_lower for term in _PATHOGENIC_TERMS)


class ClinVarScraper:
    """
    Parses ClinVar variant_summary.txt.gz from NCBI FTP.

    Args:
        force_download: Re-download even if cached file exists.
        preferred_assembly: Genome assembly to prefer ("GRCh38" or "GRCh37").
    """

    def __init__(
        self,
        force_download: bool = False,
        preferred_assembly: str = "GRCh38",
    ):
        self.force_download      = force_download
        self.preferred_assembly  = preferred_assembly

    def _ensure_file(self) -> Path:
        """Download if not cached, then return path to the gz file."""
        if _CACHE_FILE.exists() and not self.force_download:
            logger.info(f"Using cached ClinVar file: {_CACHE_FILE}")
            return _CACHE_FILE
        _download_gz(_CLINVAR_FTP_URL, _CACHE_FILE)
        return _CACHE_FILE

    def _parse(self, gz_path: Path) -> Iterator[dict]:
        """
        Stream-parse the gz file and yield pathogenic record dicts.
        Reads line by line to avoid loading the full file into memory.
        """
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
            header = None
            for line in f:
                if line.startswith("#"):
                    # Header line: strip leading '#' and split
                    header = line.lstrip("#").strip().split("\t")
                    continue
                if header is None:
                    continue

                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue

                row = dict(zip(header, parts))

                # Apply filters
                assembly     = row.get("Assembly", "")
                significance = row.get("ClinicalSignificance", "")
                gene         = row.get("GeneSymbol", "").strip()

                if assembly not in _TARGET_ASSEMBLIES:
                    continue
                if not _is_pathogenic(significance):
                    continue
                if not gene or gene in ("-", ""):
                    continue

                # Parse conditions (pipe-delimited)
                pheno_raw   = row.get("PhenotypeList", "")
                conditions  = [p.strip() for p in pheno_raw.split("|") if p.strip()]
                condition   = conditions[0] if conditions else ""
                variant_id  = row.get("AlleleID", row.get("#AlleleID", ""))
                variant_type = row.get("Type", "")
                name        = row.get("Name", "")

                condition_str = "; ".join(conditions[:3]) if conditions else "unknown condition"
                text = (
                    f"ClinVar variant in gene {gene}: {name}. "
                    f"Clinical significance: {significance}. "
                    f"Associated condition(s): {condition_str}. "
                    f"Variant type: {variant_type}."
                )

                yield {
                    "variant_id":    str(variant_id),
                    "gene":          gene,
                    "condition":     condition,
                    "significance":  significance,
                    "variant_type":  variant_type,
                    "text":          text,
                    "source_db":     "clinvar",
                    "medical_domain": "genetics",
                }

    def fetch(self, max_records: int = 20_000) -> list[dict]:
        """
        Download (once) and parse ClinVar variant_summary.txt.gz.

        Args:
            max_records: Maximum pathogenic records to return.

        Returns:
            List of record dicts.
        """
        gz_path = self._ensure_file()

        records: list[dict] = []
        for record in self._parse(gz_path):
            records.append(record)
            if len(records) % 5_000 == 0:
                logger.info(f"  Parsed {len(records):,} pathogenic records...")
            if max_records and len(records) >= max_records:
                break

        logger.info(f"ClinVar FTP parse complete. {len(records):,} pathogenic records.")
        return records
