"""
corpus/scrapers/openfda_chembl_scraper.py
==========================================
Fetches pharmacology records from two complementary free APIs:

  1. **OpenFDA** (https://api.fda.gov/drug/label.json)
     - Drug labeling: indications_and_usage, mechanism_of_action,
       warnings, adverse_reactions
     - No API key required (1,000 req/day without key)
     - Key: drug brand/generic name, SPL structured fields

  2. **ChEMBL** (https://www.ebi.ac.uk/chembl/api/data/)
     - Mechanism of action, target, action_type
     - No API key required
     - Key: molecule_chembl_id, mechanism_of_action, target_pref_name,
       action_type

Records from both APIs are combined and deduplicated by drug name.

Usage:
    from corpus.scrapers.openfda_chembl_scraper import OpenFDAChEMBLScraper
    scraper = OpenFDAChEMBLScraper()
    records = scraper.fetch()

Each record dict:
    {
        "drug_name":        str,
        "mechanism":        str,
        "indications":      str,
        "adverse_reactions": str,
        "target":           str,
        "action_type":      str,
        "text":             str,    # combined text for chunking
        "source_db":        "openfda_chembl",
        "medical_domain":   "pharmacology",
    }
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_OPENFDA_URL = "https://api.fda.gov/drug/label.json"
_CHEMBL_URL  = "https://www.ebi.ac.uk/chembl/api/data"


def _get(url: str, params: dict, timeout: int = 15, retries: int = 3) -> Optional[dict]:
    """GET with retry and exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.debug(f"GET {url} failed (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


class OpenFDAChEMBLScraper:
    """
    Fetches pharmacology records from OpenFDA and ChEMBL.

    Args:
        max_openfda:   Max drug label records from OpenFDA (batched).
        max_chembl:    Max mechanism records from ChEMBL (paginated).
        sleep_sec:     Seconds between API requests.
    """

    def __init__(
        self,
        max_openfda: int = 500,
        max_chembl: int = 1000,
        sleep_sec: float = 0.4,
    ):
        self.max_openfda = max_openfda
        self.max_chembl  = max_chembl
        self.sleep_sec   = sleep_sec

    # ------------------------------------------------------------------
    # OpenFDA
    # ------------------------------------------------------------------

    def _fetch_openfda_batch(self, skip: int, limit: int) -> list[dict]:
        """
        Fetch one page of drug labels from OpenFDA.
        Filters for records that have at least one of: mechanism_of_action
        or indications_and_usage.
        """
        params = {
            "search": "_exists_:mechanism_of_action",
            "limit":  min(limit, 100),   # OpenFDA max per request
            "skip":   skip,
        }
        data = _get(_OPENFDA_URL, params)
        if not data or "results" not in data:
            return []

        records = []
        for result in data["results"]:
            openfda  = result.get("openfda", {})
            name     = (
                openfda.get("brand_name", [None])[0]
                or openfda.get("generic_name", [None])[0]
                or ""
            )
            if not name:
                continue

            mechanism  = " ".join(result.get("mechanism_of_action", []))
            indications = " ".join(result.get("indications_and_usage", []))
            adverse     = " ".join(result.get("adverse_reactions", []))
            warnings    = " ".join(result.get("warnings", []))

            # Build RAG text — keep it focused on clinical utility
            parts = [f"{name}."]
            if mechanism:
                parts.append(f"Mechanism of action: {mechanism[:500]}")
            if indications:
                parts.append(f"Indications: {indications[:400]}")
            if adverse:
                parts.append(f"Adverse reactions: {adverse[:300]}")
            text = " ".join(parts)

            records.append({
                "drug_name":         name,
                "mechanism":         mechanism[:500],
                "indications":       indications[:400],
                "adverse_reactions": adverse[:300],
                "warnings":          warnings[:300],
                "target":            "",
                "action_type":       "",
                "text":              text,
                "source_db":         "openfda",
                "medical_domain":    "pharmacology",
            })
        return records

    def _fetch_all_openfda(self) -> list[dict]:
        """Paginate OpenFDA to collect up to max_openfda records."""
        all_records: list[dict] = []
        batch_size = 100

        for skip in range(0, self.max_openfda, batch_size):
            batch = self._fetch_openfda_batch(skip=skip, limit=batch_size)
            if not batch:
                break
            all_records.extend(batch)
            logger.info(f"  OpenFDA: {len(all_records)} records fetched...")
            time.sleep(self.sleep_sec)
            if len(all_records) >= self.max_openfda:
                break

        return all_records[:self.max_openfda]

    # ------------------------------------------------------------------
    # ChEMBL
    # ------------------------------------------------------------------

    def _fetch_chembl_page(self, offset: int, limit: int = 100) -> list[dict]:
        """Fetch one page of mechanism records from ChEMBL."""
        params = {
            "format": "json",
            "limit":  limit,
            "offset": offset,
        }
        data = _get(f"{_CHEMBL_URL}/mechanism.json", params)
        if not data:
            return []

        records = []
        for item in data.get("mechanisms", []):
            mol_id   = item.get("molecule_chembl_id", "")
            moa      = item.get("mechanism_of_action", "")
            target   = item.get("target_pref_name", "")
            action   = item.get("action_type", "")
            mol_name = item.get("molecule_name", "") or mol_id

            if not moa:
                continue

            text = (
                f"{mol_name}. "
                f"Mechanism of action: {moa}. "
                f"Target: {target}. "
                f"Action type: {action}."
            )

            records.append({
                "drug_name":         mol_name,
                "chembl_id":         mol_id,
                "mechanism":         moa,
                "indications":       "",
                "adverse_reactions": "",
                "target":            target,
                "action_type":       action,
                "text":              text,
                "source_db":         "chembl",
                "medical_domain":    "pharmacology",
            })
        return records

    def _fetch_all_chembl(self) -> list[dict]:
        """Paginate ChEMBL mechanism endpoint."""
        all_records: list[dict] = []
        limit = 100

        for offset in range(0, self.max_chembl, limit):
            batch = self._fetch_chembl_page(offset=offset, limit=limit)
            if not batch:
                break
            all_records.extend(batch)
            logger.info(f"  ChEMBL: {len(all_records)} records fetched...")
            time.sleep(self.sleep_sec)

        return all_records[:self.max_chembl]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self) -> list[dict]:
        """
        Fetch from both OpenFDA and ChEMBL, merge, and deduplicate by drug_name.

        Returns:
            Combined list of pharmacology record dicts.
        """
        logger.info("Fetching OpenFDA drug labels...")
        openfda_records = self._fetch_all_openfda()

        logger.info("Fetching ChEMBL mechanism of action data...")
        chembl_records  = self._fetch_all_chembl()

        # Merge: prefer OpenFDA records (richer text), supplement with ChEMBL
        seen_names: set[str] = set()
        merged: list[dict]   = []

        for rec in openfda_records + chembl_records:
            name_key = rec["drug_name"].lower().strip()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            merged.append(rec)

        logger.info(
            f"OpenFDA+ChEMBL complete. "
            f"{len(openfda_records)} FDA + {len(chembl_records)} ChEMBL "
            f"→ {len(merged)} deduplicated records."
        )
        return merged
