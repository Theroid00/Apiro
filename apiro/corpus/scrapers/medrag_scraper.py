"""
corpus/scrapers/medrag_scraper.py
===================================
Streams the MedRAG/pubmed dataset from HuggingFace — pre-chunked PubMed
abstracts ready for direct embedding. No NCBI email or API key required.

Dataset: https://huggingface.co/datasets/MedRAG/pubmed

CRITICAL — PMID ordering:
  MedRAG/pubmed streams records in PMID order starting from PMID 1 (1966).
  PMIDs are assigned sequentially; PMID 20,000,000 ≈ 2010.
  We set min_pmid=20_000_000 by default so only post-2010 clinical papers
  are ingested. Pre-2010 papers are basic biochemistry with little relevance
  to modern clinical diagnosis and degrade RAG quality significantly.

Each record dict is a final chunk ready for Embedder.ingest():
    {
        "chunk_id":       str,
        "pmid":           str,
        "title":          str,
        "text":           str,          # title + abstract snippet
        "source_db":      "medrag_pubmed",
        "medical_domain": str,          # one of 7 Apiro domains
        "condition_tags": str,          # comma-joined condition keywords found
        "evidence_level": int,          # always 2 (peer-reviewed journal article)
        "chunk_index":    0,
        "n_chunks":       1,
    }
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum PMID threshold — only ingest modern papers
# ---------------------------------------------------------------------------
# PMID ≥ 20_000_000 ≈ papers published from ~2010 onwards.
# This excludes the 1960s-1980s biochemistry papers that dominate the
# early part of MedRAG/pubmed and confuse clinical RAG retrieval.
MIN_MODERN_PMID = 20_000_000

# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "genetics":        ["gene", "mutation", "variant", "allele", "chromosome",
                        "genomic", "snp", "pathogenic", "brca", "tp53", "cftr"],
    "pharmacology":    ["pharmacokinetic", "pharmacodynamic", "mechanism of action",
                        "receptor", "agonist", "antagonist", "inhibitor", "ic50",
                        "bioavailability", "half-life", "cytochrome", "cyp3a4"],
    "imaging":         ["mri", "ct scan", "x-ray", "ultrasound", "radiograph",
                        "echocardiogram", "pet scan", "imaging", "opacity",
                        "consolidation", "effusion", "atelectasis"],
    "lab":             ["troponin", "creatinine", "hemoglobin", "platelet", "wbc",
                        "d-dimer", "bnp", "albumin", "bilirubin", "lactate",
                        "crp", "esr", "ferritin", "procalcitonin", "inr"],
    "treatment":       ["treatment", "therapy", "surgery", "intervention",
                        "guideline", "protocol", "first-line", "randomized",
                        "clinical trial", "efficacy", "outcomes", "prognosis"],
    "comorbidity":     ["comorbidity", "complication", "secondary", "multimorbidity",
                        "concurrent", "coexisting", "background condition"],
    "pathophysiology": ["pathophysiology", "mechanism", "pathogenesis", "etiology",
                        "inflammatory", "fibrosis", "ischemia", "necrosis",
                        "apoptosis", "autoimmune", "hypoxia"],
}


def _guess_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {
        d: sum(1 for kw in kws if kw in text_lower)
        for d, kws in _DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "pathophysiology"


# ---------------------------------------------------------------------------
# Condition tag extraction
# ---------------------------------------------------------------------------
# Clinical condition keywords for TC-1.2: condition_tags non-empty for ≥ 85% of chunks.
# Each tuple: (regex pattern, canonical tag name)
_CONDITION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(myocardial infarction|stemi|nstemi|heart attack|ami)\b", re.I), "myocardial infarction"),
    (re.compile(r"\b(heart failure|cardiac failure|left ventricular failure|lvf|chf)\b", re.I), "heart failure"),
    (re.compile(r"\b(pancreatitis)\b", re.I), "pancreatitis"),
    (re.compile(r"\b(pneumonia|community.acquired pneumonia|cap|ventilator.associated)\b", re.I), "pneumonia"),
    (re.compile(r"\b(sepsis|septic shock|bacteremia)\b", re.I), "sepsis"),
    (re.compile(r"\b(stroke|ischemic stroke|cerebral infarction|tia|transient ischemic)\b", re.I), "stroke"),
    (re.compile(r"\b(diabetes mellitus|type 2 diabetes|type 1 diabetes|hyperglycemia|diabetic)\b", re.I), "diabetes"),
    (re.compile(r"\b(hypertension|high blood pressure|antihypertensive)\b", re.I), "hypertension"),
    (re.compile(r"\b(atrial fibrillation|afib|af )\b", re.I), "atrial fibrillation"),
    (re.compile(r"\b(pulmonary embolism|pe |dvt|deep vein thrombosis|venous thromboembolism|vte)\b", re.I), "pulmonary embolism"),
    (re.compile(r"\b(copd|chronic obstructive|emphysema)\b", re.I), "COPD"),
    (re.compile(r"\b(asthma|bronchospasm)\b", re.I), "asthma"),
    (re.compile(r"\b(lupus|sle|systemic lupus erythematosus)\b", re.I), "SLE"),
    (re.compile(r"\b(rheumatoid arthritis)\b", re.I), "rheumatoid arthritis"),
    (re.compile(r"\b(acute kidney injury|aki|renal failure|chronic kidney disease|ckd)\b", re.I), "renal failure"),
    (re.compile(r"\b(liver failure|hepatic failure|cirrhosis|hepatitis)\b", re.I), "hepatic disease"),
    (re.compile(r"\b(cancer|carcinoma|malignancy|tumor|neoplasm|oncol)\b", re.I), "cancer"),
    (re.compile(r"\b(thyroid|hypothyroid|hyperthyroid|graves|hashimoto)\b", re.I), "thyroid disease"),
    (re.compile(r"\b(meningitis|encephalitis)\b", re.I), "CNS infection"),
    (re.compile(r"\b(appendicitis)\b", re.I), "appendicitis"),
    (re.compile(r"\b(cholecystitis|gallstone|cholelithiasis)\b", re.I), "gallbladder disease"),
    (re.compile(r"\b(aortic dissection|aortic aneurysm)\b", re.I), "aortic disease"),
    (re.compile(r"\b(wilson.s disease|copper metabolism)\b", re.I), "Wilson's disease"),
    (re.compile(r"\b(addison|adrenal insufficiency|adrenal crisis)\b", re.I), "adrenal insufficiency"),
    (re.compile(r"\b(sarcoidosis)\b", re.I), "sarcoidosis"),
    (re.compile(r"\b(cardiomyopathy|takotsubo|dilated cardiomyopathy|hypertrophic)\b", re.I), "cardiomyopathy"),
    (re.compile(r"\b(pulmonary hypertension)\b", re.I), "pulmonary hypertension"),
    (re.compile(r"\b(crohn|ulcerative colitis|inflammatory bowel|ibd)\b", re.I), "IBD"),
    (re.compile(r"\b(anemia|anaemia|haemoglobin|sickle cell)\b", re.I), "anemia"),
]


def _extract_condition_tags(text: str) -> str:
    """
    Extract known clinical condition keywords from text.
    Returns a comma-joined string of canonical condition names (for ChromaDB metadata).
    Returns '' if no conditions found (never None — ChromaDB rejects None).
    """
    found = []
    for pattern, tag in _CONDITION_PATTERNS:
        if pattern.search(text) and tag not in found:
            found.append(tag)
    return ", ".join(found)


# ---------------------------------------------------------------------------
# MedRAGScraper
# ---------------------------------------------------------------------------

class MedRAGScraper:
    """
    Streams MedRAG/pubmed from HuggingFace, filtering to modern papers only.

    Args:
        max_records:  Maximum records to return after filtering.
        split:        Dataset split (only 'train' exists in MedRAG/pubmed).
        min_pmid:     Minimum PMID to accept. Default 20_000_000 (~2010+).
                      Set to 0 to accept all papers (includes 1970s papers —
                      not recommended for clinical RAG).
    """

    def __init__(
        self,
        max_records: int = 100_000,
        split: str = "train",
        min_pmid: int = MIN_MODERN_PMID,
    ):
        self.max_records = max_records
        self.split       = split
        self.min_pmid    = min_pmid

    def fetch(self) -> list[dict]:
        """
        Stream MedRAG/pubmed and return up to max_records modern chunk dicts.

        Records with PMID < min_pmid are skipped entirely. The stream is
        processed until max_records modern papers are collected or the
        dataset is exhausted.

        Returns:
            List of pre-chunked record dicts ready for Embedder.ingest().
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace `datasets` library not installed. "
                "Run: pip install datasets"
            )

        logger.info(
            f"Streaming MedRAG/pubmed from HuggingFace "
            f"(max_records={self.max_records or 'unlimited'}, "
            f"min_pmid={self.min_pmid:,})..."
        )
        if self.min_pmid > 0:
            logger.info(
                f"  Skipping papers with PMID < {self.min_pmid:,} "
                f"(pre-2010 papers excluded to ensure clinical relevance)"
            )

        dataset = load_dataset(
            "MedRAG/pubmed",
            split=self.split,
            streaming=True,
            trust_remote_code=False,
        )

        records: list[dict] = []
        skipped_old  = 0
        skipped_empty = 0
        examined     = 0

        for example in dataset:
            examined += 1

            pmid_raw = example.get("PMID", "")
            try:
                pmid_int = int(pmid_raw)
            except (TypeError, ValueError):
                pmid_int = 0

            # Skip pre-modern papers
            if self.min_pmid > 0 and pmid_int < self.min_pmid:
                skipped_old += 1
                if skipped_old % 500_000 == 0:
                    logger.info(f"  Skipped {skipped_old:,} old papers so far...")
                continue

            text  = example.get("contents") or example.get("content", "")
            title = example.get("title", "")
            pmid  = str(pmid_raw)
            uid   = str(example.get("id", f"medrag_{examined:08d}"))

            if not text.strip():
                skipped_empty += 1
                continue

            records.append({
                "chunk_id":       uid,
                "pmid":           pmid,
                "title":          title,
                "text":           text,
                "source_db":      "medrag_pubmed",
                "medical_domain": _guess_domain(text),
                "condition_tags": _extract_condition_tags(text),
                "evidence_level": 2,   # peer-reviewed journal article
                "chunk_index":    0,
                "n_chunks":       1,
            })

            if len(records) % 10_000 == 0 and len(records) > 0:
                logger.info(
                    f"  Collected {len(records):,} modern records "
                    f"(examined {examined:,} total)..."
                )

            if self.max_records and len(records) >= self.max_records:
                break

        logger.info(
            f"MedRAG/pubmed fetch complete. "
            f"{len(records):,} modern records collected "
            f"({skipped_old:,} old papers skipped, {skipped_empty:,} empty)."
        )
        return records
