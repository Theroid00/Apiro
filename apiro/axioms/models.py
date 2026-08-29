from dataclasses import dataclass
from typing import Optional

@dataclass
class ClinicalAxiom:
    id: str
    text: str           # Exact extracted text from vignette
    domain: str         # "symptom" | "lab" | "vital" | "imaging" | "history" | "treatment" | "medication" | "finding"
    polarity: str       # "affirmed" | "negated" | "historical"
    value: Optional[float] # Numeric value if lab/vital (e.g. 5.0 for Troponin=5.0)
    unit: Optional[str]    # Unit string (e.g. "mg/dL", "mmHg", "C")
    weight: float       # Diagnostic specificity weight (0.1-1.0)
    snomed_cui: Optional[str] = None  # Ontology code if matched
    # The bare entity as it appeared in the vignette ("epigastric pain"), before
    # it was forged into a full sentence. Downstream stages used to recover this
    # by slicing a hard-coded sentence prefix off `text`, which broke silently
    # whenever the phrasing changed.
    raw_text: Optional[str] = None
    confidence: float = 1.0   # extractor confidence (NER score; 1.0 for regex)
