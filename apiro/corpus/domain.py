"""Small keyword-based clinical domain classifier shared by corpus adapters."""

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "genetics": ["gene", "genetic", "mutation", "allele", "chromosom", "hereditary", "inherited"],
    "pharmacology": ["drug", "medication", "dose", "prescribe", "administer", "mg", "contraindicated", "antibiotic", "statin"],
    "imaging": ["ct", "mri", "x-ray", "ultrasound", "scan", "radiograph", "echo", "echocardiogram"],
    "lab": ["blood", "serum", "plasma", "troponin", "creatinine", "bilirubin", "wbc", "rbc", "platelet", "culture"],
    "pathophysiology": ["mechanism", "pathway", "cascade", "ischemia", "inflammation", "necrosis", "apoptosis", "fibrosis"],
    "treatment": ["surgery", "procedure", "therapy", "treatment", "intervention", "resect", "catheter", "stent"],
    "comorbidity": ["comorbid", "concurrent", "coexisting", "secondary", "complication", "alongside"],
}


def classify_domain(text: str) -> str:
    """Return the first matching clinical domain, or pathophysiology."""
    lowered = text.casefold()
    scores = {
        domain: sum(keyword in lowered for keyword in keywords)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "pathophysiology"
