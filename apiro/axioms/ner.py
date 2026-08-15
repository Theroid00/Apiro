import logging
import re

from .models import ClinicalAxiom

logger = logging.getLogger(__name__)

# Minimum NER confidence for an entity to become a graph axiom.
# Axioms are seeded as "absolute certainty" (entropy ~0.01) nodes that can never
# be pruned, so a low-confidence entity is not a cheap mistake: it anchors the
# whole traversal to something the model was barely willing to tag.
MIN_NER_SCORE = 0.55

# Entity groups emitted by d4data/biomedical-ner-all that carry diagnostic
# signal. Everything else (Age, Sex, Date, Duration, Severity, Color,
# Quantitative_concept, Coreference, Subject, ...) describes the sentence rather
# than the patient's pathology; seeding those produced anchors like
# "The patient presents with the clinical finding of 45-year-old."
DIAGNOSTIC_ENTITY_GROUPS: dict[str, str] = {
    "sign_symptom":          "symptom",
    "disease_disorder":      "diagnosis",
    "diagnostic_procedure":  "lab",
    "lab_value":             "lab",
    "medication":            "pharmacology",
    "therapeutic_procedure": "treatment",
    "biological_structure":  "pathophysiology",
    "clinical_event":        "symptom",
    "history":               "history",
    "family_history":        "history",
}

# Entity strings that carry no diagnostic content on their own.
_STOP_ENTITIES = {
    "patient", "patients", "man", "woman", "male", "female", "boy", "girl",
    "history", "presentation", "admission", "examination", "exam", "hospital",
    "emergency", "department", "year", "years", "day", "days", "week", "weeks",
    "month", "months", "old", "normal", "abnormal", "mild", "moderate", "severe",
    "acute", "chronic", "left", "right", "bilateral",
}

_SENTENCE_TEMPLATES: dict[str, str] = {
    "diagnosis": "The patient has a documented diagnosis of {entity}.",
    "history":   "The patient has a history of {entity}.",
    "pharmacology": "The patient is on {entity}.",
    "treatment": "The patient has undergone {entity}.",
}
_DEFAULT_TEMPLATE = "The patient presents with the clinical finding of {entity}."


class NERExtractor:
    def __init__(self, model_name="d4data/biomedical-ner-all", min_score: float = MIN_NER_SCORE):
        logger.info(f"Loading Hugging Face NER model: {model_name}...")
        self.min_score = min_score
        try:
            from transformers import pipeline
            # Use aggregation_strategy="simple" to merge sub-word tokens (B-core, I-core) into full words
            self.nlp = pipeline("ner", model=model_name, aggregation_strategy="simple", device="cpu")
        except Exception as e:
            logger.error(f"Failed to load transformers pipeline. Error: {e}")
            self.nlp = None

    @staticmethod
    def _clean(word: str) -> str:
        """Normalise a raw NER span into a usable clinical phrase."""
        word = word.replace("##", "").strip()
        word = re.sub(r"^[^\w(]+|[^\w)%]+$", "", word)
        return re.sub(r"\s+", " ", word).strip()

    def _is_usable(self, word: str) -> bool:
        if len(word) < 3:
            return False
        if word.lower() in _STOP_ENTITIES:
            return False
        # Pure numbers / measurements without a name carry no standalone meaning
        if not re.search(r"[a-zA-Z]{3}", word):
            return False
        return True

    def extract(self, text: str) -> list[ClinicalAxiom]:
        if not self.nlp:
            logger.error("NER pipeline not initialized. Returning empty axioms.")
            return []

        try:
            entities = self.nlp(text)
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

        axioms: list[ClinicalAxiom] = []
        seen: set[str] = set()
        n_low_score = 0
        n_off_group = 0
        n_duplicate = 0

        for ent in entities:
            label = str(ent.get("entity_group") or ent.get("entity") or "finding")
            group = label.lower()
            score = float(ent.get("score", 1.0))
            word = self._clean(str(ent.get("word", "")))

            if not self._is_usable(word):
                continue
            if score < self.min_score:
                n_low_score += 1
                continue
            if group not in DIAGNOSTIC_ENTITY_GROUPS:
                n_off_group += 1
                continue

            # The same entity is mentioned several times in a typical vignette.
            # Each mention used to become its own seed node, so the graph filled
            # with identical anchors that padded the prompt, ate the node budget
            # and — because seeds all share a fixed entropy — dragged the
            # saturation window down faster.
            key = word.lower()
            if key in seen:
                n_duplicate += 1
                continue
            seen.add(key)

            domain = DIAGNOSTIC_ENTITY_GROUPS[group]
            template = _SENTENCE_TEMPLATES.get(domain, _DEFAULT_TEMPLATE)
            axioms.append(ClinicalAxiom(
                id="",
                text=template.format(entity=word),
                domain=domain,
                polarity="affirmed",
                value=None,
                unit=None,
                weight=0.0,
                snomed_cui=None,
                raw_text=word,
                confidence=round(score, 4),
            ))

        logger.info(
            f"[NER] {len(axioms)} entities kept "
            f"(dropped {n_low_score} low-confidence, {n_off_group} non-diagnostic, "
            f"{n_duplicate} duplicate)."
        )
        return axioms
