import logging
from dataclasses import dataclass
from typing import Optional

from .ner import NERExtractor
from .negation import NegationClassifier
from .lab_parser import LabParser
from .weighter import AxiomWeighter
from apiro.config import MAX_SEED_NODES

logger = logging.getLogger(__name__)

from .models import ClinicalAxiom

class AxiomExtractor:
    """
    Deterministic pipeline that runs rule-based NLP tools over a clinical vignette
    to extract immutable Clinical Axioms.
    """
    def __init__(self):
        self.ner = NERExtractor()
        self.negation = NegationClassifier()
        self.lab_parser = LabParser()
        self.weighter = AxiomWeighter()

    def extract(self, vignette: str, max_axioms: int | None = MAX_SEED_NODES) -> list[ClinicalAxiom]:
        """
        Extract deterministic clinical axioms from a vignette.

        Args:
            max_axioms: cap on the number of axioms returned. Axioms are ranked
                by diagnostic weight; lab/vital measurements are always kept
                because they are the hardest constraints available. Pass None
                to disable the cap.
        """
        logger.info("Extracting Clinical Axioms...")

        # 1. Parse labs and vitals via strict Regex (these are gold-standard facts)
        lab_axioms = self.lab_parser.parse(vignette)

        # 2. Extract clinical entities via biomedical NER
        ner_entities = self.ner.extract(vignette)

        # Filter NER entities that overlap with matched lab values to avoid duplicates
        filtered_ner = self._filter_overlaps(ner_entities, lab_axioms)

        # 3. Classify polarity (affirmed/negated) via NegEx
        polar_entities = self.negation.classify(vignette, filtered_ner)

        # Merge lab axioms and polar NER entities
        all_raw_axioms = lab_axioms + polar_entities

        # 4. Weight each axiom based on its specificity
        for raw in all_raw_axioms:
            raw.weight = self.weighter.get_weight(raw)

        # 5. Cap the seed set. Every axiom becomes an unprunable depth-0 anchor,
        # so an unbounded set floods the graph budget and the prompt with
        # low-value entities before any reasoning happens.
        axioms = self._select(all_raw_axioms, max_axioms)

        for i, ax in enumerate(axioms):
            ax.id = f"ax_{i}"

        logger.info(
            f"Extracted {len(axioms)} deterministic axioms "
            f"(from {len(all_raw_axioms)} candidates)."
        )
        return axioms

    @staticmethod
    def _select(axioms: list[ClinicalAxiom], max_axioms: int | None) -> list[ClinicalAxiom]:
        """Keep all measurements plus the highest-weighted remaining axioms."""
        if max_axioms is None or len(axioms) <= max_axioms:
            return axioms

        measurements = [a for a in axioms if a.domain in ("lab", "vital")]
        others       = [a for a in axioms if a.domain not in ("lab", "vital")]
        others.sort(key=lambda a: (a.weight, a.confidence), reverse=True)

        keep = measurements + others[: max(0, max_axioms - len(measurements))]
        kept_ids = {id(a) for a in keep}
        # Preserve the original narrative order — the graph reads better and the
        # prompt follows the vignette.
        return [a for a in axioms if id(a) in kept_ids]

    @staticmethod
    def _raw_text(axiom: ClinicalAxiom) -> str:
        prefix = "The patient presents with the clinical finding of "
        if getattr(axiom, "raw_text", None):
            return axiom.raw_text
        if axiom.text.startswith(prefix):
            return axiom.text[len(prefix):].rstrip(".")
        return axiom.text

    def _filter_overlaps(self, ner_entities: list, lab_axioms: list) -> list:
        # Prevent duplicates: if an NER entity is already contained in the
        # matched text of a lab/vital axiom, discard the duplicate NER entity.
        filtered = []
        lab_texts = [lab.text.lower() for lab in lab_axioms]
        for ner in ner_entities:
            word = self._raw_text(ner).strip().lower()
            if not word:
                continue
            if any(word in lab_text for lab_text in lab_texts):
                continue
            filtered.append(ner)
        return filtered
