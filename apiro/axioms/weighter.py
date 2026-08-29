import yaml
import logging
from pathlib import Path

from apiro.config import DATA_DIR
from .models import ClinicalAxiom

logger = logging.getLogger(__name__)


class AxiomWeighter:
    def __init__(self, weights_file: str | Path | None = None):
        # Absolute by default (config.DATA_DIR), so this loads correctly
        # regardless of the process's working directory. A relative path
        # here used to silently fail whenever Apiro was launched from
        # anywhere other than the repo root, degrading every axiom to the
        # flat default weight with only a log warning to show for it.
        weights_file = Path(weights_file) if weights_file else DATA_DIR / "axiom_weights.yaml"
        self.weights = {}
        if weights_file.exists():
            with open(weights_file, "r") as f:
                try:
                    data = yaml.safe_load(f)
                    if data:
                        # Flatten into a fast lookup dict mapping lowercase text to weight
                        for category in ["high_weight", "medium_weight", "low_weight"]:
                            if category in data:
                                for item in data[category]:
                                    self.weights[item["entity"].lower()] = item["weight"]
                except Exception as e:
                    logger.error(f"Failed to load weights from {weights_file}: {e}")
        else:
            logger.warning(f"Axiom weights file {weights_file} not found. Using default weights.")

        # Partial-match candidates ordered longest-first, so a more specific
        # phrase ("chest pain") is checked before a shorter one it contains
        # ("pain") rather than being decided by arbitrary dict/YAML order.
        self._partial_keys = sorted(self.weights.keys(), key=len, reverse=True)

    def get_weight(self, axiom: ClinicalAxiom) -> float:
        """
        Lookup the diagnostic specificity weight.
        """
        # If it's a lab value with a number, and it wasn't filtered, it generally has high specificity
        if axiom.domain in ["lab", "vital"] and axiom.value is not None:
            # A real implementation would check threshold values here
            return 0.8

        text_lower = axiom.text.lower()

        # Exact match
        if text_lower in self.weights:
            return self.weights[text_lower]

        # Partial match — longest (most specific) key wins.
        for key in self._partial_keys:
            if key in text_lower:
                return self.weights[key]

        # Default fallback weight for unknown entities
        return 0.3
