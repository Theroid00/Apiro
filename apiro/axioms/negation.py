import re
import logging
from .models import ClinicalAxiom

logger = logging.getLogger(__name__)

_LEGACY_PREFIX = "The patient presents with the clinical finding of "


def _raw_of(axiom: ClinicalAxiom) -> str:
    """
    The bare entity text for an axiom.

    Prefers the `raw_text` recorded by the extractor. Falls back to stripping
    the legacy sentence prefix for axioms built before that field existed —
    the old code sliced this prefix off blindly, so any axiom forged with a
    different template silently had its entire sentence treated as the entity.
    """
    if getattr(axiom, "raw_text", None):
        return axiom.raw_text
    if axiom.text.startswith(_LEGACY_PREFIX):
        return axiom.text[len(_LEGACY_PREFIX):].rstrip(".")
    return axiom.text


class NegationClassifier:
    def __init__(self):
        try:
            import medspacy
            # We just need the ConText component for negation and historicity
            self.nlp = medspacy.load(enable=["medspacy_context"])
            logger.info("Loaded medspacy context for negation classification.")
        except ImportError:
            logger.error("medspacy not installed. Negation classification will use regex fallback.")
            self.nlp = None

    def classify(self, text: str, axioms: list[ClinicalAxiom]) -> list[ClinicalAxiom]:
        """
        Takes raw axioms and determines if they are negated or historical based on the context.
        """
        if not axioms:
            return axioms
            
        if self.nlp:
            try:
                from medspacy.ner import TargetMatcher, TargetRule
                
                raw_words = [_raw_of(ax) for ax in axioms]

                target_matcher = TargetMatcher(self.nlp.vocab)
                rules = [TargetRule(word, "AXIOM") for word in raw_words]
                target_matcher.add(rules)
                
                doc = self.nlp.tokenizer(text)
                target_matcher(doc)
                
                # Now run context
                self.nlp.get_pipe("medspacy_context")(doc)
                
                # Map back results
                ent_map = {}
                for ent in doc.ents:
                    ent_map[ent.text.lower()] = {
                        "negated": ent._.is_negated,
                        "historical": ent._.is_historical
                    }
                    
                for ax, raw_word in zip(axioms, raw_words):
                    info = ent_map.get(raw_word.lower())
                    if info:
                        if info["historical"]:
                            ax.polarity = "historical"
                            ax.text = f"The patient has a history of {raw_word}."
                        elif info["negated"]:
                            ax.polarity = "negated"
                            ax.text = f"The patient denies the clinical finding of {raw_word}."
                        else:
                            ax.polarity = "affirmed"
                return axioms
            except Exception as e:
                logger.error(f"medspacy classification failed: {e}. Falling back to regex.")

        # Fallback to simple rule-based negation and history classifier
        text_lower = text.lower()
        neg_patterns = [
            r"\b(no|not|denies|denied|negative for|rules? out|ruled out|free of|without|absent|absence of|never|unlikely|cannot|does not|no evidence of|no sign of|no history of)\b",
        ]
        history_patterns = [
            r"\b(history of|past medical history|previously|prior episode|prior history|years ago|months ago)\b"
        ]
        
        for ax in axioms:
            raw_word = _raw_of(ax)
            word_lower = raw_word.lower()

            occurrences = [m.start() for m in re.finditer(re.escape(word_lower), text_lower)]
            if not occurrences:
                continue

            # A finding affirmed anywhere in the vignette is affirmed. The old
            # code only inspected the FIRST occurrence, so "no chest pain on
            # admission; chest pain recurred overnight" was recorded as denied.
            polarities = []
            for idx in occurrences:
                window_before = self._scope_before(text_lower, idx)
                if any(re.search(pat, window_before) for pat in history_patterns):
                    polarities.append("historical")
                elif any(re.search(pat, window_before) for pat in neg_patterns):
                    polarities.append("negated")
                else:
                    polarities.append("affirmed")

            if "affirmed" in polarities:
                ax.polarity = "affirmed"
            elif "negated" in polarities:
                ax.polarity = "negated"
                ax.text = f"The patient denies the clinical finding of {raw_word}."
            else:
                ax.polarity = "historical"
                ax.text = f"The patient has a history of {raw_word}."

        return axioms

    @staticmethod
    def _scope_before(text_lower: str, idx: int, max_chars: int = 45) -> str:
        """
        The negation scope preceding `idx`, clipped at the nearest clause or
        sentence boundary.

        A fixed 45-character lookback crosses sentence boundaries, so
        "No fever. Severe epigastric pain" marked the pain as denied — and a
        negated axiom tells the synthesizer that any diagnosis requiring that
        finding is wrong, which is exactly backwards.
        """
        window = text_lower[max(0, idx - max_chars):idx]
        boundary = max(window.rfind(c) for c in ".;!?\n")
        return window[boundary + 1:] if boundary != -1 else window
