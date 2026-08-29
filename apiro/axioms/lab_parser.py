import re
import logging
from .models import ClinicalAxiom

logger = logging.getLogger(__name__)

# Basic Regex patterns for vitals and common labs
# e.g. "Troponin 5.0 ng/mL", "BP 120/80 mmHg", "Temp 39.5 C"
# Order matters: the blood-pressure pattern runs first so the general pattern
# does not shred "BP 120/80 mmHg" into a nonsense "BP 120 /" measurement.
LAB_PATTERNS = [
    # Blood pressure: BP 120/80 mmHg
    re.compile(r'\b(BP|Blood pressure)\s*[:=]?\s*(\d{2,3}/\d{2,3})\s*(mmHg)?\b', re.IGNORECASE),
    # General pattern: [Name] [Value] [Unit]
    # Name: 1-3 words, Value: float, Unit: word with optional slashes/letters
    re.compile(r'\b([A-Za-z\-]+(?:\s+[A-Za-z\-]+){0,2})\s*[:=]?\s*(\d+(?:\.\d+)?)\s*([a-zA-Z%µ/]+)\b', re.IGNORECASE),
]

class LabParser:
    IGNORED_NAMES = {
        "a", "the", "in", "on", "at", "to", "is", "was", "for", "with", "of", "an", "and", "or", "by",
        "over", "under", "about", "around", "approximately", "history", "prior", "some", "few", "many",
        "several", "no", "not", "any", "all", "each", "every", "both", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "ten", "first", "second", "third", "last", "next",
        "previous", "old", "years", "months", "days", "weeks", "hours", "minutes", "seconds", "year",
        "month", "day", "week", "hour", "minute", "second", "age", "stone", "stones", "kg", "lbs", "male",
        "female", "woman", "man", "patient", "caucasians", "caucasian", "white", "black", "asian", "hispanic"
    }

    IGNORED_UNITS = {
        "year", "years", "month", "months", "day", "days", "week", "weeks", "hour", "hours", "minute",
        "minutes", "second", "seconds", "stone", "stones", "old", "year-old", "years-old", "month-old",
        "months-old", "yo", "y/o", "kg", "lbs",
        # Social/lifestyle quantities are not lab measurements ("drinks 2 units
        # weekly", "smoked 20 cigarettes a day").
        "unit", "units", "cigarette", "cigarettes", "pack", "packs", "drink",
        "drinks", "times", "episode", "episodes", "tablet", "tablets", "dose",
        "doses", "cm", "mm", "inch", "inches", "ft",
    }

    def __init__(self):
        pass
        
    # Filler words that sit between an analyte name and its value
    # ("Hemoglobin of 9.5 g/dL", "troponin was 5.0 ng/mL").
    _TRAILING_FILLER = {
        "of", "is", "was", "were", "are", "at", "to", "with", "and", "or",
        "the", "a", "an", "measured", "level", "levels", "value", "showed",
        "shows", "revealed", "reveals", "returned", "came", "back",
        "repeat", "repeated", "serial", "initial", "admission", "follow-up",
        "peak", "his", "her", "their",
    }

    @classmethod
    def _trim_name(cls, name: str) -> str:
        """
        Drop leading/trailing filler from a captured analyte name.

        The name group greedily swallows up to three preceding words, so real
        labs arrived as "Hemoglobin of" or "revealed a Creatinine". The old
        code rejected the whole match if ANY captured word was in IGNORED_NAMES,
        which silently discarded most naturally-phrased lab values — precisely
        the deterministic anchors this engine is built on.
        """
        words = name.split()
        while words and words[-1].lower() in cls._TRAILING_FILLER:
            words.pop()
        while words and words[0].lower() in cls._TRAILING_FILLER:
            words.pop(0)
        return " ".join(words).strip()

    def parse(self, text: str) -> list[ClinicalAxiom]:
        axioms = []
        seen: set[str] = set()
        claimed: list[tuple[int, int]] = []   # character spans already parsed

        for pattern in LAB_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue   # already covered by an earlier, more specific pattern
                name = self._trim_name(match.group(1).strip())
                val_str = match.group(2)
                unit = match.group(3) if match.lastindex >= 3 else None

                # Check for false positives in names and units
                name_lower = name.lower()
                unit_lower = unit.lower() if unit else ""

                if len(name) < 2 or name_lower in self.IGNORED_NAMES:
                    continue
                if unit_lower in self.IGNORED_UNITS:
                    continue
                if any(word in self.IGNORED_NAMES for word in name_lower.split()):
                    continue
                # A "unit" of bare punctuation ("/") means the regex clipped a
                # compound value; skip rather than emit "BP 88 /".
                if unit is not None and not re.search(r"[a-zA-Z%µ]", unit):
                    continue

                # Handle BP specifically
                if "/" in val_str:
                    val = None # Keep as string in 'text'
                else:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = None
                        
                measurement = f"{name} {val_str}{(' ' + unit) if unit else ''}".strip()
                # Key on the analyte itself, so "Potassium 5.6 mmol/L" and a
                # later "repeat potassium 5.6 mmol/L" collapse to one anchor.
                key = f"{name_lower.split()[-1]}|{val_str}|{unit_lower}"
                if key in seen:
                    # The two patterns overlap on blood pressure, and a repeated
                    # value in the vignette produced a duplicate anchor.
                    continue
                seen.add(key)
                claimed.append((start, end))

                sentence = f"The patient has a lab result or vital sign showing {measurement}."
                ax = ClinicalAxiom(
                    id="",
                    text=sentence,
                    domain="lab" if name_lower not in ["bp", "blood pressure", "temp", "temperature", "hr", "heart rate", "rr"] else "vital",
                    polarity="affirmed",
                    value=val,
                    unit=unit,
                    weight=0.0,
                    snomed_cui=None,
                    raw_text=measurement,
                )
                axioms.append(ax)

        return axioms
