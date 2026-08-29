"""
apiro/parsing.py — turning small-model prose into ranked lists
==============================================================

Every arm of every benchmark ends the same way: an 8B model is asked for a
ranked differential "one per line, no preamble, no explanation", and it
complies about half the time. What it actually emits looks like:

    Based on the provided information, I will attempt to generate three
    possible diagnoses that meet the strict rules outlined above.
    **Type 2 Diabetes Mellitus**
    **Chronic Kidney Disease**

or:

    **Diagnosis 1:**
    Acute Myeloid Leukemia (AML)
    **Diagnosis 2:**

WHY THIS MODULE EXISTS
----------------------
The old parser (``NodeExpander._parse_hypotheses``) stripped a single leading
bullet character and dropped lines matching a header-only regex. Neither
handles the two cases above, and the consequences were measured on the
committed C-NIAH run (``data/niah_eval_results.json``):

  * **57% of Apiro's top-3 slots (43 of 75) held markdown scaffolding rather
    than a diagnosis.** Five cases contained no diagnosis at all; twelve
    contained exactly one.
  * The baselines were graded over their *entire* raw output — every non-empty
    line, ~7.2 per case, uncapped — while Apiro was graded over exactly 3
    parsed slots.

So the engine was presenting an average of ~1.3 real candidates against
baselines presenting ~7, and losing comparisons it had no opportunity to win.
That is a measurement defect, not a reasoning result: any accuracy gap
computed that way is confounded by output formatting.

This module is the single parser all arms and the synthesis step share, so
they are graded on the same footing.

TWO JOBS, TWO FUNCTIONS
-----------------------
``parse_differential`` extracts short *diagnosis labels* — it cuts trailing
explanations, because "Diabetic Nephropathy: this is a complication of
long-standing diabetes..." is one label plus commentary.

``parse_claims`` extracts full *clinical sentences* — the child hypotheses
generated during expansion are supposed to be sentences, so it strips
scaffolding but keeps the prose.
"""

from __future__ import annotations

import re

__all__ = [
    "DIFFERENTIAL_SENTINEL",
    "parse_differential",
    "parse_claims",
    "strip_scaffolding",
]

#: Prefix the synthesis prompt asks the model to put on each answer line.
#: Small models follow an explicit output sentinel far more reliably than a
#: "no preamble" instruction, and when even one sentinel line appears the
#: parser can ignore everything else on the page.
DIFFERENTIAL_SENTINEL = "DX:"

_SENTINEL_RE = re.compile(r"^\s*(?:\*+\s*)?DX\s*\d*\s*[:.\-]\s*", re.IGNORECASE)

# Leading list markers: "1." "1)" "-" "*" "•" "#".
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•#>]+)\s*")

# Markdown emphasis runs at either end: ** __ * _ ` .
_EMPHASIS_RE = re.compile(r"^[*_`~]+|[*_`~]+$")

# A *paired* emphasis span anywhere in the line. Needed because the closing
# marker is often mid-line: "1. **Pulmonary embolism** - given the pleuritic
# pain" leaves a dangling "**" that the end-anchored rule above cannot see.
_INLINE_EMPHASIS_RE = re.compile(r"(\*\*|__|`|\*|_)(.+?)\1")

# A rank label the model volunteers instead of a bare name. Whatever follows on
# the same line is the real answer ("Primary Diagnosis: Acute Coronary
# Syndrome"); when nothing follows, the line is pure scaffolding.
_RANK_LABEL_RE = re.compile(
    r"^\s*(?:the\s+)?"
    r"(?:primary|secondary|tertiary|first|second|third|top|final|most\s+likely|"
    r"differential|candidate|possible|probable|working)?\s*"
    r"(?:diagnos[ie]s|dx|hypothesis|answer|option)\s*"
    r"(?:#?\d+)?\s*[:.\-]\s*",
    re.IGNORECASE,
)

# A line that is only a header ("Diagnoses:", "Output", "Here are the top 3:").
_HEADER_ONLY_RE = re.compile(
    r"^(?:hypothes[ie]s?|output|answer|diagnos[ie]s?|differential|results?|"
    r"response|list|top\s*\d*|here\s+(?:are|is)\b.*|the\s+top\b.*|my\s+top\b.*)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)

# First-person / meta commentary. A diagnosis label never contains these.
_META_PHRASES = (
    "i will", "i'll", "i have", "i am", "i can", "let me",
    "based on the provided", "based on the following", "based on the patient",
    "based on the information", "based on this", "according to the",
    "here are", "here is", "the following", "as requested", "as follows",
    "please note", "note that", "it's worth", "it is worth", "keep in mind",
    "disclaimer", "this task", "may not be", "without more context",
    "i will attempt", "would need", "cannot determine", "unable to",
)

# Openings that mark narrative rather than a label.
_PROSE_OPENERS = (
    "the patient", "this patient", "this is", "this condition", "this appears",
    "these ", "there ", "however", "although", "additionally", "furthermore",
    "in summary", "in conclusion", "overall", "given ", "considering",
    "the above", "the following", "note:", "reason:", "rationale:",
    "explanation:", "confirmed objective", "ruled-out", "ruled out",
)

# An annotation glued onto a label with punctuation: "Pulmonary embolism -
# given the pleuritic pain", "Diabetic Nephropathy: this is a complication of".
# Always split on these; the head is the label either way.
_ANNOTATION_SPLIT_RE = re.compile(r"\s*(?::\s|\s[-–—]\s|\s\|\s)")

# A subordinate clause. Split on these ONLY when the line is already too long
# to be a label: "Hemolytic anemia due to G6PD deficiency" is a diagnosis name,
# and cutting it at "due to" would throw away the half that identifies it.
_CLAUSE_SPLIT_RE = re.compile(r"\s+\(?(?:because|due to|caused by|secondary to|which|this)\b")

#: A diagnosis label longer than this is a sentence, not a name.
MAX_LABEL_WORDS = 12
#: Guard against a single stray character surviving the strippers.
MIN_LENGTH = 3


def strip_scaffolding(line: str) -> str:
    """Remove list markers, markdown emphasis and rank labels from one line.

    Applied repeatedly, because the real output nests them:
    ``"**Diagnosis 1:** Acute Coronary Syndrome"`` needs emphasis stripped
    before the rank label is visible, and the old single-pass strip left
    ``"*Diagnosis 1:**"`` behind as a "diagnosis".
    """
    text = line.strip()
    for _ in range(4):                      # bounded: each pass must shrink it
        before = text
        text = _LIST_MARKER_RE.sub("", text)
        text = _INLINE_EMPHASIS_RE.sub(r"\2", text)
        text = _EMPHASIS_RE.sub("", text).strip()
        text = _RANK_LABEL_RE.sub("", text, count=1).strip()
        text = _EMPHASIS_RE.sub("", text).strip()
        if text == before:
            break
    return text.strip(" .;,:")


def _is_noise(text: str) -> bool:
    """True when a cleaned line is commentary rather than an answer."""
    if len(text) < MIN_LENGTH:
        return True
    lowered = text.lower()
    if _HEADER_ONLY_RE.match(lowered):
        return True
    if any(phrase in lowered for phrase in _META_PHRASES):
        return True
    if lowered.startswith(_PROSE_OPENERS):
        return True
    # A label needs letters. "1", "---", "###" do not qualify.
    if not re.search(r"[a-zA-Z]{3}", text):
        return True
    return False


def parse_differential(
    raw: str,
    limit: int = 3,
    max_label_words: int = MAX_LABEL_WORDS,
) -> list[str]:
    """Parse a model's answer into a ranked list of diagnosis labels.

    Order is preserved: it *is* the ranking, and every rank-aware metric reads
    it. Duplicates are dropped case-insensitively so a model that repeats
    itself does not consume two slots with one answer.

    Args:
        raw: The model's unmodified reply.
        limit: Maximum labels to return.
        max_label_words: Lines longer than this after cleaning are treated as
            prose and dropped, unless an explanation can be split off the front.

    Returns:
        Up to ``limit`` diagnosis labels, most likely first.
    """
    if not raw or not raw.strip():
        return []

    lines = raw.strip().splitlines()

    # High-precision path: if the model honoured the sentinel anywhere, trust
    # only the sentinel lines and ignore whatever else it decided to say.
    sentinel_lines = [l for l in lines if _SENTINEL_RE.match(l)]
    if sentinel_lines:
        lines = [_SENTINEL_RE.sub("", l, count=1) for l in sentinel_lines]

    out: list[str] = []
    seen: set[str] = set()

    for line in lines:
        text = strip_scaffolding(line)
        if not text or text.startswith("```"):
            continue

        # "Pulmonary embolism - given the pleuritic pain" -> keep the head.
        # Punctuation-delimited annotations are always stripped; the head is
        # the label whether or not the whole line was over-long.
        head = _ANNOTATION_SPLIT_RE.split(text, maxsplit=1)[0].strip(" .;,:")
        head = strip_scaffolding(head)
        if head and re.search(r"[a-zA-Z]{3}", head):
            text = head

        # Still too long to be a label: try a subordinate clause, then give up.
        if len(text.split()) > max_label_words:
            head = _CLAUSE_SPLIT_RE.split(text, maxsplit=1)[0].strip(" .;,:")
            head = strip_scaffolding(head)
            if head and 0 < len(head.split()) <= max_label_words:
                text = head
            else:
                continue

        if _is_noise(text):
            continue

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break

    return out


def parse_claims(raw: str, limit: int = 3) -> list[str]:
    """Parse generated child hypotheses — full clinical sentences, not labels.

    Same scaffolding removal as :func:`parse_differential`, but no label-length
    cut and no explanation split: a child hypothesis is *supposed* to be a
    sentence ("Right coronary artery occlusion is the most likely cause of
    inferior STEMI").

    Args:
        raw: The model's unmodified reply.
        limit: Maximum claims to return.

    Returns:
        Up to ``limit`` claim sentences, in the order generated.
    """
    if not raw or not raw.strip():
        return []

    out: list[str] = []
    seen: set[str] = set()

    for line in raw.strip().splitlines():
        text = strip_scaffolding(line)
        if not text or text.startswith("```"):
            continue
        if _is_noise(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break

    return out
