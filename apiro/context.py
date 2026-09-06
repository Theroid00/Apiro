"""Auditable evidence-aware selection for bounded clinical prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_WORD = re.compile(r"[a-z][a-z0-9-]{2,}", re.I)
_STOP = {"the", "and", "with", "this", "that", "patient", "diagnosis", "primary"}
_CLINICAL_MARKERS = re.compile(
    r"\b(no|not|without|denies|negative|abnormal|elevated|reduced|low|high|"
    r"positive|critical|imaging|ct|mri|ecg|laboratory|serum|blood)\b",
    re.I,
)


@dataclass(frozen=True)
class SelectedContext:
    text: str
    spans: tuple[tuple[int, int], ...]
    original_characters: int
    truncated: bool

    def to_dict(self) -> dict:
        return {
            "spans": [list(span) for span in self.spans],
            "original_characters": self.original_characters,
            "selected_characters": len(self.text),
            "truncated": self.truncated,
        }


def select_clinical_context(
    narrative: str, hypothesis: str = "", *, max_characters: int = 12_000
) -> SelectedContext:
    """Keep anchors, relevant/abnormal statements and coverage across a long note."""
    narrative = (narrative or "").strip()
    if len(narrative) <= max_characters:
        return SelectedContext(
            text=narrative,
            spans=((0, len(narrative)),) if narrative else (),
            original_characters=len(narrative),
            truncated=False,
        )

    sentences = [(m.start(), m.end(), m.group().strip()) for m in _SENTENCE.finditer(narrative)]
    terms = {w.lower() for w in _WORD.findall(hypothesis) if w.lower() not in _STOP}

    ranked: list[tuple[int, int, int, str]] = []
    for index, (start, end, text) in enumerate(sentences):
        words = {w.lower() for w in _WORD.findall(text)}
        score = 12 * len(words & terms)
        score += 7 if _CLINICAL_MARKERS.search(text) else 0
        score += 10 if "[Deterministic Clinical Findings]" in text else 0
        score += 6 if index < 2 else 0
        ranked.append((score, start, end, text))

    # Reserve evenly spaced coverage so late evidence is not categorically hidden.
    target_samples = min(12, len(sentences))
    if target_samples:
        step = max(1, len(sentences) // target_samples)
        for index in range(0, len(sentences), step):
            score, start, end, text = ranked[index]
            ranked[index] = (max(score, 4), start, end, text)

    selected: list[tuple[int, int, str]] = []
    used = 0
    for _score, start, end, text in sorted(ranked, key=lambda item: (-item[0], item[1])):
        cost = len(text) + (1 if selected else 0)
        if used + cost > max_characters:
            continue
        selected.append((start, end, text))
        used += cost

    selected.sort()
    return SelectedContext(
        text=" ".join(text for _, _, text in selected),
        spans=tuple((start, end) for start, end, _ in selected),
        original_characters=len(narrative),
        truncated=True,
    )
