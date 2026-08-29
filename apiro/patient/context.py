"""
apiro/patient/context.py — PatientContext

.. deprecated::
    DEPRECATED AND UNUSED. Nothing in the engine, the CLI, the web app or any
    benchmark imports this module. It is retained deliberately, not by
    oversight — see the note below — but treat it as frozen: do not add
    callers, and do not extend it without first deciding it is being brought
    back into the pipeline.

Structured patient representation extracted once from a raw clinical
vignette using a single LLM call.

WHY IT IS STILL HERE
    It was built for an earlier "hypothesis-testing" architecture (structured
    PatientContext → HypothesisOracle → EvidenceMatcher → BayesianScorer) that
    was purged in commit 8a001c7. ApiroTraversal instead passes the raw
    (sanitized) vignette string directly to NodeExpander, which derives its own
    case anchor and seed context — see apiro/axioms/seeding.py and
    BeliefGraph.set_case_anchor().

    The open question recorded in docs/IMPROVEMENTS.md is whether seed
    selection would be better anchored on structured fields (age, sex, explicit
    lab dict) than on free text. This module is the working implementation of
    that idea, so it is kept as the starting point for answering it rather than
    deleted and rewritten later.

EXTRACTION:
  A single LLM call with a JSON-format prompt extracts all fields. Failures
  gracefully fall back: missing age → None, missing labs → empty dict, etc.
  The original vignette is always preserved verbatim for synthesis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

from apiro.config import OLLAMA_BASE_URL, PRIMARY_MODEL

logger = logging.getLogger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────────
_EXTRACTION_PROMPT = """\
You are a clinical data extractor. Read the following clinical case and extract structured information.

Output ONLY a valid JSON object with exactly these keys:
{{
  "chief_complaint": "<the acute primary symptom or reason for this visit, one sentence>",
  "age": <integer or null>,
  "gender": "<M or F or null>",
  "symptoms": ["<symptom 1>", "<symptom 2>", ...],
  "labs": {{"<test name>": "<value>", ...}},
  "imaging": ["<finding 1>", ...],
  "history": ["<pre-existing condition 1>", ...]
}}

Rules:
- chief_complaint MUST be the ACUTE presenting problem, not a chronic background condition.
- symptoms: only symptoms present at this visit, not historical ones.
- labs: include only labs mentioned with values (e.g. "CRP": "290 mg/L").
- history: chronic/pre-existing conditions, prior surgeries, and medical devices/implants (e.g., catheters) ONLY, not the acute problem.
- If a field has no data, use null for strings/ints or [] for arrays or {{}} for dicts.
- Output ONLY the JSON object. No preamble, no explanation.

Clinical Case:
{vignette}

JSON:"""


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class PatientContext:
    """
    Structured representation of a patient's clinical presentation.

    Extracted once from the raw vignette at the start of an Apiro run.
    All downstream components use this object rather than re-parsing text.
    """
    chief_complaint: str
    symptoms: list[str] = field(default_factory=list)
    labs: dict[str, str] = field(default_factory=dict)
    imaging: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    age: Optional[int] = None
    gender: Optional[str] = None     # "M" | "F" | None
    raw_vignette: str = ""

    def all_findings(self) -> list[str]:
        """
        Flat list of all clinical findings for evidence matching.
        Combines symptoms, labs (as strings), and imaging findings.
        """
        findings = list(self.symptoms)
        for k, v in self.labs.items():
            findings.append(f"{k}: {v}")
        findings.extend(self.imaging)
        return findings

    def summary(self) -> str:
        """Human-readable one-line summary for logging."""
        parts = []
        if self.age:
            parts.append(f"{self.age}yo")
        if self.gender:
            parts.append(self.gender)
        parts.append(f"CC={self.chief_complaint[:60]}")
        parts.append(f"symptoms={len(self.symptoms)}")
        parts.append(f"labs={len(self.labs)}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"PatientContext({self.summary()})"


# ── LLM extractor ─────────────────────────────────────────────────────────────

def extract_patient_context(
    vignette: str,
    model: str = PRIMARY_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
    timeout: int = 90,
) -> PatientContext:
    """
    Extract a structured PatientContext from a raw clinical vignette.

    Makes a single LLM call with a strict JSON-format prompt.
    Falls back gracefully on any parse failure — always returns a valid
    PatientContext, even if most fields are empty.

    Args:
        vignette:   Raw clinical case text.
        model:      Ollama model name.
        ollama_url: Ollama server URL.
        timeout:    Request timeout in seconds.

    Returns:
        PatientContext with all extracted fields populated.
    """
    prompt = _EXTRACTION_PROMPT.format(vignette=vignette.strip())
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 400},
    }

    raw_json: dict = {}
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "{}")
        raw_json = json.loads(response_text)
        logger.debug(f"[PatientContext] Extraction raw: {response_text[:200]}")
    except json.JSONDecodeError as e:
        logger.warning(f"[PatientContext] JSON parse failed: {e} — using empty context.")
    except Exception as e:
        logger.warning(f"[PatientContext] LLM extraction failed: {e} — using empty context.")

    # ── Safe field extraction with type coercion ──────────────────────────────
    def _str(key: str, default: str = "") -> str:
        val = raw_json.get(key, default)
        return str(val).strip() if val is not None else default

    def _int(key: str) -> Optional[int]:
        val = raw_json.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _list(key: str) -> list[str]:
        val = raw_json.get(key, [])
        if not isinstance(val, list):
            return []
        return [str(v).strip() for v in val if v]

    def _dict(key: str) -> dict[str, str]:
        val = raw_json.get(key, {})
        if not isinstance(val, dict):
            return {}
        return {str(k).strip(): str(v).strip() for k, v in val.items() if k and v}

    gender_raw = _str("gender")
    gender: Optional[str] = None
    if gender_raw.upper() in ("M", "MALE"):
        gender = "M"
    elif gender_raw.upper() in ("F", "FEMALE"):
        gender = "F"

    # Fall back: if chief complaint is empty, use the first symptom or the start of vignette
    cc = _str("chief_complaint")
    if not cc:
        symptoms = _list("symptoms")
        cc = symptoms[0] if symptoms else vignette[:100].strip()

    ctx = PatientContext(
        chief_complaint=cc,
        age=_int("age"),
        gender=gender,
        symptoms=_list("symptoms"),
        labs=_dict("labs"),
        imaging=_list("imaging"),
        history=_list("history"),
        raw_vignette=vignette,
    )
    logger.info(f"[PatientContext] Extracted: {ctx.summary()}")
    return ctx
