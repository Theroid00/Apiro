#!/usr/bin/env python3
"""
scripts/generate_pmc_cases.py
=============================
Builds ``data/pmc_cases.json`` from a local copy of PMC-Patients-V2 using a
four-stage LLM pipeline:

  1. Solvability filter  — reject cases whose diagnosis needs histopathology.
  2. Target extraction   — pull the acute primary diagnosis.
  3. Spoiler scrubbing   — rewrite the case, cutting everything after the
                           initial work-up so the vignette cannot leak the answer.
  4. Seed extraction     — 3-4 depth-0 findings for the belief graph.

WHAT CHANGED AND WHY
--------------------
Stage 2 used to store the model's raw reply as ``target_diagnosis``. An 8B
model asked for "ONLY the disease name" frequently answers with a preamble, and
nothing checked. Four of the ten cases in the committed ``data/pmc_cases.json``
therefore carry ground-truth labels like::

    "Here is the acute, primary presenting diagnosis:\\n\\nAppendicitis \\n\\n
     However, it's worth noting that this was initially suspected but later
     proved to be secondary to a more complex underlying condition. ..."

``apiro.eval.evaluator`` normalises that entire blob and compares a predicted
differential against it, so those cases could not be scored correctly by any
arm — they depress every reported PMC accuracy, Apiro's included.
``_clean_diagnosis_label()`` below now strips the preamble, keeps the first
real diagnosis line, and rejects anything still unusable.

The module also used to execute its whole pipeline at import time, so merely
importing it opened the dataset and started calling Ollama.

Usage:
    python scripts/generate_pmc_cases.py --n 10 --out data/pmc_cases.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

from apiro.config import OLLAMA_BASE_URL, PRIMARY_MODEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("generate_pmc_cases")

DEFAULT_SOURCE = "data/PMC-Patients-V2.json"
DEFAULT_OUT = "data/pmc_cases.json"

MIN_WORDS = 200
MAX_WORDS = 600

#: Lines the model prepends instead of answering with the label alone.
_PREAMBLE_RE = re.compile(
    r"^\s*(here (is|are)|the\s+)?(acute[, ]*)?(primary\s+)?(presenting\s+)?"
    r"(diagnosis|diagnoses|answer|disease)\b.*?:\s*$",
    re.IGNORECASE,
)

#: A label that starts one of these is commentary, not a diagnosis.
_COMMENTARY_PREFIXES = (
    "however", "note:", "note that", "it's worth", "it is worth",
    "the actual", "the original", "this was", "subsequent", "based on",
)


def _clean_diagnosis_label(raw: str) -> str | None:
    """Reduce a model reply to a single usable diagnosis label.

    Returns ``None`` when nothing usable survives, so the caller can drop the
    case rather than commit an ungradeable ground truth.
    """
    if not raw or not raw.strip():
        return None

    for line in raw.strip().splitlines():
        candidate = line.strip().strip("*_`").strip()
        candidate = re.sub(r"^\s*\d+\s*[.)]\s*|^\s*[-*•]\s*", "", candidate).strip()
        if not candidate:
            continue
        if _PREAMBLE_RE.match(candidate):
            continue
        if candidate.lower().startswith(_COMMENTARY_PREFIXES):
            continue
        # A diagnosis label is a noun phrase, not a sentence. Anything this
        # long is prose that would be normalised into meaningless tokens.
        if len(candidate.split()) > 12:
            continue
        return candidate.rstrip(".").strip()
    return None


def _ollama(prompt: str, timeout: int, fmt: str | None = None) -> str | None:
    """Single Ollama generate call. Returns None on any failure."""
    payload: dict = {"model": PRIMARY_MODEL, "prompt": prompt, "stream": False}
    if fmt:
        payload["format"] = fmt
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:  # noqa: BLE001
        # Previously a bare `except:` that swallowed KeyboardInterrupt too.
        logger.warning(f"Ollama call failed: {exc}")
        return None


def generate_seed_nodes(vignette: str) -> list[dict]:
    """Extract 3-4 depth-0 seed findings for the belief graph."""
    prompt = f"""
    You are a medical AI. Read the following clinical case report and extract 3 to 4 core initial clinical findings (symptoms, signs, or initial lab results) as seed nodes for a diagnostic engine.
    IMPORTANT: At least one seed node MUST represent the patient's primary chief complaint (the acute reason they sought care). Do not focus primarily on incidental anatomical anomalies unless they are the direct cause of the acute presentation.
    Output EXACTLY a JSON array of objects. Each object must have 'id', 'claim', 'domain', 'depth' (always 0), and 'entropy' (a float between 0.1 and 0.9 representing initial uncertainty).
    Example:
    [
      {{"id": "s1", "claim": "<extract a specific symptom from the vignette here>", "domain": "symptom", "depth": 0, "entropy": 0.8}}
    ]

    Case Report:
    {vignette}

    JSON Output:
    """
    raw = _ollama(prompt, timeout=120, fmt="json")
    if raw is None:
        return []
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        logger.warning(f"Seed JSON parse failed: {exc}")
        return []
    if isinstance(parsed, dict):
        return [parsed]
    return parsed if isinstance(parsed, list) else []


def build_cases(source: Path, n: int, skip: int) -> list[dict]:
    """Run the four-stage pipeline over the source dataset."""
    logger.info(f"Loading {source}...")
    with open(source) as fh:
        dataset = json.load(fh)

    cases: list[dict] = []
    skipped_unusable_label = 0
    remaining_skip = skip

    for row in dataset:
        if len(cases) >= n:
            break

        vignette = row.get("patient", "")
        n_words = len(vignette.split())
        if not (MIN_WORDS < n_words < MAX_WORDS):
            continue
        if remaining_skip > 0:
            remaining_skip -= 1
            continue

        # ---- Stage 1: solvability filter --------------------------------- #
        solvability = _ollama(
            "Does this case report present a diagnosis that is reasonably deducible "
            "from the initial clinical presentation (symptoms, signs, and basic "
            "labs/imaging)? If the diagnosis fundamentally requires a biopsy, "
            "pathology report, or exploratory surgery to determine (e.g. specific "
            f"histological cancer subtypes), answer NO. Otherwise, answer YES.\n\nCase: {vignette}",
            timeout=60,
        )
        if solvability is None or "YES" not in solvability.strip().upper():
            continue

        # ---- Stage 2: acute target extraction ---------------------------- #
        raw_diagnosis = _ollama(
            "Extract the ACUTE, primary presenting diagnosis that the clinicians "
            "arrive at for this specific episode. Do NOT extract chronic, "
            "pre-existing background conditions (e.g., if a patient with a history "
            "of asthma presents with a pulmonary embolism, extract 'Pulmonary "
            "embolism'). Output ONLY the disease name, nothing else.\n\n"
            f"Case: {vignette}",
            timeout=60,
        )
        if raw_diagnosis is None:
            continue

        target = _clean_diagnosis_label(raw_diagnosis)
        if target is None:
            # Dropping the case is the right call: an ungradeable ground truth
            # silently penalises every arm for the rest of the benchmark's life.
            skipped_unusable_label += 1
            logger.warning(
                f"Dropping case — no usable diagnosis label in: {raw_diagnosis[:90]!r}"
            )
            continue

        logger.info(f"Processing case {len(cases) + 1}: target={target!r}")

        # ---- Stage 3: spoiler scrubbing ---------------------------------- #
        scrubbed = _ollama(
            "Rewrite this case report as a diagnostic challenge for a medical "
            "student. Stop the narrative immediately after the initial clinical "
            "presentation, physical exam, and first-line labs/imaging. Completely "
            "remove any mention of biopsies, surgical exploration, specific "
            "treatments given, or the final diagnosis. Do not add any introductory "
            f"or concluding remarks. Output ONLY the rewritten text.\n\nCase: {vignette}",
            timeout=120,
        )
        scrubbed_vignette = (scrubbed or vignette).strip()

        # ---- Stage 4: seed extraction ------------------------------------ #
        seeds = generate_seed_nodes(scrubbed_vignette)
        if not seeds:
            continue

        cases.append({
            "case_id": f"pmc_case_{len(cases) + 1}",
            "description": "Real world case report from PMC (Scrubbed)",
            "target_diagnosis": target,
            "vignette": scrubbed_vignette,
            "seed_nodes": seeds,
        })

    if skipped_unusable_label:
        logger.warning(
            f"{skipped_unusable_label} case(s) dropped for an unusable "
            f"diagnosis label."
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate scrubbed PMC diagnostic cases for the Apiro benchmark."
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help=f"Local PMC-Patients-V2 JSON (default: {DEFAULT_SOURCE}).")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help=f"Output path (default: {DEFAULT_OUT}).")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of cases to emit (default: 10).")
    parser.add_argument("--skip", type=int, default=50,
                        help="Length-eligible rows to skip before sampling (default: 50).")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.exists():
        logger.error(
            f"Source dataset not found: {source}\n"
            f"        PMC-Patients-V2.json is not checked in (see .gitignore); "
            f"download it before running this generator."
        )
        return 2

    cases = build_cases(source, n=args.n, skip=args.skip)
    if not cases:
        logger.error("No cases generated.")
        return 1

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cases, fh, indent=2)

    logger.info(f"Saved {len(cases)} PMC cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
