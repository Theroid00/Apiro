#!/usr/bin/env python3
"""
scripts/build_niah_cases.py

Generate synthetic & perturbed Clinical Needle-In-A-Haystack (NIAH) test cases.

Five adversarial test families are produced:
  a) single_needle       - one discriminative clinical needle at varying depths
  b) contradiction_needle- a hard contradiction against an obvious wrong diagnosis
  c) multi_needle         - >= 2 co-occurring buried clues (multi-hop)
  d) red_herring          - a loud chronic comorbidity sharing non-specific symptoms
  e) negation_trap        - 'patient has no [symptom]' adjacent to relevant findings

Cases are token-length configurable and are saved to data/niah_cases.json.

Example:
    python scripts/build_niah_cases.py \
        --num-cases 200 \
        --lengths 2000 8000 16000 32000 \
        --depths 0.0 0.25 0.5 0.75 1.0 \
        --out data/niah_cases.json \
        --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"

FAMILIES = (
    "single_needle",
    "contradiction_needle",
    "multi_needle",
    "red_herring",
    "negation_trap",
)

DEFAULT_LENGTHS = (2000, 8000, 16000, 32000)
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Rough heuristic: ~0.75 tokens per whitespace word for English clinical prose.
# We over-generate words and trim to hit an approximate target token count.
WORDS_PER_TOKEN = 0.75


# ---------------------------------------------------------------------------
# Clinical content banks
# ---------------------------------------------------------------------------

# Background "hay" sentences: generic, non-discriminative clinical filler.
HAY_SENTENCES: Tuple[str, ...] = (
    "The patient was seen in the outpatient clinic for a routine follow-up visit.",
    "Vital signs were recorded on arrival and documented in the chart.",
    "The patient reports adequate sleep and no recent changes in appetite.",
    "A general review of systems was largely unremarkable at this encounter.",
    "The patient denies recent travel or known sick contacts.",
    "Medication reconciliation was performed and the list was updated.",
    "Immunization history was reviewed and found to be up to date.",
    "The patient ambulates independently without assistive devices.",
    "Skin was warm and dry with no obvious rashes or lesions noted.",
    "The patient was counseled on diet, exercise, and smoking cessation.",
    "Baseline metabolic panel values were within normal reference ranges.",
    "The patient tolerated the examination without acute distress.",
    "Family history was reviewed and no new conditions were reported.",
    "Social history includes occasional alcohol use and no illicit drug use.",
    "The patient has been compliant with previously prescribed therapy.",
    "Routine screening questionnaires were completed prior to the visit.",
    "The patient expressed understanding of the care plan and next steps.",
    "No acute cardiopulmonary complaints were elicited during the interview.",
    "The nursing staff documented intake and output over the shift.",
    "The patient was scheduled for a follow-up appointment in six weeks.",
    "Physical therapy notes indicate steady progress toward mobility goals.",
    "The patient's weight was stable compared to the prior encounter.",
    "Laboratory specimens were collected and sent to the reference lab.",
    "The patient's chronic medications were refilled without changes.",
    "A brief mental status examination was grossly nonfocal.",
    "The patient reported no new allergies since the last documented visit.",
    "Bowel and bladder function were reported as unchanged and regular.",
    "The care team discussed goals of care and documented preferences.",
    "The patient's home glucose logs were reviewed and appeared consistent.",
    "Wound care instructions were reinforced and demonstrated to the patient.",
)

# Discriminative needles keyed by a canonical diagnosis.
# Each needle is a distinctive lab or prose finding that points to the diagnosis.
NEEDLE_BANK: Dict[str, Dict[str, List[str]]] = {
    "diabetic_ketoacidosis": {
        "structured": [
            "Arterial blood gas: pH 7.12, bicarbonate 8 mEq/L, anion gap 24.",
            "Serum glucose 512 mg/dL with large serum ketones and beta-hydroxybutyrate 6.2 mmol/L.",
        ],
        "prose": [
            "The patient describes deep, labored Kussmaul breathing with a fruity odor on the breath.",
            "Over two days the patient developed polyuria, polydipsia, and abdominal pain with vomiting.",
        ],
    },
    "pulmonary_embolism": {
        "structured": [
            "CT pulmonary angiography demonstrates a filling defect in the right lower lobe segmental artery.",
            "D-dimer 3,400 ng/mL FEU with a Wells score placing the patient in the high-risk category.",
        ],
        "prose": [
            "The patient reports sudden-onset pleuritic chest pain and acute dyspnea after a long flight.",
            "There is unilateral calf swelling and tenderness following prolonged immobilization.",
        ],
    },
    "bacterial_meningitis": {
        "structured": [
            "CSF analysis: WBC 2,400/uL with 92% neutrophils, glucose 18 mg/dL, protein 220 mg/dL.",
            "Gram stain of cerebrospinal fluid shows gram-positive diplococci.",
        ],
        "prose": [
            "The patient presents with fever, severe headache, photophobia, and pronounced nuchal rigidity.",
            "A positive Kernig and Brudzinski sign were elicited on examination.",
        ],
    },
    "acute_appendicitis": {
        "structured": [
            "Abdominal CT shows a dilated appendix measuring 11 mm with periappendiceal fat stranding.",
            "WBC 16,200/uL with a left shift and 84% neutrophils.",
        ],
        "prose": [
            "Pain migrated from the periumbilical region to the right lower quadrant with rebound tenderness.",
            "A positive McBurney point tenderness and psoas sign were documented on exam.",
        ],
    },
    "hyperkalemia": {
        "structured": [
            "Serum potassium 7.1 mEq/L with peaked T waves and a widened QRS on ECG.",
            "Point-of-care potassium repeated at 6.9 mEq/L confirming true hyperkalemia.",
        ],
        "prose": [
            "The patient reports profound muscle weakness and palpitations with a history of missed dialysis.",
            "The rhythm strip shows progressive sine-wave morphology concerning for cardiac instability.",
        ],
    },
    "acute_myocardial_infarction": {
        "structured": [
            "ECG shows 3 mm ST-segment elevation in leads II, III, and aVF.",
            "High-sensitivity troponin rose from 45 to 1,220 ng/L over three hours.",
        ],
        "prose": [
            "The patient describes crushing substernal chest pressure radiating to the left arm and jaw.",
            "Symptoms were accompanied by diaphoresis, nausea, and a sense of impending doom.",
        ],
    },
}

# For contradiction cases: an "obvious but wrong" diagnosis that superficially
# fits, and a hard contradicting fact that rules it out.
CONTRADICTION_BANK: Dict[str, List[Dict[str, str]]] = {
    "diabetic_ketoacidosis": [
        {
            "wrong_diagnosis": "hyperosmolar hyperglycemic state",
            "contradiction": "However, serum and urine ketones are strongly positive with an elevated anion gap, which excludes a purely hyperosmolar nonketotic process.",
        }
    ],
    "pulmonary_embolism": [
        {
            "wrong_diagnosis": "community-acquired pneumonia",
            "contradiction": "However, the chest radiograph shows clear lung fields with no infiltrate or consolidation, arguing against pneumonia.",
        }
    ],
    "bacterial_meningitis": [
        {
            "wrong_diagnosis": "viral meningitis",
            "contradiction": "However, CSF glucose is markedly low with neutrophilic pleocytosis, which is inconsistent with a viral etiology.",
        }
    ],
    "acute_appendicitis": [
        {
            "wrong_diagnosis": "gastroenteritis",
            "contradiction": "However, focal right-lower-quadrant peritoneal signs with imaging-confirmed appendiceal inflammation are inconsistent with simple gastroenteritis.",
        }
    ],
    "hyperkalemia": [
        {
            "wrong_diagnosis": "anxiety-related palpitations",
            "contradiction": "However, the ECG demonstrates peaked T waves and QRS widening that are inconsistent with a benign anxiety etiology.",
        }
    ],
    "acute_myocardial_infarction": [
        {
            "wrong_diagnosis": "musculoskeletal chest wall pain",
            "contradiction": "However, dynamic ST elevation with a rising troponin trend is inconsistent with a musculoskeletal cause.",
        }
    ],
}

# For red-herring cases: a loud chronic comorbidity that shares non-specific
# symptoms with the acute condition and can distract a reasoner.
RED_HERRING_BANK: Dict[str, List[str]] = {
    "diabetic_ketoacidosis": [
        "The patient carries a long-standing diagnosis of generalized anxiety disorder and frequently reports nausea and rapid breathing during panic episodes, extensively documented across many prior visits.",
    ],
    "pulmonary_embolism": [
        "The patient has severe chronic obstructive pulmonary disease with baseline dyspnea and chronic chest tightness, a comorbidity that dominates the prior charting.",
    ],
    "bacterial_meningitis": [
        "The patient has a well-documented history of chronic migraine with frequent severe headaches and photophobia treated for years.",
    ],
    "acute_appendicitis": [
        "The patient has long-standing irritable bowel syndrome with recurrent diffuse abdominal pain, extensively described in prior notes.",
    ],
    "hyperkalemia": [
        "The patient has chronic fibromyalgia with pervasive muscle weakness and fatigue documented over many years of care.",
    ],
    "acute_myocardial_infarction": [
        "The patient has chronic gastroesophageal reflux disease with recurrent burning chest discomfort noted repeatedly in the record.",
    ],
}

# For multi-needle cases we require two co-occurring clues that jointly imply
# the diagnosis. We reuse structured + prose from NEEDLE_BANK.

# For negation traps: an explicit negation of a relevant symptom placed next to
# real findings, to test that the model does not over-anchor on the negation.
NEGATION_SYMPTOMS: Dict[str, List[str]] = {
    "diabetic_ketoacidosis": ["chest pain", "hematemesis"],
    "pulmonary_embolism": ["fever", "cough productive of sputum"],
    "bacterial_meningitis": ["focal weakness", "seizure activity"],
    "acute_appendicitis": ["diarrhea", "urinary symptoms"],
    "hyperkalemia": ["chest pain", "syncope"],
    "acute_myocardial_infarction": ["fever", "cough"],
}

# Questions posed per family (used to build the evaluation prompt/answer).
QUESTION_TEMPLATES: Dict[str, str] = {
    "single_needle": "Based only on the clinical note above, what is the single most likely diagnosis?",
    "contradiction_needle": "Based only on the clinical note above, what is the correct diagnosis, and which superficially plausible diagnosis is ruled out?",
    "multi_needle": "Based only on the clinical note above, what diagnosis is supported by combining the buried findings?",
    "red_herring": "Based only on the clinical note above, what is the acute diagnosis that requires attention, ignoring chronic distractors?",
    "negation_trap": "Based only on the clinical note above, what is the most likely diagnosis, accounting for any explicitly negated symptoms?",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NiahCase:
    case_id: str
    family: str
    diagnosis: str
    target_tokens: int
    approx_tokens: int
    depth_fraction: Optional[float]
    question: str
    answer: str
    needles: List[str] = field(default_factory=list)
    distractors: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    context: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Token / text helpers
# ---------------------------------------------------------------------------


def _try_import_tiktoken():
    try:
        import tiktoken  # type: ignore

        return tiktoken
    except Exception:
        return None


class TokenCounter:
    """Token counter that prefers tiktoken and falls back to a heuristic."""

    def __init__(self) -> None:
        self._enc = None
        tt = _try_import_tiktoken()
        if tt is not None:
            try:
                self._enc = tt.get_encoding("cl100k_base")
            except Exception:
                self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # Heuristic fallback based on whitespace word count.
        words = len(text.split())
        return max(1, int(round(words / WORDS_PER_TOKEN)))


def build_hay(rng: random.Random, sentence_count: int) -> List[str]:
    """Return a list of hay sentences (with light shuffling / repetition)."""
    if sentence_count <= 0:
        return []
    out: List[str] = []
    while len(out) < sentence_count:
        batch = list(HAY_SENTENCES)
        rng.shuffle(batch)
        out.extend(batch)
    return out[:sentence_count]


def insert_at_depth(
    hay_sentences: List[str], needle_blocks: List[str], depth_fraction: float
) -> List[str]:
    """
    Insert needle blocks into the hay at the requested depth fraction.

    depth_fraction 0.0 -> very top; 1.0 -> very bottom.
    Multiple needle blocks are inserted contiguously at the computed position.
    """
    n = len(hay_sentences)
    depth_fraction = min(1.0, max(0.0, depth_fraction))
    idx = int(round(depth_fraction * n))
    idx = min(n, max(0, idx))
    return hay_sentences[:idx] + list(needle_blocks) + hay_sentences[idx:]


def trim_to_target(
    counter: TokenCounter,
    sentences: List[str],
    protected_indices: List[int],
    target_tokens: int,
) -> List[str]:
    """
    Trim hay sentences so the joined text approaches target_tokens without
    ever removing protected (needle/distractor) sentences.
    """
    sentences = list(sentences)
    protected = set(protected_indices)

    def joined(seq: List[str]) -> str:
        return " ".join(seq)

    # Remove trailing/removable hay from the end first until at or below target.
    while counter.count(joined(sentences)) > target_tokens:
        removable = [i for i in range(len(sentences)) if i not in protected]
        if not removable:
            break
        # Remove from the end of the removable set to preserve early structure.
        remove_idx = removable[-1]
        del sentences[remove_idx]
        # Recompute protected indices after deletion.
        protected = {i - 1 if i > remove_idx else i for i in protected}
    return sentences


def make_case_id(family: str, diagnosis: str, target_tokens: int, salt: str) -> str:
    raw = f"{family}|{diagnosis}|{target_tokens}|{salt}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{family}-{digest}"


# ---------------------------------------------------------------------------
# Case builders (one per family)
# ---------------------------------------------------------------------------


def _diagnosis_pretty(diagnosis: str) -> str:
    return diagnosis.replace("_", " ")


def _pick_needle(rng: random.Random, diagnosis: str) -> Tuple[str, str]:
    """Return (kind, needle_text) picking randomly between structured/prose."""
    bank = NEEDLE_BANK[diagnosis]
    kind = rng.choice(["structured", "prose"])
    return kind, rng.choice(bank[kind])


def build_single_needle(
    rng: random.Random,
    counter: TokenCounter,
    diagnosis: str,
    target_tokens: int,
    depth: float,
) -> NiahCase:
    kind, needle = _pick_needle(rng, diagnosis)
    approx_hay = int((target_tokens * WORDS_PER_TOKEN) / 12) + 4
    hay = build_hay(rng, approx_hay)
    combined = insert_at_depth(hay, [needle], depth)
    protected = [combined.index(needle)]
    combined = trim_to_target(counter, combined, protected, target_tokens)
    context = " ".join(combined)
    approx = counter.count(context)

    answer = _diagnosis_pretty(diagnosis)
    return NiahCase(
        case_id=make_case_id("single_needle", diagnosis, target_tokens, f"{depth}-{needle}"),
        family="single_needle",
        diagnosis=diagnosis,
        target_tokens=target_tokens,
        approx_tokens=approx,
        depth_fraction=depth,
        question=QUESTION_TEMPLATES["single_needle"],
        answer=answer,
        needles=[needle],
        distractors=[],
        metadata={"needle_kind": kind},
        context=context,
    )


def build_contradiction_needle(
    rng: random.Random,
    counter: TokenCounter,
    diagnosis: str,
    target_tokens: int,
    depth: float,
) -> NiahCase:
    kind, needle = _pick_needle(rng, diagnosis)
    contra = rng.choice(CONTRADICTION_BANK[diagnosis])
    wrong_dx = contra["wrong_diagnosis"]
    contradiction_stmt = contra["contradiction"]

    # A "loud" statement asserting the wrong diagnosis, then the buried
    # contradiction + supporting needle to rule it out.
    wrong_claim = (
        f"The admitting team initially favored {wrong_dx} as the working diagnosis."
    )
    needle_blocks = [wrong_claim, needle, contradiction_stmt]

    approx_hay = int((target_tokens * WORDS_PER_TOKEN) / 12) + 4
    hay = build_hay(rng, approx_hay)
    combined = insert_at_depth(hay, needle_blocks, depth)
    protected = [combined.index(b) for b in needle_blocks]
    combined = trim_to_target(counter, combined, protected, target_tokens)
    context = " ".join(combined)
    approx = counter.count(context)

    answer = (
        f"{_diagnosis_pretty(diagnosis)}; ruled out: {wrong_dx}"
    )
    return NiahCase(
        case_id=make_case_id("contradiction_needle", diagnosis, target_tokens, wrong_dx),
        family="contradiction_needle",
        diagnosis=diagnosis,
        target_tokens=target_tokens,
        approx_tokens=approx,
        depth_fraction=depth,
        question=QUESTION_TEMPLATES["contradiction_needle"],
        answer=answer,
        needles=[needle, contradiction_stmt],
        distractors=[wrong_claim],
        metadata={"needle_kind": kind, "wrong_diagnosis": wrong_dx},
        context=context,
    )


def build_multi_needle(
    rng: random.Random,
    counter: TokenCounter,
    diagnosis: str,
    target_tokens: int,
    depth: float,
) -> NiahCase:
    bank = NEEDLE_BANK[diagnosis]
    # Require one structured and one prose clue at different depths.
    structured = rng.choice(bank["structured"])
    prose = rng.choice(bank["prose"])

    approx_hay = int((target_tokens * WORDS_PER_TOKEN) / 12) + 6
    hay = build_hay(rng, approx_hay)

    # Place first needle near the requested depth, second offset by ~0.4.
    depth_a = depth
    depth_b = min(1.0, max(0.0, depth + 0.4 if depth <= 0.5 else depth - 0.4))

    combined = insert_at_depth(hay, [structured], depth_a)
    combined = insert_at_depth(combined, [prose], depth_b)
    protected = [combined.index(structured), combined.index(prose)]
    combined = trim_to_target(counter, combined, protected, target_tokens)
    context = " ".join(combined)
    approx = counter.count(context)

    answer = _diagnosis_pretty(diagnosis)
    return NiahCase(
        case_id=make_case_id("multi_needle", diagnosis, target_tokens, f"{depth_a}-{depth_b}"),
        family="multi_needle",
        diagnosis=diagnosis,
        target_tokens=target_tokens,
        approx_tokens=approx,
        depth_fraction=depth,
        question=QUESTION_TEMPLATES["multi_needle"],
        answer=answer,
        needles=[structured, prose],
        distractors=[],
        metadata={"hops": 2, "depth_a": depth_a, "depth_b": depth_b},
        context=context,
    )


def build_red_herring(
    rng: random.Random,
    counter: TokenCounter,
    diagnosis: str,
    target_tokens: int,
    depth: float,
) -> NiahCase:
    kind, needle = _pick_needle(rng, diagnosis)
    herring = rng.choice(RED_HERRING_BANK[diagnosis])

    approx_hay = int((target_tokens * WORDS_PER_TOKEN) / 12) + 4
    hay = build_hay(rng, approx_hay)

    # The red herring is placed early and loudly; the true needle is buried deep.
    combined = insert_at_depth(hay, [herring], 0.05)
    combined = insert_at_depth(combined, [needle], depth)
    protected = [combined.index(herring), combined.index(needle)]
    combined = trim_to_target(counter, combined, protected, target_tokens)
    context = " ".join(combined)
    approx = counter.count(context)

    answer = _diagnosis_pretty(diagnosis)
    return NiahCase(
        case_id=make_case_id("red_herring", diagnosis, target_tokens, herring[:16]),
        family="red_herring",
        diagnosis=diagnosis,
        target_tokens=target_tokens,
        approx_tokens=approx,
        depth_fraction=depth,
        question=QUESTION_TEMPLATES["red_herring"],
        answer=answer,
        needles=[needle],
        distractors=[herring],
        metadata={"needle_kind": kind, "comorbidity_distractor": True},
        context=context,
    )


def build_negation_trap(
    rng: random.Random,
    counter: TokenCounter,
    diagnosis: str,
    target_tokens: int,
    depth: float,
) -> NiahCase:
    kind, needle = _pick_needle(rng, diagnosis)
    negated_symptom = rng.choice(NEGATION_SYMPTOMS[diagnosis])
    negation_stmt = f"The patient explicitly reports no {negated_symptom} at this time."

    # Negation is placed immediately adjacent to the real needle.
    needle_blocks = [negation_stmt, needle]

    approx_hay = int((target_tokens * WORDS_PER_TOKEN) / 12) + 4
    hay = build_hay(rng, approx_hay)
    combined = insert_at_depth(hay, needle_blocks, depth)
    protected = [combined.index(b) for b in needle_blocks]
    combined = trim_to_target(counter, combined, protected, target_tokens)
    context = " ".join(combined)
    approx = counter.count(context)

    answer = _diagnosis_pretty(diagnosis)
    return NiahCase(
        case_id=make_case_id("negation_trap", diagnosis, target_tokens, negated_symptom),
        family="negation_trap",
        diagnosis=diagnosis,
        target_tokens=target_tokens,
        approx_tokens=approx,
        depth_fraction=depth,
        question=QUESTION_TEMPLATES["negation_trap"],
        answer=answer,
        needles=[needle],
        distractors=[negation_stmt],
        metadata={"needle_kind": kind, "negated_symptom": negated_symptom},
        context=context,
    )


FAMILY_BUILDERS: Dict[str, Callable[..., NiahCase]] = {
    "single_needle": build_single_needle,
    "contradiction_needle": build_contradiction_needle,
    "multi_needle": build_multi_needle,
    "red_herring": build_red_herring,
    "negation_trap": build_negation_trap,
}


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------


def generate_cases(
    num_cases: int,
    lengths: List[int],
    depths: List[float],
    families: List[str],
    seed: int,
) -> List[NiahCase]:
    rng = random.Random(seed)
    counter = TokenCounter()
    diagnoses = list(NEEDLE_BANK.keys())

    cases: List[NiahCase] = []
    # Round-robin across (family, length, depth, diagnosis) to keep the dataset
    # balanced regardless of num_cases.
    combos: List[Tuple[str, int, float, str]] = []
    for family in families:
        for length in lengths:
            for depth in depths:
                for diagnosis in diagnoses:
                    combos.append((family, length, depth, diagnosis))
    rng.shuffle(combos)

    i = 0
    while len(cases) < num_cases:
        family, length, depth, diagnosis = combos[i % len(combos)]
        i += 1
        builder = FAMILY_BUILDERS[family]
        try:
            case = builder(rng, counter, diagnosis, length, depth)
        except Exception as exc:  # defensive: never let one combo abort the run
            print(f"[warn] skipping combo {family}/{diagnosis}: {exc}", file=sys.stderr)
            continue
        cases.append(case)

    return cases


def summarize(cases: List[NiahCase]) -> Dict[str, object]:
    by_family: Dict[str, int] = {}
    by_length: Dict[str, int] = {}
    for c in cases:
        by_family[c.family] = by_family.get(c.family, 0) + 1
        by_length[str(c.target_tokens)] = by_length.get(str(c.target_tokens), 0) + 1
    return {"total": len(cases), "by_family": by_family, "by_length": by_length}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_niah_cases.py",
        description="Generate synthetic & perturbed Clinical NIAH test cases for Apiro.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=100,
        help="Number of test cases to generate (default: 100).",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_LENGTHS),
        help="Target token lengths, space separated (default: 2000 8000 16000 32000).",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
        help="Needle depth fractions in [0,1] (default: 0.0 0.25 0.5 0.75 1.0).",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=list(FAMILIES),
        choices=list(FAMILIES),
        help="Adversarial families to include (default: all five).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/niah_cases.json",
        help="Output JSON path (default: data/niah_cases.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2).",
    )
    args = parser.parse_args(argv)

    if args.num_cases <= 0:
        parser.error("--num-cases must be a positive integer.")
    for length in args.lengths:
        if length <= 0:
            parser.error("--lengths must all be positive integers.")
    for depth in args.depths:
        if not (0.0 <= depth <= 1.0):
            parser.error("--depths must all be within [0.0, 1.0].")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    cases = generate_cases(
        num_cases=args.num_cases,
        lengths=args.lengths,
        depths=args.depths,
        families=args.families,
        seed=args.seed,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_niah_cases.py",
        "config": {
            "num_cases": args.num_cases,
            "lengths": args.lengths,
            "depths": args.depths,
            "families": args.families,
            "seed": args.seed,
        },
        "summary": summarize(cases),
        "cases": [c.to_dict() for c in cases],
    }

    out_path = args.out
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=args.indent)
        fh.write("\n")

    summary = payload["summary"]
    print(
        f"Wrote {summary['total']} cases to {out_path}\n"
        f"  by_family: {summary['by_family']}\n"
        f"  by_length: {summary['by_length']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
