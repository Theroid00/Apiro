"""
apiro/corpus/ddxplus_adapter.py — DDXPlus → Apiro evaluation cases
===================================================================

DDXPlus (Tchango et al., arXiv:2205.09148; CC-BY) is 1.3M synthetic patients
across 49 pathologies. Two things make it the most valuable external set
available to this project:

  * It ships a ground-truth **ranked differential**, not a single label. Every
    rank-aware metric in apiro/eval/metrics.py becomes meaningful against a
    real reference ordering rather than a one-hot answer.
  * Its evidences are **structured**, so a counterfactual control/trap pair can
    be constructed programmatically by swapping discriminative features —
    rather than from the hand-written bank in build_niah_cases.py. It is the
    substrate MedEinst (arXiv:2601.06636) was itself built from.

WHAT THIS MODULE DOES
    Renders one CSV row into a clinical note a model can read, plus the ground
    truth and the reference differential.

    Raw rows are not readable. EVIDENCES holds codes ("E_53",
    "E_54_@_V_179"); PATHOLOGY and DIFFERENTIAL_DIAGNOSIS may hold French
    condition names. release_evidences.json and release_conditions.json are the
    dictionaries that resolve both, which is why scripts/fetch_datasets.py
    treats them as required rather than optional.

DEFENSIVE BY DESIGN
    The dictionaries' internal key names are not standardised across DDXPlus
    mirrors — English text appears variously under "question_en", "en",
    "cond-name-eng". Every lookup here tries a list of candidates and falls
    back to the raw code rather than raising. A field that cannot be resolved
    degrades one sentence; it does not take down a benchmark run.

    Run `python scripts/fetch_datasets.py --verify-only` to see the schema
    actually on disk.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

__all__ = ["DDXPlusCase", "DDXPlusAdapter"]

#: Columns read from the CSV. scripts/fetch_datasets.py checks the file
#: against this set and fails loudly on a mismatch.
DDXPLUS_COLUMNS = {
    "AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE",
    "DIFFERENTIAL_DIAGNOSIS",
}

# Key names that have been observed to carry the same content across mirrors.
_EN_QUESTION_KEYS = ("question_en", "question_eng", "question", "en", "name")
_EN_CONDITION_KEYS = ("cond-name-eng", "cond_name_eng", "condition_name_eng",
                      "cond-name-en", "en", "condition_name", "name")
_VALUE_MEANING_KEYS = ("value_meaning", "value-meaning", "values", "possible-values")


def _first_str(mapping: Any, keys: tuple[str, ...]) -> Optional[str]:
    """First key in `keys` whose value is a non-empty string."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_listish(raw: Any) -> list:
    """Parse a stringified Python/JSON list, tolerating both encodings."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:  # noqa: BLE001 - try the next encoding
            continue
    return []


@dataclass
class DDXPlusCase:
    """One DDXPlus patient rendered for evaluation.

    Attributes:
        case_id:      Stable id derived from the row index.
        vignette:     Readable clinical note built from the resolved evidences.
        ground_truth: English pathology name (the reference answer).
        differential: Reference differential, most likely first.
        age, sex:     Demographics, already stated in the vignette.
        n_evidences:  How many evidence codes resolved into text.
        raw_pathology: The dataset's own pathology string, before translation.
    """

    case_id: str
    vignette: str
    ground_truth: str
    differential: list[str] = field(default_factory=list)
    age: Optional[int] = None
    sex: Optional[str] = None
    n_evidences: int = 0
    raw_pathology: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "vignette": self.vignette,
            "ground_truth": self.ground_truth,
            "differential": list(self.differential),
            "age": self.age,
            "sex": self.sex,
            "n_evidences": self.n_evidences,
            "raw_pathology": self.raw_pathology,
        }


class DDXPlusAdapter:
    """Loads DDXPlus and renders rows as Apiro evaluation cases.

    Args:
        data_dir: Directory holding test.csv and the two release_*.json
            dictionaries, as written by scripts/fetch_datasets.py.

    Raises:
        FileNotFoundError: If the directory or the dictionaries are absent,
            with the command that fixes it. Proceeding without them would
            yield vignettes made of raw evidence codes — technically a
            benchmark run, and a meaningless one.
    """

    def __init__(self, data_dir: str | Path = "data/datasets/ddxplus"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"DDXPlus not found at {self.data_dir}.\n"
                f"  Fetch it with: python scripts/fetch_datasets.py --only ddxplus"
            )

        self.evidences = self._load_json("release_evidences.json")
        self.conditions = self._load_json("release_conditions.json")

        # pathology string -> English name, built once.
        self._condition_en: dict[str, str] = {}
        for key, entry in self.conditions.items():
            english = _first_str(entry, _EN_CONDITION_KEYS) or key
            self._condition_en[key] = english
            self._condition_en[key.lower()] = english

        logger.info(
            f"[DDXPlus] {len(self.evidences)} evidences, "
            f"{len(self.conditions)} conditions loaded from {self.data_dir}"
        )

    def _load_json(self, filename: str) -> dict:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{filename} missing from {self.data_dir}. Without it, evidence "
                f"codes cannot be rendered as text and every vignette would be "
                f"unreadable.\n"
                f"  Fetch it with: python scripts/fetch_datasets.py --only ddxplus"
            )
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Evidence rendering
    # ------------------------------------------------------------------
    def _render_evidence(self, code: str) -> Optional[str]:
        """Turn one evidence code into an English clause.

        Handles both bare codes ("E_53") and valued codes
        ("E_54_@_V_179"), resolving the value through the evidence's own
        value-meaning table when one is present.
        """
        code = code.strip()
        if not code:
            return None

        name, _, value = code.partition("_@_")
        entry = self.evidences.get(name)
        if entry is None:
            return None                      # unknown code: drop the clause

        question = _first_str(entry, _EN_QUESTION_KEYS)
        if not question:
            return None

        if not value:
            # A bare code means the patient answered this question
            # affirmatively; the arrow makes that explicit rather than leaving
            # a bare interrogative to be read as a finding.
            return f"{question} -> yes"

        # Resolve the value code through whichever meaning table exists.
        rendered_value = value
        for key in _VALUE_MEANING_KEYS:
            table = entry.get(key)
            if isinstance(table, dict) and value in table:
                meaning = table[value]
                resolved = (
                    _first_str(meaning, ("en", "eng", "value"))
                    if isinstance(meaning, dict) else
                    (meaning if isinstance(meaning, str) else None)
                )
                if resolved:
                    rendered_value = resolved
                break

        return f"{question} -> {rendered_value}"

    def render_vignette(self, row: dict) -> tuple[str, int]:
        """Build a readable note from a CSV row. Returns (text, n_resolved)."""
        age, sex = row.get("AGE"), row.get("SEX")
        header_bits = []
        if age not in (None, ""):
            header_bits.append(f"{age}-year-old")
        if sex in ("M", "F"):
            header_bits.append({"M": "male", "F": "female"}[sex])
        header = (
            f"The patient is a {' '.join(header_bits)}."
            if header_bits else "Patient demographics are not recorded."
        )

        codes = _parse_listish(row.get("EVIDENCES"))
        initial = str(row.get("INITIAL_EVIDENCE") or "").strip()
        # The initial evidence is the presenting complaint; lead with it.
        ordered = ([initial] if initial else []) + [c for c in codes if c != initial]

        clauses = [c for c in (self._render_evidence(str(code)) for code in ordered) if c]

        if not clauses:
            return header, 0

        return (
            f"{header}\n\n"
            f"Findings elicited on history and examination (each line is a "
            f"question the patient answered, with the recorded response):\n"
            + "\n".join(f"- {clause}" for clause in clauses),
            len(clauses),
        )

    def _english_condition(self, name: str) -> str:
        name = (name or "").strip()
        return self._condition_en.get(name) or self._condition_en.get(name.lower()) or name

    def _parse_differential(self, raw: Any) -> list[str]:
        """Reference differential as English names, most likely first."""
        entries = _parse_listish(raw)
        scored: list[tuple[float, str]] = []
        for entry in entries:
            if isinstance(entry, (list, tuple)) and entry:
                name = str(entry[0])
                try:
                    prob = float(entry[1]) if len(entry) > 1 else 0.0
                except (TypeError, ValueError):
                    prob = 0.0
                scored.append((prob, self._english_condition(name)))
            elif isinstance(entry, str):
                scored.append((0.0, self._english_condition(entry)))
        # The dataset already orders these, but sorting makes the contract
        # explicit rather than inherited from file order.
        scored.sort(key=lambda t: t[0], reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for _, name in scored:
            if name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
        return out

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _iter_rows(self, split: str) -> Iterator[dict]:
        path = self.data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                f"  Fetch it with: python scripts/fetch_datasets.py "
                f"--only ddxplus --splits {split}"
            )
        with open(path, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)

    def load_cases(
        self,
        n: int = 50,
        split: str = "test",
        seed: int = 42,
        min_evidences: int = 3,
        reservoir_limit: int = 200_000,
    ) -> list[DDXPlusCase]:
        """Sample `n` renderable cases from a split.

        Reservoir-samples so the CSV is streamed once rather than held in
        memory — the test split alone is 135,000 rows.

        Args:
            n: How many cases to return.
            split: train | validate | test.
            seed: Sampling seed, so a run is reproducible.
            min_evidences: Skip rows that resolve to fewer clauses than this;
                a two-line note is not a diagnostic task.
            reservoir_limit: Stop reading after this many rows.

        Returns:
            Up to `n` DDXPlusCase objects.

        Raises:
            RuntimeError: If nothing renders — which means the dictionaries did
                not resolve the evidence codes, and every vignette would have
                been demographics only.
        """
        rng = random.Random(seed)
        reservoir: list[tuple[int, dict]] = []
        n_seen = n_skipped = 0

        for index, row in enumerate(self._iter_rows(split)):
            if index >= reservoir_limit:
                break
            n_seen += 1
            if len(reservoir) < n:
                reservoir.append((index, row))
            else:
                j = rng.randrange(n_seen)
                if j < n:
                    reservoir[j] = (index, row)

        cases: list[DDXPlusCase] = []
        for index, row in reservoir:
            vignette, n_resolved = self.render_vignette(row)
            if n_resolved < min_evidences:
                n_skipped += 1
                continue
            raw_pathology = str(row.get("PATHOLOGY") or "").strip()
            cases.append(DDXPlusCase(
                case_id=f"ddxplus_{split}_{index}",
                vignette=vignette,
                ground_truth=self._english_condition(raw_pathology),
                differential=self._parse_differential(row.get("DIFFERENTIAL_DIAGNOSIS")),
                age=int(row["AGE"]) if str(row.get("AGE", "")).isdigit() else None,
                sex=row.get("SEX") or None,
                n_evidences=n_resolved,
                raw_pathology=raw_pathology,
            ))

        if not cases:
            raise RuntimeError(
                f"No DDXPlus row rendered into a usable vignette "
                f"({n_skipped} skipped for fewer than {min_evidences} resolved "
                f"evidences, out of {n_seen} read).\n"
                f"  This means release_evidences.json did not resolve the "
                f"EVIDENCES codes — the dictionary schema differs from what "
                f"apiro/corpus/ddxplus_adapter.py expects.\n"
                f"  Run: python scripts/fetch_datasets.py --verify-only"
            )

        if n_skipped:
            logger.info(
                f"[DDXPlus] {len(cases)} cases; {n_skipped} skipped for too few "
                f"resolved evidences."
            )
        return cases
