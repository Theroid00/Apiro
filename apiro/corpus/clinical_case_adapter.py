"""
corpus/clinical_case_adapter.py
================================
Adapts HuggingFace clinical case datasets into Apiro belief-graph seed nodes.

Replaces the MIMIC adapter. Uses two datasets (both free, no credentials):

  CUPCase   — ofir408/CupCase  (3,562 rare clinical cases + correct diagnosis)
              Best for testing rabbit hole detection (unusual, ambiguous cases)
              Columns: clean_case_presentation, correct_diagnosis, distractor1-3

  VivaBench — chychiu/VivaBench  (990 clinician-validated cases with ICD-10)
              Best for path-length evaluation (structured labs/vitals/imaging +
              ground truth diagnosis per case)
              Columns: uid, vignette, specialty_group, diagnosis, clinicalcase

Data flow:
    ClinicalCaseAdapter.load_cupcase(n) → list[ClinicalCase]
    ClinicalCaseAdapter.load_vivabench(n) → list[ClinicalCase]
    ClinicalCaseAdapter.build_cases(cases) → list[EvalCase]
    Each EvalCase has: case_id, description, ground_truth, seed_nodes

Usage:
    from apiro.corpus.clinical_case_adapter import ClinicalCaseAdapter
    adapter = ClinicalCaseAdapter()
    cases = adapter.load_vivabench(n=10)
    eval_cases = adapter.build_cases(cases)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from apiro.graph.node import Node
from apiro.config import (
    SEED_ENTROPY_BY_FINDING_TYPE,
    SEED_ENTROPY_DEFAULT,
    VITAL_THRESHOLDS,
)

logger = logging.getLogger(__name__)



# ── ClinicalCase intermediate ──────────────────────────────────────────────────

@dataclass
class ClinicalCase:
    """
    Normalised clinical case parsed from either CUPCase or VivaBench.

    Fields:
        case_id:         Unique string ID.
        source:          'cupcase' or 'vivabench'.
        description:     Short description of the case.
        narrative:       Free-text clinical presentation (narrative paragraph).
        ground_truth:    Correct diagnosis string (used as eval target).
        specialty:       Medical specialty (from VivaBench or inferred).
        distractors:     Plausible wrong diagnoses (CUPCase distractor1-3).
        structured:      Dict of structured findings (labs, vitals, imaging, PMH)
                         parsed from VivaBench clinicalcase JSON.
    """
    case_id:      str
    source:       str
    description:  str
    narrative:    str
    ground_truth: str
    specialty:    str               = "unknown"
    distractors:  list[str]         = field(default_factory=list)
    structured:   dict              = field(default_factory=dict)


# ── PatientFinding ─────────────────────────────────────────────────────────────

@dataclass
class PatientFinding:
    """
    A single structured clinical finding extracted from a case record.

    finding_type: symptom | lab | vital | history | diagnosis | imaging
    value:        Text value (e.g. "Troponin I: 4.2 ng/mL", "STEMI")
    units:        Unit string if applicable.
    confidence:   1.0 for verified/structured data, lower for inferred.
    source:       Where this came from ("vivabench/labs", "cupcase/narrative").
    """
    finding_type: str
    value:        str
    units:        str   = ""
    confidence:   float = 1.0
    source:       str   = ""

    def to_claim(self) -> str:
        if self.units:
            return f"{self.value} ({self.units}) — {self.finding_type}"
        return f"{self.value} — {self.finding_type}"


# ── findings_to_seed_nodes ─────────────────────────────────────────────────────

def findings_to_seed_nodes(
    findings: list[PatientFinding],
    entropy_engine=None,
    default_entropy: float = SEED_ENTROPY_DEFAULT,
) -> list[Node]:
    """
    Convert PatientFindings into BeliefGraph seed Nodes.

    Args:
        findings:        Patient findings from the adapter.
        entropy_engine:  EntropyEngine instance for real epistemic entropy.
                         If None, seeds nodes with heuristic entropy from
                         config.SEED_ENTROPY_BY_FINDING_TYPE (much faster than
                         calling Ollama for every seed).
        default_entropy: Final fallback if finding_type not in the heuristic map.

    Returns:
        List of Node objects ready to be added to BeliefGraph as seeds.
    """
    from apiro.graph.expander import classify_domain

    nodes = []
    for i, finding in enumerate(findings):
        claim  = finding.to_claim()
        domain = classify_domain(claim)

        if entropy_engine is not None:
            try:
                entropy = entropy_engine.epistemic_certainty_entropy(claim, context_chunks=[])
            except Exception:
                # Fall back to heuristic if Ollama times out / errors
                entropy = SEED_ENTROPY_BY_FINDING_TYPE.get(finding.finding_type, default_entropy)
        else:
            # Use heuristic lookup — avoids the ~8s Ollama call per seed node.
            # This saves ~80s per 10-case batch while preserving priority ordering.
            entropy = SEED_ENTROPY_BY_FINDING_TYPE.get(finding.finding_type, default_entropy)

        node = Node(
            id=f"seed_{i}",
            claim=claim,
            domain=domain,
            entropy_score=entropy,
            resolved=False,
            depth=0,
            sources=[finding.source],
            metadata={
                "finding_type": finding.finding_type,
                "units":        finding.units,
                "confidence":   finding.confidence,
                "source":       finding.source,
            },
        )
        nodes.append(node)

    logger.info(f"[ClinicalCaseAdapter] Created {len(nodes)} seed nodes.")
    return nodes



# ── ClinicalCaseAdapter ────────────────────────────────────────────────────────

class ClinicalCaseAdapter:
    """
    Loads clinical cases from HuggingFace datasets and converts them into
    Apiro evaluation cases with seed nodes and ground truth labels.

    No credentials required. Downloads are cached locally by HuggingFace.
    """

    # ── CUPCase ────────────────────────────────────────────────────────────

    def load_cupcase(
        self,
        n: int = 10,
        specialty_filter: Optional[str] = None,
        seed: int = 42,
    ) -> list[ClinicalCase]:
        """
        Load n cases from CUPCase (ofir408/CupCase).

        CUPCase is ideal for rabbit hole testing — rare and ambiguous cases
        where BFS will chase distractors and entropy-first should resist.

        Args:
            n:                Number of cases to load.
            specialty_filter: If set, only include cases whose correct_diagnosis
                              contains this substring (case-insensitive).
            seed:             Random seed for reproducible sampling.

        Returns:
            List of ClinicalCase objects.
        """
        from datasets import load_dataset
        logger.info(f"[ClinicalCaseAdapter] Loading CUPCase (n={n})...")
        ds = load_dataset("ofir408/CupCase", split="test")

        # Optional filter
        if specialty_filter:
            ds = ds.filter(
                lambda r: specialty_filter.lower() in r["correct_diagnosis"].lower()
            )

        # Shuffle and select
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

        cases = []
        for i, row in enumerate(ds):
            cases.append(ClinicalCase(
                case_id=f"cupcase_{i:04d}",
                source="cupcase",
                description=row["correct_diagnosis"][:80],
                narrative=row["clean_case_presentation"],
                ground_truth=row["correct_diagnosis"],
                distractors=[
                    row.get("distractor1", ""),
                    row.get("distractor2", ""),
                    row.get("distractor3", ""),
                ],
            ))

        logger.info(f"[ClinicalCaseAdapter] Loaded {len(cases)} CUPCase cases.")
        return cases

    # ── VivaBench ──────────────────────────────────────────────────────────

    def load_vivabench(
        self,
        n: int = 10,
        specialty: Optional[str] = None,
        seed: int = 42,
    ) -> list[ClinicalCase]:
        """
        Load n cases from VivaBench (chychiu/VivaBench).

        VivaBench is ideal for path-length comparison — each case has a
        verified final diagnosis (with ICD-10) and rich structured data
        (labs, vitals, imaging, PMH) for granular seed nodes.

        Args:
            n:         Number of cases to load.
            specialty: If set, filter to this specialty group substring.
                       Available: 'Cardiovascular', 'Infectious Disease',
                       'Neurological', 'Gastrointestinal', 'Hematology',
                       'Endocrine', 'Pediatric', 'Respiratory', 'Musculoskeletal'
            seed:      Random seed for reproducible sampling.

        Returns:
            List of ClinicalCase objects.
        """
        from datasets import load_dataset
        logger.info(f"[ClinicalCaseAdapter] Loading VivaBench (n={n}, specialty={specialty})...")
        ds = load_dataset("chychiu/VivaBench", split="test")

        if specialty:
            ds = ds.filter(
                lambda r: specialty.lower() in r["specialty_group"].lower()
            )

        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

        cases = []
        for i, row in enumerate(ds):
            # Parse clinical case JSON
            cc = row.get("clinicalcase", {})
            if isinstance(cc, str):
                try:
                    cc = json.loads(cc)
                except json.JSONDecodeError:
                    cc = {}

            # Primary diagnosis
            diag_list = row.get("diagnosis", [])
            if isinstance(diag_list, str):
                import ast
                try:
                    diag_list = ast.literal_eval(diag_list)
                except Exception:
                    try:
                        diag_list = json.loads(diag_list)
                    except Exception:
                        diag_list = [diag_list]

            ground_truth = diag_list[0] if diag_list else "unknown"


            # Narrative: use vignette (sanitised presentation) or history_freetext
            narrative = (
                row.get("vignette", "")
                or cc.get("history_freetext", "")
                or ""
            )

            cases.append(ClinicalCase(
                case_id=row.get("uid", f"vivabench_{i:04d}"),
                source="vivabench",
                description=f"{row.get('specialty_group', '')}: {ground_truth[:60]}",
                narrative=narrative,
                ground_truth=ground_truth,
                specialty=row.get("specialty_group", "unknown"),
                structured=cc,
            ))

        logger.info(f"[ClinicalCaseAdapter] Loaded {len(cases)} VivaBench cases.")
        return cases

    # ── Findings extraction ────────────────────────────────────────────────

    def _extract_findings(self, case: ClinicalCase) -> list[PatientFinding]:
        """
        Extract structured PatientFindings from a ClinicalCase.

        For CUPCase: parse the narrative paragraph into 1-2 symptom findings.
        For VivaBench: extract labs, vitals, imaging, PMH from structured JSON.
        """
        findings: list[PatientFinding] = []

        if case.source == "cupcase":
            findings.extend(self._parse_cupcase_narrative(case))

        elif case.source == "vivabench":
            findings.extend(self._parse_vivabench_structured(case))

        if not findings:
            # Fallback: use the narrative as a single finding
            findings.append(PatientFinding(
                finding_type="symptom",
                value=case.narrative[:300].strip(),
                source=f"{case.source}/narrative",
            ))

        return findings

    def _parse_cupcase_narrative(self, case: ClinicalCase) -> list[PatientFinding]:
        """
        CUPCase has only a narrative. Split it into the chief complaint
        (first sentence) and the full presentation as separate findings.
        This gives the graph 2 seed nodes: one specific, one broad.
        """
        findings = []
        text = case.narrative.strip()

        # First sentence as chief complaint
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) > 20:
            findings.append(PatientFinding(
                finding_type="symptom",
                value=first_sentence,
                source="cupcase/chief_complaint",
                confidence=1.0,
            ))

        # Full narrative (capped at 400 chars) as clinical presentation
        if len(text) > len(first_sentence) + 20:
            findings.append(PatientFinding(
                finding_type="history",
                value=text[:400],
                source="cupcase/narrative",
                confidence=0.9,
            ))

        return findings

    def _parse_vivabench_structured(self, case: ClinicalCase) -> list[PatientFinding]:
        """
        VivaBench has a rich JSON clinicalcase. Extract:
          - Chief complaint from history.chief_complaint
          - Key symptoms from history.symptoms
          - Abnormal labs from investigations.blood / investigations.urine
          - Vitals from physical.vitals
          - Imaging from imaging reports
          - PMH from past_medical_history
        """
        cc = case.structured
        findings: list[PatientFinding] = []
        src = "vivabench"

        # 1. Chief complaint
        history = cc.get("history", {})
        cc_text = history.get("chief_complaint", "")
        if cc_text:
            findings.append(PatientFinding(
                finding_type="symptom",
                value=f"Chief complaint: {cc_text}",
                source=f"{src}/chief_complaint",
            ))

        # 2. Key symptoms (up to 3 most informative)
        symptoms = history.get("symptoms", {})
        count = 0
        for sym_name, sym_data in symptoms.items():
            if count >= 3:
                break
            if isinstance(sym_data, dict) and sym_data.get("present"):
                desc = sym_data.get("description", sym_data.get("name", sym_name))
                findings.append(PatientFinding(
                    finding_type="symptom",
                    value=str(desc)[:150],
                    source=f"{src}/symptoms",
                ))
                count += 1

        # 3. Abnormal labs from investigations
        investigations = cc.get("investigations", {})
        for fluid_type, lab_dict in investigations.items():
            if not isinstance(lab_dict, dict):
                continue
            for lab_name, lab_data in lab_dict.items():
                if not isinstance(lab_data, dict):
                    continue
                flag = lab_data.get("flag", "")
                if flag and flag.upper() in ("H", "L", "ABNORMAL", "HIGH", "LOW"):
                    name  = lab_data.get("name", lab_name)
                    value = lab_data.get("value", "")
                    units = lab_data.get("units", "")
                    findings.append(PatientFinding(
                        finding_type="lab",
                        value=f"{name}: {value}",
                        units=units,
                        source=f"{src}/labs",
                    ))

        # 4. Vitals (only flag extremes — thresholds defined in config.VITAL_THRESHOLDS)
        physical = cc.get("physical", {})
        vitals   = physical.get("vitals", {})
        for vital_name, (lo, hi) in VITAL_THRESHOLDS.items():
            val = vitals.get(vital_name)
            if val is not None:
                try:
                    numeric = float(val)
                    if numeric < lo or numeric > hi:
                        label = vital_name.replace("_", " ").title()
                        findings.append(PatientFinding(
                            finding_type="vital",
                            value=f"{label}: {val}",
                            source=f"{src}/vitals",
                        ))
                except (TypeError, ValueError):
                    pass

        # 5. Imaging (first 2 reports)
        imaging = cc.get("imaging", {})
        for img_name, img_data in list(imaging.items())[:2]:
            if isinstance(img_data, dict):
                report = img_data.get("report", "")
                if report:
                    modality = img_data.get("modality", "")
                    region   = img_data.get("region", "")
                    findings.append(PatientFinding(
                        finding_type="imaging",
                        value=f"{modality} {region}: {report[:150]}",
                        source=f"{src}/imaging",
                    ))

        # 6. PMH (first 2 active conditions)
        pmh = cc.get("past_medical_history", {})
        pmh_count = 0
        for cond_key, cond_data in pmh.items():
            if pmh_count >= 2:
                break
            if isinstance(cond_data, dict) and cond_data.get("present"):
                cond_name = cond_data.get("condition", cond_key)
                findings.append(PatientFinding(
                    finding_type="history",
                    value=f"Past medical history: {cond_name}",
                    source=f"{src}/pmh",
                ))
                pmh_count += 1

        return findings

    # ── Build eval cases ───────────────────────────────────────────────────

    def build_cases(
        self,
        cases: list[ClinicalCase],
        entropy_engine=None,
    ) -> list[dict]:
        """
        Convert ClinicalCase objects into evaluation-ready case dicts.

        Each output dict contains:
            case_id:      str
            description:  str
            source:       str ('cupcase' or 'vivabench')
            specialty:    str
            ground_truth: str  (correct diagnosis — evaluation target)
            distractors:  list[str]  (plausible wrong answers, CUPCase only)
            findings:     list[dict]  (serialisable PatientFindings)
            seed_nodes:   list[Node]  (ready for BeliefGraph)

        Args:
            cases:         ClinicalCase objects from load_cupcase/load_vivabench.
            entropy_engine: Optional EntropyEngine for real entropy on seeds.

        Returns:
            List of eval case dicts for CaseEvaluator.evaluate_all().
        """
        eval_cases = []
        for case in cases:
            try:
                findings   = self._extract_findings(case)
                seed_nodes = findings_to_seed_nodes(
                    findings,
                    entropy_engine=entropy_engine,
                )
                eval_cases.append({
                    "case_id":      case.case_id,
                    "description":  case.description,
                    "source":       case.source,
                    "specialty":    case.specialty,
                    "ground_truth": case.ground_truth,
                    "distractors":  case.distractors,
                    "findings": [
                        {
                            "finding_type": f.finding_type,
                            "value":        f.value,
                            "units":        f.units,
                            "source":       f.source,
                        }
                        for f in findings
                    ],
                    "seed_nodes": seed_nodes,
                })
                logger.debug(
                    f"[ClinicalCaseAdapter] {case.case_id}: "
                    f"{len(findings)} findings → {len(seed_nodes)} seed nodes"
                )
            except Exception as e:
                logger.warning(f"[ClinicalCaseAdapter] Skipping {case.case_id}: {e}")

        logger.info(
            f"[ClinicalCaseAdapter] Built {len(eval_cases)}/{len(cases)} eval cases."
        )
        return eval_cases
