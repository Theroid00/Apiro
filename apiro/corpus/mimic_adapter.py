"""
corpus/mimic_adapter.py
========================
Converts MIMIC-III demo data into PatientFinding lists and belief graph seed nodes.

MIMIC-III Demo dataset (100 patients, freely available without PhysioNet credentials):
  https://physionet.org/files/mimiciii-demo/1.4/

Important note on the demo dataset:
  NOTEEVENTS.csv in the demo contains only the header row (no actual clinical notes).
  This adapter therefore works from STRUCTURED data only:
    - ADMISSIONS.csv       → admission diagnosis, admission type
    - DIAGNOSES_ICD.csv    → ICD-9 codes (primary + secondary diagnoses)
    - D_ICD_DIAGNOSES.csv  → ICD-9 code → human-readable description
    - LABEVENTS.csv        → lab result values and flags
    - D_LABITEMS.csv       → ITEMID → lab test name
    - ICUSTAYS.csv         → ICU admission info
    - CHARTEVENTS.csv      → vital signs (optional, large file)

When full MIMIC-III access is available (CITI training + DUA), swap in:
    mimic_adapter.load_from_note(note_text) to parse the TEXT column from NOTEEVENTS.

Data flow:
    download_demo() → CSV files in data/mimic_demo/
    load_case(hadm_id) → list[PatientFinding]
    findings_to_seed_nodes(findings) → list[Node]

Usage:
    adapter = MimicAdapter(data_dir="data/mimic_demo")
    adapter.download_demo()
    cases = adapter.build_cases(n=10)
    # cases is a list of dicts with 'seed_nodes', 'ground_truth', 'case_id', etc.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from apiro.graph.node import Node

logger = logging.getLogger(__name__)

# ── MIMIC-III Demo download URLs ──────────────────────────────────────────────

_DEMO_BASE = "https://physionet.org/files/mimiciii-demo/1.4"

DEMO_FILES = {
    "ADMISSIONS":     f"{_DEMO_BASE}/ADMISSIONS.csv",
    "DIAGNOSES_ICD":  f"{_DEMO_BASE}/DIAGNOSES_ICD.csv",
    "D_ICD_DIAGNOSES":f"{_DEMO_BASE}/D_ICD_DIAGNOSES.csv",
    "LABEVENTS":      f"{_DEMO_BASE}/LABEVENTS.csv",
    "D_LABITEMS":     f"{_DEMO_BASE}/D_LABITEMS.csv",
    "ICUSTAYS":       f"{_DEMO_BASE}/ICUSTAYS.csv",
    "PATIENTS":       f"{_DEMO_BASE}/PATIENTS.csv",
}

# Lab items considered abnormal signals (ITEMID → description)
# From the standard MIMIC-III D_LABITEMS mapping
_KEY_LAB_ITEMIDS = {
    "50912": "Creatinine",
    "50971": "Potassium",
    "50983": "Sodium",
    "50902": "Chloride",
    "50882": "Bicarbonate",
    "51006": "Urea Nitrogen (BUN)",
    "51300": "WBC Count",
    "51221": "Hematocrit",
    "51222": "Hemoglobin",
    "51265": "Platelet Count",
    "50893": "Calcium",
    "50960": "Magnesium",
    "50970": "Phosphate",
    "51484": "Ketones (Urine)",
    "50813": "Lactate",
    "50885": "Bilirubin Total",
    "50910": "Creatine Kinase",
    "50954": "LDH",
    "50820": "pH",
    "50821": "pO2",
    "50818": "pCO2",
    # Troponin (label search needed — not a fixed ITEMID)
}


# ── PatientFinding ─────────────────────────────────────────────────────────────

@dataclass
class PatientFinding:
    """
    A single structured clinical finding from a patient record.

    finding_type: symptom | lab | vital | history | diagnosis | imaging
    value:        Text value (e.g. "4.2 ng/mL", "hypertension", "STEMI")
    units:        Unit string if applicable (e.g. "ng/mL", "mmHg")
    confidence:   1.0 for structured data, 0.7-0.9 for inferred/extracted
    source:       Where this came from ("admissions", "lab", "icd9", etc.)
    """
    finding_type: str
    value:        str
    units:        str        = ""
    confidence:   float      = 1.0
    source:       str        = ""
    metadata:     dict       = field(default_factory=dict)

    def to_claim(self) -> str:
        """Convert this finding into a belief-graph claim string."""
        if self.units:
            return f"{self.value} ({self.units}) — {self.finding_type}"
        return f"{self.value} — {self.finding_type}"


# ── findings_to_seed_nodes ─────────────────────────────────────────────────────

def findings_to_seed_nodes(
    findings: list[PatientFinding],
    entropy_engine=None,
    default_entropy: float = 0.693,  # ln(2) — maximum binary uncertainty
) -> list[Node]:
    """
    Convert a list of PatientFindings into belief graph seed Nodes.

    Args:
        findings:         Patient findings from MimicAdapter.load_case().
        entropy_engine:   EntropyEngine instance. If provided, computes real
                          epistemic entropy for each finding claim.
                          If None, seeds all nodes with default_entropy (ln(2)).
        default_entropy:  Fallback entropy when engine is None.

    Returns:
        List of Node objects ready to be added to BeliefGraph as seeds.
    """
    from apiro.graph.expander import classify_domain

    nodes = []
    for i, finding in enumerate(findings):
        claim  = finding.to_claim()
        domain = classify_domain(claim)

        # Compute entropy if engine provided, else use default (maximum uncertainty)
        if entropy_engine is not None:
            try:
                entropy = entropy_engine.epistemic_certainty_entropy(claim, context_chunks=[])
            except Exception:
                entropy = default_entropy
        else:
            entropy = default_entropy

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

    logger.info(f"[MimicAdapter] Created {len(nodes)} seed nodes from {len(findings)} findings.")
    return nodes


# ── MimicAdapter ───────────────────────────────────────────────────────────────

class MimicAdapter:
    """
    Downloads and parses the MIMIC-III demo dataset (100 patients, no credentials).

    Args:
        data_dir: Local directory to store downloaded CSVs.
    """

    def __init__(self, data_dir: str | Path = "data/mimic_demo"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Loaded tables (lazy)
        self._admissions:    Optional[list[dict]] = None
        self._diagnoses:     Optional[list[dict]] = None
        self._icd_desc:      Optional[dict]       = None   # icd9_code → description
        self._lab_items:     Optional[dict]       = None   # itemid → label
        self._labevents:     Optional[list[dict]] = None
        self._icustays:      Optional[list[dict]] = None

    # ── Download ───────────────────────────────────────────────────────────

    def download_demo(self, force: bool = False) -> None:
        """
        Download all MIMIC-III demo CSV files to data_dir.
        Skips files already present unless force=True.
        """
        logger.info(f"[MimicAdapter] Downloading MIMIC-III demo to {self.data_dir}/")
        for name, url in DEMO_FILES.items():
            dest = self.data_dir / f"{name}.csv"
            if dest.exists() and not force:
                logger.info(f"  {name}.csv — already cached.")
                continue
            logger.info(f"  Downloading {name}.csv from PhysioNet...")
            try:
                resp = requests.get(url, timeout=60, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                size_kb = dest.stat().st_size // 1024
                logger.info(f"  {name}.csv — {size_kb} KB saved.")
            except Exception as e:
                logger.error(f"  {name}.csv — FAILED: {e}")

    # ── CSV loaders ────────────────────────────────────────────────────────

    def _load_csv(self, name: str) -> list[dict]:
        path = self.data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run adapter.download_demo() first."
            )
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _admissions_table(self) -> list[dict]:
        if self._admissions is None:
            self._admissions = self._load_csv("ADMISSIONS")
        return self._admissions

    def _diagnoses_table(self) -> list[dict]:
        if self._diagnoses is None:
            self._diagnoses = self._load_csv("DIAGNOSES_ICD")
        return self._diagnoses

    def _icd_descriptions(self) -> dict:
        """Returns dict: icd9_code → short_title."""
        if self._icd_desc is None:
            rows = self._load_csv("D_ICD_DIAGNOSES")
            self._icd_desc = {
                r["ICD9_CODE"].strip(): r.get("SHORT_TITLE", r.get("LONG_TITLE", "")).strip()
                for r in rows
            }
        return self._icd_desc

    def _lab_item_names(self) -> dict:
        """Returns dict: itemid → label."""
        if self._lab_items is None:
            rows = self._load_csv("D_LABITEMS")
            self._lab_items = {
                r["ITEMID"].strip(): r.get("LABEL", "").strip()
                for r in rows
            }
        return self._lab_items

    def _labevents_table(self) -> list[dict]:
        if self._labevents is None:
            self._labevents = self._load_csv("LABEVENTS")
        return self._labevents

    def _icustays_table(self) -> list[dict]:
        if self._icustays is None:
            try:
                self._icustays = self._load_csv("ICUSTAYS")
            except FileNotFoundError:
                self._icustays = []
        return self._icustays

    # ── Case loading ───────────────────────────────────────────────────────

    def load_case(self, hadm_id: str) -> list[PatientFinding]:
        """
        Load all structured findings for a single hospital admission.

        Returns a list of PatientFinding objects covering:
          - Admission diagnosis (from ADMISSIONS.DIAGNOSIS)
          - Primary ICD-9 diagnosis (SEQ_NUM=1 from DIAGNOSES_ICD)
          - Up to 5 abnormal lab results (FLAG='abnormal' from LABEVENTS)
          - ICU stay indicator if present
        """
        findings: list[PatientFinding] = []

        # 1. Admission diagnosis
        for row in self._admissions_table():
            if row.get("HADM_ID", "").strip() == str(hadm_id):
                diag = row.get("DIAGNOSIS", "").strip()
                if diag:
                    findings.append(PatientFinding(
                        finding_type="diagnosis",
                        value=diag,
                        confidence=1.0,
                        source="admissions",
                        metadata={"hadm_id": hadm_id},
                    ))
                break

        # 2. ICD-9 diagnoses (primary first, then up to 4 secondary)
        icd_desc = self._icd_descriptions()
        icd_rows = sorted(
            [r for r in self._diagnoses_table() if r.get("HADM_ID", "").strip() == str(hadm_id)],
            key=lambda r: int(r.get("SEQ_NUM", 99)),
        )
        for r in icd_rows[:5]:
            code = r.get("ICD9_CODE", "").strip()
            desc = icd_desc.get(code, f"ICD9:{code}")
            seq  = int(r.get("SEQ_NUM", 99))
            findings.append(PatientFinding(
                finding_type="history" if seq > 1 else "diagnosis",
                value=desc,
                confidence=1.0,
                source="icd9",
                metadata={"icd9_code": code, "seq_num": seq},
            ))

        # 3. Abnormal lab results (up to 5 most recent)
        lab_names = self._lab_item_names()
        lab_rows  = [
            r for r in self._labevents_table()
            if r.get("HADM_ID", "").strip() == str(hadm_id)
               and r.get("FLAG", "").strip().lower() == "abnormal"
        ][:5]
        for r in lab_rows:
            item_id = r.get("ITEMID", "").strip()
            label   = lab_names.get(item_id, f"Lab_{item_id}")
            value   = r.get("VALUE", "").strip()
            units   = r.get("VALUEUOM", "").strip()
            if label and value:
                findings.append(PatientFinding(
                    finding_type="lab",
                    value=f"{label}: {value}",
                    units=units,
                    confidence=1.0,
                    source="labevents",
                    metadata={"itemid": item_id},
                ))

        # 4. ICU stay (if present)
        for r in self._icustays_table():
            if r.get("HADM_ID", "").strip() == str(hadm_id):
                unit = r.get("FIRST_CAREUNIT", "ICU").strip()
                los  = r.get("LOS", "").strip()
                findings.append(PatientFinding(
                    finding_type="history",
                    value=f"ICU admission ({unit})",
                    units=f"LOS: {los} days" if los else "",
                    confidence=1.0,
                    source="icustays",
                ))
                break

        logger.info(
            f"[MimicAdapter] HADM_ID {hadm_id}: {len(findings)} findings loaded "
            f"({sum(1 for f in findings if f.finding_type=='lab')} labs, "
            f"{sum(1 for f in findings if f.finding_type=='diagnosis')} diagnoses)."
        )
        return findings

    # ── Ground truth extraction ────────────────────────────────────────────

    def get_ground_truth(self, hadm_id: str) -> str:
        """
        Return the primary ICD-9 diagnosis description for an admission.
        This is used as the ground_truth string in evaluation.
        """
        icd_desc = self._icd_descriptions()
        for r in sorted(
            self._diagnoses_table(),
            key=lambda r: int(r.get("SEQ_NUM", 99)),
        ):
            if r.get("HADM_ID", "").strip() == str(hadm_id) and r.get("SEQ_NUM", "") == "1":
                code = r.get("ICD9_CODE", "").strip()
                return icd_desc.get(code, f"ICD9:{code}")
        return "unknown"

    # ── Build evaluation cases ────────────────────────────────────────────

    def build_cases(
        self,
        n: int = 10,
        entropy_engine=None,
        output_dir: Optional[str | Path] = None,
    ) -> list[dict]:
        """
        Build n evaluation cases from the demo dataset.

        Each case dict contains:
            case_id:      str (e.g. "mimic_200001")
            description:  str (admission diagnosis)
            hadm_id:      str
            ground_truth: str (primary ICD-9 description)
            findings:     list of PatientFinding as dicts
            seed_nodes:   list[Node]

        Args:
            n:             Number of cases to build (max 100 in demo).
            entropy_engine: EntropyEngine for real entropy on seeds (optional).
            output_dir:    If set, save each case to a JSON file here.

        Returns:
            List of case dicts ready for CaseEvaluator.evaluate_all().
        """
        # Get unique HADMs from admissions
        admissions = self._admissions_table()
        hadm_ids   = list({r["HADM_ID"].strip() for r in admissions if r.get("HADM_ID")})
        hadm_ids   = sorted(hadm_ids)[:n]

        cases = []
        for hadm_id in hadm_ids:
            try:
                findings     = self.load_case(hadm_id)
                ground_truth = self.get_ground_truth(hadm_id)

                # Get admission description
                desc = "unknown"
                for r in admissions:
                    if r.get("HADM_ID", "").strip() == hadm_id:
                        desc = r.get("DIAGNOSIS", "unknown").strip()
                        break

                seed_nodes = findings_to_seed_nodes(
                    findings,
                    entropy_engine=entropy_engine,
                )

                case = {
                    "case_id":      f"mimic_{hadm_id}",
                    "description":  desc,
                    "hadm_id":      hadm_id,
                    "ground_truth": ground_truth,
                    "findings":     [
                        {
                            "finding_type": f.finding_type,
                            "value":        f.value,
                            "units":        f.units,
                            "source":       f.source,
                        }
                        for f in findings
                    ],
                    "seed_nodes": seed_nodes,
                }
                cases.append(case)

                if output_dir:
                    out = Path(output_dir) / f"case_{hadm_id}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    # Serialize without seed_nodes (not JSON-serializable)
                    serializable = {k: v for k, v in case.items() if k != "seed_nodes"}
                    out.write_text(json.dumps(serializable, indent=2))

            except Exception as e:
                logger.warning(f"[MimicAdapter] Skipping HADM_ID {hadm_id}: {e}")

        logger.info(f"[MimicAdapter] Built {len(cases)} evaluation cases.")
        return cases
