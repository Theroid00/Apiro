"""
tests/test_mimic_adapter.py
============================
Unit tests for MimicAdapter and PatientFinding → Node conversion.

These tests mock the CSV data to avoid requiring an actual MIMIC download,
so they run fully offline and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apiro.corpus.mimic_adapter import (
    PatientFinding,
    findings_to_seed_nodes,
    MimicAdapter,
)
from apiro.graph.node import Node


# ── PatientFinding tests ──────────────────────────────────────────────────────

class TestPatientFinding:
    def test_to_claim_with_units(self):
        f = PatientFinding(finding_type="lab", value="Troponin I: 4.2", units="ng/mL")
        claim = f.to_claim()
        assert "Troponin I: 4.2" in claim
        assert "ng/mL" in claim
        assert "lab" in claim

    def test_to_claim_without_units(self):
        f = PatientFinding(finding_type="diagnosis", value="STEMI")
        claim = f.to_claim()
        assert "STEMI" in claim
        assert "diagnosis" in claim

    def test_default_confidence(self):
        f = PatientFinding(finding_type="lab", value="WBC elevated")
        assert f.confidence == 1.0

    def test_metadata_default_empty(self):
        f = PatientFinding(finding_type="history", value="hypertension")
        assert isinstance(f.metadata, dict)


# ── findings_to_seed_nodes tests ──────────────────────────────────────────────

class TestFindingsToSeedNodes:
    def test_produces_correct_node_count(self):
        findings = [
            PatientFinding("diagnosis", "STEMI"),
            PatientFinding("lab", "Troponin I elevated", units="ng/mL"),
            PatientFinding("history", "hypertension"),
        ]
        nodes = findings_to_seed_nodes(findings, entropy_engine=None)
        assert len(nodes) == 3

    def test_nodes_are_node_instances(self):
        findings = [PatientFinding("diagnosis", "STEMI")]
        nodes = findings_to_seed_nodes(findings, entropy_engine=None)
        assert all(isinstance(n, Node) for n in nodes)

    def test_seed_node_depth_is_zero(self):
        findings = [PatientFinding("diagnosis", "STEMI")]
        nodes = findings_to_seed_nodes(findings, entropy_engine=None)
        assert all(n.depth == 0 for n in nodes)

    def test_seed_node_not_resolved(self):
        findings = [PatientFinding("diagnosis", "STEMI")]
        nodes = findings_to_seed_nodes(findings, entropy_engine=None)
        assert all(not n.resolved for n in nodes)

    def test_default_entropy_applied(self):
        findings = [PatientFinding("diagnosis", "STEMI")]
        nodes = findings_to_seed_nodes(findings, entropy_engine=None, default_entropy=0.693)
        assert nodes[0].entropy_score == pytest.approx(0.693)

    def test_domain_classified(self):
        findings = [
            PatientFinding("lab", "Troponin I elevated", units="ng/mL"),
            PatientFinding("diagnosis", "BAG3 gene variant — pathogenic"),
        ]
        nodes = findings_to_seed_nodes(findings, entropy_engine=None)
        domains = {n.domain for n in nodes}
        # At minimum one should be lab or genetics
        assert len(domains) >= 1
        assert all(d in ["pathophysiology", "pharmacology", "genetics",
                          "imaging", "lab", "treatment", "comorbidity"]
                   for d in domains)

    def test_empty_findings_returns_empty_list(self):
        nodes = findings_to_seed_nodes([], entropy_engine=None)
        assert nodes == []


# ── MimicAdapter unit tests (mocked CSV) ─────────────────────────────────────

MOCK_ADMISSIONS = """ROW_ID,SUBJECT_ID,HADM_ID,ADMITTIME,DISCHTIME,DEATHTIME,ADMISSION_TYPE,ADMISSION_LOCATION,DISCHARGE_LOCATION,INSURANCE,LANGUAGE,RELIGION,MARITAL_STATUS,ETHNICITY,EDREGTIME,EDOUTTIME,DIAGNOSIS,HOSPITAL_EXPIRE_FLAG,HAS_CHARTEVENTS_DATA
1,10006,142345,2164-10-23 21:09:00,2164-11-01 17:15:00,,EMERGENCY,EMERGENCY ROOM ADMIT,HOME HEALTH CARE,Medicare,,UNOBTAINABLE,WIDOWED,BLACK/AFRICAN AMERICAN,2164-10-23 20:33:00,2164-10-23 21:19:00,SEPSIS,0,1
2,10011,105331,2126-08-14 22:32:00,2126-08-28 18:00:00,2126-08-28 18:00:00,EMERGENCY,EMERGENCY ROOM ADMIT,DEAD/EXPIRED,Private,,PROTESTANT QUAKER,SINGLE,UNKNOWN/NOT SPECIFIED,2126-08-14 20:03:00,2126-08-14 22:50:00,HEPATITIS B,1,1
"""

MOCK_DIAGNOSES_ICD = """ROW_ID,SUBJECT_ID,HADM_ID,SEQ_NUM,ICD9_CODE
1,10006,142345,1,99591
2,10006,142345,2,78820
3,10011,105331,1,07030
"""

MOCK_D_ICD = """ROW_ID,ICD9_CODE,SHORT_TITLE,LONG_TITLE
1,99591,Sepsis,Sepsis
2,78820,Pyrexia unknown origin,Pyrexia of unknown origin
3,07030,Hpt B wo hpt delta-actv,Viral hepatitis B without mention of hepatic coma, acute or unspecified, without mention of hepatitis delta, with hepatic necrosis
"""

MOCK_D_LABITEMS = """ROW_ID,ITEMID,LABEL,FLUID,CATEGORY,LOINC_CODE
1,50912,Creatinine,Blood,Chemistry,2160-0
2,51222,Hemoglobin,Blood,Hematology,718-7
"""

MOCK_LABEVENTS = """ROW_ID,SUBJECT_ID,HADM_ID,ITEMID,CHARTTIME,VALUE,VALUENUM,VALUEUOM,FLAG
1,10006,142345,50912,2164-10-24 14:34:00,2.1,2.1,mg/dL,abnormal
2,10006,142345,51222,2164-10-24 14:34:00,7.2,7.2,g/dL,abnormal
3,10011,105331,50912,2126-08-15 08:00:00,1.0,1.0,mg/dL,
"""

MOCK_ICUSTAYS = """ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,DBSOURCE,FIRST_CAREUNIT,LAST_CAREUNIT,FIRST_WARDID,LAST_WARDID,INTIME,OUTTIME,LOS
1,10006,142345,211552,carevue,MICU,MICU,52,52,2164-10-23 21:12:04,2164-11-01 12:05:11,8.63
"""


def make_mock_adapter(tmp_path: Path) -> MimicAdapter:
    """Write mock CSVs to tmp_path and return an adapter pointing there."""
    (tmp_path / "ADMISSIONS.csv").write_text(MOCK_ADMISSIONS)
    (tmp_path / "DIAGNOSES_ICD.csv").write_text(MOCK_DIAGNOSES_ICD)
    (tmp_path / "D_ICD_DIAGNOSES.csv").write_text(MOCK_D_ICD)
    (tmp_path / "LABEVENTS.csv").write_text(MOCK_LABEVENTS)
    (tmp_path / "D_LABITEMS.csv").write_text(MOCK_D_LABITEMS)
    (tmp_path / "ICUSTAYS.csv").write_text(MOCK_ICUSTAYS)
    return MimicAdapter(data_dir=tmp_path)


class TestMimicAdapterLoadCase:
    def test_loads_admission_diagnosis(self, tmp_path):
        adapter  = make_mock_adapter(tmp_path)
        findings = adapter.load_case("142345")
        diag_findings = [f for f in findings if f.source == "admissions"]
        assert len(diag_findings) == 1
        assert "SEPSIS" in diag_findings[0].value.upper()

    def test_loads_icd_diagnoses(self, tmp_path):
        adapter  = make_mock_adapter(tmp_path)
        findings = adapter.load_case("142345")
        icd_findings = [f for f in findings if f.source == "icd9"]
        assert len(icd_findings) >= 1

    def test_loads_abnormal_labs_only(self, tmp_path):
        adapter  = make_mock_adapter(tmp_path)
        findings = adapter.load_case("142345")
        lab_findings = [f for f in findings if f.source == "labevents"]
        assert len(lab_findings) == 2  # 2 abnormal labs for hadm 142345
        # Ensure normal labs (hadm 105331) not included
        assert all("Creatinine" in f.value or "Hemoglobin" in f.value
                   for f in lab_findings)

    def test_loads_icu_stay(self, tmp_path):
        adapter  = make_mock_adapter(tmp_path)
        findings = adapter.load_case("142345")
        icu_findings = [f for f in findings if f.source == "icustays"]
        assert len(icu_findings) == 1
        assert "MICU" in icu_findings[0].value or "ICU" in icu_findings[0].value

    def test_no_icu_stay_for_other_hadm(self, tmp_path):
        adapter  = make_mock_adapter(tmp_path)
        findings = adapter.load_case("105331")
        icu_findings = [f for f in findings if f.source == "icustays"]
        assert len(icu_findings) == 0


class TestMimicAdapterGroundTruth:
    def test_returns_primary_icd_description(self, tmp_path):
        adapter = make_mock_adapter(tmp_path)
        gt = adapter.get_ground_truth("142345")
        assert "Sepsis" in gt

    def test_returns_unknown_for_missing_hadm(self, tmp_path):
        adapter = make_mock_adapter(tmp_path)
        gt = adapter.get_ground_truth("999999")
        assert gt == "unknown"


class TestBuildCases:
    def test_returns_list_of_dicts(self, tmp_path):
        adapter = make_mock_adapter(tmp_path)
        cases   = adapter.build_cases(n=2)
        assert isinstance(cases, list)
        assert all(isinstance(c, dict) for c in cases)

    def test_case_has_required_keys(self, tmp_path):
        adapter = make_mock_adapter(tmp_path)
        cases   = adapter.build_cases(n=1)
        required = {"case_id", "description", "hadm_id", "ground_truth",
                    "findings", "seed_nodes"}
        for case in cases:
            assert required.issubset(case.keys()), f"Missing keys in {case}"

    def test_seed_nodes_are_nodes(self, tmp_path):
        adapter = make_mock_adapter(tmp_path)
        cases   = adapter.build_cases(n=1)
        for case in cases:
            assert all(isinstance(n, Node) for n in case["seed_nodes"])
