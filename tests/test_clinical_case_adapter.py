"""Clinical case adapters must never discard the source narrative."""

from apiro.corpus.clinical_case_adapter import ClinicalCase, ClinicalCaseAdapter


def test_cupcase_eval_case_preserves_late_evidence():
    late_evidence = "Decisive finding: serum metanephrines are markedly elevated."
    narrative = "Chest discomfort. " + ("Routine history is unremarkable. " * 30) + late_evidence
    case = ClinicalCase(
        case_id="late-evidence",
        source="cupcase",
        description="late evidence fixture",
        narrative=narrative,
        ground_truth="pheochromocytoma",
    )

    built = ClinicalCaseAdapter().build_cases([case])[0]

    assert built["narrative"] == narrative
    assert late_evidence in built["narrative"]
    assert len(built["narrative"]) > 400
