from apiro.context import select_clinical_context


def test_short_context_is_unchanged():
    selected = select_clinical_context("Chest pain. Troponin is elevated.", "myocardial infarction")
    assert selected.text == "Chest pain. Troponin is elevated."
    assert selected.truncated is False


def test_late_hypothesis_evidence_survives_long_context_selection():
    filler = "Routine review of systems is unchanged. " * 500
    evidence = "Serum metanephrines are markedly elevated, supporting pheochromocytoma."
    narrative = "Episodic headache and palpitations. " + filler + evidence
    selected = select_clinical_context(narrative, "pheochromocytoma", max_characters=1200)
    assert evidence in selected.text
    assert selected.truncated is True
    assert selected.spans
