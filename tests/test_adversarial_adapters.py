from scripts.run_meddistractqa_eval import clean_question, diagnosis_pairs
from scripts.run_medeinst_eval import select_pairs


def test_medeinst_selects_complete_pairs_only():
    rows = [
        {"case_id": "a", "case_type": "control", "narrative": "a", "ground_truth": "A"},
        {"case_id": "a", "case_type": "trap", "narrative": "b", "ground_truth": "B"},
        {"case_id": "incomplete", "case_type": "control", "narrative": "c", "ground_truth": "C"},
    ]
    selected = select_pairs(rows, n_pairs=10, seed=7)
    assert [(r["case_id"], r["case_type"]) for r in selected] == [("a", "control"), ("a", "trap")]


def test_meddistract_filters_to_diagnosis_and_builds_clean_pair():
    distraction = "The patient's aunt takes metformin."
    rows = [
        {"id": "d1", "question": f"A patient has episodic headaches. {distraction}",
         "question_choices": {"A": "Pheochromocytoma", "B": "Migraine"},
         "correct_answer": "A", "distracting_sentence": distraction,
         "medical_competency": "Patient Care: Diagnosis"},
        {"id": "m1", "question": "Which treatment?", "question_choices": {"A": "Drug"},
         "correct_answer": "A", "distracting_sentence": "",
         "medical_competency": "Patient Care: Management"},
    ]
    pairs = diagnosis_pairs(rows, n=10, seed=7)
    assert len(pairs) == 2
    assert distraction not in pairs[0]["narrative"]
    assert distraction in pairs[1]["narrative"]
    assert pairs[0]["ground_truth"] == "Pheochromocytoma"


def test_clean_question_removes_only_supplied_distraction():
    assert clean_question("Clinical fact. Bird story.", "Bird story.") == "Clinical fact."
