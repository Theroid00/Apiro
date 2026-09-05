from apiro.eval.adversarial import score_meddistract, score_medeinst


def match(a, b):
    return bool(a and b and a.lower() == b.lower())


def test_medeinst_counts_retaining_control_diagnosis_as_bias_trap():
    records = [
        {"case_id": "1", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "1", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["A"]}},
        {"case_id": "2", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "2", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["B"]}},
    ]
    result = score_medeinst(records, match, arms=("apiro",))["apiro"]
    assert result["bias_trap_rate"] == 0.5
    assert result["pair_resilience"] == 0.5


def test_meddistract_scores_clean_to_distracted_retention():
    records = [
        {"case_id": "1", "condition": "clean", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "1", "condition": "distracted", "ground_truth": "A", "predictions": {"apiro": ["B"]}},
    ]
    result = score_meddistract(records, match, arms=("apiro",))["apiro"]
    assert result["retention"] == 0.0
    assert result["top1_flip_rate"] == 1.0
