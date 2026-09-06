from apiro.eval.adversarial import score_meddistract, score_medeinst


def match(a, b):
    return bool(a and b and a.lower() == b.lower())


def test_medeinst_counts_rank1_control_diagnosis_retention_as_bias_trap():
    records = [
        {"case_id": "1", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "1", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["A"]}},
        {"case_id": "2", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "2", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["B"]}},
    ]
    result = score_medeinst(records, match, arms=("apiro",))["apiro"]
    assert result["bias_trap_rate"] == 0.5
    assert result["pair_resilience"] == 0.5


def test_medeinst_does_not_confuse_general_trap_errors_with_bias_traps():
    records = [
        {"case_id": "1", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "1", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["C"]}},
        {"case_id": "2", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "2", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["A"]}},
    ]
    result = score_medeinst(records, match, arms=("apiro",))["apiro"]
    assert result["bias_trap_rate"] == 0.5
    assert result["conditional_trap_error_rate"] == 1.0


def test_medeinst_primary_metrics_are_rank1_and_top3_is_secondary():
    records = [
        {"case_id": "1", "case_type": "control", "ground_truth": "A", "predictions": {"apiro": ["X", "A"]}},
        {"case_id": "1", "case_type": "trap", "ground_truth": "B", "predictions": {"apiro": ["Y", "B"]}},
    ]
    result = score_medeinst(records, match, arms=("apiro",))["apiro"]
    assert result["n_control_correct"] == 0
    assert result["bias_trap_rate"] is None
    assert result["control_accuracy"] == 0.0
    assert result["top3_control_accuracy"] == 1.0
    assert result["top3_trap_accuracy"] == 1.0
    assert result["top3_pair_resilience"] == 1.0


def test_meddistract_scores_clean_to_distracted_retention():
    records = [
        {"case_id": "1", "condition": "clean", "ground_truth": "A", "predictions": {"apiro": ["A"]}},
        {"case_id": "1", "condition": "distracted", "ground_truth": "A", "predictions": {"apiro": ["B"]}},
    ]
    result = score_meddistract(records, match, arms=("apiro",))["apiro"]
    assert result["retention"] == 0.0
    assert result["top1_flip_rate"] == 1.0
