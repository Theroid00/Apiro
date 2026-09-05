from apiro.eval.incremental import score_mint


def match(a, b):
    return bool(a and b and a.lower() == b.lower())


def test_mint_scores_commitment_revision_and_lure():
    turns = [
        {"predictions": {"apiro": []}, "abstained": {"apiro": True}},
        {"predictions": {"apiro": ["Migraine"]}, "abstained": {"apiro": False}, "is_lure": True},
        {"predictions": {"apiro": ["Pheochromocytoma"]}, "abstained": {"apiro": False}},
    ]
    result = score_mint(
        [{"case_id": "1", "ground_truth": "Pheochromocytoma", "turns": turns}],
        match, arms=("apiro",),
    )["apiro"]
    assert result["mean_first_commit_turn"] == 2
    assert result["wrong_to_correct_revisions"] == 1
    assert result["final_accuracy"] == 1.0
