"""
tests/test_parsing.py
=====================
Unit tests for apiro/parsing.py.

Most inputs below are verbatim strings taken from the Apiro arm of the
committed C-NIAH run (``data/niah_eval_results.json``), where the old parser
spent 57% of the engine's answer slots on markdown scaffolding and left five
of twenty-five cases with no diagnosis at all. Each such test is a regression
guard on a real, measured failure — not a hypothetical.

Offline: no Ollama, no ChromaDB, no model download.
"""

import pytest

from apiro.parsing import (
    DIFFERENTIAL_SENTINEL,
    parse_claims,
    parse_differential,
    strip_scaffolding,
)


# --------------------------------------------------------------------------- #
# strip_scaffolding
# --------------------------------------------------------------------------- #
class TestStripScaffolding:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("**Pulmonary Embolism**", "Pulmonary Embolism"),
            ("*Pulmonary Embolism**", "Pulmonary Embolism"),
            ("__Sepsis__", "Sepsis"),
            ("`Sepsis`", "Sepsis"),
            ("1. Pulmonary Embolism", "Pulmonary Embolism"),
            ("2) Sepsis", "Sepsis"),
            ("- Sepsis", "Sepsis"),
            ("• Sepsis", "Sepsis"),
            ("### Sepsis", "Sepsis"),
            ("Diagnosis 1: Sepsis", "Sepsis"),
            ("**Primary Diagnosis:** Acute Coronary Syndrome", "Acute Coronary Syndrome"),
            ("*Primary Diagnosis:** Acute Coronary Syndrome", "Acute Coronary Syndrome"),
            ("DX: Pulmonary embolism", "Pulmonary embolism"),
            ("  Sepsis.  ", "Sepsis"),
        ],
    )
    def test_removes_scaffolding_keeps_the_answer(self, raw, expected):
        assert strip_scaffolding(raw) == expected

    def test_nested_markers_need_more_than_one_pass(self):
        # The old parser stripped exactly one leading character, turning
        # "**Diagnosis 1:**" into "*Diagnosis 1:**" and admitting that as an
        # answer. Repeated stripping is the point.
        assert strip_scaffolding("**Diagnosis 1:**") == ""
        assert strip_scaffolding("*Diagnosis 2:**") == ""

    def test_a_bare_label_reduces_to_nothing(self):
        assert strip_scaffolding("**Diagnosis 1:**") == ""
        assert strip_scaffolding("Secondary Diagnosis:") == ""

    def test_leaves_a_clean_label_alone(self):
        assert strip_scaffolding("Acute appendicitis") == "Acute appendicitis"


# --------------------------------------------------------------------------- #
# parse_differential — the regression cases
# --------------------------------------------------------------------------- #
class TestParseDifferentialRegressions:
    """Every input here is a real Apiro output from data/niah_eval_results.json."""

    def test_sentence_preamble_does_not_consume_a_slot(self):
        raw = (
            "Based on the provided clinical findings and reasoning trace, here are "
            "three possible diagnoses:\n"
            "**Type 1 Diabetes Mellitus**\n"
            "**Spontaneous Intracranial Hypotension Syndrome**"
        )
        assert parse_differential(raw, limit=3) == [
            "Type 1 Diabetes Mellitus",
            "Spontaneous Intracranial Hypotension Syndrome",
        ]

    def test_rank_labels_do_not_consume_slots(self):
        # Old behaviour: ['*Diagnosis 1:**', 'Acute Myeloid Leukemia (AML)',
        #                 '*Diagnosis 2:**'] — one real answer out of three.
        raw = "*Diagnosis 1:**\nAcute Myeloid Leukemia (AML)\n*Diagnosis 2:**"
        assert parse_differential(raw, limit=3) == ["Acute Myeloid Leukemia (AML)"]

    def test_label_with_inline_answer_keeps_the_answer(self):
        raw = (
            "*Primary Diagnosis:** Acute Coronary Syndrome\n"
            "*Secondary Diagnosis:** Type 2 Diabetes Mellitus\n"
            "*Tertiary Diagnosis:** Chronic Kidney Disease"
        )
        assert parse_differential(raw, limit=3) == [
            "Acute Coronary Syndrome",
            "Type 2 Diabetes Mellitus",
            "Chronic Kidney Disease",
        ]

    def test_glued_explanation_is_cut_back_to_the_label(self):
        raw = (
            "**Diabetic Nephropathy**: This is a complication of long-standing "
            "diabetes mellitus, which can lead to chronic glomerular damage and "
            "basement membrane thickening."
        )
        assert parse_differential(raw, limit=3) == ["Diabetic Nephropathy"]

    def test_an_all_commentary_answer_yields_nothing(self):
        # Better an explicit empty differential (which triggers the retry) than
        # three slots of prose scored as diagnoses.
        raw = (
            "Based on the provided information, I will attempt to extract the most "
            "likely diagnoses for the patient. Please note that this is a complex "
            "task.\n"
            "*Confirmed Objective Findings:**\n"
            "The patient presents with the clinical finding of mental status examination."
        )
        assert parse_differential(raw, limit=3) == []

    def test_already_clean_output_is_untouched_except_for_emphasis(self):
        raw = "**Deep Vein Thrombosis**\n**Acute Promyelocytic Leukemia**\n**Pulmonary Embolism**"
        assert parse_differential(raw, limit=3) == [
            "Deep Vein Thrombosis",
            "Acute Promyelocytic Leukemia",
            "Pulmonary Embolism",
        ]


class TestParseDifferentialBehaviour:
    def test_order_is_preserved_because_order_is_the_ranking(self):
        raw = "Pulmonary embolism\nPneumonia\nAsthma"
        assert parse_differential(raw, limit=3) == [
            "Pulmonary embolism", "Pneumonia", "Asthma",
        ]

    def test_limit_is_respected(self):
        raw = "One disease\nTwo disease\nThree disease\nFour disease"
        assert len(parse_differential(raw, limit=2)) == 2

    def test_duplicates_do_not_consume_two_slots(self):
        raw = "Sepsis\nSEPSIS\nPneumonia"
        assert parse_differential(raw, limit=3) == ["Sepsis", "Pneumonia"]

    def test_empty_and_whitespace_input(self):
        assert parse_differential("") == []
        assert parse_differential("   \n\n  ") == []
        assert parse_differential(None) == []

    def test_code_fences_are_dropped(self):
        assert parse_differential("```\nSepsis\n```", limit=3) == ["Sepsis"]

    def test_bare_numbers_and_rules_are_dropped(self):
        assert parse_differential("1\n---\n###\nSepsis", limit=3) == ["Sepsis"]

    def test_prose_openers_are_rejected(self):
        raw = (
            "The patient likely has sepsis.\n"
            "This is a serious condition.\n"
            "Septic shock"
        )
        assert parse_differential(raw, limit=3) == ["Septic shock"]

    def test_header_only_lines_are_rejected(self):
        raw = "Diagnoses:\nOutput:\nTop 3:\nSepsis"
        assert parse_differential(raw, limit=3) == ["Sepsis"]

    def test_long_labels_without_a_split_point_are_dropped(self):
        raw = (
            "This extremely long line of narrative text goes on and on well past "
            "any plausible length for a disease name at all\n"
            "Sepsis"
        )
        assert parse_differential(raw, limit=3) == ["Sepsis"]

    def test_a_name_whose_identity_lives_after_due_to_is_preserved(self):
        # "Hemolytic anemia due to G6PD deficiency" must not be cut to
        # "Hemolytic anemia": the half after "due to" is the diagnosis the
        # benchmark grades against. Clause splitting only applies to lines
        # already too long to be a label.
        raw = "Hemolytic anemia due to G6PD deficiency"
        assert parse_differential(raw, limit=3) == ["Hemolytic anemia due to G6PD deficiency"]

    def test_a_legitimately_long_name_survives(self):
        raw = "Poorly differentiated mucinous adenocarcinoma of the colon"
        assert parse_differential(raw, limit=3) == [
            "Poorly differentiated mucinous adenocarcinoma of the colon"
        ]


class TestSentinelMode:
    def test_sentinel_lines_win_and_everything_else_is_ignored(self):
        raw = (
            "Sure! Here is my reasoning about this difficult case.\n"
            f"{DIFFERENTIAL_SENTINEL} Pulmonary embolism\n"
            "I considered pneumonia too but ruled it out.\n"
            f"{DIFFERENTIAL_SENTINEL} Aortic dissection\n"
            f"{DIFFERENTIAL_SENTINEL} Pericarditis"
        )
        assert parse_differential(raw, limit=3) == [
            "Pulmonary embolism", "Aortic dissection", "Pericarditis",
        ]

    def test_sentinel_survives_bold_and_numbering(self):
        raw = "**DX:** Sepsis\nDX 2: Pneumonia\nDX. Meningitis"
        assert parse_differential(raw, limit=3) == ["Sepsis", "Pneumonia", "Meningitis"]

    def test_partial_sentinel_use_still_takes_only_sentinel_lines(self):
        # A model that tags one line and rambles on the rest gets the tagged
        # line taken at face value; guessing at the rest is how junk gets in.
        raw = "Some preamble\nDX: Sepsis\nPneumonia maybe"
        assert parse_differential(raw, limit=3) == ["Sepsis"]

    def test_no_sentinel_falls_back_to_general_parsing(self):
        raw = "**Sepsis**\n**Pneumonia**"
        assert parse_differential(raw, limit=3) == ["Sepsis", "Pneumonia"]


# --------------------------------------------------------------------------- #
# parse_claims
# --------------------------------------------------------------------------- #
class TestParseClaims:
    def test_keeps_full_sentences(self):
        raw = (
            "Hypotheses:\n"
            "1. Right coronary artery occlusion is the most likely cause of inferior STEMI.\n"
            "2. Immediate primary PCI is indicated within 90 minutes of symptom onset."
        )
        # Terminators are kept: a claim is a sentence, and tests/
        # test_traversal_regressions.py reads the node text back.
        assert parse_claims(raw, limit=3) == [
            "Right coronary artery occlusion is the most likely cause of inferior STEMI.",
            "Immediate primary PCI is indicated within 90 minutes of symptom onset.",
        ]

    def test_does_not_truncate_at_a_colon(self):
        # Unlike parse_differential: a claim is allowed to be a long sentence,
        # and cutting it at the first colon would destroy the hypothesis.
        raw = "Elevated troponin indicates myocardial injury: this supports an ACS diagnosis."
        claims = parse_claims(raw, limit=3)
        assert len(claims) == 1
        assert "supports an ACS diagnosis" in claims[0]

    def test_strips_the_hypotheses_header(self):
        assert parse_claims("Hypotheses:\nSepsis is likely", limit=3) == ["Sepsis is likely"]

    def test_a_claim_keeps_its_full_stop(self):
        assert parse_claims("Only one real hypothesis here.", limit=3) == [
            "Only one real hypothesis here."
        ]

    def test_a_label_does_not_keep_its_full_stop(self):
        assert parse_differential("Sepsis.", limit=3) == ["Sepsis"]

    def test_rejects_meta_commentary(self):
        raw = "I will now generate three hypotheses.\nSepsis is likely given the fever"
        assert parse_claims(raw, limit=3) == ["Sepsis is likely given the fever"]

    def test_empty_input(self):
        assert parse_claims("") == []

    def test_limit_and_dedupe(self):
        raw = "Claim about sepsis\nClaim about sepsis\nClaim about pneumonia\nClaim about asthma"
        assert parse_claims(raw, limit=2) == ["Claim about sepsis", "Claim about pneumonia"]


# --------------------------------------------------------------------------- #
# The property the benchmark depends on
# --------------------------------------------------------------------------- #
def test_all_arms_can_be_parsed_by_one_function():
    """A baseline reply and an Apiro reply must yield comparable candidate lists.

    The C-NIAH harness graded baselines over every non-empty line of raw output
    (uncapped) and Apiro over three parsed slots. Whatever the formatting, the
    same parser at the same limit must now produce the same-shaped answer for
    both, or the accuracy comparison measures formatting rather than reasoning.
    """
    baseline_reply = (
        "Here are my top 3 differential diagnoses:\n\n"
        "1. **Pulmonary embolism** - given the pleuritic pain\n"
        "2. **Pneumonia**\n"
        "3. **Aortic dissection**\n\n"
        "Let me know if you need more detail."
    )
    apiro_reply = (
        "DX: Pulmonary embolism\n"
        "DX: Pneumonia\n"
        "DX: Aortic dissection"
    )
    assert parse_differential(baseline_reply, limit=3) == [
        "Pulmonary embolism", "Pneumonia", "Aortic dissection",
    ]
    assert parse_differential(apiro_reply, limit=3) == [
        "Pulmonary embolism", "Pneumonia", "Aortic dissection",
    ]
