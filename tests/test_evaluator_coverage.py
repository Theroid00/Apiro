"""
tests/test_evaluator_coverage.py
================================
The evaluator must be able to score the cases the generator produces.

This is a scoring-integrity suite, not a matching-quality one. When
`scripts/build_niah_cases.py` grew its clinical bank from 6 diagnoses to 20,
only 2 of the 20 answers had a synonym group — so 22 of 36 clinically correct
paraphrases were graded as misses, including every common abbreviation. That
depresses every arm equally and can swamp the effect under test, exactly like
the markdown-parsing defect before it.

Two properties are pinned here, and they pull in opposite directions:

  * RECALL — a correct answer in any ordinary surface form must be a hit.
  * SEPARATION — the two halves of a confusable pair must NOT match each
    other. If they did, every counterfactual trap would score as correct for
    both the control and the trap answer, and the benchmark would measure
    nothing at all.

Offline: no Ollama, no ChromaDB, no model download.
"""

import pytest

from apiro.eval.evaluator import _check_synthesis_hit, _normalize_text

# Surface forms a competent clinician (or model) might write for each answer
# the generator can emit.
PARAPHRASES = {
    "acute myocardial infarction": ["STEMI", "NSTEMI", "Acute MI", "Heart attack",
                                    "Myocardial infarction"],
    "pulmonary embolism": ["PE", "Pulmonary thromboembolism"],
    "bacterial meningitis": ["Meningitis", "Pneumococcal meningitis"],
    "diabetic ketoacidosis": ["DKA"],
    "acute appendicitis": ["Appendicitis"],
    "hyperkalemia": ["Hyperkalaemia", "High serum potassium"],
    "subarachnoid hemorrhage": ["SAH", "Subarachnoid haemorrhage"],
    "aortic dissection": ["Acute aortic dissection", "Type A aortic dissection"],
    "acute pancreatitis": ["Pancreatitis"],
    "adrenal crisis": ["Addisonian crisis", "Acute adrenal insufficiency"],
    "pheochromocytoma": ["Phaeochromocytoma"],
    "giant cell arteritis": ["Temporal arteritis", "GCA"],
    "guillain barre syndrome": ["Guillain-Barre syndrome", "GBS", "AIDP"],
    "tension pneumothorax": ["Pneumothorax"],
    "septic arthritis": ["Bacterial arthritis"],
    "acute cholangitis": ["Cholangitis", "Ascending cholangitis"],
    "salicylate toxicity": ["Aspirin overdose", "Salicylate poisoning"],
    "acute mesenteric ischemia": ["Mesenteric ischaemia", "Acute mesenteric ischaemia"],
    "carbon monoxide poisoning": ["CO poisoning", "Carbon monoxide toxicity"],
    "thyroid storm": ["Thyrotoxic crisis", "Thyrotoxicosis"],
}

#: The confusable pairs build_niah_cases.py builds counterfactual traps from.
#: These MUST stay distinct.
CONFUSABLE = [
    ("acute myocardial infarction", "aortic dissection"),
    ("bacterial meningitis", "subarachnoid hemorrhage"),
    ("acute appendicitis", "acute mesenteric ischemia"),
    ("diabetic ketoacidosis", "salicylate toxicity"),
    ("acute pancreatitis", "adrenal crisis"),
    ("thyroid storm", "pheochromocytoma"),
    ("pulmonary embolism", "tension pneumothorax"),
    ("hyperkalemia", "guillain barre syndrome"),
    ("bacterial meningitis", "carbon monoxide poisoning"),
    ("subarachnoid hemorrhage", "giant cell arteritis"),
    ("septic arthritis", "acute cholangitis"),
    ("pulmonary embolism", "acute myocardial infarction"),
]


class TestRecall:
    @pytest.mark.parametrize(
        "truth,prediction",
        [(t, p) for t, ps in PARAPHRASES.items() for p in ps],
    )
    def test_a_correct_paraphrase_is_a_hit(self, truth, prediction):
        hit, _ = _check_synthesis_hit([prediction], truth)
        assert hit, f"{prediction!r} should match ground truth {truth!r}"

    def test_every_generator_answer_has_coverage(self):
        """No answer the generator emits may be unrecognisable in every form."""
        for truth, paraphrases in PARAPHRASES.items():
            assert any(_check_synthesis_hit([p], truth)[0] for p in paraphrases), truth


class TestSeparation:
    @pytest.mark.parametrize("a,b", CONFUSABLE)
    def test_confusable_diagnoses_do_not_match_each_other(self, a, b):
        # If these collapsed, a counterfactual trap would score as correct
        # whichever way the model answered, and the benchmark would be inert.
        assert not _check_synthesis_hit([a], b)[0], f"{a!r} wrongly matched {b!r}"
        assert not _check_synthesis_hit([b], a)[0], f"{b!r} wrongly matched {a!r}"

    def test_unrelated_diagnoses_do_not_match(self):
        assert not _check_synthesis_hit(["Septic arthritis"], "thyroid storm")[0]
        assert not _check_synthesis_hit(["Carbon monoxide poisoning"], "acute cholangitis")[0]


class TestSpellingNormalization:
    @pytest.mark.parametrize(
        "british,american",
        [
            ("Subarachnoid haemorrhage", "subarachnoid hemorrhage"),
            ("Hyperkalaemia", "hyperkalemia"),
            ("Phaeochromocytoma", "pheochromocytoma"),
            ("Mesenteric ischaemia", "mesenteric ischemia"),
            ("Anaemia", "anemia"),
            ("Pulmonary oedema", "pulmonary edema"),
            ("Oesophageal spasm", "esophageal spasm"),
            ("Paediatric sepsis", "pediatric sepsis"),
        ],
    )
    def test_british_and_american_spellings_normalize_together(self, british, american):
        # The C-NIAH needle bank is written in British English, so a model
        # primed by the note's own spelling produces the British variant.
        # Marking that wrong would penalise a spelling convention.
        assert _normalize_text(british) == _normalize_text(american)

    def test_normalization_does_not_mangle_unrelated_words(self):
        # A blanket ae/oe -> e rule would break these; the targeted stem list
        # must not.
        for word in ("aerobic", "coexisting", "anaerobic", "coeruleus"):
            assert word in _normalize_text(f"finding of {word} origin"), word
