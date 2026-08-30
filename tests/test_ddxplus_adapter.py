"""
tests/test_ddxplus_adapter.py
=============================
Unit tests for apiro/corpus/ddxplus_adapter.py.

Built against a synthetic mirror of the DDXPlus schema, so the suite stays
offline: no download, no HuggingFace, no network.

The adapter is deliberately tolerant — DDXPlus mirrors disagree on the internal
key names of release_evidences.json and release_conditions.json — so these
tests pin BOTH halves of that contract:

  * tolerance: alternative key spellings still resolve;
  * loudness:  a dictionary that resolves nothing must RAISE, not quietly hand
    back vignettes containing only demographics. A benchmark that runs for
    hours on unreadable notes and reports a number is worse than one that
    refuses to start.
"""

import csv
import json

import pytest

from apiro.corpus.ddxplus_adapter import DDXPlusAdapter, DDXPlusCase

EVIDENCES = {
    "E_53": {"name": "E_53", "question_en": "Do you have chest pain?"},
    "E_54": {"name": "E_54", "question_en": "How intense is the pain (0-10)?",
             "value_meaning": {"V_179": {"en": "8 out of 10"}}},
    "E_77": {"name": "E_77", "question_en": "Where is the pain located?",
             "value_meaning": {"V_12": {"en": "substernal"}}},
    "E_99": {"name": "E_99", "question_en": "Do you smoke?", "is_antecedent": True},
}
CONDITIONS = {
    "Infarctus du myocarde": {"cond-name-eng": "Myocardial infarction"},
    "Angine instable": {"cond-name-eng": "Unstable angina"},
    "RGO": {"cond-name-eng": "GERD"},
}
ROW = {
    "AGE": "56", "SEX": "M", "PATHOLOGY": "Infarctus du myocarde",
    "EVIDENCES": "['E_53', 'E_54_@_V_179', 'E_77_@_V_12', 'E_99']",
    "INITIAL_EVIDENCE": "E_53",
    "DIFFERENTIAL_DIAGNOSIS":
        "[['Angine instable', 0.24], ['Infarctus du myocarde', 0.61], ['RGO', 0.15]]",
}


def _build(tmp_path, evidences=None, conditions=None, rows=None, n_rows=8):
    d = tmp_path / "ddxplus"
    d.mkdir(parents=True, exist_ok=True)
    (d / "release_evidences.json").write_text(json.dumps(
        EVIDENCES if evidences is None else evidences))
    (d / "release_conditions.json").write_text(json.dumps(
        CONDITIONS if conditions is None else conditions))
    rows = rows if rows is not None else [ROW] * n_rows
    with open(d / "test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROW))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return d


class TestRendering:
    def test_evidence_codes_become_english(self, tmp_path):
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        assert "chest pain" in case.vignette
        assert "Do you smoke?" in case.vignette

    def test_valued_codes_resolve_their_value(self, tmp_path):
        # "E_54_@_V_179" must render the meaning, not the code.
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        assert "8 out of 10" in case.vignette
        assert "substernal" in case.vignette
        assert "V_179" not in case.vignette

    def test_demographics_are_stated(self, tmp_path):
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        assert "56-year-old" in case.vignette and "male" in case.vignette
        assert case.age == 56 and case.sex == "M"

    def test_the_initial_evidence_leads(self, tmp_path):
        # It is the presenting complaint, so it belongs at the top of the note.
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        findings = [l for l in case.vignette.splitlines() if l.startswith("- ")]
        assert "chest pain" in findings[0]

    def test_unknown_codes_are_dropped_not_printed(self, tmp_path):
        row = dict(ROW, EVIDENCES="['E_53', 'E_UNKNOWN', 'E_99']")
        case = DDXPlusAdapter(_build(tmp_path, rows=[row] * 8)).load_cases(
            n=1, seed=1, min_evidences=2)[0]
        assert "E_UNKNOWN" not in case.vignette
        assert case.n_evidences == 2


class TestGroundTruth:
    def test_pathology_is_translated_to_english(self, tmp_path):
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        assert case.ground_truth == "Myocardial infarction"
        assert case.raw_pathology == "Infarctus du myocarde"

    def test_differential_is_english_and_ranked_by_probability(self, tmp_path):
        # The row lists Angine instable first but Infarctus has the highest
        # probability; the reference ordering is by likelihood.
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        assert case.differential == [
            "Myocardial infarction", "Unstable angina", "GERD",
        ]

    def test_json_encoded_lists_are_accepted_too(self, tmp_path):
        row = dict(ROW,
                   EVIDENCES='["E_53", "E_54_@_V_179", "E_77_@_V_12"]',
                   DIFFERENTIAL_DIAGNOSIS='[["RGO", 0.9], ["Angine instable", 0.1]]')
        case = DDXPlusAdapter(_build(tmp_path, rows=[row] * 8)).load_cases(n=1, seed=1)[0]
        assert case.differential[0] == "GERD"
        assert case.n_evidences == 3


class TestSchemaTolerance:
    def test_alternative_key_spellings_resolve(self, tmp_path):
        # Mirrors disagree; the adapter tries several names before giving up.
        evidences = {"E_53": {"question": "Do you have chest pain?"},
                     "E_99": {"en": "Do you smoke?"},
                     "E_77": {"question_eng": "Where is the pain?"}}
        conditions = {"Infarctus du myocarde": {"condition_name_eng": "Myocardial infarction"}}
        row = dict(ROW, EVIDENCES="['E_53', 'E_99', 'E_77']",
                   DIFFERENTIAL_DIAGNOSIS="[['Infarctus du myocarde', 1.0]]")
        case = DDXPlusAdapter(
            _build(tmp_path, evidences, conditions, [row] * 8)
        ).load_cases(n=1, seed=1)[0]
        assert case.ground_truth == "Myocardial infarction"
        assert case.n_evidences == 3

    def test_an_untranslatable_condition_falls_back_to_its_raw_name(self, tmp_path):
        row = dict(ROW, PATHOLOGY="Maladie inconnue")
        case = DDXPlusAdapter(_build(tmp_path, rows=[row] * 8)).load_cases(n=1, seed=1)[0]
        assert case.ground_truth == "Maladie inconnue"


class TestFailsLoudly:
    def test_a_dictionary_that_resolves_nothing_raises(self, tmp_path):
        # The failure that matters. Without this the harness would run for
        # hours on notes containing only "The patient is a 56-year-old male."
        # and report an accuracy.
        d = _build(tmp_path, evidences={"TOTALLY_DIFFERENT_KEY": {}})
        with pytest.raises(RuntimeError, match="No DDXPlus row rendered"):
            DDXPlusAdapter(d).load_cases(n=4, seed=1)

    def test_missing_dictionary_raises_with_the_fix(self, tmp_path):
        d = _build(tmp_path)
        (d / "release_evidences.json").unlink()
        with pytest.raises(FileNotFoundError, match="fetch_datasets"):
            DDXPlusAdapter(d)

    def test_missing_directory_raises_with_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_datasets"):
            DDXPlusAdapter(tmp_path / "not_downloaded")

    def test_missing_split_raises_with_the_fix(self, tmp_path):
        d = _build(tmp_path)
        with pytest.raises(FileNotFoundError, match="fetch_datasets"):
            DDXPlusAdapter(d).load_cases(n=1, split="validate")


class TestSampling:
    def test_is_reproducible_under_a_seed(self, tmp_path):
        d = _build(tmp_path, rows=[dict(ROW, AGE=str(20 + i)) for i in range(40)])
        a = [c.case_id for c in DDXPlusAdapter(d).load_cases(n=5, seed=3)]
        b = [c.case_id for c in DDXPlusAdapter(d).load_cases(n=5, seed=3)]
        assert a == b

    def test_different_seeds_sample_differently(self, tmp_path):
        d = _build(tmp_path, rows=[dict(ROW, AGE=str(20 + i)) for i in range(60)])
        a = {c.case_id for c in DDXPlusAdapter(d).load_cases(n=5, seed=1)}
        b = {c.case_id for c in DDXPlusAdapter(d).load_cases(n=5, seed=99)}
        assert a != b

    def test_to_dict_is_json_safe(self, tmp_path):
        case = DDXPlusAdapter(_build(tmp_path)).load_cases(n=1, seed=1)[0]
        json.dumps(case.to_dict())
        assert isinstance(case, DDXPlusCase)
