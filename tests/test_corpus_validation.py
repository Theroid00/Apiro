from scripts.validate_corpus import validate_records


def test_live_corpus_validator_reports_schema_drift():
    report = validate_records(["a"], ["text"], [{"source_db": "x", "medical_domain": "lab"}])
    assert report["valid"] is False
    assert "missing evidence_level" in report["errors"][0]


def test_live_corpus_validator_fingerprints_valid_records():
    report = validate_records(
        ["a"], ["clinical text"],
        [{"source_db": "pubmed", "medical_domain": "lab", "evidence_level": 2}],
    )
    assert report["valid"] is True
    assert len(report["manifest_sha256"]) == 64
