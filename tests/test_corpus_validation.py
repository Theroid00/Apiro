from scripts.validate_corpus import validate_collection, validate_records


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


def test_live_corpus_validator_reads_large_collections_in_batches():
    class Collection:
        def __init__(self):
            self.offsets = []

        def count(self):
            return 3

        def get(self, *, limit, offset, include):
            self.offsets.append(offset)
            stop = min(offset + limit, 3)
            return {
                "ids": [str(i) for i in range(offset, stop)],
                "documents": ["text"] * (stop - offset),
                "metadatas": [{
                    "source_db": "pubmed", "medical_domain": "lab",
                    "evidence_level": 2,
                }] * (stop - offset),
            }

    collection = Collection()
    report = validate_collection(collection, batch_size=2)

    assert report["valid"] is True
    assert report["document_count"] == 3
    assert collection.offsets == [0, 2]
