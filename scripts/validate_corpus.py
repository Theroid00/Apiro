#!/usr/bin/env python3
"""Validate the machine-local ChromaDB corpus as an integration check."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def validate_records(ids, documents, metadatas) -> dict:
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("chunk ids are not unique")
    sources = Counter()
    for index, (chunk_id, text, meta) in enumerate(zip(ids, documents, metadatas)):
        for field in ("source_db", "medical_domain", "evidence_level"):
            if field not in meta or meta[field] in (None, ""):
                errors.append(f"{chunk_id or index}: missing {field}")
        sources[str(meta.get("source_db", "missing"))] += 1
        if not str(text).strip():
            errors.append(f"{chunk_id or index}: empty document")
    digest = hashlib.sha256()
    for item in sorted(zip(ids, documents, metadatas), key=lambda row: row[0]):
        digest.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))
    return {
        "valid": not errors, "document_count": len(ids),
        "source_distribution": dict(sources), "manifest_sha256": digest.hexdigest(),
        "errors": errors,
    }


def main() -> int:
    from apiro.corpus.embedder import Embedder
    embedder = Embedder()
    result = embedder._collection.get(include=["documents", "metadatas"])
    report = validate_records(result.get("ids", []), result.get("documents", []), result.get("metadatas", []))
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
