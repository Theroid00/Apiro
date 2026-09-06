"""Immutable provenance records for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def content_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *, benchmark: str, dataset: str, revision: str, case_ids: list[str], config: dict
) -> dict:
    run_id = f"{benchmark}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    status = _git("status", "--porcelain")
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "dataset": {"id": dataset, "revision": revision},
        "cases": {"count": len(case_ids), "ids_sha256": content_hash(case_ids)},
        "code": {"git_commit": _git("rev-parse", "HEAD"), "dirty": bool(status)},
        "configuration": config,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def create_run_directory(manifest: dict, root: Path) -> Path:
    path = root / manifest["run_id"]
    path.mkdir(parents=True, exist_ok=False)
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
