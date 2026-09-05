#!/usr/bin/env python3
"""
scripts/fetch_datasets.py
=========================
Download and verify the external benchmark datasets.

    CUPCase   ofir408/CupCase          3,562 real clinical cases, each with
                                       three curated distractors. Drives
                                       scripts/run_cupcase_eval.py.

    DDXPlus   aai530-group6/ddxplus    CC-BY. 1.3M synthetic patients, 49
                                       pathologies, structured evidences and a
                                       ground-truth ranked differential. The
                                       substrate MedEinst was built from.
                                       Drives scripts/run_ddxplus_eval.py.

Everything lands under data/datasets/ and is skipped on re-run unless --force.

WHY THIS PRINTS THE SCHEMA
--------------------------
The DDXPlus adapter reads columns it cannot verify without downloading the
file, and a mismatch would silently produce empty vignettes rather than an
error. So this script reports the schema it actually found and checks it
against what the adapter expects. If the two disagree you will see it here,
before a multi-hour benchmark run, rather than in a results file afterwards.

Usage:
    python scripts/fetch_datasets.py                  # both, test split only
    python scripts/fetch_datasets.py --only ddxplus
    python scripts/fetch_datasets.py --splits test validate
    python scripts/fetch_datasets.py --verify-only    # check, download nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("fetch_datasets")

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

CUPCASE_REPO = "ofir408/CupCase"
DDXPLUS_REPO = "aai530-group6/ddxplus"
MEDEINST_REPO = "zhui711/MedEinst"
MEDDISTRACT_REPO = "KrithikV/MedDistractQA"

#: Files the DDXPlus adapter needs. The two JSONs are the dictionaries that
#: turn opaque evidence codes ("E_53", "E_54_@_V_179") into English text —
#: without them the CSV rows cannot be rendered as a clinical note at all.
DDXPLUS_SUPPORT_FILES = ("release_evidences.json", "release_conditions.json")
DDXPLUS_SPLIT_FILES = {"train": "train.csv", "validate": "validate.csv", "test": "test.csv"}

#: Columns apiro/corpus/ddxplus_adapter.py reads.
DDXPLUS_EXPECTED_COLUMNS = {
    "AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE",
    "DIFFERENTIAL_DIAGNOSIS",
}


def _require(module: str, pip_name: str | None = None):
    """Import a module or exit with the install command."""
    try:
        return __import__(module)
    except ImportError:
        logger.error(
            f"'{module}' is not installed.\n"
            f"        pip install {pip_name or module}"
        )
        sys.exit(1)


# --------------------------------------------------------------------------- #
# DDXPlus
# --------------------------------------------------------------------------- #
def fetch_ddxplus(splits: list[str], force: bool, verify_only: bool) -> bool:
    """Download the DDXPlus CSVs and dictionaries; report what arrived."""
    target = DATASETS_DIR / "ddxplus"
    target.mkdir(parents=True, exist_ok=True)

    wanted = list(DDXPLUS_SUPPORT_FILES) + [
        DDXPLUS_SPLIT_FILES[s] for s in splits if s in DDXPLUS_SPLIT_FILES
    ]

    if not verify_only:
        hub = _require("huggingface_hub")
        from huggingface_hub import hf_hub_download

        for filename in wanted:
            dest = target / filename
            if dest.exists() and not force:
                logger.info(f"  [skip] {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
                continue
            logger.info(f"  [get ] {filename} ...")
            try:
                path = hf_hub_download(
                    repo_id=DDXPLUS_REPO,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(target),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"  Failed to download {filename}: {exc}")
                return False
            logger.info(f"  [ok  ] {Path(path).name} "
                        f"({Path(path).stat().st_size / 1e6:.1f} MB)")

    return _verify_ddxplus(target, splits)


def _verify_ddxplus(target: Path, splits: list[str]) -> bool:
    """Check the files are present and the schema matches the adapter."""
    ok = True
    print("\n--- DDXPlus ---")

    for filename in DDXPLUS_SUPPORT_FILES:
        path = target / filename
        if not path.exists():
            logger.error(f"  MISSING {filename} — the adapter cannot render vignettes without it.")
            ok = False
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  {filename} is not readable JSON: {exc}")
            ok = False
            continue
        print(f"  {filename}: {len(data)} entries")
        sample_key = next(iter(data), None)
        if sample_key is not None and isinstance(data[sample_key], dict):
            keys = sorted(data[sample_key].keys())
            print(f"      sample entry {sample_key!r} keys: {keys}")

    import csv

    for split in splits:
        filename = DDXPLUS_SPLIT_FILES.get(split)
        if filename is None:
            continue
        path = target / filename
        if not path.exists():
            logger.error(f"  MISSING {filename}")
            ok = False
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                logger.error(f"  {filename} is empty.")
                ok = False
                continue
            n_rows = sum(1 for _ in reader)
        print(f"  {filename}: {n_rows:,} rows")
        print(f"      columns: {header}")

        missing = DDXPLUS_EXPECTED_COLUMNS - set(header)
        if missing:
            logger.error(
                f"  SCHEMA MISMATCH in {filename}: the adapter expects "
                f"{sorted(missing)} which are not present.\n"
                f"        Present columns: {header}\n"
                f"        Update DDXPLUS_COLUMNS in apiro/corpus/ddxplus_adapter.py "
                f"to match, or report this schema."
            )
            ok = False
        else:
            print("      schema matches apiro/corpus/ddxplus_adapter.py")

    return ok


# --------------------------------------------------------------------------- #
# CUPCase
# --------------------------------------------------------------------------- #
def fetch_cupcase(force: bool, verify_only: bool) -> bool:
    """Warm the HuggingFace cache for CUPCase and report its schema.

    apiro/corpus/clinical_case_adapter.py calls datasets.load_dataset()
    directly, so the useful thing here is to pull it once, up front, rather
    than have a benchmark stall on a download an hour in.
    """
    _require("datasets")
    from datasets import load_dataset

    print("\n--- CUPCase ---")
    try:
        ds = load_dataset(CUPCASE_REPO, split="test", download_mode=(
            "force_redownload" if force else "reuse_dataset_if_exists"))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"  Failed to load {CUPCASE_REPO}: {exc}")
        return False

    print(f"  rows: {len(ds):,}")
    print(f"  columns: {list(ds.column_names)}")

    # The adapter reads these; a rename upstream would silently empty the cases.
    expected = {"clean_case_presentation", "correct_diagnosis"}
    missing = expected - set(ds.column_names)
    if missing:
        logger.error(
            f"  SCHEMA MISMATCH: apiro/corpus/clinical_case_adapter.py expects "
            f"{sorted(missing)}, not present. Columns: {ds.column_names}"
        )
        return False

    n_distractors = sum(
        1 for c in ds.column_names if c.lower().startswith("distractor")
    )
    print(f"  distractor columns: {n_distractors}")
    if n_distractors == 0:
        logger.warning(
            "  No distractor columns found. run_cupcase_eval.py will still run, "
            "but its distractor-selection rate — the reason that benchmark is "
            "here — will report n/a."
        )
    print("  schema matches apiro/corpus/clinical_case_adapter.py")
    return True


def fetch_hf_rows(repo: str, split: str, expected: set[str], force: bool) -> bool:
    """Warm and schema-check a row-oriented Hugging Face dataset."""
    _require("datasets")
    from datasets import load_dataset
    print(f"\n--- {repo} ---")
    try:
        ds = load_dataset(repo, split=split, download_mode=(
            "force_redownload" if force else "reuse_dataset_if_exists"))
    except Exception as exc:
        logger.error(f"  Failed to load {repo}: {exc}")
        return False
    missing = expected - set(ds.column_names)
    print(f"  rows: {len(ds):,}")
    print(f"  columns: {list(ds.column_names)}")
    if missing:
        logger.error(f"  SCHEMA MISMATCH: missing {sorted(missing)}")
        return False
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the external benchmark datasets."
    )
    parser.add_argument("--only", choices=("cupcase", "ddxplus", "medeinst", "meddistract"), default=None,
                        help="Fetch just one dataset (default: both).")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        choices=list(DDXPLUS_SPLIT_FILES),
                        help="DDXPlus splits to download (default: test only — "
                             "train.csv is ~1M rows and the benchmark does not "
                             "need it).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file is already present.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check what is already on disk; download nothing.")
    args = parser.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Datasets directory: {DATASETS_DIR}")

    results: dict[str, bool] = {}
    if args.only in (None, "ddxplus"):
        logger.info(f"Fetching DDXPlus ({DDXPLUS_REPO}), splits={args.splits}")
        results["ddxplus"] = fetch_ddxplus(args.splits, args.force, args.verify_only)
    if args.only in (None, "cupcase"):
        if args.verify_only:
            logger.info("Skipping CUPCase: --verify-only cannot inspect the HF cache "
                        "without loading it.")
        else:
            logger.info(f"Fetching CUPCase ({CUPCASE_REPO})")
            results["cupcase"] = fetch_cupcase(args.force, args.verify_only)
    if args.only in (None, "medeinst"):
        if args.verify_only:
            logger.info("Skipping MedEinst cache inspection in --verify-only mode.")
        else:
            results["medeinst"] = fetch_hf_rows(
                MEDEINST_REPO, "test", {"case_id", "case_type", "narrative", "ground_truth"}, args.force
            )
    if args.only in (None, "meddistract"):
        if args.verify_only:
            logger.info("Skipping MedDistractQA cache inspection in --verify-only mode.")
        else:
            results["meddistract"] = fetch_hf_rows(
                MEDDISTRACT_REPO, "train",
                {"question", "question_choices", "correct_answer", "distracting_sentence", "medical_competency"},
                args.force,
            )

    print("\n" + "=" * 64)
    for name, ok in results.items():
        print(f"  {name:10} {'OK' if ok else 'FAILED'}")
    print("=" * 64)

    if not all(results.values()):
        logger.error("At least one dataset is unusable. Fix that before benchmarking: "
                     "a harness that silently gets empty cases reports a result "
                     "that never happened.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
