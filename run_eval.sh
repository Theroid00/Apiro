#!/usr/bin/env bash
#
# run_eval.sh — convenience wrapper for the Apiro benchmark suite.
#
# The previous version activated `venv/bin/activate` unconditionally (the
# repo's .gitignore names `.venv/`, and the file does not exist in a fresh
# clone, so `source` failed and the script kept going against whatever
# interpreter happened to be on PATH) and described the run as "HADCE", an
# engine purged from the codebase in July 2026.
#
# Usage:
#   ./run_eval.sh                # PMC real-world benchmark (N = 10)
#   ./run_eval.sh niah           # Clinical Needle-In-A-Haystack
#   ./run_eval.sh cupcase        # CUPCase distractor-resilience benchmark
#   ./run_eval.sh calibration    # ECE / Brier / risk-coverage over NIAH results
#
# Requires Ollama serving PRIMARY_MODEL and a populated ChromaDB corpus.
# Activate your virtualenv first; this script deliberately does not guess.

set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[!] No virtualenv active. Activate one first, e.g.:" >&2
    echo "      source .venv/bin/activate" >&2
    echo "    Continuing with $(command -v python3)." >&2
fi

TARGET="${1:-pmc}"

case "$TARGET" in
    pmc)
        echo "==> PMC real-world case-report benchmark (N = 10)"
        exec python3 scripts/run_pmc_eval.py --real "${@:2}"
        ;;
    niah)
        if [[ ! -f data/niah_cases.json ]]; then
            echo "[!] data/niah_cases.json is missing — generating it first." >&2
            python3 scripts/build_niah_cases.py
        fi
        echo "==> Clinical Needle-In-A-Haystack benchmark"
        exec python3 scripts/run_niah_eval.py --cases data/niah_cases.json --real "${@:2}"
        ;;
    cupcase)
        echo "==> CUPCase distractor-resilience benchmark"
        exec python3 scripts/run_cupcase_eval.py "${@:2}"
        ;;
    calibration)
        echo "==> Safety / calibration / selective-abstention evaluation"
        exec python3 scripts/run_safety_calibration_eval.py \
            --input data/niah_eval_results.json --tau 0.65 "${@:2}"
        ;;
    *)
        echo "Unknown target '$TARGET'. Expected: pmc | niah | cupcase | calibration" >&2
        exit 2
        ;;
esac
