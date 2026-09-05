#!/usr/bin/env bash
#
# run_eval.sh — the whole evaluation pipeline, in order.
#
#   ./run_eval.sh                 full pipeline (hours)
#   ./run_eval.sh --quick         same stages at small N, to prove it all works
#   ./run_eval.sh --dry-run       print what would run, run nothing
#   ./run_eval.sh <stage> [...]   just these stages
#
# Stages, in dependency order:
#   preflight   python, package imports, Ollama reachable, corpus non-empty
#   test        offline test suite — no Ollama, no ChromaDB, no downloads
#   fetch       download + verify external datasets
#   generate    build the C-NIAH counterfactual case set
#   niah        C-NIAH: bias trap rate, abstention, distractor selection
#   medeinst    MedEinst: external paired counterfactual Bias Trap Rate
#   meddistract MedDistractQA: clean/distracted diagnosis-pair retention
#   mint        MINT-style incremental run (requires MINT_DATASET=/path.json)
#   ddxplus     DDXPlus: external, ranked reference differential
#   cupcase     CUPCase: external, curated per-case distractors
#   calibration ECE / Brier / risk-coverage over the C-NIAH results
#
# Every stage logs to data/logs/<stage>.log and the run stops at the first
# failure. Read docs/BENCHMARKING.md before trusting any number this prints —
# in particular the section on what the synthetic cases can and cannot answer.

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
LOG_DIR="data/logs"
QUICK=0
DRY_RUN=0
STAGES=()

ALL_STAGES=(preflight test fetch generate niah medeinst meddistract ddxplus cupcase mint calibration)

# ── Tunables (env-overridable) ──────────────────────────────────────────────
NIAH_PAIRS="${NIAH_PAIRS:-40}"      # counterfactual pairs -> 2x cases + unanswerable
DDXPLUS_N="${DDXPLUS_N:-60}"
CUPCASE_N="${CUPCASE_N:-60}"
MEDEINST_PAIRS="${MEDEINST_PAIRS:-60}"
MEDDISTRACT_N="${MEDDISTRACT_N:-100}"
MINT_DATASET="${MINT_DATASET:-}"
SEED="${SEED:-7}"
TAU="${TAU:-0.65}"

# ── Arg parsing ─────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --quick)   QUICK=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            # Print the header comment block, stopping at the first line that
            # is not a comment — a hard-coded line range drifts the moment the
            # header is edited, and starts printing shell code.
            awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
            exit 0 ;;
        -*)
            echo "Unknown option: $arg" >&2; exit 2 ;;
        *)
            if [[ " ${ALL_STAGES[*]} " == *" $arg "* ]]; then
                STAGES+=("$arg")
            else
                echo "Unknown stage: $arg" >&2
                echo "Expected one of: ${ALL_STAGES[*]}" >&2
                exit 2
            fi ;;
    esac
done

if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=("${ALL_STAGES[@]}")
fi

if [[ $QUICK -eq 1 ]]; then
    # Small enough to finish in minutes. Proves the pipeline end to end; far
    # too small to support any claim — see the power analysis in
    # docs/BENCHMARKING.md.
    NIAH_PAIRS=4; DDXPLUS_N=4; CUPCASE_N=4; MEDEINST_PAIRS=4; MEDDISTRACT_N=4
fi

mkdir -p "$LOG_DIR"

# ── Helpers ─────────────────────────────────────────────────────────────────
c_bold=$'\033[1m'; c_red=$'\033[31m'; c_green=$'\033[32m'
c_yellow=$'\033[33m'; c_reset=$'\033[0m'
[[ -t 1 ]] || { c_bold=""; c_red=""; c_green=""; c_yellow=""; c_reset=""; }

STAGE_START=0
banner() {
    STAGE_START=$SECONDS
    echo
    echo "${c_bold}==============================================================${c_reset}"
    echo "${c_bold}  $1${c_reset}"
    echo "${c_bold}==============================================================${c_reset}"
}
ok()   { echo "${c_green}[ok]${c_reset}   $1 ($((SECONDS - STAGE_START))s)"; }
warn() { echo "${c_yellow}[warn]${c_reset} $1"; }
die()  { echo "${c_red}[fail]${c_reset} $1" >&2; exit 1; }

# Run a command, tee to a per-stage log, fail the script if it fails.
run() {
    local stage="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  would run: $*"
        return 0
    fi
    echo "  \$ $*"
    # `set -o pipefail` is active, so the pipeline's status is the command's
    # status even though tee succeeds. No separate PIPESTATUS check is needed
    # (and one placed after this `if` would be reading the wrong pipeline).
    if ! "$@" 2>&1 | tee "$LOG_DIR/${stage}.log"; then
        die "$stage failed. Full output: $LOG_DIR/${stage}.log"
    fi
}

has_stage() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }

# ── Stages ──────────────────────────────────────────────────────────────────
stage_preflight() {
    banner "PREFLIGHT"
    command -v "$PY" >/dev/null || die "$PY not found. Set PYTHON=/path/to/python3."
    echo "  python: $($PY --version 2>&1) at $(command -v "$PY")"

    [[ -n "${VIRTUAL_ENV:-}" ]] || warn "No virtualenv active. Continuing with $(command -v "$PY")."

    if [[ $DRY_RUN -eq 1 ]]; then echo "  (dry run: skipping live checks)"; return; fi

    "$PY" - <<'PYEOF' || die "Preflight failed. Fix the above before benchmarking."
import sys
sys.path.insert(0, ".")
problems = []

try:
    import apiro.eval.metrics, apiro.parsing, apiro.eval.evaluator      # noqa: F401
    print("  apiro package imports cleanly")
except Exception as exc:
    problems.append(f"apiro package will not import: {exc}")

# The evaluator must be able to recognise the answers the generator emits.
# Without this fix every arm is depressed by a scoring artifact large enough
# to swamp the effect under test (see docs/BENCHMARKING.md).
try:
    from apiro.eval.evaluator import _check_synthesis_hit
    probes = [("subarachnoid hemorrhage", "SAH"),
              ("diabetic ketoacidosis", "DKA"),
              ("hyperkalemia", "Hyperkalaemia")]
    missed = [p for t, p in probes if not _check_synthesis_hit([p], t)[0]]
    if missed:
        problems.append(
            f"evaluator cannot score {missed} — this checkout predates the "
            f"synonym/spelling fix and every arm will be depressed by it")
    else:
        print("  evaluator recognises abbreviations and spelling variants")
except Exception as exc:
    problems.append(f"evaluator probe failed: {exc}")

try:
    import requests
    from apiro.config import OLLAMA_BASE_URL, PRIMARY_MODEL
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5); r.raise_for_status()
    pulled = {m["name"] for m in r.json().get("models", [])}
    print(f"  ollama reachable at {OLLAMA_BASE_URL} ({len(pulled)} model(s))")
    base = PRIMARY_MODEL.split(":")[0]
    if not any(p.startswith(base) for p in pulled):
        problems.append(f"PRIMARY_MODEL '{PRIMARY_MODEL}' not pulled. "
                        f"Run: ollama pull {PRIMARY_MODEL}")
    else:
        print(f"  model '{PRIMARY_MODEL}' is available")
except Exception as exc:
    problems.append(f"ollama unreachable: {exc}  (start it with: ollama serve)")

try:
    from apiro.corpus.embedder import Embedder
    n = Embedder().count
    if n == 0:
        problems.append("ChromaDB corpus is empty. Build it with: "
                        "python -m apiro.corpus.build_corpus --sources medrag")
    else:
        print(f"  corpus: {n:,} documents")
except Exception as exc:
    problems.append(f"corpus unavailable: {exc}")

if problems:
    print("\nPREFLIGHT PROBLEMS:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
PYEOF
    ok "preflight"
}

stage_test() {
    banner "OFFLINE TEST SUITE"
    if ! "$PY" -c "import pytest" 2>/dev/null; then
        warn "pytest not installed (pip install -e '.[dev]'); skipping."
        return
    fi
    run test "$PY" -m pytest -q
    ok "tests"
}

stage_fetch() {
    banner "FETCH DATASETS"
    run fetch "$PY" scripts/fetch_datasets.py
    ok "datasets ready"
}

stage_generate() {
    banner "GENERATE C-NIAH CASES  ($NIAH_PAIRS counterfactual pairs)"
    run generate "$PY" scripts/build_niah_cases.py \
        --counterfactual --num-cases "$NIAH_PAIRS" --seed "$SEED"
    ok "data/niah_cases.json"
}

stage_niah() {
    banner "C-NIAH  — bias trap rate, abstention, distractor selection"
    [[ -f data/niah_cases.json ]] || die "data/niah_cases.json missing. Run: ./run_eval.sh generate"
    run niah "$PY" scripts/run_niah_eval.py \
        --cases data/niah_cases.json --real --out data/niah_eval_results.json
    ok "data/niah_eval_results.json"
}

stage_medeinst() {
    banner "MEDEINST — paired counterfactual Bias Trap Rate  (pairs=$MEDEINST_PAIRS)"
    run medeinst "$PY" scripts/run_medeinst_eval.py --n-pairs "$MEDEINST_PAIRS" --seed "$SEED"
    ok "MedEinst immutable run"
}

stage_meddistract() {
    banner "MEDDISTRACTQA — diagnosis-only clean/distracted pairs  (N=$MEDDISTRACT_N)"
    run meddistract "$PY" scripts/run_meddistractqa_eval.py --n "$MEDDISTRACT_N" --seed "$SEED"
    ok "MedDistractQA immutable run"
}

stage_mint() {
    banner "MINT — incremental evidence and commitment behavior"
    if [[ -z "$MINT_DATASET" ]]; then
        warn "MINT_DATASET is unset; paper has no linked public dataset, skipping optional stage."
        return
    fi
    run mint "$PY" scripts/run_mint_eval.py --dataset-json "$MINT_DATASET" --seed "$SEED"
    ok "MINT immutable run"
}

stage_ddxplus() {
    banner "DDXPLUS  — external, ranked reference differential  (N=$DDXPLUS_N)"
    run ddxplus "$PY" scripts/run_ddxplus_eval.py \
        --n "$DDXPLUS_N" --seed "$SEED" --out data/ddxplus_eval_results.json
    ok "data/ddxplus_eval_results.json"
}

stage_cupcase() {
    banner "CUPCASE  — external, curated per-case distractors  (N=$CUPCASE_N)"
    run cupcase "$PY" scripts/run_cupcase_eval.py \
        --n "$CUPCASE_N" --seed "$SEED" --out data/cupcase_eval_results.json
    ok "data/cupcase_eval_results.json"
}

stage_calibration() {
    banner "CALIBRATION  — ECE / Brier / risk-coverage"
    [[ -f data/niah_eval_results.json ]] || die "data/niah_eval_results.json missing. Run the niah stage first."
    run calibration "$PY" scripts/run_safety_calibration_eval.py \
        --input data/niah_eval_results.json --tau "$TAU"
    ok "data/calibration_eval_results.json"
}

# ── Main ────────────────────────────────────────────────────────────────────
echo "${c_bold}Apiro evaluation pipeline${c_reset}"
echo "  stages : ${STAGES[*]}"
echo "  mode   : $([[ $QUICK -eq 1 ]] && echo 'QUICK (tiny N — proves the pipeline, proves nothing else)' || echo 'full')"
echo "  logs   : $LOG_DIR/"
[[ $DRY_RUN -eq 1 ]] && echo "  ${c_yellow}DRY RUN — nothing will execute${c_reset}"

PIPELINE_START=$SECONDS
for stage in "${ALL_STAGES[@]}"; do
    has_stage "$stage" && "stage_$stage"
done

echo
echo "${c_bold}==============================================================${c_reset}"
echo "${c_green}  PIPELINE COMPLETE${c_reset} in $(( (SECONDS - PIPELINE_START) / 60 ))m $(( (SECONDS - PIPELINE_START) % 60 ))s"
echo "${c_bold}==============================================================${c_reset}"
if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Results:"
    for f in data/niah_eval_results.json data/ddxplus_eval_results.json \
             data/cupcase_eval_results.json data/calibration_eval_results.json; do
        [[ -f "$f" ]] && echo "    $f  ($(du -h "$f" | cut -f1))"
    done
    echo "  Logs:    $LOG_DIR/"
    echo
    echo "  ${c_yellow}Read the results with docs/BENCHMARKING.md open.${c_reset} In particular:"
    echo "    - check n_candidates is equal across arms before reading any accuracy"
    echo "    - the primary endpoint is bias trap rate, not aggregate accuracy"
    echo "    - quote the interval and the McNemar p, not the point estimate"
    [[ $QUICK -eq 1 ]] && echo "    - ${c_yellow}QUICK mode: these N are far too small to support a claim${c_reset}"
fi
