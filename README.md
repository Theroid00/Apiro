# Apiro

Apiro is a local research prototype for differential diagnosis in clinical
reports containing irrelevant or misleading information. It extracts facts,
retrieves biomedical evidence, ranks diagnoses, and records provenance.
It is not clinical decision-support software.

This branch contains the efficient bounded engine and the original traversal:

| Mode | Purpose |
|---|---|
| `simple` | Efficient structured medical RAG; default |
| `legacy` | Original entropy-guided graph traversal |

## Setup

Requires Python 3.11+, Ollama, and a built ChromaDB corpus.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
ollama pull llama3.1:8b
python -m apiro.corpus.build_corpus --sources textbooks
```

Configure `PRIMARY_MODEL`, `OLLAMA_BASE_URL`, and other settings in
`.env.example` when needed.

## Run

```bash
apiro --findings "49yo female with dyspnea" --mode simple
apiro --findings "49yo female with dyspnea" --mode legacy
python -m apiro.web
```

## Primary Benchmark

The main test is paired clean/distracted diagnosis reports. The same case is
evaluated before and after plausible irrelevant context is added.

```bash
APIRO_REASONING_MODE=simple python scripts/run_meddistractqa_eval.py --n 100
APIRO_REASONING_MODE=legacy python scripts/run_meddistractqa_eval.py --n 100
```

The headline metric is top-1 diagnostic retention under distraction. MedEinst
is a secondary anchoring-bias benchmark; DDXPlus and CUPCase are clean-accuracy
guardrails.

## Development

```bash
.venv/bin/pytest -q
```

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md),
[`docs/APIRO_IMPLEMENTATION_OPTIONS.md`](docs/APIRO_IMPLEMENTATION_OPTIONS.md),
and [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).
