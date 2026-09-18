# Apiro

Apiro is a local research prototype for differential diagnosis in clinical
reports containing irrelevant or misleading information. It extracts clinical
facts, retrieves biomedical evidence, ranks diagnoses, and records provenance.
It is not clinical decision-support software.

## Modes

| Mode | Purpose |
|---|---|
| `simple` | Efficient bounded reasoning; default |
| `investigator` | Bounded multi-round investigation with graph and memory support |
| `legacy` | Original entropy-guided graph traversal |

`investigator` is available on `new/complete-apiro`. The efficient branch
`new/simplified-apiro` provides `simple` and `legacy`.

## Setup

Requires Python 3.11+, a local Ollama installation, and a built ChromaDB
corpus.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
ollama pull llama3.1:8b
python -m apiro.corpus.build_corpus --sources textbooks
```

Set `PRIMARY_MODEL`, `OLLAMA_BASE_URL`, or the `APIRO_*` variables in
`.env.example` as needed.

## Run

```bash
apiro --findings "49yo female with dyspnea" --mode simple
apiro --findings "49yo female with dyspnea" --mode investigator
python -m apiro.web
```

The investigator mode also supports an optional persisted concept graph:

```bash
python scripts/build_concept_graph.py data/corpus/chunks.jsonl
```

## Primary Benchmark

Apiro's main test is paired clean/distracted diagnosis reports. The same case
is evaluated before and after plausible irrelevant context is added.

```bash
APIRO_REASONING_MODE=simple python scripts/run_meddistractqa_eval.py --n 100
APIRO_REASONING_MODE=investigator python scripts/run_meddistractqa_eval.py --n 100
APIRO_REASONING_MODE=legacy python scripts/run_meddistractqa_eval.py --n 100
```

The headline metric is top-1 diagnostic retention under distraction. MedEinst
is a secondary anchoring-bias benchmark. The DDXPlus and CUPCase runners still
use the legacy traversal and must be migrated before they can act as
clean-accuracy guardrails for `simple` or `investigator`.

## Development

```bash
.venv/bin/pytest -q
```

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md),
[`docs/APIRO_IMPLEMENTATION_OPTIONS.md`](docs/APIRO_IMPLEMENTATION_OPTIONS.md),
and [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for details.
