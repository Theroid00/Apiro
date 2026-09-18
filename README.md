# Apiro

Apiro is a local research prototype for differential diagnosis in clinical
reports containing irrelevant or misleading information. It extracts clinical
facts, retrieves biomedical evidence, ranks diagnoses, and records provenance.
It is not clinical decision-support software.

## Modes

| Mode | Purpose |
|---|---|
| `simple` | Efficient bounded reasoning; default |
| `investigator` | Bounded candidate/evidence audit with one optional revision |
| `legacy` | Original entropy-guided graph traversal |

`investigator` is available on `new/complete-apiro`. The efficient branch
`new/simplified-apiro` provides `simple` and `legacy`.

Legacy implementation files are isolated under [`apiro/legacy/`](apiro/legacy/)
for reproducible ablations; shared graph data types remain under
`apiro/graph/`.

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

Investigator mode separates patient facts from retrieved medical knowledge,
requires exact retrieved evidence spans, measures uncertainty across competing
diagnoses, and performs at most one counterfactual revision. Its graph is an
output provenance record; it does not control runtime traversal.

Simple is the fast structured-RAG baseline: it extracts facts, retrieves
medical context, generates a ranked differential, and applies deterministic
validation. Investigator adds an explicit evidence audit: it measures entropy,
single-fact dependence, and counterfactual rank stability before deciding
whether to perform one targeted revision. In the five-pair validation,
Investigator was more cautious but about 4.5 times slower; it should be treated
as an audit experiment, not an established accuracy improvement.

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
