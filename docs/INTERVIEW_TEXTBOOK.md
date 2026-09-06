# Apiro Technical and Interview Guide

This guide explains the current implementation, the evidence it has produced,
and the claims that the evidence does and does not support. Apiro is an
evaluation-ready research prototype for studying diagnostic reasoning under
misleading context. It is not a validated clinical decision-support system.

For the shortest operational summary, see [PROJECT_STATUS.md](PROJECT_STATUS.md).
For exact benchmark commands and interpretation rules, see
[BENCHMARKING.md](BENCHMARKING.md).

## 1. Research question

Apiro tests whether an explicit search process can resist diagnostic
distractors better than a bare language model or standard retrieval-augmented
generation. Its working hypothesis is that three mechanisms may help:

1. Anchor the search in findings extracted from the patient narrative.
2. Spend exploration on uncertain hypotheses.
3. Lower the priority of hypotheses that contradict patient evidence.

These are hypotheses under evaluation. Passing unit tests proves the software
implements them consistently; only adequately powered, held-out experiments
can establish whether they improve diagnostic robustness.

## 2. Current architecture

```text
clinical narrative
      |
      v
axiom extraction (regex labs/vitals + biomedical NER + negation)
      |
      v
depth-0 findings in a belief graph
      |
      v
priority frontier --> retrieve evidence --> generate hypotheses
      |                                      |
      |                                      v
      |                         patient-specific confidence score
      |                                      |
      +<----------- contradiction checks ----+
      |
      v
saturation/budget stop --> differential synthesis --> parsed top-k diagnoses
```

### Runtime isolation

Large resources such as the embedder, model client, and corpus connection can
be shared. Mutable state is created for each investigation: the belief graph,
frontier, saturation history, callback, run log, and traversal counters. This
prevents concurrent web requests or benchmark cases from leaking state into
one another.

### Axiom extraction

`apiro/axioms/` combines deterministic parsing of laboratory values and vital
signs with the `d4data/biomedical-ner-all` token-classification model. It also
tracks bounded negation and assigns heuristic specificity weights. At most 20
seed findings are retained by default.

Seeds are strong anchors, not perfect clinical facts. Regex coverage and NER
quality limit what is extracted. Negated and historical findings receive
higher initial uncertainty than direct positive findings.

### Belief graph and traversal

`BeliefGraph` stores a NetworkX graph alongside explicit node and edge
collections. Depth-0 nodes represent extracted findings. Deeper nodes represent
generated diagnostic hypotheses or refinements.

The active defaults are:

| Setting | Value |
|---|---:|
| Maximum graph nodes | 200 |
| Maximum depth | 6 |
| Exploration expansions | 24 |
| Child hypotheses per expansion | 3 |
| Output differential size | 3 |
| Minimum relevance | 0.40 |

Depth-0 priority favors specific, highly weighted anchors:

```text
priority = 2.0 - entropy + 0.1 * axiom_weight
```

For generated nodes, the frontier favors uncertainty that remains relevant to
the full case:

```text
priority = entropy * (0.4 + 0.6 * relevance) - contradiction_penalty
```

The current formula has no depth-decay multiplier.

### Retrieval and context selection

The default embedding model is `all-mpnet-base-v2`. Retrieval requests six
chunks and rejects results beyond the configured distance threshold of 0.65.
If fewer than two grounded chunks survive, expansion falls back to the model's
parametric knowledge rather than presenting weak retrieval as evidence.

Long cases are handled with evidence-aware selection. The original narrative
is preserved for all arms; prompt construction reserves clinically important
content and selects relevant spans instead of keeping only a fixed prefix.

### Entropy signal

Generated hypotheses are scored from a model's verbalized estimate of
patient-specific confidence, not from token log probabilities. If the model
returns probability `p`, Apiro computes binary entropy:

```text
H2(p) = -p log2(p) - (1-p) log2(1-p)
```

The result is rescaled into the traversal range `[0.05, 0.693]`. This makes it
a useful uncertainty heuristic, but it is not a fitted probability calibrator
and should not be described as calibrated epistemic uncertainty.

### Contradiction handling

Contradiction detection has two stages:

1. A keyword and antonym filter handles obvious conflicts and identifies
   medically related pairs.
2. An Ollama judge adjudicates related pairs that require semantic reasoning.

The implementation does not load a cross-encoder NLI model. A confirmed
contradiction subtracts `0.8` from frontier priority; it does not delete the
node or multiply its weight. This soft penalty preserves competing
differentials while making contradicted paths less attractive.

### Stopping and synthesis

Only explored nodes at depth one or greater enter the saturation history. The
default detector requires at least eight exploration observations and then
examines a five-value window. It stops when mean entropy is below `0.55`,
variance is below `0.04`, and the trend is non-increasing. Graph, depth, and
expansion budgets provide independent hard stops.

Synthesis receives the original vignette, anchors, negated findings, supporting
evidence, and contradiction annotations. A shared parser extracts at most
three diagnoses from every arm. Experimental abstention is enabled only for
benchmarks that explicitly permit it.

### Telemetry and reproducibility

Adversarial live runners write an immutable manifest containing the dataset
source and declared revision, case-selection parameters, model configuration,
Git state, and content hashes. The MedEinst loader is pinned to revision
`354f4b5`; runners that declare `main` or `local` do not provide the same level
of source immutability. The shared model scheduler records
calls, retries, failures, timeouts, prompt and completion tokens, queue time,
and inference time by purpose.

## 3. Benchmark suite

| Benchmark | What it tests | Current integration |
|---|---|---|
| MedEinst | Anchoring on a distractor disease after a discriminative cue changes | Official paired data from `zhui711/MedEinst`; rank-1 BTR is primary |
| MedDistractQA | Robustness to plausible nonliteral and bystander distractions | Diagnosis-compatible subset with explicit task validation |
| MINT-style sessions | Premature commitment as evidence arrives over turns | Incremental runner with first-commit and answer-change metrics |
| C-NIAH | Controlled long-context needles, traps, and missing evidence | Locally generated mechanism benchmark |
| CUPCase | Rare cases with curated distractors | External dataset adapter and runner |
| DDXPlus | Structured symptoms and ranked differential labels | External adapter and runner |

RABBITS and knowledge-graph-guided distractor generation are documented as
upstream multiple-choice tools. They are not presented as direct Apiro runners
because Apiro emits free-text diagnoses; a valid integration first needs a
paired narrative transformation and an endpoint compatible with every arm.

## 4. MedEinst metrics

For each control/trap pair, the primary prediction is the first diagnosis.
Let `C` be the correct control diagnosis and let `T` be the answer in the trap
version. Bias Trap Rate is:

```text
BTR = P(T = C | control prediction = C)
```

The denominator therefore contains only pairs for which the arm solved the
control at rank one. A trap is counted only when the trap prediction retains
the former control diagnosis at rank one. A correct trap diagnosis appearing
elsewhere in the top three does not cancel that rank-1 anchoring event.

Pair resilience is reported separately. Top-3 accuracy remains a secondary
capability measure and must not replace the published rank-1 BTR definition.

## 5. Current evidence

The first live MedEinst run was a smoke test of five pairs. After correcting
the scorer to use the official rank-1 definition, the result is:

| Arm | Control@1 | Trap@1 | BTR | Eligible / trapped | Pair@1 |
|---|---:|---:|---:|---:|---:|
| Apiro | 20% | 0% | 100% | 1 / 1 | 0% |
| Standard RAG | 20% | 20% | 100% | 1 / 1 | 0% |
| Bare LLM | 20% | 0% | 100% | 1 / 1 | 0% |

Secondary top-3 results were Apiro 60% control / 40% trap / 40% paired,
Standard RAG 40% / 40% / 20%, and Bare LLM 60% / 40% / 20%.

This run does not identify a winning arm. Each BTR estimate has only one
eligible pair, so all three estimates are maximally fragile. The earlier table
showing Apiro BTR at 0% resulted from applying the trap rule across top-3
predictions and is obsolete.

The run also exposed the main operational bottleneck. Across ten case variants
and all three arms, it made 1,814 model calls, used 867,582 prompt tokens, and
took about 19.35 minutes. The mean all-arm latency was 116.08 seconds per case.
Contradiction handling accounted for 1,192 calls, which makes pair reduction
and deduplication the clearest performance target.

Historical C-NIAH, CUPCase, DDXPlus, and PMC tables in the repository are
exploratory. Several predate shared answer parsing and other measurement fixes,
and their sample sizes are too small for comparative claims. Preserve them as
engineering history; do not cite them as evidence of clinical superiority.

## 6. What is complete

The engineering foundation for the adversarial evaluation phase is complete:

- per-run traversal isolation;
- preserved full narratives and evidence-aware context selection;
- shared, bounded model scheduling and detailed telemetry;
- immutable manifests and a pinned MedEinst dataset revision;
- corrected AURC and calibration-threshold handling;
- MedEinst, MedDistractQA, and MINT-style runners;
- corpus-schema validation; and
- rank-1 MedEinst scoring with a rescore path for existing results.

The research project is not complete. It does not yet have powered, held-out
evidence that Apiro improves robustness, accuracy, or calibration.

## 7. Next work, in order

1. Reduce contradiction-judge calls by deduplicating pairs and comparing new
   nodes only with anchors, ancestors, and a small nearest-neighbor set.
2. Add retrieval and graph-event counters so quality can be reported per unit
   of compute.
3. Improve rank-1 synthesis, because the smoke run often contained the correct
   answer below rank one.
4. Run `scripts/validate_corpus.py` against the actual corpus and repair or
   explicitly version any metadata mismatch.
5. Run small train-split pilots to freeze prompts, parsing, stopping settings,
   and the power plan.
6. Run a powered, unseen MedEinst evaluation and the compatible
   MedDistractQA/MINT stages without tuning on their results.
7. Fit abstention and confidence thresholds on a held-out calibration split.
8. Publish results with confidence intervals, paired tests, failure examples,
   manifests, and compute-normalized metrics.

## 8. Defensible interview answers

**What is novel about Apiro?**

The project tests a specific composition: patient-fact anchoring, uncertainty-
guided graph traversal, and contradiction penalties, measured against
compute-visible bare-LLM and RAG baselines. Novelty is the testable system
design, not a claim that its individual components are new.

**Is the entropy a true model-distribution entropy?**

No. It is binary entropy calculated from a verbalized confidence estimate and
then rescaled for traversal. It is an interpretable heuristic whose usefulness
must be validated empirically.

**Is contradiction detection NLI?**

It is an NLI-like decision task implemented by a lexical prefilter and an LLM
judge. The current runtime does not contain a cross-encoder NLI model.

**Why soft-prune instead of delete?**

Clinical alternatives can conflict with each other while remaining reasonable
differentials. A priority penalty lets strong later evidence recover a path;
hard deletion cannot.

**Why is BTR conditioned on correct controls?**

A model cannot demonstrate loss of a previously correct diagnosis if it never
solved the control. Conditioning isolates anchoring from baseline capability.

**Why did the first MedEinst table change?**

The first implementation treated any top-3 retention as a trap and suppressed
some traps when the correct answer appeared elsewhere. The benchmark defines a
single top-1 prediction, so the scorer and tests were corrected and the saved
run can be rescored reproducibly.

**What did the smoke test prove?**

It proved that the live pipeline, manifests, telemetry, and paired scorer run
end to end. It did not prove a performance advantage: only one pair per arm
qualified for BTR.

**What is the main cost problem?**

Contradiction adjudication. It consumed 1,192 of 1,814 calls in the smoke run.
The next optimization should reduce candidate pairs while measuring recall on
known contradictions.

**How will you avoid tuning on the test set?**

Freeze prompts, model versions, parser behavior, thresholds, and stopping rules
using unit fixtures and a train-split pilot. Then execute the unseen evaluation
once and retain the immutable manifest.

**When is the project complete?**

When a powered unseen run is reproducible from its manifest, all arms use equal
answer budgets and compatible inputs, uncertainty intervals and paired tests
are reported, calibration is fitted on held-out data, and the result is stated
without making clinical deployment claims.
