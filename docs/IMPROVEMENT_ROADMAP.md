# Apiro Improvement Roadmap

This document outlines the next improvements for Apiro, with emphasis on:

- correctness and operational reliability;
- lower latency and fewer model calls;
- better diagnostic results;
- stronger, reproducible benchmarks; and
- a code structure that makes experiments easier to trust.

`docs/IMPROVEMENTS.md` records earlier defects and their fixes. This roadmap
focuses on work that remains.

## Current baseline

The repository has a strong offline test foundation and useful evaluation
primitives. Its main limitations are now at the boundaries between clinical
input, long-context reasoning, concurrent execution, and experiment
reproducibility.

The following issues were confirmed during the latest review:

1. The web app shares one mutable traversal object between concurrent requests.
2. CUPCase narratives are reconstructed from findings capped at 400 characters.
3. Posterior confidence scoring sees at most the first 3,000 characters of a
   case, even when the decisive evidence is later in the note.
4. AURC depends on input order when confidence values are tied.
5. The offline test suite reads the machine's local ChromaDB when it exists; the
   current local corpus fails the expected metadata schema because some chunks
   lack `evidence_level`.
6. The calibration CLI accepts arbitrary `--tau` values but calculates only the
   hard-coded thresholds 0.50, 0.65, and 0.80. Other values raise `KeyError`.
7. Traversal duration is calculated before final synthesis, omitting a costly
   model call and any formatting retry.

One earlier concern needs qualification: `EntropyEngine.hypothesis_uncertainty`
has a lossy 200-character cache key, but the current traversal calls
`score_hypothesis` directly and bypasses that cache. The cache should still be
fixed before the wrapper is reused, but it is not currently known to invalidate
live traversal results.

## Priority plan

| Priority | Work | Expected benefit | Effort |
|---|---|---|---|
| P0 | Isolate traversal state per request and run | Correct concurrent web behavior and reliable logs | Medium |
| P0 | Preserve full clinical case text throughout evaluation | Prevent missing late evidence and biased benchmark results | Small |
| P0 | Fix AURC tie handling and calibration threshold handling | Correct safety metrics and usable CLI behavior | Small |
| P0 | Separate offline tests from local-corpus integration tests | Deterministic CI and visible corpus schema drift | Medium |
| P1 | Add immutable experiment manifests | Reproducible and comparable benchmark runs | Medium |
| P1 | Add model-call, retrieval, token, and latency instrumentation | Identify the real performance bottlenecks | Medium |
| P1 | Reduce unnecessary contradiction and entropy calls | Lower latency and model load | Medium |
| P1 | Replace prefix truncation with evidence-aware context selection | Better long-note reasoning | Medium |
| P1 | Add compute-matched ablations and repeated runs | Establish which mechanisms improve results | Medium |
| P2 | Split runtime construction, synthesis, and web transport | Easier maintenance and safer optimization | Large |
| P2 | Build a held-out confidence calibration pipeline | Meaningful selective prediction and abstention | Large |

## 1. Correctness and reliability

### 1.1 Create a traversal session per investigation

`ApiroTraversal` stores its active callback, traversal log, critic, saturation
state, and rabbit-hole events on the object. The web app initializes one global
traversal and runs up to two requests concurrently. One request can overwrite
the callback or in-memory log used by another request.

Recommended change:

- introduce a `RuntimeResources` object for safe shared resources such as the
  vector store and model client;
- create a fresh `ApiroTraversal`, `SaturationDetector`, `RabbitHoleDetector`,
  and run log for every investigation;
- assign every run a unique id and write to
  `data/logs/<run-id>/traversal.jsonl`;
- include the run id in streaming events and API responses; and
- add a concurrency test that interleaves two runs and verifies that event
  streams, graphs, stop reasons, and logs remain separate.

Acceptance criteria:

- two concurrent requests cannot receive each other's events;
- repeated requests never overwrite a previous run's log; and
- component construction remains cheap because large immutable resources are
  still shared.

### 1.2 Preserve the original clinical narrative

The CUPCase adapter reduces the full narrative to a first sentence and a
400-character history finding. The benchmark then rebuilds the input seen by
all arms from those shortened findings. Any decisive evidence after that limit
is removed before the comparison begins.

Recommended change:

- carry `narrative` or `vignette` as a first-class field in every case schema;
- use findings only for graph seeding;
- give the original narrative to bare LLM, RAG, Apiro traversal, and final
  synthesis;
- store a content hash and original character/token count in each result; and
- add a fixture whose answer is supported only by evidence after character 400.

Apply the same principle to fallback seeds. A 400-character seed may be useful
as an anchor, but it must not become the only copy of the patient context.

### 1.3 Make long-context selection explicit

Posterior scoring currently uses the first 3,000 characters. Prefix truncation
is poorly matched to C-NIAH, whose purpose is to test evidence at different
depths.

Recommended strategy:

1. Reserve space for demographics, chief complaint, and deterministic axioms.
2. Select additional sentences by semantic relevance to the hypothesis.
3. Include local context around negated and abnormal findings.
4. Fill the remaining budget with evenly spaced sections of the original note.
5. Record selected source spans so the scoring decision can be audited.

Benchmark prefix truncation against this selector on cases stratified by needle
depth. The change is successful if deep-needle accuracy improves without a
material regression on shallow cases or a large increase in prompt tokens.

### 1.4 Fix metric edge cases

For AURC, equal-confidence predictions must be treated as one threshold block.
Their order in the input file should not change the result. Report both the
empirical risk-coverage curve and an explicit tie policy.

For calibration thresholds, always calculate the user-supplied `--tau`, validate
that it lies in `[0, 1]`, and add it to the standard reporting thresholds when
needed.

Tests should cover:

- permutations of tied-confidence examples produce identical AURC;
- all-equal confidence, all-correct, all-wrong, and single-case inputs;
- arbitrary valid `--tau` values; and
- invalid values below zero or above one.

### 1.5 Make configuration imports side-effect free

`apiro/config.py` creates directories during import. Move directory creation to
application startup or the command that needs the path. This makes library
imports safe in read-only installations and removes hidden filesystem behavior
from tests.

### 1.6 Report full end-to-end duration

Calculate traversal duration after final synthesis and log writing, or report
separate values:

- extraction time;
- retrieval time;
- expansion generation time;
- entropy scoring time;
- contradiction judging time;
- synthesis time; and
- total wall-clock time.

The total reported by benchmarks must include every operation needed to produce
the final differential.

## 2. Efficiency and performance

### 2.1 Instrument before changing algorithms

Add a run-level telemetry object that counts:

- model calls by purpose;
- retries and timeouts;
- prompt and completion tokens;
- retrieved chunks and filtered chunks;
- deterministic versus LLM contradiction decisions;
- entropy scores served from cache;
- generated, merged, pruned, and expanded nodes; and
- elapsed time for each pipeline stage.

Persist these fields in benchmark results. Report quality and cost together:

- top-1 or top-3 accuracy per 100 model calls;
- accuracy per minute;
- median and p95 case latency;
- mean prompt tokens per case; and
- peak graph size.

Without these measures, a quality gain may simply reflect spending several
times more inference compute than the baseline.

### 2.2 Reduce entropy-scoring calls

Current expansion can make one generation call and up to one entropy call per
child hypothesis. Practical options, in recommended order:

1. Ask the expansion response for hypothesis name and confidence in one
   structured output, then validate confidence on only ambiguous candidates.
2. Batch several hypotheses into one scoring prompt with strict JSON output.
3. Cache by model digest, prompt version, normalized full case hash, and
   normalized hypothesis.
4. Skip rescoring semantically merged hypotheses.
5. Use a cheap local scorer to prioritize candidates and reserve the LLM scorer
   for the top frontier candidates.

Measure agreement with the current scorer before replacing it. A faster scorer
should preserve rank correlation and downstream accuracy, not merely return
plausible-looking values.

### 2.3 Bound contradiction work

Contradiction comparison can approach quadratic growth as each new node is
checked against many existing nodes. Reduce candidate pairs before invoking the
LLM judge:

- always compare against deterministic axioms and direct ancestors;
- compare against a small number of semantically nearest existing hypotheses;
- exclude unrelated domains unless an entity overlap requires escalation;
- deduplicate symmetric pairs using stable content hashes; and
- expose prefilter recall on a labeled contradiction fixture.

Track `candidate_pairs`, `prefilter_passes`, `LLM_judge_calls`, and confirmed
contradictions. The optimization succeeds when judge calls fall materially
without losing known contradiction cases.

### 2.4 Use adaptive exploration budgets

A fixed exploration budget spends the same maximum work on easy and difficult
cases. Introduce a budget policy using measurable signals:

- stop early when the top hypothesis is stable across several expansions;
- continue when new specific evidence changes the top candidates;
- cap work when retrieval repeatedly returns no grounded chunks;
- reserve extra budget for unresolved contradictions or close posterior scores;
- require a minimum exploration floor before any early stop.

Evaluate the latency-quality frontier at several budgets. Select an operating
point based on a predeclared maximum acceptable accuracy loss and p95 latency.

### 2.5 Control concurrency at the model boundary

Thread-level concurrency can overload a local Ollama instance and increase tail
latency. Add one bounded scheduler for all generation calls, measure queue time
separately from inference time, and tune concurrency for the deployed model and
hardware. Benchmark throughput at concurrency 1, 2, and 4 rather than assuming
more workers are faster.

### 2.6 Cache retrieval with a reproducible key

Use a stable key containing:

- corpus manifest hash;
- embedding model and version;
- normalized query;
- domain filter;
- top-k; and
- distance threshold.

Do not reuse retrieval results when the corpus or embedding model changes.
Persist chunk ids and distances in benchmark results.

## 3. Improving diagnostic results

### 3.1 Establish which mechanisms help

Run compute-matched ablations:

| Arm | Purpose |
|---|---|
| Bare LLM | Capability floor |
| Standard RAG | Retrieval baseline |
| Graph without entropy ordering | Value of graph expansion alone |
| Graph without contradiction pruning | Value of contradiction handling |
| Graph without relevance weighting | Value of case anchoring |
| Graph without semantic merging | Value of deduplication |
| Full Apiro | Combined system |

Match output candidate count, model, temperature, context budget, and either
model-call budget or token budget. Report both unrestricted and compute-matched
results when the full system intentionally uses more compute.

### 3.2 Improve axiom extraction coverage

The regex and NER path is central because extracted axioms are treated as
confirmed anchors. Improve it with a labeled extraction set containing:

- affirmed, negated, hypothetical, conditional, family-history, and historical
  findings;
- values with units and reference ranges;
- temporal changes such as rising troponin or resolving fever;
- medication exposure and adverse reactions; and
- coreference and shorthand common in clinical notes.

Measure span precision/recall and polarity accuracy. Incorrect confirmed axioms
are more dangerous than missing low-value entities, so optimize for precision
and expose extraction confidence to downstream logic.

### 3.3 Replace the hand-curated weight table gradually

`data/axiom_weights.yaml` is useful for prototyping but may favor concepts seen
during benchmark development. Log its coverage on every external dataset and
compare:

- the current table;
- a uniform-weight baseline;
- information-content weights derived from a separate corpus; and
- a small learned ranker trained only on development data.

Keep external test sets out of weight selection. A learned method should replace
the table only if it improves held-out seed ranking and final outcomes.

### 3.4 Improve retrieval quality rather than only retrieval quantity

Build a small labeled query-to-passage relevance set and measure recall@k and
precision@k. Then evaluate:

- biomedical embedding models;
- hybrid lexical and dense retrieval;
- metadata filters by disease, evidence type, and source;
- lightweight reranking of the top retrieved chunks; and
- query construction from the active hypothesis plus selected patient facts.

Retrieval gains should be demonstrated with passage relevance metrics before
being attributed to final diagnosis accuracy.

### 3.5 Separate exploration confidence from answer confidence

Entropy is useful for deciding what to explore. It is not automatically a
calibrated probability that the final answer is correct. Produce final answer
confidence from features available after traversal, fit a calibrator on a
development split, and evaluate it once on held-out data.

Candidate features may include:

- top-one versus top-two score margin;
- support from independent evidence sources;
- unresolved contradictions;
- stability across expansions or repeated generations;
- retrieval grounding coverage; and
- parser or generation failures.

Use explicit abstention behavior on unanswerable cases as the primary safety
signal until a held-out calibrator is available.

## 4. Better benchmarks

### 4.1 Define the role of each benchmark

| Benchmark | Recommended role | Primary endpoint |
|---|---|---|
| C-NIAH | Mechanism test for prior traps and buried evidence | Bias Trap Rate |
| CUPCase | External case and distractor robustness | Top-k accuracy; reviewed distractor capture as secondary |
| DDXPlus | External ranked differential evaluation | MRR and top-k accuracy |
| PMC | Legacy smoke fixture after label repair | No comparative performance claim |
| Unanswerable C-NIAH | Safety behavior | Fabrication rate and coverage |

C-NIAH should remain a development benchmark. Its cases, templates, and answer
logic are authored in this repository and are not independent clinical
evidence.

### 4.2 Add an immutable experiment manifest

Every run should record:

- schema version and run id;
- git commit and dirty-worktree state;
- model name, digest, temperature, seed, and generation options;
- prompt-template hashes;
- embedding model and corpus manifest hash;
- dataset id, revision, split, sampled case ids, and case hashes;
- every runtime configuration value that affects results;
- candidate budget per arm;
- completed, failed, skipped, and excluded cases; and
- environment information relevant to latency.

Write results to `data/runs/<run-id>/` and refuse to overwrite an existing run.
Store errors as result records instead of silently removing failed cases.

### 4.3 Use repeated model runs

A single local-model generation is not a stable estimate when sampling is
enabled. For each sampled case, run multiple model seeds or use deterministic
generation for the primary analysis. Report:

- mean and interval across model runs;
- case-level variance;
- rank stability; and
- the fraction of conclusions that change across seeds.

Use hierarchical or clustered resampling when several cases share the same
diagnosis, template, or counterfactual pair.

### 4.4 Predeclare endpoints and sample sizes

Before a long run, create a small run specification containing:

- primary endpoint;
- primary arm comparison;
- sample size and power assumption;
- dataset split and sampling seed;
- exclusion rules;
- statistical test; and
- secondary exploratory analyses.

This prevents endpoint selection after seeing results. Correct for multiple
comparisons when several confirmatory hypotheses are tested.

### 4.5 Make scoring independently auditable

The matcher currently uses deterministic normalization, concept aliases, an
LLM judge, and an embedding fallback. Save the match method for every candidate
and every case. For the primary result:

- use deterministic matching where possible;
- maintain a versioned diagnosis ontology or alias table;
- blind-review disagreements and a random sample of matches;
- report scorer precision and recall on an adjudicated set; and
- treat LLM-judge-only matches as a sensitivity analysis unless the judge is
  separately validated and fully recorded.

The scoring judge should not silently inherit the same model and assumptions as
the system under evaluation.

### 4.6 Validate benchmark inputs before inference

Add a mandatory validation stage that checks:

- every case has a usable diagnosis label or an explicit unanswerable label;
- no answer appears verbatim in fields that should be hidden;
- all pair ids and roles are complete;
- target diagnoses and distractors do not collapse to the same normalized
  concept;
- token lengths and needle depths match their metadata;
- external data revisions are fixed; and
- sampled case ids are unique.

Abort the run when validation fails. A warning is insufficient because a long
run can otherwise produce a polished but invalid result file.

### 4.7 Add corpus integration checks

Keep unit tests fully isolated from the developer's local ChromaDB. Add a
separate integration command that validates the actual corpus:

- unique chunk ids;
- required metadata fields;
- source and license fields;
- token-length distribution;
- embedding dimension;
- document count and source distribution; and
- a deterministic corpus manifest hash.

The current local corpus contains chunks without `evidence_level`; either
backfill the field with `scripts/repair_corpus.py` or revise the schema so that
the field is explicitly optional. The code, test, and documentation must agree.

## 5. Code organization

Refactor around stable ownership boundaries after the P0 correctness work.

Recommended extractions:

1. `apiro/application/runtime.py` for shared resource construction and
   per-run traversal creation.
2. `apiro/reasoning/synthesis.py` for final differential construction, prompt
   assembly, parsing, and retry behavior currently in `graph/expander.py`.
3. `apiro/contracts/output_parsing.py` for parsing shared by production and
   evaluation. Keep `apiro/parsing.py` as a temporary compatibility import.
4. `apiro/evaluation/benchmarks/` for importable benchmark runners, leaving
   `scripts/` as thin command-line wrappers.
5. `apiro/web/api.py`, `apiro/web/service.py`, and static assets to replace the
   single large `apiro/web/app.py` module.

Avoid moving the whole package in one commit. Use small moves with unchanged
behavior and run the offline suite after each one.

## 6. Suggested implementation sequence

### Milestone A: trustworthy execution

- isolate web traversal state;
- preserve original case narratives;
- fix AURC ties and arbitrary calibration thresholds;
- separate offline and corpus integration tests;
- correct end-to-end timing; and
- repair stale entropy and Node documentation.

Exit condition: the offline suite passes regardless of local corpus contents,
and two concurrent API runs remain isolated.

### Milestone B: measurable performance

- add stage timing, token counts, model-call counts, and retrieval diagnostics;
- persist the experiment manifest;
- introduce stable cache keys and a bounded model-call scheduler; and
- profile contradiction and entropy call volume.

Exit condition: every quality result can be paired with latency, model calls,
tokens, model identity, corpus identity, and complete case denominators.

### Milestone C: efficient reasoning

- batch or reuse entropy scoring;
- reduce contradiction candidate pairs;
- implement evidence-aware context selection; and
- tune adaptive exploration budgets.

Exit condition: median and p95 latency fall at a predeclared maximum quality
loss, measured on the same frozen case set.

### Milestone D: stronger evidence

- run compute-matched ablations;
- validate retrieval and axiom extraction independently;
- run repeated C-NIAH, DDXPlus, and CUPCase evaluations; and
- build a held-out calibration set.

Exit condition: each claimed mechanism shows an incremental benefit, results
include uncertainty and run-to-run variance, and no primary result depends on a
known-invalid historical artefact.

## Definition of done for a reportable benchmark

A benchmark result is reportable only when all of the following are true:

- input validation passes;
- the run uses live components and a frozen dataset revision;
- all arms use the same model and candidate budget unless explicitly labeled;
- the model, prompts, corpus, code, and configuration are fingerprinted;
- every requested case has a success or failure record;
- scoring methods are stored per prediction;
- stochastic variation is controlled or measured;
- the primary endpoint and comparison were declared before the run;
- confidence intervals and paired tests are reported where appropriate; and
- the result is labeled clearly as mechanism, external, exploratory, or
  confirmatory evidence.

