# GPT-6 Astra Repository Reanalysis

**Repository:** Apiro  
**Branch:** `theroid`  
**Revision reviewed:** `5991bde56fa9c74b8ef1fba1bc578b1dc514f1d8`  
**Review date:** 2026-09-05

This document records the findings from the GPT-6 Astra reanalysis of the
repository after the canonical directory refactor. It is a technical review,
not an implementation record. The corresponding implementation priorities and
acceptance criteria are developed further in `docs/IMPROVEMENT_ROADMAP.md`.

> **Implementation status:** the correctness findings in sections 1–9 were
> addressed on `feature/adversarial-benchmark-suite`. This document remains the
> audit record explaining their origin. See the roadmap's implementation update
> and `docs/IMPROVEMENTS.md` for current status. Performance recommendations
> beyond run manifests and stage timing remain open.

## Overall assessment

The tracked code was unchanged from the earlier review and remained at commit
`5991bde`. The repository has a broad offline test suite and several thoughtful
benchmarking components, but its most important remaining risks concern shared
runtime state, loss of clinical context, edge cases in safety metrics, and the
reproducibility of benchmark runs.

Directory restructuring is no longer the highest-value task. The canonical
directory commit already moved the application and parsing code into the
package. Correctness and experiment integrity should be addressed before a
larger physical reorganization.

## Correction to the earlier entropy-cache finding

The earlier repository report identified the 200-character cache key in
`EntropyEngine.hypothesis_uncertainty()` as a live source of cross-case
contamination. The cache key is lossy and should be fixed before that wrapper is
reused, but the current traversal does not call it.

`NodeExpander._batch_entropy()` calls `EntropyEngine.score_hypothesis()`
directly. A focused diagnostic confirmed that the live scoring path bypasses
`hypothesis_uncertainty()` and its cache. The earlier claim that this cache
currently invalidates live benchmark results was therefore overstated.

Recommended maintenance action:

- use a stable hash of the complete selected context, hypothesis, model digest,
  prompt version, and model options for any future confidence cache; and
- add a regression test before routing live traversal through that cache.

## Confirmed findings

### 1. Concurrent web runs share mutable traversal state

The web application constructs one global traversal and uses a thread pool with
two workers. `ApiroTraversal.run()` stores the active event callback and resets
the in-memory traversal log on the traversal instance.

Two overlapping requests can therefore interfere:

- a later request can replace the callback used by an earlier run;
- events can be delivered to the wrong request queue;
- traversal logs can contain interleaved or overwritten data;
- shared saturation and rabbit-hole detector state can leak between runs; and
- fixed case names such as `api_stream` and `api_run` overwrite log files.

Recommended fix: share only expensive immutable resources, such as the model
client and vector store. Create traversal, detector, callback, graph, and log
state per request. Assign every run a unique id and test two deliberately
interleaved requests.

### 2. CUPCase drops evidence after the first 400 characters

The CUPCase adapter emits a chief-complaint finding and a full-presentation
finding capped at 400 characters. The CUPCase benchmark reconstructs the
vignette by joining those derived findings instead of carrying the source
narrative separately.

This can remove decisive evidence late in a case before any arm sees it. The
comparison may remain superficially fair because all arms receive the same
truncated input, but it no longer measures performance on the actual case.

Recommended fix:

- retain the original narrative as an explicit case field;
- use derived findings only for graph seeding;
- send the original narrative to the bare LLM, RAG, traversal, and synthesis;
- record source and processed text hashes; and
- add a benchmark fixture whose decisive evidence appears after character 400.

### 3. Posterior scoring uses prefix truncation for long cases

The confidence prompt includes at most the first 3,000 characters of the case.
This conflicts with a benchmark specifically designed to vary the depth of
buried evidence. A hypothesis may be scored without the evidence that supports
or rules it out.

Recommended fix: replace prefix truncation with a documented context selector
that retains deterministic axioms, the chief complaint, hypothesis-relevant
sentences, abnormal and negated findings, and sampled coverage across the note.
Persist selected source spans for auditability.

### 4. AURC changes when tied predictions are reordered

A focused diagnostic used two predictions with confidence `0.5`. With labels in
one order, AURC was `0.125`; reversing only the labels produced `0.625`.

The current implementation sorts individual examples by confidence. Equal
confidence values preserve or inherit an arbitrary ordering, even though a
threshold cannot distinguish those examples. This makes AURC depend on result
file order.

Recommended fix: process equal-confidence examples as one threshold block, or
report an explicitly defined expectation under random tie-breaking. Add
permutation-invariance tests and tests for all-equal confidence.

### 5. Arbitrary calibration thresholds crash the CLI

The calibration command exposes an unrestricted `--tau` argument, but
`evaluate_arm()` calculates selective metrics only for three hard-coded
thresholds. Rendering then indexes the requested formatted threshold.

Running with `--tau 0.7` reproduced:

```text
KeyError: '0.70'
```

Recommended fix: validate that `tau` lies in `[0, 1]`, add it to the evaluated
threshold set, and test both arbitrary valid and invalid values.

### 6. The offline suite depends on the developer's local corpus

The suite was run without the pytest cache provider to avoid unrelated cache
writes. It produced:

```text
461 passed, 2 skipped, 1 failed
```

The failure occurred in `test_tc_1_2_chunk_schema_validation`. When a populated
local ChromaDB exists, the test samples that corpus instead of using its
synthetic fixture. One sampled chunk lacked the expected `evidence_level`
metadata field.

This reveals two problems:

1. A test described as offline changes behavior based on machine-local data.
2. The built corpus and the asserted metadata contract have drifted.

Recommended fix: keep unit tests hermetic and move live-corpus checks to an
explicit integration suite. Decide whether `evidence_level` is required. Then
either backfill it with the repair script or update the schema, ingestion code,
documentation, and validator to make it explicitly optional.

### 7. Reported traversal duration excludes synthesis

`ApiroTraversal.run()` calculates duration before final differential synthesis.
Synthesis includes a model call and may include a retry, so benchmark latency
underreports the actual time needed to produce an answer.

Recommended fix: calculate total duration after synthesis and log writing.
Prefer separate extraction, retrieval, expansion, entropy, contradiction,
synthesis, and total timings.

### 8. Web initialization is coupled to CLI termination behavior

The web module imports `build_components()` from the CLI. That function calls
`sys.exit(1)` for setup failures. The web module catches `Exception`, while
`SystemExit` derives from `BaseException`, so some initialization failures can
terminate module import instead of placing the service into its intended
unavailable state.

Recommended fix: extract runtime construction into an application module that
raises typed exceptions. Let the CLI convert them to exit codes and let the web
application surface them through startup health state.

### 9. Runtime documentation still describes an obsolete entropy signal

`Node.entropy_score` is documented as first-token Shannon entropy. The live
implementation uses an elicited diagnostic-breadth or posterior-confidence
heuristic mapped into an entropy-shaped range. Stale terminology makes it
harder to reason about the algorithm and encourages stronger claims than the
implementation supports.

Recommended fix: update model, function, README, and benchmark wording to use a
single precise definition for each signal. Record the configured signal in
every result manifest.

## Benchmark and evidence recommendations

The project has good metric primitives, including paired comparisons, Wilson
intervals, rank-aware scores, explicit abstention detection, and candidate
budget controls. The next benchmark improvements should focus on experimental
provenance and causal evidence for the architecture.

### Record complete run provenance

Every result should include:

- git revision and dirty status;
- model name and immutable model digest;
- generation parameters and random seed;
- prompt versions or hashes;
- corpus manifest and embedding model;
- dataset revision, split, sampled case ids, and case hashes;
- all configuration values that affect traversal;
- raw and parsed outputs;
- retrieval chunk ids and distances;
- scoring method per candidate;
- failures, retries, and exclusions; and
- stage timing, token use, and model-call counts.

Store runs under unique, immutable run directories and never overwrite an
existing completed experiment.

### Run compute-matched ablations

Comparing full Apiro only with bare LLM and standard RAG cannot establish which
part of the architecture helps. Add arms for:

- graph traversal without entropy ordering;
- graph traversal without contradiction pruning;
- graph traversal without relevance weighting;
- graph traversal without semantic merging; and
- the complete system.

Match the candidate count, model, context budget, and model-call or token budget.
Report quality beside latency and inference cost.

### Keep benchmark roles distinct

- Use C-NIAH as a mechanism and regression benchmark. It is self-authored and
  should not support a broad clinical-performance claim.
- Use DDXPlus for external ranked differential metrics such as MRR and top-k.
- Use CUPCase as an external case benchmark after preserving its full narrative.
  Treat distractor capture carefully because near-miss labels may be clinically
  related to the reference diagnosis.
- Retain PMC as a legacy smoke fixture after repairing its malformed labels.
- Use unanswerable cases for explicit fabrication and abstention behavior.

### Measure stochastic variation

If sampling remains enabled, run multiple model seeds per case. Report
run-to-run variation and cluster uncertainty by diagnosis, template, and
counterfactual pair. If deterministic generation is used for the primary run,
include repeated stochastic runs as a robustness analysis.

### Validate scoring independently

The evaluator falls back from deterministic matching to an LLM judge and then
embedding similarity. Save the method responsible for every hit. Validate the
matcher on a clinician-adjudicated sample and treat judge-only matches as a
sensitivity analysis until the judge has demonstrated suitable precision and
recall.

### Separate exploration uncertainty from answer confidence

The entropy signal controls exploration; it is not automatically a calibrated
probability that the final answer is correct. Fit any final confidence model on
a development split, lock its threshold, and evaluate it once on held-out data.
Until then, label proxy-confidence ECE and AURC results as exploratory.

## Performance recommendations

### Add stage-level instrumentation

Count model calls, retries, tokens, retrieval operations, contradiction pairs,
entropy evaluations, graph nodes, merges, and prunes. Measure median and p95
latency per stage. This provides the evidence needed to choose optimizations.

### Reduce entropy calls

Evaluate structured expansion output that includes a hypothesis confidence,
batch several hypotheses in one scoring request, cache only under full
experiment-aware keys, and avoid rescoring semantic duplicates.

### Bound contradiction comparisons

Always compare new hypotheses against deterministic anchors and ancestors, then
limit hypothesis-to-hypothesis comparisons to semantically related nodes.
Track prefilter recall and the number of LLM judge calls so speed improvements
do not silently remove useful contradiction detection.

### Tune adaptive exploration

Allocate more work when top candidates remain close, evidence changes rankings,
or important contradictions remain unresolved. Stop earlier when the ranking is
stable and retrieval repeatedly provides no grounded evidence. Select the
operating point from a measured quality-latency curve.

### Control local-model concurrency

Use one bounded scheduler around all Ollama calls and measure queue time
separately from inference time. Test throughput and tail latency at several
concurrency levels before selecting a worker count.

## Recommended implementation order

1. Isolate web request state and unique logs.
2. Preserve complete clinical narratives and add late-evidence fixtures.
3. Fix AURC ties and calibration threshold handling.
4. Separate hermetic tests from corpus integration checks and repair the corpus
   schema mismatch.
5. Correct end-to-end timing and stale entropy documentation.
6. Add run manifests and performance instrumentation.
7. Optimize entropy and contradiction calls using measured profiles.
8. Run compute-matched ablations and repeated external benchmarks.
9. Build and evaluate a held-out answer-confidence calibrator.
10. Extract runtime construction, synthesis, benchmark runners, and web service
    boundaries after the behavioral fixes are covered by tests.
