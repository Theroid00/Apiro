# What's Next

Last reviewed: 2026-09-12

Apiro is an evaluation-ready research prototype, but it does not yet have a
powered, held-out result showing that it outperforms Standard RAG or a bare LLM.
The next work should improve validity, efficiency, and rank-one quality before
adding more benchmark families or making clinical-performance claims.

## Current Snapshot

- [x] Offline test suite passes in the project virtual environment.
- [x] Per-investigation traversal state is isolated.
- [x] Model calls, tokens, queue time, and inference time are recorded.
- [x] MedEinst, MedDistractQA, and MINT-style runners are implemented.
- [x] Benchmark manifests include code and dataset provenance.
- [x] Efficient `simple`, complete `investigator`, and original `legacy` modes
      are implemented behind one application service.
- [x] Retrieval counts, reasoning-call counts, actions, and candidate histories
      are recorded for bounded modes.
- [ ] Local corpus passes schema validation.
- [ ] Ollama and the required model are available for a live run.
- [x] Bounded-mode retrieval and graph-event telemetry is complete.
- [ ] A powered held-out adversarial evaluation has been completed.
- [ ] Final-answer confidence has been fitted and validated.

## P0: Make The Evaluation Environment Valid

- [ ] Decide and document the exact meaning and allowed values of
  `evidence_level`.
- [ ] Repair or rebuild every corpus record missing `evidence_level`.
- [ ] Avoid assigning evidence levels from broad source names unless that mapping
  has a documented justification.
- [ ] Make `scripts/validate_corpus.py` read ChromaDB without loading the sentence
  transformer model or contacting Hugging Face.
- [ ] Run corpus validation and require `valid: true` before any benchmark.
- [ ] Record the validated corpus manifest hash in every live run.
- [ ] Investigate why only 106 documents are indexed despite the much larger raw
  corpus directory.
- [ ] Rebuild a representative corpus with sufficient source and domain coverage.
- [ ] Record corpus licenses, source revisions, document counts, and domain
  distributions.
- [ ] Start Ollama and verify that the configured model is installed.
- [ ] Record the exact Ollama model digest, context size, and generation options.
- [ ] Make model and embedding loading work predictably in offline mode.
- [ ] Add a clean-environment CI job for the offline test suite.
- [ ] Update stale documentation that still names the pre-merge feature branch.
- [ ] Review untracked files and either commit, ignore, archive, or remove them.

### P0 Completion Gate

- [ ] Offline tests pass in CI.
- [ ] Corpus validation passes with a stable manifest hash.
- [ ] A live smoke case completes with Ollama, ChromaDB, and the frozen model.
- [ ] The working tree and run manifest clearly identify every input artifact.

## P1: Reduce Runtime And Model Calls

### Contradiction Detection

- [ ] Count candidate pairs, lexical-filter decisions, cache hits, LLM judge
  calls, and confirmed contradictions.
- [ ] Compare each new hypothesis with deterministic anchors and direct ancestors.
- [ ] Limit hypothesis-to-hypothesis checks to a small semantic-neighbor set.
- [ ] Skip unrelated domains unless shared clinical entities require escalation.
- [ ] Deduplicate symmetric contradiction pairs with stable content hashes.
- [ ] Build a labeled contradiction fixture to measure prefilter recall.
- [ ] Reduce contradiction LLM calls by at least 60% without losing fixture recall.

### Entropy And Confidence Scoring

- [ ] Return hypothesis confidence with the expansion response using validated
  structured output.
- [ ] Batch confidence scoring for hypotheses that still need separate scoring.
- [ ] Skip scoring for hypotheses that are rejected or semantically merged.
- [ ] Cache scores by model digest, prompt version, full case hash, and normalized
  hypothesis.
- [ ] Measure rank correlation against the current scorer before replacing it.

### Exploration Budget

- [ ] Track top-diagnosis stability across expansions.
- [ ] Stop early when ranking is stable and evidence coverage is sufficient.
- [ ] Continue when new evidence changes the leading differential.
- [ ] Stop spending retrieval work after repeated ungrounded queries.
- [ ] Reserve extra budget for close scores and unresolved anchor contradictions.
- [ ] Compare fixed and adaptive budgets on the same frozen development cases.

### Performance Completion Gate

- [ ] Report median and p95 latency per case.
- [ ] Report model calls and prompt/completion tokens by purpose.
- [ ] Report quality per model call and per minute.
- [ ] Demonstrate lower cost at a predeclared maximum acceptable quality loss.

## P1: Complete Operational Telemetry

- [ ] Count retrieved chunks and chunks removed by distance filtering.
- [ ] Record retrieval cache hits and misses.
- [ ] Persist retrieved chunk IDs, sources, distances, and grounding decisions.
- [ ] Count generated, merged, pruned, failed, and expanded graph nodes.
- [ ] Record graph size, maximum depth, and frontier size over time.
- [ ] Record extraction, retrieval, generation, entropy, contradiction, synthesis,
  and log-writing durations separately.
- [ ] Store parser failures, retries, timeouts, and abstentions as explicit result
  records instead of silently dropping cases.
- [ ] Include complete requested, completed, failed, skipped, and excluded case
  denominators in every summary.

## P1: Improve Diagnostic Quality

### Rank-One Synthesis

- [ ] Build a development set of cases where the correct diagnosis appears below
  rank one.
- [ ] Separate exploration uncertainty from final-answer confidence.
- [ ] Rank final diagnoses by patient-specific support, not entropy alone.
- [ ] Reward support from independent evidence sources.
- [ ] Penalize unresolved contradictions and weak retrieval grounding.
- [ ] Include top-one versus top-two score margin and ranking stability.
- [ ] Use strict structured output with schema validation and bounded retries.
- [ ] Tune only on training or development cases.
- [ ] Freeze the synthesis prompt before evaluating unseen cases.

### Axiom Extraction

- [ ] Create a labeled clinical extraction set.
- [ ] Measure entity span precision and recall.
- [ ] Measure affirmed, negated, historical, hypothetical, and conditional
  polarity accuracy.
- [ ] Add temporal trends, medication exposure, units, reference ranges,
  coreference, and clinical shorthand cases.
- [ ] Log extraction confidence and axiom-weight-table coverage by dataset.
- [ ] Compare curated, uniform, information-content, and learned seed weights.

### Retrieval

- [ ] Create a labeled query-to-passage relevance set.
- [ ] Measure Recall@k, Precision@k, MRR, and retrieval latency.
- [ ] Compare MPNet with suitable biomedical embedding models.
- [ ] Evaluate hybrid lexical and dense retrieval.
- [ ] Evaluate lightweight reranking of retrieved passages.
- [ ] Test queries built from the active hypothesis plus selected patient facts.
- [ ] Select the retrieval configuration before the held-out diagnosis run.

## P1: Prove Which Mechanisms Help

- [ ] Run a bare-LLM baseline.
- [ ] Run a Standard RAG baseline.
- [ ] Run the graph without entropy ordering.
- [ ] Run the graph without contradiction pruning.
- [ ] Run the graph without relevance weighting.
- [ ] Run the graph without semantic merging.
- [ ] Run the complete Apiro system.
- [ ] Match model, temperature, context budget, candidate count, and compute budget
  across arms.
- [ ] Report unrestricted and compute-matched results separately.
- [ ] Use paired case-level comparisons and confidence intervals.

## P1: Freeze And Run The Main Experiment

- [ ] Predeclare the primary endpoint and primary arm comparison.
- [ ] Complete a power analysis and choose the sample size before inference.
- [ ] Freeze dataset revisions, splits, case IDs, and sampling seed.
- [ ] Freeze prompts, parser version, stopping rules, graph limits, and retrieval
  configuration.
- [ ] Freeze model digest, model options, embedding model, and corpus manifest.
- [ ] Define exclusions, retries, failures, and missing-output handling.
- [ ] Decide between deterministic generation and repeated seeded generations.
- [ ] Run small train-split pilots to find operational failures.
- [ ] Do not tune on held-out pilot or test results.
- [ ] Run the powered unseen MedEinst evaluation once.
- [ ] Run the compatible MedDistractQA evaluation.
- [ ] Run the MINT-style incremental evidence evaluation.
- [ ] Keep C-NIAH labeled as an internal mechanism benchmark.
- [ ] Keep the existing PMC set labeled as a legacy smoke fixture until its labels
  are independently repaired.

## P1: Make Scoring Auditable

- [ ] Save the matching method used for every candidate and case.
- [ ] Version the diagnosis alias and normalization tables.
- [ ] Validate matcher precision and recall on an adjudicated sample.
- [ ] Blind-review disagreements and a random sample of accepted matches.
- [ ] Treat LLM-judge-only matches as a sensitivity analysis unless independently
  validated.
- [ ] Ensure the scoring judge does not silently share the evaluated model's
  assumptions.
- [ ] Report top-1, top-3, top-5, MRR, robustness, trap, and abstention metrics only
  where each endpoint is compatible with the dataset.

## P2: Fit Confidence And Abstention

- [ ] Define final-answer confidence features using only information available
  before grading.
- [ ] Include score margin, evidence-source diversity, grounding coverage,
  unresolved contradictions, stability, and parser failures.
- [ ] Fit a calibrator on a dedicated development split.
- [ ] Select abstention thresholds on development data only.
- [ ] Evaluate ECE, Brier score, risk-coverage, AURC, coverage, and selective risk
  on a separate held-out split.
- [ ] Include explicitly unanswerable cases.
- [ ] Replace the current heuristic placeholder confidence before making safety or
  calibration claims.

## P2: Code And Repository Cleanup

- [ ] Extract final synthesis from `apiro/graph/expander.py` into a dedicated
  reasoning or synthesis module.
- [ ] Move importable benchmark implementations under `apiro/eval/benchmarks/`.
- [ ] Keep `scripts/` as thin command-line wrappers.
- [ ] Split `apiro/web/app.py` into API routes, application service, templates,
  JavaScript, and CSS assets.
- [x] Remove duplicated legacy component construction from the CLI.
- [ ] Define typed protocols for LLM, retriever, embedder, and traversal services.
- [ ] Remove or move unused dependencies to optional groups.
- [ ] Add formatting, linting, and static type checking to CI.
- [ ] Add a reproducible dependency lock strategy.
- [ ] Remove committed local-machine and generated artifacts where appropriate.

## P3: Product Hardening

This phase is needed only if Apiro is intended to become more than a research
prototype.

- [ ] Add a reproducible container or deployment environment.
- [ ] Add authentication, authorization, request limits, and input-size limits.
- [ ] Add structured audit logs without exposing patient information.
- [ ] Define data-retention and deletion behavior.
- [ ] Add health, readiness, and model-availability endpoints.
- [ ] Add cancellation, timeout, and backpressure behavior for streamed runs.
- [ ] Add monitoring for latency, failures, resource use, and model drift.
- [ ] Add browser and API integration tests.
- [ ] Perform privacy, security, clinical-safety, and human-factors reviews.
- [ ] Keep the interface clearly labeled as research software until independently
  validated for any clinical use.

## Definition Of Research-Phase Complete

- [ ] Another researcher can reproduce the environment from documented inputs.
- [ ] The corpus, model, prompts, code, configuration, and datasets are
  fingerprinted.
- [ ] The powered held-out run has complete case accounting.
- [ ] All comparison arms receive compatible inputs, output budgets, and compute
  accounting.
- [ ] Primary endpoints, confidence intervals, paired tests, and run-to-run
  variation are reported.
- [ ] Ablations show which Apiro mechanisms help, hurt, or have no measurable
  effect.
- [ ] Failure analysis includes representative errors rather than only aggregate
  scores.
- [ ] Written conclusions match the strength and limitations of the evidence.
- [ ] No clinical deployment claim is made from research-benchmark performance
  alone.
