# Simple Apiro Interview Textbook

This document describes the current `simple` mode in the active `main`
branch. It is Apiro's efficient structured-RAG engine, not the archived
entropy traversal system and not clinical decision-support software.

## One-Sentence Description

Simple Apiro extracts deterministic patient facts, retrieves a small evidence
set, makes one structured differential call, applies deterministic checks, and
optionally makes one corrective call when the result fails quality gates.

## Runtime Flow

```text
input report
  -> AxiomExtractor
  -> seed facts
  -> context selection
  -> ChromaDB retrieval
  -> one structured LLM differential
  -> parse, constrain, and rank
  -> optional correction
  -> provenance graph and InvestigationResult
```

### 1. Shared runtime

`build_runtime_resources()` creates one shared embedder, Ollama client, axiom
extractor, and model-call scheduler. `InvestigationService` selects the
SimpleReasoner without changing the resource contract used by Investigator.

### 2. Deterministic facts

`AxiomExtractor` runs lab parsing, biomedical entity extraction, negation
classification, and axiom weighting. `axioms_to_seed_nodes()` gives the LLM a
clean list of patient findings. If extraction returns no facts, the raw report
is kept as a bounded fallback fact rather than producing an empty answer.

### 3. Retrieval

`select_clinical_context()` trims long reports while preserving important
clinical statements. `Embedder.query()` retrieves a small number of biomedical
chunks from ChromaDB. Distance filtering removes weakly related results.

The query contains the report and the extracted findings. Retrieved text is
presented as general medical context, not as a new patient observation.

### 4. Structured generation

`SimpleReasoner` asks the model for a strict JSON differential. Each candidate
contains a diagnosis, confidence, supporting fact IDs, conflicting fact IDs,
and evidence IDs. The parser accepts only usable candidates and caps the
reported differential at the configured size.

### 5. Deterministic checks

Before the result is returned, Apiro:

- removes invalid or duplicate candidate references;
- checks candidate claims against patient facts;
- applies contradiction penalties;
- ranks candidates using confidence and verified support;
- supports explicit abstention when evidence is insufficient;
- records parsing fallbacks and timing telemetry.

These checks make the output more consistent and auditable. They do not turn a
language model into a medical authority.

### 6. Optional correction

The fast path uses one retrieval and one generation. If deterministic quality
checks identify a weak result, SimpleReasoner can retrieve targeted additional
evidence and make one corrective generation.

The correction is bounded. There is no recursive search, entropy frontier,
node expansion loop, or unbounded self-reflection.

## Provenance Graph

The result includes a small `BeliefGraph` containing seed facts, diagnosis
nodes, and support/expansion edges. The graph exists for output, UI display,
debugging, and JSON export. It does not perform reasoning.

## Why It Is More Than Bare RAG

Bare RAG usually combines a query, retrieved passages, and a free-form answer.
Simple Apiro adds a deterministic front and back end:

1. clinical facts are extracted before retrieval;
2. negated findings are represented explicitly;
3. the model must reference fact and evidence IDs;
4. candidate output is parsed into a fixed schema;
5. contradictions and unsupported candidates are penalized;
6. the system can make one controlled correction rather than looping.

The design goal is predictable, inexpensive reasoning that is easy to compare
with Bare LLM and Standard RAG.

## Compute Profile

Normal cases use:

- one fact-extraction pass;
- one embedding retrieval;
- one LLM generation;
- deterministic parsing and ranking.

Weak cases may use a second retrieval and generation. The model scheduler
limits concurrent Ollama requests. This makes Simple Apiro the efficient
baseline and gives every benchmark case a visible call budget.

## Evaluation

Use the active runners and compare all arms on identical cases:

- MedDistractQA is the primary clean/distracted benchmark;
- MedEinst tests paired control/trap anchoring behavior;
- MINT tests incremental evidence and lure handling.

Measure top-1 accuracy, paired retention or trap resilience, latency, model
calls, fallbacks, and confidence behavior. Tiny smoke samples test wiring only.

## Simple versus Complete

Simple uses one primary reasoning call and corrects only when quality checks
fail. Complete adds an explicit candidate/evidence audit, entropy and
counterfactual stability checks, missing-information questions, and one
contrastive revision. Complete is more inspectable and more expensive; it is
not automatically more accurate.

## Key Interview Answers

**Why not use the old graph traversal?**

It added state, budgets, frontier logic, and extra model/embedding work without
being required by the current bounded architecture. The graph is now only
provenance.

**How many model calls are normal?**

One generation normally, with at most one corrective generation.

**What happens when fact extraction fails?**

The report itself becomes a bounded fallback fact, so the system does not
silently synthesize from an empty ledger.

**Is Simple Apiro a validated clinical system?**

No. It is an efficient research baseline for testing structured reasoning in
reports containing distractors.
