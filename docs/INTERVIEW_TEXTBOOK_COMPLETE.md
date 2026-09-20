# Complete Apiro Interview Textbook

This document describes the current `investigator` mode in the active `main`
branch. It is a bounded evidence-audit system, not the archived entropy graph
traversal engine and not clinical decision-support software.

## One-Sentence Description

Complete Apiro extracts patient facts, retrieves medical evidence, asks an LLM
for competing diagnoses, audits the candidates against the fact ledger and
evidence spans, and performs at most one targeted counterfactual revision.

## Runtime Flow

```text
input report
  -> AxiomExtractor
  -> patient fact ledger
  -> context selection
  -> ChromaDB retrieval
  -> one structured LLM differential
  -> deterministic contradiction and evidence checks
  -> candidate scoring and entropy audit
  -> optional contrastive retrieval and one revision
  -> provenance graph and InvestigationResult
```

### 1. Runtime construction

`apiro.application.runtime.build_runtime_resources()` creates the shared
resources:

- `Embedder` for the persistent ChromaDB corpus;
- `OllamaLLMClient` for local model calls;
- `AxiomExtractor` for deterministic clinical facts;
- `ModelCallScheduler` for bounded concurrent requests.

`RuntimeResources.create_service()` returns the canonical
`InvestigationService`. The service selects `simple` or `investigator` while
keeping resource wiring identical.

### 2. Fact extraction

`AxiomExtractor` combines:

- `LabParser` for measurements;
- biomedical NER for clinical entities;
- negation classification;
- `AxiomWeighter` for diagnostic importance.

The result is converted by `axioms_to_seed_nodes()` into a patient fact
ledger. These facts describe the patient. Retrieved passages describe general
medical knowledge and must not silently become patient facts.

### 3. Retrieval and generation

`select_clinical_context()` keeps the useful parts of a long report. The
embedder retrieves a bounded evidence set. The first LLM call must return a
structured differential with candidate diagnoses, supporting fact IDs,
conflicting fact IDs, evidence IDs, evidence spans, confidence, and missing
patient-specific information.

Malformed output is handled by the bounded parser and recorded as a fallback;
it is not silently treated as a successful structured answer.

### 4. Candidate audit

`InvestigatorReasoner._audit()` checks whether the leading candidates are
actually supported. It measures:

- candidate-distribution entropy;
- small margins between leading candidates;
- dependence on a single patient fact;
- missing or unverifiable evidence spans;
- contradictions between candidates and patient facts;
- unresolved patient-specific questions.

Evidence is accepted only when the cited quote exists in the retrieved
passage. The audit is an evidence-use check, not a claim that the passage is
clinically correct.

### 5. One bounded revision

If the audit identifies a meaningful defect, the investigator performs one
contrastive action. It temporarily treats the influential fact as
non-discriminating, retrieves targeted evidence, and asks for one revised
differential.

The hard limits are:

- at most two retrieval rounds;
- at most two reasoning generations;
- at most one revision;
- bounded candidate and fact counts;
- no recursive graph traversal.

Entropy alone does not automatically trigger another call. The revision must
be justified by an evidence or stability defect.

## Provenance Graph

`BeliefGraph` is deliberately small. It stores facts, diagnoses, and typed
relationships for inspection, streaming, and JSON export. It does not choose
what to expand, calculate a frontier, load a second embedding model, or
control the reasoning loop.

This separation is important: the investigator is a bounded controller with
an auditable output graph, not a search algorithm disguised as a data model.

## Why It Is More Than Normal RAG

Standard RAG retrieves passages and asks an LLM to answer. Complete Apiro adds
explicit ledgers and checks:

1. patient facts are separated from general medical evidence;
2. candidates must identify which facts they use;
3. evidence spans are verified against retrieved text;
4. contradictions and single-fact dependence are measured;
5. the leading diagnosis can be stress-tested by removing an influential fact;
6. the system records uncertainty and missing information instead of hiding it.

These mechanisms improve auditability and distractor analysis. They are not
evidence of clinical accuracy by themselves.

## Compute Profile

The expensive components are embedding retrieval and local LLM generation.
The initial path uses one generation and one retrieval. A fragile case may
use one additional retrieval and one additional generation. The scheduler
limits concurrent model requests, and all benchmark arms receive the same
answer budget.

## Evaluation

The active benchmark runners compare Bare LLM, Standard RAG, and Apiro on the
same cases:

- MedDistractQA: primary clean versus distracted diagnosis retention;
- MedEinst: paired control/trap anchoring resistance;
- MINT: local incremental-evidence behavior.

The headline claim is distractor robustness, not general medical superiority.
Small smoke runs verify wiring only. Held-out, predeclared, paired samples
are required for conclusions.

## Honest Limitations

- An evidence quote can be present without semantically supporting a diagnosis.
- The model can identify a fragile answer without recovering the correct one.
- Confidence is model-derived and requires calibration.
- Local Ollama model quality strongly affects results.
- The corpus, prompts, model digest, seed, and case IDs must be frozen for a
  meaningful comparison.

## Key Interview Answers

**Why is there still a graph?**

For provenance and visualization. It is not the reasoning controller.

**Why not let the LLM answer directly?**

Direct answers do not reliably expose which patient facts or passages drove a
diagnosis. The ledgers and audit make those dependencies inspectable.

**Why only one revision?**

More rounds increase latency, cost, and opportunities for drift. One targeted
revision is a measurable compromise.

**Is this clinically validated?**

No. It is a research prototype for evaluating distractor-resistant reasoning.
