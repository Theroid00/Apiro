# Apiro Solution Analysis

Last reviewed: 2026-09-18

> Historical design analysis. Its recommended `simple` architecture is now
> implemented, and the bounded research extension is available as
> `investigator`. References to the old critical path describe `legacy` mode.

## Executive Decision

Apiro's core idea is not inherently bloated:

> Extract patient facts, retrieve medical evidence, explore a small set of
> diagnoses, reject hypotheses that conflict with the patient, and return a
> ranked differential.

The current implementation is nevertheless too complicated and compute-heavy
for the amount of evidence available. The problem is not NetworkX, Python, or
ChromaDB. The problem is that several experimental mechanisms all participate
in the same critical path before any one of them has demonstrated independent
value.

The current system combines LLM hypothesis generation, one LLM confidence call
per child, broad contradiction pairing, periodic LLM critic calls, entropy
ordering, semantic relevance weighting, saturation, rabbit-hole detection,
depth and node budgets, hand-curated axiom weights, retrieval filtering, semantic
merging, and final LLM synthesis. This makes the system expensive, difficult to
explain, and scientifically difficult to attribute.

The recommended direction is a **bounded evidence-aware reasoner** with two to
four model calls per case, a small graph used for provenance, one stopping rule,
and a narrow evaluation program. The existing implementation should remain
available as a reference until the simpler system is measured on the same cases.

## What Is Actually Large

The repository is not unusually large for an application, but it is large for a
single unvalidated research hypothesis.

| Surface | Current size |
|---|---:|
| Python package files | 57 |
| Package code | about 12,758 lines |
| Scripts | 15 Python files |
| Tests | 23 Python files |
| Package, scripts, and tests combined | about 23,001 lines |
| Configuration constants | about 40 |
| Required dependencies | 19 |

Complexity is concentrated in a few files:

| File | Lines | Responsibilities currently mixed together |
|---|---:|---|
| `apiro/graph/expander.py` | 953 | retrieval, prompt construction, generation, parsing, confidence scoring, graph mutation, domain classification, semantic merging, and final synthesis |
| `apiro/web/app.py` | 1,152 | API, runtime startup, concurrency, HTML, CSS, JavaScript, streaming, graph serialization, and presentation |
| `apiro/eval/evaluator.py` | 955 | matching, coverage, evaluation policy, normalization, and scoring fallbacks |
| `apiro/eval/metrics.py` | 949 | rank metrics, robustness metrics, calibration support, intervals, bootstrap procedures, and paired tests |
| `scripts/run_niah_eval.py` | 1,285 | dataset loading, orchestration, three evaluation arms, scoring, output, and reporting |
| `scripts/build_niah_cases.py` | 1,665 | synthetic case generation, validation, perturbation families, and CLI behavior |

Line count alone is not the defect. The defect is responsibility density: small
behavioral changes require understanding several unrelated policies in the same
module.

## Current Runtime Architecture

```text
CLI / FastAPI / benchmark runner
            |
            v
     RuntimeResources
       |      |      |
       |      |      +-- biomedical axiom extractor
       |      +--------- Ollama client and call scheduler
       +---------------- MPNet embedder and ChromaDB
            |
            v
  Extract labs, entities, polarity, and weights
            |
            v
     Seed depth-0 graph nodes
            |
            v
  Repeated traversal iteration
       1. choose frontier node
       2. rabbit-hole check
       3. retrieve evidence
       4. generate child hypotheses
       5. score every child with separate confidence calls
       6. embed and semantically merge children
       7. compare children with anchors and resolved nodes
       8. run deterministic or LLM contradiction checks
       9. periodically ask an LLM critic whether to halt
      10. separately evaluate saturation and hard budgets
            |
            v
       Final LLM synthesis
```

This path is understandable one component at a time. It is difficult to explain
as one algorithm because multiple mechanisms can reorder, suppress, or terminate
the same hypothesis.

## Why It Is Compute-Heavy

For one expansion, the current design can make:

- one hypothesis-generation call;
- up to one confidence call per generated child;
- zero to many contradiction-judge calls;
- a periodic critic call every five iterations after warm-up; and
- a final synthesis call after traversal.

Confidence calls and contradiction calls run concurrently where possible, and a
shared scheduler bounds Ollama concurrency. That improves throughput but does
not reduce the amount of inference performed.

The first MedEinst smoke run made 1,814 model calls across ten case variants,
including 1,192 contradiction calls. This is the clearest evidence that the
architecture spends too much computation adjudicating its own intermediate
state.

There is also avoidable model duplication. `Embedder` loads an MPNet model for
ChromaDB queries, while `BeliefGraph` can lazily load another process-wide MPNet
instance for relevance and semantic matching. The graph should receive and reuse
one encoder interface from the runtime.

## Essential Complexity

These parts directly support Apiro's research question and should remain:

1. **Patient facts.** A small set of explicit affirmed, negated, historical, lab,
   and vital findings makes the reasoning auditable.
2. **Evidence retrieval.** Apiro must be compared fairly with RAG and must record
   which external evidence influenced a hypothesis.
3. **A bounded set of diagnostic hypotheses.** The system needs alternatives,
   not an unconstrained stream of generated text.
4. **Patient-fact contradiction handling.** This is the strongest distinctive
   mechanism and directly addresses misleading context.
5. **A ranked differential.** The final answer must use the same candidate budget
   as every baseline.
6. **Run provenance and telemetry.** Scientific claims require model, corpus,
   prompt, dataset, cost, and failure records.

NetworkX itself should remain for now. A graph of at most 200 nodes is not a
meaningful performance bottleneck, and it provides useful traceability and UI
inspection. Replacing it with a custom graph would add risk without solving the
actual cost problem.

## Accidental Complexity

The following mechanisms are currently unproven, overlapping, duplicated, or
premature:

### Multiple stopping authorities

Traversal can stop because of exploration budget, graph budget, depth, empty
frontier, saturation, or the LLM critic. Rabbit-hole detection can also remove a
candidate from future consideration. These mechanisms operate on related signals
but have separate thresholds and failure modes.

For a primary benchmark, a fixed compute budget is easier to explain and fairer
to compare. Adaptive stopping can be a later ablation.

### LLM-derived uncertainty on every child

Generating a diagnosis and then calling the same model again to ask how confident
it is adds cost without guaranteeing calibrated or independent information. The
uncertainty score may be useful, but it should be obtained in the structured
generation response or evaluated only for ambiguous finalists.

### Broad contradiction comparison

New hypotheses are currently paired with deterministic anchors and all resolved
nodes that pass an abstraction gate. Batch execution reduces wall time, but pair
count can still grow with the graph. Most comparisons do not affect the final
answer.

### Hand-curated term weights

`data/axiom_weights.yaml` contains a small vocabulary. Unknown entities receive
a fallback weight, and quantitative labs already receive a category-based rule.
The table creates another policy to explain and validate while contributing only
a small tie-break adjustment to depth-0 priority.

### Duplicate runtime construction

The web and evaluation paths use `RuntimeResources`, but `apiro/cli.py` still
constructs the embedder, entropy engine, contradiction detector, expander,
saturation detector, rabbit-hole detector, and traversal independently. There
should be one composition root.

### Benchmark breadth before core validation

The repository supports more benchmark families than the project can currently
run at useful statistical power. Every adapter adds parsing, case-schema, scoring,
and documentation surface. The first result needs only a primary robustness set
and a clean-accuracy guardrail.

### Frontend and backend in one module

The web interface is useful, but embedding the entire HTML, CSS, JavaScript,
streaming transport, and runtime startup in one Python file makes unrelated UI
changes look like changes to the reasoning system.

### Broad core dependency set

`pandas`, `matplotlib`, `tqdm`, `requests-cache`, and `biopython` are declared as
required dependencies but have no direct imports in the current Python code.
Corpus acquisition, web serving, biomedical NER, and benchmark tooling should be
optional dependency groups rather than requirements for every installation.

## Recommended Product Definition

Apiro should be explainable in one sentence:

> Apiro extracts reliable patient facts, retrieves relevant medical evidence,
> generates a small differential, penalizes diagnoses that contradict the
> patient, and ranks the remaining diagnoses.

The graph should support that explanation. It should record facts, hypotheses,
evidence, and contradiction relationships. It should not require several
independent control systems to justify its existence.

## Simplified Target Algorithm

### Step 1: Extract a bounded fact set

- Parse objective labs and vitals deterministically.
- Detect affirmed, negated, and historical findings.
- Keep at most 10 to 12 high-confidence facts.
- Prefer measurements and explicit diagnoses before generic symptoms.
- Make transformer NER optional; retain a deterministic fallback.
- Remove term-level axiom weighting from the first simplified version.

### Step 2: Retrieve evidence once

- Query from the complete case plus selected facts.
- Keep a small top-k set after distance filtering.
- Store chunk IDs, distances, sources, and corpus hash.
- Perform another retrieval only when a refinement query is materially different.

### Step 3: Generate structured hypotheses in one call

Request three diagnoses with:

- normalized diagnosis name;
- patient-specific confidence;
- supporting fact IDs;
- conflicting fact IDs; and
- supporting evidence chunk IDs.

Validate the response against a schema. Derive uncertainty from the returned
confidence only if uncertainty-directed exploration is enabled.

### Step 4: Apply deterministic patient constraints

- Check every hypothesis against depth-0 facts.
- Use explicit negation, lab direction, laterality, and other deterministic rules
  first.
- Escalate only ambiguous, medically related pairs to one batched LLM judgment.
- Do not compare every hypothesis with every resolved graph node.

### Step 5: Perform at most one refinement round

- Select at most two uncertain but well-grounded hypotheses.
- Retrieve additional evidence only if needed.
- Ask one structured call to confirm, revise, or reject them.
- Add the results to the graph as provenance.

### Step 6: Rank deterministically, with one optional synthesis call

Use a transparent score based on:

- model confidence;
- patient-fact coverage;
- evidence grounding;
- contradiction penalty; and
- stability after refinement.

If an LLM synthesis call is retained, it may format and resolve close finalists,
but it must not silently introduce a diagnosis absent from the candidate set.

### Expected call budget

| Purpose | Calls per case |
|---|---:|
| Initial structured differential | 1 |
| Optional refinement | 0-1 |
| Batched ambiguous contradictions | 0-1 |
| Optional final synthesis | 0-1 |
| **Expected Apiro total** | **2-4** |

This is a design target, not a promised performance result. It must be compared
with the current implementation on identical development cases.

## Simplified Target Architecture

```text
                    InvestigationService
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
    FactExtractor       Retriever        ModelGateway
          |                 |                 |
          +-----------------+-----------------+
                            |
                            v
                    EvidenceReasoner
             generate -> constrain -> refine
                            |
              +-------------+-------------+
              |                           |
              v                           v
        ReasoningGraph               RankedResult
        audit/provenance             API / CLI / eval
```

Only `InvestigationService` should orchestrate an investigation. The CLI, web
API, and benchmark harness should call the same service.

Suggested ownership boundaries:

```text
apiro/
  application/
    service.py          # one investigation use case
    runtime.py          # shared resource construction
  clinical/
    facts.py            # extraction and bounded selection
  retrieval/
    service.py          # Chroma query and evidence records
  reasoning/
    search.py           # bounded one-round reasoning policy
    generation.py       # structured hypotheses
    contradiction.py    # deterministic gate + optional judge
    ranking.py          # transparent final ranking
    models.py           # facts, evidence, hypotheses, result
  infrastructure/
    ollama.py           # one model gateway
    chroma.py           # vector-store adapter
  eval/
    harness.py
    metrics.py
    benchmarks/
  web/
    api.py
    static/
```

This is a destination, not a request for a one-commit directory rewrite. Move
behavior only after the simplified path has tests and comparative results.

## What To Keep, Disable, Or Remove

| Component | Decision | Reason |
|---|---|---|
| `RuntimeResources` and model scheduler | Keep | Correct shared-resource boundary and useful telemetry |
| Axiom/fact extraction | Keep and narrow | Core to patient anchoring; cap and simplify policy |
| ChromaDB and MPNet | Keep initially | Neither is the measured bottleneck |
| NetworkX graph | Keep as provenance | Cheap at current scale and useful for inspection |
| Structured graph-controlled expansion | Replace with bounded reasoning | Current loop creates most orchestration complexity |
| Separate LLM entropy calls | Disable in simplified path | High call cost and unvalidated incremental value |
| Relevance weighting | Defer or derive from reused embeddings | Avoid a second encoder and another frontier policy |
| Broad cross-branch contradiction checking | Replace | Compare anchors, ancestors, and a tiny relevant set |
| `CriticEngine` | Feature-flag off, then ablate | Overlaps with deterministic stopping and costs calls |
| `RabbitHoleDetector` | Feature-flag off, then ablate | Fixed depth and round limits already bound exploration |
| `SaturationDetector` | Secondary mode only | Fixed budgets are easier for primary comparisons |
| Hard depth/node/call budgets | Keep | Predictable and auditable safety bounds |
| Hand-curated axiom weight table | Remove from simplified path | Sparse and unnecessary for a small bounded fact set |
| Final synthesis | Constrain or make optional | Must not override evidence and ranking invisibly |
| Streaming UI | Keep, separate assets later | Useful demonstration, not part of reasoning validity |
| Broad benchmark collection | Freeze | Use only the minimum study until the core result exists |

## Evaluation Scope After Simplification

The first evaluation should use only:

1. **Bare LLM** as the capability floor.
2. **Standard RAG** as the evidence baseline.
3. **Simplified Apiro** as the treatment.
4. **Simplified Apiro without contradiction handling** as the first mechanism
   ablation.

Use paired clean/distracted reports as the primary robustness test and
one clean free-text diagnosis set as the guardrail. Report clean accuracy,
distracted accuracy, distractor-induced accuracy drop, rank-1 pair robustness,
model calls, tokens, and latency.

Do not implement eight baseline arms, several backbones, hundreds of
clinician-reviewed families, and a full factorial study until the simpler system
shows a useful signal in a development pilot. That larger protocol remains a
valid publication plan, not the immediate engineering plan.

## Migration Plan

### Phase 0: Preserve the reference

- Tag or record the current implementation and configuration.
- Keep its tests and existing result artifacts.
- Record current calls, tokens, latency, and quality on a small frozen development
  set.

### Phase 1: Introduce a parallel simplified path

- Add `InvestigationService` as the only new orchestration entry point.
- Reuse the existing Ollama client, scheduler, extractor, embedder, and graph
  models.
- Add a `reasoning_mode = simple | investigator | legacy` configuration switch.
- Do not delete legacy traversal code yet.

### Phase 2: Remove calls before moving files

- Generate hypotheses and confidence together.
- Restrict contradiction candidates.
- Disable critic, rabbit-hole, and saturation controls in simple mode.
- Reuse the runtime embedder for graph similarity.
- Enforce the two-to-four-call case budget.

### Phase 3: Compare behavior

- Run legacy and simple modes on identical development cases.
- Compare rank-1 accuracy, robustness, call count, tokens, median latency, and p95
  latency.
- Inspect disagreements and confirm that the simple mode has not merely become
  insensitive to new evidence.

### Phase 4: Make the simple path canonical

- Switch CLI, web, and evaluation runners to `InvestigationService`.
- Keep legacy mode only for ablation and historical reproduction.
- Remove legacy components only after the comparison is documented.

### Phase 5: Reduce repository surface

- Move the frontend out of `web/app.py`.
- Make benchmark scripts thin wrappers around importable runners.
- Split large reasoning and evaluation modules by responsibility.
- Remove duplicate CLI resource construction.
- Move corpus, web, NER, and benchmark dependencies into optional groups.
- Remove truly unused dependencies.

## Acceptance Criteria

The simplification is successful when:

- the complete reasoning path can be explained accurately in under one minute;
- one service owns the production investigation workflow;
- Apiro normally uses no more than four model calls per case;
- no more than one optional refinement round is allowed;
- the graph remains small, inspectable, and tied to explicit evidence;
- contradiction work is bounded independently of graph size;
- one embedding model instance is shared by retrieval and similarity operations;
- median and p95 latency fall materially on the same hardware;
- the simple mode does not lose more than a predeclared clean-accuracy margin;
- robustness is measured against compute-visible bare-LLM and RAG baselines;
- every optional mechanism can be enabled as an ablation instead of being hidden
  in the default path; and
- documentation describes measured behavior rather than expected speedups.

## What Not To Do

- Do not replace NetworkX or ChromaDB merely because the project feels large.
  They are not the measured bottlenecks.
- Do not add LangChain, LangGraph, an agent framework, or another orchestration
  dependency. The objective is fewer control layers.
- Do not perform a whole-repository directory rewrite before simplifying runtime
  behavior.
- Do not add another LLM evaluator to decide when other LLM evaluators should run.
- Do not treat concurrent calls as reduced compute.
- Do not ingest a large new corpus without versioning, licensing, validation, and
  leakage controls.
- Do not run the full proposed publication protocol before a small development
  study shows that the simplified mechanism is promising.
- Do not delete the legacy path before producing a reproducible comparison.

## Final Assessment

Apiro is **architecturally overextended, not fundamentally overengineered**. Its
individual components are mostly reasonable. The bloat comes from making nearly
all of them mandatory in one investigation and from allowing several mechanisms
to control the same decisions.

The strongest simplification is not a cosmetic directory change. It is this:

> Turn the graph from an autonomous multi-heuristic search controller into a
> small, bounded, inspectable evidence record around two rounds of reasoning.

That preserves what makes Apiro distinct while making it faster, easier to test,
easier to explain, and much easier to evaluate scientifically.
