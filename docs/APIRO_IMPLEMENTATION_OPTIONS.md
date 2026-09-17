# Apiro Implementation Options

This document defines two possible future directions for Apiro:

1. **Efficient Apiro**: a small, predictable, production-oriented system.
2. **Complete Apiro**: a more capable investigative system that combines the
   strongest ideas found in related research, while remaining bounded and
   measurable.

The second design is deliberately more complex. It should not replace the
first design unless experiments show that the additional computation produces
meaningful diagnostic gains.

## Current Position

The `codex/simplified-apiro` branch currently implements the first half of the
efficient design:

```text
Clinical narrative
    -> deterministic axiom extraction
    -> one corpus retrieval
    -> one structured LLM differential
    -> deterministic validation and ranking
    -> shallow provenance graph
```

The legacy branch still contains an entropy-first traversal that repeatedly
expands graph nodes, retrieves evidence, calls the model, checks
contradictions, detects rabbit holes, and applies saturation logic.

The key architectural decision is whether Apiro should optimize primarily for
low cost and clarity or for deeper investigative exploration.

---

## Option A: Efficient Apiro

### Goal

Provide a clinically useful, evidence-aware differential with predictable
latency, a small code surface, and a hard model-call budget.

The default case should complete in one reasoning pass. A second pass should
be allowed only when a deterministic gate shows that the first result is weak,
contradictory, or poorly grounded.

### Architecture

```text
1. Parse the narrative
        |
2. Extract bounded clinical facts
        |
3. Retrieve biomedical evidence once
        |
4. Generate a structured differential
        |
5. Validate facts, citations, conflicts, and confidence
        |
6. Decide whether the result is adequate
       / \
      yes  no
      |    |
   return  one targeted corrective retrieval and revision
```

### Component 1: Clinical fact extraction

Keep the deterministic extraction layer. It is one of Apiro's strongest
features because labs, vitals, negations, polarity, and history can be
represented explicitly before the LLM is called.

The extractor should produce a bounded set of high-value facts:

```json
{
  "id": "ax_0",
  "text": "elevated troponin",
  "domain": "lab",
  "polarity": "affirmed",
  "value": 2.1,
  "unit": "ng/mL",
  "weight": 0.95
}
```

Recommended limits:

- Keep at most 12 facts per case.
- Always retain labs and vitals when successfully parsed.
- Preserve negated findings instead of discarding them.
- Preserve the original text span or raw text for auditability.
- Do not use an LLM for basic extraction when deterministic rules are adequate.

### Component 2: Biomedical retrieval

The first retrieval should use the full presentation plus the extracted facts.
It should return a small evidence set with stable document IDs and distances.

Recommended retrieval order:

1. Dense biomedical retrieval.
2. Optional lexical retrieval such as BM25 for exact terms, abbreviations,
   lab names, and rare diseases.
3. Optional biomedical reranking.
4. Distance or relevance filtering.

MedCPT is a particularly relevant candidate for the biomedical retriever and
reranker because it was trained for PubMed search behavior and provides both a
retriever and a cross-encoder reranker. It should be evaluated against the
current `all-mpnet-base-v2` embedding model rather than adopted blindly.

Reference: [MedCPT](https://pmc.ncbi.nlm.nih.gov/articles/PMC10627406/)

The efficient implementation should not construct or traverse a large corpus
knowledge graph at query time. Chroma, or another vector store, is sufficient
for the first version.

### Component 3: One structured reasoning call

The model receives:

- the selected clinical presentation
- the deterministic facts
- the retrieved evidence
- a fixed output schema

The response should contain only:

```json
{
  "hypotheses": [
    {
      "diagnosis": "pulmonary embolism",
      "confidence": 0.72,
      "supporting_fact_ids": ["ax_1", "ax_4"],
      "conflicting_fact_ids": [],
      "evidence_ids": ["E1", "E3"]
    }
  ]
}
```

The model should compare several candidates in one call. It should not be
asked to generate an unrestricted narrative chain of thought. Apiro should
retain concise evidence-bearing fields, not hidden reasoning text.

### Component 4: Deterministic validation

After generation, ordinary code should:

- remove duplicate diagnoses
- discard invented fact and evidence IDs
- clamp invalid confidence values
- apply deterministic contradiction rules
- penalize unsupported or contradicted candidates
- preserve evidence provenance
- optionally abstain when evidence quality is inadequate

The validation layer should not silently make new model calls. This keeps the
one-call budget real rather than merely approximate.

### Component 5: Adaptive corrective pass

The first pass should produce a quality signal using inexpensive measurements:

- average retrieval relevance
- number of usable evidence chunks
- disagreement between top candidates
- number of deterministic conflicts
- confidence margin between ranks one and two
- missing high-value facts

If the result is strong, Apiro returns it immediately. If it is weak, Apiro
may make one targeted follow-up retrieval and one revision call.

```text
Normal case:       1 retrieval + 1 generation
Difficult case:    2 retrievals + 2 generations
Hard upper bound:  no additional rounds
```

This is a small version of corrective and iterative medical RAG. CRAG uses a
retrieval evaluator to trigger corrective actions, while i-MedRAG uses
follow-up medical queries to improve difficult cases.

References: [CRAG](https://arxiv.org/abs/2401.15884),
[i-MedRAG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11997844/)

### Component 6: Provenance graph

The graph should remain, but it should be a result representation rather than
the primary reasoning controller.

```text
fact -> diagnosis -> evidence
fact -> diagnosis [contradicts]
```

The graph is useful for:

- explaining why a diagnosis was ranked
- showing conflicts
- exporting an audit trail
- analyzing errors after evaluation
- supporting the user interface

It should not automatically expand nodes or trigger more LLM calls.

### Efficient Apiro pseudocode

```text
facts = extract_facts(narrative, max_facts=12)
context = select_context(narrative, facts, max_chars=12000)
evidence = retrieve_once(context, top_k=6)

hypotheses = llm_rank(
    narrative=context,
    facts=facts,
    evidence=evidence,
    output_schema=diagnostic_schema,
)

hypotheses = validate_and_rank(hypotheses, facts, evidence)

if quality_gate(hypotheses, facts, evidence) == "adequate":
    return result(hypotheses, facts, evidence)

follow_up = choose_one_missing_or_conflicting_question(hypotheses, facts)
extra_evidence = retrieve_once(follow_up, top_k=4)
revised = llm_revise(hypotheses, facts, evidence + extra_evidence)
return validate_and_rank(revised, facts, evidence + extra_evidence)
```

### Advantages

- Easy to explain to developers and clinicians.
- Predictable latency and resource usage.
- One clear primary path.
- Easy to compare against direct LLM and basic RAG baselines.
- The graph remains available for explanation without controlling runtime.
- Failures are localized to extraction, retrieval, generation, or validation.

### Limitations

- Less likely to discover an unexpected multi-hop connection.
- Depends heavily on initial retrieval quality.
- One model call may under-explore difficult cases.
- The deterministic contradiction layer cannot replace broad medical
  knowledge.

### Recommended use

This should be Apiro's default production architecture. It is the best choice
for a local system, a research prototype with limited compute, and routine
clinical narratives where the input contains enough information to rank a
differential directly.

---

## Option B: Complete Apiro

> **Implementation status:** available as `--mode investigator` on the
> `codex/complete-apiro` branch. The implementation includes associative
> general/similar-case recall, pattern-separated evidence selection, a
> persisted typed concept graph with bounded Personalized PageRank,
> graph-linked retrieval/reranking, persistent candidate branches,
> trainable action-value weights, one bounded contradiction adjudication, and
> explicit missing-information questions. All components retain the stopping
> limits below; the mode falls back to a case-local graph and default policy
> weights when optional offline artifacts have not been built.

### Goal

Preserve the original "AI medical detective" behavior while replacing the
legacy implementation with a more disciplined combination of proven ideas:

- explicit clinical facts
- graph-aware evidence retrieval
- associative case memory
- multiple competing hypotheses
- targeted iterative investigation
- uncertainty-guided resource allocation
- contradiction and evidence checks
- bounded stopping

This is not a recommendation to activate every feature for every case. It is
a design for a separate investigator mode with explicit budgets.

### Architecture overview

```text
                         +----------------------+
                         | Medical corpus       |
                         | chunks + concepts    |
                         +----------+-----------+
                                    |
                         offline graph/indexing
                                    |
+-------------+      +-------------v-------------+
| Patient     | ---> | Clinical case state       |
| narrative   |      | facts, uncertainty, goal  |
+-------------+      +-------------+-------------+
                                    |
                +-------------------+-------------------+
                |                                       |
       associative memory                       candidate generator
       and graph retrieval                      and hypothesis set
                |                                       |
                +-------------------+-------------------+
                                    |
                         investigator controller
                         selects next action
                                    |
             +----------------------+----------------------+
             |                      |                      |
        retrieve evidence      test contradiction      ask for missing fact
             |                      |                      |
             +----------------------+----------------------+
                                    |
                           update candidate scores
                                    |
                           stop or investigate again
```

### Layer 1: Clinical case state

Replace the legacy graph's large mutable runtime state with one explicit case
state object:

```text
CaseState
    case facts
    candidate diagnoses
    evidence by candidate
    conflicts by candidate
    unresolved questions
    uncertainty scores
    action history
    token and model-call budget
```

This acts like a compact working-memory buffer. It prevents the controller
from repeatedly reconstructing the whole case prompt.

### Layer 2: Pattern memory

Use two complementary stores:

1. **General medical memory**: the stable biomedical corpus.
2. **Specific case memory**: prior cases, syndromes, or diagnostic patterns.

Borrow the computational idea of hippocampal pattern separation and pattern
completion:

- separate similar but clinically distinct patterns
- retrieve a complete pattern from partial findings
- keep rare cases from being merged into common but incorrect neighbors

HippoRAG combines LLM extraction, knowledge graphs, and Personalized PageRank
for associative retrieval. Apiro could use the same principle with a much
smaller implementation: embeddings plus a sparse concept graph and bounded
PageRank or spreading activation.

Reference: [HippoRAG](https://arxiv.org/abs/2405.14831)

Do not simulate biological neurons. Implement the computational behavior with
embeddings, sparse edges, and a few deterministic scoring operations.

### Layer 3: Medical concept and evidence graph

The complete mode may use an offline graph with nodes such as:

- symptoms
- signs
- laboratory abnormalities
- imaging findings
- diseases
- medications
- risk factors
- treatment guidelines
- source documents

Edges should have typed relationships:

```text
finding -> associated_with -> disease
finding -> argues_against -> disease
disease -> supported_by -> document
disease -> differential_of -> disease
drug -> treats -> disease
```

Graph construction should happen offline or during corpus updates. It should
not repeatedly call the LLM during every patient investigation.

GraphRAG demonstrates the value of graph extraction, community detection, and
hierarchical summaries for questions that require connecting disparate parts
of a corpus. LightRAG is a more lightweight alternative that combines graph
and vector stores. Fast Think-on-Graph adds community pruning to reduce graph
reasoning cost.

References: [Microsoft GraphRAG](https://microsoft.github.io/graphrag/),
[LightRAG](https://github.com/HKUDS/LightRAG),
[Fast Think-on-Graph](https://ojs.aaai.org/index.php/AAAI/article/view/34635)

### Layer 4: Candidate generation

Generate a diverse candidate set once, for example six diagnoses. Do not
immediately collapse to one answer.

Each candidate receives:

- prior or baseline score
- clinical support score
- evidence score
- contradiction score
- novelty or rarity flag
- uncertainty score
- unresolved-question count

The candidate set is the investigative workspace. It can be represented as a
list plus relationships; it does not require every candidate to become a
NetworkX node.

### Layer 5: Investigation controller

The controller chooses the next action using an information-value score:

```text
action_value =
    expected_uncertainty_reduction
    * clinical_relevance
    * evidence_quality
    - estimated_compute_cost
```

Possible actions:

- retrieve evidence for a candidate
- retrieve evidence for a discriminating finding
- check a candidate against a contradiction rule
- ask one targeted follow-up question
- merge duplicate candidate concepts
- stop and return the current ranking

This replaces the legacy system's unconditional frontier expansion. Entropy
can remain one input to action selection, but it should not be the only
priority signal.

### Layer 6: Deliberate hypothesis search

The complete mode can borrow from several search-based reasoning systems:

- **Tree of Thoughts**: maintain multiple coherent candidate paths and allow
  backtracking.
- **Graph of Thoughts**: merge or refine related reasoning states.
- **Think-on-Graph**: interactively explore graph entities and relations.
- **Self-RAG**: assess whether retrieval is needed and whether evidence
  supports the generated result.
- **FLARE**: retrieve when predicted generation becomes uncertain.

These ideas should be implemented as a small action loop, not as a collection
of independent agents.

References: [Tree of Thoughts](https://papers.nips.cc/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf),
[Graph of Thoughts](https://ojs.aaai.org/index.php/AAAI/article/view/29720),
[Think-on-Graph](https://arxiv.org/abs/2307.07697),
[Self-RAG](https://arxiv.org/abs/2310.11511),
[FLARE](https://aclanthology.org/2023.emnlp-main.495.pdf)

### Layer 7: Medical follow-up reasoning

For difficult cases, the controller can ask the model to generate one or two
targeted medical queries, such as:

```text
Which findings distinguish pulmonary embolism from pneumonia in this case?
```

The query is sent to the medical retriever, not blindly added as another
general-purpose graph expansion. The answer is stored as evidence for the
relevant candidates.

i-MedRAG provides a medical example of iterative follow-up queries. MedRAG
uses a hierarchical diagnostic knowledge graph to distinguish diagnoses with
similar manifestations. GPT-RagAD uses graph-based disease selection followed
by LLM reranking.

References: [i-MedRAG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11997844/),
[MedRAG](https://arxiv.org/abs/2502.04413),
[GPT-RagAD](https://proceedings.mlr.press/v297/liu26a.html)

### Layer 8: Contradiction and safety control

Contradiction checks should have two levels:

1. Fast deterministic checks for polarity, antonyms, incompatible values, and
   known exclusion rules.
2. At most one model-based adjudication for a high-impact ambiguous conflict.

Contradictions should reduce confidence and trigger targeted investigation;
they should not automatically delete a diagnosis without preserving the
reason.

The complete mode should support explicit abstention when:

- retrieval quality is low
- top candidates remain nearly tied
- a high-value fact is missing
- evidence sources disagree materially
- the case is outside the supported domain

### Layer 9: Stopping rules

The complete mode must have hard limits:

```text
Maximum initial candidates:       6
Maximum retained candidates:     3
Maximum investigation rounds:    3
Maximum targeted retrievals:     6
Maximum extra model calls:       4
Maximum graph nodes per case:    50
Maximum prompt characters:       configured bound
```

Stop early when:

- the top candidate has adequate support and a sufficient margin
- no candidate has an unresolved high-value question
- the expected value of another action is below its cost
- the token or time budget is exhausted

Saturation should be defined as a measurable lack of improvement, not as a
subjective model feeling.

### Complete Apiro pseudocode

```text
state = initialize_case_state(narrative)
state.facts = extract_facts(narrative, max_facts=20)
state.candidates = generate_candidates(narrative, max_candidates=6)

while state.rounds < 3 and state.budget_remaining():
    state = retrieve_initial_or_graph_context(state)
    state = score_evidence_and_conflicts(state)

    if state.is_adequate():
        break

    action = choose_highest_value_action(state)
    if action.kind == "retrieve":
        state = retrieve_for(action.target)
    elif action.kind == "follow_up_query":
        state = answer_one_targeted_query(action.query)
    elif action.kind == "contradiction_check":
        state = check_high_impact_conflict(action.pair)
    elif action.kind == "refine":
        state = structured_model_revision(state)

    state = rerank(state)

return audited_result(state)
```

### Advantages

- Preserves more of the original detective behavior.
- Can connect evidence across multiple retrieval steps.
- Can investigate why two diagnoses are difficult to distinguish.
- Supports rare, incomplete, and contradictory cases better than one-shot RAG
  in principle.
- Keeps a richer audit trail than a plain LLM response.

### Limitations

- More model calls and higher latency.
- More opportunities for retrieval errors and prompt drift.
- Harder to benchmark fairly.
- More difficult to debug and explain.
- Graph construction and graph quality become major dependencies.
- The extra complexity may not improve diagnosis accuracy on ordinary cases.

### Recommended use

This should be an explicit `investigator` or `deep` mode, not the default. It
should be activated only for difficult cases or research experiments.

---

## Direct Comparison

| Dimension | Efficient Apiro | Complete Apiro |
|---|---|---|
| Primary purpose | Fast evidence-backed ranking | Bounded investigative exploration |
| Default model calls | 1 | 2-6, hard capped |
| Retrieval | One dense or hybrid pass | Candidate, graph, and targeted retrieval |
| Graph | Provenance output | Corpus retrieval and case investigation |
| Reasoning | Structured single pass | Iterative candidate search |
| Uncertainty | Quality gate | Action-selection signal |
| Contradictions | Deterministic by default | Deterministic plus limited adjudication |
| Best cases | Routine and moderately complex cases | Rare, incomplete, or contradictory cases |
| Main risk | Missed multi-hop connections | Complexity and compute inflation |
| Evaluation priority | Accuracy per call and latency | Accuracy gain per additional call |

## What Should Be Implemented First

The recommended sequence is:

1. Finish and benchmark the current efficient path.
2. Improve retrieval quality before adding graph complexity.
3. Add a retrieval-quality gate and one corrective pass.
4. Evaluate a biomedical reranker such as MedCPT.
5. Add a small candidate/action state object.
6. Prototype one targeted follow-up query.
7. Only then test graph-aware retrieval or associative case memory.
8. Keep the complete investigator mode behind an explicit feature flag.

Do not implement the complete architecture as one large rewrite. Each layer
must earn its place through a paired ablation study.

## Evaluation Plan

Every architecture should be compared on the same frozen cases, model, corpus,
answer budget, and parser.

### Baselines

```text
1. Direct LLM
2. Basic vector RAG
3. Structured one-pass RAG
4. Efficient Apiro
5. Efficient Apiro + one corrective pass
6. Complete Apiro investigator mode
7. Legacy Apiro traversal
```

### Metrics

- Top-1, top-3, and top-5 diagnostic accuracy.
- Mean reciprocal rank.
- Coverage and abstention quality.
- Calibration, Brier score, and expected calibration error.
- Evidence retrieval recall and precision.
- Citation or provenance faithfulness.
- Contradiction detection precision and recall.
- Latency per case.
- Prompt and completion tokens.
- Number of retrievals.
- Number of LLM calls.
- Failure and timeout rate.

MIRAGE is a useful reference for medical RAG evaluation because it compares
corpora, retrievers, and language models across thousands of medical questions.
It is not, by itself, a complete differential-diagnosis evaluation, so Apiro
should combine it with clinical vignette and differential-diagnosis datasets.

Reference: [MIRAGE](https://aclanthology.org/2024.findings-acl.372/)

The main result should be reported as a quality-cost curve:

```text
diagnostic quality gained per additional model call
diagnostic quality gained per additional second
diagnostic quality gained per additional 1,000 tokens
```

If the complete mode is only slightly more accurate but several times more
expensive, the efficient mode should remain the default.

## Plain-Language Explanation

### Efficient Apiro

Efficient Apiro is like a careful doctor who does one well-organized review:

1. Identify the important facts.
2. Look up the most relevant medical information.
3. Compare several possible diagnoses.
4. Check the answer for contradictions and unsupported claims.
5. Return the result with an explanation of which facts and sources mattered.

It normally does this once. Only when the result is clearly weak does it make
one more focused investigation.

This version is easier to understand, faster to run, and less likely to waste
compute. It is essentially structured medical RAG with strong validation and
provenance.

### Complete Apiro

Complete Apiro is like a doctor who keeps several possibilities open and
actively investigates the differences between them:

1. Build a detailed picture of the case.
2. Recall similar medical patterns.
3. Keep several diagnoses alive.
4. Search for evidence specific to each important possibility.
5. Ask what information would distinguish the candidates.
6. Check contradictions and revise the ranking.
7. Stop when more investigation is unlikely to help.

This is closer to the original AI medical detective concept. It can be more
complete, but it costs more, has more failure modes, and is harder to prove
better. Its value must come from difficult cases where the efficient path
actually misses important connections.

## Final Recommendation

Apiro should have two modes:

```text
simple       = efficient, default, one-pass structured medical RAG
investigator = complete, bounded, opt-in investigative reasoning
```

The efficient mode should remain the product foundation. The complete mode
should be developed experimentally and activated only when measurable signals
justify its cost. This preserves Apiro's identity without forcing every case
through the most complicated architecture.

## Further Reading

- [Kohonen, The Self-Organizing Map](https://graphics.stanford.edu/courses/cs233-21-spring/ReferencedPapers/SOM.pdf)
- [O'Reilly and McClelland, Pattern Separation and Completion](https://stanford.edu/~jlmcc/papers/OReillyMcC94.pdf)
- [AMIE, Towards Conversational Diagnostic AI](https://www.nature.com/articles/s41586-025-08866-7)
- [MedRAG](https://arxiv.org/abs/2502.04413)
- [MedGraphRAG](https://arxiv.org/abs/2408.04187)
- [MedCPT](https://github.com/ncbi/MedCPT)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [Corrective RAG](https://arxiv.org/abs/2401.15884)
- [Active Retrieval Augmented Generation](https://aclanthology.org/2023.emnlp-main.495.pdf)
- [i-MedRAG](https://arxiv.org/abs/2408.00727)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [Graph of Thoughts](https://arxiv.org/abs/2308.09687)
- [Think-on-Graph 2.0](https://arxiv.org/abs/2407.10805)
