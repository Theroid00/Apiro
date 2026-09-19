# Apiro Architectures

Apiro has two active reasoning modes. `simple` is the structured-RAG baseline
and `investigator` is the bounded evidence-audit system. The original
entropy-guided graph traversal is preserved on `archive/legacy-main` for
historical ablations; it is not part of the active tree.

## Simple Apiro

`simple` uses a short fixed pipeline:

```text
clinical narrative
    -> bounded deterministic fact extraction
    -> one biomedical retrieval
    -> one structured differential
    -> deterministic citation, contradiction, and ranking checks
    -> optional one corrective retrieval and revision
    -> provenance graph
```

The graph records the result. It does not decide which node to explore. A
normal case uses one retrieval and one model call; a weak result may use one
additional retrieval and model call.

This mode is intentionally close to a strong RAG baseline. Its purpose is to
show whether the investigator's evidence audit adds value beyond retrieval,
structured output, and deterministic validation.

In practical terms, Simple asks the model for a ranked differential grounded
in retrieved context, then applies deterministic checks. It does not test
whether the leading diagnosis depends on one misleading fact.

## Complete Apiro Investigator

`investigator` implements a bounded evidence audit rather than runtime graph
traversal. It keeps the medical-detective objective while using the same hard
one/two-pass execution shape as Simple Apiro.

Both new engines enter through one service pipeline. The service validates the
request, builds the selected reasoner with shared clients and detectors, then
returns the common `InvestigationResult` contract. Engine-specific policy
starts inside the reasoner, keeping setup and telemetry consistent.

```text
raw case
    -> patient fact ledger
    -> medical knowledge retrieval
    -> competing candidate/evidence board
    -> deterministic evidence-use and uncertainty audit
       -> audit passes: return
       -> audit fails: one contrastive retrieval and revision
    -> provenance graph
```

### Patient fact ledger

Extracted facts have stable IDs, polarity, domain, and diagnostic weight. Only
these IDs may be cited as observations about the patient. Retrieved medical
text remains general knowledge and cannot silently become a patient fact.

Missing patient information is returned as an unresolved question. The engine
does not search the medical corpus and pretend that the answer was observed in
the patient.

### Candidate/evidence board

Each candidate records:

- model confidence;
- supporting patient fact IDs;
- conflicting patient fact IDs;
- retrieved evidence IDs;
- exact quoted spans from those retrieved passages;
- a deterministic evidence-weighted score.

Invented IDs are removed. A medical evidence quote counts only when it is an
exact span of the cited retrieved passage. Patient support contributes more to
the score than general medical knowledge, and conflicts receive an explicit
penalty.

### Uncertainty and counterfactual audit

Uncertainty is calculated over the normalized scores of the whole candidate
set. It is not the top candidate's verbalized confidence relabeled as entropy.

The audit checks:

- whether the leader cites patient facts;
- whether it has a verified medical evidence span;
- whether it conflicts with a patient fact;
- whether the top-two score margin is small;
- whether differential entropy remains high;
- whether removing the leader's most influential fact changes rank one.

The last check is a deterministic counterfactual influence test. It identifies
single-cue dependence, which is the failure mode targeted by MedEinst-style
traps.

### One contrastive investigation

When the audit finds a fragile result, the controller retrieves medical
knowledge that distinguishes the two leading candidates. The revision prompt
temporarily treats the most influential fact as non-discriminating and asks
whether independent patient evidence restores the prior leader.

There is no open-ended loop:

```text
Maximum candidates:    6
Maximum rounds:        2
Maximum retrievals:    2
Maximum model calls:   2
```

Values above two are clamped by the investigator. The engine may stop after
one pass when the initial candidate is independently supported, grounded, and
well separated.

The practical difference from Simple is the audit decision. Simple can perform
one corrective retrieval when its quality gates fail; Investigator first
measures candidate entropy, evidence dependence, and counterfactual rank
stability, then uses one targeted contrastive retrieval only when those checks
show fragility.

### Provenance graph

The returned graph connects patient facts to diagnoses and records supporting
and contradicting relations. It exists for inspection, export, and the user
interface. It is built after ranking and never triggers retrieval or model
calls.

## Legacy Apiro

`legacy` retains recursive belief-graph expansion, entropy-prioritized
frontiers, saturation, contradiction adjudication, and rabbit-hole handling.
It remains useful for reproducing historical results and testing whether graph
traversal adds value. It is not the current Complete architecture.

## Direct Comparison

| Dimension | Simple | Investigator | Legacy |
|---|---|---|---|
| Purpose | Structured-RAG baseline | Distractor-resistant evidence audit | Historical graph-search ablation |
| Model calls | 1, optionally 2 | 1, optionally 2 | Variable |
| Retrievals | 1, optionally 2 | 1, optionally 2 | Variable |
| Uncertainty | Quality gate | Candidate distribution | Frontier-node entropy |
| Counterfactual check | No | Removes the highest-impact fact | No |
| Evidence quotes | Optional IDs | Exact verified spans | Source metadata |
| Runtime graph traversal | No | No | Yes |

The five-pair validation illustrates the trade-off: Investigator averaged
about 28 seconds per case versus 6.7 seconds for Simple. Investigator's
false-confidence rate was 0% on MedEinst and 10% on MedDistractQA, compared
with 20% for Simple on both samples. Pair robustness was not better, so these
figures support an auditability and caution hypothesis, not an accuracy claim.

## Evaluation Contract

Compare the modes on identical cases, model, corpus, seed, decoding settings,
answer budget, and matcher. MedDistractQA is the primary available
clean/distracted check; MedEinst is the focused anchoring and single-cue trap
check. Report rank-1 accuracy, paired retention or Bias Trap Rate, top-3
accuracy, model calls, latency, parse failures, and the saved evidence audit.

The current public datasets are mechanism tests, not sufficient evidence for a
clinical-performance claim. The intended final benchmark still requires
clinician-validated clean/distracted reports, should-change controls, held-out
distractor families, a locked test split, and stronger compute-matched
baselines.

## Research Basis

The design takes evidence auditing and counterfactual reasoning from
[MedEinst](https://aclanthology.org/2026.acl-long.1847/), bounded
information-seeking from [MedClarify](https://arxiv.org/abs/2602.17308) and
[AMIE](https://www.nature.com/articles/s41586-025-08866-7), and quote-grounded
verification from [VeReaFine](https://aclanthology.org/2025.bionlp-share.34/).
Apiro deliberately omits their multi-agent, reinforcement-learning, and
evolving-graph machinery until a paired ablation shows that the two-pass
mechanism is insufficient.

The implementation is in `apiro/reasoning/investigator.py`; mechanism checks
are in `tests/test_investigator_reasoning.py`; the live paired runners are
`scripts/run_medeinst_eval.py` and `scripts/run_meddistractqa_eval.py`.
