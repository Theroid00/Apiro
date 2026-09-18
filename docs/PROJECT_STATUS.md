# Apiro Project Status

Last reviewed: 2026-09-19

Active branch: `new/complete-apiro`

## Status

Apiro is an evaluation-ready research prototype with three modes:

- `simple`: structured-RAG baseline with one optional correction;
- `investigator`: bounded patient-fact and evidence audit with one optional
  counterfactual revision;
- `legacy`: original entropy-guided runtime graph traversal.

The legacy implementation is physically isolated in `apiro/legacy/`. The
`apiro/graph/` package now contains only shared graph data structures used by
the current engines.

The new Complete architecture is implemented and tested. The research claim is
not validated: no powered held-out result shows that Investigator outperforms
Standard RAG, Bare LLM, or Simple Apiro.

## Complete Investigator Contract

Investigator now:

- separates patient fact IDs from retrieved general medical knowledge;
- keeps a scored board of competing diagnoses;
- accepts medical evidence only with an exact quote from the cited passage;
- computes entropy over the candidate-score distribution;
- removes the leading diagnosis's most influential fact to test rank stability;
- performs at most one contrastive retrieval and revision;
- records the initial and final audit in benchmark results;
- builds a shallow graph only as a provenance output.

The hard limit is two retrievals and two reasoning calls. The unused concept
graph, associative-memory, and trainable action-policy stack was removed.

The application service uses one factory for `simple` and `investigator`, so
shared dependency wiring is centralized while each reasoner owns its policy.

## Verification

- Full offline test suite passes with two expected skips.
- The 100,000-document corpus validates with manifest SHA-256
  `462dd355c2ac7db407ef70cf518f93d93d555913d07e3561aeb5e53c222ae13b`.
- Mechanism tests cover exact-span verification, candidate-distribution
  entropy, counterfactual rank flips, one-revision stopping, and unresolved
  patient questions.

## Five-Pair Pipeline Validation

The public `RuntimeResources -> InvestigationService -> reasoner` pipeline was
run from clean commit `ab47b5c` with `llama3.1:8b`, seed 7, fixed decoding, and
the validated corpus. The sample is still too small for a superiority claim.

| Benchmark and mode | First condition@1 | Distracted/trap@1 | Pair measure | Mean Apiro time | Calls | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| MedEinst Investigator | 40% | 0% | 0% pair resilience | 29.6 s | 20 | 4 |
| MedEinst Simple | 20% | 0% | 0% pair resilience | 6.7 s | 13 | 0 |
| MedDistractQA Investigator | 40% | 60% | 100% retention | 30.1 s | 19 | 2 |
| MedDistractQA Simple | 40% | 40% | 100% retention | 6.7 s | 13 | 0 |

Investigator gained one top-1 result on each five-pair sample, but did not
improve either paired robustness measure. It cost about 4.5 times as much per
case and was the only new engine to require structured-output fallbacks.

A completed Legacy MedEinst run reached 40% control, 20% trap, and 20% pair
resilience at a mean 131 seconds per Apiro case while expanding 33--82 nodes.
The five-pair Legacy MedDistractQA run was stopped before completion because of
its runtime and has no valid result.

The five completed artifacts are under `data/runs/pipeline-validation/`.
Their manifests record commit `ab47b5c` with `dirty: false`.

## Remaining Work

1. Predeclare a larger pilot size and freeze prompts, model digest, corpus,
   seed, thresholds, and case IDs.
2. Inspect the five-pair disagreements and structured-output fallbacks before
   spending compute on the larger pilot.
3. Inspect failures where the final audit still reports fragility; decide
   whether to abstain rather than return an unsupported leader.
4. Add clean diagnostic guardrails through the canonical service before using
   CUPCase or DDXPlus to compare the new engines.
5. Build the clinician-validated paired report benchmark with should-change
   controls, held-out distractor families, and a locked test split.

## Definition of Complete

This research phase is complete when another researcher can reproduce a
powered held-out run from its manifest, all arms receive compatible inputs and
answer budgets, paired uncertainty intervals and compute are reported, and the
written conclusion matches the evidence. Clinical deployment validation is
outside this phase.
