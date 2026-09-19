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
- performs at most one contrastive retrieval and revision when the audit finds
  an evidence defect or unstable leader;
- records the initial/final audit and per-round candidate history in benchmark
  results;
- caps reported confidence below 0.70 when support, verified evidence, or
  conflict checks remain weak;
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
rerun from clean commit `1e1e8d0` with `llama3.1:8b`, seed 7, fixed decoding,
and the validated corpus. The sample is still too small for a superiority
claim.

| Benchmark and mode | First condition@1 | Distracted/trap@1 | Pair measure | False-confidence@1 (≥0.70) | Mean time | Calls | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedEinst Investigator | 40% | 0% | 0% pair resilience | 0% (0/10) | 28.7 s | 18 | 3 |
| MedEinst Simple | 20% | 0% | 0% pair resilience | 20% (2/10) | 6.7 s | 13 | 0 |
| MedDistractQA Investigator | 40% | 60% | 100% retention | 10% (1/10) | 27.2 s | 17 | 2 |
| MedDistractQA Simple | 40% | 40% | 100% retention | 20% (2/10) | 6.7 s | 13 | 0 |

Investigator gained one top-1 result on each five-pair sample, but did not
improve either paired robustness measure. It cost about 4.2 times as much per
case and was the only new engine to require structured-output fallbacks. Audit
gating removed two revisions per benchmark without changing accuracy, paired
robustness, or false-confidence on these fixed cases. The
false-confidence rate counts wrong top-1 Apiro diagnoses whose matched
hypothesis confidence was at least 0.70; baseline arms do not expose a
comparable confidence signal.

A completed Legacy MedEinst run reached 40% control, 20% trap, and 20% pair
resilience at a mean 131 seconds per Apiro case while expanding 33--82 nodes.
The five-pair Legacy MedDistractQA run was stopped before completion because of
its runtime and has no valid result.

The accepted retest artifacts are
`medeinst-20260919T141345Z-aa0cb305` and
`meddistractqa-20260919T142151Z-6ddf1466` under
`data/runs/pipeline-validation/`. Their manifests record commit `1e1e8d0`
with `dirty: false`.

## Remaining Work

1. Predeclare a larger pilot size and freeze prompts, model digest, corpus,
   seed, thresholds, and case IDs.
2. Inspect failures where the final audit still reports fragility; decide
   whether to abstain rather than return an unsupported leader.
3. Add clean diagnostic guardrails through the canonical service before using
   CUPCase or DDXPlus to compare the new engines.
4. Build the clinician-validated paired report benchmark with should-change
   controls, held-out distractor families, and a locked test split.

## Definition of Complete

This research phase is complete when another researcher can reproduce a
powered held-out run from its manifest, all arms receive compatible inputs and
answer budgets, paired uncertainty intervals and compute are reported, and the
written conclusion matches the evidence. Clinical deployment validation is
outside this phase.
