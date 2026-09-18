# Apiro Project Status

Last reviewed: 2026-09-19

Active branch: `new/complete-apiro`

## Status

Apiro is an evaluation-ready research prototype with three modes:

- `simple`: structured-RAG baseline with one optional correction;
- `investigator`: bounded patient-fact and evidence audit with one optional
  counterfactual revision;
- `legacy`: original entropy-guided runtime graph traversal.

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

## Verification

- Full offline test suite passes with two expected skips.
- The 100,000-document corpus validates with manifest SHA-256
  `462dd355c2ac7db407ef70cf518f93d93d555913d07e3561aeb5e53c222ae13b`.
- Mechanism tests cover exact-span verification, candidate-distribution
  entropy, counterfactual rank flips, one-revision stopping, and unresolved
  patient questions.

## Latest Smoke Result

A two-pair integration run was completed with `llama3.1:8b`, seed 7, fixed
decoding, and the validated corpus. This sample is too small for architecture
comparison.

| Benchmark | Apiro result | RAG result | Bare LLM result |
|---|---|---|---|
| MedEinst | control@1 50%, trap@1 0%, pair@1 0% | 0%, 50%, 0% | 50%, 0%, 0% |
| MedDistractQA | clean@1 50%, distracted@1 50%, retention 100% | same | same |

Investigator used eight reasoning calls across four MedEinst variants. All four
cases reached the one-revision limit and none passed the final audit. It used
seven reasoning calls across four MedDistractQA variants: three revised, one
stopped after the initial audit, and two passed the final audit.

Two structured responses in the MedEinst run were malformed and recovered by
the parser fallback. That is a measured limitation of the current local model
and prompt, not evidence of architectural success.

## Remaining Work

1. Predeclare a larger pilot size and freeze prompts, model digest, corpus,
   seed, thresholds, and case IDs.
2. Run identical MedEinst and MedDistractQA subsets on `simple` and
   `investigator` and compare paired outcomes plus compute.
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
