# Apiro Project Status

Last reviewed: 2026-09-18

Active branch: `codex/complete-apiro`

## Status

Apiro is an evaluation-ready research prototype with three reasoning modes:
efficient `simple`, bounded complete `investigator`, and the original
entropy-first `legacy` traversal. The architecture and adversarial benchmark
framework are implemented, but the research claim is not validated. There is
no powered, held-out result showing that either Apiro mode outperforms Standard
RAG or a bare LLM.

## Completed

- Isolated mutable traversal state for each web request and benchmark case.
- Preserved complete clinical narratives and added evidence-aware context
  selection for long cases.
- Made answer parsing and output budgets consistent across all comparison arms.
- Added immutable run manifests with dataset revision fields, content hashes,
  and Git provenance; the MedEinst runner uses revision `354f4b5`.
- Added bounded model concurrency plus per-purpose call, token, queue, and
  inference telemetry.
- Corrected AURC tie handling and arbitrary calibration thresholds.
- Added MedEinst, diagnosis-compatible MedDistractQA, and MINT-style
  incremental evaluation runners.
- Added live corpus validation and kept offline tests independent of the local
  ChromaDB state.
- Corrected MedEinst Bias Trap Rate to the benchmark's rank-1 definition and
  added a rescore path for existing result files.
- Added the efficient structured-RAG mode with a hard one/two-pass budget.
- Added the bounded investigator mode with associative retrieval, sparse
  concept-graph propagation, candidate backtracking, and auditable action
  budgets.
- Added paired top-1 clean/distracted retention metrics, which are the primary
  endpoint for Apiro's central distractor-resistance claim.

## Latest live result

The first MedEinst smoke run used five control/trap pairs. Correct rank-1
scoring produced the following result:

| Arm | Control@1 | Trap@1 | BTR | Eligible pairs |
|---|---:|---:|---:|---:|
| Apiro | 20% | 0% | 100% | 1 |
| Standard RAG | 20% | 20% | 100% | 1 |
| Bare LLM | 20% | 0% | 100% | 1 |

The sample is too small to compare arms. Its value was operational: it exposed
1,814 model calls across ten case variants, including 1,192 contradiction
calls, and a mean all-arm latency of 116.08 seconds per case.

The local artifact under `data/runs/` was created from a dirty checkout before
the scoring correction. Keep it as a diagnostic artifact and use the rescore
command when inspecting it:

```bash
python scripts/run_medeinst_eval.py \
  --rescore-results data/runs/medeinst-20260905T134645Z-8a5e705c/results.json
```

Do not publish its original summary table.

## Remaining blockers

1. **Corpus validation:** the target corpus on the benchmark machine must pass
   `scripts/validate_corpus.py`, or its schema must be repaired and versioned.
2. **No frozen evaluation configuration:** prompts, model digests, stopping
   rules, and sample-size plan must be frozen before the unseen run.
3. **No powered result:** the paired distractor benchmark has not been run at a
   sample size capable of comparing `simple`, `investigator`, Standard RAG, and
   bare LLM.
4. **No contamination-resistant result:** the post-cutoff PMC paired set has
   not been assembled and run.
5. **No ablation evidence:** the complete mode's graph, memory, backtracking,
   and adjudication components have not individually demonstrated benefit.
6. **No fitted calibration:** abstention thresholds remain experimental until
   fitted and evaluated on separate splits.

## Next execution sequence

1. Move this branch to the corpus machine and validate the corpus.
2. Freeze prompts, models, seeds, distractor families, and the power plan.
3. Run small paired clean/distracted pilots in `simple` and `investigator` mode.
4. Execute the powered paired distractor run once for every comparison arm.
5. Run MedEinst as a focused anchoring-bias secondary analysis.
6. Run component ablations and clean-accuracy guardrails.
7. Fit and evaluate calibration on separate data.
8. Publish confidence intervals, paired tests, failure analysis, compute cost,
   and immutable manifests; then merge and tag the evaluation release.

## Definition of complete

The current phase is complete when another researcher can reproduce a powered
held-out run from its manifest, all arms receive compatible inputs and answer
budgets, the primary endpoints and uncertainty intervals are reported, and the
written conclusions match the strength of that evidence. Clinical deployment
validation is outside the scope of this phase.
