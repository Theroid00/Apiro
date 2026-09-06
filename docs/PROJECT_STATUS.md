# Apiro Project Status

Last reviewed: 2026-09-06

Active branch: `feature/adversarial-benchmark-suite`

## Status

Apiro is an evaluation-ready research prototype. The implementation work for
the adversarial benchmark phase is substantially complete, but the research
claim is not validated. There is no powered, held-out result showing that
Apiro outperforms Standard RAG or a bare LLM.

## Completed

- Isolated mutable traversal state for each web request and benchmark case.
- Preserved complete clinical narratives and added evidence-aware context
  selection for long cases.
- Made answer parsing and output budgets consistent across all comparison arms.
- Added immutable run manifests, dataset revisions, content hashes, and Git
  provenance.
- Added bounded model concurrency plus per-purpose call, token, queue, and
  inference telemetry.
- Corrected AURC tie handling and arbitrary calibration thresholds.
- Added MedEinst, diagnosis-compatible MedDistractQA, and MINT-style
  incremental evaluation runners.
- Added live corpus validation and kept offline tests independent of the local
  ChromaDB state.
- Corrected MedEinst Bias Trap Rate to the benchmark's rank-1 definition and
  added a rescore path for existing result files.

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

1. **Runtime cost:** contradiction comparisons dominate model calls. Candidate
   pairing must be reduced and deduplicated before a powered run.
2. **Missing operational counters:** retrieval and graph-event counts are not
   yet complete, so compute-normalized mechanism analysis is incomplete.
3. **Rank-1 quality:** correct diagnoses frequently appear below rank one.
   Synthesis and ranking need a train-split improvement pass.
4. **Corpus validation:** the target local corpus must pass
   `scripts/validate_corpus.py`, or its schema must be repaired and versioned.
5. **No frozen evaluation configuration:** prompts, model digests, stopping
   rules, and sample-size plan must be frozen before the unseen run.
6. **No powered result:** MedEinst, MedDistractQA, and MINT-style evaluations
   have not yet been run at a sample size capable of supporting comparative
   conclusions.
7. **No fitted calibration:** abstention thresholds remain experimental until
   fitted and evaluated on separate splits.

## Next execution sequence

1. Optimize contradiction pairing and add retrieval/graph telemetry.
2. Improve rank-1 synthesis on training cases only.
3. Validate and freeze the corpus, prompts, models, seeds, and power plan.
4. Run small train-split pilots to catch operational failures.
5. Execute the powered unseen MedEinst run once.
6. Run the compatible MedDistractQA and MINT-style stages.
7. Fit and evaluate calibration on separate data.
8. Publish confidence intervals, paired tests, failure analysis, compute cost,
   and immutable manifests; then merge and tag the evaluation release.

## Definition of complete

The current phase is complete when another researcher can reproduce a powered
held-out run from its manifest, all arms receive compatible inputs and answer
budgets, the primary endpoints and uncertainty intervals are reported, and the
written conclusions match the strength of that evidence. Clinical deployment
validation is outside the scope of this phase.
