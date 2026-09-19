# What's Next

## Current Systems

- `new/simplified-apiro` provides the `simple` structured-RAG baseline.
- `new/complete-apiro` provides the two-pass `investigator` evidence audit.
- The old entropy-traversal implementation is preserved on
  `archive/legacy-main` for historical comparisons.

The active tree contains only the bounded engines; `apiro/graph/` contains the
provenance data structures they share.

Investigator is no longer a recursive graph or multi-round policy engine. It
uses a patient fact ledger, candidate/evidence board, exact evidence-span
verification, candidate-distribution entropy, a deterministic counterfactual
fact-removal check, and at most one contrastive revision.

## Completed Milestone — 2026-09-19

The proposed Complete architecture is implemented. Offline tests pass, the
100,000-document corpus validates, and MedEinst plus MedDistractQA run through
the canonical three-arm harness.

The documented two-pair smoke produced:

| Benchmark and arm | Clean/control@1 | Distracted/trap@1 | Pair measure |
|---|---:|---:|---:|
| MedEinst — Investigator | 50% | 0% | 0% pair@1 |
| MedEinst — RAG | 0% | 50% | 0% pair@1 |
| MedEinst — Bare LLM | 50% | 0% | 0% pair@1 |
| MedDistractQA — Investigator | 50% | 50% | 100% retention |
| MedDistractQA — RAG | 50% | 50% | 100% retention |
| MedDistractQA — Bare LLM | 50% | 50% | 100% retention |

This is an integration result, not evidence that any arm is better. MedEinst
had only one control-correct Apiro pair and no eligible bias traps. On
MedDistractQA only one pair was solvable by each arm. Structured-output
fallbacks are stored per case in the final run.

The audit mechanism itself was observable: seven of eight case variants used
one revision, one stopped early, and one of eight passed the final audit. The
remaining cases returned a diagnosis after the hard limit while still exposing
their fragility in `traversal.evidence_audit.final`.

## Valid Live Runners

These runners use `InvestigationService` and compare Bare LLM, Standard RAG,
and the selected Apiro mode on the same narrative:

- `scripts/run_medeinst_eval.py`
- `scripts/run_meddistractqa_eval.py`
- `scripts/run_mint_eval.py` for a supplied local incremental dataset

C-NIAH, CUPCase, DDXPlus, and PMC still instantiate Legacy traversal directly.
Do not use them to compare Simple and Investigator until they are migrated to
the canonical service.

## Next Experiment

### Five-pair pipeline validation — 2026-09-19

The canonical public pipeline used the same five pairs per mode. Investigator
was retested from clean commit `1e1e8d0`; the Simple rows are the prior fixed
sample baseline.

| Benchmark and mode | Clean/control@1 | Distracted/trap@1 | Pair measure | False-confidence@1 (≥0.70) | Mean time | Calls | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedEinst Investigator | 40% | 0% | 0% pair resilience | 0% (0/10) | 28.7 s | 18 | 3 |
| MedEinst Simple | 20% | 0% | 0% pair resilience | 20% (2/10) | 6.7 s | 13 | 0 |
| MedDistractQA Investigator | 40% | 60% | 100% retention | 10% (1/10) | 27.2 s | 17 | 2 |
| MedDistractQA Simple | 40% | 40% | 100% retention | 20% (2/10) | 6.7 s | 13 | 0 |

Investigator gained one top-1 case on each tiny sample but did not improve the
paired robustness measures. It was about 4.2 times slower and produced all five
new-engine structured-output fallbacks. At a fixed 0.70 confidence threshold,
it had fewer false-confident errors than Simple in both samples, but the
denominator is only ten cases per benchmark and the threshold is provisional.
This does not establish that its extra audit pass is worth the cost. A
controller retest at commit `1e1e8d0` skipped revisions triggered only by
entropy or a small margin. On the same fixed cases it preserved all reported
accuracy, robustness, and false-confidence results while reducing calls from
20 to 18 on MedEinst and from 19 to 17 on MedDistractQA.

The accepted retest artifacts are
`medeinst-20260919T141345Z-aa0cb305` and
`meddistractqa-20260919T142151Z-6ddf1466` under
`data/runs/pipeline-validation/`. A Legacy
MedEinst comparison completed at 131 seconds per case on average; the matching
Legacy MedDistractQA run was stopped before completion and is excluded.

## Error Scan — Current Problem

The latest scan shows that Investigator's main weakness is diagnostic
interpretation, not graph traversal cost. It can identify uncertainty without
recovering the underlying diagnosis, and it can mistake a physiological
consequence for the cause the question asks for.

The clearest failure is the MedDistractQA case whose correct answer is
factitious disorder imposed on another. Investigator selected hypoglycemic
encephalopathy with confidence `0.782`, while the final audit reported no
warning. The engine recognized the low-glucose consequence but missed the
etiology. This is a high-confidence error that the current audit does not catch.

The fixed five-pair artifacts show the broader pattern:

- MedEinst ended with high differential entropy in 9/10 cases, single-fact
  dependence in 7/10, and no verified evidence span in 4/10. The audit exposes
  fragility, but the one allowed revision usually does not recover the correct
  diagnosis.
- MedDistractQA still misses somatization disorder, chronic non-bacterial
  prostatitis, and factitious-disorder cases even when the audit flags weak
  support or unstable candidates.
- Structured-output fallbacks occurred in 5/20 cases. The artifacts record the
  count, but not the raw malformed response, extracted fact ledger, or retrieved
  passages needed to explain each failure.
- Evidence validation confirms that a quoted span exists in a passage. It does
  not confirm that the passage actually supports the diagnosis.
- Candidate scoring rewards model confidence, the number of supporting facts,
  and any verified span, but has no explicit distinction between a syndrome or
  physiological consequence and its underlying etiology.

These findings change the priority order. Before a larger pilot, test the
following changes offline against the saved case outputs:

1. Add a causal-versus-manifestation check when the question asks for a cause
   or underlying diagnosis.
2. Recalibrate confidence when entropy is high or the top candidate probability
   is low; this should catch the high-confidence factitious-disorder error.
3. Validate semantic evidence support instead of checking only quote presence.
4. Persist the extracted axioms, retrieved evidence, and raw model response for
   every benchmark case.
5. Run a larger multi-seed benchmark only after these checks pass on the saved
   failures.

The five-pair results remain mechanism evidence, not a clinical-performance
claim. The branch is stable, but Investigator should not be called complete
until it either resolves these failures or reports them as explicit uncertainty.

### Larger pilot

Freeze a larger pilot before running it. Use the same case IDs, corpus hash,
model digest, decoding settings, answer budget, and scorer on both branches.
Report accuracy and compute together.

Suggested execution order:

1. Choose and record the pilot sample size before seeing results.
2. Run the same MedDistractQA pairs on `simple` and `investigator`.
3. Run the same MedEinst pairs on both modes.
4. Compare paired errors, retention/Bias Trap Rate, exact-span success, final
   audit status, model calls, tokens, and latency.
5. Review every case where the final audit remains fragile.
6. Decide from those failures whether a flagged result should be returned or
   explicitly abstained.

## Commands

```bash
git checkout new/complete-apiro
pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/validate_corpus.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
APIRO_REASONING_MODE=investigator APIRO_MODEL_SEED=7 \
  python scripts/run_medeinst_eval.py --n-pairs 2 --seed 7

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
APIRO_REASONING_MODE=investigator APIRO_MODEL_SEED=7 \
  python scripts/run_meddistractqa_eval.py --n 2 --seed 7
```

Each run directory contains `manifest.json`, `results.json`, and logs. The
evidence audit is saved per case under `traversal.evidence_audit`; parser
fallbacks are counted in `traversal.parse_fallback_count`, and per-round
candidate changes are saved in `traversal.candidate_history`.

The final clean smoke artifacts are
`medeinst-20260918T195739Z-7bfd0eec` and
`meddistractqa-20260918T195926Z-15d65de4`, both produced from commit
`3739119` with clean manifests.

## Matched Legacy Comparison — 2026-09-19

Legacy was run on the same two MedDistractQA pairs, dataset revision, model,
seed, and corpus. Its result is saved at
`data/runs/meddistractqa-20260918T202440Z-011d6423/results.json`.

| Metric | Investigator | Legacy |
|---|---:|---:|
| Clean top-1 accuracy | 50% | 0% |
| Distracted top-1 accuracy | 50% | 50% |
| Paired retention | 100% | undefined (no clean top-1 success) |
| Apiro resolution time | 15–33 s | 90–179 s |
| Runtime graph nodes | bounded provenance graph | 45–66 expanded nodes |
| Contradiction flags | deterministic candidate checks | 23–45 per case |

This is evidence about cost and runtime behavior, not a powered accuracy
comparison. Recursive graph traversal added roughly a five-fold case-time cost
here without improving the measured top-1 result.

## Final Benchmark Still Missing

The proposed primary benchmark remains a design rather than a finished final
dataset. It needs clinician-validated clean/distracted reports, should-change
pairs, multiple and held-out distractor families, post-training-cutoff cases,
leakage checks, locked development/calibration/test splits, independent
scoring validation, and compute-matched stronger baselines.

Until that exists, public benchmark results should be described as mechanism
and robustness experiments rather than proof of clinical superiority.
