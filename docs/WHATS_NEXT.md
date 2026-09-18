# What's Next

## Current Systems

- `new/simplified-apiro` provides the `simple` structured-RAG baseline.
- `new/complete-apiro` provides the two-pass `investigator` evidence audit.
- Both retain `legacy` for historical entropy-graph experiments.

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
fallbacks are counted in `traversal.parse_fallback_count`.

## Final Benchmark Still Missing

The proposed primary benchmark remains a design rather than a finished final
dataset. It needs clinician-validated clean/distracted reports, should-change
pairs, multiple and held-out distractor families, post-training-cutoff cases,
leakage checks, locked development/calibration/test splits, independent
scoring validation, and compute-matched stronger baselines.

Until that exists, public benchmark results should be described as mechanism
and robustness experiments rather than proof of clinical superiority.
