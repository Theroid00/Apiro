# What's Next

This document records the current evaluation position and the steps needed to
test the two new Apiro systems on another computer.

## Current systems

The branches are:

- `new/simplified-apiro`: the efficient bounded system. Its modes are
  `simple` and the original `legacy` mode.
- `new/complete-apiro`: the more complete bounded investigator. Its modes are
  `investigator` and `legacy`.

The efficient system normally uses one retrieval and one structured generation
call, with at most one targeted corrective pass. The complete investigator adds
bounded candidate exploration, backtracking, associative retrieval, a sparse
concept graph, graph-linked query expansion, policy-guided actions, and bounded
contradiction adjudication.

## Completed smoke comparison — 2026-09-18

The documented two-pair smoke test now passes on both branches with the same
case IDs, corpus, `llama3.1:8b` model, seed 7, and decoding settings. Both
manifests report clean Git state. The benchmark code revisions were `48dfdd7`
for Simplified and `b47b587` for Complete.

| Benchmark and engine | Clean/control@1 | Distracted/trap@1 | Pair@1 | Apiro calls | All-arm wall time |
|---|---:|---:|---:|---:|---:|
| MedEinst — Simplified | 50% | 0% | 0% | 4 | 22.4 s |
| MedEinst — Complete | 0% | 50% | 0% | 10 | 92.1 s |
| MedDistractQA — Simplified | 50% | 50% | 100% retention | 6 | 32.2 s |
| MedDistractQA — Complete | 50% | 50% | 100% retention | 6 | 63.5 s |

The sample is an integration check, not evidence that either engine is more
accurate. Complete used more compute without improving this tiny sample. The
run also verified native structured generation without parser fallback, stable
MedDistractQA case IDs, pinned dataset revisions and decoding configuration,
and a valid 100,000-document corpus with manifest hash `462dd355c2ac7db4`.

## What is already compared

`apiro/eval/live.py` evaluates three arms on the same narrative:

1. Bare LLM
2. Standard RAG
3. Apiro using the selected reasoning mode

Therefore both new systems can be compared with bare LLM and standard RAG on
identical cases. The comparison is paired at the case level, and the existing
metrics include paired accuracy and robustness statistics.

The two Apiro systems are not currently executed as two Apiro arms in one live
run. Run the same benchmark twice, once per branch, and compare the saved
results. RAG and bare-LLM outputs are regenerated for each run, so keep the
model, corpus, prompts, seed, dataset revision, and decoding settings fixed.

## What is implemented now

The runners that currently route Apiro through `InvestigationService` are:

- MedEinst control/trap pairs
- MedDistractQA clean/distracted pairs
- optional MINT-style incremental cases

The C-NIAH/NIAH, CUPCase, DDXPlus, and PMC runners still instantiate the legacy
traversal directly. Their historical results remain useful, but they do not
measure `simple` or `investigator` until migrated to the service. MedEinst and
MedDistractQA include bare LLM, RAG, and Apiro arms and are the current valid
side-by-side runners for both new engines.

## What is not finished

The newer proposed benchmark is currently a design specification, not yet a
fully assembled final dataset. It still needs:

- clinician-validated clean/distracted report pairs;
- should-change pairs, proving that the system updates when evidence really
  changes the diagnosis;
- multiple distractor families and held-out distractor families;
- post-training-cutoff cases and leakage checks;
- a locked development, calibration, and test split;
- blinded or independently validated scoring;
- compute-matched stronger baselines such as self-consistency RAG and
  iterative RAG.

The existing datasets should therefore be treated as historical and diagnostic
benchmarks, not as the sole evidence for a final superiority claim.

## Recommended benchmark decision

The proposed benchmark is approved as the final direction because it tests
Apiro's actual claim: preserving the correct diagnosis when plausible
distractors are added, without simply ignoring new evidence.

The final primary endpoint should be distractor-induced accuracy drop:

```text
DIAD = clean top-1 accuracy - distracted top-1 accuracy
```

Compare Apiro against the strongest compute-matched baseline. Also report clean
accuracy, distracted accuracy, paired retention, flip-to-distractor rate,
should-change update rate, parse failures, latency, model calls, and token cost.

MedEinst should remain a focused secondary anchoring benchmark. Clean datasets
such as CUPCase or MedCaseReasoning should remain guardrails against Apiro
losing ordinary diagnostic accuracy.

## Migration and test checklist

On the test computer:

```bash
git clone https://github.com/Theroid00/Apiro.git
cd Apiro
git fetch origin
```

Install the existing project dependencies and make sure Ollama, the configured
model, and the Apiro corpus are available. The corpus must be copied or rebuilt
on that computer; benchmark code alone is not enough for a real run.

Run the offline checks first:

```bash
git checkout new/simplified-apiro
pytest -q
python scripts/validate_corpus.py
```

Run a small benchmark smoke test:

```bash
APIRO_REASONING_MODE=simple \
  python scripts/run_medeinst_eval.py --n-pairs 2 --describe-only

APIRO_REASONING_MODE=simple \
  python scripts/run_meddistractqa_eval.py --n 2 --describe-only
```

Remove `--describe-only` to perform model calls. The resulting run directory
contains `manifest.json`, `results.json`, and logs. Keep these files for later
comparison.

Test the complete system on its own branch:

```bash
git checkout new/complete-apiro
pytest -q
python scripts/validate_corpus.py

APIRO_REASONING_MODE=investigator \
  python scripts/run_medeinst_eval.py --n-pairs 2

APIRO_REASONING_MODE=investigator \
  python scripts/run_meddistractqa_eval.py --n 2
```

Use larger, predeclared sample sizes only after the smoke run succeeds. The
complete investigator is expected to be substantially more expensive, so
record model calls, tokens, latency, and failures rather than comparing
accuracy alone.

## Minimum fair comparison

For both branches, use:

- the same dataset revision and case IDs;
- the same random seed;
- the same model and decoding settings;
- the same corpus snapshot;
- the same number of requested diagnoses;
- the same clean/distracted or control/trap pairs;
- the same scoring matcher.

Compare Apiro against `rag` and `bare_llm` from the same saved run, then
compare the efficient and complete Apiro result files across runs. Do not
compare a smoke run from one branch with a powered run from the other.

## Execution order

1. Run offline tests and corpus validation on both branches.
2. Run two-case or five-pair smoke evaluations.
3. Confirm that every `results.json` contains all three arms and telemetry.
4. Run the same powered MedEinst and MedDistractQA subsets on both branches.
5. Report accuracy, paired robustness, compute, and failure cases together.
6. Build and validate the new clinician-reviewed primary benchmark.
7. Run the locked final benchmark once, with no further prompt or threshold
   tuning on the test set.

## Current conclusion

The two new systems have passed the reproducible side-by-side smoke milestone.
They are not yet supported by the final proposed benchmark or a powered
held-out result. The next meaningful milestone is to predeclare a larger pilot,
then construct and validate the new paired distractor-report test set.
