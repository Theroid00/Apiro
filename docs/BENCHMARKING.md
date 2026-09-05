# Benchmarking Apiro

What to run, in what order, and what to expect. Written for a machine with
Ollama and a built ChromaDB corpus; the offline step needs neither.

---

## TL;DR — one command

```bash
./run_eval.sh --quick      # ~minutes: proves the whole pipeline works
./run_eval.sh              # the real run: hours
```

That is everything — preflight, tests, dataset downloads, case generation,
adversarial and general benchmarks, calibration. Stages run in dependency order, each logs to
`data/logs/<stage>.log`, and the run stops at the first failure.

## Current benchmark order

1. **MedEinst** is the primary external mechanism benchmark. Its paired control
   and trap cases directly measure retention of the prior diagnosis after
   discriminative evidence flips the correct answer. Run
   `python scripts/run_medeinst_eval.py --n-pairs 60`.
2. **MedDistractQA** measures invariance to irrelevant clinical-looking text.
   Apiro evaluates only `Patient Care: Diagnosis` rows and reports matched-pair
   retention. Run `python scripts/run_meddistractqa_eval.py --n 100`.
3. **C-NIAH** remains an internal mechanism-development benchmark. Generate it
   with `--counterfactual`; the currently committed 120-case file is not
   counterfactual and does not exercise Bias Trap Rate.
4. **DDXPlus and CUPCase** provide exploratory external accuracy. CUPCase's
   distractors are frequently near-equivalent to the answer and must not be
   described as adversarial trap evidence.
5. **MINT-style evaluation** is optional because the paper currently links no
   official public data repository. Provide `MINT_DATASET=/path/to/file.json`.

New adversarial runs write `manifest.json`, `results.json`, and isolated logs to
`data/runs/<run-id>/`; creation fails if the directory already exists. Before a
live run, execute `python scripts/validate_corpus.py` so corpus drift is an
explicit integration failure rather than a machine-dependent unit-test result.

Each live adversarial case stores `model_telemetry` for the shared runtime:
calls, retries, failures, timeouts, prompt and completion tokens, queue time,
and inference time. The scheduler limit is recorded with those counters and can
be set with `APIRO_MAX_MODEL_CONCURRENCY` (default `2`). Ollama token counts are
used directly; they are not estimated from text length.

**Do the `--quick` run first on a new machine.** It exercises every stage at a
tiny N, so a broken Ollama, an empty corpus or a failed download surfaces in
minutes instead of after a multi-hour run. Its numbers are meaningless — the
script says so when it finishes.

```bash
./run_eval.sh --dry-run    # print the plan, execute nothing
./run_eval.sh --help
./run_eval.sh fetch niah   # single stages; order is enforced, not taken from argv
```

| Stage | What it does |
|---|---|
| `preflight` | python, imports, Ollama + model, corpus, **evaluator scoring probe** |
| `test` | offline suite — no Ollama, no ChromaDB, no downloads |
| `fetch` | download and verify the supported public datasets |
| `generate` | build the C-NIAH counterfactual case set |
| `niah` | bias trap rate, abstention, distractor selection |
| `medeinst` | paired Einstellung-effect controls and traps |
| `meddistract` | clean/distracted diagnosis-only MedQA pairs |
| `ddxplus` | external, ranked reference differential |
| `cupcase` | external, curated per-case distractors |
| `mint` | incremental evidence evaluation when `MINT_DATASET` is set |
| `calibration` | ECE / Brier / risk-coverage |

Size knobs are environment variables:

```bash
NIAH_PAIRS=80 MEDEINST_PAIRS=60 MEDDISTRACT_N=100 \
DDXPLUS_N=120 CUPCASE_N=120 MINT_DATASET=/path/to/mint.json SEED=11 ./run_eval.sh
```

### The preflight probe worth knowing about

Preflight fails the run if the evaluator cannot score `SAH`, `DKA` or
`Hyperkalaemia`. That is not a style check: a checkout predating the synonym
and spelling fix marks those correct answers wrong, depressing **every** arm by
an artifact large enough to swamp the effect under test. Five seconds of
checking beats discovering it in a results file.

**Before trusting any number from step 2, read
[Are the current test cases appropriate?](#are-the-current-test-cases-appropriate-partly)** —
it records one defect that was silently depressing every arm (now fixed) and
two limitations that are not fixable by editing the generator.

Budget roughly 2–4 minutes per case per arm on a local 8B model. The default
sizes (40 counterfactual pairs = 90 C-NIAH cases, plus 60 DDXPlus and 60
CUPCase) are a multi-hour run.

---

## Choose the endpoint before choosing N

### The short version

```bash
python scripts/build_niah_cases.py --counterfactual --num-cases 40 --seed 7
python scripts/run_niah_eval.py --cases data/niah_cases.json --real --out data/niah_eval_results.json
```

That is the run to lead with. It emits 40 counterfactual (control, trap) pairs
plus 10 unanswerable cases, and produces the two endpoints that can actually
falsify this architecture's claims.

### Do these benchmarks ship their own data? Mostly no.

Short answer: **the designs and metrics are reusable, the datasets largely are
not — but the source data underneath them is.**

| Benchmark | Data obtainable? | Metric reusable? | What we do |
|---|---|---|---|
| [MedEinst](https://arxiv.org/abs/2601.06636) | Not locatable. Jan 2026 preprint; no public repo or HF release found as of Aug 2026. Built **from DDXPlus**. | Yes — Bias Trap Rate is fully specified | Reimplemented the design; see the DDXPlus route below |
| [MedAbstain](https://arxiv.org/abs/2601.12471) | Not locatable | Yes — context-omission + explicit abstention option | Reimplemented as `--unanswerable-fraction` |
| [DyReMe](https://arxiv.org/html/2510.09275) | Generates dynamically by design | Principle only | Reflected in generating cases per run |
| [CUPCase](https://huggingface.co/datasets/ofir408/CupCase) | **Yes** — public HF dataset, 3,562 cases, 3 curated distractors each | Distractor-selection rate | **Already wired**: `scripts/run_cupcase_eval.py` |
| [DDXPlus](https://huggingface.co/datasets/aai530-group6/ddxplus) | **Yes** — CC-BY, 1.3M patients, 49 pathologies, structured symptoms + **ground-truth differential** | Top-k, MRR, differential overlap | **Wired**: `scripts/run_ddxplus_eval.py` |

Both external datasets are now wired, and `./run_eval.sh fetch` downloads and
schema-checks them. DDXPlus matters most, because:

- It is the substrate MedEinst was built from, so counterfactual traps
  constructed on it are the reference design rather than an imitation of it.
- Its symptoms and antecedents are **structured**, so control/trap pairs can be
  constructed programmatically by swapping discriminative features, instead of
  from a hand-written bank.
- It ships a ground-truth **differential** (a ranked list), not a single label,
  which is what makes top-k and MRR meaningful.
- CC-BY. No access barrier.

Its evidences arrive as opaque codes (`E_53`, `E_54_@_V_179`) and its
pathologies as French names, so `apiro/corpus/ddxplus_adapter.py` resolves both
through `release_evidences.json` / `release_conditions.json`. Mirrors disagree
on those dictionaries' internal key names, so the adapter tries several
spellings — but if it resolves nothing it **raises** rather than handing back
vignettes containing only demographics. Check what you actually downloaded
with:

```bash
python scripts/fetch_datasets.py --verify-only
```

If the schema differs from what the adapter expects, that command says so
explicitly. Report the printed schema rather than working around it.

---

## Are the current test cases appropriate? Partly.

Audited rather than assumed. One blocking defect found and fixed; two
limitations remain and cannot be fixed by editing the generator.

### Fixed: the evaluator could not score its own answers

When `NEEDLE_BANK` grew from 6 diagnoses to 20, only **2 of the 20** had a
clinical synonym group. **22 of 36 clinically correct paraphrases were graded
as misses** — every common abbreviation (DKA, SAH, GCA, GBS), and every
British spelling. That last one is self-inflicted: the needle bank is written
in British English, so a model primed by the note's own spelling produced
"haemorrhage" and was marked wrong.

Recall is now 38/38, with 12/12 confusable pairs kept distinct — pinned by
`tests/test_evaluator_coverage.py`. Separation is the load-bearing half: if
"acute appendicitis" matched "acute mesenteric ischemia", every counterfactual
trap would score as correct whichever way the model answered.

**Do not run the benchmark on a checkout without this fix.** Every arm is
depressed by it, and the artifact is large enough to swamp the effect.

### Remains: the haystacks are boilerplate, not clinical prose

Measured on generated cases:

| Note size | Sentences | Distinct | Repeated | Needle density |
|---|:---:|:---:|:---:|:---:|
| 2,000 tok | 133 | 33 | 98% | 0.8% |
| 8,000 tok | 508 | 33 | 99% | 0.2% |

`HAY_SENTENCES` is a pool of 30 filler sentences, cycled. In an 8k note, 99% of
sentences are verbatim repeats and the needle is **the only sentence that
isn't**. The task is closer to "find the line that breaks the pattern" than to
"find the decisive finding among hundreds of plausible clinical statements".
Expect absolute accuracies to be optimistic, and the long-context result in
particular to overstate real performance.

### Remains: still a self-authored exam

20 diagnoses, hand-written needles, hand-written confusable pairs. The
counterfactual design fixed the worst problem — the old cases could be passed
by ignoring the vignette — but the cases are still ours. A result here is
evidence that *the mechanism fires*, not evidence about clinical performance.

### So what is it good for?

| Question | Suitable? |
|---|---|
| Does contradiction soft-pruning actually fire? | **Yes** — that is what it is built for |
| Does discriminative evidence override the prior? | **Yes** — the counterfactual traps test exactly this |
| Does the engine fabricate when evidence is absent? | **Yes** — unanswerable cases test this directly |
| Is Apiro more accurate than RAG on real patients? | **No** — use the CUPCase and DDXPlus stages |
| Comparable to published MedEinst numbers? | **No** — different cases, different construction |

Run it. Treat a positive result as "the mechanism works as designed", and read
the CUPCase and DDXPlus tables for anything stronger — those are external data
and carry the external claim.

---

### Related work these designs come from

| Benchmark | Idea taken | Applied here |
|---|---|---|
| [MedEinst](https://arxiv.org/abs/2601.06636) (2026) | Counterfactual control/trap pairs; **Bias Trap Rate** = P(wrong on trap \| right on control). 5,383 pairs, 49 diseases. Frontier models keep high accuracy yet show *severe* trap rates — they do not adjust when discriminative evidence changes. | `--counterfactual` |
| [MedAbstain](https://arxiv.org/abs/2601.12471) (2026) | Context-omission perturbations with an explicit abstention option. Even high-accuracy models fail to abstain when uncertain. | `--unanswerable-fraction` |
| [DyReMe](https://arxiv.org/html/2510.09275) (2025) | Real-world distractors, dynamic generation to avoid contamination. | `NEEDLE_BANK` widened to 20 diagnoses; cases generated per run, never checked in as a fixed set to memorise |
| [CUPCase](https://huggingface.co/datasets/ofir408/CupCase) | Curated per-case distractors (3 per case) | `scripts/run_cupcase_eval.py` |

### Why the counterfactual design replaced my first attempt

`--paired` holds the answer fixed and adds a distractor. **A model that ignores
the vignette entirely and pattern-matches the surface syndrome scores 100% on
both halves.** It measures whether a distractor *derails* a model; it cannot
detect a model that was never reading the evidence in the first place.

`--counterfactual` makes the discriminative evidence flip the answer. The
prior-driven answer is wrong by construction on the trap, so the design cannot
be passed by ignoring the note. `--paired` is kept for the narrower question of
distractor-induced degradation, but it is not the primary endpoint.

### Power, at equal compute

| Endpoint | 40 runs | 60 runs | 100 runs | Falsifies what? |
|---|:---:|:---:|:---:|---|
| **Bias trap rate** | **65%** | **89%** | **99%** | Does evidence override priors? |
| **Distractor selection** | 76% | 92% | 100% | Does it name the designed wrong answer? |
| Aggregate top-3 accuracy | 57% | 77% | 95% | Is it more accurate overall? |
| Matched-pair retention | 8% | 26% | 59% | What survives an added distractor? |

Pairs needed for 80% power on the bias trap rate, by true effect:

| Apiro escapes | RAG escapes | Gap | Pairs | Case-runs |
|:---:|:---:|:---:|:---:|:---:|
| 90% | 25% | 65 pt | 20 | 40 |
| 85% | 30% | 55 pt | 26 | 52 |
| 80% | 40% | 40 pt | 44 | 88 |
| 70% | 45% | 25 pt | 108 | 216 |
| 60% | 50% | 10 pt | >200 | — |

**If contradiction soft-pruning does anything at all, the gap here is large** —
the trap is built to punish precisely the failure the mechanism prevents. If
the gap turns out to be small, that is the most informative negative result
this project could get, and worth more than another accuracy table.

---

## The older endpoint comparison

Aggregate top-3 accuracy is the wrong primary endpoint for this architecture.
Apiro is not built to out-diagnose an 8B model in general; it is built to
**reject distractors** and to **know when to abstain**. Aggregate accuracy
buries that under the variance of case difficulty, which is large and unrelated
to the claim.

Three endpoints, with simulated power at equal compute (one traversal per
case-run):

| Endpoint | 40 runs | 60 runs | 100 runs | What it answers |
|---|:---:|:---:|:---:|---|
| **Distractor selection** | **76%** | **92%** | **100%** | Does it name the designed wrong answer? |
| Aggregate top-3 accuracy | 56% | 77% | 95% | Is it more accurate overall? |
| Matched-pair retention | 8% | 26% | 59% | Of what it could solve, what survives a distractor? |

**Lead with distractor selection.** It is both the on-thesis endpoint and the
most statistically efficient one: it conditions on nothing, so every case
contributes, and it needs one run per case. `contradiction_needle` cases
already carry `metadata.wrong_diagnosis`, and CUPCase ships three curated
distractors per case, so no new data is needed.

Runs needed for 80% power, by how large the true effect is:

| Apiro selects a distractor | vs RAG at 40% | Case-runs for 80% power |
|:---:|:---:|:---:|
| 5% | 35 pt gap | 30 |
| 10% | 30 pt gap | 40 |
| 15% | 25 pt gap | 60 |
| 20% | 20 pt gap | 90 |
| 30% | 10 pt gap | >300 |

**Matched-pair retention is the weakest of the three on power** — it costs two
runs per pair and discards pairs either arm failed clean. Run it anyway, but as
mechanism evidence rather than the headline: it is the only design that
separates resilience from raw capability, because each pair is its own control.

```bash
python scripts/build_niah_cases.py --paired --num-cases 40 --seed 7
```

emits 40 matched pairs (80 cases): same haystack, same needle, same depth,
differing only by an injected contradiction, red herring or negation.

`run_niah_eval.py` prints all three tables. Read them in the order above.

---

## Why N matters more than anything else

Power analysis on the committed N = 25 result (Apiro wins 11 of the 15 cases
where it and Standard RAG disagree):

| N | Expected discordant | Expected p | Power (P[p < 0.05]) |
|---|:---:|:---:|:---:|
| 25 | 15 | 0.119 | **35%** |
| 50 | 30 | 0.016 | 68% |
| 75 | 45 | 0.003 | 87% |
| 100 | 60 | 0.0004 | **95%** |

At N = 25 the study was more likely to fail than succeed *even if Apiro is
genuinely better*. Nothing about the algorithm changes that; it is arithmetic.
**N ≈ 100 is the single highest-value action available.**

The counterpart: at N = 25, moving from 11 to 12 wins out of 15 discordant
cases flips p from 0.119 to 0.035. Do not chase that. A result one case away
from reverting is not a result.

### Independence, and why you cannot just crank the flag

Cases built from the same diagnosis share needles, distractors and phrasing.
McNemar and the bootstrap both assume independent observations, so a p-value
computed over near-duplicates is narrower than the evidence supports.

`NEEDLE_BANK` now holds 20 diagnoses (it held 6). At N = 120 that is 6 cases
per diagnosis rather than 20. The generator warns above 8, and
`run_niah_eval.py` prints the ratio above its significance tables. If you push
N past ~160, widen the bank first.

---

## What changed that will move the numbers

Two defects were suppressing the Apiro arm specifically. Both are fixed, and
**the committed results predate the fix** — they are not a valid baseline for
comparison.

1. **57% of Apiro's answer slots held markdown, not diagnoses.** The old
   parser stripped one leading bullet character, so `**Diagnosis 1:**` entered
   the differential as a diagnosis. Five of 25 cases contained no diagnosis at
   all; twelve contained one.

2. **The arms were graded over different budgets.** Baselines were scored over
   every non-empty line of raw output (uncapped, ~7.2 per case, max 17); Apiro
   over exactly 3 parsed slots.

Combined, the engine offered ~1.3 real candidates against baselines offering
~7. All arms now share `apiro.parsing.parse_differential` at
`config.N_DIFFERENTIAL`, and each per-case record carries `n_candidates` per
arm so the asymmetry cannot silently return.

**This is not a change that favours Apiro by construction.** The baselines are
now parsed properly too, which will help them on cases where their answer was
buried in prose. Expect all three arms to move.

---

## Step 0 — the offline suite

```bash
pytest                       # ~150 tests, no Ollama, no ChromaDB, no downloads
pytest tests/test_parsing.py tests/test_eval_metrics.py tests/test_calibration.py -v
```

Run this first on any new machine. It exercises the metrics, the calibration
maths and the parser without touching a model, so a failure here is a code
problem rather than an environment problem.

---

## Step 1 — regenerate the case set

```bash
python scripts/build_niah_cases.py --num-cases 120 --seed 7
```

Reports distinct diagnoses and cases-per-diagnosis on completion. Optionally
`pip install -e '.[benchmarks]'` first for exact `tiktoken` token counts;
without it the generator uses a words/0.75 heuristic and `target_tokens` is
approximate.

---

## Step 2 — the C-NIAH run

```bash
python scripts/run_niah_eval.py --cases data/niah_cases.json --real --limit 10   # smoke test
python scripts/run_niah_eval.py --cases data/niah_cases.json --real \
    --out data/niah_eval_results.json                                            # full run
```

Emits per-case verdicts, accuracy with Wilson intervals, per-family and
length × depth breakdowns, exact McNemar between every pair of arms, and the
case-independence ratio.

**Read `n_candidates` in the output first.** If the three arms are not offering
the same number of candidates, stop — the accuracy comparison is measuring
formatting, and that was the whole point of the fix.

---

## Step 3 — calibration

```bash
python scripts/run_safety_calibration_eval.py --input data/niah_eval_results.json --tau 0.65
```

Caveat that applies to every number it prints: the confidence signal is a
hand-tuned placeholder, not a fitted calibrator, and the script says so. On the
committed run Apiro's ECE was 0.452 — the worst of the three arms — from severe
under-confidence (mean confidence 0.228 against 68% accuracy), and τ = 0.65
abstained on all 25 cases. **Re-derive τ from the new confidence distribution
before quoting any abstention figure.** The AURC result is about confidence
*ranking* and is the one part that survives.

---

## Step 4 — CUPCase

```bash
python scripts/run_cupcase_eval.py --n 20 --describe-only    # inspect, no model calls
python scripts/run_cupcase_eval.py --n 60 --out data/cupcase_eval_results.json
```

The external benchmark. CUPCase ships three curated distractors per case, so
distractor rejection is measured directly rather than inferred from an accuracy
gap. Significance on a benchmark you wrote yourself is weak evidence; this is
the one to lead with if it holds up.

---

## Empirical Benchmark Results (run of 2026-08-30)

All three suites were run at **N = 10** against live Ollama (`llama3.1:8b`) and
the built `apiro_corpus`. N = 10 is far below every power threshold in this
document — 35% power at N = 25, and less here. **Nothing below is statistically
established**, and one result is significant in the direction the project did
not want.

### Headline

| Suite | N | Apiro | Standard RAG | Bare LLM | Apiro vs Bare |
|---|:-:|:-:|:-:|:-:|---|
| C-NIAH (top-3) | 10 | 10% | 10% | **20%** | — |
| CUPCase (top-1) | 10 | 10% | 30% | **50%** | −40 pp, p = 0.219 |
| CUPCase (top-5) | 10 | 40% | 30% | **60%** | — |
| DDXPlus (top-3) | 10 | 20% | 80% | **90%** | **−70 pp, p = 0.0156** |
| DDXPlus (MRR) | 10 | 0.083 | 0.558 | **0.733** | — |

**Apiro is last on every suite.** On DDXPlus the gap against a bare LLM is
statistically significant (exact McNemar, 0 wins / 7 losses of 7 discordant
cases) — the only significant result this project has produced, and it is
against the engine.

### The primary endpoints were never exercised

`data/niah_cases.json` was generated **without `--counterfactual`**
(`"counterfactual": false` in its config block), so the case set is the classic
five families. Consequently:

```
counterfactual_traps : null      <- bias trap rate, the primary endpoint
abstention           : null      <- fabrication rate on unanswerable cases
resilience           : null      <- matched-pair retention
```

The three endpoints built to test this architecture's actual claims produced no
data. Regenerate with `--counterfactual` before drawing any conclusion about
the mechanism:

```bash
./run_eval.sh generate niah      # generate now defaults to --counterfactual
```

### Two defects found in the results, one of them mine

**1. The CUPCase distractor-selection numbers were an artifact.** The run
reported apiro 10% / rag 20% / bare_llm 40%, which reads as a 4× advantage.
Rescored with the fix in `distractor_selection_rate`, **every arm is at 0%** —
no arm ever chose a distractor over the answer.

The cause: CUPCase's curated distractors are ICD-level near-misses of the
ground truth, not clinically wrong alternatives.

```
truth      : "Primary aldosteronism and secondary Cushing's syndrome."
distractors: "Primary hyperaldosteronism", "Cushing's syndrome"
```

A correct answer matches a distractor, so the metric flagged it as a trap
capture. It was measuring accuracy, inverted — which is why the arm with the
worst accuracy looked best. Fixed; a correct answer can no longer count as a
capture, and a case whose distractors all match the truth leaves the
denominator.

This also retires an earlier recommendation in this document: **CUPCase is not
the external distractor benchmark it was chosen to be.** Naming one of its
distractors is a *granularity* error, not a trap capture.

**2. Apiro abstains on 50% of answerable cases.** On C-NIAH it replied
`INSUFFICIENT EVIDENCE` on 5 of 10 cases, all of which had a findable needle.
Neither baseline abstained once.

That is a regression introduced by rule 7 of the synthesis prompt (the
abstention option). Together with rule 6 — "weigh the confirmed objective
findings above the typical presentation" — both were written for counterfactual
traps, where the prior *is* wrong. On ordinary cases, where the prior is
usually right, they are actively harmful. **The engine was tuned for a
benchmark that was then not run.**

### The root cause: the entropy signal is degenerate

Measured over **3,782 generated hypotheses** across all 68 traversal logs:

| Entropy | Count | Share |
|---|--:|--:|
| 0.10 (“1 diagnosis”) | 2,431 | **64.3%** |
| 0.65 / 0.693 (“many”) | 1,122 | 29.6% |
| 0.25 / 0.40 / 0.55 | 176 | 4.7% |
| 0.05 | 53 | 1.4% |

The score that is supposed to be a graded uncertainty measure is in practice a
**binary flag**: "one diagnosis" (64%) or "many" (30%), with almost nothing in
between. `synthesize_differential` ranks exploration claims by ascending
entropy, so roughly two-thirds of them tie at 0.10 and the "most specific
first" ordering is decided by insertion order. The entropy-guided frontier is,
for most nodes, not guided by entropy.

Why: `differential_breadth_entropy` asks the model *"how many distinct primary
diagnoses could plausibly cause this finding?"* — but by depth ≥ 1 the "finding"
handed to it is already a full diagnostic hypothesis:

```
H=0.10  "Pulmonary embolism with associated pleuritic chest pain and cough
         due to chronic obstructive..."
H=0.10  "Pneumonia with thoracic spine involvement, likely caused by a fungal
         infection such as Histoplasma..."
```

The honest answer to "how many diagnoses explain *Pulmonary embolism with...*"
is one. The signal is measuring **whether a claim is phrased as a diagnosis**,
not how uncertain the engine is.

### What that produces

Compound, over-specific hypotheses score lowest, so they rank first into
synthesis, so the final differential is exotic:

| Ground truth | Bare LLM top-1 | Apiro top-1 |
|---|---|---|
| Acute otitis media | Middle Ear Infection ✓ | Tuberculous meningitis |
| Scombroid food poisoning | Erythema Multiforme | Eosinophilic Granulomatosis with Polyangiitis |
| Pulmonary embolism | Pulmonary Metastatic Disease | Hypereosinophilic syndrome |
| Bronchitis | Acute Bronchitis ✓ | Mycoplasma pneumoniae infection |

DDXPlus differential overlap makes the same point numerically: Apiro recovers
**14%** of the reference differential (recall@5) with **0%** top-1 agreement,
against 41% / 40% for RAG and 33% / 40% for the bare LLM.

This is the rare-disease bias recorded in `Log.md` as the "10% Accuracy
Disaster" of Apiro 2.0, reappearing through a different mechanism. Three parts
of the current design push the same way: the entropy signal rewards
specificity, synthesis ranks by ascending entropy, and synthesis rule 4
explicitly instructs "prefer 'Pheochromocytoma' over 'Hypertensive crisis'".

### Calibration

| Metric | Apiro | RAG | Bare LLM | Better |
|---|--:|--:|--:|:-:|
| Accuracy | 10% | 10% | 20% | ↑ |
| ECE | 0.425 | 0.761 | 0.666 | ↓ |
| Brier | 0.270 | 0.591 | 0.488 | ↓ |
| AURC | 0.845 | 0.945 | 0.905 | ↓ |
| Coverage at τ = 0.65 | 0.00 | 0.90 | 0.70 | — |

**These numbers do not support a calibration claim.** Every arm's confidence
comes from a different hand-written heuristic — Apiro's from traversal signals,
the baselines' from output length and hedging words — and the script stamps the
output "heuristic placeholders, replace before publishing". Comparing the ECE
of two different functions says nothing about either. Apiro's lower ECE at 10%
accuracy mostly reflects that its confidence is low and so is its accuracy.

Apiro's AURC has also risen from 0.119 in the earlier run to 0.845, close to
the other two. The earlier selective-prediction advantage did not survive.
Coverage at the documented τ = 0.65 is again 0.00: the operating point answers
nothing.

### What this run is good evidence for

- The harness works end to end: all four stages ran, parity held
  (`n_candidates` equal across arms on CUPCase, 46/50/50 on DDXPlus).
- The parsing fix held: no markdown scaffolding in any arm's output.
- The evaluator scoring fix held.
- **The entropy signal, which the architecture is named for, does not
  discriminate.** That is a real finding, measured over 3,782 observations, and
  it does not depend on N = 10.

### What it is not evidence for

Anything about relative accuracy. N = 10 on three suites, with an engine
carrying a 50% over-abstention regression and a prompt tuned for a case set
that was not generated. Fix those, generate the counterfactual set, and re-run
before comparing arms again.


## Interpreting the result honestly

- Quote the interval, not just the point estimate. 17/25 is 68% with a 95% CI of
  [48%, 83%].
- Quote the paired test. Three arms on the same cases is a paired design;
  comparing two marginal intervals is the wrong test.
- Report `n_candidates` alongside accuracy.
- Report cases-per-diagnosis alongside N.
- Name the primary endpoint **before** the run, not after. Reporting whichever
  of three endpoints happened to reach p < 0.05 is how a null result gets
  written up as a positive one.
- If a comparison does not reach significance, say so. "Consistent with a
  substantial advantage, and underpowered to demonstrate one" is a defensible
  claim. "+28% lift" without a p-value is not.
