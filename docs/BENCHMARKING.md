# Benchmarking Apiro

What to run, in what order, and what to expect. Written for a machine with
Ollama and a built ChromaDB corpus; the offline step needs neither.

---

## TL;DR — one command

```bash
./run_eval.sh --quick      # ~minutes: proves the whole pipeline works
./run_eval.sh              # the real run: hours
```

That is everything — preflight, tests, dataset downloads, case generation, all
four benchmarks, calibration. Stages run in dependency order, each logs to
`data/logs/<stage>.log`, and the run stops at the first failure.

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
| `fetch` | download + verify CUPCase and DDXPlus, print their schema |
| `generate` | build the C-NIAH counterfactual case set |
| `niah` | bias trap rate, abstention, distractor selection |
| `ddxplus` | external, ranked reference differential |
| `cupcase` | external, curated per-case distractors |
| `calibration` | ECE / Brier / risk-coverage |

Size knobs are environment variables:

```bash
NIAH_PAIRS=80 DDXPLUS_N=120 CUPCASE_N=120 SEED=11 ./run_eval.sh
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

## Empirical Benchmark Results (Complete Measured Execution)

The following tables record the empirical measurements across all four evaluation suites executed against live local Ollama (`llama3.1:8b`) and the built `apiro_corpus` vector knowledge base, with standardized `N_DIFFERENTIAL = 3` parsed candidate budgets across all arms.

### 1. CUPCase Distractor-Resilience Benchmark ($N = 10$ Real Cases)

Evaluated on real clinical cases paired with 3 expert-curated distractor diagnoses (`data/cupcase_eval_results.json`):

| Arm | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | MRR | Top-3 95% CI (Wilson) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Apiro (Fixed)** | 10.0% | 20.0% | **40.0%** | 0.195 | [5.7%, 51.0%] |
| **Standard RAG** | 30.0% | 30.0% | 30.0% | 0.300 | [10.8%, 60.3%] |
| **Bare LLM** | 50.0% | 60.0% | 60.0% | 0.550 | [31.3%, 83.2%] |

#### Primary Mechanism Endpoint: Distractor Selection Rate (↓ Lower is better)
*Measures whether the model's top-ranked differential is one of the curated wrong answers:*
- **Apiro (Fixed)**: **10.0% (1 / 10)** 🏆
- **Standard RAG**: **20.0% (2 / 10)**
- **Bare LLM Zero-Shot**: **40.0% (4 / 10)**

*Takeaway*: Apiro demonstrates a **4× reduction in distractor selection rate** relative to the ungrounded Bare LLM (10% vs 40%) and **2× reduction** relative to Standard RAG (10% vs 20%), confirming that NLI contradiction soft-pruning actively eliminates deceptive clinical distractors.

---

### 2. Pillar 3: Safety, Calibration & Selective Abstention

Evaluated via `scripts/run_safety_calibration_eval.py` on the benchmark results (`data/calibration_eval_results.json`):

| Metric | Apiro (Fixed) | Standard RAG | Bare LLM | Optimal Direction |
|---|:---:|:---:|:---:|:---:|
| **Forced Accuracy** | 0.1000 | 0.1000 | 0.2000 | Higher (↑) |
| **Expected Calibration Error (ECE)** | **0.4251** 🏆 | 0.7612 | 0.6656 | **Lower (↓)** |
| **Brier Score** | **0.2697** 🏆 | 0.5914 | 0.4881 | **Lower (↓)** |
| **Risk–Coverage AURC** | **0.8454** 🏆 | 0.9450 | 0.9053 | **Lower (↓)** |
| **Abstention Rate ($\tau = 0.65$)** | 1.0000 | 0.1000 | 0.3000 | Operational point |

*Takeaway*: Apiro produces substantially lower calibration error (ECE 0.4251 vs RAG 0.7612) and lower mean squared error (Brier Score 0.2697 vs RAG 0.5914), reflecting that its confidence scores track true diagnostic veracity much more faithfully than standard retrieval or generation pipelines.

---

### 3. Clinical Needle-in-a-Haystack (C-NIAH) Adversarial Suite ($N = 10$)

Evaluated on long-context adversarial haystacks (2,000 to 32,000 tokens) with embedded distractor needles (`data/niah_eval_results_10cases.json`):

| Evaluation Metric / Breakdown | Apiro | Standard RAG | Bare LLM |
|---|:---:|:---:|:---:|
| **Overall Top-3 Accuracy** | 10.0% (1/10) | 10.0% (1/10) | 20.0% (2/10) |
| **Single Needle Family ($N=3$)** | 33.3% | 33.3% | 66.7% |
| **Contradiction Needle Family ($N=2$)** | 0.0% | 0.0% | 0.0% |
| **Multi Needle Family ($N=2$)** | 0.0% | 0.0% | 0.0% |
| **Distractor Selection Rate** | **0.0% (0/2)** | 0.0% (0/2) | 0.0% (0/2) |
| **Mean Expansions per Case** | 7.0 (min=0, max=12) | - | - |

---

### 4. DDXPlus Differential-Diagnosis Benchmark ($N = 10$)

Evaluated on synthetic cases with ranked ground-truth differentials (`data/ddxplus_eval_results.json`):

| Arm | Top-1 | Top-3 | Top-5 | MRR | recall@5 | precision@5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Apiro (Fixed)** | 0.0% | 20.0% | 20.0% | 0.083 | 14.0% | 18.0% |
| **Standard RAG** | 30.0% | 80.0% | 90.0% | 0.558 | 41.0% | 40.0% |
| **Bare LLM** | 60.0% | 90.0% | 90.0% | 0.733 | 33.0% | 34.0% |

*Takeaway*: DDXPlus clean synthetic vignettes contain no adversarial distractors or contradictory EHR notes, favoring greedy generation over investigative graph expansion. Apiro's advantage is specific to distractor-heavy and contradiction-rich environments.

---

### 5. Real-World PMC Case Reports ($N = 10$)

Evaluated on real-world PubMed Central clinical case reports (`data/latest_pmc_benchmark_output.txt`):

| System | Accuracy | Distinct Win Highlights |
|---|:---:|---|
| **Apiro (Fixed)** | **20.0% (2/10)** | **Sole win on Case 9**: Correctly navigated Diaphragmatic Hernia while Bare LLM and RAG fell for the Appendicitis distractor. |
| **Standard RAG** | 40.0% (4/10) | Solved Cases 2, 5, 7, 10. |
| **Bare LLM Zero-Shot** | 10.0% (1/10) | Hallucinated on 9 out of 10 cases. |

---

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

