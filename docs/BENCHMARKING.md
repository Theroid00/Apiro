# Benchmarking Apiro

What to run, in what order, and what to expect. Written for a machine with
Ollama and a built ChromaDB corpus; the offline step needs neither.

---

## TL;DR — the sequence

```bash
# 0. Offline. Seconds. Needs nothing but pytest.
pytest

# 1a. PRIMARY: counterfactual traps + unanswerable cases. Lead with this.
python scripts/build_niah_cases.py --counterfactual --num-cases 40 --seed 7

# 1b. Or the classic five-family set, at a size that can resolve an effect.
#     python scripts/build_niah_cases.py --num-cases 120 --seed 7

# 2. The main run. This is the one that matters.
python scripts/run_niah_eval.py --cases data/niah_cases.json --real \
    --out data/niah_eval_results.json

# 3. Calibration over the run from step 2.
python scripts/run_safety_calibration_eval.py --input data/niah_eval_results.json --tau 0.65

# 4. External benchmark. Downloads CUPCase on first use.
python scripts/run_cupcase_eval.py --n 60 --out data/cupcase_eval_results.json
```

Steps 2 and 4 are the long ones — budget roughly 2–4 minutes per case per arm
on a local 8B model, so ~120 cases is a multi-hour run. Start with
`--limit 10` to confirm the stack is healthy before committing to the full set.

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
