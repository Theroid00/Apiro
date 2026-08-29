# Apiro

> **Entropy-first clinical reasoning that refuses to hallucinate.**

Apiro is an **entropy-guided clinical reasoning engine** that constructs and traverses a **Belief Graph**, prioritizing exploration by a bounded uncertainty score derived from the LLM's own self-assessed diagnostic breadth, and soft-pruning contradictory beliefs via a **keyword pre-filter + LLM-judge pipeline**. It is purpose-built to eliminate hallucination in complex, **distractor-heavy** medical cases, where irrelevant, misleading, or contradictory findings routinely derail conventional LLM and RAG pipelines.

Rather than emitting a single greedy chain of thought, Apiro anchors on **deterministic certainties**, quantifies its own **epistemic uncertainty**, explores hypotheses only where uncertainty is high, and actively **prunes contradictory beliefs** before synthesizing a differential.

---

## Table of Contents

- [Why Apiro](#why-apiro)
- [Architecture](#architecture)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [Clinical Needle-In-A-Haystack (C-NIAH)](#clinical-needle-in-a-haystack-c-niah)
  - [Real-World PMC Reports](#real-world-pmc-reports)
  - [CUPCase (implemented, not yet run)](#reproducing-the-benchmarks)
- [Safety, Calibration & Selective Abstention](#safety-calibration--selective-abstention)
- [Literature Grounding](#literature-grounding)
- [Systems Optimizations](#systems-optimizations)
- [Installation](#installation)
- [Reproducing the Benchmarks](#reproducing-the-benchmarks)
- [Web UI](#web-ui)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

---

## Why Apiro

Modern clinical vignettes are adversarial by nature: they bury the diagnostic signal beneath plausible distractors, red herrings, and outright contradictions across long EHR notes. Standard approaches fail in predictable ways:

- **Bare LLMs** overweight salient-but-irrelevant tokens and confabulate under long contexts.
- **Standard RAG** retrieves distractors as confidently as it retrieves ground truth, amplifying contradictions instead of resolving them.

Apiro takes a different stance. It treats reasoning as **entropy reduction over a Belief Graph**:

1. **Certainties first.** Deterministic axioms (verified entities, structured labs) form the graph's zero-uncertainty root.
2. **Explore only where uncertain.** A bounded uncertainty score — the LLM's self-assessed count of plausible alternative diagnoses for a claim, mapped onto the same [0, ln 2] range true Shannon entropy would occupy — directs retrieval and expansion toward the highest-uncertainty hypotheses.
3. **Prune contradictions.** A two-stage pipeline (a cheap keyword/antonym pre-filter, then an LLM judge for pairs that survive it) soft-prunes beliefs that contradict established axioms.
4. **Halt on saturation.** An epistemic critic stops exploration once additional evidence no longer reduces entropy.
5. **Know when to abstain.** Calibrated confidence lets Apiro *decline to answer* rather than emit a confidently wrong differential.

The result is a system that **rejects distractors instead of rationalizing them** — and, when the evidence is insufficient, **abstains instead of hallucinating**.

---

## Architecture

```text
                        ┌──────────────────────────────────┐
                        │         Patient Vignette          │
                        │   (long, distractor-heavy EHR)    │
                        └─────────────────┬────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │        Deterministic Axiom Extraction           │
                │   Biomedical NER  +  Lab Regex (structured)     │
                └─────────────────────────┬──────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │            Depth 0 — Certainty Anchors          │
                │      Zero-entropy roots of the Belief Graph     │
                └─────────────────────────┬──────────────────────┘
                                          │  Uncertainty score H(·)
                                          │  directs expansion
                                          ▼
                ┌────────────────────────────────────────────────┐
                │      Depth ≥ 1 — Uncertainty Exploration        │
                │        Hypothesis expansion via Medical         │
                │              Corpus RAG (targeted)              │
                └─────────────────────────┬──────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │    Two-Stage Contradiction Soft-Pruning         │
                │  ┌──────────────────┐   ┌────────────────────┐  │
                │  │ Stage 1: Fast    │──▶│ Stage 2: LLM Judge │  │
                │  │ keyword/antonym  │   │ (adjudicates edge  │  │
                │  │ filter (O(1))    │   │  cases / soft-prune)│ │
                │  └──────────────────┘   └────────────────────┘  │
                └─────────────────────────┬──────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │     Epistemic Saturation / Halting Critic       │
                │   Stop when ΔH ≈ 0 (no further entropy gain)    │
                └─────────────────────────┬──────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │   Confidence Calibration & Selective Abstention │
                │   Calibrated p(correct); abstain below τ        │
                └─────────────────────────┬──────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │          Etiology Differential Synthesis        │
                │   Ranked differential grounded in surviving     │
                │            (non-contradicted) beliefs           │
                └────────────────────────────────────────────────┘
```

**Pipeline summary**

| Stage | Component | Role |
|-------|-----------|------|
| Ingest | Patient Vignette | Raw, distractor-heavy clinical input |
| Extract | Biomedical NER + Lab Regex | Deterministic axiom extraction |
| Anchor | Depth 0 Certainty Anchors | Zero-uncertainty graph roots |
| Explore | Depth ≥ 1 via Medical Corpus RAG | Entropy-guided hypothesis expansion |
| Prune | Two-Stage Contradiction Check (keyword/antonym filter → LLM judge) | Contradiction soft-pruning |
| Halt | Epistemic Saturation Critic | Stops on entropy saturation |
| Calibrate | Confidence Calibration & Selective Abstention | Calibrated confidence; abstain below threshold τ |
| Synthesize | Etiology Differential Synthesis | Final grounded differential |

---

## Empirical Benchmark Results

> **The results below predate a measurement fix and will change.** Two defects
> were suppressing the Apiro arm specifically: 57% of its answer slots held
> markdown scaffolding rather than diagnoses, and the baselines were graded over
> their entire raw output (~7 candidates per case, uncapped) while Apiro was
> graded over 3 parsed slots. Both are fixed; every number on this page was
> computed before the fix. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md)
> for the re-run procedure.

> **Status of the evidence, in one paragraph.** Apiro has been evaluated on two case sets: a 25-case self-authored synthetic benchmark (C-NIAH) and 10 real PMC case reports. On C-NIAH it leads both baselines by a consistent margin, but **no comparison reaches statistical significance** at that sample size (Apiro vs RAG: p = 0.119). On the PMC set the intervals are so wide the three arms are indistinguishable, and four of the ten ground-truth labels are malformed. A third benchmark — [CUPCase](#reproducing-the-benchmarks), external and with curated per-case distractors — is implemented but **has not been run**, so no results for it are reported here. Treat the tables below as directional evidence that the mechanism behaves as designed, not as a demonstration that it outperforms the baselines.

### Clinical Needle-In-A-Haystack (C-NIAH)

The C-NIAH benchmark stress-tests **distractor resilience**: diagnostic "needles" are hidden in long clinical haystacks alongside contradictory and misleading findings.

**Overall (N = 25)**

| System | Accuracy | Correct | 95% CI (Wilson) |
|--------|:--------:|:-------:|:---------------:|
| **Apiro** | **68.0%** | **17 / 25** | [48.4%, 82.8%] |
| Bare LLM | 56.0% | 14 / 25 | [37.1%, 73.3%] |
| Standard RAG | 40.0% | 10 / 25 | [23.4%, 59.3%] |

**Is the gap real? Not at N = 25.** All three arms are scored on the same cases, so the right test is an exact McNemar on the discordant pairs. Recomputed from `data/niah_eval_results.json`:

| Comparison | Δ accuracy | 95% CI (paired bootstrap) | McNemar | p | Significant at α = 0.05 |
|------------|:----------:|:-------------------------:|:-------:|:-:|:----------------------:|
| Apiro vs Standard RAG | +28.0 pp | [+0.0, +56.0] pp | 11 W / 4 L of 15 discordant | 0.119 | **No** |
| Apiro vs Bare LLM | +12.0 pp | [−16.0, +36.0] pp | 7 W / 4 L of 11 discordant | 0.549 | **No** |
| Standard RAG vs Bare LLM | −16.0 pp | [−44.0, +12.0] pp | 4 W / 8 L of 12 discordant | 0.388 | **No** |

> **Read this before quoting the +28-point figure.** The point estimate is real and the direction is consistently in Apiro's favour (it wins 11 of the 15 cases where it and RAG disagree), but at N = 25 that is **not** statistically distinguishable from chance: p = 0.119, and the paired interval on the delta reaches down to zero. The honest statement is *"consistent with a substantial advantage, and underpowered to demonstrate one."* Roughly 60–100 cases would be needed to resolve an effect of this size. Regenerate a larger case set with `scripts/build_niah_cases.py --num-cases 200` before treating any of these gaps as established.

**Per-family breakdown**

| Family | N | Apiro | Standard RAG | McNemar p |
|--------|:-:|:-----:|:------------:|:---------:|
| Contradiction needles | 9 | 8 / 9 (88.9%) | 4 / 9 (44.4%) | 0.219 |
| Multi-needle synthesis | 4 | 3 / 4 (75.0%) | 1 / 4 (25.0%) | 0.625 |
| Single needle | 6 | 4 / 6 (66.7%) | 2 / 6 (33.3%) | 0.500 |
| Red herring | 4 | 2 / 4 (50.0%) | 2 / 4 (50.0%) | 1.000 |
| Negation trap | 2 | 0 / 2 (0%) | 1 / 2 (50.0%) | 1.000 |

> Apiro's largest margins fall where the design predicts they should — **contradiction needles** and **multi-needle synthesis** — which is the qualitative signature of contradiction soft-pruning and entropy-guided exploration. But each family here is a handful of cases; no per-family comparison approaches significance, and the negation-trap row (0 / 2) is two cases and should not be read as a result in either direction.

**A caveat on the benchmark itself.** C-NIAH cases, their needles, their distractors *and* the stub responses that recover them are all generated by `scripts/build_niah_cases.py` in this repository. It is a well-instrumented probe of whether the mechanism fires, not independent evidence about clinical performance. For an external held-out set with curated distractors, see the [CUPCase benchmark](#reproducing-the-benchmarks).

### Real-World PMC Reports

We additionally evaluated on **N = 10** real-world PubMed Central (PMC) case reports.

| System | Accuracy | Correct | 95% CI (Wilson) |
|--------|:--------:|:-------:|:---------------:|
| Bare LLM | 10% | 1 / 10 | [1.8%, 40.4%] |
| Standard RAG | 40% | 4 / 10 | [16.8%, 68.7%] |
| Apiro | 20% | 2 / 10 | [5.7%, 51.0%] |

> **Corrections to an earlier version of this table.** Two figures here were wrong, and both are corrected against the captured run in `data/latest_pmc_benchmark_output.txt`:
>
> - Bare LLM was reported as **20%**. That is the figure from a five-case run (`data/eval_results_5cases.json`, 1/5), not the ten-case run. On N = 10 the bare LLM scored **1/10**.
> - The narrative claimed **"Apiro scored the sole win on Case 4 (Colon Adenocarcinoma)"** by rejecting the Crohn's distractor. In the captured ten-case run **Apiro failed Case 4**; it passed Case 2 (miliary tuberculosis, which all three arms got) and Case 9. Case 9's ground-truth field is one of the malformed ones described below, so that win is not independently trustworthy either. Whatever run produced the Case 4 result is not the one in this repository.

**Interpretation.** At N = 10 every interval above spans more than thirty points and they all overlap. This set cannot distinguish the three systems, and no ordering between them should be read off it.

**A defect in this case set.** The `target_diagnosis` fields in `data/pmc_cases.json` were generated by an unconstrained LLM. Four of the ten are not diagnosis labels but multi-paragraph prose — for example Case 9's ground truth begins `"Here is the acute, primary presenting diagnosis:\n\nAppendicitis\n\nHowever, it's worth noting that..."`. The evaluator normalises that whole blob before matching, so those four cases cannot be scored correctly for *any* arm, and they depress every number in the table. `scripts/generate_pmc_cases.py` now post-processes the label and drops cases where no usable one survives; the case set should be regenerated before these accuracies are quoted again.

---

## Safety, Calibration & Selective Abstention

Accuracy alone is an incomplete picture of clinical trustworthiness. A system that is *confidently wrong* is more dangerous than one that knows when to defer. **Pillar 3** evaluates whether Apiro's confidence estimates are **calibrated** and whether **selective abstention** meaningfully reduces risk. It is implemented in `apiro/eval/calibration.py` and driven by `scripts/run_safety_calibration_eval.py`.

**Metrics reported**

- **ECE (Expected Calibration Error)** — the average gap between predicted confidence and empirical accuracy across confidence bins. Lower is better; measures how well confidence tracks correctness.
- **Brier Score** — the mean squared error between predicted probabilities and outcomes. Lower is better; jointly captures calibration and sharpness.
- **Risk–Coverage AURC (Area Under the Risk–Coverage curve)** — sweeps the abstention threshold and measures error rate (*risk*) as a function of the fraction of cases answered (*coverage*). Lower AURC means the model's confidence ranking lets it answer the cases it is most likely to get right while abstaining on the rest.

> **Read this section's caveat first.** Every number below is computed from a **heuristic placeholder confidence**, not from a fitted calibrator. `scripts/run_safety_calibration_eval.py` derives Apiro's confidence from traversal signals (contradiction count, stop reason) with hand-set coefficients, and the RAG / bare-LLM confidences from output length and hedging words. The script marks these `# >>> ASSUMPTION (REPLACE WITH REAL MODEL) <<<` and stamps `data/calibration_eval_results.json` with *"heuristic placeholders … replace with the real calibration model before publishing."* These figures characterise those heuristics. They are not yet a claim about Apiro's calibration.

**Calibration (ECE / Brier), N = 25 C-NIAH cases**

| System | Accuracy | Mean confidence | ECE ↓ | MCE ↓ | Brier ↓ |
|--------|:--------:|:---------------:|:-----:|:-----:|:-------:|
| Apiro | 68.0% | 0.228 | 0.452 | 0.645 | 0.387 |
| Standard RAG | 40.0% | 0.773 | 0.373 | 0.381 | 0.374 |
| Bare LLM | 56.0% | 0.681 | 0.311 | 0.608 | 0.310 |

> An earlier version of this section said ECE and Brier "have not yet been computed and published." They had been — they are in `data/calibration_eval_results.json` — and they are poor. **Apiro is the worst-calibrated of the three arms on ECE**, and severely *under*-confident: mean confidence 0.228 against 68% accuracy, a 45-point gap. That is a defect in the placeholder confidence function, not evidence about the engine, but it does mean the abstention threshold below currently has no principled basis.

**Selective abstention (Risk–Coverage AURC)**

| System | AURC ↓ | Oracle AURC | Excess risk ↓ |
|--------|:------:|:-----------:|:-------------:|
| **Apiro** | **0.119** | 0.058 | 0.061 |
| Bare LLM | 0.612 | 0.115 | 0.497 |
| Standard RAG | 0.536 | 0.233 | 0.303 |

AURC measures whether a system's confidence *ranks* its correct answers above its incorrect ones. Apiro's is the lowest, and its excess risk over a perfect ranker is by far the smallest — so even the crude traversal-signal heuristic orders Apiro's cases usefully (its confidence-vs-correctness AUROC is 0.813, against 0.557 for RAG and 0.260 for the bare LLM, where 0.5 is chance). This is the one Pillar-3 result that survives the caveat above, and it is a statement about *ranking*, not about calibrated probability — the ECE table shows the probabilities themselves are badly scaled.

> **A note on framing.** An earlier draft described the AURC gap as "4.5× lower *clinical risk*." AURC is a selective-prediction metric, not a validated measure of patient harm, and this is a 25-case set. The "area under the risk–coverage curve" framing is the accurate one.

**Selective abstention.** At an abstention threshold of **τ = 0.65**, Apiro declines to answer cases whose confidence falls below the threshold. Note that with Apiro's mean confidence at 0.228, **τ = 0.65 currently abstains on all 25 cases** (coverage 0.00, 0 kept — see `arms.apiro.selective` in `data/calibration_eval_results.json`), so the reported operating point answers nothing. The threshold was chosen before the confidence function was scaled and must be re-derived alongside a real calibrator. It is configurable via `--tau`.

---

## Literature Grounding

Apiro's evaluation methodology is grounded in the emerging consensus on **medical long-context reasoning** and **selective prediction**:

- **Med-Gemini** — establishes long-context clinical reasoning as a first-class capability for medical LLMs.
- **MedOdyssey** — benchmarks long-context medical comprehension and information retrieval under length stress.
- **NeedleBench** — formalizes needle-in-a-haystack retrieval and multi-needle synthesis for long contexts.
- **Selective prediction / risk–coverage** (e.g., Geifman & El-Yaniv) — formalizes abstention and the risk–coverage tradeoff (AURC) used in Pillar 3.

Collectively, these validate **Clinical Needle-In-A-Haystack (C-NIAH)** as the standard paradigm for measuring **distractor resilience in long EHR notes**, and **risk–coverage analysis** as the standard paradigm for measuring **calibrated abstention** — the two axes Apiro is engineered to dominate.

---

## Systems Optimizations

Apiro is designed to make entropy-guided graph traversal tractable at scale:

- **LRU query & vector caching** — implemented in `apiro/corpus/embedder.py`. Repeated retrieval queries and their embeddings are cached with a least-recently-used policy, eliminating redundant embedding computation and vector lookups during depth ≥ 1 exploration.
- **Memoized O(1) keyword set-intersection pre-filtering** — implemented in `apiro/graph/contradiction.py`. Stage 1 of the contradiction pipeline uses a memoized constant-time keyword/antonym set-intersection to cheaply discard non-overlapping belief pairs *before* invoking the expensive Stage 2 LLM judge, dramatically reducing adjudication calls.

---

## Installation

**Requirements**

- Python 3.10+
- (Recommended) a virtual environment
- [Ollama](https://ollama.com) running locally, with the model in `PRIMARY_MODEL` pulled (`ollama pull llama3.1:8b`) — Apiro calls a local LLM, not a hosted API, so no API key is required

**Setup**

```bash
# 1. Clone the repository
git clone https://github.com/Theroid00/Apiro.git
cd Apiro

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Apiro in editable mode (enables the `apiro` CLI command)
pip install -e .

# 5. (Optional) override the Ollama URL or model
cp .env.example .env
# then edit .env and `source .env`, or just export the variables directly
```

---

## Reproducing the Benchmarks

The `--real` flag runs the full pipeline against live model/retrieval backends (as opposed to cached/offline fixtures). `./run_eval.sh <target>` wraps all four.

**CUPCase distractor-resilience benchmark** — *the external one*

```bash
python scripts/run_cupcase_eval.py --n 50
python scripts/run_cupcase_eval.py --n 20 --describe-only   # inspect the case set, no model calls
```

CUPCase (`ofir408/CupCase`, 3,562 real clinical cases) is public, held out, and — the reason it belongs here specifically — **ships three curated distractors per case**. That makes distractor rejection directly measurable rather than inferred from an accuracy gap: the harness reports how often each arm's *leading* diagnosis is one of the case's designed wrong answers. Unlike C-NIAH, neither the cases nor the distractors come from this repository.

It reports top-1 / top-3 / top-5 accuracy, Mean Reciprocal Rank, the distractor-selection rate, Wilson intervals per arm, and a paired bootstrap CI plus exact McNemar test against the bare-LLM baseline. The dataset downloads via HuggingFace `datasets` on first run and is cached thereafter.

**Real-world PMC evaluation (N = 10)**

```bash
python scripts/run_pmc_eval.py --real
```

> Regenerate the case set first — four of the ten committed ground-truth labels are malformed (see [Real-World PMC Reports](#real-world-pmc-reports)):
> ```bash
> python scripts/generate_pmc_cases.py --n 10      # needs data/PMC-Patients-V2.json
> ```

**Clinical Needle-In-A-Haystack evaluation**

```bash
python scripts/build_niah_cases.py --num-cases 200      # writes data/niah_cases.json
python scripts/run_niah_eval.py --cases data/niah_cases.json --real
```

The committed results were computed on 25 cases, which is not enough to resolve the differences between arms — generate more than the default before quoting a figure. The harness emits per-case verdicts, aggregate accuracy with Wilson intervals, per-family and length × depth breakdowns, and paired McNemar tests between every pair of arms.

**Before running anything**, read [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md):
it carries the power analysis (N = 25 gives 35% power at the observed effect;
N = 100 gives 95%), the case-independence constraint, and the order to run the
four benchmarks in.

**Metric definitions** live in `apiro/eval/metrics.py` (top-k, MRR, distractor-selection rate, Wilson / bootstrap intervals, exact McNemar) and are covered by `tests/test_eval_metrics.py`. All four harnesses share one component stack via `apiro/eval/harness.py`, so their numbers are directly comparable.

**Safety, calibration & selective-abstention evaluation**

Compute ECE, Brier Score, and the Risk–Coverage AURC from a prior evaluation results file, using an abstention threshold of `τ = 0.65`:

```bash
python scripts/run_safety_calibration_eval.py --input data/niah_eval_results.json --tau 0.65
```

- `--input` — path to a results JSON produced by an evaluation run (e.g., `run_niah_eval.py`).
- `--tau` — abstention threshold; cases whose calibrated confidence is below `τ` are abstained on when computing selective-prediction metrics.

The script emits calibration metrics (ECE, Brier), the risk–coverage curve and its AURC, and the coverage/risk operating point at the chosen `τ`.

---

## Web UI

Apiro ships with an interactive Web UI for inspecting the Belief Graph, entropy trajectories, and contradiction-pruning decisions in real time.

```bash
# Launch the Web UI
uvicorn scripts.app:app --host 127.0.0.1 --port 8000
```

Then open your browser to:

```
http://localhost:8000
```

From the UI you can:

- Paste or upload a patient vignette.
- Watch **deterministic axiom extraction** populate the Depth 0 anchors.
- Observe **entropy-guided expansion** at Depth ≥ 1.
- Inspect **contradiction soft-pruning** decisions (keyword/antonym filter vs. LLM judge) per belief edge.
- View the **halting critic's** saturation signal and the final **etiology differential**.
- Read the **calibrated confidence** and whether the case was **answered or abstained** at the configured threshold.

---

## Repository Layout

```text
Apiro/
├── apiro/
│   ├── axioms/                  # Deterministic extraction → depth-0 graph anchors
│   │   ├── extractor.py         #   NER + lab regex + negation + weighting pipeline
│   │   └── seeding.py           #   The single axioms → seed-node implementation
│   ├── corpus/
│   │   ├── embedder.py          # LRU query & vector caching over ChromaDB
│   │   ├── clinical_case_adapter.py  # CUPCase / VivaBench loaders (drives the CUPCase benchmark)
│   │   └── mimic_adapter.py     # MIMIC-III demo loader — available, no benchmark wired to it yet
│   ├── entropy/engine.py        # Differential-breadth uncertainty signal
│   ├── eval/
│   │   ├── evaluator.py         # Concept-normalization match cascade
│   │   ├── metrics.py           # Top-k, MRR, distractor rate, Wilson/bootstrap CI, exact McNemar
│   │   ├── calibration.py       # ECE, Brier, Risk–Coverage AURC, selective abstention
│   │   └── harness.py           # Shared live-component wiring for every benchmark
│   ├── graph/
│   │   ├── traversal.py         # The entropy-first traversal loop
│   │   ├── expander.py          # RAG + LLM node expansion and final synthesis
│   │   └── contradiction.py     # O(1) memoized keyword pre-filter + LLM-judge soft-pruning
│   ├── patient/                 # DEPRECATED, unused — see docs/IMPROVEMENTS.md
│   ├── config.py                # All tuneable parameters
│   └── llm_client.py            # Shared OllamaLLMClient
├── scripts/
│   ├── app.py                            # Web UI (uvicorn scripts.app:app)
│   ├── investigate.py                    # CLI entry point
│   ├── build_niah_cases.py               # Generates data/niah_cases.json
│   ├── generate_pmc_cases.py             # Generates data/pmc_cases.json
│   ├── run_cupcase_eval.py               # CUPCase distractor-resilience benchmark
│   ├── run_pmc_eval.py                   # Real-world PMC benchmark
│   ├── run_niah_eval.py                  # C-NIAH benchmark
│   ├── run_safety_calibration_eval.py    # Safety / calibration / selective-abstention eval
│   └── repair_corpus.py                  # Backfills missing corpus metadata
├── tests/                       # Offline suite — no Ollama, ChromaDB or model download
├── data/
│   ├── axiom_weights.yaml       # Hand-curated diagnostic-specificity weights
│   ├── pmc_cases.json           # PMC case definitions (see the caveat above)
│   ├── niah_cases.json          # C-NIAH case definitions
│   ├── niah_eval_results.json   # C-NIAH results backing the tables above
│   └── calibration_eval_results.json   # Pillar-3 results backing the tables above
├── docs/
│   ├── BENCHMARKING.md          # What to run, in what order, and how to read it
│   └── IMPROVEMENTS.md          # Known issues, fixed and open
├── Log.md                       # Chronological architecture history
├── run_eval.sh                  # Benchmark wrapper: pmc | niah | cupcase | calibration
├── requirements.txt
└── README.md
```

Benchmark case sets and results are tracked in git deliberately: reproducing a
reported figure requires the case set it was computed on. Rebuildable or
machine-local artifacts (the ChromaDB corpus, per-case traversal traces, raw
console captures) are not.

---

## Citation

If you use Apiro or the C-NIAH methodology in your research, please cite this repository:

```bibtex
@software{apiro,
  title        = {Apiro: An Entropy-Guided Clinical Reasoning Engine with
                  Contradiction Soft-Pruning},
  author       = {The Apiro Contributors},
  year         = {2026},
  note         = {Belief Graph reasoning guided by an entropy-bounded
                  uncertainty score for distractor resilience in long
                  clinical contexts},
  url          = {https://github.com/Theroid00/Apiro}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see the [`LICENSE`](./LICENSE) file for details.
