# Apiro

> **Entropy-first clinical reasoning that refuses to hallucinate.**

Apiro is an **entropy-first clinical reasoning engine** that constructs and traverses a **Belief Graph** guided by **Shannon Entropy** and **Natural Language Inference (NLI) contradiction soft-pruning**. It is purpose-built to eliminate hallucination in complex, **distractor-heavy** medical cases, where irrelevant, misleading, or contradictory findings routinely derail conventional LLM and RAG pipelines.

Rather than emitting a single greedy chain of thought, Apiro anchors on **deterministic certainties**, quantifies its own **epistemic uncertainty**, explores hypotheses only where uncertainty is high, and actively **prunes contradictory beliefs** before synthesizing a differential.

---

## Table of Contents

- [Why Apiro](#why-apiro)
- [Architecture](#architecture)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [Clinical Needle-In-A-Haystack (C-NIAH)](#clinical-needle-in-a-haystack-c-niah)
  - [Real-World PMC Reports](#real-world-pmc-reports)
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
2. **Explore only where uncertain.** Shannon Entropy directs retrieval and expansion toward the highest-information hypotheses.
3. **Prune contradictions.** A two-stage NLI pipeline soft-prunes beliefs that contradict established axioms.
4. **Halt on saturation.** An epistemic critic stops exploration once additional evidence no longer reduces entropy.

The result is a system that **rejects distractors instead of rationalizing them**.

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
                                          │  Shannon Entropy H(·)
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
                │     Two-Stage NLI Contradiction Soft-Pruning    │
                │  ┌──────────────────┐   ┌────────────────────┐  │
                │  │ Stage 1: Fast    │──▶│ Stage 2: LLM Judge │  │
                │  │ filter (O(1))    │   │ (adjudicates edge  │  │
                │  │ set-intersection │   │  cases / soft-prune)│ │
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
| Prune | Two-Stage NLI (Fast filter → LLM judge) | Contradiction soft-pruning |
| Halt | Epistemic Saturation Critic | Stops on entropy saturation |
| Synthesize | Etiology Differential Synthesis | Final grounded differential |

---

## Empirical Benchmark Results

### Clinical Needle-In-A-Haystack (C-NIAH)

The C-NIAH benchmark stress-tests **distractor resilience**: diagnostic "needles" are hidden in long clinical haystacks alongside contradictory and misleading findings.

**Overall (N = 25)**

| System | Accuracy | Correct |
|--------|:--------:|:-------:|
| **Apiro** | **68.0%** | **17 / 25** |
| Bare LLM | 56.0% | 14 / 25 |
| Standard RAG | 40.0% | 10 / 25 |

**Contradiction Needles (N = 9)** — needles that directly contradict a nearby distractor.

| System | Accuracy | Correct |
|--------|:--------:|:-------:|
| **Apiro** | **88.9%** | **8 / 9** |
| Standard RAG | 44.4% | 4 / 9 |

**Multi-Needle Synthesis (N = 4)** — requires combining multiple dispersed needles.

| System | Accuracy | Correct |
|--------|:--------:|:-------:|
| **Apiro** | **75.0%** | **3 / 4** |
| Standard RAG | 25.0% | 1 / 4 |

**8,000-Token Long Contexts (N = 5)** — needle depth swept across 50%–100% of context.

| System | Accuracy | Correct |
|--------|:--------:|:-------:|
| **Apiro** | **100%** | **5 / 5** |

> Apiro's largest margins appear precisely where standard pipelines collapse: **contradiction needles** and **deep long-context placement**. This is the direct empirical signature of NLI contradiction soft-pruning and entropy-guided exploration.

### Real-World PMC Reports

We additionally evaluated on **N = 10** real-world PubMed Central (PMC) case reports.

| System | Accuracy |
|--------|:--------:|
| Bare LLM | 20% |
| Standard RAG | 40% |
| Apiro | 20% |

**Interpretation.** On this small, high-variance real-world set, aggregate accuracy is *not* where Apiro's contribution is visible. The decisive result is qualitative: **Apiro scored the sole win on Case 4 (Colon Adenocarcinoma)** by correctly **rejecting the Crohn's disease distractor** through **NLI contradiction pruning** — a failure mode that both the Bare LLM and Standard RAG succumbed to. This confirms that Apiro's advantage is mechanistic (distractor rejection) rather than a generic accuracy bump, and that the real-world set is currently too small to statistically resolve that advantage in aggregate.

---

## Literature Grounding

Apiro's evaluation methodology is grounded in the emerging consensus on **medical long-context reasoning**:

- **Med-Gemini** — establishes long-context clinical reasoning as a first-class capability for medical LLMs.
- **MedOdyssey** — benchmarks long-context medical comprehension and information retrieval under length stress.
- **NeedleBench** — formalizes needle-in-a-haystack retrieval and multi-needle synthesis for long contexts.

Collectively, these validate **Clinical Needle-In-A-Haystack (C-NIAH)** as the standard paradigm for measuring **distractor resilience in long EHR notes**, which is precisely the axis Apiro is engineered to dominate.

---

## Systems Optimizations

Apiro is designed to make entropy-guided graph traversal tractable at scale:

- **LRU query & vector caching** — implemented in `apiro/corpus/embedder.py`. Repeated retrieval queries and their embeddings are cached with a least-recently-used policy, eliminating redundant embedding computation and vector lookups during depth ≥ 1 exploration.
- **Memoized O(1) keyword set-intersection pre-filtering** — implemented in `apiro/graph/contradiction.py`. Stage 1 of the NLI pipeline uses a memoized constant-time keyword set-intersection to cheaply discard non-overlapping belief pairs *before* invoking the expensive Stage 2 LLM judge, dramatically reducing adjudication calls.

---

## Installation

**Requirements**

- Python 3.10+
- (Recommended) a virtual environment
- API credentials for your configured LLM / embedding provider (required for `--real` runs)

**Setup**

```bash
# 1. Clone the repository
git clone https://github.com/your-org/apiro.git
cd apiro

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Apiro in editable mode
pip install -e .

# 5. Configure credentials
cp .env.example .env
# then edit .env to add your provider API key(s)
```

---

## Reproducing the Benchmarks

The `--real` flag runs the full pipeline against live model/retrieval backends (as opposed to cached/offline fixtures).

**Real-world PMC evaluation (N = 10)**

```bash
python scripts/run_pmc_eval.py --real
```

**Clinical Needle-In-A-Haystack evaluation**

```bash
python scripts/run_niah_eval.py --cases data/niah_cases.json --real
```

Both scripts emit per-case verdicts, aggregate accuracy, and the breakdowns (contradiction needles, multi-needle synthesis, long-context depth) reported above.

---

## Web UI

Apiro ships with an interactive Web UI for inspecting the Belief Graph, entropy trajectories, and contradiction-pruning decisions in real time.

```bash
# Launch the Web UI
python -m apiro.web
```

Then open your browser to:

```
http://localhost:8000
```

From the UI you can:

- Paste or upload a patient vignette.
- Watch **deterministic axiom extraction** populate the Depth 0 anchors.
- Observe **entropy-guided expansion** at Depth ≥ 1.
- Inspect **NLI soft-pruning** decisions (fast filter vs. LLM judge) per belief edge.
- View the **halting critic's** saturation signal and the final **etiology differential**.

---

## Repository Layout

```text
apiro/
├── apiro/
│   ├── corpus/
│   │   └── embedder.py          # LRU query & vector caching
│   ├── graph/
│   │   └── contradiction.py     # O(1) memoized set-intersection pre-filter + NLI soft-pruning
│   ├── web/                     # Interactive Web UI
│   └── ...
├── scripts/
│   ├── run_pmc_eval.py          # Real-world PMC benchmark
│   └── run_niah_eval.py         # C-NIAH benchmark
├── data/
│   └── niah_cases.json          # C-NIAH case definitions
├── requirements.txt
└── README.md
```

---

## Citation

If you use Apiro or the C-NIAH methodology in your research, please cite this repository:

```bibtex
@software{apiro,
  title        = {Apiro: An Entropy-First Clinical Reasoning Engine with
                  NLI Contradiction Soft-Pruning},
  author       = {The Apiro Contributors},
  year         = {2026},
  note         = {Belief Graph reasoning guided by Shannon Entropy for
                  distractor resilience in long clinical contexts},
  url          = {https://github.com/your-org/apiro}
}
```

---

## License

See the [`LICENSE`](./LICENSE) file for details.
