# 🩺 Apiro · AI Clinical Detective

> **Apiro** is an entropy-first AI clinical reasoning engine. Instead of relying on brute-force RAG or zero-shot generation, Apiro dynamically constructs and traverses a **Belief Graph** of clinical claims, actively chasing **Shannon Entropy** (epistemic uncertainty) and executing **NLI-driven contradiction pruning** to navigate complex, distractor-heavy patient presentations toward precise differential diagnoses.

---

## 📊 Empirical Benchmark Results

Apiro has been evaluated across two independent clinical evaluation suites on local open-weights infrastructure (`llama3.1:8b`, Hugging Face biomedical-NER, and ChromaDB vector corpus):

### 1. Clinical Needle-In-A-Haystack (C-NIAH) Adversarial Suite ($N=25$)
Grounding in recent clinical AI benchmarks (**Med-Gemini**, **MedOdyssey**, **NeedleBench**), C-NIAH embeds decisive diagnostic needles and adversarial distractors inside 2,000 to 8,000 token clinical haystacks across 5 adversarial test families:

| Evaluation Arm | Accuracy | Contradiction Needles ($N=9$) | Multi-Needle Synthesis ($N=4$) | Single Needle ($N=6$) | Red Herring ($N=4$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Apiro Belief Graph** 🏆 | **68.0%** (17/25) | **88.9%** (8/9) | **75.0%** (3/4) | 66.7% (4/6) | 50.0% (2/4) |
| **Bare LLM Zero-Shot** (`llama3.1:8b`) | **56.0%** (14/25) | 66.7% (6/9) | 25.0% (1/4) | **83.3%** (5/6) | 50.0% (2/4) |
| **Standard RAG Baseline** (Top-$k$ Cosine) | **40.0%** (10/25) | 44.4% (4/9) | 25.0% (1/4) | 33.3% (2/6) | 50.0% (2/4) |

* **+28.0% Accuracy Lift over Standard RAG**: RAG collapses on distractor-heavy contexts due to blind top-$k$ similarity retrieval of irrelevant snippets.
* **Double the Accuracy on Contradictions**: Apiro reached **88.9%** vs RAG's 44.4% on adversarial contradiction needles.
* **Triple the Accuracy on Multi-Needles**: Apiro reached **75.0%** vs RAG's 25.0% when reconciling distributed diagnostic facts.
* **Long-Context Resilience at 8,000 Tokens**: Apiro achieved **100% accuracy (5/5)** when needles were buried deep (50%–100% depth) within 8k-token contexts where baseline models suffered severe context degradation.

---

### 2. Real-World PubMed Central Clinical Reports ($N=10$)
Real-world, scrubbed clinical case reports evaluating rare and complex presentations:

| Evaluation Arm | Accuracy ($N=10$) | Distractor Resilience on Case 4 |
| :--- | :---: | :--- |
| **Apiro Belief Graph** | 20.0% (2/10) | ✅ **Correctly diagnosed Colon Adenocarcinoma** (rejected Crohn's distractor) |
| **Standard RAG Baseline** | **40.0%** (4/10) | ❌ **Trapped by Crohn's distractor** (blind similarity retrieval) |
| **Bare LLM Zero-Shot** | 20.0% (2/10) | ❌ **Trapped by Crohn's distractor** (hallucinated chronic IBD) |

---

## 🚀 The Core Vision & Architecture

```
                [ Patient Clinical Vignette ]
                             │
                             ▼
            ┌─────────────────────────────────┐
            │   Deterministic Axiom Parsing   │ ◄── Biomedical NER + Regex Lab Parser
            │   (Extract Immutable Anchors)   │
            └────────────────┬────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │  Depth 0: Anchor on Certainty   │ ◄── Low Entropy First (H -> 0)
            │   (Seed Labs, Vitals, Negations)│
            └────────────────┬────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │ Depth >= 1: Chase Uncertainty   │ ◄── High Entropy First (H -> 1.0)
            │  (Corpus Retrieval & Expansions)│
            └────────────────┬────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │  NLI Contradiction Soft-Pruner  │ ◄── Fast-Filter + Two-Stage LLM Judge
            │   (Penalize Distractor Tangents)│
            └────────────────┬────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │ Saturation / Critic Stop Check  │ ◄── Epistemic Variance Floor
            └────────────────┬────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │  Etiology Differential Synthesis│ ◄── 4-Tier Node Partitioning
            └─────────────────────────────────┘
```

---

## 🧠 Core Algorithmic Components

### 1. Epistemic Uncertainty & Shannon Entropy
For any clinical claim, Apiro extracts first-token log probabilities to calculate binary Shannon entropy ($H$):
$$H = -P(\text{Yes})\log_2 P(\text{Yes}) - P(\text{No})\log_2 P(\text{No})$$
* **Low Entropy ($H \to 0$):** Model is certain about the clinical finding.
* **High Entropy ($H \to 1.0$):** Model is uncertain—Apiro targets these nodes to resolve diagnostic ambiguity.

### 2. Depth-Aware Dynamic Frontier
* **Depth 0 (Anchors):** Sorted by lowest entropy ($2.0 - H$) to expand solid patient findings first.
* **Depth $\ge 1$ (Hypotheses):** Sorted by highest entropy ($H$) to chase decision boundaries and explore competing differentials.

### 3. Two-Stage NLI Contradiction Soft-Pruner
* **Stage 1 (Fast-Filter):** Memoized keyword extraction + antonym word-boundary matching. Resolves 95% of pairs with zero network latency.
* **Stage 2 (LLM Judge):** Asynchronous batched evaluation (`ThreadPoolExecutor`) only for high-similarity ambiguous pairs. Contradicted hypotheses receive a soft penalty (`0.8`), sinking them below confirmed evidence.

### 4. 4-Tier Evaluator with Concept Normalization
Evaluates generated differential lists against gold-standard targets using a strict 4-tier cascade:
1. **Exact Match**: Case-insensitive substring match.
2. **Clinical Concept Normalization**: Curated synonym groups mapping anatomical, histological, and infectious variants (e.g., *Colon adenocarcinoma* $\leftrightarrow$ *Colorectal mucinous adenocarcinoma*, *Miliary TB* $\leftrightarrow$ *Tuberculosis*, *PE* $\leftrightarrow$ *Pulmonary embolism*).
3. **LLM-as-a-Judge**: Dual-interface (`generate`/`chat`) clinical verification.
4. **Semantic Embedding Fallback**: SentenceTransformer cosine similarity ($\ge 0.85$).

---

## ⚡ Performance Optimizations

1. **LRU Vector & Query Caching (`apiro/corpus/embedder.py`)**:
   - `_encode_cache` and `_query_cache` with 4,096-entry LRU eviction.
   - Eliminates redundant SentenceTransformer vector encodings and ChromaDB lookups during repeated BFS branch queries.
2. **Memoized NLI Keyword Extraction (`apiro/graph/contradiction.py`)**:
   - `@functools.lru_cache(maxsize=8192)` tokenization.
   - Replaces per-pair regex execution with $\mathcal{O}(1)$ frozenset `isdisjoint` short-circuiting.

---

## 🛠️ Installation & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/theroid/Apiro.git
cd Apiro
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Benchmark Suites

```bash
# Run the 10-Case Real-World PMC Evaluation
python scripts/run_pmc_eval.py --real

# Run the 25-Case Clinical Needle-In-A-Haystack (C-NIAH) Evaluation
python scripts/run_niah_eval.py --cases data/niah_cases.json --real --out data/niah_eval_results.json

# Generate new C-NIAH cases across custom lengths & depths
python scripts/build_niah_cases.py --num-cases 50 --lengths 2000 4000 8000 16000 --out data/niah_cases_50.json
```

### 3. Launch the Live Web Visualizer
```bash
uvicorn scripts.app:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` to inspect the 3-column live UI with dynamic D3 force-directed belief graph rendering.
