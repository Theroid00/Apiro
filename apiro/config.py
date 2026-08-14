"""
config.py — Apiro global constants and configuration.
All tuneable parameters live here. Import this everywhere.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOG_DIR    = DATA_DIR / "logs"

for _d in [DATA_DIR, CORPUS_DIR, CHROMA_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL  = "http://localhost:11434"
PRIMARY_MODEL    = "llama3.1:8b"        # Configured to use the locally available model
TOP_LOGPROBS     = 20                   # top-k logprobs for entropy computation (Ollama max is 20)
MAX_FIRST_TOKEN  = 1                    # only first token for entropy
MAX_ANSWER_TOKENS = 80                  # short answer generation
ENTROPY_TEMPERATURES = [0.3]            # single temperature — T=0.3 is the cleanest signal
# Temperature weights — T=0.3 is dominant (was 0.6 weight in 3-sample scheme);
# using a single sample eliminates the 3x call overhead with negligible quality loss.
ENTROPY_TEMP_WEIGHTS = {0.3: 1.0}

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL    = "all-mpnet-base-v2"
EMBED_DIM      = 768
CHROMA_COLLECTION = "apiro_corpus"
RAG_TOP_K      = 6    # chunks retrieved per query
# Minimum number of RAG chunks required before we trust corpus grounding.
# If fewer than this many chunks come back, the expander switches to parametric
# mode (LLM-only, no corpus constraint) so rare-disease nodes still expand
# meaningfully instead of recycling the same thin context.
RAG_MIN_CHUNKS_FOR_GROUNDING = 2

# ---------------------------------------------------------------------------
# Corpus chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS   = 300
CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------
MAX_TRAVERSAL_DEPTH = 8
MAX_NODES_PER_RUN   = 30              # hard cap on nodes expanded per traversal run
FRONTIER_SORT       = "entropy_desc"   # always expand highest-entropy node first
N_CHILD_HYPOTHESES  = 3                # child nodes generated per expansion

# BeliefGraph construction defaults. These were previously hard-coded in the
# BeliefGraph constructor and unreachable from config.
GRAPH_MAX_NODES = 200
GRAPH_MAX_DEPTH = 6

# Runtime bound on the *exploration* half of a run (depth >= 1 expansions).
# Seed expansions are deterministic and cheap to justify; exploration
# expansions each cost one generation call plus N_CHILD_HYPOTHESES entropy
# calls, so this is the knob that decides wall-clock per case.
# NOTE: before the saturation fix, runs halted after ~5 seed expansions and
# never reached this bound. Lower it if per-case latency matters more than
# depth of reasoning.
MAX_EXPLORATION_EXPANSIONS = 24

# Cap on how many deterministic axioms are seeded into the graph. Biomedical
# NER over a long vignette routinely yields 40+ entities, many of them
# duplicates or non-clinical; seeding all of them floods the graph budget and
# the prompt before any reasoning happens. Axioms are ranked by weight and the
# top MAX_SEED_NODES are kept.
MAX_SEED_NODES = 20

# RAG retrieval
RAG_DOMAIN_FILTER   = True            # filter ChromaDB by node.domain when True

# ---------------------------------------------------------------------------
# Path-length / diagnostic-hit evaluation
# ---------------------------------------------------------------------------
# When True, diagnostic hits are only counted in *generated* nodes (depth > 0),
# excluding seed nodes that may already contain the ground-truth diagnosis.
EVAL_EXCLUDE_SEED_HITS = True

# Secondary winner tie-breaker: if path_length is equal, EF wins if its
# entropy_auc is this fraction lower than BF's (e.g. 0.10 = 10% lower).
EVAL_AUC_TIEBREAKER_MARGIN = 0.10

# ---------------------------------------------------------------------------
# Heuristic seed entropy (used when entropy_engine=None in build_cases)
# ---------------------------------------------------------------------------
# Replaces the flat ln(2) default. Values calibrated on llama3.1:8b:
#   - symptom/history: high uncertainty (many DDx possible)
#   - lab: moderate (narrows to a set of conditions)
#   - imaging: lower (specific findings constrain heavily)
#   - vital: moderate-high
SEED_ENTROPY_BY_FINDING_TYPE: dict[str, float] = {
    "symptom":   0.80,
    "history":   0.72,
    "vital":     0.65,
    "lab":       0.58,
    "imaging":   0.32,
    "diagnosis": 0.20,   # explicit diagnosis mention is near-certain
}
SEED_ENTROPY_DEFAULT = 0.693   # ln(2) — max binary uncertainty fallback

# ---------------------------------------------------------------------------
# Vital sign thresholds (used by clinical_case_adapter.py)
# ---------------------------------------------------------------------------
VITAL_THRESHOLDS: dict[str, tuple[float, float]] = {
    "blood_pressure_systolic":  (90.0, 180.0),
    "blood_pressure_diastolic": (60.0, 120.0),
    "heart_rate":               (50.0, 120.0),
    "oxygen_saturation":        (0.0,   94.0),  # SpO2 below 94 is flagged
    "temperature":              (36.0,  38.5),
}

# ---------------------------------------------------------------------------
# Saturation stopping condition
# ---------------------------------------------------------------------------
# Theta values are calibrated empirically for llama3.1:8b using the yes/no
# verification prompt (epistemic_certainty_entropy). The model's "confident
# floor" across 4 real traversal runs is ~0.49 nats. Theta is set 0.05 nats
# above that floor so saturation fires when entropy genuinely plateaus:
#   H < theta for 5 consecutive nodes → saturated.
#
# Phase 3.4 (theta grid-search on MIMIC-III) will refine these values further.
# The genetics domain is kept lower (0.50) per the plan: "rare disease —
# explore more"; comorbidity higher (0.60) because comorbidities are
# inherently uncertain — a higher bar prevents premature stopping.
SATURATION_WINDOW       = 5      # look back at last N expanded nodes
SATURATION_MAX_VARIANCE = 0.04   # entropy variance threshold
# Depth-0 seed nodes are deterministic axioms injected with a fixed near-zero
# entropy (~0.01). Counting them in the saturation window makes the engine
# "converge" the moment the first SATURATION_WINDOW seeds are expanded — i.e.
# before any hypothesis has ever been generated. Saturation must therefore only
# look at exploration (depth >= 1) expansions.
SATURATION_EXPLORATION_ONLY = True
# Hard warm-up floor: never declare saturation before this many depth >= 1
# nodes have been expanded, regardless of how flat the entropy curve looks.
SATURATION_MIN_EXPLORATION = 8
# Guard: do NOT declare saturation when the mean RAG retrieval depth
# (chunks returned) across the window is below this value. Consistently
# sparse retrieval means the corpus is dry, not that the engine converged.
# Set to 0 to disable (default off — relies on min-chunk fallback instead).
SATURATION_CORPUS_DRY_GUARD = 0  # set >0 to enable (e.g. 2.0)
THETA_BY_DOMAIN = {
    "pathophysiology": 0.55,   # empirical: well-supported mechanism claims hit ~0.43
    "pharmacology":    0.55,   # empirical: nitroglycerin/angina hit 0.43 at depth 1
    "genetics":        0.70,   # empirical: ClinVar conflicting-classification claims
                               # plateau at 0.66-0.69 nats — model correctly uncertain
    "imaging":         0.55,
    "lab":             0.55,
    "treatment":       0.55,
    "comorbidity":     0.70,   # comorbidities inherently uncertain — higher threshold
}
DEFAULT_THETA = 0.55

# ---------------------------------------------------------------------------
# Rabbit hole detection
# ---------------------------------------------------------------------------
RABBIT_HOLE_MIN_DEPTH      = 3
RABBIT_HOLE_REVERSAL_WINDOW = 4

# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------
CONTRADICTION_MODEL    = "cross-encoder/nli-MiniLM2-L6-H768"
# EF traversal uses a tighter threshold (0.92) to avoid false-positive edges.
# BF baseline uses the spec threshold (0.85) — both values are published in the paper.
CONTRADICTION_THRESHOLD_EF  = 0.92   # entropy-first: tighter to reduce noise
CONTRADICTION_THRESHOLD_BF  = 0.92   # breadth-first: same to ensure fair comparison
CONTRADICTION_THRESHOLD     = 0.92   # default alias used by tests / standalone scripts
CONTRADICTION_PENALTY       = 0.8    # score penalty subtracted from soft-pruned nodes

# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------
DOMAINS = [
    "pathophysiology",
    "pharmacology",
    "genetics",
    "imaging",
    "lab findings",
    "treatment",
    "comorbidity",
]

# ---------------------------------------------------------------------------
# Corpus sources
# ---------------------------------------------------------------------------
PUBMED_BATCH_SIZE    = 500
PUBMED_MAX_ABSTRACTS = 500_000
OMIM_API_KEY         = ""   # Set via environment variable OMIM_API_KEY
PUBMED_SEARCH_TERMS  = [
    "diagnosis AND treatment",
    "differential diagnosis",
    "clinical presentation",
    "pathophysiology",
    "drug mechanism of action",
    "genetic disorder",
    "rare disease",
    "comorbidity",
]
