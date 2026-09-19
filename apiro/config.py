"""
config.py — Apiro global constants and configuration.
All tuneable parameters live here. Import this everywhere.

Environment overrides: OLLAMA_BASE_URL and PRIMARY_MODEL can be overridden
via environment variables of the same name (see .env.example). No .env
loader is bundled — either `export` them in your shell or `source .env`
before running, since adding python-dotenv as a dependency for two
variables wasn't worth it.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOG_DIR    = DATA_DIR / "logs"
# Paths are declarations only. Commands create the directories they write to;
# importing ``apiro.config`` must be safe in a read-only installation.

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_MODEL   = os.environ.get("PRIMARY_MODEL", "llama3.1:8b")
MAX_MODEL_CONCURRENCY = int(os.environ.get("APIRO_MAX_MODEL_CONCURRENCY", "2"))
MODEL_TEMPERATURE = float(os.environ.get("APIRO_MODEL_TEMPERATURE", "0.2"))
MODEL_SEED = int(os.environ.get("APIRO_MODEL_SEED", "7"))
MODEL_JSON_NUM_PREDICT = int(os.environ.get("APIRO_JSON_NUM_PREDICT", "768"))

# Reasoning architecture selected by CLI, web, and live evaluation.  The
# simplified engine normally uses one generation and permits one corrective
# generation when deterministic quality checks fail.
REASONING_MODE = os.environ.get("APIRO_REASONING_MODE", "simple").strip().lower()
if REASONING_MODE not in {"simple", "investigator"}:
    raise ValueError(
        "APIRO_REASONING_MODE must be 'simple' or 'investigator'"
    )

SIMPLE_MAX_FACTS = int(os.environ.get("APIRO_SIMPLE_MAX_FACTS", "12"))
SIMPLE_RAG_TOP_K = int(os.environ.get("APIRO_SIMPLE_RAG_TOP_K", "6"))
SIMPLE_MAX_CONTEXT_CHARS = int(
    os.environ.get("APIRO_SIMPLE_MAX_CONTEXT_CHARS", "12000")
)
SIMPLE_CORRECTIVE_PASS = os.environ.get(
    "APIRO_SIMPLE_CORRECTIVE_PASS", "true"
).strip().lower() in {"1", "true", "yes", "on"}

INVESTIGATOR_MAX_FACTS = int(os.environ.get("APIRO_INVESTIGATOR_MAX_FACTS", "20"))
INVESTIGATOR_MAX_CANDIDATES = int(
    os.environ.get("APIRO_INVESTIGATOR_MAX_CANDIDATES", "6")
)
INVESTIGATOR_MAX_ROUNDS = int(os.environ.get("APIRO_INVESTIGATOR_MAX_ROUNDS", "2"))
INVESTIGATOR_MAX_RETRIEVALS = int(
    os.environ.get("APIRO_INVESTIGATOR_MAX_RETRIEVALS", "2")
)
INVESTIGATOR_MAX_MODEL_CALLS = int(
    os.environ.get("APIRO_INVESTIGATOR_MAX_MODEL_CALLS", "2")
)
INVESTIGATOR_MAX_GRAPH_NODES = int(
    os.environ.get("APIRO_INVESTIGATOR_MAX_GRAPH_NODES", "50")
)
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
# Maximum cosine distance for a retrieved chunk to count as evidence.
# ChromaDB always returns the top-k nearest neighbours, however far away they
# are, so a rare-disease query still comes back with 6 confidently-formatted
# chunks about something else — which the expander then injects under
# "use ONLY what is stated here". Chunks beyond this distance are discarded,
# and if too few survive the expander falls back to parametric mode.
# Set to None to disable distance filtering.
RAG_MAX_DISTANCE = 0.65

# ---------------------------------------------------------------------------
# Corpus chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS   = 300
CHUNK_OVERLAP_TOKENS = 50

# Size of the final ranked differential. Every benchmark arm must be allowed
# the same number of candidates: the committed C-NIAH run graded the baselines
# over their entire raw output (~7 lines per case, uncapped) while capping
# Apiro at 3 parsed slots, so the arms were not answering the same question.
N_DIFFERENTIAL = 3

# BeliefGraph construction defaults. These were previously hard-coded in the
# BeliefGraph constructor and unreachable from config.
GRAPH_MAX_NODES = 200
GRAPH_MAX_DEPTH = 6

# Cap on how many deterministic axioms are seeded into the graph. Biomedical
# NER over a long vignette routinely yields 40+ entities, many of them
# duplicates or non-clinical; seeding all of them floods the graph budget and
# the prompt before any reasoning happens. Axioms are ranked by weight and the
# top MAX_SEED_NODES are kept.
MAX_SEED_NODES = 20

# Relevance weighting of the exploration frontier (see
# BeliefGraph.set_case_anchor). Exploration priority becomes
#     H * (RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * cos(claim, case))
# so a claim unrelated to this patient retains RELEVANCE_FLOOR of its raw
# entropy priority and a claim about this patient retains all of it.
# 1.0 disables relevance weighting entirely (pure entropy-first).
RELEVANCE_FLOOR = 0.4

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
