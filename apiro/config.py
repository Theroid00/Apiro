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
