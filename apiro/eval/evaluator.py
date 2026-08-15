"""apiro/eval/evaluator.py

Evaluation utilities for clinical diagnosis synthesis outputs.

This module provides tooling to determine whether a set of candidate
diagnoses ("synthesis") produced by a model contains a clinically valid
match for a known ground-truth diagnosis, and to render human-readable
evaluation summaries.

Matching proceeds through a cascade of increasingly expensive strategies:

    1. Normalized substring / exact matching (fast, deterministic).
    2. Clinical concept normalization via a curated equivalence map.
    3. LLM-as-a-Judge fallback (optional, requires an ``llm_client``).
    4. Semantic embedding similarity fallback (optional, requires an
       ``embedder`` such as a SentenceTransformer).

Each strategy returns as soon as it produces a confident hit, so the more
expensive fallbacks are only invoked when the cheaper ones are inconclusive.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Cosine-similarity threshold above which two normalized concept strings are
#: considered semantically equivalent by the embedding fallback.
DEFAULT_EMBEDDING_THRESHOLD: float = 0.82

#: Qualifiers that are stripped during normalization because they typically
#: describe *modifiers* of a diagnosis rather than the diagnostic entity
#: itself. Order does not matter because they are applied as a set of tokens.
CLINICAL_QUALIFIERS: Set[str] = {
    "acute",
    "subacute",
    "chronic",
    "recurrent",
    "relapsing",
    "remitting",
    "wild-type",
    "wild type",
    "wildtype",
    "primary",
    "secondary",
    "tertiary",
    "idiopathic",
    "essential",
    "benign",
    "malignant",
    "early",
    "late",
    "late-stage",
    "early-stage",
    "advanced",
    "metastatic",
    "localized",
    "diffuse",
    "focal",
    "mild",
    "moderate",
    "severe",
    "suspected",
    "probable",
    "possible",
    "presumed",
    "confirmed",
    "definitive",
    "provisional",
    "differential",
    "working",
    "left",
    "right",
    "bilateral",
    "unilateral",
    "upper",
    "lower",
    "distal",
    "proximal",
    "de novo",
    "new onset",
    "new-onset",
    "known",
    "underlying",
    "concurrent",
    "concomitant",
    "active",
    "inactive",
    "latent",
    "stable",
    "unstable",
    "decompensated",
    "compensated",
}

#: Filler/descriptor words removed to reduce a phrase to its clinical core.
STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "without",
    "and",
    "or",
    "due",
    "to",
    "in",
    "on",
    "at",
    "for",
    "by",
    "disease",
    "disorder",
    "syndrome",
    "condition",
    "diagnosis",
    "state",
    "type",
    "grade",
    "stage",
    "possibly",
    "likely",
    "probably",
    "clinically",
}

# ---------------------------------------------------------------------------
# Clinical concept normalization dictionary
# ---------------------------------------------------------------------------
#
# Maps a wide range of surface forms (abbreviations, synonyms, histological
# subtypes, and common clinical manifestations) to a single canonical key.
# Two concepts are considered equivalent if they normalize to the same
# canonical key. The keys are themselves canonical strings.
#
# NOTE: All keys and values are compared after passing through
# ``_normalize_text``, so they should be written in a human-readable form and
# will be normalized internally when the lookup map is built.

_CLINICAL_SYNONYM_GROUPS: List[List[str]] = [
    # Colorectal cancer and its histological subtypes
    [
        "colorectal cancer",
        "colorectal carcinoma",
        "colon cancer",
        "colon carcinoma",
        "colon adenocarcinoma",
        "colorectal adenocarcinoma",
        "rectal cancer",
        "rectal adenocarcinoma",
        "crc",
    ],
    # Tuberculosis
    [
        "tuberculosis",
        "tb",
        "pulmonary tb",
        "pulmonary tuberculosis",
        "mycobacterium tuberculosis infection",
        "consumption",
    ],
    # Pulmonary embolism
    [
        "pulmonary embolism",
        "pe",
        "pulmonary thromboembolism",
        "lung embolism",
    ],
    # Myocardial infarction
    [
        "myocardial infarction",
        "mi",
        "heart attack",
        "acute coronary syndrome",
        "acs",
        "stemi",
        "nstemi",
        "st elevation myocardial infarction",
        "non st elevation myocardial infarction",
    ],
    # Systemic lupus erythematosus
    [
        "systemic lupus erythematosus",
        "sle",
        "lupus",
        "lupus erythematosus",
    ],
    # G6PD deficiency
    [
        "g6pd deficiency",
        "glucose 6 phosphate dehydrogenase deficiency",
        "glucose-6-phosphate dehydrogenase deficiency",
        "favism",
    ],
    # Cerebrovascular accident
    [
        "stroke",
        "cva",
        "cerebrovascular accident",
        "brain attack",
        "cerebral infarction",
        "ischemic stroke",
    ],
    # Deep vein thrombosis
    [
        "deep vein thrombosis",
        "dvt",
        "deep venous thrombosis",
        "venous thromboembolism",
        "vte",
    ],
    # Chronic obstructive pulmonary disease
    [
        "chronic obstructive pulmonary disease",
        "copd",
        "emphysema",
    ],
    # Congestive heart failure
    [
        "congestive heart failure",
        "chf",
        "heart failure",
        "cardiac failure",
    ],
    # Diabetes mellitus type 2
    [
        "type 2 diabetes mellitus",
        "type 2 diabetes",
        "t2dm",
        "diabetes mellitus type 2",
        "niddm",
        "non insulin dependent diabetes mellitus",
    ],
    # Diabetes mellitus type 1
    [
        "type 1 diabetes mellitus",
        "type 1 diabetes",
        "t1dm",
        "diabetes mellitus type 1",
        "iddm",
        "insulin dependent diabetes mellitus",
    ],
    # Rheumatoid arthritis
    [
        "rheumatoid arthritis",
        "ra",
    ],
    # Urinary tract infection
    [
        "urinary tract infection",
        "uti",
        "cystitis",
        "bladder infection",
    ],
    # Gastroesophageal reflux disease
    [
        "gastroesophageal reflux disease",
        "gerd",
        "acid reflux",
        "reflux esophagitis",
    ],
    # Community acquired pneumonia
    [
        "pneumonia",
        "community acquired pneumonia",
        "cap",
        "lung infection",
    ],
    # Hepatocellular carcinoma
    [
        "hepatocellular carcinoma",
        "hcc",
        "liver cancer",
        "primary liver cancer",
    ],
    # Renal cell carcinoma
    [
        "renal cell carcinoma",
        "rcc",
        "kidney cancer",
        "clear cell renal cell carcinoma",
    ],
    # Non-Hodgkin lymphoma
    [
        "non hodgkin lymphoma",
        "nhl",
        "diffuse large b cell lymphoma",
        "dlbcl",
    ],
    # Acute kidney injury
    [
        "acute kidney injury",
        "aki",
        "acute renal failure",
        "arf",
    ],
    # Chronic kidney disease
    [
        "chronic kidney disease",
        "ckd",
        "chronic renal failure",
        "crf",
        "chronic renal insufficiency",
    ],
]


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_text(text: str) -> str:
    """Return a lower-cased, punctuation-free, whitespace-collapsed form.

    This is the base normalization used everywhere else. It intentionally
    does *not* strip clinical qualifiers or stopwords so that it can be used
    to build canonical keys that remain human-recognizable.
    """
    if text is None:
        return ""
    lowered = str(text).strip().lower()
    # Normalize a few common unicode dashes to a plain hyphen first.
    lowered = lowered.replace("\u2010", "-").replace("\u2011", "-")
    lowered = lowered.replace("\u2013", "-").replace("\u2014", "-")
    # Preserve hyphenated qualifiers such as "wild-type" before we strip
    # punctuation by temporarily converting hyphens to spaces.
    lowered = lowered.replace("-", " ")
    # Remove any remaining punctuation.
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    lowered = _WHITESPACE_RE.sub(" ", lowered).strip()
    return lowered


def _strip_qualifiers(text: str) -> str:
    """Remove clinical qualifier tokens and generic stopwords.

    Multi-word qualifiers (e.g. ``"new onset"``) are removed before single
    tokens so that phrase-level qualifiers do not leave orphan tokens behind.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    # Remove multi-word qualifiers first (longest first for greediness).
    multiword = sorted(
        (q for q in CLINICAL_QUALIFIERS if " " in _normalize_text(q)),
        key=lambda q: len(q),
        reverse=True,
    )
    for qualifier in multiword:
        norm_q = _normalize_text(qualifier)
        normalized = re.sub(rf"\b{re.escape(norm_q)}\b", " ", normalized)

    tokens = normalized.split()
    single_qualifiers = {
        _normalize_text(q) for q in CLINICAL_QUALIFIERS if " " not in _normalize_text(q)
    }

    kept: List[str] = [
        tok
        for tok in tokens
        if tok not in single_qualifiers and tok not in STOPWORDS
    ]

    stripped = _WHITESPACE_RE.sub(" ", " ".join(kept)).strip()
    # If stripping removed everything (e.g. the phrase was entirely
    # qualifiers), fall back to the un-stripped normalized form so we never
    # return an empty concept.
    return stripped or normalized


def _build_canonical_map() -> Dict[str, str]:
    """Build a mapping from every normalized surface form to a canonical key.

    The canonical key for each synonym group is the normalized form of the
    first entry in that group.
    """
    canonical_map: Dict[str, str] = {}
    for group in _CLINICAL_SYNONYM_GROUPS:
        if not group:
            continue
        canonical = _normalize_text(group[0])
        for surface in group:
            key_full = _normalize_text(surface)
            key_stripped = _strip_qualifiers(surface)
            if key_full:
                canonical_map[key_full] = canonical
            if key_stripped:
                canonical_map.setdefault(key_stripped, canonical)
    return canonical_map


#: Prebuilt lookup: normalized surface form -> canonical concept key.
_CANONICAL_MAP: Dict[str, str] = _build_canonical_map()


def _canonicalize(concept: str) -> str:
    """Resolve a concept to its canonical clinical key if known.

    Tries the full normalized form first, then the qualifier-stripped form.
    Falls back to the qualifier-stripped normalized text when the concept is
    not present in the curated synonym map.
    """
    full = _normalize_text(concept)
    if full in _CANONICAL_MAP:
        return _CANONICAL_MAP[full]

    stripped = _strip_qualifiers(concept)
    if stripped in _CANONICAL_MAP:
        return _CANONICAL_MAP[stripped]

    return stripped


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------


def _substring_match(pred_norm: str, truth_norm: str) -> bool:
    """Bidirectional token-aware substring match on normalized strings."""
    if not pred_norm or not truth_norm:
        return False
    if pred_norm == truth_norm:
        return True
    # Guard against trivially short matches (single generic tokens).
    if len(pred_norm) < 3 or len(truth_norm) < 3:
        return pred_norm == truth_norm
    # Require whole-token containment to avoid spurious substring hits
    # (e.g. "cancer" inside "pancreatic cancer" is fine, but "an" is not).
    if re.search(rf"\b{re.escape(pred_norm)}\b", truth_norm):
        return True
    if re.search(rf"\b{re.escape(truth_norm)}\b", pred_norm):
        return True
    return False


def _llm_judge(
    prediction: str,
    ground_truth: str,
    llm_client: Any,
) -> Optional[bool]:
    """Ask an LLM whether ``prediction`` is a valid clinical match.

    Supports two client interfaces:
        * ``llm_client.generate(prompt) -> str``
        * ``llm_client.chat(prompt) -> str``

    Returns ``True``/``False`` on a confident verdict, or ``None`` when the
    judge could not be invoked or produced an unparseable response.
    """
    if llm_client is None:
        return None

    prompt = (
        "You are a board-certified physician acting as a strict clinical "
        "evaluator. Determine whether the PREDICTED diagnosis should be "
        "counted as a correct match for the GROUND TRUTH diagnosis.\n\n"
        "Count it as a MATCH if the predicted diagnosis is ANY of the "
        "following relative to the ground truth:\n"
        "  1. A valid clinical synonym or abbreviation "
        "(e.g. 'MI' vs 'myocardial infarction').\n"
        "  2. A direct histological subtype "
        "(e.g. 'colon adenocarcinoma' for 'colorectal cancer').\n"
        "  3. A specific clinical manifestation of the same underlying "
        "entity (e.g. 'pulmonary TB' for 'tuberculosis').\n\n"
        "Do NOT count it as a match if it is merely a related condition, a "
        "differential, a complication, or an anatomically/etiologically "
        "distinct disease.\n\n"
        f"GROUND TRUTH: {ground_truth}\n"
        f"PREDICTED: {prediction}\n\n"
        "Answer with a single word: YES or NO."
    )

    response: Optional[str] = None
    try:
        if hasattr(llm_client, "generate") and callable(
            getattr(llm_client, "generate")
        ):
            response = llm_client.generate(prompt)
        elif hasattr(llm_client, "chat") and callable(getattr(llm_client, "chat")):
            response = llm_client.chat(prompt)
        else:
            logger.warning(
                "llm_client provided but exposes neither 'generate' nor 'chat'."
            )
            return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM judge invocation failed: %s", exc)
        return None

    if response is None:
        return None

    # Normalize possible structured responses (dict/objects) to text.
    text = _extract_llm_text(response)
    if not text:
        return None

    verdict = text.strip().lower()
    # Look for an explicit yes/no signal.
    if re.search(r"\byes\b", verdict):
        return True
    if re.search(r"\bno\b", verdict):
        return False

    logger.debug("Unparseable LLM judge response: %r", text)
    return None


def _extract_llm_text(response: Any) -> str:
    """Best-effort extraction of text content from a variety of LLM responses."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    # OpenAI-style chat completion dict.
    if isinstance(response, dict):
        for key in ("text", "content", "output", "message", "answer"):
            if key in response and isinstance(response[key], str):
                return response[key]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("text"), str):
                    return first["text"]
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
    # Object with a ``.content`` or ``.text`` attribute.
    for attr in ("content", "text", "message"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    return str(response)


def _embedding_match(
    prediction: str,
    ground_truth: str,
    embedder: Any,
    threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
) -> Tuple[Optional[bool], float]:
    """Compare two concepts using embedding cosine similarity.

    ``embedder`` is expected to be a SentenceTransformer-like object exposing
    an ``encode(list_of_texts) -> ndarray`` method. Returns a tuple of
    ``(verdict, similarity)`` where ``verdict`` is ``None`` if embeddings
    could not be produced.
    """
    if embedder is None:
        return None, 0.0

    try:
        import numpy as np  # local import: only needed for this fallback
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("numpy unavailable for embedding fallback: %s", exc)
        return None, 0.0

    try:
        embeddings = embedder.encode(
            [prediction, ground_truth],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    except TypeError:
        # Fallback for encoders that do not accept the keyword arguments.
        try:
            embeddings = embedder.encode([prediction, ground_truth])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Embedding encode failed: %s", exc)
            return None, 0.0
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Embedding encode failed: %s", exc)
        return None, 0.0

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        logger.warning("Unexpected embedding shape: %s", getattr(arr, "shape", None))
        return None, 0.0

    vec_a, vec_b = arr[0], arr[1]
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return None, 0.0
    similarity = float(np.dot(vec_a, vec_b) / denom)
    return (similarity >= threshold), similarity


# ---------------------------------------------------------------------------
# Public API: hit checking
# ---------------------------------------------------------------------------


def _check_synthesis_hit(
    synthesis: List[str],
    ground_truth: str,
    embedder: Any = None,
    llm_client: Any = None,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
) -> Tuple[bool, str]:
    """Determine whether ``synthesis`` contains a valid match for ``ground_truth``.

    The check runs a cascade of strategies for each candidate diagnosis in
    ``synthesis`` and returns as soon as any candidate produces a hit.

    Strategy cascade (per candidate):
        1. Normalized exact / substring match  -> ``"exact"``
        2. Clinical concept canonicalization    -> ``"concept_normalization"``
        3. LLM-as-a-Judge (if ``llm_client``)   -> ``"llm_judge"``
        4. Embedding similarity (if ``embedder``) -> ``"embedding"``

    Parameters
    ----------
    synthesis:
        Candidate diagnoses produced by the system under evaluation.
    ground_truth:
        The reference diagnosis.
    embedder:
        Optional SentenceTransformer-like encoder for the semantic fallback.
    llm_client:
        Optional client exposing ``generate(prompt)`` and/or ``chat(prompt)``.
    embedding_threshold:
        Cosine-similarity threshold for the embedding fallback.

    Returns
    -------
    tuple(bool, str)
        ``(hit, method)`` where ``method`` identifies the strategy that
        produced the hit, or ``"no_match"`` when nothing matched.
    """
    if not ground_truth or not isinstance(ground_truth, str):
        return False, "no_match"

    if synthesis is None:
        return False, "no_match"

    # Normalize the candidate list defensively.
    candidates: List[str] = [
        str(item) for item in synthesis if item is not None and str(item).strip()
    ]
    if not candidates:
        return False, "no_match"

    truth_norm = _normalize_text(ground_truth)
    truth_stripped = _strip_qualifiers(ground_truth)
    truth_canonical = _canonicalize(ground_truth)

    # ---- Pass 1 & 2: deterministic normalization-based matching ----
    for candidate in candidates:
        pred_norm = _normalize_text(candidate)
        pred_stripped = _strip_qualifiers(candidate)

        if _substring_match(pred_norm, truth_norm):
            return True, "exact"
        if _substring_match(pred_stripped, truth_stripped):
            return True, "exact"

        pred_canonical = _canonicalize(candidate)
        if pred_canonical and truth_canonical and pred_canonical == truth_canonical:
            return True, "concept_normalization"

    # ---- Pass 3: LLM-as-a-Judge fallback ----
    if llm_client is not None:
        for candidate in candidates:
            verdict = _llm_judge(candidate, ground_truth, llm_client)
            if verdict is True:
                return True, "llm_judge"

    # ---- Pass 4: semantic embedding fallback ----
    if embedder is not None:
        best_sim = -1.0
        best_hit = False
        for candidate in candidates:
            verdict, similarity = _embedding_match(
                candidate,
                ground_truth,
                embedder,
                threshold=embedding_threshold,
            )
            if similarity > best_sim:
                best_sim = similarity
            if verdict is True:
                best_hit = True
                break
        if best_hit:
            return True, "embedding"

    return False, "no_match"


# ---------------------------------------------------------------------------
# Public API: summary rendering
# ---------------------------------------------------------------------------


def _fmt_pct(numerator: float, denominator: float) -> str:
    """Return a formatted percentage string, guarding against zero division."""
    if not denominator:
        return "0.0%"
    return f"{(numerator / denominator) * 100.0:.1f}%"


def _print_summary(summary: Dict[str, Any]) -> None:
    """Render a human-readable evaluation summary to stdout.

    Expected (but tolerant) ``summary`` keys::

        {
            "total": int,                 # total evaluated cases
            "hits": int,                  # total correct (exact + broad)
            "exact_hits": int,            # exact / substring matches
            "broad_hits": int,            # concept/llm/embedding matches
            "method_breakdown": {         # optional per-method counts
                "exact": int,
                "concept_normalization": int,
                "llm_judge": int,
                "embedding": int,
                "no_match": int,
            },
            "misses": int,                # optional; derived if absent
        }

    Missing fields are derived where possible and default to zero otherwise.
    """
    summary = summary or {}

    total = int(summary.get("total", 0) or 0)
    exact_hits = int(summary.get("exact_hits", 0) or 0)
    broad_hits = int(summary.get("broad_hits", 0) or 0)

    # Derive total hits if not explicitly provided.
    hits = summary.get("hits")
    if hits is None:
        hits = exact_hits + broad_hits
    hits = int(hits or 0)

    misses = summary.get("misses")
    if misses is None:
        misses = max(total - hits, 0)
    misses = int(misses or 0)

    method_breakdown: Dict[str, int] = dict(summary.get("method_breakdown", {}) or {})

    width = 64
    line = "=" * width
    thin = "-" * width

    print(line)
    print("CLINICAL SYNTHESIS EVALUATION SUMMARY".center(width))
    print(line)

    print(f"  Total cases evaluated : {total}")
    print(f"  Correct (hits)        : {hits}  ({_fmt_pct(hits, total)})")
    print(f"  Incorrect (misses)    : {misses}  ({_fmt_pct(misses, total)})")
    print(thin)

    print("  Overall accuracy      : "
          f"{_fmt_pct(hits, total)}")
    print(thin)

    print("  Match-type breakdown")
    print(f"    - Exact matches     : {exact_hits}  "
          f"({_fmt_pct(exact_hits, total)} of all | "
          f"{_fmt_pct(exact_hits, hits)} of hits)")
    print(f"    - Broad matches     : {broad_hits}  "
          f"({_fmt_pct(broad_hits, total)} of all | "
          f"{_fmt_pct(broad_hits, hits)} of hits)")

    if method_breakdown:
        print(thin)
        print("  Match method breakdown")
        # Present known methods in a stable, meaningful order first.
        ordered_methods = [
            ("exact", "Exact / substring"),
            ("concept_normalization", "Concept normalization"),
            ("llm_judge", "LLM-as-a-Judge"),
            ("embedding", "Embedding similarity"),
            ("no_match", "No match"),
        ]
        seen: Set[str] = set()
        for key, label in ordered_methods:
            if key in method_breakdown:
                count = int(method_breakdown.get(key, 0) or 0)
                print(f"    - {label:<22}: {count}  "
                      f"({_fmt_pct(count, total)})")
                seen.add(key)
        # Emit any additional/custom methods not covered above.
        for key in sorted(method_breakdown):
            if key in seen:
                continue
            count = int(method_breakdown.get(key, 0) or 0)
            print(f"    - {key:<22}: {count}  ({_fmt_pct(count, total)})")

    print(line)


__all__ = [
    "_check_synthesis_hit",
    "_print_summary",
    "DEFAULT_EMBEDDING_THRESHOLD",
]
